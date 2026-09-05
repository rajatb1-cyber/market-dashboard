"""
Diversified (correlation-adjusted) 1-day VaR for the Risk tab.

Standalone VaR magnitudes come from Rajat's manual implied vols ($Risk × ivol/√256);
the correlation matrix comes from historical returns of each position's **proxy
asset** (equities/FX via yfinance, US Treasury yields via FRED). Portfolio VaR over
a window = √(vᵀ R v) with v = signed standalone VaRs (long +, short −) and R the
proxy-return correlation over that window.

Return conventions (so "long position = +proxy_return"):
  · Equities  — % price return of the index
  · FX        — % return of USD-per-unit (JPY/CNH etc. inverted from X=X)
  · Rates     — −Δyield (price-like: yield down = gain)

Everything degrades gracefully: a proxy with no data is dropped from the
correlation and added back in quadrature (treated as uncorrelated).
"""
from __future__ import annotations

import json
import math
import urllib.request

import numpy as np
import pandas as pd

WINDOWS = {"1m": 21, "3m": 63, "6m": 126, "1y": 252, "2y": 504, "5y": 1260}

# proxy → yfinance ticker (USD-per-unit already, or index level)
_YF = {
    "SPX": "^GSPC", "Nasdaq": "^IXIC", "Russell": "^RUT", "Dow": "^DJI",
    "Gold": "GC=F", "Silver": "SI=F", "WTI": "CL=F", "Brent": "BZ=F",
    "Copper": "HG=F", "NatGas": "NG=F",
    "EUR": "EURUSD=X", "GBP": "GBPUSD=X", "AUD": "AUDUSD=X", "NZD": "NZDUSD=X",
}
# proxy → yfinance ticker quoted per-USD (invert to USD-per-unit)
_YF_INV = {"JPY": "JPY=X", "CNH": "CNH=X", "CHF": "CHF=X", "CAD": "CAD=X"}
# rates proxy → (country curve in rates.py, tenor column). Daily-ish yield curves:
# US Treasury.gov, Euro ECB, Japan MOF, Australia RBA, UK BoE (BoE ~3-wk lag).
_RATE_FETCH = {
    "US2y": ("US", "2Y"), "US5y": ("US", "5Y"), "US10y": ("US", "10Y"), "US30y": ("US", "30Y"),
    "EUR2y": ("EUR", "2Y"), "EUR5y": ("EUR", "5Y"), "EUR10y": ("EUR", "10Y"), "EUR30y": ("EUR", "30Y"),
    "UK10y": ("UK", "10Y"), "JPY10y": ("JPY", "10Y"),
    "AUD2y": ("AUD", "2Y"), "AUD5y": ("AUD", "5Y"), "AUD10y": ("AUD", "10Y"),
}
# FRED daily fallback for the US tenors only.
_FRED = {"US2y": "DGS2", "US5y": "DGS5", "US10y": "DGS10", "US30y": "DGS30"}

_YC_CACHE: dict = {}   # (country, start) → yield-curve DataFrame, per process

# ticker → Close Series, primed by compute() once per call so _proxy_returns
# reads from one batched download instead of one HTTP round trip per proxy.
_PROXY_YF_BATCH: dict = {}


def _batch_daily_closes(tickers, **dl_kwargs) -> dict:
    """One batched yf.download for many tickers → {ticker: Close Series (NaNs
    dropped)}. Tickers the batch cannot resolve (missing / all-NaN) are simply
    absent from the dict, so callers fall back to their original per-ticker
    yf.Ticker().history() path for those — preserving today's flaky-ticker
    behaviour exactly. auto_adjust is forced True (FX pairs / futures / indices,
    so adjusted == raw) so values stay byte-identical to the sequential path on
    every settled bar. Mirrors the batch-and-split logic in watchlist._all_daily."""
    import yfinance as yf
    uniq = list(dict.fromkeys(t for t in tickers if t))
    out: dict = {}
    if not uniq:
        return out
    try:
        raw = yf.download(uniq, auto_adjust=True, progress=False,
                          group_by="ticker", threads=True, **dl_kwargs)
    except Exception:
        raw = None
    if raw is None or len(raw) == 0:
        return out
    cols = raw.columns
    multi = isinstance(cols, pd.MultiIndex)
    lvl0 = set(cols.get_level_values(0)) if multi else set()
    for tk in uniq:
        try:
            if multi:
                if tk not in lvl0:
                    continue
                sub = raw[tk]
            else:
                sub = raw
            if "Close" not in sub.columns:
                continue
            s = sub["Close"].dropna()
            if len(s):
                out[tk] = s
        except Exception:
            continue
    return out


def _prime_proxy_batch(proxies, start: str) -> None:
    """Pre-download every yfinance-backed proxy in ONE batched request so
    _proxy_returns reads from _PROXY_YF_BATCH instead of one HTTP round trip per
    proxy. Proxies the batch can't resolve are simply absent → _proxy_returns
    falls back to its per-ticker yf.Ticker().history() path, unchanged. Rates
    proxies (FRED/curve) are untouched — they aren't Yahoo tickers."""
    tickers = [tk for p in proxies
               if (tk := (_YF.get(p) or _YF_INV.get(p)))]
    _PROXY_YF_BATCH.clear()
    if tickers:
        _PROXY_YF_BATCH.update(_batch_daily_closes(tickers, start=start))


def _yield_curve(country: str, start: str) -> pd.DataFrame:
    """Reuse rates.py's yield-curve fetchers (memoised per process)."""
    k = (country, start)
    if k in _YC_CACHE:
        return _YC_CACHE[k]
    df = pd.DataFrame()
    try:
        import rates
        if country == "US":
            df = rates.fetch_us_treasury(months_back=30)
        elif country == "EUR":
            df = rates.fetch_ecb_curve(start=start)
        elif country == "JPY":
            df = rates.fetch_japan_jgb()
        elif country == "AUD":
            df = rates.fetch_australia_bonds()
        elif country == "UK":
            df = rates.fetch_uk_gilts(start=start)
    except Exception:
        df = pd.DataFrame()
    _YC_CACHE[k] = df
    return df


def _fred_series(series_id: str, key: str, start: str) -> pd.Series:
    url = ("https://api.stlouisfed.org/fred/series/observations"
           f"?series_id={series_id}&api_key={key}&file_type=json&observation_start={start}")
    d = json.load(urllib.request.urlopen(url, timeout=30))
    idx, val = [], []
    for o in d.get("observations", []):
        if o["value"] not in (".", ""):
            idx.append(pd.Timestamp(o["date"])); val.append(float(o["value"]))
    return pd.Series(val, index=idx)


def _proxy_returns(proxy: str, start: str, fred_key: str | None) -> pd.Series:
    try:
        if proxy in _YF or proxy in _YF_INV:
            import yfinance as yf
            tk = _YF.get(proxy) or _YF_INV[proxy]
            h = _PROXY_YF_BATCH.get(tk)
            if h is None:
                h = yf.Ticker(tk).history(start=start)["Close"].dropna()
            else:
                h = h.copy()
            if not len(h):
                return pd.Series(dtype=float)
            idx = pd.DatetimeIndex(h.index)
            if idx.tz is not None:            # tz-aware (per-ticker fallback) vs
                idx = idx.tz_localize(None)   # already-naive (batched download)
            h.index = idx
            if proxy in _YF_INV:
                h = 1.0 / h
            return h.pct_change().dropna()
        if proxy in _RATE_FETCH:
            country, tenor = _RATE_FETCH[proxy]
            # US: FRED first when a key is available — ONE request returns the full
            # multi-year daily history (needed for the 5y window), vs treasury.gov's
            # one-request-PER-MONTH curve fetch (only 30 months deep, 64 reqs for 5y).
            if country == "US" and proxy in _FRED and fred_key:
                try:
                    yv = _fred_series(_FRED[proxy], fred_key, start)
                    if len(yv):
                        return (-yv.diff()).dropna()
                except Exception:
                    pass
            df = _yield_curve(country, start)
            if df is not None and not df.empty and tenor in df.columns:
                y = pd.to_numeric(df[tenor], errors="coerce").dropna()
                y.index = pd.to_datetime(y.index)
                y = y[y.index >= pd.Timestamp(start)]
                if len(y):
                    return (-y.diff()).dropna()   # price-like: yield down = gain
            if proxy in _FRED and fred_key:        # US fallback
                yv = _fred_series(_FRED[proxy], fred_key, start)
                return (-yv.diff()).dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)


def _standalone(book, fx, sel_fut, sel_fx, products, ivols, proxies) -> pd.DataFrame:
    """[name, product, proxy, sign(+1/-1), var($, 1σ)] for the selected book."""
    rows = []
    for _, r in book.iterrows():
        s = r["Symbol"]
        if s not in sel_fut:
            continue
        if bool(r.get("is_option")):
            # options enter √(vᵀRv) via risk_options' extra_pos rows — the
            # book pass gave each a junk $0 row beside the real one (their
            # per-SYMBOL ivol is unset), duplicating every option in the
            # standalone-VaR table (Rajat 2026-09-05)
            continue
        prod = products.get(s, "Rates")
        iv = float(ivols.get(s) or 0.0)
        qty = float(r.get("Quantity") or 0.0)
        mult = float(r.get("Multiplier") or 0.0)
        fxr = float(r.get("FXRateToBase") or 1.0)
        pvb = float(r.get("position_value_base") or 0.0)
        if prod == "Rates":
            var = abs(qty) * mult * 0.01 * fxr * iv / math.sqrt(256)   # iv = bps annual normal
        else:
            var = abs(pvb) * (iv / 100.0) / math.sqrt(256)
        rows.append([s, prod, proxies.get(s, "US2y"), 1 if r["side"] == "Long" else -1, var])
    for _, r in fx.iterrows():
        c = r["Currency"]
        if c not in sel_fx:
            continue
        iv = float(ivols.get(c) or 0.0)
        exp = abs(float(r["USD_exposure"])) if pd.notna(r["USD_exposure"]) else 0.0
        var = exp * (iv / 100.0) / math.sqrt(256)
        rows.append([c, "FX", proxies.get(c, c), 1 if r["side"] == "Long" else -1, var])
    return pd.DataFrame(rows, columns=["name", "product", "proxy", "sign", "var"])


def compute(book, fx, sel_fut, sel_fx, products, ivols, proxies, fred_key=None,
            extra_pos=None) -> dict:
    """Returns {positions: DataFrame, windows: [(name, obs, diversified, undiv, benefit)],
    meta: {...}}. Correlation windows = WINDOWS.
    extra_pos: optional [[name, product, proxy, sign, var(1σ$)], …] rows appended
    to the book — used by risk_options to fold option positions into √(vᵀRv)
    (the proxy carries the correlation; var is delta-equiv or reval 1σ)."""
    pos = _standalone(book, fx, sel_fut, sel_fx, products, ivols, proxies)
    if extra_pos:
        pos = pd.concat(
            [pos, pd.DataFrame(extra_pos,
                               columns=["name", "product", "proxy", "sign", "var"])],
            ignore_index=True)
    if pos.empty:
        return {"positions": pos, "windows": [], "meta": {"error": "no positions selected"}}

    start = (pd.Timestamp.today() - pd.Timedelta(days=1900)).date().isoformat()  # ~5.2y for the 5y window
    proxies_u = list(dict.fromkeys(pos["proxy"]))   # unique, order-preserving
    _prime_proxy_batch(proxies_u, start)            # one batched download for all Yahoo proxies
    rets = {p: _proxy_returns(p, start, fred_key) for p in proxies_u}
    avail = [p for p in proxies_u if len(rets[p]) >= 60]
    dropped = [p for p in proxies_u if p not in avail]

    # signed VaR aggregated per proxy
    vmap = {}
    for _, r in pos.iterrows():
        vmap[r["proxy"]] = vmap.get(r["proxy"], 0.0) + r["sign"] * r["var"]

    R_df = pd.DataFrame({p: rets[p] for p in avail}).dropna(how="any") if avail else pd.DataFrame()
    v = np.array([vmap[p] for p in avail]) if avail else np.array([])
    drop_var = float(np.sum([vmap[p] ** 2 for p in dropped]))   # uncorrelated add-back
    undiv = float(pos["var"].abs().sum())

    out_rows = []
    for wname, wn in WINDOWS.items():
        sub = R_df.tail(wn) if not R_df.empty else R_df
        if len(sub) < max(5, wn // 3):
            out_rows.append((wname, len(sub), None, undiv, None))
            continue
        Rm = sub[avail].corr().values
        port = math.sqrt(max(0.0, float(v @ Rm @ v)) + drop_var)
        out_rows.append((wname, len(sub), port, undiv, undiv - port))

    # ── Per-asset-class VaR breakdown, per correlation window ─────────────────
    # For each window we return, per class:
    #   component  — Euler/marginal risk contribution; Σ = whole-book diversified VaR (additive,
    #                gives a true risk budget: % of book risk). Uses correlations across ALL proxies.
    #   standalone — the class's own sub-portfolio VaR √(v_cᵀ R_c v_c) (NOT additive; overstates).
    #   undiv/net  — sum |var| and signed sum within the class (window-independent).
    proxy2prod = {}
    for _, r in pos.iterrows():
        proxy2prod.setdefault(r["proxy"], r["product"])
    classes = list(dict.fromkeys(pos["product"]))
    undiv_c = {c: float(pos[pos["product"] == c]["var"].abs().sum()) for c in classes}
    net_c   = {c: float((pos[pos["product"] == c]["sign"]
                         * pos[pos["product"] == c]["var"]).sum()) for c in classes}

    by_ac_win = {}
    for wname, wn in WINDOWS.items():
        sub = R_df.tail(wn) if not R_df.empty else R_df
        if avail and len(sub) >= max(5, wn // 3):
            Rmat  = sub[avail].corr()
            Rm    = Rmat.values
            Rv    = Rm @ v
            sigma = math.sqrt(max(1e-12, float(v @ Rm @ v) + drop_var))
            comp_proxy = {p: float(v[i] * Rv[i] / sigma) for i, p in enumerate(avail)}
            for p in dropped:                                # independent add-back
                comp_proxy[p] = float(vmap[p] ** 2 / sigma)
            rows = []
            for c in classes:
                cps    = [p for p in vmap if proxy2prod.get(p) == c]
                comp_c = float(sum(comp_proxy.get(p, 0.0) for p in cps))
                cav    = [p for p in cps if p in avail]
                cdr    = [p for p in cps if p in dropped]
                if cav:
                    vc   = np.array([vmap[p] for p in cav])
                    Rc   = Rmat.loc[cav, cav].values
                    sdiv = math.sqrt(max(0.0, float(vc @ Rc @ vc))
                                     + float(np.sum([vmap[p] ** 2 for p in cdr])))
                elif cdr:
                    sdiv = math.sqrt(float(np.sum([vmap[p] ** 2 for p in cdr])))
                else:
                    sdiv = None
                rows.append((c, comp_c, comp_c / sigma * 100.0, sdiv, undiv_c[c], net_c[c]))
            rows.sort(key=lambda x: -(x[1] if x[1] is not None else 0.0))
            by_ac_win[wname] = {"obs": len(sub), "book_var": sigma, "rows": rows}
        else:
            rows = [(c, None, None, None, undiv_c[c], net_c[c]) for c in classes]
            by_ac_win[wname] = {"obs": len(sub), "book_var": None, "rows": rows}

    meta = {
        "n_days": int(len(R_df)),
        "hist_start": R_df.index.min().date().isoformat() if len(R_df) else None,
        "hist_end": R_df.index.max().date().isoformat() if len(R_df) else None,
        "dropped": dropped,
    }
    return {"positions": pos, "windows": out_rows,
            "by_asset_class_windows": by_ac_win, "meta": meta}

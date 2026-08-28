"""🧭 Core Markets — one-glance live board of the main assets Rajat follows,
with 1d / 1w / 1m / 3m / YTD changes.

Equities / FX / Commodities / Crypto — live from yfinance 5m bars
(~15-20min delay) vs previous session close; longer horizons vs daily
history (yf 2y daily; USD/CNH via watchlist's AlphaVantage path — no
Yahoo daily history for CNH).

Rates — REAL live government bond yields from CNBC's public quote API
(quote.cnbc.com, Refinitiv/Tradeweb-backed =RR feeds, near-real-time):
full US / DE (EUR) / UK / JP curves. This replaced the 2026-08-14 v1 ETF
duration proxies, which Rajat found "quite far off" — root causes: ETF
closes cut the day at 16:30/17:30 local while cash yields keep moving,
plus premium/discount noise and approximate durations.

The Δbp baseline matters (learned 2026-08-14, "DE 10Y +7bp" day):
  • US symbols: CNBC's previous_day_closing IS the US close — use it.
  • DE/UK/JP: CNBC's previous_day_closing is a late global snap taken
    after US hours, so Δ vs it misses any post-local-close move (showed
    +0.7bp on a +7bp day) — and the session open is a gappy stand-in
    (UK 30Y read +7.0 on a +8.6 day; JGBs print overnight so JP was
    overstated). Terminals measure vs yesterday's LOCAL cash close, so
    that is computed exactly from CNBC's 5D minute-bar history
    (ts-api.cnbc.com …/charts/5D.json): last bar at/before 16:30 London
    / 17:30 Berlin / 15:15 Tokyo of the prior session. Stateless, works
    even if the tab was closed all day; cached 30min per session.
    Fallbacks if the history call fails: session open, then CNBC's
    previous_day_closing.
  • 1w/1m/3m/YTD yield baselines come from the same charts API with
    range "1Y" (which actually serves ~2 years of daily bars).

CNBC endpoints are unofficial — if they ever break, the git history has
the ETF-proxy version as a fallback.
"""
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time as dtime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import ta as ta_lib

from vol_move import _rsi_cell          # shared RSI colour convention

# ── CTA positioning z column (Rajat 2026-08-28: "add slow-tilted vol-wtd
# 10y z next to RSI30"). Computed through the CTA engine's deep-history
# fetchers — the board's own hist caches are ~1y, far too short for a 10y
# window. CNBC yield codes map onto the CTA universe's tickers. ──────────────
_ENSZ_YIELD = {
    "US2Y": "^US2YT", "US5Y": "^FVX", "US10Y": "^TNX", "US30Y": "^TYX",
    "DE2Y-DE": "^ECB2Y", "DE5Y-DE": "^ECB5Y", "DE10Y-DE": "^ECB10Y",
    "DE30Y-DE": "^ECB30Y",
    "UK2Y-GB": "^UK2YT", "UK5Y-GB": "^UK5YT", "UK10Y-GB": "^UK10YT",
    "UK30Y-GB": "^UK30YT",
    "JP2Y-JP": "^JPY2Y", "JP5Y-JP": "^JPY5Y", "JP10Y-JP": "^JPY10Y",
    "JP30Y-JP": "^JPY30Y",
}


@st.cache_data(ttl=86400, show_spinner=False)
def _ens_z10(tkr: str, rates: bool):
    """Slow-tilted vol-wtd ensemble positioning z vs 10y — one value.
    years=15 matches the positioning tab's fetch tier → shared cache."""
    try:
        import cta
        h = cta._hist_signals(tkr, 126, 20, 200, 55, 63, years=15, rates=rates)
        if h.empty or len(h) < 900:
            return None
        w, vbs = cta._ENSEMBLE_WEIGHTS["Slow-tilted vol-wtd"]
        s = cta._ensemble_position(h["close"], 0.10, weights=w,
                                   vol_by_speed=vbs, arithmetic=rates)
        z = ((s - s.rolling(2520, min_periods=1260).mean())
             / s.rolling(2520, min_periods=1260).std().replace(0, np.nan)
             ).dropna()
        return float(z.iloc[-1]) if len(z) else None
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _ens_z10_sprd(tka: str, tkb: str):
    """Positioning z for a curve spread row (long-leg minus short-leg, bp)."""
    try:
        import cta
        ha = cta._hist_signals(tka, 126, 20, 200, 55, 63, years=15, rates=True)
        hb = cta._hist_signals(tkb, 126, 20, 200, 55, 63, years=15, rates=True)
        if ha.empty or hb.empty:
            return None
        sp = ((hb["close"] - ha["close"]) * 100.0).dropna()
        if len(sp) < 900:
            return None
        w, vbs = cta._ENSEMBLE_WEIGHTS["Slow-tilted vol-wtd"]
        s = cta._ensemble_position(sp, 0.10, weights=w, vol_by_speed=vbs,
                                   arithmetic=True)
        z = ((s - s.rolling(2520, min_periods=1260).mean())
             / s.rolling(2520, min_periods=1260).std().replace(0, np.nan)
             ).dropna()
        return float(z.iloc[-1]) if len(z) else None
    except Exception:
        return None


def _ensz_html(z):
    """CTA tab's Ens.Z colour convention: grey normal, amber ≥1σ, red ≥2σ."""
    try:
        from cta import _ensz_cell
        return _ensz_cell(z)
    except Exception:
        return "<span style='color:#CBD5E1'>—</span>"


def _ensz_for_row(kind: str, tkr: str):
    """Route a board row to its positioning z (None → em-dash cell)."""
    if kind == "cnbc":
        m = _ENSZ_YIELD.get(tkr)
        return _ens_z10(m, True) if m else None
    if kind == "sprd":
        a, b = tkr.split("|")
        ma, mb = _ENSZ_YIELD.get(a), _ENSZ_YIELD.get(b)
        return _ens_z10_sprd(ma, mb) if ma and mb else None
    if kind == "px":
        return _ens_z10(tkr, False)
    return None                       # synth (BBDXY) — no deep history

# ── Spec ─────────────────────────────────────────────────────────────────────
# (group, display name, ticker/symbol, kind)  kind: px = yfinance price row,
# cnbc = CNBC government bond yield row
_SPEC = [
    # ★ Watchlist — Rajat's top monitored assets, duplicated from the groups
    # below (2026-08-17). Purple shading normalises WITHIN this group only.
    ("★ Watchlist", "S&P 500",     "^GSPC",     "px"),
    ("★ Watchlist", "Nasdaq 100",  "^NDX",      "px"),
    ("★ Watchlist", "US 10Y",      "US10Y",     "cnbc"),
    ("★ Watchlist", "DE 10Y",      "DE10Y-DE",  "cnbc"),
    ("★ Watchlist", "JP 10Y",      "JP10Y-JP",  "cnbc"),
    ("★ Watchlist", "EUR/USD",     "EURUSD=X",  "px"),
    ("★ Watchlist", "USD/JPY",     "USDJPY=X",  "px"),
    ("★ Watchlist", "GBP/USD",     "GBPUSD=X",  "px"),
    ("★ Watchlist", "Gold",        "GC=F",      "px"),
    ("★ Watchlist", "Brent",       "BZ=F",      "px"),
    ("Equities", "S&P 500",        "^GSPC",     "px"),
    ("Equities", "Nasdaq 100",     "^NDX",      "px"),
    ("Equities", "Russell 2000",   "^RUT",      "px"),
    ("Equities", "SOX (semis)",    "^SOX",      "px"),
    # NB not ^SP500-4510 (S&P Software & Services index): Yahoo's daily
    # history for S&P sub-indices has month-long holes (seen 2026-08-14),
    # which poisons vol/horizon calcs. IGV tracks the same names cleanly.
    ("Equities", "Software (IGV)", "IGV",       "px"),
    ("Equities", "EuroStoxx 50",   "^STOXX50E", "px"),
    ("Equities", "FTSE 100",       "^FTSE",     "px"),
    ("Equities", "Nikkei 225",     "^N225",     "px"),
    ("Equities", "KOSPI",          "^KS11",     "px"),
    ("Equities", "Nifty 50",       "^NSEI",     "px"),
    ("FX",       "BBDXY",          "BBDXY_SYNTH", "synth"),
    ("FX",       "EUR/USD",        "EURUSD=X",  "px"),
    ("FX",       "USD/JPY",        "USDJPY=X",  "px"),
    ("FX",       "GBP/USD",        "GBPUSD=X",  "px"),
    ("FX",       "USD/CNH",        "USDCNH=X",  "px"),
    ("FX",       "USD/CAD",        "USDCAD=X",  "px"),
    ("FX",       "AUD/USD",        "AUDUSD=X",  "px"),
    ("FX",       "USD/INR",        "INR=X",     "px"),
    ("FX",       "EUR/GBP",        "EURGBP=X",  "px"),
    ("FX",       "USD/KRW",        "KRW=X",     "px"),
    ("Rates",    "US 2Y",          "US2Y",      "cnbc"),
    ("Rates",    "US 5Y",          "US5Y",      "cnbc"),
    ("Rates",    "US 10Y",         "US10Y",     "cnbc"),
    ("Rates",    "US 30Y",         "US30Y",     "cnbc"),
    ("Rates",    "DE 2Y",          "DE2Y-DE",   "cnbc"),
    ("Rates",    "DE 5Y",          "DE5Y-DE",   "cnbc"),
    ("Rates",    "DE 10Y",         "DE10Y-DE",  "cnbc"),
    ("Rates",    "DE 30Y",         "DE30Y-DE",  "cnbc"),
    ("Rates",    "UK 2Y",          "UK2Y-GB",   "cnbc"),
    ("Rates",    "UK 10Y",         "UK10Y-GB",  "cnbc"),
    ("Rates",    "UK 30Y",         "UK30Y-GB",  "cnbc"),
    ("Rates",    "JP 2Y",          "JP2Y-JP",   "cnbc"),
    ("Rates",    "JP 10Y",         "JP10Y-JP",  "cnbc"),
    ("Rates",    "JP 30Y",         "JP30Y-JP",  "cnbc"),
    # curve spreads backed out of the CNBC legs (long − short, in bp).
    # UK5Y-GB / JP5Y-JP are fetched as hidden legs — not shown as rows.
    ("Rates — Curves", "US 2s10s",       "US2Y|US10Y",         "sprd"),
    ("Rates — Curves", "US 5s30s",       "US5Y|US30Y",         "sprd"),
    ("Rates — Curves", "DE 2s10s",       "DE2Y-DE|DE10Y-DE",   "sprd"),
    ("Rates — Curves", "DE 5s30s",       "DE5Y-DE|DE30Y-DE",   "sprd"),
    ("Rates — Curves", "UK 2s10s",       "UK2Y-GB|UK10Y-GB",   "sprd"),
    ("Rates — Curves", "UK 5s30s",       "UK5Y-GB|UK30Y-GB",   "sprd"),
    ("Rates — Curves", "JP 2s10s",       "JP2Y-JP|JP10Y-JP",   "sprd"),
    ("Rates — Curves", "JP 5s30s",       "JP5Y-JP|JP30Y-JP",   "sprd"),
    ("Commodities", "Gold",        "GC=F",      "px"),
    ("Commodities", "Brent",       "BZ=F",      "px"),
    ("Commodities", "WTI",         "CL=F",      "px"),
    ("Commodities", "Silver",      "SI=F",      "px"),
    ("Commodities", "Copper",      "HG=F",      "px"),
    ("Crypto",   "Bitcoin",        "BTC-USD",   "px"),
]

# Futures fallback for equity indices outside cash hours (Rajat 2026-08-17):
# when the cash index is stale (>35min), the row switches to the CME future —
# level + 1d Δ from the future (tagged ·ES / ·NQ), longer horizons stay on
# cash closes. Only ES/NQ exist on Yahoo — FTSE/EuroStoxx/DAX/Nikkei futures
# (ICE/Eurex/OSE) aren't served, so those rows stay honest-closed. (The Vol
# Adj Move tab's Z=F/FESX=F/FDAX=F mappings are dead tickers — checked.)
_FUT_LIVE = {"^GSPC": "ES=F", "^NDX": "NQ=F", "^RUT": "RTY=F"}

# per-market timezone + local cash close (None = trust CNBC's prev close)
_MKT = {"US": ("America/New_York", None),
        "DE": ("Europe/Berlin", dtime(17, 30)),
        "GB": ("Europe/London", dtime(16, 30)),
        "JP": ("Asia/Tokyo", dtime(15, 15)),
        # FR/IT: no Core Markets rows (yet) — used by charting's country
        # yield series for correct local-date bucketing of the daily bars
        "FR": ("Europe/Paris", dtime(17, 30)),
        "IT": ("Europe/Rome", dtime(17, 30))}

_UP, _DN, _MUT = "#0D9488", "#DC2626", "#94A3B8"
_TD = "padding:4px 11px;font-size:12px;border-bottom:1px solid #E8EDF5;white-space:nowrap"
_TH = "padding:4px 11px;font-size:11px;background:#F8FAFC;font-weight:600;color:#475569;white-space:nowrap"
_GRP = ("padding:5px 11px;font-size:11px;background:#1E293B;color:#F8FAFC;"
        "font-weight:700;letter-spacing:0.4px")


# ── horizon baselines ────────────────────────────────────────────────────────
def _cuts(sess_d: date) -> dict:
    return {"1w":  sess_d - timedelta(days=7),
            "1m":  (pd.Timestamp(sess_d) - pd.DateOffset(months=1)).date(),
            "3m":  (pd.Timestamp(sess_d) - pd.DateOffset(months=3)).date(),
            "ytd": date(sess_d.year - 1, 12, 31)}


def _ref(hist, cut: date, max_stale: int = 14):
    """hist = [(date, value)] ascending → last value on/before cut.
    None if that value is >max_stale calendar days older than cut —
    protects against gappy Yahoo histories (e.g. ^SP500-4510's month-long
    hole) silently turning a 1w change into a 1m change."""
    if not hist:
        return None
    prior = [(d, v) for d, v in hist if d <= cut]
    if not prior:
        return None
    d, v = prior[-1]
    return v if (cut - d).days <= max_stale else None


# trading days per horizon for √-time vol scaling (Vol Adj Move convention)
_H_TDAYS = {"1d": 1, "3d": 3, "1w": 5, "1m": 21, "3m": 63}


def _daily_vol(hist, sess_d: date, is_bp: bool):
    """21d realized daily vol from the same history the refs use, in the
    same units as the displayed changes — % of price for px rows, bp of
    yield for rates rows. Excludes the current (partial) session. None if
    the history is too short."""
    if not hist:
        return None
    pts = [(d, v) for d, v in hist if d < sess_d][-22:]
    if len(pts) < 12:
        return None
    # only diff across gaps ≤5 calendar days — a data hole would otherwise
    # inject one huge fake "daily" move and wreck the σ
    diffs = [(v1 - v0) * 100 if is_bp else (v1 / v0 - 1) * 100
             for (d0, v0), (d1_, v1) in zip(pts[:-1], pts[1:])
             if (d1_ - d0).days <= 5]
    if len(diffs) < 8:
        return None
    # degenerate-history guard (KOSPI "69σ" incident 2026-08-20: a glitched
    # flat/stale Yahoo daily frame cached for 6h → σ≈0 → any normal move
    # shows as an absurd ratio). A near-flat series is bad DATA, not low
    # vol — suppress σ entirely rather than emit nonsense ratios.
    if sum(1 for x in diffs if x != 0) < 5:
        return None
    sd = float(np.std(np.asarray(diffs), ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return sd


def _h_days(h: str, sess_d: date) -> int:
    if h != "ytd":
        return _H_TDAYS[h]
    return max(1, int(np.busday_count(date(sess_d.year - 1, 12, 31), sess_d)))


def _rsi_vals(hist, sess_d: date) -> tuple:
    """(RSI14, RSI30) Wilder RSI on daily closes (yield levels for rates),
    excluding the current partial session — Vol Adj Move convention."""
    if not hist:
        return None, None
    vals = [v for d, v in hist if d < sess_d]
    s = pd.Series(vals, dtype=float)
    out = []
    for w in (14, 30):
        try:
            if len(s) >= w + 1:
                r = ta_lib.momentum.RSIIndicator(s, window=w).rsi().dropna()
                out.append(float(r.iloc[-1]) if not r.empty else None)
            else:
                out.append(None)
        except Exception:
            out.append(None)
    return tuple(out)


# ── yfinance price rows ──────────────────────────────────────────────────────
def _one(tkr: str):
    """(last_px, prev_session_close, last_bar_ts) from 5 days of 5m bars."""
    import yfinance as yf
    h = yf.Ticker(tkr).history(period="5d", interval="5m",
                               auto_adjust=False, prepost=False)
    c = h["Close"].dropna()
    if c.empty:
        raise ValueError("no bars")
    ts = c.index[-1]
    d0 = ts.date()
    prev = c[[i.date() < d0 for i in c.index]]
    prev_close = float(prev.iloc[-1]) if not prev.empty else None
    return float(c.iloc[-1]), prev_close, ts


def _px_row(tkr: str):
    """_one() with one retry, then a daily-history fallback (last two
    closes) so a transient yfinance hiccup (e.g. FTSE 2026-08-14) degrades
    to a stale-dotted row instead of 'no data'."""
    for _ in range(2):
        try:
            return _one(tkr)
        except Exception:
            continue
    hist = _px_hist(tkr)
    if hist and len(hist) >= 2:
        d, v = hist[-1]
        return v, hist[-2][1], pd.Timestamp(d, tz="UTC")
    raise ValueError("no data")


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_px(tickers: tuple) -> dict:
    """{ticker: (last, prev_close, ts_isoformat)} — threaded, 60s cache."""
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_px_row, t): t for t in tickers}
        for f in as_completed(futs):
            t = futs[f]
            try:
                last, prev, ts = f.result()
                out[t] = (last, prev, ts.isoformat())
            except Exception:
                out[t] = None
    return out


# NB every cached fetcher below RAISES on failure and is wrapped by a safe
# accessor: st.cache_data never caches exceptions, so a transient feed
# hiccup retries on the next render instead of pinning None in the cache
# for the ttl (the "US 5s30s blank" bug, 2026-08-17 — same anti-poison rule
# as the Vol Dash caches).
@st.cache_data(ttl=86400, show_spinner=False)
def _px_hist_c(tkr: str):
    if tkr == "USDCNH=X":                      # no Yahoo daily history
        from watchlist import fetch_chart_data
        c = fetch_chart_data(tkr, "2y", "1d")["Close"]
    else:
        import yfinance as yf
        c = yf.download(tkr, period="2y", interval="1d",
                        progress=False, auto_adjust=True)["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = c.dropna()
    if c.empty:
        raise ValueError(f"no daily history for {tkr}")
    return [(i.date() if hasattr(i, "date") else i, float(v))
            for i, v in c.items()]


def _px_hist(tkr: str):
    """[(date, close)] — 2y of daily closes for the horizon baselines."""
    try:
        return _px_hist_c(tkr)
    except Exception:
        return None


# ── FX prev-close baseline ───────────────────────────────────────────────────
# FX trades ~24h, so _one()'s "last 5m bar before today's date" baseline is a
# MIDNIGHT-UTC snapshot, not a close — any move between the daily close
# (~21-22:00 UTC) and midnight silently vanished from the 1d change, and the
# synth BBDXY live level (base × weighted 1d) understated multi-day moves
# with it ("dollar fell 3 days but BBDXY doesn't show it", 2026-08-20).
# Baseline for every FX row is therefore the last COMPLETED daily close from
# the pairs' own daily candles (NY-close convention, same basis as the
# horizon columns). One batched 10d pull, anti-poison cached.
@st.cache_data(ttl=3600, show_spinner=False)
def _fx_prevcls_c(tkrs: tuple) -> dict:
    """{fx_ticker: (last_completed_daily_close, close_date)}."""
    import yfinance as yf
    today = datetime.now(timezone.utc).date()
    out = {}
    ytk = [t for t in tkrs if t != "USDCNH=X"]
    df = yf.download(ytk, period="10d", interval="1d",
                     progress=False, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=ytk[0])
    for t in ytk:
        if t in df.columns:
            s = df[t].dropna()
            s = s[[i.date() < today for i in s.index]]
            if not s.empty:
                out[t] = (float(s.iloc[-1]), s.index[-1].date())
    if "USDCNH=X" in tkrs:                     # AlphaVantage-served history
        h = _px_hist("USDCNH=X")
        if h:
            hh = [(d0, v) for d0, v in h if d0 < today]
            if hh:
                out["USDCNH=X"] = (hh[-1][1], hh[-1][0])
    if not out:
        raise ValueError("fx prev-close batch empty")
    return out


def _fx_prevcls(tkrs: tuple) -> dict:
    try:
        return _fx_prevcls_c(tkrs)
    except Exception:
        return {}


# ── synthetic BBDXY row ──────────────────────────────────────────────────────
# Daily history = watchlist's BBDXY_SYNTH (geometric 12-pair basket, base
# 1200 at window start — the LEVEL is window-relative, changes are the
# signal). Live level/1d: same weighted Δlog computed from the component
# pairs' live 5m quotes, weights renormalised over the fresh components
# (≥8 of 12 required), applied to the last completed daily close.
@st.cache_data(ttl=86400, show_spinner=False)
def _bbdxy_hist_c():
    from watchlist import fetch_chart_data
    c = fetch_chart_data("BBDXY_SYNTH", "2y", "1d")["Close"].dropna()
    if c.empty:
        raise ValueError("no BBDXY_SYNTH history")
    return [(i.date() if hasattr(i, "date") else i, float(v))
            for i, v in c.items()]


def _bbdxy_hist():
    try:
        return _bbdxy_hist_c()
    except Exception:
        return None


def _bbdxy_live(px: dict, fxpc: dict):
    """(live_level, pct_1d, stalest_ts, sess_date) from component quotes +
    the daily hist, or (last_close, None, None, last_date) fallback.
    Per-pair baseline = completed daily close (fxpc) so the 1d matches the
    daily-close basis of hist; px's midnight prev only if the batch failed."""
    from watchlist import BBDXY_WEIGHTS
    hist = _bbdxy_hist()
    if not hist:
        return None
    dlogs, w_used, tss = [], 0.0, []
    for tkr, (w, invert) in BBDXY_WEIGHTS.items():
        d = px.get(tkr)
        if not d or not d[0]:
            continue
        last, prev, ts_iso = d
        if tkr in fxpc:
            prev = fxpc[tkr][0]
        elif fxpc:          # batch alive but this pair missing → skip the
            continue        # leg rather than mix baselines
        if not prev:
            continue
        dlogs.append((w, (-1 if invert else 1) * np.log(last / prev)))
        w_used += w
        tss.append(pd.Timestamp(ts_iso))
    if len(dlogs) < 8:                     # too few fresh legs → stale close
        d0, v0 = hist[-1]
        return v0, None, None, d0
    sess_d = max(t.date() for t in tss)
    base = [v for d0, v in hist if d0 < sess_d]
    if not base:
        return None
    pct = (np.exp(sum(w * dl for w, dl in dlogs) / w_used) - 1) * 100
    return base[-1] * (1 + pct / 100), pct, min(tss), sess_d


# ── CNBC yield rows ──────────────────────────────────────────────────────────
def _pnum(s):
    try:
        return float(str(s).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _cnbc_yields(symbols: tuple) -> dict:
    url = ("https://quote.cnbc.com/quote-html-webservice/restQuote/"
           "symbolType/symbol?symbols=" + urllib.parse.quote("|".join(symbols))
           + "&requestMethod=itv&noform=1&partnerId=2&fund=1&exthrs=1"
             "&output=json")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read().decode())
    out = {}
    for q in d.get("FormattedQuoteResult", {}).get("FormattedQuote", []):
        out[q.get("symbol")] = {
            "last": _pnum(q.get("last")),
            "open": _pnum(q.get("open")),
            "prevcls": _pnum(q.get("previous_day_closing")),
            "ts": q.get("last_time") or ""}
    return out


def _mkt_of(sym: str) -> tuple:
    cc = sym.split("-")[1] if "-" in sym else "US"
    return _MKT.get(cc, _MKT["US"])


def _cnbc_bars(sym: str, rng: str):
    """[(local_ts, close)] from the CNBC charts API."""
    tzname, _ = _mkt_of(sym)
    url = f"https://ts-api.cnbc.com/harmony/app/charts/{rng}.json?symbol={sym}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        bars = json.loads(r.read().decode())["barData"]["priceBars"]
    return [(pd.Timestamp(float(b["tradeTimeinMills"]), unit="ms",
                          tz="UTC").tz_convert(tzname), float(b["close"]))
            for b in bars]


@st.cache_data(ttl=300, show_spinner=False)
def _bars_5d_c(sym: str):
    rows = _cnbc_bars(sym, "5D")           # raises on HTTP/parse failure
    if not rows:
        raise ValueError(f"no 5D bars for {sym}")
    return rows


def _bars_5d(sym: str):
    """Cached 5D minute bars — shared by the 1d baseline and the
    missing-quote fallback. None on failure (failures are NOT cached)."""
    try:
        return _bars_5d_c(sym)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _plc_c(sym: str, sess: str, back: int):
    tzname, close_t = _mkt_of(sym)
    rows = _bars_5d_c(sym)                 # raise → this call isn't cached
    sess_d = pd.Timestamp(sess).date()
    days = sorted({t.date() for t, _v in rows if t.date() < sess_d})
    if len(days) < back:
        raise ValueError(f"only {len(days)} prior sessions for {sym}")
    prevd = days[-back]
    day = [(t, v) for t, v in rows if t.date() == prevd]
    at_close = [(t, v) for t, v in day if t.time() <= close_t]
    return (at_close or day)[-1][1]


def _prev_local_close(sym: str, sess: str, back: int = 1):
    """Local cash-close yield `back` sessions before `sess`, from CNBC 5D
    minute bars — the terminal-convention Δbp baseline (back=1 → 1d,
    back=3 → 3d). `sess` keys the cache so it rolls daily. None on any
    failure (failures are NOT cached)."""
    try:
        return _plc_c(sym, sess, back)
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _yld_hist_c(sym: str):
    rows = _cnbc_bars(sym, "1Y")           # raises on HTTP/parse failure
    out = {}
    for t, v in rows:                      # keep last print per local date
        out[t.date()] = v
    if not out:
        raise ValueError(f"empty 1Y history for {sym}")
    return sorted(out.items())


def _yld_hist(sym: str):
    """[(date, close)] daily — CNBC range '1Y' actually serves ~2 years.
    None on failure (failures are NOT cached)."""
    try:
        return _yld_hist_c(sym)
    except Exception:
        return None


def _sprd_hist(a: str, b: str):
    """[(date, long − short in % pts)] on the legs' common dates — feeds the
    horizon/σ/RSI machinery in the same units convention as yield rows
    (values in %, displayed ×100 as bp)."""
    da = dict(_yld_hist(a) or [])
    return [(d, vb - da[d]) for d, vb in (_yld_hist(b) or []) if d in da]


def _yld_rows(quotes: dict, syms: tuple) -> dict:
    """{sym: (last, dbp_1d_or_None, ts_or_None, sess_date)}. When the quote
    endpoint drops a symbol from the batch (happens transiently), the row
    is synthesized from the latest 5D minute bar instead — level, Δ and
    history columns all still work, the as-of dot just shows the bar age."""
    out = {}
    for sym in syms:
        q = quotes.get(sym) or {}
        last = q.get("last")
        tzname, close_t = _mkt_of(sym)
        ts = None
        if q.get("ts"):
            try:
                ts = pd.Timestamp(q["ts"])
                if ts.tzinfo is None:          # date-only stamps (stale JGBs)
                    ts = ts.tz_localize(tzname)
            except Exception:
                ts = None
        if last is None:                       # quote missing → bar fallback
            bars = _bars_5d(sym)
            if bars:
                ts, last = bars[-1]
        if last is None:
            out[sym] = None
            continue
        sess_d = (ts.tz_convert(tzname).date() if ts is not None
                  else pd.Timestamp.now(tz=tzname).date())
        ref3 = None
        if close_t is None:                    # US: CNBC prev close is right
            base = q.get("prevcls")
        else:
            base = (_prev_local_close(sym, str(sess_d))
                    or q.get("open") or q.get("prevcls"))
            ref3 = _prev_local_close(sym, str(sess_d), 3)
        if ref3 is None:                       # US rows / thin 5D windows
            pts = [(d0, v) for d0, v in (_yld_hist(sym) or [])
                   if d0 < sess_d]
            if len(pts) >= 3 and (sess_d - pts[-3][0]).days <= 10:
                ref3 = pts[-3][1]
        dbp = (last - base) * 100 if base else None
        d3 = (last - ref3) * 100 if ref3 else None
        out[sym] = (last, dbp, d3,
                    ts.isoformat() if ts is not None else None, sess_d)
    return out


# ── quick-chart panel (Rajat 2026-08-17: tick assets → chart on the right,
# window dropdown 1m/3m/6m/1y/2y default 6m, RSI-14 panel under the main
# chart). Asset toggles are st.pills chips — Streamlit widgets can't be
# embedded/aligned inside the HTML table rows (house lesson), so the chip
# grid lives at the top of the panel instead. ────────────────────────────────
_CH_TF = {"1m": 30, "3m": 91, "6m": 182, "1y": 365, "2y": 730}
_CH_PAL = ["#2563EB", "#0D9488", "#B45309", "#7C3AED", "#DC2626", "#64748B"]


def _rsi_series(vals, n: int = 14) -> pd.Series:
    s = pd.Series(list(vals), dtype=float)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _row_hist(tkr: str, kind: str):
    """Daily [(date, value)] for any board row — px in price, cnbc in %,
    sprd in bp."""
    if kind == "px":
        return _px_hist(tkr)
    if kind == "cnbc":
        return _yld_hist(tkr)
    if kind == "sprd":
        a, b = tkr.split("|")
        return [(d, v * 100) for d, v in _sprd_hist(a, b)]
    if kind == "synth":
        return _bbdxy_hist()
    return None


def _chart_rows() -> dict:
    """{display name: (tkr, kind)} — unique names in spec/table order
    (★ Watchlist duplicates collapse onto the same underlying data)."""
    rows = {}
    for _g, name, tkr, kind in _SPEC:
        if kind in ("px", "cnbc", "sprd", "synth") and name not in rows:
            rows[name] = (tkr, kind)
    return rows


def _chart_panel(sel):
    import plotly.graph_objects as go
    st.markdown("**📊 Quick charts**")
    tf = st.selectbox("Window", list(_CH_TF), index=2, key="_cm_ch_tf")
    if not sel:
        st.caption("click a ☐ box in the board's first column — level "
                   "chart on top, RSI-14 below (click ☑ again to clear).")
        return
    rows = _chart_rows()
    cutoff = date.today() - timedelta(days=_CH_TF[tf])
    tkr, kind = rows[sel]
    hist = _row_hist(tkr, kind)
    if not hist:
        st.caption(f"⚠ {sel}: no history")
        return
    w = [(d, v) for d, v in hist if d >= cutoff]
    if len(w) < 2:
        st.caption(f"⚠ {sel}: not enough data in window")
        return
    col = _CH_PAL[0]
    fig = go.Figure(go.Scatter(
        x=[d for d, _v in w], y=[v for _d, v in w], mode="lines", name=sel,
        line=dict(color=col, width=3.5),
        hovertemplate="%{x|%d %b %y} · %{y:,.4g}<extra>" + sel + "</extra>"))
    unit = {"px": "", "cnbc": "%", "sprd": "bp", "synth": ""}[kind]
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=28, b=10), showlegend=False,
        plot_bgcolor="#FFFFFF", xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9", title=unit),
        title=dict(text=f"{sel} — {tf}", font=dict(size=13)),
        hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # RSI on the full history for warm-up, displayed over the window
    rfig = go.Figure()
    full = pd.Series([v for _d, v in hist], index=[d for d, _v in hist])
    rs = _rsi_series(full.values)
    rs.index = full.index
    rw = rs[rs.index >= cutoff]
    rfig.add_trace(go.Scatter(
        x=list(rw.index), y=list(rw.values), mode="lines", name=sel,
        line=dict(color=col, width=1.4),
        hovertemplate="%{x|%d %b %y} · %{y:.1f}<extra>" + sel
                      + " RSI14</extra>"))
    for lv, cl, dsh in ((70, "#DC2626", "dot"), (50, "#CBD5E1", None),
                        (30, "#0D9488", "dot")):
        rfig.add_hline(y=lv, line_color=cl, line_width=1, line_dash=dsh)
    rfig.update_layout(
        height=170, margin=dict(l=10, r=10, t=6, b=10),
        showlegend=False, plot_bgcolor="#FFFFFF",
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(range=[0, 100], tickvals=[30, 50, 70], title="RSI14"),
        hovermode="x unified")
    st.plotly_chart(rfig, use_container_width=True)


# ── render ───────────────────────────────────────────────────────────────────
def _fmt_px(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.4g}" if v < 10 else f"{v:,.2f}"


def _sig_html(ratio) -> str:
    """Small σ-ratio line under a change value — Vol Adj Move colouring:
    amber ≥1σ, red ≥2σ, muted below."""
    if ratio is None:
        return ""
    a = abs(ratio)
    col = _DN if a >= 2 else ("#D97706" if a >= 1 else _MUT)
    w = 700 if a >= 1 else 400
    return (f"<br><span style='font-size:10px;color:{col};"
            f"font-weight:{w}'>{ratio:+.1f}σ</span>")


def _cell(chg_str, col, sig=""):
    # __BG__ is replaced per-row with the shade colour (Earnings-tab style)
    return (f"<td style='{_TD};text-align:right;color:{col};"
            f"font-weight:600;background:__BG__'>{chg_str}{sig}</td>")


def _chg_cells(last, hist, sess_d: date, is_bp: bool, d1, d3=None):
    """1d/3d + 1w/1m/3m/YTD change cells, each with the move/1σ ratio
    under it (σ = 21d realized daily vol × √horizon — Vol Adj Move
    convention). d1/d3 = precomputed 1d and 3d changes (bp or %) or None.
    Returns (cells_html, vol_d, {horizon: ratio}) — ratios feed the
    row-shading feature."""
    vol_d = _daily_vol(hist, sess_d, is_bp)
    ratios = {}

    # circuit-breaker: |move/σ| beyond this is data error, not markets —
    # even Mar-2020 style days print low double digits on this metric.
    # Suspect ratios show an amber ⚠ and are EXCLUDED from row shading.
    _R_MAX = 12
    _SUSPECT = ("<br><span style='font-size:10px;color:#D97706;"
                "font-weight:700' title='move/σ implausible (>12) — "
                "history or quote data suspect, ratio suppressed'>"
                "⚠ σ?</span>")

    def ratio_of(dval, h):
        if vol_d is None or dval is None or vol_d <= 0:
            return None
        return dval / (vol_d * np.sqrt(_h_days(h, sess_d)))

    def _sig_or_flag(r):
        if r is not None and abs(r) > _R_MAX:
            return None, _SUSPECT
        return r, _sig_html(r)

    cells = ""
    for h, dv in (("1d", d1), ("3d", d3)):
        r, sig = _sig_or_flag(ratio_of(dv, h))
        ratios[h] = r
        if dv is not None:
            cells += _cell(f"{dv:+.1f}bp" if is_bp else f"{dv:+.2f}%",
                           _UP if dv >= 0 else _DN, sig)
        else:
            cells += _cell("—", _MUT)
    cuts = _cuts(sess_d)
    for h in ("1w", "1m", "3m", "ytd"):
        ref = _ref(hist, cuts[h])
        if ref:
            if is_bp:
                d = (last - ref) * 100
                s = f"{d:+.1f}bp" if h == "1w" else f"{d:+.0f}bp"
            else:
                d = (last / ref - 1) * 100
                s = f"{d:+.1f}%"
            r, sig = _sig_or_flag(ratio_of(d, h))
            ratios[h] = r
            cells += _cell(s, _UP if d >= 0 else _DN, sig)
        else:
            ratios[h] = None
            cells += _cell("—", _MUT)
    return cells, vol_d, ratios


def render_core_markets():
    st.markdown(
        "<div style='background:#1E293B;color:#F8FAFC;padding:6px 12px;"
        "font-size:13px;font-weight:700;border-radius:6px;display:inline-block;"
        "margin-bottom:6px'>🧭 Core Markets"
        "&nbsp;&nbsp;<span style='font-weight:400;font-size:11px;color:#94A3B8'>"
        "live 1-day board — prices yfinance (~15-20min delay), yields CNBC "
        "(near-real-time), 60s cache</span></div>", unsafe_allow_html=True)

    c1, c2, _sp = st.columns([1, 1.6, 4.4])
    if c1.button("🔄 Refresh", key="_cm_refresh", use_container_width=True,
                 help="clears quotes AND cached daily histories — use this "
                      "if any σ/horizon figures look wrong"):
        _fetch_px.clear()
        _cnbc_yields.clear()
        # histories too: a glitched Yahoo daily frame otherwise stays
        # pinned for 6h and poisons σ/horizons (KOSPI 69σ, 2026-08-20)
        _px_hist_c.clear()
        _yld_hist_c.clear()
        _bbdxy_hist_c.clear()
        _fx_prevcls_c.clear()
        _bars_5d_c.clear()
    shade_h = c2.selectbox(
        "Shade rows by |move/σ|", ["1d", "3d", "1w", "1m", "3m", "ytd"],
        index=2, key="_cm_shade",
        help="rows shade purple by absolute vol-adjusted move over the "
             "chosen horizon — darkest = biggest |move ÷ σ| on the board")

    # dict.fromkeys dedupes while keeping order (★ Watchlist repeats tickers)
    px_tkrs = tuple(dict.fromkeys(s[2] for s in _SPEC if s[3] == "px"))
    yl_syms = tuple(dict.fromkeys(s[2] for s in _SPEC if s[3] == "cnbc"))
    _legs = {l for s in _SPEC if s[3] == "sprd" for l in s[2].split("|")}
    yl_syms = yl_syms + tuple(sorted(_legs - set(yl_syms)))
    from watchlist import BBDXY_WEIGHTS
    _extra = tuple(sorted(set(_FUT_LIVE.values()) | set(BBDXY_WEIGHTS)
                          - set(px_tkrs)))
    px = _fetch_px(px_tkrs + _extra)
    fxpc = _fx_prevcls(tuple(sorted(
        {t for t in px_tkrs if t.endswith("=X")} | set(BBDXY_WEIGHTS))))
    try:
        try:
            quotes = _cnbc_yields(yl_syms)
        except Exception:
            quotes = {}                # bar fallback can still fill rows
        ylds = _yld_rows(quotes, yl_syms)
    except Exception as ex:
        ylds = {}
        st.caption(f"⚠ CNBC yield feed failed ({type(ex).__name__}) — "
                   "rates rows unavailable this refresh")
    now = datetime.now(timezone.utc)

    # pass 1 — build each row's html (with __BG__ placeholder) + σ ratios
    recs = []
    for grp, name, tkr, kind in _SPEC:
        lvl = None
        ts = None
        ratios = {}
        if kind == "cnbc":
            d = ylds.get(tkr)
            if d:
                last, dbp, d3, ts_iso, sess_d = d
                lvl = f"{last:.3f}%"
                hist = _yld_hist(tkr)
                # consistency guard (yields): live >1.5pp from the cached
                # history's last value = bad history frame — drop it
                if hist and abs(last - hist[-1][1]) > 1.5:
                    hist = None
                cells, vol_d, ratios = _chg_cells(last, hist, sess_d,
                                                  True, dbp, d3)
                vol_s = f"{vol_d:.1f}bp" if vol_d else "—"
                rsi14, rsi30 = _rsi_vals(hist, sess_d)
                ts = pd.Timestamp(ts_iso) if ts_iso else None
        elif kind == "sprd":
            a, b = tkr.split("|")
            da_, db_ = ylds.get(a), ylds.get(b)
            if da_ and db_:
                la, d1a, d3a, tsa, sda = da_
                lb, d1b, d3b, tsb, sdb = db_
                last = lb - la                      # % pts, shown as bp
                dbp = (d1b - d1a if d1a is not None and d1b is not None
                       else None)
                d3 = (d3b - d3a if d3a is not None and d3b is not None
                      else None)
                hist = _sprd_hist(a, b)
                sess_d = max(sda, sdb)
                cells, vol_d, ratios = _chg_cells(last, hist, sess_d,
                                                  True, dbp, d3)
                lvl = f"{last * 100:.1f}bp"
                vol_s = f"{vol_d:.1f}bp" if vol_d else "—"
                rsi14, rsi30 = _rsi_vals(hist, sess_d)
                _tt = [pd.Timestamp(t) for t in (tsa, tsb) if t]
                ts = min(_tt) if _tt else None      # staler leg = honesty
        elif kind == "synth":
            r = _bbdxy_live(px, fxpc)
            if r:
                last, pct, ts, sess_d = r
                hist = _bbdxy_hist()
                pts = [(d0, v) for d0, v in (hist or []) if d0 < sess_d]
                d3 = None
                if len(pts) >= 3 and (sess_d - pts[-3][0]).days <= 10 \
                        and pts[-3][1]:
                    d3 = (last / pts[-3][1] - 1) * 100
                cells, vol_d, ratios = _chg_cells(last, hist, sess_d,
                                                  False, pct, d3)
                lvl = f"{last:,.1f}"
                vol_s = f"{vol_d:.2f}%" if vol_d else "—"
                rsi14, rsi30 = _rsi_vals(hist, sess_d)
        else:
            d = px.get(tkr)
            if d:
                last, prev, ts_iso = d
                if tkr.endswith("=X") and tkr in fxpc:
                    prev = fxpc[tkr][0]     # daily-close baseline (see
                ts = pd.Timestamp(ts_iso)   # _fx_prevcls_c) — midnight-UTC
                pct = (last / prev - 1) * 100 if prev else None
                disp_last = last
                fut_note = ""
                # cash stale + mapped future fresh → live level/1d off the
                # future; horizons/σ/RSI stay on cash closes (basis-safe)
                age = (now - ts.tz_convert("UTC")).total_seconds() / 60
                if age > 35 and tkr in _FUT_LIVE:
                    fd = px.get(_FUT_LIVE[tkr])
                    if fd and fd[1]:
                        flast, fprev, fts_iso = fd
                        fts = pd.Timestamp(fts_iso)
                        fage = (now - fts.tz_convert("UTC")
                                ).total_seconds() / 60
                        if fage <= 35:
                            disp_last, ts = flast, fts
                            pct = (flast / fprev - 1) * 100
                            fut_note = (" <span style='color:#94A3B8;font-"
                                        "size:10px'>·"
                                        + _FUT_LIVE[tkr].replace("=F", "")
                                        + "</span>")
                lvl = _fmt_px(disp_last) + fut_note
                hist = _px_hist(tkr)
                pts = [(d0, v) for d0, v in (hist or []) if d0 < ts.date()]
                # consistency guard: if the live quote is wildly off the
                # cached history's last close (>35%), the HISTORY is
                # garbage (scale glitch / stale frame) — drop it so the
                # horizon/σ/RSI columns show "—" instead of nonsense
                if pts and last and abs(last / pts[-1][1] - 1) > 0.35:
                    hist, pts = None, []
                d3 = None
                if len(pts) >= 3 and (ts.date() - pts[-3][0]).days <= 10 \
                        and pts[-3][1]:
                    d3 = (last / pts[-3][1] - 1) * 100
                cells, vol_d, ratios = _chg_cells(last, hist, ts.date(),
                                                  False, pct, d3)
                vol_s = f"{vol_d:.2f}%" if vol_d else "—"
                rsi14, rsi30 = _rsi_vals(hist, ts.date())

        if lvl is None:
            row = (f"<tr><td style='{_TD}'></td>"
                   f"<td style='{_TD}'>{name}</td><td style='{_TD};"
                   f"color:{_MUT}' colspan='12'>no data ({tkr})</td></tr>")
            recs.append((grp, row, {}))
            continue
        if ts is not None:
            ts_uk = ts.tz_convert("Europe/London")
            age_min = (now - ts.tz_convert("UTC")).total_seconds() / 60
            dot = ("<span style='color:#16A34A'>●</span>" if age_min <= 35
                   else f"<span style='color:{_MUT}'>○</span>")
            asof = f"{dot} {ts_uk.strftime('%H:%M')}"
        else:
            asof = f"<span style='color:{_MUT}'>○ —</span>"
        row = (f"<tr><td style='{_TD};background:__BG__;text-align:center;"
               f"color:#7C3AED;font-weight:700' data-key='{name}'>☐</td>"
               f"<td style='{_TD};background:__BG__'>{name}</td>"
               f"<td style='{_TD};text-align:right;background:__BG__'>{lvl}"
               f"</td>"
               f"<td style='{_TD};text-align:right;color:#64748B;"
               f"background:__BG__'>{vol_s}</td>{cells}"
               f"<td style='{_TD};text-align:right;background:__BG__'>"
               f"{_rsi_cell(rsi14)}</td>"
               f"<td style='{_TD};text-align:right;background:__BG__'>"
               f"{_rsi_cell(rsi30)}</td>"
               f"<td style='{_TD};text-align:right;background:__BG__'>"
               f"{_ensz_html(_ensz_for_row(kind, tkr))}</td>"
               f"<td style='{_TD};color:#64748B;background:__BG__'>{asof}"
               f"</td></tr>")
        recs.append((grp, row, ratios))

    # pass 2 — shade rows purple by |ratio| of the chosen horizon.
    # TWO normalisation pools (Rajat 2026-08-17): the ★ Watchlist group
    # scales against its own biggest mover; everything else against the
    # rest of the board — so the watchlist reads as its own heat block.
    _SHADE_RGB, _SHADE_AMAX = "124,58,237", 0.30      # #7C3AED, Earnings-style

    def _pool_max(pred):
        vals = [abs(r[2][shade_h]) for r in recs
                if pred(r[0]) and r[2].get(shade_h) is not None]
        return max(vals) if vals else None

    _wl = "★ Watchlist"
    rmax_by_pool = {True: _pool_max(lambda g: g == _wl),
                    False: _pool_max(lambda g: g != _wl)}
    hdr_row = f"<tr><th style='{_TH}'>📊</th>" \
              f"<th style='{_TH}'>Instrument</th>" + "".join(
        f"<th style='{_TH};text-align:right'>{h}</th>"
        for h in ("Last", "σ/day", "1d Δ", "3d", "1w", "1m", "3m", "YTD",
                  "RSI14", "RSI30", "CTAz")) + \
        f"<th style='{_TH}'>as of (UK)</th></tr>"
    html = "<table style='border-collapse:collapse'>"
    grp_seen = None
    for grp, row, ratios in recs:
        if grp != grp_seen:
            # repeat the column header under every group band so the
            # columns stay identifiable when scrolled deep into the table
            html += f"<tr><td colspan='14' style='{_GRP}'>{grp}</td></tr>"
            html += hdr_row
            grp_seen = grp
        a = 0.0
        rmax = rmax_by_pool[grp == _wl]
        if rmax and ratios.get(shade_h) is not None:
            a = _SHADE_AMAX * abs(ratios[shade_h]) / rmax
        bg = f"rgba({_SHADE_RGB},{a:.3f})" if a > 0.004 else "transparent"
        html += row.replace("__BG__", bg)
    html += "</table>"
    html = f"<div style='overflow-x:auto'>{html}</div>"
    html += ("<p style='font-size:10px;color:#94A3B8;margin-top:6px'>"
             "Prices: 1d = last 5m yfinance bar vs previous session close. "
             "S&P/Nasdaq switch to their CME futures (·ES/·NQ tag) when the "
             "cash session is closed — level and 1d Δ come from the future "
             "then; 1w+/σ/RSI always stay on cash closes. Other equity "
             "indices have no Yahoo-served futures and show their last "
             "cash close off-hours. "
             "1w/1m/3m/YTD vs daily closes (1w = 7 calendar days back, "
             "1m/3m = calendar months, YTD = last close of prior year; "
             "CNH history via AlphaVantage). BBDXY = synthetic Bloomberg "
             "dollar index (geometric 12-pair basket, base 1200 at window "
             "start — the level is window-relative, read the changes); live "
             "level/1d computed from the component pairs' live quotes, "
             "weights renormalised over fresh legs; freshness dot = stalest "
             "leg. FX 1d (pairs and BBDXY) vs the last COMPLETED daily "
             "close (NY-close convention, same basis as the horizon "
             "columns) — not the midnight-UTC snapshot. "
             "Yields: CNBC/Refinitiv feed; "
             "1d Δbp vs previous US close for US and vs previous LOCAL cash "
             "close for DE/UK/JP (last print ≤17:30 Berlin / 16:30 London / "
             "15:15 Tokyo from CNBC minute-bar history — terminal "
             "convention); longer horizons vs CNBC daily history. "
             "Curves: 2s10s = 10Y − 2Y, 5s30s = 30Y − 5Y (level and all Δ in "
             "bp; positive Δ = steepening; UK/JP 5Y legs fetched from the "
             "same feed but not shown as rows; freshness dot = the staler "
             "leg). σ/day = 21d realized daily vol (% for prices, bp of "
             "yield/spread for rates); the small figure under each change "
             "is move ÷ expected "
             "1σ for that horizon (σ·√days, Vol Adj Move convention — YTD "
             "uses elapsed trading days): <span style='color:#D97706'>"
             "≥1σ</span>, <span style='color:#DC2626'>≥2σ</span>. "
             "**CTAz** = simulated trend-follower positioning z vs 10y "
             "(Slow-tilted vol-wtd ensemble, same engine as Macro ▸ CTA "
             "Positioning; amber ≥1σ, red ≥2σ; sign = trend direction, "
             "rates in yield terms; computed once daily — first board load "
             "of the day is slower).  ·  "
             "RSI14/RSI30 = Wilder RSI on daily closes excl. the current "
             "session — for rates it runs on the YIELD, so red ≥70 means "
             "yields rich/overbought (bonds sold off). Row shading: purple "
             "depth = |move ÷ σ| for the horizon picked above — the "
             "★ Watchlist block normalises to its own biggest mover, all "
             "other groups to the biggest mover in the rest of the board. "
             "● = quote &lt;35min old; ○ = market closed / stale. JP 2Y "
             "updates infrequently on this feed.</p>")
    # board (clickable ☐ column, in-table — click_table component gives
    # perfect row alignment since the selector IS a table cell) | charts
    from click_table import click_table
    mid, right = st.columns([1.35, 1], gap="small")
    with mid:
        sel = click_table(html, selected=st.session_state.get("_cm_tbl"),
                          key="_cm_tbl")
    with right:
        _chart_panel(sel)

"""Scenario P&L for the Risk tab (Rajat 2026-09-04, built on NFP day).

User shocks a few anchor factors ("SPX −2%, US10y −12bp, EUR +0.8%"); every
other factor in the book either stays flat or takes its CONDITIONAL move
implied by the proxy-correlation framework (multivariate-normal conditional
mean E[y|x] = Σ_yx Σ_xx⁻¹ x over the chosen VaR window). Position P&L:

- futures/FX — the same linear mappings the VaR standalone uses
  (Rates: qty × mult × 0.01 × fxr $/bp — exact for the STIR book;
   others: position_value_base × %move; FX cash: USD exposure × %move)
- options   — FULL REVAL off the settlement surfaces: underlying forward
  shifted by its proxy's scenario move, per-leg IVs sticky, T fixed
  (same machinery as risk_options full-reval; vol risk NOT captured).

Shock units: rates proxies in bp of YIELD (+ = yields up), everything else
in % (+ = up; FX = USD-per-unit, so JPY +1% = yen stronger).
"""
from __future__ import annotations

_BUILD = "2026-09-04.1"

import json
import math
import os
import time

import numpy as np
import pandas as pd

import risk_div
import risk_options

# ── Event registry (Rajat 2026-09-04: "ability to add event weights and
# event specific correlations"). Each event identifies its HISTORICAL days
# via a rule or an explicit date list (past dates drive the estimation;
# future ones are harmless), plus a default weight ("NFP is worth 3 days").
# Extend by editing risk_events.json — no code change needed. ────────────────
_EVENTS_PATH = os.path.join(os.path.dirname(__file__), "risk_events.json")
_DEFAULT_EVENTS = {
    "NFP": {"rule": "first_friday", "weight": 3},
    "FOMC": {"weight": 4, "dates": [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
        "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
        "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
        "2026-09-16", "2026-10-28", "2026-12-09"]},
}


def load_events() -> dict:
    try:
        with open(_EVENTS_PATH) as f:
            d = json.load(f)
        if d:
            return d
    except Exception:
        pass
    try:
        with open(_EVENTS_PATH, "w") as f:
            json.dump(_DEFAULT_EVENTS, f, indent=2)
    except Exception:
        pass
    return dict(_DEFAULT_EVENTS)


def _event_mask(idx: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    if cfg.get("rule") == "first_friday":
        return (idx.weekday == 4) & (idx.day <= 7)
    ds = {pd.Timestamp(d).normalize() for d in cfg.get("dates", [])}
    return idx.normalize().isin(list(ds))

# proxy-returns memo so a 4-scenario batch fetches history ONCE (yf+FRED)
_RET_MEMO: dict = {}
_RET_TTL_S = 600


def _fetch_returns(fnames: list, fred_key) -> dict:
    key = (tuple(sorted(fnames)), bool(fred_key))
    now = time.time()
    ent = _RET_MEMO.get(key)
    if ent and now - ent[0] < _RET_TTL_S:
        return ent[1]
    start = (pd.Timestamp.today() - pd.Timedelta(days=900)).date().isoformat()
    risk_div._prime_proxy_batch(fnames, start)
    rets = {p: risk_div._proxy_returns(p, start, fred_key) for p in fnames}
    _RET_MEMO[key] = (now, rets)
    return rets


def _is_rate_factor(proxy: str) -> bool:
    return proxy in risk_div._RATE_FETCH or proxy in risk_div._FRED


def factor_universe(book, fx, sel_fut, sel_fx, products, proxies):
    """Ordered unique [(proxy, is_rate)] across selected futures, option
    underlyings and FX cash rows — the shockable factor list."""
    import risk as _risk
    seen: dict = {}

    def _add(p):
        if p and p not in seen:
            seen[p] = _is_rate_factor(p)

    if book is not None and not book.empty:
        for _, r in book.iterrows():
            s = r["Symbol"]
            if s not in sel_fut or bool(r.get("is_option")):
                continue
            _add(proxies.get(s) or "US2y")
    opts, _ = risk_options.option_book(book, sel_fut)
    for o in opts:
        prod = ("Rates" if o["src"] == "rates" else
                products.get(o["und"]) or _risk._guess_product(o["und"], o["und"]))
        _add(proxies.get(o["und"]) or proxies.get(o["root"])
             or _risk._guess_proxy(prod, o["root"], o["und"]))
    if fx is not None and not fx.empty:
        for _, r in fx.iterrows():
            c = r["Currency"]
            if c in sel_fx:
                _add(proxies.get(c, c))
    return [(p, ir) for p, ir in seen.items()]


def _to_ret_space(proxy: str, shock: float) -> float:
    """Natural units → proxy-return space (risk_div conventions):
    rates: +bp yield → −bp/100 pp (price-like); others: % → fraction."""
    if _is_rate_factor(proxy):
        return -shock / 100.0
    return shock / 100.0


def _to_natural(proxy: str, x: float) -> float:
    if _is_rate_factor(proxy):
        return -x * 100.0
    return x * 100.0


def compute(book, fx, sel_fut, sel_fx, products, ivols, proxies,
            shocks: dict, fred_key=None, propagate=True, window="6m",
            live=True, event: str | None = None,
            event_weight: float = 1.0) -> dict:
    """shocks = {proxy: natural-unit move} (only nonzero entries count).
    Returns {factors, rows, total, notes, obs}."""
    import risk as _risk

    notes: list = []
    factors = factor_universe(book, fx, sel_fut, sel_fx, products, proxies)
    fnames = [p for p, _ in factors]
    for p in shocks:
        if p not in fnames:
            fnames.append(p)
            factors.append((p, _is_rate_factor(p)))
    shocked = {p: _to_ret_space(p, v) for p, v in shocks.items()
               if v not in (None, 0.0)}

    # conditional moves for the unshocked factors
    x_all = dict(shocked)
    obs = 0
    unshocked = [p for p in fnames if p not in shocked]
    if propagate and shocked and unshocked:
        rets = _fetch_returns(fnames, fred_key)
        avail = [p for p in fnames if len(rets[p]) >= 60]
        R_df = (pd.DataFrame({p: rets[p] for p in avail}).dropna(how="any")
                if avail else pd.DataFrame())
        wn = risk_div.WINDOWS.get(window, 126)
        sub = R_df.tail(wn)
        obs = len(sub)
        s_av = [p for p in shocked if p in avail]
        u_av = [p for p in unshocked if p in avail]
        if obs >= max(20, wn // 3) and s_av and u_av:
            # Σ = D·R·D — correlations from the window's HISTORY, vols from
            # Rajat's ⚙ implied-vol params where one maps to the factor
            # (2026-09-04: "JPY move does not make sense given I have a higher
            # implied vol of USDJPY than EURUSD" — pure-historical cov ignored
            # his params). Historical std only where no param vol maps.
            # Event mode (Rajat 2026-09-04: "NFP is worth 3 days" → registry
            # of events): today is <event> day, so blend the correlation
            # matrix AND per-factor vol multiples toward the event-day-only
            # estimates from the FULL fetched history, both with weight
            # a = 1−1/w (shrinkage: the event-day sample is ~20-30 days).
            # In-window row-weighting was tried first and moves nothing (a 6m
            # window holds ~5 NFP days). Measured 2026-09-04: NFP-day
            # corr(EUR,JPY) 0.40 vs 0.58 all-days, corr(EUR,SPX) −0.50 vs
            # 0.00; NFP-day var multiples US2y ×5, SPX ×3, JPY ×1.7, EUR ×0.8.
            cols = s_av + u_av
            Rm = sub[cols].corr()
            wstd = {p: float(sub[p].std()) for p in cols}
            vmult = {p: 1.0 for p in cols}
            if event and event_weight and event_weight > 1:
                cfg = load_events().get(event) or {}
                dfl = pd.DataFrame({p: rets[p] for p in cols}).dropna(how="any")
                msk = _event_mask(dfl.index, cfg)
                dev, dno = dfl[msk], dfl[~msk]
                if len(dev) >= 12:
                    a = 1.0 - 1.0 / float(event_weight)
                    Rm = (1.0 - a) * Rm + a * dev.corr()
                    for p in cols:
                        sn = float(dno[p].std())
                        me = (float(dev[p].std()) / sn
                              if sn and math.isfinite(sn) else 1.0)
                        vmult[p] = 1.0 + a * (me - 1.0)
                    notes.append(
                        f"{event} mode ×{event_weight:g}: corr blended "
                        f"{a:.0%} toward {event}-day-only (n={len(dev)}); "
                        "vols × event multiples: "
                        + ", ".join(f"{p} ×{vmult[p]:.2f}" for p in cols))
                else:
                    notes.append(f"{event} mode skipped — only {len(dev)} "
                                 f"{event} day(s) in history")
            vsrc = {}

            def _fac_vol(p):
                """Daily vol in return-space units (frac for %-factors, pp for
                rates). Param ivol: currency/factor key direct, else the mean
                of mapped instruments' ivols; %ann (or bp ann) → daily /√256."""
                iv = ivols.get(p)
                if not iv:
                    c = [float(ivols[i]) for i, px in (proxies or {}).items()
                         if px == p and ivols.get(i)]
                    iv = sum(c) / len(c) if c else None
                if iv:
                    vsrc[p] = "param"
                    return float(iv) / 100.0 / 16.0 * vmult[p]  # √256 = 16
                vsrc[p] = "hist"
                return float(wstd[p]) * vmult[p]

            sig = {p: _fac_vol(p) for p in s_av + u_av}
            Dx = np.array([sig[p] for p in s_av])
            Dy = np.array([sig[p] for p in u_av])
            Cxx = Rm.loc[s_av, s_av].values * np.outer(Dx, Dx)
            Cyx = Rm.loc[u_av, s_av].values * np.outer(Dy, Dx)
            xv = np.array([shocked[p] for p in s_av])
            yv = Cyx @ np.linalg.pinv(Cxx) @ xv
            for p, y in zip(u_av, yv):
                x_all[p] = float(y)
            _hf = sorted(p for p, s in vsrc.items() if s == "hist")
            if _hf:
                notes.append("no ⚙ param vol maps to "
                             + ", ".join(_hf) + " — historical vol used there")
            for p in unshocked:
                if p not in u_av:
                    notes.append(f"{p}: no proxy history — held flat")
        elif shocked:
            notes.append(f"correlation window too thin ({obs} obs) — "
                         "unshocked factors held flat")
        for p in shocked:
            if p not in s_av and propagate:
                notes.append(f"{p}: shocked but no history — applied directly, "
                             "excluded from propagation")
    for p in fnames:
        x_all.setdefault(p, 0.0)

    frows = [(p, ir, p in shocked, _to_natural(p, x_all[p])) for p, ir in factors]

    # ── position P&L ─────────────────────────────────────────────────────────
    rows, total = [], 0.0

    def _row(name, kind, proxy, pnl, prod):
        nonlocal total
        total += pnl
        rows.append((name, kind, proxy, _to_natural(proxy, x_all.get(proxy, 0.0)),
                     _is_rate_factor(proxy), pnl, prod))

    if book is not None and not book.empty:
        for _, r in book.iterrows():
            s = r["Symbol"]
            if s not in sel_fut or bool(r.get("is_option")):
                continue
            prod = products.get(s, "Rates")
            proxy = proxies.get(s) or "US2y"
            x = x_all.get(proxy, 0.0)
            if prod == "Rates":
                usd_bp = (float(r["Quantity"]) * float(r.get("Multiplier") or 0.0)
                          * 0.01 * float(r.get("FXRateToBase") or 1.0))
                _row(s, "future", proxy, usd_bp * x * 100.0, prod)  # x pp → bp
            else:
                _row(s, "future", proxy,
                     float(r.get("position_value_base") or 0.0) * x, prod)

    opts, onotes = risk_options.option_book(book, sel_fut)
    notes += onotes
    for o in opts:
        res = risk_options._greeks(o, live)
        if res.get("err"):
            notes.append(f"{o['sym']}: {res['err']} — skipped")
            continue
        prod = ("Rates" if o["src"] == "rates" else
                products.get(o["und"]) or _risk._guess_product(o["und"], o["und"]))
        proxy = (proxies.get(o["und"]) or proxies.get(o["root"])
                 or _risk._guess_proxy(prod, o["root"], o["und"]))
        x = x_all.get(proxy, 0.0)
        F0 = float(res["F"])
        if o["src"] == "rates":
            dv01 = res.get("dv01")
            if not dv01:
                notes.append(f"{o['sym']}: no DV01 — skipped")
                continue
            if not _is_rate_factor(proxy):
                notes.append(f"{o['sym']}: proxy {proxy} is not a rates factor "
                             "— skipped")
                continue
            dF = x * 100.0 * float(dv01)        # price-like bp × pts/bp
        else:
            if _is_rate_factor(proxy):
                notes.append(f"{o['sym']}: rates proxy {proxy} on a {o['mkt']} "
                             "underlying — skipped (fix the ⚙ proxy)")
                continue
            dF = F0 * x
        v0 = risk_options._reval(o, res, F0)
        v1 = risk_options._reval(o, res, F0 + dF)
        _row(o["sym"], "option", proxy,
             (v1 - v0) * res["mult"] * o["qty"] * o["fxr"], prod)

    if fx is not None and not fx.empty:
        for _, r in fx.iterrows():
            c = r["Currency"]
            if c not in sel_fx or pd.isna(r.get("USD_exposure")):
                continue
            proxy = proxies.get(c, c)
            sign = 1.0 if str(r.get("side", "Long")) == "Long" else -1.0
            _row(c, "fx cash", proxy,
                 sign * abs(float(r["USD_exposure"])) * x_all.get(proxy, 0.0),
                 "FX")

    return {"factors": frows, "rows": rows, "total": total,
            "notes": notes, "obs": obs, "window": window,
            "propagate": propagate}

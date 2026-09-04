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

import math
import time

import numpy as np
import pandas as pd

import risk_div
import risk_options

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
            live=True) -> dict:
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
            C = sub.cov()
            Cxx = C.loc[s_av, s_av].values
            Cyx = C.loc[u_av, s_av].values
            xv = np.array([shocked[p] for p in s_av])
            yv = Cyx @ np.linalg.pinv(Cxx) @ xv
            for p, y in zip(u_av, yv):
                x_all[p] = float(y)
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

    def _row(name, kind, proxy, pnl):
        nonlocal total
        total += pnl
        rows.append((name, kind, proxy, _to_natural(proxy, x_all.get(proxy, 0.0)),
                     _is_rate_factor(proxy), pnl))

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
                _row(s, "future", proxy, usd_bp * x * 100.0)   # x pp → bp
            else:
                _row(s, "future", proxy,
                     float(r.get("position_value_base") or 0.0) * x)

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
             (v1 - v0) * res["mult"] * o["qty"] * o["fxr"])

    if fx is not None and not fx.empty:
        for _, r in fx.iterrows():
            c = r["Currency"]
            if c not in sel_fx or pd.isna(r.get("USD_exposure")):
                continue
            proxy = proxies.get(c, c)
            sign = 1.0 if str(r.get("side", "Long")) == "Long" else -1.0
            _row(c, "fx cash", proxy,
                 sign * abs(float(r["USD_exposure"])) * x_all.get(proxy, 0.0))

    return {"factors": frows, "rows": rows, "total": total,
            "notes": notes, "obs": obs, "window": window,
            "propagate": propagate}

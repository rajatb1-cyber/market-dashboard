"""
Rates Options Monitor — Databento settlement data, NORMAL (Bachelier) vol.

Rates options need a different vol model than equities/commodities: yields (and, to a
first approximation, bond-future prices) are better described by a NORMAL distribution
than a lognormal one, so implied vol here is "price-normal vol" (points/yr) and, for
bond-price-quoted contracts, a derived "yield-normal vol" (bp/yr) via an estimated
CTD duration + CBOT-style conversion factor.

Phase 1: TY (10Y UST Note options, CME). Roadmap: FV/UB/US (same CTD-duration path),
RX/OE (Eurex Bund/Bobl, same path), SOFR/Euribor/SONIA (rate-quoted — price-vol IS
yield-vol already, no CTD/duration needed).

Reuses the generic, asset-agnostic Databento plumbing from options_v2.py (definitions/
OHLCV fetch, futures-price matching, strike selection) rather than duplicating it.
"""
import math
import os
import re
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from typing import Optional
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go

import rates as _rates
from options_v2 import (
    _api_key, _trade_date, _prev_bday, _date_window, _get_range,
    _fetch_opt_defs, _fetch_opt_ohlcv, _get_fut_info, _infer_fut_pcp,
    _pick_equidistant, _sf, _STRUCTURE_TYPES, _STRIKE_LABELS, _structure_qtys,
    _STD_SIGMA_OPTS,
)

_CME  = "GLBX.MDP3"
_XEUR = "XEUR.EOBI"
_DISK_CACHE_DIR = os.path.join(os.path.dirname(__file__), "rates_options_cache")

# ── Markets ───────────────────────────────────────────────────────────────────
# ctd_years: default assumed CTD remaining-maturity proxy (editable in the UI).
# Per CME's own "Understanding Treasury Futures" reference (Table 3, Dec-2017 TY basis):
# when yields are BELOW the 6% notional coupon (as essentially always in the modern rate
# regime, incl. today, both USD and EUR), the conversion-factor bias favours SHORT-duration
# notes as CTD — i.e. toward the LOW end of each contract's deliverable window, not a
# freshly-issued note near the top of it. Defaults below sit near each window's low end.
# freq: coupon periods/yr — 2 (semiannual) for USD CME/CBOT notes & bonds, 1 (annual) for
# EUR Eurex Bunds/Bobls (confirmed via Eurex's own conversion-factor spec — a genuine
# structural difference, not just a different deliverable window).
# curve: which par-curve fetcher seeds the default yield/coupon ("us" or "eur").
_MARKETS_RATES = {
    # 2Y note (CME/IBKR root ZT, BBG TU — internal keys are bbg-style).
    # NB $200k face (not 100k): 1pt = $2,000 — pricer._MULT carries that.
    "TU": {"label": "2Y US Note (TU)",  "opt_sym": "OZT.OPT",  "fut_sym": "ZT.FUT",
           "ds": _CME,  "r": 0.045, "needs_ctd": True, "freq": 2, "curve": "us",
           "deliverable_window": (1.75, 2.0), "ctd_years": 1.79, "yf_fallback": "ZT=F"},
    "TY": {"label": "10Y US Note (TY)", "opt_sym": "OZN.OPT",  "fut_sym": "ZN.FUT",
           "ds": _CME,  "r": 0.045, "needs_ctd": True, "freq": 2, "curve": "us",
           "deliverable_window": (6.5, 8.0), "ctd_years": 7.25, "yf_fallback": "ZN=F"},
    # Classic T-Bond (ZB, BBG root US): deliverable 15-25y; with yields < 6% the
    # CTD sits at the SHORT end of the window (lowest duration) → ~15.5y.
    "US": {"label": "T-Bond 15-25y (ZB)", "opt_sym": "OZB.OPT", "fut_sym": "ZB.FUT",
           "ds": _CME,  "r": 0.045, "needs_ctd": True, "freq": 2, "curve": "us",
           "deliverable_window": (15.0, 25.0), "ctd_years": 15.5, "yf_fallback": "ZB=F"},
    "FV": {"label": "5Y US Note (FV)",  "opt_sym": "OZF.OPT",  "fut_sym": "ZF.FUT",
           "ds": _CME,  "r": 0.045, "needs_ctd": True, "freq": 2, "curve": "us",
           "deliverable_window": (4.17, 5.25), "ctd_years": 4.35},
    "UB": {"label": "Ultra T-Bond (UB)", "opt_sym": "OUB.OPT", "fut_sym": "UB.FUT",
           "ds": _CME,  "r": 0.045, "needs_ctd": True, "freq": 2, "curve": "us",
           "deliverable_window": (25.0, 30.25), "ctd_years": 25.5},
    "RX": {"label": "Euro-Bund (RX)",   "opt_sym": "OGBL.OPT", "fut_sym": "FGBL.FUT",
           "ds": _XEUR, "r": 0.030, "needs_ctd": True, "freq": 1, "curve": "eur",
           "deliverable_window": (8.5, 10.5), "ctd_years": 8.75},
    "OE": {"label": "Euro-Bobl (OE)",   "opt_sym": "OGBM.OPT", "fut_sym": "FGBM.FUT",
           "ds": _XEUR, "r": 0.030, "needs_ctd": True, "freq": 1, "curve": "eur",
           "deliverable_window": (4.5, 5.5), "ctd_years": 4.6},
    # Buxl future notional coupon is 4% (not 6%) — with 30y Bund yields near
    # 4% the CTD bias is mild; default still sits toward the window's low end.
    # Internal key "UX" because "UB" is taken by the CME Ultra (BBG: Buxl=UB).
    "UX": {"label": "Euro-Buxl (UB)",   "opt_sym": "OGBX.OPT", "fut_sym": "FGBX.FUT",
           "ds": _XEUR, "r": 0.030, "needs_ctd": True, "freq": 1, "curve": "eur",
           "deliverable_window": (24.0, 35.0), "ctd_years": 24.5},
    "DU": {"label": "Euro-Schatz (DU)", "opt_sym": "OGBS.OPT", "fut_sym": "FGBS.FUT",
           "ds": _XEUR, "r": 0.030, "needs_ctd": True, "freq": 1, "curve": "eur",
           "deliverable_window": (1.75, 2.25), "ctd_years": 1.85},

    # ── Money-market (rate-quoted: price = 100 − rate, so price-vol IS yield-vol, just
    # scaled ×100 since 1 price-point = 100bp — no CTD/duration conversion needed at all.
    # fixed_dv01=0.01 reuses the existing yield_vol = price_vol/dv01 machinery to get that
    # ×100 scaling "for free" through Chain/Expiry Curve/Pricer/Vol-Monitor unchanged.
    # raw_scale: SR3 (SOFR) reports strike/premium at 100x the true decimal scale in
    # Databento — confirmed via a live probe (strike 9500 vs future ~95.00, tick 0.25/100 =
    # 0.0025 matches CME's published SR3 option tick exactly). Euribor/SONIA are natively
    # 1:1, no raw_scale needed (confirmed via probe: strikes already ~93-97 matching futures).
    #
    # Underlying-future selection for money-market roots is rule-based (see
    # _get_fut_info_mm): from the option expiry, the next quarterly IMM month is the
    # underlying period START. Two config knobs drive it:
    #   underlying_offset_years    — 0 for the front root, +1/+2 for the 1y/2y midcurves.
    #   underlying_end_of_quarter  — False for period-START-coded futures (SOFR SR3 codes
    #                                by the START IMM; Euribor forward-looking expires at the
    #                                START IMM); True for SONIA (backward-looking — the
    #                                future expires at the END of the reference quarter, i.e.
    #                                one quarter LATER than the period start).
    # On CME the option defs carry a REAL underlying_id pointing straight at the correct
    # future (incl. the +Ny midcurve one), so _get_fut_info_mm resolves those exactly via
    # underlying_id and only uses the rule as a fallback. On ICE underlying_id points at a
    # synthetic placeholder (useless), so the rule is the sole path there. All verified
    # against put-call-parity forwards (F = K + C − P) on live settlements.
    "SOFR": {"label": "SOFR 3M (SR3)",   "vm_label": "SOFR",    "opt_sym": "SR3.OPT", "fut_sym": "SR3.FUT",
            "ds": _CME,  "r": 0.045, "needs_ctd": False, "fixed_dv01": 0.01, "raw_scale": 100,
            "underlying_offset_years": 0, "underlying_end_of_quarter": False},
    "SOFR_1Y": {"label": "SFR 1yMC (S0)", "vm_label": "SFR 1yMC", "opt_sym": "S0.OPT", "fut_sym": "SR3.FUT",
            "ds": _CME,  "r": 0.045, "needs_ctd": False, "fixed_dv01": 0.01, "raw_scale": 100,
            "underlying_offset_years": 1, "underlying_end_of_quarter": False},
    "SOFR_2Y": {"label": "SFR 2yMC (S2)", "vm_label": "SFR 2yMC", "opt_sym": "S2.OPT", "fut_sym": "SR3.FUT",
            "ds": _CME,  "r": 0.045, "needs_ctd": False, "fixed_dv01": 0.01, "raw_scale": 100,
            "underlying_offset_years": 2, "underlying_end_of_quarter": False},
    "ER":   {"label": "Euribor 3M (ER)", "vm_label": "ER",      "opt_sym": "I.OPT",   "fut_sym": "I.FUT",
            "ds": "IFLL.IMPACT", "r": 0.030, "needs_ctd": False, "fixed_dv01": 0.01,
            "underlying_offset_years": 0, "underlying_end_of_quarter": False},
    "ER_1Y": {"label": "ER 1yMC (K)",    "vm_label": "ER 1yMC", "opt_sym": "K.OPT",   "fut_sym": "I.FUT",
            "ds": "IFLL.IMPACT", "r": 0.030, "needs_ctd": False, "fixed_dv01": 0.01,
            "underlying_offset_years": 1, "underlying_end_of_quarter": False},
    "ER_2Y": {"label": "ER 2yMC (K2)",   "vm_label": "ER 2yMC", "opt_sym": "K2.OPT",  "fut_sym": "I.FUT",
            "ds": "IFLL.IMPACT", "r": 0.030, "needs_ctd": False, "fixed_dv01": 0.01,
            "underlying_offset_years": 2, "underlying_end_of_quarter": False},
    "SONIA":{"label": "SONIA 3M (SO3)",  "vm_label": "SONIA",   "opt_sym": "SO3.OPT", "fut_sym": "SO3.FUT",
            "ds": "IFLL.IMPACT", "r": 0.040, "needs_ctd": False, "fixed_dv01": 0.01,
            "underlying_offset_years": 0, "underlying_end_of_quarter": True},
    "SONIA_1Y": {"label": "SON 1yMC (SY1)", "vm_label": "SON 1yMC", "opt_sym": "SY1.OPT", "fut_sym": "SO3.FUT",
            "ds": "IFLL.IMPACT", "r": 0.040, "needs_ctd": False, "fixed_dv01": 0.01,
            "underlying_offset_years": 1, "underlying_end_of_quarter": True},
    "SONIA_2Y": {"label": "SON 2yMC (SY2)", "vm_label": "SON 2yMC", "opt_sym": "SY2.OPT", "fut_sym": "SO3.FUT",
            "ds": "IFLL.IMPACT", "r": 0.040, "needs_ctd": False, "fixed_dv01": 0.01,
            "underlying_offset_years": 2, "underlying_end_of_quarter": True},
}

_YF_FALLBACK_RATES = {k: v["yf_fallback"] for k, v in _MARKETS_RATES.items() if v.get("yf_fallback")}

_BOND_FUTURE_MARKETS = ["TU", "FV", "TY", "UB", "DU", "OE", "RX", "UX"]
# Front + 1y/2y midcurve for each of the three money-market complexes, in the
# column/selector order the user specified.
_MONEY_MKT_MARKETS   = ["SOFR", "SOFR_1Y", "SOFR_2Y",
                        "ER", "ER_1Y", "ER_2Y",
                        "SONIA", "SONIA_1Y", "SONIA_2Y"]


# ── Money-market underlying-future resolution ──────────────────────────────────
# Rate futures (SOFR SR3, Euribor, SONIA) are quarterly IMM contracts. An option (serial
# or quarterly) exercises into the quarterly future whose REFERENCE QUARTER STARTS at the
# next IMM (3rd Wed of Mar/Jun/Sep/Dec) on/after the option's own expiry — plus N years for
# the 1y/2y midcurves. All three exchanges CODE the future's symbol by that start IMM month
# (SOFR "SR3U6", Euribor "I FMU0026!", SONIA "SO3 FMU0026!" all use U=Sep for a Sep-start
# quarter), so a single "symbol-code month == next-quarterly-IMM" rule handles every case —
# the forward-vs-backward-looking difference (Euribor expires at the quarter start, SOFR/
# SONIA at the quarter end) is already absorbed by that start-coded symbology. Verified
# against put-call-parity forwards on live settlements (see the config block above).
_MM_MONTH_CODES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
                   "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
_QUARTERLY_MONTHS = [3, 6, 9, 12]


def _imm_third_wed(year: int, month: int) -> date:
    """IMM date = 3rd Wednesday of the month."""
    d0 = date(year, month, 1)
    first_wed = 1 + (2 - d0.weekday()) % 7   # weekday(): Mon=0 … Wed=2
    return date(year, month, first_wed + 14)


def _next_quarterly_imm(dt: date) -> tuple:
    """(year, month) of the first quarterly IMM (Mar/Jun/Sep/Dec, 3rd Wed) on/after `dt`."""
    for yy in (dt.year, dt.year + 1, dt.year + 2, dt.year + 3):
        for mo in _QUARTERLY_MONTHS:
            if _imm_third_wed(yy, mo) >= dt:
                return yy, mo
    return dt.year, 12


def _parse_ice_fut_ym(symbol) -> Optional[tuple]:
    """(year, month) from an ICE outright rate-future symbol, e.g. 'SO3 FMU0026!' /
    'I  FMU0026!' → (2026, 9). Matches ONLY the 'FM<code><year>!' outright form — calendar
    spreads ('I  FUH0027.Z0028'), packs/bundles ('FP…') and other composites are rejected
    so they can never be mistaken for the underlying."""
    mt = re.search(r"FM([FGHJKMNQUVXZ])(\d{2,4})!$", str(symbol).strip().upper())
    if not mt:
        return None
    yr = int(mt.group(2))
    if yr < 100:
        yr += 2000
    return yr, _MM_MONTH_CODES[mt.group(1)]


def _get_fut_info_mm(opt_expiry: date, defs_exp: pd.DataFrame, futs: pd.DataFrame,
                     m: dict, mkt_key: str) -> tuple:
    """Underlying (F, symbol) for a money-market option expiry, honouring midcurve offsets.
    CME: option defs carry the REAL underlying_id (exact even for the +Ny midcurve future) —
    delegate to _get_fut_info's underlying_id path; never symbol-parse CME (its analytic
    'SR3:AB 01Y U6' instruments would fool a code parser). ICE: underlying_id is a synthetic
    placeholder, so pick the quarterly outright by the reference-quarter-start rule."""
    if m["ds"] == _CME:
        return _get_fut_info(opt_expiry, defs_exp, futs, mkt_key)
    if futs is None or getattr(futs, "empty", True) or "symbol" not in futs.columns:
        return None, ""
    sy, sm = _next_quarterly_imm(opt_expiry)
    ty, tm = sy + int(m.get("underlying_offset_years", 0)), sm
    for _, row in futs.dropna(subset=["close"]).iterrows():
        if _parse_ice_fut_ym(row.get("symbol")) == (ty, tm):
            price = _sf(row["close"])
            if price:
                return price, str(row.get("symbol", "")).strip()
    return None, ""


def _resolve_fut_info(opt_expiry: date, defs_exp: pd.DataFrame, futs: pd.DataFrame,
                      mkt_key: str, m: dict) -> tuple:
    """Dispatch: money-market markets (front + midcurve) use the midcurve-aware resolver;
    bond-future markets keep the generic options_v2 resolver unchanged."""
    if "fixed_dv01" in m:
        return _get_fut_info_mm(opt_expiry, defs_exp, futs, m, mkt_key)
    return _get_fut_info(opt_expiry, defs_exp, futs, mkt_key)


# ── Bachelier (normal) model ───────────────────────────────────────────────────
def _bachelier(F: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    """Normal (Bachelier) price. sigma in same units as F/K (price points), annualized."""
    if sigma <= 0 or T <= 0:
        intrinsic = max(F - K, 0) if right == "C" else max(K - F, 0)
        return math.exp(-r * T) * intrinsic
    d = (F - K) / (sigma * math.sqrt(T))
    df = math.exp(-r * T)
    if right == "C":
        return df * ((F - K) * norm.cdf(d) + sigma * math.sqrt(T) * norm.pdf(d))
    return df * ((K - F) * norm.cdf(-d) + sigma * math.sqrt(T) * norm.pdf(d))


def _calc_iv_normal(price: Optional[float], F: float, K: float, T: float, r: float,
                    right: str) -> Optional[float]:
    """Solve for Bachelier sigma (price-points/yr) given a settlement premium."""
    if not (price and F and K and T and T > 0):
        return None
    try:
        intrinsic = math.exp(-r * T) * (max(F - K, 0) if right == "C" else max(K - F, 0))
        if price <= intrinsic * 1.001:
            return None
        # Bracket in price-points: wide enough for anything from quiet to very volatile regimes.
        return brentq(lambda s: _bachelier(F, K, T, r, s, right) - price,
                      1e-6, 50.0, xtol=1e-6, maxiter=100)
    except Exception:
        return None


# ── CTD proxy: modified duration + conversion factor ───────────────────────────
# freq = coupon periods/yr: 2 for USD Treasury notes/bonds (semiannual), 1 for EUR
# Bunds/Bobls (Eurex/German government bonds pay coupons ANNUALLY — a genuine structural
# difference from USD, confirmed via Eurex's own conversion-factor spec).
def _bond_price_duration(coupon: float, ytm: float, years_to_maturity: float, freq: int = 2):
    """Standard coupon-bond math (semiannual freq=2 for USD, annual freq=1 for EUR).
    coupon/ytm as decimals (e.g. 0.045). Returns (price_per_100, macaulay_yrs, mod_dur_yrs)."""
    n = max(1, round(years_to_maturity * freq))
    i = ytm / freq
    cpn = coupon / freq * 100.0
    price = 0.0
    weighted = 0.0
    for t in range(1, n + 1):
        cf = cpn + (100.0 if t == n else 0.0)
        pv = cf * (1 + i) ** (-t)
        price += pv
        weighted += pv * (t / freq)
    if price <= 0:
        return 100.0, years_to_maturity, years_to_maturity
    macaulay = weighted / price
    mod_dur = macaulay / (1 + i)
    return price, macaulay, mod_dur


def _cbot_conversion_factor(coupon: float, years_to_maturity: float) -> float:
    """CBOT-style (semiannual) conversion factor vs the standard 6% notional yield.
    Rounds remaining term down to the nearest quarter, per CBOT convention. Validated
    against CME's own worked examples — use for USD (CME/CBOT) markets."""
    total_months = round(years_to_maturity * 12)
    n = total_months // 12                  # whole years
    z = ((total_months % 12) // 3) * 3       # remaining months, rounded down to nearest quarter
    c = coupon
    a = 1.0 / (1.03 ** (z / 6.0))
    b = (c / 2.0) + (c / 0.06) + (1 - c / 0.06) / (1.03 ** (2 * n))
    return a * b - (c / 2.0) * ((6 - z) / 6.0)


def _direct_conversion_factor(coupon: float, years_to_maturity: float, freq: int) -> float:
    """Conversion factor = clean price of the bond at a 6% yield, per 100 face (CME's own
    definition) — computed directly via _bond_price_duration rather than a closed-form
    quarter/day-count formula. Used for EUR (annual, freq=1) where Eurex's exact formula
    needs real coupon dates we don't track; also cross-checks close (~0.3pp) to the CBOT
    formula for USD, so it's a reasonable general-purpose fallback."""
    price_at_6pct, _, _ = _bond_price_duration(coupon, 0.06, years_to_maturity, freq)
    return price_at_6pct / 100.0


def _estimate_dv01(coupon: float, ytm: float, years_to_maturity: float, freq: int = 2):
    """Returns (ctd_price, mod_dur_yrs, conv_factor, dv01_points_per_bp)."""
    price, _, mod_dur = _bond_price_duration(coupon, ytm, years_to_maturity, freq)
    cf = (_cbot_conversion_factor(coupon, years_to_maturity) if freq == 2
          else _direct_conversion_factor(coupon, years_to_maturity, freq))
    dv01 = mod_dur * price * 0.0001 / cf if cf else float("nan")
    return price, mod_dur, cf, dv01


# ── Bachelier greeks (analogous to options_v2._black76_greeks) ────────────────
def _bachelier_greeks(F: float, K: float, T: float, r: float, sigma: float, right: str) -> dict:
    """Normal-model price + delta/gamma/vega/theta. vega/theta/gamma are w.r.t. sigma in
    PRICE-points (the model's native unit) — same convention `_bachelier`/`_calc_iv_normal`
    already use. To display "per 1bp of yield-vol", multiply vega by DV01 (since
    price_vol = yield_vol × DV01, so d(price)/d(yield_vol) = vega_price × DV01)."""
    if sigma <= 0 or T <= 0:
        return dict(price=0.0, delta=0.0, gamma=0.0, vega=0.0, theta=0.0)
    sqT = math.sqrt(T)
    d = (F - K) / (sigma * sqT)
    df = math.exp(-r * T)
    nd = norm.pdf(d)
    if right == "C":
        price = df * ((F - K) * norm.cdf(d) + sigma * sqT * nd)
        delta = df * norm.cdf(d)
    else:
        price = df * ((K - F) * norm.cdf(-d) + sigma * sqT * nd)
        delta = -df * norm.cdf(-d)
    gamma = df * nd / (sigma * sqT)
    vega  = df * sqT * nd
    theta = (r * price - df * sigma * nd / (2 * sqT)) / 365
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)


_UST_TENOR_YEARS = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30}


def _interp_ust_yield(us_row, target_years: float) -> float:
    """Linear interpolation of a par curve row (US or EUR — same tenor columns) at an
    arbitrary maturity (years) — lets the yield/coupon default track whatever CTD maturity
    is assumed, rather than being pinned to a single fixed tenor regardless of that
    assumption."""
    pts = sorted((yrs, float(us_row[t])) for t, yrs in _UST_TENOR_YEARS.items()
                 if t in us_row.index and pd.notna(us_row.get(t)))
    if not pts:
        return 4.5
    if target_years <= pts[0][0]:
        return pts[0][1]
    if target_years >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= target_years <= x1:
            w = (target_years - x0) / (x1 - x0)
            return y0 + w * (y1 - y0)
    return pts[-1][1]


@st.cache_data(ttl=3600, show_spinner=False)
def _default_ctd_inputs(target_years: float, curve: str = "us") -> tuple:
    """(default_yield_pct, default_coupon_pct) from the US or EUR par curve, interpolated at
    the assumed CTD maturity — so the default stays consistent if that maturity is adjusted.
    curve="eur" uses the ECB curve (rates.fetch_ecb_curve) for RX/OE, same tenor columns
    (1/2/3/5/7/10/20/30Y) as the US curve — no new data source needed."""
    try:
        if curve == "eur":
            _start = (pd.Timestamp.today() - pd.Timedelta(days=60)).date().isoformat()
            df = _rates.fetch_ecb_curve(start=_start)
        else:
            df = _rates.fetch_us_treasury(months_back=2)
        y = _interp_ust_yield(df.dropna(how="all").iloc[-1], target_years)
    except Exception:
        y = 4.5 if curve == "us" else 2.7
    coupon = round(y * 8) / 8.0   # notes near par -> coupon ~ nearest 1/8 of the yield at issue
    return y, coupon


# ── Data loading (mirrors options_v2._load_data, single opt_sym — no monthly supplement
#    needed: OZN.OPT already lists both serial and quarterly expiries) ────────────────
def _disk_cache_path(mkt_key: str, date_str: str, version: int) -> str:
    os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
    return os.path.join(_DISK_CACHE_DIR, f"{mkt_key}_{date_str}_v{version}.pkl")


# ttl=24h in-memory (fast path within one running server process) PLUS an on-disk cache
# (survives dev-server restarts — st.cache_data's memory cache is wiped on every restart,
# which was making every code-reload re-pay Databento's ~5-90s cold "definition" fetch for
# this market's large option chain). Disk cache key already includes date_str (one
# settlement per trading day), so it's always safe to reuse across restarts same-day.
def _load_data_disk(mkt_key: str, date_str: str, version: int = 1) -> dict:
    """Pure (NO Streamlit) disk-cached loader — safe to call from worker threads.
    Byte-identical to the old in-line disk logic; the st.cache_data wrapper delegates here
    so the Vol Dashboard's parallel prefetch can warm the same pickles from threads."""
    path = _disk_cache_path(mkt_key, date_str, version)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    data = _load_data_fresh(mkt_key, date_str)
    # Don't persist a missing-API-key stub, and don't persist any fetch with EMPTY defs
    # (error or not) — a transient Databento failure or a session not yet rolled into
    # historical availability would otherwise poison this date's cache permanently.
    _failed = (data.get("opt_defs") is None
               or getattr(data.get("opt_defs"), "empty", True))
    if not data.get("error") and not _failed:
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass
    return data


@st.cache_data(ttl=86400, show_spinner=False)
def _load_data(mkt_key: str, date_str: str, version: int = 1) -> dict:
    return _load_data_disk(mkt_key, date_str, version)


def _fetch_fut_shared(client, ds: str, fut_sym: str, s: str, e: str, date_str: str) -> pd.DataFrame:
    """Futures settles for one root, fetched ONCE per (root, day) and shared across every
    market using that root (ER/ER_1Y/ER_2Y all need I.FUT — pre-2026-07-28 each paid for
    its own copy: 3 × $0.44/day on IFLL). `statistics` stays PRIMARY: exchanges settle
    EVERY contract daily, while ohlcv-1d only covers traded ones — untraded far-quarterly
    outrights (the midcurve underlyings!) go missing on quiet days, and ohlcv closes can
    differ from official settles by ~2.5bp (verified 2026-07-28). ohlcv-1d is fallback
    only. Restricting statistics to outright symbols was probed and saves only 23% — not
    worth it. Disk-cached per (root, dataset, day); empty results never persisted."""
    path = os.path.join(_DISK_CACHE_DIR, f"FUT_{fut_sym.replace('.', '_')}_"
                                         f"{ds.split('.')[0]}_{date_str}_v2.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    df = pd.DataFrame()
    _SETTLE = 3
    try:
        raw = _get_range(client, ds, [fut_sym], "statistics", s, e)
        d = raw.to_df().reset_index()
        d["stat_type"] = d["stat_type"].apply(int)
        settle = (d[d["stat_type"] == _SETTLE]
                  .sort_values("ts_event")
                  .groupby("instrument_id")["price"]
                  .last().reset_index()
                  .rename(columns={"price": "close"}))
        if not settle.empty and "symbol" in d.columns:
            sym_map = d[["instrument_id", "symbol"]].drop_duplicates("instrument_id")
            settle = settle.merge(sym_map, on="instrument_id", how="left")
        df = settle
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        try:
            raw = _get_range(client, ds, [fut_sym], "ohlcv-1d", s, e)
            d = raw.to_df().reset_index()
            if not d.empty and "close" in d.columns:
                df = d[["instrument_id", "close", "symbol"]].dropna(subset=["close"]).copy()
        except Exception:
            df = pd.DataFrame()

    if not df.empty:
        try:
            os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(df, f)
        except Exception:
            pass
    return df


def _load_data_fresh(mkt_key: str, date_str: str) -> dict:
    """The actual Databento fetch — only runs on a genuine cache miss (disk + memory)."""
    m = _MARKETS_RATES[mkt_key]
    key = _api_key()
    if not key:
        return {"error": "No DATABENTO_API_KEY set"}
    import databento as db
    client = db.Historical(key=key)
    s, e = _date_window(date_str)
    # Eurex (XEUR.EOBI): Databento's historical license ends at each session's 22:00 UTC
    # close — the default 23:59:59 window end crosses 2h into the live-license zone and
    # 403s ("license_not_found_unauthorized") whenever the fetch races the daily
    # availability roll. Session data all lies before 22:00, so clamping loses nothing.
    if m["ds"].startswith("XEUR"):
        e = f"{date_str}T21:59:59"
    out = {}

    try:
        defs = _fetch_opt_defs(client, m["ds"], m["opt_sym"], s, e)
        out["quarterly_expiries"] = (
            {r.date() for r in defs["expiration"].dropna()} if "expiration" in defs.columns else set()
        )
    except Exception as ex:
        defs = pd.DataFrame()
        out["quarterly_expiries"] = set()
        out["defs_err"] = str(ex)
    out["opt_defs"] = defs

    try:
        ohlcv = _fetch_opt_ohlcv(client, m["ds"], m["opt_sym"], s, e)
        if ohlcv.attrs.get("stats_err"):
            out["stats_err"] = ohlcv.attrs["stats_err"]
    except Exception as ex:
        ohlcv = pd.DataFrame()
        out["ohlcv_err"] = str(ex)
    out["opt_ohlcv"] = ohlcv

    out["fut_ohlcv"] = _fetch_fut_shared(client, m["ds"], m["fut_sym"], s, e, date_str)

    # Some money-market OPTION roots (confirmed: CME SOFR/SR3.OPT) report strike_price and
    # premium at 100x the future's actual decimal scale (e.g. strike 9500 vs future ~95.00) —
    # a raw Databento representation quirk on the OPTION side only, not a real quoting
    # difference. Confirmed the FUTURE's own settlement is already correctly scaled (~96.0,
    # not ~9600) — so only opt_defs/opt_ohlcv get corrected here, NOT fut_ohlcv. Confirmed
    # not needed at all for Euribor/SONIA (both already 1:1 on both sides).
    scale = m.get("raw_scale", 1)
    if scale != 1:
        if not out["opt_defs"].empty and "strike_price" in out["opt_defs"].columns:
            out["opt_defs"] = out["opt_defs"].copy()
            out["opt_defs"]["strike_price"] = out["opt_defs"]["strike_price"] / scale
        if not out["opt_ohlcv"].empty and "close" in out["opt_ohlcv"].columns:
            out["opt_ohlcv"] = out["opt_ohlcv"].copy()
            out["opt_ohlcv"]["close"] = out["opt_ohlcv"]["close"] / scale

    return out


# ── Chain building ──────────────────────────────────────────────────────────────
# Cached: the IV root-solving is pure CPU and re-ran on EVERY Streamlit rerun once the
# tab was loaded (any click anywhere in the app froze on it). `_data` is skip-hashed
# (big DataFrame dict); `cache_date` carries the trade date so keys roll daily.
@st.cache_data(ttl=3600, show_spinner=False)
def _build_chain(mkt_key: str, expiry: date, n_per_side: int, sigma_range: float,
                 _data: dict, r: float, dv01: float, cache_date: str = "") -> pd.DataFrame:
    data = _data
    m = _MARKETS_RATES[mkt_key]
    defs = data.get("opt_defs", pd.DataFrame())
    ohlcv = data.get("opt_ohlcv", pd.DataFrame())
    futs = data.get("fut_ohlcv", pd.DataFrame())

    if defs.empty or "expiration" not in defs.columns:
        return pd.DataFrame()
    defs_exp = defs[defs["expiration"].dt.date == expiry].copy()
    if defs_exp.empty:
        return pd.DataFrame()

    ohlcv_cols = (ohlcv[["instrument_id", "close", "volume"]] if not ohlcv.empty
                  else pd.DataFrame(columns=["instrument_id", "close", "volume"]))
    df = defs_exp.merge(ohlcv_cols, on="instrument_id", how="left")
    if df.empty:
        return pd.DataFrame()

    F, fut_sym = _resolve_fut_info(expiry, defs_exp, futs, mkt_key, m)
    if not F:
        F = _infer_fut_pcp(df.dropna(subset=["close"]), expiry, r)
        fut_sym = "put-call parity"
    if not F:
        return pd.DataFrame()

    T = max((expiry - date.today()).days, 1) / 365.0

    calls = (df[df["instrument_class"] == "C"]
             .groupby("strike_price", as_index=False).first()
             [["strike_price", "close", "volume"]]
             .rename(columns={"close": "call_p", "volume": "call_vol"}))
    puts = (df[df["instrument_class"] == "P"]
            .groupby("strike_price", as_index=False).first()
            [["strike_price", "close", "volume"]]
            .rename(columns={"close": "put_p", "volume": "put_vol"}))
    chain_full = (calls.merge(puts, on="strike_price", how="outer")
                       .rename(columns={"strike_price": "strike"})
                       .sort_values("strike").reset_index(drop=True))

    all_strikes = sorted(chain_full["strike"].tolist())
    if not all_strikes:
        return pd.DataFrame()
    atm_k = min(all_strikes, key=lambda x: abs(x - F))

    if sigma_range > 0:
        atm_row = chain_full[chain_full["strike"] == atm_k].iloc[0]
        atm_iv_c = _calc_iv_normal(_sf(atm_row.get("call_p")), F, atm_k, T, r, "C")
        atm_iv_p = _calc_iv_normal(_sf(atm_row.get("put_p")), F, atm_k, T, r, "P")
        atm_iv = atm_iv_c or atm_iv_p
        if not atm_iv:
            _settled = chain_full.dropna(subset=["call_p", "put_p"], how="all")
            if not _settled.empty:
                _near = _settled.iloc[(_settled["strike"] - F).abs().argsort()[:1]].iloc[0]
                atm_iv = (_calc_iv_normal(_sf(_near.get("call_p")), F, _near["strike"], T, r, "C") or
                          _calc_iv_normal(_sf(_near.get("put_p")), F, _near["strike"], T, r, "P"))
        if atm_iv:
            # sigma is already in price-points/yr (normal model) — no need to multiply by F.
            sigma_move = sigma_range * atm_iv * math.sqrt(T)
            upper, lower = F + sigma_move, F - sigma_move
            above = sorted([s for s in all_strikes if s > atm_k and s <= upper])
            below = sorted([s for s in all_strikes if s < atm_k and s >= lower], reverse=True)
            selected = set([atm_k]
                           + _pick_equidistant(above, atm_k, upper, n_per_side)
                           + _pick_equidistant(below, atm_k, lower, n_per_side))
        else:
            idx = all_strikes.index(atm_k)
            selected = set(all_strikes[max(0, idx - n_per_side): idx + n_per_side + 1])
    else:
        idx = all_strikes.index(atm_k)
        selected = set(all_strikes[max(0, idx - n_per_side): idx + n_per_side + 1])

    chain = (chain_full[chain_full["strike"].isin(selected)]
                       .sort_values("strike").reset_index(drop=True))
    chain["call_iv"] = chain.apply(
        lambda row: _calc_iv_normal(_sf(row.get("call_p")), F, row["strike"], T, r, "C"), axis=1)
    chain["put_iv"] = chain.apply(
        lambda row: _calc_iv_normal(_sf(row.get("put_p")), F, row["strike"], T, r, "P"), axis=1)
    chain["call_iv"] = chain["call_iv"].fillna(chain["put_iv"])
    chain["put_iv"] = chain["put_iv"].fillna(chain["call_iv"])
    # Yield-normal vol (bp/yr) = price-normal vol (points/yr) / DV01 (points per bp)
    if dv01 and np.isfinite(dv01) and dv01 > 0:
        chain["call_yvol"] = chain["call_iv"] / dv01
        chain["put_yvol"] = chain["put_iv"] / dv01
    else:
        chain["call_yvol"] = np.nan
        chain["put_yvol"] = np.nan
    chain["F"] = F
    chain["T"] = T
    chain["fut_symbol"] = fut_sym
    return chain


# ── Rendering ─────────────────────────────────────────────────────────────────
def _chain_html(chain: pd.DataFrame, F: float, dv01_ok: bool) -> str:
    th = "background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;padding:4px 8px;text-align:right"
    th_l = th.replace("text-align:right", "text-align:left")
    td = "font-size:11px;padding:3px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
    hdr = (f"<tr><th style='{th}'>Put</th><th style='{th}'>Put price-vol</th>"
           + (f"<th style='{th}'>Put yield-vol (bp)</th>" if dv01_ok else "")
           + f"<th style='{th_l}'>Strike</th>"
           + (f"<th style='{th}'>Call yield-vol (bp)</th>" if dv01_ok else "")
           + f"<th style='{th}'>Call price-vol</th><th style='{th}'>Call</th></tr>")
    rows = ""
    for _, r in chain.iterrows():
        atm = abs(r["strike"] - F) < 1e-9
        bg = "background:#F1F5F9;" if atm else ""
        cp = f"{r['call_p']:.4f}" if pd.notna(r.get("call_p")) else "—"
        pp = f"{r['put_p']:.4f}" if pd.notna(r.get("put_p")) else "—"
        civ = f"{r['call_iv']:.3f}" if pd.notna(r.get("call_iv")) else "—"
        piv = f"{r['put_iv']:.3f}" if pd.notna(r.get("put_iv")) else "—"
        cyv = f"{r['call_yvol']:.0f}" if pd.notna(r.get("call_yvol")) else "—"
        pyv = f"{r['put_yvol']:.0f}" if pd.notna(r.get("put_yvol")) else "—"
        rows += (f"<tr style='{bg}'><td style='{td}'>{pp}</td><td style='{td}'>{piv}</td>"
                 + (f"<td style='{td}'>{pyv}</td>" if dv01_ok else "")
                 + f"<td style='{td.replace('text-align:right','text-align:left')};font-weight:700'>{r['strike']:.4f}</td>"
                 + (f"<td style='{td}'>{cyv}</td>" if dv01_ok else "")
                 + f"<td style='{td}'>{civ}</td><td style='{td}'>{cp}</td></tr>")
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
            f"font-family:monospace'><thead>{hdr}</thead><tbody>{rows}</tbody></table></div>")


def _rates_moneyness_smile_chart(chain: pd.DataFrame, F: float, T: float, sigma_range: float,
                                 dv01_ok: bool) -> Optional[go.Figure]:
    """Vol smile in σ-moneyness space, mirroring options_v2's equity/commodity convention:
    x = (K − F) / (σ_ATM,price × √T). No ×F needed (unlike the lognormal case) — Bachelier
    price-vol is already in absolute price-points, not a % of F. ATM = 0 always.
    y = whichever vol is currently on display (yield-normal bp if a DV01 is set, else
    price-normal points), using the standard OTM convention (put below ATM, call above)."""
    c = chain.sort_values("strike").copy()
    c["ivol"] = c.apply(
        lambda r: (_sf(r["put_iv"]) if r["strike"] <= F else _sf(r["call_iv"]))
                  or (_sf(r["call_iv"]) if r["strike"] <= F else _sf(r["put_iv"])), axis=1)
    if dv01_ok:
        c["yvol"] = c.apply(
            lambda r: (_sf(r.get("put_yvol")) if r["strike"] <= F else _sf(r.get("call_yvol")))
                      or (_sf(r.get("call_yvol")) if r["strike"] <= F else _sf(r.get("put_yvol"))), axis=1)
        y_col, y_lbl, y_suffix = "yvol", "Yield-normal vol (bp/yr)", " bp"
    else:
        y_col, y_lbl, y_suffix = "ivol", "Price-normal vol (points/yr)", " pts"
    smile = c.dropna(subset=["ivol", y_col])
    if smile.empty:
        return None

    atm_row = smile.iloc[(smile["strike"] - F).abs().argsort()[:1]]
    atm_price_vol = float(atm_row["ivol"].iloc[0])
    if not atm_price_vol or not math.isfinite(atm_price_vol) or atm_price_vol <= 0:
        return None
    sigma_move = atm_price_vol * math.sqrt(T)   # 1σ move in price-points (Bachelier — no ×F)

    x_lim = max(float(sigma_range), 1.0)
    smile = smile.copy()
    smile["mny"] = (smile["strike"] - F) / sigma_move
    smile = smile[(smile["mny"] >= -x_lim * 1.25) & (smile["mny"] <= x_lim * 1.25)]
    if smile.empty:
        return None

    fig = go.Figure(go.Scatter(
        x=smile["mny"], y=smile[y_col], mode="lines+markers",
        line=dict(color="#2563EB", width=2), marker=dict(size=5),
        customdata=smile[["strike", y_col]].values,
        hovertemplate="K=%{customdata[0]:.4f}  " + y_lbl + "=%{customdata[1]:.1f}" + y_suffix
                     + "  %{x:.2f}σ<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color="#94A3B8",
                  annotation_text="ATM", annotation_position="top right",
                  annotation_font_size=10)
    step = 0.5 if x_lim <= 2.5 else 1.0
    tick_vals = [round(v, 2) for v in np.arange(-x_lim, x_lim + 1e-9, step)]
    tick_text = ["ATM" if abs(v) < 0.01 else f"{v:+.1f}σ" for v in tick_vals]
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f"Vol smile — Moneyness (σ units) · {y_lbl}", font=dict(size=12)),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD", font=dict(size=11, color="#1A202C"),
        xaxis=dict(title="Moneyness (σ)", gridcolor="#E8EDF5", zeroline=True,
                  zerolinecolor="#94A3B8", tickvals=tick_vals, ticktext=tick_text,
                  range=[-x_lim * 1.15, x_lim * 1.15]),
        yaxis=dict(title=y_lbl, gridcolor="#E8EDF5"), showlegend=False,
    )
    return fig


# ── Expiry Curve (mirrors options_v2._build_expiry_curve / _curve_chart) ──────
# Cached for the same reason as _build_chain — Vol-Monitor builds 8 of these across
# markets and re-ran them all on every app rerun. `_data`/`_m` skip-hashed.
@st.cache_data(ttl=3600, show_spinner=False)
def _build_expiry_curve_rates(mkt_key: str, sigma: float, _data: dict, _m: dict,
                              dv01: float, n_expiries: int = 6, cache_date: str = "") -> list:
    data, m = _data, _m
    defs  = data.get("opt_defs", pd.DataFrame())
    ohlcv = data.get("opt_ohlcv", pd.DataFrame())
    futs  = data.get("fut_ohlcv", pd.DataFrame())
    if defs.empty or ohlcv.empty or "expiration" not in defs.columns:
        return []

    dv01_ok = bool(dv01 and np.isfinite(dv01) and dv01 > 0)
    today = date.today()
    expiries = sorted({r.date() for r in defs["expiration"].dropna()
                       if 7 <= (r.date() - today).days <= 400})[:n_expiries]

    curve = []
    for exp in expiries:
        try:
            defs_exp = defs[defs["expiration"].dt.date == exp].copy()
            df = defs_exp.merge(ohlcv[["instrument_id", "close"]], on="instrument_id", how="inner")
            if df.empty:
                continue
            F, fut_sym = _resolve_fut_info(exp, defs_exp, futs, mkt_key, m)
            if not F:
                F = _infer_fut_pcp(df, exp, m["r"])
                fut_sym = "pcp"
            if not F:
                continue
            T = max((exp - today).days, 1) / 365.0
            r = m["r"]
            dte = (exp - today).days

            df["_d"] = (df["strike_price"] - F).abs()
            c_rows = df[df["instrument_class"] == "C"].sort_values("_d")
            p_rows = df[df["instrument_class"] == "P"].sort_values("_d")
            if c_rows.empty or p_rows.empty:
                continue
            atm_c, atm_p = c_rows.iloc[0], p_rows.iloc[0]

            iv_c = _calc_iv_normal(_sf(atm_c["close"]), F, atm_c["strike_price"], T, r, "C")
            iv_p = _calc_iv_normal(_sf(atm_p["close"]), F, atm_p["strike_price"], T, r, "P")
            valid = [x for x in [iv_c, iv_p] if x]
            atm_iv = sum(valid) / len(valid) if valid else None

            row = {"dte": dte, "expiry": exp.strftime("%Y-%m-%d"), "F": F,
                  "fut_sym": fut_sym or "", "atm_iv": atm_iv,
                  "atm_yvol": (atm_iv / dv01 if atm_iv and dv01_ok else None)}

            if sigma > 0 and atm_iv:
                # Bachelier: 1σ move is additive in price-points — no ×F (unlike lognormal).
                sigma_move = sigma * atm_iv * math.sqrt(T)
                call_k = F + sigma_move
                put_k  = F - sigma_move

                def _nearest_iv(right: str, target_k: float) -> Optional[float]:
                    sub = df[df["instrument_class"] == right].copy()
                    if sub.empty:
                        return None
                    sub["_dk"] = (sub["strike_price"] - target_k).abs()
                    best = sub.nsmallest(1, "_dk").iloc[0]
                    return _calc_iv_normal(_sf(best["close"]), F, best["strike_price"], T, r, right)

                cwiv = _nearest_iv("C", call_k)
                pwiv = _nearest_iv("P", put_k)
                row["call_wing_k"]  = round(call_k, 3)
                row["put_wing_k"]   = round(put_k, 3)
                row["call_wing_iv"] = cwiv
                row["put_wing_iv"]  = pwiv
                row["call_wing_yvol"] = cwiv / dv01 if cwiv and dv01_ok else None
                row["put_wing_yvol"]  = pwiv / dv01 if pwiv and dv01_ok else None

            curve.append(row)
        except Exception:
            continue

    return curve


def _curve_chart_rates(curve: list, sigma_label: str, dv01_ok: bool) -> go.Figure:
    fig = go.Figure()
    dtes   = [r["dte"] for r in curve]
    labels = [f"{r['expiry']}  {r.get('fut_sym', '')}  F={r['F']:,.2f}" for r in curve]

    if dv01_ok:
        y_key, cw_key, pw_key = "atm_yvol", "call_wing_yvol", "put_wing_yvol"
        y_lbl, y_suffix = "Yield-normal vol (bp)", " bp"
    else:
        y_key, cw_key, pw_key = "atm_iv", "call_wing_iv", "put_wing_iv"
        y_lbl, y_suffix = "Price-normal vol (pts)", " pts"

    atm_ys = [r.get(y_key) for r in curve]
    fig.add_trace(go.Scatter(x=dtes, y=atm_ys, name="ATM", mode="lines+markers",
        line=dict(color="#2563EB", width=2.5), marker=dict(size=7), text=labels,
        hovertemplate="%{text}<br>ATM: <b>%{y:.1f}" + y_suffix + "</b><extra></extra>"))

    if sigma_label != "ATM only":
        call_ys = [r.get(cw_key) for r in curve]
        put_ys  = [r.get(pw_key) for r in curve]
        if any(v for v in call_ys if v):
            fig.add_trace(go.Scatter(x=dtes, y=call_ys, name=f"+{sigma_label} Call",
                mode="lines+markers", line=dict(color="#DC2626", width=1.5, dash="dot"), marker=dict(size=5)))
        if any(v for v in put_ys if v):
            fig.add_trace(go.Scatter(x=dtes, y=put_ys, name=f"-{sigma_label} Put",
                mode="lines+markers", line=dict(color="#059669", width=1.5, dash="dot"), marker=dict(size=5)))

    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
        xaxis=dict(title="DTE", gridcolor="#E8EDF5", zeroline=False),
        yaxis=dict(title=y_lbl, gridcolor="#E8EDF5", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
    )
    return fig


def _curve_table_html_rates(curve: list, sigma: float, dv01_ok: bool) -> str:
    has_wings = sigma > 0
    if dv01_ok:
        atm_key, cw_key, pw_key = "atm_yvol", "call_wing_yvol", "put_wing_yvol"
        fmt = lambda v: f"{v:.1f} bp" if v is not None and math.isfinite(v) else "—"
    else:
        atm_key, cw_key, pw_key = "atm_iv", "call_wing_iv", "put_wing_iv"
        fmt = lambda v: f"{v:.3f} pts" if v is not None and math.isfinite(v) else "—"

    hdr = "<tr><th>Expiry</th><th>DTE</th><th>Contract</th><th>Futures</th><th>ATM</th>"
    if has_wings:
        hdr += f"<th>+{sigma}σ Strike</th><th>+{sigma}σ</th><th>-{sigma}σ Strike</th><th>-{sigma}σ</th>"
    hdr += "</tr>"

    rows = ""
    for r in curve:
        row = (f"<tr><td>{r['expiry']}</td><td>{r['dte']}</td>"
               f"<td style='font-weight:600'>{r.get('fut_sym', '')}</td>"
               f"<td>{r['F']:,.2f}</td><td>{fmt(r.get(atm_key))}</td>")
        if has_wings:
            row += (f"<td>{r.get('call_wing_k', '—')}</td><td>{fmt(r.get(cw_key))}</td>"
                    f"<td>{r.get('put_wing_k', '—')}</td><td>{fmt(r.get(pw_key))}</td>")
        row += "</tr>\n"
        rows += row

    return f"""
<style>
.crv2r{{font-size:12px;border-collapse:collapse;width:100%}}
.crv2r th{{font-size:10px;font-weight:600;color:#64748B;text-transform:uppercase;
           letter-spacing:.5px;padding:5px 10px;border-bottom:2px solid #E2E8F0;text-align:right}}
.crv2r td{{padding:4px 10px;border-bottom:1px solid #F1F5F9;text-align:right;font-size:12px}}
.crv2r tr:hover{{background:#F8FAFC}}
</style>
<table class="crv2r"><thead>{hdr}</thead><tbody>{rows}</tbody></table>"""


# ── Pricer (mirrors options_v2's surface + structure pricer) ──────────────────
# Cached — the full-surface build (12 expiries × per-strike Bachelier IV solves) is the
# single most expensive recompute in the app; date_str+dv01 key it, `_m` skip-hashed.
@st.cache_data(ttl=3600, show_spinner=False)
def _build_surface_data_rates(mkt_key: str, date_str: str, _m: dict, dv01: float) -> dict:
    """Fit a cubic spline smile (in σ-moneyness space — Bachelier-appropriate, unlike the
    log-moneyness used for lognormal markets) for every available expiry."""
    m = _m
    data = _load_data(mkt_key, date_str)
    r = m["r"]
    defs = data.get("opt_defs", pd.DataFrame())
    if defs.empty or "expiration" not in defs.columns:
        return {}

    today = date.today()
    expiries = sorted({ex.date() for ex in defs["expiration"].dropna()
                       if 7 <= (ex.date() - today).days <= 400})[:12]

    surface = {}
    for exp in expiries:
        try:
            chain = _build_chain(mkt_key, exp, 15, 3.0, data, r, dv01, cache_date=date_str)
            if chain.empty or len(chain) < 3:
                continue
            F = float(chain["F"].iloc[0])
            T = float(chain["T"].iloc[0])
            chain = chain.sort_values("strike").copy()
            chain["iv_otm"] = chain.apply(
                lambda row: (_sf(row.get("put_iv")) if row["strike"] <= F else _sf(row.get("call_iv")))
                          or (_sf(row.get("call_iv")) if row["strike"] <= F else _sf(row.get("put_iv"))),
                axis=1)
            smile = chain.dropna(subset=["iv_otm"])
            if len(smile) < 3:
                continue
            strikes = smile["strike"].tolist()
            ivs = smile["iv_otm"].tolist()
            atm_row = smile.iloc[(smile["strike"] - F).abs().argsort()[:1]]
            atm_iv = float(atm_row["iv_otm"].iloc[0])
            if not atm_iv or atm_iv <= 0:
                continue
            sigma_move = atm_iv * math.sqrt(T)
            mny = [(k - F) / sigma_move for k in strikes]
            spline = CubicSpline(mny, ivs, bc_type="natural", extrapolate=False)
            surface[exp] = dict(
                F=F, T=T, r=r, strikes=strikes, ivs=ivs, mny=mny,
                spline=spline, sigma_move=sigma_move,
                min_mny=min(mny), max_mny=max(mny),
                min_strike=min(strikes), max_strike=max(strikes),
            )
        except Exception:
            continue
    return surface


def _iv_from_surface_rates(surface: dict, expiry: date, strike: float) -> Optional[float]:
    s = surface.get(expiry)
    if not s:
        return None
    mny = (strike - s["F"]) / s["sigma_move"]
    mny_c = max(s["min_mny"], min(s["max_mny"], mny))
    iv = float(s["spline"](mny_c))
    return max(iv, 0.0001)


def _price_structure_rates(F: float, T: float, r: float,
                          strikes: list, ivs: list, stype: str) -> dict:
    right = "P" if "Put" in stype else "C"
    qtys = _structure_qtys(stype)
    legs_g = [_bachelier_greeks(F, K, T, r, iv, right) for K, iv in zip(strikes, ivs)]
    result = {k: sum(q * g[k] for q, g in zip(qtys, legs_g))
              for k in ("price", "delta", "gamma", "vega", "theta")}
    result["legs"] = [dict(K=K, iv=iv, qty=q, price=g["price"])
                      for K, iv, q, g in zip(strikes, ivs, qtys, legs_g)]
    return result


def _surface_heatmap_rates(surface: dict, dv01_ok: bool, dv01: float) -> go.Figure:
    """Vol surface: x = σ-moneyness (unified across expiries — Bachelier-appropriate),
    y = DTE."""
    if not surface:
        return go.Figure()
    today = date.today()
    expiries = sorted(surface.keys())
    dtes = [(e - today).days for e in expiries]
    scale = (1.0 / dv01) if dv01_ok else 1.0
    xs = np.linspace(-2.5, 2.5, 80)
    zmat = []
    for exp in expiries:
        s = surface[exp]
        row = []
        for x in xs:
            xc = max(s["min_mny"], min(s["max_mny"], x))
            iv = float(s["spline"](xc))
            row.append(iv * scale)
        zmat.append(row)
    z_lbl = "bp" if dv01_ok else "pts"
    fig = go.Figure(go.Heatmap(
        z=zmat, x=list(xs), y=dtes, colorscale="RdYlGn_r",
        colorbar=dict(title=z_lbl, thickness=14, len=0.8),
        hovertemplate="Moneyness: %{x:.2f}σ<br>DTE: %{y}<br>Vol: <b>%{z:.1f}" + z_lbl + "</b><extra></extra>",
    ))
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text=f"Vol Surface (σ-moneyness vs DTE) · {z_lbl}", font=dict(size=13)),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
        xaxis=dict(title="Moneyness (σ)", gridcolor="#E8EDF5"),
        yaxis=dict(title="DTE", gridcolor="#E8EDF5"),
    )
    return fig


def _smile_with_legs_rates(surface: dict, expiry: date, strike_annotations: list,
                          dv01_ok: bool, dv01: float) -> go.Figure:
    s = surface.get(expiry)
    if not s:
        return go.Figure()
    F = s["F"]
    xs_mny = np.linspace(s["min_mny"] * 1.05, s["max_mny"] * 1.05, 200)
    xs_k = [F + x * s["sigma_move"] for x in xs_mny]
    raw_ys = [float(s["spline"](max(s["min_mny"], min(s["max_mny"], x)))) for x in xs_mny]
    scale = (1.0 / dv01) if dv01_ok else 1.0
    ys = [v * scale for v in raw_ys]
    y_lbl = "Vol (bp)" if dv01_ok else "Vol (pts)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs_k, y=ys, name="Fitted smile", mode="lines",
        line=dict(color="#2563EB", width=2),
        hovertemplate="K=%{x:.3f}<br>Vol=%{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=s["strikes"], y=[v * scale for v in s["ivs"]],
        name="Settlement", mode="markers", marker=dict(size=5, color="#64748B"),
        hovertemplate="K=%{x:.3f}<br>Vol=%{y:.1f}<extra></extra>"))
    for K, label, col in strike_annotations:
        iv = _iv_from_surface_rates(surface, expiry, K)
        if iv:
            fig.add_trace(go.Scatter(
                x=[K], y=[iv * scale], name=label, mode="markers",
                marker=dict(size=12, symbol="diamond", color=col, line=dict(color="#fff", width=1.5)),
                hovertemplate=f"{label}<br>Vol=%{{y:.1f}}<extra></extra>"))
    fig.add_vline(x=F, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"F {F:,.3f}", annotation_font_size=10, annotation_position="top")
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
        xaxis=dict(title="Strike", gridcolor="#E8EDF5", zeroline=False),
        yaxis=dict(title=y_lbl, gridcolor="#E8EDF5", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
    )
    return fig


def _render_chain_tab_rates(mkt: str, data: dict, m: dict, dv01: float, dv01_ok: bool,
                            tdate_str: str = ""):
    defs = data.get("opt_defs", pd.DataFrame())
    if defs.empty or "expiration" not in defs.columns:
        st.warning("No option data returned. Check API key / availability.")
        if data.get("defs_err"):
            with st.expander("Debug", expanded=True):
                st.code(data["defs_err"])
        return
    today = date.today()
    exp_dates = sorted({r.date() for r in defs["expiration"].dropna() if (r.date() - today).days >= 1})[:8]
    if not exp_dates:
        st.warning("No upcoming expiries found.")
        return

    # Widget keys (including the form's own key) are parameterized by `mkt` — otherwise
    # Streamlit treats "Option Expiry"/"Strikes per side"/"Range (σ)" as the SAME widget
    # across every market (identical label+position => identical implicit key), which can
    # make the form appear frozen/stale when switching markets instead of resetting.
    _idx_clamped = min(st.session_state.get(f"_ro_exp_idx_{mkt}", 0), len(exp_dates) - 1)
    with st.form(f"ro_chain_form_{mkt}"):
        exp_labels = {e: f"{e.strftime('%a %b %d %Y')}  ({(e - today).days} DTE)" for e in exp_dates}
        sel_exp = st.selectbox("Option Expiry", list(exp_labels.keys()), format_func=lambda e: exp_labels[e],
                               index=_idx_clamped, key=f"_ro_expsel_{mkt}")
        c1, c2, c3 = st.columns([2, 2, 1])
        n_per_side = c1.select_slider("Strikes per side", options=[2, 3, 5, 7, 10, 14],
                                      value=st.session_state.get(f"_ro_nps_{mkt}", 5),
                                      key=f"_ro_npssel_{mkt}")
        sigma_range = c2.select_slider("Range (σ)", options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                                       value=st.session_state.get(f"_ro_sig_{mkt}", 2.0),
                                       format_func=lambda x: f"{x}σ", key=f"_ro_sigsel_{mkt}")
        submitted = c3.form_submit_button("⟳  Refresh", use_container_width=True,
                                          key=f"_ro_refresh_{mkt}")

    if submitted:
        st.session_state[f"_ro_exp_idx_{mkt}"] = exp_dates.index(sel_exp)
        st.session_state[f"_ro_nps_{mkt}"] = n_per_side
        st.session_state[f"_ro_sig_{mkt}"] = sigma_range
        st.session_state[f"_ro_exp_active_{mkt}"] = sel_exp
        st.session_state[f"_ro_nps_active_{mkt}"] = n_per_side
        st.session_state[f"_ro_sig_active_{mkt}"] = sigma_range

    exp_active = st.session_state.get(f"_ro_exp_active_{mkt}", sel_exp)
    nps_active = st.session_state.get(f"_ro_nps_active_{mkt}", n_per_side)
    sig_active = st.session_state.get(f"_ro_sig_active_{mkt}", sigma_range)

    with st.spinner("Building chain..."):
        chain = _build_chain(mkt, exp_active, nps_active, sig_active, data, m["r"], dv01,
                             cache_date=tdate_str)

    if chain.empty:
        st.warning("No chain data for selected expiry.")
        return

    F = float(chain["F"].iloc[0])
    T = float(chain["T"].iloc[0])
    fut_lbl = str(chain["fut_symbol"].iloc[0])
    st.caption(f"Underlying future: **{fut_lbl}** &nbsp;|&nbsp; Settle used for ATM: **{F:,.4f}**"
              + (f" &nbsp;|&nbsp; DV01 **{dv01:.4f}** pts/bp" if dv01_ok else " &nbsp;|&nbsp; "
                 "⚠ yield-vol needs a DV01 (set the CTD assumption above)"))
    st.markdown(_chain_html(chain, F, dv01_ok), unsafe_allow_html=True)

    # ── Smile chart ───────────────────────────────────────────────────────────
    if dv01_ok:
        smile = chain.dropna(subset=["call_yvol", "put_yvol"], how="all").copy()
        smile["yvol"] = smile["put_yvol"].where(smile["strike"] <= F, smile["call_yvol"])
        y_lbl, y_suffix = "Yield-normal vol (bp/yr)", " bp"
        y_col = "yvol"
    else:
        smile = chain.dropna(subset=["call_iv", "put_iv"], how="all").copy()
        smile["ivol"] = smile["put_iv"].where(smile["strike"] <= F, smile["call_iv"])
        y_lbl, y_suffix = "Price-normal vol (points/yr)", ""
        y_col = "ivol"
    if not smile.empty:
        fig = go.Figure(go.Scatter(
            x=smile["strike"], y=smile[y_col], mode="lines+markers",
            line=dict(color="#2563EB", width=2),
            hovertemplate="Strike %{x}: %{y:.1f}" + y_suffix + "<extra></extra>"))
        fig.add_vline(x=F, line_dash="dot", line_color="#94A3B8", annotation_text="F (ATM)")
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text=f"Vol smile — {y_lbl}", font=dict(size=12)),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD", font=dict(size=11, color="#1A202C"),
            xaxis=dict(title="Strike", gridcolor="#E8EDF5"),
            yaxis=dict(title=y_lbl, gridcolor="#E8EDF5"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Smile chart — moneyness (σ units), matching the equity/commodity convention ──
    mny_fig = _rates_moneyness_smile_chart(chain, F, T, sig_active, dv01_ok)
    if mny_fig is not None:
        st.plotly_chart(mny_fig, use_container_width=True)


def _render_curve_tab_rates(mkt: str, tdate_str: str, m: dict, dv01: float, dv01_ok: bool):
    if not dv01_ok:
        st.info("Set a DV01 in the CTD box above to see this curve in yield-normal (bp) "
               "terms — showing price-normal (points) for now.")
    with st.form(f"ro_curve_form_{mkt}"):
        c1, c2, c3 = st.columns([2, 2, 1])
        sigma_label = c1.selectbox("Vol wings", list(_STD_SIGMA_OPTS.keys()),
                                   index=st.session_state.get(f"_ro_curve_sigidx_{mkt}", 0),
                                   key=f"_ro_curve_sigsel_{mkt}")
        n_exp = c2.select_slider("Expiries", options=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                                 value=st.session_state.get(f"_ro_curve_nexp_{mkt}", 6),
                                 key=f"_ro_curve_nexpsel_{mkt}")
        curve_submitted = c3.form_submit_button("⟳  Refresh", key=f"_ro_curve_refresh_{mkt}")

    if curve_submitted:
        st.session_state[f"_ro_curve_sigidx_{mkt}"]   = list(_STD_SIGMA_OPTS.keys()).index(sigma_label)
        st.session_state[f"_ro_curve_sigactive_{mkt}"] = sigma_label
        st.session_state[f"_ro_curve_nexp_{mkt}"]      = n_exp

    sigma_active = st.session_state.get(f"_ro_curve_sigactive_{mkt}", sigma_label)
    n_exp_active = st.session_state.get(f"_ro_curve_nexp_{mkt}", 6)
    sigma = _STD_SIGMA_OPTS[sigma_active]

    with st.spinner("Building expiry curve..."):
        data_curve = _load_data(mkt, tdate_str)
        curve = _build_expiry_curve_rates(mkt, sigma, data_curve, m, dv01,
                                          n_expiries=n_exp_active, cache_date=tdate_str)

    if not curve:
        st.warning("No expiry curve data available.")
        return
    if not any(r.get("atm_iv") for r in curve):
        st.warning("Could not compute IVs for any expiry (check data availability).")
        return

    st.plotly_chart(_curve_chart_rates(curve, sigma_active, dv01_ok), use_container_width=True)
    st.markdown(_curve_table_html_rates(curve, sigma, dv01_ok), unsafe_allow_html=True)


def _render_pricer_tab_rates(mkt: str, tdate_str: str, m: dict, dv01: float, dv01_ok: bool):
    st.markdown("#### Rates Options Pricer")
    st.caption(
        "Builds a synthetic vol surface (cubic spline over σ-moneyness — Bachelier-"
        "appropriate) from settlement vols. Choose a structure, enter strikes, and hit Price."
    )

    with st.spinner("Building vol surface..."):
        surface = _build_surface_data_rates(mkt, tdate_str, m, dv01)

    if not surface:
        st.warning("Could not build vol surface — no settlement vol data available.")
        return

    expiries = sorted(surface.keys())
    today = date.today()

    st.plotly_chart(_surface_heatmap_rates(surface, dv01_ok, dv01), use_container_width=True)
    st.divider()

    col_stype, col_exp = st.columns([2, 3])
    with col_stype:
        stype = st.selectbox("Structure", _STRUCTURE_TYPES, key=f"_ro_pricer_stype_{mkt}")
    with col_exp:
        exp_labels = {e: f"{e.strftime('%a %b %d %Y')}  ({(e - today).days} DTE)" for e in expiries}
        sel_exp = st.selectbox("Expiry", expiries, format_func=lambda e: exp_labels[e],
                               key=f"_ro_pricer_exp_{mkt}")

    s = surface[sel_exp]
    F, T, r = s["F"], s["T"], s["r"]

    # Round default strikes to a sensible tick — the smallest gap actually seen in the
    # listed strikes for this expiry (rather than options_v2's fixed "nearest 25", which
    # assumes equity-index-sized strikes).
    strikes_sorted = sorted(set(s["strikes"]))
    tick = min((b - a for a, b in zip(strikes_sorted, strikes_sorted[1:])), default=0.5)
    tick = tick if tick > 0 else 0.5

    def _rtick(x: float) -> float:
        return float(round(x / tick) * tick)

    _defaults: dict[str, list[float]] = {
        "Outright Call":  [_rtick(F)],
        "Outright Put":   [_rtick(F)],
        "Call Spread":    [_rtick(F - 5 * tick), _rtick(F + 5 * tick)],
        "Put Spread":     [_rtick(F - 5 * tick), _rtick(F + 5 * tick)],
        "Call Butterfly": [_rtick(F - 10 * tick), _rtick(F),            _rtick(F + 10 * tick)],
        "Put Butterfly":  [_rtick(F - 10 * tick), _rtick(F),            _rtick(F + 10 * tick)],
        "Call Condor":    [_rtick(F - 15 * tick), _rtick(F - 5 * tick), _rtick(F + 5 * tick), _rtick(F + 15 * tick)],
        "Put Condor":     [_rtick(F - 15 * tick), _rtick(F - 5 * tick), _rtick(F + 5 * tick), _rtick(F + 15 * tick)],
    }
    labels   = _STRIKE_LABELS[stype]
    defaults = _defaults[stype]
    n_legs   = len(labels)

    strike_cols = st.columns(n_legs)
    strikes: list[float] = []
    for i, (col, lbl, dflt) in enumerate(zip(strike_cols, labels, defaults)):
        with col:
            strikes.append(st.number_input(lbl, value=dflt, step=tick, format="%.3f",
                                           key=f"_ro_pricer_k{i}_{mkt}"))

    qtys = _structure_qtys(stype)
    leg_colors = ["#059669" if q > 0 else "#DC2626" for q in qtys]
    annotations = [(K, f"K{i+1}={K:,.3f} ({'L' if q>0 else 'S'}{abs(q) if abs(q)>1 else ''})", col)
                  for i, (K, q, col) in enumerate(zip(strikes, qtys, leg_colors))]
    st.plotly_chart(_smile_with_legs_rates(surface, sel_exp, annotations, dv01_ok, dv01),
                    use_container_width=True)

    if st.button("▶  Price", key=f"_ro_pricer_run_{mkt}"):
        if sorted(strikes) != strikes or len(set(strikes)) < n_legs:
            st.error("Strikes must be in strictly increasing order (K1 < K2 < …).")
        else:
            ivs = [_iv_from_surface_rates(surface, sel_exp, K) for K in strikes]
            if any(iv is None for iv in ivs):
                st.error("Could not interpolate vol for one or more strikes.")
            else:
                result = _price_structure_rates(F, T, r, strikes, ivs, stype)

                def _row(label, val):
                    return (f"<tr><td>{label}</td>"
                            f"<td style='text-align:right'><b>{val}</b></td></tr>")

                # vega is computed w.r.t. price-vol (points); "per 1bp of yield-vol" =
                # vega_per_point × DV01 (since price_vol = yield_vol × DV01).
                vega_scale = dv01 if dv01_ok else 1.0
                vega_unit  = "1bp yield-vol" if dv01_ok else "1pt price-vol"

                rows_html = _row("Structure", stype)
                rows_html += _row("Forward (F)", f"{F:,.3f}")
                rows_html += _row("DTE", f"{int(T * 365)}")
                rows_html += _row("Price (pts)", f"{result['price']:.4f}")
                if n_legs == 2:
                    width = abs(strikes[1] - strikes[0])
                    rows_html += _row("Price (% of width)", f"{result['price'] / width * 100:.1f}%")
                elif "Butterfly" in stype:
                    wing = strikes[2] - strikes[0]
                    rows_html += _row("Price (% of wing width)", f"{result['price'] / wing * 100:.2f}%")
                rows_html += _row("Delta", f"{result['delta']:.4f}")
                rows_html += _row("Gamma", f"{result['gamma']:.6f}")
                rows_html += _row(f"Vega (per {vega_unit})", f"{result['vega'] * vega_scale:.4f}")
                rows_html += _row("Theta (per day)", f"{result['theta']:.4f}")

                st.markdown(f"""
<style>
.pricer-tbl-r {{border-collapse:collapse;width:360px;font-size:13px}}
.pricer-tbl-r td {{padding:4px 10px;border-bottom:1px solid #334155;color:#CBD5E1}}
.pricer-tbl-r tr:first-child td {{color:#F1F5F9;font-size:14px;border-bottom:2px solid #475569}}
</style>
<table class="pricer-tbl-r">{rows_html}</table>""", unsafe_allow_html=True)

                with st.expander("Leg detail", expanded=False):
                    for i, leg in enumerate(result["legs"]):
                        _v = f"{leg['iv']*dv01:.1f} bp" if dv01_ok else f"{leg['iv']:.3f} pts"
                        st.caption(f"K{i+1}={leg['K']:,.3f}  qty={leg['qty']:+d}  "
                                  f"vol={_v}  leg price={leg['price']:.4f}")


# ── Vol-Monitor: ATM yield-vol grid, aggregated across all built markets ──────
_VOL_MONITOR_TENORS  = [("1m", 30), ("2m", 60), ("3m", 90)]


def _get_market_dv01(mkt: str, m: dict) -> tuple[float, bool]:
    """DV01 for a market independent of whichever market is selected in the top dropdown —
    reuses that market's own CTD-box widget state if it's already been configured
    (st.session_state persists across tabs/fragments within a session), else computes a
    fresh default from the config."""
    if "fixed_dv01" in m:
        v = m["fixed_dv01"]
        return float(v), bool(v and np.isfinite(v) and v > 0)
    if not m.get("needs_ctd"):
        return float("nan"), False
    key = f"_ro_dv01_{mkt}"
    if key in st.session_state:
        v = st.session_state[key]
        if v and np.isfinite(v) and v > 0:
            return float(v), True
    ctd_years = st.session_state.get(f"_ro_ctdyrs_{mkt}", m["ctd_years"])
    dy, dc = _default_ctd_inputs(ctd_years, curve=m["curve"])
    _, _, _, dv01_est = _estimate_dv01(dc / 100, dy / 100, ctd_years, freq=m["freq"])
    return float(dv01_est), bool(dv01_est and np.isfinite(dv01_est) and dv01_est > 0)


def _interp_curve_at_dte(curve: list, target_dte: int, key: str) -> Optional[float]:
    """Linear interpolation (clamped at the ends) of `key` (e.g. 'atm_yvol') vs DTE."""
    pts = sorted((r["dte"], r[key]) for r in curve if r.get(key) is not None)
    if not pts:
        return None
    if target_dte <= pts[0][0]:
        return pts[0][1]
    if target_dte >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= target_dte <= x1:
            w = (target_dte - x0) / (x1 - x0)
            return y0 + w * (y1 - y0)
    return pts[-1][1]


def _nearest_expiry_row(curve: list, target: date, max_gap_days: int = 15) -> Optional[dict]:
    """Row in `curve` whose 'expiry' date is closest to `target` (within max_gap_days).
    Used to align each market's own listed expiry to a shared reference date — needed
    because some markets (e.g. Eurex Bund/RX) list dense weekly expiries for the front
    couple of months alongside the standard monthly ones, so "take the Nth listed expiry"
    doesn't reliably pick the monthly-aligned date the way it does for CME's clean
    monthly-only serials (TY/FV/UB)."""
    if not curve:
        return None
    best, best_gap = None, None
    for r in curve:
        try:
            d = date.fromisoformat(r["expiry"])
        except Exception:
            continue
        gap = abs((d - target).days)
        if best_gap is None or gap < best_gap:
            best, best_gap = r, gap
    if best is not None and best_gap is not None and best_gap <= max_gap_days:
        return best
    return None


def _vm_table_html(hdr_lbl: str, row_labels: list, markets: list,
                   cell_fn, sub_fn=None) -> str:
    """Generic compact table: rows=row_labels, cols=markets. cell_fn(mkt, row_idx) -> str|None
    formatted value; sub_fn(mkt, row_idx) -> optional small sub-line (e.g. the actual date)."""
    th  = "background:#1E293B;color:#F8FAFC;font-size:12px;font-weight:600;padding:6px 14px;text-align:right;white-space:nowrap"
    thl = th.replace("text-align:right", "text-align:left")
    td  = "font-size:12px;padding:6px 14px;border-bottom:1px solid #E2E8F0;text-align:right;white-space:nowrap"
    tdl = td.replace("text-align:right", "text-align:left")
    hdr = f"<tr><th style='{thl}'>{hdr_lbl}</th>" + "".join(f"<th style='{th}'>{mkt}</th>" for mkt in markets) + "</tr>"
    body = ""
    for i, lbl in enumerate(row_labels):
        body += f"<tr><td style='{tdl}'><b>{lbl}</b></td>"
        for mkt in markets:
            val = cell_fn(mkt, i)
            sub = sub_fn(mkt, i) if sub_fn else None
            if val is None:
                body += f"<td style='{td};color:#94A3B8'>—</td>"
            else:
                sub_html = f"<br><span style='font-size:10px;color:#94A3B8'>{sub}</span>" if sub else ""
                body += f"<td style='{td};font-weight:700'>{val}{sub_html}</td>"
        body += "</tr>"
    # No width:100% — let the table size to its content instead of stretching across the
    # whole page with wide, mostly-empty columns.
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;"
           f"font-family:monospace'><thead>{hdr}</thead><tbody>{body}</tbody></table></div>")


def _vm_load_gate(group_id: str, n_markets: int) -> bool:
    """Lazy-load gate for the Vol-Monitor sub-tabs. Building the monitor loads EVERY
    market in the group (~40s each on a cold day-cache — several minutes total), and
    st.tabs executes all tab bodies on every rerun — without this gate, merely
    selecting a market in the Chain tab stalls the whole app behind the monitor's
    multi-market fetch (hit 2026-07-28)."""
    key = f"_vm_loaded_{group_id}"
    if st.session_state.get(key):
        return True
    _, col, _ = st.columns([1, 2, 1])
    with col:
        if st.button(
                f"Load Vol-Monitor ({n_markets} markets — first load of the day can "
                f"take a few minutes; instant once cached)",
                key=f"_vm_load_btn_{group_id}", use_container_width=True):
            st.session_state[key] = True
            st.rerun()
    return False


def _render_vol_monitor_tab(markets: list, anchor_mkt: str):
    if not _vm_load_gate("bond", len(markets)):
        return
    st.markdown("#### Vol-Monitor &nbsp;·&nbsp; ATM term structure across markets")
    st.caption(
        f"Aggregated **ATM yield-normal vol (bp)** across all {len(markets)} markets in this "
        "group. Uses each market's current CTD/DV01 assumption from its own Chain tab if "
        "you've already set one there (session-persisted), otherwise a fresh default "
        "estimate. Values are directly comparable in bp vol-space even across currencies "
        "(and, for the money-market group, no CTD conversion applies at all)."
    )

    curves: dict[str, list] = {}
    keys: dict[str, str] = {}
    notes = []
    with st.spinner(f"Loading all {len(markets)} markets..."):
        for mkt in markets:
            m = _MARKETS_RATES[mkt]
            tdate = _trade_date(m["ds"])
            tdate_str = str(tdate)
            data = _load_data(mkt, tdate_str)
            _derr = data.get("defs_err", "")
            _defs_empty = (data.get("opt_defs") is None
                           or getattr(data.get("opt_defs"), "empty", True))
            if _defs_empty or (_derr and ("dataset_unavailable_range" in _derr
                                          or "data_end_after_available_end" in _derr
                                          or "license_not_found_unauthorized" in _derr)):
                # Session not yet in Databento historical (Eurex rolls at 22:00 UTC) or a
                # license/availability error — fall back one business day.
                tdate_str = str(_prev_bday(tdate, 1))
                data = _load_data(mkt, tdate_str)

            dv01, dv01_ok = _get_market_dv01(mkt, m)
            key = "atm_yvol" if dv01_ok else "atm_iv"
            curve = _build_expiry_curve_rates(mkt, 0.0, data, m, dv01, n_expiries=8,
                                              cache_date=tdate_str)
            curves[mkt] = curve
            keys[mkt] = key
            if not curve or not any(r.get(key) is not None for r in curve):
                notes.append(f"{mkt}: no usable curve data")

    # ── Table 1: interpolated at fixed calendar tenors (1m/2m/3m) ────────────
    def _cell_interp(mkt, i):
        lbl, days = _VOL_MONITOR_TENORS[i]
        v = _interp_curve_at_dte(curves.get(mkt, []), days, keys.get(mkt, "atm_yvol"))
        return f"{v:.1f} bp" if v is not None else None

    st.markdown(_vm_table_html("Tenor", [lbl for lbl, _ in _VOL_MONITOR_TENORS],
                              markets, _cell_interp), unsafe_allow_html=True)

    with st.expander("DV01 / CTD assumption used per market", expanded=False):
        for mkt in markets:
            m = _MARKETS_RATES[mkt]
            dv01, dv01_ok = _get_market_dv01(mkt, m)
            src = "your Chain-tab override" if f"_ro_dv01_{mkt}" in st.session_state else "default estimate"
            st.caption(f"**{mkt}**: DV01 = {dv01:.4f} pts/bp ({src})" if dv01_ok else f"**{mkt}**: no DV01")
    if notes:
        st.caption("⚠ " + "; ".join(notes))

    # ── Table 2: actual 1st/2nd/3rd MONTHLY expiry (no interpolation) ─────────
    # Anchor on TY's dates (CME lists clean monthly-only serials, no weekly clutter) and
    # find each market's own nearest listed expiry to those — needed because some markets
    # (Eurex Bund/RX in particular) list dense WEEKLY expiries for the front couple of
    # months, so naively taking "the next 3 listed expiries" would pick weeklies for RX,
    # not the monthly dates that actually align with CME's.
    st.markdown("---")
    st.markdown("##### By actual monthly expiry (1st / 2nd / 3rd upcoming)")
    st.caption(
        f"Same markets, but instead of interpolating at a fixed day-count, this reads the "
        f"**ATM vol observed directly** off each market's next three MONTHLY expiries, "
        f"anchored on **{anchor_mkt}**'s dates (a clean monthly-only serial, no weekly "
        f"clutter) — some markets also list dense weekly expiries for the front couple of "
        f"months, so \"take the next 3 listed\" would otherwise pick weeklies there instead "
        f"of the monthly-aligned date. A market's actual date only appears under its value "
        f"when it **differs** from the anchor (an exact match needs no callout); a missing "
        f"cell means no close monthly match that period."
    )

    anchor_curve = curves.get(anchor_mkt) or next((curves[k] for k in markets if curves.get(k)), [])
    anchor_dates = [date.fromisoformat(r["expiry"]) for r in anchor_curve[:3]]

    def _cell_actual(mkt, i):
        if i >= len(anchor_dates):
            return None
        row = _nearest_expiry_row(curves.get(mkt, []), anchor_dates[i])
        if not row:
            return None
        v = row.get(keys.get(mkt, "atm_yvol"))
        return f"{v:.1f} bp" if v is not None else None

    def _sub_actual(mkt, i):
        # Only surface the actual date when it DIFFERS from the anchor — that's the only
        # case worth a reader's attention (an exact match is already shown by the row label).
        if i >= len(anchor_dates):
            return None
        row = _nearest_expiry_row(curves.get(mkt, []), anchor_dates[i])
        if not row:
            return "no match"
        if row["expiry"] == anchor_dates[i].isoformat():
            return None
        return f"actual: {row['expiry']} ({row['dte']}d)"

    st.markdown(_vm_table_html(f"Anchor ({anchor_mkt})", [d.isoformat() for d in anchor_dates],
                              markets, _cell_actual, _sub_actual),
               unsafe_allow_html=True)


def _mm_monitor_row_dates(anchor_curve: list, today: date) -> list:
    """Row expiries for the money-market monitor: the anchor's first 4 listed (serial/
    monthly) expiries, then only its quarterly-cycle (Mar/Jun/Sep/Dec) expiries out to
    ~370 days — e.g. Aug, Sep, Oct, Nov, Dec, Mar-27, Jun-27."""
    dates = []
    for r in sorted(anchor_curve, key=lambda x: x["dte"]):
        try:
            dates.append(date.fromisoformat(r["expiry"]))
        except Exception:
            continue
    if not dates:
        return []
    selected = list(dates[:4])
    for d in dates[4:]:
        if d.month in _QUARTERLY_MONTHS and (d - today).days <= 370 and d not in selected:
            selected.append(d)
    return sorted(set(selected))


def _render_mm_vol_monitor_tab(markets: list, anchor_mkt: str):
    """Money-market Vol-Monitor: a compact expiry (rows) × market (cols) grid of ATM
    yield-normal vol (bp). Rows come from the anchor (SOFR) — first 4 serials then
    quarterlies to ~1y; each other market's cell is read off its own nearest listed expiry
    (ICE and CME expiry dates differ by a day or two, so a small tolerance is applied)."""
    if not _vm_load_gate("mm", len(markets)):
        return
    st.markdown("#### Vol-Monitor &nbsp;·&nbsp; Money-market ATM yield-vol grid")
    st.caption(
        "**ATM yield-normal vol (bp)** for every money-market market — front and 1y/2y "
        "midcurves — across a shared expiry ladder. Rows are the **first 4 serial/monthly** "
        f"expiries of the anchor (**{anchor_mkt}**) followed by its **quarterly** expiries out "
        "to ~1 year. Each cell is that market's ATM vol at its own nearest listed expiry "
        "(±a few days across ICE/CME); a market's actual date is shown under the value only "
        "when it differs from the anchor's. Rate contracts are quoted price = 100 − rate, so "
        "price-vol already **is** yield-vol (×100) — directly comparable across currencies."
    )

    order = [k for k in _MONEY_MKT_MARKETS if k in markets]
    labels = [_MARKETS_RATES[k].get("vm_label", k) for k in order]

    curves: dict[str, list] = {}
    notes = []
    with st.spinner(f"Loading all {len(order)} money-market markets..."):
        for mkt in order:
            m = _MARKETS_RATES[mkt]
            tdate_str = str(_trade_date(m["ds"]))
            data = _load_data(mkt, tdate_str)
            _derr = data.get("defs_err", "")
            _defs_empty = (data.get("opt_defs") is None
                           or getattr(data.get("opt_defs"), "empty", True))
            if _defs_empty or (_derr and ("dataset_unavailable_range" in _derr
                                          or "data_end_after_available_end" in _derr
                                          or "license_not_found_unauthorized" in _derr)):
                tdate_str = str(_prev_bday(_trade_date(m["ds"]), 1))
                data = _load_data(mkt, tdate_str)
            dv01, _ = _get_market_dv01(mkt, m)
            # n_expiries deliberately large: the builder returns the sorted expiries within
            # 7..400 days, and we need serials + quarterlies out to ~1y — not just the first
            # 6 (which would be all serials and reach no quarterly beyond the front months).
            curves[mkt] = _build_expiry_curve_rates(mkt, 0.0, data, m, dv01, n_expiries=16,
                                                    cache_date=tdate_str)
            if not curves[mkt] or not any(r.get("atm_yvol") is not None for r in curves[mkt]):
                notes.append(f"{_MARKETS_RATES[mkt].get('vm_label', mkt)}: no usable data")

    today = date.today()
    anchor_curve = curves.get(anchor_mkt) or next((curves[k] for k in order if curves.get(k)), [])
    row_dates = _mm_monitor_row_dates(anchor_curve, today)
    if not row_dates:
        st.warning("No expiry ladder available for the anchor market.")
        return

    def _cell(label, i):
        mkt = order[labels.index(label)]
        row = _nearest_expiry_row(curves.get(mkt, []), row_dates[i])
        if not row:
            return None
        v = row.get("atm_yvol")
        return f"{v:.1f} bp" if v is not None else None

    def _sub(label, i):
        mkt = order[labels.index(label)]
        row = _nearest_expiry_row(curves.get(mkt, []), row_dates[i])
        if not row:
            return "no match"
        if row["expiry"] == row_dates[i].isoformat():
            return None
        return f"{row['expiry']} ({row['dte']}d)"

    row_labels = [d.strftime("%b '%y") for d in row_dates]
    st.markdown(_vm_table_html("Expiry", row_labels, labels, _cell, _sub),
                unsafe_allow_html=True)

    st.caption(f"Anchor: **{anchor_mkt}** — rows {row_labels[0]} … {row_labels[-1]} "
               f"({len(row_labels)} expiries). Blank cell = no close expiry match for that market.")
    if notes:
        st.caption("⚠ " + "; ".join(notes))


def _render_market_group(group_keys: list, group_id: str, vol_anchor_mkt: str):
    """Everything for one market group (market selector + CTD box + Chain/Expiry Curve/
    Pricer/Vol-Monitor sub-tabs). Called once per top-level tab (Bond Future-Options,
    Money Mkt-Options) — group_id keeps widget keys distinct between the two calls."""
    mkt_labels = {k: _MARKETS_RATES[k]["label"] for k in group_keys}
    mkt = st.selectbox("Market", list(mkt_labels.keys()), format_func=lambda k: mkt_labels[k],
                       key=f"_ro_mkt_sel_{group_id}")
    m = _MARKETS_RATES[mkt]
    tdate = _trade_date(m["ds"])
    tdate_str = str(tdate)

    with st.spinner("Loading market data — first load of the day can take 60-90s "
                    "(this market's option chain has thousands of listed strikes); "
                    "cached after that, so later reloads are instant..."):
        data = _load_data(mkt, tdate_str)
        _derr = data.get("defs_err", "")
        _defs_empty = (data.get("opt_defs") is None
                       or getattr(data.get("opt_defs"), "empty", True))
        if _defs_empty or (_derr and ("dataset_unavailable_range" in _derr
                                      or "data_end_after_available_end" in _derr
                                      or "license_not_found_unauthorized" in _derr)):
            # Session not yet in Databento historical (Eurex rolls at 22:00 UTC) or a
            # license/availability error — fall back one business day.
            tdate = _prev_bday(tdate, 1)
            tdate_str = str(tdate)
            data = _load_data(mkt, tdate_str)
    st.caption(f"Data: Databento settlement — {tdate.strftime('%a %b %d, %Y')}")

    # ── CTD assumption box (editable) — shared across Chain / Expiry Curve / Pricer ──
    # Bond-future markets need the full editable CTD/duration estimate (needs_ctd=True).
    # Money-market contracts are rate-quoted (price = 100 − rate) — price-vol IS yield-vol,
    # just scaled ×100 (1 price-point = 100bp), no CTD/duration involved — fixed_dv01=0.01
    # reuses the SAME yield_vol = price_vol/dv01 machinery to get that scaling "for free".
    dv01 = float("nan")
    if m.get("needs_ctd"):
        with st.expander("⚙️  Assumed CTD & duration (editable)", expanded=False):
            _lo, _hi = m.get("deliverable_window", (1.0, 31.0))
            st.caption(
                f"Per CME's *Understanding Treasury Futures* reference: when yields are **below** "
                f"the 6% notional coupon (as now), the conversion-factor bias favours a **short-"
                f"duration** note as CTD — i.e. toward the **low end** of the {_lo:g}–{_hi:g}y "
                f"deliverable window, not a freshly-issued note near the top of it. Default below "
                f"reflects that bias; override if you have a better CTD estimate."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                ctd_years = st.number_input("Assumed CTD maturity (yrs)", value=float(m["ctd_years"]),
                                           min_value=1.0, max_value=31.0, step=0.25, key=f"_ro_ctdyrs_{mkt}")
            dy, dc = _default_ctd_inputs(ctd_years, curve=m["curve"])
            with c2:
                ctd_coupon = st.number_input("Assumed coupon (%)", value=float(dc), min_value=0.0,
                                            max_value=15.0, step=0.125, key=f"_ro_ctdcpn_{mkt}") / 100.0
            with c3:
                ctd_yield = st.number_input("Assumed yield (%)", value=float(dy), min_value=0.0,
                                           max_value=20.0, step=0.05, key=f"_ro_ctdyld_{mkt}") / 100.0
            price, mod_dur, cf, dv01_est = _estimate_dv01(ctd_coupon, ctd_yield, ctd_years, freq=m["freq"])
            c4, c5 = st.columns(2)
            with c4:
                _freq_lbl = "semiannual" if m["freq"] == 2 else "annual"
                st.caption(f"→ Price **{price:.2f}** · Modified Duration **{mod_dur:.2f}y** · "
                          f"Conversion Factor **{cf:.4f}** ({_freq_lbl} coupon)")
            with c5:
                dv01 = st.number_input("DV01 override (price-points per bp)", value=float(dv01_est),
                                       min_value=0.0001, step=0.001, format="%.4f",
                                       key=f"_ro_dv01_{mkt}",
                                       help="Pre-filled from the assumed CTD above. Overwrite if you "
                                            "have the actual current CTD's duration/DV01.")
            _curve_lbl = "ECB EUR" if m["curve"] == "eur" else "UST"
            st.caption(f"Estimate only — yield/coupon interpolated off the {_curve_lbl} curve at the "
                      "assumed maturity; maturity itself is a judgment call, not an identified CTD. "
                      "Override any input, including the DV01 directly. **Yield-vol scales as "
                      "1/DV01** — the absolute bp level below is sensitive to this assumption; the "
                      "shape of the smile and its move over time are far more robust than the level.")
    elif "fixed_dv01" in m:
        dv01 = m["fixed_dv01"]
        st.caption(
            f"**{m['label']}** is rate-quoted (price = 100 − rate) — price-vol already **is** "
            f"yield-vol, just ×{1/dv01:.0f} scaled (1 price-point = {1/dv01:.0f}bp). No CTD/"
            f"duration conversion needed; nothing to configure here."
        )

    dv01_ok = bool(dv01 and np.isfinite(dv01) and dv01 > 0)

    tab_chain, tab_curve, tab_pricer, tab_monitor = st.tabs(
        ["Chain", "Expiry Curve", "Pricer", "Vol-Monitor"])
    with tab_chain:
        _render_chain_tab_rates(mkt, data, m, dv01, dv01_ok, tdate_str=tdate_str)
    with tab_curve:
        _render_curve_tab_rates(mkt, tdate_str, m, dv01, dv01_ok)
    with tab_pricer:
        _render_pricer_tab_rates(mkt, tdate_str, m, dv01, dv01_ok)
    with tab_monitor:
        if group_id == "mm":
            _render_mm_vol_monitor_tab(group_keys, vol_anchor_mkt)
        else:
            _render_vol_monitor_tab(group_keys, vol_anchor_mkt)


@st.fragment
def render_rates_options():
    with st.expander("📊  Databento usage & costs", expanded=False):
        try:
            import data_costs
            data_costs.render_panel()
        except Exception:
            st.caption("cost ledger unavailable")

    st.markdown("#### 📐 Rates-Vol &nbsp;·&nbsp; Normal (Bachelier) vol")
    st.caption(
        "Rates options use a **normal**, not lognormal, vol model. Implied vol is computed "
        "directly on the futures price (Bachelier) as **price-vol** (points/yr) — that part "
        "needs no bond-reference data. **Bond Future-Options** (TY/FV/UB/RX/OE) additionally "
        "convert to **yield-normal vol (bp/yr)** via an estimated CTD duration & conversion "
        "factor (fully editable — Databento has no deliverable-basket data). USD contracts use "
        "CBOT's semiannual-coupon convention; EUR contracts use Eurex's annual-coupon convention "
        "— a genuine structural difference, handled separately. **Money Mkt-Options** "
        "(SOFR/Euribor/SONIA) are rate-quoted, so price-vol already **is** yield-vol (×100 "
        "scaled) — no CTD conversion needed at all."
    )

    if not _api_key():
        st.error("Set DATABENTO_API_KEY in Streamlit secrets or environment variable.")
        return

    top_bond, top_mm = st.tabs(["Bond Future-Options", "Money Mkt-Options"])
    with top_bond:
        _render_market_group(_BOND_FUTURE_MARKETS, "bond", vol_anchor_mkt="TY")
    with top_mm:
        _render_market_group(_MONEY_MKT_MARKETS, "mm", vol_anchor_mkt="SOFR")

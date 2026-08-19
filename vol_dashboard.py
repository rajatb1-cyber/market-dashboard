"""
Cross-asset Vol Dashboard — CVOL-style panel grid.

One row per product: interpolated constant-tenor ATM vol (1m/3m/6m dropdown),
change columns vs 1d/1w/1m (from our own history DB), front-future price and
its 1d change. Panels by asset class:

    Equities (options_v2, Black-76 IV %)  ·  Rates Yield-Vol (rates_options,
    Bachelier bp/yr)  ·  Rates Price-Vol (same loaded data, pts/yr)  ·  STIRs
    (bp/yr)  ·  FX (IV %)  ·  Commodities (IV %)

All pricing/IV machinery is REUSED from options_v2.py / rates_options.py —
this module only loads (through their cached loaders), interpolates the expiry
curves at fixed calendar tenors, persists daily values to
vol_dashboard_history.db and renders the grid.

Methodology note: interpolated ATM vol at constant tenor from settlement
chains — our own measure, comparable to but not identical with CME CVOL
(which is variance-style over the full strike curve).
"""

import concurrent.futures
import math
import os
import sqlite3
import threading
from datetime import date, timedelta
from typing import Optional

import numpy as np
import streamlit as st

import options_v2 as _ov2
import rates_options as _ro
from options_v2 import _api_key, _trade_date, _prev_bday, _strike_fmt

# ── Config ────────────────────────────────────────────────────────────────────
_TENORS = [("1m", 30), ("3m", 90), ("6m", 180)]
_TENOR_TOL = 10        # last listed expiry may sit up to this many days SHORT of the
                       # tenor target and still fill the cell (e.g. a 172-dte expiry
                       # serves the 180d/6m point); beyond that the cell is blank —
                       # we never extrapolate past the listed curve.
_N_EXPIRIES = 16       # deep enough to reach ~6m even on markets with dense weekly
                       # front expiries (Eurex RX); pure-CPU cost only, builders cached.
_CHG_HORIZONS = [("Δ1d", 1), ("Δ1w", 7), ("Δ1m", 30)]

# Rough $ per market for one uncached settlement-day fetch (lo, hi). BRN's old $6.2
# IFEU futures-statistics landmine was fixed at source 2026-07-29 (options_v2 now uses
# ohlcv-1d first on ICE, ~$0.04) — BRN is just mildly pricier than CME markets now.
_EST_COST_DEFAULT = (0.01, 0.03)
_EST_COST_MKT = {"BRN": (0.05, 0.12)}

# measure -> which expiry-curve row key feeds it, display scaling and decimals.
#   iv_pct  : options_v2 Black-76 ATM IV (decimal)  -> %.
#   yvol_bp : rates_options Bachelier yield-vol     -> bp/yr (already bp).
#   pvol_pts: rates_options Bachelier price-vol     -> points/yr.
# wing_call/wing_put name the ±1σ wing-IV row keys the builders emit when called with
# sigma=1.0 — read out in the SAME units as curve_key (options_v2: decimal Black-76 IV;
# rates yvol: bp/yr already divided by dv01 by the builder; rates pvol: price pts). The
# The ±1σ wing columns show these as ABSOLUTE vols next to the ATM, interpolated like it.
_MEASURES = {
    "iv_pct":   {"curve_key": "atm_iv",   "wing_call": "call_wing_iv",
                 "wing_put": "put_wing_iv",   "scale": 100.0, "dp": 1},
    "yvol_bp":  {"curve_key": "atm_yvol", "wing_call": "call_wing_yvol",
                 "wing_put": "put_wing_yvol", "scale": 1.0,   "dp": 1},
    "pvol_pts": {"curve_key": "atm_iv",   "wing_call": "call_wing_iv",
                 "wing_put": "put_wing_iv",   "scale": 1.0,   "dp": 2},
}

# Declarative panel grid: (title, unit label, source module, measure, [(mkt, label)]).
# Adding a market later is a one-line change here, provided the key exists in the
# source module's config (_MARKETS_V2 / _MARKETS_RATES) — validated at import below.
# NOTE: "Rates — Yield Vol" and "Rates — Price Vol" reference the SAME five markets;
# the underlying data/curve is loaded and built ONCE and read out through two measures.
_RATES_MKTS = [("FV", "5Y UST (FV)"), ("TY", "10Y UST (TY)"),
               ("US", "T-Bond (ZB)"), ("UB", "Ultra Bond (UB)"),
               ("DU", "Euro-Schatz (DU)"), ("OE", "Euro-Bobl (OE)"),
               ("RX", "Euro-Bund (RX)"), ("UX", "Euro-Buxl (UB)")]

# Bloomberg roots for the Rates—Yield panel's BBG column (Rajat 2026-08-04;
# NB our internal keys are mostly bbg-style already — the exceptions are UB
# (bbg WN) and the ZB future living under key "US").
_BBG_ROOTS = {"FV": "FV", "TY": "TY", "US": "US", "UB": "WN", "OE": "OE",
              "RX": "RX", "UX": "UB", "DU": "DU"}

_PANELS = [
    ("Equities",          "ATM IV · %/yr",          "v2",    "iv_pct",
     [("ES", "S&P 500 (ES)"), ("NQ", "Nasdaq 100 (NQ)"), ("RTY", "Russell 2000 (RTY)"),
      ("ESTX", "EuroStoxx 50 (OESX)"), ("DAX", "DAX (ODAX)")]),
    ("Rates — Yield Vol", "ATM normal vol · bp/yr", "rates", "yvol_bp",  _RATES_MKTS),
    ("FX",                "ATM IV · %/yr",          "v2",    "iv_pct",
     [("EUR", "EUR/USD (6E)"), ("JPY", "USD/JPY (6J)"), ("GBP", "GBP/USD (6B)"), ("AUD", "AUD/USD (6A)"),
      ("CAD", "USD/CAD (6C)"), ("CHF", "USD/CHF (6S)"), ("MXN", "USD/MXN (6M)"), ("NZD", "NZD/USD (6N)")]),
    ("STIRs",             "ATM normal vol · bp/yr", "rates", "yvol_bp",
     [("SOFR", "SOFR 3M (SR3)"), ("SOFR_1Y", "SOFR 1y MC (S0)"), ("SOFR_2Y", "SOFR 2y MC (S2)"),
      ("ER", "Euribor 3M (ER)"), ("ER_1Y", "Euribor 1y MC (K)"), ("ER_2Y", "Euribor 2y MC (K2)"),
      ("SONIA", "SONIA 3M (SO3)"), ("SONIA_1Y", "SONIA 1y MC (SY1)"), ("SONIA_2Y", "SONIA 2y MC (SY2)")]),
    ("Commodities",       "ATM IV · %/yr",          "v2",    "iv_pct",
     [("GC", "Gold (GC)"), ("BRN", "Brent (BRN)"),
      ("CL", "WTI (CL)"), ("SI", "Silver (SI)"), ("HG", "Copper (HG)")]),
]

# Fail fast (at import) on any panel market missing from its source config —
# keeps the "one-line add" promise honest.
for _t, _u, _src, _meas, _mkts in _PANELS:
    _cfg = _ov2._MARKETS_V2 if _src == "v2" else _ro._MARKETS_RATES
    for _k, _lbl in _mkts:
        if _k not in _cfg:
            raise KeyError(f"vol_dashboard panel '{_t}': market '{_k}' not in "
                           f"{'_MARKETS_V2' if _src == 'v2' else '_MARKETS_RATES'}")
    if _meas not in _MEASURES:
        raise KeyError(f"vol_dashboard panel '{_t}': unknown measure '{_meas}'")

# rates_options._load_data current disk-pickle version (its `version=1` default —
# see rates_options_cache/*_v1.pkl). Only used for the pre-load cache check.
_RO_DISK_VERSION = 1

_AVAIL_ERRS = ("dataset_unavailable_range", "data_end_after_available_end",
               "license_not_found_unauthorized")

# Markets whose FUTURE is quoted inversely to market convention (CME 6J = JPY/USD,
# convention = USDJPY). Display-only: FUT column and the distribution charts' price
# axis show 1/x (vols are inversion-invariant and stay untouched). Densities are
# transformed PROPERLY with the Jacobian (pdf_y = pdf_x * x^2 for y = 1/x).
_DISPLAY_INVERT = {"v2:JPY", "v2:CAD", "v2:CHF", "v2:MXN"}


def _disp_fut(src_key: str, v):
    """Display transform for a future price (None-safe)."""
    if v is None:
        return None
    return (1.0 / float(v)) if src_key in _DISPLAY_INVERT and float(v) != 0 else float(v)


def _disp_fut_chg(src_key: str, pct):
    """% change of the DISPLAYED (possibly inverted) future. For y=1/x the exact
    transform is pct_y = -pct_x / (1 + pct_x/100) (in percent units)."""
    if pct is None:
        return None
    if src_key in _DISPLAY_INVERT:
        return -pct / (1.0 + pct / 100.0)
    return pct


def _unique_loads() -> list:
    """De-duplicated [(source, mkt)] across all panels, in panel order — the two
    rates panels collapse to one load per market."""
    seen, out = set(), []
    for _t, _u, src, _meas, mkts in _PANELS:
        for k, _lbl in mkts:
            if (src, k) not in seen:
                seen.add((src, k))
                out.append((src, k))
    return out


# ── Synthetic FX crosses (Rajat 2026-08-03: "recreate EURGBP implied vol from
# the USD legs") — NO market options behind these rows: ATM/wings are the
# triangle combination σx² = σa² + σb² − 2ρσaσb of the two USD legs' vols at
# trailing-90d REALIZED correlation (implied corr unobservable without cross
# options; typically ρ_impl < ρ_realized so true cross vols likely a touch
# higher). Wings pair A-call with B-put (cross up = A up / B down) and vice
# versa. Rows render amber-tinted in the FX panel; deliberately NOT in the
# charting dropdown. Cross forward = Fa/Fb (triangle arbitrage, assumption-free).
_FX_CROSSES = [("EURGBP", "EUR/GBP", "EUR", "GBP"),
               ("EURJPY", "EUR/JPY", "EUR", "JPY"),
               ("GBPJPY", "GBP/JPY", "GBP", "JPY"),
               ("AUDNZD", "AUD/NZD", "AUD", "NZD")]


def _cross_rho(a: str, b: str, window: int = 90):
    """Trailing realized correlation of the two legs' daily log returns (native
    USD-per-unit quotes from the rolling underlying stores). None when thin."""
    try:
        import pandas as pd
        da, _e1 = _underlying_bars("v2", a)
        db_, _e2 = _underlying_bars("v2", b)
        if da is None or db_ is None:
            return None
        m = pd.merge(da[["date", "close"]], db_[["date", "close"]], on="date",
                     suffixes=("_a", "_b")).tail(window + 1)
        if len(m) < 30:
            return None
        ra = np.diff(np.log(m["close_a"].to_numpy(dtype=float)))
        rb = np.diff(np.log(m["close_b"].to_numpy(dtype=float)))
        r = float(np.corrcoef(ra, rb)[0, 1])
        return r if math.isfinite(r) else None
    except Exception:
        return None


def _add_fx_crosses(build: dict) -> None:
    """Inject synthetic cross series/markets into a finished build (keys 'x:NAME' /
    'NAME:iv_pct'). Because they live in build['series'], the EXISTING history
    upsert and Δ machinery handle them with no further wiring. Never raises."""
    for name, _lbl, a, b in _FX_CROSSES:
        try:
            sa = build["series"].get(f"{a}:iv_pct")
            sb = build["series"].get(f"{b}:iv_pct")
            ma = build["markets"].get(f"v2:{a}") or {}
            mb = build["markets"].get(f"v2:{b}") or {}
            fa, fb = ma.get("fut"), mb.get("fut")
            if not sa or not sb or not fa or not fb:
                continue
            rho = _cross_rho(a, b)
            if rho is None:
                continue

            def _comb(v1, v2):
                if v1 is None or v2 is None:
                    return None
                x = v1 * v1 + v2 * v2 - 2.0 * rho * v1 * v2
                return math.sqrt(x) if x > 0 else None

            vols = {tn: _comb(sa["vols"].get(tn), sb["vols"].get(tn))
                    for tn, _d in _TENORS}
            wcs = {tn: _comb((sa.get("wcs") or {}).get(tn),
                             (sb.get("wps") or {}).get(tn)) for tn, _d in _TENORS}
            wps = {tn: _comb((sa.get("wps") or {}).get(tn),
                             (sb.get("wcs") or {}).get(tn)) for tn, _d in _TENORS}
            build["markets"][f"x:{name}"] = {
                "tdate": ma.get("tdate"), "err": None,
                "fut": float(fa) / float(fb),
                "fut_sym": (f"{ma.get('fut_sym', '')}/{mb.get('fut_sym', '')} "
                            f"· ρ90d={rho:.2f} (derived)"),
                "max_dte": min(ma.get("max_dte") or 0, mb.get("max_dte") or 0) or None,
            }
            build["series"][f"{name}:iv_pct"] = {"mkey": f"x:{name}", "vols": vols,
                                                 "wcs": wcs, "wps": wps}
        except Exception:
            continue


# ── History DB (short-lived connections + lock, per data_costs.py pattern) ───
_DB_PATH = os.path.join(os.path.dirname(__file__), "vol_dashboard_history.db")
_DB_LOCK = threading.Lock()


def _hist_conn():
    c = sqlite3.connect(_DB_PATH, timeout=5)
    c.execute("""CREATE TABLE IF NOT EXISTS vol_hist (
        day TEXT, mkt TEXT, tenor TEXT, vol REAL, fut REAL,
        PRIMARY KEY (day, mkt, tenor))""")
    return c


def _hist_upsert(build: dict, day: Optional[str] = None) -> None:
    """Persist today's values for EVERY series x tenor (vol may be NULL for a blank
    6m cell — the row still carries the future price for the fut-Δ1d lookup).
    `mkt` column stores the series key '<MKT>:<measure>' so the two rates measures
    keep independent histories under the spec's single-vol-column schema."""
    day = day or date.today().isoformat()
    rows = []
    for skey, s in build["series"].items():
        mk = build["markets"].get(s["mkey"], {})
        fut = mk.get("fut")
        if fut is None and all(v is None for v in s["vols"].values()):
            continue   # market failed entirely — don't write an empty row
        wcs = s.get("wcs") or {}
        wps = s.get("wps") or {}
        for tn, _days in _TENORS:
            rows.append((day, skey, tn, s["vols"].get(tn), fut))
            # Wing history under distinct series keys '<MKT>:<measure>:c1s'/':p1s' in the
            # SAME table (RR derivable as c1s − p1s) — accumulates so wing Δ columns can be
            # added later. _hist_changes only ever gets the ATM keys, so these are invisible
            # to the ATM Δ math.
            rows.append((day, f"{skey}:c1s", tn, wcs.get(tn), fut))
            rows.append((day, f"{skey}:p1s", tn, wps.get(tn), fut))
    if not rows:
        return
    try:
        with _DB_LOCK:
            c = _hist_conn()
            c.executemany("INSERT OR REPLACE INTO vol_hist VALUES (?,?,?,?,?)", rows)
            c.commit()
            c.close()
    except Exception:
        pass


def _hist_changes(day: str, series_keys: list) -> dict:
    """{(skey, tenor): {1: Δvol_1d, 7: Δvol_1w, 30: Δvol_1m, 'fut': Δfut_1d_pct}}
    against the nearest stored day <= day - N. '—' (None) when no prior exists."""
    hist: dict = {}
    try:
        with _DB_LOCK:
            c = _hist_conn()
            qmarks = ",".join("?" * len(series_keys))
            rows = c.execute(
                f"SELECT day, mkt, tenor, vol, fut FROM vol_hist "
                f"WHERE mkt IN ({qmarks}) ORDER BY day", series_keys).fetchall()
            c.close()
    except Exception:
        rows = []
    by_st: dict = {}
    for d, mkt, tn, vol, fut in rows:
        by_st.setdefault((mkt, tn), []).append((d, vol, fut))

    today_d = date.fromisoformat(day)

    def _prior(skey, tn, back, field):
        target = (today_d - timedelta(days=back)).isoformat()
        idx = 1 if field == "vol" else 2
        for d, vol, fut in reversed(by_st.get((skey, tn), [])):
            row = (d, vol, fut)
            if d <= target and row[idx] is not None:
                return row[idx]
        return None

    out: dict = {}
    for skey in series_keys:
        for tn, _days in _TENORS:
            cur = next((v for d, v, f in reversed(by_st.get((skey, tn), []))
                        if d == day), None)
            cur_fut = next((f for d, v, f in reversed(by_st.get((skey, tn), []))
                            if d == day), None)
            cell: dict = {}
            for _lbl, back in _CHG_HORIZONS:
                prev = _prior(skey, tn, back, "vol")
                cell[back] = (cur - prev) if (cur is not None and prev is not None) else None
            pf = _prior(skey, tn, 1, "fut")
            cell["fut"] = ((cur_fut / pf - 1.0) * 100.0
                           if (cur_fut is not None and pf) else None)
            out[(skey, tn)] = cell
    return out


# ── Curve interpolation ───────────────────────────────────────────────────────
def _interp_pairs(pairs: list, target_days: int, tol: int = _TENOR_TOL) -> Optional[float]:
    """Linear interpolation of [(dte, value)] at target_days. Short end clamps to the
    front expiry's value; long end returns None (blank) once the target exceeds the
    last listed expiry by more than `tol` days — no extrapolation."""
    pts = sorted((d, v) for d, v in pairs if v is not None and d is not None)
    if not pts:
        return None
    if target_days <= pts[0][0]:
        return pts[0][1]
    if target_days > pts[-1][0]:
        return pts[-1][1] if (target_days - pts[-1][0]) <= tol else None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= target_days <= x1:
            if x1 == x0:
                return y0
            return y0 + (target_days - x0) * (y1 - y0) / (x1 - x0)
    return pts[-1][1]


# ── Market loading (reuses the source modules' cached loaders/builders) ──────
def _market_dv01(mkt: str, m: dict) -> tuple:
    """rates_options._get_market_dv01, with a session-state-free fallback so the
    offline verify script (bare mode, no Streamlit runtime) also works."""
    try:
        return _ro._get_market_dv01(mkt, m)
    except Exception:
        if "fixed_dv01" in m:
            v = float(m["fixed_dv01"])
            return v, bool(v and np.isfinite(v) and v > 0)
        if not m.get("needs_ctd"):
            return float("nan"), False
        dy, dc = _ro._default_ctd_inputs(m["ctd_years"], curve=m["curve"])
        _, _, _, dv01 = _ro._estimate_dv01(dc / 100, dy / 100, m["ctd_years"], freq=m["freq"])
        return float(dv01), bool(dv01 and np.isfinite(dv01) and dv01 > 0)


# ── Disk cache for options_v2 loads ──────────────────────────────────────────
# The per-(market, day) pickle layer now lives INSIDE options_v2 (options_v2._load_data_disk,
# reusing this module's vol_dash_cache dir + filename scheme) so it benefits the Options tab
# too. The dashboard just calls _ov2._load_data / _ov2._load_data_disk; rates_options gained
# the same _load_data_disk extraction. No duplicate disk layer here anymore.


def _sanitize_v2_defs(data: dict) -> dict:
    """Drop option-definition rows whose strike is implausibly far from the market's
    own futures settles before curve building.

    Why: options_v2's RTY config merges 'SR3.OPT' as its monthly supplement, but SR3
    is CME SOFR — its 100x-raw-scaled strikes (~9000-10300 vs RTY future ~2964) land
    in RTY's chain, and the merged-in expiries produce absurd ATM IVs (584%/365%/...,
    hit 2026-07-29). For an ATM term-structure only strikes near the future matter,
    so keeping strikes within [ref/2.5, ref*2.5] of the median OUTRIGHT futures settle
    (spread rows like 'RTYU6-RTYM9' settle near 0 and must be excluded from the
    reference) removes the pollution and is harmless for every other market — deep
    wings are irrelevant to the ATM pick. Returns a shallow-copied data dict; never
    mutates the source module's cached object."""
    defs = data.get("opt_defs")
    futs = data.get("fut_ohlcv")
    try:
        if (defs is None or getattr(defs, "empty", True)
                or futs is None or getattr(futs, "empty", True)
                or "strike_price" not in defs.columns
                or "close" not in futs.columns or "symbol" not in futs.columns):
            return data
        outright = futs[~futs["symbol"].astype(str).str.contains("-", na=False)]
        if outright.empty:
            return data
        ref = float(outright["close"].median())
        if not (math.isfinite(ref) and ref > 0):
            return data
        keep = defs["strike_price"].between(ref / 2.5, ref * 2.5)
        if keep.all() or not keep.any():
            return data
        data = dict(data)
        data["opt_defs"] = defs[keep].reset_index(drop=True)
        return data
    except Exception:
        return data


def _needs_fallback(data: dict) -> bool:
    derr = data.get("defs_err", "") or ""
    defs_empty = (data.get("opt_defs") is None
                  or getattr(data.get("opt_defs"), "empty", True))
    return defs_empty or any(e in derr for e in _AVAIL_ERRS)


_MISS_TTL_S = 4 * 3600   # remember "date not available yet" this long, then retry


def _miss_path(src: str, mkt: str, date_str: str) -> str:
    return os.path.join(os.path.dirname(__file__), "vol_dash_cache",
                        f"MISS_{src}_{mkt}_{date_str}.flag")


def _date_known_missing(src: str, mkt: str, date_str: str) -> bool:
    """True if this (market, date) recently came back empty/embargoed. Empty results
    are deliberately never data-cached (anti-poison rule), which after the midnight
    date-rollover caused EVERY rerun to re-pay a doomed fetch before falling back
    (hit 2026-08-01 00:30 — settlements for the 'new' trade date publish hours later).
    A short-TTL miss marker lets us skip straight to the fallback date, retrying only
    after the TTL when the data may genuinely have published."""
    p = _miss_path(src, mkt, date_str)
    try:
        import time as _t
        return os.path.exists(p) and (_t.time() - os.path.getmtime(p)) < _MISS_TTL_S
    except Exception:
        return False


def _mark_date_missing(src: str, mkt: str, date_str: str) -> None:
    try:
        os.makedirs(os.path.dirname(_miss_path(src, mkt, date_str)), exist_ok=True)
        with open(_miss_path(src, mkt, date_str), "w") as f:
            f.write("")
    except Exception:
        pass


def _load_market(src: str, mkt: str, sigma: float = 1.0) -> tuple:
    """(curve_rows, tdate_str, err) — same empty-defs/license fallback the source
    modules use in their own renderers (one business day back), with a short-TTL
    negative cache so a not-yet-published date isn't re-fetched on every rerun."""
    try:
        if src == "rates":
            m = _ro._MARKETS_RATES[mkt]
            tdate = _trade_date(m["ds"])
            tdate_str = str(tdate)
            if _date_known_missing(src, mkt, tdate_str):
                tdate_str = str(_prev_bday(tdate, 1))       # skip the doomed fetch
                data = _ro._load_data(mkt, tdate_str)
            else:
                data = _ro._load_data(mkt, tdate_str)
                if _needs_fallback(data):
                    _mark_date_missing(src, mkt, tdate_str)
                    tdate_str = str(_prev_bday(tdate, 1))
                    data = _ro._load_data(mkt, tdate_str)
            if data.get("error"):
                return [], tdate_str, data["error"]
            dv01, _dv01_ok = _market_dv01(mkt, m)
            # sigma=1.0 turns on the builders' ±1σ wing-IV machinery (wing columns).
            # This is a NEW cache key inside the builder — the Expiry-Curve/Vol-Monitor
            # callers (sigma=0.0) keep their own cached entries, untouched.
            curve = _ro._build_expiry_curve_rates(mkt, sigma, data, m, dv01,
                                                  n_expiries=_N_EXPIRIES,
                                                  cache_date=tdate_str)
        else:
            m = _ov2._MARKETS_V2[mkt]
            tdate = _trade_date(m["ds"])
            tdate_str = str(tdate)
            if _date_known_missing(src, mkt, tdate_str):
                tdate_str = str(_prev_bday(tdate, 1))       # skip the doomed fetch
                data = _ov2._load_data(mkt, tdate_str)
            else:
                data = _ov2._load_data(mkt, tdate_str)
                if _needs_fallback(data):
                    _mark_date_missing(src, mkt, tdate_str)
                    tdate_str = str(_prev_bday(tdate, 1))
                    data = _ov2._load_data(mkt, tdate_str)
            if data.get("error"):
                return [], tdate_str, data["error"]
            data = _sanitize_v2_defs(data)
            # sigma=1.0: see note in the rates branch — enables the ±1σ wing IVs.
            curve = _ov2._build_expiry_curve(mkt, sigma, data,
                                             n_expiries=_N_EXPIRIES,
                                             cache_date=tdate_str)
        return curve, tdate_str, (None if curve else "no usable curve data")
    except Exception as ex:
        return [], "", f"{type(ex).__name__}: {ex}"


# ── Dashboard build (cached per day + market set — reruns after load are free) ──
@st.cache_data(ttl=3600, show_spinner=False)
def _build_dashboard(cache_day: str, loads_sig: tuple) -> dict:
    """{"markets": {"src:MKT": {tdate, err, fut, fut_sym, max_dte}},
        "series":  {"MKT:measure": {"mkey": "src:MKT", "vols": {tenor: value}}}}

    Disk-persisted per (day, load-set): the build is minutes of pure-CPU IV solving
    whose inputs (daily settlements) don't change intraday — an app restart should
    NOT re-pay it (Rajat 2026-07-31). Only fully-clean builds (zero market errors)
    are persisted, so a transiently-failed market retries on the next build instead
    of being frozen into the pickle."""
    import pickle as _pkl
    import hashlib as _hl
    _sig = _hl.md5(repr(sorted(loads_sig)).encode()).hexdigest()[:10]
    _bpath = os.path.join(os.path.dirname(__file__), "vol_dash_cache",
                          f"BUILD_{cache_day}_{_sig}_v1.pkl")
    if os.path.exists(_bpath):
        try:
            with open(_bpath, "rb") as _f:
                _prev = _pkl.load(_f)
            # Staleness check: a build persisted overnight (from fallback/T-1 data,
            # before embargoes cleared) must NOT be served all day once fresh
            # settlements exist (caught 2026-07-31 morning). Serve the pickle only
            # if every market's stored trade date still matches what a load would
            # use NOW (current _trade_date, or its T-1 fallback when the miss
            # marker says the current date is unavailable).
            _stale = False
            for _s2, _m2 in loads_sig:
                _cfg = _ro._MARKETS_RATES[_m2] if _s2 == "rates" else _ov2._MARKETS_V2[_m2]
                _exp_td = str(_trade_date(_cfg["ds"]))
                if _date_known_missing(_s2, _m2, _exp_td):
                    _exp_td = str(_prev_bday(_trade_date(_cfg["ds"]), 1))
                _stored = (_prev.get("markets", {}).get(f"{_s2}:{_m2}", {}) or {}).get("tdate")
                if _stored and _stored != _exp_td:
                    _stale = True
                    break
            if not _stale:
                # builds pickled before the crosses feature lack them — backfill once
                if not any(k.startswith("x:") for k in _prev.get("markets", {})):
                    _add_fx_crosses(_prev)
                    if any(k.startswith("x:") for k in _prev.get("markets", {})):
                        try:
                            with open(_bpath, "wb") as _f:
                                _pkl.dump(_prev, _f)
                        except Exception:
                            pass
                return _prev
        except Exception:
            pass
    out = {"markets": {}, "series": {}}
    for src, mkt in loads_sig:
        curve, tdate_str, err = _load_market(src, mkt)
        front = min(curve, key=lambda r: r["dte"]) if curve else None
        mkey = f"{src}:{mkt}"
        out["markets"][mkey] = {
            "tdate": tdate_str, "err": err,
            "fut": front.get("F") if front else None,
            "fut_sym": (front.get("fut_sym") or "") if front else "",
            "max_dte": max((r["dte"] for r in curve), default=None),
        }
        for _t, _u, psrc, meas, mkts in _PANELS:
            if psrc != src or mkt not in (k for k, _l in mkts):
                continue
            spec = _MEASURES[meas]
            pairs = [(r.get("dte"), r.get(spec["curve_key"])) for r in curve]
            vols = {}
            for tn, days in _TENORS:
                v = _interp_pairs(pairs, days)
                vols[tn] = (v * spec["scale"]) if v is not None else None
            # ±1σ wing vols shown as ABSOLUTE levels next to the ATM (Rajat 2026-07-30:
            # easier to read than the RR difference — skew is visible by eye). Each wing
            # interpolated independently, same units as ATM; an expiry missing that wing
            # contributes no pair; ≥2 usable pairs or the cell blanks (no clamp-to-front)
            # — missing wings NEVER affect the ATM `vols` above.
            wing_c_pairs, wing_p_pairs = [], []
            for r in curve:
                dte = r.get("dte")
                cw = r.get(spec["wing_call"])
                pw = r.get(spec["wing_put"])
                if dte is not None and cw is not None and math.isfinite(cw):
                    wing_c_pairs.append((dte, cw * spec["scale"]))
                if dte is not None and pw is not None and math.isfinite(pw):
                    wing_p_pairs.append((dte, pw * spec["scale"]))
            wcs, wps = {}, {}
            for tn, days in _TENORS:
                wcs[tn] = _interp_pairs(wing_c_pairs, days) if len(wing_c_pairs) >= 2 else None
                wps[tn] = _interp_pairs(wing_p_pairs, days) if len(wing_p_pairs) >= 2 else None
            out["series"][f"{mkt}:{meas}"] = {"mkey": mkey, "vols": vols,
                                              "wcs": wcs, "wps": wps}

    _add_fx_crosses(out)
    # Persist only clean builds — a failed market must retry next build, not fossilize.
    if not any(m.get("err") for m in out["markets"].values()):
        try:
            os.makedirs(os.path.dirname(_bpath), exist_ok=True)
            with open(_bpath, "wb") as _f:
                _pkl.dump(out, _f)
        except Exception:
            pass
    return out


# ── Cache-awareness for the Load button ───────────────────────────────────────
def _cache_status(loads: list) -> tuple:
    """([cached (src, mkt)], [needs-fetch (src, mkt)]) for TODAY's trade date, by
    disk-pickle existence — rates_options' own cache dir for rates markets, this
    module's vol_dash_cache for v2 markets (plus the session marker for v2 markets
    already loaded this session but whose pickle write failed)."""
    v2_done = st.session_state.get("_vd_v2_session", set())
    cached, need = [], []
    for src, mkt in loads:
        if src == "rates":
            m = _ro._MARKETS_RATES[mkt]
            td = str(_trade_date(m["ds"]))
            ok = os.path.exists(_ro._disk_cache_path(mkt, td, _RO_DISK_VERSION))
        else:
            m = _ov2._MARKETS_V2[mkt]
            td = str(_trade_date(m["ds"]))
            ok = os.path.exists(_ov2._load_data_disk_path(mkt, td)) or mkt in v2_done
        (cached if ok else need).append((src, mkt))
    return cached, need


def _est_fetch_cost(need: list) -> tuple:
    """Rough (lo, hi) $ estimate for fetching the uncached markets."""
    lo = sum(_EST_COST_MKT.get(mkt, _EST_COST_DEFAULT)[0] for _src, mkt in need)
    hi = sum(_EST_COST_MKT.get(mkt, _EST_COST_DEFAULT)[1] for _src, mkt in need)
    return lo, hi


# ── Parallel prefetch (warms every market's disk cache before the sequential build) ──
def _load_tasks(loads: list) -> list:
    """Group [(src, mkt)] into concurrent tasks. Rates markets that SHARE a futures root
    (SOFR/SOFR_1Y/SOFR_2Y → SR3.FUT; ER family → I.FUT; SONIA family → SO3.FUT) collapse
    into ONE task so worker threads never race on the same _fetch_fut_shared pickle; every
    other market is its own singleton task. Returns [(task_label, [(src, mkt), …])] in the
    input order of each group's first member."""
    groups: dict = {}
    order: list = []
    for src, mkt in loads:
        if src == "rates":
            m = _ro._MARKETS_RATES[mkt]
            key = f"rates:{m.get('ds', '')}:{m.get('fut_sym', mkt)}"
        else:
            key = f"v2:{mkt}"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((src, mkt))
    return [(k, groups[k]) for k in order]


def _prefetch_member(src: str, mkt: str) -> None:
    """Warm one market's disk cache (today's settlement, plus the 1-bday fallback when today
    is unavailable/empty — mirrors _load_market). PURE: no Streamlit calls, so it is
    thread-safe. Per-call db.Historical clients live inside the fetch functions; the shared
    fut-root pickle is guarded by task grouping; data_costs.record_cost is lock-guarded."""
    if src == "rates":
        m = _ro._MARKETS_RATES[mkt]
        tdate = _trade_date(m["ds"])
        if _date_known_missing(src, mkt, str(tdate)):
            _ro._load_data_disk(mkt, str(_prev_bday(tdate, 1)), _RO_DISK_VERSION)
            return
        data = _ro._load_data_disk(mkt, str(tdate), _RO_DISK_VERSION)
        if _needs_fallback(data):
            _mark_date_missing(src, mkt, str(tdate))
            _ro._load_data_disk(mkt, str(_prev_bday(tdate, 1)), _RO_DISK_VERSION)
    else:
        m = _ov2._MARKETS_V2[mkt]
        tdate = _trade_date(m["ds"])
        if _date_known_missing(src, mkt, str(tdate)):
            _ov2._load_data_disk(mkt, str(_prev_bday(tdate, 1)))
            return
        data = _ov2._load_data_disk(mkt, str(tdate))
        if _needs_fallback(data):
            _mark_date_missing(src, mkt, str(tdate))
            _ov2._load_data_disk(mkt, str(_prev_bday(tdate, 1)))


def _prefetch_task(members: list) -> str:
    """Process one task's members SEQUENTIALLY — shared-fut-root families must not race on
    the same _fetch_fut_shared pickle. Returns the member labels for the status line."""
    for src, mkt in members:
        _prefetch_member(src, mkt)
    return ", ".join(mkt for _s, mkt in members)


def _run_prefetch(loads: list) -> dict:
    """Concurrent disk-cache warm-up with a progress bar + status line. Per-task exceptions
    are collected and returned (never raised) — the build then applies its normal per-market
    fallback/blank treatment, so a failed prefetch never crashes the dashboard."""
    tasks = _load_tasks(loads)
    total = len(loads)
    done = 0
    errors: dict = {}
    prog = st.progress(0.0)
    status = st.empty()
    # 12 workers, not more: probed 2026-07-30 — Databento serves ~5-6 requests per key
    # truly concurrently and queues the rest server-side (per-request latency INFLATES
    # 14s→22s at width 12, 504s appear); width 24 gained nothing over 12.
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        fmap = {ex.submit(_prefetch_task, members): (label, members)
                for label, members in tasks}
        for fut in concurrent.futures.as_completed(fmap):
            label, members = fmap[fut]
            done += len(members)
            try:
                names = fut.result()
                status.caption(f"{done}/{total} · loaded {names}")
            except Exception as ex_:
                errors[label] = f"{type(ex_).__name__}: {ex_}"
                status.caption(f"{done}/{total} · {label} failed "
                               f"(build will fall back per-market)")
            prog.progress(min(done / total, 1.0) if total else 1.0)
    prog.empty()
    status.empty()
    return errors


# ── HTML rendering (house compact-table style) ────────────────────────────────
_GREEN, _RED, _GREY = "#059669", "#DC2626", "#94A3B8"
_TH = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
       "padding:5px 10px;text-align:right;white-space:nowrap")
_TD = ("font-size:12px;padding:5px 10px;border-bottom:1px solid #E2E8F0;"
       "text-align:right;white-space:nowrap;font-family:monospace")


def _fmt_val(v, dp: int) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return f"<span style='color:{_GREY}'>—</span>"
    return f"<b>{v:.{dp}f}</b>"


def _fmt_chg(v, dp: int, up_color: str, down_color: str, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return f"<span style='color:{_GREY}'>—</span>"
    col = up_color if v > 0 else (down_color if v < 0 else "#64748B")
    return f"<span style='color:{col};font-weight:600'>{v:+.{dp}f}{suffix}</span>"


def _ctd_yield_from_price(m: dict, F: float) -> float:
    """Implied CTD yield (%) for one bond-future price: solve
    bond_price(coupon, y, ctd_years) = F × conversion_factor with the same CTD
    assumptions the dv01 machinery uses. Raises on failure (callers wrap)."""
    from scipy.optimize import brentq
    _dy, dc = _ro._default_ctd_inputs(m["ctd_years"], curve=m["curve"])
    years, freq = float(m["ctd_years"]), int(m["freq"])
    cf = (_ro._cbot_conversion_factor(dc / 100.0, years) if freq == 2
          else _ro._direct_conversion_factor(dc / 100.0, years, freq))
    if not (cf and math.isfinite(cf) and cf > 0):
        raise ValueError("bad conversion factor")
    tgt = float(F) * cf
    y = brentq(lambda yy: _ro._bond_price_duration(dc / 100.0, yy, years, freq)[0] - tgt,
               1e-4, 0.25, xtol=1e-8, maxiter=100)
    return float(y) * 100.0


def _rate_level(psrc: str, mkt: str, fut) -> Optional[float]:
    """Rate level (%) implied by the futures price, for the rates panels' RATE column.
    STIR: 100 − F directly. Bond futures: the implied CTD yield — solve
    bond_price(coupon, y, ctd_years) = F × conversion_factor with the same CTD
    assumptions the dv01 machinery uses (Rajat: "you have done it before")."""
    if psrc != "rates" or fut is None:
        return None
    try:
        m = _ro._MARKETS_RATES[mkt]
        F = float(fut)
        if "fixed_dv01" in m:
            return 100.0 - F
        if not m.get("needs_ctd"):
            return None
        return _ctd_yield_from_price(m, F)
    except Exception:
        return None


def _yield_axis_fns(src: str, mkt: str, pmin: float, pmax: float):
    """(price→yield%, yield%→price) vectorized monotone maps over [pmin, pmax] —
    powers the distribution charts' "x-axis in yield" checkbox. STIR: exact 100−p.
    Bond futures: implied CTD yield sampled on a coarse price grid + monotone
    interpolation (yield is DECREASING in price, so display arrays get reversed
    exactly like the USDJPY 1/x inversion). None when unavailable."""
    if src != "rates" or not (math.isfinite(pmin) and math.isfinite(pmax) and pmax > pmin):
        return None
    try:
        m = _ro._MARKETS_RATES[mkt]
        if "fixed_dv01" in m:
            return (lambda p: 100.0 - np.asarray(p, dtype=float),
                    lambda y: 100.0 - np.asarray(y, dtype=float))
        if not m.get("needs_ctd"):
            return None
        pad = 0.02 * (pmax - pmin) + 1e-9
        pg = np.linspace(pmin - pad, pmax + pad, 25)
        yg = np.array([_ctd_yield_from_price(m, float(p)) for p in pg])
        if not (np.all(np.isfinite(yg)) and np.all(np.diff(yg) < 0)):
            return None
        yr, pr = yg[::-1], pg[::-1]      # increasing-x views for the inverse interp
        return (lambda p: np.interp(np.asarray(p, dtype=float), pg, yg),
                lambda y: np.interp(np.asarray(y, dtype=float), yr, pr))
    except Exception:
        return None


def _panel_html(title: str, unit: str, src: str, meas: str, mkts: list,
                build: dict, changes: dict, tenor: str) -> str:
    dp = _MEASURES[meas]["dp"]
    thl = _TH.replace("text-align:right", "text-align:left")
    tdl = _TD.replace("text-align:right", "text-align:left") \
             .replace("font-family:monospace", "")
    _rate_col = (meas == "yvol_bp")
    _bbg_col = (title == "Rates — Yield Vol")
    hdr = (f"<tr><th style='{thl}'>PRODUCT</th>"
           + (f"<th style='{_TH}'>BBG</th>" if _bbg_col else "")
           + f"<th style='{_TH}'>VOL {tenor}</th>"
           + f"<th style='{_TH}'>+1σ</th><th style='{_TH}'>−1σ</th>"
           + "".join(f"<th style='{_TH}'>{lbl}</th>" for lbl, _n in _CHG_HORIZONS)
           + f"<th style='{_TH}'>FUT</th>"
           + (f"<th style='{_TH}'>RATE</th>" if _rate_col else "")
           + f"<th style='{_TH}'>Δ1d</th></tr>")
    # normal rows, then (FX panel only) the amber-tinted DERIVED cross rows —
    # crosses live under src "x" in the build and are deliberately absent from
    # the charting dropdown (Rajat 2026-08-03).
    row_defs = [(mkt, label, src, False) for mkt, label in mkts]
    _has_cross = False
    if title == "FX":
        for _xn, _xl, _a2, _b2 in _FX_CROSSES:
            if f"x:{_xn}" in (build.get("markets") or {}):
                row_defs.append((_xn, _xl + " ×", "x", True))
                _has_cross = True
    body = ""
    for mkt, label, rsrc, _tint in row_defs:
        _bg = ";background:#FFF6E0" if _tint else ""
        td = _TD + _bg
        tdl_r = tdl + _bg
        skey = f"{mkt}:{meas}"
        s = build["series"].get(skey, {})
        mk = build["markets"].get(f"{rsrc}:{mkt}", {})
        ch = changes.get((skey, tenor), {})
        vol = (s.get("vols") or {}).get(tenor)
        wc = (s.get("wcs") or {}).get(tenor)
        wp = (s.get("wps") or {}).get(tenor)
        fut = mk.get("fut")
        fut_sym = mk.get("fut_sym") or ""
        body += (
            f"<tr><td style='{tdl_r}'><b>{label}</b></td>"
            + (f"<td style='{td}'>{_BBG_ROOTS.get(mkt, '—')}</td>" if _bbg_col else "")
            + f"<td style='{td}'>{_fmt_val(vol, dp)}</td>"
            # ±1σ wing vols (absolute levels — call wing then put wing)
            f"<td style='{td}'>{_fmt_val(wc, dp)}</td>"
            f"<td style='{td}'>{_fmt_val(wp, dp)}</td>"
            # vol changes: up=red / down=green (risk convention, as in the rates tables)
            + "".join(f"<td style='{td}'>{_fmt_chg(ch.get(n), dp, _RED, _GREEN)}</td>"
                      for _lbl, n in _CHG_HORIZONS)
            + f"<td style='{td}' title='{fut_sym}'>"
              f"{_strike_fmt(_disp_fut(f'{rsrc}:{mkt}', fut)) if fut is not None else '—'}</td>"
            + ((lambda _rl: f"<td style='{td}'>{f'{_rl:.3f}%' if _rl is not None else '—'}</td>")(
                _rate_level(rsrc, mkt, fut)) if _rate_col else "")
            # future price change: up=green / down=red (price convention, as in Watchlist)
            + f"<td style='{td}'>{_fmt_chg(_disp_fut_chg(f'{rsrc}:{mkt}', ch.get('fut')), 2, _GREEN, _RED, '%')}</td></tr>"
        )
    _foot = ""
    if _has_cross:
        _foot = ("<div style='font-size:10px;color:#B45309;margin-top:2px'>"
                 "× amber rows = DERIVED crosses — triangle of the USD legs' vols at "
                 "trailing-90d realized ρ (hover FUT for legs &amp; ρ); no cross "
                 "options behind these numbers.</div>")
    head_bar = (
        f"<div style='background:#1E293B;color:#F8FAFC;padding:6px 12px;font-size:13px;"
        f"font-weight:700;border-radius:6px 6px 0 0;display:inline-block;"
        f"white-space:nowrap'>{title}"
        f"&nbsp;&nbsp;<span style='font-weight:400;font-size:11px;color:#94A3B8'>{unit}</span></div>"
    )
    return (f"<div style='overflow-x:auto;margin-bottom:14px'>{head_bar}"
            f"<table style='border-collapse:collapse'>"
            f"<thead>{hdr}</thead><tbody>{body}</tbody></table>{_foot}</div>")


# ── Render ────────────────────────────────────────────────────────────────────
def _split_normal_curve(src: str, mkt: str, tenor_days: int, mkinfo: dict):
    """Split-normal implied price-density pieces at the horizon for one market — the
    SHARED math behind both the simple distribution chart and the light dashed comparison
    overlaid on the Breeden–Litzenberger density. Centre = current future F; left
    half-width σ_put = −1σ wing vol, right σ_call = +1σ wing vol, converted to ABSOLUTE
    price std-devs at the tenor (rates: Bachelier price-vol × √T; lognormal markets:
    iv × F × √T). Returns {F, xs, pdf (skewed), sym (ATM-only), s_put, s_call, s_atm,
    unit} or None when ATM/both wings/future price are missing. Reuses the cached curve
    builders via _load_market (no new fetches)."""
    F = mkinfo.get("fut")
    if F is None or not math.isfinite(float(F)) or float(F) <= 0:
        return None
    F = float(F)
    curve, _td, _err = _load_market(src, mkt)
    if not curve:
        return None
    T = tenor_days / 365.0

    # Interpolate ATM + both wings at the tenor IN PRICE-VOL terms (rates carry the
    # Bachelier price-vol under atm_iv/call_wing_iv/put_wing_iv; v2 = decimal IV).
    def _interp_key(key):
        prs = [(r.get("dte"), r.get(key)) for r in curve
               if r.get("dte") is not None and r.get(key) is not None
               and math.isfinite(r.get(key))]
        return _interp_pairs(prs, tenor_days) if len(prs) >= 2 else None

    atm_pv = _interp_key("atm_iv")
    cw_pv = _interp_key("call_wing_iv")
    pw_pv = _interp_key("put_wing_iv")
    if atm_pv is None or cw_pv is None or pw_pv is None:
        return None

    if src == "rates":
        s_atm, s_call, s_put = (atm_pv * math.sqrt(T), cw_pv * math.sqrt(T),
                                pw_pv * math.sqrt(T))
        unit = "pts"
    else:
        s_atm, s_call, s_put = (F * atm_pv * math.sqrt(T), F * cw_pv * math.sqrt(T),
                                F * pw_pv * math.sqrt(T))
        unit = ""
    if min(s_atm, s_call, s_put) <= 0:
        return None

    smax = max(s_call, s_put, s_atm)
    xs = np.linspace(F - 4 * smax, F + 4 * smax, 481)
    A = 2.0 / (math.sqrt(2 * math.pi) * (s_put + s_call))
    pdf = np.where(xs < F,
                   A * np.exp(-((xs - F) ** 2) / (2 * s_put ** 2)),
                   A * np.exp(-((xs - F) ** 2) / (2 * s_call ** 2)))
    sym = (1.0 / (math.sqrt(2 * math.pi) * s_atm)
           * np.exp(-((xs - F) ** 2) / (2 * s_atm ** 2)))
    return dict(F=F, xs=xs, pdf=pdf, sym=sym, s_put=s_put, s_call=s_call,
                s_atm=s_atm, unit=unit)


def _implied_dist_figure(src: str, mkt: str, meas: str, label: str,
                         tenor: str, tenor_days: int, mkinfo: dict,
                         yield_axis: bool = False):
    """Split-normal implied price distribution at the horizon for one market.
    pdf = 2/(√2π (σl+σr)) · exp(−(x−F)²/2σ²_side) — continuous at F, integrates to 1.
    Returns a plotly Figure or None when inputs are missing. Thin wrapper over the shared
    _split_normal_curve helper (no new fetches)."""
    import plotly.graph_objects as go

    c = _split_normal_curve(src, mkt, tenor_days, mkinfo)
    if c is None:
        return None
    F, xs, pdf, sym = c["F"], c["xs"], c["pdf"], c["sym"]
    s_put, s_call, unit = c["s_put"], c["s_call"], c["unit"]

    # Display inversion (e.g. 6J JPY/USD -> USDJPY convention): proper change of
    # variables y = 1/x with Jacobian pdf_y = pdf_x * x^2; vols untouched.
    _inv = f"{src}:{mkt}" in _DISPLAY_INVERT
    _yfns = (_yield_axis_fns(src, mkt, float(xs.min()), float(xs.max()))
             if yield_axis else None)
    _xfmt = (lambda v: f"{v:.3f}%") if _yfns else _strike_fmt
    if _yfns:
        # y = yield(x), decreasing: pdf_y = pdf_x·|dx/dy|, arrays reversed; the σ
        # sides swap (higher price = lower yield), labels follow the display side.
        _p2y, _y2p = _yfns
        y_arr = np.asarray(_p2y(xs))
        _J = np.abs(np.gradient(xs, y_arr))
        pdf = (pdf * _J)[::-1]
        sym = (sym * _J)[::-1]
        xs = y_arr[::-1]
        _F_d = float(_p2y(F))
        _lo_d, _hi_d = float(_p2y(F + s_call)), float(_p2y(F - s_put))
        _vlines = ((_lo_d, f"−1σ {_xfmt(_lo_d)}"),
                   (_hi_d, f"+1σ {_xfmt(_hi_d)}"))
    elif _inv:
        pdf = (pdf * xs ** 2)[::-1]
        sym = (sym * xs ** 2)[::-1]
        xs = (1.0 / xs)[::-1]
        _F_d = 1.0 / F
        _lo_d, _hi_d = 1.0 / (F + s_call), 1.0 / (F - s_put)   # swap sides under 1/x
        _vlines = ((_lo_d, f"−1σ {_strike_fmt(_lo_d)}"),
                   (_hi_d, f"+1σ {_strike_fmt(_hi_d)}"))
    else:
        _F_d = F
        _vlines = ((F - s_put, f"−1σ {_strike_fmt(F - s_put)}"),
                   (F + s_call, f"+1σ {_strike_fmt(F + s_call)}"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=pdf, mode="lines", fill="tozeroy",
                             line=dict(color="#2563EB", width=2),
                             fillcolor="rgba(37,99,235,0.15)",
                             name="skewed (wings)",
                             hovertemplate="%{x:,.6~f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=sym, mode="lines",
                             line=dict(color="#94A3B8", width=1.5, dash="dash"),
                             name="symmetric (ATM only)",
                             hovertemplate="%{x:,.6~f}<extra></extra>"))
    _und = mkinfo.get("fut_sym")
    fig.add_vline(x=_F_d, line_color="#1E293B", line_width=1.5,
                  annotation_text=f"F {_xfmt(_F_d)}" + (f" · {_und}" if _und else ""),
                  annotation_font_size=11)
    for x_, txt in _vlines:
        fig.add_vline(x=x_, line_color="#CBD5E1", line_width=1, line_dash="dot",
                      annotation_text=txt, annotation_font_size=10,
                      annotation_position="bottom")
    fig.update_layout(
        title=dict(text=f"{label} — implied distribution at {tenor} "
                        f"(σ± {('%.4g' % s_put)}/{('%.4g' % s_call)} {unit})",
                   font_size=13),
        height=340, margin=dict(l=30, r=30, t=48, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right",
                    font_size=10),
        xaxis=dict(gridcolor="#F1F5F9",
                   title="implied yield · %" if _yfns else None),
        yaxis=dict(visible=False),
        plot_bgcolor="#FFFFFF",
    )
    return fig


# ── Breeden–Litzenberger implied density ──────────────────────────────────────
def _compute_bl_density(src: str, mkt: str, tenor: str, tenor_days: int,
                        n_grid: int = 400, min_strikes: int = 7) -> dict:
    """Model-free risk-neutral density f(K) = e^{rT}·∂²C/∂K² from the source module's
    CACHED smile surface, for the LISTED expiry nearest the tenor target (BL needs a real
    expiry — no cross-expiry interpolation). No new fetches, no extrapolation beyond quoted
    strikes. PURE (no Streamlit calls) so the offline verify script can call it.

    Pipeline: pick the surface expiry minimising |dte − tenor_days|; read F, T, r straight
    off the fitted surface (it carries them per expiry); build a dense ~n_grid strike grid
    spanning ONLY [min_strike, max_strike] of that expiry; for each K read IV from the
    surface readout and price a call (Black-76 for v2, Bachelier for rates); take the
    central second finite difference and scale by e^{rT}. Negatives (spline noise) are
    clipped to 0 and their integrated magnitude reported as `clipped_mass`; the density is
    integrated over the in-range grid and reported as `total_mass` (< 1 — tails truncated;
    NOT renormalised). Returns a result dict; ok=False + error on insufficient data so the
    caller falls back to the split-normal chart."""
    try:
        curve, tdate_str, err = _load_market(src, mkt)
        if not tdate_str:
            return {"ok": False, "error": err or "market failed / no data"}
        if src == "rates":
            m = _ro._MARKETS_RATES[mkt]
            dv01, _dv01_ok = _market_dv01(mkt, m)
            surface = _ro._build_surface_data_rates(mkt, tdate_str, m, dv01)
            iv_fn, pricer, world = (_ro._iv_from_surface_rates, _ro._bachelier, "Bachelier")
        else:
            surface = _ov2._build_surface_data(mkt, tdate_str)
            iv_fn, pricer, world = (_ov2._iv_from_surface, _ov2._b76, "Black-76")
        if not surface:
            return {"ok": False, "error": "no smile surface (chain unavailable)"}

        today = date.today()
        exp = min(surface.keys(), key=lambda e: abs((e - today).days - tenor_days))
        dte = (exp - today).days

        # WIDE chain for the smile fit — the Pricer surface only spans ±3σ strikes,
        # which truncates exactly the fat tail that skew creates (Rajat 2026-07-31:
        # "S&P put-side vol is higher but the metric shows the opposite" — the far
        # left tail was being cut off, understating put-side probability). Build a
        # ±6σ / 30-per-side chain (cached by the builders' own keys) and fit a
        # shape-preserving PCHIP through the OTM smile (no cubic-spline overshoot —
        # this also fixes TY's >100% mass artifact).
        if src == "rates":
            m2 = _ro._MARKETS_RATES[mkt]
            data2 = _ro._load_data(mkt, tdate_str)
            wide = _ro._build_chain(mkt, exp, 30, 6.0, data2, m2["r"], dv01,
                                    cache_date=tdate_str)
            r = float(m2["r"])
        else:
            m2 = _ov2._MARKETS_V2[mkt]
            data2 = _sanitize_v2_defs(_ov2._load_data(mkt, tdate_str))
            wide = _ov2._build_chain(mkt, exp, 30, 6.0, data2, cache_date=tdate_str)
            r = float(m2["r"])
        if wide is None or wide.empty:
            return {"ok": False, "error": "wide chain unavailable for this expiry"}
        F = float(wide["F"].iloc[0])
        T = float(wide["T"].iloc[0])
        und_sym = (str(wide["fut_symbol"].iloc[0])
                   if "fut_symbol" in wide.columns else None)

        # OTM smile points: puts below F, calls above (most liquid side), fallback
        # to the other side when missing.
        # Premium-quality floor: deep-OTM settlements pinned at minimum tick produce
        # garbage IVs that wreck the spline curvature (TY showed 27% clipped mass at
        # ±6σ). Keep a smile point only if its OTM premium exceeds 3× the smallest
        # positive premium in the chain (≈ 3 ticks).
        _prem_all = []
        for _c in ("call_p", "put_p"):
            if _c in wide.columns:
                _prem_all += [float(v) for v in wide[_c].dropna() if float(v) > 0]
        _min_prem = 2.0 * min(_prem_all) if _prem_all else 0.0
        pts = []
        for _, row2 in wide.iterrows():
            k2 = float(row2["strike"])
            if k2 < F:
                iv2, pr2 = row2.get("put_iv"), row2.get("put_p")
            else:
                iv2, pr2 = row2.get("call_iv"), row2.get("call_p")
            if iv2 is None or not (isinstance(iv2, float) and math.isfinite(iv2) and iv2 > 0):
                iv2, pr2 = ((row2.get("call_iv"), row2.get("call_p")) if k2 < F
                            else (row2.get("put_iv"), row2.get("put_p")))
            try:
                _pok = pr2 is not None and float(pr2) > _min_prem
            except Exception:
                _pok = False
            if (_pok and iv2 is not None and isinstance(iv2, float)
                    and math.isfinite(iv2) and iv2 > 0):
                pts.append((k2, float(iv2), float(pr2)))
        pts = sorted({k: (v, pr) for k, v, pr in pts}.items())
        pts = [(k, v, pr) for k, (v, pr) in pts]
        n_listed = len(pts)
        kmin = pts[0][0] if pts else 0.0
        kmax = pts[-1][0] if pts else 0.0
        base = {"expiry": exp.isoformat(), "dte": dte, "n_strikes": n_listed,
                "F": F, "T": T, "world": world, "tdate": tdate_str, "tenor": tenor,
                "kmin": kmin, "kmax": kmax, "und_sym": und_sym}
        if n_listed < min_strikes or not (kmax > kmin > 0) or not (T > 0):
            return {**base, "ok": False,
                    "error": f"only {n_listed} usable strikes / bad range "
                             f"(need ≥ {min_strikes})"}

        # Smooth the SMILE, not (only) the density: settlement IVs on coarse-tick
        # markets (TY 1/64s, JPY 5-pip strikes) jitter point-to-point; an
        # interpolating fit (PCHIP) preserves that noise and differentiating it
        # produced visible squiggles (Rajat 2026-07-31). A weighted least-squares
        # quartic in standardized moneyness rejects the noise by construction —
        # weights ~ sqrt(premium) favour reliable near-ATM quotes over wing scraps.
        _k_pts = np.array([k for k, _v, _p in pts])
        _iv_pts = np.array([v for _k, v, _p in pts])
        _pr_pts = np.array([max(p2, 0.0) for _k, _v, p2 in pts])
        _atm0 = float(_iv_pts[int(np.argmin(np.abs(_k_pts - F)))])
        if world == "Bachelier":
            _m_pts = (_k_pts - F) / max(_atm0 * math.sqrt(T), 1e-12)
        else:
            _m_pts = np.log(_k_pts / F) / max(_atm0 * math.sqrt(T), 1e-12)
        _w = np.sqrt(_pr_pts)
        _w = np.where(_w > 0, _w, _w[_w > 0].min() if (_w > 0).any() else 1.0)
        _deg = 4 if n_listed >= 12 else 2
        _coef = np.polyfit(_m_pts, _iv_pts, _deg, w=_w)

        def _smile(k_arr):
            k_arr = np.asarray(k_arr, dtype=float)
            if world == "Bachelier":
                m_ = (k_arr - F) / max(_atm0 * math.sqrt(T), 1e-12)
            else:
                m_ = np.log(k_arr / F) / max(_atm0 * math.sqrt(T), 1e-12)
            iv_ = np.polyval(_coef, m_)
            return np.clip(iv_, 1e-4, None)

        Ks = np.linspace(kmin, kmax, n_grid)
        Cs = np.full(n_grid, np.nan)
        _ivg = _smile(Ks)
        for i in range(n_grid):
            iv = _ivg[i]
            if iv is not None and math.isfinite(iv) and iv > 0:
                try:
                    Cs[i] = pricer(F, float(Ks[i]), T, r, float(iv), "C")
                except Exception:
                    pass
        good = np.isfinite(Cs)
        if good.sum() < min_strikes:
            return {**base, "ok": False, "error": "call-price grid mostly undefined"}
        if not good.all():
            # Fill isolated gaps so the 2nd difference is defined everywhere in range.
            Cs = np.interp(Ks, Ks[good], Cs[good])

        dK = Ks[1] - Ks[0]
        d2C = np.full(n_grid, np.nan)
        d2C[1:-1] = (Cs[2:] - 2.0 * Cs[1:-1] + Cs[:-2]) / (dK * dK)
        dens = math.exp(r * T) * d2C
        Ki = Ks[1:-1]
        fi = dens[1:-1]

        # Survival curve P(S > K) from the FIRST derivative (digital-call prices):
        # P(S>K) = −e^{rT}·dC/dK — an order of magnitude more stable than the
        # second-derivative density against settlement noise, and the standard way
        # desks quote implied probabilities. Clipped to [0,1] and forced monotone
        # non-increasing (tiny convexity violations in the fitted smile show up as
        # sub-bp wiggles here rather than the density's wild spikes).
        dC = np.gradient(Cs, dK)
        surv = np.clip(-math.exp(r * T) * dC, 0.0, 1.0)
        surv = np.minimum.accumulate(surv)

        _trapz = getattr(np, "trapezoid", np.trapz)   # np≥2.0 renamed trapz→trapezoid
        neg = fi < 0
        clipped_mass = float(_trapz(np.where(neg, -fi, 0.0), Ki))  # integrated |negatives|
        fi_c = np.where(neg, 0.0, fi)
        total_mass = float(_trapz(fi_c, Ki))
        mode = float(Ki[int(np.argmax(fi_c))]) if len(Ki) else float("nan")
        try:
            base["atm_iv"] = float(_smile(F))
        except Exception:
            base["atm_iv"] = None
        base["smile_k"] = _k_pts.tolist()
        base["smile_iv"] = _iv_pts.tolist()
        base["smile_fit_k"] = Ks.tolist()
        base["smile_fit_iv"] = np.asarray(_ivg, dtype=float).tolist()
        if src == "rates":
            base["dv01"] = float(dv01) if dv01 else None
        return {**base, "ok": True, "Ks": Ki, "dens": fi_c,
                "surv_K": Ks, "surv": surv,
                "total_mass": total_mass, "clipped_mass": clipped_mass, "mode": mode}
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}


def _bl_density_figure(bl: dict, label: str, tenor: str, src: str, mkt: str,
                       tenor_days: int, mkinfo: dict, yield_axis: bool = False):
    """Market-implied PROBABILITY view of the BL density (Rajat 2026-07-31: the raw
    pdf curve was unreadable). Two visual layers from the same density:
      1. round-number price buckets, each bar = smile-implied probability of
         settling there (bucket masses from the BL CDF, renormalized to in-range
         mass so the bars sum to 100% of what the listed strikes cover);
      2. the percentile cone — 5/25/50/75/95 levels as dashed vlines with labels.
    Display-inversion (USDJPY convention) applied with the proper Jacobian."""
    import plotly.graph_objects as go

    if "surv" not in bl:      # stale pre-redesign session cache — force a re-click
        return None
    F = float(bl["F"])
    _inv = f"{src}:{mkt}" in _DISPLAY_INVERT
    # Probabilities from the SURVIVAL curve (digital-call first derivative — stable),
    # not the noisy second-derivative density. CDF(K) = 1 − P(S>K).
    sK = np.asarray(bl["surv_K"], dtype=float)
    sv = np.asarray(bl["surv"], dtype=float)
    _yfns = (_yield_axis_fns(src, mkt, float(sK.min()), float(sK.max()))
             if yield_axis else None)
    if _yfns:
        # display y = yield(K), decreasing: P(display ≤ y) = P(price ≥ K) = surv(K)
        _p2y, _y2p = _yfns
        Ks = np.asarray(_p2y(sK))[::-1]
        cdf = sv[::-1]
        F = float(_p2y(F))
    elif _inv:
        # display y = 1/K: P(display ≤ y) = P(native ≥ 1/y) = surv(1/y)
        Ks = (1.0 / sK)[::-1]
        cdf = sv[::-1]
        F = 1.0 / F
    else:
        Ks = sK
        cdf = 1.0 - sv
    _xfmt = (lambda v: f"{v:.3f}%") if _yfns else _strike_fmt
    # normalize the covered range so the density integrates to the shown mass
    c0, c1 = float(cdf[0]), float(cdf[-1])
    if not (c1 > c0):
        return None
    cdf = (cdf - c0) / (c1 - c0)

    # ── Smooth continuous density from the (stable, monotone) CDF ────────────
    # Resample onto a uniform display grid, differentiate, light Gaussian smooth —
    # full-smile accuracy with the readable shape of the original split-normal
    # chart (Rajat 2026-07-31: "continuous and intuitive like the first graph but
    # more accurate, with a normal overlay to see how skew moves probability").
    xs_u = np.linspace(float(Ks[0]), float(Ks[-1]), 400)
    cdf_u = np.interp(xs_u, Ks, cdf)
    dens_u = np.clip(np.gradient(cdf_u, xs_u), 0.0, None)
    _kw = 7   # ~kernel std in grid points (400-pt grid → ~2% of range)
    _g = np.exp(-0.5 * (np.arange(-3 * _kw, 3 * _kw + 1) / _kw) ** 2)
    _g /= _g.sum()
    dens_s = np.convolve(dens_u, _g, mode="same")

    # Percentile cone from the CDF
    pct_levels = [5, 25, 50, 75, 95]
    pcts = {q: float(np.interp(q / 100.0, cdf_u, xs_u)) for q in pct_levels}

    # No-skew overlay: flat-ATM-vol distribution in DISPLAY space (lognormal for
    # v2 / normal for rates; inverted with the Jacobian when displaying USDJPY).
    bench = None
    _atm = bl.get("atm_iv")
    if _atm and _atm > 0:
        try:
            T_ = float(bl["T"])
            F_n = float(bl["F"])            # native forward
            if _yfns:
                k_nat = np.asarray(_y2p(xs_u))
            elif _inv:
                k_nat = 1.0 / xs_u
            else:
                k_nat = xs_u
            s_ = _atm * math.sqrt(T_)
            if bl["world"] == "Bachelier":
                pdf_nat = np.exp(-0.5 * ((k_nat - F_n) / s_) ** 2) / (s_ * math.sqrt(2 * math.pi))
            else:
                pdf_nat = (np.exp(-0.5 * ((np.log(k_nat / F_n) + 0.5 * s_ * s_) / s_) ** 2)
                           / (k_nat * s_ * math.sqrt(2 * math.pi)))
            # Jacobian into display space — 1/x: pdf_y = pdf_K·K² (k_nat = 1/y);
            # yield: pdf_y = pdf_K·|dp/dy| (numerical, from the inverse map).
            if _inv:
                bench = pdf_nat * k_nat ** 2
            elif _yfns:
                bench = pdf_nat * np.abs(np.gradient(k_nat, xs_u))
            else:
                bench = pdf_nat
        except Exception:
            bench = None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_u, y=dens_s, mode="lines", fill="tozeroy",
                             line=dict(color="#2563EB", width=2),
                             fillcolor="rgba(37,99,235,0.15)",
                             name="smile-implied (full chain)",
                             hovertemplate="%{x:,.6~f}<extra></extra>"))
    if bench is not None:
        fig.add_trace(go.Scatter(x=xs_u, y=bench, mode="lines",
                                 line=dict(color="#94A3B8", width=1.5, dash="dash"),
                                 name="no-skew (flat ATM vol)",
                                 hovertemplate="%{x:,.6~f}<extra></extra>"))
    # Symmetric NORMAL centred exactly at F (Rajat: separates the lognormal "lean"
    # from true skew). Lognormal worlds only — for rates the dashed overlay already
    # IS a symmetric normal at F.
    if bl["world"] != "Bachelier" and _atm and _atm > 0:
        _sd = float(F) * _atm * math.sqrt(float(bl["T"]))
        _symn = (np.exp(-0.5 * ((xs_u - float(F)) / _sd) ** 2)
                 / (_sd * math.sqrt(2 * math.pi)))
        fig.add_trace(go.Scatter(x=xs_u, y=_symn, mode="lines",
                                 line=dict(color="#0D9488", width=1.5, dash="dot"),
                                 name="symmetric normal @ F",
                                 hovertemplate="%{x:,.6~f}<extra></extra>"))
    _und = bl.get("und_sym")
    fig.add_vline(x=F, line_color="#1E293B", line_width=1.5,
                  annotation_text=f"F {_xfmt(F)}" + (f" · {_und}" if _und else ""),
                  annotation_font_size=11)
    for q in (25, 50, 75):
        fig.add_vline(x=pcts[q], line_color="#CBD5E1", line_width=1, line_dash="dot",
                      annotation_text=f"P{q} {_xfmt(pcts[q])}",
                      annotation_font_size=9, annotation_position="bottom")
    _cone = "  ·  ".join(f"P{q} {_xfmt(pcts[q])}" for q in pct_levels)
    fig.add_annotation(text=f"cone: {_cone}", xref="paper", yref="paper",
                       x=0.5, y=1.13, showarrow=False, font_size=11,
                       font_color="#475569")
    fig.update_layout(
        title=dict(text=(f"{label} — market-implied distribution (full smile) · "
                         f"expiry {bl['expiry']} ({bl['dte']}d"
                         + (f", underlying {_und}" if _und else "")
                         + f") · covers {bl['total_mass'] * 100:.0f}% "
                           f"of listed-strike mass"),
                   font_size=13),
        height=380, margin=dict(l=30, r=30, t=70, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right",
                    font_size=10),
        xaxis=dict(gridcolor="#F1F5F9",
                   title="implied yield · %" if _yfns else None),
        yaxis=dict(visible=False),
        plot_bgcolor="#FFFFFF",
    )
    return fig


# ── Underlying futures history (Rajat 2026-07-31: "click TY → 6m of the underlying") ──
# Midcurve STIR options underlie a DEFERRED quarterly (SOFR_1Y → SR3U7, not front
# SR3U6 — Rajat caught the mismatch 2026-08-02): these markets fetch the build's
# actual underlying contract (raw symbol) instead of the volume-rolled front.
_MIDCURVE_UND = {"SOFR_1Y", "SOFR_2Y", "ER_1Y", "ER_2Y", "SONIA_1Y", "SONIA_2Y"}


def _underlying_root(src: str, mkt: str):
    m = (_ro._MARKETS_RATES if src == "rates" else _ov2._MARKETS_V2)[mkt]
    return m["ds"], m["fut_sym"].split(".")[0]


_UND_KEEP_ROWS = 560        # ~2.2y of daily bars (CTA MA-200 + warmup needs ~210)
_UND_DEPTH = 2              # bump → existing shallower stores full-refetch once
_UND_RECHECK_S = 4 * 3600   # retry window when the newest settlement isn't out yet


def _und_path(root: str) -> str:
    return os.path.join(os.path.dirname(__file__), "vol_dash_cache", f"UND_{root}.pkl")


def _und_is_fresh(src: str, mkt: str, contract: str = None) -> bool:
    """True when the rolling store already covers the last business day (or was
    checked recently) — used by the class prefetch and the spinner wording."""
    import pickle as _pkl
    import time as _t
    try:
        _ds, root = _underlying_root(src, mkt)
        with open(_und_path(contract or root), "rb") as fh:
            blob = _pkl.load(fh)
        if blob.get("depth") != _UND_DEPTH:
            return False     # pre-CTA shallow store → needs the one-time deep refetch
        if blob["df"]["date"].iloc[-1] >= _prev_bday(date.today(), 1):
            return True
        return (_t.time() - blob.get("checked_at", 0)) < _UND_RECHECK_S
    except Exception:
        return False


def _underlying_bars(src: str, mkt: str, contract: str = None):
    """Daily closes for the market's underlying future via Databento CONTINUOUS
    front-month symbology ({root}.v.0, volume-rolled — resolves on GLBX/IFEU/IFLL/
    XEUR alike). NEVER use parent symbology here: ICE parents bill the whole complex
    (the $6.2 BRN lesson — parent BRN.FUT ohlcv-1d probes at $6.74).

    INCREMENTAL rolling store (Rajat 2026-07-31: "next time only fill the gap"):
    ONE pickle per root ({df, checked_at, contract}); first build fetches ~6m, after
    that only the missing tail since the last stored bar is fetched (typically 1 day,
    fractions of a cent). A 4h recheck guard stops rerun storms while the newest
    settlement isn't published yet. History trimmed to ~10 months. Costs auto-recorded
    via _ov2._get_range's ledger hook. Returns (DataFrame[date, close, ...], err)."""
    import pickle as _pkl
    import databento as db
    import pandas as pd
    import glob as _glob
    import time as _t
    try:
        ds, root = _underlying_root(src, mkt)
    except Exception as ex:
        return None, f"no config: {ex}"
    _dir = os.path.join(os.path.dirname(__file__), "vol_dash_cache")
    fp = _und_path(contract or root)   # midcurves: per-contract store (e.g. SR3U7)

    # ICE/Eurex ohlcv-1d emits per-SESSION bars (2 per date) and CME includes the
    # Sunday-evening partial — collapse to one close per date (last bar wins).
    def _one_per_date(d):
        return d.groupby("date", as_index=False).last()

    def _fetch(start_iso: str):
        client = db.Historical(key=_api_key())
        _sym = contract if contract else f"{root}.v.0"
        _stin = "raw_symbol" if contract else "continuous"
        try:
            raw = _ov2._get_range(client, ds, [_sym], "ohlcv-1d",
                                  start_iso, date.today().isoformat(),
                                  stype_in=_stin)
        except Exception as ex:
            # Young datasets (XEUR.EOBI starts 2025-03-10) 422 on a deep
            # backfill instead of returning the partial range — clamp to the
            # advertised available start from the error and retry once.
            # Killed the RX/OE underlying charts before 2026-08-14.
            import re as _re
            _m = _re.search(
                r"available start of dataset \S+ \('(\d{4}-\d{2}-\d{2})",
                str(ex))
            if not _m or _m.group(1) >= date.today().isoformat():
                raise
            raw = _ov2._get_range(client, ds, [_sym], "ohlcv-1d",
                                  _m.group(1), date.today().isoformat(),
                                  stype_in=_stin)
        df = raw.to_df().reset_index()
        if df.empty:
            return None, None
        df["date"] = pd.to_datetime(df["ts_event"]).dt.date
        keep = ["date", "close"] + [c for c in ("symbol", "instrument_id")
                                    if c in df.columns]
        out = _one_per_date(df[keep].dropna(subset=["close"]).reset_index(drop=True))
        if out.empty:
            return None, None
        if contract:               # fixed-contract fetch: label is the contract itself
            return out, contract
        ctr = None
        if "instrument_id" in out.columns:
            try:   # which actual contract is the front right now (free symbology call)
                iid = str(int(out["instrument_id"].iloc[-1]))
                res = client.symbology.resolve(
                    dataset=ds, symbols=[iid], stype_in="instrument_id",
                    stype_out="raw_symbol",
                    start_date=out["date"].iloc[-1].isoformat())
                mapping = (res or {}).get("result", {}).get(iid, [])
                if mapping:
                    ctr = mapping[0].get("s", "")
                    # ICE raw outrights read "SO3 FMU0026!" — prettify to SO3U6
                    import re as _re
                    _m = _re.match(r"^(\S+) FM([FGHJKMNQUVXZ])(\d{4})!$", ctr or "")
                    if _m:
                        ctr = f"{_m.group(1)}{_m.group(2)}{_m.group(3)[-1]}"
            except Exception:
                pass
        return out, ctr

    blob = None
    try:
        with open(fp, "rb") as fh:
            blob = _pkl.load(fh)
        if blob.get("depth") != _UND_DEPTH:
            blob = None          # shallower store (pre-CTA) → one-time full refetch
    except Exception:
        # migrate a legacy per-day pickle (UND_{root}_{date}.pkl) if one exists
        legacy = sorted(_glob.glob(os.path.join(_dir, f"UND_{root}_20*.pkl")))
        if legacy:
            try:
                with open(legacy[-1], "rb") as fh:
                    ldf = _one_per_date(_pkl.load(fh))
                ctr = (str(ldf["contract"].iloc[-1])
                       if "contract" in ldf.columns else None)
                blob = {"df": ldf.drop(columns=["contract"], errors="ignore"),
                        "checked_at": 0, "contract": ctr}
            except Exception:
                blob = None

    def _with_contract(d, ctr):
        d = d.copy()
        if ctr:
            d["contract"] = ctr
        return d

    try:
        if blob is None:
            out, contract = _fetch((date.today() - timedelta(days=820)).isoformat())
            if out is None:
                return None, "no bars returned"
            blob = {"df": out.drop(columns=["contract"], errors="ignore"),
                    "checked_at": _t.time(), "contract": contract,
                    "depth": _UND_DEPTH}
        else:
            last = blob["df"]["date"].iloc[-1]
            stale = last < _prev_bday(date.today(), 1)
            recheck_ok = (_t.time() - blob.get("checked_at", 0)) >= _UND_RECHECK_S
            if stale and recheck_ok:
                out, contract = _fetch((last + timedelta(days=1)).isoformat())
                blob["checked_at"] = _t.time()
                if out is not None:
                    new = out[out["date"] > last]
                    if len(new):
                        blob["df"] = _one_per_date(
                            pd.concat([blob["df"], new], ignore_index=True)
                        ).sort_values("date").reset_index(drop=True)
                        blob["df"] = blob["df"].iloc[-_UND_KEEP_ROWS:].reset_index(drop=True)
                        if contract:
                            blob["contract"] = contract
        os.makedirs(_dir, exist_ok=True)
        with open(fp, "wb") as fh:      # never persists empty (anti-poison rule)
            _pkl.dump(blob, fh)
        return _with_contract(blob["df"], blob.get("contract")), None
    except Exception as ex:
        if blob is not None and blob.get("df") is not None and len(blob["df"]):
            # serve the stored history even when the gap-fill fetch failed
            return _with_contract(blob["df"], blob.get("contract")), None
        return None, f"{type(ex).__name__}: {str(ex)[:160]}"


def _horizon_totvar(src: str, mkt: str, hs, key: str = "atm_iv", sigma: float = 1.0):
    """TOTAL VARIANCE σ²·t at each horizon in `hs` (days), interpolated LINEAR IN
    TIME between the cached curve's listed-expiry knots (curve column `key`:
    atm_iv / call_wing_iv / put_wing_iv from the sigma-level build). Linear-in-σ
    interpolation kinks the forward-variance subtraction at every knot (UB's
    zigzag channel, Rajat 2026-08-02) — linear total variance makes forward
    variance piecewise-constant instead. Flat-vol extrapolation before the first
    knot, flat-FORWARD-vol after the last. None when missing."""
    curve, _td, _err = _load_market(src, mkt, sigma) if sigma != 1.0 else _load_market(src, mkt)
    if not curve:
        return None
    prs = sorted({float(r["dte"]): float(r[key]) for r in curve
                  if r.get("dte") is not None
                  and isinstance(r.get(key), float)
                  and math.isfinite(r.get(key)) and r.get(key) > 0}.items())
    if not prs:
        return None
    ds_ = np.array([p[0] for p in prs])
    vs_ = np.array([p[1] for p in prs])
    W = vs_ ** 2 * ds_
    hs = np.asarray(hs, dtype=float)
    out = np.interp(hs, ds_, W)
    lo = hs < ds_[0]
    out[lo] = vs_[0] ** 2 * hs[lo]
    hi = hs > ds_[-1]
    out[hi] = W[-1] + vs_[-1] ** 2 * (hs[hi] - ds_[-1])
    return out


def _underlying_history_figure(src: str, mkt: str, label: str, df, months: int,
                               yield_axis: bool = False,
                               rvol_n: int = 10, rvol_lbl: str = "2w",
                               fixed_contract: bool = False):
    """Underlying daily closes + layered REALIZED-VOL move band (2w MVA ± k·2w
    expected move at trailing realized vol — Rajat 2026-08-02: "not textbook
    bollinger is fine, I just want a realized vol measure"; the ±kσ width is
    close·σ_real·√(14/365), the EXACT realized twin of the implied cone's 2w point,
    so band-vs-cone width = clean realized-vs-implied vol premium read) + the SAME
    2w-move band projected forward to ~3m at FORWARD 2w implied vols from the term
    structure (σ_fwd(h,h+14) via σ²·t subtraction — NOT a terminal σ·√h fan; Rajat
    2026-08-02 "same methodology consistently for both"). The implied channel is
    PURE from its first day; the LAST week of the realized band eases into the
    implied starting levels (backward blend — Rajat's call, keeps the implied side
    undistorted). Shading darkens with sigma level. Display conventions follow the
    section: 1/x inversion, price→yield."""
    import plotly.graph_objects as go
    cutoff = date.today() - timedelta(days=months * 31)
    d = df[df["date"] >= cutoff].reset_index(drop=True)
    if len(d) < 5:
        return None
    xs = list(d["date"])
    closes = np.asarray(d["close"], dtype=float)

    # ── native-space pieces ──────────────────────────────────────────────────
    _HZN = 91                              # forward-cone horizon (~3 months)
    _BLEND_D = 14                          # band→cone smoothstep ramp (bars) — 7 felt abrupt (Rajat)
    _MVA_N = 10                            # 2 trading weeks
    _BAND_HZN = 14                         # band width horizon (2w move, calendar)
    roll = d["close"].rolling(_MVA_N, min_periods=_MVA_N // 2)
    mva = roll.mean().to_numpy().copy()   # copy: backward blend writes into it
    # trailing realized vol of DAILY moves, annualized, then scaled to a 2w move —
    # the same construction as the implied cone's 2w point (realized vs implied):
    # rates: absolute pt changes (Bachelier); lognormal markets: log returns.
    # realized vol over the USER-SELECTED trailing window (2w/1m/2m/3m); the move
    # horizon stays 2w so the band remains comparable with the implied channel
    chg = d["close"].diff() if src == "rates" else np.log(d["close"]).diff()
    sig_ann = (chg.rolling(rvol_n, min_periods=max(rvol_n // 2, 5)).std().to_numpy()
               * math.sqrt(252.0))
    w1 = sig_ann * math.sqrt(_BAND_HZN / 365.0)
    if src != "rates":
        w1 = w1 * closes
    up1, dn1 = mva + w1, mva - w1
    up2, dn2 = mva + 2.0 * w1, mva - 2.0 * w1

    # forward band: SAME 2w-move methodology as the realized band (Rajat 2026-08-02:
    # "same methodology consistently for both" — NOT a terminal σ·√h fan). At each
    # forward date h the width is the 2-week move at the FORWARD 2w vol extracted
    # from the term structure: σ_fwd(h,h+14)² = (σ(h+14)²·(h+14) − σ(h)²·h)/14.
    # At h=0 this is today's 2w implied move — directly comparable to the realized
    # band's last width; forward event/meeting vol shows up as width bumps.
    c0, last_d = float(closes[-1]), xs[-1]
    hs = np.arange(0, _HZN + 1)
    fx = [last_d + timedelta(days=int(h)) for h in hs]
    m_last = float(mva[-1]) if math.isfinite(mva[-1]) else c0
    b_last = float(w1[-1]) if math.isfinite(w1[-1]) else 0.0

    # Width smoothing: forward variance off sparse noisy knots is inherently steppy
    # (UB stayed lumpy through two rounds of filtering). Rajat 2026-08-02: "make
    # intelligent assumptions — approximate is fine for a visual tool, just not
    # misleading". Assumption: a forward-vol WIDTH profile is a smooth low-frequency
    # object — fit a QUADRATIC to the (median-robustified) width series: level =
    # premium, slope = contango/decay, single bow = event cluster. A cubic still
    # S-waved on UB's sparse noisy knots (two turning points ≠ anything real in a
    # 3m window); a quadratic cannot wave at all.
    def _smooth(w):
        from scipy.ndimage import median_filter
        wm = median_filter(w, size=7, mode="nearest")
        co = np.polyfit(hs.astype(float), wm, 2)
        fit = np.polyval(co, hs.astype(float))
        return np.maximum(fit, 0.2 * float(np.median(wm)))   # positivity floor

    def _fwd_w(key, sg):
        """2w-move width per forward date from the `key` vol term structure at
        sigma level `sg`: forward variance via TOTAL-VARIANCE subtraction (smooth
        by construction), floored against inverted-structure pinch, then lightly
        Gaussian-smoothed. None if the curve/column is unusable."""
        W0 = _horizon_totvar(src, mkt, hs, key, sg)
        W14 = _horizon_totvar(src, mkt, hs + _BAND_HZN, key, sg)
        if W0 is None or W14 is None:
            return None
        vf = (W14 - W0) / float(_BAND_HZN)
        spot_var = W14 / np.maximum(hs + _BAND_HZN, 1)     # σ² at the far point
        vf = np.maximum(vf, 0.25 * spot_var)
        if not np.all(np.isfinite(vf)):
            return None
        w = np.sqrt(vf) * math.sqrt(_BAND_HZN / 365.0)
        w = _smooth(w)
        return w if src == "rates" else w * c0

    # SKEW-AWARE widths (Rajat 2026-08-02 "incorporate the implied skew"): upper
    # edges use the CALL-wing vols, lower edges the PUT-wing vols, each at the
    # matching sigma level's strikes (σ=1/σ=2 curve builds — already cached by the
    # 7-point machinery, zero fetches). ATM fallback per side when wings missing.
    w_atm = _fwd_w("atm_iv", 1.0)
    cone = None
    if w_atm is not None:
        w_up, w_dn = {}, {}
        for k, sg in ((1, 1.0), (2, 2.0)):
            wu = _fwd_w("call_wing_iv", sg)
            wd = _fwd_w("put_wing_iv", sg)
            w_up[k] = wu if wu is not None else w_atm
            w_dn[k] = wd if wd is not None else w_atm
        # PURE implied from day zero (Rajat 2026-08-02: "from the time the implied
        # cloud starts it should represent the actual vol, not the blend — blend
        # backwards inside the hist cloud instead")
        cone = {"center": np.full(len(hs), c0)}
        for k in (1, 2):
            cone[f"up{k}"] = c0 + k * w_up[k]
            cone[f"dn{k}"] = c0 - k * w_dn[k]
        # backward blend: the LAST _BLEND_D bars of the realized band ease into the
        # implied channel's starting levels, so the junction is seamless without
        # distorting the implied side
        B = min(_BLEND_D, len(closes) - 1)
        cone["B"] = B if B >= 2 else 0
        # keep the UNBLENDED band edges over the blend zone — drawn as dotted lines
        # so the user sees where the realized cloud would have been (Rajat 2026-08-02)
        cone["raw"] = {nm: arr[-(B + 1):].copy() for nm, arr in
                       (("up1", up1), ("dn1", dn1), ("up2", up2), ("dn2", dn2))}
        if B >= 2:
            sb = np.arange(B + 1) / float(B)
            sb = sb * sb * (3.0 - 2.0 * sb)        # smoothstep 0→1
            for arr, tgt in ((up1, cone["up1"][0]), (dn1, cone["dn1"][0]),
                             (up2, cone["up2"][0]), (dn2, cone["dn2"][0]),
                             (mva, c0)):
                seg = arr[-(B + 1):]
                ok = np.isfinite(seg)
                seg[ok] = (1.0 - sb[ok]) * seg[ok] + sb[ok] * tgt
                arr[-(B + 1):] = seg

    # ── display transform (built over the FULL range incl. band/cone) ────────
    _allv = [closes, up2[np.isfinite(up2)], dn2[np.isfinite(dn2)]]
    if cone is not None:
        _allv += [cone["up2"], cone["dn2"]]
    _gmin = min(float(np.nanmin(a)) for a in _allv if len(a))
    _gmax = max(float(np.nanmax(a)) for a in _allv if len(a))
    _yfns = _yield_axis_fns(src, mkt, _gmin, _gmax) if yield_axis else None
    _inv = f"{src}:{mkt}" in _DISPLAY_INVERT
    if _yfns:
        _t = lambda a: np.asarray(_yfns[0](a)) if a is not None else None
        ytitle = "implied yield · %"
        _fmt = lambda v: f"{v:.3f}%"
    elif _inv:
        _t = lambda a: (1.0 / np.asarray(a)) if a is not None else None
        ytitle = "price"
        _fmt = _strike_fmt
    else:
        _t = lambda a: a
        ytitle = "price"
        _fmt = _strike_fmt
    ys = _t(closes)
    mva_d = _t(mva)
    b_up1, b_dn1, b_up2, b_dn2 = _t(up1), _t(dn1), _t(up2), _t(dn2)

    fig = go.Figure()
    # ── layered realized fills: light core (±1σ), darker rings (1σ→2σ) ───────
    # The blend zone (last B bars) is drawn as per-day segments whose fill colour
    # graduates blue→teal, so the realized→implied hand-off is visible (Rajat
    # 2026-08-02: "blend the colour too so the user gets an idea what's happening").
    _B_IN, _B_OUT = "rgba(37,99,235,0.07)", "rgba(37,99,235,0.16)"
    _bz = int(cone.get("B", 0)) if cone is not None else 0
    _n = len(xs)
    _e = _n - 1 - _bz if _bz else _n - 1          # last index of the solid-blue part
    fig.add_trace(go.Scatter(x=xs[:_e + 1], y=b_dn2[:_e + 1], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs[:_e + 1], y=b_dn1[:_e + 1], mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor=_B_OUT,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs[:_e + 1], y=b_up1[:_e + 1], mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor=_B_IN,
                             name=f"realized ±1σ (2w move, {rvol_lbl} vol)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs[:_e + 1], y=b_up2[:_e + 1], mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor=_B_OUT,
                             name="realized ±2σ", hoverinfo="skip"))
    if _bz:
        # blue(37,99,235) → teal(13,148,136), alpha ramps toward the cone's values
        def _mixc(s_, a0, a1):
            return (f"rgba({round(37 + (13 - 37) * s_)},"
                    f"{round(99 + (148 - 99) * s_)},"
                    f"{round(235 + (136 - 235) * s_)},{a0 + (a1 - a0) * s_:.3f})")

        def _seg(j, ylo, yhi, colr):
            fig.add_trace(go.Scatter(x=[xs[j], xs[j + 1]], y=[ylo[j], ylo[j + 1]],
                                     mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[xs[j], xs[j + 1]], y=[yhi[j], yhi[j + 1]],
                                     mode="lines", line=dict(width=0),
                                     fill="tonexty", fillcolor=colr,
                                     showlegend=False, hoverinfo="skip"))

        for j in range(_e, _n - 1):
            sm = ((j - _e) + 0.5) / float(_bz)
            sm = sm * sm * (3.0 - 2.0 * sm)
            _seg(j, b_dn2, b_dn1, _mixc(sm, 0.16, 0.22))
            _seg(j, b_dn1, b_up1, _mixc(sm, 0.07, 0.10))
            _seg(j, b_up1, b_up2, _mixc(sm, 0.16, 0.22))
        # dotted ghost of the UNBLENDED realized edges across the blend zone —
        # ±1σ thin, ±2σ slightly heavier, one legend entry per sigma level
        _raw = cone.get("raw") if cone is not None else None
        if _raw:
            _xz = xs[-(_bz + 1):]
            for _nm2, _arr2 in _raw.items():
                _is2 = _nm2.endswith("2")
                fig.add_trace(go.Scatter(
                    x=_xz, y=_t(_arr2), mode="lines",
                    line=dict(color="#2563EB", width=1.1 if _is2 else 1.8,
                              dash="dot"),
                    name=("realized ±2σ w/o blend" if _is2
                          else "realized ±1σ w/o blend"),
                    showlegend=_nm2.startswith("up"),
                    hovertemplate="%{x} · %{y:,.6~f}<extra></extra>"))
    if mva_d is not None:
        fig.add_trace(go.Scatter(x=xs, y=mva_d, mode="lines",
                                 line=dict(color="#64748B", width=1.2, dash="dash"),
                                 name="2w MVA",
                                 hovertemplate="%{x} · %{y:,.6~f}<extra></extra>"))
    # ── implied forward cone (term structure to ~3m), same light→dark layering ──
    if cone is not None:
        _C_IN, _C_OUT = "rgba(13,148,136,0.10)", "rgba(13,148,136,0.22)"
        c_up1, c_dn1 = _t(cone["up1"]), _t(cone["dn1"])
        c_up2, c_dn2 = _t(cone["up2"]), _t(cone["dn2"])
        c_ctr = _t(cone["center"])
        fig.add_trace(go.Scatter(x=fx, y=c_dn2, mode="lines",
                                 line=dict(color="#0D9488", width=1, dash="dot"),
                                 showlegend=False,
                                 hovertemplate="%{x} · %{y:,.6~f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fx, y=c_dn1, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_C_OUT,
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fx, y=c_up1, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=_C_IN,
                                 name="implied 2w move ±1σ (fwd vol, to 3m)",
                                 hovertemplate="%{x} · %{y:,.6~f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fx, y=c_up2, mode="lines",
                                 line=dict(color="#0D9488", width=1, dash="dot"),
                                 fill="tonexty", fillcolor=_C_OUT,
                                 name="implied ±2σ",
                                 hovertemplate="%{x} · %{y:,.6~f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fx, y=c_ctr, mode="lines",
                                 line=dict(color="#94A3B8", width=1, dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color="#000000", width=3.6),
                             name="close",
                             hovertemplate="%{x} · %{y:,.6~f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[xs[-1]], y=[float(ys[-1])], mode="markers+text",
                             marker=dict(size=7, color="#000000"),
                             text=[_fmt(float(ys[-1]))], textposition="middle right",
                             textfont_size=11, showlegend=False,
                             hoverinfo="skip"))
    _ctr_sym = (str(df["contract"].iloc[-1] or "")
                if "contract" in df.columns else "")
    if fixed_contract and _ctr_sym:
        _src_note = f"option underlying {_ctr_sym}"     # midcurves: deferred quarterly
    else:
        _src_note = ("continuous front month"
                     + (f" · currently {_ctr_sym}" if _ctr_sym else ""))
    fig.update_layout(
        title=dict(text=f"{label} — underlying future, last {months}m "
                        f"({_src_note})", font_size=13),
        height=520, margin=dict(l=30, r=70, t=58, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right",
                    font_size=10),
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(title=ytitle, gridcolor="#F1F5F9",
                   tickformat=".6~f" if max(abs(ys)) < 1 else None),
        plot_bgcolor="#FFFFFF",
    )
    return fig


# ── Optional technicals under the cloud chart (Rajat 2026-08-04): RSI (14/30d)
# and the CTA-tab Combined momentum signal, computed on the SAME rolling
# underlying store (display-convention prices, so USDJPY reads conventionally).
def _disp_closes(src: str, mkt: str, df):
    c = df["close"].astype(float)
    return (1.0 / c) if f"{src}:{mkt}" in _DISPLAY_INVERT else c


def _rsi_figure(src: str, mkt: str, label: str, df, months: int, n: int):
    import plotly.graph_objects as go
    import pandas as pd
    import ta as ta_lib
    df = df[[d.weekday() < 5 for d in df["date"]]].reset_index(drop=True)
    closes = pd.Series(_disp_closes(src, mkt, df).to_numpy(dtype=float))
    if len(closes) < n + 5:
        return None
    rsi = ta_lib.momentum.RSIIndicator(closes, window=n).rsi().to_numpy()
    cutoff = date.today() - timedelta(days=months * 31)
    m = np.array([d >= cutoff for d in df["date"]]) & np.isfinite(rsi)
    if m.sum() < 5:
        return None
    xs = [d for d, k in zip(df["date"], m) if k]
    ys = rsi[m]
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(220,38,38,0.05)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(22,163,74,0.05)", line_width=0)
    for lv, c_ in ((70, "#DC2626"), (30, "#16A34A")):
        fig.add_hline(y=lv, line_color=c_, line_width=1, line_dash="dot")
    fig.add_hline(y=50, line_color="#CBD5E1", line_width=1)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color="#7C3AED", width=1.8), name=f"RSI {n}d",
                             hovertemplate="%{x} · %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[xs[-1]], y=[float(ys[-1])], mode="markers+text",
                             marker=dict(size=6, color="#7C3AED"),
                             text=[f"{ys[-1]:.1f}"], textposition="middle right",
                             textfont_size=11, showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        title=dict(text=f"{label} — RSI ({n}d)", font_size=13),
        height=210, margin=dict(l=30, r=70, t=40, b=24), showlegend=False,
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(range=[0, 100], gridcolor="#F1F5F9", dtick=25),
        plot_bgcolor="#FFFFFF")
    return fig


# CTA-tab default parameters (cta.py render defaults) — keep in sync so this
# mini-chart matches the CTA tab's Combined signal for the same asset.
_CTA_P = dict(tsmom=126, ma_fast=20, ma_slow=200, don=55, ewma=63, vol_days=21)


def _cta_figure(src: str, mkt: str, label: str, df, months: int):
    import plotly.graph_objects as go
    import pandas as pd
    p = _CTA_P
    df = df[[d.weekday() < 5 for d in df["date"]]].reset_index(drop=True)
    close = pd.Series(_disp_closes(src, mkt, df).to_numpy(dtype=float))
    min_valid = max(p["tsmom"], p["ma_slow"] + 10, p["don"],
                    p["ewma"] + p["vol_days"] + 5)
    if len(close) <= min_valid + 5:
        return None
    tsmom_s = close.pct_change(p["tsmom"]).map(
        lambda x: (1 if x > 0 else -1 if x < 0 else 0) if pd.notna(x) else 0)
    ema_f = close.ewm(span=p["ma_fast"], adjust=False).mean()
    ema_s = close.ewm(span=p["ma_slow"], adjust=False).mean()
    ma_s = (ema_f - ema_s).map(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    don_hi = close.shift(1).rolling(p["don"]).max()
    don_lo = close.shift(1).rolling(p["don"]).min()
    don_s = pd.Series(np.where(close > don_hi, 1,
                               np.where(close < don_lo, -1, 0)), index=close.index)
    rets = close.pct_change()
    ewma_ret = rets.ewm(span=p["ewma"], adjust=False).mean()
    rvol = rets.rolling(p["vol_days"]).std()
    ewma_s = (ewma_ret / rvol.replace(0, np.nan)).map(
        lambda x: (1 if x > 0 else -1 if x < 0 else 0) if pd.notna(x) else 0)
    combined = ((tsmom_s + ma_s + don_s + ewma_s) / 4.0).to_numpy(dtype=float).copy()
    combined[:min_valid] = np.nan
    cutoff = date.today() - timedelta(days=months * 31)
    m = np.array([d >= cutoff for d in df["date"]]) & np.isfinite(combined)
    if m.sum() < 5:
        return None
    xs = [d for d, k in zip(df["date"], m) if k]
    ys = combined[m]
    fig = go.Figure()
    fig.add_hline(y=0, line_color="#94A3B8", line_width=1)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy",
                             line=dict(color="#B45309", width=1.8, shape="hv"),
                             fillcolor="rgba(180,83,9,0.12)", name="combined",
                             hovertemplate="%{x} · %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[xs[-1]], y=[float(ys[-1])], mode="markers+text",
                             marker=dict(size=6, color="#B45309"),
                             text=[f"{ys[-1]:+.2f}"], textposition="middle right",
                             textfont_size=11, showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        title=dict(text=f"{label} — CTA momentum (Combined: TSMOM{p['tsmom']} · "
                        f"MA{p['ma_fast']}/{p['ma_slow']} · Don{p['don']} · "
                        f"EWMA{p['ewma']})", font_size=13),
        height=210, margin=dict(l=30, r=70, t=40, b=24), showlegend=False,
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(range=[-1.08, 1.08], gridcolor="#F1F5F9", dtick=0.5),
        plot_bgcolor="#FFFFFF")
    return fig


def _seven_point_data(src: str, mkt: str, tenor_days: int):
    """(pts_m, pts_v, atm, cubic_coef) — the 7 tenor-interpolated smile points
    (ATM + ±1/2/3σ wings) and their cubic fit; None when σ2/σ3 wings unavailable.
    Shared by the 7-point density figure and its companion smile chart."""
    def _ik(curve, key):
        prs = [(rw.get("dte"), rw.get(key)) for rw in curve
               if rw.get("dte") is not None and rw.get(key) is not None
               and isinstance(rw.get(key), float) and math.isfinite(rw.get(key))]
        return _interp_pairs(prs, tenor_days) if len(prs) >= 2 else None

    curve1, _td, _err = _load_market(src, mkt, 1.0)
    if not curve1:
        return None
    atm = _ik(curve1, "atm_iv")
    if atm is None or atm <= 0:
        return None
    pts_m, pts_v = [0.0], [atm]
    for s_lvl, curve_s in ((1.0, curve1),
                           (2.0, _load_market(src, mkt, 2.0)[0]),
                           (3.0, _load_market(src, mkt, 3.0)[0])):
        if not curve_s:
            return None
        cw = _ik(curve_s, "call_wing_iv")
        pw = _ik(curve_s, "put_wing_iv")
        if cw is None or pw is None:
            return None
        pts_m += [s_lvl, -s_lvl]
        pts_v += [cw, pw]
    coef = np.polyfit(np.array(pts_m), np.array(pts_v), 3)
    return pts_m, pts_v, atm, coef


def _seven_point_smile_figure(src: str, mkt: str, label: str, tenor: str,
                              tenor_days: int, mkinfo: dict,
                              yield_axis: bool = False):
    """Companion smile for the 7-POINT view: the 7 tenor-interpolated vol points
    (markers) + cubic fit (line) vs STRIKE, F marked — needs no BL build, so it
    renders after any reload. Units per market; USDJPY inversion applied."""
    import plotly.graph_objects as go
    F = mkinfo.get("fut")
    if F is None or not math.isfinite(float(F)) or float(F) <= 0:
        return None
    F = float(F)
    sp = _seven_point_data(src, mkt, tenor_days)
    if sp is None:
        return None
    pts_m, pts_v, atm, coef = sp
    world = "Bachelier" if src == "rates" else "Black-76"
    T = tenor_days / 365.0
    sT = math.sqrt(T)
    ms = np.linspace(-3.2, 3.2, 200)
    vols = np.clip(np.polyval(coef, ms), 1e-4, None)
    if world == "Bachelier":
        Ks = F + ms * atm * sT
        Kp = F + np.array(pts_m) * atm * sT
        dv01, _ok = _market_dv01(mkt, _ro._MARKETS_RATES[mkt])
        scale, unit = ((1.0 / dv01, "yield vol · bp/yr") if dv01
                       else (1.0, "price vol · pts/yr"))
    else:
        Ks = F * np.exp(ms * atm * sT)
        Kp = F * np.exp(np.array(pts_m) * atm * sT)
        scale, unit = 100.0, "implied vol · %/yr"
    v_line = vols * scale
    v_pts = np.array(pts_v) * scale
    F_d = F
    _yfns = (_yield_axis_fns(src, mkt, float(Ks.min()), float(Ks.max()))
             if yield_axis else None)
    _xfmt = (lambda v: f"{v:.3f}%") if _yfns else _strike_fmt
    if _yfns:
        _p2y, _y2p = _yfns
        Ks = np.asarray(_p2y(Ks))[::-1]
        v_line = v_line[::-1]
        Kp = np.asarray(_p2y(Kp))
        F_d = float(_p2y(F))
    elif f"{src}:{mkt}" in _DISPLAY_INVERT:
        Ks = (1.0 / Ks)[::-1]
        v_line = v_line[::-1]
        Kp = 1.0 / Kp
        F_d = 1.0 / F
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ks, y=v_line, mode="lines",
                             line=dict(color="#2563EB", width=2),
                             name="cubic fit (7 points)",
                             hovertemplate="K %{x:,.6~f} · %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=Kp, y=v_pts, mode="markers",
                             marker=dict(size=9, color="#1D4ED8", symbol="diamond"),
                             name="ATM ±1/2/3σ points",
                             hovertemplate="K %{x:,.6~f} · %{y:.1f}<extra></extra>"))
    _und = mkinfo.get("fut_sym")
    fig.add_vline(x=F_d, line_color="#1E293B", line_width=1.5,
                  annotation_text=(f"F {_xfmt(F_d)}" + (f" · {_und}" if _und else "")
                                   + f" · ATM {atm * scale:.1f}"),
                  annotation_font_size=11)
    fig.update_layout(
        title=dict(text=f"{label} — 7-point tenor smile ({tenor}"
                        + (f" · {_und}" if _und else "") + ")", font_size=13),
        height=440, margin=dict(l=30, r=30, t=44, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right",
                    font_size=10),
        xaxis=dict(title="strike yield · %" if _yfns else "strike",
                   gridcolor="#F1F5F9"),
        yaxis=dict(title=unit, gridcolor="#F1F5F9"),
        plot_bgcolor="#FFFFFF",
    )
    return fig


def _seven_point_figure(src: str, mkt: str, meas: str, label: str,
                        tenor: str, tenor_days: int, mkinfo: dict,
                        yield_axis: bool = False):
    """7-point constant-tenor implied distribution: ATM + wings at ±1/2/3σ, each
    interpolated to the tenor across expiries (matching the grid's convention —
    unlike the BL chart, which is tied to one listed expiry). A cubic fit through
    the 7 (moneyness, vol) points captures smile CURVATURE the 3-point split-normal
    cannot (tail wings), then the same digital-call → survival → smooth-density
    pipeline renders it. Returns None when σ2/σ3 wings are unavailable (caller
    falls back to the split-normal)."""
    import plotly.graph_objects as go

    F = mkinfo.get("fut")
    if F is None or not math.isfinite(float(F)) or float(F) <= 0:
        return None
    F = float(F)
    world = "Bachelier" if src == "rates" else "Black-76"
    r_cfg = (_ro._MARKETS_RATES if src == "rates" else _ov2._MARKETS_V2)[mkt]
    r = float(r_cfg.get("r", 0.045))
    T = tenor_days / 365.0

    sp = _seven_point_data(src, mkt, tenor_days)
    if sp is None:
        return None
    pts_m, pts_v, atm, _coef = sp
    ms = np.linspace(-3.5, 3.5, 400)
    vols = np.clip(np.polyval(_coef, ms), 1e-4, None)
    sT = math.sqrt(T)
    if world == "Bachelier":
        Ks = F + ms * atm * sT
        pricer = _ro._bachelier
    else:
        Ks = F * np.exp(ms * atm * sT)
        pricer = _ov2._b76
    Cs = np.array([pricer(F, float(k), T, r, float(v), "C") for k, v in zip(Ks, vols)])
    good = np.isfinite(Cs)
    if good.sum() < 20:
        return None
    Cs = np.interp(Ks, Ks[good], Cs[good])
    dC = np.gradient(Cs, Ks)
    surv = np.minimum.accumulate(np.clip(-math.exp(r * T) * dC, 0.0, 1.0))

    _inv = f"{src}:{mkt}" in _DISPLAY_INVERT
    _yfns = (_yield_axis_fns(src, mkt, float(Ks.min()), float(Ks.max()))
             if yield_axis else None)
    if _yfns:
        _p2y, _y2p = _yfns
        # yield decreases in price → reverse, same shape as the 1/x inversion;
        # P(price > K) = P(yield < y(K)) = the display-space CDF directly.
        xs_n = np.asarray(_p2y(Ks))[::-1]
        cdf_n = surv[::-1]
        F_d = float(_p2y(F))
    elif _inv:
        xs_n = (1.0 / Ks)[::-1]
        cdf_n = surv[::-1]
        F_d = 1.0 / F
    else:
        xs_n = Ks
        cdf_n = 1.0 - surv
        F_d = F
    _xfmt = (lambda v: f"{v:.3f}%") if _yfns else _strike_fmt
    c0, c1 = float(cdf_n[0]), float(cdf_n[-1])
    if not (c1 > c0):
        return None
    cdf_n = (cdf_n - c0) / (c1 - c0)
    xs_u = np.linspace(float(xs_n[0]), float(xs_n[-1]), 400)
    cdf_u = np.interp(xs_u, xs_n, cdf_n)
    dens_u = np.clip(np.gradient(cdf_u, xs_u), 0.0, None)
    _g = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)
    _g /= _g.sum()
    dens_s = np.convolve(dens_u, _g, mode="same")
    pcts = {q: float(np.interp(q / 100.0, cdf_u, xs_u)) for q in (5, 25, 50, 75, 95)}

    # no-skew overlay (flat ATM) — computed at the NATIVE price for each display x,
    # then Jacobian-corrected into display space (1/x: ·K²; yield: ·|dp/dy|).
    if _yfns:
        k_nat = np.asarray(_y2p(xs_u))
    elif _inv:
        k_nat = 1.0 / xs_u
    else:
        k_nat = xs_u
    s_ = atm * sT
    if world == "Bachelier":
        bench = np.exp(-0.5 * ((k_nat - F) / s_) ** 2) / (s_ * math.sqrt(2 * math.pi))
    else:
        bench = (np.exp(-0.5 * ((np.log(k_nat / F) + 0.5 * s_ * s_) / s_) ** 2)
                 / (k_nat * s_ * math.sqrt(2 * math.pi)))
    if _inv:
        bench = bench * k_nat ** 2
    elif _yfns:
        bench = bench * np.abs(np.gradient(k_nat, xs_u))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_u, y=dens_s, mode="lines", fill="tozeroy",
                             line=dict(color="#2563EB", width=2),
                             fillcolor="rgba(37,99,235,0.15)",
                             name="7-point (tenor smile)",
                             hovertemplate="%{x:,.6~f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs_u, y=bench, mode="lines",
                             line=dict(color="#94A3B8", width=1.5, dash="dash"),
                             name="no-skew (flat ATM vol)",
                             hovertemplate="%{x:,.6~f}<extra></extra>"))
    # Symmetric normal at F — lognormal worlds only (rates overlay already is one).
    if world != "Bachelier" and atm and atm > 0:
        _sd = float(F_d) * atm * sT
        _symn = (np.exp(-0.5 * ((xs_u - float(F_d)) / _sd) ** 2)
                 / (_sd * math.sqrt(2 * math.pi)))
        fig.add_trace(go.Scatter(x=xs_u, y=_symn, mode="lines",
                                 line=dict(color="#0D9488", width=1.5, dash="dot"),
                                 name="symmetric normal @ F",
                                 hovertemplate="%{x:,.6~f}<extra></extra>"))
    _und = mkinfo.get("fut_sym")
    fig.add_vline(x=F_d, line_color="#1E293B", line_width=1.5,
                  annotation_text=f"F {_xfmt(F_d)}" + (f" · {_und}" if _und else ""),
                  annotation_font_size=11)
    for q in (25, 50, 75):
        fig.add_vline(x=pcts[q], line_color="#CBD5E1", line_width=1, line_dash="dot",
                      annotation_text=f"P{q} {_xfmt(pcts[q])}",
                      annotation_font_size=9, annotation_position="bottom")
    _cone = "  ·  ".join(f"P{q} {_xfmt(pcts[q])}" for q in (5, 25, 50, 75, 95))
    fig.add_annotation(text=f"cone: {_cone}", xref="paper", yref="paper",
                       x=0.5, y=1.13, showarrow=False, font_size=11,
                       font_color="#475569")
    fig.update_layout(
        title=dict(text=f"{label} — implied distribution at {tenor} "
                        f"(7-point: ATM ±1/2/3σ, constant tenor)", font_size=13),
        height=360, margin=dict(l=30, r=30, t=70, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right",
                    font_size=10),
        xaxis=dict(gridcolor="#F1F5F9",
                   title="implied yield · %" if _yfns else None),
        yaxis=dict(visible=False),
        plot_bgcolor="#FFFFFF",
    )
    return fig


def _bl_smile_figure(bl: dict, label: str, src: str, mkt: str,
                     yield_axis: bool = False):
    """Absolute-vol smile for the BL expiry: raw settlement IV points + the fitted
    (noise-rejecting) curve, F marked. Units follow the panel: % for lognormal
    markets; bp/yr yield-vol for rates (price-vol ÷ dv01); USDJPY x-inversion for
    display_invert markets."""
    import plotly.graph_objects as go
    ks = np.asarray(bl.get("smile_k") or [], dtype=float)
    ivs = np.asarray(bl.get("smile_iv") or [], dtype=float)
    fk = np.asarray(bl.get("smile_fit_k") or [], dtype=float)
    fiv = np.asarray(bl.get("smile_fit_iv") or [], dtype=float)
    if len(ks) < 3 or len(fk) < 3:
        return None
    F = float(bl["F"])
    if bl["world"] == "Bachelier":
        dv01 = bl.get("dv01")
        if dv01:
            scale, unit = 1.0 / dv01, "yield vol · bp/yr"
        else:
            scale, unit = 1.0, "price vol · pts/yr"
    else:
        scale, unit = 100.0, "implied vol · %/yr"
    ivs_d, fiv_d = ivs * scale, fiv * scale
    _inv = f"{src}:{mkt}" in _DISPLAY_INVERT
    _yfns = (_yield_axis_fns(src, mkt, float(fk.min()), float(fk.max()))
             if yield_axis else None)
    _xfmt = (lambda v: f"{v:.3f}%") if _yfns else _strike_fmt
    if _yfns:
        _p2y, _y2p = _yfns
        ks = np.asarray(_p2y(ks))[::-1]
        ivs_d = ivs_d[::-1]
        fk = np.asarray(_p2y(fk))[::-1]
        fiv_d = fiv_d[::-1]
        F = float(_p2y(F))
    elif _inv:
        ks = (1.0 / ks)[::-1]
        ivs_d = ivs_d[::-1]
        fk = (1.0 / fk)[::-1]
        fiv_d = fiv_d[::-1]
        F = 1.0 / F
    atm = bl.get("atm_iv")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fk, y=fiv_d, mode="lines",
                             line=dict(color="#2563EB", width=2),
                             name="fitted smile",
                             hovertemplate="K %{x:,.6~f} · %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ks, y=ivs_d, mode="markers",
                             marker=dict(size=6, color="#94A3B8", symbol="circle-open"),
                             name="settlement IVs",
                             hovertemplate="K %{x:,.6~f} · %{y:.1f}<extra></extra>"))
    _und = bl.get("und_sym")
    fig.add_vline(x=F, line_color="#1E293B", line_width=1.5,
                  annotation_text=(f"F {_xfmt(F)}" + (f" · {_und}" if _und else "")
                                   + (f" · ATM {atm * scale:.1f}" if atm else "")),
                  annotation_font_size=11)
    fig.update_layout(
        title=dict(text=f"{label} — smile at expiry {bl['expiry']} ({bl['dte']}d"
                        + (f", underlying {_und}" if _und else "") + ")",
                   font_size=13),
        height=440, margin=dict(l=30, r=30, t=44, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right",
                    font_size=10),
        xaxis=dict(title="strike yield · %" if _yfns else "strike",
                   gridcolor="#F1F5F9"),
        yaxis=dict(title=unit, gridcolor="#F1F5F9"),
        plot_bgcolor="#FFFFFF",
    )
    return fig


@st.fragment
def render_vol_dashboard():
    if not _api_key():
        st.error("Set DATABENTO_API_KEY in Streamlit secrets or environment variable.")
        return

    with st.expander("📊  Databento usage & costs", expanded=False):
        try:
            import data_costs
            data_costs.render_panel()
        except Exception:
            st.caption("cost ledger unavailable")

    st.markdown("#### 🌐 Vol Dashboard &nbsp;·&nbsp; cross-asset constant-tenor ATM vol")

    # Monitor's vol-indices table on top — free yfinance data (shared watchlist
    # cache), so it renders even before the paid-Databento load gate.
    try:
        from volatility import render_indices_table
        render_indices_table(header="#### Vol Indices")
    except Exception as _ex:
        st.caption(f"⚠ vol indices table unavailable: "
                   f"{type(_ex).__name__} {str(_ex)[:80]}")

    loads = _unique_loads()

    # ── Smart load gate (house _vm_load_gate pattern) — data is NEVER fetched by
    #    merely rendering this tab. Shows what's cached vs what costs money first. ──
    if not st.session_state.get("_vd_loaded"):
        cached, need = _cache_status(loads)
        lo, hi = _est_fetch_cost(need)
        if need:
            _brn_note = ""   # BRN's $6.2 ICE statistics landmine fixed at source 2026-07-29
            st.caption(
                f"**{len(cached)} of {len(loads)}** markets already cached today "
                f"(disk/session). Loading the remaining **{len(need)}** "
                f"({', '.join(k for _s, k in need)}) means paid Databento fetches — "
                f"roughly **${lo:.2f}–{hi:.2f}** (actuals land in the cost ledger above)."
                f"{_brn_note} First cold load can take a few minutes; instant once cached."
            )
        else:
            st.caption(f"All **{len(loads)}** markets already cached today — load is free "
                       f"and near-instant.")
        _sp1, col, _sp2 = st.columns([1, 2, 1])
        with col:
            if st.button(
                    f"Load dashboard — {len(cached)} of {len(loads)} markets cached today"
                    + (f"; ~${lo:.2f}–{hi:.2f} for the rest" if need else ""),
                    key="_vd_load_btn", use_container_width=True):
                st.session_state["_vd_loaded"] = True
                st.rerun(scope="fragment")
        return

    # ── Parallel prefetch: warm every market's disk cache concurrently (max 6 workers,
    #    shared-fut-root families kept in one sequential task) so the cached build below
    #    hits memory/disk instantly. Runs once per session-day; reruns (e.g. tenor change)
    #    skip straight to the cached build. ──
    _pf_day = date.today().isoformat()
    # Gate includes the load-set size so panel-config changes re-trigger the prefetch
    # for newly added markets within the same session.
    _pf_key = f"{_pf_day}|{len(loads)}"
    if st.session_state.get("_vd_prefetch_day") != _pf_key:
        _run_prefetch(loads)
        st.session_state["_vd_prefetch_day"] = _pf_key

    with st.spinner(f"Building dashboard across {len(loads)} markets — first load of "
                    f"the day can take a few minutes; instant once cached..."):
        build = _build_dashboard(str(date.today()), tuple(loads))

    # Session marker for the v2 (no-disk-cache) markets: after a successful build they
    # are held by st.cache_data, so the load gate can report them as cached.
    st.session_state["_vd_v2_session"] = (
        st.session_state.get("_vd_v2_session", set())
        | {mkt for src, mkt in loads if src == "v2"
           and not build["markets"].get(f"v2:{mkt}", {}).get("err")})

    # Persist today's values (all three tenors) once per session-day; idempotent upsert.
    today_iso = date.today().isoformat()
    if st.session_state.get("_vd_hist_day") != today_iso:
        _hist_upsert(build, today_iso)
        st.session_state["_vd_hist_day"] = today_iso

    tenor = st.selectbox("Tenor", [t for t, _d in _TENORS], index=0, key="_vd_tenor",
                         help="Constant calendar tenor for the interpolated ATM vol. "
                              "Changing tenor re-reads the already-loaded curves — no refetch.")

    changes = _hist_changes(today_iso, list(build["series"].keys()))

    cols = st.columns(2)
    for i, (title, unit, src, meas, mkts) in enumerate(_PANELS):
        with cols[i % 2]:
            st.markdown(_panel_html(title, unit, src, meas, mkts, build, changes, tenor),
                        unsafe_allow_html=True)

    # Footnotes: data dates, failures, blanks.
    notes = []
    tdates = sorted({m["tdate"] for m in build["markets"].values() if m.get("tdate")})
    if tdates:
        notes.append("settlement date(s): " + ", ".join(tdates))
    fails = [k.split(":", 1)[1] for k, m in build["markets"].items() if m.get("err")]
    if fails:
        notes.append("failed: " + ", ".join(fails))
    short6 = [k.split(":", 1)[1] for k, m in build["markets"].items()
              if not m.get("err") and m.get("max_dte") is not None
              and m["max_dte"] < _TENORS[-1][1] - _TENOR_TOL]
    if short6:
        notes.append("6m blank (listed expiries too short): " + ", ".join(short6))
    st.caption(
        "Interpolated ATM vol at constant tenor from settlement chains — our own "
        "measure, comparable to but not identical with CME CVOL (variance-style, "
        "full-curve). Vol Δ columns vs nearest stored day ≤ 1d/1w/1m back "
        "(vol up = red); FUT Δ1d from yesterday's stored settle (price up = green). "
        "+1σ / −1σ = absolute wing vols at strikes one ATM-σ above/below the forward, "
        "interpolated to the tenor (σ-strike convention, not 25-delta); wing > ATM on "
        "that side = skew toward it. "
        + (" · ".join(notes) if notes else "")
    )
    if fails:
        for k, m in build["markets"].items():
            if m.get("err"):
                st.caption(f"⚠ {k}: {m['err']}")

    # ── Implied distribution chart (skewed split-normal from ATM + wings) ─────
    st.markdown("---")
    _by_class, _class_order, _seen_d = {}, [], set()
    for _t2, _u2, _src2, _meas2, _mkts2 in _PANELS:
        if _t2 not in _by_class:
            _by_class[_t2] = []
            _class_order.append(_t2)
        for _k2, _lbl2 in _mkts2:
            if (_src2, _k2) not in _seen_d:
                _seen_d.add((_src2, _k2))
                _by_class[_t2].append((_src2, _k2, _meas2, _lbl2))
    # Params + RUN gate (Rajat 2026-08-02: "it should not recalc on every select —
    # only once I've picked all params"): widgets are staged; charts render from the
    # SNAPSHOT stored when Run is pressed, so changing a dropdown recomputes nothing.
    _RVOL_N = {"2w": 10, "1m": 21, "2m": 42, "3m": 63}
    _dc1, _dc2, _dc3, _dc4, _dc5 = st.columns([1.0, 1.25, 0.6, 0.75, 0.5])
    _ac = _dc1.selectbox("📈 Implied distribution — asset class", _class_order,
                         key="_vd_dist_ac")
    _opts_ac = _by_class[_ac]
    _dsel = _dc2.selectbox(
        "Instrument",
        range(len(_opts_ac)), format_func=lambda i: _opts_ac[i][3],
        key=f"_vd_dist_mkt_{_ac}",
        help="Skewed price distribution at the selected tenor: centre = current future, "
             "left/right widths from the −1σ / +1σ wing vols. Dashed = symmetric "
             "ATM-only curve, so the skew deformation is visible.")
    _sel_src, _sel_mkt, _sel_meas, _sel_lbl = _opts_ac[_dsel]
    _rv_lbl = _dc3.selectbox("Realized vol window", list(_RVOL_N), key="_vd_rvol_win",
                             help="Trailing window for estimating realized vol in the "
                                  "history band. The band width is ALWAYS the 2-week "
                                  "move at that vol (comparable with the implied side).")
    _sel_yld = False
    if _sel_src == "rates":
        _dc4.write("")   # spacer — aligns the checkbox with the dropdowns
        _sel_yld = _dc4.checkbox(
            "x-axis in yield (%)", key="_vd_dist_yld",
            help="Flip the distribution and smile x-axes from futures price to implied "
                 "rate: STIRs 100−price; bond futures the implied CTD yield (same "
                 "machinery as the grid's RATE column). Densities are transformed with "
                 "the proper Jacobian, so probabilities are preserved.")
    _dc5.write("")
    if _dc5.button("▶ Run", key="_vd_dist_go", type="primary",
                   use_container_width=True):
        st.session_state["_vd_dist_snap"] = dict(
            src=_sel_src, mkt=_sel_mkt, meas=_sel_meas, lbl=_sel_lbl,
            yld=_sel_yld, rvl=_rv_lbl)

    # Background prefetch of the WHOLE selected asset class's underlying bars runs
    # on LIVE class selection (before the Run gate) so the data is usually on disk
    # by the time Run is pressed — daemon threads, results land as disk pickles;
    # otherwise every first click on a market pays a 12-30s Databento queue wait
    # that reads as a hang (Rajat hit it switching to Rates).
    _und_pf_key = f"_vd_und_pf_{_ac}_{date.today().isoformat()}"
    if not st.session_state.get(_und_pf_key):
        st.session_state[_und_pf_key] = True
        _pf_targets, _seen_roots = [], set()
        for _s3, _k3, _m3, _l3 in _opts_ac:
            try:
                _ds3, _root3 = _underlying_root(_s3, _k3)
            except Exception:
                continue
            if _root3 in _seen_roots or (_s3, _k3) == (_sel_src, _sel_mkt):
                continue
            _seen_roots.add(_root3)
            _ctr3 = (build["markets"].get(f"{_s3}:{_k3}", {}) or {}).get("fut_sym") \
                if _k3 in _MIDCURVE_UND else None
            if not _und_is_fresh(_s3, _k3, _ctr3):
                _pf_targets.append((_s3, _k3, _ctr3))
        if _pf_targets:
            import threading
            from concurrent.futures import ThreadPoolExecutor

            def _und_prefetch(targets):
                with ThreadPoolExecutor(max_workers=5) as _ex:
                    list(_ex.map(lambda t: _underlying_bars(*t), targets))

            threading.Thread(target=_und_prefetch, args=(_pf_targets,),
                             daemon=True).start()

    # everything below renders from the RUN snapshot only
    _snap = st.session_state.get("_vd_dist_snap")
    if not _snap:
        st.info("Pick asset class, instrument and params above, then press ▶ Run.")
        return
    _dsrc, _dmkt = _snap["src"], _snap["mkt"]
    _dmeas, _dlbl = _snap["meas"], _snap["lbl"]
    _yld_axis = bool(_snap.get("yld"))
    _rvl = _snap.get("rvl", "2w")
    _rvn = _RVOL_N.get(_rvl, 10)
    _tdays = dict(_TENORS)[tenor]
    _mkinfo = build["markets"].get(f"{_dsrc}:{_dmkt}", {})

    # ── Full option-implied density (Breeden–Litzenberger) ────────────────────
    # Cached per (market, tenor, day) in session_state so tenor-flips / reruns don't
    # recompute; the button forces a recompute for the current selection and its result
    # REPLACES the simple split-normal chart. Insufficient data → clean fallback below.
    _bl_cache = st.session_state.setdefault("_vd_bl_cache", {})
    _bl_key = f"{_dsrc}:{_dmkt}|{tenor}|{today_iso}"
    if st.button("Build full implied density (Breeden–Litzenberger)",
                 key="_vd_bl_btn", use_container_width=True,
                 help="Model-free risk-neutral density f(K)=e^(rT)·∂²C/∂K² by finite "
                      "differences over the LISTED-strike smile of the expiry nearest the "
                      "tenor. Uses only the already-cached surface — no new data fetch."):
        with st.spinner("Computing Breeden–Litzenberger density from the cached smile…"):
            _bl_cache[_bl_key] = _compute_bl_density(_dsrc, _dmkt, tenor, _tdays)

    _bl = _bl_cache.get(_bl_key)
    _view = "full"
    if _bl and _bl.get("ok"):
        _view = st.radio("Distribution view", ["Full smile (expiry)", "7-point (tenor)"],
                         horizontal=True, key=f"_vd_bl_view_{_bl_key}",
                         label_visibility="collapsed")
        _view = "full" if _view.startswith("Full") else "simple"

    # ── Underlying futures history FIRST (Rajat: "move it to the top") ──
    _und_rng = st.radio("Underlying range", ["3m", "6m"], horizontal=True, index=1,
                        key="_vd_und_rng", label_visibility="collapsed")
    _und_ctr = _mkinfo.get("fut_sym") if _dmkt in _MIDCURVE_UND else None
    _und_cached = _und_is_fresh(_dsrc, _dmkt, _und_ctr)
    with st.spinner(f"Loading {_dlbl} underlying history…"
                    + ("" if _und_cached else
                       " (first fetch today — Databento queue can take 15-30s; "
                       "the rest of this asset class is pre-loading in background)")):
        _und_df, _und_err = _underlying_bars(_dsrc, _dmkt, _und_ctr)
    if _und_df is None:
        st.caption(f"⚠ Underlying history unavailable: {_und_err}")
    else:
        _und_fig = _underlying_history_figure(
            _dsrc, _dmkt, _dlbl, _und_df, 3 if _und_rng == "3m" else 6,
            yield_axis=_yld_axis, rvol_n=_rvn, rvol_lbl=_rvl,
            fixed_contract=bool(_und_ctr))
        if _und_fig is None:
            st.caption("⚠ Not enough underlying bars in the selected window.")
        else:
            st.plotly_chart(_und_fig, use_container_width=True)
            st.caption(
                "Daily closes of the volume-rolled continuous front month from Databento "
                "(unadjusted at rolls; fetched once per market per day, ~$0.001–0.04). "
                "**One methodology throughout**: the band is always the 2-week "
                "expected move (±1σ/±2σ, darker = further ring). **Blue (history)**: "
                "at trailing REALIZED vol (symmetric). **Teal (next 3m)**: at the "
                "FORWARD 2w implied vol from the option term structure, SKEWED — "
                "upper edges use call-wing vols, lower use put-wing (asymmetry = "
                "implied skew; width bumps ahead = priced-in event vol). The teal "
                "side is pure implied from day one; the final week of the blue band "
                "eases into it, so compare widths just either side of the junction "
                "for the vol risk premium. "
                + ("Yield view applies TODAY's CTD assumptions across the whole history — "
                   "indicative, not a true historical yield series. " if _yld_axis else "")
                + ("Displayed in inverted (USD-first) convention. "
                   if f"{_dsrc}:{_dmkt}" in _DISPLAY_INVERT else ""))

    # ── Optional technicals: RSI / CTA momentum (Rajat 2026-08-04) — charts 2 & 3,
    # live toggles (no Run needed: computed off the already-cached bars) ──
    _oc1, _oc2, _oc3, _sp = st.columns([0.7, 0.7, 1.2, 3.4])
    _show_rsi = _oc1.checkbox("RSI", key="_vd_show_rsi")
    _rsi_n = _oc2.selectbox("RSI window", [14, 30], key="_vd_rsi_n",
                            format_func=lambda n: f"{n}d",
                            label_visibility="collapsed")
    _show_cta = _oc3.checkbox("CTA momentum", key="_vd_show_cta")
    if _und_df is not None and (_show_rsi or _show_cta):
        _mn = 3 if _und_rng == "3m" else 6
        if _show_rsi:
            _rf = _rsi_figure(_dsrc, _dmkt, _dlbl, _und_df, _mn, _rsi_n)
            if _rf is None:
                st.caption(f"⚠ RSI({_rsi_n}d): not enough history for this market.")
            else:
                st.plotly_chart(_rf, use_container_width=True)
        if _show_cta:
            _cf = _cta_figure(_dsrc, _dmkt, _dlbl, _und_df, _mn)
            if _cf is None:
                st.caption("⚠ CTA momentum: needs ~210 bars of history "
                           "(thin/new contracts may not have it).")
            else:
                st.plotly_chart(_cf, use_container_width=True)
                st.caption("Same Combined signal as the CTA tab (average of the 4 "
                           "sign signals at the tab's default parameters), computed "
                           "on this continuous-front/underlying series.")

    # Smile chart next (Rajat 2026-07-31: "strike vs vol on top, distribution below").
    # Always shown; the view decides the source: full-smile view → raw settlement IVs
    # + fit at the BL expiry; otherwise → the 7 tenor-interpolated points + cubic fit
    # (no build needed, renders straight after a reload).
    _shown_smile = False
    if _bl and _bl.get("ok") and _view == "full" and _bl.get("smile_k"):
        _sm_fig = _bl_smile_figure(_bl, _dlbl, _dsrc, _dmkt, yield_axis=_yld_axis)
        if _sm_fig is not None:
            st.plotly_chart(_sm_fig, use_container_width=True)
            st.caption("Smile at the BL expiry — grey circles are raw settlement IVs per "
                       "strike; blue line is the noise-rejecting fit. The slope/curvature "
                       "IS the skew shaping the distribution below.")
            _shown_smile = True
    if not _shown_smile:
        _sm7 = _seven_point_smile_figure(_dsrc, _dmkt, _dlbl, tenor, _tdays, _mkinfo,
                                         yield_axis=_yld_axis)
        if _sm7 is not None:
            st.plotly_chart(_sm7, use_container_width=True)
            st.caption("7-point tenor smile — the ATM and ±1/2/3σ wing vols (diamonds) "
                       "with their cubic fit; this is exactly what shapes the 7-point "
                       "distribution below. Build the full density for the every-strike "
                       "settlement smile at a listed expiry.")

    if _bl and _bl.get("ok") and _view == "full":
        st.plotly_chart(
            _bl_density_figure(_bl, _dlbl, tenor, _dsrc, _dmkt, _tdays, _mkinfo,
                               yield_axis=_yld_axis),
            use_container_width=True)
        st.caption(
            f"Breeden–Litzenberger risk-neutral density from the {_bl['world']} smile of "
            f"the listed expiry **{_bl['expiry']}** ({_bl['dte']} dte) — the real expiry "
            f"nearest the {tenor} target (no cross-expiry interpolation). "
            f"f(K)=e^(rT)·∂²C/∂K² by central finite differences over ~400 strikes spanning "
            f"only the quoted range [{_strike_fmt(_bl['kmin'])}, {_strike_fmt(_bl['kmax'])}] "
            f"(no extrapolation). Mass shown: **{_bl['total_mass'] * 100:.0f}%** — tails "
            f"beyond listed strikes are truncated (NOT renormalised); clipped negative "
            f"mass: {_bl['clipped_mass'] * 100:.2f}%. Mode {_strike_fmt(_bl['mode'])} vs "
            f"F {_strike_fmt(_bl['F'])}. Dashed grey = split-normal wing fit for comparison.")
    else:
        if _bl and not _bl.get("ok"):
            st.caption(f"⚠ Breeden–Litzenberger unavailable ({_bl.get('error')}) — showing "
                       f"the split-normal fit instead.")
        _fig = _seven_point_figure(_dsrc, _dmkt, _dmeas, _dlbl, tenor, _tdays, _mkinfo,
                                   yield_axis=_yld_axis)
        if _fig is None:      # σ2/σ3 wings unavailable → old 3-point split-normal
            _fig = _implied_dist_figure(_dsrc, _dmkt, _dmeas, _dlbl, tenor, _tdays,
                                        _mkinfo, yield_axis=_yld_axis)
        if _fig is None:
            st.info("Not enough data for this market/tenor (needs ATM + both wing vols "
                    "and a future price).")
        else:
            st.plotly_chart(_fig, use_container_width=True)
            st.caption(
                "7-point implied distribution at constant tenor (ATM + ±1/2/3σ wing vols, "
                "each tenor-interpolated; falls back to the 3-point split-normal when σ2/σ3 "
                "wings are unavailable). Dashed grey = symmetric flat-ATM. Click the button "
                "above for the full-chain density at the nearest listed expiry.")

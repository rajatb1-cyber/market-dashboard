"""
Options Monitor V2 — Databento settlement data.
No IBKR connection. Data = previous trading day settlement.
IVs computed via Black-76.
"""

import streamlit as st
import pandas as pd
import numpy as np
import databento as db
import math, os, re, pickle
from datetime import date, timedelta, datetime
from typing import Optional
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go

# ── Markets ───────────────────────────────────────────────────────────────────
_CME = "GLBX.MDP3"
_ICE = "IFEU.IMPACT"
_XEUR = "XEUR.EOBI"

_MARKETS_V2 = {
    "ES":  {"label": "S&P 500 (ES)",       "opt_sym": "ES.OPT",  "monthly_opt_sym": "EW3.OPT", "fut_sym": "ES.FUT",  "ds": _CME, "r": 0.045},
    "NQ":  {"label": "Nasdaq 100 (NQ)",    "opt_sym": "NQ.OPT",  "monthly_opt_sym": "QN3.OPT", "fut_sym": "NQ.FUT",  "ds": _CME, "r": 0.045},
    # monthly root fixed 2026-07-29: was "SR3.OPT" (SOFR! — its 100x-scaled strikes
    # polluted RTY chains with garbage IVs). RTM = Russell 2000 end-of-month options.
    "RTY": {"label": "Russell 2000 (RTY)", "opt_sym": "RTO.OPT", "monthly_opt_sym": "RTM.OPT", "fut_sym": "RTY.FUT", "ds": _CME, "r": 0.045},
    "GC":  {"label": "Gold (GC)",          "opt_sym": "OG.OPT",  "fut_sym": "GC.FUT",  "ds": _CME, "r": 0.045},
    "BTC": {"label": "Bitcoin (MBT)",      "opt_sym": "WM.OPT",  "fut_sym": "MBT.FUT", "ds": _CME, "r": 0.045},
    "BRN": {"label": "Brent Crude (BRN)",  "opt_sym": "BRN.OPT", "fut_sym": "BRN.FUT", "ds": _ICE, "r": 0.045},
    # FX options on CME FX futures (probed 2026-07-29: monthly first-Friday expiries,
    # underlying = the quarterly future via real underlying_id, native decimal strikes,
    # no scale quirks). Same Black-76 engine — FX vol quotes are lognormal like equities.
    "EUR": {"label": "EUR/USD (6E)", "opt_sym": "EUU.OPT", "fut_sym": "6E.FUT", "ds": _CME, "r": 0.045},
    # display_invert: 6J settles JPY/USD (~0.006314) but market convention is USDJPY
    # (~158.38). Strikes/F are SHOWN as 1/x (USDJPY) in Chain/Smile/Expiry Curve only;
    # all computation (IV, ATM, PCP, surface) stays native. Premiums & Pricer stay native.
    "JPY": {"label": "JPY/USD (6J) — shown as USDJPY", "opt_sym": "JPU.OPT", "fut_sym": "6J.FUT", "ds": _CME, "r": 0.045, "display_invert": True},
    "GBP": {"label": "GBP/USD (6B)", "opt_sym": "GBU.OPT", "fut_sym": "6B.FUT", "ds": _CME, "r": 0.045},
    "AUD": {"label": "AUD/USD (6A)", "opt_sym": "ADU.OPT", "fut_sym": "6A.FUT", "ds": _CME, "r": 0.045},
    # More CME FX (probed 2026-07-31). CAD/CHF/MXN settle foreign/USD but market
    # convention is USD-per-foreign — shown as 1/x (USDCAD/USDCHF/USDMXN) in
    # Chain/Smile/Expiry Curve only; all computation stays native (see JPY note).
    "CAD": {"label": "CAD/USD (6C) — shown as USDCAD", "opt_sym": "CAU.OPT", "fut_sym": "6C.FUT", "ds": _CME, "r": 0.045, "display_invert": True},
    "CHF": {"label": "CHF/USD (6S) — shown as USDCHF", "opt_sym": "CHU.OPT", "fut_sym": "6S.FUT", "ds": _CME, "r": 0.045, "display_invert": True},
    "MXN": {"label": "MXN/USD (6M) — shown as USDMXN", "opt_sym": "6M.OPT", "fut_sym": "6M.FUT", "ds": _CME, "r": 0.045, "display_invert": True},
    "NZD": {"label": "NZD/USD (6N)", "opt_sym": "6N.OPT", "fut_sym": "6N.FUT", "ds": _CME, "r": 0.045},
    # Eurex index options (XEUR.EOBI). Real underlying_id links resolve OESX/ODAX to
    # FESX/FDAX (unlike ICE); cash-settled index options priced against the future as
    # forward proxy. r=0.02 (EUR). XEUR historical license ends at 22:00 UTC — the
    # fetch end is clamped in _load_data_fresh to avoid a doomed 403.
    "ESTX": {"label": "EuroStoxx 50 (OESX)", "opt_sym": "OESX.OPT", "fut_sym": "FESX.FUT", "ds": _XEUR, "r": 0.02},
    "DAX":  {"label": "DAX (ODAX)",          "opt_sym": "ODAX.OPT", "fut_sym": "FDAX.FUT", "ds": _XEUR, "r": 0.02},
    # CME energy/metals (probed 2026-07-31).
    "CL": {"label": "WTI Crude (CL)", "opt_sym": "LO.OPT",  "fut_sym": "CL.FUT", "ds": _CME, "r": 0.045},
    "SI": {"label": "Silver (SI)",    "opt_sym": "SO.OPT",  "fut_sym": "SI.FUT", "ds": _CME, "r": 0.045},
    "HG": {"label": "Copper (HG)",    "opt_sym": "HXE.OPT", "fut_sym": "HG.FUT", "ds": _CME, "r": 0.045},
}

_MONTH_CODES = {"F":1,"G":2,"H":3,"J":4,"K":5,"M":6,"N":7,"Q":8,"U":9,"V":10,"X":11,"Z":12}
_YF_FALLBACK = {"BRN": "BZ=F", "GC": "GC=F", "BTC": "BTC-USD",
                "EUR": "6E=F", "JPY": "6J=F", "GBP": "6B=F", "AUD": "6A=F",
                "CAD": "6C=F", "CHF": "6S=F", "MXN": "6M=F", "NZD": "6N=F",
                "CL": "CL=F", "SI": "SI=F", "HG": "HG=F",
                "ESTX": "^STOXX50E", "DAX": "^GDAXI"}
_STD_SIGMA_OPTS = {"ATM only": 0.0, "0.5σ": 0.5, "1σ": 1.0, "2σ": 2.0}

# ── Auth ──────────────────────────────────────────────────────────────────────
def _api_key() -> str:
    try:
        return st.secrets["DATABENTO_API_KEY"]
    except Exception:
        return os.environ.get("DATABENTO_API_KEY", "")

# ── Date helpers ──────────────────────────────────────────────────────────────
def _trade_date(ds: str) -> date:
    """Last date for which data is available. ICE has ~18hr delay."""
    n = 2 if ds == _ICE else 1
    d = date.today() - timedelta(days=n)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d

def _prev_bday(d: date, n: int) -> date:
    """Go back n business days from d."""
    result = d
    while n > 0:
        result -= timedelta(days=1)
        if result.weekday() < 5:
            n -= 1
    return result

def _date_window(d: date) -> tuple[str, str]:
    return f"{d}T00:00:00", f"{d}T23:59:59"

# ── Black-76 ──────────────────────────────────────────────────────────────────
def _b76(F: float, K: float, T: float, r: float, sigma: float, right: str) -> float:
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    df = math.exp(-r * T)
    if right == "C":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def _calc_iv(price: Optional[float], F: float, K: float, T: float, r: float, right: str) -> Optional[float]:
    if not (price and F and K and T and T > 0 and F > 0 and K > 0):
        return None
    try:
        intrinsic = math.exp(-r * T) * max(F - K, 0) if right == "C" else math.exp(-r * T) * max(K - F, 0)
        if price <= intrinsic * 1.001:
            return None
        return brentq(lambda s: _b76(F, K, T, r, s, right) - price, 1e-5, 20.0, xtol=1e-6, maxiter=100)
    except Exception:
        return None

def _sf(v) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) and f > 0 else None
    except Exception:
        return None

# ── Data loading (all cached) ─────────────────────────────────────────────────
_LOAD_DATA_VERSION = 10  # bump to bust cache when loading logic changes

def _clamp_end(ex: Exception) -> Optional[str]:
    """Parse the available-end timestamp from Databento availability errors."""
    msg = str(ex)
    # Pattern 1: data_end_after_available_end — "available up to 'TIMESTAMP'"
    if "data_end_after_available_end" in msg:
        m = re.search(r"available up to '([0-9 :T.+-]+)'", msg)
        if m:
            return m.group(1).split("+")[0].strip().replace(" ", "T")
    # Pattern 2: dataset_unavailable_range — "end time before TIMESTAMP"
    if "dataset_unavailable_range" in msg:
        m = re.search(r"end time before ([0-9T:.\-Z]+)", msg)
        if m:
            ts = m.group(1).replace("Z", "")
            return ts.split(".")[0]  # strip sub-second precision
    # Pattern 3: license_not_found_unauthorized — "a live data license is required to
    # access DATASET data after TIMESTAMP" (Eurex/ICE venues: historical license ends at
    # the session close; a window past it 403s). Clamp to the licensed boundary.
    if "license_not_found_unauthorized" in msg:
        m = re.search(r"data after ([0-9T:.\-Z]+)", msg)
        if m:
            ts = m.group(1).replace("Z", "")
            return ts.split(".")[0]
    # Pattern 4: data_schema_not_fully_available — "Schema X for dataset Y is only
    # available between START and END" (hit on IFLL ohlcv-1d just after the UTC day
    # rolls, 2026-08-02). Clamp to the SECOND timestamp.
    if "data_schema_not_fully_available" in msg:
        m = re.search(r"between [0-9T:.\-Z]+ and ([0-9T:.\-Z]+)", msg)
        if m:
            ts = m.group(1).replace("Z", "")
            return ts.split(".")[0]
    return None


def _is_transient_server_error(ex: Exception) -> bool:
    msg = str(ex)
    return any(k in msg for k in ("504", "502", "503", "gateway", "Gateway"))


def _record_fetch_cost(client, dataset, symbols, schema, start, end, stype_in):
    """Best-effort: ask Databento what the fetch just cost (free metadata call) and
    append it to the spend ledger. Runs once per REAL fetch — the disk cache means
    ~once per market per day, so no meaningful latency or call-count is added."""
    try:
        from data_costs import record_cost
        usd = client.metadata.get_cost(dataset=dataset, symbols=symbols, schema=schema,
                                       start=start, end=end, stype_in=stype_in)
        record_cost(dataset, schema, symbols, float(usd))
    except Exception:
        pass


def _get_range(client, dataset: str, symbols: list, schema: str,
               start: str, end: str, stype_in: str = "parent"):
    """timeseries.get_range with automatic end-clamp on availability errors, and a
    short retry-with-backoff on transient Databento server errors (502/503/504) —
    large definition queries (e.g. OZN's 12k+ option instruments) occasionally hit a
    gateway timeout on Databento's side; a quick retry usually succeeds."""
    import time
    _attempts = 3
    for attempt in range(_attempts):
        try:
            raw = client.timeseries.get_range(dataset=dataset, symbols=symbols,
                                              schema=schema, start=start, end=end,
                                              stype_in=stype_in)
            _record_fetch_cost(client, dataset, symbols, schema, start, end, stype_in)
            return raw
        except Exception as ex:
            clamped = _clamp_end(ex)
            if clamped:
                raw = client.timeseries.get_range(dataset=dataset, symbols=symbols,
                                                  schema=schema, start=start, end=clamped,
                                                  stype_in=stype_in)
                _record_fetch_cost(client, dataset, symbols, schema, start, clamped, stype_in)
                return raw
            if _is_transient_server_error(ex) and attempt < _attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise


def _fetch_opt_defs(client, ds: str, sym: str, s: str, e: str) -> pd.DataFrame:
    raw = _get_range(client, ds, [sym], "definition", s, e)
    df = raw.to_df()
    if "instrument_class" in df.columns:
        df = df[df["instrument_class"].isin(["C", "P"])].copy()
    keep = [c for c in ["instrument_id", "strike_price", "instrument_class",
                         "expiration", "underlying_id"] if c in df.columns]
    return df[keep].drop_duplicates(subset=["instrument_id"]).reset_index(drop=True)


def _fetch_opt_ohlcv(client, ds: str, sym: str, s: str, e: str) -> pd.DataFrame:
    """Fetch option settlement prices.
    Primary: statistics schema (SETTLEMENT_PRICE) — wider strike coverage than ohlcv-1d.
    Fallback: ohlcv-1d.
    Returns a tuple (DataFrame, source_str) for debugging.
    """
    _SETTLE  = 3   # StatType.SETTLEMENT_PRICE
    _CLEARED = 6   # StatType.CLEARED_VOLUME
    _stats_err = None
    try:
        raw  = _get_range(client, ds, [sym], "statistics", s, e)
        df   = raw.to_df().reset_index()
        # stat_type may be a Databento enum — normalise to plain int for comparison
        df["stat_type"] = df["stat_type"].apply(int)
        settle = (df[df["stat_type"] == _SETTLE]
                  .sort_values("ts_event")
                  .groupby("instrument_id")["price"]
                  .last().reset_index()
                  .rename(columns={"price": "close"}))
        if not settle.empty:
            vol = (df[df["stat_type"] == _CLEARED][["instrument_id", "quantity"]]
                   .rename(columns={"quantity": "volume"})
                   .groupby("instrument_id")["volume"].sum().reset_index())
            result = settle.merge(vol, on="instrument_id", how="left")
            result["volume"] = result["volume"].fillna(0).astype(int)
            return result
        _stats_err = f"statistics schema returned 0 SETTLEMENT_PRICE rows for {sym}"
    except Exception as ex:
        _stats_err = f"statistics fetch failed for {sym}: {ex}"
    # Fallback: ohlcv-1d (raise _stats_err so caller can surface it)
    raw = _get_range(client, ds, [sym], "ohlcv-1d", s, e)
    df  = raw.to_df().reset_index()
    result = df[["instrument_id", "close", "volume"]].copy()
    result.attrs["stats_err"] = _stats_err   # stash for debugging
    return result


# ── Disk cache (shared with the Vol Dashboard's vol_dash_cache) ────────────────
# _load_data is @st.cache_data (memory, 1h ttl) — every fresh server process/hour
# otherwise re-pays the Databento fetch (ES was refetched 6× across restarts in one
# evening). Add a pickle-per-(market, day) disk layer, REUSING the same dir + filename
# scheme the Vol Dashboard already writes (vol_dash_cache/{mkt}_{date}_v{version}.pkl),
# so existing pickles keep working. Keyed on _LOAD_DATA_VERSION so cache busts bust this
# too. Stores the RAW loader output; never persists an empty/error result (a transient
# failure / availability embargo must not poison the date — same rule rates_options uses).
_V2_DISK_DIR = os.path.join(os.path.dirname(__file__), "vol_dash_cache")


def _load_data_disk_path(mkt_key: str, date_str: str,
                         version: int = _LOAD_DATA_VERSION) -> str:
    os.makedirs(_V2_DISK_DIR, exist_ok=True)
    return os.path.join(_V2_DISK_DIR, f"{mkt_key}_{date_str}_v{version}.pkl")


def _load_data_fresh(mkt_key: str, date_str: str) -> dict:
    """Load option defs + option OHLCV + futures OHLCV for one trading day."""
    m = _MARKETS_V2[mkt_key]
    key = _api_key()
    if not key:
        return {"error": "No DATABENTO_API_KEY set"}
    client = db.Historical(key=key)
    s, e = _date_window(date_str)
    # Eurex (XEUR.EOBI): Databento's historical license ends at each session's 22:00 UTC
    # close — the default 23:59:59 window end crosses 2h into the live-license zone and
    # 403s ("license_not_found_unauthorized") whenever the fetch races the daily
    # availability roll. Session data all lies before 22:00, so clamping loses nothing.
    # (_clamp_end already recovers from the 403, but the explicit clamp avoids paying the
    # failed attempt.) Mirrors rates_options._load_data_fresh.
    if m["ds"].startswith("XEUR"):
        e = f"{date_str}T21:59:59"
    out = {}
    monthly_sym = m.get("monthly_opt_sym")

    # Option definitions (primary quarterly symbol)
    try:
        defs = _fetch_opt_defs(client, m["ds"], m["opt_sym"], s, e)
        out["quarterly_expiries"] = (
            {r.date() for r in defs["expiration"].dropna()} if "expiration" in defs.columns else set()
        )
    except Exception as ex:
        defs = pd.DataFrame()
        out["quarterly_expiries"] = set()
        out["defs_err"] = str(ex)

    # Monthly supplement (e.g. EW3.OPT for ES — adds non-quarterly 3rd-Friday expiries)
    if monthly_sym:
        try:
            monthly_defs = _fetch_opt_defs(client, m["ds"], monthly_sym, s, e)
            if not monthly_defs.empty:
                defs = (pd.concat([defs, monthly_defs], ignore_index=True)
                          .drop_duplicates(subset=["instrument_id"])
                          .reset_index(drop=True))
        except Exception:
            pass

    out["opt_defs"] = defs

    # Option OHLCV (primary)
    try:
        ohlcv = _fetch_opt_ohlcv(client, m["ds"], m["opt_sym"], s, e)
        if ohlcv.attrs.get("stats_err"):
            out["stats_err"] = ohlcv.attrs["stats_err"]
    except Exception as ex:
        ohlcv = pd.DataFrame()
        out["ohlcv_err"] = str(ex)

    # Monthly OHLCV supplement
    if monthly_sym:
        try:
            monthly_ohlcv = _fetch_opt_ohlcv(client, m["ds"], monthly_sym, s, e)
            if not monthly_ohlcv.empty:
                ohlcv = (pd.concat([ohlcv, monthly_ohlcv], ignore_index=True)
                           .drop_duplicates(subset=["instrument_id"])
                           .reset_index(drop=True))
        except Exception:
            pass

    out["opt_ohlcv"] = ohlcv

    # Futures settlements. Schema order is COST-CRITICAL on ICE: the BRN.FUT parent
    # `statistics` query bills ~$6.2/day (drags the entire Brent complex incl. spreads)
    # vs ~$0.03 for ohlcv-1d, and the liquid Brent strip trades every listed month so
    # ohlcv coverage is fine there. CME statistics is near-free, and its full per-contract
    # settle coverage matters for far-dated underlyings — so: ICE → ohlcv-1d first with
    # statistics fallback; CME → statistics first (original behaviour, unchanged values).
    _SETTLE = 3  # StatType.SETTLEMENT_PRICE

    def _fut_via_statistics():
        raw = _get_range(client, m["ds"], [m["fut_sym"]], "statistics", s, e)
        df  = raw.to_df().reset_index()
        df["stat_type"] = df["stat_type"].apply(int)
        settle = (df[df["stat_type"] == _SETTLE]
                  .sort_values("ts_event")
                  .groupby("instrument_id")["price"]
                  .last().reset_index()
                  .rename(columns={"price": "close"}))
        if settle.empty:
            raise ValueError("no futures settlements in statistics response")
        if "symbol" in df.columns:
            sym_map = df[["instrument_id", "symbol"]].drop_duplicates("instrument_id")
            settle  = settle.merge(sym_map, on="instrument_id", how="left")
        return settle

    def _fut_via_ohlcv():
        raw = _get_range(client, m["ds"], [m["fut_sym"]], "ohlcv-1d", s, e)
        df  = raw.to_df().reset_index()
        df = df[["instrument_id", "close", "symbol"]].dropna(subset=["close"]).copy()
        if m["ds"] == _ICE:
            # ohlcv (unlike settlement statistics) carries bars for calendar spreads
            # (close ~0.4) and TAS '_Z' instruments (close ~0.03) — poison for F
            # selection/sanitising (BRN showed 1043% IV, 2026-07-29). ICE OUTRIGHT
            # symbols uniquely end with '!' — keep only those.
            df = df[df["symbol"].astype(str).str.strip().str.endswith("!")]
        if df.empty:
            raise ValueError("no futures bars in ohlcv-1d response")
        return df

    order = ((_fut_via_ohlcv, _fut_via_statistics) if m["ds"] == _ICE
             else (_fut_via_statistics, _fut_via_ohlcv))
    out["fut_ohlcv"] = pd.DataFrame()
    for fn in order:
        try:
            out["fut_ohlcv"] = fn()
            break
        except Exception:
            continue

    return out


def _load_data_disk(mkt_key: str, date_str: str,
                    version: int = _LOAD_DATA_VERSION) -> dict:
    """Pure (NO Streamlit) disk-cached loader — safe to call from worker threads.
    Returns the pickle if present; otherwise fetches fresh and persists it, but NEVER
    persists an error / empty-opt_defs result. This is what the Vol Dashboard's parallel
    prefetch and build call; `_load_data` is the in-memory wrapper over it."""
    path = _load_data_disk_path(mkt_key, date_str, version)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    data = _load_data_fresh(mkt_key, date_str)
    failed = (data.get("error")
              or data.get("opt_defs") is None
              or getattr(data.get("opt_defs"), "empty", True))
    if not failed:
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def _load_data(mkt_key: str, date_str: str, version: int = _LOAD_DATA_VERSION) -> dict:
    """In-memory (st.cache_data) wrapper over the disk layer — unchanged signature/behaviour
    for the Options tab; now also survives server restarts via the shared pickle."""
    return _load_data_disk(mkt_key, date_str, version)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_hist_data(mkt_key: str, date_str: str, version: int = _LOAD_DATA_VERSION) -> dict:
    """Load option OHLCV + futures OHLCV for a historical date (smiles)."""
    m = _MARKETS_V2[mkt_key]
    key = _api_key()
    if not key:
        return {}
    client = db.Historical(key=key)
    s, e = _date_window(date_str)
    if m["ds"].startswith("XEUR"):   # Eurex license ends 22:00 UTC — see _load_data_fresh
        e = f"{date_str}T21:59:59"
    out = {}
    monthly_sym = m.get("monthly_opt_sym")

    try:
        ohlcv = _fetch_opt_ohlcv(client, m["ds"], m["opt_sym"], s, e)[["instrument_id", "close"]].copy()
    except Exception:
        ohlcv = pd.DataFrame()

    if monthly_sym:
        try:
            monthly_ohlcv = _fetch_opt_ohlcv(client, m["ds"], monthly_sym, s, e)[["instrument_id", "close"]].copy()
            if not monthly_ohlcv.empty:
                ohlcv = (pd.concat([ohlcv, monthly_ohlcv], ignore_index=True)
                           .drop_duplicates(subset=["instrument_id"])
                           .reset_index(drop=True))
        except Exception:
            pass

    out["opt_ohlcv"] = ohlcv
    try:
        raw = client.timeseries.get_range(dataset=m["ds"], symbols=[m["fut_sym"]],
                                           schema="ohlcv-1d", start=s, end=e, stype_in="parent")
        out["fut_ohlcv"] = raw.to_df().reset_index()[["instrument_id", "close", "symbol"]].copy()
    except Exception:
        out["fut_ohlcv"] = pd.DataFrame()
    return out

# ── Futures price helpers ─────────────────────────────────────────────────────

def _fut_price_yf(mkt_key: str) -> Optional[float]:
    sym = _YF_FALLBACK.get(mkt_key)
    if not sym:
        return None
    try:
        import yfinance as yf
        return float(yf.Ticker(sym).fast_info.last_price)
    except Exception:
        return None

def _parse_fut_delivery(symbol: str) -> Optional[date]:
    """Parse 'RTYM6' → approx delivery date (15th of contract month).

    Also handles Eurex-style descriptive future symbols that carry an embedded
    YYYYMMDD delivery date (e.g. 'FESX SI 20260918 CS' → 2026-09-18). XEUR index
    futures (FESX/FDAX) arrive with no usable underlying_id on the OESX/ODAX option
    defs (it settles to 0), so layer-1 can't link them — this lets layer-2's
    nearest-delivery logic resolve F from the real Databento future settlement
    instead of falling through to the yfinance cash index. No existing CME/ICE
    future symbol contains an 8-digit 20xxxxxx run, so this is a no-op for them."""
    s = str(symbol).strip()
    m = re.match(r"^[A-Z]+([FGHJKMNQUVXZ])(\d)$", s)
    if m:
        try:
            month = _MONTH_CODES[m.group(1)]
            year  = 2020 + int(m.group(2))
            return date(year, month, 15)
        except Exception:
            pass
    m2 = re.search(r"(20\d{2})(\d{2})(\d{2})", s)
    if m2:
        try:
            return date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
        except Exception:
            pass
    return None

def _get_fut_price(opt_expiry: date, defs_exp: pd.DataFrame, fut_ohlcv: pd.DataFrame,
                   mkt_key: str) -> Optional[float]:
    """Futures settlement for a given option expiry. Three fallback layers."""
    return _get_fut_info(opt_expiry, defs_exp, fut_ohlcv, mkt_key)[0]


def _get_fut_info(opt_expiry: date, defs_exp: pd.DataFrame, fut_ohlcv: pd.DataFrame,
                  mkt_key: str) -> tuple:
    """Returns (futures_price, futures_symbol_label). Three fallback layers."""
    # 1. underlying_id match (exact)
    if "underlying_id" in defs_exp.columns and not fut_ohlcv.empty:
        for uid in defs_exp["underlying_id"].dropna().unique():
            try:
                row = fut_ohlcv[fut_ohlcv["instrument_id"] == int(uid)]
                if not row.empty:
                    price = _sf(row.iloc[0]["close"])
                    sym   = str(row.iloc[0].get("symbol", "")) if "symbol" in row.columns else ""
                    if price:
                        return price, sym
            except Exception:
                pass

    # 2. Symbol-based nearest delivery
    if not fut_ohlcv.empty and "symbol" in fut_ohlcv.columns:
        df_f = fut_ohlcv.copy()
        df_f["_del"] = df_f["symbol"].apply(_parse_fut_delivery)
        df_f = df_f.dropna(subset=["_del"])
        if not df_f.empty:
            candidates = df_f[df_f["_del"] >= opt_expiry]
            if not candidates.empty:
                # Nearest contract expiring on or after the option expiry
                row = candidates.sort_values("_del").iloc[0]
            else:
                # All available futures expire before the option — pick the latest
                # (closest from below, better than the front month which may be years earlier)
                row = df_f.sort_values("_del").iloc[-1]
            price = _sf(row["close"])
            if price:
                return price, str(row.get("symbol", "nearest"))

    # 3. yfinance
    yf_sym = _YF_FALLBACK.get(mkt_key, "yfinance")
    return _fut_price_yf(mkt_key), yf_sym


def _infer_fut_pcp(df: pd.DataFrame, expiry: date, r: float) -> Optional[float]:
    """Infer futures price from put-call parity at available strikes."""
    if df.empty or "instrument_class" not in df.columns:
        return None
    calls = df[df["instrument_class"] == "C"].set_index("strike_price")["close"]
    puts  = df[df["instrument_class"] == "P"].set_index("strike_price")["close"]
    common = calls.index.intersection(puts.index)
    if common.empty:
        return None
    T = max((expiry - date.today()).days, 1) / 365.0
    estimates = []
    for k in common:
        c, p = _sf(calls[k]), _sf(puts[k])
        if c and p:
            estimates.append(k + (c - p) * math.exp(r * T))
    return float(np.median(estimates)) if estimates else None

# ── Chain building ─────────────────────────────────────────────────────────────

def _pick_equidistant(candidates: list, center: float, limit: float, n: int) -> list:
    """Pick n strikes spaced evenly between center and limit from candidates."""
    if not candidates or n == 0:
        return []
    if len(candidates) <= n:
        return list(candidates)
    targets   = [center + (limit - center) * (i + 1) / n for i in range(n)]
    remaining = list(candidates)
    picked    = []
    for t in targets:
        if not remaining:
            break
        best = min(remaining, key=lambda x: abs(x - t))
        picked.append(best)
        remaining.remove(best)
    return picked


# Cached: pure-CPU IV solving that otherwise re-runs on EVERY app rerun once the tab
# is loaded (loaded st.tabs bodies execute even while hidden). `_data` skip-hashed;
# `cache_date` keys the trade date so entries roll daily.
@st.cache_data(ttl=3600, show_spinner=False)
def _build_chain(mkt_key: str, expiry: date, n_per_side: int, sigma_range: float,
                 _data: dict, cache_date: str = "") -> pd.DataFrame:
    data = _data
    m = _MARKETS_V2[mkt_key]
    defs  = data.get("opt_defs", pd.DataFrame())
    ohlcv = data.get("opt_ohlcv", pd.DataFrame())
    futs  = data.get("fut_ohlcv", pd.DataFrame())

    if defs.empty or "expiration" not in defs.columns:
        return pd.DataFrame()

    defs_exp = defs[defs["expiration"].dt.date == expiry].copy()
    if defs_exp.empty:
        return pd.DataFrame()

    # Left join: keep all defined strikes; prices are NaN where CME has no settlement.
    # This ensures near-ATM strikes always appear even when only deep-OTM has settlements.
    ohlcv_cols = ohlcv[["instrument_id", "close", "volume"]] if not ohlcv.empty else pd.DataFrame(columns=["instrument_id", "close", "volume"])
    df = defs_exp.merge(ohlcv_cols, on="instrument_id", how="left")
    if df.empty:
        return pd.DataFrame()

    F, fut_sym = _get_fut_info(expiry, defs_exp, futs, mkt_key)
    if not F:
        F = _infer_fut_pcp(df.dropna(subset=["close"]), expiry, m["r"])
        fut_sym = "put-call parity"
    if not F:
        return pd.DataFrame()

    T = max((expiry - date.today()).days, 1) / 365.0
    r = m["r"]

    # Full call/put pivot across all defined strikes (prices NaN where unsettled)
    calls = (df[df["instrument_class"] == "C"]
             .groupby("strike_price", as_index=False).first()
             [["strike_price", "close", "volume"]]
             .rename(columns={"close": "call_p", "volume": "call_vol"}))
    puts  = (df[df["instrument_class"] == "P"]
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

    # Strike selection: sigma defines boundary, n_per_side picks equidistant within it
    if sigma_range > 0:
        atm_row  = chain_full[chain_full["strike"] == atm_k].iloc[0]
        atm_iv_c = _calc_iv(_sf(atm_row.get("call_p")), F, atm_k, T, r, "C")
        atm_iv_p = _calc_iv(_sf(atm_row.get("put_p")),  F, atm_k, T, r, "P")
        atm_iv   = atm_iv_c or atm_iv_p

        # ATM strike may have no settlement (far-dated sparse data); use nearest settled IV
        if not atm_iv:
            _settled = chain_full.dropna(subset=["call_p", "put_p"], how="all")
            if not _settled.empty:
                _near = _settled.iloc[(_settled["strike"] - F).abs().argsort()[:1]].iloc[0]
                atm_iv = (_calc_iv(_sf(_near.get("call_p")), F, _near["strike"], T, r, "C") or
                          _calc_iv(_sf(_near.get("put_p")),  F, _near["strike"], T, r, "P"))

        if atm_iv:
            sigma_move  = sigma_range * atm_iv * math.sqrt(T) * F
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

    chain["call_iv"]    = chain.apply(
        lambda row: _calc_iv(_sf(row.get("call_p")), F, row["strike"], T, r, "C"), axis=1)
    chain["put_iv"]     = chain.apply(
        lambda row: _calc_iv(_sf(row.get("put_p")),  F, row["strike"], T, r, "P"), axis=1)
    # CME doesn't publish settlements for deep ITM options, so one side may be missing.
    # By put-call parity IV(call) = IV(put) at the same strike — fill from the other side.
    chain["call_iv"] = chain["call_iv"].fillna(chain["put_iv"])
    chain["put_iv"]  = chain["put_iv"].fillna(chain["call_iv"])
    chain["F"]          = F
    chain["T"]          = T
    chain["fut_symbol"] = fut_sym
    return chain


def _build_hist_smile(mkt_key: str, expiry: date, defs: pd.DataFrame, hist: dict,
                       hist_date: Optional[date] = None) -> pd.DataFrame:
    """IV smile for one historical date, reusing today's definitions.

    hist_date: the settlement date of the historical data — used to compute
               the correct DTE at that time (not today's DTE).
    Returns columns: strike, iv, F  (F is the historical forward price).
    """
    m = _MARKETS_V2[mkt_key]
    hohlcv = hist.get("opt_ohlcv", pd.DataFrame())
    hfuts  = hist.get("fut_ohlcv", pd.DataFrame())
    if defs.empty or hohlcv.empty or "expiration" not in defs.columns:
        return pd.DataFrame()

    defs_exp = defs[defs["expiration"].dt.date == expiry].copy()
    if defs_exp.empty:
        return pd.DataFrame()

    df = defs_exp.merge(hohlcv[["instrument_id", "close"]], on="instrument_id", how="inner")
    if df.empty:
        return pd.DataFrame()

    F = _get_fut_price(expiry, defs_exp, hfuts, mkt_key)
    if not F:
        F = _infer_fut_pcp(df, expiry, m["r"])
    if not F:
        return pd.DataFrame()

    # Use DTE as of the historical date so IVs are correct, not inflated by shorter today-T
    ref_date = hist_date or date.today()
    T = max((expiry - ref_date).days, 1) / 365.0
    r = m["r"]

    calls = df[df["instrument_class"] == "C"].groupby("strike_price")["close"].first()
    puts  = df[df["instrument_class"] == "P"].groupby("strike_price")["close"].first()
    all_strikes = sorted(set(calls.index) | set(puts.index))

    rows = []
    for k in all_strikes:
        c = _sf(calls.get(k)) if k in calls.index else None
        p = _sf(puts.get(k))  if k in puts.index  else None
        iv_c = _calc_iv(c, F, k, T, r, "C")
        iv_p = _calc_iv(p, F, k, T, r, "P")
        iv = iv_c or iv_p
        if iv:
            rows.append({"strike": k, "iv": iv, "F": F})

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Vol surface ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _build_surface_data(mkt_key: str, date_str: str,
                        version: int = _LOAD_DATA_VERSION) -> dict:
    """Fit a cubic spline smile for every available expiry. Returns surface dict."""
    data = _load_data(mkt_key, date_str)
    m    = _MARKETS_V2[mkt_key]
    r    = m["r"]
    defs = data.get("opt_defs", pd.DataFrame())
    if defs.empty or "expiration" not in defs.columns:
        return {}

    today    = date.today()
    expiries = sorted({ex.date() for ex in defs["expiration"].dropna()
                       if 7 <= (ex.date() - today).days <= 400})[:12]

    surface = {}
    for exp in expiries:
        try:
            chain = _build_chain(mkt_key, exp, 15, 3.0, data, cache_date=date_str)
            if chain.empty or len(chain) < 3:
                continue
            F = float(chain["F"].iloc[0])
            T = float(chain["T"].iloc[0])
            chain = chain.sort_values("strike").copy()
            chain["iv_otm"] = chain.apply(
                lambda row: (
                    (_sf(row.get("put_iv")) if row["strike"] <= F else _sf(row.get("call_iv")))
                    or (_sf(row.get("call_iv")) if row["strike"] <= F else _sf(row.get("put_iv")))
                ), axis=1)
            smile = chain.dropna(subset=["iv_otm"])
            if len(smile) < 3:
                continue
            strikes  = smile["strike"].tolist()
            ivs      = smile["iv_otm"].tolist()
            log_m    = [math.log(k / F) for k in strikes]
            spline   = CubicSpline(log_m, ivs, bc_type="natural", extrapolate=False)
            surface[exp] = dict(
                F=F, T=T, r=r,
                strikes=strikes, ivs=ivs,
                log_m=log_m, spline=spline,
                min_log_m=min(log_m), max_log_m=max(log_m),
                min_strike=min(strikes), max_strike=max(strikes),
            )
        except Exception:
            continue
    return surface


def _iv_from_surface(surface: dict, expiry: date, strike: float) -> Optional[float]:
    """Interpolate (or flat-extrapolate) IV from the fitted surface."""
    s = surface.get(expiry)
    if not s:
        return None
    F      = s["F"]
    lm     = math.log(strike / F)
    spline = s["spline"]
    lm_c   = max(s["min_log_m"], min(s["max_log_m"], lm))   # clamp → flat extrapolation
    iv     = float(spline(lm_c))
    return max(iv, 0.001)


def _black76_greeks(F, K, T, r, sigma, right):
    """Black-76 price + delta + gamma + vega(per 1pp IV) + theta(per day)."""
    sqT = math.sqrt(T)
    d1  = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqT)
    d2  = d1 - sigma * sqT
    df  = math.exp(-r * T)
    if right == "C":
        price = df * (F * norm.cdf(d1)  - K * norm.cdf(d2))
        delta = df * norm.cdf(d1)
    else:
        price = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
        delta = -df * norm.cdf(-d1)
    gamma = df * norm.pdf(d1) / (F * sigma * sqT)
    vega  = df * F * norm.pdf(d1) * sqT / 100        # per 1 vol point (1pp)
    theta = (-df * F * norm.pdf(d1) * sigma / (2 * sqT)
             - r * price) / 365                       # per calendar day
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)


def _price_spread(F, K1, K2, T, r, iv1, iv2, right: str) -> dict:
    """
    Price a vertical spread.
    Call spread: long K1 call / short K2 call  (K1 < K2, bullish).
    Put  spread: long K2 put  / short K1 put   (K1 < K2, bearish).
    """
    leg1 = _black76_greeks(F, K1, T, r, iv1, right)
    leg2 = _black76_greeks(F, K2, T, r, iv2, right)
    if right == "C":
        sign1, sign2 = +1, -1     # long lower call, short upper call
    else:
        sign1, sign2 = -1, +1     # short lower put,  long upper put
    return dict(
        price  = sign1 * leg1["price"] + sign2 * leg2["price"],
        delta  = sign1 * leg1["delta"] + sign2 * leg2["delta"],
        gamma  = sign1 * leg1["gamma"] + sign2 * leg2["gamma"],
        vega   = sign1 * leg1["vega"]  + sign2 * leg2["vega"],
        theta  = sign1 * leg1["theta"] + sign2 * leg2["theta"],
        iv1=iv1, iv2=iv2,
        price1=leg1["price"], price2=leg2["price"],
    )


_STRUCTURE_TYPES = [
    "Call Spread", "Put Spread",
    "Call Butterfly", "Put Butterfly",
    "Call Condor", "Put Condor",
    "Outright Call", "Outright Put",
]

_STRIKE_LABELS: dict[str, list[str]] = {
    "Call Spread":    ["K1 (long, lower)", "K2 (short, upper)"],
    "Put Spread":     ["K1 (short, lower)", "K2 (long, upper)"],
    "Call Butterfly": ["K1 (wing)", "K2 (body, short ×2)", "K3 (wing)"],
    "Put Butterfly":  ["K1 (wing)", "K2 (body, short ×2)", "K3 (wing)"],
    "Call Condor":    ["K1 (long)", "K2 (short)", "K3 (short)", "K4 (long)"],
    "Put Condor":     ["K1 (long)", "K2 (short)", "K3 (short)", "K4 (long)"],
    "Outright Call":  ["Strike (K)"],
    "Outright Put":   ["Strike (K)"],
}

def _structure_qtys(stype: str) -> list[int]:
    if "Outright" in stype:
        return [1]
    if "Spread" in stype:
        return [1, -1] if "Call" in stype else [-1, 1]
    if "Butterfly" in stype:
        return [1, -2, 1]
    if "Condor" in stype:
        return [1, -1, -1, 1]
    return []


def _price_structure(F: float, T: float, r: float,
                     strikes: list, ivs: list, stype: str) -> dict:
    right = "P" if "Put" in stype else "C"
    qtys  = _structure_qtys(stype)
    legs_g = [_black76_greeks(F, K, T, r, iv, right)
              for K, iv in zip(strikes, ivs)]
    result = {k: sum(q * g[k] for q, g in zip(qtys, legs_g))
              for k in ("price", "delta", "gamma", "vega", "theta")}
    result["legs"] = [
        dict(K=K, iv=iv, qty=q, price=g["price"])
        for K, iv, q, g in zip(strikes, ivs, qtys, legs_g)
    ]
    return result


def _surface_heatmap(surface: dict) -> go.Figure:
    """IV heatmap: x = log-moneyness grid, y = DTE."""
    if not surface:
        return go.Figure()
    today = date.today()
    expiries = sorted(surface.keys())
    dtes     = [(e - today).days for e in expiries]

    # Unified log-moneyness grid centred on 0, ±0.25
    xs   = np.linspace(-0.20, 0.20, 80)
    zmat = []
    for exp in expiries:
        s = surface[exp]
        F = s["F"]
        row = []
        for lm in xs:
            lm_c = max(s["min_log_m"], min(s["max_log_m"], lm))
            row.append(float(s["spline"](lm_c)) * 100)
        zmat.append(row)

    xticks = [f"{x:+.0%}" for x in xs[::10]]

    fig = go.Figure(go.Heatmap(
        z=zmat,
        x=list(xs),
        y=dtes,
        colorscale="RdYlGn_r",
        colorbar=dict(title="IV %", thickness=14, len=0.8),
        hovertemplate="Moneyness: %{x:.3f}<br>DTE: %{y}<br>IV: <b>%{z:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="Implied Vol Surface  (log-moneyness vs DTE)",
                   font=dict(size=13, color=_C["text"])),
        paper_bgcolor=_C["paper"], plot_bgcolor=_C["plot"],
        xaxis=dict(title="Log-moneyness (ln K/F)", gridcolor=_C["grid"],
                   tickvals=list(xs[::10]), ticktext=xticks),
        yaxis=dict(title="DTE", gridcolor=_C["grid"]),
    )
    return fig


def _smile_with_legs(surface: dict, expiry: date,
                     strike_annotations: list) -> go.Figure:
    """Fitted smile for one expiry with annotated strikes.
    strike_annotations: list of (K, label, color) tuples."""
    s = surface.get(expiry)
    if not s:
        return go.Figure()
    F     = s["F"]
    xs_lm = np.linspace(s["min_log_m"] * 1.05, s["max_log_m"] * 1.05, 200)
    xs_k  = [F * math.exp(lm) for lm in xs_lm]
    ys_iv = [float(s["spline"](max(s["min_log_m"], min(s["max_log_m"], lm)))) * 100
             for lm in xs_lm]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs_k, y=ys_iv, name="Fitted smile",
        mode="lines", line=dict(color=_C["blue"], width=2),
        hovertemplate="K=%{x:,.6~f}<br>IV=%{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=s["strikes"], y=[v * 100 for v in s["ivs"]],
        name="Settlement IVs", mode="markers",
        marker=dict(size=5, color=_C["slate"]),
        hovertemplate="K=%{x:,.6~f}<br>IV=%{y:.1f}%<extra></extra>",
    ))
    for K, label, col in strike_annotations:
        iv = _iv_from_surface(surface, expiry, K)
        if iv:
            fig.add_trace(go.Scatter(
                x=[K], y=[iv * 100], name=label, mode="markers",
                marker=dict(size=12, symbol="diamond", color=col,
                            line=dict(color="#fff", width=1.5)),
                hovertemplate=f"{label}<br>IV=%{{y:.1f}}%<extra></extra>",
            ))
    fig.add_vline(x=F, line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"F {_strike_fmt(F)}", annotation_font_size=10,
                  annotation_position="top")
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=_C["paper"], plot_bgcolor=_C["plot"],
        xaxis=dict(title="Strike", gridcolor=_C["grid"], zeroline=False),
        yaxis=dict(title="IV %",   gridcolor=_C["grid"], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
    )
    return fig


# ── Expiry curve ───────────────────────────────────────────────────────────────

# Cached for the same rerun-recompute reason as _build_chain.
@st.cache_data(ttl=3600, show_spinner=False)
def _build_expiry_curve(mkt_key: str, sigma: float, _data: dict,
                        n_expiries: int = 6, cache_date: str = "") -> list:
    data  = _data
    m     = _MARKETS_V2[mkt_key]
    defs  = data.get("opt_defs", pd.DataFrame())
    ohlcv = data.get("opt_ohlcv", pd.DataFrame())
    futs  = data.get("fut_ohlcv", pd.DataFrame())

    if defs.empty or ohlcv.empty or "expiration" not in defs.columns:
        return []

    today    = date.today()
    expiries = sorted({r.date() for r in defs["expiration"].dropna()
                       if 7 <= (r.date() - today).days <= 400})[:n_expiries]

    curve = []
    for exp in expiries:
        try:
            defs_exp = defs[defs["expiration"].dt.date == exp].copy()
            df = defs_exp.merge(ohlcv[["instrument_id", "close"]], on="instrument_id", how="inner")
            if df.empty:
                continue

            F, fut_sym = _get_fut_info(exp, defs_exp, futs, mkt_key)
            if not F:
                F = _infer_fut_pcp(df, exp, m["r"])
                fut_sym = "pcp"
            if not F:
                continue

            T = max((exp - today).days, 1) / 365.0
            r = m["r"]
            dte = (exp - today).days

            df["_d"] = (df["strike_price"] - F).abs()
            atm_c = df[df["instrument_class"] == "C"].sort_values("_d").iloc[0]
            atm_p = df[df["instrument_class"] == "P"].sort_values("_d").iloc[0]

            iv_c = _calc_iv(_sf(atm_c["close"]), F, atm_c["strike_price"], T, r, "C")
            iv_p = _calc_iv(_sf(atm_p["close"]), F, atm_p["strike_price"], T, r, "P")
            valid = [x for x in [iv_c, iv_p] if x]
            atm_iv = sum(valid) / len(valid) if valid else None

            row = {"dte": dte, "expiry": exp.strftime("%Y-%m-%d"), "F": F,
                   "fut_sym": fut_sym or "", "atm_iv": atm_iv}

            if sigma > 0 and atm_iv:
                call_k = F * math.exp(+sigma * atm_iv * math.sqrt(T))
                put_k  = F * math.exp(-sigma * atm_iv * math.sqrt(T))

                def _nearest_iv(right: str, target_k: float) -> Optional[float]:
                    sub = df[df["instrument_class"] == right].copy()
                    if sub.empty:
                        return None
                    sub["_dk"] = (sub["strike_price"] - target_k).abs()
                    best = sub.nsmallest(1, "_dk").iloc[0]
                    return _calc_iv(_sf(best["close"]), F, best["strike_price"], T, r, right)

                # 6 significant figures, not 0 decimals — round(0.0064, 0) destroyed
                # JPY's wing strikes to 0.0 (display-only bug; wing IVs were computed
                # from the unrounded targets and were always correct).
                row["call_wing_k"]  = float(f"{call_k:.6g}")
                row["put_wing_k"]   = float(f"{put_k:.6g}")
                row["call_wing_iv"] = _nearest_iv("C", call_k)
                row["put_wing_iv"]  = _nearest_iv("P", put_k)

            curve.append(row)
        except Exception:
            continue

    return curve

# ── UI helpers ─────────────────────────────────────────────────────────────────

_C = {"paper":"#FFFFFF","plot":"#FAFBFD","grid":"#E8EDF5","text":"#1A202C",
      "green":"#059669","red":"#DC2626","blue":"#2563EB","amber":"#F59E0B",
      "purple":"#A855F7","sky":"#0EA5E9","slate":"#64748B"}

def _pct(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v*100:.1f}%"

def _price_fmt(v, digits=2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{digits}f}"


def _strike_fmt(k) -> str:
    """Adaptive strike precision: index/commodity strikes are 100s-1000s (0dp), FX
    majors ~0.5-1.6 (4dp), JPY ~0.006 (6dp). One formatter serves all markets."""
    try:
        k = float(k)
    except Exception:
        return "—"
    if k >= 1000:
        return f"{k:,.0f}"
    if k >= 5:
        return f"{k:,.2f}"          # Brent-style 0.5-spaced strikes (65.50) keep their halves
    if k >= 0.05:
        return f"{k:,.4f}"
    return f"{k:,.6f}"


def _disp_px(m: dict, v):
    """Display-convention transform for a price/strike level (native -> shown).
    No-op unless the market carries display_invert (e.g. JPY 6J: JPY/USD -> USDJPY).
    Vols are inversion-invariant and never pass through here."""
    if v is None or not m.get("display_invert"):
        return v
    try:
        return 1.0 / float(v) if float(v) != 0 else v
    except Exception:
        return v


def _chain_html(df: pd.DataFrame, F: float, trade_date_str: str, m: dict = None) -> str:
    m = m or {}
    _invert_note = (
        f'<p style="font-size:10px;color:{_C["amber"]};margin:2px 0 0">'
        "Strikes/F shown in USDJPY convention; premiums in native contract units "
        "(JPY/USD points).</p>"
    ) if m.get("display_invert") else ""
    rows = ""
    for _, row in df.iterrows():
        k  = row["strike"]
        atm = abs(k - F) / F < 0.008
        strike_style = "font-weight:700;background:#FEF9C3;" if atm else ""

        cp = _sf(row.get("call_p"))
        pp = _sf(row.get("put_p"))
        civ = row.get("call_iv")
        piv = row.get("put_iv")
        cvol = row.get("call_vol")
        pvol = row.get("put_vol")

        cs = f"color:{_C['green']};" if cp else f"color:{_C['slate']};"
        ps = f"color:{_C['red']};"   if pp else f"color:{_C['slate']};"

        c_pvol = f"{int(cvol):,}" if cvol and not math.isnan(float(cvol)) else "—"
        p_pvol = f"{int(pvol):,}" if pvol and not math.isnan(float(pvol)) else "—"

        rows += (
            f"<tr>"
            f"<td style='{cs}'>{_price_fmt(cp)}</td>"
            f"<td style='{cs}'>{_pct(civ)}</td>"
            f"<td style='{cs}'>{c_pvol}</td>"
            f"<td style='{strike_style}'>{_strike_fmt(_disp_px(m, k))}</td>"
            f"<td style='{ps}'>{_price_fmt(pp)}</td>"
            f"<td style='{ps}'>{_pct(piv)}</td>"
            f"<td style='{ps}'>{p_pvol}</td>"
            f"</tr>\n"
        )

    return f"""
<style>
.ov2{{font-size:12px;border-collapse:collapse;width:100%;font-family:'SF Mono',Consolas,monospace}}
.ov2 th{{font-size:10px;font-weight:600;color:{_C['slate']};text-transform:uppercase;
         letter-spacing:.5px;padding:5px 8px;border-bottom:2px solid #E2E8F0;text-align:right}}
.ov2 td{{padding:3px 8px;border-bottom:1px solid #F1F5F9;text-align:right}}
.ov2 tr:hover{{background:#F8FAFC}}
</style>
<table class="ov2">
<thead>
<tr>
  <th colspan="3" style="color:{_C['green']};text-align:center">— Calls —</th>
  <th style="color:{_C['text']}">Strike</th>
  <th colspan="3" style="color:{_C['red']};text-align:center">— Puts —</th>
</tr>
<tr>
  <th>Settle</th><th>IV</th><th>Vol</th>
  <th></th>
  <th>Settle</th><th>IV</th><th>Vol</th>
</tr>
</thead>
<tbody>{rows}</tbody>
</table>
<p style="font-size:10px;color:{_C['slate']};margin:6px 0 0">
  Settlement {trade_date_str} &nbsp;|&nbsp; F = {_disp_px(m, F):,.2f} &nbsp;|&nbsp; IVs via Black-76
</p>{_invert_note}"""


def _smile_chart(df_chain: pd.DataFrame,
                 overlay: Optional[pd.DataFrame] = None,
                 overlay_label: str = "", m: dict = None) -> go.Figure:
    m = m or {}
    fig = go.Figure()
    F = df_chain["F"].iloc[0]

    # OTM convention: put IV below ATM, call IV above ATM — single smile line
    # Use _sf so np.nan is treated as missing (nan is truthy so plain `or` fails)
    chain = df_chain.sort_values("strike").copy()
    chain["iv"] = chain.apply(
        lambda r: (_sf(r["put_iv"]) if r["strike"] <= F else _sf(r["call_iv"]))
                  or (_sf(r["call_iv"]) if r["strike"] <= F else _sf(r["put_iv"])),
        axis=1)
    smile = chain.dropna(subset=["iv"])

    fig.add_trace(go.Scatter(x=smile["strike"].map(lambda v: _disp_px(m, v)), y=smile["iv"] * 100,
        name="Today", mode="lines+markers",
        line=dict(color=_C["blue"], width=2), marker=dict(size=5)))

    if overlay is not None and not overlay.empty:
        s = overlay.dropna(subset=["iv"]).sort_values("strike")
        # Clip to current smile strike range so deep ITM/OTM junk doesn't widen the axis
        # (compare in NATIVE units — display inversion is applied only at plot time)
        if not smile.empty:
            s = s[(s["strike"] >= smile["strike"].min()) &
                  (s["strike"] <= smile["strike"].max())]
        if not s.empty:
            fig.add_trace(go.Scatter(x=s["strike"].map(lambda v: _disp_px(m, v)), y=s["iv"] * 100,
                name=overlay_label, mode="lines+markers",
                line=dict(color=_C["amber"], width=1.5, dash="dash"),
                marker=dict(size=4)))

    fig.add_vline(x=_disp_px(m, F), line_dash="dot", line_color="#94A3B8",
                  annotation_text=f"F {_strike_fmt(_disp_px(m, F))}", annotation_position="top",
                  annotation_font_size=10)

    if overlay is not None and not overlay.empty and "F" in overlay.columns:
        hist_F = float(overlay["F"].iloc[0])
        if hist_F != F:
            fig.add_vline(x=_disp_px(m, hist_F), line_dash="dot", line_color=_C["amber"],
                          annotation_text=f"F {_strike_fmt(_disp_px(m, hist_F))} ({overlay_label})",
                          annotation_position="top left",
                          annotation_font_size=10,
                          annotation_font_color=_C["amber"])

    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=_C["paper"], plot_bgcolor=_C["plot"],
        xaxis=dict(title="Strike", gridcolor=_C["grid"], zeroline=False),
        yaxis=dict(title="IV %",   gridcolor=_C["grid"], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(255,255,255,.85)"),
        hovermode="x unified",
    )
    return fig


def _moneyness_smile_chart(df_chain: pd.DataFrame, sigma_range: float,
                            overlay: Optional[pd.DataFrame] = None,
                            overlay_label: str = "", m: dict = None) -> Optional[go.Figure]:
    m = m or {}
    """Vol smile in σ-moneyness space: x = (K − F) / (F × σ_ATM × √T).

    ATM = 0 always, so current and historical smiles align even when F moves.
    sigma_range controls x-axis limits (same slider as the chain range).
    overlay must have columns: strike, iv, F  (F = historical forward).
    """
    F = float(df_chain["F"].iloc[0])
    T = float(df_chain["T"].iloc[0])

    # OTM-convention IV: put below ATM, call above ATM
    chain = df_chain.sort_values("strike").copy()
    chain["iv"] = chain.apply(
        lambda r: (_sf(r["put_iv"]) if r["strike"] <= F else _sf(r["call_iv"]))
                  or (_sf(r["call_iv"]) if r["strike"] <= F else _sf(r["put_iv"])),
        axis=1)
    smile = chain.dropna(subset=["iv"])
    if smile.empty:
        return None

    # ATM IV = IV of the strike nearest to F
    atm_iv = float(smile.iloc[(smile["strike"] - F).abs().argsort()[:1]]["iv"].iloc[0])
    if not atm_iv or not math.isfinite(atm_iv) or atm_iv <= 0:
        return None

    sigma_move = F * atm_iv * math.sqrt(T)   # 1σ move in price units
    x_lim = max(float(sigma_range), 1.0)

    smile = smile.copy()
    smile["mny"] = (smile["strike"] - F) / sigma_move
    smile = smile[(smile["mny"] >= -x_lim * 1.25) & (smile["mny"] <= x_lim * 1.25)]

    fig = go.Figure()

    if not smile.empty:
        fig.add_trace(go.Scatter(
            x=smile["mny"], y=smile["iv"] * 100,
            name="Today", mode="lines+markers",
            line=dict(color=_C["blue"], width=2),
            marker=dict(size=5),
            customdata=np.column_stack([smile["strike"].map(lambda v: _disp_px(m, v)), smile["iv"]]),
            hovertemplate="K=%{customdata[0]:,.6~f}  IV=%{customdata[1]:.1%}  %{x:.2f}σ<extra>Today</extra>",
        ))

    # Historical overlay — normalised with its own F and ATM IV
    if overlay is not None and not overlay.empty and "F" in overlay.columns:
        hist_F = float(overlay["F"].iloc[0])
        hist_near = overlay.iloc[(overlay["strike"] - hist_F).abs().argsort()[:1]]
        hist_atm_iv = float(hist_near["iv"].iloc[0]) if not hist_near.empty else None
        if hist_atm_iv and math.isfinite(hist_atm_iv) and hist_atm_iv > 0:
            hist_sigma_move = hist_F * hist_atm_iv * math.sqrt(T)
            ov = overlay.copy()
            ov["mny"] = (ov["strike"] - hist_F) / hist_sigma_move
            ov = ov[(ov["mny"] >= -x_lim * 1.25) & (ov["mny"] <= x_lim * 1.25)]
            if not ov.empty:
                fig.add_trace(go.Scatter(
                    x=ov["mny"], y=ov["iv"] * 100,
                    name=overlay_label, mode="lines+markers",
                    line=dict(color=_C["amber"], width=1.5, dash="dash"),
                    marker=dict(size=4),
                    customdata=np.column_stack([ov["strike"].map(lambda v: _disp_px(m, v)), ov["iv"]]),
                    hovertemplate="K=%{customdata[0]:,.6~f}  IV=%{customdata[1]:.1%}  %{x:.2f}σ<extra>" + overlay_label + "</extra>",
                ))

    # ATM reference line
    fig.add_vline(x=0, line_dash="dot", line_color="#94A3B8",
                  annotation_text="ATM", annotation_position="top right",
                  annotation_font_size=10)

    # Tick grid at 0.5σ intervals
    step = 0.5 if x_lim <= 2.5 else 1.0
    raw = np.arange(-x_lim, x_lim + 1e-9, step)
    tick_vals = [round(v, 2) for v in raw]
    tick_text = ["ATM" if abs(v) < 0.01 else f"{v:+.1f}σ" for v in tick_vals]

    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="Vol Smile — Moneyness (σ units)", font=dict(size=13, color=_C["text"])),
        paper_bgcolor=_C["paper"], plot_bgcolor=_C["plot"],
        xaxis=dict(
            title="Moneyness (σ)",
            gridcolor=_C["grid"], zeroline=True, zerolinecolor="#94A3B8",
            tickvals=tick_vals, ticktext=tick_text,
            range=[-x_lim * 1.15, x_lim * 1.15],
        ),
        yaxis=dict(title="IV %", gridcolor=_C["grid"], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(255,255,255,.85)"),
        hovermode="x unified",
    )
    return fig


def _curve_chart(curve: list, sigma_label: str, m: dict = None) -> go.Figure:
    m = m or {}
    fig = go.Figure()
    dtes    = [r["dte"] for r in curve]
    labels  = [f"{r['expiry']}  {r.get('fut_sym', '')}  F={_disp_px(m, r['F']):,.1f}" for r in curve]

    # ATM line
    atm_ivs = [r["atm_iv"] * 100 if r["atm_iv"] else None for r in curve]
    fig.add_trace(go.Scatter(x=dtes, y=atm_ivs, name="ATM IV",
        mode="lines+markers", line=dict(color=_C["blue"], width=2.5),
        marker=dict(size=7),
        text=labels, hovertemplate="%{text}<br>ATM IV: <b>%{y:.1f}%</b><extra></extra>"))

    if sigma_label != "ATM only":
        call_ivs = [r.get("call_wing_iv", 0) * 100 if r.get("call_wing_iv") else None for r in curve]
        put_ivs  = [r.get("put_wing_iv",  0) * 100 if r.get("put_wing_iv")  else None for r in curve]
        if any(v for v in call_ivs if v):
            fig.add_trace(go.Scatter(x=dtes, y=call_ivs, name=f"+{sigma_label} Call",
                mode="lines+markers", line=dict(color=_C["red"], width=1.5, dash="dot"),
                marker=dict(size=5)))
        if any(v for v in put_ivs if v):
            fig.add_trace(go.Scatter(x=dtes, y=put_ivs, name=f"-{sigma_label} Put",
                mode="lines+markers", line=dict(color=_C["green"], width=1.5, dash="dot"),
                marker=dict(size=5)))

    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor=_C["paper"], plot_bgcolor=_C["plot"],
        xaxis=dict(title="DTE", gridcolor=_C["grid"], zeroline=False),
        yaxis=dict(title="IV %", gridcolor=_C["grid"], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        hovermode="x unified",
    )
    return fig


def _curve_table_html(curve: list, sigma: float, m: dict = None) -> str:
    m = m or {}
    has_wings = sigma > 0
    hdr = "<tr><th>Expiry</th><th>DTE</th><th>Contract</th><th>Futures</th><th>ATM IV</th>"
    if has_wings:
        hdr += f"<th>+{sigma}σ Strike</th><th>+{sigma}σ IV</th><th>-{sigma}σ Strike</th><th>-{sigma}σ IV</th>"
    hdr += "</tr>"

    rows = ""
    for r in curve:
        atm_iv = _pct(r["atm_iv"])
        fut_sym = r.get("fut_sym", "")
        row = (f"<tr><td>{r['expiry']}</td><td>{r['dte']}</td>"
               f"<td style='font-weight:600'>{fut_sym}</td>"
               f"<td>{_disp_px(m, r['F']):,.1f}</td><td>{atm_iv}</td>")
        if has_wings:
            cwk = r.get("call_wing_k")
            pwk = r.get("put_wing_k")
            row += (f"<td>{_disp_px(m, cwk) if cwk is not None else '—'}</td>"
                    f"<td>{_pct(r.get('call_wing_iv'))}</td>"
                    f"<td>{_disp_px(m, pwk) if pwk is not None else '—'}</td>"
                    f"<td>{_pct(r.get('put_wing_iv'))}</td>")
        row += "</tr>\n"
        rows += row

    return f"""
<style>
.crv2{{font-size:12px;border-collapse:collapse;width:100%}}
.crv2 th{{font-size:10px;font-weight:600;color:{_C['slate']};text-transform:uppercase;
           letter-spacing:.5px;padding:5px 10px;border-bottom:2px solid #E2E8F0;text-align:right}}
.crv2 td{{padding:4px 10px;border-bottom:1px solid #F1F5F9;text-align:right;font-size:12px}}
.crv2 tr:hover{{background:#F8FAFC}}
</style>
<table class="crv2"><thead>{hdr}</thead><tbody>{rows}</tbody></table>"""


# ── Tab helpers (return-safe: return only exits the helper, not render_options_v2) ──

def _render_chain_tab(mkt: str, tdate: date, data: dict):
    err = data.get("defs_err") or data.get("ohlcv_err")
    if err:
        st.error(f"Data error: {err}")
    if data.get("stats_err"):
        with st.expander("⚠ Statistics schema fallback", expanded=False):
            st.caption(data["stats_err"])

    defs  = data.get("opt_defs", pd.DataFrame())

    if not defs.empty and "expiration" in defs.columns:
        today = date.today()
        all_exp = sorted({r.date() for r in defs["expiration"].dropna()
                          if (r.date() - today).days >= 1})
        quarterly = data.get("quarterly_expiries", set())
        qtly_by_month = {e.month: e for e in quarterly if (e - today).days >= 1}
        all_exp = [e for e in all_exp
                   if e in quarterly or e.month not in qtly_by_month]
        first4 = all_exp[:4]
        extra_qtly = [e for e in sorted(quarterly)
                      if e not in first4 and (e - today).days >= 1]
        exp_dates = first4 + extra_qtly[:2]
    else:
        exp_dates = []

    if not exp_dates:
        st.warning("No option expiries found. Check API key and market data availability.")
        if data.get("defs_err"):
            with st.expander("Debug", expanded=True):
                st.code(data.get("defs_err"))
        return

    with st.form("v2_chain_form"):
        exp_labels = {e: f"{e.strftime('%a %b %d %Y')}  ({(e - date.today()).days} DTE)"
                      for e in exp_dates}
        default_exp_idx = st.session_state.get("_v2_exp_idx", 0)
        sel_exp_str = st.selectbox("Option Expiry", list(exp_labels.keys()),
                                   format_func=lambda e: exp_labels[e],
                                   index=min(default_exp_idx, len(exp_dates) - 1))
        c1, c2, c3 = st.columns([2, 2, 1])
        n_per_side = c1.select_slider(
            "Strikes per side", options=[2, 3, 5, 7],
            value=st.session_state.get("_v2_n_per_side", 5))
        sigma_range = c2.select_slider(
            "Range (σ)", options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            value=st.session_state.get("_v2_sigma_range", 2.0),
            format_func=lambda x: f"{x}σ")
        submitted = c3.form_submit_button("⟳  Refresh", use_container_width=True)

    if submitted:
        st.session_state["_v2_exp_idx"]           = exp_dates.index(sel_exp_str)
        st.session_state["_v2_n_per_side"]         = n_per_side
        st.session_state["_v2_sigma_range"]        = sigma_range
        st.session_state["_v2_exp_active"]         = sel_exp_str
        st.session_state["_v2_n_per_side_active"]  = n_per_side
        st.session_state["_v2_sigma_active_chain"] = sigma_range

    exp_active         = st.session_state.get("_v2_exp_active",         sel_exp_str)
    n_per_side_active  = st.session_state.get("_v2_n_per_side_active",  n_per_side)
    sigma_range_active = st.session_state.get("_v2_sigma_active_chain", sigma_range)

    with st.spinner("Building chain..."):
        df_chain = _build_chain(mkt, exp_active, n_per_side_active, sigma_range_active, data,
                                cache_date=str(tdate))

    if df_chain.empty:
        st.warning("No chain data for selected expiry. Try a different expiry or click ⟳.")
        if defs.empty:
            st.info("Definitions empty — check API key / data availability above.")
        return

    m           = _MARKETS_V2[mkt]
    F           = float(df_chain["F"].iloc[0])
    T           = float(df_chain["T"].iloc[0])
    fut_sym_lbl = str(df_chain["fut_symbol"].iloc[0])

    st.caption(
        f"Underlying future: **{fut_sym_lbl}** &nbsp;|&nbsp; "
        f"Close used for ATM: **{_disp_px(m, F):,.2f}**"
    )

    n_available = len(df_chain)
    n_requested = 2 * n_per_side_active + 1
    if n_available < n_requested:
        st.caption(
            f"Only {n_available} strikes with CME settlement data for this expiry "
            f"({n_requested} requested). Far-dated options have fewer settled strikes."
        )
    st.markdown(_chain_html(df_chain, F, tdate.strftime("%b %d %Y"), m), unsafe_allow_html=True)

    _HIST_DAYS = {"1d": 1, "1w": 7, "1m": 30, "3m": 91}
    c_lbl, c_cust, c_run = st.columns([3, 2, 1])
    with c_lbl:
        hist_period = st.radio(
            "Compare vs", ["None", "1d", "1w", "1m", "3m", "Custom"],
            horizontal=True, key="_v2_hist_period")
    custom_date = None
    if hist_period == "Custom":
        with c_cust:
            custom_date = st.date_input(
                "Date", value=_prev_bday(tdate, 7),
                max_value=tdate, key="_v2_hist_custom_date",
                label_visibility="collapsed")
    with c_run:
        st.write("")
        smile_run = st.button("▶ Run", key="_v2_smile_run", use_container_width=True)

    if smile_run:
        if hist_period == "None":
            st.session_state["_v2_overlay_smile"] = None
            st.session_state["_v2_overlay_label"] = ""
        else:
            hdate = (custom_date if hist_period == "Custom" and custom_date
                     else _prev_bday(tdate, _HIST_DAYS.get(hist_period, 7)))
            with st.spinner(f"Loading {hdate.strftime('%b %d')} smile..."):
                hist_data = _load_hist_data(mkt, str(hdate))
                _ov = _build_hist_smile(mkt, exp_active, defs, hist_data, hist_date=hdate)
            st.session_state["_v2_overlay_smile"] = _ov
            st.session_state["_v2_overlay_label"] = hdate.strftime("%b %d")

    overlay_smile = st.session_state.get("_v2_overlay_smile")
    overlay_label = st.session_state.get("_v2_overlay_label", "")

    st.plotly_chart(_smile_chart(df_chain, overlay_smile, overlay_label, m),
                    use_container_width=True)

    mny_fig = _moneyness_smile_chart(df_chain, sigma_range_active, overlay_smile, overlay_label, m)
    if mny_fig:
        st.plotly_chart(mny_fig, use_container_width=True)
        st.caption(
            "X-axis: (K − F) / (F × σ_ATM × √T)  ·  "
            "ATM = 0 always, so smiles align across dates even when the forward moves."
        )

    with st.expander("Chain stats"):
        n_c   = df_chain["call_iv"].notna().sum()
        n_p   = df_chain["put_iv"].notna().sum()
        avg_c = df_chain["call_iv"].dropna().mean()
        avg_p = df_chain["put_iv"].dropna().mean()
        st.markdown(
            f"Strikes: **{len(df_chain)}** ({n_per_side_active}/side, {sigma_range_active}σ cutoff) "
            f"&nbsp;|&nbsp; Calls with IV: **{n_c}** &nbsp;|&nbsp; Puts with IV: **{n_p}** "
            f"&nbsp;|&nbsp; Avg call IV: **{_pct(avg_c)}** &nbsp;|&nbsp; "
            f"Avg put IV: **{_pct(avg_p)}** &nbsp;|&nbsp; DTE: **{int(T * 365)}**"
        )


def _render_curve_tab(mkt: str, tdate: date, tdate_str: str):
    with st.form("v2_curve_form"):
        c1, c2, c3 = st.columns([2, 2, 1])
        sigma_label = c1.selectbox("Vol wings", list(_STD_SIGMA_OPTS.keys()),
                                   index=st.session_state.get("_v2_sigma_idx", 0))
        n_exp = c2.select_slider(
            "Expiries", options=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            value=st.session_state.get("_v2_n_expiries", 6))
        curve_submitted = c3.form_submit_button("⟳  Refresh")

    if curve_submitted:
        st.session_state["_v2_sigma_idx"]    = list(_STD_SIGMA_OPTS.keys()).index(sigma_label)
        st.session_state["_v2_sigma_active"] = sigma_label
        st.session_state["_v2_n_expiries"]   = n_exp

    sigma_active = st.session_state.get("_v2_sigma_active", sigma_label)
    n_exp_active = st.session_state.get("_v2_n_expiries", 6)
    sigma        = _STD_SIGMA_OPTS[sigma_active]

    with st.spinner("Building expiry curve..."):
        data_curve = _load_data(mkt, tdate_str)
        curve = _build_expiry_curve(mkt, sigma, data_curve, n_expiries=n_exp_active,
                                    cache_date=tdate_str)

    if not curve:
        st.warning("No expiry curve data available.")
        return

    filled = [r for r in curve if r.get("atm_iv")]
    if not filled:
        st.warning("Could not compute IVs for any expiry (check data availability).")
        return

    m = _MARKETS_V2[mkt]
    st.plotly_chart(_curve_chart(curve, sigma_active, m), use_container_width=True)
    st.markdown(_curve_table_html(curve, sigma, m), unsafe_allow_html=True)


def _render_pricer_tab(mkt: str, tdate: date, tdate_str: str, data: dict):
    st.markdown("#### Options Pricer")
    st.caption(
        "Builds a synthetic vol surface from settlement IVs. "
        "Choose a structure, enter strikes, and hit Price."
    )
    if _MARKETS_V2[mkt].get("display_invert"):
        st.caption(
            "Pricer works in native JPY/USD units (multiply by nothing — strikes as "
            "listed, e.g. 0.006125); USDJPY-convention display applies to Chain/Expiry "
            "Curve only."
        )

    with st.spinner("Building vol surface..."):
        surface = _build_surface_data(mkt, tdate_str)

    if not surface:
        st.warning("Could not build vol surface — no settlement IV data available.")
        return

    expiries = sorted(surface.keys())
    today    = date.today()

    st.plotly_chart(_surface_heatmap(surface), use_container_width=True)
    st.divider()

    # ── Structure and expiry selectors ────────────────────────────────────────
    col_stype, col_exp = st.columns([2, 3])
    with col_stype:
        stype = st.selectbox("Structure", _STRUCTURE_TYPES, key="_pricer_stype")
    with col_exp:
        exp_labels = {e: f"{e.strftime('%a %b %d %Y')}  ({(e - today).days} DTE)"
                      for e in expiries}
        sel_exp = st.selectbox("Expiry", expiries,
                               format_func=lambda e: exp_labels[e], key="_pricer_exp")

    s = surface[sel_exp]
    F = s["F"]
    T = s["T"]
    r = s["r"]

    # ── Strike inputs (number depends on structure) ───────────────────────────
    def _r25(x: float) -> float:
        return float(round(x / 25) * 25)

    _defaults: dict[str, list[float]] = {
        "Outright Call":  [_r25(F)],
        "Outright Put":   [_r25(F)],
        "Call Spread":    [_r25(F * 0.98), _r25(F * 1.02)],
        "Put Spread":     [_r25(F * 0.98), _r25(F * 1.02)],
        "Call Butterfly": [_r25(F * 0.97), _r25(F),        _r25(F * 1.03)],
        "Put Butterfly":  [_r25(F * 0.97), _r25(F),        _r25(F * 1.03)],
        "Call Condor":    [_r25(F * 0.96), _r25(F * 0.98), _r25(F * 1.02), _r25(F * 1.04)],
        "Put Condor":     [_r25(F * 0.96), _r25(F * 0.98), _r25(F * 1.02), _r25(F * 1.04)],
    }
    labels   = _STRIKE_LABELS[stype]
    defaults = _defaults[stype]
    n_legs   = len(labels)

    strike_cols = st.columns(n_legs)
    strikes: list[float] = []
    for i, (col, lbl, dflt) in enumerate(zip(strike_cols, labels, defaults)):
        with col:
            strikes.append(st.number_input(lbl, value=dflt, step=25.0,
                                           format="%.0f", key=f"_pricer_k{i}"))

    # ── Smile preview (always shown) ─────────────────────────────────────────
    qtys = _structure_qtys(stype)
    leg_colors = [_C["green"] if q > 0 else _C["red"] for q in qtys]
    annotations = [(K, f"K{i+1}={_strike_fmt(K)} ({'L' if q>0 else 'S'}{abs(q) if abs(q)>1 else ''})",
                    col)
                   for i, (K, q, col) in enumerate(zip(strikes, qtys, leg_colors))]
    st.plotly_chart(_smile_with_legs(surface, sel_exp, annotations),
                    use_container_width=True)

    # ── Price button ─────────────────────────────────────────────────────────
    if st.button("▶  Price", key="_pricer_run"):
        # Validation: strikes must be strictly increasing
        if sorted(strikes) != strikes or len(set(strikes)) < n_legs:
            st.error("Strikes must be in strictly increasing order (K1 < K2 < …).")
        else:
            ivs = [_iv_from_surface(surface, sel_exp, K) for K in strikes]
            if any(iv is None for iv in ivs):
                st.error("Could not interpolate IV for one or more strikes.")
            else:
                result = _price_structure(F, T, r, strikes, ivs, stype)

                def _row(label, val):
                    return (f"<tr><td>{label}</td>"
                            f"<td style='text-align:right'><b>{val}</b></td></tr>")

                rows_html = _row("Structure", stype)
                rows_html += _row("Forward (F)", f"{F:,.2f}")
                rows_html += _row("DTE", f"{int(T * 365)}")
                rows_html += _row("Price (pts)", f"{result['price']:.4f}")

                # Width % only meaningful for 2-leg structures
                if n_legs == 2:
                    width = strikes[1] - strikes[0]
                    rows_html += _row("Price (% of width)",
                                      f"{result['price'] / width * 100:.1f}%")
                elif "Butterfly" in stype:
                    wing = strikes[2] - strikes[0]
                    rows_html += _row("Price (% of wing width)",
                                      f"{result['price'] / wing * 100:.2f}%")

                rows_html += _row("Delta", f"{result['delta']:.4f}")
                rows_html += _row("Gamma", f"{result['gamma']:.6f}")
                rows_html += _row("Vega (per 1pp IV)", f"{result['vega']:.4f}")
                rows_html += _row("Theta (per day)", f"{result['theta']:.4f}")

                st.markdown(f"""
<style>
.pricer-tbl {{border-collapse:collapse;width:360px;font-size:13px}}
.pricer-tbl td {{padding:4px 10px;border-bottom:1px solid #334155;color:#CBD5E1}}
.pricer-tbl tr:first-child td {{color:#F1F5F9;font-size:14px;border-bottom:2px solid #475569}}
</style>
<table class="pricer-tbl">{rows_html}</table>""", unsafe_allow_html=True)

                with st.expander("Leg detail", expanded=False):
                    for leg in result["legs"]:
                        q    = leg["qty"]
                        side = f"{'Long' if q > 0 else 'Short'}{f' ×{abs(q)}' if abs(q) > 1 else ''}"
                        st.markdown(
                            f"**{side}** K={_strike_fmt(leg['K'])}  "
                            f"IV={leg['iv']*100:.2f}%  "
                            f"price={leg['price']:.4f}  "
                            f"→ contribution **{q * leg['price']:.4f}**"
                        )


# ── Main render ────────────────────────────────────────────────────────────────

@st.fragment
def render_options_v2():
    if not _api_key():
        st.error("Set DATABENTO_API_KEY in Streamlit secrets or environment variable.")
        return

    with st.expander("📊  Databento usage & costs", expanded=False):
        try:
            import data_costs
            data_costs.render_panel()
        except Exception:
            st.caption("cost ledger unavailable")

    # ── Market selector ───────────────────────────────────────────────────────
    mkt_labels = {k: v["label"] for k, v in _MARKETS_V2.items()}
    prev_mkt = st.session_state.get("_v2_mkt", "ES")
    mkt = st.selectbox("Market", list(mkt_labels.keys()),
                       format_func=lambda k: mkt_labels[k],
                       index=list(mkt_labels.keys()).index(prev_mkt),
                       key="_v2_mkt_sel")
    if mkt != prev_mkt:
        for k in [k for k in st.session_state if k.startswith("_v2_") and k != "_v2_mkt_sel"]:
            del st.session_state[k]
        st.session_state["_v2_mkt"] = mkt
        st.rerun()
    st.session_state["_v2_mkt"] = mkt

    m      = _MARKETS_V2[mkt]
    tdate  = _trade_date(m["ds"])
    tdate_str = str(tdate)

    # Load data — fall back to T-2 if T-1 is not yet fully available on Databento
    # (availability error, license-boundary 403, or empty defs when the session
    # hasn't rolled into historical yet).
    _AVAIL_ERRS = ("dataset_unavailable_range", "data_end_after_available_end",
                   "license_not_found_unauthorized")
    with st.spinner("Loading market data..."):
        data = _load_data(mkt, tdate_str)
        _derr = data.get("defs_err", "")
        _defs_empty = (data.get("opt_defs") is None
                       or getattr(data.get("opt_defs"), "empty", True))
        if _defs_empty or (_derr and any(e in _derr for e in _AVAIL_ERRS)):
            tdate = _prev_bday(tdate, 1)
            tdate_str = str(tdate)
            data = _load_data(mkt, tdate_str)

    st.caption(f"Data: Databento settlement — {tdate.strftime('%a %b %d, %Y')} &nbsp;·&nbsp; Historical plan (previous close)")

    tab_chain, tab_curve, tab_pricer = st.tabs(["Chain", "Expiry Curve", "Pricer"])

    # ════════════════════════════════════════════════════════════════════════
    # Chain tab
    # ════════════════════════════════════════════════════════════════════════
    with tab_chain:
        _render_chain_tab(mkt, tdate, data)

    with tab_curve:
        _render_curve_tab(mkt, tdate, tdate_str)

    with tab_pricer:
        _render_pricer_tab(mkt, tdate, tdate_str, data)

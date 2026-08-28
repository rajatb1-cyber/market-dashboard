"""
CTA Signals Tab
Implements 4 systematic trend-following signals across Macro assets.
Signals: TSMOM | MA Crossover | Donchian Breakout | EWMA
Risk allocation via vol-scaling (target vol / realised vol × signal).
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import yfinance as yf
import ta as ta_lib
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from watchlist import (
    _raw_daily, load_config, CLASS_BG, CLASS_FG,
    FRED_MAP, ECB_MAP, JGB_MAP, ALPHAVANTAGE_FX_MAP,
    _fetch_fred_df, _fetch_ecb_df, _fetch_jgb_df, _fetch_alphavantage_fx,
    _load_cs_json, _fetch_custom_df_cached,
)

@st.cache_data(ttl=1800, show_spinner=False)
def _custom_is_rates(cs_id: str) -> bool:
    """Rates classification for a CUSTOM series from its components (Rajat
    2026-08-28: 'any rates related product should be arithmetic' — the
    convention follows the PRODUCT, not a data heuristic). True when any
    component var is a rates source: sovereign curve / CNBC yield specs,
    STIR db contracts, RATES-prefixed registry labels, or the ^..Y/^..YT
    yield pseudo-tickers."""
    try:
        cs = next((c for c in _load_cs_json() if c["id"] == cs_id), None)
        if not cs:
            return False
        for var in cs.get("series", {}).values():
            sp = var.get("spec") or {}
            nm = str(var.get("name", "")).upper()
            code = str(sp.get("code", var.get("key", "")))
            if sp.get("src") in ("curve", "cnbcy"):
                return True
            if nm.startswith("RATES"):
                return True
            if code.startswith("stir:"):
                return True
            if re.match(r"^\^(US|UK|AU|ECB|JPY|FVX|TNX|TYX)", code):
                return True
        return False
    except Exception:
        return False


def _ticker_is_rates(tkr: str) -> bool:
    """Custom series inherit rates-ness from their components."""
    return tkr.startswith("custom:") and _custom_is_rates(tkr[7:])


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_custom_full(cs_id: str) -> pd.DataFrame:
    """Custom series with FULL history via the analyzer's modern evaluator
    (handles new spec-format entries + charting-registry sources). The old
    watchlist._fetch_custom_df_cached legacy path returned ~1y only — US
    2s10s came back 245 rows and the positioning tab said 'no usable
    history' (Rajat 2026-08-28)."""
    try:
        import analyzer
        s = analyzer._fetch_custom_series(cs_id, "max")
        if s is None or s.empty:
            return pd.DataFrame()
        s = s.dropna()
        s.index = pd.to_datetime(s.index).normalize()
        return pd.DataFrame({"Close": s.astype(float)})
    except Exception:
        return pd.DataFrame()


# ── CTA asset universe ─────────────────────────────────────────────────────────
CTA_ASSETS = [
    {"name": "S&P 500",        "ticker": "^GSPC",     "class": "Equity"},
    {"name": "NASDAQ",         "ticker": "^IXIC",     "class": "Equity"},
    {"name": "Dow Jones",      "ticker": "^DJI",      "class": "Equity"},
    {"name": "Russell 2000",   "ticker": "^RUT",      "class": "Equity"},
    {"name": "FTSE 100",       "ticker": "^FTSE",     "class": "Equity"},
    {"name": "DAX",            "ticker": "^GDAXI",    "class": "Equity"},
    {"name": "Euro Stoxx 50",  "ticker": "^STOXX50E", "class": "Equity"},
    {"name": "Nikkei 225",     "ticker": "^N225",     "class": "Equity"},
    {"name": "NIFTY 50",       "ticker": "^NSEI",     "class": "Equity"},
    {"name": "KOSPI",          "ticker": "^KS11",     "class": "Equity"},
    {"name": "Taiwan",         "ticker": "^TWII",     "class": "Equity"},
    {"name": "Hang Seng",      "ticker": "^HSI",      "class": "Equity"},
    {"name": "CSI 300",        "ticker": "000300.SS", "class": "Equity"},
    {"name": "EUR/USD",        "ticker": "EURUSD=X",  "class": "FX"},
    {"name": "GBP/USD",        "ticker": "GBPUSD=X",  "class": "FX"},
    {"name": "EUR/GBP",        "ticker": "EURGBP=X",  "class": "FX"},
    {"name": "USD/JPY",        "ticker": "JPY=X",     "class": "FX"},
    {"name": "AUD/USD",        "ticker": "AUDUSD=X",  "class": "FX"},
    {"name": "USD/CHF",        "ticker": "CHF=X",     "class": "FX"},
    {"name": "USD/CAD",        "ticker": "CAD=X",     "class": "FX"},
    {"name": "USD/CNH",        "ticker": "USDCNH=X",  "class": "FX"},
    {"name": "USD/INR",        "ticker": "INR=X",     "class": "FX"},
    {"name": "USD/KRW",        "ticker": "KRW=X",     "class": "FX"},
    {"name": "USD/BRL",        "ticker": "BRL=X",     "class": "FX"},
    {"name": "Gold",           "ticker": "GC=F",      "class": "Commodity"},
    {"name": "WTI Oil",        "ticker": "CL=F",      "class": "Commodity"},
    {"name": "Brent Crude",    "ticker": "BZ=F",      "class": "Commodity"},
    {"name": "Silver",         "ticker": "SI=F",      "class": "Commodity"},
    {"name": "Copper",         "ticker": "HG=F",      "class": "Commodity"},
    {"name": "US 2Y Yield",    "ticker": "^US2YT",    "class": "Rates"},
    {"name": "US 5Y Yield",    "ticker": "^FVX",      "class": "Rates"},
    {"name": "US 10Y Yield",   "ticker": "^TNX",      "class": "Rates"},
    {"name": "US 30Y Yield",   "ticker": "^TYX",      "class": "Rates"},
    {"name": "EUR 2Y Yld",     "ticker": "^ECB2Y",    "class": "Rates"},
    {"name": "EUR 5Y Yld",     "ticker": "^ECB5Y",    "class": "Rates"},
    {"name": "EUR 10Y Yld",    "ticker": "^ECB10Y",   "class": "Rates"},
    {"name": "EUR 30Y Yld",    "ticker": "^ECB30Y",   "class": "Rates"},
    # UK/JP/AU ladders = the same standard curve set as the Charting tab
    # (Rajat 2026-08-28); US/EUR above use deeper direct sources
    {"name": "UK 2Y Yield",    "ticker": "^UK2YT",    "class": "Rates"},
    {"name": "UK 5Y Yield",    "ticker": "^UK5YT",    "class": "Rates"},
    {"name": "UK 10Y Yield",   "ticker": "^UK10YT",   "class": "Rates"},
    {"name": "UK 30Y Yield",   "ticker": "^UK30YT",   "class": "Rates"},
    {"name": "JPY 2Y",         "ticker": "^JPY2Y",    "class": "Rates"},
    {"name": "JPY 5Y",         "ticker": "^JPY5Y",    "class": "Rates"},
    {"name": "JPY 10Y",        "ticker": "^JPY10Y",   "class": "Rates"},
    {"name": "JPY 30Y",        "ticker": "^JPY30Y",   "class": "Rates"},
    {"name": "AU 2Y Yield",    "ticker": "^AU2YT",    "class": "Rates"},
    {"name": "AU 5Y Yield",    "ticker": "^AU5YT",    "class": "Rates"},
    {"name": "AU 10Y Yield",   "ticker": "^AU10YT",   "class": "Rates"},
    {"name": "US 5Y Real",     "ticker": "^US5YR",    "class": "Rates"},
    {"name": "US 10Y Real",    "ticker": "^US10YR",   "class": "Rates"},
    {"name": "US 5Y Breakeven","ticker": "^US5YBE",   "class": "Rates"},
    {"name": "US 10Y Breakeven","ticker": "^US10YBE", "class": "Rates"},
    {"name": "Bitcoin",        "ticker": "BTC-USD",   "class": "Crypto"},
    {"name": "Ethereum",       "ticker": "ETH-USD",   "class": "Crypto"},
]

_PERIOD_YEARS = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "15Y": 15, "20Y": 20}

_SIGNAL_OPTIONS = ["Combined", "TSMOM", "MA Cross", "Donchian", "EWMA"]

# column name and display colour for each named signal
_SIG_META = {
    "TSMOM":    ("tsmom",    "#0EA5E9"),
    "MA Cross": ("ma_cross", "#F59E0B"),
    "Donchian": ("donchian", "#10B981"),
    "EWMA":     ("ewma",     "#A855F7"),
    "Combined": ("combined", "#1E293B"),
}

# ── Point-in-time signal helpers ───────────────────────────────────────────────

def _sign(x) -> int:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _tsmom(close, lb, arithmetic=False):
    # arithmetic=True for rates (yield levels): bp moves, never pct —
    # pct breaks across zero (JGB 2019-21). Others stay %.
    if len(close) < lb + 1:
        return 0
    if arithmetic:
        return _sign(close.iloc[-1] - close.iloc[-lb])
    return _sign(close.iloc[-1] / close.iloc[-lb] - 1)


def _ma_cross(close, fast, slow):
    if len(close) < slow + 10:
        return 0
    ef = close.ewm(span=fast, adjust=False).mean().iloc[-1]
    es = close.ewm(span=slow, adjust=False).mean().iloc[-1]
    if not (math.isfinite(ef) and math.isfinite(es)):
        return 0
    return 1 if ef > es else -1


def _donchian(close, n):
    if len(close) < n + 1:
        return 0
    w = close.iloc[-n-1:-1]
    px = close.iloc[-1]
    if px > w.max():  return 1
    if px < w.min():  return -1
    return 0


def _ewma_signal(close, span, vol_days=21, arithmetic=False):
    if len(close) < max(span, vol_days) + 5:
        return 0
    rets = (close.diff() if arithmetic else close.pct_change()).dropna()
    if len(rets) < vol_days:
        return 0
    ev  = rets.ewm(span=span, adjust=False).mean().iloc[-1]
    rv  = rets.iloc[-vol_days:].std()
    if rv == 0 or not math.isfinite(rv):
        return 0
    return _sign(ev / rv)


def _rvol_ann(close, vol_days=21, arithmetic=False):
    """Annualised vol — % terms for prices, yield POINTS for rates
    (arithmetic; display ×100 = bp/yr)."""
    if len(close) < vol_days + 2:
        return float("nan")
    rets = (close.diff() if arithmetic else close.pct_change()).dropna()
    return float(rets.iloc[-vol_days:].std() * math.sqrt(252))


# ── Point-in-time signal table ─────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _compute_signals(tickers, tsmom_days, ma_fast, ma_slow, donchian_n, ewma_span,
                     rates_tickers: tuple = ()):
    rows = []
    for tkr in tickers:
        if tkr.startswith("custom:"):
            df = _fetch_custom_full(tkr[7:])
        else:
            df = _raw_daily(tkr)
        if df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        if len(close) < 30:
            continue

        # rates: bp math, never pct — the convention follows the PRODUCT
        # (custom series classified from their components, e.g. US 2s10s)
        _rt = tkr in rates_tickers or _ticker_is_rates(tkr)
        vol = _rvol_ann(close, arithmetic=_rt)

        # ── Continuous raw values for cross-sectional scoring ─────────────────
        # TSMOM: vol-normalised trailing return (bp-based for rates)
        if len(close) >= tsmom_days + 1 and math.isfinite(vol) and vol > 0:
            ret = (float(close.iloc[-1] - close.iloc[-tsmom_days]) if _rt
                   else float(close.iloc[-1] / close.iloc[-tsmom_days] - 1))
            tsmom_raw = ret / vol
        else:
            tsmom_raw = 0.0

        # MA Cross: normalised EMA spread
        if len(close) >= ma_slow + 10:
            ef = float(close.ewm(span=ma_fast, adjust=False).mean().iloc[-1])
            es = float(close.ewm(span=ma_slow, adjust=False).mean().iloc[-1])
            px = float(close.iloc[-1])
            ma_raw = (ef - es) / px if px > 0 else 0.0
        else:
            ma_raw = 0.0

        # Donchian: price position within channel (0=bottom, 1=top)
        if len(close) >= donchian_n + 1:
            hi = float(close.iloc[-donchian_n-1:-1].max())
            lo = float(close.iloc[-donchian_n-1:-1].min())
            px = float(close.iloc[-1])
            don_raw = float(np.clip((px - lo) / (hi - lo) if hi > lo else 0.5, 0.0, 1.0))
        else:
            don_raw = 0.5

        # EWMA: vol-normalised EWMA of returns (bp-based for rates)
        if len(close) >= max(ewma_span, 21) + 5:
            rets = (close.diff() if _rt else close.pct_change()).dropna()
            ev  = float(rets.ewm(span=ewma_span, adjust=False).mean().iloc[-1])
            rv  = float(rets.iloc[-21:].std())
            ewma_raw = ev / rv if rv > 0 and math.isfinite(rv) else 0.0
        else:
            ewma_raw = 0.0

        try:
            _rsi = ta_lib.momentum.RSIIndicator(close, window=14).rsi().dropna()
            rsi14 = float(_rsi.iloc[-1]) if not _rsi.empty else float("nan")
        except Exception:
            rsi14 = float("nan")

        # Multi-speed ensemble positioning z (slow-tilted weights) — table
        # column (Rajat 2026-08-26). z is invariant to the vol-target scalar.
        # _raw_daily's 670d window gives ~200d of position history after the
        # 252d burn-in — slightly shorter trailing sample than the chart's
        # 3y+ fetch, so the two can differ by a few hundredths.
        try:
            _w, _vbs = _ENSEMBLE_WEIGHTS["Slow-tilted (10/20/30/40)"]
            _pos = _ensemble_position(close, 0.10, weights=_w,
                                      vol_by_speed=_vbs,
                                      arithmetic=tkr in rates_tickers)
            _zs = ((_pos - _pos.rolling(252, min_periods=126).mean())
                   / _pos.rolling(252, min_periods=126).std().replace(0, np.nan)
                   ).dropna()
            ens_z = float(_zs.iloc[-1]) if len(_zs) else float("nan")
        except Exception:
            ens_z = float("nan")

        rows.append({
            "ticker":    tkr,
            "tsmom":     _tsmom(close, tsmom_days, arithmetic=_rt),
            "ma_cross":  _ma_cross(close, ma_fast, ma_slow),
            "donchian":  _donchian(close, donchian_n),
            "ewma":      _ewma_signal(close, ewma_span, arithmetic=_rt),
            "vol_ann":   vol,
            "rsi14":     rsi14,
            "tsmom_raw": tsmom_raw,
            "ma_raw":    ma_raw,
            "don_raw":   don_raw,
            "ewma_raw":  ewma_raw,
            "ens_z":     ens_z,
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


# ── Extended price history ──────────────────────────────────────────────────────

# Some tickers (e.g. ^US2YT) only work via Ticker.history(), not yf.download().
# Mirror the three-level fallback used by _raw_daily in watchlist.py.
_YF_PERIOD_FOR_YEARS = {1: "2y", 2: "5y", 3: "5y", 5: "5y", 7: "10y", 10: "10y", 15: "max", 20: "max", 30: "max"}


@st.cache_data(ttl=3600, show_spinner=False)
def _raw_daily_ext(ticker: str, years: int) -> pd.DataFrame:
    """Fetch `years` of daily OHLCV. Mirrors _raw_daily's source routing exactly:
    FRED → ECB API → JGB MOF → yfinance (3-level fallback).
    Restart-proof: read-through daily_store disk layer (Rajat 2026-08-28 —
    "store it in a local database so a restart doesn't recompute").
    """
    import daily_store
    _dk = f"rawext|{ticker}|{years}"
    _cached = daily_store.get_df(_dk)
    if _cached is not None:
        return _cached
    df = _raw_daily_ext_fetch(ticker, years)
    if not df.empty:
        daily_store.put_df(_dk, df)
    return df


def _raw_daily_ext_fetch(ticker: str, years: int) -> pd.DataFrame:
    start_str = (date.today() - timedelta(days=years * 365 + 60)).strftime("%Y-%m-%d")
    cutoff    = pd.Timestamp(start_str)
    period    = _YF_PERIOD_FOR_YEARS.get(years, "10y")

    # ── Custom series ────────────────────────────────────────────────────────
    if ticker.startswith("custom:"):
        return _fetch_custom_full(ticker[7:])

    # ── Sovereign-curve yields (Rajat 2026-08-28): UK gilts via BoE spot
    # curve (+recent splice), JGBs via the MOF curve — charting's curve
    # fetchers; no yfinance/FRED daily source exists for these
    _curve_route = {
        "^UK2YT": ("UK", "2Y"), "^UK5YT": ("UK", "5Y"),
        "^UK10YT": ("UK", "10Y"), "^UK30YT": ("UK", "30Y"),
        "^JPY2Y": ("JP", "2Y"), "^JPY5Y": ("JP", "5Y"),
        "^JPY30Y": ("JP", "30Y"),      # ^JPY10Y stays on JGB_MAP below
        "^AU2YT": ("AU", "2Y"), "^AU5YT": ("AU", "5Y"),
        "^AU10YT": ("AU", "10Y"),      # RBA F2 curve has no 30Y
    }
    if ticker in _curve_route:
        try:
            import charting as _ch
            cty, mat = _curve_route[ticker]
            s = _ch._curve_series(cty, mat)
            if cty == "UK":
                s = _ch._uk_recent_splice(s, mat)
            s = s.dropna()
            s.index = pd.DatetimeIndex(s.index).normalize()
            df = pd.DataFrame({"Close": s.astype(float)})
            return df[df.index >= cutoff].copy()
        except Exception:
            return pd.DataFrame()

    # ── Special non-yfinance sources (same as watchlist._raw_daily) ──────────
    if ticker in FRED_MAP:
        return _fetch_fred_df(FRED_MAP[ticker], start=start_str)
    if ticker in JGB_MAP:
        return _fetch_jgb_df(JGB_MAP[ticker], start=start_str)
    if ticker in ECB_MAP:
        return _fetch_ecb_df(ECB_MAP[ticker], start=start_str)
    if ticker in ALPHAVANTAGE_FX_MAP:
        from_sym, to_sym = ALPHAVANTAGE_FX_MAP[ticker]
        return _fetch_alphavantage_fx(from_sym, to_sym, start=start_str)

    # ── yfinance — four-strategy fallback ────────────────────────────────────
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        df = df[df["Close"].notna()].copy()
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        df.index = idx.normalize()
        df = df[~df.index.duplicated(keep="last")]
        return df[df.index >= cutoff].copy()

    def _covers(df: pd.DataFrame) -> bool:
        # Accept only if data actually reaches back close to the requested cutoff.
        # Some FX tickers silently return less history than requested.
        return not df.empty and df.index[0] <= cutoff + pd.Timedelta(days=90)

    try:
        df = _clean(yf.download(ticker, start=start_str, interval="1d",
                                 auto_adjust=True, progress=False, multi_level_index=False))
        if _covers(df):
            return df
    except Exception:
        pass

    try:
        df = _clean(yf.download(ticker, period=period, interval="1d",
                                 auto_adjust=True, progress=False, multi_level_index=False))
        if _covers(df):
            return df
    except Exception:
        pass

    try:
        df = _clean(yf.Ticker(ticker).history(period=period, interval="1d",
                                               auto_adjust=True))
        if _covers(df):
            return df
    except Exception:
        pass

    # Final fallback: pull maximum available history and filter to cutoff
    try:
        return _clean(yf.Ticker(ticker).history(period="max", interval="1d",
                                                 auto_adjust=True))
    except Exception:
        return pd.DataFrame()


# ── Vectorised historical signal computation ────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _hist_signals(ticker: str, tsmom_days: int, ma_fast: int, ma_slow: int,
                  donchian_n: int, ewma_span: int, years: int = 1,
                  rates: bool = False) -> pd.DataFrame:
    """Vectorized signals across full price history.

    Returns: close, ema_fast, ema_slow, don_high, don_low,
             tsmom, ma_cross, donchian, ewma, combined, rvol_ann
    Risk allocation is NOT stored here — computed live in the chart
    so it always reflects the current vol-target slider.
    """
    raw = _raw_daily_ext(ticker, years)
    if raw.empty or "Close" not in raw.columns:
        return pd.DataFrame()
    close = raw["Close"].dropna()
    if len(close) < 30:
        return pd.DataFrame()

    vol_days = 21

    # rates=True: yield levels → bp math throughout (pct breaks across zero).
    # Custom series inherit rates-ness from their components (2s10s etc.).
    rates = rates or _ticker_is_rates(ticker)
    tsmom_s = ((close.diff(tsmom_days) if rates else close.pct_change(tsmom_days))
               .map(lambda x: _sign(x) if pd.notna(x) and math.isfinite(x) else 0))

    ema_f = close.ewm(span=ma_fast,  adjust=False).mean()
    ema_s = close.ewm(span=ma_slow, adjust=False).mean()
    ma_s  = (ema_f - ema_s).map(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    don_hi = close.shift(1).rolling(donchian_n).max()
    don_lo = close.shift(1).rolling(donchian_n).min()
    don_s  = pd.Series(
        np.where(close > don_hi, 1, np.where(close < don_lo, -1, 0)),
        index=close.index,
    )

    rets     = close.diff() if rates else close.pct_change()
    ewma_ret = rets.ewm(span=ewma_span, adjust=False).mean()
    rvol     = rets.rolling(vol_days).std()
    ewma_s   = (ewma_ret / rvol.replace(0, np.nan)).map(
        lambda x: _sign(x) if pd.notna(x) and math.isfinite(x) else 0)

    combined = (tsmom_s + ma_s + don_s + ewma_s) / 4.0
    rvol_ann  = rvol * math.sqrt(252)

    rsi14 = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
    rsi30 = ta_lib.momentum.RSIIndicator(close, window=30).rsi()

    # Norm.Signal: historical time series of Norm.Score
    # combined maps -1→0, 0→50, +1→100; flip below 50 to abs-conviction scale
    mom_100_ts  = (combined + 1) / 2 * 100
    adj_mom_ts  = mom_100_ts.where(mom_100_ts >= 50, 100 - mom_100_ts)
    adj_rsi_ts  = rsi14.where(rsi14 >= 50, 100 - rsi14)
    norm_signal = adj_mom_ts - adj_rsi_ts  # range -50 to +50

    result = pd.DataFrame({
        "close":       close,
        "ema_fast":    ema_f,
        "ema_slow":    ema_s,
        "don_high":    don_hi,
        "don_low":     don_lo,
        "tsmom":       tsmom_s.astype(float),
        "ma_cross":    ma_s.astype(float),
        "donchian":    don_s.astype(float),
        "ewma":        ewma_s.astype(float),
        "combined":    combined,
        "rvol_ann":    rvol_ann,
        "rsi14":       rsi14,
        "rsi30":       rsi30,
        "norm_signal": norm_signal,
    })

    min_valid = max(tsmom_days, ma_slow + 10, donchian_n, ewma_span + vol_days + 5)
    return result.iloc[min_valid:].dropna(subset=["close"]).copy()


# ── Detail chart ────────────────────────────────────────────────────────────────

def _plot_asset_detail(name: str, hist: pd.DataFrame, ma_fast: int, ma_slow: int,
                       don_n: int, signal_mode: str = "Combined",
                       vol_tgt: float = 0.10) -> go.Figure:
    """6-row chart: price | norm.signal | norm.deviation | chosen signal | RSI | risk allocation."""

    sig_col, sig_color = _SIG_META[signal_mode]
    sig_series = hist[sig_col].fillna(0)

    rvol_ann = hist["rvol_ann"].replace(0, np.nan)
    risk = sig_series * (vol_tgt / rvol_ann) * 100

    sig_title = (
        "Combined Signal  (±1 = fully long / short)"
        if signal_mode == "Combined"
        else f"{signal_mode} Signal  (+1 = Long · 0 = Neutral · −1 = Short)"
    )

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]],
        row_heights=[0.44, 0.10, 0.10, 0.14, 0.11, 0.11],
        vertical_spacing=0.13,
        subplot_titles=[f"{name} — Price",
                        "Norm.Signal  (adj momentum − adj RSI, range −50 → +50)",
                        "Norm.Signal − Period Avg",
                        sig_title, "RSI", "Risk Allocation  (%)"],
    )

    # ── Row 1: price + EMA lines + Donchian channel ───────────────────────────
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["close"],
        name="Price", line=dict(color="#1E40AF", width=2.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["ema_fast"],
        name=f"EMA {ma_fast}", line=dict(color="#F59E0B", width=1.2, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["ema_slow"],
        name=f"EMA {ma_slow}", line=dict(color="#A855F7", width=1.2, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["don_high"],
        name=f"Don({don_n}) Hi",
        line=dict(color="rgba(100,116,139,0.45)", width=1),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["don_low"],
        name=f"Don({don_n}) Lo",
        line=dict(color="rgba(100,116,139,0.45)", width=1),
        fill="tonexty", fillcolor="rgba(100,116,139,0.06)",
    ), row=1, col=1)

    # Combined signal overlaid on price chart (right secondary axis, black line)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["combined"].fillna(0),
        name="Combined (overlay)", mode="lines",
        line=dict(color="#000000", width=1.8),
        opacity=0.55,
    ), row=1, col=1, secondary_y=True)

    # ── Row 2: Norm.Signal ────────────────────────────────────────────────────
    if "norm_signal" in hist.columns:
        ns = hist["norm_signal"].fillna(0)
        fig.add_trace(go.Scatter(
            x=hist.index, y=ns.clip(lower=0),
            name="Norm+", mode="lines",
            line=dict(color="#059669", width=0),
            fill="tozeroy", fillcolor="rgba(5,150,105,0.25)",
            showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=ns.clip(upper=0),
            name="Norm−", mode="lines",
            line=dict(color="#DC2626", width=0),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.25)",
            showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=ns,
            name="Norm.Signal", line=dict(color="#6366F1", width=1.6),
        ), row=2, col=1)

    # ── Row 3: Norm.Signal − period mean ─────────────────────────────────────
    if "norm_signal" in hist.columns:
        ns_avg = hist["norm_signal"].mean()
        nd = hist["norm_signal"].fillna(0) - ns_avg
        fig.add_trace(go.Scatter(
            x=hist.index, y=nd.clip(lower=0),
            name="NormDev+", mode="lines",
            line=dict(color="#059669", width=0),
            fill="tozeroy", fillcolor="rgba(5,150,105,0.25)",
            showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=nd.clip(upper=0),
            name="NormDev−", mode="lines",
            line=dict(color="#DC2626", width=0),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.25)",
            showlegend=False,
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=hist.index, y=nd,
            name="Norm.Dev", line=dict(color="#8B5CF6", width=1.6),
        ), row=3, col=1)

    # ── Row 4: chosen signal fill + line ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=hist.index, y=sig_series.clip(lower=0),
        name="Long", mode="lines",
        line=dict(color="#059669", width=0),
        fill="tozeroy", fillcolor="rgba(5,150,105,0.30)",
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=sig_series.clip(upper=0),
        name="Short", mode="lines",
        line=dict(color="#DC2626", width=0),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.30)",
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=hist.index, y=sig_series,
        name=signal_mode, line=dict(color=sig_color, width=1.8),
    ), row=4, col=1)

    if signal_mode == "Combined":
        for label, (col_name, color) in {
            k: v for k, v in _SIG_META.items() if k != "Combined"
        }.items():
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist[col_name],
                name=label, mode="lines",
                line=dict(color=color, width=0.9, dash="dot"),
                opacity=0.65,
                visible="legendonly",
            ), row=4, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["combined"],
            name="Combined (ref)", mode="lines",
            line=dict(color="#94A3B8", width=1, dash="dot"),
            opacity=0.7,
            visible="legendonly",
        ), row=4, col=1)

    # ── Row 5: RSI ────────────────────────────────────────────────────────────
    x_ends = [hist.index[0], hist.index[-1]]
    for lvl, col in [(70, "rgba(220,38,38,0.30)"), (50, "rgba(148,163,184,0.35)"), (30, "rgba(5,150,105,0.30)")]:
        fig.add_trace(go.Scatter(
            x=x_ends, y=[lvl, lvl], mode="lines",
            line=dict(color=col, width=1, dash="dot"),
            showlegend=False,
        ), row=5, col=1)
    if "rsi30" in hist.columns:
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["rsi30"],
            name="RSI 30", mode="lines",
            line=dict(color="#F59E0B", width=1.2, dash="dot"),
            opacity=0.8,
        ), row=5, col=1)
    if "rsi14" in hist.columns:
        fig.add_trace(go.Scatter(
            x=hist.index, y=hist["rsi14"],
            name="RSI 14", mode="lines",
            line=dict(color="#0EA5E9", width=1.6),
        ), row=5, col=1)

    # ── Row 6: risk allocation bars ───────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=hist.index, y=risk.fillna(0),
        name="Risk Alloc %",
        marker_color=["#059669" if v >= 0 else "#DC2626" for v in risk.fillna(0)],
        showlegend=False,
    ), row=6, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=1160,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFBFD",
        font=dict(color="#1A202C", family="Inter, Segoe UI, sans-serif", size=11),
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(
            orientation="h", y=1.05, x=1,
            xanchor="right", yanchor="bottom",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0", borderwidth=1,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0",
                        font=dict(color="#1A202C", size=11)),
    )

    _grid = dict(gridcolor="#E8EDF5", zeroline=False, linecolor="#E2E8F0")
    for r in [1, 2, 3, 4, 5, 6]:
        fig.update_xaxes(**_grid, row=r, col=1)
        fig.update_yaxes(**_grid, row=r, col=1)

    fig.update_yaxes(zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1.2,
                     range=[-52, 52], row=2, col=1)
    fig.update_yaxes(zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1.2,
                     row=3, col=1)
    fig.update_yaxes(zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1.2,
                     range=[-1.15, 1.15], row=4, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=5, col=1)
    fig.update_yaxes(zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1.2,
                     row=6, col=1)
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.04, bgcolor="#F1F5F9",
                         bordercolor="#E2E8F0", borderwidth=1),
        row=1, col=1,
    )

    # ── Override subplot domains: wide gap row 1-2 (slider), tight rows 2-6 ──
    _rp    = [0.44, 0.10, 0.10, 0.14, 0.11, 0.11]   # row proportions (sum=1.0)
    _g12   = 0.10                                      # row 1-2 gap (slider clearance)
    _gi    = 0.03                                      # inner gap rows 2-6
    _avail = 1.0 - _g12 - 4 * _gi                    # available height for plots
    _h     = [r * _avail for r in _rp]
    _bots  = [0.0] * 6;  _tops = [0.0] * 6
    _tops[5] = round(_h[5], 4);            _bots[5] = 0.0
    _bots[4] = round(_tops[5] + _gi, 4);  _tops[4] = round(_bots[4] + _h[4], 4)
    _bots[3] = round(_tops[4] + _gi, 4);  _tops[3] = round(_bots[3] + _h[3], 4)
    _bots[2] = round(_tops[3] + _gi, 4);  _tops[2] = round(_bots[2] + _h[2], 4)
    _bots[1] = round(_tops[2] + _gi, 4);  _tops[1] = round(_bots[1] + _h[1], 4)
    _bots[0] = round(_tops[1] + _g12, 4); _tops[0] = 1.0
    fig.update_layout(
        yaxis =dict(domain=[_bots[0], _tops[0]]),
        # yaxis2 = secondary axis for row 1 (auto-created by secondary_y=True)
        yaxis3=dict(domain=[_bots[1], _tops[1]]),   # row 2 (was yaxis2)
        yaxis4=dict(domain=[_bots[2], _tops[2]]),   # row 3
        yaxis5=dict(domain=[_bots[3], _tops[3]]),   # row 4
        yaxis6=dict(domain=[_bots[4], _tops[4]]),   # row 5
        yaxis7=dict(domain=[_bots[5], _tops[5]]),   # row 6
    )
    # Style the secondary y-axis: range ±1, no ticks, no grid, just a zero line
    fig.update_yaxes(range=[-1.5, 1.5], showticklabels=False, showgrid=False,
                     zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1,
                     row=1, col=1, secondary_y=True)
    # Reposition subplot title annotations to match new row tops
    for _i in range(min(6, len(fig.layout.annotations))):
        fig.layout.annotations[_i].y = _tops[_i]

    return fig


# ── Formatting helpers ──────────────────────────────────────────────────────────

def _sig_cell(s):
    if s == 1:  return '<span style="color:#059669;font-weight:600">Long</span>'
    if s == -1: return '<span style="color:#DC2626;font-weight:600">Short</span>'
    return '<span style="color:#94A3B8">—</span>'

def _risk_cell(risk):
    if not math.isfinite(risk) or risk == 0:
        return '<span style="color:#94A3B8">—</span>'
    col = "#059669" if risk > 0 else "#DC2626"
    return f'<span style="color:{col};font-weight:600">{risk:+.1f}%</span>'

def _vol_fmt(v, rates=False):
    if not math.isfinite(v):
        return "—"
    # rates vol is in yield POINTS/yr (bp math) → display as bp; others %/yr
    return f"{v*100:.0f}bp" if rates else f"{v*100:.1f}%"

def _rsi_cell(rsi):
    if not math.isfinite(rsi):
        return '<span style="color:#94A3B8">—</span>'
    if rsi >= 70:   col = "#DC2626"
    elif rsi >= 60: col = "#F87171"
    elif rsi <= 30: col = "#059669"
    elif rsi <= 40: col = "#10B981"
    else:           col = "#94A3B8"
    return f'<span style="color:{col};font-weight:600">{rsi:.1f}</span>'

def _ensz_cell(z):
    """Multi-speed ensemble positioning z (slow-tilted): grey when normal,
    amber ≥1σ, red ≥2σ — |z| is the stretch, sign is the trend direction."""
    if z is None or not math.isfinite(z):
        return '<span style="color:#CBD5E1">—</span>'
    if abs(z) >= 2:
        c, w = "#DC2626", 700
    elif abs(z) >= 1:
        c, w = "#B45309", 700
    else:
        c, w = "#64748B", 500
    return f'<span style="color:{c};font-weight:{w};font-size:11px">{z:+.2f}</span>'


def _norm_score_cell(ns):
    if not math.isfinite(ns):
        return '<span style="color:#94A3B8">—</span>'
    # range -50 to +50; 0 = both equally neutral/stretched
    if ns >= 25:   col = "#059669"
    elif ns >= 10: col = "#10B981"
    elif ns >= -10: col = "#94A3B8"
    elif ns >= -25: col = "#F87171"
    else:           col = "#DC2626"
    return f'<span style="color:{col};font-weight:600">{ns:+.0f}</span>'

def _signal_score_cell(score_100, direction):
    """1-100 cross-sectional rank (bold top) + Long/Short/— sub-label."""
    if score_100 >= 75:   sc = "#059669"
    elif score_100 >= 55: sc = "#10B981"
    elif score_100 >= 45: sc = "#94A3B8"
    elif score_100 >= 25: sc = "#F87171"
    else:                 sc = "#DC2626"
    if direction == 1:    dl = '<span style="color:#059669;font-size:9px">Long</span>'
    elif direction == -1: dl = '<span style="color:#DC2626;font-size:9px">Short</span>'
    else:                 dl = '<span style="color:#94A3B8;font-size:9px">—</span>'
    return (f'<div style="line-height:1.3">'
            f'<span style="color:{sc};font-weight:700;font-size:12px">{score_100}</span>'
            f'<br>{dl}</div>')

def _total_score_cell(total_400, old_sum):
    """Combined score normalised to 1-100 (bold top) + old 4-signal direction label (sub-text)."""
    score_100 = round(total_400 / 4)
    if score_100 >= 75:   tc = "#059669"
    elif score_100 >= 62: tc = "#10B981"
    elif score_100 >= 38: tc = "#94A3B8"
    elif score_100 >= 25: tc = "#F87171"
    else:                 tc = "#DC2626"
    _lbl = {4:"4/4 Long",3:"3/4 Long",2:"2/4 Mixed",1:"1/4 Mixed",
            0:"Neutral",-1:"1/4 Mixed",-2:"2/4 Mixed",-3:"3/4 Short",-4:"4/4 Short"}
    old_txt = _lbl.get(old_sum, "Mixed")
    oc = ("#059669" if old_sum > 2 else "#10B981" if old_sum > 0 else
          "#DC2626" if old_sum < -2 else "#F87171" if old_sum < 0 else "#94A3B8")
    return (f'<div style="line-height:1.3">'
            f'<span style="color:{tc};font-weight:700;font-size:12px">{score_100}/100</span>'
            f'<br><span style="color:{oc};font-size:9px">{old_txt}</span></div>')


# ── Main render ─────────────────────────────────────────────────────────────────

# ── Positioning z-score chart (Rajat 2026-08-26, styled after Citadel Sec's
# "CTA Simulation Implies Long End Positioning is Stretched") ────────────────
_ZCOLORS = ["#1E3A8A", "#2563EB", "#60A5FA", "#0E7490", "#94A3B8", "#334155",
            "#7C3AED", "#B45309"]


_ENSEMBLE_WEIGHTS = {
    # label: (per-speed weights for (21, 63, 126, 252)d, vol_by_speed).
    # Slow-tilted mirrors where trend-fund AUM actually sits (3-12m signals
    # dominate; fast = small sleeve) — Rajat 2026-08-26: default. X-slow
    # (Rajat 2026-08-28): only the 6m/12m models, each sized off its OWN
    # horizon's realised vol (not the common 21d) — sizing as slow as the
    # signals. Sums needn't be 1 (normalised).
    "Slow-tilted (10/20/30/40)": ((0.10, 0.20, 0.30, 0.40), False),
    # same AUM weights but each speed sized off its OWN horizon's realised
    # vol (Rajat 2026-08-28) — slow sleeves shrug off short vol spikes
    "Slow-tilted vol-wtd": ((0.10, 0.20, 0.30, 0.40), True),
    "Equal": ((0.25, 0.25, 0.25, 0.25), False),
    "Fast-tilted (40/30/20/10)": ((0.40, 0.30, 0.20, 0.10), False),
    "X-slow (126/252) vol-wtd": ((0.0, 0.0, 0.5, 0.5), True),
}


def _ensemble_position(close: pd.Series, vol_tgt: float,
                       speeds=(21, 63, 126, 252),
                       weights=None, vol_by_speed: bool = False,
                       sized: bool = True,
                       arithmetic: bool = False) -> pd.Series:
    """Multi-speed CTA positioning proxy (Rajat 2026-08-26, after Citadel's
    'collection of trend-following frameworks with varying adjustment
    speeds'): at each speed L, a TSMOM sign and a vol-normalised EWMA sign,
    each vol-scaled into a position (× vol_tgt/realised vol); weighted sum
    across speeds (weights ~ industry AUM by default), normalised to a
    per-unit-weight average. Continuous-ish level series that can make fresh
    extremes even when the slow signal is saturated — unlike the bounded
    4-signal `combined`."""
    if weights is None:
        weights = (1.0,) * len(speeds)
    wsum = float(sum(weights)) or 1.0
    # arithmetic=True (rates: yield LEVELS): use level diffs — pct_change on
    # a series that crosses zero is garbage (JGB 10y 2019-21 sat at −0.3..
    # +0.17 with 36 zero crossings → infinite "returns"; Rajat 2026-08-28)
    rets = close.diff() if arithmetic else close.pct_change()
    rvol = rets.rolling(21).std().replace(0, np.nan)   # signal norm stays 21d
    scale_21 = vol_tgt / (rvol * math.sqrt(252))
    pos = pd.Series(0.0, index=close.index)
    for L, w in zip(speeds, weights):
        if not w:
            continue
        if not sized:                      # signal-only: direction crowding,
            scale = 1.0                    # no vol-targeting size channel
        elif vol_by_speed:                 # size off the horizon's own vol
            scale = vol_tgt / (rets.rolling(L).std()
                               * math.sqrt(252)).replace(0, np.nan)
        else:
            scale = scale_21
        # CONTINUOUS responses (Rajat 2026-08-28 — hard sign() made the
        # 2-speed X-slow book jump 25% per flip at trend inflections, e.g.
        # US30y Jun23-Jun24 with 9 zero-crossings of the 126d return):
        # trend t-stat clipped to ±2 and scaled to ±1, so positions build
        # and fade through zero instead of snapping.
        trail = (close.diff(L) if arithmetic else close.pct_change(L))
        ts = (trail / (rvol * math.sqrt(L))).clip(-2, 2) / 2.0
        ew = (rets.ewm(span=L, adjust=False).mean() / rvol
              * math.sqrt(L / 2.0)).clip(-2, 2) / 2.0
        pos = pos + w * (ts.fillna(0) + ew.fillna(0)) / 2.0 * scale
    return pos / wsum


def _plot_positioning_z(series_by_name: dict, zwin_lbl: str,
                        rsi_by_name: dict | None = None,
                        overlay_by_name: dict | None = None) -> go.Figure:
    """Multi-asset overlay of positioning z-scores (dotted ±1/±2σ guides,
    zero line, right-edge last-value labels, most stretched reading in red)
    with a 30d-RSI panel below. Rajat 2026-08-28 layout: the z panel is the
    MAIN chart (big), RSI second; flows render as separate charts below.
    overlay_by_name: companion series per asset drawn dashed/dotted in the
    same colour (signal-only z — the gap to the main line is vol-sizing)."""
    has_rsi = bool(rsi_by_name)
    # single-asset styling — position z THICK DARK BLUE, signal-only dotted
    # light purple. multi-asset keeps per-asset colours.
    single = len(series_by_name) == 1
    _rows = 2 if has_rsi else 1
    _rsi_row = 2 if has_rsi else None
    _flow_row = None
    fig = make_subplots(
        rows=_rows, cols=1, shared_xaxes=True,
        row_heights=[0.76, 0.24] if has_rsi else None,
        vertical_spacing=0.06,
        subplot_titles=(None, "RSI (30d)") if has_rsi else None)
    ext_name, ext_val = None, 0.0
    for i, (name, s) in enumerate(series_by_name.items()):
        col = "#1E3A8A" if single else _ZCOLORS[i % len(_ZCOLORS)]
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines",
            line=dict(color=col, width=3.6 if single else 1.6),
            hovertemplate=f"{name}: %{{y:.2f}}σ<extra></extra>"),
            row=1, col=1)
        last = float(s.iloc[-1])
        if abs(last) > abs(ext_val):
            ext_name, ext_val = name, last
        fig.add_annotation(x=s.index[-1], y=last, xanchor="left", xshift=4,
                           text=f"{last:+.2f}", showarrow=False,
                           font=dict(size=10, color=col), row=1, col=1)
        ov = (overlay_by_name or {}).get(name)
        if ov is not None and len(ov):
            fig.add_trace(go.Scatter(
                x=ov.index, y=ov.values, name=f"{name} signal-only",
                mode="lines", showlegend=single,
                line=dict(color="#A78BFA" if single else col,
                          width=2.2 if single else 1.1, dash="dot"),
                opacity=0.95 if single else 0.75,
                hovertemplate=f"{name} signal-only: %{{y:.2f}}σ<extra></extra>"),
                row=1, col=1)
        rs = (rsi_by_name or {}).get(name)
        if rs is not None and len(rs) and _rsi_row:
            fig.add_trace(go.Scatter(
                x=rs.index, y=rs.values, name=f"{name} RSI30", mode="lines",
                line=dict(color=col, width=1.2), showlegend=False,
                hovertemplate=f"{name} RSI30: %{{y:.0f}}<extra></extra>"),
                row=_rsi_row, col=1)
    if ext_name is not None:      # re-annotate the most stretched in red bold
        s = series_by_name[ext_name]
        fig.add_annotation(x=s.index[-1], y=float(s.iloc[-1]), xanchor="left",
                           xshift=4, yshift=12,
                           text=f"<b>{float(s.iloc[-1]):+.2f}</b>",
                           showarrow=False, font=dict(size=11, color="#DC2626"),
                           row=1, col=1)
    for lv in (1.0, -1.0, 2.0, -2.0):
        fig.add_hline(y=lv, line=dict(color="#DC2626", width=1, dash="dot"),
                      opacity=1.0 if abs(lv) < 1.5 else 0.6, row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#94A3B8", width=1), row=1, col=1)
    if has_rsi:
        fig.add_hline(y=50, line=dict(color="#CBD5E1", width=1),
                      row=_rsi_row, col=1)
        for lv in (30, 70):
            fig.add_hline(y=lv, line=dict(color="#94A3B8", width=1, dash="dot"),
                          row=_rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], row=_rsi_row, col=1)
    fig.update_layout(
        height=700 if has_rsi else 560, template="plotly_white",
        margin=dict(l=10, r=60, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11)),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=f"z-score vs {zwin_lbl} history", row=1, col=1)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEF1F6", zeroline=False)
    return fig


@st.fragment
def render_cta():
    st.markdown("#### CTA Signals &nbsp;·&nbsp; Systematic Trend-Following")
    st.caption(
        "Signals: **TSMOM** (trailing return momentum) · **MA Cross** (EMA fast/slow) · "
        "**Breakout** (Donchian channel) · **EWMA** (vol-normalised EWMA of returns).  "
        "Risk allocation = signal × (target vol / realised vol). "
        "Rates signals are on yield direction (Long = yields rising = bearish bonds)."
    )

    # ── Parameter controls ────────────────────────────────────────────────────
    with st.expander("⚙  Signal Parameters", expanded=False):
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        tsmom_lb = c1.selectbox("TSMOM lookback", [21, 63, 126, 252], index=2,
                                 format_func=lambda x: {21:"1m",63:"3m",126:"6m",252:"12m"}[x])
        ma_fast  = c2.selectbox("MA fast (days)", [10, 20, 50], index=1)
        ma_slow  = c3.selectbox("MA slow (days)", [40, 100, 200], index=2)
        don_n    = c4.selectbox("Breakout window", [20, 55, 100], index=1)
        ewma_sp  = c5.selectbox("EWMA span", [21, 63, 126], index=1,
                                 format_func=lambda x: {21:"1m",63:"3m",126:"6m"}[x])
        vol_tgt  = c6.selectbox("Vol target", [0.05, 0.10, 0.15, 0.20], index=1,
                                 format_func=lambda x: f"{x*100:.0f}%")

    # ── Build asset list: static CTA_ASSETS + active custom series ───────────
    cfg        = load_config()
    all_cs     = _load_cs_json()
    cs_name_map = {cs["id"]: cs["name"] for cs in all_cs}
    sel_cs_ids = [cid for cid in cfg.get("custom_series_ids", []) if cid in cs_name_map]
    custom_assets = [
        {"name": cs_name_map[cid], "ticker": f"custom:{cid}", "class": "Custom"}
        for cid in sel_cs_ids
    ]
    all_assets = list(CTA_ASSETS) + custom_assets

    # ── Asset class filter ────────────────────────────────────────────────────
    classes = sorted({a["class"] for a in all_assets})
    sel_classes = st.multiselect("Asset classes", classes, default=classes, key="_cta_classes")
    assets = [a for a in all_assets if a["class"] in sel_classes]
    if not assets:
        st.info("Select at least one asset class.")
        return

    tickers   = tuple(a["ticker"] for a in assets)
    name_map  = {a["ticker"]: a["name"]  for a in assets}
    class_map = {a["ticker"]: a["class"] for a in assets}

    with st.spinner("Computing CTA signals…"):
        sigs = _compute_signals(
            tickers, tsmom_lb, ma_fast, ma_slow, don_n, ewma_sp,
            rates_tickers=tuple(t for t in tickers
                                if class_map.get(t) == "Rates"))

    if sigs.empty:
        st.warning("No signal data available — check data sources.")
        return

    # ── Cross-sectional 1-100 scores ──────────────────────────────────────────
    def _xrank(series):
        n = len(series)
        if n <= 1:
            return pd.Series([50] * n, index=series.index, dtype=int)
        return series.rank(pct=True).mul(99).add(1).round().clip(1, 100).astype(int)

    for _rc, _sc in [("tsmom_raw","tsmom_score"),("ma_raw","ma_score"),("ewma_raw","ewma_score")]:
        sigs[_sc] = _xrank(sigs[_rc]) if _rc in sigs.columns else 50
    if "don_raw" in sigs.columns:
        sigs["don_score"] = (sigs["don_raw"] * 99 + 1).round().clip(1, 100).astype(int)
    else:
        sigs["don_score"] = 50
    sigs["total_score"] = sigs[["tsmom_score","ma_score","don_score","ewma_score"]].sum(axis=1)

    # ── Sort controls ─────────────────────────────────────────────────────────
    _SORT_OPTS = ["Asset", "Mom.Score", "Norm.Score", "Ens.Z", "RSI (14)", "Vol (21d)", "TSMOM", "MA Cross", "Breakout", "EWMA", "Risk Alloc"]
    c_sort, c_dir = st.columns([3, 1])
    with c_sort:
        sort_col = st.selectbox("Sort by", _SORT_OPTS, key="_cta_sort_col",
                                label_visibility="collapsed")
    with c_dir:
        _asc = st.session_state.get("_cta_sort_asc", False)
        if st.button("↑ Asc" if _asc else "↓ Desc", key="_cta_dir_btn", use_container_width=True):
            st.session_state["_cta_sort_asc"] = not _asc
            st.rerun()
    sort_asc: bool = st.session_state.get("_cta_sort_asc", False)

    # ── Signal table ──────────────────────────────────────────────────────────
    th   = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
            "padding:5px 8px;text-align:center;white-space:nowrap;border:none")
    th_l = th.replace("text-align:center","text-align:left")
    th_a = th + ";background:#334155"  # active sort column highlight
    td   = "font-size:11px;padding:4px 8px;border-bottom:1px solid #E2E8F0;text-align:center"
    td_l = td.replace("text-align:center","text-align:left")

    lbl_tsmom = {21:"1m",63:"3m",126:"6m",252:"12m"}[tsmom_lb]
    _col_sort_map = {
        "Asset": "Asset", "Mom.Score": "Mom.Score", "Norm.Score": "Norm.Score",
        "Ens.Z": "Ens.Z",
        "RSI (14)": "RSI (14)", "Vol (21d)": "Vol (21d)",
        f"TSMOM {lbl_tsmom}": "TSMOM",
        f"{ma_fast}/{ma_slow} EMA": "MA Cross",
        f"{don_n}d Breakout": "Breakout",
        f"EWMA {ewma_sp}d": "EWMA",
        "Risk Alloc": "Risk Alloc",
    }

    def _hdr(label, style_base):
        key = _col_sort_map.get(label, label)
        arrow = (" ▲" if sort_asc else " ▼") if key == sort_col else ""
        s = (th_a if key == sort_col else style_base)
        return f'<th style="{s}">{label}{arrow}</th>'

    header = (
        f'<tr>'
        f'{_hdr("Asset", th_l)}'
        f'{_hdr("Class", th)}'
        f'{_hdr("Mom.Score", th)}'
        f'{_hdr("Norm.Score", th)}'
        f'{_hdr("Ens.Z", th)}'
        f'{_hdr("RSI (14)", th)}'
        f'{_hdr("Vol (21d)", th)}'
        f'{_hdr(f"TSMOM {lbl_tsmom}", th)}'
        f'{_hdr(f"{ma_fast}/{ma_slow} EMA", th)}'
        f'{_hdr(f"{don_n}d Breakout", th)}'
        f'{_hdr(f"EWMA {ewma_sp}d", th)}'
        f'{_hdr("Risk Alloc", th)}'
        f'</tr>'
    )

    # ── Collect row data, sort, then render ────────────────────────────────────
    row_data = []
    for tkr in tickers:
        if tkr not in sigs.index:
            continue
        row  = sigs.loc[tkr]
        name = name_map[tkr]; cls = class_map[tkr]
        bg   = CLASS_BG.get(cls, "#F8FAFC"); fg = CLASS_FG.get(cls, "#475569")
        s1, s2, s3, s4 = int(row["tsmom"]), int(row["ma_cross"]), int(row["donchian"]), int(row["ewma"])
        sc1, sc2 = int(row["tsmom_score"]), int(row["ma_score"])
        sc3, sc4 = int(row["don_score"]),   int(row["ewma_score"])
        total_400 = int(row["total_score"])
        old_sum = s1 + s2 + s3 + s4; vol = float(row["vol_ann"])
        rsi14 = float(row["rsi14"]) if "rsi14" in row.index else float("nan")
        risk  = (old_sum/4.0 * vol_tgt/vol * 100
                 if math.isfinite(vol) and vol > 0 and old_sum != 0 else float("nan"))
        # Norm.Score: both on 0-100 scale, flipped to absolute conviction (50-100), then subtracted
        mom_100 = total_400 / 4
        adj_mom = mom_100 if mom_100 >= 50 else (100 - mom_100)
        if math.isfinite(rsi14):
            adj_rsi = rsi14 if rsi14 >= 50 else (100 - rsi14)
            norm_score = float(adj_mom - adj_rsi)
        else:
            norm_score = float("nan")
        badge = (f'<span style="background:{bg};color:{fg};font-size:10px;'
                 f'font-weight:600;padding:2px 6px;border-radius:4px">{cls}</span>')
        _ez = (float(row["ens_z"]) if "ens_z" in row.index else float("nan"))
        row_data.append({
            "name": name, "cls": cls, "badge": badge,
            "total_400": total_400, "old_sum": old_sum,
            "vol": vol, "rsi14": rsi14, "norm_score": norm_score,
            "ens_z": _ez,
            "sc1": sc1, "sc2": sc2, "sc3": sc3, "sc4": sc4,
            "s1": s1, "s2": s2, "s3": s3, "s4": s4, "risk": risk,
        })

    _sort_keys = {
        "Asset":       lambda r: r["name"].lower(),
        "Mom.Score":   lambda r: r["total_400"],
        "Norm.Score":  lambda r: r["norm_score"] if math.isfinite(r["norm_score"]) else -1.0,
        "Ens.Z":       lambda r: abs(r["ens_z"]) if math.isfinite(r["ens_z"]) else -1.0,
        "RSI (14)":    lambda r: r["rsi14"] if math.isfinite(r["rsi14"]) else -1.0,
        "Vol (21d)":  lambda r: r["vol"] if math.isfinite(r["vol"]) else -1.0,
        "TSMOM":      lambda r: r["sc1"],
        "MA Cross":   lambda r: r["sc2"],
        "Breakout":   lambda r: r["sc3"],
        "EWMA":       lambda r: r["sc4"],
        "Risk Alloc": lambda r: r["risk"] if math.isfinite(r["risk"]) else 0.0,
    }
    row_data.sort(key=_sort_keys.get(sort_col, lambda r: r["name"].lower()), reverse=not sort_asc)

    rows_html = []
    for rd in row_data:
        rows_html.append(
            f'<tr style="background:#FFFFFF">'
            f'<td style="{td_l};font-weight:600">{rd["name"]}</td>'
            f'<td style="{td}">{rd["badge"]}</td>'
            f'<td style="{td}">{_total_score_cell(rd["total_400"], rd["old_sum"])}</td>'
            f'<td style="{td}">{_norm_score_cell(rd["norm_score"])}</td>'
            f'<td style="{td}">{_ensz_cell(rd["ens_z"])}</td>'
            f'<td style="{td}">{_rsi_cell(rd["rsi14"])}</td>'
            f'<td style="{td}">{_vol_fmt(rd["vol"], rd["cls"] == "Rates")}</td>'
            f'<td style="{td}">{_signal_score_cell(rd["sc1"], rd["s1"])}</td>'
            f'<td style="{td}">{_signal_score_cell(rd["sc2"], rd["s2"])}</td>'
            f'<td style="{td}">{_signal_score_cell(rd["sc3"], rd["s3"])}</td>'
            f'<td style="{td}">{_signal_score_cell(rd["sc4"], rd["s4"])}</td>'
            f'<td style="{td}">{_risk_cell(rd["risk"])}</td></tr>'
        )

    st.markdown(
        '<div style="overflow-x:auto">'
        '<table style="border-collapse:collapse;width:100%;font-family:monospace">'
        f'<thead>{header}</thead><tbody>{"".join(rows_html)}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )

    all_old = [
        int(sigs.loc[t,"tsmom"]) + int(sigs.loc[t,"ma_cross"]) +
        int(sigs.loc[t,"donchian"]) + int(sigs.loc[t,"ewma"])
        for t in tickers if t in sigs.index
    ]
    if all_old:
        st.caption(
            f"**{sum(s > 0 for s in all_old)}** net long &nbsp;·&nbsp; "
            f"**{sum(s < 0 for s in all_old)}** net short &nbsp;·&nbsp; "
            f"**{sum(s == 0 for s in all_old)}** neutral &nbsp;·&nbsp; "
            f"Score = cross-sectional rank 1–100 per signal, total 4–400 "
            f"(≥300 strongly bullish, ≤100 strongly bearish) &nbsp;·&nbsp; "
            f"Risk alloc = signal × ({vol_tgt*100:.0f}% target vol / realised vol) &nbsp;·&nbsp; "
            f"Ens.Z = multi-speed ensemble positioning z (slow-tilted weights, 1y window; "
            f"amber ≥1σ, red ≥2σ — sortable by |z| to surface stretched trends)"
        )

    # ── Signal history chart ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### Signal History")

    available_names = [a["name"] for a in assets if a["ticker"] in sigs.index]

    # ── Control row: asset | period | signal | Run ────────────────────────────
    c_asset, c_period, c_signal, c_run = st.columns([4, 1, 2, 1])

    with c_asset:
        sel_name = st.selectbox(
            "Asset", ["— choose an asset —"] + available_names,
            key="_cta_detail_sel", label_visibility="collapsed",
        )
    with c_period:
        period_label = st.selectbox(
            "Period", list(_PERIOD_YEARS.keys()),
            index=0, key="_cta_period", label_visibility="collapsed",
        )
    with c_signal:
        signal_mode = st.selectbox(
            "Signal", _SIGNAL_OPTIONS,
            index=0, key="_cta_signal_mode", label_visibility="collapsed",
        )
    with c_run:
        st.write("")
        run_clicked = st.button(
            "▶ Run", type="primary", key="_cta_run_btn",
            disabled=(sel_name == "— choose an asset —"),
            use_container_width=True,
        )

    # ── Data fetch (only on Run) ──────────────────────────────────────────────
    if run_clicked and sel_name != "— choose an asset —":
        sel_tkr = next(a["ticker"] for a in all_assets if a["name"] == sel_name)
        years   = _PERIOD_YEARS[period_label]
        _rt_sel = class_map.get(sel_tkr) == "Rates"
        with st.spinner(f"Loading {period_label} of {sel_name} data…"):
            hist = _hist_signals(sel_tkr, tsmom_lb, ma_fast, ma_slow, don_n, ewma_sp, years,
                                 rates=_rt_sel)
            if hist.empty:
                # Stale empty cache (e.g. from before AlphaVantage routing was added)
                _hist_signals.clear()
                _raw_daily_ext.clear()
                hist = _hist_signals(sel_tkr, tsmom_lb, ma_fast, ma_slow, don_n, ewma_sp, years,
                                     rates=_rt_sel)
        st.session_state["_cta_hist"]  = hist
        st.session_state["_cta_hmeta"] = {
            "name": sel_name, "period": period_label,
            "ma_fast": ma_fast, "ma_slow": ma_slow, "don_n": don_n,
        }

    # ── Chart render (signal_mode and vol_tgt are always live — no Run needed) ─
    hist = st.session_state.get("_cta_hist")
    meta = st.session_state.get("_cta_hmeta", {})

    # Evict stale session-state cached from an older code version
    if hist is not None and not hist.empty and not {"rvol_ann", "rsi14", "norm_signal"}.issubset(hist.columns):
        st.session_state.pop("_cta_hist", None)
        st.session_state.pop("_cta_hmeta", None)
        hist = None
        st.info("Cached signal data is from an older version — click ▶ Run to reload.")

    if hist is not None:
        if hist.empty:
            st.warning("No historical data available for this asset.")
        else:
            st.caption(
                f"**{meta.get('period','')}** · **{meta.get('name','')}** · "
                f"Signal: **{signal_mode}** · "
                f"Risk alloc uses current vol target ({vol_tgt*100:.0f}%). "
                f"Toggle individual signals in the legend."
            )
            fig = _plot_asset_detail(
                meta.get("name", ""), hist,
                meta.get("ma_fast", ma_fast),
                meta.get("ma_slow", ma_slow),
                meta.get("don_n",   don_n),
                signal_mode=signal_mode,
                vol_tgt=vol_tgt,
            )
            st.plotly_chart(fig, use_container_width=True)


# GS Futures-Flow-Monitor naming/colours (Rajat 2026-08-28: "the GS CTA
# template"): Up Big green, Up Small blue, Flat amber, Down Small near-black,
# Down Big red; history = grey line over a shaded band.
_SCN_SIGMAS = ((2, "Up Big", "#059669"), (1, "Up Small", "#2563EB"),
               (0, "Flat Market", "#D97706"), (-1, "Down Small", "#111827"),
               (-2, "Down Big", "#DC2626"))


def _plot_asset_price(px_by_name: dict, cls_by_name: dict,
                      mva_by_name: dict | None = None) -> go.Figure:
    """Underlying asset chart, drawn ABOVE the main z chart when its checkbox
    is on (Rajat 2026-08-28), same span as the z chart. Single asset = raw
    level (yield % for rates); multi = change since span start (% for
    prices, bp for rates) so mixed scales share one axis."""
    single = len(px_by_name) == 1
    fig = go.Figure()
    for i, (name, s) in enumerate(px_by_name.items()):
        rt = cls_by_name.get(name) == "Rates"
        col = "#1E3A8A" if single else _ZCOLORS[i % len(_ZCOLORS)]
        if single:
            y, unit = s, ("%" if rt else "")
            hv = f"{name}: %{{y:.3f}}{unit}<extra></extra>"
        elif rt:
            y = (s - s.iloc[0]) * 100.0
            hv = f"{name}: %{{y:+.0f}}bp<extra></extra>"
        else:
            y = (s / s.iloc[0] - 1) * 100.0
            hv = f"{name}: %{{y:+.1f}}%<extra></extra>"
        fig.add_trace(go.Scatter(
            x=s.index, y=y, name=name, mode="lines",
            line=dict(color=col, width=2.2 if single else 1.6),
            hovertemplate=hv))
        # 3m MVA — red dotted (Rajat 2026-08-28); asset-coloured when
        # several assets share the chart so lines stay attributable.
        # Prefer the full-history MVA passed in (no burn-in gap at span
        # start); transform it the same way as the price line.
        _m = (mva_by_name or {}).get(name)
        if _m is not None and len(_m):
            if single:
                _mva = _m
            elif rt:
                _mva = (_m - s.iloc[0]) * 100.0
            else:
                _mva = (_m / s.iloc[0] - 1) * 100.0
        else:
            _mva = y.rolling(63).mean()
        fig.add_trace(go.Scatter(
            x=_mva.index, y=_mva.values, name=f"{name} 3m MVA", mode="lines",
            line=dict(color="#DC2626" if single else col, width=1.4,
                      dash="dot"),
            opacity=0.9 if single else 0.6, showlegend=single,
            hovertemplate=f"{name} 3m MVA: %{{y:.3f}}<extra></extra>"))
    ttl = ("Underlying — level · 3m MVA dotted" if single
           else "Underlying — change since span start (% px · bp rates)")
    fig.update_layout(
        height=300, template="plotly_white",
        title=dict(text=ttl, font=dict(size=13)),
        margin=dict(l=10, r=20, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11)),
        hovermode="x unified")
    fig.update_yaxes(gridcolor="#EEF1F6", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def _plot_flows_hist(flows_by_name: dict, flow_lbl: str) -> go.Figure:
    """Standalone simulated-flows chart (4th chart in the tab layout, Rajat
    2026-08-28): bars of N-day net position change, % of vol-target book.
    Single asset: green = adding / red = cutting; multi: per-asset colours."""
    single = len(flows_by_name) == 1
    fig = go.Figure()
    for i, (name, fl) in enumerate(flows_by_name.items()):
        col = _ZCOLORS[i % len(_ZCOLORS)]
        bcol = (["#059669" if v >= 0 else "#DC2626" for v in fl.values]
                if single else col)
        fig.add_trace(go.Bar(
            x=fl.index, y=fl.values, name=name,
            marker=dict(color=bcol, line=dict(width=0)),
            opacity=0.85 if single else 0.6, showlegend=not single,
            hovertemplate=f"{name} flow: %{{y:+.1f}}%<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#94A3B8", width=1))
    fig.update_layout(
        height=280, template="plotly_white",
        title=dict(text=f"Simulated flows — {flow_lbl} net position change "
                        "(% of vol-target book)", font=dict(size=13)),
        margin=dict(l=10, r=20, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11)),
        hovermode="x unified", barmode="relative")
    fig.update_yaxes(gridcolor="#EEF1F6", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def _scenario_flows(close: pd.Series, vol_tgt: float, weights, vbs: bool,
                    horizon: int = 21, arithmetic: bool = False) -> dict:
    """Projected CTA flows (Rajat 2026-08-28): extend the price path
    `horizon` bdays under a k·σ TOTAL move (spread evenly, σ = current 21d
    realised vol scaled to the horizon), re-run the ensemble on each
    hypothetical path, difference vs today's position. Returns
    {k: (cum_flow_series_pct_of_book, total_move_pct)}."""
    rets = close.diff() if arithmetic else close.pct_change()
    vol_d = float(rets.rolling(21).std().iloc[-1])
    pos_now = float(_ensemble_position(close, vol_tgt, weights=weights,
                                       vol_by_speed=vbs,
                                       arithmetic=arithmetic).iloc[-1])
    fut_idx = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1),
                             periods=horizon)
    # Paths need NOISE at the current vol around the drift: a deterministic
    # drift path has zero realised vol, which collapses the vol-targeting
    # denominator and saturates the t-stats (first cut showed −1σ = +16% of
    # BUYING — garbage). Average over seeded noise paths → expected flow;
    # fixed rng seed keeps the chart reproducible run-to-run.
    n_paths = 25
    rng = np.random.default_rng(7)
    eps = rng.standard_normal((n_paths, horizon))
    eps = eps - eps.mean(axis=1, keepdims=True)        # pin the total move
    scn = {}
    for k, _lbl, _c in _SCN_SIGMAS:
        move = k * vol_d * math.sqrt(horizon)          # total k·σ over horizon
        acc = None
        for p in range(n_paths):
            if arithmetic:                             # yield levels: additive
                dr = move / horizon + vol_d * eps[p]
                path = float(close.iloc[-1]) + np.cumsum(dr)
            else:
                r_d = (1 + move) ** (1.0 / horizon) - 1
                dr = r_d + vol_d * eps[p]
                path = float(close.iloc[-1]) * np.cumprod(1 + dr)
            ext = pd.concat([close, pd.Series(path, index=fut_idx)])
            pos = _ensemble_position(ext, vol_tgt, weights=weights,
                                     vol_by_speed=vbs,
                                     arithmetic=arithmetic).iloc[-horizon:]
            acc = pos if acc is None else acc + pos
        pos_avg = acc / n_paths
        # move display: % for price series, bp for yield levels
        scn[k] = (pos_avg * 100.0,
                  move * 100.0 if arithmetic else move * 100.0)
    hist = (_ensemble_position(close, vol_tgt, weights=weights,
                               vol_by_speed=vbs, arithmetic=arithmetic)
            * 100.0).dropna().iloc[-85:]
    return {"hist": hist, "scn": scn, "now": pos_now * 100.0,
            "unit": "bp" if arithmetic else "%"}


def _plot_scenario_flows(res: dict, name: str) -> go.Figure:
    """GS Futures-Flow-Monitor style: grey realized net-length history over a
    shaded band, conditional scenario level-paths fanning out from today."""
    fig = go.Figure()
    hist, scn = res["hist"], res["scn"]
    _u = res.get("unit", "%")
    t0 = hist.index[-1]
    fig.add_vrect(x0=hist.index[0], x1=t0, fillcolor="#94A3B8", opacity=0.13,
                  line_width=0)
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, name="Simulated net length",
        mode="lines", line=dict(color="#6B7280", width=2),
        hovertemplate="net length: %{y:+.0f}%<extra></extra>"))
    for k, lbl, col in _SCN_SIGMAS:
        if k not in scn:
            continue
        s, mv = scn[k]
        xs = [t0] + list(s.index)
        ys = [float(hist.iloc[-1])] + list(s.values)
        nm = f"{lbl} ({mv:+.1f}{_u})" if _u == "%" else f"{lbl} ({mv:+.0f}{_u})"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name=nm, mode="lines",
            line=dict(color=col, width=2.2 if abs(k) == 2 else 1.7),
            hovertemplate=nm + ": %{y:+.0f}%<extra></extra>"))
        fig.add_annotation(x=s.index[-1], y=float(s.iloc[-1]), xanchor="left",
                           xshift=4, text=f"{float(s.iloc[-1]):+.0f}",
                           showarrow=False, font=dict(size=10, color=col))
    fig.add_vline(x=t0, line=dict(color="#94A3B8", width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color="#CBD5E1", width=1))
    _ymid = float(min(hist.min(), min(s.min() for s, _ in scn.values())))
    fig.add_annotation(x=hist.index[len(hist) // 2], y=_ymid,
                       text="<i>Simulated<br>Realized Flows</i>",
                       showarrow=False, font=dict(size=11, color="#9CA3AF"))
    _sc0 = scn[0][0] if 0 in scn else list(scn.values())[0][0]
    fig.add_annotation(x=_sc0.index[len(_sc0) // 2], y=_ymid,
                       text="<i>Conditional<br>Expected Flows</i>",
                       showarrow=False, font=dict(size=11, color="#059669"))
    fig.update_layout(
        height=420, template="plotly_white",
        title=dict(text=f"{name} — CTA/Trend 1-Month Conditional Projections "
                        "(% of vol-target book)", font=dict(size=13)),
        margin=dict(l=10, r=55, t=45, b=10),
        legend=dict(x=0.01, y=0.99, yanchor="top", font=dict(size=10.5),
                    bgcolor="rgba(255,255,255,0.7)"),
        yaxis_title="Net length (% of book)",
        hovermode="x unified")
    fig.update_yaxes(gridcolor="#EEF1F6", zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


# ── CTA Positioning — its own Macro sub-tab (Rajat 2026-08-28: "make a
# separate tab, call it CTA Positioning"). Signal params fixed at the tab
# defaults (126/20/200/55/63); the z is scale-invariant to vol target. ──────
@st.fragment
def render_cta_positioning():
    st.markdown("#### CTA Positioning — how stretched is the trend book?")
    st.caption(
        "Simulated trend-follower positioning per asset, z-scored against its "
        "own trailing history — the sell-side 'CTA positioning' chart, on our "
        "own engine. Solid = position z (direction × vol-targeted sizing); "
        "dashed overlay = signal-only z (pure direction crowding); the gap is "
        "the vol-sizing channel. RSI(30) below for the same assets."
    )
    tsmom_lb, ma_fast, ma_slow, don_n, ewma_sp, vol_tgt = 126, 20, 200, 55, 63, 0.10

    cfg = load_config()
    all_cs = _load_cs_json()
    cs_name_map = {cs["id"]: cs["name"] for cs in all_cs}
    sel_cs_ids = [cid for cid in cfg.get("custom_series_ids", []) if cid in cs_name_map]
    all_assets = list(CTA_ASSETS) + [
        {"name": cs_name_map[cid], "ticker": f"custom:{cid}", "class": "Custom"}
        for cid in sel_cs_ids
    ]
    classes = sorted({a["class"] for a in all_assets})
    sel_classes = st.multiselect("Asset classes", classes, default=classes,
                                 key="_ctap_classes")
    assets = [a for a in all_assets if a["class"] in sel_classes]
    if not assets:
        st.info("Select at least one asset class.")
        return

    zc1, zc2, zc3, zc4, zc5 = st.columns([2.7, 1.3, 1.3, 0.9, 0.9])
    _zsel = zc1.multiselect(
        "Assets", [a["name"] for a in assets], key="_cta_z_assets",
        help="Overlay each asset's simulated CTA positioning as a z-score vs "
             "its own trailing history — stretched readings (beyond ±1σ) "
             "imply asymmetric unwind risk, à la the sell-side CTA charts.")
    _zbasis = zc2.selectbox("Basis", ["Multi-speed ensemble",
                                      "Multi-speed signal-only",
                                      "Risk allocation (sized)",
                                      "Combined signal"],
                            key="_cta_z_basis",
                            help="Multi-speed ensemble = TSMOM + EWMA responses at "
                                 "21/63/126/252d, each vol-scaled into a position "
                                 "and summed (closest to sell-side CTA sims; z "
                                 "reflects BOTH trend direction and vol-driven "
                                 "sizing — e.g. the 2017-18 low-vol extremes). "
                                 "Signal-only = same responses UNSIZED — pure "
                                 "direction crowding, immune to vol-regime "
                                 "inflation. Risk allocation = the tab's combined "
                                 "signal × (vol target / realised vol). "
                                 "Combined = raw ±1.")
    _zwts_lbl = zc3.selectbox("Speed weights", list(_ENSEMBLE_WEIGHTS),
                              key="_cta_z_wts",
                              help="How the 21/63/126/252d models are weighted "
                                   "in the Multi-speed ensemble basis (ignored "
                                   "for the other bases). Slow-tilted ≈ where "
                                   "trend-fund AUM sits.")
    _zwin = zc4.selectbox("z window", [126, 252, 504, 1260, 2520], index=4,
                          key="_cta_z_win",
                          format_func=lambda x: {126: "6m", 252: "1y",
                                                 504: "2y", 1260: "5y",
                                                 2520: "10y"}[x],
                          help="10y = long-memory stretch (default); short "
                               "windows flag FRESH regime builds and can "
                               "print big values (2s10s +4.4σ Jun-25 on 1y "
                               "as the steepener built from nothing).")
    _zspan = zc5.selectbox("Chart span", [1, 2, 3, 5, 10, 15, 20], index=1,
                           key="_cta_z_span",
                           format_func=lambda x: f"{x}y")
    _oc1, _oc3, _oc2 = st.columns([2.2, 1.1, 1.3])
    _zpx_on = _oc3.checkbox("underlying chart on top", value=False,
                            key="_cta_z_px",
                            help="Draw the actual asset (level for one "
                                 "asset; %-px / bp-rates change for several) "
                                 "above the main chart, same span.")
    _zovl_on = _oc1.checkbox(
        "overlay signal-only z (dotted) — the gap to the main line is the "
        "vol-sizing channel", value=True, key="_cta_z_ovl") \
        if _zbasis == "Multi-speed ensemble" else False
    _zflw_lbl = _oc2.selectbox(
        "Flows panel", ["1w", "1d", "1m", "off"], key="_cta_z_flw",
        help="Simulated CTA flows = N-day net change in the simulated "
             "position (% of the vol-target book): what the trend crowd "
             "was buying/selling. Green = adding, red = cutting (in the "
             "asset's own direction convention — rates are yield-direction).")
    _FLW_D = {"1d": 1, "1w": 5, "1m": 21}
    if _zsel:
        _tk_by_name = {a["name"]: a["ticker"] for a in assets}
        _cls_by_name = {a["name"]: a["class"] for a in assets}
        _zseries, _zrsi, _zovl, _zflw, _zpx, _zpxm = {}, {}, {}, {}, {}, {}

        def _zof(_s):
            _mu = _s.rolling(_zwin, min_periods=_zwin // 2).mean()
            _sd = _s.rolling(_zwin, min_periods=_zwin // 2).std().replace(0, np.nan)
            _z = ((_s - _mu) / _sd).dropna()
            _cut = pd.Timestamp(date.today() - timedelta(days=_zspan * 365))
            return _z[_z.index >= _cut]

        with st.spinner("Computing positioning history…"):
            for _nm in _zsel:
                _need = _zspan + max(1, _zwin // 252) + 1
                _yrs = next((k for k in (3, 5, 7, 10, 15, 20, 30) if k >= _need), 30)
                _arith = _cls_by_name.get(_nm) == "Rates"   # yield levels
                _h = _hist_signals(_tk_by_name[_nm], tsmom_lb, ma_fast, ma_slow,
                                   don_n, ewma_sp, years=_yrs, rates=_arith)
                if _h.empty:
                    continue
                # custom rates products (spreads etc.) → arithmetic too
                _arith = _arith or _ticker_is_rates(_tk_by_name[_nm])
                if _zbasis.startswith("Multi"):
                    _w, _vbs = _ENSEMBLE_WEIGHTS[_zwts_lbl]
                    _s = _ensemble_position(_h["close"], vol_tgt,
                                            weights=_w, vol_by_speed=_vbs,
                                            sized="signal-only" not in _zbasis,
                                            arithmetic=_arith)
                    if _zovl_on:
                        _z2 = _zof(_ensemble_position(
                            _h["close"], vol_tgt, weights=_w,
                            vol_by_speed=_vbs, sized=False,
                            arithmetic=_arith))
                        if len(_z2):
                            _zovl[_nm] = _z2
                elif _zbasis.startswith("Risk"):
                    _s = (_h["combined"]
                          * (vol_tgt / _h["rvol_ann"].replace(0, np.nan)))
                else:
                    _s = _h["combined"]
                _z = _zof(_s)
                if len(_z):
                    _zseries[_nm] = _z
                    _zrsi[_nm] = _h["rsi30"].reindex(_z.index).dropna()
                    if _zpx_on:
                        _px = _h["close"].reindex(_z.index).dropna()
                        if len(_px):
                            _zpx[_nm] = _px
                            _zpxm[_nm] = (_h["close"].rolling(63).mean()
                                          .reindex(_z.index).dropna())
                    if _zflw_lbl != "off":
                        _f = (_s.diff(_FLW_D[_zflw_lbl]) * 100.0
                              ).reindex(_z.index).dropna()
                        if len(_f):
                            _zflw[_nm] = _f
        if _zseries:
            _zw_lbl = {126: "6m", 252: "1y", 504: "2y", 1260: "5y",
                       2520: "10y"}[_zwin]
            # chart order (Rajat 2026-08-28): 0 underlying (optional, on top)
            # → 1 main z (big, RSI as its 2nd panel) → 3 projected flows →
            # 4 simulated flows
            if _zpx:
                st.plotly_chart(_plot_asset_price(_zpx, _cls_by_name,
                                                  _zpxm or None),
                                use_container_width=True)
            st.plotly_chart(_plot_positioning_z(_zseries, _zw_lbl, _zrsi,
                                                _zovl or None),
                            use_container_width=True)
            # ── Projected 1m flows under scenarios (single asset only) ─────
            if len(_zsel) == 1 and _zbasis.startswith("Multi"):
                _nm1 = _zsel[0]
                st.markdown(f"##### Projected 1m flows — {_nm1}")
                _h1 = _hist_signals(_tk_by_name[_nm1], tsmom_lb, ma_fast,
                                    ma_slow, don_n, ewma_sp, years=5,
                                    rates=_cls_by_name.get(_nm1) == "Rates")
                if not _h1.empty:
                    _w1, _vbs1 = _ENSEMBLE_WEIGHTS[_zwts_lbl]
                    _ar1 = (_cls_by_name.get(_nm1) == "Rates"
                            or _ticker_is_rates(_tk_by_name[_nm1]))
                    with st.spinner("Simulating scenario paths…"):
                        _scn = _scenario_flows(
                            _h1["close"], vol_tgt, _w1, _vbs1,
                            arithmetic=_ar1)
                    st.plotly_chart(_plot_scenario_flows(_scn, _nm1),
                                    use_container_width=True)
                    st.caption(
                        "GS Futures-Flow-Monitor format: grey = simulated "
                        "realized net length (the ensemble position, % of "
                        "vol-target book); coloured fan = expected net length "
                        "under a **k·σ total move over the next 21 business "
                        "days** (σ from current 21d realised vol; Up/Down "
                        "Small = ±1σ, Big = ±2σ), averaged over 25 seeded "
                        "noise paths so realised vol is preserved. The "
                        "distance each line travels from today's level IS the "
                        "anticipated flow. Rates are in yield direction "
                        "(net length + = short bonds). Drift-plus-noise "
                        "central scenarios — not a path-dependent forecast.")
            elif len(_zsel) > 1:
                st.caption("Projected-flows scenarios render when exactly "
                           "one asset is selected.")
            if _zflw:
                st.plotly_chart(_plot_flows_hist(_zflw, _zflw_lbl),
                                use_container_width=True)
            st.caption(
                "z-score of the simulated CTA position vs its own trailing "
                f"{_zw_lbl} (mean/σ). Beyond the dotted ±1σ the positioning is "
                "stretched: further moves in the trend direction add little, "
                "while a reversal forces a mechanical unwind — the red figure "
                "marks the most stretched reading. Rates assets are in YIELD "
                "direction (negative = stretched short yields = long duration "
                "crowd). Uses the parameter set from ⚙ above.")
        else:
            st.info("No usable history for the selected assets.")
    else:
        st.caption("Pick assets above to draw the chart — e.g. the rates "
                   "complex to see how stretched the duration trend book is.")

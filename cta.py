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
    {"name": "JPY 10Y",        "ticker": "^JPY10Y",   "class": "Rates"},
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


def _tsmom(close, lb):
    if len(close) < lb + 1:
        return 0
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


def _ewma_signal(close, span, vol_days=21):
    if len(close) < max(span, vol_days) + 5:
        return 0
    rets = close.pct_change().dropna()
    if len(rets) < vol_days:
        return 0
    ev  = rets.ewm(span=span, adjust=False).mean().iloc[-1]
    rv  = rets.iloc[-vol_days:].std()
    if rv == 0 or not math.isfinite(rv):
        return 0
    return _sign(ev / rv)


def _rvol_ann(close, vol_days=21):
    if len(close) < vol_days + 2:
        return float("nan")
    return float(close.pct_change().dropna().iloc[-vol_days:].std() * math.sqrt(252))


# ── Point-in-time signal table ─────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _compute_signals(tickers, tsmom_days, ma_fast, ma_slow, donchian_n, ewma_span):
    rows = []
    for tkr in tickers:
        if tkr.startswith("custom:"):
            df = _fetch_custom_df_cached(tkr[7:])
        else:
            df = _raw_daily(tkr)
        if df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].dropna()
        if len(close) < 30:
            continue

        vol = _rvol_ann(close)

        # ── Continuous raw values for cross-sectional scoring ─────────────────
        # TSMOM: vol-normalised trailing return
        if len(close) >= tsmom_days + 1 and math.isfinite(vol) and vol > 0:
            ret = float(close.iloc[-1] / close.iloc[-tsmom_days] - 1)
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

        # EWMA: vol-normalised EWMA of returns
        if len(close) >= max(ewma_span, 21) + 5:
            rets = close.pct_change().dropna()
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

        rows.append({
            "ticker":    tkr,
            "tsmom":     _tsmom(close, tsmom_days),
            "ma_cross":  _ma_cross(close, ma_fast, ma_slow),
            "donchian":  _donchian(close, donchian_n),
            "ewma":      _ewma_signal(close, ewma_span),
            "vol_ann":   vol,
            "rsi14":     rsi14,
            "tsmom_raw": tsmom_raw,
            "ma_raw":    ma_raw,
            "don_raw":   don_raw,
            "ewma_raw":  ewma_raw,
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


# ── Extended price history ──────────────────────────────────────────────────────

# Some tickers (e.g. ^US2YT) only work via Ticker.history(), not yf.download().
# Mirror the three-level fallback used by _raw_daily in watchlist.py.
_YF_PERIOD_FOR_YEARS = {1: "2y", 2: "5y", 3: "5y", 5: "5y", 7: "10y", 10: "10y", 15: "max", 20: "max"}


@st.cache_data(ttl=3600, show_spinner=False)
def _raw_daily_ext(ticker: str, years: int) -> pd.DataFrame:
    """Fetch `years` of daily OHLCV. Mirrors _raw_daily's source routing exactly:
    FRED → ECB API → JGB MOF → yfinance (3-level fallback).
    """
    start_str = (date.today() - timedelta(days=years * 365 + 60)).strftime("%Y-%m-%d")
    cutoff    = pd.Timestamp(start_str)
    period    = _YF_PERIOD_FOR_YEARS.get(years, "10y")

    # ── Custom series ────────────────────────────────────────────────────────
    if ticker.startswith("custom:"):
        return _fetch_custom_df_cached(ticker[7:])

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
                  donchian_n: int, ewma_span: int, years: int = 1) -> pd.DataFrame:
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

    tsmom_s = (close.pct_change(tsmom_days)
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

    rets     = close.pct_change()
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

def _vol_fmt(v):
    return "—" if not math.isfinite(v) else f"{v*100:.1f}%"

def _rsi_cell(rsi):
    if not math.isfinite(rsi):
        return '<span style="color:#94A3B8">—</span>'
    if rsi >= 70:   col = "#DC2626"
    elif rsi >= 60: col = "#F87171"
    elif rsi <= 30: col = "#059669"
    elif rsi <= 40: col = "#10B981"
    else:           col = "#94A3B8"
    return f'<span style="color:{col};font-weight:600">{rsi:.1f}</span>'

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
        sigs = _compute_signals(tickers, tsmom_lb, ma_fast, ma_slow, don_n, ewma_sp)

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
    _SORT_OPTS = ["Asset", "Mom.Score", "Norm.Score", "RSI (14)", "Vol (21d)", "TSMOM", "MA Cross", "Breakout", "EWMA", "Risk Alloc"]
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
        row_data.append({
            "name": name, "cls": cls, "badge": badge,
            "total_400": total_400, "old_sum": old_sum,
            "vol": vol, "rsi14": rsi14, "norm_score": norm_score,
            "sc1": sc1, "sc2": sc2, "sc3": sc3, "sc4": sc4,
            "s1": s1, "s2": s2, "s3": s3, "s4": s4, "risk": risk,
        })

    _sort_keys = {
        "Asset":       lambda r: r["name"].lower(),
        "Mom.Score":   lambda r: r["total_400"],
        "Norm.Score":  lambda r: r["norm_score"] if math.isfinite(r["norm_score"]) else -1.0,
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
            f'<td style="{td}">{_rsi_cell(rd["rsi14"])}</td>'
            f'<td style="{td}">{_vol_fmt(rd["vol"])}</td>'
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
            f"Risk alloc = signal × ({vol_tgt*100:.0f}% target vol / realised vol)"
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
        with st.spinner(f"Loading {period_label} of {sel_name} data…"):
            hist = _hist_signals(sel_tkr, tsmom_lb, ma_fast, ma_slow, don_n, ewma_sp, years)
            if hist.empty:
                # Stale empty cache (e.g. from before AlphaVantage routing was added)
                _hist_signals.clear()
                _raw_daily_ext.clear()
                hist = _hist_signals(sel_tkr, tsmom_lb, ma_fast, ma_slow, don_n, ewma_sp, years)
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

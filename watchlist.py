"""
Bloomberg-style watchlist — clickable rows, inline charts, persistent config.
"""

import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import ta as ta_lib
import yfinance as yf
from datetime import datetime, timedelta, date

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist_config.json")

ALL_COLUMNS     = ["Price", "Change %", "Change", "Weekly %", "Monthly %", "RSI", "52W High", "52W Low", "Volume"]
DEFAULT_COLUMNS = ["Price", "Change %", "Weekly %", "Monthly %", "RSI", "52W High", "52W Low"]
ASSET_CLASSES   = ["Equity", "FX", "Rates", "Commodity", "Crypto", "Other"]

CHART_TIMEFRAMES = {
    "1D":  ("1d",  "5m"),
    "1W":  ("5d",  "15m"),
    "1M":  ("1mo", "1h"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "1Y":  ("1y",  "1d"),
    "2Y":  ("2y",  "1wk"),
    "Custom": (None, "1d"),
}

DEFAULT_INSTRUMENTS = [
    {"name": "S&P 500",      "ticker": "^GSPC",    "class": "Equity"},
    {"name": "NASDAQ",       "ticker": "^IXIC",    "class": "Equity"},
    {"name": "FTSE 100",     "ticker": "^FTSE",    "class": "Equity"},
    {"name": "DAX",          "ticker": "^GDAXI",   "class": "Equity"},
    {"name": "Dow Jones",    "ticker": "^DJI",     "class": "Equity"},
    {"name": "EUR/USD",      "ticker": "EURUSD=X", "class": "FX"},
    {"name": "GBP/USD",      "ticker": "GBPUSD=X", "class": "FX"},
    {"name": "USD/JPY",      "ticker": "JPY=X",    "class": "FX"},
    {"name": "USD/CHF",      "ticker": "CHF=X",    "class": "FX"},
    {"name": "US 10Y",       "ticker": "^TNX",     "class": "Rates"},
    {"name": "US 2Y",        "ticker": "^IRX",     "class": "Rates"},
    {"name": "US 30Y",       "ticker": "^TYX",     "class": "Rates"},
    {"name": "UK 10Y Gilt",  "ticker": "^GUKG10",  "class": "Rates"},
    {"name": "Gold",         "ticker": "GC=F",     "class": "Commodity"},
    {"name": "WTI Oil",      "ticker": "CL=F",     "class": "Commodity"},
    {"name": "Brent Crude",  "ticker": "BZ=F",     "class": "Commodity"},
    {"name": "Silver",       "ticker": "SI=F",     "class": "Commodity"},
    {"name": "Bitcoin",      "ticker": "BTC-USD",  "class": "Crypto"},
    {"name": "Ethereum",     "ticker": "ETH-USD",  "class": "Crypto"},
]

CLASS_BG = {
    "Equity":    "#F0F5FF",
    "FX":        "#F0FDF4",
    "Rates":     "#FFFBEB",
    "Commodity": "#FFF4ED",
    "Crypto":    "#FAF5FF",
    "Other":     "#F8FAFC",
}
CLASS_FG = {
    "Equity":    "#1D4ED8",
    "FX":        "#047857",
    "Rates":     "#B45309",
    "Commodity": "#C2410C",
    "Crypto":    "#6D28D9",
    "Other":     "#475569",
}
UP   = "#059669"
DOWN = "#DC2626"
MUTE = "#94A3B8"


# ── Config persistence ─────────────────────────────────────────────────────────
def load_config() -> dict:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
            cfg.setdefault("instruments", DEFAULT_INSTRUMENTS)
            cfg.setdefault("columns",     DEFAULT_COLUMNS)
            return cfg
    except Exception:
        pass
    return {"instruments": DEFAULT_INSTRUMENTS.copy(), "columns": DEFAULT_COLUMNS.copy()}


def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ── Watchlist data ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def fetch_batch(tickers: tuple) -> dict:
    if not tickers:
        return {}
    try:
        raw = yf.download(
            list(tickers), period="1y", interval="1d",
            auto_adjust=True, progress=False, group_by="ticker",
        )
        out = {}
        for tkr in tickers:
            try:
                df = raw[tkr].copy() if len(tickers) > 1 else raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.dropna(how="all", inplace=True)
                if len(df) > 5:
                    out[tkr] = df
            except Exception:
                pass
        return out
    except Exception:
        return {}


def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return None


def compute_row(inst: dict, df, columns: list) -> dict:
    row = {"Name": inst["name"], "Class": inst["class"], "_ticker": inst["ticker"]}
    if df is None or df.empty:
        for c in columns: row[c] = None
        return row

    close = df["Close"].squeeze().dropna().astype(float)
    if close.empty:
        for c in columns: row[c] = None
        return row

    last = _f(close.iloc[-1])
    if last is None:
        for c in columns: row[c] = None
        return row

    prev  = _f(close.iloc[-2]) if len(close) > 1 else last
    d_chg = last - prev if prev else 0
    d_pct = (d_chg / prev * 100) if prev else 0

    for c in columns:
        if c == "Price":
            row[c] = last
        elif c == "Change %":
            row[c] = d_pct
        elif c == "Change":
            row[c] = d_chg
        elif c == "Weekly %":
            ref = _f(close.iloc[-6]) if len(close) > 5 else _f(close.iloc[0])
            row[c] = ((last - ref) / ref * 100) if ref else None
        elif c == "Monthly %":
            ref = _f(close.iloc[-22]) if len(close) > 21 else _f(close.iloc[0])
            row[c] = ((last - ref) / ref * 100) if ref else None
        elif c == "RSI":
            try:
                rsi = ta_lib.momentum.RSIIndicator(close, window=14).rsi().dropna()
                row[c] = _f(rsi.iloc[-1]) if not rsi.empty else None
            except Exception:
                row[c] = None
        elif c == "52W High":
            row[c] = _f(close.max())
        elif c == "52W Low":
            row[c] = _f(close.min())
        elif c == "Volume":
            try:
                vol = df["Volume"].squeeze().dropna()
                row[c] = _f(vol.iloc[-1]) if not vol.empty else None
            except Exception:
                row[c] = None
    return row


# ── Table formatting ───────────────────────────────────────────────────────────
def _fmt(val, col: str) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if col in ("Change %", "Weekly %", "Monthly %"):
        return f"{val:+.2f}%"
    if col == "Change":
        return f"{val:+.4f}" if -1 < val < 1 else f"{val:+.2f}"
    if col == "RSI":
        return f"{val:.1f}"
    if col == "Volume":
        if val >= 1e9: return f"{val/1e9:.1f}B"
        if val >= 1e6: return f"{val/1e6:.1f}M"
        if val >= 1e3: return f"{val/1e3:.0f}K"
        return f"{val:.0f}"
    if abs(val) < 10:   return f"{val:.4f}"
    if abs(val) < 10000: return f"{val:,.2f}"
    return f"{val:,.0f}"


def build_display_df(df_rows: pd.DataFrame, active_cols: list) -> pd.DataFrame:
    """Build a string-formatted display DataFrame for st.dataframe."""
    out = pd.DataFrame()
    out["Name"]  = df_rows["Name"]
    out["Class"] = df_rows["Class"]
    for c in active_cols:
        if c in df_rows.columns:
            out[c] = df_rows[c].apply(lambda v: _fmt(v, c))
    return out.reset_index(drop=True)


def style_table(df_display: pd.DataFrame, active_cols: list):
    """Apply green/red text to change columns via pandas Styler."""
    pct_cols = [c for c in active_cols if c in ("Change %", "Change", "Weekly %", "Monthly %")]
    rsi_cols = [c for c in active_cols if c == "RSI"]

    def color_change(val: str):
        if isinstance(val, str) and val.startswith("+"):
            return f"color: {UP}; font-weight: 600"
        if isinstance(val, str) and val.startswith("-"):
            return f"color: {DOWN}; font-weight: 600"
        return ""

    def color_rsi(val: str):
        try:
            v = float(val)
            if v > 70: return f"color: {DOWN}; font-weight: 600"
            if v < 30: return f"color: {UP}; font-weight: 600"
        except Exception:
            pass
        return ""

    def color_class(val: str):
        return f"color: {CLASS_FG.get(val, MUTE)}; font-weight: 600"

    styler = df_display.style
    if pct_cols:
        styler = styler.map(color_change, subset=pct_cols)
    if rsi_cols:
        styler = styler.map(color_rsi, subset=rsi_cols)
    styler = styler.map(color_class, subset=["Class"])
    styler = styler.set_properties(**{
        "font-family": "Inter, Segoe UI, sans-serif",
        "font-size":   "13px",
    })
    styler = styler.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#1B3A6B"),
            ("color", "white"),
            ("font-weight", "700"),
            ("font-size", "12px"),
            ("padding", "8px 12px"),
            ("text-align", "left"),
        ]},
        {"selector": "td", "props": [("padding", "7px 12px")]},
        {"selector": "tr:hover td", "props": [("background-color", "#EFF6FF !important")]},
    ])
    return styler


# ── Chart for selected instrument ──────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_chart_data(ticker: str, period: str | None, interval: str,
                     rsi_period: int = 14,
                     start: str | None = None, end: str | None = None) -> pd.DataFrame:
    try:
        if start and end:
            df = yf.download(ticker, start=start, end=end, interval=interval,
                             auto_adjust=True, progress=False)
        else:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        if not df.empty:
            close = df["Close"].squeeze().astype(float)
            df["SMA20"]  = ta_lib.trend.SMAIndicator(close, window=20).sma_indicator()
            df["SMA50"]  = ta_lib.trend.SMAIndicator(close, window=50).sma_indicator()
            df["SMA100"] = ta_lib.trend.SMAIndicator(close, window=100).sma_indicator()
            df["SMA200"] = ta_lib.trend.SMAIndicator(close, window=200).sma_indicator()
            df["EMA20"] = ta_lib.trend.EMAIndicator(close, window=20).ema_indicator()
            bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
            df["BB_upper"] = bb.bollinger_hband()
            df["BB_lower"] = bb.bollinger_lband()
            df["RSI"] = ta_lib.momentum.RSIIndicator(close, window=rsi_period).rsi()
        return df
    except Exception:
        return pd.DataFrame()


def get_rangebreaks(df: pd.DataFrame) -> list:
    breaks = [dict(bounds=["sat", "mon"])]
    if df.empty or len(df) < 2:
        return breaks
    idx = pd.DatetimeIndex(df.index)
    delta_seconds = (idx[1] - idx[0]).total_seconds()
    if delta_seconds >= 86_400:
        try:
            dates = idx.normalize()
            all_bdays = pd.bdate_range(dates.min(), dates.max())
            holidays = all_bdays.difference(dates)
            if len(holidays):
                breaks.append(dict(values=holidays.strftime("%Y-%m-%d").tolist()))
        except Exception:
            pass
    else:
        try:
            hours = idx.hour + idx.minute / 60
            lo, hi = hours.min(), hours.max()
            if hi - lo < 18:
                breaks.append(dict(bounds=[hi + 0.25, lo - 0.25], pattern="hour"))
        except Exception:
            pass
    return breaks


def build_instrument_chart(df: pd.DataFrame, name: str, ticker: str,
                            chart_type: str = "Candlestick",
                            overlays: list = None,
                            rsi_period: int = 14) -> go.Figure:
    if overlays is None:
        overlays = ["SMA 20", "SMA 50"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
    )

    close = df["Close"].squeeze()

    # ── Price trace ────────────────────────────────────────────────────────
    if chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=df.index, y=close, name="Price",
            line=dict(color="#0F172A", width=2),
        ), row=1, col=1)

    elif chart_type == "OHLC Bar":
        fig.add_trace(go.Ohlc(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name="Price",
            increasing_line_color="#059669",
            decreasing_line_color="#DC2626",
        ), row=1, col=1)

    else:  # Candlestick (default)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name="Price",
            increasing_line_color="#059669", decreasing_line_color="#DC2626",
            increasing_fillcolor="#059669",  decreasing_fillcolor="#DC2626",
        ), row=1, col=1)

    # ── Overlays ───────────────────────────────────────────────────────────
    if "SMA 20" in overlays and "SMA20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA 20",
            line=dict(color="#F59E0B", width=1.5, dash="dot")), row=1, col=1)

    if "SMA 50" in overlays and "SMA50" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50",
            line=dict(color="#A855F7", width=1.5, dash="dot")), row=1, col=1)

    if "SMA 100" in overlays and "SMA100" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA100"], name="SMA 100",
            line=dict(color="#06B6D4", width=1.5, dash="dot")), row=1, col=1)

    if "SMA 200" in overlays and "SMA200" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200",
            line=dict(color="#EF4444", width=2, dash="dot")), row=1, col=1)

    if "EMA 20" in overlays and "EMA20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20",
            line=dict(color="#EC4899", width=1.5, dash="dot")), row=1, col=1)

    if "Bollinger Bands" in overlays and "BB_upper" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="rgba(100,116,139,0.6)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="rgba(100,116,139,0.6)", width=1),
            fill="tonexty", fillcolor="rgba(100,116,139,0.05)"), row=1, col=1)

    # ── RSI ────────────────────────────────────────────────────────────────
    rsi_row = 2
    if "RSI" in df:
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(220,38,38,0.05)",
                      line_width=0, row=rsi_row, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(5,150,105,0.05)",
                      line_width=0, row=rsi_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#DC2626",
                      line_width=1.5, row=rsi_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#059669",
                      line_width=1.5, row=rsi_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            name=f"RSI({rsi_period})",
            line=dict(color="#0EA5E9", width=1.8),
            fill="tozeroy", fillcolor="rgba(14,165,233,0.06)",
        ), row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1,
                         tickfont=dict(size=10), gridcolor="#E8EDF5")

    fig.update_layout(
        title=dict(
            text=f"<b>{name}</b>  <span style='font-size:13px;color:#64748B'>({ticker})  ·  {chart_type}  ·  RSI({rsi_period})</span>",
            font=dict(size=16, color="#1A202C", family="Inter, Segoe UI, sans-serif"),
        ),
        xaxis_rangeslider_visible=False,
        height=600,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
        font=dict(color="#1A202C", family="Inter, Segoe UI, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0",
                        font=dict(color="#1A202C", size=12)),
    )
    fig.update_xaxes(gridcolor="#E8EDF5", zeroline=False, linecolor="#E2E8F0",
                     showspikes=True, spikecolor="#94A3B8", spikethickness=1,
                     rangebreaks=get_rangebreaks(df))
    fig.update_yaxes(gridcolor="#E8EDF5", zeroline=False, linecolor="#E2E8F0",
                     showspikes=True, spikecolor="#94A3B8", spikethickness=1,
                     rangemode="normal", autorange=True, row=1, col=1)
    return fig


# ── Main render ────────────────────────────────────────────────────────────────
def render_watchlist():
    if "wl_config" not in st.session_state:
        st.session_state.wl_config = load_config()
    if "wl_selected" not in st.session_state:
        st.session_state.wl_selected = None

    cfg         = st.session_state.wl_config
    instruments = cfg["instruments"]
    sel_cols    = cfg["columns"]

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])

    with c1:
        new_cols = st.multiselect("Columns", ALL_COLUMNS, default=sel_cols,
                                  help="Choose which metrics to display")
        if set(new_cols) != set(sel_cols):
            cfg["columns"] = new_cols
            save_config(cfg)
            st.session_state.wl_config = cfg

    active_cols = new_cols if new_cols else sel_cols

    with c2:
        sort_col = st.selectbox("Sort by", ["Name", "Class"] + active_cols)

    with c3:
        sort_dir = st.radio("Order", ["↑ Asc", "↓ Desc"], horizontal=True)
        sort_asc = sort_dir.startswith("↑")

    with c4:
        st.markdown("<div style='margin-top:26px'>", unsafe_allow_html=True)
        if st.button("⟳ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Asset class filter pills ──────────────────────────────────────────────
    present_classes = sorted({i["class"] for i in instruments})
    selected_classes = st.pills(
        "Filter by asset class", options=present_classes,
        default=present_classes, selection_mode="multi",
    )
    if not selected_classes:
        selected_classes = present_classes

    # ── Fetch data ────────────────────────────────────────────────────────────
    tickers = tuple(i["ticker"] for i in instruments)
    with st.spinner("Loading watchlist…"):
        batch = fetch_batch(tickers)

    rows = [compute_row(inst, batch.get(inst["ticker"]), active_cols) for inst in instruments]
    df_rows = pd.DataFrame(rows)

    # Filter & sort
    df_rows = df_rows[df_rows["Class"].isin(selected_classes)]
    if sort_col in df_rows.columns:
        # Sort on numeric values before formatting
        df_rows = df_rows.sort_values(sort_col, ascending=sort_asc, na_position="last")

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:11px;color:#94A3B8;margin-bottom:4px'>"
        "👆 Click a row to chart it</p>",
        unsafe_allow_html=True,
    )

    df_display = build_display_df(df_rows, active_cols)
    styled     = style_table(df_display, active_cols)

    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=min(600, 38 + 35 * len(df_display)),
    )

    st.caption(
        f"Last updated: {datetime.now().strftime('%H:%M:%S')}  ·  "
        f"{len(df_rows)} instruments shown  ·  Data: Yahoo Finance"
    )

    # ── Inline chart ─────────────────────────────────────────────────────────
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        row_idx  = selected_rows[0]
        row_data = df_display.iloc[row_idx]
        inst_name = row_data["Name"]

        # Look up ticker
        inst = next((i for i in instruments if i["name"] == inst_name), None)
        if inst:
            st.session_state.wl_selected = inst

    sel = st.session_state.wl_selected
    if sel:
        st.markdown("---")
        st.markdown(
            f"<h4 style='color:#1B3A6B;margin-bottom:4px'>"
            f"{sel['name']} "
            f"<span style='font-size:13px;color:#64748B;font-weight:400'>"
            f"({sel['ticker']})</span></h4>",
            unsafe_allow_html=True,
        )

        # ── Row 1: Timeframe + Close ──────────────────────────────────────
        tf_cols = st.columns([7, 1])
        with tf_cols[0]:
            tf = st.segmented_control(
                "Timeframe",
                options=list(CHART_TIMEFRAMES.keys()),
                default="1M",
                key=f"tf_{sel['ticker']}",
            )
        with tf_cols[1]:
            st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
            if st.button("✕ Close", key="close_chart", use_container_width=True):
                st.session_state.wl_selected = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 2: Chart type · RSI period · Overlays ─────────────────────
        opt_c1, opt_c2, opt_c3 = st.columns([2, 2, 4])

        with opt_c1:
            chart_type = st.selectbox(
                "Chart type",
                ["Candlestick", "OHLC Bar", "Line"],
                key=f"ct_{sel['ticker']}",
            )

        with opt_c2:
            rsi_period = st.number_input(
                "RSI period", min_value=2, max_value=50,
                value=14, step=1,
                key=f"rsi_{sel['ticker']}",
                help="Standard is 14. Higher = smoother, lower = more sensitive.",
            )

        with opt_c3:
            overlays = st.multiselect(
                "Overlays",
                ["SMA 20", "SMA 50", "SMA 100", "SMA 200", "EMA 20", "Bollinger Bands"],
                default=["SMA 20", "SMA 50"],
                key=f"ov_{sel['ticker']}",
            )

        # ── Custom date range ─────────────────────────────────────────────
        if tf == "Custom":
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                start_date = st.date_input("From", value=date.today() - timedelta(days=180),
                                           max_value=date.today() - timedelta(days=2))
            with d_col2:
                end_date = st.date_input("To", value=date.today(),
                                         min_value=start_date + timedelta(days=1),
                                         max_value=date.today())
            period    = None
            interval  = "1d"
            start_str = str(start_date)
            end_str   = str(end_date)
        else:
            period, interval = CHART_TIMEFRAMES[tf]
            start_str = end_str = None

        with st.spinner(f"Loading {sel['name']} chart…"):
            df_chart = fetch_chart_data(
                sel["ticker"], period, interval,
                rsi_period=int(rsi_period),
                start=start_str, end=end_str,
            )

        if not df_chart.empty:
            # Key encodes all options so Streamlit always redraws when anything changes
            chart_key = (
                f"wl_{sel['ticker']}_{tf}_{chart_type}_{rsi_period}"
                f"_{'_'.join(sorted(overlays or []))}"
                f"_{start_str}_{end_str}"
            )
            st.plotly_chart(
                build_instrument_chart(
                    df_chart, sel["name"], sel["ticker"],
                    chart_type=chart_type,
                    overlays=overlays,
                    rsi_period=int(rsi_period),
                ),
                use_container_width=True,
                key=chart_key,
            )
        else:
            st.warning("No chart data available for this instrument / timeframe.")

    st.markdown("---")

    # ── Add / Remove ──────────────────────────────────────────────────────────
    col_add, col_remove = st.columns(2)

    with col_add:
        with st.expander("➕  Add instrument"):
            new_name   = st.text_input("Display name", placeholder="e.g. Apple Inc")
            new_ticker = st.text_input("Yahoo Finance ticker", placeholder="e.g. AAPL  /  ^FTSE  /  GC=F")
            new_class  = st.selectbox("Asset class", ASSET_CLASSES, key="add_class")
            if st.button("Add to watchlist", type="primary", key="btn_add"):
                name   = new_name.strip()
                ticker = new_ticker.upper().strip()
                if name and ticker:
                    entry = {"name": name, "ticker": ticker, "class": new_class}
                    if ticker not in [i["ticker"] for i in cfg["instruments"]]:
                        cfg["instruments"].append(entry)
                        save_config(cfg)
                        st.session_state.wl_config = cfg
                        st.cache_data.clear()
                        st.success(f"✓ Added {name} ({ticker})")
                        st.rerun()
                    else:
                        st.warning(f"{ticker} is already in the watchlist.")
                else:
                    st.warning("Please fill in both name and ticker.")

    with col_remove:
        with st.expander("➖  Remove instruments"):
            names = [i["name"] for i in cfg["instruments"]]
            to_remove = st.multiselect("Select to remove", names)
            if st.button("Remove selected", type="secondary", key="btn_remove"):
                if to_remove:
                    cfg["instruments"] = [i for i in cfg["instruments"]
                                          if i["name"] not in to_remove]
                    save_config(cfg)
                    st.session_state.wl_config = cfg
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("Select at least one instrument to remove.")

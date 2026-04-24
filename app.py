import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta as ta_lib
from datetime import datetime, date, timedelta
import time
import json
import urllib.request
import finnhub
from watchlist import render_watchlist
from rates import render_rates
from correl import render_correl

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium fintech CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #FFFFFF;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Top header bar ── */
[data-testid="stHeader"] { background-color: #FFFFFF; border-bottom: 1px solid #E2E8F0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #F7F9FC;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] h1 {
    color: #1B2B4B;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.3px;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E8EDF5;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 12px rgba(37,99,235,0.10);
    border-color: #BFDBFE;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    text-transform: uppercase !important;
    color: #64748B !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.18rem !important;
    font-weight: 700 !important;
    color: #1A202C !important;
    letter-spacing: -0.3px !important;
}

/* ── Section headings ── */
h2, h3 { color: #1B2B4B !important; font-weight: 700 !important; letter-spacing: -0.4px; }

/* ── Dividers ── */
hr { border: none; border-top: 1px solid #E2E8F0 !important; margin: 0.5rem 0; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    background: #FAFBFD !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #2563EB; }

/* ── Selectbox & inputs ── */
[data-testid="stSelectbox"] > div,
[data-testid="stTextInput"] > div > div {
    border-radius: 8px !important;
    border-color: #E2E8F0 !important;
}

/* ── Positive delta green, negative red ── */
[data-testid="stMetricDelta"] > div { font-weight: 600 !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Finnhub client ─────────────────────────────────────────────────────────────
def get_finnhub():
    try:
        key = st.secrets["FINNHUB_KEY"]
        if key:
            return finnhub.Client(api_key=key)
    except Exception:
        pass
    return None

# ── Market data definitions ────────────────────────────────────────────────────
INDICES = {
    "S&P 500":      "^GSPC",
    "NASDAQ":       "^IXIC",
    "Dow Jones":    "^DJI",
    "Russell 2000": "^RUT",
    "VIX":          "^VIX",
}

BONDS = {
    "US 2Y":     "^IRX",
    "US 5Y":     "^FVX",
    "US 10Y":    "^TNX",
    "US 30Y":    "^TYX",
}

# Tickers sourced from FRED instead of yfinance (Yahoo no longer carries them)
FRED_MAP = {
    "^GUKG10": "IRLTLT01GBM156N",   # UK 10-Year Government Bond Yield (monthly, OECD via FRED)
}

COMMODITIES = {
    "Gold":        "GC=F",
    "Silver":      "SI=F",
    "WTI Oil":     "CL=F",
    "Brent Crude": "BZ=F",
    "Nat Gas":     "NG=F",
}

CURRENCIES = {
    "DXY":     "DX-Y.NYB",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CNY": "CNY=X",
}

CRYPTO = {
    "Bitcoin":  "BTC-USD",
    "Ethereum": "ETH-USD",
    "BNB":      "BNB-USD",
    "SOL":      "SOL-USD",
}

# Flat name→ticker map for the dropdown (grouped by category)
TICKER_OPTIONS = {
    # Indices
    "S&P 500 (^GSPC)":      "^GSPC",
    "NASDAQ (^IXIC)":       "^IXIC",
    "Dow Jones (^DJI)":     "^DJI",
    "Russell 2000 (^RUT)":  "^RUT",
    "VIX (^VIX)":           "^VIX",
    # US Bonds
    "US 2Y Treasury (^IRX)":  "^IRX",
    "US 5Y Treasury (^FVX)":  "^FVX",
    "US 10Y Treasury (^TNX)": "^TNX",
    "US 30Y Treasury (^TYX)": "^TYX",
    # Commodities
    "Gold (GC=F)":         "GC=F",
    "Silver (SI=F)":       "SI=F",
    "WTI Oil (CL=F)":      "CL=F",
    "Brent Crude (BZ=F)":  "BZ=F",
    "Nat Gas (NG=F)":      "NG=F",
    # Currencies
    "DXY Dollar Index":    "DX-Y.NYB",
    "EUR/USD":             "EURUSD=X",
    "GBP/USD":             "GBPUSD=X",
    "USD/JPY":             "JPY=X",
    "USD/CNY":             "CNY=X",
    # Crypto
    "Bitcoin (BTC-USD)":   "BTC-USD",
    "Ethereum (ETH-USD)":  "ETH-USD",
    "BNB (BNB-USD)":       "BNB-USD",
    "Solana (SOL-USD)":    "SOL-USD",
    # Stocks & ETFs
    "Apple (AAPL)":        "AAPL",
    "Microsoft (MSFT)":    "MSFT",
    "Google (GOOGL)":      "GOOGL",
    "Amazon (AMZN)":       "AMZN",
    "NVIDIA (NVDA)":       "NVDA",
    "Meta (META)":         "META",
    "Tesla (TSLA)":        "TSLA",
    "S&P 500 ETF (SPY)":   "SPY",
    "NASDAQ ETF (QQQ)":    "QQQ",
    "Gold ETF (GLD)":      "GLD",
}

TIMEFRAMES = {
    "1 Day":    ("1d",  "5m"),
    "5 Days":   ("5d",  "15m"),
    "1 Month":  ("1mo", "1h"),
    "3 Months": ("3mo", "1d"),
    "6 Months": ("6mo", "1d"),
    "1 Year":   ("1y",  "1d"),
    "2 Years":  ("2y",  "1wk"),
}

COLORS = {
    "up":          "#059669",
    "down":        "#DC2626",
    "volume_up":   "rgba(5,150,105,0.25)",
    "volume_down": "rgba(220,38,38,0.25)",
    "sma20":       "#F59E0B",   # amber/gold
    "sma50":       "#A855F7",   # purple
    "ema20":       "#EC4899",   # pink
    "bb_upper":    "rgba(100,116,139,0.6)",  # slate
    "bb_lower":    "rgba(100,116,139,0.6)",
    "bb_fill":     "rgba(100,116,139,0.05)",
    # chart backgrounds
    "paper":       "#FFFFFF",
    "plot":        "#FAFBFD",
    "grid":        "#E8EDF5",
    "text":        "#1A202C",
    "spike":       "#94A3B8",
    # indicator lines
    "rsi_line":    "#0EA5E9",   # sky blue
    "rsi_fill":    "rgba(14,165,233,0.06)",
    "macd_line":   "#0EA5E9",   # sky blue
    "signal_line": "#F59E0B",   # amber
}

# ── FRED helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def _fetch_fred_df(series_id: str, start: str = "2020-01-01") -> pd.DataFrame:
    """Fetch a FRED time series and return as an OHLCV-shaped DataFrame."""
    try:
        key = st.secrets["FRED_KEY"]
    except Exception:
        return pd.DataFrame()
    try:
        url = (
            "https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={key}&file_type=json"
            f"&sort_order=asc&observation_start={start}"
        )
        with urllib.request.urlopen(url, timeout=10) as r:
            obs = json.loads(r.read().decode())["observations"]
        rows = [(o["date"], float(o["value"])) for o in obs if o["value"] != "."]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["Date", "Close"])
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"]  = df["Close"]
        df["Volume"] = 0
        return df
    except Exception:
        return pd.DataFrame()


def _fred_period_start(period: str) -> str:
    """Convert a yfinance-style period string to a FRED observation_start date."""
    days = {"1d": 60, "5d": 60, "1mo": 90, "3mo": 100,
            "6mo": 190, "1y": 370, "2y": 740, "5y": 1830}
    return (date.today() - timedelta(days=days.get(period, 370))).isoformat()


# ── Data helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_fred_quote(ticker: str) -> dict:
    """Latest quote for a FRED-sourced ticker (daily data, updates once a day)."""
    series_id = FRED_MAP.get(ticker)
    if not series_id:
        return {}
    df = _fetch_fred_df(series_id, start=_fred_period_start("1mo"))
    if df.empty:
        return {}
    close = df["Close"].dropna()
    if len(close) < 2:
        return {}
    price  = float(close.iloc[-1])
    prev   = float(close.iloc[-2])
    change = price - prev
    pct    = (change / prev * 100) if prev else 0
    return {
        "price":      price,
        "prev_close": prev,
        "change":     change,
        "change_pct": pct,
        "high":  None, "low": None, "open": None, "volume": None,
        "source": "FRED (daily)",
    }


@st.cache_data(ttl=15)
def fetch_realtime_quote(ticker: str) -> dict:
    """Real-time quote via Finnhub for stocks; fallback to yfinance or FRED."""
    if ticker in FRED_MAP:
        return fetch_fred_quote(ticker)
    fh = get_finnhub()
    # Finnhub only supports plain stock tickers (no ^ or = or -)
    use_finnhub = fh and not any(c in ticker for c in ["^", "=", "-", "."])
    if use_finnhub:
        try:
            q = fh.quote(ticker)
            if q and q.get("c"):
                price  = q["c"]
                prev   = q["pc"]
                change = q["d"]
                pct    = q["dp"]
                return {
                    "price":      price,
                    "prev_close": prev,
                    "change":     change,
                    "change_pct": pct,
                    "high":       q.get("h"),
                    "low":        q.get("l"),
                    "open":       q.get("o"),
                    "volume":     None,
                    "source":     "Finnhub (real-time)",
                }
        except Exception:
            pass
    return fetch_yf_quote(ticker)


@st.cache_data(ttl=60)
def fetch_yf_quote(ticker: str) -> dict:
    """Quote via yfinance (15-min delay for stocks). Falls back to FRED if needed."""
    if ticker in FRED_MAP:
        return fetch_fred_quote(ticker)
    try:
        info  = yf.Ticker(ticker).fast_info
        price = info.last_price
        prev  = info.previous_close
        if not price or not prev:
            return {}
        change = price - prev
        pct    = (change / prev) * 100
        return {
            "price":      price,
            "prev_close": prev,
            "change":     change,
            "change_pct": pct,
            "high":       getattr(info, "day_high",    None),
            "low":        getattr(info, "day_low",     None),
            "open":       getattr(info, "open",        None),
            "volume":     getattr(info, "last_volume", None),
            "source":     "Yahoo Finance",
        }
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """OHLCV + technical indicators via yfinance (or FRED for select tickers)."""
    if ticker in FRED_MAP:
        df = _fetch_fred_df(FRED_MAP[ticker], start=_fred_period_start(period))
        if df.empty:
            return df
        df.dropna(inplace=True)
    else:
        df = yf.download(ticker, period=period, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)

    close = df["Close"].squeeze()

    df["SMA20"]  = ta_lib.trend.SMAIndicator(close, window=20).sma_indicator()
    df["SMA50"]  = ta_lib.trend.SMAIndicator(close, window=50).sma_indicator()
    df["SMA100"] = ta_lib.trend.SMAIndicator(close, window=100).sma_indicator()
    df["SMA200"] = ta_lib.trend.SMAIndicator(close, window=200).sma_indicator()
    df["EMA20"] = ta_lib.trend.EMAIndicator(close, window=20).ema_indicator()

    bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_mid"]   = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()

    df["RSI"] = ta_lib.momentum.RSIIndicator(close, window=14).rsi()

    macd = ta_lib.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_hist"]   = macd.macd_diff()
    df["MACD_signal"] = macd.macd_signal()

    return df


# ── Chart helpers ──────────────────────────────────────────────────────────────
def get_rangebreaks(df: pd.DataFrame) -> list:
    """
    Build Plotly rangebreaks by detecting every gap directly from the data.
    No holiday calendar needed — works for any asset, interval, or timezone.
    A gap between consecutive bars that is >1.5x the normal interval is hidden.
    """
    if df.empty or len(df) < 2:
        return []

    idx = pd.DatetimeIndex(df.index)

    # Strip timezone for arithmetic — tz_localize(None) removes label without conversion
    try:
        idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    except Exception:
        idx_naive = idx

    # Infer the expected bar spacing = smallest positive diff between consecutive bars
    try:
        diffs = idx_naive.to_series().diff().dt.total_seconds().dropna()
        pos = diffs[diffs > 0]
        if pos.empty:
            return []
        expected = pos.min()
    except Exception:
        return []

    threshold = expected * 1.5
    breaks = []

    try:
        for i in range(len(idx_naive) - 1):
            gap = (idx_naive[i + 1] - idx_naive[i]).total_seconds()
            if gap > threshold:
                gap_start = idx_naive[i] + pd.Timedelta(seconds=expected)
                gap_end   = idx_naive[i + 1]
                breaks.append(dict(bounds=[
                    gap_start.strftime("%Y-%m-%dT%H:%M:%S"),
                    gap_end.strftime("%Y-%m-%dT%H:%M:%S"),
                ]))
    except Exception:
        pass

    return breaks


# ── Chart builders ─────────────────────────────────────────────────────────────
def build_main_chart(df, ticker, overlays, show_volume):
    row_heights = [0.7, 0.3] if show_volume else [1.0]
    fig = make_subplots(
        rows=2 if show_volume else 1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="Price",
        increasing_line_color=COLORS["up"],
        decreasing_line_color=COLORS["down"],
        increasing_fillcolor=COLORS["up"],
        decreasing_fillcolor=COLORS["down"],
    ), row=1, col=1)

    if "SMA 20" in overlays and "SMA20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA 20",
            line=dict(color=COLORS["sma20"], width=1.5, dash="dot")), row=1, col=1)
    if "SMA 50" in overlays and "SMA50" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50",
            line=dict(color=COLORS["sma50"], width=1.5, dash="dot")), row=1, col=1)
    if "SMA 100" in overlays and "SMA100" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA100"], name="SMA 100",
            line=dict(color="#06B6D4", width=1.5, dash="dot")), row=1, col=1)
    if "SMA 200" in overlays and "SMA200" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200",
            line=dict(color="#EF4444", width=2, dash="dot")), row=1, col=1)
    if "EMA 20" in overlays and "EMA20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20",
            line=dict(color=COLORS["ema20"], width=1.5, dash="dot")), row=1, col=1)
    if "Bollinger Bands" in overlays and "BB_upper" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color=COLORS["bb_upper"], width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color=COLORS["bb_lower"], width=1),
            fill="tonexty", fillcolor=COLORS["bb_fill"]), row=1, col=1)

    if show_volume and "Volume" in df:
        vol_colors = [
            COLORS["volume_up"] if c >= o else COLORS["volume_down"]
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, showlegend=False), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1, tickfont=dict(size=10))

    fig.update_layout(
        title=dict(text=ticker, font=dict(size=18, color=COLORS["text"], family="Inter, Segoe UI, sans-serif")),
        xaxis_rangeslider_visible=False,
        height=480,
        paper_bgcolor=COLORS["paper"], plot_bgcolor=COLORS["plot"],
        font=dict(color=COLORS["text"], family="Inter, Segoe UI, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#E2E8F0", borderwidth=1),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0",
                        font=dict(color="#1A202C", size=12)),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False, linecolor="#E2E8F0",
                     showspikes=True, spikecolor=COLORS["spike"], spikethickness=1,
                     rangebreaks=get_rangebreaks(df))
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False, linecolor="#E2E8F0",
                     showspikes=True, spikecolor=COLORS["spike"], spikethickness=1)
    return fig


def build_rsi_chart(df):
    fig = go.Figure()
    if "RSI" not in df:
        return fig
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(220,38,38,0.06)", line_width=0,
                  annotation_text="Overbought", annotation_position="top left",
                  annotation=dict(font_color="#DC2626", font_size=11))
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(5,150,105,0.06)", line_width=0,
                  annotation_text="Oversold", annotation_position="bottom left",
                  annotation=dict(font_color="#059669", font_size=11))
    fig.add_hline(y=70, line_dash="dot", line_color="#DC2626", line_width=1.5)
    fig.add_hline(y=30, line_dash="dot", line_color="#059669", line_width=1.5)
    fig.add_hline(y=50, line_dash="dot",  line_color="#CBD5E1", line_width=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)",
        line=dict(color=COLORS["rsi_line"], width=2),
        fill="tozeroy", fillcolor=COLORS["rsi_fill"]))
    fig.update_layout(
        title=dict(text="RSI (14)", font=dict(size=14, color=COLORS["text"])),
        height=220, paper_bgcolor=COLORS["paper"], plot_bgcolor=COLORS["plot"],
        font=dict(color=COLORS["text"]), margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 100], gridcolor=COLORS["grid"], zeroline=False),
        xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        showlegend=False, hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", font=dict(color="#1A202C")),
    )
    return fig


def build_macd_chart(df):
    fig = go.Figure()
    if "MACD" not in df:
        return fig
    hist_colors = [COLORS["up"] if v >= 0 else COLORS["down"]
                   for v in df["MACD_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Histogram",
        marker_color=hist_colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
        line=dict(color=COLORS["macd_line"], width=1.8)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
        line=dict(color=COLORS["signal_line"], width=1.8)))
    fig.add_hline(y=0, line_color="#CBD5E1", line_width=1)
    fig.update_layout(
        title=dict(text="MACD (12, 26, 9)", font=dict(size=14, color=COLORS["text"])),
        height=220, paper_bgcolor=COLORS["paper"], plot_bgcolor=COLORS["plot"],
        font=dict(color=COLORS["text"]), margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="right", x=1, font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#E2E8F0", borderwidth=1),
        xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0", font=dict(color="#1A202C")),
    )
    return fig


# ── UI helpers ─────────────────────────────────────────────────────────────────
def fmt_large(n):
    if n is None:
        return "—"
    if abs(n) >= 1e12: return f"{n/1e12:.2f}T"
    if abs(n) >= 1e9:  return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:  return f"{n/1e6:.2f}M"
    return f"{n:,.0f}"


def render_overview_row(label: str, tickers: dict):
    """Render a labelled row of metric cards for a group of tickers."""
    st.markdown(f"**{label}**")
    cols = st.columns(len(tickers))
    for col, (name, symbol) in zip(cols, tickers.items()):
        q = fetch_yf_quote(symbol)
        if not q:
            col.metric(label=name, value="—")
            continue
        price = q["price"]
        delta = q["change_pct"]
        arrow = "▲" if delta >= 0 else "▼"
        # Bonds show yield (%), currencies 4dp, others 2dp
        if symbol in BONDS.values():
            price_str = f"{price:.3f}%"
        elif symbol in CURRENCIES.values():
            price_str = f"{price:.4f}"
        elif price < 10:
            price_str = f"{price:.4f}"
        else:
            price_str = f"{price:,.2f}"
        col.metric(
            label=f"{name}",
            value=price_str,
            delta=f"{arrow} {abs(delta):.2f}%",
            delta_color="normal" if delta >= 0 else "inverse",
        )


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.title("📈 Market Dashboard")
    st.sidebar.markdown("---")

    custom = st.sidebar.text_input(
        "Custom ticker", placeholder="e.g. AAPL, ^TNX, GC=F"
    ).upper().strip()
    selected_label = st.sidebar.selectbox(
        "Select ticker", list(TICKER_OPTIONS.keys())
    )
    selected = TICKER_OPTIONS[selected_label]
    ticker = custom if custom else selected

    timeframe = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=2)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Chart Overlays**")
    overlays = []
    if st.sidebar.checkbox("SMA 20",          value=True):  overlays.append("SMA 20")
    if st.sidebar.checkbox("SMA 50",          value=True):  overlays.append("SMA 50")
    if st.sidebar.checkbox("SMA 100",         value=False): overlays.append("SMA 100")
    if st.sidebar.checkbox("SMA 200",         value=False): overlays.append("SMA 200")
    if st.sidebar.checkbox("EMA 20",          value=False): overlays.append("EMA 20")
    if st.sidebar.checkbox("Bollinger Bands", value=False): overlays.append("Bollinger Bands")
    show_volume = st.sidebar.checkbox("Volume bars", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Indicators**")
    show_rsi  = st.sidebar.checkbox("RSI (14)",          value=True)
    show_macd = st.sidebar.checkbox("MACD (12, 26, 9)", value=True)

    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

    st.sidebar.markdown("---")
    fh = get_finnhub()
    src = "Finnhub (real-time) + Yahoo Finance" if fh else "Yahoo Finance (15-min delay)"
    st.sidebar.caption(f"Data: {src}")
    st.sidebar.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

    return ticker, timeframe, overlays, show_volume, show_rsi, show_macd, auto_refresh


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ticker, timeframe, overlays, show_volume, show_rsi, show_macd, auto_refresh = render_sidebar()

    tab_charts, tab_macro, tab_rates, tab_correl = st.tabs(["📊  Charts & Indicators", "📋  Macro", "📈  Rates", "🔗  Correl"])

    with tab_macro:
        render_watchlist()

    with tab_rates:
        render_rates()

    with tab_correl:
        render_correl()

    with tab_charts:
        # ── Market overview ──────────────────────────────────────────────────
        st.subheader("Market Overview")
        render_overview_row("Indices",       INDICES)
        st.markdown("")
        render_overview_row("Bonds / Rates", BONDS)
        st.markdown("")

        col_left, col_right = st.columns(2)
        with col_left:
            render_overview_row("Commodities", COMMODITIES)
        with col_right:
            render_overview_row("Currencies",  CURRENCIES)

        st.markdown("")
        render_overview_row("Crypto", CRYPTO)
        st.markdown("---")

        # ── Ticker detail ────────────────────────────────────────────────────
        q = fetch_realtime_quote(ticker)
        if q:
            price      = q["price"]
            change     = q["change"]
            change_pct = q["change_pct"]
            arrow      = "▲" if change >= 0 else "▼"
            source     = q.get("source", "")

            c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
            c1.markdown(f"### {ticker}")
            c1.caption(source)
            price_fmt = f"{price:,.4f}" if price < 10 else f"{price:,.2f}"
            c2.metric("Price",    price_fmt)
            c3.metric("Change",   f"{arrow} {abs(change_pct):.2f}%",
                      delta=f"{change:+.4f}" if price < 10 else f"{change:+.2f}",
                      delta_color="normal" if change >= 0 else "inverse")
            c4.metric("Day High", f"{q['high']:,.2f}" if q.get("high") else "—")
            c5.metric("Day Low",  f"{q['low']:,.2f}"  if q.get("low")  else "—")
        else:
            st.warning(f"Could not fetch data for **{ticker}**. Check the ticker symbol.")

        # ── Chart ────────────────────────────────────────────────────────────
        period, interval = TIMEFRAMES[timeframe]
        with st.spinner("Loading chart…"):
            df = fetch_history(ticker, period, interval)

        if df.empty:
            st.error("No historical data returned. Try a different ticker or timeframe.")
            return

        st.plotly_chart(build_main_chart(df, ticker, overlays, show_volume),
                        use_container_width=True)

        # ── Indicators ───────────────────────────────────────────────────────
        if show_rsi and show_macd:
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(build_rsi_chart(df),  use_container_width=True)
            with c2: st.plotly_chart(build_macd_chart(df), use_container_width=True)
        elif show_rsi:
            st.plotly_chart(build_rsi_chart(df),  use_container_width=True)
        elif show_macd:
            st.plotly_chart(build_macd_chart(df), use_container_width=True)

        # ── Summary stats ────────────────────────────────────────────────────
        with st.expander("Summary statistics"):
            last = df.tail(1).iloc[0]
            stats = {
                "Open":   f"{last['Open']:.4f}"  if last['Open']  < 10 else f"{last['Open']:,.2f}",
                "High":   f"{last['High']:.4f}"  if last['High']  < 10 else f"{last['High']:,.2f}",
                "Low":    f"{last['Low']:.4f}"   if last['Low']   < 10 else f"{last['Low']:,.2f}",
                "Close":  f"{last['Close']:.4f}" if last['Close'] < 10 else f"{last['Close']:,.2f}",
                "Volume": fmt_large(last.get("Volume")),
            }
            if "RSI" in df:
                v = df["RSI"].dropna().iloc[-1]
                zone = "Overbought 🔴" if v > 70 else "Oversold 🟢" if v < 30 else "Neutral ⚪"
                stats["RSI (14)"] = f"{v:.2f}  —  {zone}"
            if "MACD" in df:
                m   = df["MACD"].dropna().iloc[-1]
                sig = df["MACD_signal"].dropna().iloc[-1]
                stats["MACD"] = f"{m:.4f}  (Signal: {sig:.4f})"
            if "SMA20" in df:
                stats["SMA 20"] = f"{df['SMA20'].dropna().iloc[-1]:,.2f}"
            if "SMA50" in df:
                stats["SMA 50"] = f"{df['SMA50'].dropna().iloc[-1]:,.2f}"
            st.table(pd.DataFrame(stats.items(), columns=["Indicator", "Value"]).set_index("Indicator"))

        # ── Raw data ─────────────────────────────────────────────────────────
        with st.expander("Raw OHLCV data (last 50 rows)"):
            display_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            st.dataframe(
                df[display_cols].tail(50).sort_index(ascending=False).round(2),
                use_container_width=True,
            )

        # ── Auto-refresh ─────────────────────────────────────────────────────
        if auto_refresh:
            time.sleep(30)
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()

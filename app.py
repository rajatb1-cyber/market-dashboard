import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta as ta_lib
from datetime import datetime, timedelta
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
INDICES = {
    "S&P 500":      "^GSPC",
    "NASDAQ":       "^IXIC",
    "Dow Jones":    "^DJI",
    "Russell 2000": "^RUT",
    "VIX":          "^VIX",
}

POPULAR_TICKERS = [
    "^GSPC", "^IXIC", "^DJI", "^RUT",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD",
]

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
    "up":         "#26A69A",
    "down":       "#EF5350",
    "neutral":    "#9E9E9E",
    "volume_up":  "rgba(38,166,154,0.4)",
    "volume_down":"rgba(239,83,80,0.4)",
    "sma20":      "#FF9800",
    "sma50":      "#2196F3",
    "ema20":      "#CE93D8",
    "bb_upper":   "rgba(100,181,246,0.7)",
    "bb_lower":   "rgba(100,181,246,0.7)",
    "bb_fill":    "rgba(100,181,246,0.07)",
}

# ── Data helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_quote(ticker: str) -> dict:
    """Fetch current price info for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = info.last_price
        prev  = info.previous_close
        change     = price - prev
        change_pct = (change / prev) * 100 if prev else 0
        return {
            "price":      price,
            "prev_close": prev,
            "change":     change,
            "change_pct": change_pct,
            "high":       info.day_high,
            "low":        info.day_low,
            "volume":     info.last_volume,
        }
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch OHLCV history and compute technical indicators."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return df

    # Flatten multi-level columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)

    close = df["Close"].squeeze()

    # Moving averages
    df["SMA20"] = ta_lib.trend.SMAIndicator(close, window=20).sma_indicator()
    df["SMA50"] = ta_lib.trend.SMAIndicator(close, window=50).sma_indicator()
    df["EMA20"] = ta_lib.trend.EMAIndicator(close, window=20).ema_indicator()

    # Bollinger Bands
    bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_mid"]   = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()

    # RSI
    df["RSI"] = ta_lib.momentum.RSIIndicator(close, window=14).rsi()

    # MACD
    macd = ta_lib.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_hist"]   = macd.macd_diff()
    df["MACD_signal"] = macd.macd_signal()

    return df


# ── Chart builders ─────────────────────────────────────────────────────────────
def build_main_chart(df: pd.DataFrame, ticker: str, overlays: list,
                     show_volume: bool) -> go.Figure:
    """Candlestick chart with optional overlays and volume."""
    row_heights = [0.7, 0.3] if show_volume else [1.0]
    rows = 2 if show_volume else 1

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    # Candlestick
    colors_up   = [COLORS["up"]   if c >= o else COLORS["down"]
                   for o, c in zip(df["Open"], df["Close"])]
    colors_down = colors_up

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

    # Overlays
    if "SMA 20" in overlays and "SMA20" in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA20"], name="SMA 20",
            line=dict(color=COLORS["sma20"], width=1.5),
        ), row=1, col=1)

    if "SMA 50" in overlays and "SMA50" in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["SMA50"], name="SMA 50",
            line=dict(color=COLORS["sma50"], width=1.5),
        ), row=1, col=1)

    if "EMA 20" in overlays and "EMA20" in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA20"], name="EMA 20",
            line=dict(color=COLORS["ema20"], width=1.5, dash="dot"),
        ), row=1, col=1)

    if "Bollinger Bands" in overlays and "BB_upper" in df:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color=COLORS["bb_upper"], width=1),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color=COLORS["bb_lower"], width=1),
            fill="tonexty", fillcolor=COLORS["bb_fill"],
        ), row=1, col=1)

    # Volume
    if show_volume and "Volume" in df:
        vol_colors = [
            COLORS["volume_up"] if c >= o else COLORS["volume_down"]
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1,
                         tickfont=dict(size=10))

    fig.update_layout(
        title=dict(text=ticker, font=dict(size=18)),
        xaxis_rangeslider_visible=False,
        height=480,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#FAFAFA"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(
        gridcolor="#2D333B", zeroline=False,
        showspikes=True, spikecolor="#555", spikethickness=1,
    )
    fig.update_yaxes(
        gridcolor="#2D333B", zeroline=False,
        showspikes=True, spikecolor="#555", spikethickness=1,
    )
    return fig


def build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if "RSI" not in df:
        return fig

    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.1)",
                  line_width=0, annotation_text="Overbought",
                  annotation_position="top left",
                  annotation=dict(font_color="#EF5350", font_size=11))
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.1)",
                  line_width=0, annotation_text="Oversold",
                  annotation_position="bottom left",
                  annotation=dict(font_color="#26A69A", font_size=11))
    fig.add_hline(y=70, line_dash="dash", line_color="#EF5350", line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26A69A", line_width=1)
    fig.add_hline(y=50, line_dash="dot",  line_color="#666",     line_width=1)

    rsi_colors = [
        COLORS["up"] if v <= 30 else COLORS["down"] if v >= 70 else "#00D4FF"
        for v in df["RSI"].fillna(50)
    ]
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI(14)",
        line=dict(color="#00D4FF", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)",
    ))

    fig.update_layout(
        title=dict(text="RSI (14)", font=dict(size=14)),
        height=220,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#FAFAFA"),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 100], gridcolor="#2D333B", zeroline=False),
        xaxis=dict(gridcolor="#2D333B", zeroline=False),
        showlegend=False,
        hovermode="x unified",
    )
    return fig


def build_macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if "MACD" not in df:
        return fig

    hist_colors = [
        COLORS["up"] if v >= 0 else COLORS["down"]
        for v in df["MACD_hist"].fillna(0)
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_hist"], name="Histogram",
        marker_color=hist_colors, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], name="MACD",
        line=dict(color="#00D4FF", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_signal"], name="Signal",
        line=dict(color="#FF9800", width=1.8),
    ))
    fig.add_hline(y=0, line_color="#555", line_width=1)

    fig.update_layout(
        title=dict(text="MACD (12, 26, 9)", font=dict(size=14)),
        height=220,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        font=dict(color="#FAFAFA"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(gridcolor="#2D333B", zeroline=False),
        yaxis=dict(gridcolor="#2D333B", zeroline=False),
        hovermode="x unified",
    )
    return fig


# ── UI helpers ─────────────────────────────────────────────────────────────────
def delta_color(val: float) -> str:
    return "normal" if val >= 0 else "inverse"


def fmt_large(n: float) -> str:
    if n is None:
        return "—"
    if abs(n) >= 1e12:
        return f"{n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    return f"{n:,.0f}"


def render_market_overview():
    st.subheader("Market Overview")
    cols = st.columns(len(INDICES))
    for col, (name, symbol) in zip(cols, INDICES.items()):
        q = fetch_quote(symbol)
        if not q:
            col.metric(label=name, value="—")
            continue
        price = q["price"]
        delta = q["change_pct"]
        arrow = "▲" if delta >= 0 else "▼"
        col.metric(
            label=f"{name}  ({symbol})",
            value=f"{price:,.2f}" if price else "—",
            delta=f"{arrow} {abs(delta):.2f}%  ({q['change']:+.2f})",
            delta_color="normal" if delta >= 0 else "inverse",
        )


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    st.sidebar.title("📈 Market Dashboard")
    st.sidebar.markdown("---")

    custom = st.sidebar.text_input(
        "Custom ticker", placeholder="e.g. AAPL, BTC-USD"
    ).upper().strip()

    selected_ticker = st.sidebar.selectbox(
        "Select ticker",
        options=POPULAR_TICKERS,
        format_func=lambda x: x,
    )

    ticker = custom if custom else selected_ticker

    timeframe = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=2)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Chart Overlays**")
    overlays = []
    if st.sidebar.checkbox("SMA 20",          value=True):  overlays.append("SMA 20")
    if st.sidebar.checkbox("SMA 50",          value=True):  overlays.append("SMA 50")
    if st.sidebar.checkbox("EMA 20",          value=False): overlays.append("EMA 20")
    if st.sidebar.checkbox("Bollinger Bands", value=False): overlays.append("Bollinger Bands")
    show_volume = st.sidebar.checkbox("Volume bars", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Indicators**")
    show_rsi  = st.sidebar.checkbox("RSI (14)",          value=True)
    show_macd = st.sidebar.checkbox("MACD (12, 26, 9)", value=True)

    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    st.sidebar.caption("Data: Yahoo Finance via yfinance")

    return ticker, timeframe, overlays, show_volume, show_rsi, show_macd, auto_refresh


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ticker, timeframe, overlays, show_volume, show_rsi, show_macd, auto_refresh = render_sidebar()

    # ── Market overview ──────────────────────────────────────────────────────
    render_market_overview()
    st.markdown("---")

    # ── Ticker detail header ─────────────────────────────────────────────────
    q = fetch_quote(ticker)
    if q:
        price      = q["price"]
        change     = q["change"]
        change_pct = q["change_pct"]
        color      = COLORS["up"] if change >= 0 else COLORS["down"]
        arrow      = "▲" if change >= 0 else "▼"

        c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 2, 2])
        c1.markdown(f"### {ticker}")
        c2.metric("Price",  f"{price:,.4f}" if price < 10 else f"{price:,.2f}")
        c3.metric("Change", f"{arrow} {abs(change_pct):.2f}%",
                  delta=f"{change:+.2f}",
                  delta_color="normal" if change >= 0 else "inverse")
        c4.metric("Day High",  f"{q['high']:,.2f}" if q.get("high") else "—")
        c5.metric("Day Low",   f"{q['low']:,.2f}"  if q.get("low")  else "—")
    else:
        st.warning(f"Could not fetch data for **{ticker}**. Check the ticker symbol.")

    # ── Chart ────────────────────────────────────────────────────────────────
    period, interval = TIMEFRAMES[timeframe]
    with st.spinner("Loading chart data…"):
        df = fetch_history(ticker, period, interval)

    if df.empty:
        st.error("No historical data returned. Try a different ticker or timeframe.")
        return

    st.plotly_chart(
        build_main_chart(df, ticker, overlays, show_volume),
        use_container_width=True,
    )

    # ── Technical indicator charts ───────────────────────────────────────────
    if show_rsi or show_macd:
        ind_cols = st.columns(2) if (show_rsi and show_macd) else [st.container()]

        if show_rsi and show_macd:
            with ind_cols[0]:
                st.plotly_chart(build_rsi_chart(df),  use_container_width=True)
            with ind_cols[1]:
                st.plotly_chart(build_macd_chart(df), use_container_width=True)
        elif show_rsi:
            st.plotly_chart(build_rsi_chart(df),  use_container_width=True)
        else:
            st.plotly_chart(build_macd_chart(df), use_container_width=True)

    # ── Summary stats table ──────────────────────────────────────────────────
    with st.expander("Summary statistics"):
        last = df.tail(1).iloc[0]
        stats = {
            "Open":   f"{last['Open']:.4f}"   if last['Open']   < 10 else f"{last['Open']:,.2f}",
            "High":   f"{last['High']:.4f}"   if last['High']   < 10 else f"{last['High']:,.2f}",
            "Low":    f"{last['Low']:.4f}"    if last['Low']    < 10 else f"{last['Low']:,.2f}",
            "Close":  f"{last['Close']:.4f}"  if last['Close']  < 10 else f"{last['Close']:,.2f}",
            "Volume": fmt_large(last.get("Volume")),
        }
        if "RSI" in df:
            rsi_val = df["RSI"].dropna().iloc[-1]
            zone = "Overbought 🔴" if rsi_val > 70 else "Oversold 🟢" if rsi_val < 30 else "Neutral ⚪"
            stats["RSI (14)"] = f"{rsi_val:.2f}  —  {zone}"
        if "MACD" in df and "MACD_signal" in df:
            m   = df["MACD"].dropna().iloc[-1]
            sig = df["MACD_signal"].dropna().iloc[-1]
            stats["MACD"] = f"{m:.4f}  (Signal: {sig:.4f})"
        if "SMA20" in df:
            stats["SMA 20"] = f"{df['SMA20'].dropna().iloc[-1]:,.2f}"
        if "SMA50" in df:
            stats["SMA 50"] = f"{df['SMA50'].dropna().iloc[-1]:,.2f}"

        st.table(pd.DataFrame(stats.items(), columns=["Indicator", "Value"]).set_index("Indicator"))

    # ── Raw data ─────────────────────────────────────────────────────────────
    with st.expander("Raw OHLCV data (last 50 rows)"):
        display_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        st.dataframe(
            df[display_cols].tail(50).sort_index(ascending=False).round(2),
            use_container_width=True,
        )

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(60)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()

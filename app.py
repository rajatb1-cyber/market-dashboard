import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta as ta_lib
from datetime import datetime
import time
import finnhub

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Finnhub client ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_finnhub():
    key = st.secrets.get("FINNHUB_KEY", "")
    return finnhub.Client(api_key=key) if key else None

# ── Market data definitions ────────────────────────────────────────────────────
INDICES = {
    "S&P 500":      "^GSPC",
    "NASDAQ":       "^IXIC",
    "Dow Jones":    "^DJI",
    "Russell 2000": "^RUT",
    "VIX":          "^VIX",
}

BONDS = {
    "US 2Y":  "^IRX",
    "US 5Y":  "^FVX",
    "US 10Y": "^TNX",
    "US 30Y": "^TYX",
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

POPULAR_TICKERS = [
    "^GSPC", "^IXIC", "^DJI", "^RUT",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "SPY", "QQQ", "GLD", "BTC-USD", "ETH-USD",
    "^TNX", "GC=F", "CL=F", "EURUSD=X",
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
    "up":          "#26A69A",
    "down":        "#EF5350",
    "volume_up":   "rgba(38,166,154,0.4)",
    "volume_down": "rgba(239,83,80,0.4)",
    "sma20":       "#FF9800",
    "sma50":       "#2196F3",
    "ema20":       "#CE93D8",
    "bb_upper":    "rgba(100,181,246,0.7)",
    "bb_lower":    "rgba(100,181,246,0.7)",
    "bb_fill":     "rgba(100,181,246,0.07)",
}

# ── Data helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def fetch_realtime_quote(ticker: str) -> dict:
    """Real-time quote via Finnhub for stocks; fallback to yfinance."""
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
    """Quote via yfinance (15-min delay for stocks)."""
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
    """OHLCV + technical indicators via yfinance."""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)

    close = df["Close"].squeeze()

    df["SMA20"] = ta_lib.trend.SMAIndicator(close, window=20).sma_indicator()
    df["SMA50"] = ta_lib.trend.SMAIndicator(close, window=50).sma_indicator()
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
            line=dict(color=COLORS["sma20"], width=1.5)), row=1, col=1)
    if "SMA 50" in overlays and "SMA50" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50",
            line=dict(color=COLORS["sma50"], width=1.5)), row=1, col=1)
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
        title=dict(text=ticker, font=dict(size=18)),
        xaxis_rangeslider_visible=False,
        height=480,
        paper_bgcolor="#0E1117", plot_bgcolor="#161B22",
        font=dict(color="#FAFAFA"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#2D333B", zeroline=False,
                     showspikes=True, spikecolor="#555", spikethickness=1)
    fig.update_yaxes(gridcolor="#2D333B", zeroline=False,
                     showspikes=True, spikecolor="#555", spikethickness=1)
    return fig


def build_rsi_chart(df):
    fig = go.Figure()
    if "RSI" not in df:
        return fig
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.1)", line_width=0,
                  annotation_text="Overbought", annotation_position="top left",
                  annotation=dict(font_color="#EF5350", font_size=11))
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.1)", line_width=0,
                  annotation_text="Oversold", annotation_position="bottom left",
                  annotation=dict(font_color="#26A69A", font_size=11))
    fig.add_hline(y=70, line_dash="dash", line_color="#EF5350", line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26A69A", line_width=1)
    fig.add_hline(y=50, line_dash="dot",  line_color="#666",    line_width=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI(14)",
        line=dict(color="#00D4FF", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"))
    fig.update_layout(
        title=dict(text="RSI (14)", font=dict(size=14)),
        height=220, paper_bgcolor="#0E1117", plot_bgcolor="#161B22",
        font=dict(color="#FAFAFA"), margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 100], gridcolor="#2D333B", zeroline=False),
        xaxis=dict(gridcolor="#2D333B", zeroline=False),
        showlegend=False, hovermode="x unified",
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
        line=dict(color="#00D4FF", width=1.8)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
        line=dict(color="#FF9800", width=1.8)))
    fig.add_hline(y=0, line_color="#555", line_width=1)
    fig.update_layout(
        title=dict(text="MACD (12, 26, 9)", font=dict(size=14)),
        height=220, paper_bgcolor="#0E1117", plot_bgcolor="#161B22",
        font=dict(color="#FAFAFA"), margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="top", y=-0.15,
                    xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(gridcolor="#2D333B", zeroline=False),
        yaxis=dict(gridcolor="#2D333B", zeroline=False),
        hovermode="x unified",
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
    selected = st.sidebar.selectbox("Select ticker", POPULAR_TICKERS)
    ticker = custom if custom else selected

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

    # ── Market overview ──────────────────────────────────────────────────────
    st.subheader("Market Overview")
    render_overview_row("Indices",     INDICES)
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

    # ── Ticker detail ────────────────────────────────────────────────────────
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

    # ── Chart ────────────────────────────────────────────────────────────────
    period, interval = TIMEFRAMES[timeframe]
    with st.spinner("Loading chart…"):
        df = fetch_history(ticker, period, interval)

    if df.empty:
        st.error("No historical data returned. Try a different ticker or timeframe.")
        return

    st.plotly_chart(build_main_chart(df, ticker, overlays, show_volume),
                    use_container_width=True)

    # ── Indicators ───────────────────────────────────────────────────────────
    if show_rsi and show_macd:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(build_rsi_chart(df),  use_container_width=True)
        with c2: st.plotly_chart(build_macd_chart(df), use_container_width=True)
    elif show_rsi:
        st.plotly_chart(build_rsi_chart(df),  use_container_width=True)
    elif show_macd:
        st.plotly_chart(build_macd_chart(df), use_container_width=True)

    # ── Summary stats ────────────────────────────────────────────────────────
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

    # ── Raw data ─────────────────────────────────────────────────────────────
    with st.expander("Raw OHLCV data (last 50 rows)"):
        display_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        st.dataframe(
            df[display_cols].tail(50).sort_index(ascending=False).round(2),
            use_container_width=True,
        )

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(30)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()

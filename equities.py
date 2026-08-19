"""
Equities tab — four fixed tables: Global Indices, SPX Sectors,
SPX Style Box, SPX Top 20 Market Cap.
"""

from datetime import date, timedelta
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf

from watchlist import (
    ALL_COLUMNS, CHART_TIMEFRAMES,
    fetch_batch, fetch_chart_data,
    compute_row, build_display_df, style_table,
    build_instrument_chart,
)

# ── Instrument lists ───────────────────────────────────────────────────────────

_INDICES = [
    {"name": "S&P 500",       "ticker": "^GSPC",     "class": "US"},
    {"name": "NASDAQ 100",    "ticker": "^NDX",      "class": "US"},
    {"name": "Dow Jones",     "ticker": "^DJI",      "class": "US"},
    {"name": "Russell 2000",  "ticker": "^RUT",      "class": "US"},
    {"name": "Euro Stoxx 50", "ticker": "^STOXX50E", "class": "Europe"},
    {"name": "FTSE 100",      "ticker": "^FTSE",     "class": "Europe"},
    {"name": "DAX",           "ticker": "^GDAXI",    "class": "Europe"},
    {"name": "CAC 40",        "ticker": "^FCHI",     "class": "Europe"},
    {"name": "Nikkei 225",    "ticker": "^N225",     "class": "Asia"},
    {"name": "Hang Seng",     "ticker": "^HSI",      "class": "Asia"},
    {"name": "NIFTY 50",      "ticker": "^NSEI",     "class": "Asia"},
    {"name": "KOSPI",         "ticker": "^KS11",     "class": "Asia"},
    {"name": "ASX 200",       "ticker": "^AXJO",     "class": "Asia"},
]

_SECTORS = [
    {"name": "Technology",      "ticker": "XLK",  "class": "Sector"},
    {"name": "Communication",   "ticker": "XLC",  "class": "Sector"},
    {"name": "Consumer Disc",   "ticker": "XLY",  "class": "Sector"},
    {"name": "Consumer Staples","ticker": "XLP",  "class": "Sector"},
    {"name": "Financials",      "ticker": "XLF",  "class": "Sector"},
    {"name": "Healthcare",      "ticker": "XLV",  "class": "Sector"},
    {"name": "Industrials",     "ticker": "XLI",  "class": "Sector"},
    {"name": "Energy",          "ticker": "XLE",  "class": "Sector"},
    {"name": "Materials",       "ticker": "XLB",  "class": "Sector"},
    {"name": "Real Estate",     "ticker": "XLRE", "class": "Sector"},
    {"name": "Utilities",       "ticker": "XLU",  "class": "Sector"},
]

_FACTORS = [
    # Core factor indices
    {"name": "Momentum",            "ticker": "MTUM", "class": "Core Factor"},
    {"name": "Quality",             "ticker": "QUAL", "class": "Core Factor"},
    {"name": "Value",               "ticker": "VLUE", "class": "Core Factor"},
    {"name": "Growth",              "ticker": "IVW",  "class": "Core Factor"},
    {"name": "Low Volatility",      "ticker": "USMV", "class": "Core Factor"},
    {"name": "High Beta",           "ticker": "SPHB", "class": "Core Factor"},
    {"name": "Dividend Aristocrats","ticker": "NOBL", "class": "Core Factor"},
    # Weighting-based factors
    {"name": "Equal Weight",        "ticker": "RSP",  "class": "Weighting"},
    {"name": "Revenue-Weighted",    "ticker": "RWL",  "class": "Weighting"},
    {"name": "Buyback",             "ticker": "PKW",  "class": "Weighting"},
]

_TOP20 = [
    {"name": "Apple",       "ticker": "AAPL",  "class": "Tech"},
    {"name": "NVIDIA",      "ticker": "NVDA",  "class": "Tech"},
    {"name": "Microsoft",   "ticker": "MSFT",  "class": "Tech"},
    {"name": "Amazon",      "ticker": "AMZN",  "class": "Consumer"},
    {"name": "Alphabet",    "ticker": "GOOGL", "class": "Tech"},
    {"name": "Meta",        "ticker": "META",  "class": "Tech"},
    {"name": "Tesla",       "ticker": "TSLA",  "class": "Consumer"},
    {"name": "Broadcom",    "ticker": "AVGO",  "class": "Tech"},
    {"name": "Berkshire B", "ticker": "BRK-B", "class": "Finance"},
    {"name": "JPMorgan",    "ticker": "JPM",   "class": "Finance"},
    {"name": "Eli Lilly",   "ticker": "LLY",   "class": "Healthcare"},
    {"name": "Visa",        "ticker": "V",     "class": "Finance"},
    {"name": "UnitedHealth","ticker": "UNH",   "class": "Healthcare"},
    {"name": "Exxon Mobil", "ticker": "XOM",   "class": "Energy"},
    {"name": "Costco",      "ticker": "COST",  "class": "Consumer"},
    {"name": "Mastercard",  "ticker": "MA",    "class": "Finance"},
    {"name": "Oracle",      "ticker": "ORCL",  "class": "Tech"},
    {"name": "Walmart",     "ticker": "WMT",   "class": "Consumer"},
    {"name": "Netflix",     "ticker": "NFLX",  "class": "Tech"},
    {"name": "Home Depot",  "ticker": "HD",    "class": "Consumer"},
]

_DEFAULT_COLS = ["Price", "Change %", "Weekly %", "Monthly %", "YTD %", "52W High", "52W Low", "RSI (14)"]

# ── P/E data ───────────────────────────────────────────────────────────────────

# Index tickers carry no P/E on Yahoo — map to a tracking ETF for trailing P/E
_INDEX_PE_PROXY = {
    "^GSPC":     "SPY",
    "^NDX":      "QQQ",
    "^DJI":      "DIA",
    "^RUT":      "IWM",
    "^STOXX50E": "FEZ",
    "^FTSE":     "EWU",
    "^GDAXI":    "EWG",
    "^FCHI":     "EWQ",
    "^N225":     "EWJ",
    "^HSI":      "EWH",
    "^NSEI":     "INDA",
    "^KS11":     "EWY",
    "^AXJO":     "EWA",
}

# GICS sector name → sector ETF ticker
_GICS_TO_SECTOR_ETF = {
    "Information Technology":  "XLK",
    "Communication Services":  "XLC",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Industrials":             "XLI",
    "Energy":                  "XLE",
    "Materials":               "XLB",
    "Real Estate":             "XLRE",
    "Utilities":               "XLU",
}
_SECTOR_ETF_TO_GICS = {v: k for k, v in _GICS_TO_SECTOR_ETF.items()}


@st.cache_data(ttl=3600 * 6)
def _fetch_pe_both(ticker: str) -> tuple:
    """Returns (trailingPE, forwardPE). Cached 6 hours."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("trailingPE"), info.get("forwardPE")
    except Exception:
        return None, None


def _trailing_pe_val(ticker: str) -> float:
    lookup = _INDEX_PE_PROXY.get(ticker, ticker)
    trail, _ = _fetch_pe_both(lookup)
    return float(trail) if trail else float("nan")


# ── Constituent-based forward P/E ──────────────────────────────────────────────

@st.cache_data(ttl=3600 * 24)
def _fetch_sp500_constituents() -> pd.DataFrame:
    """S&P 500 tickers + GICS sectors from Wikipedia. Cached 24h."""
    try:
        import requests, io as _io
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=20,
        )
        df = pd.read_html(_io.StringIO(r.text))[0][["Symbol", "GICS Sector"]].copy()
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df
    except Exception:
        return pd.DataFrame(columns=["Symbol", "GICS Sector"])


@st.cache_data(ttl=3600 * 24)
def _fetch_ndx100_tickers() -> tuple:
    """NASDAQ-100 tickers from Wikipedia. Cached 24h."""
    try:
        import requests, io as _io
        r = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=20,
        )
        for t in pd.read_html(_io.StringIO(r.text)):
            if "Ticker" in t.columns:
                return tuple(t["Ticker"].str.replace(".", "-", regex=False).tolist())
    except Exception:
        pass
    return ()


@st.cache_data(ttl=3600 * 6)
def _fetch_constituent_pe(tickers: tuple) -> dict:
    """Parallel-fetch forwardPE + marketCap for a list of tickers. Cached 6h."""
    from concurrent.futures import ThreadPoolExecutor

    def _one(tkr):
        try:
            info = yf.Ticker(tkr).info
            return tkr, info.get("forwardPE"), info.get("marketCap")
        except Exception:
            return tkr, None, None

    with ThreadPoolExecutor(max_workers=20) as ex:
        rows = list(ex.map(_one, tickers))
    return {tkr: {"fpe": fpe, "mkt": mkt} for tkr, fpe, mkt in rows}


def _aggregate_pe(pe_data: dict, ticker_set: set) -> float:
    """Cap-weighted aggregate forward P/E for a subset of tickers."""
    total_mkt = total_earn = 0.0
    for tkr, d in pe_data.items():
        if tkr not in ticker_set:
            continue
        fpe, mkt = d.get("fpe"), d.get("mkt")
        if fpe and mkt and fpe > 0 and mkt > 0:
            total_mkt += mkt
            total_earn += mkt / fpe
    if total_earn == 0:
        return float("nan")
    return total_mkt / total_earn


# ── Table renderer ─────────────────────────────────────────────────────────────

def _render_table(instruments: list, batch: dict, active_cols: list,
                  title: str, table_key: str,
                  fwd_pe_map: dict | None = None) -> dict | None:
    """Renders one instrument table. Returns the selected instrument dict or None."""
    st.markdown(f"##### {title}")
    rows    = [compute_row(inst, batch.get(inst["ticker"]), active_cols) for inst in instruments]
    df_rows = pd.DataFrame(rows)

    # Keep all values numeric so st.dataframe column-click sort works correctly.
    df_display = pd.DataFrame()
    df_display["Name"]  = df_rows["Name"]
    df_display["Class"] = df_rows["Class"]
    for c in active_cols:
        if c in df_rows.columns:
            df_display[c] = pd.to_numeric(df_rows[c], errors="coerce")

    df_display["P/E (T)"] = [_trailing_pe_val(i["ticker"]) for i in instruments]
    df_display["Fwd P/E"] = [
        fwd_pe_map.get(i["ticker"], float("nan")) if fwd_pe_map is not None
        else float("nan")
        for i in instruments
    ]

    styled = style_table(df_display, active_cols)

    # Display formatters — underlying values stay numeric for sorting
    def _nan(v, s): return "—" if pd.isna(v) else s
    def _pct(v):
        try: return _nan(v := float(v), f"{v:+.2f}%")
        except: return "—"
    def _chg(v):
        try: return _nan(v := float(v), f"{v:+.4f}" if abs(v) < 1 else f"{v:+.2f}")
        except: return "—"
    def _price(v):
        try:
            v = float(v)
            if pd.isna(v): return "—"
            if abs(v) < 10:    return f"{v:.4f}"
            if abs(v) < 10000: return f"{v:,.2f}"
            return f"{v:,.0f}"
        except: return "—"
    def _rsi(v):
        try: return _nan(v := float(v), f"{v:.1f}")
        except: return "—"
    def _vol(v):
        try:
            v = float(v)
            if pd.isna(v): return "—"
            if v >= 1e9: return f"{v/1e9:.1f}B"
            if v >= 1e6: return f"{v/1e6:.1f}M"
            if v >= 1e3: return f"{v/1e3:.0f}K"
            return f"{v:.0f}"
        except: return "—"
    def _pe(v):
        try: return _nan(v := float(v), f"{v:.1f}x")
        except: return "—"

    fmt: dict = {"P/E (T)": _pe, "Fwd P/E": _pe}
    for c in active_cols:
        if   c in ("Change %", "Weekly %", "Monthly %", "YTD %"): fmt[c] = _pct
        elif c == "Change":                                         fmt[c] = _chg
        elif c in ("RSI (14)", "RSI (30)"):                        fmt[c] = _rsi
        elif c in ("Price", "52W High", "52W Low"):                fmt[c] = _price
        elif c == "Volume":                                         fmt[c] = _vol
    styled = styled.format(fmt, na_rep="—")

    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=38 + 35 * len(df_display),
        key=table_key,
    )

    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        inst_name = df_display.iloc[selected_rows[0]]["Name"]
        return next((i for i in instruments if i["name"] == inst_name), None)
    return None


# ── Main render ────────────────────────────────────────────────────────────────

@st.fragment
def render_equities():
    st.markdown("### Equities")

    if "eq_selected" not in st.session_state:
        st.session_state.eq_selected = None

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([4, 2, 1])
    with ctrl_c1:
        active_cols = st.multiselect("Columns", ALL_COLUMNS,
                                     default=_DEFAULT_COLS, key="eq_cols")
        if not active_cols:
            active_cols = _DEFAULT_COLS
    with ctrl_c2:
        sort_col = st.selectbox("Sort by", ["Name", "Class"] + active_cols, key="eq_sort_col")
    with ctrl_c3:
        st.markdown("<div style='margin-top:26px'>", unsafe_allow_html=True)
        if st.button("⟳ Refresh", use_container_width=True, key="eq_refresh"):
            # Targeted refresh: only the caches this tab reads.
            from watchlist import _all_daily
            _all_daily.clear()                  # watchlist: shared daily OHLCV (fetch_batch)
            fetch_chart_data.clear()            # watchlist: selected-instrument chart bars
            _fetch_pe_both.clear()              # trailing/forward P/E
            _fetch_sp500_constituents.clear()   # S&P 500 constituents scrape
            _fetch_ndx100_tickers.clear()       # NDX-100 constituents scrape
            _fetch_constituent_pe.clear()       # constituent-level P/E
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    sort_asc = True

    # ── Fetch all tickers in one batch ────────────────────────────────────────
    all_instruments = _INDICES + _SECTORS + _FACTORS + _TOP20
    tickers = tuple(i["ticker"] for i in all_instruments)
    with st.spinner("Loading equities data…"):
        batch = fetch_batch(tickers)

    # ── Constituent-based forward P/E ─────────────────────────────────────────
    with st.spinner("Loading constituent P/E data (cached 6h)…"):
        sp500_df  = _fetch_sp500_constituents()
        ndx_tkrs  = _fetch_ndx100_tickers()
        all_const = tuple(set(sp500_df["Symbol"].tolist()) | set(ndx_tkrs))
        cpe       = _fetch_constituent_pe(all_const) if all_const else {}

    sp500_set = set(sp500_df["Symbol"].tolist())
    ndx_set   = set(ndx_tkrs)

    def _fwd_pe(ticker: str) -> float:
        # S&P 500 index / large-cap blend ETF
        if ticker in ("^GSPC", "IVV"):
            return _aggregate_pe(cpe, sp500_set)
        # NASDAQ-100
        if ticker in ("^NDX",):
            return _aggregate_pe(cpe, ndx_set)
        # S&P 500 sector ETFs — filter constituents by GICS sector
        if ticker in _SECTOR_ETF_TO_GICS:
            gics = _SECTOR_ETF_TO_GICS[ticker]
            sector_set = set(sp500_df.loc[sp500_df["GICS Sector"] == gics, "Symbol"])
            return _aggregate_pe(cpe, sector_set)
        # Individual stocks — pull from constituent cache if present, else direct fetch
        if ticker in cpe:
            fpe = cpe[ticker].get("fpe")
            return float(fpe) if fpe else float("nan")
        _, fwd = _fetch_pe_both(ticker)
        return float(fwd) if fwd else float("nan")

    fwd_pe_map = {i["ticker"]: _fwd_pe(i["ticker"]) for i in all_instruments}

    st.caption(
        f"Last updated: {datetime.now().strftime('%H:%M:%S')}  ·  Data: Yahoo Finance  ·  "
        "P/E (T) = trailing P/E (proxy ETF for indices)  ·  "
        "Fwd P/E = cap-weighted from constituents for S&P 500 / NASDAQ 100 / sectors; "
        "direct analyst estimate for stocks  ·  cached 6h"
    )
    st.markdown("<p style='font-size:11px;color:#94A3B8;margin-bottom:2px'>"
                "👆 Click any row to chart it</p>", unsafe_allow_html=True)

    # ── Four tables ───────────────────────────────────────────────────────────
    sel = None

    result = _render_table(_INDICES, batch, active_cols, "Global Indices", "eq_tbl_indices", fwd_pe_map)
    if result:
        sel = result

    st.markdown("")
    result = _render_table(_SECTORS, batch, active_cols, "S&P 500 — Sectors", "eq_tbl_sectors", fwd_pe_map)
    if result:
        sel = result

    st.markdown("")
    result = _render_table(_FACTORS, batch, active_cols, "S&P 500 — Factor Indices", "eq_tbl_factors", fwd_pe_map)
    if result:
        sel = result

    st.markdown("")
    result = _render_table(_TOP20, batch, active_cols, "S&P 500 — Top 20 by Market Cap", "eq_tbl_top20", fwd_pe_map)
    if result:
        sel = result

    if sel:
        st.session_state.eq_selected = sel

    # ── Inline chart ──────────────────────────────────────────────────────────
    chosen = st.session_state.eq_selected
    if chosen:
        st.markdown("---")
        st.markdown(
            f"<h4 style='color:#1B3A6B;margin-bottom:4px'>"
            f"{chosen['name']} "
            f"<span style='font-size:13px;color:#64748B;font-weight:400'>"
            f"({chosen['ticker']})</span></h4>",
            unsafe_allow_html=True,
        )

        tf_cols = st.columns([7, 1])
        with tf_cols[0]:
            tf = st.segmented_control(
                "Timeframe", options=list(CHART_TIMEFRAMES.keys()),
                default="3M", key=f"eq_tf_{chosen['ticker']}",
            )
        with tf_cols[1]:
            st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
            if st.button("✕ Close", key="eq_close_chart", use_container_width=True):
                st.session_state.eq_selected = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        opt_c1, opt_c2, opt_c3 = st.columns([2, 2, 4])
        with opt_c1:
            chart_type = st.selectbox("Chart type",
                                      ["Candlestick", "Line", "OHLC Bar"],
                                      key=f"eq_ct_{chosen['ticker']}")
        with opt_c2:
            rsi_period = st.number_input("RSI period", min_value=2, max_value=50,
                                         value=14, step=1,
                                         key=f"eq_rsi_{chosen['ticker']}")
        with opt_c3:
            overlays = st.multiselect(
                "Overlays",
                ["SMA 20", "SMA 50", "SMA 100", "SMA 200", "EMA 20", "Bollinger Bands"],
                default=["SMA 20", "SMA 50"],
                key=f"eq_ov_{chosen['ticker']}",
            )

        if tf == "Custom":
            d_c1, d_c2 = st.columns(2)
            with d_c1:
                start_date = st.date_input("From",
                                           value=date.today() - timedelta(days=180),
                                           max_value=date.today() - timedelta(days=2),
                                           key="eq_start")
            with d_c2:
                end_date = st.date_input("To", value=date.today(),
                                         min_value=start_date + timedelta(days=1),
                                         max_value=date.today(), key="eq_end")
            period, interval = None, "1d"
            start_str, end_str = str(start_date), str(end_date)
        else:
            period, interval = CHART_TIMEFRAMES[tf]
            start_str = end_str = None

        with st.spinner(f"Loading {chosen['name']} chart…"):
            df_chart = fetch_chart_data(
                chosen["ticker"], period, interval,
                rsi_period=int(rsi_period),
                start=start_str, end=end_str,
            )

        if not df_chart.empty:
            chart_key = (
                f"eq_{chosen['ticker']}_{tf}_{chart_type}_{rsi_period}"
                f"_{'_'.join(sorted(overlays or []))}"
                f"_{start_str}_{end_str}"
            )
            st.plotly_chart(
                build_instrument_chart(
                    df_chart, chosen["name"], chosen["ticker"],
                    chart_type=chart_type,
                    overlays=overlays,
                    rsi_period=int(rsi_period),
                ),
                use_container_width=True,
                key=chart_key,
            )
        else:
            st.warning("No chart data available for this instrument / timeframe.")

"""
Bloomberg-style watchlist — data fetching, table rendering, config persistence.
"""

import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import ta as ta_lib
import yfinance as yf
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist_config.json")

ALL_COLUMNS     = ["Price", "Change %", "Change", "Weekly %", "Monthly %", "RSI", "52W High", "52W Low", "Volume"]
DEFAULT_COLUMNS = ["Price", "Change %", "Weekly %", "Monthly %", "RSI", "52W High", "52W Low"]
ASSET_CLASSES   = ["Equity", "FX", "Rates", "Commodity", "Crypto", "Other"]

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

# Row background tints per asset class
CLASS_BG = {
    "Equity":    "#F0F5FF",
    "FX":        "#F0FDF4",
    "Rates":     "#FFFBEB",
    "Commodity": "#FFF4ED",
    "Crypto":    "#FAF5FF",
    "Other":     "#F8FAFC",
}

# Badge text colours per asset class
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
            # Back-fill any missing keys
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


# ── Data fetching ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)
def fetch_batch(tickers: tuple) -> dict:
    """Download 1 year of daily data for all tickers in one yfinance call."""
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


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return None


def compute_row(inst: dict, df: pd.DataFrame | None, columns: list) -> dict:
    row: dict = {"Name": inst["name"], "Class": inst["class"], "_ticker": inst["ticker"]}

    if df is None or df.empty:
        for c in columns:
            row[c] = None
        return row

    close = df["Close"].squeeze().dropna().astype(float)
    if close.empty:
        for c in columns:
            row[c] = None
        return row

    last = _safe_float(close.iloc[-1])
    if last is None:
        for c in columns:
            row[c] = None
        return row

    prev  = _safe_float(close.iloc[-2]) if len(close) > 1 else last
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
            ref = _safe_float(close.iloc[-6]) if len(close) > 5 else _safe_float(close.iloc[0])
            row[c] = ((last - ref) / ref * 100) if ref else None
        elif c == "Monthly %":
            ref = _safe_float(close.iloc[-22]) if len(close) > 21 else _safe_float(close.iloc[0])
            row[c] = ((last - ref) / ref * 100) if ref else None
        elif c == "RSI":
            try:
                rsi = ta_lib.momentum.RSIIndicator(close, window=14).rsi().dropna()
                row[c] = _safe_float(rsi.iloc[-1]) if not rsi.empty else None
            except Exception:
                row[c] = None
        elif c == "52W High":
            row[c] = _safe_float(close.max())
        elif c == "52W Low":
            row[c] = _safe_float(close.min())
        elif c == "Volume":
            try:
                vol = df["Volume"].squeeze().dropna()
                row[c] = _safe_float(vol.iloc[-1]) if not vol.empty else None
            except Exception:
                row[c] = None

    return row


# ── Formatting ─────────────────────────────────────────────────────────────────
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
    if abs(val) < 10:
        return f"{val:.4f}"
    if abs(val) < 10_000:
        return f"{val:,.2f}"
    return f"{val:,.0f}"


def _cell_font_color(val, col: str) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return MUTE
    if col in ("Change %", "Change", "Weekly %", "Monthly %"):
        return UP if val >= 0 else DOWN
    if col == "RSI":
        if val > 70: return DOWN
        if val < 30: return UP
    return "#1A202C"


# ── Plotly table ───────────────────────────────────────────────────────────────
def build_table(df_rows: pd.DataFrame, columns: list) -> go.Figure:
    display_cols = ["Name", "Class"] + columns

    # Collect per-column cell lists
    cell_vals   = {c: [] for c in display_cols}
    cell_colors = {c: [] for c in display_cols}
    cell_fonts  = {c: [] for c in display_cols}

    for _, row in df_rows.iterrows():
        cls    = row.get("Class", "Other")
        row_bg = CLASS_BG.get(cls, "#F8FAFC")

        for c in display_cols:
            raw = row.get(c)
            if c == "Name":
                cell_vals[c].append(f"<b>{raw}</b>" if raw else "—")
                cell_colors[c].append(row_bg)
                cell_fonts[c].append("#1A202C")
            elif c == "Class":
                cell_vals[c].append(str(raw) if raw else "—")
                cell_colors[c].append(row_bg)
                cell_fonts[c].append(CLASS_FG.get(cls, "#475569"))
            else:
                cell_vals[c].append(_fmt(raw, c))
                cell_colors[c].append(row_bg)
                cell_fonts[c].append(_cell_font_color(raw, c))

    col_widths = []
    for c in display_cols:
        if c == "Name":      col_widths.append(130)
        elif c == "Class":   col_widths.append(85)
        elif c == "Volume":  col_widths.append(80)
        elif c == "52W High": col_widths.append(95)
        elif c == "52W Low":  col_widths.append(95)
        else:                col_widths.append(90)

    align_header = ["left", "left"] + ["right"] * len(columns)
    align_cells  = align_header

    fig = go.Figure(data=[go.Table(
        columnwidth=col_widths,
        header=dict(
            values=[f"<b>{c}</b>" for c in display_cols],
            fill_color="#1B3A6B",
            font=dict(color="white", size=12, family="Inter, Segoe UI, sans-serif"),
            align=align_header,
            height=38,
            line_color="#2D5A8E",
        ),
        cells=dict(
            values=[cell_vals[c] for c in display_cols],
            fill_color=[cell_colors[c] for c in display_cols],
            font=dict(
                color=[cell_fonts[c] for c in display_cols],
                size=12,
                family="Inter, Segoe UI, sans-serif",
            ),
            align=align_cells,
            height=34,
            line_color="#E2E8F0",
        ),
    )])

    n_rows = len(df_rows)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#FFFFFF",
        height=max(200, 38 + 34 * n_rows + 16),
    )
    return fig


# ── Legend bar ────────────────────────────────────────────────────────────────
def render_legend():
    badges = " &nbsp; ".join(
        f'<span style="background:{CLASS_BG[c]};color:{CLASS_FG[c]};'
        f'border:1px solid {CLASS_FG[c]}33;border-radius:4px;'
        f'padding:2px 8px;font-size:11px;font-weight:600;">{c}</span>'
        for c in ASSET_CLASSES[:-1]
    )
    st.markdown(badges, unsafe_allow_html=True)


# ── Main render ────────────────────────────────────────────────────────────────
def render_watchlist():
    # ── Load config into session state once ──────────────────────────────────
    if "wl_config" not in st.session_state:
        st.session_state.wl_config = load_config()

    cfg         = st.session_state.wl_config
    instruments = cfg["instruments"]
    sel_cols    = cfg["columns"]

    # ── Control bar ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])

    with c1:
        new_cols = st.multiselect(
            "Columns", ALL_COLUMNS, default=sel_cols,
            help="Choose which metrics to display",
        )
        if set(new_cols) != set(sel_cols):
            cfg["columns"] = new_cols
            save_config(cfg)
            st.session_state.wl_config = cfg

    active_cols = new_cols if new_cols else sel_cols

    with c2:
        sort_options = ["Name", "Class"] + active_cols
        sort_col = st.selectbox("Sort by", sort_options)

    with c3:
        sort_dir = st.radio("Order", ["↑ Ascending", "↓ Descending"],
                            horizontal=True, label_visibility="visible")
        sort_asc = sort_dir.startswith("↑")

    with c4:
        st.markdown("<div style='margin-top:26px'>", unsafe_allow_html=True)
        if st.button("⟳ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    render_legend()
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)

    # ── Fetch and build rows ──────────────────────────────────────────────────
    tickers = tuple(i["ticker"] for i in instruments)

    with st.spinner("Loading watchlist data…"):
        batch = fetch_batch(tickers)

    rows = [compute_row(inst, batch.get(inst["ticker"]), active_cols) for inst in instruments]
    df_rows = pd.DataFrame(rows)

    # Sort
    if sort_col in df_rows.columns:
        df_rows = df_rows.sort_values(sort_col, ascending=sort_asc, na_position="last")

    # ── Render table ─────────────────────────────────────────────────────────
    st.plotly_chart(build_table(df_rows, active_cols), use_container_width=True)

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}  ·  "
               f"{len(instruments)} instruments  ·  Data: Yahoo Finance")

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
                    existing = [i["ticker"] for i in cfg["instruments"]]
                    if ticker not in existing:
                        cfg["instruments"].append(entry)
                        save_config(cfg)
                        st.session_state.wl_config = cfg
                        st.cache_data.clear()
                        st.success(f"✓ Added **{name}** ({ticker})")
                        st.rerun()
                    else:
                        st.warning(f"{ticker} is already in the watchlist.")
                else:
                    st.warning("Please fill in both name and ticker.")

    with col_remove:
        with st.expander("➖  Remove instruments"):
            names = [i["name"] for i in cfg["instruments"]]
            to_remove = st.multiselect("Select instruments to remove", names)
            if st.button("Remove selected", type="secondary", key="btn_remove"):
                if to_remove:
                    cfg["instruments"] = [i for i in cfg["instruments"]
                                          if i["name"] not in to_remove]
                    save_config(cfg)
                    st.session_state.wl_config = cfg
                    st.cache_data.clear()
                    st.success(f"Removed {len(to_remove)} instrument(s).")
                    st.rerun()
                else:
                    st.warning("Select at least one instrument to remove.")

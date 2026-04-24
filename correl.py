import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import yfinance as yf

from watchlist import (fetch_batch, load_config, save_config,
                       FRED_MAP, ECB_MAP, ALPHAVANTAGE_FX_MAP,
                       _fetch_fred_df, _fetch_ecb_df, _fetch_alphavantage_fx)

_DETAIL_DAYS = {
    "1M": 30, "3M": 91, "6M": 182,
    "1Y": 365, "2Y": 730, "5Y": 1825, "10Y": 3650,
}


@st.cache_data(ttl=300)
def _fetch_detail_series(ticker: str, start_str: str) -> pd.Series:
    """Daily close for one ticker going back to start_str (handles all sources)."""
    if ticker in FRED_MAP:
        df = _fetch_fred_df(FRED_MAP[ticker], start=start_str)
    elif ticker in ECB_MAP:
        df = _fetch_ecb_df(ECB_MAP[ticker], start=start_str)
    elif ticker in ALPHAVANTAGE_FX_MAP:
        from_sym, to_sym = ALPHAVANTAGE_FX_MAP[ticker]
        df = _fetch_alphavantage_fx(from_sym, to_sym, start=start_str)
    else:
        try:
            df = yf.download(ticker, start=start_str, interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[df["Close"].notna()]
            if not df.empty:
                idx = pd.DatetimeIndex(df.index)
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                df.index = idx.normalize()
                df = df[~df.index.duplicated(keep="last")]
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        return pd.Series(dtype=float)
    return df["Close"].squeeze().astype(float)


def _detail_charts(asset1: str, asset2: str,
                   name_to_inst: dict, name_to_ticker: dict):
    """Render the dual-axis price chart and rolling correlation chart."""
    st.markdown("---")
    st.markdown(f"**{asset1}  ×  {asset2}**")

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([4, 2, 2])
    with c1:
        tf = st.segmented_control(
            "Chart history",
            list(_DETAIL_DAYS.keys()) + ["Custom"],
            default="1Y",
            key="corr_detail_tf",
        )
    with c2:
        roll_window = st.number_input(
            "Rolling window (days)", min_value=5, max_value=252,
            value=30, step=5, key="corr_roll",
        )
    with c3:
        detail_mode = st.radio(
            "Correlate on",
            ["Returns", "Levels"],
            horizontal=True,
            key="corr_detail_mode",
        )

    if tf == "Custom":
        dc1, dc2 = st.columns(2)
        with dc1:
            start_d = st.date_input("From",
                                    value=date.today() - timedelta(days=365),
                                    key="corr_d_start")
        with dc2:
            end_d = st.date_input("To", value=date.today(), key="corr_d_end")
    else:
        end_d   = date.today()
        start_d = end_d - timedelta(days=_DETAIL_DAYS[tf])

    start_str = start_d.isoformat()
    tkr1 = name_to_ticker[asset1]
    tkr2 = name_to_ticker[asset2]
    cls1 = name_to_inst[asset1].get("class", "")
    cls2 = name_to_inst[asset2].get("class", "")

    with st.spinner("Loading detail data…"):
        s1 = _fetch_detail_series(tkr1, start_str)
        s2 = _fetch_detail_series(tkr2, start_str)

    if s1.empty or s2.empty:
        st.warning("Could not load data for one or both assets.")
        return

    s1 = s1[s1.index <= pd.Timestamp(end_d)]
    s2 = s2[s2.index <= pd.Timestamp(end_d)]

    # ── Chart 1: dual-axis price ───────────────────────────────────────────────
    fig_price = make_subplots(specs=[[{"secondary_y": True}]])

    fig_price.add_trace(
        go.Scatter(x=s1.index, y=s1.values, name=asset1,
                   line=dict(color="#0EA5E9", width=2)),
        secondary_y=False,
    )
    fig_price.add_trace(
        go.Scatter(x=s2.index, y=s2.values, name=asset2,
                   line=dict(color="#F59E0B", width=2)),
        secondary_y=True,
    )

    fig_price.update_layout(
        title=dict(text=f"{asset1}  vs  {asset2}",
                   font=dict(size=14, color="#1A202C")),
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        legend=dict(orientation="h", y=1.08, x=0,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#E2E8F0", borderwidth=1),
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
    )
    fig_price.update_xaxes(gridcolor="#E8EDF5", zeroline=False)
    fig_price.update_yaxes(title_text=asset1, gridcolor="#E8EDF5",
                           zeroline=False, secondary_y=False,
                           title_font=dict(color="#0EA5E9"),
                           tickfont=dict(color="#0EA5E9"))
    fig_price.update_yaxes(title_text=asset2, gridcolor="#E8EDF5",
                           zeroline=False, secondary_y=True,
                           title_font=dict(color="#F59E0B"),
                           tickfont=dict(color="#F59E0B"))

    st.plotly_chart(fig_price, use_container_width=True)

    # ── Rolling correlation ────────────────────────────────────────────────────
    if detail_mode == "Levels":
        r1 = s1
        r2 = s2
    else:
        r1 = s1.diff() if cls1 == "Rates" else s1.pct_change()
        r2 = s2.diff() if cls2 == "Rates" else s2.pct_change()

    combined     = pd.concat([r1.rename("a"), r2.rename("b")], axis=1).dropna()
    rolling_corr = (combined["a"].rolling(window=roll_window)
                    .corr(combined["b"]) * 100)

    clean = rolling_corr.dropna()
    last_corr = float(clean.iloc[-1]) if not clean.empty else np.nan
    pad   = max(3.0, (float(clean.max()) - float(clean.min())) * 0.05) if not clean.empty else 5.0
    y_lo  = float(clean.min()) - pad if not clean.empty else -100
    y_hi  = float(clean.max()) + pad if not clean.empty else 100

    line_color = "#059669" if last_corr >= 0 else "#DC2626"

    fig_roll = go.Figure()
    fig_roll.add_hrect(y0=0,   y1=y_hi, fillcolor="rgba(5,150,105,0.04)",  line_width=0)
    fig_roll.add_hrect(y0=y_lo, y1=0,   fillcolor="rgba(220,38,38,0.04)", line_width=0)
    fig_roll.add_hline(y=0, line_color="#94A3B8", line_width=1)

    fig_roll.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr.values,
        name=f"{roll_window}d rolling corr",
        line=dict(color=line_color, width=2),
        fill="tozeroy",
        fillcolor=("rgba(5,150,105,0.08)" if last_corr >= 0
                   else "rgba(220,38,38,0.08)"),
    ))

    fig_roll.update_layout(
        title=dict(
            text=(f"Rolling {roll_window}-day Correlation  "
                  f"<span style='font-size:13px;color:#64748B'>"
                  f"latest: {last_corr:.0f}%</span>"),
            font=dict(size=14, color="#1A202C"),
        ),
        height=280,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        yaxis=dict(range=[y_lo, y_hi], gridcolor="#E8EDF5",
                   zeroline=False, tickformat=".0f", ticksuffix="%"),
        xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
        showlegend=False,
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
    )
    st.plotly_chart(fig_roll, use_container_width=True)
    corr_basis = "outright levels" if detail_mode == "Levels" else "daily returns"
    st.caption(
        f"Price chart: outright levels on independent axes  ·  "
        f"Rolling correlation: {roll_window}-day window on {corr_basis}"
    )


def render_correl():
    cfg         = st.session_state.get("wl_config") or load_config()
    instruments = cfg["instruments"]

    name_to_inst   = {i["name"]: i           for i in instruments}
    name_to_ticker = {i["name"]: i["ticker"] for i in instruments}
    all_names      = list(name_to_ticker.keys())

    st.markdown("### Correlation Matrix")

    # ── Seed session state from saved config on first load ────────────────────
    if "corr_rows" not in st.session_state:
        saved = cfg.get("correl_rows", all_names[:min(5, len(all_names))])
        st.session_state["corr_rows"] = [n for n in saved if n in all_names]
    if "corr_cols" not in st.session_state:
        saved = cfg.get("correl_cols", all_names[:min(5, len(all_names))])
        st.session_state["corr_cols"] = [n for n in saved if n in all_names]

    # ── Asset selectors ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([4, 4, 1])
    with c1:
        row_names = st.multiselect(
            "Row assets (up to 10)", all_names,
            key="corr_rows",
        )
    with c2:
        col_names = st.multiselect(
            "Column assets (up to 10)", all_names,
            key="corr_cols",
        )
    with c3:
        st.markdown("<div style='margin-top:26px'>", unsafe_allow_html=True)
        if st.button("💾 Save Grid", use_container_width=True):
            cfg["correl_rows"] = st.session_state.get("corr_rows", [])
            cfg["correl_cols"] = st.session_state.get("corr_cols", [])
            save_config(cfg)
            st.session_state.wl_config = cfg
            st.toast("Grid saved!", icon="✅")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Timeframe ─────────────────────────────────────────────────────────────
    tf_opts = ["1W", "2W", "1M", "3M", "6M", "1Y", "Custom"]
    tf = st.segmented_control("Timeframe", tf_opts, default="3M", key="corr_tf")

    if tf == "Custom":
        dc1, dc2 = st.columns(2)
        with dc1:
            start_date = st.date_input("From",
                                       value=date.today() - timedelta(days=91),
                                       key="corr_start")
        with dc2:
            end_date = st.date_input("To", value=date.today(), key="corr_end")
    else:
        days_map   = {"1W": 7, "2W": 14, "1M": 30, "3M": 91, "6M": 182, "1Y": 365}
        end_date   = date.today()
        start_date = end_date - timedelta(days=days_map[tf])

    # ── Returns vs Levels toggle ──────────────────────────────────────────────
    corr_mode = st.radio(
        "Correlate on",
        ["Daily returns", "Outright levels"],
        horizontal=True,
        key="corr_mode",
    )

    if not row_names or not col_names:
        st.info("Select at least one asset for both rows and columns.")
        return

    # ── Fetch data ────────────────────────────────────────────────────────────
    needed_names   = list(dict.fromkeys(row_names + col_names))
    needed_tickers = tuple(name_to_ticker[n] for n in needed_names)

    with st.spinner("Computing correlations…"):
        batch = fetch_batch(needed_tickers)

    series = {}
    for name in needed_names:
        tkr = name_to_ticker[name]
        df  = batch.get(tkr)
        if df is None or df.empty:
            continue
        close = df["Close"].squeeze().astype(float)
        close = close[
            (close.index >= pd.Timestamp(start_date)) &
            (close.index <= pd.Timestamp(end_date))
        ]
        if len(close) < 5:
            continue
        if corr_mode == "Outright levels":
            series[name] = close
        else:
            inst_class = name_to_inst[name].get("class", "")
            chg = close.diff() if inst_class == "Rates" else close.pct_change()
            series[name] = chg.dropna()

    if len(series) < 2:
        st.warning("Not enough data — try a longer timeframe or different assets.")
        return

    ret_df    = pd.DataFrame(series).dropna(how="all")
    row_avail = [n for n in row_names if n in ret_df.columns]
    col_avail = [n for n in col_names if n in ret_df.columns]

    if not row_avail or not col_avail:
        st.warning("No data available for the selected combination.")
        return

    # ── Correlation ───────────────────────────────────────────────────────────
    all_avail = list(dict.fromkeys(row_avail + col_avail))
    corr      = ret_df[all_avail].corr()
    sub       = corr.loc[row_avail, col_avail]
    pct       = (sub * 100).round(1)
    n_days    = len(ret_df.dropna(how="all"))

    # ── Heatmap ───────────────────────────────────────────────────────────────
    z    = pct.values.tolist()
    text = [[f"{v:.0f}%" for v in row] for row in pct.values]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=col_avail,
        y=row_avail,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=13, family="Inter, Segoe UI, sans-serif"),
        colorscale=[
            [0.00, "#DC2626"],
            [0.25, "#FCA5A5"],
            [0.50, "#F8FAFC"],
            [0.75, "#6EE7B7"],
            [1.00, "#059669"],
        ],
        zmin=-100, zmax=100,
        colorbar=dict(
            title=dict(text="Corr %", font=dict(size=12)),
            tickvals=[-100, -50, 0, 50, 100],
            ticktext=["-100%", "-50%", "0%", "50%", "100%"],
            thickness=14,
        ),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Correlation: %{text}<extra></extra>",
    ))
    fig.update_layout(
        height=max(350, len(row_avail) * 50 + 130),
        margin=dict(l=10, r=10, t=70, b=10),
        xaxis=dict(side="top", tickangle=-30, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
    )

    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")

    mode_label = (
        "Outright price levels"
        if corr_mode == "Outright levels"
        else "Daily returns (Rates: absolute Δ, others: % change)"
    )
    st.caption(
        f"Period: **{start_date}** → **{end_date}**  ·  "
        f"**{n_days}** trading days  ·  {mode_label}"
    )

    # ── Detail charts ─────────────────────────────────────────────────────────
    st.markdown("#### Detail charts")
    d1, d2 = st.columns(2)
    with d1:
        detail_a1 = st.selectbox("Asset 1", ["— select —"] + row_avail,
                                 key="corr_d1")
    with d2:
        detail_a2 = st.selectbox("Asset 2", ["— select —"] + col_avail,
                                 key="corr_d2")

    if detail_a1 != "— select —" and detail_a2 != "— select —":
        _detail_charts(detail_a1, detail_a2, name_to_inst, name_to_ticker)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from watchlist import fetch_batch, load_config


def render_correl():
    cfg         = st.session_state.get("wl_config") or load_config()
    instruments = cfg["instruments"]

    name_to_inst   = {i["name"]: i            for i in instruments}
    name_to_ticker = {i["name"]: i["ticker"]  for i in instruments}
    all_names      = list(name_to_ticker.keys())

    st.markdown("### Correlation Matrix")

    # ── Asset selectors ───────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        row_names = st.multiselect(
            "Row assets (up to 10)", all_names,
            default=all_names[:min(5, len(all_names))],
            key="corr_rows",
        )
    with c2:
        col_names = st.multiselect(
            "Column assets (up to 10)", all_names,
            default=all_names[:min(5, len(all_names))],
            key="corr_cols",
        )

    # ── Timeframe ─────────────────────────────────────────────────────────────
    tf_opts = ["1W", "2W", "1M", "3M", "6M", "1Y", "Custom"]
    tf = st.segmented_control("Timeframe", tf_opts, default="3M", key="corr_tf")

    if tf == "Custom":
        dc1, dc2 = st.columns(2)
        with dc1:
            start_date = st.date_input("From", value=date.today() - timedelta(days=91),
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

    returns = {}
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
            returns[name] = close
        else:
            inst_class = name_to_inst[name].get("class", "")
            chg = close.diff() if inst_class == "Rates" else close.pct_change()
            returns[name] = chg.dropna()

    if len(returns) < 2:
        st.warning("Not enough data — try a longer timeframe or different assets.")
        return

    ret_df    = pd.DataFrame(returns).dropna(how="all")
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

    st.plotly_chart(fig, use_container_width=True)
    mode_label = (
        "Outright price levels"
        if corr_mode == "Outright levels"
        else "Daily returns (Rates: absolute Δ, others: % change)"
    )
    st.caption(
        f"Period: **{start_date}** → **{end_date}**  ·  "
        f"**{n_days}** trading days  ·  {mode_label}"
    )

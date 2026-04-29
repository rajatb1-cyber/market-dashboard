import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import urllib.request
from datetime import date

_BARCHART_BASE = "https://ondemand.websol.barchart.com"

_MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}
_MONTH_NAMES = {3: "Mar", 6: "Jun", 9: "Sep", 12: "Dec"}


def _active_contracts(n: int = 8) -> list:
    """Generate the next n quarterly Euribor futures contracts from today."""
    today = date.today()
    qm, qy = today.month, today.year
    for q in (3, 6, 9, 12):
        if q >= qm:
            qm, qy = q, qy
            break
    else:
        qm, qy = 3, qy + 1

    contracts = []
    for _ in range(n):
        yy = str(qy)[-2:]
        contracts.append({
            "ticker": f"ER{_MONTH_CODES[qm]}{yy}",
            "label":  f"{_MONTH_NAMES[qm]} {str(qy)[-2:]}",
        })
        qm += 3
        if qm > 12:
            qm, qy = 3, qy + 1
    return contracts


@st.cache_data(ttl=60)
def _barchart_quotes(api_key: str, symbols: str) -> list:
    url = (
        f"{_BARCHART_BASE}/getQuote.json?apikey={api_key}"
        f"&symbols={symbols}"
        f"&fields=lastPrice,previousClose,change,volume,openInterest,high,low"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    return data.get("results") or []


@st.cache_data(ttl=300)
def _barchart_history(api_key: str, symbol: str, period: str = "3m") -> pd.DataFrame:
    max_bars = {"1m": 31, "3m": 92, "6m": 183, "1y": 365}.get(period, 92)
    url = (
        f"{_BARCHART_BASE}/getHistory.json?apikey={api_key}"
        f"&symbol={symbol}&type=daily&maxBars={max_bars}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    results = data.get("results") or []
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    date_col = next((c for c in ("tradingDay", "timestamp", "date") if c in df.columns), None)
    if not date_col:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
    df = df.set_index("date").sort_index()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["implied_rate"] = 100 - df["close"]
    return df[["close", "implied_rate"]].dropna()


# ── Main render ────────────────────────────────────────────────────────────────

def render_stir():
    st.markdown("### STIR — Short Term Interest Rate Futures")

    api_key = ""
    try:
        api_key = st.secrets.get("BARCHART_API_KEY", "")
    except Exception:
        pass

    if not api_key:
        st.warning(
            "**Barchart API key required.**  \n"
            "Sign up for a free key at **barchart.com/ondemand** then add it to "
            "Streamlit secrets:  \n```\nBARCHART_API_KEY = \"your-key-here\"\n```"
        )
        return

    contracts = _active_contracts(8)
    symbols   = ",".join(c["ticker"] for c in contracts)

    with st.spinner("Loading Euribor futures…"):
        try:
            results = _barchart_quotes(api_key, symbols)
        except Exception as e:
            st.error(f"Barchart fetch error: {e}")
            return

    if not results:
        st.warning("No quotes returned — check that your API key is valid.")
        return

    quote_map = {r["symbol"]: r for r in results}

    rows = []
    for c in contracts:
        q = quote_map.get(c["ticker"])
        if not q:
            continue
        last = q.get("lastPrice") or q.get("last")
        prev = q.get("previousClose")
        vol  = q.get("volume")
        oi   = q.get("openInterest")
        if last is None:
            continue
        impl   = 100 - last
        chg_p  = (last - prev) if prev is not None else None
        chg_b  = chg_p * 100   if chg_p is not None else None
        rows.append({
            "Contract":   c["label"],
            "Ticker":     c["ticker"],
            "Price":      f"{last:.3f}",
            "Impl. Rate": f"{impl:.3f}%",
            "Chg (pts)":  f"{chg_p:+.3f}" if chg_p is not None else "—",
            "Chg (bps)":  f"{chg_b:+.1f}" if chg_b is not None else "—",
            "Volume":     f"{int(vol):,}"  if vol else "—",
            "Open Int":   f"{int(oi):,}"   if oi  else "—",
            "_impl":      impl,
            "_chg_b":     chg_b,
        })

    if not rows:
        st.warning("No valid quotes found.")
        return

    # ── Table ──────────────────────────────────────────────────────────────────
    st.markdown("#### Euribor 3M Futures (ICE)")
    st.caption("Price = 100 − Implied Rate · Source: Barchart (15-min delayed)")

    display_cols = ["Price", "Impl. Rate", "Chg (pts)", "Chg (bps)", "Volume", "Open Int"]
    df_tbl = pd.DataFrame(rows).set_index("Contract")[display_cols]

    # price up = implied rate down (dovish); colour price-change cols green on up
    # colour Impl. Rate inverse: red if rate rising, green if falling
    chg_colors  = []
    rate_colors = []
    for r in rows:
        v = r["_chg_b"]
        if v is not None:
            chg_colors.append("color:#059669;font-weight:600" if v > 0 else
                              ("color:#DC2626;font-weight:600" if v < 0 else ""))
            rate_colors.append("color:#DC2626;font-weight:600" if v > 0 else
                               ("color:#059669;font-weight:600" if v < 0 else ""))
        else:
            chg_colors.append("")
            rate_colors.append("")

    styled = (
        df_tbl.style
        .apply(lambda _: chg_colors,  subset=["Chg (pts)", "Chg (bps)"])
        .apply(lambda _: rate_colors, subset=["Impl. Rate"])
    )
    st.dataframe(styled, use_container_width=True)

    # ── Implied rate curve ─────────────────────────────────────────────────────
    st.markdown("#### Implied ECB Rate Path")
    labels = [r["Contract"] for r in rows]
    rates  = [r["_impl"]    for r in rows]
    chgs   = [r["_chg_b"]   for r in rows]

    dot_colors = [
        "#DC2626" if (c or 0) > 0 else ("#059669" if (c or 0) < 0 else "#94A3B8")
        for c in chgs
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=rates,
        mode="lines+markers",
        line=dict(color="#0EA5E9", width=2),
        marker=dict(size=9, color=dot_colors, line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="<b>%{x}</b><br>Implied: %{y:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        yaxis=dict(ticksuffix="%", gridcolor="#E8EDF5", zeroline=False),
        xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dot colour: red = rate priced higher vs prev. close · green = lower")

    # ── Contract history ───────────────────────────────────────────────────────
    st.markdown("#### Contract history")
    hist_labels  = [r["Contract"] for r in rows]
    hist_tickers = [r["Ticker"]   for r in rows]

    hc1, hc2 = st.columns([3, 2])
    with hc1:
        sel_lbl = st.selectbox("Contract", hist_labels, key="stir_hist_sel")
    with hc2:
        hist_period = st.selectbox(
            "Period",
            ["1m", "3m", "6m", "1y"],
            format_func=lambda x: {"1m": "1M", "3m": "3M", "6m": "6M", "1y": "1Y"}[x],
            index=1, key="stir_hist_period",
        )

    sel_tkr = hist_tickers[hist_labels.index(sel_lbl)]
    with st.spinner(f"Loading {sel_lbl} history…"):
        try:
            df_hist = _barchart_history(api_key, sel_tkr, hist_period)
        except Exception as e:
            st.error(f"History fetch error: {e}")
            df_hist = pd.DataFrame()

    if not df_hist.empty:
        first_rate = float(df_hist["implied_rate"].iloc[0])
        last_rate  = float(df_hist["implied_rate"].iloc[-1])
        line_color = "#DC2626" if last_rate > first_rate else "#059669"
        fill_color = ("rgba(220,38,38,0.07)" if line_color == "#DC2626"
                      else "rgba(5,150,105,0.07)")

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_hist.index, y=df_hist["implied_rate"],
            mode="lines",
            line=dict(color=line_color, width=2),
            fill="tozeroy", fillcolor=fill_color,
            hovertemplate="%{x|%d %b %Y}<br><b>%{y:.3f}%</b><extra></extra>",
        ))
        fig2.update_layout(
            title=dict(
                text=f"{sel_lbl} — Implied Rate  "
                     f"<span style='font-size:12px;color:#64748B'>"
                     f"current: {last_rate:.3f}%</span>",
                font=dict(size=13, color="#1A202C"),
            ),
            height=280,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
            yaxis=dict(ticksuffix="%", gridcolor="#E8EDF5", zeroline=False),
            xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
            hovermode="x unified",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No history available for this contract / period.")

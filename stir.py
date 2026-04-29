import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date
import time

import asyncio

# Streamlit runs scripts in a thread with no event loop; ib_insync needs one at import time
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    import nest_asyncio
    nest_asyncio.apply()
    from ib_insync import IB, Future, util
    _IB_AVAILABLE = True
except Exception:
    _IB_AVAILABLE = False

_MONTH_NAMES = {3: "Mar", 6: "Jun", 9: "Sep", 12: "Dec"}
_DURATION_MAP = {"1M": "1 M", "3M": "3 M", "6M": "6 M", "1Y": "1 Y"}


def _active_contracts(n: int = 12) -> list[tuple]:
    """Return n quarterly Euribor contracts starting from the most recent past quarter."""
    today = date.today()
    qm, qy = today.month, today.year
    for q in (12, 9, 6, 3):
        if q <= qm:
            qm = q
            break
    else:
        qm, qy = 12, qy - 1

    out = []
    for _ in range(n):
        out.append((f"{_MONTH_NAMES[qm]} {str(qy)[-2:]}", f"{qy}{qm:02d}"))
        qm += 3
        if qm > 12:
            qm, qy = 3, qy + 1
    return out


def _v(val):
    """Return val if it is a valid positive finite float, else None."""
    try:
        f = float(val)
        return f if (f == f and f > 0) else None  # NaN guard + positive check
    except Exception:
        return None


@st.cache_data(ttl=15)
def _fetch_quotes(host: str, port: int, contracts: tuple) -> tuple:
    """Fetch Euribor futures quotes from IBKR.
    Returns (rows: list[dict], debug: list[dict]).
    """
    ib = IB()
    rows, debug = [], []
    try:
        ib.connect(host, port, clientId=15, timeout=10, readonly=True)

        # ── Batch qualify all contracts in one API round-trip ──────────────────
        fut_objects = [
            Future(symbol="I", exchange="ICEEU", currency="EUR",
                   lastTradeDateOrContractMonth=expiry)
            for _, expiry in contracts
        ]
        try:
            qualified_list = ib.qualifyContracts(*fut_objects)
        except Exception:
            qualified_list = []

        if not qualified_list:
            return [], []

        # After qualification IBKR fills lastTradeDateOrContractMonth as YYYYMMDD
        # (e.g. "20261215").  Match back to our labels using the YYYYMM prefix.
        label_by_exp6 = {exp: lbl for lbl, exp in contracts}  # "202612" → "Dec 26"

        ticker_triples = []  # (ticker, label, exp6)
        for qc in qualified_list:
            exp6 = qc.lastTradeDateOrContractMonth[:6]
            label = label_by_exp6.get(exp6)
            if label:
                t = ib.reqMktData(qc, "", False, False)
                ticker_triples.append((t, label, exp6))

        if not ticker_triples:
            return [], []

        # ── Wait for data ──────────────────────────────────────────────────────
        # Frozen mode (2): returns last available data even when market is closed.
        # IBKR will automatically upgrade to live (1) if the market is currently open.
        ib.reqMarketDataType(2)
        ib.sleep(5)

        # ── Extract prices ─────────────────────────────────────────────────────
        for ticker, label, expiry in ticker_triples:
            last  = _v(ticker.last)
            close = _v(ticker.close)
            bid   = _v(ticker.bid)
            ask   = _v(ticker.ask)
            vol   = ticker.volume
            mdt   = getattr(ticker, "marketDataType", "?")

            debug.append({
                "Contract": label,
                "last":     last,
                "bid":      bid,
                "ask":      ask,
                "close":    close,
                "mktDataType": mdt,
            })

            mid        = (bid + ask) / 2 if (bid and ask) else None
            live_price = last or mid
            src        = "last" if last else ("mid" if mid else "close")

            # For curve display, fall back to close if no live price
            display_price = live_price or close
            if display_price is None:
                continue

            chg_p = (live_price - close) if (live_price and close) else None
            chg_b = chg_p * 100         if chg_p is not None else None

            rows.append({
                "label":     label,
                "expiry":    expiry,
                "price":     live_price,       # None when market closed
                "close":     close,
                "impl_rate": 100 - display_price,
                "chg_p":     chg_p,
                "chg_b":     chg_b,
                "volume":    int(vol) if vol and vol > 0 else None,
                "src":       src,
            })

        for t, _, _ in ticker_triples:
            try:
                ib.cancelMktData(t.contract)
            except Exception:
                pass

    finally:
        if ib.isConnected():
            ib.disconnect()

    return rows, debug


@st.cache_data(ttl=300)
def _fetch_history(host: str, port: int, expiry: str, duration: str) -> pd.DataFrame:
    ib = IB()
    try:
        ib.connect(host, port, clientId=16, timeout=10, readonly=True)

        contract = Future(symbol="I", exchange="ICEEU", currency="EUR",
                          lastTradeDateOrContractMonth=expiry)
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            return pd.DataFrame()

        bars = ib.reqHistoricalData(
            qualified[0],
            endDateTime="",
            durationStr=duration,
            barSizeSetting="1 day",
            whatToShow="LAST",
            useRTH=True,
        )
        if not bars:
            return pd.DataFrame()

        df = util.df(bars)[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").sort_index()
        df["implied_rate"] = 100 - df["close"]
        return df.dropna()

    finally:
        if ib.isConnected():
            ib.disconnect()


def _fmt(v, fmt_str, fallback="—"):
    try:
        return fmt_str.format(v) if v is not None else fallback
    except Exception:
        return fallback


def render_stir():
    st.markdown("### STIR — Short Term Interest Rate Futures")

    if not _IB_AVAILABLE:
        st.info(
            "**This tab requires a local IBKR connection and is not available on Streamlit Cloud.**  \n\n"
            "To use it, run the app locally with TWS open:  \n"
            "```\npip install ib_insync nest_asyncio\nstreamlit run app.py\n```"
        )
        return

    # ── Connection settings ────────────────────────────────────────────────────
    with st.expander("⚙️  IBKR connection", expanded=False):
        cc1, cc2 = st.columns(2)
        with cc1:
            host = st.text_input("Host", value="127.0.0.1", key="stir_host")
        with cc2:
            port = st.number_input(
                "Port  (TWS: 7496 · IB Gateway: 4001)",
                value=7496, min_value=1, max_value=65535,
                step=1, key="stir_port",
            )

    contracts = _active_contracts(12)

    # ── Refresh button ─────────────────────────────────────────────────────────
    if st.button("⟳ Refresh", key="stir_refresh_btn"):
        _fetch_quotes.clear()
        st.rerun()

    # ── Fetch quotes ───────────────────────────────────────────────────────────
    fetch_ts = time.time()
    with st.spinner("Connecting to IBKR…"):
        try:
            rows, debug = _fetch_quotes(host, int(port), tuple(contracts))
        except Exception as e:
            st.error(
                f"**Could not connect to IBKR** (`{host}:{port}`).  \n"
                f"Make sure IB Gateway / TWS is running and API connections are enabled.  \n"
                f"Error: `{e}`"
            )
            return

    if not rows:
        st.warning("Connected but no quotes returned — check your Euribor futures market data subscription in TWS.")
        return

    # ── Debug expander ─────────────────────────────────────────────────────────
    with st.expander("Debug — raw IBKR tick values", expanded=False):
        st.caption("mktDataType: 1=live, 2=frozen, 3=delayed, 4=delayed-frozen")
        st.dataframe(pd.DataFrame(debug), use_container_width=True)

    # ── Table ──────────────────────────────────────────────────────────────────
    st.markdown("#### Euribor 3M Futures (ICE / LIFFE)")

    src_counts: dict = {}
    for r in rows:
        src_counts[r["src"]] = src_counts.get(r["src"], 0) + 1
    src_str = " · ".join(f"{v}× {k}" for k, v in src_counts.items())
    updated = time.strftime("%H:%M:%S", time.localtime(fetch_ts))
    st.caption(f"IBKR · Price = 100 − Rate · Updated {updated} · {src_str}")

    table_rows, chg_colors, rate_colors = [], [], []

    for r in rows:
        chg_b = r["chg_b"]
        if chg_b is not None:
            chg_colors.append("color:#059669;font-weight:600" if chg_b > 0 else
                               ("color:#DC2626;font-weight:600" if chg_b < 0 else ""))
            rate_colors.append("color:#DC2626;font-weight:600" if chg_b > 0 else
                                ("color:#059669;font-weight:600" if chg_b < 0 else ""))
        else:
            chg_colors.append("")
            rate_colors.append("")

        price_str = (_fmt(r["price"], "{:.3f}") if r["price"] is not None
                     else _fmt(r["close"], "{:.3f}*"))  # * = prev close, no live data

        table_rows.append({
            "Contract":   r["label"],
            "Price":      price_str,
            "Impl. Rate": _fmt(r["impl_rate"], "{:.3f}%"),
            "Chg (pts)":  _fmt(r["chg_p"], "{:+.3f}"),
            "Chg (bps)":  _fmt(r["chg_b"], "{:+.1f}"),
            "Volume":     _fmt(r["volume"], "{:,}"),
        })

    df_tbl = pd.DataFrame(table_rows).set_index("Contract")
    styled = (
        df_tbl.style
        .apply(lambda _: chg_colors,  subset=["Chg (pts)", "Chg (bps)"])
        .apply(lambda _: rate_colors, subset=["Impl. Rate"])
    )
    st.dataframe(styled, use_container_width=True)

    # ── Implied rate curve ─────────────────────────────────────────────────────
    st.markdown("#### Implied ECB Rate Path")

    labels     = [r["label"]     for r in rows]
    impl_rates = [r["impl_rate"] for r in rows]
    chg_bs     = [r["chg_b"]     for r in rows]

    dot_colors = [
        "#DC2626" if (c or 0) > 0 else ("#059669" if (c or 0) < 0 else "#94A3B8")
        for c in chg_bs
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=impl_rates,
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
    st.caption("Dot colour: red = rate priced higher vs prev close · green = lower")

    # ── Contract history ───────────────────────────────────────────────────────
    st.markdown("#### Contract history")

    hc1, hc2 = st.columns([3, 2])
    with hc1:
        sel_label = st.selectbox("Contract", [r["label"] for r in rows], key="stir_hist_sel")
    with hc2:
        hist_period = st.selectbox("Period", ["1M", "3M", "6M", "1Y"], index=1, key="stir_hist_period")

    sel_expiry = next(r["expiry"] for r in rows if r["label"] == sel_label)
    ibkr_dur   = _DURATION_MAP[hist_period]

    with st.spinner(f"Loading {sel_label} history…"):
        try:
            df_hist = _fetch_history(host, int(port), sel_expiry, ibkr_dur)
        except Exception as e:
            st.error(f"History fetch error: `{e}`")
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
                text=(f"{sel_label} — Implied Rate  "
                      f"<span style='font-size:12px;color:#64748B'>"
                      f"current: {last_rate:.3f}%</span>"),
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
        st.info("No history data available for this contract / period.")

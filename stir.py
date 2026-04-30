import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date
import time

import asyncio

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

# Contract specs for each curve
_CURVES = {
    "Euribor": {
        "symbol":        "I",
        "exchange":      "ICEEU",
        "currency":      "EUR",
        "tradingClass":  "",
        "subtitle":      "Euribor 3M Futures (ICE/LIFFE)",
        "cb_label":      "Implied ECB Rate Path",
    },
    "SONIA": {
        "symbol":        "SONIA",
        "exchange":      "ICEEU",
        "currency":      "GBP",
        "tradingClass":  "SONIA.N",   # as shown in TWS: SONIA.N Dec'26@ICEEU
        "subtitle":      "SONIA Futures (ICE/LIFFE)",
        "cb_label":      "Implied BoE Rate Path",
    },
    "SOFR": {
        "symbol":        "SR3",
        "exchange":      "CME",
        "currency":      "USD",
        "tradingClass":  "",
        "subtitle":      "SOFR 3M Futures (CME)",
        "cb_label":      "Implied Fed Rate Path",
    },
}


def _active_contracts(n: int = 12) -> list[tuple]:
    """Return n quarterly contracts starting from the most recent past quarter."""
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
    try:
        f = float(val)
        return f if (f == f and f > 0) else None
    except Exception:
        return None


def _fmt(v, fmt_str, fallback="—"):
    try:
        return fmt_str.format(v) if v is not None else fallback
    except Exception:
        return fallback


def _resolve(ib, symbol, exchange, currency, exp6, trading_class=""):
    """Return a qualified contract via reqContractDetails (handles ambiguous results)."""
    kwargs = dict(symbol=symbol, exchange=exchange, currency=currency,
                  lastTradeDateOrContractMonth=exp6)
    if trading_class:
        kwargs["tradingClass"] = trading_class
    fut = Future(**kwargs)
    try:
        details = ib.reqContractDetails(fut)
    except Exception:
        return None, 0
    if not details:
        return None, 0
    exact = [d for d in details if d.contract.lastTradeDateOrContractMonth[:6] == exp6]
    chosen = exact or details
    return chosen[0].contract, len(details)


@st.cache_data(ttl=15)
def _fetch_quotes(host: str, port: int, contracts: tuple,
                  symbol: str, exchange: str, currency: str,
                  trading_class: str = "") -> tuple:
    """Returns (rows, debug) for one curve."""
    ib = IB()
    rows  = []
    debug = {"qual": [], "ticks": []}
    try:
        ib.connect(host, port, clientId=15, timeout=10, readonly=True)

        label_by_exp6 = {exp: lbl for lbl, exp in contracts}
        ticker_quads  = []

        for exp6, label in label_by_exp6.items():
            qc, n_matches = _resolve(ib, symbol, exchange, currency, exp6, trading_class)
            debug["qual"].append({
                "exp": exp6, "label": label,
                "status": "ok" if qc else "no match",
                "matches": n_matches,
            })
            if qc:
                t = ib.reqMktData(qc, "", False, False)
                ticker_quads.append((t, qc, label, exp6))

        if not ticker_quads:
            return [], debug

        ib.reqMarketDataType(2)
        ib.sleep(5)

        for ticker, qc, label, expiry in ticker_quads:
            last  = _v(ticker.last)
            close = _v(ticker.close)
            bid   = _v(ticker.bid)
            ask   = _v(ticker.ask)
            vol   = ticker.volume
            mdt   = getattr(ticker, "marketDataType", "?")

            mid        = (bid + ask) / 2 if (bid and ask) else None
            live_price = last or mid

            hist_price = None
            if not live_price:
                try:
                    bars = ib.reqHistoricalData(
                        qc, endDateTime="",
                        durationStr="5 D", barSizeSetting="1 day",
                        whatToShow="LAST", useRTH=False,
                    )
                    if bars:
                        hist_price = _v(bars[-1].close)
                        if not close and len(bars) >= 2:
                            close = _v(bars[-2].close)
                except Exception:
                    pass

            live_price = live_price or hist_price
            src = "last" if last else ("mid" if mid else ("hist" if hist_price else "close"))

            debug["ticks"].append({
                "Contract": label, "last": last, "bid": bid,
                "ask": ask, "close": close, "hist": hist_price, "mktDataType": mdt,
            })

            display_price = live_price or close
            if display_price is None:
                continue

            chg_p = (live_price - close) if (live_price and close) else None
            chg_b = chg_p * 100         if chg_p is not None else None

            rows.append({
                "label":     label,
                "expiry":    expiry,
                "price":     live_price,
                "close":     close,
                "impl_rate": 100 - display_price,
                "chg_p":     chg_p,
                "chg_b":     chg_b,
                "volume":    int(vol) if vol and vol > 0 else None,
                "src":       src,
            })

        for t, _, _, _ in ticker_quads:
            try:
                ib.cancelMktData(t.contract)
            except Exception:
                pass

    finally:
        if ib.isConnected():
            ib.disconnect()

    return rows, debug


@st.cache_data(ttl=300)
def _fetch_history(host: str, port: int, expiry: str, duration: str,
                   symbol: str, exchange: str, currency: str,
                   trading_class: str = "") -> tuple:
    """Returns (df, error_str)."""
    ib = IB()
    try:
        ib.connect(host, port, clientId=16, timeout=10, readonly=True)

        resolved, _ = _resolve(ib, symbol, exchange, currency, expiry, trading_class)
        if resolved is None:
            return pd.DataFrame(), f"Could not resolve contract {symbol} {expiry} on {exchange}"

        bars = None
        for show in ("MIDPOINT", "LAST"):
            try:
                bars = ib.reqHistoricalData(
                    resolved, endDateTime="",
                    durationStr=duration, barSizeSetting="1 day",
                    whatToShow=show, useRTH=False,
                )
            except Exception:
                bars = None
            if bars:
                break

        if not bars:
            return pd.DataFrame(), (
                f"No bars returned for conId={resolved.conId} "
                f"expiry={resolved.lastTradeDateOrContractMonth} dur={duration}"
            )

        df = util.df(bars)[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").sort_index()
        df["implied_rate"] = 100 - df["close"]
        return df.dropna(), ""

    except Exception as e:
        return pd.DataFrame(), str(e)
    finally:
        if ib.isConnected():
            ib.disconnect()


@st.cache_data(ttl=300)
def _scan_symbol(host: str, port: int, symbol: str, exchange: str) -> list:
    """Probe IBKR with reqContractDetails using minimal constraints to find the right spec."""
    ib = IB()
    results = []
    try:
        ib.connect(host, port, clientId=17, timeout=10, readonly=True)

        # Try a range of symbol variants — no expiry so we get whatever is listed
        candidates = [symbol, symbol.replace(".", ""), symbol.split(".")[0]]
        seen = set()
        for sym in candidates:
            if sym in seen:
                continue
            seen.add(sym)
            try:
                fut = Future(symbol=sym, exchange=exchange)
                details = ib.reqContractDetails(fut)
                for d in details[:5]:   # cap at 5 per variant
                    c = d.contract
                    results.append({
                        "tried_symbol": sym,
                        "conId":        c.conId,
                        "symbol":       c.symbol,
                        "localSymbol":  c.localSymbol,
                        "tradingClass": c.tradingClass,
                        "exchange":     c.exchange,
                        "currency":     c.currency,
                        "expiry":       c.lastTradeDateOrContractMonth,
                    })
            except Exception as e:
                results.append({"tried_symbol": sym, "error": str(e)})

    except Exception as e:
        results.append({"error": str(e)})
    finally:
        if ib.isConnected():
            ib.disconnect()

    return results


def _render_curve(host: str, port: int, curve_name: str, tab_key: str):
    """Render the full table + chart + history for one curve."""
    cfg            = _CURVES[curve_name]
    symbol         = cfg["symbol"]
    exchange       = cfg["exchange"]
    currency       = cfg["currency"]
    trading_class  = cfg.get("tradingClass", "")
    contracts      = _active_contracts(12)

    if st.button("⟳ Refresh", key=f"stir_refresh_{tab_key}"):
        _fetch_quotes.clear()
        st.rerun()

    fetch_ts = time.time()
    with st.spinner(f"Connecting to IBKR for {curve_name}…"):
        try:
            rows, debug = _fetch_quotes(
                host, port, tuple(contracts), symbol, exchange, currency, trading_class
            )
        except Exception as e:
            st.error(
                f"**Could not connect to IBKR** (`{host}:{port}`).  \n"
                f"Error: `{e}`"
            )
            return

    if not rows:
        st.warning(f"No quotes returned for {curve_name} — could not resolve contracts.")
        with st.expander("Debug — qualification", expanded=True):
            st.dataframe(pd.DataFrame(debug.get("qual", [])), use_container_width=True)
        with st.expander(f"Symbol scanner — probing '{cfg['symbol']}' on {cfg['exchange']}", expanded=True):
            scan = _scan_symbol(host, port, cfg["symbol"], cfg["exchange"])
            if scan:
                st.dataframe(pd.DataFrame(scan), use_container_width=True)
            else:
                st.info("No results — check exchange name or symbol.")
        return

    missing = {lbl for lbl, _ in contracts} - {r["label"] for r in rows}
    if missing:
        st.warning(f"Missing contracts: {', '.join(sorted(missing))}")

    with st.expander("Debug — raw IBKR values", expanded=False):
        st.caption("Qualification")
        st.dataframe(pd.DataFrame(debug.get("qual", [])), use_container_width=True)
        st.caption("Tick data  (mktDataType: 1=live, 2=frozen)")
        st.dataframe(pd.DataFrame(debug.get("ticks", [])), use_container_width=True)

    # ── Quotes table ───────────────────────────────────────────────────────────
    st.markdown(f"#### {cfg['subtitle']}")
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
                     else _fmt(r["close"], "{:.3f}*"))

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

    # ── Rate curve ─────────────────────────────────────────────────────────────
    st.markdown(f"#### {cfg['cb_label']}")
    dot_colors = [
        "#DC2626" if (r["chg_b"] or 0) > 0 else
        ("#059669" if (r["chg_b"] or 0) < 0 else "#94A3B8")
        for r in rows
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["label"] for r in rows],
        y=[r["impl_rate"] for r in rows],
        mode="lines+markers",
        line=dict(color="#0EA5E9", width=2),
        marker=dict(size=9, color=dot_colors, line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="<b>%{x}</b><br>Implied: %{y:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        yaxis=dict(ticksuffix="%", gridcolor="#E8EDF5", zeroline=False),
        xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
        hovermode="x unified",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dot: red = rate priced higher vs prev close · green = lower")

    # ── Contract history ───────────────────────────────────────────────────────
    st.markdown("#### Contract history")
    hc1, hc2 = st.columns([3, 2])
    with hc1:
        sel_label = st.selectbox("Contract", [r["label"] for r in rows],
                                 key=f"stir_hist_sel_{tab_key}")
    with hc2:
        hist_period = st.selectbox("Period", ["1M", "3M", "6M", "1Y"], index=1,
                                   key=f"stir_hist_period_{tab_key}")

    sel_expiry = next(r["expiry"] for r in rows if r["label"] == sel_label)
    ibkr_dur   = _DURATION_MAP[hist_period]

    with st.spinner(f"Loading {sel_label} history…"):
        try:
            df_hist, hist_err = _fetch_history(
                host, port, sel_expiry, ibkr_dur, symbol, exchange, currency, trading_class
            )
        except Exception as e:
            st.error(f"History fetch error: `{e}`")
            df_hist, hist_err = pd.DataFrame(), ""

    if hist_err:
        st.error(f"History error: {hist_err}")

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
            height=280, margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
            yaxis=dict(ticksuffix="%", gridcolor="#E8EDF5", zeroline=False),
            xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
            hovermode="x unified",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
        )
        st.plotly_chart(fig2, use_container_width=True)
    elif not hist_err:
        st.info("No history data available for this contract / period.")


def render_stir():
    st.markdown("### STIR — Short Term Interest Rate Futures")

    if not _IB_AVAILABLE:
        st.info(
            "**This tab requires a local IBKR connection and is not available on Streamlit Cloud.**  \n\n"
            "To use it, run the app locally with TWS open:  \n"
            "```\npip install ib_insync nest_asyncio\nstreamlit run app.py\n```"
        )
        return

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

    tab_eur, tab_son, tab_sofr = st.tabs(["Euribor", "SONIA", "SOFR"])

    with tab_eur:
        _render_curve(host, int(port), "Euribor", "eur")

    with tab_son:
        _render_curve(host, int(port), "SONIA", "son")

    with tab_sofr:
        _render_curve(host, int(port), "SOFR", "sofr")

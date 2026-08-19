"""
Long-Term Holdings risk tab.

The mirror image of the speculative Risk/VaR book: keeps ONLY the ETF holdings
(SubCategory == "ETF") and shows how much notional is allocated to each exposure
(S&P 500, Nasdaq, Gold, US Treasuries, Bitcoin, …), rolled up by asset class.

Positions come from the same local Flex DB the Risk/VaR and P&L tabs populate —
refresh them from either of those tabs' "Update from IBKR" buttons.
"""
import json
import math
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pnl_db import load_sections

_ETF_SUBCATEGORY = "ETF"
_LT_SUBCATEGORIES = {"ETF", "COMMON"}   # ETFs + single stocks — the long-term book

# Asset classes for the overall allocation — "Illiquid" (real estate, PE, …) on top.
_ALLOC_CLASSES = ["Illiquid", "Equity", "Rates", "Commodity", "Crypto", "Cash"]
_MANUAL_PATH = os.path.join(os.path.dirname(__file__), "manual_holdings.json")


def _load_manual() -> pd.DataFrame:
    """External (non-IBKR) holdings entered manually: Account / Asset Class / $ Amount."""
    try:
        with open(_MANUAL_PATH) as f:
            df = pd.DataFrame(json.load(f))
        if not df.empty:
            return df.reindex(columns=["Account", "Asset Class", "$ Amount"])
    except Exception:
        pass
    return pd.DataFrame(columns=["Account", "Asset Class", "$ Amount"])


def _save_manual(df: pd.DataFrame):
    try:
        d = df.copy()
        d["$ Amount"] = pd.to_numeric(d["$ Amount"], errors="coerce")
        d = d[d["$ Amount"].notna() & (d["$ Amount"] != 0)]
        with open(_MANUAL_PATH, "w") as f:
            json.dump(d.to_dict("records"), f, indent=2)
    except Exception:
        pass

# Symbol → (exposure, asset class). Derived from the actual holdings; extend as needed.
_ETF_EXPOSURE = {
    "SPY":   ("S&P 500",            "Equity"),
    "VUSA":  ("S&P 500",            "Equity"),
    "CSPX":  ("S&P 500",            "Equity"),
    "SXRV":  ("Nasdaq 100",         "Equity"),
    "CNDX":  ("Nasdaq 100",         "Equity"),
    "XDJP":  ("Nikkei 225 / Japan", "Equity"),
    "TLT":   ("US Treasury 20Y+",   "Rates"),
    "LTPZ":  ("US TIPS 15Y+",       "Rates"),
    "VDST":  ("US T-Bills 0-1Y (cash)", "Cash"),
    "PHGP":  ("Gold",               "Commodity"),
    "PHGPL": ("Gold",               "Commodity"),
    "CPER":  ("Copper",             "Commodity"),
    "EZBC":  ("Bitcoin",            "Crypto"),
}

_AC_COLOUR = {
    "Equity": "#2563EB", "Rates": "#059669", "Commodity": "#D97706",
    "Crypto": "#7C3AED", "Cash": "#0891B2", "Illiquid": "#92400E", "Other": "#64748B",
}


def _classify(symbol: str, description: str, subcategory: str = "") -> tuple[str, str]:
    """(exposure, asset_class) — table lookup, then description keywords. Single
    stocks (SubCategory COMMON) get their own exposure line under Equity."""
    s = str(symbol).upper().strip()
    if s in _ETF_EXPOSURE:
        return _ETF_EXPOSURE[s]
    if str(subcategory).upper().strip() == "COMMON":
        return (s or "Single Stock", "Equity")
    d = str(description).upper()
    if "S&P 500" in d or "S&P500" in d or "SP500" in d:
        return ("S&P 500", "Equity")
    if "NASDAQ" in d:
        return ("Nasdaq 100", "Equity")
    if "NIKKEI" in d or "JAPAN" in d or "TOPIX" in d:
        return ("Nikkei 225 / Japan", "Equity")
    if "GOLD" in d:
        return ("Gold", "Commodity")
    if "COPPER" in d:
        return ("Copper", "Commodity")
    if "BITCOIN" in d or "BTC" in d:
        return ("Bitcoin", "Crypto")
    if "TIPS" in d:
        return ("US TIPS", "Rates")
    if "TREASURY" in d or "TRBD" in d or "GILT" in d or "BOND" in d:
        return ("Bonds / Rates", "Rates")
    return (s or "Other", "Other")


def build_ltr_book(positions: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per ETF symbol with USD notional, unrealised PnL, exposure & asset class."""
    if positions is None:
        positions = load_sections().get("positions", pd.DataFrame())
    if positions is None or positions.empty:
        return pd.DataFrame()

    df = positions.copy()
    for c in ("Quantity", "MarkPrice", "PositionValue", "FXRateToBase", "FifoPnlUnrealized"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["FXRateToBase"] = df["FXRateToBase"].fillna(1.0) if "FXRateToBase" in df.columns else 1.0

    # LT book = ETFs + single stocks (SubCategory COMMON, e.g. NVDA, SpaceX Cl-A) —
    # both are long-term holdings; the speculative Risk/VaR book excludes them.
    if "SubCategory" in df.columns:
        df = df[df["SubCategory"].fillna("").str.upper().isin(_LT_SUBCATEGORIES)]
    if df.empty:
        return pd.DataFrame()

    df["notional_usd"] = df["PositionValue"] * df["FXRateToBase"]
    df["upnl_usd"] = df.get("FifoPnlUnrealized", 0.0) * df["FXRateToBase"]
    _cls = df.apply(lambda r: _classify(r["Symbol"], r.get("Description", ""),
                                        r.get("SubCategory", "")), axis=1)
    df["Exposure"]   = [c[0] for c in _cls]
    df["AssetClass2"] = [c[1] for c in _cls]

    def _first(s):
        return s.dropna().iloc[0] if s.notna().any() else ""

    g = df.groupby("Symbol", as_index=False).agg(
        Description=("Description", _first),
        Exposure=("Exposure", "first"),
        AssetClass2=("AssetClass2", "first"),
        Currency=("CurrencyPrimary", _first) if "CurrencyPrimary" in df.columns else ("Symbol", _first),
        Quantity=("Quantity", "sum"),
        MarkPrice=("MarkPrice", _first),
        notional_usd=("notional_usd", "sum"),
        upnl_usd=("upnl_usd", "sum"),
    )
    return g.sort_values("notional_usd", ascending=False).reset_index(drop=True)


# ── HTML helpers ──────────────────────────────────────────────────────────────
def _f0(v):
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"${v:,.0f}"


def _pct(v):
    return "—" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:,.1f}%"


def _cc(v):
    return "#059669" if (v or 0) >= 0 else "#DC2626"


def _summary_table(rows: list[tuple[str, float, float, float]], first_col: str, total_lbl: str) -> str:
    """rows = [(label, notional, pct_of_book, upnl)]; renders a compact HTML table."""
    sth  = "background:#0F172A;color:#F8FAFC;font-size:12px;font-weight:700;padding:6px 10px;text-align:right"
    sthl = sth.replace("text-align:right", "text-align:left")
    std  = "font-size:12px;padding:5px 10px;border-bottom:1px solid #E2E8F0;text-align:right"
    stdl = std.replace("text-align:right", "text-align:left")
    stf  = "font-size:12px;padding:6px 10px;border-top:2px solid #475569;font-weight:800;text-align:right"
    stfl = stf.replace("text-align:right", "text-align:left")
    body = ""
    for lbl, notl, pct, upnl in rows:
        body += (f"<tr><td style='{stdl}'><b>{lbl}</b></td>"
                 f"<td style='{std}'>{_f0(notl)}</td>"
                 f"<td style='{std}'>{_pct(pct)}</td>"
                 f"<td style='{std};color:{_cc(upnl)}'>{_f0(upnl)}</td></tr>")
    tn = sum(r[1] for r in rows)
    tu = sum(r[3] for r in rows)
    body += (f"<tr><td style='{stfl}'>{total_lbl}</td>"
             f"<td style='{stf}'>{_f0(tn)}</td><td style='{stf}'>100.0%</td>"
             f"<td style='{stf};color:{_cc(tu)}'>{_f0(tu)}</td></tr>")
    hdr = (f"<tr><th style='{sthl}'>{first_col}</th><th style='{sth}'>Notional (USD)</th>"
           f"<th style='{sth}'>% of book</th><th style='{sth}'>Unreal. P&amp;L</th></tr>")
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
            f"font-family:monospace'><thead>{hdr}</thead><tbody>{body}</tbody></table></div>")


# ── Main render ───────────────────────────────────────────────────────────────
@st.fragment
def render_ltr():
    st.markdown("#### 💼 Long-Term Holdings &nbsp;·&nbsp; ETF book")
    st.caption(
        "Your **long-term ETF holdings only** (speculative futures/options are excluded — see the "
        "Risk / VaR tab for those). Notional = market value in USD."
    )

    # ── Update from IBKR (EOD) — self-contained; shares the Risk/P&L pull cooldown ──
    _pc1, _pc2 = st.columns([1, 3])
    with _pc1:
        if st.button("⟳  Update from IBKR (EOD)", key="_ltr_flex_pull", use_container_width=True,
                     help="Pull the latest end-of-day IBKR Flex statement (positions, NAV, cash) into "
                          "the local DB. Same pull as the Risk/VaR & P&L tabs, so it respects the shared "
                          "rate-limit cooldown."):
            import risk
            _lvl, _msg = risk._do_flex_pull()          # runs BEFORE load_sections below → fresh read
            st.session_state["_ltr_pull_msg"] = (_lvl, _msg)
    _pm = st.session_state.get("_ltr_pull_msg")
    if _pm:
        with _pc2:
            {"success": st.success, "warning": st.warning, "info": st.info}.get(_pm[0], st.info)(_pm[1])

    sections = load_sections()
    positions = sections.get("positions", pd.DataFrame())
    book = build_ltr_book(positions)

    if book.empty:
        st.info("No ETF holdings found in the local positions snapshot. "
                "Pull an IBKR statement from the Risk / VaR or P&L tab first.")
        return

    # positions-as-of date
    _asof = ""
    if not positions.empty and "ReportDate" in positions.columns:
        _d = pd.to_datetime(positions["ReportDate"], errors="coerce").dropna()
        if not _d.empty:
            _asof = _d.max().strftime("%Y-%m-%d")
    if _asof:
        st.caption(f"Positions as of **{_asof}**")

    total_notl = float(book["notional_usd"].sum())
    total_upnl = float(book["upnl_usd"].sum())

    # ── KPI row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total notional", _f0(total_notl))
    with k2:
        st.metric("Holdings", f"{len(book)}")
    with k3:
        st.metric("Unrealised P&L", _f0(total_upnl))
    with k4:
        _eq = float(book[book["AssetClass2"] == "Equity"]["notional_usd"].sum())
        st.metric("Equity weight", _pct(100 * _eq / total_notl if total_notl else float("nan")))

    # ── Account cash reconciliation: Total NAV − ETF holdings = cash ─────────
    nav = sections.get("nav", pd.DataFrame())
    total_nav = nav_cash = cash_residual = float("nan")
    if not nav.empty and "Total" in nav.columns:
        _nrow = nav.sort_values("ReportDate").iloc[-1]
        total_nav = float(pd.to_numeric(_nrow.get("Total"), errors="coerce"))
        if "Cash" in nav.columns:
            nav_cash = float(pd.to_numeric(_nrow.get("Cash"), errors="coerce"))
    if math.isfinite(total_nav):
        cash_residual = total_nav - total_notl
        _std  = "font-size:12px;padding:5px 12px;border-bottom:1px solid #E2E8F0"
        _stf  = "font-size:12px;padding:6px 12px;border-top:2px solid #475569;font-weight:800"
        _xtra = (f" &nbsp;·&nbsp; <span style='color:#64748B'>IBKR cash line: {_f0(nav_cash)}</span>"
                 if math.isfinite(nav_cash) else "")
        st.markdown(
            f"<table style='border-collapse:collapse;font-family:monospace'>"
            f"<tr><td style='{_std}'>Total account NAV</td>"
            f"<td style='{_std};text-align:right'>{_f0(total_nav)}</td></tr>"
            f"<tr><td style='{_std}'>&minus; ETF holdings (this book)</td>"
            f"<td style='{_std};text-align:right'>{_f0(total_notl)}</td></tr>"
            f"<tr><td style='{_stf}'>= Cash sitting in account</td>"
            f"<td style='{_stf};text-align:right;color:{_cc(cash_residual)}'>{_f0(cash_residual)}</td></tr>"
            f"</table>",
            unsafe_allow_html=True)
        st.caption(
            "Cash = whole-account **NAV − ETF holdings**. Matches IBKR's cash line closely; any small "
            "gap is options premium + interest/dividend accruals. Note this cash also backs the margin "
            "on your speculative futures book (Risk / VaR tab)." + _xtra)

        # ── Live margin: how much of that cash is tied up as futures margin ──────
        if st.button("⟳  Fetch margin / available funds (live TWS)", key="_ltr_margin_btn",
                     help="Pull live margin & liquidity from IBKR TWS to see how much cash is free "
                          "after the futures-book margin. Requires TWS/Gateway running."):
            import risk_prices as rp
            _m, _mnote = rp.live_account_margin()
            st.session_state["_ltr_margin"] = _m
            st.session_state["_ltr_margin_note"] = _mnote

        _m = st.session_state.get("_ltr_margin", {})
        if _m:
            _netliq = _m.get("NetLiquidation", float("nan"))
            _tcash  = _m.get("TotalCashValue", float("nan"))
            _init   = _m.get("FullInitMarginReq", float("nan"))
            _maint  = _m.get("FullMaintMarginReq", float("nan"))
            _avail  = _m.get("AvailableFunds", float("nan"))
            _excess = _m.get("ExcessLiquidity", float("nan"))
            _md = "font-size:12px;padding:5px 12px;border-bottom:1px solid #E2E8F0"
            _mf = "font-size:12px;padding:6px 12px;border-top:2px solid #475569;font-weight:800"
            # Deployable capital = NetLiq − initial margin (NOT cash − margin: the ETFs are
            # collateral too, so subtracting margin from cash alone understates capacity).
            _mrows = [
                ("Net liquidation (account equity)", _netliq, _md),
                ("&minus; Initial margin req (futures book)", _init, _md),
                ("= Available funds (deployable)", _avail, _mf),
            ]
            _html = "<table style='border-collapse:collapse;font-family:monospace'>"
            for lbl, val, sty in _mrows:
                col = _cc(val) if sty == _mf else "#1A202C"
                _html += (f"<tr><td style='{sty}'>{lbl}</td>"
                          f"<td style='{sty};text-align:right;color:{col}'>{_f0(val)}</td></tr>")
            _html += "</table>"
            st.markdown(_html, unsafe_allow_html=True)
            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                st.metric("Literal cash", _f0(_tcash),
                          help="Actual cash balance (TotalCashValue). Much smaller than deployable "
                               "capital because your ETFs also count as margin collateral.")
            with _c2:
                st.metric("Excess liquidity", _f0(_excess),
                          help="NetLiq − maintenance margin. Buffer before forced liquidation.")
            with _c3:
                st.metric("Init margin used", _f0(_init))
            st.caption(
                f"Your **deployable capital is Available Funds ≈ {_f0(_avail)}** — far above the **literal "
                f"cash ({_f0(_tcash)})**, because your ETF holdings also serve as margin collateral, not "
                f"just cash. The futures book uses **{_f0(_init)}** of initial margin; **excess liquidity "
                f"{_f0(_excess)}** is your cushion before forced liquidation. (Note: subtracting margin "
                f"from cash *alone* would understate capacity and can go negative — the ETFs back the "
                f"margin too.) " + str(st.session_state.get("_ltr_margin_note", "")))
        elif st.session_state.get("_ltr_margin_note"):
            st.caption("⚠️ " + str(st.session_state["_ltr_margin_note"]))

    # ── By exposure (S&P 500, Nasdaq, Gold, …) ───────────────────────────────
    st.markdown("##### By exposure")
    _exp = (book.groupby("Exposure")
                .agg(notional=("notional_usd", "sum"), upnl=("upnl_usd", "sum"),
                     ac=("AssetClass2", "first"))
                .sort_values("notional", ascending=False))
    exp_rows = [(idx, r["notional"], 100 * r["notional"] / total_notl if total_notl else 0.0, r["upnl"])
                for idx, r in _exp.iterrows()]
    st.markdown(_summary_table(exp_rows, "Exposure", "TOTAL"), unsafe_allow_html=True)

    # ── Horizontal bar of notional by exposure (coloured by asset class) ─────
    _bar = _exp.sort_values("notional")
    fig = go.Figure(go.Bar(
        orientation="h",
        y=_bar.index.tolist(),
        x=_bar["notional"].tolist(),
        marker_color=[_AC_COLOUR.get(ac, "#64748B") for ac in _bar["ac"]],
        text=[f"${v:,.0f}" for v in _bar["notional"]],
        textposition="auto",
        hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(220, 40 * len(_bar) + 60),
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
        font=dict(size=11, color="#1A202C"),
        xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#E8EDF5"),
        yaxis=dict(gridcolor="#E8EDF5"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── By asset class (Equity / Rates / Commodity / Crypto) ─────────────────
    st.markdown("##### By asset class")
    _ac = (book.groupby("AssetClass2")
               .agg(notional=("notional_usd", "sum"), upnl=("upnl_usd", "sum"))
               .sort_values("notional", ascending=False))
    ac_rows = [(idx, r["notional"], 100 * r["notional"] / total_notl if total_notl else 0.0, r["upnl"])
               for idx, r in _ac.iterrows()]
    st.markdown(_summary_table(ac_rows, "Asset class", "TOTAL"), unsafe_allow_html=True)

    # ── Per-holding detail ───────────────────────────────────────────────────
    with st.expander("📄  Per-holding detail", expanded=False):
        th  = "background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;padding:4px 8px;text-align:right"
        thl = th.replace("text-align:right", "text-align:left")
        td  = "font-size:11px;padding:3px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
        tdl = td.replace("text-align:right", "text-align:left")
        body = ""
        for _, r in book.iterrows():
            pct = 100 * r["notional_usd"] / total_notl if total_notl else 0.0
            body += (
                f"<tr><td style='{tdl}'><b>{r['Symbol']}</b></td>"
                f"<td style='{tdl};color:#64748B'>{str(r['Description'])[:28]}</td>"
                f"<td style='{tdl}'>{r['Exposure']}</td>"
                f"<td style='{td}'>{r['Quantity']:,.0f}</td>"
                f"<td style='{td}'>{r['MarkPrice']:,.2f} {r['Currency']}</td>"
                f"<td style='{td}'>{_f0(r['notional_usd'])}</td>"
                f"<td style='{td}'>{_pct(pct)}</td>"
                f"<td style='{td};color:{_cc(r['upnl_usd'])}'>{_f0(r['upnl_usd'])}</td></tr>"
            )
        hdr = (f"<tr><th style='{thl}'>Symbol</th><th style='{thl}'>Description</th>"
               f"<th style='{thl}'>Exposure</th><th style='{th}'>Qty</th><th style='{th}'>Mark</th>"
               f"<th style='{th}'>Notional (USD)</th><th style='{th}'>%</th><th style='{th}'>Unreal. P&amp;L</th></tr>")
        st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
                    f"font-family:monospace'><thead>{hdr}</thead><tbody>{body}</tbody></table></div>",
                    unsafe_allow_html=True)
        st.caption("VDST (short T-bills) is effectively a cash/parking position; TLT & LTPZ carry the "
                   "duration/inflation exposure. Notional is USD market value (local × FX-to-USD).")

    # ── Overall asset allocation (IBKR book + external holdings) ─────────────
    st.markdown("---")
    st.markdown("##### 🌐 Overall asset allocation — incl. holdings outside IBKR")
    st.caption(
        "Add your **non-IBKR** holdings below (real estate, PE, other brokers, bank cash). Use "
        "**Illiquid** for real estate / private equity / anything not marked daily. These are added to "
        "your IBKR book (ETF holdings by asset class **+ account cash**) for the overall pie. Speculative "
        "futures are a leveraged overlay, so they're excluded from the wealth allocation."
    )

    _man = _load_manual()
    if _man.empty:
        _man = pd.DataFrame([{"Account": "", "Asset Class": "Illiquid", "$ Amount": 0.0}])
    _man["$ Amount"] = pd.to_numeric(_man["$ Amount"], errors="coerce").fillna(0.0)

    edited = st.data_editor(
        _man, num_rows="dynamic", use_container_width=True, hide_index=True,
        key="_ltr_manual_editor",
        column_config={
            "Account": st.column_config.TextColumn("Account", help="e.g. Home, PE fund, Vanguard ISA, Bank"),
            "Asset Class": st.column_config.SelectboxColumn("Asset Class", options=_ALLOC_CLASSES, required=True),
            "$ Amount": st.column_config.NumberColumn("$ Amount", format="$%d", min_value=0.0),
        },
    )
    if st.button("💾  Save external holdings", key="_ltr_manual_save"):
        _save_manual(edited)
        st.success("Saved external holdings.")

    # Optional look-through: count the speculative book's EQUITY futures notional as
    # Equity exposure (funded from Cash — margin collateral), keeping total unchanged.
    # Signed: a net-short equity book DECREASES Equity and increases Cash. Default OFF —
    # the wealth pie normally treats speculative futures as a leveraged overlay.
    _spec_eq = 0.0
    _incl_spec = st.checkbox(
        "Include speculative equity futures exposure in the allocation "
        "(shifts notional from Cash → Equity; total unchanged)",
        value=False, key="_ltr_incl_spec_eq")
    if _incl_spec:
        from risk import _guess_product

        def _equity_notional(bk) -> float:
            if bk is None or bk.empty:
                return 0.0
            eq = bk[(~bk["is_option"]) &
                    (bk.apply(lambda r: _guess_product(r["Symbol"],
                              r.get("Underlying", "")), axis=1) == "Equities")]
            return float(eq["position_value_base"].sum())

        _live = st.session_state.get("_ltr_spec_live")   # {"notional": float, "ts": str}
        _src = "EOD (Flex)"
        if _live is not None:
            _spec_eq, _src = _live["notional"], f"LIVE {_live['ts']}"
        else:
            try:
                from risk import build_speculative_book
                _spec_eq = _equity_notional(
                    build_speculative_book(load_sections().get("positions")))
            except Exception:
                _spec_eq = 0.0

        _cc1, _cc2 = st.columns([3, 1])
        with _cc1:
            if _spec_eq:
                st.caption(f"Speculative equity futures notional: **{_f0(_spec_eq)}** "
                           f"(net, signed, **{_src}**) — moved from Cash to Equity below.")
            else:
                st.caption(f"No speculative equity futures found ({_src}).")
        with _cc2:
            if st.button("⟳ Live (TWS)", key="_ltr_spec_live_btn", use_container_width=True,
                         help="Re-mark the speculative equity notional from live TWS "
                              "positions & prices instead of the EOD Flex snapshot."):
                try:
                    import risk_prices as _rp
                    with st.spinner("Fetching live positions from TWS…"):
                        _lb, _lnote = _rp.live_positions()
                    if _lb is not None and not _lb.empty:
                        import datetime as _dt
                        st.session_state["_ltr_spec_live"] = {
                            "notional": _equity_notional(_lb),
                            "ts": _dt.datetime.now().strftime("%H:%M:%S")}
                        st.rerun()
                    else:
                        st.warning(f"TWS returned no live book — {_lnote}; keeping EOD.")
                except Exception as _ex:
                    st.warning(f"Live fetch failed ({_ex}) — keeping EOD.")

    # Combine: IBKR ETF book by asset class + IBKR account cash + external manual entries.
    alloc: dict = {}
    for ac, v in book.groupby("AssetClass2")["notional_usd"].sum().items():
        alloc[ac] = alloc.get(ac, 0.0) + float(v)
    if math.isfinite(cash_residual):
        alloc["Cash"] = alloc.get("Cash", 0.0) + cash_residual
    if _spec_eq:
        alloc["Equity"] = alloc.get("Equity", 0.0) + _spec_eq
        alloc["Cash"] = alloc.get("Cash", 0.0) - _spec_eq
    _ext_total = 0.0
    for _, r in edited.iterrows():
        ac = str(r.get("Asset Class") or "").strip()
        amt = pd.to_numeric(r.get("$ Amount"), errors="coerce")
        if ac and pd.notna(amt) and amt != 0:
            alloc[ac] = alloc.get(ac, 0.0) + float(amt)
            _ext_total += float(amt)
    alloc = {k: v for k, v in alloc.items() if abs(v) > 1}
    grand = sum(alloc.values())

    if grand > 0:
        # order by the predefined class order, then any extras
        _labels = ([c for c in _ALLOC_CLASSES if c in alloc]
                   + [c for c in alloc if c not in _ALLOC_CLASSES])
        _vals = [alloc[k] for k in _labels]

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        st.markdown("**Overall allocation**")
        cpie, ctab = st.columns([3, 2])
        with cpie:
            fig_pie = go.Figure(go.Pie(
                labels=_labels, values=_vals, hole=0.5, sort=False,
                marker=dict(colors=[_AC_COLOUR.get(k, "#64748B") for k in _labels],
                            line=dict(color="#FFFFFF", width=1)),
                textinfo="label+percent", textfont=dict(size=12),
                hovertemplate="%{label}: $%{value:,.0f} (%{percent})<extra></extra>",
            ))
            fig_pie.update_layout(
                height=340, margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="#FFFFFF", font=dict(size=11, color="#1A202C"), showlegend=False,
                annotations=[dict(text=f"${grand/1e6:,.2f}M<br>total", x=0.5, y=0.5,
                                  font=dict(size=15, color="#1A202C"), showarrow=False)],
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with ctab:
            _hd = "font-size:12px;padding:5px 10px;background:#0F172A;color:#F8FAFC;font-weight:700"
            _rd = "font-size:12px;padding:5px 10px;border-bottom:1px solid #E2E8F0"
            _ft = "font-size:12px;padding:6px 10px;border-top:2px solid #475569;font-weight:800"
            _h = ("<table style='border-collapse:collapse;width:100%;font-family:monospace'>"
                  f"<tr><th style='{_hd};text-align:left'>Class</th>"
                  f"<th style='{_hd};text-align:right'>USD</th>"
                  f"<th style='{_hd};text-align:right'>%</th></tr>")
            for k in _labels:
                dot = _AC_COLOUR.get(k, "#64748B")
                _h += (f"<tr><td style='{_rd};text-align:left'><span style='color:{dot}'>●</span> {k}</td>"
                       f"<td style='{_rd};text-align:right'>{_f0(alloc[k])}</td>"
                       f"<td style='{_rd};text-align:right'>{100*alloc[k]/grand:,.1f}%</td></tr>")
            _h += (f"<tr><td style='{_ft};text-align:left'>TOTAL</td>"
                   f"<td style='{_ft};text-align:right'>{_f0(grand)}</td>"
                   f"<td style='{_ft};text-align:right'>100.0%</td></tr></table>")
            st.markdown(_h, unsafe_allow_html=True)

        st.caption(
            f"Total net worth (incl. external): **{_f0(grand)}**  =  IBKR **{_f0(grand - _ext_total)}**  +  "
            f"external **{_f0(_ext_total)}**.  Illiquid = real estate / PE / other assets not marked daily.")

        # ── Trading-capital view: net worth with capital earmarked for trading removed ──
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        _tc = st.columns([1.4, 1, 1.6])
        _adj_on = _tc[0].checkbox("Set aside trading capital", value=False, key="_ltr_tc_on",
                                  help="Show your net-worth allocation with the capital that backs your "
                                       "trading positions removed — Cash is reduced by this amount and "
                                       "everything else keeps the same notional; percentages re-base.")
        _trade_cap = _tc[1].number_input("Trading capital (USD)", min_value=0.0, value=500000.0,
                                         step=50000.0, format="%.0f", key="_ltr_tc_amt")
        if _adj_on and _trade_cap > 0:
            _cash_now = alloc.get("Cash", 0.0)
            _applied = min(_trade_cap, max(_cash_now, 0.0))
            if _trade_cap > _cash_now:
                st.warning(
                    f"Trading capital ({_f0(_trade_cap)}) exceeds your Cash ({_f0(_cash_now)}). "
                    f"Cash floored at 0 and net worth reduced by {_f0(_applied)} — to set aside more "
                    "than your cash, the rest would be backed by invested assets, not cash.")
            alloc2 = {k: v for k, v in dict(alloc, Cash=_cash_now - _applied).items() if abs(v) > 1}
            grand2 = grand - _applied
            if grand2 > 0:
                st.markdown("**Overall allocation — excluding trading capital**")
                _cp2, _ct2 = st.columns([3, 2])
                _l2 = ([c for c in _ALLOC_CLASSES if c in alloc2]
                       + [c for c in alloc2 if c not in _ALLOC_CLASSES])
                _v2 = [alloc2[k] for k in _l2]
                with _cp2:
                    _fig2 = go.Figure(go.Pie(
                        labels=_l2, values=_v2, hole=0.5, sort=False,
                        marker=dict(colors=[_AC_COLOUR.get(k, "#64748B") for k in _l2],
                                    line=dict(color="#FFFFFF", width=1)),
                        textinfo="label+percent", textfont=dict(size=12),
                        hovertemplate="%{label}: $%{value:,.0f} (%{percent})<extra></extra>",
                    ))
                    _fig2.update_layout(
                        height=340, margin=dict(l=10, r=10, t=20, b=10),
                        paper_bgcolor="#FFFFFF", font=dict(size=11, color="#1A202C"), showlegend=False,
                        annotations=[dict(text=f"${grand2/1e6:,.2f}M<br>ex-trading", x=0.5, y=0.5,
                                          font=dict(size=15, color="#1A202C"), showarrow=False)],
                    )
                    st.plotly_chart(_fig2, use_container_width=True, key="_ltr_alloc_adj")
                with _ct2:
                    _hd = "font-size:12px;padding:5px 10px;background:#0F172A;color:#F8FAFC;font-weight:700"
                    _rd = "font-size:12px;padding:5px 10px;border-bottom:1px solid #E2E8F0"
                    _ft = "font-size:12px;padding:6px 10px;border-top:2px solid #475569;font-weight:800"
                    _h2 = ("<table style='border-collapse:collapse;width:100%;font-family:monospace'>"
                           f"<tr><th style='{_hd};text-align:left'>Class</th>"
                           f"<th style='{_hd};text-align:right'>USD</th>"
                           f"<th style='{_hd};text-align:right'>%</th></tr>")
                    for k in _l2:
                        dot = _AC_COLOUR.get(k, "#64748B")
                        _h2 += (f"<tr><td style='{_rd};text-align:left'><span style='color:{dot}'>●</span> {k}</td>"
                                f"<td style='{_rd};text-align:right'>{_f0(alloc2[k])}</td>"
                                f"<td style='{_rd};text-align:right'>{100*alloc2[k]/grand2:,.1f}%</td></tr>")
                    _h2 += (f"<tr><td style='{_ft};text-align:left'>TOTAL</td>"
                            f"<td style='{_ft};text-align:right'>{_f0(grand2)}</td>"
                            f"<td style='{_ft};text-align:right'>100.0%</td></tr></table>")
                    st.markdown(_h2, unsafe_allow_html=True)
                st.caption(
                    f"Net worth **excluding {_f0(_applied)} trading capital**: **{_f0(grand2)}**  "
                    f"(= {_f0(grand)} total − {_f0(_applied)}, removed from Cash). Every other holding is "
                    "unchanged in notional; percentages are re-based on the smaller total.")

    # 🧪 stress test at the bottom of the tab (Rajat 2026-08-07: WHOLE net
    # worth, not just the IBKR ETF book — reuse the overall allocation above)
    if grand > 0:
        _render_stress(alloc, spec_eq=_spec_eq, trade_cap=_trade_cap,
                       tc_on_default=_adj_on)


# ── 🧪 Portfolio stress test (Rajat 2026-08-07) ───────────────────────────────
# Class-level: user gives per-class PROXY + IMPLIED VOL and three EQUITY-shock
# scenarios; other classes move via rho(class, equity) x (sigma_cls/sigma_eq) x
# eq_move (rho from 3y of daily proxy returns, sigma = the INPUT implied vols —
# vol ratio + correlation propagate the shock; horizon-free by construction).
# Plus a historical replay of the CURRENT class weights on the proxies' long
# histories (daily-rebalanced blend).
_STRESS_DEFAULTS = {   # class → (default proxy, default implied vol %, descr)
    "Equity": ("^GSPC", 15.0, "S&P 500 index"),
    "Rates": ("TLT", 14.0, "20+yr Treasury ETF"),
    "Commodity": ("GC=F", 16.0, "Gold futures"),
    "Crypto": ("BTC-USD", 45.0, "Bitcoin"),
    "Other": ("^GSPC", 15.0, "S&P 500 index"),
}


# Proxy dropdown options (Rajat 2026-08-07: wanted a short-duration bond
# alternative to TLT — SHY/VGSH 1-3y, SHV 0-1y — plus the usual suspects)
_PROXY_OPTIONS = [
    "^GSPC (S&P 500 index)", "^NDX (Nasdaq 100)", "ES=F (S&P 500 futures)",
    "TLT (20+yr Treasury ETF)", "IEF (7-10yr Treasury ETF)",
    "SHY (1-3yr Treasury ETF)", "VGSH (1-3yr Treasury ETF)",
    "SHV (0-1yr T-bill ETF)",
    "ZT=F (UST 2y futures)", "ZF=F (UST 5y futures)", "ZN=F (UST 10y futures)",
    "GC=F (Gold futures)", "SI=F (Silver futures)", "CL=F (WTI futures)",
    "BTC-USD (Bitcoin)", "ETH-USD (Ethereum)", "DX-Y.NYB (Dollar index)",
]


def _proxy_ticker(v: str) -> str:
    """'GC=F (Gold futures)' → 'GC=F' — the bracket is display-only."""
    return str(v).split(" (")[0].strip()


def _render_stress(alloc: dict | None = None, spec_eq: float = 0.0,
                   trade_cap: float = 0.0, tc_on_default: bool = False):
    import math
    import numpy as np
    import plotly.graph_objects as go
    from charting import _yf_series

    if not alloc:
        book = build_ltr_book()
        if book.empty:
            return
        alloc = book.groupby("AssetClass2")["notional_usd"].sum().to_dict()
    alloc = dict(alloc)
    st.markdown("---")
    st.markdown(
        "<div style='background:#1E293B;color:#F8FAFC;padding:6px 12px;"
        "font-size:13px;font-weight:700;border-radius:6px;display:inline-block;"
        "margin-bottom:6px'>🧪 Stress Test"
        "&nbsp;&nbsp;<span style='font-weight:400;font-size:11px;color:#94A3B8'>"
        "whole net worth · equity-shock scenarios propagated by proxy correlations · historical replay of current weights</span></div>", unsafe_allow_html=True)

    # ── toggles mirroring the views above (Rajat 2026-08-07) ─────────────────
    _tg1, _tg2, _tg3 = st.columns([1.7, 1.9, 1.8])
    inc_spec = _tg1.checkbox(
        "Include speculative Equity futures", value=bool(spec_eq),
        key="_lt_st_spec",
        help="Look-through: the speculative futures' equity notional shifted "
             "from Cash into Equity (amount from the section above).")
    if spec_eq:
        if not inc_spec:      # alloc arrives WITH the shift applied — reverse it
            alloc["Equity"] = alloc.get("Equity", 0.0) - spec_eq
            alloc["Cash"] = alloc.get("Cash", 0.0) + spec_eq
    elif inc_spec:
        _tg3.caption("⚠ enable the speculative look-through above to compute "
                     "the futures notional first")
    set_tc = _tg2.checkbox(
        f"Set aside trading capital (${trade_cap:,.0f})",
        value=bool(tc_on_default), key="_lt_st_tc",
        help="Remove the trading capital (number from the box above) from Cash "
             "before stressing — percentages re-base on the smaller total.")
    if set_tc and trade_cap > 0:
        _cash_now = alloc.get("Cash", 0.0)
        alloc["Cash"] = _cash_now - min(trade_cap, max(_cash_now, 0.0))

    w = pd.Series(alloc)
    w = w[w > 0]
    total = float(w.sum())
    # Cash AND Illiquid sit FLAT in scenarios and earn 0 in the replay — stress
    # only the marked-to-market risk classes (their weight still dilutes the
    # portfolio-level numbers)
    _FLAT = {"CASH", "ILLIQUID"}
    classes = [c for c in w.index if c.upper() not in _FLAT]
    cash_w = float(sum(v for c, v in w.items() if c.upper() in _FLAT))

    # ── editable config: proxy + implied vol per class ───────────────────────
    cfg = pd.DataFrame({
        "Class": classes,
        "Weight %": [100 * float(w[c]) / total for c in classes],
        "Proxy": [f"{_STRESS_DEFAULTS.get(c, ('^GSPC', 15.0, ''))[0]} "
                  f"({_STRESS_DEFAULTS.get(c, ('^GSPC', 15.0, 'S&P 500'))[2]})"
                  for c in classes],
        "Impl vol %": [_STRESS_DEFAULTS.get(c, ("^GSPC", 15.0, ""))[1]
                       for c in classes],
        "ρ vs eq": [None] * len(classes),
    })
    edited = st.data_editor(
        cfg, hide_index=True, use_container_width=True,
        # key includes the class list — row-position edit state would otherwise
        # survive structural changes and put e.g. Commodity's GC=F proxy on
        # whatever class now occupies that row (Rajat hit it 2026-08-07)
        key=f"_lt_stress_cfg_{'_'.join(classes)}",
        disabled=["Class"],
        column_config={
            # Weight % is EDITABLE and may exceed 100 (Rajat: a low-duration
            # bond proxy like SHY needs a levered allocation to match TLT's
            # rate exposure) — financing of the excess is NOT modelled
            "Weight %": st.column_config.NumberColumn(
                format="%.1f", min_value=0.0, max_value=400.0,
                help="editable what-if weight; >100% allowed (implicitly "
                     "financed from cash at 0%)"),
            "Proxy": st.column_config.SelectboxColumn(
                options=_PROXY_OPTIONS, width="medium"),
            "Impl vol %": st.column_config.NumberColumn(format="%.1f"),
            "ρ vs eq": st.column_config.NumberColumn(
                format="%.2f", min_value=-1.0, max_value=1.0,
                help="correlation to Equity used in the scenarios. BLANK = "
                     "auto (trailing 3y historical — POSITIVE stock-bond ρ "
                     "since 2022, so bonds barely hedge). For a 2020-style "
                     "Fed-cut crash, set Rates to −0.4…−0.6."),
        })
    # apply what-if weight overrides ($ = pct × ORIGINAL total)
    w = w.copy()
    for _i, r in edited.iterrows():
        try:
            w[r["Class"]] = float(r["Weight %"]) / 100.0 * total
        except Exception:
            pass
    sc1, sc2, sc3, _sp = st.columns([0.7, 0.7, 0.7, 2.4])
    m1 = sc1.number_input("Equity scen 1 (%)", value=-5.0, step=1.0,
                          key="_lt_scn1")
    m2 = sc2.number_input("Equity scen 2 (%)", value=-10.0, step=1.0,
                          key="_lt_scn2")
    m3 = sc3.number_input("Equity scen 3 (%)", value=-20.0, step=1.0,
                          key="_lt_scn3")

    # ── proxy histories → correlations (3y daily) + replay series ────────────
    rets = {}
    for _i, r in edited.iterrows():
        try:
            s = _yf_series(_proxy_ticker(r["Proxy"]))
            rets[r["Class"]] = s.pct_change().dropna()
        except Exception as ex:
            st.caption(f"⚠ proxy {r['Proxy']} ({r['Class']}): "
                       f"{type(ex).__name__} {str(ex)[:80]}")
    if "Equity" not in rets:
        st.warning("stress needs a working Equity proxy")
        return
    rdf = pd.DataFrame(rets).dropna()
    corr3 = rdf.tail(756).corr()

    ivol = {r["Class"]: float(r["Impl vol %"]) for _i, r in edited.iterrows()}
    eq_vol = max(ivol.get("Equity", 15.0), 1e-6)

    # ── scenario chart + implied-move table ──────────────────────────────────
    scen = [m1, m2, m3]
    pl_by_cls, mv_by_cls = {}, {}
    rho_over = {}
    for _i, r in edited.iterrows():
        _rv = pd.to_numeric(r.get("ρ vs eq"), errors="coerce")
        if pd.notna(_rv):
            rho_over[r["Class"]] = float(_rv)
    for c in classes:
        rho = rho_over.get(
            c, float(corr3.loc[c, "Equity"]) if c in corr3.index else 1.0)
        if c == "Equity":
            rho = 1.0
        beta = rho * ivol.get(c, 15.0) / eq_vol
        mv = [(m if c == "Equity" else beta * m) for m in scen]
        mv_by_cls[c] = mv
        pl_by_cls[c] = [float(w[c]) * m / 100.0 for m in mv]
    totals = [sum(pl_by_cls[c][j] for c in classes) for j in range(3)]

    fig = go.Figure()
    _cols = {"Equity": "#2563EB", "Rates": "#0D9488", "Commodity": "#B45309",
             "Crypto": "#7C3AED", "Other": "#64748B"}
    xlab = [f"S{j + 1}: eq {scen[j]:+.0f}%" for j in range(3)]
    for c in classes:
        fig.add_trace(go.Bar(
            x=xlab, y=pl_by_cls[c], name=c,
            marker_color=_cols.get(c, "#64748B"),
            hovertemplate=("%{x} · " + c + " %{customdata:+.1f}%% · "
                           "$%{y:,.0f}<extra></extra>"),
            customdata=mv_by_cls[c]))
    for j, t in enumerate(totals):
        fig.add_annotation(x=xlab[j], y=t, text=f"<b>${t:,.0f}</b>",
                           showarrow=False, yshift=14 if t >= 0 else -14,
                           font=dict(size=12))
    fig.update_layout(
        barmode="relative", height=380,
        margin=dict(l=30, r=30, t=40, b=30),
        title=dict(text="Scenario P&L by asset class (equity shock propagated "
                        "via correlation × vol-ratio)", font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1,
                    xanchor="right", font_size=10),
        yaxis=dict(gridcolor="#F1F5F9", tickprefix="$", tickformat=",.0f"),
        plot_bgcolor="#FFFFFF")
    st.plotly_chart(fig, use_container_width=True)
    _rows = "".join(
        "<tr><td style='padding:3px 9px;font-size:12px;text-align:left'>"
        f"<b>{c}</b> <span style='color:#94A3B8'>(ρ "
        f"{rho_over.get(c, float(corr3.loc[c, 'Equity']) if c in corr3.index else 1.0):+.2f}"
        f"{'*' if c in rho_over else ''})"
        "</span></td>"
        + "".join(f"<td style='padding:3px 9px;font-size:12px;"
                  f"text-align:right;font-family:monospace'>"
                  f"{mv_by_cls[c][j]:+.1f}%</td>" for j in range(3))
        + "</tr>"
        for c in classes)
    st.markdown(
        "<table style='border-collapse:collapse'><thead><tr>"
        "<th style='padding:3px 9px;font-size:11px;color:#64748B;"
        "text-align:left'>implied move</th>"
        + "".join(f"<th style='padding:3px 9px;font-size:11px;"
                  f"color:#64748B'>{x}</th>" for x in xlab)
        + f"</tr></thead><tbody>{_rows}</tbody></table>",
        unsafe_allow_html=True)

    # ── post-scenario allocation (no rebalancing — Rajat 2026-08-07) ─────────
    st.markdown("**Allocation after each scenario** — assuming no new trades "
                "(risk classes move, Cash/Illiquid unchanged, percentages "
                "re-base on the new total)")
    _all_cls = list(w.index)
    _th2 = ("padding:3px 10px;font-size:11px;color:#64748B;text-align:right;"
            "border-bottom:1px solid #E2E8F0;white-space:nowrap")
    _td2 = ("padding:3px 10px;font-size:12px;text-align:right;"
            "font-family:monospace;border-bottom:1px solid #F1F5F9")
    _new_tot = []
    _new_w = {}
    for j in range(3):
        nw = {}
        for c in _all_cls:
            mv = mv_by_cls.get(c, [0, 0, 0])[j] if c in mv_by_cls else 0.0
            nw[c] = float(w[c]) * (1.0 + mv / 100.0)
        _new_w[j] = nw
        _new_tot.append(sum(nw.values()))
    hdr2 = (f"<tr><th style='{_th2};text-align:left'>ALLOCATION</th>"
            f"<th style='{_th2}'>current</th>"
            + "".join(f"<th style='{_th2}'>{x}</th>" for x in xlab) + "</tr>")
    rows2 = ""
    for c in _all_cls:
        cur = 100.0 * float(w[c]) / total
        cells = ""
        for j in range(3):
            npc = 100.0 * _new_w[j][c] / _new_tot[j] if _new_tot[j] else 0.0
            dcol = ("#16A34A" if npc > cur + 0.05 else
                    "#DC2626" if npc < cur - 0.05 else "#334155")
            cells += (f"<td style='{_td2};color:{dcol}'>{npc:.1f}% "
                      f"<span style='font-size:10px;color:#94A3B8'>"
                      f"({npc - cur:+.1f})</span></td>")
        rows2 += (f"<tr><td style='{_td2};text-align:left'><b>{c}</b></td>"
                  f"<td style='{_td2}'>{cur:.1f}%</td>{cells}</tr>")
    rows2 += (f"<tr><td style='{_td2};text-align:left'><b>TOTAL ($)</b></td>"
              f"<td style='{_td2}'>${total:,.0f}</td>"
              + "".join(f"<td style='{_td2}'>${t:,.0f} "
                        f"<span style='font-size:10px;color:#94A3B8'>"
                        f"({100 * (t / total - 1):+.1f}%)</span></td>"
                        for t in _new_tot) + "</tr>")
    st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:"
                f"collapse'><thead>{hdr2}</thead><tbody>{rows2}</tbody></table>"
                f"</div>", unsafe_allow_html=True)

    # ── historical replay of current weights ─────────────────────────────────
    st.markdown("**Historical replay** — current class weights, daily "
                "rebalanced, on the proxies' common history")
    hw1, hw2 = st.columns([0.8, 3.2])
    hyrs = hw1.selectbox("Window", ["1y", "2y", "3y", "5y", "10y", "20y", "max"],
                         index=5, key="_lt_hist_yrs")
    # per-class overlay checkboxes (Rajat 2026-08-07): tick classes to see each
    # one's own history + drawdown alongside the blended portfolio
    _avail = [c for c in classes if c in rdf.columns]
    _cbcols = hw2.columns(max(len(_avail), 1))
    show_cls = [c for i, c in enumerate(_avail)
                if _cbcols[i].checkbox(c, key=f"_lt_hcb_{c}")]

    wts = {c: float(w[c]) / total for c in classes if c in rdf.columns}
    rwin = rdf
    if hyrs != "max":
        rwin = rdf[rdf.index >= pd.Timestamp.today()
                   - pd.DateOffset(years=int(hyrs[:-1]))]
    if len(rwin) < 60:
        st.caption("not enough common history for the chosen proxies")
        return
    port = sum(rwin[c] * wt for c, wt in wts.items())
    cum = (1 + port).cumprod() * 100
    dd = (cum / cum.cummax() - 1) * 100
    # overlays = WEIGHT-SCALED contribution paths (Rajat 2026-08-07: raw proxy
    # lines made a 200%-weighted VGSH sleeve look like a flatline — the overlay
    # now shows what that sleeve actually contributes at YOUR weight)
    cls_cum, cls_dd, cls_lbl = {}, {}, {}
    for c in show_cls:
        _wt = wts.get(c, 0.0)
        cc_ = (1 + rwin[c] * _wt).cumprod() * 100
        cls_cum[c] = cc_
        cls_dd[c] = (cc_ / cc_.cummax() - 1) * 100
        cls_lbl[c] = f"{c} (wt {_wt * 100:.0f}%)"

    # ── chart 1: cumulative performance ──────────────────────────────────────
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines",
                              name="portfolio",
                              line=dict(color="#0F172A", width=2.4),
                              hovertemplate="%{x|%b %y} · %{y:,.1f}"
                                            "<extra>portfolio</extra>"))
    for c in show_cls:
        figp.add_trace(go.Scatter(
            x=cls_cum[c].index, y=cls_cum[c].values, mode="lines",
            name=cls_lbl[c],
            line=dict(color=_cols.get(c, "#64748B"), width=1.3),
            hovertemplate="%{x|%b %y} · %{y:,.1f}<extra>"
                          + cls_lbl[c] + "</extra>"))
    figp.update_layout(
        height=360, margin=dict(l=30, r=30, t=40, b=25),
        title=dict(text="Cumulative performance (idx = 100 at window start)",
                   font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1,
                    xanchor="right", font_size=10),
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9"),
        hovermode="x unified", plot_bgcolor="#FFFFFF")
    st.plotly_chart(figp, use_container_width=True)

    # ── chart 2: drawdowns ───────────────────────────────────────────────────
    figd = go.Figure()
    figd.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines",
                              name="portfolio",
                              line=dict(color="#0F172A", width=2),
                              fill="tozeroy",
                              fillcolor="rgba(15,23,42,0.07)",
                              hovertemplate="%{x|%b %y} · %{y:.1f}%"
                                            "<extra>portfolio</extra>"))
    for c in show_cls:
        figd.add_trace(go.Scatter(
            x=cls_dd[c].index, y=cls_dd[c].values, mode="lines",
            name=cls_lbl[c],
            line=dict(color=_cols.get(c, "#64748B"), width=1.2),
            hovertemplate="%{x|%b %y} · %{y:.1f}%<extra>"
                          + cls_lbl[c] + "</extra>"))
    figd.update_layout(
        height=300, margin=dict(l=30, r=30, t=40, b=25),
        title=dict(text="Drawdown from peak (%)", font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1,
                    xanchor="right", font_size=10),
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9", ticksuffix="%"),
        hovermode="x unified", plot_bgcolor="#FFFFFF")
    st.plotly_chart(figd, use_container_width=True)

    yrs_n = max((port.index[-1] - port.index[0]).days / 365.25, 1e-9)
    cagr = (float(cum.iloc[-1]) / 100) ** (1 / yrs_n) - 1
    _cls_stats = ""
    for c in show_cls:
        _cyrs = yrs_n
        _ccagr = (float(cls_cum[c].iloc[-1]) / 100) ** (1 / _cyrs) - 1
        _cls_stats += (f" · {c}: {_ccagr * 100:+.1f}% CAGR, "
                       f"maxDD {float(cls_dd[c].min()):.1f}%")
    st.caption(
        f"Portfolio: CAGR **{cagr * 100:+.1f}%** · ann. vol "
        f"**{float(port.std()) * math.sqrt(252) * 100:.1f}%** · max drawdown "
        f"**{float(dd.min()):.1f}%** (weights: "
        + ", ".join(f"{c} {wt * 100:.0f}%" for c, wt in wts.items())
        + (f", Cash+Illiquid {100 * cash_w / total:.0f}% flat" if cash_w else "")
        + ")" + _cls_stats
        + ". ρ from 3y daily proxy returns; scenario moves use YOUR implied "
          "vols (only the ratio matters) — correlations are historical, and in "
          "a crash they usually rise. Cash/margin not modelled.")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from watchlist import fetch_batch, fetch_chart_data, build_instrument_chart
import rates as _rates

# ── Rates implied-vol monitor (MOVE-inferred) ──────────────────────────────────
# MOVE = ICE BofA implied 1m yield vol on the UST 2/5/10/30 basket: 10y=40%, rest 20%.
_MOVE_WEIGHTS = {"2Y": 0.20, "5Y": 0.20, "10Y": 0.40, "30Y": 0.20}
_RV_WINDOWS   = {"1w": 5, "2w": 10, "1m": 21, "3m": 63}


# CBOE constant-maturity yield indices — 0.1bp resolution (vs treasury.gov's 1bp, which
# quantizes the calm long-end to lots of 0.00 daily changes and understates realized vol).
_CBOE_US = {"5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}


@st.cache_data(ttl=3600, show_spinner=False)
def _cboe_us_yields() -> pd.DataFrame:
    out: dict = {}
    for ten, tk in _CBOE_US.items():
        try:
            s = yf.Ticker(tk).history(period="1y")["Close"].dropna()
            s.index = pd.to_datetime(s.index).tz_localize(None)
            out[ten] = s
        except Exception:
            pass
    return pd.DataFrame(out)


@st.cache_data(ttl=3600, show_spinner=False)
def _rates_ivol_data() -> dict:
    """MOVE index series + US & EUR par yields (for realized-vol inference)."""
    out: dict = {"move": pd.Series(dtype=float), "us": pd.DataFrame(), "eur": pd.DataFrame()}
    try:
        mv = yf.Ticker("^MOVE").history(period="1y")["Close"].dropna()
        mv.index = pd.to_datetime(mv.index).tz_localize(None)
        out["move"] = mv
    except Exception:
        pass
    try:
        us_t = _rates.fetch_us_treasury(months_back=6)
        cboe = _cboe_us_yields()
        if not us_t.empty:
            idx = us_t.index
            cols: dict = {}
            for ten in ("1Y", "2Y", "5Y", "10Y", "30Y"):   # 1Y is inferred-only (not in MOVE)
                if ten in cboe.columns and cboe[ten].notna().sum() > 10:
                    cols[ten] = cboe[ten].reindex(idx, method="ffill")   # finer 0.1bp series
                elif ten in us_t.columns:
                    cols[ten] = pd.to_numeric(us_t[ten], errors="coerce")
            out["us"] = pd.DataFrame(cols, index=idx)
    except Exception:
        pass
    try:
        _start = (pd.Timestamp.today() - pd.Timedelta(days=200)).date().isoformat()
        out["eur"] = _rates.fetch_ecb_curve(start=_start)
    except Exception:
        pass
    return out


def _realized_bp_vol(yield_series: pd.Series, window: int) -> float:
    """Annualized realized NORMAL vol in bp/yr from the last `window` daily yield changes.
    Yields are in percent → ×100 to basis points; annualize with √252."""
    s = pd.to_numeric(yield_series, errors="coerce").dropna()
    d = s.diff().dropna()
    if len(d) > window:
        d = d.iloc[-window:]
    if len(d) < 3:
        return float("nan")
    return float(d.std(ddof=1) * 100.0 * np.sqrt(252))


def _flat_days(yield_series: pd.Series, window: int) -> tuple[int, int]:
    """(#zero daily changes, #obs) over the window — flags 1bp-rounding-deflated realized vol."""
    s = pd.to_numeric(yield_series, errors="coerce").dropna()
    d = s.diff().dropna()
    if len(d) > window:
        d = d.iloc[-window:]
    return int((d.abs() < 1e-9).sum()), int(len(d))


def _ratio_history(mv: pd.Series, us: pd.DataFrame, window: int) -> pd.Series:
    """Rolling MOVE-implied / US-basket-realized ratio (same `window` as the table)."""
    tenors = [t for t in _MOVE_WEIGHTS if t in us.columns]
    if mv.empty or not tenors:
        return pd.Series(dtype=float)
    diffs = us[tenors].apply(pd.to_numeric, errors="coerce").diff()
    rv = diffs.rolling(window).std(ddof=1) * 100.0 * np.sqrt(252)          # per-tenor rolling bp vol
    basket = sum(_MOVE_WEIGHTS[t] * rv[t] for t in tenors)                 # MOVE-weighted basket
    basket = basket[basket > 0].dropna()
    if basket.empty:
        return pd.Series(dtype=float)
    move_aligned = mv.reindex(basket.index, method="ffill")               # MOVE onto basket dates
    ratio = (move_aligned / basket).replace([np.inf, -np.inf], np.nan).dropna()
    return ratio.iloc[-180:]                                              # ~last 9 months

# ── Instrument definitions ─────────────────────────────────────────────────────
VOL_INSTRUMENTS = [
    {"name": "VIX",   "ticker": "^VIX",   "group": "Equity",    "desc": "S&P 500 30-day implied vol"},
    {"name": "VVIX",  "ticker": "^VVIX",  "group": "Equity",    "desc": "VIX of VIX — vol of vol"},
    {"name": "VXN",   "ticker": "^VXN",   "group": "Equity",    "desc": "NASDAQ 100 implied vol"},
    {"name": "VXD",   "ticker": "^VXD",   "group": "Equity",    "desc": "Dow Jones implied vol"},
    {"name": "VIX3M", "ticker": "^VIX3M", "group": "VIX Term",  "desc": "S&P 500 3-month implied vol"},
    {"name": "SKEW",  "ticker": "^SKEW",  "group": "Skew",      "desc": "S&P 500 tail risk index"},
    {"name": "MOVE",  "ticker": "^MOVE",  "group": "Rates",     "desc": "ICE BofA bond market implied vol"},
    {"name": "OVX",   "ticker": "^OVX",   "group": "Commodity", "desc": "Crude Oil implied vol"},
    {"name": "GVZ",   "ticker": "^GVZ",   "group": "Commodity", "desc": "Gold implied vol"},
]

_TICKERS = tuple(i["ticker"] for i in VOL_INSTRUMENTS)
_NAME_MAP = {i["ticker"]: i["name"] for i in VOL_INSTRUMENTS}
_DESC_MAP = {i["ticker"]: i["desc"] for i in VOL_INSTRUMENTS}

# ── Realized vol instruments ───────────────────────────────────────────────────
RVOL_INSTRUMENTS = [
    {"name": "S&P 500", "ticker": "^GSPC",      "class": "Equity",    "rates": False, "iv_ticker": "^VIX"},
    {"name": "NASDAQ",  "ticker": "^IXIC",      "class": "Equity",    "rates": False, "iv_ticker": "^VXN"},
    {"name": "BBDXY",   "ticker": "BBDXY_SYNTH","class": "FX",        "rates": False, "iv_ticker": None},
    {"name": "EUR/USD", "ticker": "EURUSD=X",   "class": "FX",        "rates": False, "iv_ticker": None},
    {"name": "GBP/USD", "ticker": "GBPUSD=X",   "class": "FX",        "rates": False, "iv_ticker": None},
    {"name": "USD/JPY", "ticker": "JPY=X",      "class": "FX",        "rates": False, "iv_ticker": None},
    {"name": "US 10Y",  "ticker": "^TNX",       "class": "Rates",     "rates": True,  "iv_ticker": None},
    {"name": "US 2Y",   "ticker": "^US2YT",     "class": "Rates",     "rates": True,  "iv_ticker": None},
    {"name": "EUR 10Y", "ticker": "^ECB10Y",    "class": "Rates",     "rates": True,  "iv_ticker": None},
    {"name": "EUR 2Y",  "ticker": "^ECB2Y",     "class": "Rates",     "rates": True,  "iv_ticker": None},
    {"name": "Brent",   "ticker": "BZ=F",       "class": "Commodity", "rates": False, "iv_ticker": None},
    {"name": "WTI",     "ticker": "CL=F",       "class": "Commodity", "rates": False, "iv_ticker": "^OVX"},
    {"name": "Gold",    "ticker": "GC=F",       "class": "Commodity", "rates": False, "iv_ticker": "^GVZ"},
    {"name": "Bitcoin", "ticker": "BTC-USD",    "class": "Crypto",    "rates": False, "iv_ticker": None},
]


def _fetch_rvol_series(ticker: str) -> pd.Series:
    try:
        df = fetch_chart_data(ticker, period="2y", interval="1d")
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df["Close"].squeeze().astype(float).dropna()
    except Exception:
        return pd.Series(dtype=float)


def _ann_rvol(close: pd.Series, window: int, is_rates: bool) -> float | None:
    s = close.dropna()
    if len(s) < window + 1:
        return None
    if is_rates:
        daily = s.diff().dropna() * 100        # % points → bps
    else:
        daily = np.log(s / s.shift(1)).dropna() * 100   # log returns in %
    tail = daily.iloc[-window:]
    if len(tail) < window:
        return None
    rv = float(tail.std() * np.sqrt(252))
    return rv if np.isfinite(rv) else None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _vix_regime(level: float) -> tuple:
    """Return (label, color) for a VIX level."""
    if level < 15:  return "Low",      "#059669"
    if level < 20:  return "Normal",   "#0EA5E9"
    if level < 30:  return "Elevated", "#D97706"
    if level < 40:  return "High",     "#DC2626"
    return              "Extreme",     "#7C2D12"


def _pct_change(series: pd.Series, n: int) -> float | None:
    s = series.dropna()
    if len(s) <= n:
        return None
    return float((s.iloc[-1] - s.iloc[-1 - n]) / s.iloc[-1 - n] * 100)


def _abs_change(series: pd.Series, n: int) -> float | None:
    s = series.dropna()
    if len(s) <= n:
        return None
    return float(s.iloc[-1] - s.iloc[-1 - n])


def _fmt_change(val: float | None, suffix: str = "") -> str:
    if val is None:
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.2f}{suffix}"


def _col_change(val: float | None) -> str:
    if val is None:
        return "#64748B"
    return "#059669" if val >= 0 else "#DC2626"


def _render_rates_ivol_monitor():
    """MOVE-inferred implied vol for USD & EUR 2/5/10/30, from the implied/realized ratio."""
    st.markdown("#### Rates Implied Vol Monitor &nbsp;·&nbsp; MOVE-inferred")

    win_lbl = st.radio("Realized window", list(_RV_WINDOWS.keys()), index=2, horizontal=True,
                       key="_rates_ivol_win")
    window = _RV_WINDOWS[win_lbl]

    data = _rates_ivol_data()
    mv, us, eur = data["move"], data["us"], data["eur"]
    if mv.empty or us.empty:
        st.info("MOVE index or US Treasury data unavailable right now.")
        return

    move_implied = float(mv.iloc[-1])
    _basket_tenors = ["2Y", "5Y", "10Y", "30Y"]        # MOVE constituents
    _disp_tenors   = ["1Y"] + _basket_tenors           # 1Y = inferred-only, not in basket

    rv_us = {t: _realized_bp_vol(us[t], window) for t in _disp_tenors if t in us.columns}
    basket_rv = sum(_MOVE_WEIGHTS[t] * rv_us[t]
                    for t in _basket_tenors if np.isfinite(rv_us.get(t, np.nan)))
    if not np.isfinite(basket_rv) or basket_rv <= 0:
        st.info("Could not compute the US-basket realized vol for this window.")
        return
    ratio = move_implied / basket_rv
    rv_eur = ({t: _realized_bp_vol(eur[t], window) for t in _disp_tenors if t in eur.columns}
              if not eur.empty else {})

    st.caption(
        f"**MOVE implied {move_implied:.1f} bp**  ·  US-basket realized ({win_lbl}) "
        f"**{basket_rv:.1f} bp**  ·  implied/realized ratio **{ratio:.2f}×** — applied to every "
        f"bucket below. Realized = annualized bp normal vol of daily yield changes over the window; "
        f"inferred implied = realized × ratio (USD 2/5/10/30 reweighted reconstruct MOVE exactly; "
        f"1Y rides the same premium but is outside the MOVE basket)."
    )

    rows, flagged = [], []
    for curve, rv, frame in (("USD", rv_us, us), ("EUR", rv_eur, eur)):
        for t in _disp_tenors:
            if frame.empty or t not in frame.columns:
                continue
            r = rv.get(t, float("nan"))
            inf = r * ratio if np.isfinite(r) else float("nan")
            nz, n = _flat_days(frame[t], window)
            is_flag = n >= 5 and nz >= 3 and nz >= 0.30 * n     # rounding-deflated realized
            if is_flag:
                flagged.append(f"{curve} {t} ({nz}/{n} flat days)")
            rows.append({"Curve": curve, "Tenor": f"{t} ⚠" if is_flag else t,
                         "Realized (bp)": r, "Inferred Implied (bp)": inf,
                         "Impl − Real (bp)": inf - r if np.isfinite(r) else float("nan"),
                         "Weight": (f"{_MOVE_WEIGHTS[t]*100:.0f}%"
                                    if (curve == "USD" and t in _basket_tenors) else "—")})
    df = pd.DataFrame(rows).set_index(["Curve", "Tenor"])

    def _spread_colour(col):
        return [f"color: {'#059669' if v >= 0 else '#DC2626'}; font-weight:600"
                if np.isfinite(v) else "" for v in col]

    styled = (df.style
                .format({"Realized (bp)": "{:.1f}", "Inferred Implied (bp)": "{:.1f}",
                         "Impl − Real (bp)": "{:+.1f}"}, na_rep="—")
                .set_properties(subset=["Inferred Implied (bp)"], **{"font-weight": "700"})
                .apply(_spread_colour, subset=["Impl − Real (bp)"]))
    st.dataframe(styled, use_container_width=True)
    if flagged:
        st.warning(
            "⚠ Realized vol likely **understated by 1bp source rounding** (many zero-change days) for: "
            + "; ".join(flagged) + ". These sit on the coarser treasury.gov / ECB feed; when a tenor goes "
            "very quiet, sub-basis-point moves round to 0.00 and deflate the realized std. Treat the "
            "flagged buckets' realized (and their inferred implied) with caution.")
    st.caption("EUR buckets inherit the same USD implied/realized premium (no EUR options index used). "
               "MOVE ≈ 1-month implied; **Impl − Real** > 0 = a positive vol premium (implied above realized). "
               "A ratio < 1 means realized has recently run above the implied basis. "
               "**1Y** rows are inferred-only (outside the MOVE basket).")

    # ── Implied/realized ratio history (rolling) ──────────────────────────────
    rh = _ratio_history(mv, us, window)
    if not rh.empty and len(rh) > 5:
        fig = go.Figure(go.Scatter(
            x=rh.index, y=rh.values, mode="lines", line=dict(color="#2563EB", width=2),
            hovertemplate="%{x|%d %b %y}: %{y:.2f}×<extra></extra>"))
        fig.add_hline(y=1.0, line_dash="dot", line_color="#94A3B8",
                      annotation_text="parity (implied = realized)", annotation_position="top left")
        _cur = float(rh.iloc[-1])
        fig.update_layout(
            height=200, margin=dict(l=10, r=10, t=28, b=10),
            title=dict(text=f"MOVE implied / basket realized ({win_lbl}) — now {_cur:.2f}×", font=dict(size=12)),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD", font=dict(size=10, color="#1A202C"),
            yaxis=dict(gridcolor="#E8EDF5"), xaxis=dict(gridcolor="#E8EDF5"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Rolling implied/realized ratio on the US basket. Above 1× = market pricing vol "
                   "richer than recently realized (typical); below 1× = realized outrunning implied.")


def _build_vol_stats() -> dict:
    """{ticker: level / 1d / 1w / 1m / 52w stats} for every VOL_INSTRUMENTS
    member — shared by the Monitor and the Vol Dash indices table."""
    batch = fetch_batch(_TICKERS)
    stats = {}
    for inst in VOL_INSTRUMENTS:
        tkr = inst["ticker"]
        df  = batch.get(tkr)
        if df is None or df.empty:
            continue
        close = df["Close"].squeeze().astype(float).dropna()
        if close.empty:
            continue
        stats[tkr] = {
            "level":  float(close.iloc[-1]),
            "d1":     _abs_change(close, 1),
            "d1pct":  _pct_change(close, 1),
            "d5":     _abs_change(close, 5),
            "d21":    _abs_change(close, 21),
            "hi52":   float(close.tail(252).max()),
            "lo52":   float(close.tail(252).min()),
            "close":  close,
        }
    return stats


def render_indices_table(stats: dict | None = None,
                         header: str = "#### All Indices") -> None:
    """The Monitor's vol-indices summary table, reusable from other tabs
    (the 🌐 Vol Dash shows it on top). Fetches its own stats when not given."""
    if stats is None:
        with st.spinner("Loading vol indices…"):
            stats = _build_vol_stats()
    if not stats:
        st.caption("⚠ vol indices unavailable")
        return
    if header:
        st.markdown(header)
    rows = []
    for inst in VOL_INSTRUMENTS:
        tkr = inst["ticker"]
        s   = stats.get(tkr)
        if not s:
            continue
        hi52, lo52 = s["hi52"], s["lo52"]
        pct_of_range = (s["level"] - lo52) / (hi52 - lo52) * 100 if hi52 != lo52 else 50

        rows.append({
            "Index":       inst["name"],
            "Description": inst["desc"],
            "Level":       f"{s['level']:.2f}",
            "1D Δ (pts)":  _fmt_change(s["d1"]),
            "1D %":        _fmt_change(s["d1pct"], "%"),
            "1W Δ (pts)":  _fmt_change(s["d5"]),
            "1M Δ (pts)":  _fmt_change(s["d21"]),
            "52W High":    f"{hi52:.2f}",
            "52W Low":     f"{lo52:.2f}",
            "52W %ile":    f"{pct_of_range:.0f}%",
        })

    if not rows:
        return
    df_tbl = pd.DataFrame(rows).set_index("Index")

    def _style_row(row):
        styles = [""] * len(row)
        for i, col in enumerate(row.index):
            if col in ("1D Δ (pts)", "1D %", "1W Δ (pts)", "1M Δ (pts)"):
                try:
                    v = float(str(row[col]).replace("%", "").replace("+", ""))
                    styles[i] = f"color: {'#059669' if v >= 0 else '#DC2626'}; font-weight: 600"
                except Exception:
                    pass
        return styles

    styled = df_tbl.style.apply(_style_row, axis=1)
    st.dataframe(styled, use_container_width=True)


# ── Main render ────────────────────────────────────────────────────────────────

@st.fragment
def render_volatility():
    hdr_col, btn_col = st.columns([5, 1])
    with hdr_col:
        st.markdown("### Volatility Monitor")
    with btn_col:
        if st.button("⟳ Refresh", use_container_width=True, help="Reload latest data"):
            # Targeted refresh: only the caches this tab reads.
            from watchlist import _all_daily
            _all_daily.clear()          # watchlist: shared daily OHLCV (fetch_batch)
            fetch_chart_data.clear()    # watchlist: VIX/instrument chart bars
            _cboe_us_yields.clear()     # local: CBOE US yield curve
            _rates_ivol_data.clear()    # local: rates implied-vol monitor
            st.rerun()

    with st.spinner("Loading volatility data…"):
        stats = _build_vol_stats()

    if not stats:
        st.warning("Could not load volatility data.")
        return

    # ── Hero cards: VIX, VVIX, OVX, GVZ ──────────────────────────────────────
    hero_tickers = ["^VIX", "^VVIX", "^OVX", "^GVZ"]
    hero_cols    = st.columns(len(hero_tickers))

    for col, tkr in zip(hero_cols, hero_tickers):
        s = stats.get(tkr)
        if not s:
            col.metric(_NAME_MAP[tkr], "—")
            continue
        level    = s["level"]
        d1       = s["d1"]
        d1_str   = _fmt_change(d1, "pts")
        regime, rcolor = _vix_regime(level) if tkr == "^VIX" else ("", "")

        col.markdown(
            f"<div style='background:#FFFFFF;border:1px solid #E8EDF5;border-radius:10px;"
            f"padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,0.05)'>"
            f"<div style='font-size:0.72rem;font-weight:600;letter-spacing:0.4px;"
            f"text-transform:uppercase;color:#64748B'>{_NAME_MAP[tkr]}</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:#1A202C;margin:4px 0'>"
            f"{level:.2f}</div>"
            f"<div style='font-size:0.82rem;font-weight:600;color:{_col_change(d1)}'>"
            f"{d1_str} today</div>"
            + (f"<div style='margin-top:6px;display:inline-block;padding:2px 8px;"
               f"border-radius:12px;background:{rcolor}22;color:{rcolor};"
               f"font-size:0.75rem;font-weight:700'>{regime}</div>" if regime else "")
            + f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── VIX Term Structure bar ─────────────────────────────────────────────────
    vix_s   = stats.get("^VIX")
    vix3m_s = stats.get("^VIX3M")
    if vix_s and vix3m_s:
        vix_l   = vix_s["level"]
        vix3m_l = vix3m_s["level"]
        ratio   = vix3m_l / vix_l if vix_l else 1.0
        structure = "Contango" if ratio >= 1 else "Backwardation"
        struct_color = "#0EA5E9" if ratio >= 1 else "#DC2626"
        st.markdown(
            f"**VIX Term Structure** — VIX: **{vix_l:.2f}**  ·  VIX3M: **{vix3m_l:.2f}**  ·  "
            f"Ratio (3M/Spot): **{ratio:.3f}**  ·  "
            f"<span style='color:{struct_color};font-weight:700'>{structure}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Contango (ratio > 1) = normal; Backwardation (ratio < 1) = stress signal, spot vol > forward vol")
        st.markdown("")

    # ── Summary table (shared renderer — also shown on top of the Vol Dash) ────
    render_indices_table(stats)

    # ── Rates implied-vol monitor (MOVE-inferred) ──────────────────────────────
    _render_rates_ivol_monitor()

    # ── Realised Vol table ─────────────────────────────────────────────────────
    st.markdown("#### Realised Volatility")
    st.caption(
        "Annualized realized vol — Equities / FX / Commodities / Crypto in **%** · "
        "Rates in **bps** (daily yield changes). "
        "Windows: 1W = 5d, 1M = 21d, 3M = 63d, 6M = 126d."
    )

    try:
        def _fmt_rv(v: float | None) -> str:
            return f"{v:.1f}" if v is not None else "—"

        rvol_rows = []
        for inst in RVOL_INSTRUMENTS:
            is_rates = inst["rates"]
            unit     = "bps" if is_rates else "%"
            iv_tkr   = inst.get("iv_ticker")
            iv_level = stats[iv_tkr]["level"] if iv_tkr and iv_tkr in stats else None
            try:
                close = _fetch_rvol_series(inst["ticker"])
                rv1w  = _ann_rvol(close, 5,   is_rates)
                rv1m  = _ann_rvol(close, 21,  is_rates)
                rv3m  = _ann_rvol(close, 63,  is_rates)
                rv6m  = _ann_rvol(close, 126, is_rates)
            except Exception:
                rv1w = rv1m = rv3m = rv6m = None
            rvol_rows.append({
                "Asset":    inst["name"],
                "Class":    inst["class"],
                "Unit":     unit,
                "Impl. Vol": f"{iv_level:.2f}" if iv_level is not None else "—",
                "1W RV":    _fmt_rv(rv1w),
                "1M RV":    _fmt_rv(rv1m),
                "3M RV":    _fmt_rv(rv3m),
                "6M RV":    _fmt_rv(rv6m),
            })

        df_rv = pd.DataFrame(rvol_rows).set_index("Asset")

        colors_1m = []
        for r in rvol_rows:
            try:
                v1m = float(r["1M RV"]) if r["1M RV"] != "—" else None
                v3m = float(r["3M RV"]) if r["3M RV"] != "—" else None
                if v1m is not None and v3m is not None and v3m > 0:
                    if v1m > v3m * 1.2:
                        colors_1m.append("color: #DC2626; font-weight: 700")
                    elif v1m < v3m * 0.8:
                        colors_1m.append("color: #059669; font-weight: 700")
                    else:
                        colors_1m.append("")
                else:
                    colors_1m.append("")
            except Exception:
                colors_1m.append("")

        styled_rv = df_rv.style.apply(lambda _: colors_1m, subset=["1M RV"])  # type: ignore[arg-type]
        st.dataframe(styled_rv, use_container_width=True)

    except Exception as e:
        st.error(f"Realised vol error: {e}")

    st.markdown("")

    # ── Historical chart ───────────────────────────────────────────────────────
    st.markdown("#### Historical chart")
    avail_names = [i["name"] for i in VOL_INSTRUMENTS if i["ticker"] in stats]
    avail_tickers = [i["ticker"] for i in VOL_INSTRUMENTS if i["ticker"] in stats]

    ch1, ch2, ch3 = st.columns([3, 2, 2])
    with ch1:
        sel_name = st.selectbox("Index", avail_names, key="vol_sel")
    with ch2:
        tf = st.segmented_control(
            "Timeframe", ["3M", "6M", "1Y", "2Y", "5Y"], default="1Y",
            key="vol_tf",
        )
    with ch3:
        compare = st.selectbox(
            "Compare with",
            ["None"] + [n for n in avail_names if n != sel_name],
            key="vol_compare",
        )

    _tf_map = {"3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"}
    period   = _tf_map.get(tf, "1y")
    sel_tkr  = avail_tickers[avail_names.index(sel_name)]

    with st.spinner("Loading chart data…"):
        df_chart = fetch_chart_data(sel_tkr, period=period, interval="1d")

    if df_chart.empty:
        st.warning("No chart data available.")
        return

    close = df_chart["Close"].squeeze().astype(float)

    # Build figure
    fig = go.Figure()

    # Main vol index line
    fig.add_trace(go.Scatter(
        x=df_chart.index, y=close,
        name=sel_name,
        line=dict(color="#0EA5E9", width=2),
        fill="tozeroy", fillcolor="rgba(14,165,233,0.06)",
        hovertemplate=f"<b>{sel_name}</b>: %{{y:.2f}}<extra></extra>",
    ))

    # Optional comparison (secondary y-axis)
    cmp_data = None
    if compare != "None":
        cmp_tkr = avail_tickers[avail_names.index(compare)]
        with st.spinner(f"Loading {compare}…"):
            df_cmp = fetch_chart_data(cmp_tkr, period=period, interval="1d")
        if not df_cmp.empty:
            cmp_data = df_cmp["Close"].squeeze().astype(float)
            fig.add_trace(go.Scatter(
                x=df_cmp.index,
                y=cmp_data,
                name=compare,
                yaxis="y2",
                line=dict(color="#F59E0B", width=2),
                hovertemplate=f"<b>{compare}</b>: %{{y:.2f}}<extra></extra>",
            ))

    # VIX regime bands (only when showing VIX)
    if sel_tkr == "^VIX":
        for y0, y1, color, label in [
            (0,  15, "rgba(5,150,105,0.05)",   "Low (<15)"),
            (15, 20, "rgba(14,165,233,0.05)",  "Normal (15-20)"),
            (20, 30, "rgba(217,119,6,0.05)",   "Elevated (20-30)"),
            (30, 40, "rgba(220,38,38,0.05)",   "High (30-40)"),
            (40, 90, "rgba(124,45,18,0.05)",   "Extreme (>40)"),
        ]:
            fig.add_trace(go.Scatter(
                x=[df_chart.index[0], df_chart.index[-1], df_chart.index[-1], df_chart.index[0]],
                y=[y0, y0, y1, y1],
                fill="toself", fillcolor=color,
                line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ))
        for y, label, color in [(15, "15", "#94A3B8"), (20, "20", "#D97706"),
                                 (30, "30", "#DC2626"), (40, "40", "#7C2D12")]:
            fig.add_trace(go.Scatter(
                x=[df_chart.index[0], df_chart.index[-1]], y=[y, y],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))

    layout_kwargs = dict(
        title=dict(
            text=f"{sel_name} — {_DESC_MAP[sel_tkr]}",
            font=dict(size=14, color="#1A202C"),
        ),
        height=380,
        margin=dict(l=10, r=10 if cmp_data is None else 60, t=45, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        legend=dict(orientation="h", y=1.06, x=0,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#E2E8F0", borderwidth=1),
        hovermode="x unified",
        xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
        yaxis=dict(gridcolor="#E8EDF5", zeroline=False,
                   title=dict(text=sel_name, font=dict(color="#0EA5E9")),
                   tickfont=dict(color="#0EA5E9")),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
    )
    if cmp_data is not None:
        layout_kwargs["yaxis2"] = dict(
            overlaying="y", side="right",
            gridcolor="#E8EDF5", zeroline=False,
            title=dict(text=compare, font=dict(color="#F59E0B")),
            tickfont=dict(color="#F59E0B"),
        )
    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True)

    # ── VIX vs VIX3M term structure chart ─────────────────────────────────────
    if vix_s and vix3m_s:
        st.markdown("#### VIX term structure history")
        with st.spinner("Loading term structure…"):
            df_vix   = fetch_chart_data("^VIX",   period="1y", interval="1d")
            df_vix3m = fetch_chart_data("^VIX3M", period="1y", interval="1d")

        if not df_vix.empty and not df_vix3m.empty:
            combined = pd.concat([
                df_vix["Close"].squeeze().rename("VIX"),
                df_vix3m["Close"].squeeze().rename("VIX3M"),
            ], axis=1).dropna()
            combined["Ratio"] = combined["VIX3M"] / combined["VIX"]

            fig2 = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.05, row_heights=[0.6, 0.4],
            )
            fig2.add_trace(go.Scatter(
                x=combined.index, y=combined["VIX"],
                name="VIX (spot)", line=dict(color="#0EA5E9", width=2),
            ), row=1, col=1)
            fig2.add_trace(go.Scatter(
                x=combined.index, y=combined["VIX3M"],
                name="VIX3M", line=dict(color="#F59E0B", width=2),
            ), row=1, col=1)

            ratio_colors = ["#059669" if v >= 1 else "#DC2626"
                            for v in combined["Ratio"]]
            fig2.add_trace(go.Bar(
                x=combined.index, y=combined["Ratio"],
                name="Ratio (3M/Spot)",
                marker_color=ratio_colors, opacity=0.7,
            ), row=2, col=1)
            # Reference line at 1.0 for ratio panel
            fig2.add_trace(go.Scatter(
                x=[combined.index[0], combined.index[-1]], y=[1.0, 1.0],
                mode="lines",
                line=dict(color="#94A3B8", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ), row=2, col=1)

            fig2.update_layout(
                title=dict(text="VIX Spot vs 3-Month · Ratio (green = contango, red = backwardation)",
                           font=dict(size=13, color="#1A202C")),
                height=380,
                margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                legend=dict(orientation="h", y=1.06, x=0,
                            font=dict(size=11),
                            bgcolor="rgba(255,255,255,0.85)",
                            bordercolor="#E2E8F0", borderwidth=1),
                hovermode="x unified",
                font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
            )
            fig2.update_xaxes(gridcolor="#E8EDF5", zeroline=False)
            fig2.update_yaxes(gridcolor="#E8EDF5", zeroline=False)
            st.plotly_chart(fig2, use_container_width=True)

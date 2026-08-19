"""Vol Adj Move tab — realized move vs vol-scaled 1σ for each horizon."""

import math
import concurrent.futures as _cf
from datetime import date as _date
import numpy as np
import pandas as pd
import streamlit as st
import ta as ta_lib
import yfinance as yf

from watchlist import (
    _raw_daily,
    CLASS_BG,
    CLASS_FG,
)

# ── Vol Adj Move instrument list — managed independently from the Macro watchlist ──
_VM_INSTRUMENTS = [
    {"name": "S&P 500",        "ticker": "^GSPC",    "class": "Equity"},
    {"name": "NASDAQ 100",      "ticker": "^NDX",     "class": "Equity"},
    {"name": "Dow Jones",      "ticker": "^DJI",     "class": "Equity"},
    {"name": "Russell 2000",   "ticker": "^RUT",     "class": "Equity"},
    {"name": "FTSE 100",       "ticker": "^FTSE",    "class": "Equity"},
    {"name": "DAX",            "ticker": "^GDAXI",   "class": "Equity"},
    {"name": "Euro Stoxx 50",  "ticker": "^STOXX50E","class": "Equity"},
    {"name": "Nikkei 225",     "ticker": "^N225",    "class": "Equity"},
    {"name": "NIFTY 50",       "ticker": "^NSEI",    "class": "Equity"},
    {"name": "KOSPI",          "ticker": "^KS11",    "class": "Equity"},
    {"name": "Taiwan",         "ticker": "^TWII",    "class": "Equity"},
    {"name": "Hang Seng",      "ticker": "^HSI",     "class": "Equity"},
    {"name": "CSI 300",        "ticker": "000300.SS","class": "Equity"},
    {"name": "EUR/USD",        "ticker": "EURUSD=X", "class": "FX"},
    {"name": "GBP/USD",        "ticker": "GBPUSD=X", "class": "FX"},
    {"name": "EUR/GBP",        "ticker": "EURGBP=X", "class": "FX"},
    {"name": "USD/JPY",        "ticker": "JPY=X",    "class": "FX"},
    {"name": "AUD/USD",        "ticker": "AUDUSD=X", "class": "FX"},
    {"name": "USD/CHF",        "ticker": "CHF=X",    "class": "FX"},
    {"name": "USD/CAD",        "ticker": "CAD=X",    "class": "FX"},
    {"name": "USD/CNH",        "ticker": "USDCNH=X", "class": "FX"},
    {"name": "USD/INR",        "ticker": "INR=X",    "class": "FX"},
    {"name": "USD/KRW",        "ticker": "KRW=X",    "class": "FX"},
    {"name": "USD/BRL",        "ticker": "BRL=X",    "class": "FX"},
    {"name": "Gold",           "ticker": "GC=F",     "class": "Commodity"},
    {"name": "WTI Oil",        "ticker": "CL=F",     "class": "Commodity"},
    {"name": "Brent Crude",    "ticker": "BZ=F",     "class": "Commodity"},
    {"name": "Silver",         "ticker": "SI=F",     "class": "Commodity"},
    {"name": "Copper",         "ticker": "HG=F",     "class": "Commodity"},
    {"name": "Bitcoin",        "ticker": "BTC-USD",  "class": "Crypto"},
    {"name": "Ethereum",       "ticker": "ETH-USD",  "class": "Crypto"},
    {"name": "US 10Y Ultra",   "ticker": "TN=F",     "class": "Rates"},
    {"name": "US 30Y Bond",    "ticker": "ZB=F",     "class": "Rates"},
    {"name": "US Ultra Bond",  "ticker": "UB=F",     "class": "Rates"},
]

_CLASS_BG = {**CLASS_BG}
_CLASS_FG = {**CLASS_FG}

# (label, annualisation divisor, trading-day lookback)
# "Today" is computed separately using live price — not in this list
_HIST_HORIZONS = [
    ("3d",  84,  3),
    ("5d",  50,  5),
    ("1m",  12, 21),
    ("3m",   4, 63),
]

_VOL_THRESHOLDS = {
    "Equity":    (0.12, 0.25),
    "FX":        (0.05, 0.12),
    "Rates":     (0.05, 0.12),
    "Commodity": (0.15, 0.30),
    "Crypto":    (0.40, 0.80),
    "Custom":    (0.15, 0.30),
    "Other":     (0.15, 0.30),
}

# Equity index → liquid futures ticker for live price outside cash-market hours.
# Futures trade ~23h/day so prices are available pre/post-market and overnight.
# CME contracts ONLY — Yahoo does not serve ICE/Eurex/OSE/HKEX index futures.
# Cleaned 2026-08-17: Z=F (FTSE), FDAX=F, FESX=F, NK=F, HSI=F were dead
# tickers (404/no data since written) that silently fell back to cash — the
# non-US indices always used their cash quote and still do, now honestly.
_FUTURES_LIVE = {
    "^GSPC":      "ES=F",     # S&P 500 → E-mini S&P 500 (same index)
    "^NDX":       "NQ=F",     # Nasdaq 100 → E-mini Nasdaq 100 (same index)
    "^DJI":       "YM=F",     # Dow Jones → E-mini Dow
    "^RUT":       "RTY=F",    # Russell 2000 → E-mini Russell
}

_GREEN = "#059669"
_AMBER = "#D97706"
_RED   = "#DC2626"
_MUTE  = "#94A3B8"

_SORT_KEYS: dict[str, callable] = {
    "Asset":      lambda r: r["name"].lower(),
    "Class":      lambda r: r["class"],
    "Last Price": lambda r: r["last"],
    "Ann Vol":    lambda r: r["ann_vol"],
    "RSI 14":     lambda r: r.get("rsi14") or 50,
    "1d/1σ":      lambda r: abs(r["ratios"].get("Today") or 0),
    "3d/1σ":      lambda r: abs(r["ratios"].get("3d")    or 0),
    "5d/1σ":      lambda r: abs(r["ratios"].get("5d")    or 0),
    "1m/1σ":      lambda r: abs(r["ratios"].get("1m")    or 0),
    "3m/1σ":      lambda r: abs(r["ratios"].get("3m")    or 0),
}

_HEADER_TO_SORT = {
    "Asset":    "Asset",
    "Class":    "Class",
    "Last":     "Last Price",
    "Ann Vol":  "Ann Vol",
    "RSI 14":   "RSI 14",
    "1d/1σ":    "1d/1σ",
    "3d/1σ":    "3d/1σ",
    "5d/1σ":    "5d/1σ",
    "1m/1σ":    "1m/1σ",
    "3m/1σ":    "3m/1σ",
}


def _vol_color(ann_vol: float, asset_class: str) -> str:
    lo, hi = _VOL_THRESHOLDS.get(asset_class, (0.15, 0.30))
    if ann_vol < lo:
        return _GREEN
    if ann_vol < hi:
        return _AMBER
    return _RED


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_hist_row(ticker: str, name: str, asset_class: str, lookback: int = 21) -> dict | None:
    """Historical vol + week/month/3M moves. Cached 1hr per (ticker, lookback)."""
    try:
        df = _raw_daily(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].squeeze().dropna().sort_index()
        if len(close) < lookback + 2:
            return None

        # Drop today's partial bar if present (e.g. 24/7 crypto always has one)
        last_bar_date = close.index[-1].date() if hasattr(close.index[-1], "date") else close.index[-1]
        if last_bar_date >= _date.today():
            close = close.iloc[:-1]
        if len(close) < 3:
            return None

        yesterday_close = float(close.iloc[-1])
        prev_close      = float(close.iloc[-2]) if len(close) >= 2 else None

        rets = close.pct_change().dropna().iloc[-lookback:]
        if len(rets) < max(5, lookback // 2):
            return None

        ann_vol = float(rets.std() * math.sqrt(252))

        ratios:  dict[str, float | None] = {}
        actuals: dict[str, float | None] = {}
        bases:   dict[str, float | None] = {}

        for label, ann_div, lookback in _HIST_HORIZONS:
            sigma = ann_vol / math.sqrt(ann_div)
            if len(close) > lookback and sigma > 0:
                base   = float(close.iloc[-(lookback + 1)])
                move   = float(close.iloc[-1] / base - 1)
                ratios[label]  = move / sigma
                actuals[label] = move
                bases[label]   = base
            else:
                ratios[label]  = None
                actuals[label] = None
                bases[label]   = None

        last_date = close.index[-1]
        if hasattr(last_date, "date"):
            last_date = last_date.date()

        try:
            rsi_series = ta_lib.momentum.RSIIndicator(close.astype(float), window=14).rsi().dropna()
            rsi14 = float(rsi_series.iloc[-1]) if not rsi_series.empty else None
        except Exception:
            rsi14 = None

        return {
            "name":             name,
            "ticker":           ticker,
            "class":            asset_class,
            "yesterday_close":  yesterday_close,
            "prev_close":       prev_close,
            "ann_vol":          ann_vol,
            "rsi14":            rsi14,
            "ratios":           ratios,
            "actuals":          actuals,
            "bases":            bases,
            "last_date":        last_date,
        }
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_live_prices_batch(src_tickers: tuple) -> dict[str, float]:
    """One yf.download call for all tickers — far faster than individual fast_info calls.
    Returns {ticker: last_close}. Uses 5m bars over 1d so the last bar is near-live."""
    try:
        def _dl():
            return yf.download(list(src_tickers), period="1d", interval="5m",
                               auto_adjust=True, progress=False, threads=True)
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            df = ex.submit(_dl).result(timeout=20)
        if df is None or df.empty:
            return {}
        close = df["Close"] if "Close" in df.columns else df.xs("Close", axis=1, level=0)
        if isinstance(close, pd.Series):
            close = close.to_frame(name=src_tickers[0])
        out = {}
        for col in close.columns:
            s = close[col].dropna()
            if not s.empty:
                out[str(col)] = float(s.iloc[-1])
        return out
    except Exception:
        return {}


def _fetch_live_price(ticker: str, prices_cache: dict) -> tuple[float | None, str]:
    """Look up live price from the pre-fetched batch cache.
    Tries futures ticker first, then spot ticker."""
    futures_ticker = _FUTURES_LIVE.get(ticker)
    for src in ([futures_ticker, ticker] if futures_ticker else [ticker]):
        p = prices_cache.get(src)
        if p:
            return float(p), src
    return None, ticker


def _fmt_price(val: float) -> str:
    if val >= 1000:
        return f"{val:,.2f}"
    if val >= 10:
        return f"{val:.4f}"
    if val >= 0.1:
        return f"{val:.5f}"
    return f"{val:.6f}"


def _rsi_cell(rsi: float | None) -> str:
    if rsi is None:
        return f'<span style="color:{_MUTE}">—</span>'
    if rsi >= 70:
        color = _RED
    elif rsi >= 60:
        color = _AMBER
    elif rsi <= 30:
        color = _GREEN
    elif rsi <= 40:
        color = "#0EA5E9"
    else:
        color = "#1E293B"
    return f'<span style="color:{color};font-weight:700">{rsi:.1f}</span>'


def _ratio_cell(ratio: float | None, actual: float | None) -> str:
    if ratio is None or actual is None:
        return f'<span style="color:{_MUTE}">—</span>'
    abs_r = abs(ratio)
    if abs_r >= 2.0:
        color = _RED
    elif abs_r >= 1.0:
        color = _AMBER
    else:
        color = "#1E293B"
    sign = "+" if ratio >= 0 else ""
    return (
        f'<span style="color:{color};font-weight:700">{sign}{ratio:.2f}σ</span>'
        f'<br><span style="color:#64748B;font-size:10px">{sign}{actual*100:.2f}%</span>'
    )


def _build_table_html(rows: list[dict], sort_col: str, sort_asc: bool, vol_window: int = 21) -> str:
    h_base   = ("background:#F8FAFC;color:#475569;font-size:11px;font-weight:600;"
                "text-transform:uppercase;letter-spacing:.5px;padding:6px 10px;"
                "border-bottom:2px solid #E2E8F0;text-align:right;white-space:nowrap;")
    h_left   = h_base.replace("text-align:right", "text-align:left")
    h_active = "background:#EFF6FF;color:#1D4ED8;"
    td_style = ("padding:6px 10px;border-bottom:1px solid #F1F5F9;font-size:12px;"
                "font-family:monospace;text-align:right;vertical-align:top;")
    td_left  = td_style.replace("text-align:right", "text-align:left")

    headers = ["Asset", "Class", "Last", f"{vol_window}d Ann Vol", "RSI 14",
               "1d/1σ", "3d/1σ", "5d/1σ", "1m/1σ", "3m/1σ"]

    th_parts = []
    for i, h in enumerate(headers):
        sort_name = _HEADER_TO_SORT.get(h) or (_HEADER_TO_SORT.get("Ann Vol") if "Ann Vol" in h else "")
        is_active = sort_name == sort_col
        arrow     = (" ▲" if sort_asc else " ▼") if is_active else ""
        base_s    = h_left if i == 0 else h_base
        extra_s   = h_active if is_active else ""
        th_parts.append(f'<th style="{base_s}{extra_s}">{h}{arrow}</th>')

    today = _date.today()
    body_rows = []
    for r in rows:
        ac     = r["class"]
        bg     = _CLASS_BG.get(ac, "#F8FAFC")
        fg_cls = _CLASS_FG.get(ac, "#475569")
        vc     = _vol_color(r["ann_vol"], ac)

        badge    = (
            f'<span style="background:{bg};color:{fg_cls};font-size:10px;'
            f'font-weight:600;padding:2px 6px;border-radius:4px">{ac}</span>'
        )
        vol_html = f'<span style="color:{vc};font-weight:700">{r["ann_vol"]*100:.1f}%</span>'

        last_date  = r.get("last_date", today)
        bdays_old  = int(np.busday_count(last_date, today)) if last_date else 0
        date_color = _AMBER if bdays_old > 1 else _MUTE
        date_str   = last_date.strftime("%d %b").lstrip("0") if last_date else "—"
        if r.get("live"):
            lbl = r.get("live_label", "")
            live_tag = (
                f'<span style="color:#059669;font-size:9px;font-weight:600">'
                f' LIVE {"via " + lbl if lbl else ""}</span>'
            )
        else:
            live_tag = f'<span style="color:{date_color};font-size:10px"> as of {date_str}</span>'

        cells = [
            f'<td style="{td_left}"><span style="font-weight:600;color:#1E293B">'
            f'{r["name"]}</span><br>'
            f'<span style="color:{_MUTE};font-size:10px">{r["ticker"]}</span></td>',
            f'<td style="{td_style}">{badge}</td>',
            f'<td style="{td_style}"><span style="font-weight:600">{_fmt_price(r["last"])}</span>'
            f'{live_tag}</td>',
            f'<td style="{td_style}">{vol_html}</td>',
            f'<td style="{td_style}">{_rsi_cell(r.get("rsi14"))}</td>',
        ]
        for label in ["Today", "3d", "5d", "1m", "3m"]:
            cells.append(
                f'<td style="{td_style}">'
                f'{_ratio_cell(r["ratios"].get(label), r["actuals"].get(label))}'
                f'</td>'
            )

        body_rows.append(f'<tr>{"".join(cells)}</tr>')

    return (
        '<table style="width:100%;border-collapse:collapse;border:1px solid #E2E8F0;'
        'border-radius:8px;overflow:hidden">'
        f'<thead><tr>{"".join(th_parts)}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table>'
    )


@st.fragment
def render_vol_adj_move() -> None:
    st.markdown(
        '<p style="color:#64748B;font-size:12px;margin-bottom:4px">'
        "Realized move expressed in units of expected 1σ (21d ann. vol). "
        "Today uses live price vs yesterday close. σ ratio on top, actual % below. "
        "<span style='color:#D97706'>●</span> 1–2σ &nbsp;"
        "<span style='color:#DC2626'>●</span> &gt;2σ."
        "</p>",
        unsafe_allow_html=True,
    )

    instruments = _VM_INSTRUMENTS

    if not instruments:
        st.info("No instruments configured.")
        return

    _VOL_WINDOWS = [5, 10, 21, 42, 63]
    all_classes  = sorted({i["class"] for i in _VM_INSTRUMENTS})
    sel_classes  = st.multiselect(
        "Asset classes", all_classes, default=all_classes, key="_vm_classes"
    )

    sort_options = list(_SORT_KEYS.keys())
    c_win, c_sort, c_dir, c_run, c_ref = st.columns([1, 3, 1, 1, 1])

    with c_win:
        vol_window = st.selectbox(
            "Vol window", _VOL_WINDOWS,
            index=_VOL_WINDOWS.index(st.session_state.get("_vm_vol_window", 21)),
            key="_vm_vol_window",
            format_func=lambda x: f"{x}d",
            label_visibility="collapsed",
        )

    with c_sort:
        sort_col = st.selectbox(
            "Sort by", sort_options,
            index=sort_options.index(st.session_state.get("_vm_sort_col", "Ann Vol") if st.session_state.get("_vm_sort_col", "Ann Vol") in sort_options else "Ann Vol"),
            key="_vm_sort_col",
            label_visibility="collapsed",
        )

    with c_dir:
        asc_label = "↑ Asc" if st.session_state.get("_vm_sort_asc", False) else "↓ Desc"
        if st.button(asc_label, key="_vm_dir_btn", use_container_width=True):
            st.session_state["_vm_sort_asc"] = not st.session_state.get("_vm_sort_asc", False)
            st.rerun()

    sort_asc: bool = st.session_state.get("_vm_sort_asc", False)

    with c_run:
        run = st.button("▶ Run", key="_vm_run", use_container_width=True,
                        type="primary")

    with c_ref:
        if st.button("⟳ Reset", key="_vm_refresh", use_container_width=True):
            # Targeted refresh: only the caches this tab reads.
            from watchlist import _all_daily
            _all_daily.clear()                  # watchlist: shared daily OHLCV (via _raw_daily)
            _compute_hist_row.clear()           # local: per-ticker vol/move rows
            _fetch_live_prices_batch.clear()    # local: intraday live prices
            st.session_state.pop("_vm_rows", None)
            st.session_state.pop("_vm_failed", None)
            st.rerun()

    if run:
        all_rows: list[dict] = []
        failed:   list[str]  = []

        # Build the full set of tickers to batch-fetch live prices for
        _live_src_set: set[str] = set()
        for inst in instruments:
            t = inst["ticker"]
            ft = _FUTURES_LIVE.get(t)
            _live_src_set.update([ft, t] if ft else [t])
        live_cache = _fetch_live_prices_batch(tuple(sorted(_live_src_set)))

        progress = st.progress(0, text="Loading…")
        n = len(instruments)
        for i, inst in enumerate(instruments):
            progress.progress((i + 1) / n, text=f"Loading {inst['name']}…")
            ticker = inst["ticker"]
            name   = inst["name"]
            ac     = inst.get("class", "Other")

            hist = _compute_hist_row(ticker, name, ac, vol_window)
            if not hist:
                failed.append(name)
                continue

            live_price, live_src = _fetch_live_price(ticker, live_cache)
            ann_vol    = hist["ann_vol"]
            sigma_day  = ann_vol / math.sqrt(256)

            if live_price and hist["yesterday_close"] and sigma_day > 0:
                current      = live_price
                today_actual = live_price / hist["yesterday_close"] - 1
                today_ratio  = today_actual / sigma_day
                last_price   = live_price
                last_date    = _date.today()
                is_live      = True
                live_label   = live_src if live_src != ticker else ""
            else:
                current = hist["yesterday_close"]
                yc = hist["yesterday_close"]
                pc = hist["prev_close"]
                if pc and sigma_day > 0:
                    today_actual = yc / pc - 1
                    today_ratio  = today_actual / sigma_day
                else:
                    today_actual = None
                    today_ratio  = None
                last_price  = yc
                last_date   = hist["last_date"]
                is_live     = False
                live_label  = ""

            # Recompute all horizon moves using current (live) price as the numerator
            ratios  = {"Today": today_ratio}
            actuals = {"Today": today_actual}
            for label, ann_div, _lb in _HIST_HORIZONS:
                base = hist["bases"].get(label)
                if base and current and base > 0:
                    sigma = ann_vol / math.sqrt(ann_div)
                    move  = current / base - 1
                    ratios[label]  = move / sigma if sigma > 0 else None
                    actuals[label] = move
                else:
                    ratios[label]  = None
                    actuals[label] = None

            all_rows.append({
                **hist,
                "last":       last_price,
                "last_date":  last_date,
                "live":       is_live,
                "live_label": live_label,
                "ratios":     ratios,
                "actuals":    actuals,
            })

        progress.empty()
        st.session_state["_vm_rows"]   = all_rows
        st.session_state["_vm_failed"] = failed

    all_rows = st.session_state.get("_vm_rows")
    failed   = st.session_state.get("_vm_failed", [])

    # Stale-cache check: if stored rows are missing any current horizon key, evict
    _expected_keys = {label for label, *_ in _HIST_HORIZONS}
    if all_rows and not _expected_keys.issubset(set(all_rows[0].get("ratios", {}).keys())):
        st.session_state.pop("_vm_rows", None)
        st.session_state.pop("_vm_failed", None)
        # Only _compute_hist_row holds the stale horizon-keyed schema; evict just it.
        _compute_hist_row.clear()
        all_rows = None
        st.info("Data columns changed — click ▶ Run to reload.")

    if not all_rows:
        st.info("Click ▶ Run to load vol data.")
        return

    filtered = [r for r in all_rows if r["class"] in sel_classes] if sel_classes else all_rows

    key_fn = _SORT_KEYS.get(sort_col, lambda r: r["ann_vol"])
    filtered.sort(key=key_fn, reverse=not sort_asc)

    stored_window = st.session_state.get("_vm_vol_window", 21)
    st.markdown(_build_table_html(filtered, sort_col, sort_asc, stored_window), unsafe_allow_html=True)

    if failed:
        st.caption(f"Could not compute vol for: {', '.join(failed)}")

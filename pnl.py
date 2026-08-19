"""
P&L Analytics Tab — IBKR Flex Query performance analysis.
Upload a Flex CSV → get Sharpe, VaR, drawdown, win rate, trade breakdown.
"""

import json
import math
import os
import re
from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flex_parser import parse_flex_csv
import pnl_db

_EXCLUDED_PATH = os.path.join(os.path.dirname(__file__), "pnl_excluded.json")
_ANN = 252  # trading days per year


# ── Exclusion list helpers ─────────────────────────────────────────────────────

def _load_excluded() -> list[str]:
    try:
        with open(_EXCLUDED_PATH) as f:
            return json.load(f).get("excluded_symbols", [])
    except Exception:
        return []


def _save_excluded(syms: list[str]):
    data = {"_note": "Symbols excluded from trading P&L analysis (long-term ETF holds).",
            "excluded_symbols": sorted(set(syms))}
    with open(_EXCLUDED_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _sharpe(returns: pd.Series) -> float:
    s = returns.std()
    # Guard against a degenerate (near-zero) dispersion that would explode the ratio.
    if not (s > 1e-9):
        return float("nan")
    return float(returns.mean() / s * math.sqrt(_ANN))


def _sortino(returns: pd.Series) -> float:
    neg = returns[returns < 0]
    # Need at least two negative days, and a non-degenerate downside deviation.
    # A couple of near-identical tiny negative days gives ds ≈ 1e-18, which would
    # otherwise blow the ratio up to ~1e16 over short windows.
    if len(neg) < 2:
        return float("nan")
    ds = neg.std()
    if not (ds > 1e-9):
        return float("nan")
    return float(returns.mean() / ds * math.sqrt(_ANN))


def _max_drawdown(nav: pd.Series) -> float:
    roll_max = nav.expanding().max()
    dd = (nav - roll_max) / roll_max
    return float(dd.min())


def _max_drawdown_dollar(nav: pd.Series) -> float:
    return float((nav - nav.expanding().max()).min())


def _drawdown_series_dollar(nav: pd.Series) -> pd.Series:
    return nav - nav.expanding().max()


def _var_cvar(returns: pd.Series, confidence: float = 0.95):
    var_pct = returns.quantile(1 - confidence)
    cvar_pct = returns[returns <= var_pct].mean()
    return float(var_pct), float(cvar_pct)


def _var_cvar_dollar(pnl: pd.Series, confidence: float = 0.95):
    var_d = pnl.quantile(1 - confidence)
    cvar_d = pnl[pnl <= var_d].mean()
    return float(var_d), float(cvar_d)


def _calmar_dollar(ann_pnl: float, max_dd_dollar: float) -> float:
    return ann_pnl / abs(max_dd_dollar) if max_dd_dollar != 0 else float("nan")


def _rolling_sharpe(returns: pd.Series, window: int = 30) -> pd.Series:
    roll = returns.rolling(window)
    return roll.mean() / roll.std() * math.sqrt(_ANN)


def _fx_native_balances(sections: dict) -> dict:
    """{currency: native cash balance} for non-USD currencies (from cash_summary)."""
    cs = sections.get("cash_summary", pd.DataFrame())
    out = {}
    if cs.empty or "CurrencyPrimary" not in cs.columns:
        return out
    df = cs[cs["LevelOfDetail"].astype(str).str.lower() == "currency"] if "LevelOfDetail" in cs.columns else cs
    for _, r in df.iterrows():
        ccy = str(r.get("CurrencyPrimary") or "").strip()
        if ccy in ("", "USD", "BASE_SUMMARY"):
            continue
        b = pd.to_numeric(r.get("EndingCash"), errors="coerce")
        if pd.notna(b) and abs(b) > 1e-6:
            out[ccy] = float(b)
    return out


def _batch_daily_closes(tickers, **dl_kwargs) -> dict:
    """One batched yf.download for many tickers → {ticker: Close Series (NaNs
    dropped)}. Tickers the batch cannot resolve (missing / all-NaN) are simply
    absent from the dict, so callers fall back to their original per-ticker
    yf.Ticker().history() path for those — preserving today's flaky-ticker
    behaviour exactly. auto_adjust is forced True (these are FX pairs / futures,
    so adjusted == raw) so values stay byte-identical to the sequential path on
    every settled bar. Mirrors the batch-and-split logic in watchlist._all_daily."""
    import yfinance as yf
    uniq = list(dict.fromkeys(t for t in tickers if t))
    out: dict = {}
    if not uniq:
        return out
    try:
        raw = yf.download(uniq, auto_adjust=True, progress=False,
                          group_by="ticker", threads=True, **dl_kwargs)
    except Exception:
        raw = None
    if raw is None or len(raw) == 0:
        return out
    cols = raw.columns
    multi = isinstance(cols, pd.MultiIndex)
    lvl0 = set(cols.get_level_values(0)) if multi else set()
    for tk in uniq:
        try:
            if multi:
                if tk not in lvl0:
                    continue
                sub = raw[tk]
            else:
                sub = raw
            if "Close" not in sub.columns:
                continue
            s = sub["Close"].dropna()
            if len(s):
                out[tk] = s
        except Exception:
            continue
    return out


def _naive_norm(idx):
    """tz_localize(None).normalize(), safe whether idx is tz-aware (per-ticker
    history fallback) or already tz-naive (batched yf.download). Reproduces the
    original `idx.tz_localize(None).normalize()` exactly for the tz-aware case."""
    idx = pd.DatetimeIndex(idx)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _fx_align_ibkr(idx):
    """Shift yfinance FX daily-bar dates back ONE business day to align with IBKR
    statement dates. Empirical (2026-07-31): the Thu Jul-30 daytime yen rally
    (USDJPY 163.30->159.73) sits in yfinance's bar LABELLED Jul-31, while IBKR books
    that trading under ReportDate Jul-30 — unshifted, a Jul-29->30 statement window
    missed the entire move (showed $2.2k instead of ~$19k on a Y124M balance)."""
    return pd.DatetimeIndex(idx) - pd.tseries.offsets.BDay(1)


@st.cache_data(ttl=1800, show_spinner=False)
def _fx_translation_daily(balances_items: tuple, start: str, end: str) -> pd.Series:
    """Daily FX mark-to-market P&L (USD) = Σ_ccy balance × Δ(USD-per-unit), like a
    future's daily settlement. Rates from yfinance. balances_items = ((ccy, bal), ...)."""
    import yfinance as yf
    direct = {"EUR", "GBP", "AUD", "NZD"}
    tickers = [f"{str(ccy).upper()}USD=X" if str(ccy).upper() in direct
               else f"{str(ccy).upper()}=X" for ccy, _ in balances_items]
    # pad end: statement-day D's move lives in the yfinance bar labelled D+1 (see
    # _fx_align_ibkr) — fetch past the window so the last day's move is captured.
    _e_pad = (pd.Timestamp(end) + pd.Timedelta(days=4)).date().isoformat()
    batch = _batch_daily_closes(tickers, start=start, end=_e_pad)
    total = None
    for ccy, bal in balances_items:
        c = str(ccy).upper()
        tk = f"{c}USD=X" if c in direct else f"{c}=X"
        inv = c not in direct
        try:
            h = batch.get(tk)
            if h is None:
                h = yf.Ticker(tk).history(start=start, end=_e_pad)["Close"].dropna()
            else:
                h = h.copy()
            if h.empty:
                continue
            if inv:
                h = 1.0 / h
            h.index = _fx_align_ibkr(_naive_norm(h.index))
            h = h[h.index <= pd.Timestamp(end)]
            mtm = float(bal) * h.diff()          # daily P&L = balance × Δrate
            total = mtm if total is None else total.add(mtm, fill_value=0.0)
        except Exception:
            continue
    return total.dropna() if total is not None else pd.Series(dtype=float)


def _fx_daily_balances(sections: dict) -> dict:
    """{ccy: daily ending native balance (Series)} from statement_of_funds (per-currency
    ledger, LevelOfDetail='Currency'). Empty if the section isn't per-currency."""
    sof = sections.get("statement_of_funds", pd.DataFrame())
    if sof.empty or not {"CurrencyPrimary", "Date", "Balance", "LevelOfDetail"}.issubset(sof.columns):
        return {}
    df = sof[sof["LevelOfDetail"].astype(str).str.lower() == "currency"].copy()
    if df.empty:
        return {}
    df["_d"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", errors="coerce")
    df["_bal"] = pd.to_numeric(df["Balance"], errors="coerce")
    df = df.dropna(subset=["_d", "_bal"])
    out = {}
    for ccy, g in df.groupby("CurrencyPrimary"):
        c = str(ccy).strip()
        if c in ("", "USD", "BASE_SUMMARY"):
            continue
        out[c] = g.sort_values("_d").groupby("_d")["_bal"].last()   # ending balance per day
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def _fx_rates_daily(currencies: tuple, start: str, end: str) -> dict:
    """{ccy: USD-per-unit daily Series} via yfinance (CNH falls back to onshore CNY)."""
    import yfinance as yf
    direct = {"EUR", "GBP", "AUD", "NZD"}

    def _tks(c):
        t = [f"{c}USD=X" if c in direct else f"{c}=X"]
        if c == "CNH":
            t.append("CNY=X")                       # offshore CNH is flaky on yfinance
        return t

    all_tks = []
    for ccy in currencies:
        all_tks += _tks(str(ccy).upper())
    # pad end: statement-day D's move lives in the yfinance bar labelled D+1
    _end_pad = (pd.Timestamp(end) + pd.Timedelta(days=4)).date().isoformat()
    batch = _batch_daily_closes(all_tks, start=start, end=_end_pad)

    out = {}
    for ccy in currencies:
        c = str(ccy).upper()
        tks = _tks(c)
        inv = c not in direct
        for tk in tks:
            try:
                h = batch.get(tk)
                if h is None:
                    h = yf.Ticker(tk).history(start=start, end=_end_pad)["Close"].dropna()
                else:
                    h = h.copy()
                if h.empty:
                    continue
                if inv:
                    h = 1.0 / h
                h.index = _fx_align_ibkr(_naive_norm(h.index))
                out[ccy] = h
                break
            except Exception:
                continue
    return out


def _fx_mtm_from_balances(balhist: dict, start: str, end: str) -> pd.Series:
    """Exact daily FX MTM (USD) = Σ_ccy (prior-day ending balance) × Δrate, using the
    real per-day balances from the Statement of Funds ledger."""
    rates = _fx_rates_daily(tuple(sorted(balhist)), start, end)
    total = None
    for ccy, bal in balhist.items():
        r = rates.get(ccy)
        if r is None or r.empty:
            continue
        bal = bal.sort_index()
        held = (bal.reindex(r.index.union(bal.index)).sort_index()
                   .ffill().reindex(r.index).shift(1))     # balance held into each day
        mtm = held * r.diff()                                # × that day's rate move
        total = mtm if total is None else total.add(mtm, fill_value=0.0)
    return total.dropna() if total is not None else pd.Series(dtype=float)


def _fx_mtm_by_ccy(balhist: dict, start: str, end: str, ts_from, ts_to) -> dict:
    """{ccy: windowed FX MTM (USD)} — the per-currency version of _fx_mtm_from_balances."""
    rates = _fx_rates_daily(tuple(sorted(balhist)), start, end)
    out = {}
    for ccy, bal in balhist.items():
        r = rates.get(ccy)
        if r is None or r.empty:
            continue
        held = (bal.sort_index().reindex(r.index.union(bal.index)).sort_index()
                   .ffill().reindex(r.index).shift(1))
        mtm = held * r.diff()
        w = mtm[(mtm.index >= ts_from) & (mtm.index <= ts_to)]
        out[ccy] = float(w.sum())
    return out


_THEME_BY_ROOT = {
    "MBT": "Bitcoin", "BTC": "Bitcoin", "MET": "Ethereum", "ETH": "Ethereum",
    "ZQ": "Fed Funds", "GE": "Eurodollar",
    "ZT": "US 2Y", "ZF": "US 5Y", "ZN": "US 10Y", "TN": "US Ultra 10Y", "ZB": "US 30Y",
    "UB": "US Ultra Bond", "OZN": "US 10Y (opt)", "OZB": "US 30Y (opt)",
    "GC": "Gold", "MGC": "Gold", "SI": "Silver", "HG": "Copper", "PL": "Platinum", "PA": "Palladium",
    "CL": "WTI Crude", "MCL": "WTI Crude", "BZ": "Brent", "BRN": "Brent", "NG": "NatGas",
    "RB": "Gasoline", "HO": "Heating Oil",
    "ES": "S&P 500", "MES": "S&P 500", "NQ": "Nasdaq", "MNQ": "Nasdaq", "RTY": "Russell",
    "M2K": "Russell", "YM": "Dow", "MYM": "Dow", "NKD": "Nikkei", "DAX": "DAX",
    "FDAX": "DAX", "FESX": "Euro Stoxx",
}


def _root_simple(s: str) -> str:
    s = re.split(r"\s+", str(s).upper().strip())[0]
    return re.sub(r"[FGHJKMNQUVXZ]\d{1,2}$", "", s)


def _theme_of(symbol: str, underlying: str = "") -> str:
    """Human-readable product theme for a symbol (Bitcoin, Euribor, Gold, S&P 500, …)."""
    s = str(symbol).upper().strip()
    u = str(underlying).upper()
    if re.match(r"^I[FGHJKMNQUVXZ]\d{0,2}$", s) or s.startswith("ER") or "EURIBOR" in u:
        return "Euribor"          # ICE Euribor code is "I" + month + year (IZ6, IH7, …)
    if s.startswith(("SO3", "SOA")) or "SONIA" in u:
        return "SONIA"
    if s.startswith("SR") or "SOFR" in u:
        return "SOFR"
    for r in (_root_simple(s), _root_simple(u)):
        if r in _THEME_BY_ROOT:
            return _THEME_BY_ROOT[r]
    m = re.match(r"[A-Z]+", s)                # fallback: leading letters (e.g. IH7 → IH)
    return m.group() if m else s


# Map a product theme → broad asset class (Rates / Commodities / Equities / Crypto / FX).
_AC_RATES = {"Fed Funds", "Eurodollar", "Euribor", "SONIA", "SOFR",
             "US 2Y", "US 5Y", "US 10Y", "US Ultra 10Y", "US 30Y", "US Ultra Bond",
             "US 10Y (opt)", "US 30Y (opt)"}
_AC_COMMOD = {"Gold", "Silver", "Copper", "Platinum", "Palladium",
              "WTI Crude", "Brent", "NatGas", "Gasoline", "Heating Oil"}
_AC_EQUITY = {"S&P 500", "Nasdaq", "Russell", "Dow", "Nikkei", "DAX", "Euro Stoxx"}
_AC_CRYPTO = {"Bitcoin", "Ethereum"}


def _asset_class_of(theme: str) -> str:
    """Broad asset class for a product theme (as returned by _theme_of, or an FX row)."""
    t = str(theme)
    if t.endswith("(FX)"):
        return "FX"
    if t in _AC_RATES:
        return "Rates"
    if t in _AC_COMMOD:
        return "Commodities"
    if t in _AC_EQUITY:
        return "Equities"
    if t in _AC_CRYPTO:
        return "Crypto"
    return "Other"


def _futures_pnl_by_symbol(sections: dict, ts_from, ts_to, sym_filter, excluded) -> dict:
    """{symbol: windowed settled P&L (PriorMtmPnl × FX)} from prior_period_pnl, honouring
    the symbol filter / ETF exclusion so it matches the tab's active book."""
    pp = sections.get("prior_period_pnl", pd.DataFrame())
    if pp.empty or not {"Symbol", "Date", "PriorMtmPnl"}.issubset(pp.columns):
        return {}
    df = pp.copy()
    df["_d"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[(df["_d"] >= ts_from) & (df["_d"] <= ts_to)]
    if sym_filter:
        df = df[df["Symbol"].isin(sym_filter)]
    elif excluded:
        df = df[~df["Symbol"].isin(excluded)]
    df["_p"] = (pd.to_numeric(df["PriorMtmPnl"], errors="coerce").fillna(0)
                * pd.to_numeric(df.get("FXRateToBase"), errors="coerce").fillna(1))
    return {k: float(v) for k, v in df.groupby("Symbol")["_p"].sum().items() if abs(v) > 0.5}


def _pnl_by_symbol_full(trades_sym: pd.DataFrame,
                        prior_df:   pd.DataFrame,
                        sym_filter: list[str],
                        excluded:   list[str]) -> dict:
    """
    Per-symbol P&L that mirrors ``_build_daily_pnl`` exactly (all three components),
    so the breakdown SUMS TO the window Total P&L KPI — not just the settlement piece.

    Components (identical to _build_daily_pnl, kept per-symbol):
      1. prior_period_pnl.PriorMtmPnl × FX  — overnight settlements.
      2. trades.MtmPnl + commission (FUT/FOP on prior-period symbols) — execution slippage.
      3. _net_per_trade for symbols NOT in prior_period — new mid-period positions/rolls.

    ``trades_sym`` is assumed already windowed + symbol-filtered/excluded (as the caller
    prepares it); ``prior_df`` is the raw windowed prior-period frame (filter/exclude here).
    """
    out: dict[str, float] = {}
    def _add(sym, v):
        if sym is None:
            return
        out[str(sym)] = out.get(str(sym), 0.0) + float(v)

    # Drop spot-FX conversion trades (CASH) — FX P&L is the balance MTM (toggle), not
    # these rows; keeps this in lock-step with _build_daily_pnl so the breakdown ties out.
    if trades_sym is not None and not trades_sym.empty and "AssetClass" in trades_sym.columns:
        trades_sym = trades_sym[trades_sym["AssetClass"] != "CASH"]

    # Fallback: no prior_period section → trades-only (matches _build_daily_pnl)
    if prior_df is None or prior_df.empty or "Symbol" not in prior_df.columns:
        if trades_sym is not None and not trades_sym.empty:
            t = trades_sym.copy()
            t["_net"] = _net_per_trade(t)
            for sym, v in t.groupby("Symbol")["_net"].sum().items():
                _add(sym, v)
        return {k: v for k, v in out.items() if abs(v) > 0.5}

    prior_syms = set(prior_df["Symbol"].unique())
    if sym_filter:
        prior_syms = prior_syms & set(sym_filter)

    # Component 1 — overnight settlements per symbol
    pf = prior_df.copy()
    if excluded:
        pf = pf[~pf["Symbol"].isin(excluded)]
    if sym_filter:
        pf = pf[pf["Symbol"].isin(sym_filter)]
    if not pf.empty:
        pf["_p"] = (pd.to_numeric(pf["PriorMtmPnl"], errors="coerce").fillna(0)
                    * pd.to_numeric(pf.get("FXRateToBase"), errors="coerce").fillna(1.0))
        for sym, v in pf.groupby("Symbol")["_p"].sum().items():
            _add(sym, v)

    # Component 2 — futures execution slippage on prior-period symbols
    if trades_sym is not None and not trades_sym.empty and "AssetClass" in trades_sym.columns:
        fut_prior = trades_sym[
            trades_sym["AssetClass"].isin(["FUT", "FOP"]) &
            trades_sym["Symbol"].isin(prior_syms)
        ].copy()
        if not fut_prior.empty:
            fut_prior["_net"] = (fut_prior["MtmPnl"].fillna(0)
                                 + fut_prior["IBCommission"].fillna(0))
            for sym, v in fut_prior.groupby("Symbol")["_net"].sum().items():
                _add(sym, v)

    # Component 3 — new mid-period positions (symbols not in prior_period)
    if trades_sym is not None and not trades_sym.empty:
        new_trades = trades_sym[~trades_sym["Symbol"].isin(prior_syms)]
        if not new_trades.empty:
            nt = new_trades.copy()
            nt["_net"] = _net_per_trade(nt)
            for sym, v in nt.groupby("Symbol")["_net"].sum().items():
                _add(sym, v)

    return {k: v for k, v in out.items() if abs(v) > 0.5}


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _colour(val: float, good_positive: bool = True) -> str:
    if not math.isfinite(val):
        return "#94A3B8"
    if good_positive:
        return "#059669" if val >= 0 else "#DC2626"
    else:
        return "#059669" if val <= 0 else "#DC2626"


def _fmt(val: float, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
    if not math.isfinite(val):
        return "—"
    return f"{prefix}{val:,.{decimals}f}{suffix}"


# ── KPI card ──────────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, colour: str = "#1A202C", sub: str = ""):
    sub_html = f'<div style="font-size:10px;color:#94A3B8;margin-top:2px">{sub}</div>' if sub else ""
    st.markdown(
        f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;'
        f'padding:10px 14px;text-align:center">'
        f'<div style="font-size:10px;font-weight:600;color:#64748B;text-transform:uppercase;'
        f'letter-spacing:0.5px">{label}</div>'
        f'<div style="font-size:20px;font-weight:700;color:{colour};margin-top:4px">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


# ── Asset-class-aware net P&L per trade row ───────────────────────────────────

def _net_per_trade(df: pd.DataFrame) -> pd.Series:
    """
    Futures/FOP: MtmPnl + commission  (daily settlement MTM — best proxy from trades).
    All others:  FifoPnlRealized + commission (FIFO cost basis, correct for stocks).
    Note: for futures the daily chart is approximate; use realized_pnl section for
    accurate period totals.
    """
    net = pd.Series(0.0, index=df.index)
    if "AssetClass" in df.columns:
        fut = df["AssetClass"].isin(["FUT", "FOP"])
        net[fut]  = df.loc[fut,  "MtmPnl"].fillna(0)          + df.loc[fut,  "IBCommission"].fillna(0)
        net[~fut] = df.loc[~fut, "FifoPnlRealized"].fillna(0) + df.loc[~fut, "IBCommission"].fillna(0)
    else:
        net = df["FifoPnlRealized"].fillna(0) + df["IBCommission"].fillna(0)
    return net


# ── Accurate P&L totals from the realized_pnl section ─────────────────────────

def _pnl_from_realized(realized_df: pd.DataFrame,
                        sym_filter: list[str],
                        excluded:   list[str]) -> pd.DataFrame:
    """
    Returns DataFrame [Symbol, AssetClass, TotalFifoPnl] from the realized_pnl
    section.  TotalFifoPnl = realized + unrealized on open positions, computed from
    original entry price — this matches IBKR's official period P&L report.
    """
    if realized_df.empty:
        return pd.DataFrame(columns=["Symbol", "AssetClass", "TotalFifoPnl"])
    df = realized_df.copy()
    for c in ["TotalFifoPnl", "TotalRealizedPnl", "TotalUnrealizedPnl"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df = df[df["Symbol"].str.strip().ne("")]   # drop aggregate/summary rows
    if "AssetClass" in df.columns:
        df = df[df["AssetClass"] != "CASH"]    # FX balances not trading P&L
    if excluded:
        df = df[~df["Symbol"].isin(excluded)]
    if sym_filter:
        df = df[df["Symbol"].isin(sym_filter)]
    cols = [c for c in ["Symbol", "AssetClass", "SubCategory", "TotalFifoPnl",
                        "TotalRealizedPnl", "TotalUnrealizedPnl"] if c in df.columns]
    return df[cols].copy()


# ── Trade stats (win/loss counts and timing — uses trade-level data) ──────────

def _trade_stats(trades: pd.DataFrame, excluded: list[str]) -> dict:
    df = trades.copy()
    if excluded:
        df = df[~df["Symbol"].isin(excluded)]

    # For futures: every trade has a daily MTM P&L — use all trades
    # For non-futures: only closing trades carry realised P&L
    if "AssetClass" in df.columns and "Open/CloseIndicator" in df.columns:
        fut_mask = df["AssetClass"].isin(["FUT", "FOP"])
        closing  = pd.concat([
            df[fut_mask],                                         # all futures trades
            df[~fut_mask & (df["Open/CloseIndicator"] == "C")],  # closing non-futures only
        ])
    elif "Open/CloseIndicator" in df.columns:
        closing = df[df["Open/CloseIndicator"] == "C"].copy()
    else:
        closing = df.copy()

    if closing.empty:
        return {}

    closing = closing.copy()
    closing["_net"] = _net_per_trade(closing)
    wins   = closing[closing["_net"] > 0]
    losses = closing[closing["_net"] < 0]
    total  = len(closing)

    gross_win  = float(wins["_net"].sum())
    gross_loss = float(losses["_net"].sum())

    return {
        "total_trades":   total,
        "wins":           len(wins),
        "losses":         len(losses),
        "win_rate":       len(wins) / total if total else 0,
        "avg_win":        float(wins["_net"].mean())   if len(wins)   else 0,
        "avg_loss":       float(losses["_net"].mean()) if len(losses) else 0,
        "profit_factor":  gross_win / abs(gross_loss)  if gross_loss != 0 else float("inf"),
        "commissions":    float(df["IBCommission"].sum()),
        "best_trade":     float(closing["_net"].max()),
        "worst_trade":    float(closing["_net"].min()),
        "closing_df":     closing,
    }


# ── Daily trading P&L (from trade-level data) ─────────────────────────────────

def _daily_trade_pnl(trades: pd.DataFrame, excluded: list[str]) -> pd.Series:
    df = trades.copy()
    if excluded:
        df = df[~df["Symbol"].isin(excluded)]
    df["_net"] = _net_per_trade(df)
    daily = df.groupby("TradeDate")["_net"].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


# ── Daily P&L from Prior Period Positions section (accurate overnight MTM) ─────

def _daily_prior_period_pnl(prior_df: pd.DataFrame,
                             sym_filter: list[str],
                             excluded:   list[str]) -> pd.Series:
    """
    Prior Period Positions section has one row per symbol per trading day.
    PriorMtmPnl is in local currency; multiply by FXRateToBase to get USD.
    Returns daily Series of base-currency P&L.
    """
    df = prior_df.copy()
    if excluded:
        df = df[~df["Symbol"].isin(excluded)]
    if sym_filter:
        df = df[df["Symbol"].isin(sym_filter)]
    if df.empty:
        return pd.Series(dtype=float)
    df["_pnl_base"] = df["PriorMtmPnl"].fillna(0) * df["FXRateToBase"].fillna(1.0)
    daily = df.groupby("Date")["_pnl_base"].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def _build_daily_pnl(trades_sym:   pd.DataFrame,
                     prior_df:     pd.DataFrame,
                     sym_filter:   list[str],
                     excluded:     list[str]) -> pd.Series:
    """
    Best-available daily P&L combining two complementary sources:

    1. prior_period_pnl.PriorMtmPnl × FXRateToBase
       = overnight variation-margin settlements for positions open at period start.

    2. trades.MtmPnl + commission  (FUTURES ONLY, from prior-period symbols)
       = execution-time slippage vs prior settlement — no overlap with (1).

    3. _daily_trade_pnl for symbols NOT in prior_period_pnl
       = new positions opened mid-period; uses MtmPnl for futures,
         FifoPnlRealized for equities.

    Fallback: trades-only when prior_period_pnl section is absent.

    Spot-FX conversion trades (AssetClass CASH, e.g. EUR.USD) are dropped here — their
    economic P&L is the balance mark-to-market, handled separately by the "Count FX P&L"
    toggle; counting the CASH trade rows too would double-count FX.
    """
    if not trades_sym.empty and "AssetClass" in trades_sym.columns:
        trades_sym = trades_sym[trades_sym["AssetClass"] != "CASH"]
    if prior_df.empty or "Symbol" not in prior_df.columns:
        return _daily_trade_pnl(trades_sym, [])

    prior_syms = set(prior_df["Symbol"].unique())
    if sym_filter:
        prior_syms = prior_syms & set(sym_filter)

    # Component 1 — overnight settlements (prior-period positions)
    daily_prior = _daily_prior_period_pnl(prior_df, sym_filter, excluded)

    # Component 2 — execution slippage for FUTURES trades on prior-period symbols
    # MtmPnl measures (execution_price - prev_settlement) × qty, which is
    # complementary to PriorMtmPnl and does not overlap.
    daily_fut_exec = pd.Series(dtype=float)
    if not trades_sym.empty and "AssetClass" in trades_sym.columns:
        fut_prior = trades_sym[
            trades_sym["AssetClass"].isin(["FUT", "FOP"]) &
            trades_sym["Symbol"].isin(prior_syms)
        ].copy()
        if not fut_prior.empty:
            fut_prior["_net"] = (fut_prior["MtmPnl"].fillna(0)
                                 + fut_prior["IBCommission"].fillna(0))
            daily_fut_exec = (fut_prior.groupby("TradeDate")["_net"].sum()
                              .pipe(lambda s: s.set_axis(pd.to_datetime(s.index)))
                              .sort_index())

    # Component 3 — new mid-period positions (not in prior_period data)
    new_trades = trades_sym[~trades_sym["Symbol"].isin(prior_syms)]
    daily_new = (_daily_trade_pnl(new_trades, [])
                 if not new_trades.empty else pd.Series(dtype=float))

    # Merge all three components
    idx = daily_prior.index
    for s in [daily_fut_exec, daily_new]:
        if not s.empty:
            idx = idx.union(s.index)

    result = daily_prior.reindex(idx, fill_value=0)
    for s in [daily_fut_exec, daily_new]:
        if not s.empty:
            result = result + s.reindex(idx, fill_value=0)

    return result.sort_index()


@st.cache_data(ttl=1800, show_spinner=False)
def _fx_window_rates(currencies: tuple, start_iso: str, end_iso: str) -> dict:
    """USD-per-unit rate at the start and end of the window, per currency (yfinance)."""
    import yfinance as yf
    direct = {"EUR", "GBP", "AUD", "NZD"}   # quoted CCYUSD
    out = {}
    end_pad = (pd.Timestamp(end_iso) + pd.Timedelta(days=4)).date().isoformat()
    tickers = [f"{str(ccy).upper()}USD=X" if str(ccy).upper() in direct
               else f"{str(ccy).upper()}=X" for ccy in currencies]
    batch = _batch_daily_closes(tickers, start=start_iso, end=end_pad)
    for ccy in currencies:
        c = str(ccy).upper()
        tk = f"{c}USD=X" if c in direct else f"{c}=X"
        inv = c not in direct
        try:
            h = batch.get(tk)
            if h is None:
                h = yf.Ticker(tk).history(start=start_iso, end=end_pad)["Close"].dropna()
            if h is not None and len(h):
                h = h.copy()
                h.index = _fx_align_ibkr(_naive_norm(h.index))
                w = h[(h.index >= pd.Timestamp(start_iso)) & (h.index <= pd.Timestamp(end_iso))]
                if len(w) >= 2:
                    r0, r1 = float(w.iloc[0]), float(w.iloc[-1])
                    if inv:
                        r0, r1 = 1.0 / r0, 1.0 / r1
                    out[ccy] = {"r0": r0, "r1": r1}
        except Exception:
            pass
    return out


# ── Main render ───────────────────────────────────────────────────────────────

@st.fragment
def render_pnl():
    st.markdown("#### P&L Analytics &nbsp;·&nbsp; IBKR Flex Query")
    st.caption(
        "Pull your latest data straight from IBKR (Flex Web Service). "
        "Long-term ETF holds are excluded from trading metrics via the exclusion list below."
    )

    # ── Auto-load from local DB on page start (no re-upload needed after refresh) ──
    if "pnl_sections" not in st.session_state:
        _db_sections = pnl_db.load_sections()
        if _db_sections:
            st.session_state["pnl_sections"] = _db_sections

    # ── Data source: pull from IBKR (Flex Web Service) ─────────────────────────
    with st.expander("📂  Data — pull from IBKR", expanded="pnl_sections" not in st.session_state):
        c_pull, c_pull_info = st.columns([1, 3])
        with c_pull:
            if st.button("🔄  Update from IBKR", key="_pnl_ibkr", use_container_width=True):
                try:
                    from flex_web import update_portfolio
                    with st.spinner("Fetching latest Flex statement from IBKR…"):
                        res = update_portfolio()
                    st.session_state["pnl_sections"] = pnl_db.load_sections()
                    st.session_state.pop("_pnl_run_params", None)
                    st.success(f"Updated from IBKR — sections: {res}")
                    st.rerun()
                except Exception as e:
                    st.error(f"IBKR update failed — {e}")
        with c_pull_info:
            st.caption("Pulls your latest positions & activity via the Flex Web Service token "
                       "(no manual download needed).")

        # ── DB stats ──────────────────────────────────────────────────────────
        _stats = pnl_db.db_stats()
        if _stats:
            _stat_parts = []
            for t, info in _stats.items():
                if "from" in info:
                    _stat_parts.append(
                        f"**{t}**: {info['rows']:,} rows  "
                        f"({str(info['from'])[:10]} → {str(info['to'])[:10]})"
                    )
                else:
                    _stat_parts.append(f"**{t}**: {info['rows']:,} rows")
            st.caption("DB contents:  " + "   ·   ".join(_stat_parts))
        else:
            st.caption("No local database yet — upload a Flex CSV to create one.")

        if _stats and st.button("🗑  Clear database", key="_pnl_clear_db"):
            pnl_db.clear_db()
            st.session_state.pop("pnl_sections", None)
            st.session_state.pop("_pnl_file_id", None)
            st.session_state.pop("_pnl_run_params", None)
            st.rerun()

    if "pnl_sections" not in st.session_state:
        st.info("Click **🔄 Update from IBKR** above to pull your data.")
        return

    sections: dict = st.session_state["pnl_sections"]

    # ── ETF exclusion management ──────────────────────────────────────────────
    with st.expander("🚫  ETF Exclusion List", expanded=False):
        excluded_cur = _load_excluded()
        st.caption("Symbols in this list are excluded from all trading P&L and risk metrics.")
        new_sym = st.text_input("Add symbol to exclude", key="_pnl_add_sym",
                                placeholder="e.g. VUSA", label_visibility="collapsed")
        c_add, c_rem, _ = st.columns([1, 1, 4])
        with c_add:
            if st.button("+ Add", key="_pnl_btn_add") and new_sym.strip():
                excluded_cur = list(set(excluded_cur) | {new_sym.strip().upper()})
                _save_excluded(excluded_cur)
                st.rerun()
        with c_rem:
            rem_sym = st.selectbox("Remove", ["—"] + sorted(excluded_cur),
                                   key="_pnl_rem_sel", label_visibility="collapsed")
            if st.button("− Remove", key="_pnl_btn_rem") and rem_sym != "—":
                excluded_cur = [s for s in excluded_cur if s != rem_sym]
                _save_excluded(excluded_cur)
                st.rerun()
        if excluded_cur:
            st.markdown(
                " ".join(f'<span style="background:#FEF2F2;color:#DC2626;font-size:11px;'
                         f'padding:2px 8px;border-radius:12px;font-weight:600">{s}</span>'
                         for s in sorted(excluded_cur)),
                unsafe_allow_html=True,
            )

    excluded = _load_excluded()

    # ── Pull key DataFrames ───────────────────────────────────────────────────
    nav_df              = sections.get("nav",              pd.DataFrame())
    trades_df           = sections.get("trades",           pd.DataFrame())
    pos_df              = sections.get("positions",        pd.DataFrame())
    div_df              = sections.get("dividends",        pd.DataFrame())
    realized_pnl_raw    = sections.get("realized_pnl",    pd.DataFrame())
    mtm_pnl_raw         = sections.get("mtm_pnl",         pd.DataFrame())
    prior_period_raw    = sections.get("prior_period_pnl", pd.DataFrame())

    if nav_df.empty and trades_df.empty:
        st.warning("No NAV or trade data found — check the Flex report includes NAV and Trades sections.")
        return

    # ── Date range bounds (from raw data, before any filtering) ──────────────
    _all_dates = []
    if not nav_df.empty and "ReportDate" in nav_df.columns:
        _all_dates += nav_df["ReportDate"].dropna().tolist()
    if not trades_df.empty and "TradeDate" in trades_df.columns:
        _all_dates += trades_df["TradeDate"].dropna().tolist()
    _min_date = min(_all_dates).date() if _all_dates else date.today() - timedelta(days=365)
    _max_date = max(_all_dates).date() if _all_dates else date.today()

    # Active symbols from full (unfiltered) data — populate form options
    _active_syms: list[str] = sorted(
        trades_df[~trades_df["Symbol"].isin(excluded)]["Symbol"].dropna().unique().tolist()
    ) if not trades_df.empty and "Symbol" in trades_df.columns else []

    # Auto-initialise on first load so analysis renders immediately after upload
    if "_pnl_run_params" not in st.session_state:
        st.session_state["_pnl_run_params"] = {
            "date_from": _min_date,
            "date_to":   _max_date,
            "sym_filter": [],
            "exclude_syms": [],
        }

    # ── Controls — Run captures current widget values immediately ────────────
    st.markdown("")
    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 2, 3, 3, 1])
    _saved = st.session_state["_pnl_run_params"]
    with fc1:
        _form_date_from = st.date_input(
            "From", value=_saved["date_from"],
            min_value=_min_date, max_value=_max_date, key="_pnl_date_from")
    with fc2:
        _form_date_to = st.date_input(
            "To", value=_saved["date_to"],
            min_value=_min_date, max_value=_max_date, key="_pnl_date_to")
    with fc3:
        _valid_defaults = [s for s in _saved.get("sym_filter", []) if s in _active_syms]
        _form_sym = st.multiselect(
            "Filter by symbol (blank = all)", options=_active_syms,
            default=_valid_defaults, placeholder="All active", key="_pnl_sym_filter")
    with fc4:
        _valid_excl = [s for s in _saved.get("exclude_syms", []) if s in _active_syms]
        _form_excl = st.multiselect(
            "Exclude symbol(s)", options=_active_syms,
            default=_valid_excl, placeholder="None", key="_pnl_excl_filter")
    with fc5:
        st.write("")
        _submitted = st.button("▶  Run", use_container_width=True, type="primary", key="_pnl_run_btn")

    if _submitted:
        st.session_state["_pnl_run_params"] = {
            "date_from": _form_date_from,
            "date_to":   _form_date_to,
            "sym_filter": _form_sym,
            "exclude_syms": _form_excl,
        }

    # Analysis uses whichever params were committed on the last Run click
    _params    = st.session_state["_pnl_run_params"]
    date_from  = _params["date_from"]
    date_to    = _params["date_to"]
    _adhoc_excl = [s for s in _params.get("exclude_syms", []) if s in _active_syms]
    # sym_filter (include-only), with any ad-hoc-excluded symbols removed from it
    sym_filter = [s for s in _params.get("sym_filter", []) if s in _active_syms and s not in _adhoc_excl]
    # merge ad-hoc exclusions into the standing (ETF) exclusion — flows to every calc
    excluded = list(excluded) + _adhoc_excl

    st.caption(f"Data available: **{_min_date}** → **{_max_date}**  ·  "
               f"Showing: **{date_from}** → **{date_to}**")

    ts_from = pd.Timestamp(date_from)
    ts_to   = pd.Timestamp(date_to)

    # Optional: treat foreign-currency balances as FX positions, marked-to-market daily
    # (balance × Δrate) — the unrealised translation P&L, like a future's EOD MTM.
    include_fx = st.checkbox(
        "➕  Count FX P&L as trading — mark foreign balances to market daily (incl. unrealised)",
        value=st.session_state.get("_pnl_incl_fx_v", True), key="_pnl_incl_fx_v")   # default ON (Rajat 2026-07-31: FX is first-class P&L)

    # Daily FX MTM series (USD) = Σ_ccy balance × Δrate over the window.
    # Prefer EXACT per-day balances from the Statement of Funds ledger; fall back to
    # the current-balance approximation if the per-currency ledger isn't available.
    fx_daily = pd.Series(dtype=float)
    fx_src = ""
    if include_fx:
        _fx_start = (ts_from - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        _fx_end = (ts_to + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        _balhist = _fx_daily_balances(sections)
        if _balhist:
            fx_daily = _fx_mtm_from_balances(_balhist, _fx_start, _fx_end)
            fx_src = "exact daily balances"
        else:
            _bal = _fx_native_balances(sections)
            if _bal:
                fx_daily = _fx_translation_daily(tuple(sorted(_bal.items())), _fx_start, _fx_end)
                fx_src = "current-balance approx"
        if not fx_daily.empty:
            fx_daily = fx_daily[(fx_daily.index >= ts_from) & (fx_daily.index <= ts_to)]
    fx_pnl_window = float(fx_daily.sum()) if not fx_daily.empty else 0.0

    # Apply date filter to all DataFrames
    if not nav_df.empty and "ReportDate" in nav_df.columns:
        nav_df = nav_df[(nav_df["ReportDate"] >= ts_from) & (nav_df["ReportDate"] <= ts_to)].copy()
    if not trades_df.empty and "TradeDate" in trades_df.columns:
        trades_df = trades_df[(trades_df["TradeDate"] >= ts_from) & (trades_df["TradeDate"] <= ts_to)].copy()
    if not prior_period_raw.empty and "Date" in prior_period_raw.columns:
        prior_period_raw = prior_period_raw[
            (prior_period_raw["Date"] >= ts_from) & (prior_period_raw["Date"] <= ts_to)
        ].copy()

    # If symbols selected, restrict trade data to those symbols only
    trades_sym = trades_df.copy()
    if sym_filter:
        trades_sym = trades_sym[trades_sym["Symbol"].isin(sym_filter)]
    else:
        if excluded:
            trades_sym = trades_sym[~trades_sym["Symbol"].isin(excluded)]

    _sym_label = f"{', '.join(sym_filter)}" if sym_filter else "all active symbols"

    # ── realized_pnl section filtered by symbol/exclusion (accurate totals) ──────
    # NOTE: realized_pnl covers the full flex report period, not the date-picker range.
    # For date-accurate totals, upload a flex covering exactly the period of interest.
    rp_sym = _pnl_from_realized(realized_pnl_raw, sym_filter, excluded)
    rp_total_pnl = float(rp_sym["TotalFifoPnl"].sum()) if not rp_sym.empty else float("nan")

    # ── NAV series (used for starting capital and total NAV display only) ────────
    current_nav = float("nan")
    nav_series  = pd.Series(dtype=float)
    if not nav_df.empty and "Total" in nav_df.columns:
        nav_series  = nav_df.set_index("ReportDate")["Total"].dropna()
        current_nav = float(nav_series.iloc[-1])

    # ── Active-trading equity curve (symbol-filtered, realized P&L + commissions) ──
    # Returns are expressed as % of starting NAV so Sharpe / VaR are meaningful.
    act_returns = pd.Series(dtype=float)
    act_nav_s   = pd.Series(dtype=float)
    _daily_pnl_nonzero = pd.Series(dtype=float)
    act_sharpe = act_sortino = act_max_dd = act_max_dd_dollar = act_var95 = act_cvar95 = float("nan")
    act_var1sig = float("nan")   # ~1σ downside = 16th-%ile (84% one-tailed) daily P&L
    _avg_daily_pnl  = float("nan")
    _avg_daily_days = 0
    _ann_daily_vol  = float("nan")
    _window_pnl     = float("nan")   # P&L within the selected date range (matches chart)

    # Compute from settlements even when there are no new trades in the window
    # (held positions still have daily mark-to-market via prior_period_pnl).
    if not nav_series.empty:
        _start_nav      = float(nav_series.iloc[0])
        _daily_pnl_raw  = _build_daily_pnl(trades_sym, prior_period_raw, sym_filter, excluded)
        if include_fx and not fx_daily.empty:           # fold realized FX into the daily series
            _daily_pnl_raw = _daily_pnl_raw.add(fx_daily, fill_value=0.0)  # so Sharpe/Sortino/vol/VaR incl FX
        _window_pnl     = float(_daily_pnl_raw.sum())   # = cumulative chart endpoint (incl FX if toggled)
        _full_idx       = nav_series.index
        _daily_pnl      = _daily_pnl_raw.reindex(_full_idx, fill_value=0)
        act_nav_s    = _start_nav + _daily_pnl.cumsum()
        act_returns  = _daily_pnl / _start_nav
        _daily_pnl_nonzero = _daily_pnl[_daily_pnl != 0]
        act_returns  = act_returns[act_returns != 0]
        if len(act_returns) >= 2:
            act_sharpe  = _sharpe(act_returns)
            act_sortino = _sortino(act_returns)
            act_var95, act_cvar95 = _var_cvar_dollar(_daily_pnl_nonzero)
            # ~1σ downside: 16th-%ile (one-tailed 84%), i.e. the -1 standard-deviation loss.
            # (The ±1σ band is 68% two-sided → 16% in each tail, not 32%.)
            act_var1sig = _var_cvar_dollar(_daily_pnl_nonzero, 0.8413)[0]
        act_max_dd_dollar = _max_drawdown_dollar(act_nav_s)
        act_max_dd = _max_drawdown(act_nav_s)  # kept for Calmar ratio denominator
        if not _daily_pnl_nonzero.empty:
            _avg_daily_days = len(_daily_pnl_nonzero)
            # Daily P&L vol = RMS (√mean of squared daily P&L) — a proper volatility that
            # weights big/lumpy days (e.g. FX loss days) correctly, not a mean-of-absolutes.
            _avg_daily_pnl  = float(math.sqrt((_daily_pnl_nonzero ** 2).mean()))
            _ann_daily_vol  = _avg_daily_pnl * math.sqrt(_ANN)

    # Trading P&L for the window = active P&L (already incl. realized FX in the daily
    # series when toggled). Fall back to FX-only if there's no NAV window.
    _trading_pnl = (_window_pnl if math.isfinite(_window_pnl)
                    else (fx_pnl_window if include_fx else 0.0))

    # ── Trade-level metrics ───────────────────────────────────────────────────
    tstats: dict = {}
    if not trades_sym.empty:
        tstats = _trade_stats(trades_sym, [])

    # ── Top KPI row ───────────────────────────────────────────────────────────
    st.markdown("---")
    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
    with k1:
        _fx_note = (f" · incl FX MTM {_fmt(fx_pnl_window, '$', decimals=0)} ({fx_src})"
                    if include_fx else "")
        _kpi("Total P&L (range)",
             _fmt(_trading_pnl, "$", decimals=0),
             _colour(_trading_pnl),
             sub=f"selected dates{_fx_note}<br>lifetime (from entry): {_fmt(rp_total_pnl, '$', decimals=0)}")
    with k2:
        _vol_sub = (f"RMS  ({_avg_daily_days}d)<br>ann: {_fmt(_ann_daily_vol, '$', decimals=0)}"
                    if _avg_daily_days else "—")
        _kpi("Realised Daily P&L Vol",
             _fmt(_avg_daily_pnl, "$", decimals=0),
             "#6366F1",
             sub=_vol_sub)
    with k3:
        _kpi("Sharpe (active)",
             _fmt(act_sharpe, decimals=2),
             _colour(act_sharpe),
             sub=f"annualised ({len(act_returns)}d)" if len(act_returns) else "—")
    with k4:
        _kpi("Sortino (active)",
             _fmt(act_sortino, decimals=2),
             _colour(act_sortino),
             sub=f"downside vol ({len(act_returns)}d)" if len(act_returns) else "—")
    with k5:
        _kpi("Max Drawdown",
             _fmt(act_max_dd_dollar if not trades_sym.empty and not nav_series.empty else float("nan"), "$", decimals=0),
             "#DC2626",
             sub="active trading")
    with k6:
        _kpi("VaR 95% (1d)",
             _fmt(act_var95 if math.isfinite(act_var95) else float("nan"), "$", decimals=0),
             "#DC2626",
             sub=f"~1σ {_fmt(act_var1sig if math.isfinite(act_var1sig) else float('nan'), '$', decimals=0)}"
                 f" · CVaR {_fmt(act_cvar95 if math.isfinite(act_cvar95) else float('nan'), '$', decimals=0)}")
    with k7:
        _kpi("Win Rate",
             _fmt(tstats.get("win_rate", float("nan")) * 100, suffix="%", decimals=1),
             _colour(tstats.get("win_rate", 0) - 0.5),
             sub=f"{tstats.get('wins',0)}W / {tstats.get('losses',0)}L")
    with k8:
        _kpi("Profit Factor",
             _fmt(tstats.get("profit_factor", float("nan")), decimals=2),
             _colour(tstats.get("profit_factor", 0) - 1.0))

    # ── Sub-tabs ──────────────────────────────────────────────────────────────
    st.markdown("")
    t_overview, t_risk, t_trades, t_positions = st.tabs(
        ["📈  P&L Overview", "⚠️  Risk", "🎯  Trade Analysis", "📋  Positions"]
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with t_overview:
        if not nav_df.empty and "Total" in nav_df.columns:
            nav_s = nav_df.set_index("ReportDate")["Total"].dropna()
            pnl_s = nav_s - nav_s.iloc[0]

            # ── Data-coverage check — warn if the daily-P&L source has gaps ────
            _bdays = pd.bdate_range(ts_from, ts_to)
            _data_days = set()
            if not prior_period_raw.empty and "Date" in prior_period_raw.columns:
                _data_days |= set(pd.to_datetime(prior_period_raw["Date"]).dt.normalize())
            if not trades_df.empty and "TradeDate" in trades_df.columns:
                _data_days |= set(pd.to_datetime(trades_df["TradeDate"]).dt.normalize())
            _n_cov = sum(1 for d in _bdays if d in _data_days)
            if len(_bdays) >= 5 and _n_cov < 0.6 * len(_bdays):
                st.warning(
                    f"⚠ Daily P&L data covers only **{_n_cov} of {len(_bdays)}** business days in this "
                    f"range — the chart and window metrics will look sparse or misleading. Your Flex "
                    f"pull has date gaps (e.g. June 2026 is missing). Set your Flex query's period to "
                    f"**“Last 365 Days”** and hit **🔄 Update from IBKR** to backfill, or pick a date "
                    f"range that's fully covered."
                )

            # ── Cumulative P&L chart ──────────────────────────────────────────
            fig_nav = go.Figure()

            # Zero anchor one day before the range start, so the curve begins at 0
            _anchor = nav_s.index[0] - pd.Timedelta(days=1)

            # Active / symbol-filtered line (default visible)
            if not trades_sym.empty or not prior_period_raw.empty:
                daily_trade = _build_daily_pnl(trades_sym, prior_period_raw, sym_filter, excluded)
                if include_fx and not fx_daily.empty:
                    daily_trade = daily_trade.add(fx_daily, fill_value=0)
                all_dates   = nav_s.index.union(daily_trade.index).sort_values()
                daily_trade = daily_trade.reindex(all_dates, fill_value=0)
                cum_trade   = daily_trade.cumsum().reindex(nav_s.index, method="ffill").fillna(0)
                # Anchor at 0 at the start so the line originates from zero (not the first day's P&L)
                cum_trade   = pd.concat([pd.Series([0.0], index=[_anchor]), cum_trade])

                fig_nav.add_trace(go.Scatter(
                    x=cum_trade.index, y=cum_trade.clip(lower=0),
                    mode="lines", line=dict(width=0),
                    fill="tozeroy", fillcolor="rgba(5,150,105,0.15)",
                    showlegend=False, hoverinfo="skip",
                ))
                fig_nav.add_trace(go.Scatter(
                    x=cum_trade.index, y=cum_trade.clip(upper=0),
                    mode="lines", line=dict(width=0),
                    fill="tozeroy", fillcolor="rgba(220,38,38,0.15)",
                    showlegend=False, hoverinfo="skip",
                ))
                fig_nav.add_trace(go.Scatter(
                    x=cum_trade.index, y=cum_trade,
                    name=f"P&L — {_sym_label}",
                    line=dict(color="#059669", width=2),
                ))

            # Total NAV line — hidden by default, click legend to show
            pnl_s_anchored = pd.concat([pd.Series([0.0], index=[_anchor]), pnl_s])
            fig_nav.add_trace(go.Scatter(
                x=pnl_s_anchored.index, y=pnl_s_anchored,
                name="Total Portfolio P&L (incl. ETF holds)",
                line=dict(color="#94A3B8", width=1.5, dash="dot"),
                visible="legendonly",
            ))
            _chart_title = (
                f"Cumulative P&L — {_sym_label}"
                if sym_filter else
                "Cumulative Active Trading P&L (excl. long-term ETF holds, realized only)"
            )
            # Render the title as a Streamlit heading ABOVE the chart (not a Plotly
            # in-chart title) so it can't overlap the plot / horizontal legend.
            st.markdown(f"**{_chart_title}**")
            fig_nav.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=30, b=10),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
                font=dict(size=11, color="#1A202C"),
                yaxis=dict(tickprefix="$", gridcolor="#E8EDF5"),
                xaxis=dict(gridcolor="#E8EDF5"),
            )
            st.plotly_chart(fig_nav, use_container_width=True)
            st.caption(
                "Green = active-trading P&L **within the selected dates** (excl. long-term ETF holds, "
                "incl. overnight settlements & execution slippage) — this now matches the "
                "**Total P&L (range)** KPI above. Grey (toggle in the legend) = whole-portfolio P&L "
                "incl. ETF holds. The KPI's *lifetime (from entry)* figure is the full-period P&L from "
                "`realized_pnl` — larger because it includes gains made *before* your start date."
            )

        # ── P&L breakdown (this window): by instrument + by currency ─────────
        with st.expander("📊  P&L breakdown — this window (by instrument & currency)", expanded=True):
            # Full per-symbol P&L (settlements + execution slippage + rolls/new positions),
            # mirroring _build_daily_pnl so the breakdown sums to the Total P&L KPI.
            _fbs = _pnl_by_symbol_full(trades_sym, prior_period_raw, sym_filter, excluded)

            # symbol → underlying map, so options (e.g. WY4K6 C1100 on ZNU6) theme by
            # their underlying future (US 10Y) instead of their raw option root ("WY").
            _undl: dict = {}
            for _d in (prior_period_raw, trades_df):
                if not _d.empty and {"Symbol", "UnderlyingSymbol"}.issubset(_d.columns):
                    for _sy, _un in zip(_d["Symbol"].astype(str), _d["UnderlyingSymbol"].astype(str)):
                        if _un and _un.lower() != "nan":
                            _undl.setdefault(_sy, _un)
            _fxbc = {}
            if include_fx:
                _bh2 = _fx_daily_balances(sections)
                if _bh2:
                    _fxbc = _fx_mtm_by_ccy(
                        _bh2, (ts_from - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                        (ts_to + pd.Timedelta(days=2)).strftime("%Y-%m-%d"), ts_from, ts_to)

            # ── Themed summary (Bitcoin, Euribor, Gold, FX pairs, …) ─────────
            _theme: dict = {}
            for _s, _v in _fbs.items():
                _k = _theme_of(_s, _undl.get(_s, "")); _theme[_k] = _theme.get(_k, 0.0) + _v
            for _c, _v in _fxbc.items():
                _k = f"{_c} (FX)"; _theme[_k] = _theme.get(_k, 0.0) + _v
            _cc = lambda v: "#059669" if v >= 0 else "#DC2626"
            _f2 = lambda v: _fmt(v, "$", decimals=0)
            _sth = "background:#0F172A;color:#F8FAFC;font-size:12px;font-weight:700;padding:6px 10px;text-align:right"
            _sthl = _sth.replace("text-align:right", "text-align:left")
            _std = "font-size:12px;padding:5px 10px;border-bottom:1px solid #E2E8F0;text-align:right"
            _stdl = _std.replace("text-align:right", "text-align:left")
            _stf = "font-size:12px;padding:6px 10px;border-top:2px solid #475569;font-weight:800;text-align:right"
            _stfl = _stf.replace("text-align:right", "text-align:left")

            # ── By asset class (Rates / Commodities / Equities / Crypto / FX) ──
            _ac: dict = {}
            for _k, _v in _theme.items():
                _acls = _asset_class_of(_k); _ac[_acls] = _ac.get(_acls, 0.0) + _v
            _acrows = ""
            for _k, _v in sorted(_ac.items(), key=lambda x: -abs(x[1])):
                _acrows += (f"<tr><td style='{_stdl}'><b>{_k}</b></td>"
                            f"<td style='{_std};color:{_cc(_v)};font-weight:700'>{_f2(_v)}</td></tr>")
            _act = sum(_ac.values())
            _acrows += (f"<tr><td style='{_stfl}'>TOTAL</td>"
                        f"<td style='{_stf};color:{_cc(_act)}'>{_f2(_act)}</td></tr>")
            _achdr = f"<tr><th style='{_sthl}'>Asset class</th><th style='{_sth}'>P&amp;L (window)</th></tr>"
            st.markdown("**By asset class**")
            st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
                        f"font-family:monospace'><thead>{_achdr}</thead><tbody>{_acrows}</tbody></table></div>",
                        unsafe_allow_html=True)

            # ── By theme (Bitcoin, Euribor, Gold, FX pairs, …) ────────────────
            st.markdown("**By theme**")
            _srows = ""
            for _k, _v in sorted(_theme.items(), key=lambda x: -abs(x[1])):
                _srows += (f"<tr><td style='{_stdl}'><b>{_k}</b></td>"
                           f"<td style='{_std};color:{_cc(_v)};font-weight:700'>{_f2(_v)}</td></tr>")
            _gt = sum(_theme.values())
            _srows += (f"<tr><td style='{_stfl}'>TOTAL</td>"
                       f"<td style='{_stf};color:{_cc(_gt)}'>{_f2(_gt)}</td></tr>")
            _shdr = f"<tr><th style='{_sthl}'>Theme</th><th style='{_sth}'>P&amp;L (window)</th></tr>"
            st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
                        f"font-family:monospace'><thead>{_shdr}</thead><tbody>{_srows}</tbody></table></div>",
                        unsafe_allow_html=True)
            # one-line narrative
            _los = [(k, v) for k, v in sorted(_theme.items(), key=lambda x: x[1]) if v < 0][:3]
            _gan = [(k, v) for k, v in sorted(_theme.items(), key=lambda x: -x[1]) if v > 0][:2]
            _parts = []
            if _los:
                _parts.append("biggest drag: " + ", ".join(f"**{k}** {_f2(v)}" for k, v in _los))
            if _gan:
                _parts.append("offset by " + ", ".join(f"**{k}** {_f2(v)}" for k, v in _gan))
            _dr = f"{ts_from.strftime('%d %b')} → {ts_to.strftime('%d %b %Y')}"
            st.markdown(f"🧠 **{_dr}:** net **{_f2(_gt)}** — " + "; ".join(_parts) + "."
                        if _parts else f"🧠 **{_dr}:** net **{_f2(_gt)}**.")
            st.markdown("**Per-instrument detail**")

            _th = "background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;padding:4px 8px;text-align:right"
            _thl = _th.replace("text-align:right", "text-align:left")
            _td = "font-size:11px;padding:3px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
            _tdl = _td.replace("text-align:right", "text-align:left")
            _cc = lambda v: "#059669" if v >= 0 else "#DC2626"
            _rows = ""
            _fut_tot = 0.0
            for _sym, _v in sorted(_fbs.items(), key=lambda x: x[1]):
                _fut_tot += _v
                _rows += (f"<tr><td style='{_tdl}'><b>{_sym}</b></td><td style='{_tdl};color:#94A3B8'>Futures/Opt</td>"
                          f"<td style='{_td};color:{_cc(_v)}'>{_fmt(_v, '$', decimals=0)}</td></tr>")
            _fx_tot = 0.0
            for _ccy, _v in sorted(_fxbc.items(), key=lambda x: x[1]):
                _fx_tot += _v
                _rows += (f"<tr><td style='{_tdl}'><b>{_ccy}</b></td><td style='{_tdl};color:#94A3B8'>FX MTM</td>"
                          f"<td style='{_td};color:{_cc(_v)}'>{_fmt(_v, '$', decimals=0)}</td></tr>")
            _tf = "font-size:11px;padding:5px 8px;border-top:2px solid #475569;font-weight:700;text-align:right"
            _tfl = _tf.replace("text-align:right", "text-align:left")
            _foot = (f"<tr><td style='{_tfl}'>Futures / Options subtotal</td><td style='{_tf}'></td>"
                     f"<td style='{_tf};color:{_cc(_fut_tot)}'>{_fmt(_fut_tot, '$', decimals=0)}</td></tr>")
            if include_fx:
                _foot += (f"<tr><td style='{_tfl}'>FX MTM subtotal</td><td style='{_tf}'></td>"
                          f"<td style='{_tf};color:{_cc(_fx_tot)}'>{_fmt(_fx_tot, '$', decimals=0)}</td></tr>")
            _grand = _fut_tot + _fx_tot
            _foot += (f"<tr><td style='{_tfl}'>TOTAL</td><td style='{_tf}'></td>"
                      f"<td style='{_tf};color:{_cc(_grand)}'>{_fmt(_grand, '$', decimals=0)}</td></tr>")
            _hdr = (f"<tr><th style='{_thl}'>Instrument</th><th style='{_thl}'>Type</th>"
                    f"<th style='{_th}'>P&L (window)</th></tr>")
            st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
                        f"font-family:monospace'><thead>{_hdr}</thead><tbody>{_rows}{_foot}</tbody></table></div>",
                        unsafe_allow_html=True)
            st.caption("**Futures/Options** = full windowed P&L by symbol — overnight settlements "
                       "**plus** execution slippage and mid-period rolls/new positions "
                       "(same three components as the Total). **FX MTM** = exact daily mark-to-market "
                       "by currency (shown when *Count FX* is on). Futures/Options subtotal now "
                       "matches the **Total P&L (range)** KPI above (pre-FX).")

        # ── P&L Bridge / Waterfall reconciliation ────────────────────────────
        # Decompose the WHOLE-PORTFOLIO NAV change over the selected window into
        # consistent, all-windowed buckets (previously this mixed full-period
        # realized with a windowed NAV change, so "Other" was a garbage residual).
        total_nav_pnl = float(pnl_s.iloc[-1]) if not nav_df.empty and "Total" in nav_df.columns else 0.0

        # Active trading P&L (excl. ETFs) — same windowed series as the chart/KPI
        active_pnl = _window_pnl if math.isfinite(_window_pnl) else 0.0

        # Long-term ETF holdings P&L (windowed) — prior_period MTM on excluded symbols
        etf_pnl = 0.0
        if not prior_period_raw.empty and {"Symbol", "PriorMtmPnl"}.issubset(prior_period_raw.columns):
            _e  = prior_period_raw[prior_period_raw["Symbol"].isin(excluded)]
            _fx = pd.to_numeric(_e.get("FXRateToBase"), errors="coerce").fillna(1.0)
            etf_pnl = float((pd.to_numeric(_e["PriorMtmPnl"], errors="coerce").fillna(0) * _fx).sum())

        # Dividends (date-filtered)
        dividends_total = 0.0
        if not div_df.empty and "Amount" in div_df.columns:
            d = div_df.copy()
            dt_col = "Date/Time" if "Date/Time" in d.columns else None
            if dt_col:
                d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
                d = d[(d[dt_col] >= ts_from) & (d[dt_col] <= ts_to)]
            if "Type" in d.columns:
                d = d[d["Type"].str.contains("Dividend|dividend", na=False)]
            dividends_total = float(d["Amount"].sum())

        # Slimmed NAV bridge: a single Trading bar (per-instrument / asset-class / FX detail
        # lives in the breakdown tables above) → ETF → Dividends → non-trading residual → NAV.
        # trading_total already includes the FX balance-MTM when the toggle is on (it's folded
        # into _window_pnl), so we must NOT add FX again here — that was a double-count.
        trading_total = active_pnl
        # Non-trading residual = interest, FX translation on balances, deposits/withdrawals
        # (plus realized/translation FX when the FX toggle is off, since it's not in trading then).
        residual = total_nav_pnl - trading_total - etf_pnl - dividends_total

        _fx_note = " (incl. FX)" if include_fx else ""
        bridge_labels  = [f"Trading total{_fx_note}", "Long-term ETF\nholdings", "Dividends",
                          "Non-trading\nresidual", "Total\n(NAV change)"]
        bridge_values  = [trading_total, etf_pnl, dividends_total, residual, total_nav_pnl]
        bridge_measure = ["relative"] * (len(bridge_values) - 1) + ["total"]

        fig_bridge = go.Figure(go.Waterfall(
            orientation="v",
            measure=bridge_measure,
            x=bridge_labels,
            y=bridge_values,
            connector=dict(line=dict(color="#94A3B8", width=1, dash="dot")),
            increasing=dict(marker_color="#059669"),
            decreasing=dict(marker_color="#DC2626"),
            totals=dict(marker_color="#1E40AF"),
            text=[f"${v:,.0f}" for v in bridge_values],
            textposition="outside",
        ))
        _bridge_title = (
            f"P&L Bridge — {_sym_label}" if sym_filter else
            "P&L Bridge — how NAV change is composed (excl. long-term ETF holds)"
        )
        fig_bridge.update_layout(
            title=_bridge_title,
            height=340, margin=dict(l=10, r=10, t=50, b=30),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
            font=dict(size=11, color="#1A202C"),
            yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#E8EDF5"),
            xaxis=dict(gridcolor="#E8EDF5"),
            showlegend=False,
        )
        st.plotly_chart(fig_bridge, use_container_width=True)
        st.caption(
            "How your **whole-portfolio NAV change** over the selected dates breaks down. "
            "**Trading total** matches the Total P&L KPI / chart / breakdown above (includes FX when the "
            "*Count FX* box is ticked). **ETF holdings** = MTM on the excluded long-term ETF holds. "
            "**Non-trading residual** = interest, FX translation on balances, and deposits/withdrawals "
            "(plus realized FX when the *Count FX* box is off). **Total = NAV change**. "
            "Per-instrument / asset-class trading detail is in the breakdown above; open *Investigate the "
            "residual* below to decompose the residual bar."
        )

        # ── Investigate the residual: NAV change → Cash vs Stock, then Cash breakdown ──
        with st.expander("🔍  Investigate the residual — Cash vs ETF, FX & cash flows"):
            # (1) Windowed NAV component deltas (clean, straight from the nav section)
            cash_d = stock_d = total_d = float("nan")
            if not nav_df.empty and {"Cash", "Stock", "Total"}.issubset(nav_df.columns):
                _n = nav_df.sort_values("ReportDate")
                _f, _l = _n.iloc[0], _n.iloc[-1]

                def _delta(col):
                    return float(pd.to_numeric(_l[col], errors="coerce")
                                 - pd.to_numeric(_f[col], errors="coerce"))
                cash_d, stock_d, total_d = _delta("Cash"), _delta("Stock"), _delta("Total")
                st.markdown(
                    f"**NAV split over the window** (from IBKR's daily NAV):\n"
                    f"- **Cash**: {_fmt(cash_d, '$', decimals=0)} — holds trading settlements, FX, "
                    f"interest, dividends & deposits/withdrawals\n"
                    f"- **Stock / long-term ETF holds**: {_fmt(stock_d, '$', decimals=0)} — ETF price moves "
                    f"(+ any buys/sells)\n"
                    f"- **Total NAV change**: {_fmt(total_d, '$', decimals=0)}"
                )

            # (2) Attribute the Cash change to what we can window precisely
            realized_fx = 0.0
            fa = sections.get("fx_activity", pd.DataFrame())
            if not fa.empty and {"RealizedP/L", "ReportDate"}.issubset(fa.columns):
                _fa = fa.copy()
                _fa["_d"] = pd.to_datetime(_fa["ReportDate"].astype(str), format="%Y%m%d", errors="coerce")
                _fa = _fa[(_fa["_d"] >= ts_from) & (_fa["_d"] <= ts_to)]
                realized_fx = float(pd.to_numeric(_fa["RealizedP/L"], errors="coerce").fillna(0).sum())
            _active     = _window_pnl if math.isfinite(_window_pnl) else 0.0
            _explained  = _active + realized_fx + dividends_total
            _unexplained = (cash_d - _explained) if math.isfinite(cash_d) else float("nan")

            st.markdown("**What's inside the Cash change (windowed where possible):**")
            _rows = [
                ("Active trading P&L (futures → cash)", _active),
                ("Realized FX (conversions booked)", realized_fx),
                ("Dividends received", dividends_total),
                ("Unexplained → FX translation + interest + deposits/withdrawals", _unexplained),
                ("= Total Cash change", cash_d),
            ]
            _th = "font-size:12px;padding:4px 10px;border-bottom:1px solid #E2E8F0"
            _html = "<table style='border-collapse:collapse'>"
            for lbl, val in _rows:
                bold = "font-weight:700" if lbl.startswith("=") else ""
                col = "#059669" if (val or 0) >= 0 else "#DC2626"
                _html += (f"<tr><td style='{_th};{bold}'>{lbl}</td>"
                          f"<td style='{_th};text-align:right;color:{col};{bold}'>{_fmt(val, '$', decimals=0)}</td></tr>")
            _html += "</table>"
            st.markdown(_html, unsafe_allow_html=True)

            # (3) FX-book context — the likely driver of the "unexplained" line
            cs = sections.get("cash_summary", pd.DataFrame())
            if not cs.empty and "CurrencyPrimary" in cs.columns:
                _cur = cs[cs["LevelOfDetail"].astype(str).str.lower() == "currency"] if "LevelOfDetail" in cs.columns else cs
                _fx_rows = []
                for _, r in _cur.iterrows():
                    c = str(r.get("CurrencyPrimary") or "")
                    if c in ("", "USD", "BASE_SUMMARY"):
                        continue
                    b = pd.to_numeric(r.get("EndingCash"), errors="coerce")
                    if pd.notna(b) and abs(b) > 1:
                        _fx_rows.append((c, float(b)))
                if _fx_rows:
                    _dep = None
                    _base = cs[cs["CurrencyPrimary"] == "BASE_SUMMARY"] if "CurrencyPrimary" in cs.columns else pd.DataFrame()
                    if not _base.empty and "Deposit/Withdrawals" in cs.columns:
                        _dep = pd.to_numeric(_base["Deposit/Withdrawals"], errors="coerce").iloc[0]
                    st.markdown(
                        "**Why 'unexplained' is usually large for you — your foreign-cash book:** "
                        + ", ".join(f"{c} {b:,.0f}" for c, b in _fx_rows)
                        + ". As the dollar moves, the USD value of these balances swings (FX translation), "
                        "landing in Cash without being trading P&L."
                    )
                    if _dep is not None and abs(_dep) > 1:
                        st.markdown(
                            f"**Cash flows:** net deposits/withdrawals of **{_fmt(float(_dep), '$', decimals=0)}** "
                            "over the flex-query period (not date-windowed). If any landed inside your selected "
                            "dates, they moved Cash without being P&L."
                        )
            st.caption(
                "For an exact, IBKR-computed split (Deposits · MTM · Realized · Dividends · Interest · "
                "**FX Translation** · Withdrawals), add the **“Change in NAV”** section to your Flex query — "
                "then I can replace the 'unexplained' line with precise figures."
            )

        # ── FX P&L by currency (vs USD), selected period ─────────────────────
        st.markdown("##### 💱  FX P&L by currency (vs USD) — selected period")
        _fa_all = sections.get("fx_activity", pd.DataFrame())
        _cs_all = sections.get("cash_summary", pd.DataFrame())
        _rz = {}
        if not _fa_all.empty and {"RealizedP/L", "FXCurrency", "ReportDate"}.issubset(_fa_all.columns):
            _ff = _fa_all.copy()
            _ff["_d"] = pd.to_datetime(_ff["ReportDate"].astype(str), format="%Y%m%d", errors="coerce")
            _ff = _ff[(_ff["_d"] >= ts_from) & (_ff["_d"] <= ts_to)]
            _rz = (pd.to_numeric(_ff["RealizedP/L"], errors="coerce").fillna(0)
                   .groupby(_ff["FXCurrency"]).sum().to_dict())
        _bal = {}
        if not _cs_all.empty and "CurrencyPrimary" in _cs_all.columns:
            _cc = (_cs_all[_cs_all["LevelOfDetail"].astype(str).str.lower() == "currency"]
                   if "LevelOfDetail" in _cs_all.columns else _cs_all)
            for _, r in _cc.iterrows():
                c = str(r.get("CurrencyPrimary") or "")
                if c in ("", "USD", "BASE_SUMMARY"):
                    continue
                b = pd.to_numeric(r.get("EndingCash"), errors="coerce")
                if pd.notna(b):
                    _bal[c] = float(b)
        _ccys = sorted((set(_rz) | set(_bal)) - {"USD", ""})
        _rates = (_fx_window_rates(tuple(_ccys), ts_from.date().isoformat(), ts_to.date().isoformat())
                  if _ccys else {})
        # EXACT per-currency daily MTM from the Statement-of-Funds ledger — prior-day
        # balance × each day's rate move. This is the ECONOMIC FX P&L (matches the
        # Risk-tab methodology). The old "balance × window Δrate" estimate used the
        # CURRENT balance, which is badly wrong whenever the balance changed during
        # the window (2026-07-31: Rajat held ¥124M through a 2.2% yen rally then sold
        # most of it — estimate said ~$0.6k on the remnant vs ~$19k economic truth).
        _balhist = _fx_daily_balances(sections)
        _mtm = {}
        if _balhist:
            _rstart = (ts_from - pd.Timedelta(days=7)).date().isoformat()
            _rend = (ts_to + pd.Timedelta(days=1)).date().isoformat()
            _mtm = _fx_mtm_by_ccy(_balhist, _rstart, _rend, ts_from, ts_to)

        if not _ccys:
            st.caption("No non-USD activity or balances in this period.")
        else:
            _th = "background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;padding:5px 8px;text-align:right"
            _thl = _th.replace("text-align:right", "text-align:left")
            _td = "font-size:11px;padding:4px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
            _tdl = _td.replace("text-align:right", "text-align:left")

            def _usd(v):
                if v is None:
                    return "—"
                return f"<span style='color:{'#059669' if v >= 0 else '#DC2626'}'>${v:,.0f}</span>"

            _exact = bool(_mtm)
            _fx_col = "FX P&L (daily MTM)" if _exact else "Transl. est. (rough)"
            _hdr = (f"<tr><th style='{_thl}'>Ccy</th>"
                    f"<th style='{_th}'>{_fx_col}</th>"
                    f"<th style='{_th}'>Rate start</th><th style='{_th}'>Rate end</th><th style='{_th}'>Move</th>"
                    f"<th style='{_th}'>Balance now</th>"
                    f"<th style='{_th}'>IBKR realized (ref)</th></tr>")
            _rows = ""
            _t_fx = _t_real = 0.0
            for c in _ccys:
                rl = float(_rz.get(c, 0.0))
                b  = _bal.get(c)
                rr = _rates.get(c)
                r0 = rr["r0"] if rr else None
                r1 = rr["r1"] if rr else None
                if _exact:
                    fxv = _mtm.get(c)
                else:   # fallback estimate on the current balance — labeled rough
                    fxv = (b * (r1 - r0)) if (b is not None and r0 is not None
                                              and r1 is not None) else None
                mv = ((r1 / r0 - 1) * 100) if (r0 and r1) else None
                _t_fx += (fxv or 0.0)
                _t_real += rl
                _rows += (
                    f"<tr><td style='{_tdl}'><b>{c}</b></td>"
                    f"<td style='{_td};font-weight:700'>{_usd(fxv)}</td>"
                    f"<td style='{_td}'>{('%.5g' % r0) if r0 else '—'}</td>"
                    f"<td style='{_td}'>{('%.5g' % r1) if r1 else '—'}</td>"
                    f"<td style='{_td}'>{('%+.2f%%' % mv) if mv is not None else '—'}</td>"
                    f"<td style='{_td}'>{f'{b:,.0f}' if b is not None else '—'}</td>"
                    f"<td style='{_td}'>{_usd(rl)}</td></tr>"
                )
            _rows += (
                f"<tr><td style='{_tdl};font-weight:700'>Total</td>"
                f"<td style='{_td};font-weight:700'>{_usd(_t_fx)}</td>"
                f"<td style='{_td}'></td><td style='{_td}'></td><td style='{_td}'></td><td style='{_td}'></td>"
                f"<td style='{_td};font-weight:700'>{_usd(_t_real)}</td></tr>"
            )
            st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
                        f"font-family:monospace'><thead>{_hdr}</thead><tbody>{_rows}</tbody></table></div>",
                        unsafe_allow_html=True)
            st.caption(
                ("**FX P&L (daily MTM)** = Σ over each day of (prior-day ending balance × that day's rate move), "
                 "using EXACT per-day balances from the IBKR Statement-of-Funds ledger — the economic FX P&L of "
                 "the whole book (same methodology as the Risk-tab FX numbers). Balance changes enter at their "
                 "transaction-day rate, so conversions are correctly embedded — do NOT add the realized column "
                 "on top. **IBKR realized (ref)** = IBKR's own virtual currency-pair lot accounting "
                 "(`fx_activity`); bookkeeping reference only — it tracks only explicitly-traded FX lots and "
                 "their IBKR cost bases, not the economics of balances built up from margin/dividend flows. "
                 if _exact else
                 "**No per-currency Statement-of-Funds ledger in the Flex data** — falling back to the rough "
                 "estimate: current balance × window rate move (WRONG when the balance changed mid-window; add "
                 "the Statement of Funds section to your Flex query for exact daily MTM). ")
                + "Rates are USD per 1 unit (yfinance)."
            )

        # ── Monthly P&L (date-filtered trades, asset-class aware) ────────────
        if not trades_sym.empty and "TradeDate" in trades_sym.columns:
            _mon = trades_sym.copy()
            _mon["_net"] = _net_per_trade(_mon)
            _mon["_month"] = _mon["TradeDate"].dt.to_period("M").dt.to_timestamp()
            monthly_trade = _mon.groupby("_month")["_net"].sum()

            fig_m = go.Figure(go.Bar(
                x=monthly_trade.index.strftime("%b %Y"),
                y=monthly_trade.values,
                marker_color=["#059669" if v >= 0 else "#DC2626" for v in monthly_trade.values],
                text=[f"${v:,.0f}" for v in monthly_trade.values],
                textposition="outside",
            ))
            fig_m.update_layout(
                title=f"Monthly P&L — {_sym_label} (approx for futures: MtmPnl per trade, date-filtered)",
                height=280, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
                font=dict(size=11, color="#1A202C"),
                yaxis=dict(tickprefix="$", gridcolor="#E8EDF5"),
                xaxis=dict(gridcolor="#E8EDF5"),
            )
            st.plotly_chart(fig_m, use_container_width=True)

        # ── P&L by asset class — from realized_pnl section (accurate) ────────
        if not rp_sym.empty and "AssetClass" in rp_sym.columns:
            by_cls = rp_sym.groupby("AssetClass")["TotalFifoPnl"].sum().sort_values()
            fig_cls = go.Figure(go.Bar(
                x=by_cls.values, y=by_cls.index,
                orientation="h",
                marker_color=["#059669" if v >= 0 else "#DC2626" for v in by_cls.values],
                text=[f"${v:,.0f}" for v in by_cls.values],
                textposition="outside",
            ))
            fig_cls.update_layout(
                title=f"Total P&L by Asset Class — {_sym_label} (realized_pnl section, full flex period)",
                height=220, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
                font=dict(size=11, color="#1A202C"),
                xaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#E8EDF5"),
                yaxis=dict(gridcolor="#E8EDF5"),
            )
            st.plotly_chart(fig_cls, use_container_width=True)

        # ── P&L Reconciliation: realized_pnl.TotalFifoPnl vs mtm_pnl.Total ──────
        with st.expander("🔍  P&L Reconciliation — realized_pnl vs MTM", expanded=False):
            if mtm_pnl_raw.empty:
                st.caption("mtm_pnl section not found in this flex report.")
            elif rp_sym.empty:
                st.caption("realized_pnl section not found in this flex report.")
            else:
                # Prep mtm_pnl: numeric coercion, filter empties and CASH rows
                mtm_work = mtm_pnl_raw.copy()
                for c in ["Total", "TransactionMtmPnl", "PriorOpenMtmPnl"]:
                    if c in mtm_work.columns:
                        mtm_work[c] = pd.to_numeric(mtm_work[c], errors="coerce").fillna(0)
                mtm_work = mtm_work[mtm_work["Symbol"].str.strip().ne("")]
                if "AssetClass" in mtm_work.columns:
                    mtm_work = mtm_work[mtm_work["AssetClass"] != "CASH"]
                if excluded:
                    mtm_work = mtm_work[~mtm_work["Symbol"].isin(excluded)]
                if sym_filter:
                    mtm_work = mtm_work[mtm_work["Symbol"].isin(sym_filter)]

                mtm_cols = [c for c in ["Symbol", "Total", "TransactionMtmPnl"] if c in mtm_work.columns]
                mtm_work = mtm_work[mtm_cols].rename(columns={"Total": "MtmTotal"})

                # Merge with realized_pnl (filter out summary rows and FX cash rows)
                rp_cols = [c for c in ["Symbol", "AssetClass", "TotalFifoPnl"] if c in rp_sym.columns]
                rp_recon = rp_sym[rp_cols].copy()
                rp_recon = rp_recon[rp_recon["Symbol"].str.strip().ne("")]
                if "AssetClass" in rp_recon.columns:
                    rp_recon = rp_recon[rp_recon["AssetClass"] != "CASH"]
                merged = pd.merge(rp_recon, mtm_work, on="Symbol", how="outer")
                merged["TotalFifoPnl"] = pd.to_numeric(merged["TotalFifoPnl"], errors="coerce").fillna(0)
                merged["MtmTotal"]     = pd.to_numeric(merged.get("MtmTotal", 0), errors="coerce").fillna(0)
                merged["Diff"]         = merged["TotalFifoPnl"] - merged["MtmTotal"]
                merged = merged.sort_values("TotalFifoPnl", key=lambda s: s.abs(), ascending=False)

                th_s = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
                        "padding:6px 10px;text-align:right")
                th_l = th_s.replace("text-align:right", "text-align:left")
                td_s = "font-size:11px;padding:5px 10px;border-bottom:1px solid #E2E8F0"

                def _diff_colour(d: float) -> str:
                    return "#059669" if abs(d) < 200 else ("#F59E0B" if abs(d) < 2000 else "#DC2626")

                rows_html = ""
                for _, r in merged.iterrows():
                    ac = r.get("AssetClass", "") or ""
                    ac_html = (f'<span style="background:#EFF6FF;color:#1D4ED8;font-size:10px;'
                               f'padding:1px 6px;border-radius:8px">{ac}</span>') if ac else ""
                    fifo_col = "#059669" if r["TotalFifoPnl"] >= 0 else "#DC2626"
                    mtm_col  = "#059669" if r["MtmTotal"] >= 0 else "#DC2626"
                    rows_html += (
                        f'<tr>'
                        f'<td style="{td_s};text-align:left;font-weight:600">{r["Symbol"]}</td>'
                        f'<td style="{td_s};text-align:center">{ac_html}</td>'
                        f'<td style="{td_s};text-align:right;color:{fifo_col};font-weight:700">'
                        f'${r["TotalFifoPnl"]:,.0f}</td>'
                        f'<td style="{td_s};text-align:right;color:{mtm_col};font-weight:700">'
                        f'${r["MtmTotal"]:,.0f}</td>'
                        f'<td style="{td_s};text-align:right;color:{_diff_colour(r["Diff"])};font-weight:700">'
                        f'${r["Diff"]:,.0f}</td>'
                        f'</tr>'
                    )

                tot_fifo = float(merged["TotalFifoPnl"].sum())
                tot_mtm  = float(merged["MtmTotal"].sum())
                tot_diff = tot_fifo - tot_mtm
                rows_html += (
                    f'<tr style="background:#F1F5F9">'
                    f'<td style="{td_s};text-align:left;font-weight:700">TOTAL</td>'
                    f'<td style="{td_s}"></td>'
                    f'<td style="{td_s};text-align:right;color:{"#059669" if tot_fifo>=0 else "#DC2626"};font-weight:700">'
                    f'${tot_fifo:,.0f}</td>'
                    f'<td style="{td_s};text-align:right;color:{"#059669" if tot_mtm>=0 else "#DC2626"};font-weight:700">'
                    f'${tot_mtm:,.0f}</td>'
                    f'<td style="{td_s};text-align:right;color:{_diff_colour(tot_diff)};font-weight:700">'
                    f'${tot_diff:,.0f}</td>'
                    f'</tr>'
                )

                html_tbl = (
                    '<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%">'
                    f'<thead><tr>'
                    f'<th style="{th_l}">Symbol</th>'
                    f'<th style="{th_s}">Class</th>'
                    f'<th style="{th_s}">TotalFifoPnl<br><small style="font-weight:400">(realized_pnl)</small></th>'
                    f'<th style="{th_s}">MTM Total<br><small style="font-weight:400">(mtm_pnl)</small></th>'
                    f'<th style="{th_s}">Diff</th>'
                    f'</tr></thead><tbody>'
                    f'{rows_html}'
                    '</tbody></table></div>'
                )
                st.markdown(html_tbl, unsafe_allow_html=True)
                st.caption(
                    "**TotalFifoPnl** (realized_pnl section): entry-price P&L including open unrealized — "
                    "matches IBKR's official period report. "
                    "**MTM Total** (mtm_pnl section): sum of daily variation-margin settlements. "
                    "Diff color: 🟢 <$200 · 🟡 <$2k · 🔴 ≥$2k. "
                    "Large differences in futures usually reflect pre-period open positions "
                    "not visible to the MTM section."
                )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with t_risk:
        if act_returns.empty:
            st.info("No active trade data found for risk metrics.")
        else:
            # ── Drawdown chart (active trading equity curve) ──────────────────
            dd_s = _drawdown_series_dollar(act_nav_s)
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_s.index, y=dd_s,
                fill="tozeroy", fillcolor="rgba(220,38,38,0.18)",
                line=dict(color="#DC2626", width=1.5),
                name="Drawdown $",
            ))
            fig_dd.update_layout(
                title=f"Drawdown ($) — {_sym_label}",
                height=260, margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
                font=dict(size=11, color="#1A202C"),
                yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#E8EDF5"),
                xaxis=dict(gridcolor="#E8EDF5"),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            # ── Rolling Sharpe + daily return distribution side by side ───────
            c_rs, c_hist = st.columns(2)
            with c_rs:
                rs30 = _rolling_sharpe(act_returns, 30)
                fig_rs = go.Figure(go.Scatter(
                    x=rs30.index, y=rs30,
                    line=dict(color="#6366F1", width=1.8),
                    name="30d Rolling Sharpe",
                    fill="tozeroy",
                    fillcolor="rgba(99,102,241,0.10)",
                ))
                fig_rs.add_hline(y=1,  line_dash="dot", line_color="#059669", opacity=0.6)
                fig_rs.add_hline(y=0,  line_dash="dot", line_color="#94A3B8", opacity=0.4)
                fig_rs.add_hline(y=-1, line_dash="dot", line_color="#DC2626", opacity=0.6)
                fig_rs.update_layout(
                    title="30-Day Rolling Sharpe (active trades only)",
                    height=280, margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
                    font=dict(size=11, color="#1A202C"),
                    yaxis=dict(gridcolor="#E8EDF5"),
                    xaxis=dict(gridcolor="#E8EDF5"),
                )
                st.plotly_chart(fig_rs, use_container_width=True)

            with c_hist:
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=_daily_pnl_nonzero.values,
                    nbinsx=50,
                    marker_color="#1E40AF", opacity=0.75,
                    name="Daily P&L",
                ))
                fig_h.add_vline(x=act_var95,  line_dash="dash",
                                line_color="#DC2626",
                                annotation_text=f"VaR 95%=${act_var95:,.0f}")
                fig_h.add_vline(x=act_cvar95, line_dash="dot",
                                line_color="#F87171",
                                annotation_text=f"CVaR=${act_cvar95:,.0f}")
                fig_h.update_layout(
                    title="Daily P&L Distribution (active trade days only)",
                    height=280, margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
                    font=dict(size=11, color="#1A202C"),
                    xaxis=dict(tickprefix="$", tickformat=",.0f", title="Daily P&L ($)", gridcolor="#E8EDF5"),
                    yaxis=dict(gridcolor="#E8EDF5"),
                )
                st.plotly_chart(fig_h, use_container_width=True)

            # ── Risk summary table ────────────────────────────────────────────
            _start = float(nav_series.iloc[0]) if not nav_series.empty else float("nan")
            ann_pnl  = float(act_returns.mean() * _ANN * _start) if math.isfinite(_start) else float("nan")
            vol_dollar = float(act_returns.std() * math.sqrt(_ANN) * _start) if math.isfinite(_start) else float("nan")
            var99_d  = _var_cvar_dollar(_daily_pnl_nonzero, 0.99)[0]
            calmar   = _calmar_dollar(ann_pnl, act_max_dd_dollar)

            th = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
                  "padding:6px 12px;text-align:center")
            td = "font-size:11px;padding:5px 12px;border-bottom:1px solid #E2E8F0;text-align:center"

            rows = [
                ("Annualised P&L (active)",  _fmt(ann_pnl,           "$", decimals=0), _colour(ann_pnl)),
                ("Annualised Volatility ($)", _fmt(vol_dollar,        "$", decimals=0), "#1A202C"),
                ("Sharpe Ratio",              _fmt(act_sharpe),                         _colour(act_sharpe)),
                ("Sortino Ratio",             _fmt(act_sortino),                        _colour(act_sortino)),
                ("Calmar Ratio",              _fmt(calmar),                             _colour(calmar)),
                ("Max Drawdown",              _fmt(act_max_dd_dollar, "$", decimals=0), "#DC2626"),
                ("VaR ~1σ (1d, 16%ile)",      _fmt(act_var1sig,       "$", decimals=0), "#DC2626"),
                ("VaR 95% (1d)",              _fmt(act_var95,         "$", decimals=0), "#DC2626"),
                ("CVaR 95% (1d)",             _fmt(act_cvar95,        "$", decimals=0), "#DC2626"),
                ("VaR 99% (1d)",              _fmt(var99_d,           "$", decimals=0), "#DC2626"),
            ]
            html = ('<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%">'
                    f'<thead><tr><th style="{th};text-align:left">Metric</th>'
                    f'<th style="{th}">Value</th></tr></thead><tbody>')
            for lbl, val, col in rows:
                html += (f'<tr><td style="{td.replace("text-align:center","text-align:left")}">'
                         f'{lbl}</td>'
                         f'<td style="{td};color:{col};font-weight:700">{val}</td></tr>')
            html += "</tbody></table></div>"
            st.markdown(html, unsafe_allow_html=True)
            st.caption(
                "Dollar metrics use daily realized P&L from active trades only. "
                "Sharpe/Sortino are dimensionless (daily P&L ÷ starting NAV). "
                "Zero-return (no-trade) days excluded from all calculations. "
                "Annualised Sharpe/Sortino show **—** when the window has too few P&L days "
                "(or too few/near-identical loss days) to estimate reliably — widen the date range for a stable figure."
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with t_trades:
        if not tstats:
            st.info("No trade data found.")
        else:
            # ── Stats summary ─────────────────────────────────────────────────
            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                _kpi("Total Trades",  str(tstats["total_trades"]))
            with s2:
                _kpi("Total P&L",
                     _fmt(rp_total_pnl, "$", decimals=0),
                     _colour(rp_total_pnl),
                     sub="realized_pnl section")
            with s3:
                _kpi("Avg Win",  _fmt(tstats["avg_win"],  "$"), "#059669")
            with s4:
                _kpi("Avg Loss", _fmt(tstats["avg_loss"], "$"), "#DC2626")
            with s5:
                _kpi("Commissions", _fmt(tstats["commissions"], "$"), "#F59E0B")

            st.markdown("")

            # ── P&L by Symbol — from realized_pnl section (accurate) ──────────
            if not rp_sym.empty:
                by_sym = rp_sym.set_index("Symbol")["TotalFifoPnl"].sort_values(ascending=False)
                colours = ["#059669" if v >= 0 else "#DC2626" for v in by_sym.values]
                fig_sym = go.Figure(go.Bar(
                    x=by_sym.index, y=by_sym.values,
                    marker_color=colours,
                    text=[f"${v:,.0f}" for v in by_sym.values],
                    textposition="outside",
                ))
                fig_sym.update_layout(
                    title=f"Total P&L by Symbol — {_sym_label} (realized_pnl section, full flex period)",
                    height=300, margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
                    font=dict(size=11, color="#1A202C"),
                    yaxis=dict(tickprefix="$", tickformat=",.0f", gridcolor="#E8EDF5"),
                    xaxis=dict(gridcolor="#E8EDF5"),
                )
                st.plotly_chart(fig_sym, use_container_width=True)

            # ── Monthly P&L (date-filtered trades, asset-class aware) ────────
            cdf = tstats["closing_df"].copy()
            if not cdf.empty and "TradeDate" in cdf.columns:
                cdf["_month"] = cdf["TradeDate"].dt.to_period("M").dt.to_timestamp()
                by_month = cdf.groupby("_month")["_net"].sum()
                fig_tm = go.Figure(go.Bar(
                    x=by_month.index.strftime("%b %Y"),
                    y=by_month.values,
                    marker_color=["#059669" if v >= 0 else "#DC2626" for v in by_month.values],
                    text=[f"${v:,.0f}" for v in by_month.values],
                    textposition="outside",
                ))
                fig_tm.update_layout(
                    title=f"Monthly P&L — {_sym_label} (date-filtered, approx for futures)",
                    height=260, margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
                    font=dict(size=11, color="#1A202C"),
                    yaxis=dict(tickprefix="$", gridcolor="#E8EDF5"),
                    xaxis=dict(gridcolor="#E8EDF5"),
                )
                st.plotly_chart(fig_tm, use_container_width=True)

            # ── Best / Worst trades table ─────────────────────────────────────
            c_best, c_worst = st.columns(2)
            show_cols = ["TradeDate", "Symbol", "Buy/Sell", "Quantity", "TradePrice", "_net"]
            col_labels = {"_net": "Net P&L"}

            def _trade_table(df_t: pd.DataFrame, title: str, colour: str):
                st.markdown(f"**{title}**")
                if df_t.empty:
                    st.caption("—")
                    return
                disp = df_t[[c for c in show_cols if c in df_t.columns]].copy()
                disp = disp.rename(columns=col_labels)
                disp["TradeDate"] = disp["TradeDate"].dt.strftime("%Y-%m-%d")
                disp["Net P&L"] = disp["Net P&L"].apply(lambda x: f"${x:,.0f}")
                if "TradePrice" in disp.columns:
                    disp["TradePrice"] = disp["TradePrice"].apply(
                        lambda x: f"{x:,.4f}" if pd.notna(x) else "—")
                st.dataframe(disp, hide_index=True, use_container_width=True)

            with c_best:
                _trade_table(
                    cdf.nlargest(10, "_net")[show_cols if all(c in cdf.columns for c in show_cols) else ["TradeDate","Symbol","_net"]],
                    "Top 10 Winning Trades", "#059669")
            with c_worst:
                _trade_table(
                    cdf.nsmallest(10, "_net")[show_cols if all(c in cdf.columns for c in show_cols) else ["TradeDate","Symbol","_net"]],
                    "Top 10 Losing Trades", "#DC2626")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with t_positions:
        if pos_df.empty:
            st.info("No open position data found.")
        else:
            # Aggregate lots → one row per symbol
            grp_cols = ["Symbol", "Description", "AssetClass", "SubCategory"]
            agg = {
                "Quantity":         "sum",
                "PositionValue":    "sum",
                "CostBasisMoney":   "sum",
                "FifoPnlUnrealized":"sum",
                "PercentOfNAV":     "sum",
            }
            valid_agg = {k: v for k, v in agg.items() if k in pos_df.columns}
            valid_grp = [c for c in grp_cols if c in pos_df.columns]
            if valid_grp:
                summary = (pos_df.groupby(valid_grp, dropna=False)
                           .agg(valid_agg)
                           .reset_index()
                           .sort_values("PositionValue", ascending=False))

                # Flag excluded
                summary["Excluded"] = summary["Symbol"].isin(excluded).map(
                    {True: "✗ Hold", False: "✓ Active"})

                # Format for display
                disp = summary.copy()
                if "PositionValue"     in disp: disp["PositionValue"]     = disp["PositionValue"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
                if "CostBasisMoney"    in disp: disp["CostBasisMoney"]    = disp["CostBasisMoney"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
                if "FifoPnlUnrealized" in disp: disp["FifoPnlUnrealized"] = disp["FifoPnlUnrealized"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
                if "PercentOfNAV"      in disp: disp["PercentOfNAV"]      = disp["PercentOfNAV"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "—")

                st.dataframe(disp, hide_index=True, use_container_width=True,
                             column_config={
                                 "PositionValue":     st.column_config.TextColumn("Mkt Value"),
                                 "CostBasisMoney":    st.column_config.TextColumn("Cost Basis"),
                                 "FifoPnlUnrealized": st.column_config.TextColumn("Unrealized P&L"),
                                 "PercentOfNAV":      st.column_config.TextColumn("% NAV"),
                             })

                # Unrealized P&L chart — active positions only (excludes long-term ETF holds)
                if "FifoPnlUnrealized" in summary.columns:
                    active = summary[~summary["Symbol"].isin(excluded)] if excluded else summary
                    upnl = active.set_index("Symbol")["FifoPnlUnrealized"].sort_values(ascending=False)
                    fig_u = go.Figure(go.Bar(
                        x=upnl.index, y=upnl.values,
                        marker_color=["#059669" if v >= 0 else "#DC2626" for v in upnl.values],
                        text=[f"${v:,.0f}" for v in upnl.values],
                        textposition="outside",
                    ))
                    fig_u.update_layout(
                        title="Unrealized P&L by Active Position (excl. long-term ETF holds)",
                        height=280, margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
                        font=dict(size=11, color="#1A202C"),
                        yaxis=dict(tickprefix="$", gridcolor="#E8EDF5"),
                        xaxis=dict(gridcolor="#E8EDF5"),
                    )
                    st.plotly_chart(fig_u, use_container_width=True)

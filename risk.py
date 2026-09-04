"""
Speculative-book risk engine for IBKR positions.

Phase 1 (this file): build the speculative book (exclude ETFs), compute
base-currency exposures and unrealised PnL, and summary metrics.
Phase 2 (next): parametric & historical 1-day VaR (95%/99%) using a blend of
implied (options_v2 surfaces) and realised (Databento/price history) vol.

Data source: pnl_db sections (fed by manual Flex upload today, or flex_web.py
automated Flex Web Service pull). Base currency is inferred from FXRateToBase
(rate that converts each position's CurrencyPrimary into the account base).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

from pnl_db import load_sections
import ibkr_conn
import risk_prices as rp
import risk_div

_RISK_SEL_PATH = os.path.join(os.path.dirname(__file__), "risk_selection.json")


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


_PRODUCTS = ["Rates", "FX", "Equities", "Commod"]


# Futures/option **roots** → product. Matched against the month/year-stripped root
# (not a loose substring), so we never false-match the "NQ" inside "OZNQ6".
_PRODUCT_BY_ROOT = {
    # US rates: Treasuries + STIRs
    "ZT": "Rates", "Z3N": "Rates", "ZF": "Rates", "ZN": "Rates", "TN": "Rates",
    "ZB": "Rates", "UB": "Rates", "TWE": "Rates",
    "ZQ": "Rates", "GE": "Rates", "SR1": "Rates", "SR3": "Rates",
    "SO1": "Rates", "SO3": "Rates", "ER": "Rates", "FF": "Rates",
    # Equity indices
    "ES": "Equities", "MES": "Equities", "NQ": "Equities", "MNQ": "Equities",
    "RTY": "Equities", "M2K": "Equities", "YM": "Equities", "MYM": "Equities",
    "EMD": "Equities", "NKD": "Equities", "DAX": "Equities", "FDAX": "Equities",
    "FESX": "Equities", "VG": "Equities",
    # Commodities
    "GC": "Commod", "MGC": "Commod", "SI": "Commod", "SIL": "Commod", "HG": "Commod",
    "PL": "Commod", "PA": "Commod", "CL": "Commod", "MCL": "Commod", "QM": "Commod",
    "BZ": "Commod", "BRN": "Commod", "NG": "Commod", "RB": "Commod", "HO": "Commod",
    "MBT": "Commod", "BTC": "Commod", "ETH": "Commod", "MET": "Commod",
    # FX futures + FX option roots (option roots ≠ future roots: EUU/JPU/GBU —
    # 2026-09-04: EUUU6 puts guessed "Rates"/US2y and were skipped)
    "6E": "FX", "M6E": "FX", "EUU": "FX", "EUR": "FX",
    "6J": "FX", "JPU": "FX", "JPY": "FX",
    "6B": "FX", "M6B": "FX", "GBU": "FX", "GBP": "FX",
    "6A": "FX", "M6A": "FX", "AUD": "FX",
    "6C": "FX", "CAD": "FX", "6S": "FX", "CHF": "FX",
}
_RATES_KW = ("SOFR", "SONIA", "ESTR", "EURIBOR", "STIR", "TREASUR", "T-NOTE", "T-BOND",
             "GILT", "BUND", "BOBL", "SCHATZ", "JGB", "FED FUND")
_EQUITY_KW = ("S&P", "SPX", "NASDAQ", "NDX", "RUSSELL", "DOW JONES", "NIKKEI",
              "EURO STOXX", "EUROSTOXX", "FTSE", "E-MINI", "E-MICRO")
_COMMOD_KW = ("GOLD", "SILVER", "COPPER", "PLATIN", "PALLAD", "CRUDE", "BRENT", "WTI",
              "GASOLINE", "HEATING OIL", "NATURAL GAS", "NAT GAS", "BITCOIN", "ETHER")


def _root_of(sym: str) -> str:
    """Contract root with option-strike suffix and month-code+year stripped:
    ZNU6→ZN, MESU6→MES, SR3M6→SR3, 'OZNQ6 C1100'→OZN."""
    s = re.split(r"\s+", str(sym).upper().strip())[0]        # drop 'C1100' strike suffix
    s = re.sub(r"[FGHJKMNQUVXZ]\d{1,2}$", "", s)             # drop month-code + year
    return s


def _guess_product(symbol, underlying="") -> str:
    """Product bucket from the symbol/underlying root; user can override in the dropdown.
    For options the *underlying* future carries the true product (OZN option → ZN → Rates)."""
    roots = []
    if underlying:
        roots.append(_root_of(underlying))
    roots.append(_root_of(symbol))
    for r in roots:
        if r in _PRODUCT_BY_ROOT:
            return _PRODUCT_BY_ROOT[r]
    blob = (str(symbol) + " " + str(underlying)).upper()
    if any(k in blob for k in _RATES_KW):
        return "Rates"
    if any(k in blob for k in _COMMOD_KW):
        return "Commod"
    if any(k in blob for k in _EQUITY_KW):
        return "Equities"
    return "Rates"


# ── Proxy assets (for the correlation / diversification calc) ─────────────────
_PROXY_RATES = ["US2y", "US5y", "US10y", "US30y",
                "EUR2y", "EUR5y", "EUR10y", "EUR30y", "UK10y", "JPY10y",
                "AUD2y", "AUD5y", "AUD10y"]
_PROXY_EQUITY = ["SPX", "Russell", "Nasdaq", "Dow"]
_PROXY_COMMOD = ["Gold", "Silver", "WTI", "Brent", "Copper", "NatGas"]


def _proxy_options(product, name, ccy=None):
    """Proxy-asset choices for a given product. FX just uses its own pair."""
    if product == "Rates":
        return _PROXY_RATES
    if product == "Equities":
        return _PROXY_EQUITY
    if product == "Commod":
        return _PROXY_COMMOD
    # FX: cash rows pass ccy (their own pair); futures/option rows (6EU6 —
    # Rajat 2026-09-05) resolve their currency via the root map. Resolved
    # currency first, the rest selectable as overrides.
    _all = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNH", "NZD"]
    _c = ccy or _guess_proxy("FX", name, name)
    if _c in _all:
        return [_c] + [x for x in _all if x != _c]
    return [_c]


def _guess_proxy(product, name, underlying=""):
    """Best-guess proxy from the symbol; user overrides in the dropdown."""
    b = (str(name) + " " + str(underlying)).upper()
    if product == "Rates":
        if any(k in b for k in ("EURIBOR", "IZ", "FES", "EUR")):
            return "EUR2y"
        if any(k in b for k in ("SONIA", "SO3", "SOA")) or b.startswith("L"):
            return "UK10y"
        if "JPY" in b or "TONA" in b:
            return "JPY10y"
        if "ZT" in b:
            return "US2y"
        if "ZF" in b:
            return "US5y"
        if "ZN" in b:
            return "US10y"
        if "ZB" in b or "UB" in b:
            return "US30y"
        return "US2y"      # SOFR / front-end default
    if product == "Equities":
        if any(k in b for k in ("MNQ", "NQ", "NASDAQ", "NDX")):
            return "Nasdaq"
        if any(k in b for k in ("M2K", "RTY", "RUSSELL")):
            return "Russell"
        if any(k in b for k in ("MYM", "YM", "DOW", "DJIA")):
            return "Dow"
        return "SPX"
    if product == "Commod":
        if any(k in b for k in ("MGC", "GC", "GOLD", "XAU")):
            return "Gold"
        if any(k in b for k in ("SI", "SILVER")):
            return "Silver"
        if any(k in b for k in ("BZ", "BRN", "BRENT")):
            return "Brent"
        if any(k in b for k in ("CL", "WTI")):
            return "WTI"
        if any(k in b for k in ("HG", "COPPER")):
            return "Copper"
        if any(k in b for k in ("NG", "NATGAS", "GAS")):
            return "NatGas"
        return "Gold"
    # FX: futures/option roots → the CURRENCY proxy. Checks underlying AND the
    # symbol (delivered "6EU6" futures carry Underlying "EUR", whose root
    # missed the old map and returned the raw symbol — Rajat 2026-09-05).
    _FX_ROOT = {"EUU": "EUR", "6E": "EUR", "M6E": "EUR", "EUR": "EUR",
                "JPU": "JPY", "6J": "JPY", "JPY": "JPY",
                "GBU": "GBP", "6B": "GBP", "M6B": "GBP", "GBP": "GBP",
                "6A": "AUD", "M6A": "AUD", "AUD": "AUD",
                "6C": "CAD", "CAD": "CAD", "6S": "CHF", "CHF": "CHF",
                "CNH": "CNH"}
    for cand in (underlying, name):
        r = _root_of(cand)
        if r in _FX_ROOT:
            return _FX_ROOT[r]
    return name


_CLASS_RANK = {"Rates": 0, "FX": 1, "Equities": 2, "Commod": 3}


def _inst_sort_key(name: str, prod: str):
    """Class-grouped, family-grouped ordering for position tables (Rajat
    2026-09-04: "EUUU6 P1162 and EUUU6 P1157 should be together ... rates in
    one group then fx then equities"). Sorts by asset class, then contract
    family (Eurex 'C OGBL 20261023 125 M' → OGBL), then expiry date token,
    then strike — so options line up beside their future, strikes in order."""
    s = str(name).strip()
    toks = s.split()
    base = toks[0]
    if base in ("C", "P") and len(toks) > 1:       # Eurex flex style
        base = toks[1]
    strike = 0.0
    m = re.findall(r"[PC](\d+(?:\.\d+)?)", s)      # CME style 'P1157'
    if m:
        strike = float(m[-1])
    else:                                          # Eurex: bare strike token
        nums = [t for t in toks[1:]
                if re.fullmatch(r"\d+(?:\.\d+)?", t) and len(t) < 8]
        if nums:
            strike = float(nums[-1])
    m8 = re.search(r"\b(\d{8})\b", s)              # Eurex expiry yyyymmdd
    exp_n = int(m8.group(1)) if m8 else 0
    return (_CLASS_RANK.get(prod, 9), base, exp_n, strike, s)


def _load_risk_selection():
    """Return (futures_set, fx_set, products, ivols, proxies, exists_bool)."""
    try:
        with open(_RISK_SEL_PATH, "r") as f:
            d = json.load(f)
        return (set(d.get("futures", [])), set(d.get("fx", [])),
                dict(d.get("products", {})), dict(d.get("ivols", {})),
                dict(d.get("proxies", {})), True)
    except Exception:
        return set(), set(), {}, {}, {}, False


def _save_risk_selection(futures, fx, products=None, ivols=None, proxies=None):
    with open(_RISK_SEL_PATH, "w") as f:
        json.dump({"futures": sorted(futures), "fx": sorted(fx),
                   "products": products or {}, "ivols": ivols or {},
                   "proxies": proxies or {}}, f, indent=2)


# ── Saved scenario sets for the 🎯 Scenario tab (Rajat 2026-09-04) ────────────
_SCN_SETS_PATH = os.path.join(os.path.dirname(__file__), "risk_scenarios.json")


def _load_scn_sets() -> dict:
    try:
        with open(_SCN_SETS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_scn_sets(d: dict) -> None:
    with open(_SCN_SETS_PATH, "w") as f:
        json.dump(d, f, indent=2)


# ── IBKR Flex pull guard (rate-limit / IP-block protection) ──────────────────
# Flex Web Service is meant for infrequent pulls; rapid repeats → a ~10-min IP
# penalty box. We persist the last-pull time (and any back-off) to a file so the
# guard survives Streamlit restarts.
_FLEX_PULL_PATH = os.path.join(os.path.dirname(__file__), "flex_last_pull.json")
_FLEX_COOLDOWN_S = 120      # minimum seconds between IBKR Flex pulls
_FLEX_BLOCK_S = 600         # back-off window when IBKR signals a rate-limit / block


def _load_flex_pull_state() -> dict:
    try:
        with open(_FLEX_PULL_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_flex_pull_state(ts=None, blocked_until=None):
    d = _load_flex_pull_state()
    if ts is not None:
        d["ts"] = ts
    if blocked_until is not None:
        d["blocked_until"] = blocked_until
    try:
        with open(_FLEX_PULL_PATH, "w") as f:
            json.dump(d, f)
    except Exception:
        pass


def _flex_pull_gate(now: float):
    """(allowed, reason). Enforce block back-off first, then the cooldown."""
    d = _load_flex_pull_state()
    bu = float(d.get("blocked_until", 0) or 0)
    last = float(d.get("ts", 0) or 0)
    if now < bu:
        return False, f"IBKR is backing off after a rate-limit — retry in {int(bu - now)}s."
    if now - last < _FLEX_COOLDOWN_S:
        return False, (f"Skipped IBKR pull (last was {int(now - last)}s ago). "
                       f"Next allowed in {int(_FLEX_COOLDOWN_S - (now - last))}s.")
    return True, ""


def _do_flex_pull():
    """Gated IBKR Flex pull. Returns (level, message); stamps success/back-off."""
    import time
    now = time.time()
    ok, why = _flex_pull_gate(now)
    if not ok:
        return "info", why
    try:
        from flex_web import update_portfolio
        with st.spinner("Fetching latest IBKR Flex statement…"):
            update_portfolio()
        _save_flex_pull_state(ts=time.time())
        return "success", "Pulled latest IBKR statement (positions + settlements)."
    except Exception as e:
        emsg = str(e).lower()
        if any(k in emsg for k in ("block", "429", "too many", "rate", "penalt", "temporarily")):
            _save_flex_pull_state(blocked_until=time.time() + _FLEX_BLOCK_S)
            return "warning", (f"IBKR signalled a rate-limit — backing off "
                               f"{_FLEX_BLOCK_S // 60} min before the next attempt. ({e})")
        return "warning", f"IBKR pull failed — {e}"

# Normal-distribution z-scores for one-tailed VaR confidence levels.
_Z = {0.95: 1.6448536269514722, 0.99: 2.3263478740408408}

# IBKR Flex flags ETFs via SubCategory == 'ETF'. Keep a manual override too.
_ETF_SUBCATEGORY = "ETF"
_LT_SUBCATEGORIES = {"ETF", "COMMON"}   # the LT Holdings book: ETFs + single stocks
_NON_SPECULATIVE_ASSET_CLASSES = {"CASH"}   # forex balances etc.

_NUM_COLS = [
    "Quantity", "MarkPrice", "PositionValue", "FifoPnlUnrealized",
    "Multiplier", "FXRateToBase", "CostBasisMoney", "Strike",
]


def load_positions() -> pd.DataFrame:
    """Latest positions snapshot from the local Flex DB."""
    return load_sections().get("positions", pd.DataFrame())


def build_speculative_book(
    positions: pd.DataFrame | None = None,
    exclude_etfs: bool = True,
    extra_exclude_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate lot-level positions into net positions per instrument, excluding
    ETFs (and cash), with base-currency exposure and unrealised PnL.

    Returns one row per instrument, sorted by gross base exposure desc.
    """
    if positions is None:
        positions = load_positions()
    if positions is None or positions.empty:
        return pd.DataFrame()

    df = positions.copy()
    for c in _NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "FXRateToBase" in df.columns:
        df["FXRateToBase"] = df["FXRateToBase"].fillna(1.0)
    else:
        df["FXRateToBase"] = 1.0

    # ── Filter to the speculative book ────────────────────────────────────────
    # The long-term book (LT Holdings tab) owns ETFs AND single stocks (SubCategory
    # COMMON, e.g. NVDA / SpaceX Cl-A bought 2026-07) — neither belongs in futures VaR.
    if exclude_etfs and "SubCategory" in df.columns:
        df = df[~df["SubCategory"].fillna("").str.upper().isin(_LT_SUBCATEGORIES)]
    if "AssetClass" in df.columns:
        df = df[~df["AssetClass"].isin(_NON_SPECULATIVE_ASSET_CLASSES)]
    if extra_exclude_symbols:
        df = df[~df["Symbol"].isin(extra_exclude_symbols)]
    if df.empty:
        return pd.DataFrame()

    # ── Base-currency values ──────────────────────────────────────────────────
    df["position_value_base"] = df["PositionValue"] * df["FXRateToBase"]
    df["upnl_base"] = df.get("FifoPnlUnrealized", 0.0) * df["FXRateToBase"]

    # ── Aggregate lots → net position per instrument ─────────────────────────
    if "Put/Call" not in df.columns:      # older Flex templates lack the field
        df["Put/Call"] = None
    key = "Conid" if "Conid" in df.columns and df["Conid"].notna().any() else "Symbol"

    def _first(s):
        return s.dropna().iloc[0] if s.notna().any() else None

    agg = (
        df.groupby(key, as_index=False)
        .agg(
            Symbol=("Symbol", _first),
            Description=("Description", _first),
            AssetClass=("AssetClass", _first),
            SubCategory=("SubCategory", _first),
            Underlying=("UnderlyingSymbol", _first),
            Currency=("CurrencyPrimary", _first),
            Exchange=("ListingExchange", _first),
            Expiry=("Expiry", _first),
            Strike=("Strike", _first),
            PutCall=("Put/Call", _first),
            Multiplier=("Multiplier", _first),
            MarkPrice=("MarkPrice", _first),
            Quantity=("Quantity", "sum"),
            position_value_base=("position_value_base", "sum"),
            upnl_base=("upnl_base", "sum"),
            FXRateToBase=("FXRateToBase", _first),
        )
    )

    agg = agg[agg["Quantity"].fillna(0) != 0].copy()
    agg["side"] = agg["Quantity"].apply(
        lambda q: "Long" if q > 0 else ("Short" if q < 0 else "Flat")
    )
    # Options (futures options FOP / equity options OPT) — flag for separate
    # handling. Eurex physically-settled futures options come through as
    # FSFOP (seen live 2026-08-24: "P OGBL … " Bund put) → match any *FOP*.
    _ac = agg["AssetClass"].fillna("").astype(str).str.upper()
    agg["is_option"] = (_ac.str.contains("FOP")
                        | _ac.isin({"OPT", "FUTOPT", "OPTFUT"}))
    agg["gross_base"] = agg["position_value_base"].abs()
    return agg.sort_values("gross_base", ascending=False).reset_index(drop=True)


def book_metrics(book: pd.DataFrame) -> dict:
    """Portfolio-level summary of the speculative book (base currency)."""
    if book is None or book.empty:
        return {}
    longs = book[book["Quantity"] > 0]
    shorts = book[book["Quantity"] < 0]
    return {
        "n_positions": int(len(book)),
        "net_base": float(book["position_value_base"].sum()),
        "gross_base": float(book["gross_base"].sum()),
        "long_base": float(longs["position_value_base"].sum()),
        "short_base": float(shorts["position_value_base"].sum()),
        "upnl_base": float(book["upnl_base"].sum()),
        "by_asset_class": book.groupby("AssetClass")["gross_base"].sum().round(0).to_dict(),
    }


# ── FX exposure ──────────────────────────────────────────────────────────────────
#
# Base currency is USD, so every non-USD cash balance is an open FX position.
# Exposure comes from the Cash Report (cash_summary.EndingCash per currency);
# the USD/unit rate is taken from the most recent fx_activity row (|Proceeds|/|Qty|,
# which is USD-per-unit since FunctionalCurrency is USD) — self-contained, no
# external calls. Realized FX PnL is summed from fx_activity.RealizedP/L (USD).

def fx_balances(sections: dict | None = None) -> dict:
    """Per-currency ending cash (native units), excluding USD and the base summary."""
    secs = sections if sections is not None else load_sections()
    cs = secs.get("cash_summary", pd.DataFrame())
    if cs.empty or "CurrencyPrimary" not in cs.columns:
        return {}
    df = cs
    if "LevelOfDetail" in cs.columns:
        df = cs[cs["LevelOfDetail"].astype(str).str.lower() == "currency"]
    out = {}
    for _, r in df.iterrows():
        ccy = str(r.get("CurrencyPrimary") or "").strip()
        if ccy in ("", "USD", "BASE_SUMMARY"):
            continue
        bal = pd.to_numeric(r.get("EndingCash"), errors="coerce")
        if pd.notna(bal) and abs(bal) > 1e-6:
            out[ccy] = float(bal)
    return out


def fx_usd_rates(sections: dict | None = None) -> dict:
    """USD per 1 unit of each FX currency, from the latest fx_activity row."""
    secs = sections if sections is not None else load_sections()
    fa = secs.get("fx_activity", pd.DataFrame())
    if fa.empty or "FXCurrency" not in fa.columns:
        return {}
    fa = fa.copy()
    fa["_q"] = pd.to_numeric(fa.get("Quantity"), errors="coerce").abs()
    fa["_p"] = pd.to_numeric(fa.get("Proceeds"), errors="coerce").abs()
    fa["_dt"] = pd.to_datetime(fa.get("DateTime"), errors="coerce")
    fa = fa[(fa["_q"] > 0) & (fa["_p"] > 0)].sort_values("_dt")
    rates = {}
    for ccy, g in fa.groupby("FXCurrency"):
        last = g.iloc[-1]
        rates[str(ccy)] = float(last["_p"] / last["_q"])
    return rates


@st.cache_data(ttl=600, show_spinner=False)
def fx_spot_quotes(currencies: tuple) -> dict:
    """{ccy: {'live': usd_per_unit_now, 'prev': usd_per_unit_prev_close}} via yfinance
    (cached 10 min). Both are USD-per-unit; prev is the last session's close."""
    import yfinance as yf

    direct = {"EUR", "GBP", "AUD", "NZD"}   # quoted CCYUSD (price already USD/unit)

    # Every candidate ticker that could be queried, known up front. The 5d-closes
    # fallback (below) only fires when fast_info is missing data — so we batch it
    # LAZILY on first need and cache within this call: zero extra round trips on
    # the common (fast_info works) path, one batch instead of N sequential
    # history() calls on the degraded path.
    _all_candidates = []
    for _ccy in currencies:
        _c = str(_ccy).upper()
        if _c in direct:
            _all_candidates.append(f"{_c}USD=X")
        _all_candidates += [f"{_c}=X", f"USD{_c}=X"]
    _cl_batch: dict = {}
    _cl_primed = [False]

    def _closes(tk: str):
        """5d Close series for tk — from the shared batch, else per-ticker."""
        if not _cl_primed[0]:
            _cl_batch.update(_batch_daily_closes(_all_candidates, period="5d"))
            _cl_primed[0] = True
        s = _cl_batch.get(tk)
        if s is None:
            s = yf.Ticker(tk).history(period="5d")["Close"].dropna()
        return s

    def _raw(tk: str):
        """Return (last, prev_close) raw pair prices, or (None, None)."""
        last = prev = None
        try:
            fi = yf.Ticker(tk).fast_info
            last = fi.get("last_price")
            prev = fi.get("previous_close")
        except Exception:
            pass
        if last is None or prev is None:
            try:
                closes = _closes(tk)
                if last is None and len(closes) >= 1:
                    last = float(closes.iloc[-1])
                if prev is None and len(closes) >= 2:
                    prev = float(closes.iloc[-2])
            except Exception:
                pass
        return (float(last) if last else None, float(prev) if prev else None)

    out = {}
    for ccy in currencies:
        c = str(ccy).upper()
        candidates = []
        if c in direct:
            candidates.append((f"{c}USD=X", False))     # USD per unit directly
        candidates += [(f"{c}=X", True), (f"USD{c}=X", True)]   # USD/CCY → invert
        for tk, invert in candidates:
            last, prev = _raw(tk)
            if last and last > 0:
                lv = (1.0 / last) if invert else last
                pv = ((1.0 / prev) if invert else prev) if (prev and prev > 0) else None
                out[ccy] = {"live": lv, "prev": pv}
                break
    return out


def fx_realized_pnl(sections: dict | None = None) -> dict:
    """Total realized FX PnL per currency (USD), summed over fx_activity."""
    secs = sections if sections is not None else load_sections()
    fa = secs.get("fx_activity", pd.DataFrame())
    if fa.empty or "RealizedP/L" not in fa.columns:
        return {}
    fa = fa.copy()
    fa["_r"] = pd.to_numeric(fa["RealizedP/L"], errors="coerce").fillna(0.0)
    return fa.groupby("FXCurrency")["_r"].sum().to_dict()


def fx_activity_span(sections: dict | None = None):
    """(start_date, end_date) covered by the accumulated fx_activity rows."""
    secs = sections if sections is not None else load_sections()
    fa = secs.get("fx_activity", pd.DataFrame())
    if fa.empty or "ReportDate" not in fa.columns:
        return (None, None)
    d = pd.to_datetime(fa["ReportDate"].astype(str), format="%Y%m%d", errors="coerce").dropna()
    if d.empty:
        return (None, None)
    return (d.min().date(), d.max().date())


def build_fx_book(sections: dict | None = None, use_live: bool = True,
                  balances: dict | None = None) -> pd.DataFrame:
    """FX positions: native balance, USD/unit rate, USD exposure, realized PnL (USD).
    Uses live spot rates where available, falling back to the last-activity rate.
    `balances` overrides the Flex cash balances (e.g. live TWS balances)."""
    secs = sections if sections is not None else load_sections()
    bal = balances if balances is not None else fx_balances(secs)
    act = fx_usd_rates(secs)
    rpnl = fx_realized_pnl(secs)
    quotes = {}
    if use_live and bal:
        try:
            quotes = fx_spot_quotes(tuple(sorted(bal)))
        except Exception:
            quotes = {}
    rows = []
    for ccy, b in bal.items():
        q = quotes.get(ccy) or {}
        lv = q.get("live")
        pv = q.get("prev")
        rate = lv if lv else act.get(ccy)
        src = "live" if lv else ("activity" if act.get(ccy) else "—")
        today_pnl = (b * (lv - pv)) if (lv is not None and pv is not None) else None
        rows.append({
            "Currency": ccy,
            "Balance": b,
            "side": "Long" if b > 0 else "Short",
            "USD_per_unit": rate,
            "prev_close": pv,
            "rate_source": src,
            "USD_exposure": (b * rate) if rate else None,
            "Today_PnL_USD": today_pnl,
            "Realized_PnL_USD": float(rpnl.get(ccy, 0.0)),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("USD_exposure", key=lambda s: s.abs(), ascending=False,
                            na_position="last").reset_index(drop=True)
    return df


# ── VaR engine ──────────────────────────────────────────────────────────────────
#
# Everything is driven by a matrix of daily *price changes* (Δmark) per instrument,
# because the book mixes rate futures (price ~96, % returns meaningless) with crypto.
# Daily P&L in base currency for instrument i on day t:
#     pnl[i,t] = Quantity_i * Multiplier_i * Δmark[i,t] * FXRateToBase_i
# From that single P&L matrix, historical and parametric VaR are just two readings
# of the same object: historical = empirical tail percentile; parametric = z·σ under
# a normal assumption (optionally with implied-vol σ overriding the realised σ while
# keeping the realised correlation structure).


def pnl_matrix(book: pd.DataFrame, dmark_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a matrix of daily Δmark (index=date, columns=Symbol, in each contract's
    own price units) into daily P&L in base currency per instrument.

    Only instruments present in both `book` and `dmark_hist` are kept.
    """
    if book is None or book.empty or dmark_hist is None or dmark_hist.empty:
        return pd.DataFrame()
    scale = {}
    for _, r in book.iterrows():
        sym = r["Symbol"]
        if sym in dmark_hist.columns:
            scale[sym] = float(r["Quantity"]) * float(r["Multiplier"]) * float(r["FXRateToBase"])
    if not scale:
        return pd.DataFrame()
    cols = list(scale)
    return dmark_hist[cols].mul(pd.Series(scale)).dropna(how="all")


def historical_var(pnl_mat: pd.DataFrame, cls=(0.95, 0.99)) -> dict:
    """Historical-simulation VaR: empirical lower-tail percentile of portfolio P&L.
    Returns {confidence: loss_amount_positive}."""
    if pnl_mat is None or pnl_mat.empty:
        return {}
    port = pnl_mat.sum(axis=1).dropna()
    return {cl: float(-np.percentile(port, (1 - cl) * 100)) for cl in cls}


def parametric_var(pnl_mat: pd.DataFrame, cls=(0.95, 0.99),
                   sigma_override: dict | None = None) -> dict:
    """
    Parametric (variance–covariance) VaR = z · σ_portfolio, mean-zero convention.
    `sigma_override` maps Symbol → daily P&L σ (base ccy) from implied vol; when
    given, it replaces the realised per-instrument σ but keeps the realised
    correlation matrix (the implied/realised blend).
    """
    if pnl_mat is None or pnl_mat.empty:
        return {}
    cov = pnl_mat.cov()
    if sigma_override:
        emp_sd = np.sqrt(np.diag(cov.values))
        emp_sd[emp_sd == 0] = np.nan
        corr = cov.values / np.outer(emp_sd, emp_sd)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        sd = np.array([float(sigma_override.get(c, emp_sd[i]))
                       for i, c in enumerate(cov.columns)])
        cov_use = corr * np.outer(sd, sd)
    else:
        cov_use = cov.values
    port_sd = float(np.sqrt(max(cov_use.sum(), 0.0)))   # 1'Σ1 (P&L series already scaled)
    return {cl: _Z[cl] * port_sd for cl in cls}


def var_breakdown(book: pd.DataFrame, dmark_hist: pd.DataFrame,
                  method: str = "historical", cl: float = 0.95,
                  sigma_override: dict | None = None) -> dict:
    """
    Full VaR report for the book at one confidence level:
      - portfolio VaR (chosen method)
      - standalone VaR per instrument
      - undiversified sum and the diversification benefit
    """
    pm = pnl_matrix(book, dmark_hist)
    if pm.empty:
        return {}

    def _one(series_df):
        if method == "parametric":
            return parametric_var(series_df, cls=(cl,), sigma_override=sigma_override)[cl]
        return historical_var(series_df, cls=(cl,))[cl]

    standalone = {c: _one(pm[[c]]) for c in pm.columns}
    port = _one(pm)
    undiv = sum(standalone.values())
    return {
        "method": method,
        "cl": cl,
        "portfolio_var": port,
        "standalone": standalone,
        "undiversified_sum": undiv,
        "diversification_benefit": undiv - port,
    }


# ── Manual-vol parametric VaR (interim, until price history is wired) ────────────
#
# Lets the user hand-enter a daily price-move (Δmark) vol per instrument and a
# correlation for same-underlying legs (e.g. the Euribor calendar spread), then
# builds the P&L covariance with *signed* position scales so a long/short spread
# offsets correctly. This is the "manual/intelligent vol" playground; it will sit
# beside the history-driven version once Databento is connected.

def default_dmark_sigma(book: pd.DataFrame) -> dict:
    """Sensible starting daily Δmark vols (in each contract's own price units)."""
    out = {}
    for _, r in book.iterrows():
        u = str(r.get("Underlying") or "").upper()
        mark = float(r["MarkPrice"])
        if 90.0 <= mark <= 101.0:                          # 3M rate future (price ~100)
            sig = 0.02                                     # ≈ 2bp/day
        elif any(k in u for k in ("BTC", "ETH", "MBT", "MET")) or u.startswith("BT"):
            sig = round(mark * 0.03, 2)                    # crypto ≈ 3%/day
        elif any(k in u for k in ("GC", "GOLD", "SI", "SILVER", "MGC")):
            sig = round(mark * 0.01, 2)                    # metals ≈ 1%/day
        else:
            sig = round(mark * 0.015, 2)                   # equity index / other ≈ 1.5%/day
        out[r["Symbol"]] = sig
    return out


def parametric_var_manual(book: pd.DataFrame, dmark_sigma: dict,
                          same_underlying_rho: float = 0.99,
                          cls=(0.95, 0.99)) -> dict:
    """Parametric VaR from manual per-instrument Δmark vols + same-underlying corr."""
    syms = list(book["Symbol"])
    scale = {r["Symbol"]: float(r["Quantity"]) * float(r["Multiplier"]) * float(r["FXRateToBase"])
             for _, r in book.iterrows()}                # signed
    und = {r["Symbol"]: (r.get("Underlying") or None) for _, r in book.iterrows()}
    sig = {s: float(dmark_sigma.get(s, 0.0)) for s in syms}

    n = len(syms)
    cov = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            sa, sb = syms[a], syms[b]
            if a == b:
                rho = 1.0
            elif und[sa] is not None and und[sa] == und[sb]:
                rho = same_underlying_rho
            else:
                rho = 0.0
            cov[a, b] = scale[sa] * scale[sb] * rho * sig[sa] * sig[sb]
    port_sd = float(np.sqrt(max(cov.sum(), 0.0)))
    sd_pnl = {s: abs(scale[s]) * sig[s] for s in syms}   # 1-sigma standalone P&L

    out = {"sigma_pnl": sd_pnl}
    for cl in cls:
        z = _Z[cl]
        standalone = {s: z * sd_pnl[s] for s in syms}
        undiv = sum(standalone.values())
        port = z * port_sd
        out[cl] = {"portfolio": port, "standalone": standalone,
                   "undiversified_sum": undiv,
                   "diversification_benefit": undiv - port}
    return out


# ── Streamlit tab ────────────────────────────────────────────────────────────────

def _pnl_color(v):
    return "#059669" if v >= 0 else "#DC2626"


def _book_table_html(book: pd.DataFrame) -> str:
    tot = book["gross_base"].sum() or 1.0
    th = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
          "padding:5px 8px;text-align:right;white-space:nowrap")
    th_l = th.replace("text-align:right", "text-align:left")
    td = "font-size:11px;padding:4px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
    td_l = td.replace("text-align:right", "text-align:left")
    header = (f"<tr><th style='{th_l}'>Symbol</th><th style='{th_l}'>Description</th>"
              f"<th style='{th}'>Side</th><th style='{th}'>Qty</th><th style='{th}'>Mark</th>"
              f"<th style='{th}'>Exposure (base)</th><th style='{th}'>uPnL</th>"
              f"<th style='{th}'>% book</th></tr>")
    rows = ""
    for _, r in book.iterrows():
        sc = _pnl_color(r["Quantity"])
        pc = _pnl_color(r["upnl_base"])
        rows += (
            f"<tr><td style='{td_l}'><b>{r['Symbol']}</b></td>"
            f"<td style='{td_l};color:#94A3B8'>{str(r['Description'] or '')[:24]}</td>"
            f"<td style='{td};color:{sc};font-weight:600'>{r['side']}</td>"
            f"<td style='{td}'>{r['Quantity']:,.0f}</td>"
            f"<td style='{td}'>{r['MarkPrice']:,.3f}</td>"
            f"<td style='{td}'>${r['position_value_base']:,.0f}</td>"
            f"<td style='{td};color:{pc}'>${r['upnl_base']:,.0f}</td>"
            f"<td style='{td}'>{r['gross_base']/tot*100:.0f}%</td></tr>"
        )
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
            f"font-family:monospace'><thead>{header}</thead><tbody>{rows}</tbody></table></div>")


def _var_table_html(book: pd.DataFrame, rep: dict) -> str:
    th = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
          "padding:5px 8px;text-align:right")
    th_l = th.replace("text-align:right", "text-align:left")
    td = "font-size:11px;padding:4px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
    td_l = td.replace("text-align:right", "text-align:left")
    header = (f"<tr><th style='{th_l}'>Symbol</th><th style='{th}'>σ P&L/day</th>"
              f"<th style='{th}'>Standalone 95%</th><th style='{th}'>Standalone 99%</th></tr>")
    rows = ""
    order = sorted(book["Symbol"], key=lambda s: -rep[0.95]["standalone"][s])
    for s in order:
        rows += (
            f"<tr><td style='{td_l}'><b>{s}</b></td>"
            f"<td style='{td}'>${rep['sigma_pnl'][s]:,.0f}</td>"
            f"<td style='{td}'>${rep[0.95]['standalone'][s]:,.0f}</td>"
            f"<td style='{td}'>${rep[0.99]['standalone'][s]:,.0f}</td></tr>"
        )
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
            f"font-family:monospace'><thead>{header}</thead><tbody>{rows}</tbody></table></div>")


def _fx_table_html(fx: pd.DataFrame) -> str:
    th = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
          "padding:5px 8px;text-align:right")
    th_l = th.replace("text-align:right", "text-align:left")
    td = "font-size:11px;padding:4px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
    td_l = td.replace("text-align:right", "text-align:left")
    header = (f"<tr><th style='{th_l}'>Currency</th><th style='{th}'>Balance</th>"
              f"<th style='{th}'>Side</th><th style='{th}'>USD/unit</th>"
              f"<th style='{th}'>USD exposure</th><th style='{th}'>Today PnL</th>"
              f"<th style='{th}'>Realised FX PnL</th></tr>")
    rows = ""
    for _, r in fx.iterrows():
        sc = _pnl_color(r["Balance"])
        pc = _pnl_color(r["Realized_PnL_USD"])
        rate = f"{r['USD_per_unit']:.6g}" if pd.notna(r["USD_per_unit"]) else "—"
        exp = f"${r['USD_exposure']:,.0f}" if pd.notna(r["USD_exposure"]) else "—"
        tp = r.get("Today_PnL_USD")
        tp_cell = (f"<span style='color:{_pnl_color(tp)}'>${tp:,.0f}</span>"
                   if pd.notna(tp) else "—")
        rows += (
            f"<tr><td style='{td_l}'><b>{r['Currency']}</b></td>"
            f"<td style='{td}'>{r['Balance']:,.0f}</td>"
            f"<td style='{td};color:{sc};font-weight:600'>{r['side']}</td>"
            f"<td style='{td}'>{rate}</td>"
            f"<td style='{td}'>{exp}</td>"
            f"<td style='{td}'>{tp_cell}</td>"
            f"<td style='{td};color:{pc}'>${r['Realized_PnL_USD']:,.0f}</td></tr>"
        )
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
            f"font-family:monospace'><thead>{header}</thead><tbody>{rows}</tbody></table></div>")


def futures_multiday_pnl(symbols, sections=None) -> dict:
    """{symbol: {1: pnl1d, 3: pnl3d, 5: pnl5d}} — sum of IBKR daily settlement P&L
    (prior_period_pnl.PriorMtmPnl × FX) over the last N business days, base USD."""
    secs = sections if sections is not None else load_sections()
    pp = secs.get("prior_period_pnl", pd.DataFrame())
    out = {s: {1: 0.0, 3: 0.0, 5: 0.0} for s in symbols}
    if pp.empty or not {"Date", "Symbol", "PriorMtmPnl"}.issubset(pp.columns):
        return out
    pp = pp.copy()
    pp["Date"] = pd.to_datetime(pp["Date"], errors="coerce")
    pp["_pnl"] = (pd.to_numeric(pp["PriorMtmPnl"], errors="coerce").fillna(0.0)
                  * pd.to_numeric(pp.get("FXRateToBase"), errors="coerce").fillna(1.0))
    for s in symbols:
        ss = pp[pp["Symbol"] == s].dropna(subset=["Date"]).sort_values("Date")
        if ss.empty:
            continue
        for n in (1, 3, 5):
            out[s][n] = float(ss["_pnl"].tail(n).sum())
    return out


def futures_pnl_asof(sections=None):
    """Date of the latest IBKR settlement in prior_period_pnl (how fresh the 1d/3d/5d
    futures PnL is). Returns a Timestamp or None."""
    secs = sections if sections is not None else load_sections()
    pp = secs.get("prior_period_pnl", pd.DataFrame())
    if pp.empty or "Date" not in pp.columns:
        return None
    d = pd.to_datetime(pp["Date"], errors="coerce").max()
    return d if pd.notna(d) else None


@st.cache_data(ttl=600, show_spinner=False)
def _fx_closes(currencies: tuple) -> dict:
    """{ccy: [USD-per-unit daily closes, most recent last]} via yfinance (~2 weeks)."""
    import yfinance as yf
    direct = {"EUR", "GBP", "AUD", "NZD"}
    start = (pd.Timestamp.today() - pd.Timedelta(days=16)).date().isoformat()
    tickers = [f"{str(ccy).upper()}USD=X" if str(ccy).upper() in direct
               else f"{str(ccy).upper()}=X" for ccy in currencies]
    batch = _batch_daily_closes(tickers, start=start)
    out = {}
    for ccy in currencies:
        c = str(ccy).upper()
        tk = f"{c}USD=X" if c in direct else f"{c}=X"
        inv = c not in direct
        try:
            h = batch.get(tk)
            if h is None:
                h = yf.Ticker(tk).history(start=start)["Close"].dropna()
            if len(h):
                out[ccy] = [(1.0 / v) if inv else float(v) for v in h.tolist()]
        except Exception:
            pass
    return out


def _ivol_txt(product, v) -> str:
    """Format implied vol: bps for Rates, % annual vol otherwise. '—' if unset."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or float(v) == 0.0:
        return "—"
    return f"{float(v):,.1f} bps" if product == "Rates" else f"{float(v):,.1f}%"


_VAR_ANNUAL_DAYS = 256   # daily vol = annual vol / sqrt(256)


def _one_day_var(product, ivol, risk):
    """1-day risk ≈ $risk × daily vol.
    Rates:  $DV01 × (annual_bps / √256).
    Others: notional × (annual_% / 100 / √256).  Returns None if vol/risk unset."""
    if (risk is None or ivol is None
            or (isinstance(risk, float) and pd.isna(risk)) or float(ivol) == 0.0):
        return None
    daily = float(ivol) / (_VAR_ANNUAL_DAYS ** 0.5)
    if product == "Rates":
        return abs(float(risk)) * daily
    return abs(float(risk)) * (daily / 100.0)


def _business_dte(expiry):
    """Business (working) days from today to expiry — Fri→Mon = 1, not 3. None if unparseable."""
    try:
        # live-snapshot json round-trip turns "20260915" into the int 20260915,
        # which pd.to_datetime reads as NANOSECONDS since epoch → 01-Jan-70
        # (Rajat 2026-09-05) — normalize to the yyyymmdd string first
        if expiry is not None and not isinstance(expiry, str) and pd.notna(expiry):
            expiry = str(int(expiry))
        exp = pd.to_datetime(expiry, errors="coerce")
        if pd.isna(exp):
            return None
        return int(np.busday_count(pd.Timestamp.today().normalize().date(), exp.date()))
    except Exception:
        return None


def _opt_prem_str(pvb, expiry) -> str:
    """'$<premium> (<business-days-to-expiry>d)' for an option row (premium = |market value|)."""
    prem = abs(float(pvb)) if pvb is not None else 0.0
    dte = _business_dte(expiry)
    return f"${prem:,.0f}" + (f" ({dte}d)" if dte is not None else "")


@st.cache_data(ttl=600, show_spinner=False)
def _underlying_levels(roots: tuple) -> dict:
    """{root: last futures level} via yfinance continuous-future tickers (~active contract)."""
    import yfinance as yf
    yfmap = {"ZT": "ZT=F", "Z3N": "ZN=F", "ZF": "ZF=F", "ZN": "ZN=F", "TN": "TN=F",
             "ZB": "ZB=F", "UB": "UB=F",
             "ES": "ES=F", "MES": "ES=F", "NQ": "NQ=F", "MNQ": "NQ=F", "RTY": "RTY=F",
             "M2K": "RTY=F", "YM": "YM=F", "MYM": "YM=F",
             "GC": "GC=F", "MGC": "GC=F", "SI": "SI=F", "HG": "HG=F", "PL": "PL=F",
             "CL": "CL=F", "MCL": "CL=F", "BZ": "BZ=F", "NG": "NG=F", "RB": "RB=F", "HO": "HO=F"}
    batch = _batch_daily_closes([yfmap[r] for r in roots if yfmap.get(r)], period="5d")
    out = {}
    for root in roots:
        tk = yfmap.get(root)
        if not tk:
            continue
        try:
            h = batch.get(tk)
            if h is None:
                h = yf.Ticker(tk).history(period="5d")["Close"].dropna()
            if len(h):
                out[root] = float(h.iloc[-1])
        except Exception:
            pass
    return out


def _options_box_html(opt_rows: list) -> str:
    """opt_rows: (name, underlying, strike, expiry_str, und_level, prem_signed, dte_business,
    pst_signed). Prem & Prem/√t are SIGNED (long +, short −); adds a Total row."""
    th = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
          "padding:5px 8px;text-align:right")
    th_l = th.replace("text-align:right", "text-align:left")
    td = "font-size:11px;padding:4px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
    td_l = td.replace("text-align:right", "text-align:left")
    header = (f"<tr><th style='{th_l}'>Option</th><th style='{th_l}'>Underlying</th>"
              f"<th style='{th}'>Strike</th><th style='{th}'>Expiry</th><th style='{th}'>Fut Level</th>"
              f"<th style='{th}'>Prem</th><th style='{th}'>Days to Exp (bus)</th>"
              f"<th style='{th}'>Prem / √t</th></tr>")

    def _sd(v, style=td):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return f"<td style='{style};color:#64748B'>—</td>"
        return f"<td style='{style};color:{_pnl_color(v)}'>{'+' if v >= 0 else '−'}${abs(v):,.0f}</td>"

    # Expiry cell: indigo urgency ramp (Rajat 2026-09-05) — darker = closer,
    # fading to pale lavender at the book's furthest expiry
    _dtes = [r[6] for r in opt_rows if r[6] is not None]
    _dmax = max(max(_dtes), 1) if _dtes else 1

    def _exp_cell(expy, dte):
        if dte is None:
            return f"<td style='{td};color:#64748B'>{expy}</td>"
        t = min(max(dte, 0) / _dmax, 1.0)
        c0, c1 = (76, 63, 191), (236, 234, 249)     # #4C3FBF → #ECEAF9
        rgb = tuple(round(a + (b - a) * t) for a, b in zip(c0, c1))
        fg = "#FFFFFF" if t < 0.45 else "#3A3468"
        return (f"<td style='{td};background:rgb{rgb};color:{fg};"
                f"font-weight:600'>{expy}</td>")

    body = ""
    tot_prem, tot_pst = 0.0, 0.0
    for name, und, strike, expy, lvl, prem, dte, pst in opt_rows:
        tot_prem += prem or 0.0
        tot_pst += pst or 0.0
        strike_txt = f"{strike:g}" if (strike is not None and not (isinstance(strike, float) and pd.isna(strike))) else "—"
        lvl_txt = f"{lvl:,.3f}" if lvl is not None else "—"
        lvl_col = td if lvl is not None else f"{td};color:#64748B"
        dte_txt = f"{dte}" if dte is not None else "—"
        body += (f"<tr><td style='{td_l}'><b>{name}</b></td>"
                 f"<td style='{td_l};color:#94A3B8'>{und or '—'}</td>"
                 f"<td style='{td}'>{strike_txt}</td>"
                 f"{_exp_cell(expy, dte)}"
                 f"<td style='{lvl_col}'>{lvl_txt}</td>"
                 f"{_sd(prem)}<td style='{td}'>{dte_txt}</td>{_sd(pst)}</tr>")

    tf = ("font-size:11px;padding:5px 8px;border-top:2px solid #475569;"
          "text-align:right;font-weight:700")
    tf_l = tf.replace("text-align:right", "text-align:left")
    body += (f"<tr><td style='{tf_l}'>Total</td><td style='{tf_l}'></td><td style='{tf}'></td>"
             f"<td style='{tf}'></td><td style='{tf}'></td>{_sd(tot_prem, tf)}"
             f"<td style='{tf}'></td>{_sd(tot_pst, tf)}</tr>")
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
            f"font-family:monospace'><thead>{header}</thead><tbody>{body}</tbody></table></div>")


def _agg_table_html(rows: list) -> str:
    """rows: (type, instrument, product, ivol, side, usd_exposure, pnl1d, pnl3d, pnl5d,
    dollar_risk, option_prem_str, lots, mark_src, mark_asof).
    mark_src ∈ {'live','stale','settled','none'} drives the Marks badge, row tint, and
    whether the PnL is shown as authoritative (live) or muted (fallback)."""
    th = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
          "padding:5px 8px;text-align:right")
    th_l = th.replace("text-align:right", "text-align:left")
    td = "font-size:11px;padding:4px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
    td_l = td.replace("text-align:right", "text-align:left")
    header = (f"<tr><th style='{th_l}'>Instrument</th><th style='{th_l}'>Type</th>"
              f"<th style='{th_l}'>Product</th><th style='{th_l}'>Marks</th>"
              f"<th style='{th}'>Implied Vol</th><th style='{th}'>1d VaR</th>"
              f"<th style='{th}'>1d PnL</th><th style='{th}'>3d PnL</th><th style='{th}'>5d PnL</th>"
              f"<th style='{th}'>Side</th><th style='{th}'>USD exposure</th><th style='{th}'>$ Risk</th>"
              f"<th style='{th}'>Option Prem</th><th style='{th}'>Lots</th></tr>")

    # Per-source styling. Intraday marks (delayed 15-min = the standard, or real-time
    # if ever available) are authoritative → coloured + counted. Everything from an
    # earlier day (stale / settled / prev-close-only) is muted grey so it can never be
    # mistaken for a current intraday PnL.
    _SRC = {
        "live":    {"bg": "#F0FDF4", "badge": "🟢 live",      "bcol": "#16A34A", "muted": False},
        "delayed": {"bg": "",        "badge": "🕒 15m",       "bcol": "#2563EB", "muted": False},
        "closed":  {"bg": "",        "badge": "🌙 closed",    "bcol": "#0891B2", "muted": False},
        "pclose":  {"bg": "#FFFBEB", "badge": "⚪ prev-close", "bcol": "#B45309", "muted": True},
        "stale":   {"bg": "#FFFBEB", "badge": "🟡 stale",     "bcol": "#B45309", "muted": True},
        "settled": {"bg": "#F1F5F9", "badge": "⚪ settled",    "bcol": "#64748B", "muted": True},
        # option PnL estimated as delta × underlying move (first order only:
        # no gamma/vega/theta) — informative but never authoritative → muted
        "dest":    {"bg": "#F1F5F9", "badge": "≈ Δ-est",      "bcol": "#64748B", "muted": True},
        "none":    {"bg": "#FEF2F2", "badge": "✖ no mark",    "bcol": "#B91C1C", "muted": True},
    }
    # live/delayed = ticking; closed = market shut, on today's close mark. All are a
    # current mark → counted in the intraday Total (closed rows just won't move).
    _INTRADAY = ("live", "delayed", "closed")
    _MUTE = "#94A3B8"   # grey for non-authoritative numbers

    def _pnl_cell(v, muted):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return f"<td style='{td};color:#CBD5E1'>—</td>"
        col = _MUTE if muted else _pnl_color(v)
        return f"<td style='{td};color:{col}'>${v:,.0f}</td>"

    def _num_cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return f"<td style='{td};color:#64748B'>—</td>"
        return f"<td style='{td}'>${v:,.0f}</td>"

    body = ""
    tvar = 0.0
    for row in sorted(rows, key=lambda x: -abs(x[9] or 0.0)):
        typ, name, product, ivol, side, exp, p1, p3, p5, risk, oprem, lots = row[:12]
        src = row[12] if len(row) > 12 else "settled"
        asof = row[13] if len(row) > 13 else ""
        style = _SRC.get(src, _SRC["settled"])
        muted = style["muted"]
        rbg = f"background:{style['bg']};" if style["bg"] else ""     # tint only the fallbacks
        badge = (f"<span style='color:{style['bcol']};font-weight:600'>{style['badge']}</span>"
                 + (f"<span style='color:#94A3B8'> {asof}</span>" if asof else ""))
        sc = _pnl_color(1 if side == "Long" else -1)
        exp_txt = f"${exp:,.0f}" if typ == "FX" else "—"
        exp_col = td if typ == "FX" else f"{td};color:#64748B"
        iv_txt = _ivol_txt(product, ivol)
        iv_col = td if iv_txt != "—" else f"{td};color:#64748B"
        var = _one_day_var(product, ivol, risk)
        tvar += var or 0.0
        prem_cell = (f"<td style='{td}'>{oprem}</td>" if oprem
                     else f"<td style='{td};color:#64748B'>—</td>")
        lots_cell = (f"<td style='{td}'>{lots:,.0f}</td>" if lots is not None
                     else f"<td style='{td};color:#64748B'>—</td>")
        body += (
            f"<tr style='{rbg}'><td style='{td_l}'><b>{name}</b></td>"
            f"<td style='{td_l};color:#94A3B8'>{typ}</td>"
            f"<td style='{td_l}'>{product or '—'}</td>"
            f"<td style='{td_l};font-size:10px'>{badge}</td>"
            f"<td style='{iv_col}'>{iv_txt}</td>{_num_cell(var)}"
            f"{_pnl_cell(p1, muted)}{_pnl_cell(p3, muted)}{_pnl_cell(p5, muted)}"
            f"<td style='{td};color:{sc};font-weight:600'>{side}</td>"
            f"<td style='{exp_col}'>{exp_txt}</td>{_num_cell(risk)}{prem_cell}{lots_cell}</tr>"
        )

    # Totals row — intraday-marked rows only (mixing current marks with stale/settled
    # figures in a total is wrong). No total for $ Risk (mixed units, non-sensical).
    _live_rows = [x for x in rows if (x[12] if len(x) > 12 else "settled") in _INTRADAY]
    t1 = sum((x[6] or 0.0) for x in _live_rows)
    t3 = sum((x[7] or 0.0) for x in _live_rows)
    t5 = sum((x[8] or 0.0) for x in _live_rows)
    # Net USD position = −(sum of non-USD exposures): short EUR/GBP/… ⇒ long
    # USD, shown positive (Rajat 2026-08-21: the total is MY USD position,
    # not the foreign-ccy sum). Exposure only meaningful for FX rows.
    tusd = -sum(x[5] for x in rows if x[0] == "FX")
    _tlabel = f"Total (intraday: {len(_live_rows)}/{len(rows)})"
    _pnl_or_dash = lambda t: (f"<span style='color:{_pnl_color(t)}'>${t:,.0f}</span>"
                              if _live_rows else "<span style='color:#CBD5E1'>—</span>")
    tf = ("font-size:11px;padding:5px 8px;border-top:2px solid #475569;"
          "text-align:right;font-weight:700")
    tf_l = tf.replace("text-align:right", "text-align:left")
    body += (
        f"<tr><td style='{tf_l}'>{_tlabel}</td><td style='{tf_l}'></td>"
        f"<td style='{tf_l}'></td><td style='{tf_l}'></td><td style='{tf}'></td>"
        f"<td style='{tf}'></td>"          # 1d VaR total blanked (undiversified sum is non-sensical)
        f"<td style='{tf}'>{_pnl_or_dash(t1)}</td>"
        f"<td style='{tf}'>{_pnl_or_dash(t3)}</td>"
        f"<td style='{tf}'>{_pnl_or_dash(t5)}</td>"
        f"<td style='{tf}'></td>"
        f"<td style='{tf}' title='net USD position: + = long USD, − = short USD'>"
        f"{'+' if tusd >= 0 else '−'}${abs(tusd):,.0f}"
        f" <span style='color:#94A3B8;font-weight:500;font-size:9px'>USD</span></td>"
        f"<td style='{tf}'></td><td style='{tf}'></td><td style='{tf}'></td></tr>"
    )
    # ── second Total incl. option Δ-estimates (Rajat 2026-08-26): grey/≈ so
    # the mixed authoritative+estimated sum is never mistaken for a mark ─────
    _dest_rows = [x for x in rows if (x[12] if len(x) > 12 else "") == "dest"]
    if _dest_rows:
        d1 = t1 + sum((x[6] or 0.0) for x in _dest_rows)
        d3 = t3 + sum((x[7] or 0.0) for x in _dest_rows)
        d5 = t5 + sum((x[8] or 0.0) for x in _dest_rows)
        tf2 = tf.replace("border-top:2px solid #475569",
                         "border-top:1px dashed #94A3B8") + ";color:#64748B"
        tf2_l = tf2.replace("text-align:right", "text-align:left")
        _est_cell = lambda t: f"≈ ${t:,.0f}"
        body += (
            f"<tr><td style='{tf2_l}'>Total incl. options (Δ-est: "
            f"{len(_dest_rows)})</td><td style='{tf2_l}'></td>"
            f"<td style='{tf2_l}'></td><td style='{tf2_l}'></td>"
            f"<td style='{tf2}'></td><td style='{tf2}'></td>"
            f"<td style='{tf2}'>{_est_cell(d1)}</td>"
            f"<td style='{tf2}'>{_est_cell(d3)}</td>"
            f"<td style='{tf2}'>{_est_cell(d5)}</td>"
            f"<td style='{tf2}'></td><td style='{tf2}'></td>"
            f"<td style='{tf2}'></td><td style='{tf2}'></td><td style='{tf2}'></td></tr>"
        )
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%;"
            f"font-family:monospace'><thead>{header}</thead><tbody>{body}</tbody></table></div>")


_AGG_INTRADAY = ("live", "delayed", "closed")   # authoritative mark flags


def _class_pnl_html(rows: list) -> str:
    """Total PnL incl. option Δ-estimates, broken by asset class (Rajat
    2026-08-26). Same inclusion rules as the table totals: intraday-marked
    rows + Δ-est option rows; stale/settled excluded. Classes containing any
    estimate are shown grey with ≈ (mixed mark quality), pure-intraday
    classes keep PnL colouring."""
    th, th_l, td, td_l, tf, tf_l = _VTH, _VTHL, _VTD, _VTDL, _VTF, _VTFL
    by_cls: dict = {}
    for x in rows:
        flag = x[12] if len(x) > 12 else "settled"
        if flag not in _AGG_INTRADAY and flag != "dest":
            continue
        c = by_cls.setdefault(x[2] or "Other",
                              {"p": [0.0, 0.0, 0.0], "est": False, "n": 0})
        for i, xi in enumerate((6, 7, 8)):
            c["p"][i] += (x[xi] or 0.0)
        c["est"] = c["est"] or flag == "dest"
        c["n"] += 1
    if not by_cls:
        return ""
    order = ["Rates", "FX", "Equities", "Commod", "Crypto"]
    classes = ([c for c in order if c in by_cls]
               + [c for c in by_cls if c not in order])
    h = (f"<tr><th style='{th_l}'>Asset class</th><th style='{th}'>1d PnL</th>"
         f"<th style='{th}'>3d PnL</th><th style='{th}'>5d PnL</th>"
         f"<th style='{th}'>rows</th></tr>")
    b = ""
    tot = [0.0, 0.0, 0.0]

    def _cell(v, est):
        if est:
            return f"<td style='{td};color:#64748B'>≈ ${v:,.0f}</td>"
        return f"<td style='{td};color:{_pnl_color(v)}'>${v:,.0f}</td>"
    any_est = False
    for cls in classes:
        c = by_cls[cls]
        any_est = any_est or c["est"]
        for i in range(3):
            tot[i] += c["p"][i]
        b += (f"<tr><td style='{td_l}'><b>{cls}</b>"
              + (" <span style='color:#94A3B8;font-size:9px'>Δ-est</span>"
                 if c["est"] else "")
              + f"</td>{_cell(c['p'][0], c['est'])}{_cell(c['p'][1], c['est'])}"
              f"{_cell(c['p'][2], c['est'])}"
              f"<td style='{td};color:#94A3B8'>{c['n']}</td></tr>")
    b += (f"<tr><td style='{tf_l}'>Total{' (incl. Δ-est)' if any_est else ''}</td>"
          f"<td style='{tf}'>{'≈ ' if any_est else ''}${tot[0]:,.0f}</td>"
          f"<td style='{tf}'>{'≈ ' if any_est else ''}${tot[1]:,.0f}</td>"
          f"<td style='{tf}'>{'≈ ' if any_est else ''}${tot[2]:,.0f}</td>"
          f"<td style='{tf}'></td></tr>")
    return (f"<div style='overflow-x:auto;max-width:560px'>"
            f"<b style='font-size:12px'>PnL by asset class · intraday + option Δ-est</b>"
            f"<table style='border-collapse:collapse;width:100%;font-family:monospace'>"
            f"<thead>{h}</thead><tbody>{b}</tbody></table></div>")


_VTH  = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
         "padding:5px 8px;text-align:right")
_VTHL = _VTH.replace("text-align:right", "text-align:left")
_VTD  = "font-size:11px;padding:4px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
_VTDL = _VTD.replace("text-align:right", "text-align:left")
_VTF  = "font-size:11px;padding:5px 8px;border-top:2px solid #475569;font-weight:800;text-align:right"
_VTFL = _VTF.replace("text-align:right", "text-align:left")


_SPLIT_ORDER = ["FX", "Equities", "Rates", "Commod", "Crypto"]


def _net_split_html(split: dict) -> str:
    """Net delta risk by asset class × underlying (Rajat 2026-08-24): per
    class a table of unique underlyings sorted by |total|, columns = linear
    Delta risk / Option Δ risk / Total. Rates in $/bp, everything else $
    notional. Generic over classes — Commod/Crypto appear when positions do."""
    th, th_l, td, td_l, tf, tf_l = _VTH, _VTHL, _VTD, _VTDL, _VTF, _VTFL
    cards = ""
    classes = ([c for c in _SPLIT_ORDER if c in split]
               + [c for c in split if c not in _SPLIT_ORDER])
    for cls in classes:
        unit = "$/bp" if cls == "Rates" else "$ notional"
        rows = sorted(split[cls].items(),
                      key=lambda kv: -abs(kv[1][0] + kv[1][1]))
        h = (f"<tr><th style='{th_l}'>Underlying</th>"
             f"<th style='{th}'>Delta risk</th>"
             f"<th style='{th}'>Option Δ risk</th>"
             f"<th style='{th}'>Total</th></tr>")
        b, t_f, t_o = "", 0.0, 0.0

        def _sv(v):
            if not v:
                return f"<span style='color:#CBD5E1'>—</span>"
            return (f"<span style='color:{_pnl_color(1 if v > 0 else -1)}'>"
                    f"{'+' if v > 0 else '−'}${abs(v):,.0f}</span>")
        for key, (fv, ov) in rows:
            t_f += fv
            t_o += ov
            tot = fv + ov
            b += (f"<tr><td style='{td_l}'><b>{key}</b></td>"
                  f"<td style='{td}'>{_sv(fv)}</td>"
                  f"<td style='{td}'>{_sv(ov)}</td>"
                  f"<td style='{td};font-weight:700'>{_sv(tot)}</td></tr>")
        if cls == "FX":
            # Rajat 2026-08-25: FX Net row in MY-USD-position terms — short
            # foreign ccys ⇒ LONG USD shows positive (per-ccy rows keep
            # their own sign, − = short that ccy). Same convention as the
            # Net USD position metric in the FX Exposure section.
            b += (f"<tr><td style='{tf_l}' title='+ = net long USD'>"
                  f"Net USD</td><td style='{tf}'>{_sv(-t_f)}</td>"
                  f"<td style='{tf}'>{_sv(-t_o)}</td>"
                  f"<td style='{tf}'>{_sv(-(t_f + t_o))}</td></tr>")
        else:
            b += (f"<tr><td style='{tf_l}'>Net</td><td style='{tf}'>{_sv(t_f)}</td>"
                  f"<td style='{tf}'>{_sv(t_o)}</td>"
                  f"<td style='{tf}'>{_sv(t_f + t_o)}</td></tr>")
        cards += (f"<div style='flex:1 1 240px;min-width:240px'>"
                  f"<b style='font-size:12px'>{cls} <span style='color:#94A3B8;"
                  f"font-weight:400'>· {unit}</span></b>"
                  f"<table style='border-collapse:collapse;width:100%;"
                  f"font-family:monospace'><thead>{h}</thead>"
                  f"<tbody>{b}</tbody></table></div>")
    note = ("<div style='font-size:10px;color:#94A3B8;padding:3px 2px'>"
            "<b>Delta risk</b> = linear positions (futures $ notional / $DV01, "
            "FX cash balances). <b>Option Δ risk</b> = options mapped to their "
            "underlying-equivalent (delta × lots × mult, × DV01 → $/bp for "
            "rates; × F → notional otherwise) — requires an options mode in "
            "the dropdown. Signs: + long / − short the underlying; the FX "
            "<b>Net USD</b> row is sign-flipped to YOUR USD position "
            "(+ = net long USD).</div>")
    return ("<b style='font-size:12px'>Net delta risk by underlying</b>"
            f"<div style='display:flex;flex-wrap:wrap;gap:18px'>{cards}</div>"
            + note)


def _opt_var_html(ors: dict) -> str:
    """Options VaR box (risk_options result) — per-position + reval total."""
    th, th_l, td, td_l, tf, tf_l = _VTH, _VTHL, _VTD, _VTDL, _VTF, _VTFL
    lbl = ("Delta-equivalent mapping" if ors["mode"] == "delta"
           else "Full-revaluation historical (price risk, vol held constant)")
    h = (f"<tr><th style='{th_l}'>Option</th><th style='{th_l}'>Mkt</th>"
         f"<th style='{th}'>Leg</th><th style='{th}'>Expiry</th>"
         f"<th style='{th}'>Lots</th><th style='{th}'>Δ-equiv</th>"
         f"<th style='{th}'>1σ $</th><th style='{th}'>VaR 95%</th>"
         f"<th style='{th}'>VaR 99%</th><th style='{th}'>obs</th></tr>")
    b = ""
    for (sym, mkt, right, K, exp, qty, de, sign, v1, v95, v99, nobs) in ors["rows"]:
        _f = lambda v: f"${v:,.0f}" if v is not None else "—"
        sc = _pnl_color(sign)
        b += (f"<tr><td style='{td_l}'><b>{sym}</b></td>"
              f"<td style='{td_l}'>{mkt}</td>"
              f"<td style='{td}'>{K:g}{right.lower()}</td>"
              f"<td style='{td}'>{exp}</td><td style='{td}'>{qty:+,.0f}</td>"
              f"<td style='{td};color:{sc}'>{de}</td>"
              f"<td style='{td}'>{_f(v1)}</td>"
              f"<td style='{td};font-weight:700'>{_f(v95)}</td>"
              f"<td style='{td}'>{_f(v99)}</td>"
              f"<td style='{td};color:#64748B'>{nobs or '—'}</td></tr>")
    tot = ors.get("total")
    if tot:
        b += (f"<tr><td style='{tf_l}' colspan='6'>Options book — summed daily "
              f"vectors ({tot['n']} days, cross-option correlation exact)</td>"
              f"<td style='{tf}'>${tot['sigma']:,.0f}</td>"
              f"<td style='{tf}'>${tot['v95']:,.0f}</td>"
              f"<td style='{tf}'>${tot['v99']:,.0f}</td><td style='{tf}'></td></tr>")
    note = ("<div style='font-size:10px;color:#94A3B8;padding:3px 2px'>"
            "Priced off the same settlement surfaces as the Pricer tab. "
            + ("<b>Δ-equiv</b> = delta × lots × mult (× DV01 → $/bp for rates); "
               "1σ = Δ-equiv × underlying vol / √256 (manual ⚙ ivol if saved "
               "for the underlying, else the leg's fitted surface IV) — linear, "
               "understates gamma near strikes/expiry." if ors["mode"] == "delta" else
               "Each structure repriced under the last ~250 daily underlying moves "
               "(sticky per-leg IVs, T fixed) — VaR from the P&L percentiles, gamma "
               "exact. <b>Vega risk is NOT included.</b>")
            + " These positions also enter the diversified √(vᵀRv) report above "
              "via their underlying's proxy.</div>")
    return (f"<div style='overflow-x:auto'><b style='font-size:12px'>Options VaR — "
            f"{lbl}</b>"
            f"<table style='border-collapse:collapse;width:100%;font-family:monospace'>"
            f"<thead>{h}</thead><tbody>{b}</tbody></table>{note}</div>")


def _win_var_html(res: dict) -> str:
    """Diversified 1-day VaR by correlation window."""
    th, th_l, td, td_l = _VTH, _VTHL, _VTD, _VTDL
    wh = (f"<tr><th style='{th_l}'>Corr window</th><th style='{th}'>days</th>"
          f"<th style='{th}'>Diversified (1σ)</th><th style='{th}'>Diversified 95%</th>"
          f"<th style='{th}'>Undiversified (1σ)</th><th style='{th}'>Div benefit</th></tr>")
    wb = ""
    for wname, obs, port, undiv, ben in res["windows"]:
        if port is None:
            wb += (f"<tr><td style='{td_l}'><b>{wname}</b></td><td style='{td}'>{obs}</td>"
                   f"<td style='{td};color:#64748B'>—</td><td style='{td};color:#64748B'>—</td>"
                   f"<td style='{td}'>${undiv:,.0f}</td><td style='{td};color:#64748B'>—</td></tr>")
            continue
        pct = (ben / undiv * 100.0) if undiv else 0.0
        wb += (f"<tr><td style='{td_l}'><b>{wname}</b></td><td style='{td}'>{obs}</td>"
               f"<td style='{td};font-weight:700'>${port:,.0f}</td>"
               f"<td style='{td}'>${port*1.645:,.0f}</td>"
               f"<td style='{td}'>${undiv:,.0f}</td>"
               f"<td style='{td};color:{_pnl_color(1)}'>${ben:,.0f} ({pct:.0f}%)</td></tr>")
    return (f"<div style='overflow-x:auto'><b style='font-size:12px'>Diversified 1-day VaR "
            f"by correlation window</b>"
            f"<table style='border-collapse:collapse;width:100%;font-family:monospace'>"
            f"<thead>{wh}</thead><tbody>{wb}</tbody></table></div>")


def _scn_multi_html(results: list) -> str:
    """Scenario P&L report (risk_scenario): results = [(label, res), …].
    Two tables — factor moves (rows = factors, cols = scenarios; inputs bold,
    implied grey) and per-position P&L (rows = positions, cols = scenarios,
    sign-coloured, total footer)."""
    th, th_l, td, td_l, tf, tf_l = _VTH, _VTHL, _VTD, _VTDL, _VTF, _VTFL
    labels = [lbl for lbl, _ in results]
    # union of factors, order = first appearance
    fmeta: dict = {}
    for _, sr in results:
        for p, is_rate, _sh, _mv in sr["factors"]:
            fmeta.setdefault(p, is_rate)
    fmap = [{p: (mv, sh) for p, _ir, sh, mv in sr["factors"]}
            for _, sr in results]
    fh = (f"<tr><th style='{th_l}'>Factor</th>"
          + "".join(f"<th style='{th}'>{l}</th>" for l in labels) + "</tr>")
    fb = ""
    for p, is_rate in fmeta.items():
        unit = "bp" if is_rate else "%"
        cells = ""
        for fm in fmap:
            if p not in fm:
                cells += f"<td style='{td};color:#64748B'>—</td>"
                continue
            mv, sh = fm[p]
            sty = "font-weight:700" if sh else "color:#64748B"
            cells += f"<td style='{td};{sty}'>{mv:+.2f}{unit}</td>"
        fb += f"<tr><td style='{td_l}'><b>{p}</b></td>{cells}</tr>"
    # by-asset-class matrix (Rajat 2026-09-04: "breakdown by asset class")
    cmeta: list = []
    for _, sr in results:
        for row in sr["rows"]:
            if row[6] not in cmeta:
                cmeta.append(row[6])
    cmap = []
    for _, sr in results:
        agg: dict = {}
        for row in sr["rows"]:
            agg[row[6]] = agg.get(row[6], 0.0) + row[5]
        cmap.append(agg)
    ch = (f"<tr><th style='{th_l}'>Asset class</th>"
          + "".join(f"<th style='{th}'>{l}</th>" for l in labels) + "</tr>")
    cb = ""
    for cls in cmeta:
        cells = ""
        for am in cmap:
            if cls not in am:
                cells += f"<td style='{td};color:#64748B'>—</td>"
                continue
            pnl = am[cls]
            cells += (f"<td style='{td};color:{_pnl_color(1 if pnl >= 0 else -1)};"
                      f"font-weight:600'>${pnl:+,.0f}</td>")
        cb += f"<tr><td style='{td_l}'><b>{cls}</b></td>{cells}</tr>"
    ctcells = ""
    for _, sr in results:
        tot = sr["total"]
        ctcells += (f"<td style='{tf};color:{_pnl_color(1 if tot >= 0 else -1)};"
                    f"font-weight:700'>${tot:+,.0f}</td>")
    cb += f"<tr><td style='{tf_l}'><b>Total</b></td>{ctcells}</tr>"

    # union of positions, class-grouped + family-grouped (Rajat 2026-09-04)
    pmeta: dict = {}
    for _, sr in results:
        for name, kind, proxy, _mv, _ir, _pnl, _prod in sr["rows"]:
            pmeta.setdefault(name, (kind, proxy, _prod))
    pmeta = dict(sorted(pmeta.items(),
                        key=lambda kv: _inst_sort_key(kv[0], kv[1][2])))
    pmap = [{name: pnl for name, _k, _x, _mv, _ir, pnl, _pr in sr["rows"]}
            for _, sr in results]
    ph = (f"<tr><th style='{th_l}'>Position</th><th style='{th}'>kind</th>"
          f"<th style='{th}'>factor</th>"
          + "".join(f"<th style='{th}'>{l}</th>" for l in labels) + "</tr>")
    pb = ""
    for name, (kind, proxy, _prod) in pmeta.items():
        cells = ""
        for pm in pmap:
            if name not in pm:
                cells += f"<td style='{td};color:#64748B'>—</td>"
                continue
            pnl = pm[name]
            cells += (f"<td style='{td};color:{_pnl_color(1 if pnl >= 0 else -1)};"
                      f"font-weight:600'>${pnl:+,.0f}</td>")
        pb += (f"<tr><td style='{td_l}'>{name}</td>"
               f"<td style='{td};color:#64748B'>{kind}</td>"
               f"<td style='{td};color:#64748B'>{proxy}</td>{cells}</tr>")
    tcells = ""
    for _, sr in results:
        tot = sr["total"]
        tcells += (f"<td style='{tf};color:{_pnl_color(1 if tot >= 0 else -1)};"
                   f"font-weight:700'>${tot:+,.0f}</td>")
    pb += (f"<tr><td style='{tf_l}'><b>Total</b></td><td style='{tf}'></td>"
           f"<td style='{tf}'></td>{tcells}</tr>")
    sr0 = results[0][1]
    cap = (f"corr window {sr0['window']} ({sr0['obs']} obs) — bold = your "
           "input, grey = correlation-implied"
           if sr0["propagate"] else "no propagation — unshocked factors flat")
    return (
        f"<div style='overflow-x:auto;margin-bottom:14px'>"
        f"<b style='font-size:12px'>Factor moves</b> "
        f"<span style='font-size:10px;color:#64748B'>({cap})</span>"
        f"<table style='border-collapse:collapse;width:100%;"
        f"font-family:monospace'><thead>{fh}</thead><tbody>{fb}</tbody>"
        f"</table></div>"
        f"<div style='overflow-x:auto;margin-bottom:14px'>"
        f"<b style='font-size:12px'>P&L by asset class</b>"
        f"<table style='border-collapse:collapse;width:100%;font-family:monospace'>"
        f"<thead>{ch}</thead><tbody>{cb}</tbody></table></div>"
        f"<div style='overflow-x:auto'><b style='font-size:12px'>"
        f"Scenario P&L — by position</b>"
        f"<table style='border-collapse:collapse;width:100%;font-family:monospace'>"
        f"<thead>{ph}</thead><tbody>{pb}</tbody></table></div>")


def _ac_var_html(res: dict, wname: str) -> str:
    """VaR by asset class for one correlation window — Component (Euler) VaR + % of book risk,
    plus each class's standalone diversified VaR, undiversified sum and net signed."""
    th, th_l, td, td_l, tf, tf_l = _VTH, _VTHL, _VTD, _VTDL, _VTF, _VTFL
    acw = res.get("by_asset_class_windows", {})
    d = acw.get(wname)
    if not d or not d.get("rows"):
        return ""
    book = d.get("book_var")
    ah = (f"<tr><th style='{th_l}'>Asset class</th>"
          f"<th style='{th}'>Component VaR</th><th style='{th}'>% of book risk</th>"
          f"<th style='{th}'>Standalone (1σ)</th><th style='{th}'>Undiv (1σ)</th>"
          f"<th style='{th}'>Net (signed)</th></tr>")
    ab = ""
    t_comp = t_sdiv = t_undv = 0.0
    for prod, comp, pct, sdiv, undv, net in d["rows"]:
        t_undv += undv
        t_sdiv += (sdiv if sdiv is not None else 0.0)
        comp_s  = f"${comp:,.0f}" if comp is not None else "—"
        pct_s   = f"{pct:.0f}%"   if pct  is not None else "—"
        sdiv_s  = f"${sdiv:,.0f}" if sdiv is not None else "—"
        nc      = _pnl_color(1 if net >= 0 else -1)
        cc      = _pnl_color(1 if (comp or 0) >= 0 else -1)
        if comp is not None:
            t_comp += comp
        ab += (f"<tr><td style='{td_l}'><b>{prod}</b></td>"
               f"<td style='{td};font-weight:700;color:{cc}'>{comp_s}</td>"
               f"<td style='{td};font-weight:700'>{pct_s}</td>"
               f"<td style='{td}'>{sdiv_s}</td>"
               f"<td style='{td}'>${undv:,.0f}</td>"
               f"<td style='{td};color:{nc}'>${net:,.0f}</td></tr>")
    book_s = f"${book:,.0f}" if book else "—"
    ab += (f"<tr><td style='{tf_l}'>Book VaR ({wname})</td>"
           f"<td style='{tf}'>{book_s}</td><td style='{tf}'>100%</td>"
           f"<td style='{tf}'>${t_sdiv:,.0f}</td><td style='{tf}'>${t_undv:,.0f}</td>"
           f"<td style='{tf}'></td></tr>")
    note = ("<div style='font-size:10px;color:#94A3B8;padding:3px 2px'>"
            "<b>Component VaR</b> = Euler risk contribution; the column <b>sums to the whole-book "
            "diversified VaR</b>, so <b>% of book risk</b> is a true risk budget (adds to 100%; a "
            "hedging class can go negative). <b>Standalone</b> = each class's own diversified VaR "
            "(not additive — Σ overstates). <b>Net</b> = signed sum (long +/short −).</div>")
    return (f"<div style='overflow-x:auto'><b style='font-size:12px'>VaR by asset class "
            f"· {wname} window</b>"
            f"<table style='border-collapse:collapse;width:100%;font-family:monospace'>"
            f"<thead>{ah}</thead><tbody>{ab}</tbody></table>{note}</div>")


def _pos_var_html(res: dict) -> str:
    """Standalone VaR by position."""
    th, th_l, td, td_l = _VTH, _VTHL, _VTD, _VTDL
    pos = res["positions"]
    ph = (f"<tr><th style='{th_l}'>Instrument</th><th style='{th_l}'>Product</th>"
          f"<th style='{th_l}'>Proxy</th><th style='{th}'>Side</th>"
          f"<th style='{th}'>Standalone 1d VaR (1σ)</th></tr>")
    pb = ""
    _p2 = pos.copy()
    _p2["_sk"] = [_inst_sort_key(n, p) for n, p in zip(_p2["name"],
                                                       _p2["product"])]
    for _, r in _p2.sort_values("_sk").iterrows():
        sc = _pnl_color(1 if r["sign"] > 0 else -1)
        pb += (f"<tr><td style='{td_l}'><b>{r['name']}</b></td>"
               f"<td style='{td_l};color:#94A3B8'>{r['product']}</td>"
               f"<td style='{td_l}'>{r['proxy']}</td>"
               f"<td style='{td};color:{sc};font-weight:600'>{'Long' if r['sign']>0 else 'Short'}</td>"
               f"<td style='{td}'>${r['var']:,.0f}</td></tr>")
    return (f"<div style='overflow-x:auto'><b style='font-size:12px'>Standalone VaR by position</b>"
            f"<table style='border-collapse:collapse;width:100%;font-family:monospace'>"
            f"<thead>{ph}</thead><tbody>{pb}</tbody></table></div>")


# ── IBKR usage telemetry tables (compact HTML, house style) ──────────────────
_UTH  = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
         "padding:5px 8px;text-align:right;white-space:nowrap")
_UTHL = _UTH.replace("text-align:right", "text-align:left")
_UTD  = ("font-size:11px;padding:4px 8px;border-bottom:1px solid #E2E8F0;"
         "text-align:right;white-space:nowrap")
_UTDL = _UTD.replace("text-align:right", "text-align:left")


def _usage_pct_color(p):
    return "#DC2626" if p >= 100 else ("#B45309" if p >= 80 else "#059669")


def _usage_today_html(tod: dict, by_tag: dict) -> str:
    """Today's counts vs soft budgets + a per-tag breakdown."""
    th, th_l, td, td_l = _UTH, _UTHL, _UTD, _UTDL
    header = (f"<tr><th style='{th_l}'>Kind</th><th style='{th}'>Today</th>"
              f"<th style='{th}'>Soft budget</th><th style='{th}'>% used</th></tr>")
    rows = ""
    for k, klabel in (("mktdata", "Quote subs"), ("hist", "Hist reqs")):
        cur = tod.get(k, 0)
        bud = tod["budgets"].get(k) or 0
        pct = tod["pct"].get(k, 0.0)
        rows += (f"<tr><td style='{td_l}'>{klabel}</td><td style='{td}'>{cur:,}</td>"
                 f"<td style='{td}'>{bud:,}</td>"
                 f"<td style='{td};color:{_usage_pct_color(pct)};font-weight:700'>{pct:.0f}%</td></tr>")
    rows += (f"<tr><td style='{td_l}'>Contract lookups</td>"
             f"<td style='{td}'>{tod.get('secdef', 0):,}</td>"
             f"<td style='{td};color:#64748B'>—</td><td style='{td};color:#64748B'>—</td></tr>")
    t1 = (f"<table style='border-collapse:collapse;font-family:monospace'>"
          f"<thead>{header}</thead><tbody>{rows}</tbody></table>")
    if by_tag:
        h2 = (f"<tr><th style='{th_l}'>Tag</th><th style='{th}'>Quote subs</th>"
              f"<th style='{th}'>Hist</th><th style='{th}'>Secdef</th></tr>")
        b2 = ""
        for tag in sorted(by_tag):
            v = by_tag[tag]
            b2 += (f"<tr><td style='{td_l}'>{tag}</td>"
                   f"<td style='{td}'>{v.get('mktdata', 0):,}</td>"
                   f"<td style='{td}'>{v.get('hist', 0):,}</td>"
                   f"<td style='{td}'>{v.get('secdef', 0):,}</td></tr>")
        t2 = (f"<div style='margin-top:8px'><table style='border-collapse:collapse;"
              f"font-family:monospace'><thead>{h2}</thead><tbody>{b2}</tbody></table></div>")
    else:
        t2 = "<div style='font-size:11px;color:#64748B;margin-top:6px'>No tagged usage yet today.</div>"
    return f"<div style='overflow-x:auto'>{t1}{t2}</div>"


def _usage_history_html(hist: list) -> str:
    """14-day per-day totals with a ⚡ marker on breakage days."""
    th, th_l, td, td_l = _UTH, _UTHL, _UTD, _UTDL
    header = (f"<tr><th style='{th_l}'>Day</th><th style='{th}'>Quote subs</th>"
              f"<th style='{th}'>Hist</th><th style='{th}'>Secdef</th>"
              f"<th style='{th_l}'>Breakage</th></tr>")
    rows = ""
    for r in hist:
        rbg = "background:#FEF2F2;" if r["breakage"] else ""
        mark = "⚡" if r["breakage"] else ""
        rows += (f"<tr style='{rbg}'><td style='{td_l}'>{r['day']}</td>"
                 f"<td style='{td}'>{r['mktdata']:,}</td><td style='{td}'>{r['hist']:,}</td>"
                 f"<td style='{td}'>{r['secdef']:,}</td><td style='{td_l}'>{mark}</td></tr>")
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;"
            f"font-family:monospace'><thead>{header}</thead><tbody>{rows}</tbody></table></div>")


# ── Intraday trades P&L (live mode) ──────────────────────────────────────────
# The tab's live 1d PnL marks every live position from PRIOR CLOSE — correct for a
# lot carried from yesterday, WRONG for anything traded today (a lot bought this
# morning is P&L'd from yesterday's close; a round-trip closed intraday is invisible).
# These helpers decompose the day for symbols traded today:
#     Total day = Carried + Intraday trades − Commissions
#     Carried(sym)  = (live_qty − Σ fill_qty) × (mark − prior_close) × mult × fx
#     Trades(sym)   = Σ_fills  fill_qty × (mark − fill_price)       × mult × fx
# The marks/prior_closes are the SAME ones the aggregated table shows (futures: the
# risk_prices live/settlement cache; options: the EOD-settled mark), so the box
# reconciles with the table above. All pure functions → unit-testable off-line.

def _clean_fill_records(recs) -> list:
    """Make fill dicts JSON-serializable (numpy scalars → py, NaN → None, ts → iso)."""
    def _c(v):
        if v is None:
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            f = float(v)
            return f if f == f else None
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, float):
            return v if v == v else None
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return v
    return [{k: _c(v) for k, v in dict(r).items()} for r in (recs or [])]


def _save_live_fills(fills_recs) -> None:
    """Additively persist today's fills into the live snapshot json under a 'fills' key.
    Never touches rp.save_live_snapshot — call AFTER it has written book/fxbal/ts."""
    path = rp._LIVE_SNAP_PATH
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        d = {}
    d["fills"] = _clean_fill_records(fills_recs)
    try:
        with open(path, "w") as f:
            json.dump(d, f)
    except Exception:
        pass


def _load_live_fills() -> list:
    """Fills persisted alongside the live snapshot (empty list if none / unreadable)."""
    try:
        with open(rp._LIVE_SNAP_PATH) as f:
            d = json.load(f)
        return list(d.get("fills", []))
    except Exception:
        return []


def _intraday_marks(book, price_cache, eod_mark, today_iso):
    """(marks, prior_closes) per Symbol, matching the aggregated table's 1d column.

    Futures — replicate live_multiday_pnl's anchor logic on the risk_prices cache:
      · open/delayed tick → mark = live intraday, prior_close = latest close < today
      · closed/prev-close → mark = latest settlement, prior_close = the close before it
      falls back to the book's MarkPrice for the mark when the cache has nothing.
    Options — the EOD-settled mark (same policy the Options-Prem box uses); options'
      1d PnL is Flex-settled, not a (mark − close) formula, so prior_close is left None."""
    marks, prior_closes = {}, {}
    if book is None or book.empty:
        return marks, prior_closes
    for _, r in book.iterrows():
        sym = str(r.get("Symbol"))
        mp = r.get("MarkPrice")
        mp = float(mp) if (mp is not None and not (isinstance(mp, float) and pd.isna(mp))) else None
        if bool(r.get("is_option")):
            m = eod_mark.get(sym)
            marks[sym] = float(m) if m is not None else mp
            prior_closes[sym] = None
            continue
        ent = price_cache.get(sym) or {}
        live = ent.get("live")
        closes = ent.get("closes", {}) or {}
        lsrc = ent.get("live_src")
        mark = pc = None
        if live is not None and closes:
            sc = sorted(closes.items())                       # (date, close) oldest→newest
            if lsrc in ("closed", "prev-close"):
                anchor_d, anchor_c = sc[-1]
                mark = anchor_c
                vals = [c for d, c in sc if d < anchor_d]
            else:
                mark = live
                vals = [c for d, c in sc if d < today_iso]
            pc = vals[-1] if vals else None
        marks[sym] = float(mark) if mark is not None else mp
        prior_closes[sym] = float(pc) if pc is not None else None
    return marks, prior_closes


def _fx_fills_rows(cash_fills, fxq, balances) -> tuple:
    """Rows for spot-FX (IDEALPRO) fills — the FULL day decomposition per currency
    (Rajat 2026-07-30: 'SOD level needs to be compared vs the fill'):
      day P&L(ccy) = SOD_balance × (live − SOD rate)  ← carried leg
                   + Σ trade_deltas × (live − fill rate)  ← trades leg
    fxq = {ccy: {'live': usd_per_unit_now, 'prev': usd_per_unit_prior_close}} (both
    USD-per-unit, from fx_spot_quotes); balances = {ccy: CURRENT balance} (live TWS);
    SOD balance is reconstructed as current − today's fill deltas. Pure/testable.
    Returns (rows, trades_total_usd, carried_total_usd, commissions_as_is, skipped)."""
    from collections import OrderedDict
    groups: "OrderedDict" = OrderedDict()
    deltas: dict = {}          # ccy → net balance change from today's fills
    for f in (cash_fills or []):
        pair = str(f.get("Symbol") or "")
        groups.setdefault(pair, []).append(f)
        try:
            base, quote = [x.upper() for x in pair.split(".")]
            q, px = float(f.get("Qty") or 0.0), float(f.get("Price") or 0.0)
            deltas[base] = deltas.get(base, 0.0) + q
            deltas[quote] = deltas.get(quote, 0.0) - q * px
        except Exception:
            pass

    def _u(ccy, key="live"):
        if ccy == "USD":
            return 1.0
        d = fxq.get(ccy) or {}
        v = d.get(key)
        return float(v) if v else None

    rows, t_total, c_total, comm, skipped = [], 0.0, 0.0, 0.0, []
    for pair, fs in groups.items():
        comm += sum(abs(float(f.get("Commission") or 0.0)) for f in fs)  # IDEALPRO: USD
        try:
            base, quote = [x.upper() for x in pair.split(".")]
            u_b, u_q = _u(base), _u(quote)
            cross = u_b / u_q
        except Exception:
            skipped.append(pair)
            continue
        net = wnum = gross = pnl_q = 0.0
        for f in fs:
            q, px = float(f.get("Qty") or 0.0), float(f.get("Price") or 0.0)
            net += q
            gross += abs(q)
            wnum += abs(q) * px
            pnl_q += q * (cross - px)
        pnl_usd = pnl_q * u_q
        rows.append({
            "Symbol": pair, "n_fills": len(fs), "net_qty": net,
            "avg_px": (wnum / gross) if gross else None, "trades_pnl": pnl_usd,
            "carried_qty": None, "carried_pnl": None, "mark": cross,
            "prior_close": None, "mult": 1.0, "fx": u_q, "live_qty": None,
        })
        t_total += pnl_usd

    # Carried leg: SOD balance of each TRADED currency × (live − SOD USD-per-unit).
    # Untouched currencies stay with the FX section (avoids double-display).
    for ccy, delta in deltas.items():
        if ccy == "USD":
            continue        # base currency of the account — no FX P&L on USD itself
        u_live, u_prev = _u(ccy, "live"), _u(ccy, "prev")
        if u_live is None or u_prev is None:
            skipped.append(f"{ccy} (no SOD rate)")
            continue
        sod_bal = float((balances or {}).get(ccy, 0.0)) - delta
        if abs(sod_bal) < 1.0:
            continue
        carried = sod_bal * (u_live - u_prev)
        rows.append({
            "Symbol": f"{ccy} bal (SOD)", "n_fills": 0, "net_qty": None,
            "avg_px": None, "trades_pnl": None, "carried_qty": sod_bal,
            "carried_pnl": carried, "mark": u_live, "prior_close": u_prev,
            "mult": 1.0, "fx": 1.0, "live_qty": None,
        })
        c_total += carried
    return rows, t_total, c_total, comm, skipped


def _intraday_decomposition(fills, book, marks, prior_closes) -> dict:
    """Decompose today's P&L for traded symbols. Pure — takes already-resolved marks &
    prior_closes so it is trivially unit-testable.

    fills: list of dicts (Conid, Symbol, SecType, Qty[signed], Price, Time, Commission, ExecId).
    Matches each fill to a book row by Conid (fallback Symbol) to read Multiplier /
    FXRateToBase; fills whose contract isn't in the book are skipped and noted (their
    commission is still summed, as-is since the currency is then unknown). Commissions on
    matched fills are converted to base via that row's FXRateToBase."""
    by_conid, by_symbol = {}, {}
    if book is not None and not book.empty:
        for _, r in book.iterrows():
            conid = r.get("Conid")
            if conid is not None and not (isinstance(conid, float) and pd.isna(conid)):
                try:
                    by_conid[int(conid)] = r
                except Exception:
                    pass
            by_symbol[str(r.get("Symbol"))] = r

    from collections import OrderedDict
    groups: "OrderedDict" = OrderedDict()
    unmatched, commissions = [], 0.0
    _flat_cands: dict = {}
    # currency → FXRateToBase map from the book, for flat round-trips in non-USD contracts
    _ccy_fx = {}
    if book is not None and not book.empty and "Currency" in book.columns:
        for _, _r in book.iterrows():
            c_ = str(_r.get("Currency") or "").upper()
            if c_ and c_ not in _ccy_fx:
                _ccy_fx[c_] = float(_r.get("FXRateToBase") or 1.0)
    for f in (fills or []):
        sym = str(f.get("Symbol"))
        row = None
        conid = f.get("Conid")
        if conid is not None:
            try:
                row = by_conid.get(int(conid))
            except Exception:
                row = None
        if row is None:
            row = by_symbol.get(sym)
        comm = abs(float(f.get("Commission") or 0.0))
        if row is None:
            # Not in the live book — the important case is a round-trip closed FLAT by
            # snapshot time (flat positions vanish from the book, but a completed
            # day-trade is exactly what this box exists for). Its P&L is pure realized
            # (mark-independent), and the fill carries its own Multiplier/Currency, so
            # it can be computed with no book row at all — collect for a second pass.
            _flat_cands.setdefault(sym, []).append(f)
            continue
        commissions += comm * float(row.get("FXRateToBase") or 1.0)   # instrument ccy → base
        groups.setdefault(sym, {"row": row, "fills": []})["fills"].append(f)

    rows, total_trades, total_carried, carried_missing = [], 0.0, 0.0, []
    for sym, g in groups.items():
        row = g["row"]
        mult = float(row.get("Multiplier") or 1.0)
        fx = float(row.get("FXRateToBase") or 1.0)
        live_qty = float(row.get("Quantity") or 0.0)
        mark = marks.get(sym)
        mark = float(mark) if mark is not None else None
        net_qty = gross_abs = wnum = tpnl = 0.0
        for f in g["fills"]:
            q = float(f.get("Qty") or 0.0)
            px = float(f.get("Price") or 0.0)
            net_qty += q
            gross_abs += abs(q)
            wnum += abs(q) * px
            if mark is not None:
                tpnl += q * (mark - px) * mult * fx
        avg_px = (wnum / gross_abs) if gross_abs else None
        trades_pnl = tpnl if mark is not None else None
        pc = prior_closes.get(sym)
        pc = float(pc) if pc is not None else None
        carried_qty = live_qty - net_qty
        if mark is not None and pc is not None:
            carried_pnl = carried_qty * (mark - pc) * mult * fx
        else:
            carried_pnl = None
            if abs(carried_qty) > 1e-9:
                carried_missing.append(sym)
        rows.append({
            "Symbol": sym, "n_fills": len(g["fills"]), "net_qty": net_qty,
            "avg_px": avg_px, "trades_pnl": trades_pnl, "carried_qty": carried_qty,
            "carried_pnl": carried_pnl, "mark": mark, "prior_close": pc,
            "mult": mult, "fx": fx, "live_qty": live_qty,
        })
        total_trades += trades_pnl or 0.0
        total_carried += carried_pnl or 0.0

    # Second pass — contracts absent from the live book. FLAT round-trips are fully
    # computable (realized-only: Σq=0 makes the mark cancel out; use the fills' own
    # Multiplier/Currency). Non-flat unmatched fills (anomaly) keep the old skip+note.
    for sym, fs in _flat_cands.items():
        net = sum(float(f.get("Qty") or 0.0) for f in fs)
        comm_sum = sum(abs(float(f.get("Commission") or 0.0)) for f in fs)
        if abs(net) > 1e-9:
            unmatched.append(sym)
            commissions += comm_sum                          # currency unknown → as-is
            continue
        mult = float(fs[0].get("Multiplier") or 1.0)
        ccy = str(fs[0].get("Currency") or "USD").upper()
        fx = _ccy_fx.get(ccy, 1.0)
        commissions += comm_sum * fx
        realized = -sum(float(f.get("Qty") or 0.0) * float(f.get("Price") or 0.0)
                        for f in fs) * mult * fx
        gross_abs = sum(abs(float(f.get("Qty") or 0.0)) for f in fs)
        wnum = sum(abs(float(f.get("Qty") or 0.0)) * float(f.get("Price") or 0.0) for f in fs)
        rows.append({
            "Symbol": sym, "n_fills": len(fs), "net_qty": 0.0,
            "avg_px": (wnum / gross_abs) if gross_abs else None,
            "trades_pnl": realized, "carried_qty": 0.0, "carried_pnl": None,
            "mark": None, "prior_close": None, "mult": mult, "fx": fx, "live_qty": 0.0,
        })
        total_trades += realized

    return {
        "rows": rows,
        "carried_pnl": total_carried,
        "trades_pnl": total_trades,
        "commissions": commissions,
        "total_day": total_carried + total_trades - commissions,
        "unmatched": unmatched,
        "carried_missing": carried_missing,
    }


def _intraday_box_html(decomp: dict) -> str:
    """Compact per-symbol fills table + the Carried/Intraday/Commissions/Total decomposition.
    House style (usage-table consts: compact, nowrap, no width:100%)."""
    th, th_l, td, td_l = _UTH, _UTHL, _UTD, _UTDL
    tf = ("font-size:11px;padding:5px 8px;border-top:2px solid #475569;"
          "text-align:right;font-weight:700;white-space:nowrap")
    tf_l = tf.replace("text-align:right", "text-align:left")
    header = (f"<tr><th style='{th_l}'>Symbol</th><th style='{th}'>fills</th>"
              f"<th style='{th}'>net qty traded</th><th style='{th}'>avg px</th>"
              f"<th style='{th}'>Trades P&L</th></tr>")
    body = ""
    for r in sorted(decomp["rows"], key=lambda x: -abs(x.get("trades_pnl") or 0.0)):
        tp = r.get("trades_pnl")
        tp_cell = (f"<td style='{td};color:{_pnl_color(tp)}'>${tp:,.0f}</td>"
                   if tp is not None else f"<td style='{td};color:#64748B'>—</td>")
        avg = r.get("avg_px")
        avg_txt = f"{avg:,.4f}" if avg is not None else "—"
        # FX SOD-carried rows have no fills/net-qty/avg — show the carried P&L in the
        # Trades column slot with a "carried" tag, and "—" for the trade-only fields.
        nq = r.get("net_qty")
        nq_txt = f"{nq:,.0f}" if nq is not None else "—"
        cp = r.get("carried_pnl")
        if tp is None and cp is not None:
            tp_cell = (f"<td style='{td};color:{_pnl_color(cp)}'>${cp:,.0f}"
                       f" <span style='color:#94A3B8;font-weight:400'>(carried)</span></td>")
        nf = r.get("n_fills") or 0
        body += (f"<tr><td style='{td_l}'><b>{r['Symbol']}</b></td>"
                 f"<td style='{td}'>{nf if nf else '—'}</td>"
                 f"<td style='{td}'>{nq_txt}</td>"
                 f"<td style='{td}'>{avg_txt}</td>{tp_cell}</tr>")
    _tt = decomp["trades_pnl"]
    body += (f"<tr><td style='{tf_l}'>Total trades</td><td style='{tf}'></td>"
             f"<td style='{tf}'></td><td style='{tf}'></td>"
             f"<td style='{tf};color:{_pnl_color(_tt)}'>${_tt:,.0f}</td></tr>")
    table = (f"<table style='border-collapse:collapse;font-family:monospace'>"
             f"<thead>{header}</thead><tbody>{body}</tbody></table>")

    car, tra = decomp["carried_pnl"], decomp["trades_pnl"]
    com, tot = decomp["commissions"], decomp["total_day"]

    def _line(label, v, color_val=None, strong=False):
        col = _pnl_color(color_val if color_val is not None else v)
        wt = "700" if strong else "600"
        return (f"<div style='display:flex;justify-content:space-between;gap:24px;"
                f"font-size:12px;font-family:monospace;padding:2px 2px'>"
                f"<span{' style=font-weight:700' if strong else ''}>{label}</span>"
                f"<span style='color:{col};font-weight:{wt}'>${v:,.0f}</span></div>")

    decomp_html = (
        "<div style='margin-top:8px;max-width:380px'>"
        + _line("Carried P&L (held × Δ vs prior close)", car)
        + _line("Intraday trades P&L (fills × Δ vs fill px)", tra)
        + _line("Commissions", -com, color_val=-1)      # a cost → shown negative & red
        + "<div style='border-top:2px solid #475569;margin-top:2px'></div>"
        + _line("Total day", tot, strong=True)
        + "</div>"
    )
    return f"<div style='overflow-x:auto'>{table}{decomp_html}</div>"


@st.fragment
def _scenario_inputs():
    """(book, fx, eff_fut, eff_fx, products, ivols, proxies, src_txt) — the
    same positions source + saved-selection resolution render_risk uses, minus
    the refresh buttons (the Risk / VaR tab owns pulling; this is read-only).
    Option rows' position_value_base is left as-is: the scenario engine
    reprices options off the surfaces and never reads their premium."""
    raw_pos = load_positions()
    _live_book = st.session_state.get("_risk_live_book")
    if _live_book is None:                  # restore persisted LIVE snapshot
        _lb, _lfb, _lts = rp.load_live_snapshot()
        if _lb is not None and not _lb.empty:
            _live_book = _lb
            st.session_state["_risk_live_book"] = _lb
            st.session_state["_risk_live_fxbal"] = _lfb
            st.session_state["_risk_live_ts"] = _lts
    _is_live = (_live_book is not None and hasattr(_live_book, "empty")
                and not _live_book.empty)
    if _is_live:
        book = _live_book
        _lfb = st.session_state.get("_risk_live_fxbal") or None
        fx = build_fx_book(balances=_lfb) if _lfb else build_fx_book()
        src = "📡 LIVE snapshot"
    else:
        book = build_speculative_book(raw_pos)
        fx = build_fx_book()
        src = "Flex EOD"
    (saved_fut, saved_fx, products, ivols, proxies,
     saved_exists) = _load_risk_selection()
    if saved_exists:
        eff_fut, eff_fx = set(saved_fut), set(saved_fx)
    else:
        eff_fut = set(book["Symbol"]) if not book.empty else set()
        eff_fx = set(fx["Currency"]) if not fx.empty else set()
    if _is_live and not book.empty:
        eff_fut = set(book["Symbol"])       # live book: show ALL live positions
    return (book, fx, eff_fut, eff_fx, dict(products), dict(ivols),
            dict(proxies), src)


def render_scenario():
    """🎯 Scenario sub-tab (Rajat 2026-09-04, NFP day): shock a few anchor
    factors, propagate to the rest via the VaR correlation framework,
    full-reval the options — engine in risk_scenario.py."""
    import risk_scenario
    st.markdown("#### Scenario P&L — shock the book")
    (book, fx, eff_fut, eff_fx, eff_products, eff_ivols, eff_proxies,
     _src) = _scenario_inputs()
    st.caption(
        f"Positions: **{_src}** · products/vols/proxies from the Risk / VaR "
        "tab's saved params. Enter shocks only for the factors you have a "
        "view on — equities/FX/commod in **%** (+ = up; FX is USD-per-unit, "
        "so JPY +1% = yen STRONGER), rates in **bp of yield** (+ = yields "
        "up). Unshocked factors take their correlation-implied conditional "
        "move over the chosen window (untick to hold them flat). Futures/FX "
        "map linearly (same conventions as the VaR); options are FULLY "
        "REPRICED off the settlement surfaces at the shifted underlying — "
        "IV sticky, so vol moves are NOT captured.")
    _scf = risk_scenario.factor_universe(
        book, fx, set(eff_fut), set(eff_fx), eff_products, eff_proxies)
    if not _scf and book.empty and fx.empty:
        st.warning("No positions — pull the book in the Risk / VaR tab first.")
    else:
        # Factor picker (Rajat 2026-09-04: "let me set the factors and you
        # back out the others using correl") — default his anchor five; any
        # proxy is shockable even with no position mapped to it (US10y).
        # apply a pending Load BEFORE any _rsc widget instantiates (session
        # state can't be written for a widget already created this run)
        _pend = st.session_state.pop("_rsc_pending_load", None)
        if _pend:
            for _k in [k for k in st.session_state
                       if isinstance(k, str) and k.startswith("_rsc_g_")]:
                st.session_state[_k] = 0.0
            st.session_state["_rsc_facs"] = list(_pend.get("factors", []))
            for _j, _col in enumerate(_pend.get("shocks", [])[:4]):
                for _p, _v in _col.items():
                    st.session_state[f"_rsc_g_{_p}_{_j}"] = float(_v)
            st.session_state["_rsc_prop"] = bool(_pend.get("propagate", True))
            if _pend.get("window") in risk_div.WINDOWS:
                st.session_state["_rsc_win"] = _pend["window"]
            st.session_state["_rsc_evw"] = float(_pend.get("event_weight", 3.0))
            _pev = _pend.get("event", "none")
            if _pev == "none" or _pev in risk_scenario.load_events():
                st.session_state["_rsc_ev"] = _pev
        _bookf = [p for p, _ in _scf]
        _allf = list(dict.fromkeys(
            _bookf + list(risk_div._RATE_FETCH) + list(risk_div._YF)
            + list(risk_div._YF_INV)))
        _DEF = [p for p in ("US2y", "US10y", "EUR2y", "EUR", "SPX")
                if p in _allf]
        _facs = st.multiselect(
            "factors to set — everything else is backed out via correlations",
            _allf, key="_rsc_facs",
            **({} if "_rsc_facs" in st.session_state else {"default": _DEF}))
        # grid: rows = factors, columns = Scenario 1-4
        _NS = 4
        _hd = st.columns([1.1] + [1.0] * _NS)
        _hd[0].markdown("**Factor**")
        for _j in range(_NS):
            _hd[_j + 1].markdown(f"**Scenario {_j + 1}**")
        for _p in _facs:
            _ir = risk_scenario._is_rate_factor(_p)
            _cc = st.columns([1.1] + [1.0] * _NS)
            # USD-per-unit convention: for pairs conventionally quoted the
            # other way (JPY/CHF/CAD/CNH) spell out the direction on the row
            if _p in risk_div._YF_INV:
                _cc[0].markdown(f"`{_p}` (%, + = {_p}↑ = USD{_p}↓)")
            else:
                _cc[0].markdown(f"`{_p}` ({'bp' if _ir else '%'})")
            for _j in range(_NS):
                _gk = f"_rsc_g_{_p}_{_j}"
                _cc[_j + 1].number_input(
                    f"{_p} scenario {_j + 1}",
                    step=1.0 if _ir else 0.25, format="%.2f",
                    key=_gk, label_visibility="collapsed",
                    **({} if _gk in st.session_state else {"value": 0.0}))
        _evreg = risk_scenario.load_events()
        _evopts = ["none"] + sorted(_evreg)
        _sc1, _sc2, _scv, _sce, _sc3 = st.columns([1.15, 0.75, 0.85, 0.7, 0.9])
        _prop = _sc1.checkbox(
            "propagate via correlations", key="_rsc_prop",
            **({} if "_rsc_prop" in st.session_state else {"value": True}))
        _swin = _sc2.selectbox(
            "corr window", list(risk_div.WINDOWS), key="_rsc_win",
            **({} if "_rsc_win" in st.session_state else {"index": 2}))
        # default today's event: NFP on a first-Friday, FOMC on a listed date
        _tdy = pd.Timestamp.today().normalize()
        _evdef = "none"
        for _en, _ec in _evreg.items():
            if bool(risk_scenario._event_mask(
                    pd.DatetimeIndex([_tdy]), _ec)[0]):
                _evdef = _en
                break
        _evsel = _scv.selectbox(
            "event today", _evopts, key="_rsc_ev",
            help="Blends correlations AND vol multiples toward this event's "
                 "historical event-day behaviour. Registry: risk_events.json "
                 "(add events/dates there — or ask Claude).",
            **({} if "_rsc_ev" in st.session_state
               else {"index": _evopts.index(_evdef)}))
        _evw = _sce.number_input(
            "worth ×N days", min_value=1.0, max_value=10.0, step=1.0,
            key="_rsc_evw",
            help="Shrinkage weight 1−1/N toward the event-day estimates "
                 "(N=1 → plain history).",
            **({} if "_rsc_evw" in st.session_state else
               {"value": float((_evreg.get(_evdef) or {}).get("weight", 3))}))
        _sc3.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if _sc3.button("🎯 Run scenarios", key="_rsc_run"):
            _scns = []
            for _j in range(_NS):
                _shk = {p: st.session_state.get(f"_rsc_g_{p}_{_j}", 0.0)
                        for p in _facs}
                _shk = {p: v for p, v in _shk.items() if v}
                if _shk:
                    _scns.append((f"Scenario {_j + 1}", _shk))
            if not _scns:
                st.warning("Set at least one nonzero shock in some column.")
            else:
                try:
                    _fred_s = st.secrets.get("FRED_KEY")
                except Exception:
                    _fred_s = None
                _out = []
                with st.spinner(f"Running {len(_scns)} scenario(s) — "
                                "propagating shocks & repricing options…"):
                    for _lbl, _shk in _scns:
                        _out.append((_lbl, risk_scenario.compute(
                            book, fx, set(eff_fut), set(eff_fx), eff_products,
                            eff_ivols, eff_proxies, _shk, _fred_s,
                            propagate=_prop, window=_swin,
                            event=(None if _evsel == "none" else _evsel),
                            event_weight=_evw)))
                st.session_state["_risk_scn_multi"] = _out
        # ── saved scenario sets (persisted to risk_scenarios.json) ───────────
        _sets = _load_scn_sets()
        _sv = st.columns([1.5, 0.55, 0.45, 1.5, 0.55])
        _pick = _sv[0].selectbox(
            "saved sets", ["—"] + sorted(_sets), key="_rsc_set_pick",
            label_visibility="collapsed",
            help="Pick a saved scenario set, then 📂 Load")
        if _sv[1].button("📂 Load", key="_rsc_set_load",
                         use_container_width=True) and _pick != "—":
            st.session_state["_rsc_pending_load"] = _sets[_pick]
            st.rerun()
        if _sv[2].button("🗑️", key="_rsc_set_del", use_container_width=True,
                         help="Delete the selected set") and _pick != "—":
            _sets.pop(_pick, None)
            _save_scn_sets(_sets)
            st.rerun()
        _sname = _sv[3].text_input(
            "set name", key="_rsc_set_name", placeholder="name it — e.g. NFP day",
            label_visibility="collapsed")
        if _sv[4].button("💾 Save", key="_rsc_set_save",
                         use_container_width=True):
            if not _sname.strip():
                st.warning("Give the set a name first.")
            else:
                _shocks = []
                for _j in range(_NS):
                    _col = {p: st.session_state.get(f"_rsc_g_{p}_{_j}", 0.0)
                            for p in _facs}
                    _shocks.append({p: v for p, v in _col.items() if v})
                _sets[_sname.strip()] = {
                    "factors": list(_facs), "shocks": _shocks,
                    "propagate": bool(_prop), "window": _swin,
                    "event": _evsel, "event_weight": float(_evw)}
                _save_scn_sets(_sets)
                st.success(f"Saved '{_sname.strip()}' "
                           f"({sum(1 for s in _shocks if s)} scenario(s)).")
        _srs = st.session_state.get("_risk_scn_multi")
        if _srs:
            st.markdown(_scn_multi_html(_srs), unsafe_allow_html=True)
            _nts = list(dict.fromkeys(
                n for _, _r in _srs for n in _r["notes"]))
            if _nts:
                st.caption("Notes: " + " · ".join(_nts))


def render_risk():
    st.markdown("#### Speculative Book — Risk & PnL")
    st.caption("IBKR positions **excluding ETFs**. Base currency USD. VaR is 1-day, 95% & 99%.")

    # ── Data freshness + one-click refresh from IBKR Flex Web Service ─────────
    raw_pos = load_positions()
    asof = None
    if not raw_pos.empty and "ReportDate" in raw_pos.columns:
        try:
            asof = pd.to_datetime(raw_pos["ReportDate"]).max()
        except Exception:
            asof = None
    _live_book = st.session_state.get("_risk_live_book")
    if _live_book is None:                       # restore persisted LIVE snapshot (survives reopen)
        _lb, _lfb, _lts = rp.load_live_snapshot()
        if _lb is not None and not _lb.empty:
            _live_book = _lb
            st.session_state["_risk_live_book"] = _lb
            st.session_state["_risk_live_fxbal"] = _lfb
            st.session_state["_risk_live_ts"] = _lts
    _is_live = _live_book is not None and hasattr(_live_book, "empty") and not _live_book.empty
    rc1, rc2 = st.columns([3, 1])
    with rc1:
        if _is_live:
            import datetime as _dt
            _lts = st.session_state.get("_risk_live_ts")
            _lts_txt = _dt.datetime.fromtimestamp(_lts).strftime("%Y-%m-%d %H:%M:%S") if _lts else "?"
            st.caption(f"Positions: 📡 **LIVE snapshot** — {_lts_txt} (IBKR TWS, incl. today's trades; "
                       "settled 1d/3d/5d PnL still Flex EOD).")
        else:
            asof_txt = asof.date().isoformat() if asof is not None and pd.notna(asof) else "unknown"
            st.caption(f"Positions as of **{asof_txt}** (IBKR Flex, end-of-day).")
    with rc2:
        # Pre-flight soft-budget warning for the LIVE pull (subscribes ~1 md-line per
        # position). Estimate from the live snapshot if present, else the Flex book size.
        _est_live = (len(_live_book) if (_is_live and _live_book is not None)
                     else (int(raw_pos["Symbol"].nunique()) if not raw_pos.empty else 0))
        _ulvl, _umsg = ibkr_conn.usage_warning("mktdata", max(_est_live, 1))
        if _ulvl != "ok":
            st.warning(_umsg)
        if st.button("🔄 Update from IBKR (EOD)", use_container_width=True, key="_risk_refresh",
                     help="Pull the end-of-day Flex report into the DB (positions + settled PnL)."):
            st.session_state["_risk_pull_msg"] = _do_flex_pull()      # Flex rate-limit gate
            st.session_state.pop("_risk_live_book", None)             # revert to EOD source
            st.session_state.pop("_risk_live_fxbal", None)
            rp.clear_live_snapshot()                                  # drop persisted live snapshot
            st.rerun()
        if st.button("📡 Update — LIVE", use_container_width=True, key="_risk_live_btn",
                     help="Pull LIVE positions + FX balances from IBKR TWS (incl. today's trades). "
                          "On-demand snapshot — held until you click again (no auto-refresh)."):
            import time
            _lb, _ln = rp.live_positions()
            if _lb is None or _lb.empty:
                st.session_state["_risk_pull_msg"] = ("warning", f"Live positions unavailable — {_ln}.")
            else:
                _lf, _fn = rp.live_fx_balances()
                _ts = time.time()
                st.session_state["_risk_live_book"] = _lb
                st.session_state["_risk_live_fxbal"] = _lf or {}
                st.session_state["_risk_live_ts"] = _ts
                rp.save_live_snapshot(_lb, _lf or {}, _ts)            # persist across reopens
                # ALSO pull today's fills so the Intraday-trades box can decompose the day.
                # Failures NEVER block the live update — store an empty list + the reason.
                try:
                    _fdf, _fnote = rp.todays_fills()
                    _frecs = (_fdf.to_dict("records")
                              if _fdf is not None and not _fdf.empty else [])
                except Exception as _fe:
                    _frecs, _fnote = [], f"fills fetch crashed — {type(_fe).__name__}: {_fe}"
                _frecs = _clean_fill_records(_frecs)
                st.session_state["_risk_live_fills"] = _frecs
                st.session_state["_risk_live_fills_note"] = _fnote
                _save_live_fills(_frecs)                              # into the same snapshot json
                st.session_state["_risk_pull_msg"] = (
                    "success", f"LIVE snapshot — {_ln}; {_fn if _lf else 'FX from Flex EOD'}; {_fnote}.")
            st.rerun()

    sel_fut_syms: set = set()   # futures ticked (pending, until Save)
    sel_fx_ccys: set = set()    # FX pairs ticked (pending, until Save)

    # Positions source: a LIVE TWS snapshot (if 'Update — LIVE' was clicked) else Flex EOD.
    if _is_live:
        book = _live_book
        _lfb = st.session_state.get("_risk_live_fxbal") or None
        fx = build_fx_book(balances=_lfb) if _lfb else build_fx_book()
    else:
        book = build_speculative_book(raw_pos)
        fx = build_fx_book()

    # Options: ALWAYS use the EOD/Flex-settled mark for premium, even when the rest of the
    # book is live — near-expiry, deep-OTM options are often too thin for a reliable live
    # bid/ask; a live TWS quote can sit on a stale `last` print for a while (confirmed
    # 2026-07-22: inflated OZNQ6 C1095's premium ~15x even after a retry/poll fix). The EOD
    # settlement price is the exchange's own official value and doesn't have that problem.
    # Quantity still comes from whichever book is active, so same-day trades are reflected —
    # only the per-unit price is swapped to the EOD mark (falls back to the live/current
    # value for a brand-new option not yet in an EOD Flex pull).
    if not book.empty and "is_option" in book.columns and book["is_option"].any():
        _eod_book = book if not _is_live else build_speculative_book(raw_pos)
        _eod_mark = (_eod_book.set_index("Symbol")["MarkPrice"].to_dict()
                    if not _eod_book.empty else {})

        def _eod_prem(row):
            if not row.get("is_option") or row["Symbol"] not in _eod_mark:
                return row.get("position_value_base")
            mult = row.get("Multiplier") or 1.0
            fx_r = row.get("FXRateToBase") or 1.0
            return float(row["Quantity"]) * float(_eod_mark[row["Symbol"]]) * float(mult) * float(fx_r)

        book = book.copy()
        book["position_value_base"] = book.apply(_eod_prem, axis=1)

    # Saved selection drives the top live table; the checkboxes below edit it.
    (saved_fut, saved_fx, saved_products, saved_ivols,
     saved_proxies, saved_exists) = _load_risk_selection()
    if saved_exists:
        eff_fut, eff_fx = saved_fut, saved_fx
    else:
        eff_fut = set(book["Symbol"]) if not book.empty else set()
        eff_fx  = set(fx["Currency"]) if not fx.empty else set()
    if _is_live and not book.empty:
        # Live book: the saved selection is Flex-era (may be stale after a roll) —
        # show ALL current live positions instead of filtering by it.
        eff_fut = set(book["Symbol"])
    eff_products = dict(saved_products)   # {instrument: "Rates"/"FX"/...}
    eff_ivols = dict(saved_ivols)         # {instrument: float}
    eff_proxies = dict(saved_proxies)     # {instrument: proxy asset}

    def _persist_params():
        """Read Product / Implied Vol / Proxy straight from widget state (robust to
        the top-button-vs-below-expander ordering) and save to disk. Returns the tuple."""
        pf = st.session_state.get("_risk_pending_fut", eff_fut)
        px = st.session_state.get("_risk_pending_fx", eff_fx)
        pp = dict(eff_products); pi = dict(eff_ivols); pr = dict(eff_proxies)
        # Option positions collapse to UNDERLYING rows in the editor (ES/GBL/
        # SOFR3…) whose names are in neither pf nor px, so sweeping only the
        # selection lost their vols on save — fine same-session (widget state
        # carried them) but 0 the next morning (Rajat 2026-09-04). Sweep every
        # rendered editor row via its widget key instead.
        _edited = {k[len("_riskiv_"):] for k in st.session_state
                   if isinstance(k, str) and k.startswith("_riskiv_")}
        for name in set(pf) | set(px) | _edited:
            if f"_riskprod_{name}" in st.session_state:
                pp[name] = st.session_state[f"_riskprod_{name}"]
            if f"_riskiv_{name}" in st.session_state:
                _v = st.session_state[f"_riskiv_{name}"]
                pi[name] = float(_v) if _v else None
            _xk = f"_riskproxy_{name}_{pp.get(name)}"   # proxy key suffixed with product
            if _xk in st.session_state:
                pr[name] = st.session_state[_xk]
        _save_risk_selection(pf, px, pp, pi, pr)
        return pf, px, pp, pi, pr

    # ── Save selection + refresh-PnL buttons (top) ───────────────────────────
    _b1, _b2, _b3 = st.columns([1, 1, 4])
    with _b1:
        if st.button("💾  Save selection", type="primary", key="_risk_save_sel",
                     use_container_width=True):
            pf, px, pp, pi, pr = _persist_params()
            eff_fut, eff_fx = set(pf), set(px)   # reflect immediately in the table below
            eff_products, eff_ivols, eff_proxies = pp, pi, pr
            st.success(f"Saved {len(pf)} futures + {len(px)} FX pairs.")
    with _b2:
        # Pre-flight soft-budget warning for Ref PnL, which re-subscribes every future
        # in the book (est = non-option count, floored at 4 per the telemetry spec).
        _nf = (int((~book["is_option"].fillna(False)).sum())
               if (not book.empty and "is_option" in book.columns) else 0)
        _rlvl, _rmsg = ibkr_conn.usage_warning("mktdata", max(_nf, 4))
        if _rlvl != "ok":
            st.warning(_rmsg)
        if st.button("🔃  Ref PnL", key="_risk_ref_pnl", use_container_width=True,
                     help="Re-price live FX + pull LIVE futures prices from IBKR TWS (market data, "
                          "not the Flex statement) and recompute 1d/3d/5d PnL vs cached daily closes. "
                          "Does NOT touch positions — safe to run many times a day."):
            try:
                fx_spot_quotes.clear(); _fx_closes.clear()   # always re-price live FX
            except Exception:
                pass
            # Fetches daily closes sequentially per future (up to 15s each) — can genuinely
            # take 30-60s+ with several positions. No spinner here looked exactly like a
            # frozen UI (Rajat, 2026-07-22) — this makes the wait visible instead.
            with st.spinner("Fetching live prices from IBKR TWS — can take up to a minute "
                            "with several positions..."):
                _cache, _note = rp.refresh_prices(book)   # 15-min delayed TWS marks (no Flex)
            if _note.startswith("0/") or "no marks" in _note:
                # Got NOTHING — even 15-min delayed failed. That's a TWS-side problem, not
                # positions. Tell the user what to check rather than a bland "refreshed".
                _lvl = "warning"
                _note = (_note + ". Check IBKR TWS is running & logged in, the API is enabled "
                         "(Config → API → Enable ActiveX/Socket clients), and the data-farm "
                         "indicators (bottom-right of TWS) are green. Rows below stay on IBKR "
                         "**settled** PnL, clearly flagged, until marks arrive.")
            elif "TWS unavailable" in _note:
                _lvl = "warning"
            elif "unavailable" in _note or "prev-close" in _note:
                _lvl = "info"     # partial — some priced, some not
            else:
                _lvl = "success"
            st.session_state["_risk_pull_msg"] = (_lvl, f"Marks refreshed — {_note}.")
            st.rerun()

    _msg = st.session_state.pop("_risk_pull_msg", None)
    if _msg:
        getattr(st, _msg[0])(_msg[1])

    # ── TOP: aggregated saved positions, live PnL & exposure ─────────────────
    st.markdown("##### 📌 Aggregated positions — live (saved selection)")
    _price_cache = rp.load_price_cache()
    _live_md = rp.live_multiday_pnl(book, _price_cache) if not book.empty else {}   # TWS live
    _settled_md = futures_multiday_pnl(list(book["Symbol"])) if not book.empty else {}  # Flex fallback
    _fx_cl = _fx_closes(tuple(fx["Currency"])) if not fx.empty else {}

    # Freshness banner: live-price time + settled-fallback date.
    _live_ts, _closes_date = rp.prices_asof(_price_cache)
    _bits = []
    if _live_ts:
        import datetime as _dt
        _bits.append(f"live prices **{_dt.datetime.fromtimestamp(_live_ts).strftime('%H:%M:%S')}**"
                     + (f", closes to {_closes_date}" if _closes_date else ""))
    _fut_asof = futures_pnl_asof()
    if _fut_asof is not None:
        _bits.append(f"settled-PnL fallback to {_fut_asof.date().isoformat()}")
    if _bits:
        st.caption("⏱️ Futures PnL: " + " · ".join(_bits)
                   + ".  Click **🔃 Ref PnL** for fresh live prices (no position pull).")
    if not _live_md and _live_ts is None:
        st.caption("ℹ️ No live prices yet — click **🔃 Ref PnL** (needs IBKR TWS running). "
                   "Showing IBKR settled PnL until then.")

    # ── Per-row mark source & freshness ──────────────────────────────────────
    # Every row carries where its 1d/3d/5d PnL came from, so the table can HIGHLIGHT
    # trustworthy live marks and MUTE stale/settled fallbacks instead of showing them
    # as if they were live. Classes: "live" (fresh TWS today), "stale" (cached live
    # mark from a previous day), "settled" (no live mark → IBKR Flex settlement),
    # "none" (no data at all).
    import datetime as _dt
    _today_d = _dt.date.today()
    _settled_asof = _fut_asof.date().isoformat() if _fut_asof is not None else "—"

    def _fut_src(sym, used_live):
        ent = _price_cache.get(sym, {})
        lts = ent.get("live_ts")
        lsrc = ent.get("live_src")
        if used_live and lts:
            d = _dt.datetime.fromtimestamp(lts)
            if d.date() == _today_d:
                if lsrc == "prev-close":
                    return "pclose", d.strftime("%H:%M")    # only prev close, no intraday tick
                if lsrc == "live":
                    return "live", d.strftime("%H:%M")      # real-time (CME etc.)
                if lsrc == "closed":
                    return "closed", d.strftime("%H:%M")    # market shut → close/last-trade mark
                return "delayed", d.strftime("%H:%M")       # 15-min delayed
            return "stale", d.date().isoformat()            # cached mark, but from an earlier day
        return "settled", _settled_asof                     # Flex settlement fallback

    _agg = []
    _opt_pnl_est = {}
    if not book.empty and "is_option" in book.columns and book["is_option"].any():
        try:
            import risk_options as _rop_est
            _opt_pnl_est = _rop_est.est_pnl(book, set(eff_fut))
        except Exception:
            _opt_pnl_est = {}
    if not book.empty:
        for _, r in book[book["Symbol"].isin(eff_fut)].iterrows():
            sym = r["Symbol"]
            _und = str(r.get("Underlying") or "")
            if bool(r.get("is_option")):
                # options: the UNDERLYING carries the product — the ⚙ editor
                # saves by underlying now, and legacy per-SYMBOL saves misfile
                # FX options as Rates ("EUUU6 P1157": "Rates" in the old json,
                # which parked them inside the Rates block — Rajat 2026-09-04)
                prod = (eff_products.get(_und)
                        or _guess_product(_und or sym, _und or sym))
            else:
                prod = eff_products.get(sym) or _guess_product(sym, _und)
            pvb = float(r["position_value_base"])
            if bool(r.get("is_option")):
                # Options: rough delta×move PnL off the vol-market surfaces
                # when the underlying has history (Rajat 2026-08-26 — "better
                # than blank, keep it grey": first-order only, no gamma/vega/
                # theta, so muted ≈ Δ-est badge, excluded from the Total).
                # Falls back to Flex settled marks, then blank.
                _est = _opt_pnl_est.get(sym)
                if _est:
                    _agg.append(("Option", sym, prod, None, r["side"], pvb,
                                 _est.get(1), _est.get(3), _est.get(5), None,
                                 _opt_prem_str(pvb, r.get("Expiry")),
                                 float(r["Quantity"]), "dest", "delta est"))
                    continue
                omd = _settled_md.get(sym, {})
                _src = ("settled", _settled_asof) if omd else ("none", "")
                _agg.append(("Option", sym, prod, None, r["side"], pvb,
                             omd.get(1), omd.get(3), omd.get(5), None,
                             _opt_prem_str(pvb, r.get("Expiry")), float(r["Quantity"]),
                             _src[0], _src[1]))
                continue
            live_md = _live_md.get(sym)
            used_live = live_md is not None
            md = live_md if used_live else _settled_md.get(sym, {})   # prefer live, fall back
            if not md:
                _src = ("none", "")
            else:
                _src = _fut_src(sym, used_live)
            iv = eff_ivols.get(sym)
            if prod == "Rates":
                # $DV01 = |lots| × multiplier × 0.01 (1bp = 0.01 price pts) × FX, signed by side
                qty = float(r["Quantity"]); mult = float(r.get("Multiplier") or 0.0)
                fxr = float(r.get("FXRateToBase") or 1.0)
                risk = qty * mult * 0.01 * fxr
            else:                       # Equities / Commod (and any non-rates future) → $ notional
                risk = pvb
            _agg.append(("Future", sym, prod, iv, r["side"], pvb,
                         md.get(1), md.get(3), md.get(5), risk, None, float(r["Quantity"]),
                         _src[0], _src[1]))
    if not fx.empty:
        for _, r in fx[fx["Currency"].isin(eff_fx)].iterrows():
            _e   = float(r["USD_exposure"]) if pd.notna(r["USD_exposure"]) else 0.0
            b    = float(r["Balance"])
            live = r["USD_per_unit"]
            p1 = float(r["Today_PnL_USD"]) if pd.notna(r["Today_PnL_USD"]) else None
            p3 = p5 = None
            closes = _fx_cl.get(r["Currency"])
            if closes and pd.notna(live):
                if len(closes) >= 4:
                    p3 = b * (float(live) - closes[-4])   # vs close 3 business days ago
                if len(closes) >= 6:
                    p5 = b * (float(live) - closes[-6])   # vs close 5 business days ago
            ccy = r["Currency"]
            prod = eff_products.get(ccy, "FX")
            iv = eff_ivols.get(ccy)
            _fxsrc = ("live", "spot") if pd.notna(live) else ("none", "")
            _agg.append(("FX", ccy, prod, iv, r["side"], _e, p1, p3, p5, _e, None, None,
                         _fxsrc[0], _fxsrc[1]))  # FX $ risk = USD exp

    if not _agg:
        st.info("No saved positions yet — tick futures/FX below and click **💾 Save selection**.")
    else:
        # Freshness banner: how many rows have a current intraday mark vs old fallbacks.
        _n_live = sum(1 for x in _agg if x[12] in ("live", "delayed", "closed"))
        _n_bad  = sum(1 for x in _agg if x[12] in ("pclose", "stale", "settled", "none", "dest"))
        if _n_bad:
            _detail = []
            _n_pcl     = sum(1 for x in _agg if x[12] == "pclose")
            _n_stale   = sum(1 for x in _agg if x[12] == "stale")
            _n_settled = sum(1 for x in _agg if x[12] == "settled")
            _n_none    = sum(1 for x in _agg if x[12] == "none")
            if _n_pcl:     _detail.append(f"{_n_pcl} prev-close only")
            if _n_stale:   _detail.append(f"{_n_stale} stale (earlier day)")
            if _n_settled: _detail.append(f"{_n_settled} on IBKR settled (as of {_settled_asof})")
            if _n_none:    _detail.append(f"{_n_none} with no price")
            _n_dest = sum(1 for x in _agg if x[12] == "dest")
            if _n_dest:    _detail.append(f"{_n_dest} option Δ-estimates")
            _bmsg = (f"⚠️ **{_n_live}/{len(_agg)} positions have a current intraday mark.** "
                     f"The rest are NOT intraday — " + "; ".join(_detail) + ". "
                     "Their PnL is muted (grey) below and **excluded from the Total**. "
                     "Click **🔃 Ref PnL** with IBKR TWS running to refresh (15-min delayed by default).")
            (st.warning if _n_live == 0 else st.info)(_bmsg)
        _agg.sort(key=lambda x: _inst_sort_key(x[1], x[2]))
        st.markdown(_agg_table_html(_agg), unsafe_allow_html=True)
        _cls_html = _class_pnl_html(_agg)
        if _cls_html:
            st.markdown("<div style='height:8px'></div>" + _cls_html,
                        unsafe_allow_html=True)
    st.caption("**Marks** column = where each row's PnL comes from: **🟢 live** (real-time, e.g. CME) · "
               "**🕒 15m** (15-min delayed) · **🌙 closed** (that contract's market is shut — priced off "
               "its close/last trade, so its PnL is the completed-session move and won't tick) · "
               "**⚪ prev-close** (only yesterday's close) · "
               "**🟡 stale** (a mark from an earlier day) · **⚪ settled** (IBKR Flex settlement) · "
               "**≈ Δ-est** (option PnL ≈ delta × underlying move off the vol surfaces — first order "
               "only, no gamma/vega/theta) · "
               "**✖ no mark**. Only 🟢/🕒/🧊 current marks are coloured and counted in the **Total**; "
               "the rest are greyed so old figures are never mistaken for a current intraday PnL.  ·  "
               "**Futures PnL** = live IBKR TWS price × position vs the **close N business days ago** "
               "(Qty × Mult × Δprice × FX); refreshed by **🔃 Ref PnL** (market data, not the Flex statement — "
               "positions stay put). Falls back to IBKR **settled** PnL if TWS is unavailable. "
               "**FX PnL** = balance × rate move (1d = live vs prior close; 3d/5d = live vs "
               "close N days ago). '—' = no history yet for that window.  ·  "
               "**Implied Vol** (manual): Rates = annualized **normal** vol in bps, else % annual vol.  ·  "
               "**$ Risk**: FX = USD exposure; Rates = $DV01 (|lots| × multiplier × 0.01 × FX); "
               "Equities/Commod = USD notional (no total — mixed units).  ·  "
               "**1d VaR** = $Risk × (implied vol / √256), per position. For a portfolio VaR "
               "(correlation/diversification) use the **🎲 Run VaR Risk** report below.")

    # ── Intraday trades — today's day decomposition (LIVE mode only) ─────────
    # The 1d PnL above marks live positions from PRIOR CLOSE — right for carried lots,
    # wrong for anything traded today. This box splits the day into Carried + Intraday
    # trades − Commissions, using the SAME marks/closes the table above uses.
    if _is_live:
        _fills = st.session_state.get("_risk_live_fills")
        if _fills is None:                                   # restored snapshot → read from file
            _fills = _load_live_fills()
        _fills_note = st.session_state.get("_risk_live_fills_note", "")
        st.markdown("##### 🔁 Intraday trades — today's day decomposition")
        if not _fills:
            _nl = _fills_note.lower()
            if _fills_note and any(k in _nl for k in ("unavailable", "failed", "crashed")):
                st.caption(f"Intraday trades unavailable — {_fills_note}.")
            else:
                st.caption("No intraday trades today.")
        else:
            _eod_bk = build_speculative_book(raw_pos)        # EOD marks for options
            _eod_mk = (_eod_bk.set_index("Symbol")["MarkPrice"].to_dict()
                       if not _eod_bk.empty else {})
            _imarks, _ipc = _intraday_marks(book, _price_cache, _eod_mk, _today_d.isoformat())
            _cashf = [f for f in _fills if str(f.get("SecType", "")).upper() == "CASH"]
            _derivf = [f for f in _fills if str(f.get("SecType", "")).upper() != "CASH"]
            _decomp = _intraday_decomposition(_derivf, book, _imarks, _ipc)
            if _cashf:
                # Spot-FX day trades (IDEALPRO): marked vs live yfinance crosses.
                _ccys = set()
                for f in _cashf:
                    try:
                        b_, q_ = str(f.get("Symbol") or "").split(".")
                        _ccys.update((b_.upper(), q_.upper()))
                    except Exception:
                        pass
                _ccys.discard("USD")
                _fxq = fx_spot_quotes(tuple(sorted(_ccys))) if _ccys else {}
                _fxbal = st.session_state.get("_risk_live_fxbal") or {}
                _fxrows, _fxtot, _fxcar, _fxcomm, _fxskip = _fx_fills_rows(
                    _cashf, _fxq, _fxbal)
                _decomp["rows"] += _fxrows
                _decomp["trades_pnl"] += _fxtot
                _decomp["carried_pnl"] += _fxcar
                _decomp["commissions"] += _fxcomm
                _decomp["total_day"] = (_decomp["carried_pnl"] + _decomp["trades_pnl"]
                                        - _decomp["commissions"])
                _decomp["unmatched"] += _fxskip
            if not _decomp["rows"]:
                _um = sorted(set(_decomp["unmatched"]))
                st.caption("Today's fills aren't in the current live book (contract mismatch — "
                           "e.g. a position fully closed out): "
                           + (", ".join(_um) if _um else "—") + ".")
            else:
                st.markdown(_intraday_box_html(_decomp), unsafe_allow_html=True)
                _notes = ["**Total day = Carried + Intraday trades − Commissions.** "
                          "Carried = qty held from yesterday × (mark − prior close); "
                          "Intraday = each fill × (mark − fill price). Marks match the table "
                          "above (futures: live/settlement; options: EOD-settled)."]
                if _decomp["unmatched"]:
                    _notes.append("Skipped fills not in the current book (fully closed out / "
                                  "contract mismatch): " + ", ".join(sorted(set(_decomp["unmatched"]))) + ".")
                if _decomp["carried_missing"]:
                    _notes.append("No prior-close for " + ", ".join(sorted(set(_decomp["carried_missing"])))
                                  + " → carried P&L omitted for these (shown only under Intraday trades).")
                if _decomp["commissions"]:
                    _notes.append("Commissions summed as reported — converted to USD via each fill's "
                                  "FXRateToBase where the contract matched the book, else summed as-is.")
                st.caption("  ·  ".join(_notes))

                # ── Whole-book day total: ONE number, no double-counting. Traded
                # symbols/ccys use the box's corrected decomposition; everything else
                # gets the plain qty×(mark − prior close) / balance×Δ leg. ─────────
                _traded_syms = {r["Symbol"] for r in _decomp["rows"]}
                _untraded = 0.0
                _untr_miss = []
                for _, _br in book.iterrows():
                    _s = _br["Symbol"]
                    if _s in _traded_syms:
                        continue
                    _mk, _pc = _imarks.get(_s), _ipc.get(_s)
                    if _mk is None or _pc is None:
                        if not _br.get("is_option"):
                            _untr_miss.append(_s)
                        continue
                    _untraded += (float(_br.get("Quantity") or 0) * (_mk - _pc)
                                  * float(_br.get("Multiplier") or 1.0)
                                  * float(_br.get("FXRateToBase") or 1.0))
                _fxb_all = st.session_state.get("_risk_live_fxbal") or {}
                _tccys = set()
                for _f in _cashf:
                    try:
                        _b2, _q2 = str(_f.get("Symbol") or "").split(".")
                        _tccys.update((_b2.upper(), _q2.upper()))
                    except Exception:
                        pass
                _fx_untr = 0.0
                _untr_c = sorted(c for c in _fxb_all
                                 if str(c).upper() not in _tccys and str(c).upper() != "USD")
                if _untr_c:
                    for _c3, _d3 in fx_spot_quotes(tuple(_untr_c)).items():
                        if _d3.get("live") and _d3.get("prev"):
                            _fx_untr += (float(_fxb_all.get(_c3, 0.0))
                                         * (float(_d3["live"]) - float(_d3["prev"])))
                _whole = _decomp["total_day"] + _untraded + _fx_untr
                st.markdown(
                    f"<div style='margin-top:10px;padding:8px 14px;border:1px solid #CBD5E1;"
                    f"border-radius:8px;font-family:monospace;font-size:14px;font-weight:700;"
                    f"display:flex;justify-content:space-between;max-width:520px'>"
                    f"<span>📅 Whole-book day P&L (incl. intraday trades)</span>"
                    f"<span style='color:{_pnl_color(_whole)}'>${_whole:,.0f}</span></div>",
                    unsafe_allow_html=True)
                st.caption(
                    f"= traded book (box above) ${_decomp['total_day']:,.0f} "
                    f"+ untraded positions ${_untraded:,.0f} "
                    f"+ untraded FX balances ${_fx_untr:,.0f}."
                    + (f"  No mark/prior-close for: {', '.join(_untr_miss)} — excluded."
                       if _untr_miss else "")
                    + "  Options' day change excluded unless traded today (EOD-marked).")

    # ── TWS diagnostics (paste-to-debug) ─────────────────────────────────────
    with st.expander("🩺  TWS diagnostics — run this if marks look wrong, then paste the output",
                     expanded=False):
        st.caption("One-off probe of the live-price path: connection details, per-symbol contract "
                   "qualify timing, the **market-data type TWS actually serves** (live / frozen / "
                   "delayed), the bid/ask/last/close it returns, and any IBKR data-farm / "
                   "market-data-permission messages. Copy the whole block to share for diagnosis. "
                   "Needs IBKR TWS running; takes ~5–15s.")
        if st.button("Run TWS diagnostic", key="_risk_tws_diag"):
            with st.spinner("Probing TWS (qualify + delayed market-data per symbol)…"):
                try:
                    _rep = rp.tws_diagnostics(book)
                except Exception as _e:
                    _rep = f"diagnostic crashed: {type(_e).__name__}: {_e}"
            st.session_state["_risk_tws_diag_out"] = _rep
        _diag = st.session_state.get("_risk_tws_diag_out")
        if _diag:
            st.code(_diag, language="text")

    # ── IBKR usage & limits (request telemetry + soft-budget warnings) ────────
    with st.expander("📊  IBKR usage & limits", expanded=False):
        _tod   = ibkr_conn.usage_today()
        _bytag = ibkr_conn.usage_today_by_tag()
        st.markdown("**Today's IBKR request usage vs soft budgets**")
        st.markdown(_usage_today_html(_tod, _bytag), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>**Last 14 days**", unsafe_allow_html=True)
        st.markdown(_usage_history_html(ibkr_conn.usage_history(14)), unsafe_allow_html=True)
        _bkc1, _bkc2 = st.columns([3, 1])
        with _bkc1:
            _bk_note = st.text_input(
                "Breakage note (what broke?)", key="_ibkr_breakage_note",
                placeholder="e.g. delayed data stopped in TWS — 'subscribe' prompts on unrelated products")
        with _bkc2:
            st.write("")
            if st.button("⚠️ Mark breakage now", key="_ibkr_mark_breakage",
                         use_container_width=True):
                ibkr_conn.mark_breakage(_bk_note or "")
                st.success("Breakage stamped with the current usage counts.")
        st.caption(
            "Soft budgets live in **ibkr_budgets.json** (project dir) — edit them and they reload "
            "within a minute; no restart. These are **warnings only, never blocks** (the daily IBKR "
            "delayed-data limit is unpublished, so we can't enforce it). **Learning period:** when "
            "delayed data breaks in TWS, click **⚠️ Mark breakage now** — the ⚡ marks the day above. "
            "After a few incidents, tighten each budget to just under the lowest daily usage that "
            "preceded a breakage.")

    # ── Options detail (collapsible, below the positions box) ────────────────
    _opts = book[book["is_option"]] if (not book.empty and "is_option" in book.columns) else None
    if _opts is not None and not _opts.empty:
        with st.expander(f"🎯  Options ({len(_opts)}) — premium & time-to-expiry", expanded=False):
            _uroots = tuple(sorted({_root_of(str(r.get("Underlying") or ""))
                                    for _, r in _opts.iterrows()}))
            _ulvl = _underlying_levels(_uroots)
            _orows = []
            for _, r in _opts.iterrows():
                prem = float(r["position_value_base"])   # SIGNED: long +, short −
                dte = _business_dte(r.get("Expiry"))
                pst = (prem / (dte ** 0.5)) if (dte and dte > 0) else None
                und = str(r.get("Underlying") or "")
                lvl = _ulvl.get(_root_of(und))
                _strike = r.get("Strike")
                strike = float(_strike) if pd.notna(_strike) else None
                _expv = r.get("Expiry")
                if (_expv is not None and not isinstance(_expv, str)
                        and pd.notna(_expv)):
                    _expv = str(int(_expv))          # int yyyymmdd → epoch bug
                _exp = pd.to_datetime(_expv, errors="coerce")
                expy = _exp.strftime("%d-%b-%y") if pd.notna(_exp) else "—"
                _orows.append((r["Symbol"], und, strike, expy, lvl, prem, dte, pst))
            # class+family order like the other position tables (Rajat
            # 2026-09-05; was soonest-expiry-first) — product via underlying
            _orows.sort(key=lambda x: _inst_sort_key(
                x[0], _guess_product(x[1] or x[0], x[1] or x[0])))
            st.markdown(_options_box_html(_orows), unsafe_allow_html=True)
            st.caption("**Prem** & **Prem / √t** are **signed** (long +, short −); the **Total** row is the "
                       "net book premium.  ·  **Fut Level** = underlying future's ~live level (yfinance active "
                       "contract).  ·  **Days to Exp** = business/working days (Fri→Mon = 1).  ·  "
                       "**Prem / √t** = premium ÷ √(business days).")

    # ── Save Params (co-located with the editor) ─────────────────────────────
    if st.button("💾  Save Params", key="_risk_save_params",
                 help="Save the products / implied vols / proxies set below, then run a fresh "
                      "VaR report. (Same as 'Save selection' at the top.)"):
        pf, px, pp, pi, pr = _persist_params()
        eff_fut, eff_fx = set(pf), set(px)
        eff_products, eff_ivols, eff_proxies = pp, pi, pr
        st.session_state["_risk_pull_msg"] = (
            "success", f"Params saved ({len(pi)} vols). Now click 🎲 Run VaR Risk for a fresh report.")
        st.rerun()

    # ── Manual Product / Implied Vol / Proxy inputs ──────────────────────────
    if not book.empty or not fx.empty:
        # Auto-open ONLY for an unconfigured book (no saved vols yet). Anything
        # dynamic here is a trap: the fragment re-mounts this element on every
        # in-tab button click, discarding the user's manual open/closed state
        # in favour of this default — and on the Run-VaR click-rerun the old
        # "_risk_var_result not in ss" check was still True (the result is
        # only stored BELOW, later in the same run), so the box jumped open
        # even when manually closed (Rajat 2026-08-20). ss["_risk_run_var"]
        # is the Run button's own key — True during its click-rerun.
        _iv_exp_default = (not eff_ivols
                           and "_risk_var_result" not in st.session_state
                           and not st.session_state.get("_risk_run_var"))
        with st.expander("⚙️  Set Product, Implied Vol & Proxy (manual)",
                         expanded=_iv_exp_default):
            st.caption("**Implied Vol** — Rates → annualized **normal** vol in **bps** (66 = 66 bp/yr); "
                       "FX / Equities / Commod → **% annual** vol (5 = 5%).  ·  "
                       "**Proxy** — the asset used for the correlation / diversification calc "
                       "(FX just uses its own pair).  ·  Then hit **💾 Save selection** above.")
            _pp = dict(st.session_state.get("_risk_pending_products", eff_products))
            _pi = dict(st.session_state.get("_risk_pending_ivols", eff_ivols))
            _px = dict(st.session_state.get("_risk_pending_proxies", eff_proxies))
            _h = st.columns([3, 3, 2, 3])
            _h[0].markdown("**Instrument**")
            _h[1].markdown("**Product**")
            _h[2].markdown("**Implied Vol**")
            _h[3].markdown("**Proxy**")

            def _iv_row(name, default_prod, ccy=None):
                cc = st.columns([3, 3, 2, 3])
                cc[0].markdown(f"`{name}`")
                cur_prod = eff_products.get(name, default_prod)
                if cur_prod not in _PRODUCTS:
                    cur_prod = default_prod
                p = cc[1].selectbox("Product", _PRODUCTS, index=_PRODUCTS.index(cur_prod),
                                    key=f"_riskprod_{name}", label_visibility="collapsed")
                v = cc[2].number_input("Implied Vol", min_value=0.0, step=0.5, format="%.2f",
                                       value=float(eff_ivols.get(name) or 0.0),
                                       key=f"_riskiv_{name}", label_visibility="collapsed")
                # Proxy options depend on the *current* product selection; key is
                # suffixed with the product so switching product never leaves a
                # stale value that isn't in the new option list.
                popts = _proxy_options(p, name, ccy)
                cur_px = eff_proxies.get(name) or _guess_proxy(p, name)
                if cur_px not in popts:
                    cur_px = popts[0]
                xval = cc[3].selectbox("Proxy", popts, index=popts.index(cur_px),
                                       key=f"_riskproxy_{name}_{p}", label_visibility="collapsed")
                _pp[name] = p
                _pi[name] = float(v) if v else None
                _px[name] = xval

            _shown = False
            _done = set()          # every _iv_row name rendered — a name may
            _opt_unds = {}         # reach us via several paths; render ONCE
            if not book.empty:
                _u = book["Underlying"].values if "Underlying" in book.columns else [""] * len(book)
                _io = (book["is_option"].fillna(False).values
                       if "is_option" in book.columns else [False] * len(book))
                # options collapse to their UNIQUE UNDERLYINGS (Rajat
                # 2026-08-24: "ESU6 P7600 / P7500 → just ask for ESU6") —
                # risk_options looks vols/proxies up by underlying symbol
                for s, u, io_ in zip(book["Symbol"].values, _u, _io):
                    if s not in eff_fut:
                        continue
                    if bool(io_):
                        und = str(u or "").strip()
                        if und:
                            _opt_unds.setdefault(und, _guess_product(und, und))
                        continue
                    _iv_row(s, _guess_product(s, u))
                    _done.add(s)
                    _shown = True
                # FX-cash rows render below under the same name (e.g. an option
                # on the EUR future ↔ the EUR cash-balance row — same EUR/USD
                # vol): let the FX row be the single source, skip the dup here
                _fx_names = (set(fx["Currency"]) & set(eff_fx)
                             if not fx.empty else set())
                for und, prod_ in _opt_unds.items():
                    if und in _done or und in _fx_names:
                        continue
                    _iv_row(und, prod_)
                    _done.add(und)
                    _shown = True
            if not fx.empty:
                for c_ in fx["Currency"].values:
                    if c_ in eff_fx and c_ not in _done:
                        _iv_row(c_, "FX", ccy=c_)
                        _done.add(c_)
                        _shown = True
            if not _shown:
                st.caption("No positions in the saved selection yet — tick ✓ in the tables below "
                           "and **💾 Save selection** first.")
            st.session_state["_risk_pending_products"] = _pp
            st.session_state["_risk_pending_ivols"] = _pi
            st.session_state["_risk_pending_proxies"] = _px

    # ── Diversified VaR — runs ONLY on button click (not on page load) ────────
    _vc1, _vc2, _vcsp = st.columns([1.0, 1.5, 2.0])
    _opt_mode_lbl = _vc2.selectbox(
        "Options risk", ["No options risk", "Delta-equivalent mapping",
                         "Full-revaluation historical VaR"],
        key="_risk_opt_mode",
        help="How option positions (FOPs in the book) enter the VaR report. "
             "Delta-equivalent: delta × lots × mult of the underlying, linear, "
             "into √(vᵀRv) via the underlying's proxy. Full-reval: each structure "
             "repriced under ~250 historical daily underlying moves (sticky IV) — "
             "gamma-exact percentile VaR; vol risk not included in either.")
    _vc1.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if _vc1.button("🎲  Run VaR Risk", key="_risk_run_var",
                   help="Estimate correlation-adjusted portfolio VaR across 1m/3m/6m/1y windows. "
                        "Fetches proxy history from yfinance/FRED — runs only when clicked."):
        _rprod, _rprox = {}, {}
        if not book.empty:
            _uu = book["Underlying"].values if "Underlying" in book.columns else [""] * len(book)
            for s, u in zip(book["Symbol"].values, _uu):
                if s in eff_fut:
                    _rprod[s] = eff_products.get(s) or _guess_product(s, u)
                    _rprox[s] = eff_proxies.get(s) or _guess_proxy(_rprod[s], s, u)
        if not fx.empty:
            for c_ in fx["Currency"].values:
                if c_ in eff_fx:
                    _rprod[c_] = eff_products.get(c_, "FX")
                    _rprox[c_] = eff_proxies.get(c_, c_)
        try:
            _fred = st.secrets.get("FRED_KEY")
        except Exception:
            _fred = None
        _mode_key = {"Delta-equivalent mapping": "delta",
                     "Full-revaluation historical VaR": "reval"}.get(_opt_mode_lbl)
        _ors = None
        if _mode_key:
            import risk_options
            with st.spinner("Pricing options off settlement surfaces…"):
                try:
                    _ors = risk_options.compute(book, _mode_key, eff_products,
                                                eff_ivols, eff_proxies, _fred,
                                                sel=set(eff_fut))
                except Exception as _oe:
                    st.error(f"⛔ Options risk failed ({type(_oe).__name__}: {_oe}) "
                             "— this run is FUTURES/FX ONLY, same as No-options.")
        st.session_state["_risk_opt_stamp"] = (
            f"options mode: **{_opt_mode_lbl}** · "
            + (f"{len(_ors['extra_pos'])} option row(s) in √(vᵀRv) "
               f"(engine v{risk_options._BUILD})" if _ors else "0 option rows")
            )
        st.session_state["_risk_opt_var"] = _ors
        with st.spinner("Fetching proxy history & computing diversified VaR…"):
            st.session_state["_risk_var_result"] = risk_div.compute(
                book, fx, set(eff_fut), set(eff_fx), _rprod, eff_ivols, _rprox, _fred,
                extra_pos=(_ors or {}).get("extra_pos"))

    _vres = st.session_state.get("_risk_var_result")
    if _vres and _vres.get("windows"):
        _stamp = st.session_state.get("_risk_opt_stamp")
        if _stamp:
            st.caption(_stamp)
        st.markdown(_win_var_html(_vres), unsafe_allow_html=True)
        # ── VaR by asset class — window-selectable (default 1y) ──────────────
        _acw = _vres.get("by_asset_class_windows", {})
        if _acw:
            _wopts = list(_acw.keys())
            _wsel = st.radio(
                "Asset-class VaR — correlation window", _wopts,
                index=_wopts.index("1y") if "1y" in _wopts else len(_wopts) - 1,
                horizontal=True, key="_risk_ac_window")
            st.markdown(_ac_var_html(_vres, _wsel), unsafe_allow_html=True)
        # ── Net delta risk by asset class × underlying (Rajat 2026-08-24) ────
        import risk_options as _rop
        _split: dict = {}

        def _sp_add(cls, key, i, v):
            _split.setdefault(cls, {}).setdefault(key, [0.0, 0.0])[i] += v
        if not book.empty:
            for _, _r in book[book["Symbol"].isin(eff_fut)].iterrows():
                if bool(_r.get("is_option")):
                    continue                       # options via _ors exposures
                _sym = _r["Symbol"]
                _prod = (eff_products.get(_sym)
                         or _guess_product(_sym, _r.get("Underlying", "")))
                # rates: per contract (SR3M6, SR3U6, …); rest: complex (ES…)
                _key = (_rop.underlying_contract(_sym) if _prod == "Rates"
                        else _rop.underlying_key(_sym))
                if _prod == "Rates":
                    _v = (float(_r["Quantity"]) * float(_r.get("Multiplier") or 0.0)
                          * 0.01 * float(_r.get("FXRateToBase") or 1.0))
                else:
                    _v = float(_r["position_value_base"])
                _sp_add(_prod, _key, 0, _v)
        if not fx.empty:
            for _, _r in fx[fx["Currency"].isin(eff_fx)].iterrows():
                if pd.notna(_r["USD_exposure"]):
                    _sp_add("FX", _r["Currency"], 0, float(_r["USD_exposure"]))
        _ors_e = st.session_state.get("_risk_opt_var")
        for _cls, _key, _v in ((_ors_e or {}).get("exposures") or []):
            _sp_add(_cls, _key, 1, _v)
        if _split:
            st.markdown("<div style='height:10px'></div>"
                        + _net_split_html(_split), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>" + _pos_var_html(_vres),
                    unsafe_allow_html=True)
        _ors = st.session_state.get("_risk_opt_var")
        if _ors and (_ors["rows"] or _ors["notes"]):
            if not _ors["extra_pos"]:
                st.warning("⚠️ An options mode was selected but **no option "
                           "contributed to the VaR above** — every position was "
                           "skipped (reasons below). The report equals the "
                           "no-options run.")
            st.markdown("<div style='height:10px'></div>" + _opt_var_html(_ors),
                        unsafe_allow_html=True)
            if _ors["notes"]:
                st.caption("Options skipped/warnings: " + " · ".join(_ors["notes"]))
        _m = _vres.get("meta", {})
        _note = (f"Correlations from proxy returns over **{_m.get('n_days','?')}** aligned days "
                 f"({_m.get('hist_start','?')} → {_m.get('hist_end','?')}). "
                 "Magnitudes from your manual implied vols; 1σ daily, 95% = ×1.645. "
                 "Portfolio VaR = √(vᵀRv), v = signed standalone VaRs.")
        if _m.get("dropped"):
            _note += f"  Proxies with no data (added uncorrelated): {', '.join(_m['dropped'])}."
        st.caption(_note)
    elif _vres:
        st.warning("Could not compute — check the saved selection has positions and implied vols.")
    st.divider()

    if book.empty:
        st.warning("No speculative positions found. Upload an IBKR Flex statement in the P&L tab "
                   "(or wire the Flex Web Service) so positions are in the local DB.")
    else:
        m = book_metrics(book)
        c = st.columns(2)
        c[0].metric("Positions", m["n_positions"])
        c[1].metric("Unrealised PnL", f"${m['upnl_base']:,.0f}")

        _disp = pd.DataFrame({
            "Include": [s in eff_fut for s in book["Symbol"]],
            "Symbol": book["Symbol"].values,
            "Side": book["side"].values,
            "Qty": [f"{q:,.0f}" for q in book["Quantity"]],
            "Mark": [f"{p:,.3f}" for p in book["MarkPrice"]],
            "Exposure (USD)": [f"${v:,.0f}" for v in book["position_value_base"]],
            "uPnL (USD)": [f"${v:,.0f}" for v in book["upnl_base"]],
        })
        _ed = st.data_editor(
            _disp, hide_index=True, use_container_width=True, key="_risk_book_sel",
            column_config={"Include": st.column_config.CheckboxColumn(
                "✓", help="Include in the Aggregated positions table above")},
            disabled=["Symbol", "Side", "Qty", "Mark", "Exposure (USD)", "uPnL (USD)"],
        )
        sel_fut_syms = set(_ed.loc[_ed["Include"] == True, "Symbol"])
        st.session_state["_risk_pending_fut"] = sel_fut_syms
        st.caption("Tick ✓ to include a position, then **💾 Save selection** at the top. Set **Product** "
                   "+ **Implied Vol** in the **⚙️ expander above**. Notional/exposure ≠ risk for STIR; "
                   "the **🎲 Run VaR Risk** report is the meaningful risk measure.")

    # ── FX exposure (non-USD cash balances = open FX positions vs USD) ────────
    st.divider()
    st.markdown("##### FX Exposure — cash balances vs USD")
    if fx.empty:
        st.caption("No non-USD cash balances found (add the Cash Report section to your Flex query).")
    else:
        exp = fx["USD_exposure"].dropna()
        today = fx["Today_PnL_USD"].dropna()
        fx_start, fx_end = fx_activity_span()
        span_txt = (f"{fx_start} → {fx_end}" if fx_start else "unknown range")
        fc = st.columns(4)
        # sign flipped vs the per-ccy rows: short foreign ccys ⇒ LONG USD
        _usd_pos = -exp.sum()
        fc[0].metric("Net USD position",
                     f"{'+' if _usd_pos >= 0 else '−'}${abs(_usd_pos):,.0f}",
                     help="−(sum of non-USD exposures): + = net long USD, "
                          "− = net short USD. Per-ccy rows below keep their "
                          "own sign (− = short that ccy).")
        fc[1].metric("Gross FX exposure (USD)", f"${exp.abs().sum():,.0f}")
        fc[2].metric("Today FX PnL (MTM)", f"${today.sum():,.0f}" if len(today) else "—",
                     help="Mark-to-market: FX exposure × (live rate − prior session close), per currency.")
        fc[3].metric("Realised FX PnL — cumulative", f"${fx['Realized_PnL_USD'].sum():,.0f}",
                     help=f"IBKR's realised FX conversions in the DB ({span_txt}). NOT a daily figure.")
        _fxd = pd.DataFrame({
            "Include": [c in eff_fx for c in fx["Currency"]],
            "Ccy": fx["Currency"].values,
            "Side": fx["side"].values,
            "Balance": [f"{b:,.0f}" for b in fx["Balance"]],
            "USD/unit": [f"{r:.6g}" if pd.notna(r) else "—" for r in fx["USD_per_unit"]],
            "USD exposure": [f"${v:,.0f}" if pd.notna(v) else "—" for v in fx["USD_exposure"]],
            "Today PnL": [f"${v:,.0f}" if pd.notna(v) else "—" for v in fx["Today_PnL_USD"]],
            "Realised FX": [f"${v:,.0f}" for v in fx["Realized_PnL_USD"]],
        })
        _fxed = st.data_editor(
            _fxd, hide_index=True, use_container_width=True, key="_risk_fx_sel",
            column_config={"Include": st.column_config.CheckboxColumn(
                "✓", help="Include in the Aggregated positions table above")},
            disabled=["Ccy", "Side", "Balance", "USD/unit", "USD exposure", "Today PnL", "Realised FX"],
        )
        sel_fx_ccys = set(_fxed.loc[_fxed["Include"] == True, "Ccy"])
        st.session_state["_risk_pending_fx"] = sel_fx_ccys
        n_live = int((fx["rate_source"] == "live").sum())
        n_fb = int((fx["rate_source"] == "activity").sum())
        rate_note = f"live spot ({n_live} live" + (f", {n_fb} fell back to last activity" if n_fb else "") + ")"
        st.caption(
            f"USD/unit rates: **{rate_note}**, refreshed every 10 min. "
            f"**Realised FX PnL is cumulative over {span_txt}** — not daily. "
            "The FX PnL that matters *daily* is the mark-to-market on these balances (unrealised), "
            "which comes with the price-history feed. Unrealised MTM also needs the FX Lots section in your Flex query."
        )

    st.divider()
    st.caption("Tick the ✓ boxes in the **Futures** and **FX** tables above, then hit **💾 Save selection** "
               "at the top — the live **Aggregated positions** table will track your saved set (no re-selecting "
               "on refresh). Update from IBKR + re-tick only when positions change.")


if __name__ == "__main__":
    # 1) Book self-test against the live DB.
    bk = build_speculative_book()
    cols = ["Symbol", "Underlying", "AssetClass", "Currency", "side",
            "Quantity", "MarkPrice", "position_value_base", "upnl_base"]
    print("--- SPECULATIVE BOOK ---")
    print(bk[cols].to_string(index=False))
    print()
    for k, v in book_metrics(bk).items():
        print(f"{k}: {v}")

    # 2) VaR math self-test on a synthetic Δmark matrix (until the Databento
    #    history provider is wired in Phase 2). Demonstrates that the Euribor
    #    calendar spread's correlated legs collapse the portfolio VaR.
    print("\n--- VaR SELF-TEST (synthetic dMark, 500 days) ---")
    rng = np.random.default_rng(0)
    n = 500
    # daily price-change vols (in each contract's price units), roughly realistic:
    #   STIR ~1.5bp = 0.015 pts; BTC ~ $2500; Gold ~ $45
    euri = rng.normal(0, 0.015, n)           # shared Euribor factor
    dmark = pd.DataFrame({
        "SO3H6": rng.normal(0, 0.015, n),
        "IZ7":   euri + rng.normal(0, 0.002, n),   # ~0.99 corr with IZ6
        "IZ6":   euri + rng.normal(0, 0.002, n),
        "MBTM6": rng.normal(0, 2500.0, n),
        "MGCV6": rng.normal(0, 45.0, n),
    })
    for meth in ("historical", "parametric"):
        rep = var_breakdown(bk, dmark, method=meth, cl=0.95)
        if not rep:
            continue
        print(f"\n[{meth}] 95% 1-day VaR")
        print(f"  portfolio            : ${rep['portfolio_var']:,.0f}")
        print(f"  undiversified sum    : ${rep['undiversified_sum']:,.0f}")
        print(f"  diversification saved: ${rep['diversification_benefit']:,.0f}")
        for s, v in sorted(rep["standalone"].items(), key=lambda x: -x[1]):
            print(f"    {s:7} standalone : ${v:,.0f}")

"""
Bloomberg-style watchlist — clickable rows, inline charts, persistent config.
"""

import io
import json
import os
import pathlib
import sqlite3
import urllib.request
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import ta as ta_lib
import yfinance as yf
from datetime import datetime, timedelta, date

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist_config.json")

ALL_COLUMNS     = ["Price", "Change %", "Change", "Weekly %", "Monthly %", "YTD %", "RSI (14)", "RSI (30)", "52W High", "52W Low", "Volume"]
DEFAULT_COLUMNS = ["Price", "Change %", "Weekly %", "Monthly %", "YTD %", "RSI (14)", "RSI (30)", "52W High", "52W Low"]
ASSET_CLASSES   = ["Equity", "FX", "Rates", "Commodity", "Crypto", "Custom", "Other"]

CHART_TIMEFRAMES = {
    "1D":  ("1d",  "5m"),
    "1W":  ("5d",  "15m"),
    "1M":  ("1mo", "1h"),
    "3M":  ("3mo", "1d"),
    "6M":  ("6mo", "1d"),
    "1Y":  ("1y",  "1d"),
    "2Y":  ("2y",  "1d"),
    "5Y":  ("5y",  "1d"),
    "10Y": ("10y", "1d"),
    "Custom": (None, "1d"),
}

DEFAULT_INSTRUMENTS = [
    {"name": "S&P 500",      "ticker": "^GSPC",    "class": "Equity"},
    {"name": "NASDAQ",       "ticker": "^IXIC",    "class": "Equity"},
    {"name": "FTSE 100",     "ticker": "^FTSE",    "class": "Equity"},
    {"name": "DAX",          "ticker": "^GDAXI",   "class": "Equity"},
    {"name": "Dow Jones",    "ticker": "^DJI",     "class": "Equity"},
    {"name": "EUR/USD",      "ticker": "EURUSD=X", "class": "FX"},
    {"name": "GBP/USD",      "ticker": "GBPUSD=X", "class": "FX"},
    {"name": "USD/JPY",      "ticker": "JPY=X",    "class": "FX"},
    {"name": "USD/CHF",      "ticker": "CHF=X",    "class": "FX"},
    {"name": "US 2Y",        "ticker": "^US2YT",   "class": "Rates"},
    {"name": "US 5Y",        "ticker": "^FVX",     "class": "Rates"},
    {"name": "US 10Y",       "ticker": "^TNX",     "class": "Rates"},
    {"name": "US 30Y",       "ticker": "^TYX",     "class": "Rates"},
    {"name": "EUR 2Y",       "ticker": "^ECB2Y",   "class": "Rates"},
    {"name": "EUR 5Y",       "ticker": "^ECB5Y",   "class": "Rates"},
    {"name": "EUR 10Y",      "ticker": "^ECB10Y",  "class": "Rates"},
    {"name": "EUR 30Y",      "ticker": "^ECB30Y",  "class": "Rates"},
    {"name": "Gold",         "ticker": "GC=F",     "class": "Commodity"},
    {"name": "WTI Oil",      "ticker": "CL=F",     "class": "Commodity"},
    {"name": "Brent Crude",  "ticker": "BZ=F",     "class": "Commodity"},
    {"name": "Silver",       "ticker": "SI=F",     "class": "Commodity"},
    {"name": "Bitcoin",      "ticker": "BTC-USD",  "class": "Crypto"},
    {"name": "Ethereum",     "ticker": "ETH-USD",  "class": "Crypto"},
]

CLASS_BG = {
    "Equity":    "#F0F5FF",
    "FX":        "#F0FDF4",
    "Rates":     "#FFFBEB",
    "Commodity": "#FFF4ED",
    "Crypto":    "#FAF5FF",
    "Custom":    "#F0FAFA",
    "Other":     "#F8FAFC",
}
CLASS_FG = {
    "Equity":    "#1D4ED8",
    "FX":        "#047857",
    "Rates":     "#B45309",
    "Commodity": "#C2410C",
    "Crypto":    "#6D28D9",
    "Custom":    "#0F766E",
    "Other":     "#475569",
}
UP   = "#059669"
DOWN = "#DC2626"
MUTE = "#94A3B8"

# ── Custom series helpers (avoid circular import with analyzer.py) ─────────────
_HERE_WL    = pathlib.Path(__file__).parent
_CUSTOM_FILE_WL = str(_HERE_WL / "custom_series.json")
_STIR_DB_WL     = str(_HERE_WL / "stir_bars.db")
_COMMOD_DB_WL   = str(_HERE_WL / "commodities_bars.db")
_CS_SAFE = {"abs": abs, "log": np.log, "log10": np.log10, "exp": np.exp, "sqrt": np.sqrt}


def _load_cs_json() -> list[dict]:
    try:
        with open(_CUSTOM_FILE_WL, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _wl_fetch_key(key: str) -> pd.Series:
    """Fetch a single component series for a custom-series formula."""
    start = (date.today() - timedelta(days=_DAILY_LOOKBACK)).isoformat()
    if key.startswith("stir:") or key.startswith("commod:"):
        parts = key.split(":")
        _, sym, exch, exp6 = parts
        db = _STIR_DB_WL if key.startswith("stir:") else _COMMOD_DB_WL
        table = "stir_bars" if key.startswith("stir:") else "bars"
        try:
            conn = sqlite3.connect(db)
            rows = conn.execute(
                f"SELECT bar_date, close FROM {table} "
                "WHERE symbol=? AND exchange=? AND exp6=? ORDER BY bar_date",
                (sym, exch, exp6),
            ).fetchall()
            conn.close()
        except Exception:
            return pd.Series(dtype=float)
        if not rows:
            return pd.Series(dtype=float)
        df = pd.DataFrame(rows, columns=["bar_date", "close"])
        df["bar_date"] = pd.to_datetime(df["bar_date"])
        df.set_index("bar_date", inplace=True)
        s = df["close"].astype(float)
        return s[s.index >= pd.Timestamp(start)]
    # Yahoo Finance / FRED / ECB etc — reuse _raw_daily (defined later in this file)
    df = _raw_daily(key)
    if df.empty:
        return pd.Series(dtype=float)
    col = df["Close"]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col.astype(float)


@st.cache_data(ttl=300)
def _fetch_custom_df_cached(cs_id: str) -> pd.DataFrame:
    """Evaluate a custom series formula and return a Close-column DataFrame."""
    cs = next((c for c in _load_cs_json() if c["id"] == cs_id), None)
    if cs is None:
        return pd.DataFrame()
    raw: dict[str, pd.Series] = {}
    for var, spec in cs.get("series", {}).items():
        s = _wl_fetch_key(spec["key"])
        if not s.empty:
            s = s.copy()
            s.index = pd.to_datetime(s.index).normalize()
            raw[var] = s
    if not raw:
        return pd.DataFrame()
    combined = pd.concat(raw, axis=1).dropna()
    if combined.empty:
        return pd.DataFrame()
    local_ns = {k: combined[k] for k in raw if k in combined.columns}
    try:
        result = eval(cs["formula"], {"__builtins__": None}, {**_CS_SAFE, **local_ns})
        if not isinstance(result, pd.Series):
            return pd.DataFrame()
        df = result.to_frame(name="Close")
        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"]  = df["Close"]
        df["Volume"] = 0
        return df
    except Exception:
        return pd.DataFrame()


# ── FRED helpers ───────────────────────────────────────────────────────────────
FRED_MAP = {
    "^GUKG10": "IRLTLT01GBM156N",   # UK 10Y (monthly, OECD via FRED)
    "^US2YT":  "DGS2",              # US 2Y constant maturity (daily, FRED)
    "^US5YR":  "DFII5",             # US 5Y real yield / TIPS (daily, FRED)
    "^US10YR": "DFII10",            # US 10Y real yield / TIPS (daily, FRED)
    "^US5YBE": "T5YIE",             # US 5Y breakeven inflation (daily, FRED)
    "^US10YBE":"T10YIE",            # US 10Y breakeven inflation (daily, FRED)
    "CL=F":    "DCOILWTICO",        # WTI crude oil spot (daily, FRED — better history than yfinance CL=F)
    "BZ=F":    "DCOILBRENTEU",      # Brent crude oil spot (daily, FRED)
}

JGB_MAP = {
    "^JPY10Y": "10Y",   # Japan 10Y JGB — MOF daily via rates.fetch_japan_jgb
}

ECB_MAP = {
    "^ECB2Y":  "SR_2Y",    # ECB AAA euro area spot rate 2Y
    "^ECB5Y":  "SR_5Y",    # ECB AAA euro area spot rate 5Y
    "^ECB10Y": "SR_10Y",   # ECB AAA euro area spot rate 10Y
    "^ECB30Y": "SR_30Y",   # ECB AAA euro area spot rate 30Y
}

ALPHAVANTAGE_FX_MAP = {
    "USDCNH=X": ("USD", "CNH"),   # offshore yuan — no Yahoo Finance history
}

# Synthetic BBDXY: geometric weighted basket of 12 USD pairs.
# invert=True means Yahoo ticker is CCY/USD (e.g. EUR/USD) so we flip to USD/CCY.
BBDXY_WEIGHTS = {
    "EURUSD=X": (0.2947, True),
    "JPY=X":    (0.1238, False),
    "CAD=X":    (0.1165, False),
    "GBPUSD=X": (0.1027, True),
    "MXN=X":    (0.0962, False),
    "USDCNH=X": (0.0700, False),
    "CHF=X":    (0.0447, False),
    "AUDUSD=X": (0.0439, True),
    "KRW=X":    (0.0316, False),
    "INR=X":    (0.0283, False),
    "SGD=X":    (0.0261, False),
    "TWD=X":    (0.0215, False),
}


@st.cache_data(ttl=3600)
def _fetch_fred_df_cached(series_id: str, start: str) -> pd.DataFrame:
    """Raises on any failure so Streamlit does not cache empty/error results."""
    key = st.secrets["FRED_KEY"]          # KeyError propagates → not cached
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={key}&file_type=json"
        f"&sort_order=asc&observation_start={start}"
    )
    with urllib.request.urlopen(url, timeout=10) as r:
        obs = json.loads(r.read().decode())["observations"]
    rows = [(o["date"], float(o["value"])) for o in obs if o["value"] != "."]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df["Open"]   = df["Close"]
    df["High"]   = df["Close"]
    df["Low"]    = df["Close"]
    df["Volume"] = 0
    return df


def _fetch_fred_df(series_id: str, start: str = "2020-01-01") -> pd.DataFrame:
    """Graceful wrapper — returns empty DataFrame on any error."""
    try:
        return _fetch_fred_df_cached(series_id, start)
    except Exception:
        return pd.DataFrame()


def _fred_period_start(period: str | None) -> str:
    days = {"1d": 60, "5d": 60, "1mo": 90, "3mo": 100,
            "6mo": 190, "1y": 370, "2y": 740, "5y": 1830, "10y": 3650,
            "20y": 7320, "max": 36500}
    return (date.today() - timedelta(days=days.get(period or "1y", 370))).isoformat()


@st.cache_data(ttl=3600)
def _fetch_jgb_df(col: str, start: str = "2020-01-01") -> pd.DataFrame:
    """Convert rates.fetch_japan_jgb() into the standard OHLCV frame used by watchlist."""
    try:
        from rates import fetch_japan_jgb
        df_jgb = fetch_japan_jgb()
    except Exception:
        return pd.DataFrame()
    if df_jgb.empty or col not in df_jgb.columns:
        return pd.DataFrame()
    s = df_jgb[col].dropna()
    s = s[s.index >= pd.Timestamp(start)]
    if s.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"Close": s, "Open": s, "High": s, "Low": s})
    out["Volume"] = 0
    return out


@st.cache_data(ttl=3600)
def _fetch_ecb_df(maturity_code: str, start: str = "2020-01-01") -> pd.DataFrame:
    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        f"YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{maturity_code}"
        f"?format=csvdata&startPeriod={start}"
    )
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            tmp = pd.read_csv(io.StringIO(r.read().decode()))
        tmp = tmp[["TIME_PERIOD", "OBS_VALUE"]].copy()
        tmp.columns = ["Date", "Close"]
        tmp["Date"] = pd.to_datetime(tmp["Date"])
        tmp["Close"] = pd.to_numeric(tmp["Close"], errors="coerce")
        tmp.set_index("Date", inplace=True)
        tmp.dropna(inplace=True)
        tmp["Open"]   = tmp["Close"]
        tmp["High"]   = tmp["Close"]
        tmp["Low"]    = tmp["Close"]
        tmp["Volume"] = 0
        return tmp
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _fetch_alphavantage_fx(from_sym: str, to_sym: str, start: str = "2020-01-01") -> pd.DataFrame:
    try:
        key = st.secrets.get("ALPHAVANTAGE_KEY", "SRJZBTK9RHQ2KDJE")
        url = (
            f"https://www.alphavantage.co/query?function=FX_DAILY"
            f"&from_symbol={from_sym}&to_symbol={to_sym}"
            f"&outputsize=full&apikey={key}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        ts = data.get("Time Series FX (Daily)", {})
        if not ts:
            return pd.DataFrame()
        rows = []
        for d, v in ts.items():
            rows.append({
                "Date":   pd.Timestamp(d),
                "Open":   float(v["1. open"]),
                "High":   float(v["2. high"]),
                "Low":    float(v["3. low"]),
                "Close":  float(v["4. close"]),
                "Volume": 0,
            })
        df = pd.DataFrame(rows).set_index("Date").sort_index()
        df = df[df.index >= pd.Timestamp(start)]
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def _compute_bbdxy(start: str) -> pd.DataFrame:
    """Synthetic BBDXY: geometric weighted mean of 12 USD pairs, base 1200."""
    frames: dict = {}
    weights_used: dict = {}
    for tkr, (w, invert) in BBDXY_WEIGHTS.items():
        try:
            if tkr in ALPHAVANTAGE_FX_MAP:
                from_sym, to_sym = ALPHAVANTAGE_FX_MAP[tkr]
                df = _fetch_alphavantage_fx(from_sym, to_sym, start=start)
            else:
                df = yf.download(tkr, start=start, interval="1d",
                                 auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[df["Close"].notna()]
            if df.empty:
                continue
            _sc = df["Close"]
            s = (_sc.iloc[:, 0] if isinstance(_sc, pd.DataFrame) else _sc).astype(float)
            if invert:
                s = 1.0 / s
            idx = pd.DatetimeIndex(s.index)
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            s.index = idx.normalize()
            s = s[~s.index.duplicated(keep="last")]
            frames[tkr] = s
            weights_used[tkr] = w
        except Exception:
            continue

    if len(frames) < 6:
        return pd.DataFrame()

    combined = pd.DataFrame(frames).dropna(how="any")
    if combined.empty or len(combined) < 5:
        return pd.DataFrame()

    total_w = sum(weights_used[t] for t in combined.columns)
    log_index = sum(
        (weights_used[tkr] / total_w) * np.log(combined[tkr])
        for tkr in combined.columns
    )
    product = np.exp(log_index)
    index_vals = product / float(product.iloc[0]) * 1200.0

    return pd.DataFrame({
        "Open":   index_vals,
        "High":   index_vals,
        "Low":    index_vals,
        "Close":  index_vals,
        "Volume": pd.Series(0, index=index_vals.index),
    })


# ── Single source of truth for daily OHLCV ────────────────────────────────────
_DAILY_LOOKBACK = 670   # 300-day SMA warm-up + 370-day 1Y window

def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    """tz-strip → normalize to midnight → dedupe by date (keep last).
    Exactly the treatment the original _raw_daily applied to every frame."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    try:
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        df.index = idx.normalize()
        return df[~df.index.duplicated(keep="last")]
    except Exception:
        return df


def _yahoo_daily_single(ticker: str, start: str) -> pd.DataFrame:
    """Per-ticker Yahoo fallback chain: start= → period='1y' → Ticker.history.
    Identical to the original _raw_daily download logic, used to retry the
    hard-to-fetch tickers that came back empty from the batch download."""
    try:
        df = yf.download(ticker, start=start, interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[df["Close"].notna()]
        if df.empty:
            # Some tickers don't support start= — fall back to period
            df = yf.download(ticker, period="1y", interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[df["Close"].notna()]
        if df.empty:
            # Last resort: Ticker.history() uses a different API path
            df = yf.Ticker(ticker).history(period="1y", interval="1d",
                                           auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[df["Close"].notna()]
        return _normalize_daily(df)
    except Exception:
        return pd.DataFrame()


def _extract_ticker_frame(raw, ticker: str) -> pd.DataFrame:
    """Pull a single ticker's OHLCV frame out of a multi-ticker yf.download
    result (group_by='ticker'). Handles MultiIndex keyed by ticker at either
    column level and the flat-column shape a single-element list can yield.
    Returns an empty frame for a failed (all-NaN) ticker."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    try:
        cols = raw.columns
        if isinstance(cols, pd.MultiIndex):
            lvl0 = set(cols.get_level_values(0))
            lvl1 = set(cols.get_level_values(1))
            if ticker in lvl0:
                sub = raw[ticker].copy()
            elif ticker in lvl1:
                sub = raw.xs(ticker, axis=1, level=1).copy()
            else:
                return pd.DataFrame()
        else:
            # Flat columns — only happens when effectively one ticker came back
            sub = raw.copy()
        if "Close" not in sub.columns:
            return pd.DataFrame()
        # Failed tickers in a batch come back all-NaN → this drops every row
        sub = sub[sub["Close"].notna()]
        return sub
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _all_daily(tickers: tuple) -> dict:
    """Single source of truth for daily OHLCV. Returns {ticker: OHLCV DataFrame}.

    Special sources route to their own cached helpers; every plain Yahoo ticker
    is fetched in ONE batched yf.download and split per-ticker. Failed tickers
    are retried through the per-ticker fallback chain so hard-to-fetch symbols
    keep working exactly as before. Only frames with >5 rows are kept.
    Because table and chart both read this same cache entry, RSI/indicators are
    always computed from byte-identical price data in both places."""
    if not tickers:
        return {}
    start = (date.today() - timedelta(days=_DAILY_LOOKBACK)).isoformat()
    out: dict = {}
    yahoo: list = []

    for tkr in tickers:
        if tkr == "BBDXY_SYNTH":
            df = _compute_bbdxy(start=start)
            if df is not None and not df.empty:
                out[tkr] = df
        elif tkr in FRED_MAP:
            df = _fetch_fred_df(FRED_MAP[tkr], start=start)
            if not df.empty:
                out[tkr] = df
        elif tkr in JGB_MAP:
            df = _fetch_jgb_df(JGB_MAP[tkr], start=start)
            if not df.empty:
                out[tkr] = df
        elif tkr in ECB_MAP:
            df = _fetch_ecb_df(ECB_MAP[tkr], start=start)
            if not df.empty:
                out[tkr] = df
        elif tkr in ALPHAVANTAGE_FX_MAP:
            from_sym, to_sym = ALPHAVANTAGE_FX_MAP[tkr]
            df = _fetch_alphavantage_fx(from_sym, to_sym, start=start)
            if not df.empty:
                out[tkr] = df
        else:
            yahoo.append(tkr)

    # ── One batched request for every plain Yahoo ticker ─────────────────────
    missing: list = []
    if yahoo:
        try:
            raw = yf.download(
                yahoo, start=start, interval="1d", auto_adjust=True,
                progress=False, group_by="ticker", threads=True,
            )
        except Exception:
            raw = None
        for tkr in yahoo:
            df = _normalize_daily(_extract_ticker_frame(raw, tkr))
            if df is not None and not df.empty:
                out[tkr] = df
            else:
                missing.append(tkr)

    # ── Retry the tickers the batch could not resolve, one at a time ─────────
    for tkr in missing:
        df = _yahoo_daily_single(tkr, start)
        if not df.empty:
            out[tkr] = df

    return {t: d for t, d in out.items() if d is not None and not d.empty and len(d) > 5}


def _watchlist_tickers() -> tuple:
    """The full ticker tuple the watchlist table fetches — same config the table
    reads. Used so _raw_daily hits the exact _all_daily entry the table populated."""
    try:
        return tuple(i["ticker"] for i in load_config().get("instruments", []))
    except Exception:
        return ()


def _raw_daily(ticker: str) -> pd.DataFrame:
    """670 days of clean daily OHLCV for one ticker. Reads from the shared
    _all_daily cache so table and chart always use byte-identical price data.

    If the ticker is on the watchlist it returns its slice of the table's own
    batch entry (a cache hit — zero extra network). Custom-series formulas may
    reference arbitrary tickers not on the watchlist; those fall back to a
    single-ticker batch."""
    wl = _watchlist_tickers()
    if ticker in wl:
        return _all_daily(wl).get(ticker, pd.DataFrame())
    return _all_daily((ticker,)).get(ticker, pd.DataFrame())


# ── Config persistence ─────────────────────────────────────────────────────────
def load_config() -> dict:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
            cfg.setdefault("instruments",       DEFAULT_INSTRUMENTS)
            cfg.setdefault("columns",           DEFAULT_COLUMNS)
            cfg.setdefault("custom_series_ids", [])
            return cfg
    except Exception:
        pass
    return {"instruments": DEFAULT_INSTRUMENTS.copy(), "columns": DEFAULT_COLUMNS.copy()}


def save_config(cfg: dict) -> str:
    """Save config locally and, if GITHUB_TOKEN is set, commit to the repo.
    Returns 'remote' | 'local' | 'error'."""
    import base64
    import urllib.error

    content_str = json.dumps(cfg, indent=2)

    # ── GitHub API (persistent across Streamlit Cloud restarts) ──────────────
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo  = st.secrets.get("GITHUB_REPO", "rajatb1-cyber/market-dashboard")
        if token:
            api_url = f"https://api.github.com/repos/{repo}/contents/watchlist_config.json"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "Content-Type": "application/json",
            }
            # Fetch current SHA (required for update)
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                sha = json.loads(resp.read())["sha"]

            payload = json.dumps({
                "message": "chore: update watchlist config",
                "content": base64.b64encode(content_str.encode()).decode(),
                "sha": sha,
            }).encode()
            req = urllib.request.Request(api_url, data=payload, method="PUT",
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=20):
                pass

            # Also write locally so the current process sees it immediately
            try:
                with open(CONFIG_FILE, "w") as f:
                    f.write(content_str)
            except Exception:
                pass
            return "remote"
    except Exception:
        pass

    # ── Local fallback ────────────────────────────────────────────────────────
    try:
        with open(CONFIG_FILE, "w") as f:
            f.write(content_str)
        return "local"
    except Exception:
        return "error"


# ── Watchlist data ─────────────────────────────────────────────────────────────
def fetch_batch(tickers: tuple) -> dict:
    """Thin wrapper over the shared _all_daily cache (the single source of truth).
    Not itself cached — _all_daily is the cache; no caller relies on
    fetch_batch.clear()."""
    if not tickers:
        return {}
    return _all_daily(tuple(tickers))


def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) else v
    except Exception:
        return None


def compute_row(inst: dict, df, columns: list) -> dict:
    row = {"Name": inst["name"], "Class": inst["class"], "_ticker": inst["ticker"]}
    if df is None or df.empty:
        for c in columns: row[c] = None
        return row

    _c = df["Close"]
    if isinstance(_c, pd.DataFrame):
        _c = _c.iloc[:, 0]
    close = _c.dropna().astype(float)
    if close.empty:
        for c in columns: row[c] = None
        return row

    last = _f(close.iloc[-1])
    if last is None:
        for c in columns: row[c] = None
        return row

    prev  = _f(close.iloc[-2]) if len(close) > 1 else last
    d_chg = last - prev if prev else 0
    d_pct = (d_chg / prev * 100) if prev else 0

    is_rate = inst.get("class") == "Rates"

    for c in columns:
        if c == "Price":
            row[c] = last
        elif c == "Change %":
            row[c] = d_chg * 100 if is_rate else d_pct
        elif c == "Change":
            row[c] = d_chg * 100 if is_rate else d_chg
        elif c == "Weekly %":
            ref = _f(close.iloc[-6]) if len(close) > 5 else _f(close.iloc[0])
            if is_rate:
                row[c] = ((last - ref) * 100) if ref is not None else None
            else:
                row[c] = ((last - ref) / ref * 100) if ref else None
        elif c == "Monthly %":
            ref = _f(close.iloc[-22]) if len(close) > 21 else _f(close.iloc[0])
            if is_rate:
                row[c] = ((last - ref) * 100) if ref is not None else None
            else:
                row[c] = ((last - ref) / ref * 100) if ref else None
        elif c == "YTD %":
            cur_year = close.index[-1].year
            prev_year = close[close.index.year < cur_year]
            ref = _f(prev_year.iloc[-1]) if not prev_year.empty else _f(close.iloc[0])
            if is_rate:
                row[c] = ((last - ref) * 100) if ref is not None else None
            else:
                row[c] = ((last - ref) / ref * 100) if ref else None
        elif c == "RSI (14)":
            try:
                rsi = ta_lib.momentum.RSIIndicator(close, window=14).rsi().dropna()
                row[c] = _f(rsi.iloc[-1]) if not rsi.empty else None
            except Exception:
                row[c] = None
        elif c == "RSI (30)":
            try:
                rsi = ta_lib.momentum.RSIIndicator(close, window=30).rsi().dropna()
                row[c] = _f(rsi.iloc[-1]) if not rsi.empty else None
            except Exception:
                row[c] = None
        elif c == "52W High":
            row[c] = _f(close.max())
        elif c == "52W Low":
            row[c] = _f(close.min())
        elif c == "Volume":
            try:
                _v = df["Volume"]
                vol = (_v.iloc[:, 0] if isinstance(_v, pd.DataFrame) else _v).dropna()
                row[c] = _f(vol.iloc[-1]) if not vol.empty else None
            except Exception:
                row[c] = None
    return row


# ── Table formatting ───────────────────────────────────────────────────────────
def _fmt(val, col: str, cls: str = "") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if col in ("Change %", "Weekly %", "Monthly %", "YTD %"):
        if cls == "Rates":
            return f"{int(round(val)):+d}bp"
        return f"{val:+.2f}%"
    if col == "Change":
        if cls == "Rates":
            return f"{int(round(val)):+d}bp"
        return f"{val:+.4f}" if -1 < val < 1 else f"{val:+.2f}"
    if col in ("RSI (14)", "RSI (30)"):
        return f"{val:.1f}"
    if col == "Volume":
        if val >= 1e9: return f"{val/1e9:.1f}B"
        if val >= 1e6: return f"{val/1e6:.1f}M"
        if val >= 1e3: return f"{val/1e3:.0f}K"
        return f"{val:.0f}"
    if abs(val) < 10:   return f"{val:.4f}"
    if abs(val) < 10000: return f"{val:,.2f}"
    return f"{val:,.0f}"


def build_display_df(df_rows: pd.DataFrame, active_cols: list) -> pd.DataFrame:
    """Build a string-formatted display DataFrame for st.dataframe."""
    out = pd.DataFrame()
    out["Name"]  = df_rows["Name"]
    out["Class"] = df_rows["Class"]
    for c in active_cols:
        if c in df_rows.columns:
            out[c] = [_fmt(v, c, cls) for v, cls in zip(df_rows[c], df_rows["Class"])]
    return out.reset_index(drop=True)


def style_table(df_display: pd.DataFrame, active_cols: list):
    """Apply green/red text to change columns via pandas Styler."""
    pct_cols = [c for c in active_cols if c in ("Change %", "Change", "Weekly %", "Monthly %", "YTD %")]
    rsi_cols = [c for c in active_cols if c in ("RSI (14)", "RSI (30)")]

    def color_change(val):
        try:
            f = float(val)
            if not np.isnan(f):
                if f > 0: return f"color: {UP}; font-weight: 600"
                if f < 0: return f"color: {DOWN}; font-weight: 600"
        except (TypeError, ValueError):
            pass
        return ""

    def color_rsi(val: str):
        try:
            v = float(val)
            if v > 70: return f"color: {DOWN}; font-weight: 600"
            if v < 30: return f"color: {UP}; font-weight: 600"
        except Exception:
            pass
        return ""

    def color_class(val: str):
        return f"color: {CLASS_FG.get(val, MUTE)}; font-weight: 600"

    styler = df_display.style
    if pct_cols:
        styler = styler.map(color_change, subset=pct_cols)
    if rsi_cols:
        styler = styler.map(color_rsi, subset=rsi_cols)
    styler = styler.map(color_class, subset=["Class"])
    styler = styler.set_properties(**{
        "font-family": "Inter, Segoe UI, sans-serif",
        "font-size":   "13px",
    })
    styler = styler.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#1B3A6B"),
            ("color", "white"),
            ("font-weight", "700"),
            ("font-size", "12px"),
            ("padding", "8px 12px"),
            ("text-align", "left"),
        ]},
        {"selector": "td", "props": [("padding", "7px 12px")]},
        {"selector": "tr:hover td", "props": [("background-color", "#EFF6FF !important")]},
    ])
    return styler


# ── Chart for selected instrument ──────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_chart_data(ticker: str, period: str | None, interval: str,
                     rsi_period: int = 14,
                     start: str | None = None, end: str | None = None) -> pd.DataFrame:
    try:
        trim_to = None
        _period_days_map = {
            "1d": 2, "5d": 7, "1mo": 35, "3mo": 100, "6mo": 190,
            "1y": 370, "2y": 740, "5y": 1830, "10y": 3660,
            "20y": 7320, "max": 36500,
        }
        # FRED / ECB / BBDXY_SYNTH: always daily data regardless of requested interval
        if ticker == "BBDXY_SYNTH":
            lookback = _period_days_map.get(period, 370) + 300
            bbdxy_start = start if start else (date.today() - timedelta(days=lookback)).isoformat()
            df = _compute_bbdxy(start=bbdxy_start)
            if not df.empty:
                trim_to = date.today() - timedelta(days=_period_days_map.get(period, 370)) if period else None
                if end:
                    df = df[df.index <= pd.Timestamp(end)]
        elif ticker in FRED_MAP:
            fred_start = start if start else _fred_period_start(period)
            df = _fetch_fred_df(FRED_MAP[ticker], start=fred_start)
            if not df.empty and end:
                df = df[df.index <= pd.Timestamp(end)]
            df.dropna(inplace=True)
        elif ticker in JGB_MAP:
            jgb_start = start if start else _fred_period_start(period)
            df = _fetch_jgb_df(JGB_MAP[ticker], start=jgb_start)
            if not df.empty and end:
                df = df[df.index <= pd.Timestamp(end)]
            df.dropna(inplace=True)
        elif ticker in ECB_MAP:
            ecb_start = start if start else _fred_period_start(period)
            df = _fetch_ecb_df(ECB_MAP[ticker], start=ecb_start)
            if not df.empty and end:
                df = df[df.index <= pd.Timestamp(end)]
            df.dropna(inplace=True)
        elif start and end:
            df = yf.download(ticker, start=start, end=end, interval=interval,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            trim_to = None
        else:
            _period_days = {
                "1d": 2, "5d": 7, "1mo": 35, "3mo": 100, "6mo": 190,
                "1y": 370, "2y": 740, "5y": 1830, "10y": 3660,
                "20y": 7320, "max": 36500,
            }
            # For daily ≤1Y views reuse the same cached daily data that the
            # watchlist table uses — this guarantees RSI/indicators are computed
            # from identical rows so table and chart values always match.
            if interval == "1d" and period in ("3mo", "6mo", "1y"):
                df = _raw_daily(ticker).copy()
                trim_to = date.today() - timedelta(days=_period_days[period])
            else:
                # 2Y/5Y/10Y and intraday: fetch an extended window so SMA(200)
                # is fully warmed up across the entire visible range.
                _warmup = {"1d": 300}
                warmup = _warmup.get(interval, 0)
                if interval == "1d" and ticker in ALPHAVANTAGE_FX_MAP:
                    # Alpha Vantage tickers: use full history, trim after indicators
                    from_sym, to_sym = ALPHAVANTAGE_FX_MAP[ticker]
                    ext_start = (date.today() - timedelta(days=_period_days.get(period, 370) + warmup)).isoformat()
                    df = _fetch_alphavantage_fx(from_sym, to_sym, start=ext_start)
                    trim_to = date.today() - timedelta(days=_period_days.get(period, 370))
                elif warmup and period in _period_days:
                    ext_start = (date.today() - timedelta(days=_period_days[period] + warmup)).isoformat()
                    df = yf.download(ticker, start=ext_start, interval=interval,
                                     auto_adjust=True, progress=False)
                    trim_to = date.today() - timedelta(days=_period_days[period])
                else:
                    df = yf.download(ticker, period=period, interval=interval,
                                     auto_adjust=True, progress=False)
                    trim_to = None
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.dropna(inplace=True)
        # Deduplicate: yfinance can return two rows for the same calendar date
        # with different tz offsets (e.g. +00:00 vs -03:00). Convert to UTC,
        # strip tz, then deduplicate so every date appears exactly once.
        if not df.empty and interval in ("1d", "1wk", "1mo"):
            try:
                idx = pd.DatetimeIndex(df.index)
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                df.index = idx.normalize()
                df = df[~df.index.duplicated(keep="last")]
            except Exception:
                pass
        if not df.empty:
            _cc = df["Close"]
            close = (_cc.iloc[:, 0] if isinstance(_cc, pd.DataFrame) else _cc).astype(float)
            df["SMA20"]  = ta_lib.trend.SMAIndicator(close, window=20).sma_indicator()
            df["SMA50"]  = ta_lib.trend.SMAIndicator(close, window=50).sma_indicator()
            df["SMA100"] = ta_lib.trend.SMAIndicator(close, window=100).sma_indicator()
            df["SMA200"] = ta_lib.trend.SMAIndicator(close, window=200).sma_indicator()
            df["EMA20"]  = ta_lib.trend.EMAIndicator(close, window=20).ema_indicator()
            bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
            df["BB_upper"] = bb.bollinger_hband()
            df["BB_lower"] = bb.bollinger_lband()
            # RSI always on daily bars regardless of chart interval so it
            # matches the watchlist table value.
            if interval in ("1d", "1wk", "1mo"):
                df["RSI"]   = ta_lib.momentum.RSIIndicator(close, window=rsi_period).rsi()
                df["RSI30"] = ta_lib.momentum.RSIIndicator(close, window=30).rsi()
            else:
                # Intraday chart: place daily RSI only at the last bar of each
                # trading day (NaN elsewhere). Plotly connects the dots with a
                # line, giving clean daily-resolution RSI on an intraday chart.
                _d = _raw_daily(ticker)
                if not _d.empty:
                    _dc = (_d["Close"].iloc[:, 0] if isinstance(_d["Close"], pd.DataFrame) else _d["Close"]).astype(float)
                    _r14 = ta_lib.momentum.RSIIndicator(_dc, window=rsi_period).rsi()
                    _r30 = ta_lib.momentum.RSIIndicator(_dc, window=30).rsi()
                    _didx = pd.DatetimeIndex(_r14.index)
                    if _didx.tz is not None:
                        _didx = _didx.tz_convert("UTC").tz_localize(None)
                    _map14 = dict(zip(_didx.normalize(), _r14.values))
                    _map30 = dict(zip(_didx.normalize(), _r30.values))
                    _cidx = pd.DatetimeIndex(df.index)
                    if _cidx.tz is not None:
                        _cdates = _cidx.tz_convert("UTC").tz_localize(None).normalize()
                    else:
                        _cdates = _cidx.normalize()
                    rsi14_col = [np.nan] * len(df)
                    rsi30_col = [np.nan] * len(df)
                    for i, d in enumerate(_cdates):
                        if i == len(_cdates) - 1 or _cdates[i + 1] != d:
                            rsi14_col[i] = _map14.get(d, np.nan)
                            rsi30_col[i] = _map30.get(d, np.nan)
                    df["RSI"]   = rsi14_col
                    df["RSI30"] = rsi30_col
                else:
                    df["RSI"]   = ta_lib.momentum.RSIIndicator(close, window=rsi_period).rsi()
                    df["RSI30"] = ta_lib.momentum.RSIIndicator(close, window=30).rsi()
            # Trim warm-up rows — indicators are fully computed, drop the extra history
            if trim_to is not None:
                idx = pd.DatetimeIndex(df.index)
                trim_ts = pd.Timestamp(trim_to)
                if idx.tz is not None:
                    trim_ts = trim_ts.tz_localize(idx.tz)
                df = df[idx >= trim_ts]
        return df
    except Exception:
        return pd.DataFrame()


def get_rangebreaks(df: pd.DataFrame) -> list:
    """
    Detect every gap directly from the data and return Plotly rangebreaks.
    No holiday calendar needed — works for any asset, interval, or timezone.
    """
    if df.empty or len(df) < 2:
        return []
    idx = pd.DatetimeIndex(df.index)
    try:
        idx_naive = idx.tz_localize(None) if idx.tz is not None else idx
    except Exception:
        idx_naive = idx
    try:
        diffs = idx_naive.to_series().diff().dt.total_seconds().dropna()
        pos = diffs[diffs > 0]
        if pos.empty:
            return []
        expected = pos.min()
    except Exception:
        return []
    threshold = expected * 1.5
    breaks = []
    try:
        for i in range(len(idx_naive) - 1):
            gap = (idx_naive[i + 1] - idx_naive[i]).total_seconds()
            if gap > threshold:
                gap_start = idx_naive[i] + pd.Timedelta(seconds=expected)
                gap_end   = idx_naive[i + 1]
                breaks.append(dict(bounds=[
                    gap_start.strftime("%Y-%m-%dT%H:%M:%S"),
                    gap_end.strftime("%Y-%m-%dT%H:%M:%S"),
                ]))
    except Exception:
        pass
    return breaks


def build_instrument_chart(df: pd.DataFrame, name: str, ticker: str,
                            chart_type: str = "Candlestick",
                            overlays: list = None,
                            rsi_period: int = 14) -> go.Figure:
    if overlays is None:
        overlays = ["SMA 20", "SMA 50"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.58, 0.21, 0.21],
    )

    _cc = df["Close"]
    close = _cc.iloc[:, 0] if isinstance(_cc, pd.DataFrame) else _cc

    # ── Price trace ────────────────────────────────────────────────────────
    if chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=df.index, y=close, name="Price",
            line=dict(color="#0F172A", width=3),
        ), row=1, col=1)

    elif chart_type == "OHLC Bar":
        fig.add_trace(go.Ohlc(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name="Price",
            increasing_line_color="#059669",
            decreasing_line_color="#DC2626",
        ), row=1, col=1)

    else:  # Candlestick (default)
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name="Price",
            increasing_line_color="#059669", decreasing_line_color="#DC2626",
            increasing_fillcolor="#059669",  decreasing_fillcolor="#DC2626",
        ), row=1, col=1)

    # ── Overlays ───────────────────────────────────────────────────────────
    if "SMA 20" in overlays and "SMA20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA 20",
            line=dict(color="#F59E0B", width=1.5, dash="dot")), row=1, col=1)

    if "SMA 50" in overlays and "SMA50" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50",
            line=dict(color="#A855F7", width=1.5, dash="dot")), row=1, col=1)

    if "SMA 100" in overlays and "SMA100" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA100"], name="SMA 100",
            line=dict(color="#06B6D4", width=1.5, dash="dot")), row=1, col=1)

    if "SMA 200" in overlays and "SMA200" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA 200",
            line=dict(color="#EF4444", width=2, dash="dot")), row=1, col=1)

    if "EMA 20" in overlays and "EMA20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20",
            line=dict(color="#EC4899", width=1.5, dash="dot")), row=1, col=1)

    if "Bollinger Bands" in overlays and "BB_upper" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="rgba(100,116,139,0.6)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="rgba(100,116,139,0.6)", width=1),
            fill="tonexty", fillcolor="rgba(100,116,139,0.05)"), row=1, col=1)

    # ── RSI (14) ───────────────────────────────────────────────────────────
    if "RSI" in df:
        x_ends = [df.index[0], df.index[-1]]
        fig.add_trace(go.Scatter(
            x=x_ends, y=[75, 75], mode="lines",
            line=dict(color="#DC2626", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=x_ends, y=[25, 25], mode="lines",
            line=dict(color="#059669", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            name=f"RSI({rsi_period})",
            line=dict(color="#0EA5E9", width=1.8),
            connectgaps=True,
            fill="tozeroy", fillcolor="rgba(14,165,233,0.06)",
        ), row=2, col=1)
    # ── RSI (30) ───────────────────────────────────────────────────────────
    if "RSI30" in df:
        x_ends = [df.index[0], df.index[-1]]
        fig.add_trace(go.Scatter(
            x=x_ends, y=[70, 70], mode="lines",
            line=dict(color="#DC2626", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=x_ends, y=[30, 30], mode="lines",
            line=dict(color="#059669", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI30"],
            name="RSI(30)",
            line=dict(color="#A855F7", width=1.8),
            connectgaps=True,
            fill="tozeroy", fillcolor="rgba(168,85,247,0.06)",
        ), row=3, col=1)
    fig.update_layout(
        title=dict(
            text=f"<b>{name}</b>  <span style='font-size:13px;color:#64748B'>({ticker})  ·  {chart_type}  ·  RSI({rsi_period})</span>",
            font=dict(size=16, color="#1A202C", family="Inter, Segoe UI, sans-serif"),
        ),
        xaxis_rangeslider_visible=False,
        height=750,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFBFD",
        font=dict(color="#1A202C", family="Inter, Segoe UI, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
        margin=dict(l=10, r=10, t=60, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0",
                        font=dict(color="#1A202C", size=12)),
    )
    fig.update_xaxes(gridcolor="#E8EDF5", zeroline=False, linecolor="#E2E8F0",
                     showspikes=True, spikecolor="#94A3B8", spikethickness=1,
                     rangebreaks=get_rangebreaks(df))
    fig.update_yaxes(gridcolor="#E8EDF5", zeroline=False, linecolor="#E2E8F0",
                     showspikes=True, spikecolor="#94A3B8", spikethickness=1)
    fig.update_yaxes(rangemode="normal", autorange=True, row=1, col=1)
    # Pin RSI panel ranges — always include threshold lines with padding
    def _rsi_axis_range(series: pd.Series, oversold: float, overbought: float,
                        pad: float = 5.0):
        s = series.dropna()
        data_lo = float(s.min()) if not s.empty else oversold
        data_hi = float(s.max()) if not s.empty else overbought
        lo = min(data_lo, oversold) - pad
        hi = max(data_hi, overbought) + pad
        return [max(0.0, lo), min(100.0, hi)]

    if "RSI" in df:
        fig.update_layout(yaxis2=dict(
            range=_rsi_axis_range(df["RSI"], oversold=25, overbought=75),
            tickfont=dict(size=10), gridcolor="#E8EDF5",
        ))
    if "RSI30" in df:
        fig.update_layout(yaxis3=dict(
            range=_rsi_axis_range(df["RSI30"], oversold=30, overbought=70),
            tickfont=dict(size=10), gridcolor="#E8EDF5",
        ))
    return fig


# ── Main render ────────────────────────────────────────────────────────────────
@st.fragment
def render_watchlist():
    if "wl_config" not in st.session_state:
        st.session_state.wl_config = load_config()
    if "wl_selected" not in st.session_state:
        st.session_state.wl_selected = None

    cfg         = st.session_state.wl_config
    instruments = cfg["instruments"]
    sel_cols    = cfg["columns"]

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])

    with c1:
        new_cols = st.multiselect("Columns", ALL_COLUMNS, default=sel_cols,
                                  help="Choose which metrics to display")
        if set(new_cols) != set(sel_cols):
            cfg["columns"] = new_cols
            save_config(cfg)
            st.session_state.wl_config = cfg

    active_cols = new_cols if new_cols else sel_cols

    with c2:
        sort_col = st.selectbox("Sort by", ["Name", "Class"] + active_cols)

    with c3:
        sort_dir = st.radio("Order", ["↑ Asc", "↓ Desc"], horizontal=True)
        sort_asc = sort_dir.startswith("↑")

    with c4:
        st.markdown("<div style='margin-top:26px'>", unsafe_allow_html=True)
        if st.button("⟳ Refresh", use_container_width=True):
            # Targeted invalidation — refresh THIS tab's prices only, never wipe
            # other tabs' expensive Databento/ECB/Wikipedia caches.
            _all_daily.clear()
            fetch_chart_data.clear()
            _fetch_custom_df_cached.clear()
            _fetch_fred_df_cached.clear()
            _fetch_jgb_df.clear()
            _fetch_ecb_df.clear()
            _fetch_alphavantage_fx.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Custom series selector ────────────────────────────────────────────────
    all_cs      = _load_cs_json()
    cs_name_map = {cs["id"]: cs["name"] for cs in all_cs}
    sel_cs_ids  = cfg.get("custom_series_ids", [])
    # Keep only IDs that still exist in custom_series.json
    sel_cs_ids  = [cid for cid in sel_cs_ids if cid in cs_name_map]

    if all_cs:
        chosen_names = st.multiselect(
            "Custom series in Macro",
            options=[cs["name"] for cs in all_cs],
            default=[cs_name_map[cid] for cid in sel_cs_ids],
            placeholder="None — pick series to add",
        )
        new_sel_ids = [cs["id"] for cs in all_cs if cs["name"] in chosen_names]
        if new_sel_ids != sel_cs_ids:
            cfg["custom_series_ids"] = new_sel_ids
            save_config(cfg)
            st.session_state.wl_config = cfg
            sel_cs_ids = new_sel_ids

    active_cs = [cs for cs in all_cs if cs["id"] in sel_cs_ids]

    # ── Asset class filter pills ──────────────────────────────────────────────
    present_classes = sorted({i["class"] for i in instruments} | ({"Custom"} if active_cs else set()))
    selected_classes = st.pills(
        "Filter by asset class", options=present_classes,
        default=present_classes, selection_mode="multi",
    )
    if not selected_classes:
        selected_classes = present_classes

    # ── Fetch data ────────────────────────────────────────────────────────────
    tickers = tuple(i["ticker"] for i in instruments)
    with st.spinner("Loading watchlist…"):
        batch = fetch_batch(tickers)

    rows = [compute_row(inst, batch.get(inst["ticker"]), active_cols) for inst in instruments]

    # Append selected custom series rows
    if active_cs and "Custom" in selected_classes:
        for cs in active_cs:
            cs_df = _fetch_custom_df_cached(cs["id"])
            cs_inst = {"name": cs["name"], "class": "Custom", "ticker": f"custom:{cs['id']}"}
            row = compute_row(cs_inst, cs_df if not cs_df.empty else None, active_cols)
            rows.append(row)

    df_rows = pd.DataFrame(rows)

    # Filter & sort
    df_rows = df_rows[df_rows["Class"].isin(selected_classes)]
    if sort_col in df_rows.columns:
        # Sort on numeric values before formatting
        df_rows = df_rows.sort_values(sort_col, ascending=sort_asc, na_position="last")

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:11px;color:#94A3B8;margin-bottom:4px'>"
        "👆 Click a row to chart it</p>",
        unsafe_allow_html=True,
    )

    df_display = build_display_df(df_rows, active_cols)
    styled     = style_table(df_display, active_cols)

    event = st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=min(600, 38 + 35 * len(df_display)),
    )

    st.caption(
        f"Last updated: {datetime.now().strftime('%H:%M:%S')}  ·  "
        f"{len(df_rows)} instruments shown  ·  Data: Yahoo Finance"
    )

    # ── Inline chart ─────────────────────────────────────────────────────────
    selected_rows = event.selection.rows if event and event.selection else []
    if selected_rows:
        row_idx  = selected_rows[0]
        row_data = df_display.iloc[row_idx]
        inst_name = row_data["Name"]

        # Custom series — no ticker chart available; clear any previous selection
        if any(cs["name"] == inst_name for cs in active_cs):
            st.session_state.wl_selected = None
        else:
            inst = next((i for i in instruments if i["name"] == inst_name), None)
            if inst:
                st.session_state.wl_selected = inst

    sel = st.session_state.wl_selected
    if sel:
        st.markdown("---")
        st.markdown(
            f"<h4 style='color:#1B3A6B;margin-bottom:4px'>"
            f"{sel['name']} "
            f"<span style='font-size:13px;color:#64748B;font-weight:400'>"
            f"({sel['ticker']})</span></h4>",
            unsafe_allow_html=True,
        )

        # ── Row 1: Timeframe + Close ──────────────────────────────────────
        tf_cols = st.columns([7, 1])
        with tf_cols[0]:
            tf = st.segmented_control(
                "Timeframe",
                options=list(CHART_TIMEFRAMES.keys()),
                default="3M",
                key=f"tf_{sel['ticker']}",
            )
        with tf_cols[1]:
            st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
            if st.button("✕ Close", key="close_chart", use_container_width=True):
                st.session_state.wl_selected = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Row 2: Chart type · RSI period · Overlays ─────────────────────
        opt_c1, opt_c2, opt_c3 = st.columns([2, 2, 4])

        with opt_c1:
            chart_type = st.selectbox(
                "Chart type",
                ["Line", "Candlestick", "OHLC Bar"],
                key=f"ct_{sel['ticker']}",
            )

        with opt_c2:
            rsi_period = st.number_input(
                "RSI period", min_value=2, max_value=50,
                value=14, step=1,
                key=f"rsi_{sel['ticker']}",
                help="Standard is 14. Higher = smoother, lower = more sensitive.",
            )

        with opt_c3:
            overlays = st.multiselect(
                "Overlays",
                ["SMA 20", "SMA 50", "SMA 100", "SMA 200", "EMA 20", "Bollinger Bands"],
                default=["SMA 20", "SMA 50"],
                key=f"ov_{sel['ticker']}",
            )

        # ── Custom date range ─────────────────────────────────────────────
        if tf == "Custom":
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                start_date = st.date_input("From", value=date.today() - timedelta(days=180),
                                           max_value=date.today() - timedelta(days=2))
            with d_col2:
                end_date = st.date_input("To", value=date.today(),
                                         min_value=start_date + timedelta(days=1),
                                         max_value=date.today())
            period    = None
            interval  = "1d"
            start_str = str(start_date)
            end_str   = str(end_date)
        else:
            period, interval = CHART_TIMEFRAMES[tf]
            start_str = end_str = None

        with st.spinner(f"Loading {sel['name']} chart…"):
            df_chart = fetch_chart_data(
                sel["ticker"], period, interval,
                rsi_period=int(rsi_period),
                start=start_str, end=end_str,
            )

        if not df_chart.empty:
            # Key encodes all options so Streamlit always redraws when anything changes
            chart_key = (
                f"wl_{sel['ticker']}_{tf}_{chart_type}_{rsi_period}"
                f"_{'_'.join(sorted(overlays or []))}"
                f"_{start_str}_{end_str}"
            )
            st.plotly_chart(
                build_instrument_chart(
                    df_chart, sel["name"], sel["ticker"],
                    chart_type=chart_type,
                    overlays=overlays,
                    rsi_period=int(rsi_period),
                ),
                use_container_width=True,
                key=chart_key,
            )
        else:
            st.warning("No chart data available for this instrument / timeframe.")

    st.markdown("---")

    # ── Add / Remove ──────────────────────────────────────────────────────────
    col_add, col_remove = st.columns(2)

    with col_add:
        with st.expander("➕  Add instrument"):
            new_name   = st.text_input("Display name", placeholder="e.g. Apple Inc")
            new_ticker = st.text_input("Yahoo Finance ticker", placeholder="e.g. AAPL  /  ^FTSE  /  GC=F")
            new_class  = st.selectbox("Asset class", ASSET_CLASSES, key="add_class")
            if st.button("Add to watchlist", type="primary", key="btn_add"):
                name   = new_name.strip()
                ticker = new_ticker.upper().strip()
                if name and ticker:
                    entry = {"name": name, "ticker": ticker, "class": new_class}
                    if ticker not in [i["ticker"] for i in cfg["instruments"]]:
                        cfg["instruments"].append(entry)
                        save_config(cfg)
                        st.session_state.wl_config = cfg
                        # Ticker tuple changed → _all_daily gets a fresh cache
                        # key automatically; no clear() needed.
                        st.success(f"✓ Added {name} ({ticker})")
                        st.rerun()
                    else:
                        st.warning(f"{ticker} is already in the watchlist.")
                else:
                    st.warning("Please fill in both name and ticker.")

    with col_remove:
        with st.expander("➖  Remove instruments"):
            names = [i["name"] for i in cfg["instruments"]]
            to_remove = st.multiselect("Select to remove", names)
            if st.button("Remove selected", type="secondary", key="btn_remove"):
                if to_remove:
                    cfg["instruments"] = [i for i in cfg["instruments"]
                                          if i["name"] not in to_remove]
                    save_config(cfg)
                    st.session_state.wl_config = cfg
                    # Ticker tuple changed → _all_daily gets a fresh cache key
                    # automatically; no clear() needed.
                    st.rerun()
                else:
                    st.warning("Select at least one instrument to remove.")

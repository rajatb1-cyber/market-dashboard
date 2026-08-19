import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
import time
import sqlite3
import pathlib

import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    import nest_asyncio
    nest_asyncio.apply()
    from ib_insync import Future, util
    _IB_AVAILABLE = True
except Exception:
    _IB_AVAILABLE = False

# All TWS I/O routes through the ONE shared connection (clientId 15) — never open
# ad-hoc IB() clients here (the random clientId used to collide with clientId 15,
# and connection churn exhausted the account-wide market-data line pool).
from ibkr_conn import (get_conn, quotes, hist_bars, PacingBlocked,
                       usage_today, usage_warning)

# ── Module-level singletons ────────────────────────────────────────────────────
_CONTRACT_STORE: dict = {}

_DB_PATH  = str(pathlib.Path(__file__).parent / "commodities_bars.db")
_DB_READY: bool = False


# ── DB helpers (same pattern as stir.py) ──────────────────────────────────────

def _ensure_db() -> None:
    global _DB_READY
    if _DB_READY:
        return
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS bars (
        symbol TEXT NOT NULL, exchange TEXT NOT NULL,
        exp6 TEXT NOT NULL, bar_date TEXT NOT NULL, close REAL NOT NULL,
        PRIMARY KEY (symbol, exchange, exp6, bar_date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY, value TEXT)""")
    conn.commit()
    conn.close()
    _DB_READY = True


def _bars_are_fresh(symbol: str, exchange: str) -> bool:
    _ensure_db()
    try:
        conn = sqlite3.connect(_DB_PATH)
        cur  = conn.execute("SELECT value FROM meta WHERE key=?",
                            (f"last_updated_{symbol}_{exchange}",))
        row  = cur.fetchone()
        conn.close()
        if not row:
            return False
        last_upd = datetime.fromisoformat(row[0])
        now      = datetime.now()
        cutoff   = now.replace(hour=17, minute=30, second=0, microsecond=0)
        if now >= cutoff:
            return last_upd >= cutoff
        return last_upd >= cutoff - timedelta(days=1)
    except Exception:
        return False


def _read_bars_from_db(symbol: str, exchange: str, contracts: tuple) -> dict:
    _ensure_db()
    conn   = sqlite3.connect(_DB_PATH)
    result = {}
    try:
        for _, exp6 in contracts:
            cur    = conn.execute(
                "SELECT close FROM bars WHERE symbol=? AND exchange=? AND exp6=? ORDER BY bar_date",
                (symbol, exchange, exp6))
            closes = [r[0] for r in cur.fetchall()]
            if closes:
                result[exp6] = closes
    finally:
        conn.close()
    return result


def _write_bars_to_db(symbol: str, exchange: str, raw_bars: dict) -> None:
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH)
    try:
        for exp6, bars in raw_bars.items():
            for bar in bars:
                dv = bar.date
                ds = dv.strftime('%Y-%m-%d') if hasattr(dv, 'strftime') else str(dv)[:10]
                conn.execute("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?)",
                             (symbol, exchange, exp6, ds, float(bar.close)))
        conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                     (f"last_updated_{symbol}_{exchange}", datetime.now().isoformat()))
        conn.commit()
    finally:
        conn.close()


def _bars_last_updated(symbol: str, exchange: str) -> str:
    try:
        _ensure_db()
        conn = sqlite3.connect(_DB_PATH)
        cur  = conn.execute("SELECT value FROM meta WHERE key=?",
                            (f"last_updated_{symbol}_{exchange}",))
        row  = cur.fetchone()
        conn.close()
        if not row:
            return ""
        return datetime.fromisoformat(row[0]).strftime("%d %b %Y %H:%M")
    except Exception:
        return ""


def _force_bars_refresh(symbol: str, exchange: str) -> None:
    try:
        _ensure_db()
        conn = sqlite3.connect(_DB_PATH)
        conn.execute("DELETE FROM meta WHERE key=?",
                     (f"last_updated_{symbol}_{exchange}",))
        conn.commit()
        conn.close()
    except Exception:
        pass
    _get_change_bars.clear()


# ── Contract helpers ──────────────────────────────────────────────────────────

_MONTH_NAMES = {
    1: "Jan", 2: "Feb",  3: "Mar", 4: "Apr",  5: "May", 6: "Jun",
    7: "Jul", 8: "Aug",  9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
_MONTH_CODES = {
    "01": "F", "02": "G", "03": "H", "04": "J", "05": "K", "06": "M",
    "07": "N", "08": "Q", "09": "U", "10": "V", "11": "X", "12": "Z",
}
_DURATION_MAP = {"1M": "1 M", "3M": "3 M", "6M": "6 M", "1Y": "1 Y"}

_CURVES = {
    "Brent": {
        "symbol":           "COIL",
        "exchange":         "IPE",
        "currency":         "USD",
        "local_sym_prefix": "COIL",
        "subtitle":         "ICE Brent Crude Futures (IPE)",
        "color":            "#F59E0B",
        "unit":             "$/bbl",
    },
    "WTI": {
        "symbol":           "WTI",
        "exchange":         "IPE",
        "currency":         "USD",
        "local_sym_prefix": "WTI",
        "subtitle":         "ICE WTI Crude Futures (IPE)",
        "color":            "#0EA5E9",
        "unit":             "$/bbl",
    },
    "NatGas": {
        "symbol":           "NG",
        "exchange":         "NYMEX",
        "currency":         "USD",
        "local_sym_prefix": "NG",
        "subtitle":         "NYMEX Natural Gas Futures (Henry Hub)",
        "color":            "#10B981",
        "unit":             "$/MMBtu",
    },
    "UKGas": {
        "symbol":           "NGF",
        "exchange":         "IPE",
        "currency":         "GBP",
        "local_sym_prefix": "NGF",
        "subtitle":         "ICE UK Natural Gas Futures (NBP)",
        "color":            "#8B5CF6",
        "unit":             "p/therm",
    },
    "TTF": {
        "symbol":           "TFM",
        "exchange":         "ENDEX",
        "currency":         "EUR",
        "local_sym_prefix": "TFM",
        "subtitle":         "ICE TTF Natural Gas Futures (Dutch)",
        "color":            "#F97316",
        "unit":             "€/MWh",
    },
}



def _active_contracts(n: int = 12) -> list[tuple]:
    today = date.today()
    y, m  = today.year, today.month + 2   # Brent/WTI front months expire ~6wks early; skip two months back
    if m > 12:
        m, y = 1, y + 1
    out = []
    for _ in range(n):
        out.append((f"{_MONTH_NAMES[m]} {str(y)[-2:]}", f"{y}{m:02d}"))
        m += 1
        if m > 12:
            m, y = 1, y + 1
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


def _resolve(ib, symbol, exchange, currency, exp6, local_sym_prefix=""):
    from ib_insync import Contract
    if local_sym_prefix:
        mc = _MONTH_CODES.get(exp6[4:6], "")
        if mc:
            local_sym = f"{local_sym_prefix}{mc}{exp6[3]}"
            try:
                det = ib.reqContractDetails(
                    Contract(secType="FUT", localSymbol=local_sym, exchange=exchange)
                )
                if det:
                    exact  = [d for d in det if d.contract.lastTradeDateOrContractMonth[:6] == exp6]
                    chosen = exact or det
                    return chosen[0].contract, len(det)
            except Exception:
                pass
    fut = Future(symbol=symbol, exchange=exchange, currency=currency,
                 lastTradeDateOrContractMonth=exp6)
    try:
        details = ib.reqContractDetails(fut)
    except Exception:
        return None, 0
    if not details:
        return None, 0
    exact  = [d for d in details if d.contract.lastTradeDateOrContractMonth[:6] == exp6]
    chosen = exact or details
    return chosen[0].contract, len(details)


def _resolve_fast(ib, host, port, symbol, exchange, currency, exp6, local_sym_prefix):
    key = (host, port, symbol, exchange, currency, local_sym_prefix, exp6)
    if key in _CONTRACT_STORE:
        return _CONTRACT_STORE[key], 1
    qc, n = _resolve(ib, symbol, exchange, currency, exp6, local_sym_prefix)
    if qc:
        _CONTRACT_STORE[key] = qc
    return qc, n


# ── Cached data fetchers ───────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _get_change_bars(host: str, port: int, contracts: tuple,
                     symbol: str, exchange: str, currency: str,
                     local_sym_prefix: str = "") -> dict:
    """DB-first bar fetch — {exp6: [close, ...]}; IBKR only when stale."""
    if _bars_are_fresh(symbol, exchange):
        return _read_bars_from_db(symbol, exchange, contracts)

    ibl, err = get_conn()
    if ibl is None:
        return _read_bars_from_db(symbol, exchange, contracts)

    result:   dict = {}
    raw_bars: dict = {}
    try:
        for lbl, exp6 in contracts:
            # Build the contract directly and hand it to hist_bars — IBKR resolves
            # symbol+exchange+expiry by itself, so we SKIP reqContractDetails, which hangs
            # ~20s per contract in this TWS setup and would freeze the shared loop thread.
            qc = Future(symbol=symbol, exchange=exchange, currency=currency,
                        lastTradeDateOrContractMonth=exp6)
            bars = None
            for show in ("TRADES", "MIDPOINT"):   # LAST is rejected (Error 321) for these futures
                try:
                    b = hist_bars(
                        qc, endDateTime="", durationStr="1 Y",
                        barSizeSetting="1 day", whatToShow=show, useRTH=False,
                        timeout_s=12, ibl=ibl, tag="commodities",
                    )
                    if b:
                        bars = b
                        break
                except PacingBlocked:
                    raise                 # PacingBlocked → serve DB bars (handled below)
                except Exception:
                    pass
            if bars:
                result[exp6]   = [float(bar.close) for bar in bars]
                raw_bars[exp6] = bars
            else:
                result[exp6]   = []
    except PacingBlocked:
        return _read_bars_from_db(symbol, exchange, contracts)
    except Exception:
        pass

    if raw_bars:
        try:
            _write_bars_to_db(symbol, exchange, raw_bars)
        except Exception:
            pass

    return result if result else _read_bars_from_db(symbol, exchange, contracts)


@st.cache_data(ttl=None)
def _fetch_quotes(host: str, port: int, contracts: tuple,
                  symbol: str, exchange: str, currency: str,
                  local_sym_prefix: str = "") -> tuple:
    """Live prices + change columns. Returns (rows, debug)."""
    bars_by_exp = _get_change_bars(host, port, contracts, symbol, exchange,
                                   currency, local_sym_prefix)
    ibl, err = get_conn()
    if ibl is None:
        return [], {"qual": [], "ticks": [], "error": err}

    rows  = []
    debug = {"qual": [], "ticks": []}
    try:
        from ib_insync import Contract
        label_by_exp6 = {exp: lbl for lbl, exp in contracts}
        quad_meta     = []   # (qc, label, exp6) parallel to the quotes() input

        for exp6, label in label_by_exp6.items():
            # Build the contract directly (skip reqContractDetails — it hangs here). reqMktData
            # inside quotes() is non-blocking and resolves the contract by itself.
            mc = _MONTH_CODES.get(exp6[4:6], "")
            if local_sym_prefix and mc:
                qc = Contract(secType="FUT", exchange=exchange, currency=currency,
                              localSymbol=f"{local_sym_prefix}{mc}{exp6[3]}")
            else:
                qc = Future(symbol=symbol, exchange=exchange, currency=currency,
                            lastTradeDateOrContractMonth=exp6)
            debug["qual"].append({"exp": exp6, "label": label, "status": "direct", "matches": 1})
            quad_meta.append((qc, label, exp6))

        if not quad_meta:
            return [], debug

        # mdtype=2 (frozen) + 2s settle preserve the old reqMarketDataType(2)+sleep(2);
        # quotes() subscribes in chunks and always cancels the lines in a finally.
        tickers = quotes([qc for qc, _, _ in quad_meta], mdtype=2, settle_s=2.0, ibl=ibl,
                         tag="commodities")

        for (qc, label, expiry), ticker in zip(quad_meta, tickers):
            last  = _v(ticker.last)   if ticker is not None else None
            close = _v(ticker.close)  if ticker is not None else None
            bid   = _v(ticker.bid)    if ticker is not None else None
            ask   = _v(ticker.ask)    if ticker is not None else None
            vol   = ticker.volume     if ticker is not None else None
            mdt   = getattr(ticker, "marketDataType", "?") if ticker is not None else "?"

            mid        = (bid + ask) / 2 if (bid and ask) else None
            live_price = last or mid

            bc        = bars_by_exp.get(expiry, [])
            bar_close = _v(bc[-1]) if len(bc) >= 1 else None
            bar_prev  = _v(bc[-2]) if len(bc) >= 2 else None

            hist_price = None
            if not live_price and bar_close:
                hist_price = bar_close
                live_price = hist_price
            close = bar_prev or close or bar_close

            src          = "last" if last else ("mid" if mid else ("hist" if hist_price else "close"))
            display_price = live_price or close
            if display_price is None:
                continue

            debug["ticks"].append({
                "Contract": label, "last": last, "bid": bid, "ask": ask,
                "close": close, "hist": hist_price, "mktDataType": mdt,
            })

            ref_1d = close
            ref_1w = _v(bc[-6])  if len(bc) >= 6  else None
            ref_1m = _v(bc[-22]) if len(bc) >= 22 else None
            ref_3m = _v(bc[-65]) if len(bc) >= 65 else (_v(bc[0]) if bc else None)

            def _chg(ref):
                return ((live_price - ref) / ref * 100) if (live_price and ref) else None

            rows.append({
                "label":  label,
                "expiry": expiry,
                "price":  live_price,
                "close":  close,
                "chg_1d": _chg(ref_1d),
                "chg_1w": _chg(ref_1w),
                "chg_1m": _chg(ref_1m),
                "chg_3m": _chg(ref_3m),
                "volume": int(vol) if vol and vol > 0 else None,
                "src":    src,
            })

    except Exception:
        pass

    return rows, debug


_HIST_DURATION_DAYS = {"1 M": 31, "3 M": 93, "6 M": 186, "1 Y": 366}


def _read_history_from_db(symbol: str, exchange: str, exp6: str,
                           duration_str: str) -> tuple:
    _ensure_db()
    try:
        conn    = sqlite3.connect(_DB_PATH)
        cur     = conn.execute(
            "SELECT bar_date, close FROM bars WHERE symbol=? AND exchange=? AND exp6=?"
            " ORDER BY bar_date",
            (symbol, exchange, exp6))
        rows_db = cur.fetchall()
        conn.close()
        if not rows_db:
            return pd.DataFrame(), "no DB data"
        df = pd.DataFrame(rows_db, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        days      = _HIST_DURATION_DAYS.get(duration_str, 93)
        cutoff    = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        df_period = df[df.index >= cutoff]
        return (df_period if not df_period.empty else df).dropna(), ""
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=None)
def _fetch_history(host: str, port: int, expiry: str, duration: str,
                   symbol: str, exchange: str, currency: str,
                   local_sym_prefix: str = "") -> tuple:
    """Returns (df, error_str) — DB first, IBKR fallback."""
    df_db, _ = _read_history_from_db(symbol, exchange, expiry, duration)
    if not df_db.empty:
        return df_db, ""

    ibl, err = get_conn()
    if ibl is None:
        return pd.DataFrame(), err
    try:
        # Build the contract directly (skip _resolve/reqContractDetails — it hangs here);
        # hist_bars resolves symbol+exchange+expiry itself.
        resolved = Future(symbol=symbol, exchange=exchange, currency=currency,
                          lastTradeDateOrContractMonth=expiry)
        bars = None
        for show in ("TRADES", "LAST", "MIDPOINT"):
            try:
                bars = hist_bars(
                    resolved, endDateTime="", durationStr=duration,
                    barSizeSetting="1 day", whatToShow=show, useRTH=False, ibl=ibl,
                    tag="commodities",
                )
            except PacingBlocked as e:
                return pd.DataFrame(), str(e)   # PacingBlocked → empty (DB already tried above)
            except Exception:
                bars = None
            if bars:
                break
        if not bars:
            return pd.DataFrame(), f"No bars returned for {expiry}"
        df = util.df(bars)[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df.set_index("date").sort_index().dropna(), ""
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=300)
def _scan_symbol(host: str, port: int, symbol: str, exchange: str,
                 currency: str = "USD") -> list:
    """Contract-spec scanner — DISABLED under the shared-connection module.

    Its only operation is reqContractDetails, which hangs in this TWS setup and
    would block the one shared loop thread every other tab depends on; the frozen
    ibkr_conn API deliberately exposes no reqContractDetails path (see Wave-2 spec).
    Update the _CURVES config by hand (localSymbol / exchange) if a curve stops
    resolving, rather than probing live."""
    return [{
        "info": "Contract scanner disabled — reqContractDetails hangs this TWS setup "
                "and is unavailable via the shared connection. Edit _CURVES manually "
                "if a curve stops resolving.",
        "symbol": symbol, "exchange": exchange, "currency": currency,
    }]


@st.cache_data(ttl=None)
def _fetch_ref_prices_curve(host: str, port: int, contracts: tuple, ref_date_str: str,
                             symbol: str, exchange: str, currency: str,
                             local_sym_prefix: str = "") -> dict:
    """Price for each contract at ref_date — DB first, IBKR fallback."""
    _ensure_db()
    label_by_exp6 = {exp: lbl for lbl, exp in contracts}
    results: dict = {}

    # DB path: single query per contract, closest bar <= ref_date
    try:
        conn = sqlite3.connect(_DB_PATH)
        for exp6, label in label_by_exp6.items():
            cur = conn.execute(
                "SELECT close FROM bars WHERE symbol=? AND exchange=? AND exp6=?"
                " AND bar_date <= ? ORDER BY bar_date DESC LIMIT 1",
                (symbol, exchange, exp6, ref_date_str))
            row = cur.fetchone()
            if row:
                results[label] = row[0]
        conn.close()
    except Exception:
        pass

    if len(results) >= max(len(contracts) // 2, 1):
        return results

    # IBKR fallback for any missing contracts
    ibl, err = get_conn()
    if ibl is None:
        return results
    try:
        ref_dt    = pd.Timestamp(ref_date_str)
        days_back = max((pd.Timestamp.now().normalize() - ref_dt).days + 10, 15)
        duration  = f"{min(days_back, 365)} D" if days_back <= 365 else f"{days_back // 365 + 1} Y"
        for exp6, label in label_by_exp6.items():
            if label in results:
                continue
            # Build the contract directly (skip _resolve/reqContractDetails — it hangs here).
            qc = Future(symbol=symbol, exchange=exchange, currency=currency,
                        lastTradeDateOrContractMonth=exp6)
            bars = None
            for show in ("TRADES", "LAST", "MIDPOINT"):
                try:
                    b = hist_bars(qc, endDateTime="", durationStr=duration,
                                  barSizeSetting="1 day", whatToShow=show, useRTH=False, ibl=ibl,
                                  tag="commodities")
                    if b:
                        bars = b
                        break
                except PacingBlocked:
                    return results          # PacingBlocked → serve whatever DB already gave us
                except Exception:
                    pass
            if bars:
                df = util.df(bars)[["date", "close"]].copy()
                df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                df = df.set_index("date").sort_index()
                cands = df[df.index <= ref_dt]
                if not cands.empty:
                    results[label] = float(cands["close"].iloc[-1])
    except Exception:
        pass
    return results


# ── Render helpers ─────────────────────────────────────────────────────────────

def _chg_color(v):
    if v is None:
        return ""
    return "color:#059669;font-weight:600" if v > 0 else (
           "color:#DC2626;font-weight:600" if v < 0 else "")


def _render_curve(host: str, port: int, curve_name: str, tab_key: str):
    cfg              = _CURVES[curve_name]
    symbol           = cfg["symbol"]
    exchange         = cfg["exchange"]
    currency         = cfg["currency"]
    local_sym_prefix = cfg["local_sym_prefix"]
    color            = cfg["color"]
    unit             = cfg.get("unit", "$/bbl")
    contracts        = _active_contracts(12)

    btn_c1, btn_c2 = st.columns([2, 3])
    with btn_c1:
        if st.button("⟳ Refresh prices", key=f"com_refresh_{tab_key}"):
            _fetch_quotes.clear()
            st.rerun()
    with btn_c2:
        if st.button("↺ Update bar data from IBKR", key=f"com_barrefresh_{tab_key}"):
            _force_bars_refresh(symbol, exchange)
            _fetch_quotes.clear()
            st.rerun()

    t0 = time.perf_counter()
    fetch_ts = time.time()
    with st.spinner(f"Fetching live prices ({curve_name})…"):
        try:
            rows, debug = _fetch_quotes(
                host, port, tuple(contracts), symbol, exchange, currency,
                local_sym_prefix,
            )
        except Exception as e:
            st.error(f"**Could not connect to IBKR** (`{host}:{port}`).  \nError: `{e}`")
            return
    quotes_ms  = (time.perf_counter() - t0) * 1000
    quotes_src = "cache" if quotes_ms < 100 else "IBKR"

    if not rows:
        err = debug.get("error", "")
        if err:
            st.error(f"**IBKR connection error:** `{err}`")
        else:
            st.warning(f"No quotes returned for {curve_name} — could not resolve contracts.")
        with st.expander("Debug — qualification", expanded=True):
            st.dataframe(pd.DataFrame(debug.get("qual", [])), use_container_width=True)
        if not err:
            with st.expander(f"Contract scanner — probing '{symbol}' on {exchange}", expanded=True):
                with st.spinner("Scanning IBKR for contract spec…"):
                    # Use the example Dec'26 localSymbol for the scan
                    mc  = _MONTH_CODES.get("12", "Z")
                    lsym = f"{local_sym_prefix}{mc}6"
                    scan = _scan_symbol(host, port, lsym, exchange, currency)
                if scan:
                    st.dataframe(pd.DataFrame(scan), use_container_width=True)
                    st.caption("Use the exchange/symbol values above to update _CURVES config.")
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

    # ── Prices table ──────────────────────────────────────────────────────────
    st.markdown(f"#### {cfg['subtitle']}")
    src_counts: dict = {}
    for r in rows:
        src_counts[r["src"]] = src_counts.get(r["src"], 0) + 1
    src_str  = " · ".join(f"{v}× {k}" for k, v in src_counts.items())
    updated  = time.strftime("%H:%M:%S", time.localtime(fetch_ts))
    bars_ts  = _bars_last_updated(symbol, exchange)
    bars_note = f" · Bars updated {bars_ts}" if bars_ts else " · Bars: not yet cached"
    st.caption(
        f"IBKR · Prices {updated} · {src_str}{bars_note}"
        f" · quotes {quotes_ms:.0f}ms [{quotes_src}]"
    )

    col_1d, col_1w, col_1m, col_3m = [], [], [], []
    table_rows = []
    for r in rows:
        col_1d.append(_chg_color(r["chg_1d"]))
        col_1w.append(_chg_color(r["chg_1w"]))
        col_1m.append(_chg_color(r["chg_1m"]))
        col_3m.append(_chg_color(r["chg_3m"]))
        price_str = _fmt(r["price"], "{:.2f}") if r["price"] else _fmt(r["close"], "{:.2f}*")
        table_rows.append({
            "Contract": r["label"],
            "Price":    price_str,
            "1D (%)":   _fmt(r["chg_1d"], "{:+.2f}%"),
            "1W (%)":   _fmt(r["chg_1w"], "{:+.2f}%"),
            "1M (%)":   _fmt(r["chg_1m"], "{:+.2f}%"),
            "3M (%)":   _fmt(r["chg_3m"], "{:+.2f}%"),
            "Volume":   _fmt(r["volume"], "{:,}"),
        })

    df_tbl = pd.DataFrame(table_rows).set_index("Contract")
    styled  = (
        df_tbl.style
        .apply(lambda _: col_1d, subset=["1D (%)"])
        .apply(lambda _: col_1w, subset=["1W (%)"])
        .apply(lambda _: col_1m, subset=["1M (%)"])
        .apply(lambda _: col_3m, subset=["3M (%)"])
    )
    st.dataframe(styled, use_container_width=True)

    # ── Forward curve ─────────────────────────────────────────────────────────
    st.markdown("#### Forward Curve")

    cmp_c1, cmp_c2 = st.columns([5, 3])
    with cmp_c1:
        cmp_period = st.radio(
            "Compare vs", ["—", "1D", "1W", "1M", "3M", "Custom"],
            index=0, horizontal=True, key=f"com_curve_cmp_{tab_key}",
        )
    cmp_date = None
    if cmp_period == "Custom":
        with cmp_c2:
            cmp_date = st.date_input(
                "Date",
                value=date.today() - timedelta(days=30),
                max_value=date.today() - timedelta(days=1),
                key=f"com_curve_cmp_date_{tab_key}",
            )

    _cmp_chg_key = {"1D": "chg_1d", "1W": "chg_1w", "1M": "chg_1m", "3M": "chg_3m"}
    hist_curve = None

    if cmp_period in _cmp_chg_key:
        chg_k = _cmp_chg_key[cmp_period]
        hist_curve = [
            (r["label"],
             r["price"] / (1 + r[chg_k] / 100)
             if (r["price"] is not None and r[chg_k] is not None) else None)
            for r in rows
        ]
    elif cmp_period == "Custom" and cmp_date:
        with st.spinner(f"Loading curve as of {cmp_date}…"):
            try:
                ref_prices = _fetch_ref_prices_curve(
                    host, port, tuple(contracts), str(cmp_date),
                    symbol, exchange, currency, local_sym_prefix,
                )
            except Exception:
                ref_prices = {}
        hist_curve = [(r["label"], ref_prices.get(r["label"])) for r in rows]

    dot_colors = [
        "#059669" if (r["chg_1d"] or 0) > 0 else
        ("#DC2626" if (r["chg_1d"] or 0) < 0 else "#94A3B8")
        for r in rows
    ]
    show_legend = hist_curve is not None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["label"] for r in rows],
        y=[r["price"]  for r in rows],
        mode="lines+markers",
        name="Current",
        showlegend=show_legend,
        line=dict(color=color, width=2),
        marker=dict(size=9, color=dot_colors, line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate=f"<b>Current</b>  %{{x}}<br>%{{y:.2f}} {unit}<extra></extra>",
    ))

    if hist_curve:
        valid_pts = [(lbl, p) for lbl, p in hist_curve if p is not None]
        if valid_pts:
            cmp_label = cmp_period if cmp_period != "Custom" else str(cmp_date)
            fig.add_trace(go.Scatter(
                x=[lbl for lbl, _ in valid_pts],
                y=[p   for _, p   in valid_pts],
                mode="lines+markers",
                name=cmp_label,
                line=dict(color="#DC2626", width=1.5, dash="dot"),
                marker=dict(size=6, color="#DC2626"),
                hovertemplate=f"<b>{cmp_label}</b>  %{{x}}<br>%{{y:.2f}} {unit}<extra></extra>",
            ))

    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        yaxis=dict(tickprefix="$", gridcolor="#E8EDF5", zeroline=False),
        xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1) if show_legend else {},
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dot: green = price higher vs prev close · red = lower")

    # ── Contract history ───────────────────────────────────────────────────────
    st.markdown("#### Contract history")
    hc1, hc2 = st.columns([3, 2])
    with hc1:
        sel_label = st.selectbox("Contract", [r["label"] for r in rows],
                                 key=f"com_hist_sel_{tab_key}")
    with hc2:
        hist_period = st.selectbox("Period", ["1M", "3M", "6M", "1Y"], index=1,
                                   key=f"com_hist_period_{tab_key}")

    sel_expiry = next(r["expiry"] for r in rows if r["label"] == sel_label)
    ibkr_dur   = _DURATION_MAP[hist_period]

    t1 = time.perf_counter()
    with st.spinner(f"Loading {sel_label} history…"):
        try:
            df_hist, hist_err = _fetch_history(
                host, port, sel_expiry, ibkr_dur, symbol, exchange, currency,
                local_sym_prefix,
            )
        except Exception as e:
            st.error(f"History fetch error: `{e}`")
            df_hist, hist_err = pd.DataFrame(), ""
    hist_ms  = (time.perf_counter() - t1) * 1000
    hist_src = "cache" if hist_ms < 100 else "IBKR"

    if hist_err:
        st.error(f"History error: {hist_err}")

    if not df_hist.empty:
        first_p = float(df_hist["close"].iloc[0])
        last_p  = float(df_hist["close"].iloc[-1])
        line_color  = "#059669" if last_p >= first_p else "#DC2626"
        fill_color  = ("rgba(5,150,105,0.07)" if line_color == "#059669"
                       else "rgba(220,38,38,0.07)")

        y_vals = df_hist["close"].dropna()
        pad    = max((y_vals.max() - y_vals.min()) * 0.15, 0.5)
        y_lo   = float(y_vals.min()) - pad
        y_hi   = float(y_vals.max()) + pad

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(df_hist.index) + list(df_hist.index[::-1]),
            y=list(y_vals) + [y_lo] * len(y_vals),
            fill="toself", fillcolor=fill_color,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig2.add_trace(go.Scatter(
            x=df_hist.index, y=df_hist["close"],
            mode="lines",
            line=dict(color=line_color, width=2),
            hovertemplate="%{x|%d %b %Y}<br><b>$%{y:.2f}</b><extra></extra>",
        ))
        fig2.update_layout(
            title=dict(
                text=(f"{sel_label} — {curve_name}  "
                      f"<span style='font-size:12px;color:#64748B'>"
                      f"{last_p:.2f} {unit}  ·  {hist_ms:.0f}ms [{hist_src}]</span>"),
                font=dict(size=13, color="#1A202C"),
            ),
            height=280, margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
            yaxis=dict(tickprefix="$", gridcolor="#E8EDF5", zeroline=False,
                       range=[y_lo, y_hi]),
            xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
            hovermode="x unified",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
        )
        st.plotly_chart(fig2, use_container_width=True)
    elif not hist_err:
        st.info("No history data available for this contract / period.")


def _render_overview(host: str, port: int):
    st.markdown("#### Crude Oil & Natural Gas — Forward Curves Overview")

    gbtn_c1, gbtn_c2 = st.columns([2, 3])
    with gbtn_c1:
        if st.button("⟳ Refresh prices", key="com_refresh_ovw"):
            _fetch_quotes.clear()
            st.rerun()
    with gbtn_c2:
        if st.button("↺ Update bar data from IBKR", key="com_barrefresh_ovw"):
            for name in _CURVES:
                cfg = _CURVES[name]
                _force_bars_refresh(cfg["symbol"], cfg["exchange"])
            _fetch_quotes.clear()
            st.rerun()

    contracts       = _active_contracts(12)
    contract_labels = [lbl for lbl, _ in contracts]

    all_rows: dict[str, dict] = {}
    for curve_name in _CURVES:
        cfg = _CURVES[curve_name]
        with st.spinner(f"Loading {curve_name}…"):
            try:
                rows, _ = _fetch_quotes(
                    host, port, tuple(contracts),
                    cfg["symbol"], cfg["exchange"], cfg["currency"],
                    cfg["local_sym_prefix"],
                )
                all_rows[curve_name] = {r["label"]: r for r in rows}
            except Exception:
                all_rows[curve_name] = {}

    bar_parts = [
        f"{n}: {_bars_last_updated(_CURVES[n]['symbol'], _CURVES[n]['exchange']) or 'not cached'}"
        for n in _CURVES
    ]
    st.caption("Bars last updated — " + " · ".join(bar_parts))

    empty_curves = [n for n in _CURVES if not all_rows.get(n)]
    if empty_curves:
        st.warning(
            f"No data for: {', '.join(empty_curves)}. "
            "Switch to the individual tab (Brent / WTI / NatGas) for contract diagnostics."
        )

    # ── Changes table ─────────────────────────────────────────────────────────
    st.markdown("##### Changes (%)")
    pc1, pc2 = st.columns([4, 3])
    with pc1:
        period = st.radio("Period", ["1D", "1W", "1M", "3M"],
                          index=0, horizontal=True, key="com_ovw_period")
    _chg_key = {"1D": "chg_1d", "1W": "chg_1w", "1M": "chg_1m", "3M": "chg_3m"}
    key      = _chg_key[period]

    disp, styles = {}, {}
    for curve_name in _CURVES:
        disp[curve_name] = {
            lbl: (_fmt(r[key], "{:+.2f}%") if (r := all_rows.get(curve_name, {}).get(lbl)) else "—")
            for lbl in contract_labels
        }
        styles[curve_name] = {
            lbl: _chg_color(r[key] if (r := all_rows.get(curve_name, {}).get(lbl)) else None)
            for lbl in contract_labels
        }

    df_chg = pd.DataFrame(disp, index=contract_labels).T
    df_chg.index.name = "Curve"
    styled_chg = df_chg.style.apply(
        lambda row: pd.Series(
            [styles.get(row.name, {}).get(col, "") for col in df_chg.columns],
            index=df_chg.columns,
        ),
        axis=1,
    )
    st.dataframe(styled_chg, use_container_width=True)

    # ── Overlaid forward curves — Crude ───────────────────────────────────────
    st.markdown("#### Forward Curves — Brent vs WTI")
    fig = go.Figure()
    for curve_name, color in [("Brent", "#F59E0B"), ("WTI", "#0EA5E9")]:
        rows_map = all_rows.get(curve_name, {})
        labels   = [lbl for lbl in contract_labels if lbl in rows_map]
        prices   = [rows_map[lbl]["price"] for lbl in labels]
        if not labels:
            continue
        fig.add_trace(go.Scatter(
            x=labels, y=prices,
            mode="lines+markers",
            name=curve_name,
            line=dict(color=color, width=2),
            marker=dict(size=7, color=color, line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate=f"<b>{curve_name}</b>  %{{x}}<br>%{{y:.2f}} $/bbl<extra></extra>",
        ))
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        yaxis=dict(tickprefix="$", gridcolor="#E8EDF5", zeroline=False),
        xaxis=dict(gridcolor="#E8EDF5", zeroline=False, categoryorder="array", categoryarray=contract_labels),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Combined Gas forward curves (3-axis) ──────────────────────────────────
    st.markdown("#### Forward Curves — Natural Gas (Henry Hub / ICE NBP / TTF)")
    gas_specs = [
        ("NatGas", "Henry Hub", "#10B981", "yaxis",  "$/MMBtu", "{:.3f} $/MMBtu", dict(tickprefix="$",   gridcolor="#E8EDF5", zeroline=False, title=dict(text="$/MMBtu",  font=dict(color="#10B981")), tickfont=dict(color="#10B981"))),
        ("UKGas",  "ICE NBP",   "#8B5CF6", "yaxis2", "p/therm", "{:.2f} p/therm", dict(ticksuffix="p",   gridcolor="#E8EDF5", zeroline=False, title=dict(text="p/therm",  font=dict(color="#8B5CF6")), tickfont=dict(color="#8B5CF6"), overlaying="y", side="right")),
        ("TTF",    "TTF",       "#F97316", "yaxis3", "€/MWh",   "{:.2f} €/MWh",   dict(tickprefix="€",   gridcolor="#E8EDF5", zeroline=False, title=dict(text="€/MWh",    font=dict(color="#F97316")), tickfont=dict(color="#F97316"), overlaying="y", side="right", anchor="free", position=1.0)),
    ]
    fig_gas = go.Figure()
    any_gas = False
    for curve_name, label, color, yref, unit_str, hover_fmt, _ in gas_specs:
        rm = all_rows.get(curve_name, {})
        lbls   = [lbl for lbl in contract_labels if lbl in rm]
        prices = [rm[lbl]["price"] for lbl in lbls]
        if not lbls:
            continue
        any_gas = True
        hover_text = [hover_fmt.format(p) for p in prices]
        fig_gas.add_trace(go.Scatter(
            x=lbls, y=prices,
            mode="lines+markers",
            name=label,
            yaxis=yref.replace("yaxis", "y"),
            line=dict(color=color, width=2),
            marker=dict(size=7, color=color, line=dict(color="#FFFFFF", width=1.5)),
            customdata=hover_text,
            hovertemplate=f"<b>{label}</b>  %{{x}}<br>%{{customdata}}<extra></extra>",
        ))
    if any_gas:
        fig_gas.update_layout(
            height=320, margin=dict(l=10, r=80, t=20, b=10),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
            xaxis=dict(gridcolor="#E8EDF5", zeroline=False, categoryorder="array", categoryarray=contract_labels, domain=[0, 0.92]),
            yaxis =gas_specs[0][6],
            yaxis2=gas_specs[1][6],
            yaxis3=gas_specs[2][6],
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
        )
        st.plotly_chart(fig_gas, use_container_width=True)
    else:
        st.info("No gas data — refresh prices or check individual gas tabs.")

    # ── Brent–WTI spread ──────────────────────────────────────────────────────
    st.markdown("#### Brent–WTI Spread ($/bbl)")
    spread_labels, spread_vals = [], []
    brent_map = all_rows.get("Brent", {})
    wti_map   = all_rows.get("WTI", {})
    for lbl in contract_labels:
        b = brent_map.get(lbl)
        w = wti_map.get(lbl)
        if b and w and b["price"] and w["price"]:
            spread_labels.append(lbl)
            spread_vals.append(b["price"] - w["price"])

    if spread_labels:
        spread_colors = ["#059669" if v >= 0 else "#DC2626" for v in spread_vals]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=spread_labels, y=spread_vals,
            marker_color=spread_colors,
            hovertemplate="<b>%{x}</b><br>Spread: $%{y:.2f}<extra></extra>",
        ))
        fig3.add_hline(y=0, line_color="#94A3B8", line_width=1)
        fig3.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
            yaxis=dict(tickprefix="$", gridcolor="#E8EDF5", zeroline=False),
            xaxis=dict(gridcolor="#E8EDF5", zeroline=False, categoryorder="array", categoryarray=contract_labels),
            font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Brent − WTI · positive = Brent premium (normal) · negative = WTI premium (unusual)")
    else:
        st.info("Spread unavailable — need prices for both curves.")


# ── Entry point ────────────────────────────────────────────────────────────────

def _usage_banner():
    """One-line IBKR usage caption + soft-budget warning for this tab's load.
    The Overview loads every curve: ~len(_CURVES) × 12 quote subscriptions."""
    try:
        tod = usage_today()
        pct = tod["pct"].get("mktdata", 0.0)
        st.caption(f"IBKR today: {tod['mktdata']:,} quote subs · {tod['hist']:,} hist "
                   f"— {pct:.0f}% of soft budget")
        lvl, msg = usage_warning("mktdata", len(_CURVES) * 12)
        if lvl != "ok":
            st.warning(msg)
    except Exception:
        pass


@st.fragment
def render_commodities():
    st.markdown("### Commodities — Crude Oil & Natural Gas Futures")

    if not _IB_AVAILABLE:
        st.info(
            "**This tab requires a local IBKR connection.**  \n\n"
            "Run the app locally with TWS open:  \n"
            "```\npip install ib_insync nest_asyncio\nstreamlit run app.py\n```"
        )
        return

    _usage_banner()

    with st.expander("⚙️  IBKR connection", expanded=False):
        cc1, cc2 = st.columns(2)
        with cc1:
            host = st.text_input("Host", value="127.0.0.1", key="com_host")
        with cc2:
            port = st.number_input(
                "Port  (TWS: 7496 · IB Gateway: 4001)",
                value=7496, min_value=1, max_value=65535,
                step=1, key="com_port",
            )

    tab_ovw, tab_brent, tab_wti, tab_ng, tab_uk, tab_ttf = st.tabs(
        ["Overview", "Brent", "WTI", "Nat Gas", "UK Gas", "TTF"]
    )

    with tab_ovw:
        _render_overview(host, int(port))

    with tab_brent:
        _render_curve(host, int(port), "Brent", "brent")

    with tab_wti:
        _render_curve(host, int(port), "WTI", "wti")

    with tab_ng:
        _render_curve(host, int(port), "NatGas", "ng")

    with tab_uk:
        _render_curve(host, int(port), "UKGas", "ukgas")

    with tab_ttf:
        _render_curve(host, int(port), "TTF", "ttf")

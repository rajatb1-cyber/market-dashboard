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

# All TWS I/O goes through the one shared connection (clientId 15, own loop thread).
# Never open our own IB()/clientId here — throwaway clients in the 10-49 range collide
# with the shared connection and exhaust the account-wide market-data line pool.
from ibkr_conn import (get_conn, quotes, hist_bars, PacingBlocked,
                       record_usage, usage_today, usage_warning)

_CONTRACT_STORE: dict = {}   # module-level singleton; accessible from any thread

_DB_PATH    = str(pathlib.Path(__file__).parent / "stir_bars.db")
_PREFS_PATH = str(pathlib.Path(__file__).parent / "stir_prefs.json")
_LOG_PATH   = str(pathlib.Path(__file__).parent / "stir_debug.log")
_DB_READY: bool = False


def _log(msg: str):
    """Append a timestamped line to stir_debug.log in the project folder."""
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


def _load_spread_prefs(n: int = 4) -> tuple[list, list]:
    import json
    try:
        with open(_PREFS_PATH) as f:
            p = json.load(f)
        return p.get("g3_c1", [None] * n), p.get("g3_c2", [None] * n)
    except Exception:
        return [None] * n, [None] * n


def _save_spread_prefs(c1: list, c2: list) -> None:
    import json
    try:
        with open(_PREFS_PATH, "w") as f:
            json.dump({"g3_c1": c1, "g3_c2": c2}, f)
    except Exception:
        pass


def _ensure_db() -> None:
    global _DB_READY
    if _DB_READY:
        return
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS stir_bars (
        symbol TEXT NOT NULL, exchange TEXT NOT NULL,
        exp6 TEXT NOT NULL, bar_date TEXT NOT NULL, close REAL NOT NULL,
        PRIMARY KEY (symbol, exchange, exp6, bar_date))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS stir_meta (
        key TEXT PRIMARY KEY, value TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS stir_con_ids (
        symbol TEXT NOT NULL, exchange TEXT NOT NULL, exp6 TEXT NOT NULL,
        con_id INTEGER NOT NULL,
        PRIMARY KEY (symbol, exchange, exp6))""")
    conn.commit()
    conn.close()
    _DB_READY = True


def _bars_are_fresh(symbol: str, exchange: str) -> bool:
    """True if bars were fetched after today's 17:30, or after yesterday's 17:30 if before 17:30 now."""
    _ensure_db()
    try:
        conn = sqlite3.connect(_DB_PATH)
        cur = conn.execute("SELECT value FROM stir_meta WHERE key=?",
                           (f"last_updated_{symbol}_{exchange}",))
        row = cur.fetchone()
        conn.close()
        if not row:
            return False
        last_upd = datetime.fromisoformat(row[0])
        now = datetime.now()
        cutoff = now.replace(hour=17, minute=30, second=0, microsecond=0)
        if now >= cutoff:
            return last_upd >= cutoff
        return last_upd >= cutoff - timedelta(days=1)
    except Exception:
        return False


def _read_bars_from_db(symbol: str, exchange: str, contracts: tuple) -> dict:
    """Returns {exp6: [(date_str, close), ...]} ordered by date ascending."""
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH)
    result = {}
    try:
        for _, exp6 in contracts:
            cur = conn.execute(
                "SELECT bar_date, close FROM stir_bars "
                "WHERE symbol=? AND exchange=? AND exp6=? ORDER BY bar_date",
                (symbol, exchange, exp6))
            rows = cur.fetchall()
            if rows:
                result[exp6] = [(r[0], float(r[1])) for r in rows]
    finally:
        conn.close()
    return result


def _write_bars_to_db(symbol: str, exchange: str, raw_bars: dict) -> None:
    """raw_bars: {exp6: list of ib_insync Bar objects}"""
    _ensure_db()
    conn = sqlite3.connect(_DB_PATH)
    try:
        for exp6, bars in raw_bars.items():
            for bar in bars:
                dv = bar.date
                ds = dv.strftime('%Y-%m-%d') if hasattr(dv, 'strftime') else str(dv)[:10]
                conn.execute("INSERT OR REPLACE INTO stir_bars VALUES (?,?,?,?,?)",
                             (symbol, exchange, exp6, ds, float(bar.close)))
        conn.execute("INSERT OR REPLACE INTO stir_meta VALUES (?,?)",
                     (f"last_updated_{symbol}_{exchange}", datetime.now().isoformat()))
        conn.commit()
    finally:
        conn.close()


def _missing_duration(symbol: str, exchange: str, exp6: str) -> str:
    """Return IBKR durationStr covering only bars missing since the last DB entry for this contract."""
    _ensure_db()
    try:
        conn = sqlite3.connect(_DB_PATH)
        cur = conn.execute(
            "SELECT MAX(bar_date) FROM stir_bars WHERE symbol=? AND exchange=? AND exp6=?",
            (symbol, exchange, exp6),
        )
        row = cur.fetchone()
        conn.close()
        if not row or not row[0]:
            return "1 Y"
        days = (pd.Timestamp.now().normalize() - pd.Timestamp(row[0]).normalize()).days + 3
        days = max(days, 5)
        return "1 Y" if days >= 366 else f"{days} D"
    except Exception:
        return "1 Y"


def _bars_last_updated(symbol: str, exchange: str) -> str:
    try:
        _ensure_db()
        conn = sqlite3.connect(_DB_PATH)
        cur = conn.execute("SELECT value FROM stir_meta WHERE key=?",
                           (f"last_updated_{symbol}_{exchange}",))
        row = cur.fetchone()
        conn.close()
        if not row:
            return ""
        return datetime.fromisoformat(row[0]).strftime("%d %b %Y %H:%M")
    except Exception:
        return ""


def _bars_data_through(symbol: str, exchange: str) -> str:
    """Return the latest bar_date stored in DB for this symbol+exchange, formatted for display."""
    _ensure_db()
    try:
        conn = sqlite3.connect(_DB_PATH)
        cur = conn.execute(
            "SELECT MAX(bar_date) FROM stir_bars WHERE symbol=? AND exchange=?",
            (symbol, exchange),
        )
        row = cur.fetchone()
        conn.close()
        if not row or not row[0]:
            return ""
        return pd.Timestamp(row[0]).strftime("%d %b %Y")
    except Exception:
        return ""


def _render_bar_status(symbol: str, exchange: str, label: str = "") -> None:
    """Render a compact one-line data freshness indicator."""
    through   = _bars_data_through(symbol, exchange)
    updated   = _bars_last_updated(symbol, exchange)
    prefix    = f"{label} " if label else ""
    if through:
        msg = f"{prefix}Bar data through **{through}**"
        if updated:
            msg += f" · last fetched {updated}"
    else:
        msg = f"{prefix}No bar data in DB yet — click **↺ Update bar data** to load."
    st.caption(msg)


def _force_bars_refresh(symbol: str, exchange: str) -> None:
    """Clear DB freshness flag so next _get_change_bars call re-fetches from IBKR."""
    try:
        _ensure_db()
        conn = sqlite3.connect(_DB_PATH)
        conn.execute("DELETE FROM stir_meta WHERE key=?",
                     (f"last_updated_{symbol}_{exchange}",))
        conn.commit()
        conn.close()
    except Exception:
        pass
    _get_change_bars.clear()


def _contract_store() -> dict:
    return _CONTRACT_STORE


def _read_con_id(symbol: str, exchange: str, exp6: str) -> int | None:
    try:
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute(
            "SELECT con_id FROM stir_con_ids WHERE symbol=? AND exchange=? AND exp6=?",
            (symbol, exchange, exp6),
        ).fetchone()
        conn.close()
        return int(row[0]) if row else None
    except Exception:
        return None


def _write_con_id(symbol: str, exchange: str, exp6: str, con_id: int) -> None:
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO stir_con_ids VALUES (?,?,?,?)",
            (symbol, exchange, exp6, con_id),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _req_contract_details(ibl, contract, timeout_s: int = 15):
    """Run reqContractDetails on the shared connection's loop, time-bounded.

    reqContractDetails can hang in this TWS setup (the historical-bar path avoids it via
    the cached-conId fast path); the ibl.submit timeout plays the role the old throwaway
    client's RequestTimeout=15 did, so a hung lookup can never stall the shared loop.
    Returns [] on any failure — callers treat empty and error identically (fall through)."""
    if ibl is None:
        return []
    record_usage("secdef", 1, tag="stir")
    try:
        return list(ibl.submit(lambda ib: ib.reqContractDetailsAsync(contract),
                               timeout_s=timeout_s) or [])
    except Exception:
        return []


def _resolve_fast(ibl, host: str, port: int,
                  symbol: str, exchange: str, currency: str,
                  exp6: str, trading_class: str, local_sym_prefix: str):
    """_resolve with two-level cache: in-process dict first, then DB conId, then a
    contract-details lookup on the shared connection."""
    from ib_insync import Contract
    store = _contract_store()
    key   = (host, port, symbol, exchange, currency, trading_class, local_sym_prefix, exp6)
    if key in store:
        _log(f"[stir]   {exp6}: in-process cache hit conId={store[key].conId} sym={store[key].localSymbol}")
        return store[key], 1

    # Use DB-cached conId to get fully-qualified contract; the lookup is time-bounded on
    # the shared loop so a hung reqContractDetails can't stall the connection.
    con_id = _read_con_id(symbol, exchange, exp6)
    if con_id:
        _log(f"[stir]   {exp6}: cached conId={con_id}, reqContractDetails…")
        details = _req_contract_details(ibl, Contract(conId=con_id, exchange=exchange))
        if details:
            qc = details[0].contract
            store[key] = qc
            return qc, 1
        _log(f"[stir]   {exp6}: conId lookup empty, falling to full resolve")

    _log(f"[stir]   {exp6}: full resolve…")
    qc, n = _resolve(ibl, symbol, exchange, currency, exp6, trading_class, local_sym_prefix)
    if qc:
        store[key] = qc
        _write_con_id(symbol, exchange, exp6, qc.conId)
    return qc, n


_MONTH_NAMES  = {3: "Mar", 6: "Jun", 9: "Sep", 12: "Dec"}
_MONTH_CODES  = {"03": "H", "06": "M", "09": "U", "12": "Z"}  # IBKR futures month codes
_DURATION_MAP = {"1M": "1 M", "3M": "3 M", "6M": "6 M", "1Y": "1 Y"}

# Contract specs for each curve
_CURVES = {
    "Euribor": {
        "symbol":           "I",
        "exchange":         "ICEEU",
        "currency":         "EUR",
        "tradingClass":     "",
        "local_sym_prefix": "I",
        "subtitle":         "Euribor 3M Futures (ICE/LIFFE)",
        "cb_label":         "Implied ECB Rate Path",
    },
    "SONIA": {
        "symbol":          "SO3",
        "exchange":        "ICEEU",
        "currency":        "GBP",
        "tradingClass":    "",
        "local_sym_prefix": "SO3",   # localSymbol = SO3 + month_code + year_digit, e.g. SO3Z6
        "subtitle":        "SONIA Futures (ICE/LIFFE)",
        "cb_label":        "Implied BoE Rate Path",
    },
    "SOFR": {
        "symbol":          "SR3",
        "exchange":        "CME",
        "currency":        "USD",
        "tradingClass":    "",
        "local_sym_prefix": "SR3",   # localSymbol = SR3 + month_code + year_digit, e.g. SR3Z6
        "subtitle":        "SOFR 3M Futures (CME)",
        "cb_label":        "Implied Fed Rate Path",
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


def _resolve(ibl, symbol, exchange, currency, exp6, trading_class="", local_sym_prefix=""):
    """Return a qualified contract via a contract-details lookup on the shared connection
    (handles ambiguous results). If local_sym_prefix is set, tries localSymbol lookup
    first (e.g. SO3Z6 for SONIA)."""
    from ib_insync import Contract

    if local_sym_prefix:
        mc = _MONTH_CODES.get(exp6[4:6], "")
        if mc:
            local_sym = f"{local_sym_prefix}{mc}{exp6[3]}"
            _log(f"[stir]     localSym lookup: {local_sym}")
            det = _req_contract_details(
                ibl, Contract(secType="FUT", localSymbol=local_sym, exchange=exchange))
            if det:
                exact = [d for d in det if d.contract.lastTradeDateOrContractMonth[:6] == exp6]
                chosen = exact or det
                _log(f"[stir]     localSym ok: conId={chosen[0].contract.conId} sym={chosen[0].contract.localSymbol}")
                return chosen[0].contract, len(det)
            else:
                _log(f"[stir]     localSym {local_sym}: empty")

    _log(f"[stir]     full lookup: {symbol}/{exchange}/{exp6}")
    kwargs = dict(symbol=symbol, exchange=exchange, currency=currency,
                  lastTradeDateOrContractMonth=exp6)
    if trading_class:
        kwargs["tradingClass"] = trading_class
    fut = Future(**kwargs)
    details = _req_contract_details(ibl, fut)
    if not details:
        _log(f"[stir]     full lookup: empty")
        return None, 0
    exact = [d for d in details if d.contract.lastTradeDateOrContractMonth[:6] == exp6]
    chosen = exact or details
    _log(f"[stir]     full ok: conId={chosen[0].contract.conId} sym={chosen[0].contract.localSymbol}")
    return chosen[0].contract, len(details)


def _fetch_bars_from_ibkr(host: str, port: int, contracts: tuple,
                          symbol: str, exchange: str, currency: str,
                          trading_class: str = "", local_sym_prefix: str = "") -> int:
    """Fetch daily bars from IBKR (shared connection) and write to DB. Returns number of
    contracts with bars. This is the DB *updater*: if the connection is down or pacing
    blocks a request, the existing DB bars simply stand."""
    from ib_insync import Contract
    _raw: dict = {}

    ibl, err = get_conn()
    if ibl is None:
        _log(f"[stir] {symbol}/{exchange} no TWS connection: {err}")
        return 0

    _log(f"[stir] {symbol}/{exchange} connected, resolving {len(contracts)} contracts…")
    for lbl, exp6 in contracts:
        # Fast path: hand the cached conId straight to reqHistoricalData — IBKR resolves
        # by conId, so we SKIP reqContractDetails (which hangs ~15s per contract in this
        # TWS setup). Full resolve only when we have no cached conId yet.
        con_id = _read_con_id(symbol, exchange, exp6)
        if con_id:
            qc = Contract(conId=int(con_id), exchange=exchange)
            _log(f"[stir]   {exp6}: cached conId={con_id} → hist_bars direct")
        else:
            qc, _ = _resolve_fast(ibl, host, port, symbol, exchange, currency,
                                  exp6, trading_class, local_sym_prefix)
        if qc is None:
            _log(f"[stir]   {exp6}: resolve=None, skipping")
            continue
        duration = _missing_duration(symbol, exchange, exp6)
        bars = None
        for show in ("TRADES", "LAST", "MIDPOINT"):
            try:
                b = hist_bars(qc, endDateTime="", durationStr=duration,
                              barSizeSetting="1 day", whatToShow=show,
                              useRTH=False, timeout_s=15, ibl=ibl, tag="stir")
            except PacingBlocked as _pe:
                # Account-wide pacing budget hit — keep existing DB bars for this contract.
                _log(f"[stir]   {exp6}: paced — keeping DB bars: {_pe}")
                break
            except Exception as _be:
                _log(f"[stir]   {exp6} {show} failed: {_be}")
                continue
            if b:
                bars = b
                _log(f"[stir]   {exp6}: {len(b)} bars via {show}")
                break
            else:
                _log(f"[stir]   {exp6} {show}: empty response")
        if not bars:
            _log(f"[stir]   {exp6}: no bars from any show type")
        else:
            _raw[exp6] = bars
    _log(f"[stir] {symbol}/{exchange} done: {len(_raw)} contracts with bars")

    if _raw:
        try:
            _write_bars_to_db(symbol, exchange, _raw)
        except Exception:
            pass

    return len(_raw)


@st.cache_data(ttl=300)
def _get_change_bars(symbol: str, exchange: str, contracts: tuple) -> dict:
    """DB-only bar reader for change calculations. Call _fetch_bars_from_ibkr to update."""
    return _read_bars_from_db(symbol, exchange, contracts)



@st.cache_data(ttl=None)
def _fetch_quotes(host: str, port: int, contracts: tuple,
                  symbol: str, exchange: str, currency: str,
                  trading_class: str = "", local_sym_prefix: str = "") -> tuple:
    """Returns (rows, debug) with live prices and change columns."""
    bars_by_exp = _get_change_bars(symbol, exchange, contracts)

    _rows:  list = []
    _debug: dict = {"qual": [], "ticks": []}

    # Live prices come off the one shared connection; quotes() subscribes in bounded
    # chunks and always frees the market-data lines in a finally.
    ibl, err = get_conn()
    if ibl is None:
        _debug["error"] = err
        return _rows, _debug

    label_by_exp6 = {exp: lbl for lbl, exp in contracts}
    store = _contract_store()
    quads = []   # (qc, label, exp6) parallel to the tickers we request
    for exp6, label in label_by_exp6.items():
        # Use cached qualified contract if available; otherwise construct directly
        # from known fields — reqMktData resolves it, so no reqContractDetails round-trip.
        key = (host, port, symbol, exchange, currency,
               trading_class, local_sym_prefix, exp6)
        if key in store:
            qc     = store[key]
            status = "cached"
        else:
            qc = Future(symbol=symbol, exchange=exchange, currency=currency,
                        lastTradeDateOrContractMonth=exp6)
            if trading_class:
                qc.tradingClass = trading_class
            status = "direct"
        _debug["qual"].append({"exp": exp6, "label": label, "status": status})
        quads.append((qc, label, exp6))

    if not quads:
        return _rows, _debug

    # mdtype=2 (frozen) + settle_s=2.0 matches the old reqMarketDataType(2)+sleep(2);
    # every subscription is cancelled inside quotes() so lines never leak.
    tickers = quotes([qc for qc, _, _ in quads], mdtype=2, settle_s=2.0, ibl=ibl, tag="stir")

    for (qc, label, expiry), ticker in zip(quads, tickers):
        if ticker is None:
            continue
        last  = _v(ticker.last)
        close = _v(ticker.close)
        bid   = _v(ticker.bid)
        ask   = _v(ticker.ask)
        vol   = ticker.volume
        mdt   = getattr(ticker, "marketDataType", "?")

        mid        = (bid + ask) / 2 if (bid and ask) else None
        live_price = last or mid

        bc_raw = bars_by_exp.get(expiry, [])
        today_ts = pd.Timestamp.now(tz=None).normalize()

        if bc_raw:
            bc_dated = [(pd.Timestamp(d), float(c)) for d, c in bc_raw]
            bar_close = bc_dated[-1][1]
            prev_bars = [(d, c) for d, c in bc_dated if d < today_ts]
            bar_prev  = prev_bars[-1][1] if prev_bars else None

            def _ref_at(days, _bc=bc_dated):
                tgt = today_ts - pd.Timedelta(days=days)
                cands = [c for d, c in _bc if d <= tgt]
                return cands[-1] if cands else None

            ref_1w = _ref_at(7)
            ref_1m = _ref_at(30)
            ref_3m = _ref_at(90)
        else:
            bar_close = bar_prev = ref_1w = ref_1m = ref_3m = None

        hist_price = None
        if not live_price and bar_close:
            hist_price = bar_close
            live_price = hist_price
        close = bar_prev or close or bar_close

        src = "last" if last else ("mid" if mid else ("hist" if hist_price else "close"))

        _debug["ticks"].append({
            "Contract": label, "last": last, "bid": bid,
            "ask": ask, "close": close, "hist": hist_price, "mktDataType": mdt,
        })

        display_price = live_price or close
        if display_price is None:
            continue

        ref_1d = close

        def _bps(ref, _lp=live_price):
            return (ref - _lp) * 100 if (_lp and ref) else None

        _rows.append({
            "label":     label,
            "expiry":    expiry,
            "price":     live_price,
            "close":     close,
            "impl_rate": 100 - display_price,
            "chg_1d":    _bps(ref_1d),
            "chg_1w":    _bps(ref_1w),
            "chg_1m":    _bps(ref_1m),
            "chg_3m":    _bps(ref_3m),
            "volume":    int(vol) if vol and vol > 0 else None,
            "src":       src,
        })

    return _rows, _debug


_HIST_DURATION_DAYS = {"1 M": 31, "3 M": 93, "6 M": 186, "1 Y": 366}


def _ref_prices_from_db(contracts: tuple, ref_date_str: str,
                        symbol: str, exchange: str) -> dict:
    """Look up close price at or before ref_date for each contract from stir_bars DB."""
    ref_dt = pd.Timestamp(ref_date_str).normalize()
    results: dict = {}
    _ensure_db()
    try:
        conn = sqlite3.connect(_DB_PATH)
        for _lbl, exp6 in contracts:
            cur = conn.execute(
                "SELECT close FROM stir_bars "
                "WHERE symbol=? AND exchange=? AND exp6=? AND bar_date<=? "
                "ORDER BY bar_date DESC LIMIT 1",
                (symbol, exchange, exp6, ref_dt.strftime("%Y-%m-%d")),
            )
            row = cur.fetchone()
            if row:
                results[_lbl] = float(row[0])
        conn.close()
    except Exception:
        pass
    return results


def _read_history_from_db(symbol: str, exchange: str, exp6: str, duration_str: str):
    """Read contract history from stir_bars DB table. Returns (df, error_str).
    Returns whatever data exists — caller falls back to IBKR only if df is completely empty."""
    _ensure_db()
    try:
        conn = sqlite3.connect(_DB_PATH)
        cur = conn.execute(
            "SELECT bar_date, close FROM stir_bars WHERE symbol=? AND exchange=? AND exp6=?"
            " ORDER BY bar_date",
            (symbol, exchange, exp6))
        rows_db = cur.fetchall()
        conn.close()
        if not rows_db:
            return pd.DataFrame(), "no DB data"
        df = pd.DataFrame(rows_db, columns=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        days = _HIST_DURATION_DAYS.get(duration_str, 93)
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        df_period = df[df.index >= cutoff]
        # Return whatever we have — full period if covered, partial if not
        df_out = df_period if not df_period.empty else df
        df_out["implied_rate"] = 100 - df_out["close"]
        return df_out.dropna(), ""
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=None)
def _fetch_history(host: str, port: int, expiry: str, duration: str,
                   symbol: str, exchange: str, currency: str,
                   trading_class: str = "", local_sym_prefix: str = "") -> tuple:
    """Returns (df, error_str) — DB first, IBKR fallback."""
    df_db, _ = _read_history_from_db(symbol, exchange, expiry, duration)
    if not df_db.empty:
        return df_db, ""

    ibl, err = get_conn()
    if ibl is None:
        return pd.DataFrame(), err or "TWS not connected"

    resolved, _ = _resolve_fast(ibl, host, port, symbol, exchange, currency,
                                expiry, trading_class, local_sym_prefix)
    if resolved is None:
        return pd.DataFrame(), f"Could not resolve contract {symbol} {expiry} on {exchange}"

    bars = None
    for show in ("TRADES", "LAST", "MIDPOINT"):
        try:
            bars = hist_bars(resolved, endDateTime="", durationStr=duration,
                             barSizeSetting="1 day", whatToShow=show,
                             useRTH=False, timeout_s=15, ibl=ibl, tag="stir")
        except PacingBlocked as e:
            # Paced account-wide — serve whatever the DB has (empty here → "no data").
            _log(f"[stir] paced — serving DB bars: {e}")
            return df_db, ""
        except Exception:
            bars = None
        if bars:
            break

    if not bars:
        return pd.DataFrame(), (
            f"No bars returned for conId={resolved.conId} "
            f"expiry={resolved.lastTradeDateOrContractMonth} dur={duration}"
        )

    try:
        df = util.df(bars)[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date").sort_index()
        df["implied_rate"] = 100 - df["close"]
        return df.dropna(), ""
    except Exception as e:
        return pd.DataFrame(), str(e)


@st.cache_data(ttl=300)
def _scan_symbol(host: str, port: int, symbol: str, exchange: str) -> list:
    """Probe IBKR with reqContractDetails (on the shared connection) to find the right spec.
    Diagnostic path only — runs when a curve fails to resolve; each lookup is time-bounded
    so a hung reqContractDetails can't stall the shared loop."""
    from ib_insync import Contract
    results: list = []

    ibl, err = get_conn()
    if ibl is None:
        results.append({"error": err or "TWS not connected"})
        return results

    # If a known conId is provided, probe it first — most reliable path
    known_con_ids = {
        "SONIA": 522330875,   # SONIA.N Dec'26 — used to discover full spec
    }
    if symbol in known_con_ids:
        details = _req_contract_details(
            ibl, Contract(conId=known_con_ids[symbol], exchange=exchange))
        if details:
            for d in details[:1]:
                ct = d.contract
                results.append({
                    "tried":        f"conId={known_con_ids[symbol]}",
                    "conId":        ct.conId,
                    "symbol":       ct.symbol,
                    "localSymbol":  ct.localSymbol,
                    "tradingClass": ct.tradingClass,
                    "exchange":     ct.exchange,
                    "currency":     ct.currency,
                    "expiry":       ct.lastTradeDateOrContractMonth,
                })
        else:
            results.append({"tried": f"conId={known_con_ids[symbol]}", "error": "no details returned"})

    def _probe(label, contract_kwargs):
        details = _req_contract_details(ibl, Contract(**contract_kwargs))
        for d in details[:3]:
            c = d.contract
            results.append({
                "tried":        label,
                "conId":        c.conId,
                "symbol":       c.symbol,
                "localSymbol":  c.localSymbol,
                "tradingClass": c.tradingClass,
                "exchange":     c.exchange,
                "currency":     c.currency,
                "expiry":       c.lastTradeDateOrContractMonth,
            })

    # 1. localSymbol lookup — TWS shows "SONIA.N" so try that directly
    _probe(f"localSymbol={symbol}@{exchange}",
           dict(secType="FUT", localSymbol=symbol, exchange=exchange))
    _probe(f"localSymbol={symbol}@any",
           dict(secType="FUT", localSymbol=symbol))

    # 2. symbol variants × exchange variants
    for sym in [symbol, symbol.replace(".", ""), symbol.split(".")[0]]:
        for exch in [exchange, ""]:
            _probe(f"symbol={sym}@{exch or 'any'}",
                   dict(secType="FUT", symbol=sym, exchange=exch) if exch
                   else dict(secType="FUT", symbol=sym))

    return results


@st.fragment
def _render_curve(host: str, port: int, curve_name: str, tab_key: str):
    """Render the full table + chart + history for one curve."""
    cfg              = _CURVES[curve_name]
    symbol           = cfg["symbol"]
    exchange         = cfg["exchange"]
    currency         = cfg["currency"]
    trading_class    = cfg.get("tradingClass", "")
    local_sym_prefix = cfg.get("local_sym_prefix", "")
    contracts        = _active_contracts(12)

    btn_c1, btn_c2 = st.columns([2, 3])
    with btn_c1:
        if st.button("⟳ Refresh prices", key=f"stir_refresh_{tab_key}"):
            _fetch_quotes.clear()
    with btn_c2:
        if st.button("↺ Update bar data from IBKR", key=f"stir_barrefresh_{tab_key}"):
            with st.spinner(f"Fetching {curve_name} bars from IBKR…"):
                n = _fetch_bars_from_ibkr(host, port, tuple(contracts),
                                          symbol, exchange, currency,
                                          trading_class, local_sym_prefix)
            _get_change_bars.clear()
            _fetch_quotes.clear()
            if n:
                st.toast(f"Updated {n} contracts. Prices refreshing…", icon="✅")
            else:
                st.toast("No new bars returned — check IBKR connection.", icon="⚠️")
    _render_bar_status(symbol, exchange)

    fetch_ts = time.time()
    t0 = time.perf_counter()
    with st.spinner(f"Fetching live prices ({curve_name})…"):
        try:
            rows, debug = _fetch_quotes(
                host, port, tuple(contracts), symbol, exchange, currency,
                trading_class, local_sym_prefix
            )
        except Exception as e:
            st.error(
                f"**Could not connect to IBKR** (`{host}:{port}`).  \n"
                f"Error: `{e}`"
            )
            return
    quotes_ms = (time.perf_counter() - t0) * 1000

    if not rows:
        err = debug.get("error", "")
        if err:
            st.error(f"**IBKR connection error:** `{err}`")
        else:
            st.warning(f"No quotes returned for {curve_name} — could not resolve contracts.")
        with st.expander("Debug — qualification", expanded=True):
            st.dataframe(pd.DataFrame(debug.get("qual", [])), use_container_width=True)
        if not err:
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
    bars_ts = _bars_last_updated(symbol, exchange)
    bars_note = f" · Bars updated {bars_ts}" if bars_ts else " · Bars: not yet cached"
    quotes_src = "cache" if quotes_ms < 100 else "IBKR"
    st.caption(f"IBKR · Price = 100 − Rate · Prices {updated} · {src_str}{bars_note} · quotes {quotes_ms:.0f}ms [{quotes_src}]")

    def _chg_color(bps):
        if bps is None:
            return ""
        return ("color:#DC2626;font-weight:600" if bps > 0 else
                "color:#059669;font-weight:600" if bps < 0 else "")

    table_rows  = []
    col_1d, col_1w, col_1m, col_3m, rate_colors = [], [], [], [], []

    for r in rows:
        col_1d.append(_chg_color(r["chg_1d"]))
        col_1w.append(_chg_color(r["chg_1w"]))
        col_1m.append(_chg_color(r["chg_1m"]))
        col_3m.append(_chg_color(r["chg_3m"]))
        rate_colors.append(_chg_color(r["chg_1d"]))

        price_str = (_fmt(r["price"], "{:.3f}") if r["price"] is not None
                     else _fmt(r["close"], "{:.3f}*"))

        table_rows.append({
            "Contract":   r["label"],
            "Price":      price_str,
            "Impl. Rate": _fmt(r["impl_rate"], "{:.3f}%"),
            "1D (bps)":   _fmt(r["chg_1d"], "{:+.1f}"),
            "1W (bps)":   _fmt(r["chg_1w"], "{:+.1f}"),
            "1M (bps)":   _fmt(r["chg_1m"], "{:+.1f}"),
            "3M (bps)":   _fmt(r["chg_3m"], "{:+.1f}"),
            "Volume":     _fmt(r["volume"], "{:,}"),
        })

    df_tbl = pd.DataFrame(table_rows).set_index("Contract")
    styled = (
        df_tbl.style
        .apply(lambda _: col_1d,      subset=["1D (bps)"])
        .apply(lambda _: col_1w,      subset=["1W (bps)"])
        .apply(lambda _: col_1m,      subset=["1M (bps)"])
        .apply(lambda _: col_3m,      subset=["3M (bps)"])
        .apply(lambda _: rate_colors, subset=["Impl. Rate"])
    )
    st.dataframe(styled, use_container_width=True)

    # ── Rate curve ─────────────────────────────────────────────────────────────
    st.markdown(f"#### {cfg['cb_label']}")

    # Historical curve overlay selector
    cmp_c1, cmp_c2 = st.columns([5, 3])
    with cmp_c1:
        cmp_period = st.radio(
            "Compare vs", ["—", "1D", "1W", "1M", "3M", "Custom"],
            index=0, horizontal=True, key=f"curve_cmp_{tab_key}",
        )
    cmp_date = None
    if cmp_period == "Custom":
        with cmp_c2:
            cmp_date = st.date_input(
                "Date",
                value=date.today() - timedelta(days=30),
                max_value=date.today() - timedelta(days=1),
                key=f"curve_cmp_date_{tab_key}",
            )

    # Derive historical implied rates
    _cmp_chg_key = {"1D": "chg_1d", "1W": "chg_1w", "1M": "chg_1m", "3M": "chg_3m"}
    hist_curve = None  # list of (label, rate | None)

    if cmp_period in _cmp_chg_key:
        key = _cmp_chg_key[cmp_period]
        hist_curve = [
            (r["label"],
             r["impl_rate"] - r[key] / 100
             if (r["impl_rate"] is not None and r[key] is not None) else None)
            for r in rows
        ]
    elif cmp_period == "Custom" and cmp_date:
        with st.spinner(f"Loading curve as of {cmp_date}…"):
            try:
                ref_prices = _fetch_ref_prices_curve(
                    host, port, tuple(contracts), str(cmp_date),
                    symbol, exchange, currency, trading_class, local_sym_prefix,
                )
            except Exception:
                ref_prices = {}
        hist_curve = [
            (r["label"],
             100 - ref_prices[r["label"]] if r["label"] in ref_prices else None)
            for r in rows
        ]

    dot_colors = [
        "#DC2626" if (r["chg_1d"] or 0) > 0 else
        ("#059669" if (r["chg_1d"] or 0) < 0 else "#94A3B8")
        for r in rows
    ]  # red = rates priced higher, green = lower

    show_legend = hist_curve is not None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["label"] for r in rows],
        y=[r["impl_rate"] for r in rows],
        mode="lines+markers",
        name="Current",
        showlegend=show_legend,
        line=dict(color="#0EA5E9", width=2),
        marker=dict(size=9, color=dot_colors, line=dict(color="#FFFFFF", width=1.5)),
        hovertemplate="<b>Current</b>  %{x}<br>%{y:.3f}%<extra></extra>",
    ))

    if hist_curve:
        valid_pts = [(lbl, r) for lbl, r in hist_curve if r is not None]
        if valid_pts:
            cmp_label = cmp_period if cmp_period != "Custom" else str(cmp_date)
            fig.add_trace(go.Scatter(
                x=[lbl for lbl, _ in valid_pts],
                y=[r   for _, r   in valid_pts],
                mode="lines+markers",
                name=cmp_label,
                line=dict(color="#DC2626", width=1.5, dash="dot"),
                marker=dict(size=6, color="#DC2626"),
                hovertemplate=f"<b>{cmp_label}</b>  %{{x}}<br>%{{y:.3f}}%<extra></extra>",
            ))

    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        yaxis=dict(ticksuffix="%", gridcolor="#E8EDF5", zeroline=False),
        xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1) if show_legend else {},
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

    t1 = time.perf_counter()
    with st.spinner(f"Loading {sel_label} history…"):
        try:
            df_hist, hist_err = _fetch_history(
                host, port, sel_expiry, ibkr_dur, symbol, exchange, currency,
                trading_class, local_sym_prefix
            )
        except Exception as e:
            st.error(f"History fetch error: `{e}`")
            df_hist, hist_err = pd.DataFrame(), ""
    hist_ms = (time.perf_counter() - t1) * 1000

    if hist_err:
        st.error(f"History error: {hist_err}")

    if not df_hist.empty:
        first_rate = float(df_hist["implied_rate"].iloc[0])
        last_rate  = float(df_hist["implied_rate"].iloc[-1])
        line_color = "#DC2626" if last_rate > first_rate else "#059669"
        fill_color = ("rgba(220,38,38,0.07)" if line_color == "#DC2626"
                      else "rgba(5,150,105,0.07)")

        y_vals = df_hist["implied_rate"].dropna()
        y_min  = float(y_vals.min())
        y_max  = float(y_vals.max())
        pad    = max((y_max - y_min) * 0.15, 0.02)
        y_lo   = y_min - pad
        y_hi   = y_max + pad

        fig2 = go.Figure()
        # shaded area anchored to y_lo (not zero) so axis stays tight
        fig2.add_trace(go.Scatter(
            x=list(df_hist.index) + list(df_hist.index[::-1]),
            y=list(y_vals) + [y_lo] * len(y_vals),
            fill="toself", fillcolor=fill_color,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig2.add_trace(go.Scatter(
            x=df_hist.index, y=df_hist["implied_rate"],
            mode="lines", name=sel_label,
            line=dict(color=line_color, width=2),
            hovertemplate="%{x|%d %b %Y}<br><b>%{y:.3f}%</b><extra></extra>",
        ))
        hist_src = "cache" if hist_ms < 100 else "IBKR"
        fig2.update_layout(
            title=dict(
                text=(f"{sel_label} — Implied Rate  "
                      f"<span style='font-size:12px;color:#64748B'>"
                      f"current: {last_rate:.3f}%  ·  {hist_ms:.0f}ms [{hist_src}]</span>"),
                font=dict(size=13, color="#1A202C"),
            ),
            height=280, margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
            yaxis=dict(ticksuffix="%", gridcolor="#E8EDF5", zeroline=False,
                       range=[y_lo, y_hi]),
            xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
            hovermode="x unified",
            font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
        )
        st.plotly_chart(fig2, use_container_width=True)
    elif not hist_err:
        st.info("No history data available for this contract / period.")

    # ── Spread chart ───────────────────────────────────────────────────────────
    st.markdown("#### Spread")
    sp1, sp2, sp3 = st.columns([3, 3, 2])
    labels_list = [r["label"] for r in rows]
    with sp1:
        sprd_label1 = st.selectbox("Contract 1", labels_list, index=0,
                                   key=f"stir_sprd1_{tab_key}")
    with sp2:
        sprd_label2 = st.selectbox("Contract 2", labels_list,
                                   index=min(1, len(labels_list) - 1),
                                   key=f"stir_sprd2_{tab_key}")
    with sp3:
        sprd_period = st.selectbox("Period", ["1M", "3M", "6M", "1Y"], index=1,
                                   key=f"stir_sprd_period_{tab_key}")

    if sprd_label1 == sprd_label2:
        st.info("Select two different contracts to see the spread.")
    else:
        sprd_exp1 = next(r["expiry"] for r in rows if r["label"] == sprd_label1)
        sprd_exp2 = next(r["expiry"] for r in rows if r["label"] == sprd_label2)
        sprd_dur  = _DURATION_MAP[sprd_period]

        t2 = time.perf_counter()
        df1, err1 = _fetch_history(host, port, sprd_exp1, sprd_dur,
                                   symbol, exchange, currency, trading_class, local_sym_prefix)
        df2, err2 = _fetch_history(host, port, sprd_exp2, sprd_dur,
                                   symbol, exchange, currency, trading_class, local_sym_prefix)
        sprd_ms = (time.perf_counter() - t2) * 1000

        if err1 or err2:
            st.error(f"Spread history error: {err1 or err2}")
        elif df1.empty or df2.empty:
            st.info("Not enough data for spread calculation.")
        else:
            df_sprd = pd.DataFrame({"c1": df1["close"], "c2": df2["close"]}).dropna()
            df_sprd["spread_bps"] = ((df_sprd["c1"] - df_sprd["c2"]) * 100).round(2)

            y_vals   = df_sprd["spread_bps"].dropna()
            first_s  = float(y_vals.iloc[0])
            last_s   = float(y_vals.iloc[-1])
            line_col = "#DC2626" if last_s > first_s else "#059669"
            fill_col = ("rgba(220,38,38,0.07)" if line_col == "#DC2626"
                        else "rgba(5,150,105,0.07)")
            pad  = max((float(y_vals.max()) - float(y_vals.min())) * 0.15, 0.5)
            y_lo = float(y_vals.min()) - pad
            y_hi = float(y_vals.max()) + pad

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=list(df_sprd.index) + list(df_sprd.index[::-1]),
                y=list(y_vals) + [y_lo] * len(y_vals),
                fill="toself", fillcolor=fill_col,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            sprd_name  = f"{sprd_label2}−{sprd_label1} Spread"
            hover_text = [f"{v:+.2f}" for v in y_vals]
            fig3.add_trace(go.Scatter(
                x=df_sprd.index, y=y_vals,
                mode="lines", name=sprd_name,
                line=dict(color=line_col, width=2),
                customdata=hover_text,
                hovertemplate="%{x|%d %b %Y}<br><b>%{customdata} bps</b><extra></extra>",
            ))
            sprd_src = "cache" if sprd_ms < 100 else "IBKR"
            fig3.update_layout(
                title=dict(
                    text=(f"Spread: {sprd_label2} − {sprd_label1}  "
                          f"<span style='font-size:12px;color:#64748B'>"
                          f"current: {last_s:+.1f} bps  ·  {sprd_ms:.0f}ms [{sprd_src}]</span>"),
                    font=dict(size=13, color="#1A202C"),
                ),
                height=280, margin=dict(l=10, r=10, t=45, b=10),
                paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                yaxis=dict(ticksuffix=" bps", tickformat=".2f", gridcolor="#E8EDF5",
                           zeroline=True, zerolinecolor="#94A3B8", zerolinewidth=1,
                           range=[y_lo, y_hi]),
                xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
                hovermode="x unified",
                font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
            )
            st.plotly_chart(fig3, use_container_width=True)


@st.cache_data(ttl=300)
def _fetch_ref_prices_curve(host: str, port: int, contracts: tuple,
                             ref_date_str: str, symbol: str, exchange: str,
                             currency: str, trading_class: str = "",
                             local_sym_prefix: str = "") -> dict:
    """Return {label: price_at_ref_date} for all contracts (for custom change calc).
    Uses stir_bars DB first (covers the last 1Y); falls back to IBKR only for gaps."""
    # DB covers ~1Y of daily bars — answer directly without touching IBKR
    results = _ref_prices_from_db(contracts, ref_date_str, symbol, exchange)

    missing = [lbl for lbl, _ in contracts if lbl not in results]
    if not missing:
        return results

    # IBKR fallback for contracts not in DB (e.g. date older than 1Y)
    ibl, err = get_conn()
    if ibl is None:
        return results

    ref_dt = pd.Timestamp(ref_date_str)
    days_back = max((pd.Timestamp.now().normalize() - ref_dt).days + 10, 15)
    duration = f"{min(days_back, 365)} D" if days_back <= 365 else f"{days_back // 365 + 1} Y"

    missing_set = set(missing)
    label_by_exp6 = {exp: lbl for lbl, exp in contracts}
    for exp6, label in label_by_exp6.items():
        if label not in missing_set:
            continue
        qc, _ = _resolve_fast(ibl, host, port, symbol, exchange, currency,
                              exp6, trading_class, local_sym_prefix)
        if qc is None:
            continue
        bars = None
        for show in ("TRADES", "LAST", "MIDPOINT"):
            try:
                b = hist_bars(qc, endDateTime="", durationStr=duration,
                              barSizeSetting="1 day", whatToShow=show,
                              useRTH=False, timeout_s=15, ibl=ibl, tag="stir")
            except PacingBlocked as e:
                # Paced — leave this contract to its DB value (may stay missing).
                _log(f"[stir] paced — serving DB ref prices: {e}")
                break
            except Exception:
                continue
            if b:
                bars = b
                break
        if bars:
            df = util.df(bars)[["date", "close"]].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.set_index("date").sort_index()
            candidates = df[df.index <= ref_dt]
            if not candidates.empty:
                results[label] = float(candidates["close"].iloc[-1])

    return results


_G3_CURVES = [
    ("USD", "SOFR",    "#0EA5E9"),
    ("EUR", "Euribor", "#8B5CF6"),
    ("GBP", "SONIA",   "#F59E0B"),
]


@st.fragment
def _render_g3(host: str, port: int):
    st.markdown("#### G3 Implied Rate Overview")

    gbtn_c1, gbtn_c2 = st.columns([2, 3])
    with gbtn_c1:
        if st.button("⟳ Refresh prices", key="stir_refresh_g3"):
            _fetch_quotes.clear()
    with gbtn_c2:
        if st.button("↺ Update bar data from IBKR", key="stir_barrefresh_g3"):
            g3_contracts = tuple(_active_contracts(12))
            total = 0
            for _, curve_name2, _ in _G3_CURVES:
                cfg2 = _CURVES[curve_name2]
                with st.spinner(f"Fetching {curve_name2} bars from IBKR…"):
                    total += _fetch_bars_from_ibkr(
                        host, port, g3_contracts,
                        cfg2["symbol"], cfg2["exchange"], cfg2["currency"],
                        cfg2.get("tradingClass", ""), cfg2.get("local_sym_prefix", ""),
                    )
            _get_change_bars.clear()
            _fetch_quotes.clear()
            if total:
                st.toast(f"Updated {total} contracts across G3.", icon="✅")
            else:
                st.toast("No new bars returned — check IBKR connection.", icon="⚠️")

    for ccy, curve_name, _ in _G3_CURVES:
        cfg = _CURVES[curve_name]
        _render_bar_status(cfg["symbol"], cfg["exchange"], label=f"{ccy}:")

    contracts = _active_contracts(12)
    contract_labels = [lbl for lbl, _ in contracts]

    all_rows: dict[str, dict] = {}
    for ccy, curve_name, _ in _G3_CURVES:
        cfg = _CURVES[curve_name]
        with st.spinner(f"Loading {ccy} ({curve_name})…"):
            try:
                rows, _ = _fetch_quotes(
                    host, port, tuple(contracts),
                    cfg["symbol"], cfg["exchange"], cfg["currency"],
                    cfg.get("tradingClass", ""), cfg.get("local_sym_prefix", ""),
                )
                all_rows[ccy] = {r["label"]: r for r in rows}
            except Exception:
                all_rows[ccy] = {}

    bar_parts = []
    for ccy, curve_name, _ in _G3_CURVES:
        cfg = _CURVES[curve_name]
        ts = _bars_last_updated(cfg["symbol"], cfg["exchange"])
        bar_parts.append(f"{ccy}: {ts or 'not cached'}")
    st.caption("Bars last updated — " + " · ".join(bar_parts))

    # ── Changes table ─────────────────────────────────────────────────────────
    st.markdown("##### Changes (bps)")
    pc1, pc2 = st.columns([4, 3])
    with pc1:
        period = st.radio("Period", ["1D", "1W", "1M", "3M", "Custom"],
                          index=0, horizontal=True, key="g3_chg_period")
    custom_date = None
    if period == "Custom":
        with pc2:
            custom_date = st.date_input(
                "Reference date",
                value=date.today() - timedelta(days=30),
                max_value=date.today() - timedelta(days=1),
                key="g3_custom_date",
            )

    _chg_key = {"1D": "chg_1d", "1W": "chg_1w", "1M": "chg_1m", "3M": "chg_3m"}

    chg_vals: dict[str, dict] = {}
    if period in _chg_key:
        key = _chg_key[period]
        for ccy, _, _ in _G3_CURVES:
            chg_vals[ccy] = {
                lbl: (r[key] if (r := all_rows.get(ccy, {}).get(lbl)) else None)
                for lbl in contract_labels
            }
    else:
        ref_str = str(custom_date)
        for ccy, curve_name, _ in _G3_CURVES:
            cfg = _CURVES[curve_name]
            with st.spinner(f"Loading custom history for {ccy}…"):
                try:
                    ref_prices = _fetch_ref_prices_curve(
                        host, port, tuple(contracts), ref_str,
                        cfg["symbol"], cfg["exchange"], cfg["currency"],
                        cfg.get("tradingClass", ""), cfg.get("local_sym_prefix", ""),
                    )
                except Exception:
                    ref_prices = {}
            row_chg = {}
            for lbl in contract_labels:
                r = all_rows.get(ccy, {}).get(lbl)
                ref_p = ref_prices.get(lbl)
                row_chg[lbl] = (ref_p - r["price"]) * 100 if (r and r["price"] and ref_p) else None
            chg_vals[ccy] = row_chg

    # Build styled dataframe
    disp, styles = {}, {}
    for ccy, _, _ in _G3_CURVES:
        disp[ccy]   = {lbl: (f"{v:+.1f}" if v is not None else "—")
                       for lbl, v in chg_vals[ccy].items()}
        styles[ccy] = {
            lbl: ("color:#DC2626;font-weight:600" if v and v > 0 else
                  "color:#059669;font-weight:600" if v and v < 0 else "")
            for lbl, v in chg_vals[ccy].items()
        }

    df_chg = pd.DataFrame(disp).T
    df_chg.index.name = "Ccy"
    styled_chg = df_chg.style.apply(
        lambda row: pd.Series(
            [styles.get(row.name, {}).get(col, "") for col in df_chg.columns],
            index=df_chg.columns,
        ),
        axis=1,
    )
    st.dataframe(styled_chg, use_container_width=True)

    # ── Spread table ──────────────────────────────────────────────────────────
    sp_title, sp_btn = st.columns([3, 1])
    sp_title.markdown("##### Spreads (bps)")

    # Initialise session state — load saved prefs first, fall back to defaults
    if "g3_c1" not in st.session_state or len(st.session_state.g3_c1) != 5:
        saved_c1, saved_c2 = _load_spread_prefs(5)
        _sep26 = next((l for l in contract_labels if l.startswith("Sep") and l.endswith("26")),
                      contract_labels[min(2, len(contract_labels) - 1)])
        _sep27 = next((l for l in contract_labels if l.startswith("Sep") and l.endswith("27")),
                      contract_labels[min(6, len(contract_labels) - 1)])
        _dec26 = next((l for l in contract_labels if l.startswith("Dec") and l.endswith("26")),
                      contract_labels[min(3, len(contract_labels) - 1)])
        _dec27 = next((l for l in contract_labels if l.startswith("Dec") and l.endswith("27")),
                      contract_labels[min(7, len(contract_labels) - 1)])
        _c1_def = [contract_labels[min(i, len(contract_labels) - 1)] for i in range(3)] + [_sep26, _dec26]
        _c2_def = [contract_labels[min(i + 1, len(contract_labels) - 1)] for i in range(3)] + [_sep27, _dec27]
        # Pad saved prefs to 5 if they came from old 4-row file
        saved_c1 = (saved_c1 + [None])[:5]
        saved_c2 = (saved_c2 + [None])[:5]
        st.session_state.g3_c1 = [
            s if s in contract_labels else _c1_def[i]
            for i, s in enumerate(saved_c1)
        ]
        st.session_state.g3_c2 = [
            s if s in contract_labels else _c2_def[i]
            for i, s in enumerate(saved_c2)
        ]

    c1_sels = st.session_state.g3_c1
    c2_sels = st.session_state.g3_c2

    # Compute spread values from current selections
    def _sprd(ccy, c1, c2):
        r1 = all_rows.get(ccy, {}).get(c1)
        r2 = all_rows.get(ccy, {}).get(c2)
        if r1 and r2 and r1.get("price") and r2.get("price"):
            return round((r1["price"] - r2["price"]) * 100, 2)
        return None

    sprd_df = pd.DataFrame({
        "Con 1": c1_sels,
        "Con 2": c2_sels,
        "USD (bps)": [_sprd("USD", c1, c2) for c1, c2 in zip(c1_sels, c2_sels)],
        "EUR (bps)": [_sprd("EUR", c1, c2) for c1, c2 in zip(c1_sels, c2_sels)],
        "GBP (bps)": [_sprd("GBP", c1, c2) for c1, c2 in zip(c1_sels, c2_sels)],
    }, index=["Col 1", "Col 2", "Col 3", "Col 4", "Col 5"])

    sprd_col, _ = st.columns([1, 1])
    with sprd_col:
        edited = st.data_editor(
            sprd_df,
            column_config={
                "Con 1": st.column_config.SelectboxColumn(
                    "Con 1", options=contract_labels, required=True),
                "Con 2": st.column_config.SelectboxColumn(
                    "Con 2", options=contract_labels, required=True),
                "USD (bps)": st.column_config.NumberColumn("USD (bps)", format="%+.2f"),
                "EUR (bps)": st.column_config.NumberColumn("EUR (bps)", format="%+.2f"),
                "GBP (bps)": st.column_config.NumberColumn("GBP (bps)", format="%+.2f"),
            },
            disabled=["USD (bps)", "EUR (bps)", "GBP (bps)"],
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="g3_sprd_editor",
        )

    # Update session state if selections changed, then rerun to refresh spreads
    new_c1 = list(edited["Con 1"])
    new_c2 = list(edited["Con 2"])
    if new_c1 != c1_sels or new_c2 != c2_sels:
        st.session_state.g3_c1 = new_c1
        st.session_state.g3_c2 = new_c2
        st.rerun(scope="fragment")

    with sp_btn:
        st.write("")  # nudge button down to align with title
        if st.button("💾 Save", key="g3_sprd_save", help="Save current spread selections"):
            _save_spread_prefs(st.session_state.g3_c1, st.session_state.g3_c2)
            st.toast("Spread preferences saved.", icon="✅")

    # ── Overlaid rate curves ───────────────────────────────────────────────────
    st.markdown("#### Implied Rate Curves — G3")
    fig = go.Figure()
    for ccy, _, color in _G3_CURVES:
        rows_map = all_rows.get(ccy, {})
        labels = [lbl for lbl in contract_labels if lbl in rows_map]
        rates  = [rows_map[lbl]["impl_rate"] for lbl in labels]
        if not labels:
            continue
        fig.add_trace(go.Scatter(
            x=labels, y=rates,
            mode="lines+markers",
            name=ccy,
            line=dict(color=color, width=2),
            marker=dict(size=7, color=color,
                        line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate=f"<b>{ccy}</b>  %{{x}}<br>%{{y:.3f}}%<extra></extra>",
        ))

    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        yaxis=dict(ticksuffix="%", gridcolor="#E8EDF5", zeroline=False),
        xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("USD = SOFR · EUR = Euribor · GBP = SONIA")


def _usage_banner():
    """One-line IBKR usage caption + soft-budget warning for this tab's load.
    STIR ≈ 3 curves × 12 quarterly contracts of quote subscriptions per full load."""
    try:
        tod = usage_today()
        pct = tod["pct"].get("mktdata", 0.0)
        st.caption(f"IBKR today: {tod['mktdata']:,} quote subs · {tod['hist']:,} hist "
                   f"— {pct:.0f}% of soft budget")
        lvl, msg = usage_warning("mktdata", 3 * 12)
        if lvl != "ok":
            st.warning(msg)
    except Exception:
        pass


def render_stir():
    st.markdown("### STIR — Short Term Interest Rate Futures")

    if not _IB_AVAILABLE:
        st.info(
            "**This tab requires a local IBKR connection and is not available on Streamlit Cloud.**  \n\n"
            "To use it, run the app locally with TWS open:  \n"
            "```\npip install ib_insync nest_asyncio\nstreamlit run app.py\n```"
        )
        return

    _usage_banner()

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

    tab_g3, tab_eur, tab_son, tab_sofr = st.tabs(["G3", "Euribor", "SONIA", "SOFR"])

    with tab_g3:
        _render_g3(host, int(port))

    with tab_eur:
        _render_curve(host, int(port), "Euribor", "eur")

    with tab_son:
        _render_curve(host, int(port), "SONIA", "son")

    with tab_sofr:
        _render_curve(host, int(port), "SOFR", "sofr")

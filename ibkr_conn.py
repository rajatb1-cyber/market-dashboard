"""
Single owner of Interactive Brokers TWS connectivity for the whole dashboard.

WHY THIS MODULE EXISTS
----------------------
Historically several tabs each opened their own throwaway TWS client with a
random clientId. That churn is expensive: IBKR market-data *lines* are an
account-wide pool (default 100) shared between the TWS GUI and every API client,
so connection churn plus unbounded concurrent ``reqMktData`` subscriptions
exhaust the pool and make the GUI pop "subscribe?" prompts on unrelated
products. IBKR also hard-limits historical data pacing: at most **60
reqHistoricalData per rolling 10 minutes** account-wide, **no identical request
within 15s**, and **BID_ASK requests count double**; violations throttle or
disconnect the API session. There are only ~100 shared market-data lines.

To respect all of that this module owns ONE persistent, cached connection
(``_IBLoop`` on its own asyncio thread, fixed clientId 15, ``@st.cache_resource``)
and exposes a small, safe public surface:

    get_conn()   -> (ibl_or_None, err_str)   self-healing connection accessor
    quotes(...)  -> list[Ticker|None]        chunked snapshot quotes, lines always freed
    hist_bars(...) -> list                   reqHistoricalData with a global pacing ledger
    PacingBlocked                            raised by hist_bars when a request would breach limits

Callers of ``hist_bars`` are expected to catch ``PacingBlocked`` and fall back
to their own local DB caches rather than hammer TWS.

Importing this module must NOT open a connection — connections happen only inside
``_get_ibl()`` / ``get_conn()`` at runtime.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import pathlib
import sqlite3
import threading
import time
from typing import Any, Callable, Optional

import streamlit as st

try:
    from ib_insync import IB, Future, FuturesOption, Option as IBOption, util
    _IB_AVAILABLE = True
except ImportError:
    IB = Future = FuturesOption = IBOption = util = None  # type: ignore
    _IB_AVAILABLE = False

_HOST      = "127.0.0.1"
_PORT      = 7496
_CLIENT_ID = 15


# ══ Request-telemetry ledger + soft-budget warnings ═══════════════════════════
#
# WHY: Rajat hits an *unpublished* daily IBKR heuristic — after heavy automated
# DELAYED-data use, TWS stops serving delayed data for unsubscribed products
# ("subscribe?" prompts) until the next day. The published limits (100 md-lines,
# 60 hist/10min) are already respected elsewhere; this system instead MEASURES
# usage, lets the user MARK the breakage moment, and WARNS near a configurable
# soft budget. It NEVER hard-blocks — warnings only.
#
# Ledger: SQLite ``ibkr_usage.db`` in the project dir. Short-lived connections
# per op (this codebase's pattern), guarded by a module-level lock. Every public
# function swallows DB errors (telemetry must never break a data path).

_USAGE_DB_PATH = pathlib.Path(__file__).parent / "ibkr_usage.db"
_BUDGETS_PATH  = pathlib.Path(__file__).parent / "ibkr_budgets.json"
_USAGE_LOCK    = threading.Lock()

_DEFAULT_BUDGETS = {"mktdata": 300, "hist": 250}
_budgets_cache   = {"ts": 0.0, "val": None}   # re-read the JSON at most once/min


def _today_str() -> str:
    """Local-date ISO string — daily grouping key (DST-proof, no UTC drift)."""
    return datetime.date.today().isoformat()


def _usage_open() -> sqlite3.Connection:
    """Short-lived connection with the table ensured. Call inside _USAGE_LOCK."""
    conn = sqlite3.connect(str(_USAGE_DB_PATH))
    conn.execute("CREATE TABLE IF NOT EXISTS usage "
                 "(ts REAL, day TEXT, kind TEXT, n INTEGER, tag TEXT, note TEXT DEFAULT '')")
    return conn


def _load_budgets() -> dict:
    """Soft budgets from ibkr_budgets.json (written with defaults if missing).
    Cached for 60s so edits apply without a Streamlit restart."""
    now = time.time()
    if _budgets_cache["val"] is not None and (now - _budgets_cache["ts"]) < 60:
        return _budgets_cache["val"]
    b = dict(_DEFAULT_BUDGETS)
    try:
        if _BUDGETS_PATH.exists():
            with open(_BUDGETS_PATH) as f:
                data = json.load(f)
            for k in ("mktdata", "hist"):
                if k in data:
                    b[k] = int(data[k])
        else:
            with open(_BUDGETS_PATH, "w") as f:
                json.dump(_DEFAULT_BUDGETS, f, indent=2)
    except Exception:
        pass
    _budgets_cache["val"] = b
    _budgets_cache["ts"]  = now
    return b


def record_usage(kind: str, n: int, tag: str = "", note: str = "") -> None:
    """Append one usage row. Never raises (telemetry must not break a data path).
    kind ∈ 'mktdata'|'hist'|'secdef'|'breakage'; n is the count (hist BID_ASK = 2)."""
    try:
        n = int(n)
    except Exception:
        return
    if n <= 0 and kind != "breakage":
        return
    try:
        with _USAGE_LOCK:
            conn = _usage_open()
            try:
                conn.execute("INSERT INTO usage (ts, day, kind, n, tag, note) "
                             "VALUES (?,?,?,?,?,?)",
                             (time.time(), _today_str(), kind, n, tag, note))
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass


def usage_today() -> dict:
    """{'mktdata','hist','secdef': int, 'budgets': {...}, 'pct': {...}} for today."""
    counts = {"mktdata": 0, "hist": 0, "secdef": 0}
    try:
        with _USAGE_LOCK:
            conn = _usage_open()
            try:
                cur = conn.execute("SELECT kind, SUM(n) FROM usage WHERE day=? GROUP BY kind",
                                   (_today_str(),))
                for kind, tot in cur.fetchall():
                    if kind in counts:
                        counts[kind] = int(tot or 0)
            finally:
                conn.close()
    except Exception:
        pass
    budgets = _load_budgets()
    pct = {}
    for k in ("mktdata", "hist"):
        bud = budgets.get(k) or 0
        pct[k] = (counts[k] / bud * 100.0) if bud else 0.0
    return {"mktdata": counts["mktdata"], "hist": counts["hist"], "secdef": counts["secdef"],
            "budgets": {"mktdata": budgets.get("mktdata"), "hist": budgets.get("hist")},
            "pct": pct}


def usage_today_by_tag() -> dict:
    """{tag: {'mktdata','hist','secdef': int}} for today (breakage rows excluded)."""
    out: dict = {}
    try:
        with _USAGE_LOCK:
            conn = _usage_open()
            try:
                cur = conn.execute(
                    "SELECT tag, kind, SUM(n) FROM usage "
                    "WHERE day=? AND kind!='breakage' GROUP BY tag, kind", (_today_str(),))
                for tag, kind, tot in cur.fetchall():
                    t = out.setdefault(tag or "(untagged)",
                                       {"mktdata": 0, "hist": 0, "secdef": 0})
                    if kind in t:
                        t[kind] = int(tot or 0)
            finally:
                conn.close()
    except Exception:
        pass
    return out


def usage_warning(kind: str, planned_n: int) -> tuple[str, str]:
    """('ok'|'warn'|'over', message). 'warn' when current+planned ≥ 80% of the soft
    budget, 'over' when ≥ 100%. NEVER a hard block — callers only surface the text."""
    budgets = _load_budgets()
    bud = budgets.get(kind) or 0
    try:
        planned = int(planned_n)
    except Exception:
        planned = 0
    if bud <= 0:
        return ("ok", "")
    cur   = usage_today().get(kind, 0)
    would = cur + planned
    frac  = would / bud
    label = ("quote subscriptions" if kind == "mktdata"
             else "historical requests" if kind == "hist" else kind)
    msg = (f"IBKR usage: this will add ~{planned} {label} "
           f"(would be {frac*100:.0f}% of today's soft budget)")
    if frac >= 1.0:
        return ("over", msg)
    if frac >= 0.8:
        return ("warn", msg)
    return ("ok", msg)


def mark_breakage(note: str = "") -> None:
    """Stamp a user-observed breakage incident (kind='breakage', n=1) with the
    current usage counts folded into the note so the row is self-describing."""
    tod = usage_today()
    stamp = (f"mktdata={tod['mktdata']} hist={tod['hist']} secdef={tod['secdef']} "
             f"at {datetime.datetime.now().strftime('%H:%M')}")
    full = f"{note} ({stamp})" if note else stamp
    record_usage("breakage", 1, tag="breakage", note=full)


def usage_history(days: int = 14) -> list:
    """Per-day totals for the last ``days`` (newest first), each with a 'breakage'
    bool for whether an incident was marked that day."""
    per_day: dict = {}
    try:
        with _USAGE_LOCK:
            conn = _usage_open()
            try:
                cur = conn.execute("SELECT day, kind, SUM(n) FROM usage GROUP BY day, kind")
                for day, kind, tot in cur.fetchall():
                    d = per_day.setdefault(
                        day, {"mktdata": 0, "hist": 0, "secdef": 0, "breakage": False})
                    if kind == "breakage":
                        d["breakage"] = True
                    elif kind in d:
                        d[kind] = int(tot or 0)
            finally:
                conn.close()
    except Exception:
        pass
    today = datetime.date.today()
    out = []
    for i in range(days):
        d = (today - datetime.timedelta(days=i)).isoformat()
        rec = per_day.get(d, {"mktdata": 0, "hist": 0, "secdef": 0, "breakage": False})
        out.append({"day": d, "mktdata": rec["mktdata"], "hist": rec["hist"],
                    "secdef": rec["secdef"], "breakage": rec["breakage"]})
    return out


# ── 60-second quote memo (usage reduction for quotes() ONLY) ──────────────────
_QUOTE_MEMO: dict = {}          # {key: (ts, ticker)}
_QUOTE_MEMO_LOCK  = threading.Lock()
_QUOTE_MEMO_TTL_S = 60


# ── Dedicated asyncio event loop thread ───────────────────────────────────────

class _IBLoop:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._ib: Optional[IB] = None
        self._conn_done  = threading.Event()
        self._conn_error: Optional[str] = None

        t = threading.Thread(target=self._thread_main, daemon=True, name="ib-loop")
        t.start()

    def _thread_main(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._do_connect())
        self._loop.run_forever()

    async def _do_connect(self):
        try:
            self._ib = IB()
            await self._ib.connectAsync(
                _HOST, _PORT, clientId=_CLIENT_ID, readonly=True, timeout=15
            )
            self._ib.reqMarketDataType(2)  # frozen: fall back to settlement when live unavailable
        except Exception as exc:
            self._conn_error = f"{type(exc).__name__}: {exc}" or "connection failed"
        finally:
            self._conn_done.set()

    def is_connected(self) -> bool:
        try:
            return bool(self._ib and self._ib.isConnected())
        except Exception:
            return False

    def reconnect(self) -> bool:
        """Disconnect and reconnect on the event loop thread — always force-cycles
        the connection so zombie subscriptions from timed-out coros are cleared."""
        async def _do():
            try:
                if self._ib:
                    try:
                        self._ib.disconnect()
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)  # let TWS release clientId
                self._ib = IB()
                await self._ib.connectAsync(
                    _HOST, _PORT, clientId=_CLIENT_ID, readonly=True, timeout=15
                )
                self._ib.reqMarketDataType(2)
                self._conn_error = None
                return True
            except Exception as exc:
                self._conn_error = str(exc)
                return False
        future = asyncio.run_coroutine_threadsafe(_do(), self._loop)
        try:
            return bool(future.result(timeout=25))
        except Exception as exc:
            self._conn_error = str(exc)
            return False

    def submit(self, coro_fn: Callable, timeout_s: int = 60) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro_fn(self._ib), self._loop)
        return future.result(timeout=timeout_s)


@st.cache_resource
def _get_ibl() -> Optional[_IBLoop]:
    if not _IB_AVAILABLE:
        return None
    ibl = _IBLoop()
    ibl._conn_done.wait(timeout=20)
    return ibl


def _ping_tws(ibl: _IBLoop) -> bool:
    """Verify the connection is actually live (not just cached as connected)."""
    async def _coro(ib):
        return await ib.reqCurrentTimeAsync()
    try:
        ibl.submit(_coro, timeout_s=5)
        return True
    except Exception:
        return False


# ── Public connection accessor ────────────────────────────────────────────────

def get_conn():
    """Return (ibl, err). ibl is None if TWS/ib_insync unavailable.

    Self-heals: if the cached connection is dead (TWS restarted, network blip, or
    another client grabbed our clientId — they share clientId 15) try one
    reconnect before giving up."""
    if not _IB_AVAILABLE:
        return None, "ib_insync not installed"
    ibl = _get_ibl()
    if ibl is None:
        return None, "could not create IB loop"
    if not ibl.is_connected():
        try:
            if ibl.reconnect() and ibl.is_connected():
                return ibl, ""
        except Exception:
            pass
        return None, f"TWS not connected ({getattr(ibl, '_conn_error', '') or 'is TWS running on :7496?'})"
    return ibl, ""


# ── Snapshot quotes ───────────────────────────────────────────────────────────

def _quote_memo_key(c, mdtype):
    """conId if set, else (symbol, exchange, ltdocm, secType, mdtype)."""
    cid = getattr(c, "conId", 0)
    if cid:
        return ("id", cid, mdtype)
    return ("sym", getattr(c, "symbol", ""), getattr(c, "exchange", ""),
            getattr(c, "lastTradeDateOrContractMonth", ""), getattr(c, "secType", ""), mdtype)


def quotes(contracts, mdtype=3, settle_s=3.0, chunk=20, timeout_s=60, ibl=None, tag=""):
    """Snapshot-quote a list of ib_insync Contracts. Returns a list of Ticker
    objects parallel to the input (None where a contract failed).

    Contracts may be unqualified — do NOT qualifyContracts/reqContractDetails
    (they hang in this TWS setup); reqMktData resolves the contract itself.

    Contracts are processed in chunks of at most ``chunk`` (default 20) to bound
    concurrent market-data-line usage; every subscription is cancelled in a
    ``finally`` so lines are released even on exception. ``mdtype`` is the
    per-call market-data type (3 = delayed, needs no subscription); the
    connection-wide default (2 = frozen) is restored afterwards.

    A 60-second memo (module-level) short-circuits contracts quoted <60s ago:
    the prior (cancelled = static) ticker is reused, NOT re-subscribed and NOT
    counted in telemetry. ``tag`` labels the actual subscriptions in the usage
    ledger; only newly-subscribed contracts are recorded.
    """
    contracts = list(contracts)
    n = len(contracts)
    if n == 0:
        return []

    if ibl is None:
        ibl, err = get_conn()
        if ibl is None:
            return [None] * n

    async def _coro(ib):
        out: list = [None] * n
        # Reuse fresh memo entries; only the misses get subscribed.
        now0 = time.time()
        to_sub: list = []
        with _QUOTE_MEMO_LOCK:
            for i, c in enumerate(contracts):
                ent = _QUOTE_MEMO.get(_quote_memo_key(c, mdtype))
                if ent and (now0 - ent[0]) < _QUOTE_MEMO_TTL_S:
                    out[i] = ent[1]                    # reuse the static snapshot
                else:
                    to_sub.append((i, c))
        n_subscribed = 0
        for start in range(0, len(to_sub), chunk):
            batch = to_sub[start:start + chunk]
            subscribed: list = []
            try:
                try:
                    ib.reqMarketDataType(mdtype)
                except Exception:
                    pass
                for idx, c in batch:
                    try:
                        t = ib.reqMktData(c, "", snapshot=False)
                        subscribed.append((idx, c, t))
                    except Exception:
                        pass
                await asyncio.sleep(settle_s)
                memo_ts = time.time()
                with _QUOTE_MEMO_LOCK:
                    for idx, c, t in subscribed:
                        out[idx] = t
                        _QUOTE_MEMO[_quote_memo_key(c, mdtype)] = (memo_ts, t)
                        n_subscribed += 1
            finally:
                for idx, c, t in subscribed:
                    try:
                        ib.cancelMktData(c)
                    except Exception:
                        pass
        try:
            ib.reqMarketDataType(2)  # restore connection-wide default
        except Exception:
            pass
        if n_subscribed:
            record_usage("mktdata", n_subscribed, tag)
        return out

    try:
        return ibl.submit(_coro, timeout_s=timeout_s)
    except Exception:
        return [None] * n


# ── Historical bars with a global pacing ledger ───────────────────────────────

class PacingBlocked(RuntimeError):
    """Raised by hist_bars when a request would breach IBKR pacing limits.
    Callers should catch this and fall back to their local DB cache."""


_PACING_PATH = pathlib.Path(__file__).parent / "ibkr_pacing.json"
_PACING_LOCK = threading.Lock()

_PACING_WINDOW_S   = 600   # IBKR pacing window: 60 requests per rolling 10 minutes
_PACING_MAX_REQS   = 50    # safety margin under IBKR's account-wide 60/10min
_PACING_IDENTICAL_S = 15   # no identical request within 15s


def _pacing_load() -> list:
    """Return the pruned list of [ts, key] entries; corrupt/missing → fresh."""
    try:
        with open(_PACING_PATH) as f:
            data = json.load(f)
        reqs = data.get("reqs", [])
    except Exception:
        reqs = []
    cutoff = time.time() - _PACING_WINDOW_S
    return [r for r in reqs if isinstance(r, (list, tuple)) and len(r) == 2 and r[0] >= cutoff]


def _pacing_save(reqs: list) -> None:
    try:
        with open(_PACING_PATH, "w") as f:
            json.dump({"reqs": reqs}, f)
    except Exception:
        pass


def hist_bars(contract, endDateTime="", durationStr="1 Y", barSizeSetting="1 day",
              whatToShow="TRADES", useRTH=False, timeout_s=20, ibl=None, tag=""):
    """reqHistoricalDataAsync wrapper guarded by a global (cross-process) pacing
    ledger. Returns the bars list ([] on empty). Raises ``PacingBlocked`` when a
    request would breach IBKR limits — callers should fall back to local caches.
    Genuine IBKR errors propagate as their own exceptions (distinct from
    PacingBlocked)."""
    conId    = getattr(contract, "conId", 0)
    symbol   = getattr(contract, "symbol", "")
    exchange = getattr(contract, "exchange", "")
    ltdocm   = getattr(contract, "lastTradeDateOrContractMonth", "")
    key = f"{conId or symbol}|{exchange}|{ltdocm}|{durationStr}|{barSizeSetting}|{whatToShow}"
    cost = 2 if whatToShow == "BID_ASK" else 1  # BID_ASK requests count double

    with _PACING_LOCK:
        reqs = _pacing_load()
        now = time.time()

        # (a) identical request within 15s
        for ts, k in reqs:
            if k == key and (now - ts) < _PACING_IDENTICAL_S:
                raise PacingBlocked("identical request within 15s")

        # (b) rolling-10-min request budget (BID_ASK counts double)
        used = sum(2 if k.endswith("|BID_ASK") else 1 for _, k in reqs)
        if used + cost > _PACING_MAX_REQS:
            oldest = min((ts for ts, _ in reqs), default=now)
            retry_n = int(max(0, _PACING_WINDOW_S - (now - oldest))) + 1
            raise PacingBlocked(f"historical-data budget exhausted — retry in {retry_n}s")

        # Record before sending so concurrent callers see it.
        reqs.append([now, key])
        _pacing_save(reqs)

    # Telemetry mirrors the pacing ledger: only requests actually sent are counted
    # (PacingBlocked ones raised above and never reach here). BID_ASK counts double.
    record_usage("hist", cost, tag)

    if ibl is None:
        ibl, err = get_conn()
        if ibl is None:
            raise RuntimeError(err or "TWS not connected")

    async def _coro(ib):
        bars = await ib.reqHistoricalDataAsync(
            contract, endDateTime=endDateTime, durationStr=durationStr,
            barSizeSetting=barSizeSetting, whatToShow=whatToShow, useRTH=useRTH,
        )
        return list(bars) if bars else []

    return ibl.submit(_coro, timeout_s=timeout_s)

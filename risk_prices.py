"""
Live futures prices for the Risk/VaR tab.

Design (per Rajat, 2026-07): positions come from IBKR **Flex** only when he clicks
*Update from IBKR*. The daily 1d/3d/5d PnL instead uses **live prices from IBKR
TWS** (market-data API — NOT the Flex statement, so it is safe to call many times
a day) against **reference daily closes cached locally** in ``risk_prices.json``.

    PnL_Nd = Quantity × Multiplier × (live − close_N_business_days_ago) × FXRateToBase

The multiplier is the contract's point value, so this Δprice formula is unit-correct
for STIR (price ~96, mult 2500 → $25/bp) and equity futures alike.

TWS access reuses options.py's shared, cached connection (``_get_ibl``), so we do
not open a second client. Everything degrades gracefully: if TWS is down or a
contract has no data, the caller falls back to the settled Flex PnL.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from datetime import date

import pandas as pd

# Telemetry only — these fetchers keep their own (painfully-debugged) TWS logic;
# we merely append usage rows. Guarded so a missing ibkr_conn never breaks pricing.
try:
    from ibkr_conn import record_usage
except Exception:
    def record_usage(*_a, **_k):
        pass

_PRICES_PATH = os.path.join(os.path.dirname(__file__), "risk_prices.json")


# ── local cache ──────────────────────────────────────────────────────────────
def load_price_cache() -> dict:
    """{symbol: {"live": float, "live_src": str, "live_ts": epoch,
                 "closes": {iso_date: close}, "closes_date": iso_date}}"""
    try:
        with open(_PRICES_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_price_cache(d: dict) -> None:
    try:
        with open(_PRICES_PATH, "w") as f:
            json.dump(d, f, indent=2)
    except Exception:
        pass


# ── conId → exchange resolution (once ever, persisted) ───────────────────────
# Flex book rows carry an EMPTY Exchange field, and reqMktData with conId + no
# exchange gets answered from TWS's frozen snapshot cache instead of streaming —
# marks showed 🌙 closed while markets were open (2026-07-27). The old qualify step
# masked this by filling the exchange in; qualify was removed (it hangs). So:
# resolve each conId's exchange ONCE via a time-bounded contract-details lookup
# (same pattern stir.py uses) and persist it — the hang-prone call runs at most
# once per contract, off the hot path forever after.
_CONID_EXCH_PATH = os.path.join(os.path.dirname(__file__), "conid_exchange.json")


def _exchange_for(ibl, conid) -> str:
    try:
        with open(_CONID_EXCH_PATH) as f:
            m = json.load(f)
    except Exception:
        m = {}
    key = str(int(conid))
    if m.get(key):
        return m[key]

    async def _coro(ib):
        from ib_insync import Contract
        cds = await asyncio.wait_for(
            ib.reqContractDetailsAsync(Contract(conId=int(conid))), timeout=4.0)
        return cds[0].contract.exchange if cds else ""
    try:
        exch = ibl.submit(_coro, timeout_s=6) or ""
    except Exception:
        exch = ""
    if exch:
        m[key] = exch
        try:
            with open(_CONID_EXCH_PATH, "w") as f:
                json.dump(m, f, indent=2)
        except Exception:
            pass
        try:
            record_usage("secdef", 1, tag="risk-refpnl", note=f"conid {key} -> {exch}")
        except Exception:
            pass
    return exch


# ── TWS fetch (reuses ibkr_conn.py's shared connection) ──────────────────────
def _get_conn():
    """Return (ibl, err). ibl is None if TWS/ib_insync unavailable.
    Thin delegation to ibkr_conn.get_conn() — the single owner of the shared TWS
    connection (does its own self-healing reconnect)."""
    try:
        from ibkr_conn import get_conn
    except Exception as e:
        return None, f"ibkr_conn TWS module unavailable ({e})"
    return get_conn()


def _fetch_quotes(ibl, futures, want_daily):
    """futures: [(symbol, conid, exchange)]; want_daily: set of symbols needing a
    daily-bar refresh. Returns {symbol: (live, src, closes_dict)} where src ∈
    {'live','delayed','closed','prev-close',None}.

    Routes through ``ibkr_conn.quotes()`` / ``hist_bars()`` — the shared, memoized,
    market-data-line-safe API — instead of raw ``reqMktData``. This is what fixes
    "Ref PnL works once then not again": raw re-subscription of the same contracts on
    every click left lingering subscriptions / churned the shared line pool, so the 2nd
    call failed. quotes() reuses a <60s snapshot (correct for a 15-min-delayed feed),
    always cancels + frees lines, and needs no qualify (which hangs in this TWS setup).
    """
    try:
        from ibkr_conn import quotes as _conn_quotes, hist_bars as _conn_hist, PacingBlocked
    except Exception:
        return {}
    try:
        from ib_insync import Future
    except Exception:
        return {}

    def sf(v):
        try:
            f = float(v)
            return f if (f > 0 and not math.isnan(f)) else None
        except Exception:
            return None

    # Build conId contracts WITH a real exchange — the Flex book's Exchange field is
    # empty, and conId-only reqMktData gets frozen-snapshot answers even in open
    # markets. _exchange_for resolves each conId once (persisted) — no per-click qualify.
    conts = {}
    for sym, conid, exch in futures:
        try:
            conts[sym] = Future(conId=int(conid),
                                exchange=(exch or _exchange_for(ibl, conid) or ""))
        except Exception:
            conts[sym] = None

    # ── Marks: one memoized, line-safe delayed snapshot for all contracts ────────
    valid = [(s, conts[s]) for s, _, _ in futures if conts.get(s) is not None]
    # mdtype=1 (real-time — book fully subscribed); see live_positions for why not delayed(3).
    tks = _conn_quotes([c for _, c in valid], mdtype=1, tag="risk-refpnl", ibl=ibl) if valid else []
    tkr_by_sym = {s: t for (s, _), t in zip(valid, tks)}

    _MDT2SRC = {1: "live", 3: "delayed"}
    out = {}
    for sym, conid, exch in futures:
        c = conts.get(sym)
        if c is None:
            out[sym] = (None, None, {})
            continue
        tkr = tkr_by_sym.get(sym)
        mk = last = cl = mdt = None
        if tkr is not None:
            bid, ask = sf(tkr.bid), sf(tkr.ask)
            last, cl = sf(tkr.last), sf(tkr.close)
            mk = (bid + ask) / 2 if (bid and ask) else last     # mid, else last
            mdt = getattr(tkr, "marketDataType", None)           # 1 live/2 frozen/3 delayed/4 dlyd-frozen

        if mdt in (2, 4):
            # Market CLOSED (frozen). Use last trade (= session close), not the frozen
            # bid/ask mid; `cl` (tkr.close = PRIOR settlement) is the fallback.
            live = last if last is not None else cl
            src = "closed" if live is not None else None
        elif mk is not None:
            live, src = mk, _MDT2SRC.get(mdt, "delayed")
        elif last is not None:
            live, src = last, _MDT2SRC.get(mdt, "delayed")
        elif cl:
            live, src = cl, "prev-close"
        else:
            live, src = None, None

        # ── Daily bars (pacing-guarded; skipped once cached for the day) ─────────
        closes = {}
        if sym in want_daily:
            try:
                bars = _conn_hist(c, durationStr="12 D", barSizeSetting="1 day",
                                  whatToShow="TRADES", useRTH=False, tag="risk-refpnl", ibl=ibl)
                for b in bars:
                    d = b.date.isoformat() if hasattr(b.date, "isoformat") else str(b.date)
                    closes[d] = float(b.close)
            except PacingBlocked:
                closes = {}      # keep cached closes (refresh_prices reconciles against them)
            except Exception:
                closes = {}

        # Weak/failed tick but we have fresh daily bars → use latest settlement as the
        # mark (avoids the prev-close trap: tkr.close is the PRIOR settlement → false $0).
        if src in (None, "prev-close") and closes:
            live, src = closes[max(closes)], "closed"
        out[sym] = (live, src, closes)
    return out


def refresh_prices(book: pd.DataFrame, force_daily: bool = False):
    """Fetch live prices (always) + daily closes (once/day unless force_daily) for
    the futures in *book*; update the local cache. Returns (cache, source_note)."""
    cache = load_price_cache()
    if book is None or book.empty:
        return cache, "no futures"
    ibl, err = _get_conn()
    if ibl is None:
        return cache, f"TWS unavailable — {err}"

    today = date.today().isoformat()
    futures, want_daily = [], set()
    for _, r in book.iterrows():
        if r.get("is_option"):          # options handled separately (analysis deferred)
            continue
        conid = r.get("Conid")
        if pd.isna(conid) if conid is not None else True:
            continue
        sym = r["Symbol"]
        futures.append((sym, conid, r.get("Exchange", "")))
        ent = cache.get(sym, {})
        if force_daily or ent.get("closes_date") != today or not ent.get("closes"):
            want_daily.add(sym)
    if not futures:
        return cache, "no Conid on positions — re-pull from IBKR"

    import time
    res = _fetch_quotes(ibl, futures, want_daily)
    now = time.time()
    n_priced = n_pclose = 0
    for sym, (live, src, closes) in res.items():
        ent = cache.get(sym, {})
        if closes:
            ent["closes"], ent["closes_date"] = closes, today
        # Reconcile a weak tick (no intraday mark, or "prev-close" = the PRIOR settlement)
        # against the daily bars we HOLD (incl. same-day cached ones _fetch_quotes didn't
        # refetch): use the latest settlement as the mark → a real closed-market mark, not
        # a prior close that shows a false $0 1d PnL.
        _cl = ent.get("closes") or {}
        if src in (None, "prev-close") and _cl:
            _ld = max(_cl)
            live, src = _cl[_ld], "closed"
        if live is not None:
            ent["live"], ent["live_src"], ent["live_ts"] = live, src, now
            if src in ("live", "delayed", "closed"):
                n_priced += 1
            elif src == "prev-close":
                n_pclose += 1
        cache[sym] = ent
    _save_price_cache(cache)
    n = len(futures)
    parts = []
    if n_priced:
        parts.append(f"{n_priced} priced")
    if n_pclose:
        parts.append(f"{n_pclose} prev-close only")
    miss = n - n_priced - n_pclose
    if miss:
        parts.append(f"{miss} unavailable")
    return cache, (", ".join(parts) + f" — {n} positions") if parts else f"0/{n} — no marks from TWS"


def tws_diagnostics(book) -> str:
    """Probe TWS and return a COPY-PASTEABLE diagnostic report so live-price failures
    can be diagnosed precisely. Reports: local price-cache state, connection details,
    and per-symbol contract-qualify timing + the market-data type TWS actually serves
    (1 live / 2 frozen / 3 delayed / 4 delayed-frozen) + bid/ask/last/close, plus any
    IBKR error/notice codes emitted during the probe (data-farm status & market-data
    permission messages surface here — e.g. 2103/2105 farm broken, 354 no permission)."""
    import time, datetime as _dt
    L = [f"=== TWS DIAGNOSTICS  {_dt.datetime.now().isoformat(timespec='seconds')} ==="]

    # 1) Local price-cache summary (no TWS needed) ─────────────────────────────
    cache = load_price_cache()
    L.append(f"price cache: {len(cache)} symbols")
    for sym, e in sorted(cache.items()):
        lts = e.get("live_ts")
        when = _dt.datetime.fromtimestamp(lts).strftime("%Y-%m-%d %H:%M") if lts else "—"
        L.append(f"  {sym:10} src={str(e.get('live_src','—')):11} mark={e.get('live')}  @{when}  "
                 f"closes={len(e.get('closes',{}))} (asof {e.get('closes_date','—')})")

    # 2) Positions we would price ──────────────────────────────────────────────
    futs = []
    if book is not None and not book.empty:
        for _, r in book.iterrows():
            if r.get("is_option"):
                continue
            conid = r.get("Conid")
            if conid is None or (isinstance(conid, float) and conid != conid):
                continue
            futs.append((r["Symbol"], conid, r.get("Exchange", "")))
    L.append(f"futures in book: {[s for s, _, _ in futs]}")

    ibl, err = _get_conn()
    if ibl is None:
        L.append(f"CONNECTION: FAILED — {err}")
        L.append("→ TWS/IB Gateway not reachable. Check it is running, logged in, and API enabled "
                 "(Config → API → Enable ActiveX and Socket Clients) on the expected port.")
        return "\n".join(L)

    async def _coro(ib):
        out = []
        cl = ib.client
        try:
            out.append(f"CONNECTION: OK  host={cl.host} port={cl.port} clientId={cl.clientId} "
                       f"serverVersion={cl.serverVersion()}")
        except Exception as e:
            out.append(f"CONNECTION: OK (client details unavailable: {e})")
        try:
            out.append(f"managedAccounts={ib.managedAccounts()}")
        except Exception as e:
            out.append(f"managedAccounts: err {e}")

        # Capture IBKR error/notice codes during the probe — farm status lives here.
        errs = []
        def _on_err(reqId=None, code=None, msg=None, contract=None, *a):
            cid = getattr(contract, "conId", None) if contract is not None else None
            errs.append(f"  [{code}] {msg}" + (f"  (conId={cid})" if cid else ""))
        try:
            ib.errorEvent += _on_err
        except Exception:
            pass

        from ib_insync import Future
        _MDT = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed-frozen"}
        for sym, conid, exch in futs:
            t0 = time.time()
            c = Future(conId=int(conid), exchange=(exch or ""))
            try:
                await asyncio.wait_for(ib.qualifyContractsAsync(c), timeout=4.0)
                q = (f"qualify OK {time.time()-t0:.1f}s  localSym={getattr(c,'localSymbol','')} "
                     f"mult={getattr(c,'multiplier','')}")
            except Exception as e:
                q = f"qualify FAIL/timeout {time.time()-t0:.1f}s ({type(e).__name__})"
            try:
                ib.reqMarketDataType(3)     # ask for delayed (the app's default)
                tkr = ib.reqMktData(c, "", snapshot=False)
                await asyncio.sleep(3.0)
                mdt = _MDT.get(getattr(tkr, "marketDataType", None),
                               getattr(tkr, "marketDataType", None))
                out.append(f"{sym}  conid={conid} exch={exch or '—'}  {q}")
                out.append(f"    served={mdt}  bid={tkr.bid} ask={tkr.ask} "
                           f"last={tkr.last} close={tkr.close}")
                try:
                    ib.cancelMktData(c)
                except Exception:
                    pass
            except Exception as e:
                out.append(f"{sym}  {q}")
                out.append(f"    reqMktData ERROR: {type(e).__name__}: {e}")
        try:
            ib.reqMarketDataType(1)
        except Exception:
            pass
        try:
            ib.errorEvent -= _on_err
        except Exception:
            pass
        out.append("IBKR messages during probe:" if errs else "IBKR messages during probe: (none)")
        out += errs
        return out

    try:
        L += ibl.submit(_coro, timeout_s=120)
    except Exception as e:
        L.append(f"PROBE FAILED: {type(e).__name__}: {e}")
    L.append("=== END ===  (paste this whole block to diagnose)")
    return "\n".join(L)


# ── Live positions from TWS (incl. today's trades) ───────────────────────────
def live_positions():
    """Build the speculative book from LIVE TWS positions (portfolio) — includes
    today's fills, unlike the Flex EOD snapshot. Returns (book_df, note); empty df
    + note on failure so the caller can fall back to Flex. Same schema as
    risk.build_speculative_book (Symbol, AssetClass, Underlying, Strike, Expiry,
    Multiplier, MarkPrice, Quantity, position_value_base, upnl_base, FXRateToBase,
    side, is_option, gross_base, Conid, Exchange, Currency)."""
    ibl, err = _get_conn()
    if ibl is None:
        return pd.DataFrame(), f"TWS unavailable — {err}"

    async def _coro(ib):
        positions = await ib.reqPositionsAsync()          # fast; no account-update subscription
        picks = [p for p in positions
                 if (getattr(p.contract, "secType", "") or "").upper() in ("FUT", "FOP", "OPT")
                 and float(p.position or 0) != 0]
        try:
            # REAL-TIME (type 1), not delayed(3): the book is fully covered by live subs
            # (CME + ICE LIFFE L2), and API delayed-mode requests are the last remaining
            # suspect for flipping TWS's delayed-data state → "subscribe" prompts on
            # unrelated products (2 incidents 2026-07-28, both ~15-25min after an
            # Update—LIVE; volume + options subs already ruled out). If prompts persist
            # even with type 1, the app is exonerated entirely — it's TWS farm state.
            ib.reqMarketDataType(1)
        except Exception:
            pass
        # FUTURES ONLY — no live-quote subscriptions for options. Three reasons
        # (2026-07-28): the Options Prem box always uses the EOD/Flex-settled mark
        # anyway (2026-07-23 fix), the thin FOP quotes never refresh in practice
        # (Rajat), and delayed-type requests on FOPs — which mostly have NO delayed
        # feed — are the prime suspect for tripping TWS's "subscribe" prompts on
        # unrelated products (prompts appeared ~30min after the day's only
        # Update—LIVE burst, at just 21 total subs — daily volume ruled out).
        tickers = {}
        for p in picks:                                    # request marks (delayed)
            if (getattr(p.contract, "secType", "") or "").upper() != "FUT":
                continue
            try:
                tickers[p.contract.conId] = ib.reqMktData(p.contract, "", snapshot=False)
            except Exception:
                tickers[p.contract.conId] = None
        record_usage("mktdata", sum(1 for _t in tickers.values() if _t is not None),
                     tag="risk-live")

        # Poll for up to ~6s (checking every 0.5s) instead of a fixed 2.5s sleep — thin,
        # near-expiry contracts (deep-OTM weekly-style options especially) can have one
        # side of the quote (e.g. bid) arrive well before the other (ask). A fixed short
        # sleep risked reading a half-populated ticker and falling back to a stale `last`
        # print even when a real live market (e.g. bid=0/ask=0.015625) was about to land.
        def _has_quote(tkr):
            try:
                b, a = float(tkr.bid), float(tkr.ask)
                return b == b and a == a and b >= 0 and a >= 0   # not-NaN, non-negative
            except Exception:
                return False
        for _ in range(12):
            await asyncio.sleep(0.5)
            if all(_has_quote(t) for t in tickers.values() if t is not None):
                break
        try:
            ib.reqMarketDataType(1)   # restore real-time default for other tabs on the conn
        except Exception:
            pass

        def sf(v):
            """Strictly-positive float, else None. For last/close: 0 there means "no data",
            not a real trade — a real trade/settlement price is never exactly 0."""
            try:
                f = float(v)
                return f if (f == f and f > 0) else None
            except Exception:
                return None

        def sfq(v):
            """Non-negative float, else None. For bid/ask specifically: a quote of exactly
            0 is a REAL, valid market (e.g. a deep-OTM option near expiry with no bid) — must
            NOT be treated the same as "no data", unlike sf() above."""
            try:
                f = float(v)
                return f if (f == f and f >= 0) else None
            except Exception:
                return None

        out = []
        for p in picks:
            c = p.contract
            tkr = tickers.get(c.conId)
            mark = None
            if tkr is not None:
                try:
                    ib.cancelMktData(c)
                except Exception:
                    pass
                b, a, last, close = sfq(tkr.bid), sfq(tkr.ask), sf(tkr.last), sf(tkr.close)
                # NOTE: `b`/`a` can legitimately be 0.0 (falsy) — must check `is not None`,
                # not `(b and a)`, or a real zero bid gets treated as missing and falls
                # through to a stale `last` print (the bug: OZNQ6 C1095 showed bid=0/ask=1/64
                # live in TWS, worth ~0.008, but a stale last=0.132 was used instead, inflating
                # the premium ~15x).
                mark = ((b + a) / 2) if (b is not None and a is not None) else (last or close)
            out.append((c.conId, getattr(c, "localSymbol", "") or getattr(c, "symbol", ""),
                        getattr(c, "symbol", ""), (getattr(c, "secType", "") or "").upper(),
                        getattr(c, "currency", ""),
                        getattr(c, "exchange", "") or getattr(c, "primaryExchange", ""),
                        getattr(c, "lastTradeDateOrContractMonth", "") or "",
                        float(c.strike) if getattr(c, "strike", 0) else None,
                        float(c.multiplier) if getattr(c, "multiplier", None) else 1.0,
                        float(p.position or 0), mark, float(p.avgCost or 0)))
        return out

    try:
        recs = ibl.submit(_coro, timeout_s=45)
    except Exception as e:
        return pd.DataFrame(), f"TWS positions fetch failed — {type(e).__name__}: {e}"
    if not recs:
        return pd.DataFrame(), "no live futures/options positions"

    rows = []
    for (conid, localsym, sym, sec, ccy, exch, expiry, strike, mult, pos, mark, avgcost) in recs:
        fx = 1.0                                           # USD book; non-USD needs a rate
        m = mark if mark is not None else ((avgcost / mult) if (avgcost and mult) else 0.0)
        pvb = pos * m * mult * fx
        rows.append({
            "Conid": conid,
            "Symbol": localsym,
            "Description": sym,
            "AssetClass": sec,
            "SubCategory": "",
            "Underlying": sym,
            "Currency": ccy,
            "Exchange": exch,
            "Expiry": expiry,
            "Strike": strike,
            "Multiplier": mult,
            "MarkPrice": m,
            "Quantity": pos,
            "position_value_base": pvb,
            "upnl_base": 0.0,
            "FXRateToBase": fx,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, "no live futures/options positions (or account not subscribed)"
    df["side"] = df["Quantity"].apply(lambda q: "Long" if q > 0 else "Short")
    df["is_option"] = df["AssetClass"].isin(["FOP", "OPT"])
    df["gross_base"] = df["position_value_base"].abs()
    df = df.sort_values("gross_base", ascending=False).reset_index(drop=True)
    return df, f"{len(df)} live positions from TWS"


def todays_fills():
    """Today's executions from TWS via the shared connection. Returns (DataFrame, note);
    empty df + note on failure. Columns: Conid, Symbol (localSymbol), SecType, Qty
    (signed: BOT +, SLD −), Price, Time (iso str), Commission, ExecId.

    Uses ``reqExecutionsAsync`` (default filter = today's executions). This consumes NO
    market-data lines (no reqMktData) — so, per the telemetry spec, nothing is recorded.
    Degrades gracefully: TWS down / fetch failure → (empty df, reason)."""
    _cols = ["Conid", "Symbol", "SecType", "Qty", "Price", "Time", "Commission", "ExecId",
             "Multiplier", "Currency"]
    ibl, err = _get_conn()
    if ibl is None:
        return pd.DataFrame(columns=_cols), f"TWS unavailable — {err}"

    async def _coro(ib):
        fills = await asyncio.wait_for(ib.reqExecutionsAsync(), timeout=20.0)
        out = []
        for fl in fills:
            try:
                c = fl.contract
                ex = fl.execution
                cr = getattr(fl, "commissionReport", None)
                sec = (getattr(c, "secType", "") or "").upper()
                side = (getattr(ex, "side", "") or "").upper()
                shares = float(getattr(ex, "shares", 0) or 0)
                qty = shares if side == "BOT" else -shares   # signed: BOT +, SLD −
                t = getattr(ex, "time", None)
                tstr = (t.isoformat() if hasattr(t, "isoformat")
                        else (str(t) if t is not None else ""))
                # commissionReport may be missing / None, and ib_insync fills its
                # `.commission` with an UNSET sentinel (~1.8e308) when the report has
                # not arrived yet — guard against that so we never sum garbage.
                comm = None
                if cr is not None:
                    try:
                        cv = float(getattr(cr, "commission", None))
                        if cv == cv and abs(cv) < 1e17:
                            comm = cv
                    except Exception:
                        comm = None
                out.append((
                    int(getattr(c, "conId", 0) or 0),
                    getattr(c, "localSymbol", "") or getattr(c, "symbol", ""),
                    sec, qty,
                    float(getattr(ex, "price", 0) or 0),
                    tstr, comm,
                    getattr(ex, "execId", "") or "",
                    float(getattr(c, "multiplier", 0) or 0) or 1.0,
                    getattr(c, "currency", "") or "USD",
                ))
            except Exception:
                continue
        return out

    try:
        recs = ibl.submit(_coro, timeout_s=25)
    except Exception as e:
        return pd.DataFrame(columns=_cols), f"TWS fills fetch failed — {type(e).__name__}: {e}"

    df = pd.DataFrame(recs, columns=_cols) if recs else pd.DataFrame(columns=_cols)
    if df.empty:
        return df, "no executions today"
    # Dedupe EXACT ExecId repeats only — partial fills carry distinct execIds and are
    # legitimately separate executions, so each is kept.
    df = df.drop_duplicates(subset=["ExecId"]).reset_index(drop=True)
    # Derivatives + spot FX (Rajat day-trades IDEALPRO pairs too — 2026-07-30).
    df = df[df["SecType"].isin(["FUT", "FOP", "OPT", "CASH"])].reset_index(drop=True)
    if df.empty:
        return df, "no futures/options/FX executions today"
    return df, f"{len(df)} fills today"


def live_fx_balances():
    """{currency: native cash balance} for non-USD currencies, live from TWS account
    values (incl. today's FX conversions). Returns (dict, note); {} on failure."""
    ibl, err = _get_conn()
    if ibl is None:
        return {}, f"TWS unavailable — {err}"

    async def _coro(ib):
        accts = ib.managedAccounts()
        acct = accts[0] if accts else ""
        vals = ib.accountValues(acct)                      # auto-populated on connect
        if not any(getattr(v, "tag", "") == "CashBalance" for v in vals):
            await ib.reqAccountUpdatesAsync(acct)          # fallback: subscribe then read
            vals = ib.accountValues(acct)
        return [(getattr(v, "tag", ""), getattr(v, "currency", "") or "", getattr(v, "value", ""))
                for v in vals]

    try:
        rows = ibl.submit(_coro, timeout_s=25)
    except Exception as e:
        return {}, f"TWS FX balances failed — {type(e).__name__}: {e}"

    out = {}
    for tag, ccy, val in rows:
        if tag == "CashBalance" and ccy not in ("", "BASE", "USD"):
            try:
                b = float(val)
            except Exception:
                continue
            if abs(b) > 1e-6:
                out[ccy] = b
    return out, f"{len(out)} live FX balances from TWS"


def live_account_margin():
    """Live account margin / liquidity summary from TWS (base currency).

    Returns (dict, note). Keys (whichever TWS reports): NetLiquidation, TotalCashValue,
    AvailableFunds, ExcessLiquidity, FullInitMarginReq, FullMaintMarginReq, BuyingPower.
    AvailableFunds = NetLiq − initial margin; ExcessLiquidity = NetLiq − maintenance margin.
    """
    ibl, err = _get_conn()
    if ibl is None:
        return {}, f"TWS unavailable — {err}"

    _TAGS = {"NetLiquidation", "TotalCashValue", "AvailableFunds", "ExcessLiquidity",
             "FullInitMarginReq", "FullMaintMarginReq", "BuyingPower"}

    async def _coro(ib):
        accts = ib.managedAccounts()
        acct = accts[0] if accts else ""
        vals = ib.accountValues(acct)                      # auto-populated on connect
        if not any(getattr(v, "tag", "") in _TAGS for v in vals):
            await ib.reqAccountUpdatesAsync(acct)          # fallback: subscribe then read
            vals = ib.accountValues(acct)
        return [(getattr(v, "tag", ""), getattr(v, "currency", "") or "", getattr(v, "value", ""))
                for v in vals]

    try:
        rows = ibl.submit(_coro, timeout_s=25)
    except Exception as e:
        return {}, f"TWS margin fetch failed — {type(e).__name__}: {e}"

    out = {}
    for tag, ccy, val in rows:
        if tag in _TAGS and ccy in ("", "BASE", "USD"):
            try:
                out[tag] = float(val)
            except Exception:
                pass
    if not out:
        return {}, "TWS returned no margin fields"
    return out, f"margin summary from TWS ({len(out)} fields)"


# ── Persist the LIVE snapshot so it survives tab reopen / reload ──────────────
_LIVE_SNAP_PATH = os.path.join(os.path.dirname(__file__), "risk_live_snapshot.json")


def save_live_snapshot(book, fxbal, ts):
    try:
        payload = {"ts": ts, "fxbal": fxbal or {},
                   "book": book.to_json(orient="records")}
        with open(_LIVE_SNAP_PATH, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def load_live_snapshot():
    """Return (book_df|None, fxbal_dict, ts)."""
    try:
        import io
        with open(_LIVE_SNAP_PATH) as f:
            d = json.load(f)
        book = pd.read_json(io.StringIO(d["book"]), orient="records")
        return book, dict(d.get("fxbal", {})), d.get("ts")
    except Exception:
        return None, {}, None


def clear_live_snapshot():
    try:
        os.remove(_LIVE_SNAP_PATH)
    except Exception:
        pass


# ── PnL from cached prices ───────────────────────────────────────────────────
def live_multiday_pnl(book: pd.DataFrame, cache: dict | None = None) -> dict:
    """{symbol: {1: pnl, 3: pnl, 5: pnl}} from live price vs the close N business
    days ago (Qty × Mult × Δprice × FX). Only symbols with cached data appear."""
    cache = cache if cache is not None else load_price_cache()
    out = {}
    if book is None or book.empty:
        return out
    today = date.today().isoformat()
    for _, r in book.iterrows():
        if r.get("is_option"):          # options deferred
            continue
        sym = r["Symbol"]
        ent = cache.get(sym)
        if not ent:
            continue
        live = ent.get("live")
        closes = ent.get("closes", {})
        lsrc = ent.get("live_src")
        if live is None or not closes:
            continue
        sc = sorted(closes.items())          # (date, close) oldest→newest
        if lsrc in ("closed", "prev-close"):
            # Market shut: anchor on the latest daily settlement, NOT `today` or the
            # frozen/prev-close snapshot (which can carry the PRIOR close → a false 0).
            # Mark = the latest settlement; compare it to sessions strictly before it.
            # This is also date-rollover-proof (uses the bar date, not the wall clock).
            anchor_d, anchor_c = sc[-1]
            live = anchor_c
            vals = [c for d, c in sc if d < anchor_d]
        else:
            # Open/delayed: intraday mark vs completed sessions strictly before today.
            vals = [c for d, c in sc if d < today]
        if not vals:
            continue
        qty = float(r.get("Quantity") or 0.0)
        mult = float(r.get("Multiplier") or 0.0)
        fxr = float(r.get("FXRateToBase") or 1.0)
        res = {}
        for n in (1, 3, 5):
            if len(vals) >= n:
                res[n] = qty * mult * (float(live) - vals[-n]) * fxr
        if res:
            out[sym] = res
    return out


def prices_asof(cache: dict | None = None):
    """(latest live_ts epoch or None, latest closes_date iso or None)."""
    cache = cache if cache is not None else load_price_cache()
    live_ts = [e.get("live_ts") for e in cache.values() if e.get("live_ts")]
    cdates = [e.get("closes_date") for e in cache.values() if e.get("closes_date")]
    return (max(live_ts) if live_ts else None, max(cdates) if cdates else None)

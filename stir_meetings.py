"""Hikes/cuts priced by central-bank meeting, ~1y out (Rajat 2026-08-27).

US: CBOT 30-Day Fed Funds futures (ZQ, GLBX.MDP3) — price = 100 − monthly
    average EFFR, so each month pins the day-weighted average policy rate.
EU: ICE One-Month €STR futures (EON, IFLL.IMPACT) — 100 − monthly compounded
    €STR; compounding ≈ arithmetic average at these levels (sub-0.1bp).

Bootstrap (standard): policy rate piecewise constant, changing only on
meeting EFFECTIVE dates — Fed: decision day + 1; ECB: decision Thursday + 6
days (rates historically apply from the following Wednesday). Iterate months
chronologically: a meeting-free month calibrates the running rate, a month
containing one effective date solves that meeting's post-rate from the
day-weighted average. Months with two effective dates solve under an
equal-step assumption and are flagged ≈.

Data: last settlement via ohlcv-1d (cost-guarded, ~$0.01/snapshot), disk-
cached per trade date in vol_dash_cache/. Meeting dates verified 2026-08-27
against the Fed and ECB published calendars (through Dec-2027).
"""
from __future__ import annotations

import math
import os
import pickle
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

_CACHE_DIR = str(Path(__file__).parent / "vol_dash_cache")

# decision-day (day 2 of each meeting); effective = decision + offset
_FOMC = [date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
         date(2027, 1, 27), date(2027, 3, 17), date(2027, 4, 28),
         date(2027, 6, 9), date(2027, 7, 28), date(2027, 9, 15),
         date(2027, 10, 27), date(2027, 12, 8)]
_ECB = [date(2026, 9, 10), date(2026, 10, 29), date(2026, 12, 17),
        date(2027, 2, 4), date(2027, 3, 18), date(2027, 4, 29),
        date(2027, 6, 10), date(2027, 7, 22), date(2027, 9, 9),
        date(2027, 10, 28), date(2027, 12, 16)]
# BoE MPC announcement days (Bank Rate applies from the decision day →
# eff_off 0). 2026 from the event calendar; 2027 from the BoE's published
# schedule (verified 2026-08-27).
_MPC = [date(2026, 9, 17), date(2026, 11, 5), date(2026, 12, 17),
        date(2027, 2, 4), date(2027, 3, 18), date(2027, 4, 29),
        date(2027, 6, 17), date(2027, 7, 29), date(2027, 9, 16),
        date(2027, 11, 4), date(2027, 12, 16)]
# BoJ decision days (day 2 of each MPM; Fri decisions → effective next
# business day). 2026 from the event calendar; 2027 verified vs BoJ's
# published schedule 2026-08-27.
_BOJ = [date(2026, 9, 18), date(2026, 10, 30), date(2026, 12, 18),
        date(2027, 1, 22), date(2027, 3, 18), date(2027, 4, 28),
        date(2027, 6, 11), date(2027, 7, 22), date(2027, 9, 22),
        date(2027, 10, 29), date(2027, 12, 17)]

_CB = {
    # ZQ trades actively → ohlcv-1d; EON is settlement-driven with thin
    # trading (no trade bars) → ICE settles via the statistics schema
    "FOMC": {"meetings": _FOMC, "eff_off": 1, "ds": "GLBX.MDP3",
             "parent": "ZQ.FUT", "rate": "EFFR", "schema": "ohlcv-1d"},
    "ECB":  {"meetings": _ECB, "eff_off": 6, "ds": "IFLL.IMPACT",
             "parent": "EON.FUT", "rate": "€STR", "schema": "statistics"},
    # ICE One-Month SONIA (SOA, £2500/pt, monthly EOM expiries — probed
    # 2026-08-27); Bank Rate effective on the decision day itself
    "BoE":  {"meetings": _MPC, "eff_off": 0, "ds": "IFLL.IMPACT",
             "parent": "SOA.FUT", "rate": "SONIA", "schema": "statistics"},
}
_SETTLE_STAT = 3
_MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


def _api_key() -> str:
    import options_v2 as _ov2
    k = _ov2._api_key()
    if k:
        return k
    # bare-mode fallback (st.secrets is cwd-dependent outside streamlit)
    try:
        m = re.search(r'DATABENTO_API_KEY\s*=\s*"([^"]+)"',
                      (Path(__file__).parent / ".streamlit" /
                       "secrets.toml").read_text())
        return m.group(1) if m else ""
    except OSError:
        return ""


def _parse_month(raw: str):
    """Contract month (y, m) from a raw symbol: 'ZQU6'→(2026,9);
    'EON FMQ0026!' (ICE style)→(2026,8)."""
    raw = str(raw).strip()
    m = re.search(r"FM([FGHJKMNQUVXZ])00(\d{2})", raw)
    if m:
        return 2000 + int(m.group(2)), _MONTH_CODE[m.group(1)]
    m = re.match(r"^[A-Z0-9]+?([FGHJKMNQUVXZ])(\d{1,2})$", raw)
    if m:
        y = int(m.group(2))
        y = 2000 + y if y > 9 else 2020 + y
        if y < date.today().year - 1:
            y += 10
        return y, _MONTH_CODE[m.group(1)]
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_strips() -> dict:
    """{cb: {(y,m): implied_avg_rate}}, settle date; disk-cached per day."""
    today = date.today()
    fp = os.path.join(_CACHE_DIR, f"MEET_{today.isoformat()}.pkl")
    if os.path.exists(fp):
        try:
            with open(fp, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    import databento as db
    import vol_dashboard as _vd
    client = db.Historical(key=_api_key())
    out = {"asof": None}
    for cb, cfg in _CB.items():
        try:
            # per-dataset licensed trade date (IMPACT datasets embargo ~24h —
            # end=today 422s on IFLL; _trade_date knows each feed's window)
            tdate = _vd._trade_date(cfg["ds"])
            df = None
            for _back in range(3):    # step back on the license's ~24h embargo
                start = (tdate - timedelta(days=7)).isoformat()
                end = (tdate + timedelta(days=1)).isoformat()
                try:
                    cost = client.metadata.get_cost(
                        dataset=cfg["ds"], symbols=[cfg["parent"]],
                        stype_in="parent", schema=cfg["schema"],
                        start=start, end=end)
                    if cost > 2.0:    # never fetch blind (the $25 parent trap)
                        raise RuntimeError(f"query cost ${cost:.2f} — aborted")
                    df = client.timeseries.get_range(
                        dataset=cfg["ds"], symbols=[cfg["parent"]],
                        stype_in="parent", schema=cfg["schema"], start=start,
                        end=end).to_df(map_symbols=True)
                    break
                except Exception as _fe:
                    if "unavailable_range" in str(_fe) and _back < 2:
                        tdate -= timedelta(days=1)
                        continue
                    raise
            if df is None:
                out[cb] = {"err": "no licensed window found"}
                continue
            if df.empty:
                out[cb] = {"err": "no bars returned"}
                continue
            df = df.sort_index()
            if cfg["schema"] == "statistics":        # ICE settles
                df = df.reset_index()
                df["stat_type"] = df["stat_type"].apply(int)
                df = (df[df["stat_type"] == _SETTLE_STAT]
                      .sort_values("ts_event")
                      .rename(columns={"price": "close"}))
                df = df.set_index("ts_event")
            strips = {}
            for sym, sub in df.groupby("symbol"):
                ym = _parse_month(sym)
                if ym is None:
                    continue
                px = float(sub["close"].iloc[-1])
                if 85 <= px <= 100:                  # sanity: rate 0-15%
                    strips[ym] = 100.0 - px
            out[cb] = {"months": strips, "asof": str(df.index[-1].date())}
            out["asof"] = str(df.index[-1].date())
        except Exception as e:
            out[cb] = {"err": f"{type(e).__name__}: {e}"}
    # cache only COMPLETE snapshots — a pickled half-failure would otherwise
    # freeze the error for the rest of the day (hit 2026-08-27)
    if all("months" in (out.get(cb) or {}) for cb in _CB):
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            with open(fp, "wb") as fh:
                pickle.dump(out, fh)
        except Exception:
            pass
    return out


def _days_in_month(y, m):
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return (nxt - date(y, m, 1)).days


def bootstrap(months: dict, cb: str, horizon_days: int = 370) -> dict:
    """Per-meeting implied policy path from monthly average-rate futures.
    Returns {r0, rows: [(decision, effective, pre, post, dbp, cum, approx)]}."""
    cfg = _CB[cb]
    today = date.today()
    effs = [(dec, dec + timedelta(days=cfg["eff_off"]))
            for dec in cfg["meetings"]
            if today < dec <= today + timedelta(days=horizon_days)]
    mkeys = sorted(k for k in months if date(k[0], k[1], 1)
                   <= (effs[-1][1] if effs else today))
    r, rows, cum = None, [], 0.0
    for (y, m) in mkeys:
        avg = months[(y, m)]
        n = _days_in_month(y, m)
        m_start, m_end = date(y, m, 1), date(y, m, n)
        in_m = [e for e in effs if m_start <= e[1] <= m_end]
        if not in_m:
            if r is None:
                r = avg                          # calibrates the running rate
            continue
        if r is None:
            r = avg                              # front month w/ meeting: approx
        if len(in_m) == 1:
            dec, eff = in_m[0]
            d = (eff - m_start).days             # days at pre-rate
            w_pre = d / n
            # Late-month meetings are ill-conditioned (dividing by the few
            # post-meeting days amplifies noise ×10+). When the FOLLOWING
            # month is meeting-free its contract IS the post-rate — prefer it
            # whenever the meeting sits in the back third of the month.
            ny, nm_ = (y + (m == 12), m % 12 + 1)
            nxt_clean = ((ny, nm_) in months
                         and not any(date(ny, nm_, 1) <= e[1]
                                     <= date(ny, nm_, _days_in_month(ny, nm_))
                                     for e in effs))
            if w_pre > 0.65 and nxt_clean:
                post = months[(ny, nm_)]
            elif w_pre < 1:
                post = (avg - w_pre * r) / (1 - w_pre)
            else:
                post = r
            dbp = (post - r) * 100
            cum += dbp
            rows.append((dec, eff, r, post, dbp, cum, False))
            r = post
        else:                                    # 2 meetings: equal-step assumption
            bounds = [m_start] + [e[1] for e in in_m] + [m_end + timedelta(days=1)]
            segs = [(bounds[i + 1] - bounds[i]).days / n
                    for i in range(len(bounds) - 1)]
            # avg = seg0*r + seg1*(r+Δ) + seg2*(r+2Δ)  →  solve Δ
            k = sum(s * i for i, s in enumerate(segs))
            delta = (avg - r) / k if k > 0 else 0.0
            for i, (dec, eff) in enumerate(in_m, start=1):
                post = r + delta
                dbp = delta * 100
                cum += dbp
                rows.append((dec, eff, r, post, dbp, cum, True))
                r = post
    return {"r0": (months.get(min(months)) if months else None) if not rows
            else rows[0][2], "rows": rows}


# ── BoJ leg: OSE 3M TONA via the shared IBKR connection ─────────────────────
# Databento carries neither OSE nor a TONA future (the ICE "TOA" root is an
# emissions product — probed 2026-08-27), so the JPY leg reads Rajat's OSE
# TOA3M quarterly settles through ibkr_conn (a handful of hist_bars).
def _imm_wed(y: int, m: int) -> date:
    d = date(y, m, 15)                       # 3rd Wednesday
    return d + timedelta(days=(2 - d.weekday()) % 7)


def _next_bday(d: date) -> date:
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_tona(host: str = "127.0.0.1", port: int = 7496) -> dict:
    """{(y,m): implied avg TONA over the contract's ref quarter} for the next
    ~5 quarterly TOA3M contracts (ref window = IMM Wed → IMM Wed + 3m)."""
    try:
        from ib_insync import Future
        import ibkr_conn
    except Exception as e:
        return {"err": f"ib_insync unavailable ({e})"}
    today = date.today()
    # quarterly cycle Mar/Jun/Sep/Dec; keep the 5 whose ref window (IMM Wed →
    # IMM Wed +3m) hasn't ended — includes the in-progress contract. (The
    # first cut generated Jul/Oct/Jan months — nonexistent contracts, so
    # nothing resolved; fixed 2026-08-27.)
    cands = [(y, m) for y in range(today.year - 1, today.year + 3)
             for m in (3, 6, 9, 12)]
    quarters = [q for q in cands
                if _imm_wed(q[0] + (q[1] > 9), (q[1] + 2) % 12 + 1) > today][:5]
    # Rajat 2026-08-27: quotes, not bars — one frozen snapshot for the whole
    # strip via quotes() (mdtype 2 = frozen close even when OSE is shut; no
    # hist-pacing cost, reqMktData resolves the contract itself). OSE carries
    # NUMERIC IB symbols (contract detail showed Symbol=161060091) — try
    # candidate symbols until one batch resolves.
    import math as _m

    def _px(t):
        for a in ("last", "close"):
            v = getattr(t, a, None)
            if v is not None and _m.isfinite(v) and 90 <= v <= 100:
                return float(v)
        return None

    dbg = [f"today={today} quarters={quarters}"]
    ibl, cerr = ibkr_conn.get_conn()
    if ibl is None:
        return {"err": f"IBKR connection failed: {cerr}",
                "debug": "\n".join(dbg + [f"get_conn -> None ({cerr})"])}
    dbg.append("get_conn OK (shared clientId conn)")

    # capture TWS error codes for the debug report (200 = no security def,
    # 354 = not subscribed, 10167/10168 = delayed-data config, …)
    _errs: list = []

    def _on_err(reqId, code, msg, contract=None, *a):
        try:
            _c = f" [{getattr(contract, 'localSymbol', '') or getattr(contract, 'symbol', '')}]" if contract else ""
            _errs.append(f"err {code}{_c}: {str(msg)[:90]}")
        except Exception:
            pass
    try:
        ibl._ib.errorEvent += _on_err
    except Exception:
        dbg.append("(could not hook errorEvent)")

    def _snap(cons, note):
        # mdtype 4 = delayed-frozen: Rajat's OSE data is the delayed feed
        # (TWS shows it automatically; the API must request delayed —
        # frozen(2) got err 10168 "delayed not enabled", 2026-08-27)
        try:
            ts = ibkr_conn.quotes(cons, mdtype=4, settle_s=4.0, ibl=ibl,
                                  tag="tona_meetings")
        except Exception as e:
            dbg.append(f"{note}: quotes RAISED {type(e).__name__}: {e}")
            return [None] * len(cons)
        return ts

    def _fmt(t):
        _c = getattr(t, "contract", None)
        return (f"conId={getattr(_c, 'conId', '?')} "
                f"local='{getattr(_c, 'localSymbol', '')}' "
                f"last={getattr(t, 'last', None)} close={getattr(t, 'close', None)}")

    # Resolution settled 2026-08-27 (see memory): plain symbol=TOA3M +
    # ref-quarter contract month resolves; the whole earlier failure was
    # mdtype (frozen→10168) — NB ib_insync never backfills conId on the
    # request contract, so success is judged by PRICE, never by conId.
    cs = [Future(symbol="TOA3M", exchange="OSE.JPN", currency="JPY",
                 tradingClass="TOA3M",
                 lastTradeDateOrContractMonth=f"{qy:04d}{qm_:02d}")
          for (qy, qm_) in quarters]
    ts = _snap(cs, "strip")
    for q, t in zip(quarters, ts):
        dbg.append(f"strip {q}: " + (_fmt(t) if t is not None else "None"))
    out = {q: _px(t) for q, t in zip(quarters, ts) if t is not None}
    out = {q: 100.0 - p for q, p in out.items() if p is not None}
    try:
        ibl._ib.errorEvent -= _on_err
    except Exception:
        pass
    if _errs:
        seen = list(dict.fromkeys(_errs))
        dbg.append("TWS errors during snapshots:")
        dbg.extend("  " + e for e in seen[:12])
    if out:
        return {"windows": out, "debug": "\n".join(dbg)}
    return {"err": "no TOA3M quotes resolved — see debug",
            "debug": "\n".join(dbg)}


def bootstrap_windows(windows: dict, decisions: list,
                      horizon_days: int = 370) -> dict:
    """BoJ per-meeting attribution from QUARTERLY compounded windows via
    ridge least-squares — quarterly windows cannot uniquely pin each meeting
    (more meetings than contracts), so steps are regularised toward zero and
    every row is flagged ≈. Window of contract (y,m) = IMM Wed(m) → +3m."""
    import numpy as np
    today = date.today()
    effs = [(dec, _next_bday(dec)) for dec in decisions
            if today < dec <= today + timedelta(days=horizon_days)]
    if not effs or not windows:
        return {"r0": None, "rows": []}
    k = len(effs)
    rows_A, rows_b = [], []
    r0_anchor = None
    for (y, m), avg in sorted(windows.items()):
        ws, we = _imm_wed(y, m), _imm_wed(y + (m > 9), (m + 2) % 12 + 1)
        if we <= today:
            continue
        if ws <= today:                       # in-progress window ≈ r0 anchor
            r0_anchor = avg
            continue
        n = (we - ws).days
        coef = [1.0] + [max(0, (we - max(ws, eff)).days) / n
                        for _dec, eff in effs]
        rows_A.append(coef)
        rows_b.append(avg)
    if not rows_A:
        return {"r0": r0_anchor, "rows": []}
    lam = 0.03                                # ridge on steps (rate %-units)
    A = np.array(rows_A)
    b = np.array(rows_b)
    if r0_anchor is not None:                 # soft prior on r0
        A = np.vstack([A, [3.0] + [0.0] * k])
        b = np.append(b, 3.0 * r0_anchor)
    reg = np.hstack([np.zeros((k, 1)), lam * np.eye(k)])
    A = np.vstack([A, reg])
    b = np.append(b, np.zeros(k))
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = float(x[0])
    rows, cum = [], 0.0
    for i, (dec, eff) in enumerate(effs):
        dbp = float(x[1 + i]) * 100
        cum += dbp
        rows.append((dec, eff, r, r + dbp / 100, dbp, cum, True))
        r += dbp / 100
    return {"r0": float(x[0]), "rows": rows}


# ── Live overlay (Rajat 2026-08-28: "add a button for live prices when
# needed") — quotes the ZQ/EON/SOA strips through the shared IBKR conn and
# overrides the settlement months where a live/frozen price resolves. ────────
_LIVE_SPECS = {
    "FOMC": ("ZQ", "CBOT", "USD"),
    "ECB":  ("EON", "ICEEU", "EUR"),
    "BoE":  ("SOA", "ICEEU", "GBP"),
}


def fetch_live_strips() -> dict:
    """{cb: {(y,m): rate}} from live IBKR quotes (mdtype 4 = delayed-frozen —
    live where entitled, last close where shut). Partial results are fine:
    the caller overlays onto settles month-by-month."""
    try:
        from ib_insync import Future
        import ibkr_conn
    except Exception as e:
        return {"err": f"ib_insync unavailable ({e})"}
    ibl, cerr = ibkr_conn.get_conn()
    if ibl is None:
        return {"err": f"IBKR connection failed: {cerr}"}
    today = date.today()
    months = []
    y, m = today.year, today.month
    for _ in range(14):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    out = {}
    for cb, (sym, exch, ccy) in _LIVE_SPECS.items():
        cs = [Future(symbol=sym, exchange=exch, currency=ccy,
                     lastTradeDateOrContractMonth=f"{qy:04d}{qm:02d}")
              for (qy, qm) in months]
        try:
            ts = ibkr_conn.quotes(cs, mdtype=4, settle_s=4.0, ibl=ibl,
                                  tag="meetings_live")
        except Exception as e:
            out[cb] = {"err": f"{type(e).__name__}: {e}"}
            continue
        strip = {}
        for q, t in zip(months, ts):
            if t is None:
                continue
            px = None
            for a in ("last", "close"):
                v = getattr(t, a, None)
                if v is not None and math.isfinite(v) and 85 <= v <= 100:
                    px = float(v)
                    break
            if px is not None:
                strip[q] = 100.0 - px
        out[cb] = {"months": strip}
    return out


# ── UI ───────────────────────────────────────────────────────────────────────
_TH = ("background:#1E293B;color:#F8FAFC;font-size:11px;font-weight:600;"
       "padding:5px 8px;text-align:right;white-space:nowrap")
_THL = _TH.replace("text-align:right", "text-align:left")
_TD = "font-size:11.5px;padding:4px 8px;border-bottom:1px solid #E2E8F0;text-align:right"
_TDL = _TD.replace("text-align:right", "text-align:left")


def _dbp_cell(v, approx=False):
    if v is None or not math.isfinite(v):
        return f"<td style='{_TD};color:#CBD5E1'>—</td>"
    c = "#B91C1C" if v > 2 else ("#047857" if v < -2 else "#64748B")
    pre = "≈ " if approx else ""
    return (f"<td style='{_TD};color:{c};font-weight:700'>{pre}{v:+.1f}</td>")


def _cb_table(cb: str, res: dict) -> str:
    h = (f"<tr><th style='{_THL}'>{cb} meeting</th><th style='{_TH}'>eff.</th>"
         f"<th style='{_TH}'>Δbp priced</th><th style='{_TH}'>cum bp</th>"
         f"<th style='{_TH}'>implied after</th></tr>")
    b = ""
    for dec, eff, pre, post, dbp, cum, approx in res["rows"]:
        b += (f"<tr><td style='{_TDL}'><b>{dec.strftime('%d %b %y')}</b></td>"
              f"<td style='{_TD};color:#94A3B8'>{eff.strftime('%d %b')}</td>"
              f"{_dbp_cell(dbp, approx)}"
              f"<td style='{_TD}'>{cum:+.1f}</td>"
              f"<td style='{_TD};font-weight:600'>{post:.3f}%</td></tr>")
    return (f"<table style='border-collapse:collapse;width:100%;"
            f"font-family:monospace'><thead>{h}</thead><tbody>{b}</tbody></table>")


def _path_chart(results: dict):
    import plotly.graph_objects as go
    fig = go.Figure()
    cols = {"FOMC": "#1E40AF", "ECB": "#0E7490", "BoE": "#7C3AED",
            "BoJ": "#B45309"}
    today = date.today()
    for cb, res in results.items():
        if not res["rows"]:
            continue
        xs = [today]
        ys = [res["rows"][0][2]]
        for _dec, eff, _pre, post, _dbp, _cum, _ap in res["rows"]:
            xs.append(eff)
            ys.append(post)
        lbl = f"{cb} ({_CB[cb]['rate'] if cb in _CB else 'TONA'})"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, name=lbl, mode="lines+markers",
            line=dict(color=cols.get(cb, "#64748B"), width=2, shape="hv"),
            marker=dict(size=5),
            hovertemplate="%{x|%d %b %y}: %{y:.3f}%<extra>" + lbl + "</extra>"))
    fig.update_layout(
        height=380, template="plotly_white",
        margin=dict(l=10, r=20, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis_title="implied policy-linked o/n rate (%)",
        hovermode="x unified")
    fig.update_yaxes(gridcolor="#EEF1F6")
    return fig


def render_meetings(host: str = "127.0.0.1", port: int = 7496):
    st.markdown("#### Meetings — hikes & cuts priced, next 12 months")
    _bc1, _bc2 = st.columns([1.1, 4.9])
    if _bc1.button("⚡ Live", key="_meet_live_btn",
                   help="Overlay LIVE strip quotes from IBKR (ZQ/EON/SOA, "
                        "delayed-frozen) onto the settlement bootstrap. "
                        "Settles remain the default on load; live lasts "
                        "until the tab reruns cold."):
        with st.spinner("Quoting strips via IBKR…"):
            st.session_state["_meet_live"] = fetch_live_strips()
    _live = st.session_state.get("_meet_live")
    data = fetch_strips()
    results = {}
    _live_n = {}
    for cb in _CB:
        d = data.get(cb) or {}
        months = dict(d.get("months") or {})
        if _live and "months" in (_live.get(cb) or {}):
            _lv = _live[cb]["months"]
            months.update(_lv)              # live overrides settle, per month
            _live_n[cb] = len(_lv)
        if months:
            results[cb] = bootstrap(months, cb)
        elif "err" in d:
            st.warning(f"{cb}: {d.get('err', 'no data')}")
    if _live:
        _bits = [f"{cb} {n}m" for cb, n in _live_n.items() if n]
        _errs = [f"{cb}: {(_live.get(cb) or {}).get('err')}"
                 for cb in _LIVE_SPECS
                 if "err" in (_live.get(cb) or {})]
        _bc2.caption("⚡ live overlay: "
                     + (" · ".join(_bits) if _bits else "no live quotes")
                     + ((" · " + " · ".join(_errs)) if _errs else "")
                     + (f" · {_live.get('err')}" if _live.get("err") else ""))
    tona = fetch_tona(host, port)
    if "windows" in tona:
        boj = bootstrap_windows(tona["windows"], _BOJ)
        if boj["rows"]:
            results["BoJ"] = boj
    else:
        st.caption(f"BoJ leg unavailable — {tona.get('err', '')} "
                   "(OSE TOA3M needs TWS running)")
    if tona.get("debug") and "windows" not in tona:
        with st.expander("🐛 BoJ debug report — paste this to Claude"):
            st.code(tona["debug"])
    if not results:
        return
    st.plotly_chart(_path_chart(results), use_container_width=True)
    _rate_lbl = {"FOMC": "EFFR", "ECB": "€STR", "BoE": "SONIA", "BoJ": "TONA"}
    cols = st.columns(len(results))
    for col, cb in zip(cols, [c for c in ("FOMC", "ECB", "BoE", "BoJ")
                              if c in results]):
        with col:
            r0 = results[cb]["r0"]
            st.markdown(f"**{cb}** · {_rate_lbl[cb]} now ≈ "
                        f"**{r0:.3f}%**" if r0 else f"**{cb}**",
                        unsafe_allow_html=True)
            st.markdown(_cb_table(cb, results[cb]), unsafe_allow_html=True)
    _asofs = " · ".join(f"{cb} {(data.get(cb) or {}).get('asof', '?')}"
                        for cb in _CB if "months" in (data.get(cb) or {}))
    st.caption(
        f"Settles: **{_asofs}** — ZQ (CBOT Fed Funds, "
        "100 − monthly avg EFFR) and EON (ICE 1-Month €STR, 100 − monthly "
        "compounded €STR; compounding ≈ average at these levels). Bootstrap: "
        "policy rate assumed constant between meeting **effective** dates "
        "(Fed decision +1 day; ECB decision +6 days — rates apply from the "
        "following Wednesday). Meeting-free months calibrate the running "
        "rate; a month with two effective dates is solved with an equal-step "
        "assumption and flagged ≈. **BoE**: ICE 1-Month SONIA (SOA), Bank "
        "Rate effective on decision day. **BoJ**: OSE 3M TONA quarterly settles via "
        "IBKR — quarterly compounded windows cannot uniquely pin each MPM, so "
        "meeting steps come from a ridge least-squares attribution and every "
        "row is flagged ≈ (treat as indicative). Hikes red, cuts green "
        "(>±2bp). Meeting dates verified against the Fed/ECB/BoJ calendars "
        "through Dec-2027. Snapshot cached per day (~$0.01/refresh).")


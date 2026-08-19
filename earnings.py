"""📅 Earnings — calendar for the top-50 names by market cap: S&P + SPCX + TSM
(Rajat 2026-08).

Views:
- **Quarter** (default, dropdown of the last 4 earnings quarters — e.g. Q2 26,
  Q1 26, Q4 25, Q3 25): each company's report for that quarter with EPS est vs
  actual, surprise % and the 1-day stock reaction. Scorecard strip on top:
  x/50 reported + market-cap-weighted 1-day reaction (+ beats count). The
  ongoing quarter shows unreported names as grey "due" rows.
- **Future**: everyone's next scheduled report with EPS estimate + fwd P/E.

Dates are TRIPLE-CHECKED: yfinance × Finnhub per-symbol calendar × NASDAQ's
public Zacks endpoint (agreeing pair ±1d wins, else earliest; ✱ + hover on
disagreement). Finnhub also BACKFILLS recent reports yfinance missed entirely
(DIS/AMD/LLY/BKNG lesson, 2026-08-06). Per-ticker event history (~8 quarters)
with reactions computed from one 18-month price pull. Disk-cached per day."""

import math
import os
import pickle
from datetime import date, timedelta

import pandas as pd
import streamlit as st

_CANDIDATES = [
    # SPCX (SpaceX) added 2026-08-06 per Rajat — not in the S&P but ~$1.45T;
    # the live top-50 cap ranking naturally bumps the smallest name out.
    # TSM (TSMC ADR) added 2026-08-11 per Rajat — non-S&P but top-10 world cap.
    "SPCX", "TSM",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B",
    "LLY", "JPM", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "NFLX", "JNJ",
    "ABBV", "BAC", "CRM", "ORCL", "CVX", "WMT", "KO", "AMD", "PEP", "ADBE",
    "CSCO", "ACN", "MRK", "LIN", "TMO", "MCD", "ABT", "PM", "WFC", "IBM",
    "GE", "TXN", "QCOM", "INTU", "CAT", "DIS", "AXP", "MS", "VZ", "AMGN",
    "GS", "ISRG", "NOW", "UBER", "T", "PLTR", "BKNG", "SPGI", "PFE", "HON",
]

_CACHE_FP = os.path.join(os.path.dirname(__file__), "earnings_cache.pkl")
_CACHE_V = 6


# ── Earnings-quarter helpers ─────────────────────────────────────────────────
def _latest_eq(d: date) -> tuple:
    """(q, yy) of the LATEST earnings quarter (the one whose season is running):
    calendar quarter before today's."""
    q = (d.month - 1) // 3 + 1
    y = d.year
    q -= 1
    if q == 0:
        q, y = 4, y - 1
    return q, y


def _eq_list(n: int = 4) -> list:
    """Last n earnings quarters, latest first: [(q, y), ...]."""
    q, y = _latest_eq(date.today())
    out = []
    for _ in range(n):
        out.append((q, y))
        q -= 1
        if q == 0:
            q, y = 4, y - 1
    return out


def _eq_label(q: int, y: int) -> str:
    return f"Q{q} {y % 100:02d}"


def _eq_window(q: int, y: int) -> tuple:
    """Reporting-season window [start, end) for earnings quarter (q, y): the
    following calendar quarter (Q2-26 results are reported during Jul-Sep 26)."""
    sq, sy = q + 1, y
    if sq == 5:
        sq, sy = 1, sy + 1
    start = date(sy, (sq - 1) * 3 + 1, 1)
    eq_, ey = sq + 1, sy
    if eq_ == 5:
        eq_, ey = 1, ey + 1
    end = date(ey, (eq_ - 1) * 3 + 1, 1)
    return start, end


# ── Cross-check sources ──────────────────────────────────────────────────────
def _fh_key():
    try:
        return st.secrets["FINNHUB_KEY"]
    except Exception:
        try:
            import toml
            for p in (os.path.expanduser("~/.streamlit/secrets.toml"),
                      os.path.join(os.path.dirname(__file__), ".streamlit",
                                   "secrets.toml")):
                if os.path.exists(p):
                    return toml.load(p).get("FINNHUB_KEY")
        except Exception:
            pass
    return None


def _fh_events(sym: str, key: str) -> list:
    """[(date, hour, eps_est, eps_act)] from Finnhub, −45d..+150d. [] on error."""
    import requests
    try:
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/earnings",
            params={"from": (date.today() - timedelta(days=45)).isoformat(),
                    "to": (date.today() + timedelta(days=150)).isoformat(),
                    "symbol": sym, "token": key}, timeout=15)
        return sorted(
            (date.fromisoformat(c["date"]), c.get("hour") or "",
             c.get("epsEstimate"), c.get("epsActual"))
            for c in r.json().get("earningsCalendar", []) if c.get("date"))
    except Exception:
        return []


def _nasdaq_next(sym: str):
    import re
    import requests
    try:
        r = requests.get(
            f"https://api.nasdaq.com/api/analyst/{sym}/earnings-date",
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=10)
        txt = ((r.json().get("data") or {}).get("announcement")) or ""
        m = re.search(r"([A-Z][a-z]{2}) (\d{1,2}), (\d{4})", txt)
        if m:
            import datetime as _dt
            return _dt.datetime.strptime(
                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%b %d %Y").date()
    except Exception:
        pass
    return None


# ── Per-ticker fetch ─────────────────────────────────────────────────────────
def _fetch_one(tkr: str, fh_key: str = None) -> dict:
    import yfinance as yf
    tk = yf.Ticker(tkr)
    info = tk.info or {}
    today = date.today()
    row = {
        "ticker": tkr,
        "name": info.get("shortName") or tkr,
        "cap": info.get("marketCap"),
        "fwd_pe": info.get("forwardPE"),
        "edate": None, "next_est": None, "ntm_pe": None,
        "events": [],            # past reports: dicts d/est/act/surp/react
        "flag": False, "srcs": "",
    }
    # NTM P/E: time-weighted blend of FY1/FY2 consensus EPS (the desk NTM
    # convention — unlike Yahoo's forwardPE which uses NEXT-fiscal-year EPS
    # only, misleading for offset-FY names like NVDA)
    try:
        est_df = tk.earnings_estimate
        fy1 = float(est_df.loc["0y", "avg"])
        fy2 = float(est_df.loc["+1y", "avg"])
        import datetime as _dt
        fye = info.get("nextFiscalYearEnd")
        fy_end = _dt.date.fromtimestamp(fye) if fye else None
        dl = (fy_end - today).days if fy_end else 182
        while dl < 0:
            dl += 365
        f_ = min(max(dl / 365.0, 0.0), 1.0)
        ntm_eps = fy1 * f_ + fy2 * (1.0 - f_)
        px = info.get("currentPrice") or info.get("regularMarketPrice")
        if ntm_eps and ntm_eps > 0 and px:
            row["ntm_pe"] = float(px) / ntm_eps
    except Exception:
        pass
    try:
        ed = tk.get_earnings_dates(limit=12).sort_index()
    except Exception:
        ed = None
    try:
        hist = tk.history(start=today - timedelta(days=550), auto_adjust=True)
        closes = hist["Close"]
        hdts = [x.date() for x in closes.index]
    except Exception:
        closes, hdts = None, []

    def _react(ts, amc: bool):
        """(pct, reaction_day) — the day whose close-to-close pair is the move."""
        try:
            pos = max(j for j, hd in enumerate(hdts) if hd <= ts)
            if amc:
                if pos + 1 < len(closes):
                    return (float(closes.iloc[pos + 1] / closes.iloc[pos] - 1) * 100,
                            hdts[pos + 1])
            elif pos >= 1 and hdts[pos] == ts:
                return (float(closes.iloc[pos] / closes.iloc[pos - 1] - 1) * 100,
                        hdts[pos])
        except Exception:
            pass
        return (None, None)

    if ed is not None and len(ed):
        for i, ts_ in enumerate(ed.index):
            d_ = ts_.date()
            est = ed.iloc[i].get("EPS Estimate")
            est = float(est) if pd.notna(est) else None
            if d_ >= today:
                if row["edate"] is None:
                    row["edate"] = d_
                    row["next_est"] = est
                continue
            act = ed.iloc[i].get("Reported EPS")
            surp = ed.iloc[i].get("Surprise(%)")
            rct, rdy = (_react(d_, ts_.hour >= 12) if closes is not None
                        else (None, None))
            row["events"].append({
                "d": d_, "est": est,
                "act": float(act) if pd.notna(act) else None,
                "surp": float(surp) if pd.notna(surp) else None,
                "react": rct, "rday": rdy,
            })
    row["events"].sort(key=lambda e: e["d"])
    if row["edate"] is None:
        try:
            cal = tk.calendar
            eds = (cal or {}).get("Earnings Date") if isinstance(cal, dict) else None
            if eds:
                futc = [d_ for d_ in eds if d_ >= today]
                row["edate"] = min(futc) if futc else max(eds)
        except Exception:
            pass

    # cross-check + backfill
    fh = _fh_events(tkr, fh_key) if fh_key else []
    fh_past = [e for e in fh if e[0] < today]
    fh_next = next((e for e in fh if e[0] >= today), None)
    nd_next = _nasdaq_next(tkr)
    if fh_past:
        fts, fhr, fest, fact = fh_past[-1]
        have = {e["d"] for e in row["events"]}
        if fts not in have and not any(abs((fts - d_).days) <= 2 for d_ in have):
            surp = ((fact / fest - 1) * 100
                    if (fest and fact and fest != 0) else None)
            rct, rdy = (_react(fts, fhr != "bmo") if closes is not None
                        else (None, None))
            row["events"].append({
                "d": fts,
                "est": float(fest) if fest is not None else None,
                "act": float(fact) if fact is not None else None,
                "surp": surp,
                "react": rct, "rday": rdy,
            })
            row["events"].sort(key=lambda e: e["d"])
    cands = {"yf": row["edate"], "fh": fh_next[0] if fh_next else None,
             "nd": nd_next}
    avail = {k: v for k, v in cands.items() if v is not None}
    row["srcs"] = " · ".join(f"{k} {v}" for k, v in cands.items() if v) or "none"
    if avail:
        pair = None
        ks = list(avail)
        for a in range(len(ks)):
            for b_ in range(a + 1, len(ks)):
                if abs((avail[ks[a]] - avail[ks[b_]]).days) <= 1:
                    pair = min(avail[ks[a]], avail[ks[b_]])
        if pair is not None:
            row["flag"] = (row["edate"] is not None
                           and abs((pair - row["edate"]).days) > 1)
            row["edate"] = pair
        else:
            row["flag"] = len(avail) > 1
            row["edate"] = min(avail.values())
        if fh_next and row["edate"] == fh_next[0] and row["next_est"] is None \
                and fh_next[2] is not None:
            row["next_est"] = float(fh_next[2])
    return row


def _load_earnings(force: bool = False) -> pd.DataFrame:
    today = date.today().isoformat()
    if not force and os.path.exists(_CACHE_FP):
        try:
            with open(_CACHE_FP, "rb") as fh:
                blob = pickle.load(fh)
            if (blob.get("day") == today and blob.get("v") == _CACHE_V
                    and len(blob.get("df", [])) >= 40):
                return blob["df"]
        except Exception:
            pass
    rows = []
    fhk = _fh_key()
    prog = st.progress(0.0, text="Fetching earnings calendar…")
    for i, t in enumerate(_CANDIDATES):
        try:
            rows.append(_fetch_one(t, fhk))
        except Exception:
            rows.append({"ticker": t, "name": t, "cap": None, "fwd_pe": None,
                         "edate": None, "next_est": None, "ntm_pe": None,
                         "events": [], "flag": False, "srcs": ""})
        prog.progress((i + 1) / len(_CANDIDATES),
                      text=f"Fetching earnings… {t} ({i + 1}/{len(_CANDIDATES)})")
    prog.empty()
    df = pd.DataFrame(rows).dropna(subset=["cap"])
    df = df.sort_values("cap", ascending=False).head(50).reset_index(drop=True)
    df["weight"] = df["cap"] / df["cap"].sum() * 100.0
    try:
        with open(_CACHE_FP, "wb") as fh:
            pickle.dump({"day": today, "v": _CACHE_V, "df": df}, fh)
    except Exception:
        pass
    return df


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def _spy_rets() -> dict:
    """{date: same-day close-to-close % return} for SPY, ~2y."""
    try:
        import yfinance as yf
        h = yf.Ticker("SPY").history(period="2y", auto_adjust=True)["Close"]
        r = (h / h.shift(1) - 1) * 100
        return {ts.date(): float(v) for ts, v in r.dropna().items()}
    except Exception:
        return {}


# ── Formatting ───────────────────────────────────────────────────────────────
def _fmt_cap(v) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    return f"${v / 1e12:.2f}T" if v >= 1e12 else f"${v / 1e9:.0f}B"


def _fmt_num(v, dp=2, suffix=""):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{dp}f}{suffix}"


_TH = ("padding:3px 9px;font-size:11px;color:#64748B;text-align:right;"
       "border-bottom:1px solid #E2E8F0;white-space:nowrap")
_TD = ("padding:3px 9px;font-size:12px;text-align:right;font-family:monospace;"
       "border-bottom:1px solid #F1F5F9;white-space:nowrap")
_GREEN, _RED = "#16A34A", "#DC2626"
# Row shading by market-cap share (Rajat 2026-08-06: biggest = darkest, fades to
# ~white for the smallest; "blue first, might change the colour later"):
_CAP_RGB = "37,99,235"      # blue — swap RGB here to retheme
_CAP_AMAX = 0.50            # alpha of the biggest cap


def _chip(lbl, val, col="#0F172A"):
    return (f"<span style='display:inline-block;padding:3px 12px;margin-right:"
            f"10px;border-radius:6px;background:#F1F5F9;font-size:12px'>"
            f"<span style='color:#64748B;font-size:10.5px;text-transform:"
            f"uppercase;letter-spacing:.05em'>{lbl}</span>&nbsp;&nbsp;"
            f"<b style='font-family:monospace;color:{col}'>{val}</b></span>")


# ── Render ───────────────────────────────────────────────────────────────────
def render_earnings():
    st.markdown(
        "<div style='background:#1E293B;color:#F8FAFC;padding:6px 12px;"
        "font-size:13px;font-weight:700;border-radius:6px;display:inline-block;"
        "margin-bottom:6px'>📅 Earnings — top 50 mega-caps (S&P + SPCX + TSM)"
        "&nbsp;&nbsp;<span style='font-weight:400;font-size:11px;color:#94A3B8'>"
        "yfinance × Finnhub × NASDAQ · cached daily</span></div>",
        unsafe_allow_html=True)
    eqs = _eq_list(4)
    c0, c1, c2, c3, c4, _sp = st.columns([0.8, 0.9, 1.0, 0.55, 0.7, 1.3])
    view = c0.radio("View", ["Quarter", "Future"], horizontal=True,
                    key="_earn_view")
    _cur = view == "Quarter"
    if _cur:
        qsel = c1.selectbox("Quarter", range(len(eqs)),
                            format_func=lambda i: _eq_label(*eqs[i]),
                            key="_earn_q")
        q, y = eqs[qsel]
        w0, w1 = _eq_window(q, y)
        ongoing = qsel == 0
    sort_opts = (["Reported date", "Market cap", "Surprise %", "1d change",
                  "Fwd P/E", "NTM P/E", "Ticker"] if _cur else
                 ["Earnings date", "Market cap", "Fwd P/E", "NTM P/E", "Ticker"])
    sort_by = c2.selectbox("Sort by", sort_opts, key=f"_earn_sort_{view}")
    asc = c3.checkbox("asc", value=not _cur, key=f"_earn_asc_{view}")
    if c4.button("🔄 Refetch", key="_earn_refetch",
                 help="Bypass today's cache (~4-5 min)."):
        df = _load_earnings(force=True)
    else:
        df = _load_earnings()
    if df.empty:
        st.warning("no earnings data — sources may be rate-limiting; retry later")
        return
    df = df.copy()

    if _cur:
        def _qevent(evts):
            hits = [e for e in evts if w0 <= e["d"] < w1]
            return hits[-1] if hits else None
        df["ev"] = df["events"].map(_qevent)
        df["rep"] = df["ev"].map(lambda e: e is not None)
        df["sdate"] = df.apply(
            lambda r: r["ev"]["d"] if r["rep"] else
            (r["edate"] if ongoing else None), axis=1)
        for f_ in ("est", "act", "surp", "react"):
            df["q_" + f_] = df["ev"].map(lambda e, f__=f_: e[f__] if e else None)

        # ── scorecard ────────────────────────────────────────────────────────
        rep_df = df[df["rep"]]
        n_rep = len(rep_df)
        rw = rep_df[rep_df["q_react"].notna()]
        wtd = (float((rw["q_react"] * rw["weight"]).sum() / rw["weight"].sum())
               if len(rw) and rw["weight"].sum() > 0 else None)
        spy = _spy_rets()
        exc_n = exc_d = 0.0
        for _j, rr_ in rw.iterrows():
            sret = spy.get(rr_["ev"].get("rday"))
            if sret is not None:
                exc_n += (rr_["q_react"] - sret) * rr_["weight"]
                exc_d += rr_["weight"]
        exc = exc_n / exc_d if exc_d > 0 else None
        beats = int((rep_df["q_surp"] > 0).sum())
        cap_pct = (float(rep_df["weight"].sum())
                   if len(rep_df) else 0.0)   # weight is already % of top-50 cap
        sc = (_chip(_eq_label(q, y), f"{n_rep}/50 reported")
              + _chip("cap reported", f"{cap_pct:.1f}%")
              + _chip("mkt-wtd 1d reaction",
                      _fmt_num(wtd, 2, "%"),
                      _GREEN if (wtd or 0) >= 0 else _RED)
              + _chip("vs SPY (excess)",
                      _fmt_num(exc, 2, "%"),
                      _GREEN if (exc or 0) >= 0 else _RED)
              + _chip("beats", f"{beats}/{n_rep}" if n_rep else "—"))
        st.markdown(f"<div style='margin:2px 0 8px 0'>{sc}</div>",
                    unsafe_allow_html=True)
        key = {"Reported date": "sdate", "Market cap": "cap",
               "Surprise %": "q_surp", "1d change": "q_react",
               "Fwd P/E": "fwd_pe", "NTM P/E": "ntm_pe",
               "Ticker": "ticker"}[sort_by]
    else:
        df["days"] = df["edate"].map(
            lambda d: (d - date.today()).days if d is not None else None)
        key = {"Earnings date": "edate", "Market cap": "cap",
               "Fwd P/E": "fwd_pe", "NTM P/E": "ntm_pe",
               "Ticker": "ticker"}[sort_by]
    df = df.sort_values(key, ascending=asc, na_position="last")

    if _cur:
        cols = ("TICKER", "COMPANY", "MKT CAP", "WGT %", "DATE",
                "EPS EST", "EPS ACT", "SURP %", "1D CHG %", "FWD P/E",
                "NTM P/E")
    else:
        cols = ("TICKER", "COMPANY", "MKT CAP", "WGT %", "EARNINGS", "DAYS",
                "EPS EST", "FWD P/E", "NTM P/E")
    hdr = "".join(f"<th style='{_TH}'>{h}</th>" for h in cols)
    body = ""
    capmax = float(df["cap"].max())
    for _i, r in df.iterrows():
        pe = _fmt_num(r["fwd_pe"], 1)
        ntm = _fmt_num(r.get("ntm_pe"), 1)
        _a = _CAP_AMAX * float(r["cap"]) / capmax if capmax else 0.0
        shade = f";background:rgba({_CAP_RGB},{_a:.3f})"
        if _cur:
            if r["rep"]:
                ev = r["ev"]
                dt_cell = ev["d"].strftime("%a %d %b")
                est, act = _fmt_num(ev["est"]), _fmt_num(ev["act"])
                sc_ = _GREEN if (ev["surp"] or 0) >= 0 else _RED
                surp = _fmt_num(ev["surp"], 1)
                rc = _GREEN if (ev["react"] or 0) >= 0 else _RED
                react = (_fmt_num(ev["react"], 1, "%")
                         if ev["react"] is not None else "—")
            else:
                if ongoing and r["edate"]:
                    _fl = (f"<span style='color:#B45309' "
                           f"title='{r.get('srcs', '')}'>✱</span>"
                           if r.get("flag") else "")
                    dt_cell = (f"<span style='color:#94A3B8;font-style:italic' "
                               f"title='{r.get('srcs', '')}'>due "
                               f"{r['edate'].strftime('%d %b')}</span>{_fl}")
                    est = _fmt_num(r["next_est"])
                else:
                    dt_cell, est = "—", "—"
                act, surp, react = "—", "—", "—"
                sc_ = rc = "#94A3B8"
            body += (
                f"<tr><td style='{_TD};text-align:left{shade}'><b>{r['ticker']}</b></td>"
                f"<td style='{_TD};text-align:left{shade}'>{r['name']}</td>"
                f"<td style='{_TD}{shade}'>{_fmt_cap(r['cap'])}</td>"
                f"<td style='{_TD}{shade}'>{r['weight']:.1f}</td>"
                f"<td style='{_TD}{shade}'>{dt_cell}</td>"
                f"<td style='{_TD}{shade}'>{est}</td>"
                f"<td style='{_TD}{shade}'>{act}</td>"
                f"<td style='{_TD}{shade};color:{sc_}'>{surp}</td>"
                f"<td style='{_TD}{shade};color:{rc}'>{react}</td>"
                f"<td style='{_TD}{shade}'>{pe}</td>"
                f"<td style='{_TD}{shade}'>{ntm}</td></tr>")
        else:
            soon = r["days"] is not None and 0 <= r["days"] <= 7
            # amber flag lives on the date/days cells only — the row itself
            # carries the cap shade
            bg = ";background:#FFF6E0" if soon else shade
            _fl = (f"<span style='color:#B45309' title='{r.get('srcs', '')}'>✱"
                   f"</span>" if r.get("flag") else "")
            ed = (f"<span title='{r.get('srcs', '')}'>"
                  f"{r['edate'].strftime('%a %d %b')}</span>{_fl}"
                  if r["edate"] else "—")
            dd = f"{int(r['days'])}d" if r["days"] is not None else "—"
            body += (
                f"<tr><td style='{_TD};text-align:left{shade}'><b>{r['ticker']}</b></td>"
                f"<td style='{_TD};text-align:left{shade}'>{r['name']}</td>"
                f"<td style='{_TD}{shade}'>{_fmt_cap(r['cap'])}</td>"
                f"<td style='{_TD}{shade}'>{r['weight']:.1f}</td>"
                f"<td style='{_TD}{bg}'>{ed}</td>"
                f"<td style='{_TD}{bg}'>{dd}</td>"
                f"<td style='{_TD}{shade}'>{_fmt_num(r['next_est'])}</td>"
                f"<td style='{_TD}{shade}'>{pe}</td>"
                f"<td style='{_TD}{shade}'>{ntm}</td></tr>")
    st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:"
                f"collapse'><thead><tr>{hdr}</tr></thead><tbody>{body}</tbody>"
                f"</table></div>", unsafe_allow_html=True)
    if _cur:
        st.caption(
            f"{_eq_label(q, y)} results, reported {w0:%d %b %y} → "
            f"{w1 - timedelta(days=1):%d %b %y}. "
            + ("Grey 'due' rows are still to come this season. " if ongoing
               else "'—' rows = no report found in the window (fiscal-calendar "
                    "offsets can shift a name's report outside it). ")
            + "1D CHG = close-to-close around the report (AMC → next day; BMO → "
              "same day). Scorecard 1d reaction is cap-weighted over reported "
              "names. Dates triple-checked (✱ = sources disagree, hover). "
              "NTM P/E = price / time-weighted FY1-FY2 consensus EPS (true "
              "next-12m basis; FWD P/E is Yahoo's next-fiscal-year basis). "
              "Cached once per day.")
    else:
        st.caption(
            "Next scheduled report per name; amber = within 7 days. EPS EST = "
            "upcoming consensus. Dates are consensus of yfinance/Finnhub/NASDAQ "
            "(agreeing pair wins, else earliest; ✱ = disagreement, hover). "
            "Cached once per day.")

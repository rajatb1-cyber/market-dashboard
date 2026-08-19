"""📅 Event Calendar — macro events driven by event_calendar.json.

The json is the single source of truth: ask Claude in chat to add / move /
remove events (it edits the file), or edit it by hand — the tab re-reads it
every render. Seeded 2026-08-11 from the official BLS release schedule and
the Fed / ECB / BoE / BoJ meeting calendars.
"""

import json
import os
from datetime import date, datetime, time as dtime

import streamlit as st

try:
    from zoneinfo import ZoneInfo
    _UK = ZoneInfo("Europe/London")
except Exception:            # tzdata missing — fall back to the native label
    ZoneInfo = None
    _UK = None

_FP = os.path.join(os.path.dirname(__file__), "event_calendar.json")

_CAT_COL = {"US Data": "#2563EB", "FOMC": "#7C3AED", "Fed": "#7C3AED",
            "ECB": "#0D9488", "BoE": "#B45309", "BoJ": "#DC2626",
            "JP Data": "#F43F5E", "Earnings": "#0EA5E9", "Other": "#64748B"}

_TH = ("padding:4px 10px;font-size:11px;color:#64748B;text-align:left;"
       "border-bottom:1px solid #E2E8F0;white-space:nowrap")
_TD = ("padding:4px 10px;font-size:12.5px;text-align:left;"
       "border-bottom:1px solid #F1F5F9;white-space:nowrap")


def _load_events() -> list:
    try:
        with open(_FP, "r", encoding="utf-8") as fh:
            return json.load(fh).get("events", [])
    except Exception:
        return []


def _earnings_events(today: date, top_n: int = 10,
                     horizon_days: int = 92) -> tuple:
    """(events, cache_day) — next earnings for the top-N caps, read from the
    Earnings tab's daily cache (its triple-checked dates; READ-ONLY, never
    triggers that tab's ~4-min fetch). ([], None) when no cache exists yet."""
    import pickle
    from datetime import timedelta
    try:
        from earnings import _CACHE_FP as _efp
        with open(_efp, "rb") as fh:
            blob = pickle.load(fh)
        df, cache_day = blob["df"], blob.get("day")
    except Exception:
        return [], None
    out = []
    try:
        for _, r in df.sort_values("cap", ascending=False).head(top_n).iterrows():
            ed = r.get("edate")
            if ed is None or not (today <= ed <= today
                                  + timedelta(days=horizon_days)):
                continue
            out.append({
                "date": ed.isoformat(),
                "name": f"{r['ticker']} earnings — {r['name']}",
                "cat": "Earnings", "time": "",
                "notes": ("✱ sources disagree — see Earnings tab"
                          if r.get("flag") else "")})
    except Exception:
        return [], cache_day
    return out, cache_day


def _fmt_span(d0: date, d1: date | None) -> str:
    if d1 is None or d1 == d0:
        return d0.strftime("%a %d %b")
    if d0.month == d1.month:
        return f"{d0.strftime('%a %d')}–{d1.strftime('%d %b')}"
    return f"{d0.strftime('%d %b')}–{d1.strftime('%d %b')}"


def _uk_time(e: dict, d0: date, d1: date | None) -> str | None:
    """UK wall-clock time of the release/decision, converted from the event's
    native tz ON ITS OWN DATE — so BST/GMT vs US/EU DST transitions are right
    (e.g. 8:30 ET = 12:30 UK during the Oct 26–31 mismatch week, 13:30 rest
    of the year). Uses the decision day (end) for multi-day meetings."""
    tl, tz = e.get("time_local"), e.get("tz")
    if not (tl and tz and ZoneInfo):
        return None
    try:
        hh, mm = (int(x) for x in tl.split(":"))
        d_dec = d1 or d0
        loc = datetime.combine(d_dec, dtime(hh, mm), tzinfo=ZoneInfo(tz))
        return ("~" if e.get("approx") else "") + \
            loc.astimezone(_UK).strftime("%H:%M")
    except Exception:
        return None


def _countdown(d0: date, d1: date | None, today: date) -> tuple:
    """(label, css_color) — 'today'/'now' for live events, 'in Nd' ahead."""
    d1 = d1 or d0
    if d0 <= today <= d1:
        return ("today" if d0 == d1 else "now"), "#DC2626"
    if today < d0:
        n = (d0 - today).days
        return ("tomorrow" if n == 1 else f"in {n}d"), \
               ("#B45309" if n <= 7 else "#64748B")
    return f"{(today - d1).days}d ago", "#94A3B8"


def render_event_calendar():
    st.markdown(
        "<div style='background:#1E293B;color:#F8FAFC;padding:6px 12px;"
        "font-size:13px;font-weight:700;border-radius:6px;display:inline-block;"
        "margin-bottom:6px'>📅 Event Calendar"
        "&nbsp;&nbsp;<span style='font-weight:400;font-size:11px;color:#94A3B8'>"
        "US data + central-bank meetings — ask Claude in chat to add or change "
        "dates (edits event_calendar.json)</span></div>",
        unsafe_allow_html=True)

    today = date.today()
    events = _load_events()
    earn_evts, earn_day = _earnings_events(today)
    events = events + earn_evts
    if not events:
        st.info("event_calendar.json is empty or missing — ask Claude in chat "
                "to add events.")
        return

    show_past = st.checkbox("show past events", value=False, key="_ec_past")
    if not earn_evts:
        st.caption("💡 Earnings rows (top-10 caps, next 3m) appear once the "
                   "🏦 Equities > 📅 Earnings tab has been loaded — this tab "
                   "reuses its cache rather than refetching.")

    rows = []
    for e in events:
        try:
            d0 = date.fromisoformat(e["date"])
            d1 = date.fromisoformat(e["end"]) if e.get("end") else None
        except Exception:
            continue
        rows.append((d0, d1, e))
    rows.sort(key=lambda r: (r[0], r[1] or r[0]))
    if not show_past:
        rows = [r for r in rows if (r[1] or r[0]) >= today]
    if not rows:
        st.info("no upcoming events — tick 'show past events' or ask Claude "
                "to add more.")
        return

    html = [f"<table style='border-collapse:collapse;width:auto'>"
            f"<tr><th style='{_TH}'>Date</th><th style='{_TH}'>Event</th>"
            f"<th style='{_TH}'>Type</th><th style='{_TH}'>UK time</th>"
            f"<th style='{_TH}'>Notes</th>"
            f"<th style='{_TH};text-align:right'>In</th></tr>"]
    cur_month = None
    for d0, d1, e in rows:
        mk = d0.strftime("%B %Y")
        if mk != cur_month:
            cur_month = mk
            html.append(
                f"<tr><td colspan='6' style='padding:8px 10px 3px 10px;"
                f"font-size:11px;font-weight:700;letter-spacing:0.06em;"
                f"text-transform:uppercase;color:#334155;background:#F8FAFC;"
                f"border-bottom:1px solid #E2E8F0'>{mk}</td></tr>")
        cnt, ccol = _countdown(d0, d1, today)
        col = _CAT_COL.get(e.get("cat", "Other"), _CAT_COL["Other"])
        soon = today <= d0 and (d0 - today).days <= 7 or \
            (d0 <= today <= (d1 or d0))
        row_bg = "background:#FFFBEB;" if soon else ""
        chip = (f"<span style='display:inline-block;padding:1px 8px;"
                f"border-radius:10px;font-size:10.5px;font-weight:700;"
                f"color:{col};background:{col}18'>{e.get('cat', 'Other')}"
                f"</span>")
        _uk = _uk_time(e, d0, d1)
        _nat = e.get("time", "")
        if _uk:
            _tcell = (f"<b style='font-family:monospace;color:#0F172A'>{_uk}"
                      f"</b>"
                      + (f" <span style='color:#94A3B8;font-size:10.5px'>"
                         f"· {_nat}</span>" if _nat else ""))
        else:
            _tcell = (f"<span style='color:#64748B;font-size:11.5px'>{_nat}"
                      f"</span>")
        html.append(
            f"<tr style='{row_bg}'>"
            f"<td style='{_TD};font-family:monospace'>{_fmt_span(d0, d1)}</td>"
            f"<td style='{_TD};font-weight:600;color:#0F172A'>"
            f"{e.get('name', '?')}</td>"
            f"<td style='{_TD}'>{chip}</td>"
            f"<td style='{_TD}'>{_tcell}</td>"
            f"<td style='{_TD};color:#64748B;font-size:11.5px'>"
            f"{e.get('notes', '')}</td>"
            f"<td style='{_TD};text-align:right;font-weight:700;"
            f"color:{ccol}'>{cnt}</td></tr>")
    html.append("</table>")
    st.markdown("".join(html), unsafe_allow_html=True)

    st.caption(
        "Times shown in **UK wall clock**, converted from each event's native "
        "timezone on its own date — so the late-Oct week where UK clocks have "
        "gone back but US/EU haven't shows the true (earlier) UK time. Native "
        "time in grey; ~ = not a fixed time (BoJ statements). Sources: BLS "
        "2026 release schedule · Fed, ECB, BoE and BoJ official calendars, "
        "seeded through Dec 2026. Earnings rows = top-10 market caps' next "
        "report within ~3 months, pulled live from the 📅 Earnings tab's "
        "daily cache (yfinance × Finnhub × NASDAQ triple-check"
        + (f", as of {earn_day}" if earn_day else "") + ") — they follow "
        "that tab automatically as dates firm up. Rows within 7 days are "
        "highlighted. To add anything else — auctions, OPEC, elections, "
        "personal reminders — just tell Claude the date and name in chat.")

"""Databento spend ledger — records the USD cost of every historical fetch.

Lives in the same SQLite file as the IBKR usage ledger (ibkr_usage.db) so the
Risk tab's usage panel can show both providers in one place. Costs come from
Databento's free metadata.get_cost endpoint, queried best-effort after each
real fetch (the disk cache in options_v2/rates_options means each unique
fetch happens ~once per day, so this adds one cheap metadata call per day per
market — not per click).

The point is VISIBILITY: daily loads cost ~$0.01/market (measured 2026-07-27);
the real budget risks are one-off full-venue scans (ALL_SYMBOLS enumerations
are orders of magnitude bigger). This ledger makes any spend spike obvious.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import date

_DB_PATH = os.path.join(os.path.dirname(__file__), "ibkr_usage.db")
_LOCK = threading.Lock()


def _conn():
    c = sqlite3.connect(_DB_PATH, timeout=5)
    c.execute("""CREATE TABLE IF NOT EXISTS databento_cost (
        ts REAL, day TEXT, dataset TEXT, schema TEXT, symbols TEXT, usd REAL)""")
    return c


def record_cost(dataset: str, schema: str, symbols, usd: float) -> None:
    """Append one fetch's cost. Never raises."""
    try:
        with _LOCK:
            c = _conn()
            c.execute("INSERT INTO databento_cost VALUES (?,?,?,?,?,?)",
                      (time.time(), date.today().isoformat(), str(dataset),
                       str(schema), ",".join(map(str, symbols))[:120], float(usd)))
            c.commit()
            c.close()
    except Exception:
        pass


def cost_today() -> float:
    try:
        with _LOCK:
            c = _conn()
            row = c.execute("SELECT COALESCE(SUM(usd),0) FROM databento_cost WHERE day=?",
                            (date.today().isoformat(),)).fetchone()
            c.close()
        return float(row[0] or 0.0)
    except Exception:
        return 0.0


def cost_today_detail() -> list:
    """[(dataset, schema, symbols, usd)] for today, biggest first."""
    try:
        with _LOCK:
            c = _conn()
            rows = c.execute(
                "SELECT dataset, schema, symbols, SUM(usd) FROM databento_cost "
                "WHERE day=? GROUP BY dataset, schema, symbols ORDER BY SUM(usd) DESC",
                (date.today().isoformat(),)).fetchall()
            c.close()
        return rows
    except Exception:
        return []


def cost_by_day(days: int = 14) -> list:
    """[(day, usd)] newest first."""
    try:
        with _LOCK:
            c = _conn()
            rows = c.execute(
                "SELECT day, SUM(usd) FROM databento_cost GROUP BY day "
                "ORDER BY day DESC LIMIT ?", (int(days),)).fetchall()
            c.close()
        return rows
    except Exception:
        return []


# ── Streamlit panel (rendered inside the Databento-using tabs) ────────────────
_TD = "padding:3px 8px; font-size:11px; border-bottom:1px solid #E8EDF5; white-space:nowrap"
_TH = ("background:#1B3A6B; color:#FFFFFF; font-size:11px; font-weight:700; "
       "padding:4px 8px; white-space:nowrap")


def render_panel() -> None:
    """Compact spend panel — call inside an st.expander on tabs that fetch Databento."""
    import streamlit as st
    st.markdown(f"**Databento spend — today: ${cost_today():.3f}**")
    det = cost_today_detail()
    if det:
        rows = "".join(
            f"<tr><td style='{_TD}'>{d}</td><td style='{_TD}'>{s}</td>"
            f"<td style='{_TD}'>{sym}</td>"
            f"<td style='{_TD};text-align:right'>${u:.4f}</td></tr>"
            for d, s, sym, u in det[:12])
        st.markdown(
            f"<table style='border-collapse:collapse'><tr>"
            f"<th style='{_TH}'>Dataset</th><th style='{_TH}'>Schema</th>"
            f"<th style='{_TH}'>Symbols</th><th style='{_TH}'>USD</th></tr>"
            f"{rows}</table>", unsafe_allow_html=True)
    else:
        st.caption("No paid fetches yet today — the disk cache is serving everything.")
    days = cost_by_day(14)
    if days:
        dtot = "  ·  ".join(f"{d[5:]}: ${u:.2f}" for d, u in days[:7])
        st.caption(f"Daily totals: {dtot}. Normal full-day load ≈ $0.20-0.30; a spike means "
                   f"an unusual query (full-venue scans are the budget killers).")

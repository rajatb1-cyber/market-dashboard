"""Tiny restart-proof daily cache (Rajat 2026-08-28: "store it in a local
database so a restart does not need to recompute for the day").

SQLite key→JSON store keyed by (name, day). st.cache_data survives reruns
but dies with the process; values that are computed ONCE PER DAY (CTAz
column, etc.) belong here so an app restart re-reads instead of recomputing.
Use as a read-through layer UNDER st.cache_data:

    v = daily_store.get("ensz|^TNX")
    if v is None:
        v = expensive_compute()
        daily_store.put("ensz|^TNX", v)

Values must be JSON-serialisable. Rows older than ~10 days are pruned
opportunistically on write. Derived data only — deliberately NOT in the
OneDrive backup lists.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

_DB = str(Path(__file__).parent / "daily_cache.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB, timeout=10)
    c.execute("CREATE TABLE IF NOT EXISTS daily_cache ("
              "name TEXT NOT NULL, day TEXT NOT NULL, value TEXT,"
              "PRIMARY KEY (name, day))")
    return c


def get(name: str, day: date | None = None):
    """Today's stored value for `name`, or None."""
    d = (day or date.today()).isoformat()
    try:
        c = _conn()
        row = c.execute("SELECT value FROM daily_cache WHERE name=? AND day=?",
                        (name, d)).fetchone()
        c.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def put(name: str, value, day: date | None = None) -> None:
    d = (day or date.today()).isoformat()
    try:
        c = _conn()
        c.execute("INSERT OR REPLACE INTO daily_cache (name, day, value) "
                  "VALUES (?,?,?)", (name, d, json.dumps(value)))
        c.execute("DELETE FROM daily_cache WHERE day < ?",
                  ((date.today() - timedelta(days=10)).isoformat(),))
        c.commit()
        c.close()
    except Exception:
        pass


# ── DataFrame variant (pickle BLOBs, separate table) — for the deep-history
# price fetches that every CTA surface shares ────────────────────────────────
def _conn_df() -> sqlite3.Connection:
    c = sqlite3.connect(_DB, timeout=10)
    c.execute("CREATE TABLE IF NOT EXISTS daily_df ("
              "name TEXT NOT NULL, day TEXT NOT NULL, blob BLOB,"
              "PRIMARY KEY (name, day))")
    return c


def get_df(name: str, day: date | None = None):
    import pickle
    d = (day or date.today()).isoformat()
    try:
        c = _conn_df()
        row = c.execute("SELECT blob FROM daily_df WHERE name=? AND day=?",
                        (name, d)).fetchone()
        c.close()
        return pickle.loads(row[0]) if row else None
    except Exception:
        return None


def put_df(name: str, df, day: date | None = None) -> None:
    import pickle
    d = (day or date.today()).isoformat()
    try:
        c = _conn_df()
        c.execute("INSERT OR REPLACE INTO daily_df (name, day, blob) "
                  "VALUES (?,?,?)", (name, d, pickle.dumps(df)))
        c.execute("DELETE FROM daily_df WHERE day < ?",
                  ((date.today() - timedelta(days=3)).isoformat(),))
        c.commit()
        c.close()
    except Exception:
        pass

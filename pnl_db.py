"""
Local SQLite accumulator for IBKR Flex P&L sections.

Append strategy per section type:
  ACCUMULATE  — append new rows, dedup by natural key (trades, nav, prior_period_pnl, dividends, interest)
  SYM_REPLACE — keep latest value per symbol; accumulates all symbols ever traded (realized_pnl, mtm_pnl)
  SNAPSHOT    — full replace on every upload (positions, pnl_summary, etc.)
"""
import hashlib
import os
import sqlite3
import pandas as pd
from flex_parser import _clean_sections

_DB_PATH = os.path.join(os.path.dirname(__file__), "pnl_data.db")

# Sections to accumulate — dedup key columns (None → hash-dedup all columns)
_ACCUMULATE: dict[str, list[str] | None] = {
    "trades":           ["TradeID"],
    "nav":              ["ReportDate"],
    "prior_period_pnl": None,
    "dividends":        None,
    "interest":         None,
    "fx_activity":      None,
}

# Sections where we keep the latest value per symbol key (so open-position P&L updates correctly)
_SYM_REPLACE: dict[str, list[str]] = {
    "realized_pnl": ["Symbol", "AssetClass"],
    "mtm_pnl":      ["Symbol"],
}

# Sections fully replaced on each upload (point-in-time snapshots)
_SNAPSHOT = {"positions", "period_pnl", "pnl_summary", "cash_summary", "mtm_asset_class",
             "statement_of_funds"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _col(name: str) -> str:
    """Quote a column name for use in SQL."""
    return f'"{name}"'


def _row_hashes(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda r: hashlib.md5(str(tuple(r.values)).encode()).hexdigest(), axis=1
    )


def _ensure_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame):
    """Create table from DataFrame schema if it doesn't exist yet."""
    df.head(0).to_sql(name, conn, if_exists="append", index=False)


# ── Per-section upsert strategies ─────────────────────────────────────────────

def _upsert_accumulate(conn: sqlite3.Connection, name: str, df: pd.DataFrame,
                       keys: list[str] | None) -> int:
    df = df.copy()

    if keys:
        # Create table from df schema (no extra columns)
        _ensure_table(conn, name, df)
        try:
            quoted_keys = ", ".join(_col(k) for k in keys)
            existing = pd.read_sql(f"SELECT {quoted_keys} FROM {_col(name)}", conn)
            # Compare as strings to handle date-format differences between CSV and SQLite
            ex_tuples = set(existing.astype(str).apply(tuple, axis=1))
            df_key_tuples = df[keys].astype(str).apply(tuple, axis=1)
            new_rows = df[~df_key_tuples.isin(ex_tuples)]
        except Exception:
            new_rows = df
    else:
        # Hash-dedup: stamp each row with its content hash BEFORE creating the table
        # so the schema includes _hash from the start.
        df["_hash"] = _row_hashes(df.drop(columns=["_hash"], errors="ignore"))
        _ensure_table(conn, name, df)
        # Table may exist from an older upload that predates the _hash column.
        # Add it now so inserts don't fail with a schema mismatch.
        try:
            existing_cols = [r[1] for r in conn.execute(
                f"PRAGMA table_info({_col(name)})"
            ).fetchall()]
            if "_hash" not in existing_cols:
                conn.execute(f'ALTER TABLE {_col(name)} ADD COLUMN "_hash" TEXT')
        except Exception:
            pass
        try:
            existing_hashes = set(
                pd.read_sql(f"SELECT _hash FROM {_col(name)}", conn)["_hash"]
            )
        except Exception:
            existing_hashes = set()
        new_rows = df[~df["_hash"].isin(existing_hashes)]

    if not new_rows.empty:
        new_rows.to_sql(name, conn, if_exists="append", index=False)
    return len(new_rows)


def _upsert_sym_replace(conn: sqlite3.Connection, name: str, df: pd.DataFrame,
                        keys: list[str]) -> int:
    """Load existing, replace rows matching new keys, append new symbols."""
    _ensure_table(conn, name, df)
    try:
        existing = pd.read_sql(f"SELECT * FROM {_col(name)}", conn)
    except Exception:
        existing = pd.DataFrame()

    if existing.empty:
        df.to_sql(name, conn, if_exists="replace", index=False)
        return len(df)

    # Drop rows in existing that appear in new data (will be replaced)
    merged = existing.merge(df[keys].drop_duplicates(), on=keys, how="left", indicator=True)
    kept = existing[merged["_merge"].values == "left_only"]

    combined = pd.concat([kept, df], ignore_index=True)
    combined.to_sql(name, conn, if_exists="replace", index=False)
    return len(df)


# ── Public API ─────────────────────────────────────────────────────────────────

def upsert_sections(sections: dict) -> dict[str, int]:
    """
    Persist new Flex sections to the local DB.
    Returns {section_name: rows_added_or_replaced}.
    """
    conn = _conn()
    result: dict[str, int] = {}
    try:
        for name, df in sections.items():
            if df.empty:                       # keep _raw_ sections too (capture new/unknown ones)
                continue

            if name in _ACCUMULATE:
                n = _upsert_accumulate(conn, name, df, _ACCUMULATE[name])
                result[name] = n

            elif name in _SYM_REPLACE:
                n = _upsert_sym_replace(conn, name, df, _SYM_REPLACE[name])
                result[name] = n

            elif name in _SNAPSHOT:
                df.to_sql(name, conn, if_exists="replace", index=False)
                result[name] = len(df)

        # Compact nav to one row per ReportDate (latest). The accumulate-by-key
        # dedup misses these due to date-string formatting, so they pile up and
        # would double the cumulative-P&L chart. Keep the DB clean here.
        try:
            _nav = pd.read_sql('SELECT * FROM "nav"', conn)
            if not _nav.empty and "ReportDate" in _nav.columns:
                _nav_c = _nav.drop_duplicates(subset=["ReportDate"], keep="last")
                if len(_nav_c) < len(_nav):
                    _nav_c.to_sql("nav", conn, if_exists="replace", index=False)
        except Exception:
            pass

        conn.commit()
    finally:
        conn.close()
    return result


def load_sections() -> dict:
    """
    Load all accumulated sections from DB and re-apply type coercion.
    Returns empty dict if DB does not exist or has no tables.
    """
    if not os.path.exists(_DB_PATH):
        return {}
    conn = _conn()
    try:
        tables = [
            r[0] for r in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .fetchall()
        ]
        if not tables:
            return {}
        sections: dict = {}
        for t in tables:
            df = pd.read_sql(f"SELECT * FROM {_col(t)}", conn)
            # Drop internal hash column used for dedup
            if "_hash" in df.columns:
                df = df.drop(columns=["_hash"])
            sections[t] = df
    finally:
        conn.close()

    # Re-apply type coercion (handles both "20250115" from CSV and "2025-01-15" from SQLite)
    _clean_sections(sections)
    return sections


def db_stats() -> dict:
    """
    Returns row counts and date ranges for the key time-series tables.
    {table: {rows, date_from, date_to}}
    """
    if not os.path.exists(_DB_PATH):
        return {}
    conn = _conn()
    stats: dict = {}
    try:
        for table, date_col in [
            ("trades",           "TradeDate"),
            ("nav",              "ReportDate"),
            ("prior_period_pnl", "Date"),
        ]:
            try:
                row = conn.execute(
                    f'SELECT COUNT(*), MIN({_col(date_col)}), MAX({_col(date_col)}) '
                    f'FROM {_col(table)}'
                ).fetchone()
                stats[table] = {"rows": row[0], "from": row[1], "to": row[2]}
            except Exception:
                pass
        # realized_pnl symbol count
        try:
            n = conn.execute("SELECT COUNT(*) FROM \"realized_pnl\"").fetchone()[0]
            stats["realized_pnl"] = {"rows": n}
        except Exception:
            pass
    finally:
        conn.close()
    return stats


def clear_db():
    """Delete the database file (full reset)."""
    if os.path.exists(_DB_PATH):
        os.remove(_DB_PATH)

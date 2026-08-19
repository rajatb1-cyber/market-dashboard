"""
IBKR Flex Web Service client — pull the latest portfolio/activity statement
programmatically and load it into the local pnl_data.db (same tables the manual
CSV upload feeds).

ONE-TIME SETUP (in IBKR Client Portal → Settings → Account Settings → Reporting):
  1. Flex Web Service → Enable → generate a TOKEN (valid ~1 year). Copy it.
  2. Flex Queries → create an *Activity* Flex Query including at least
     "Open Positions" (add Trades, Change in NAV, Cash Report, MTM Performance
     Summary, etc. for full P&L). Set the query FORMAT to **CSV/Text**.
     Note its QUERY ID.
  3. Put both in Streamlit secrets (.streamlit/secrets.toml) OR environment vars:
        FLEX_TOKEN="........"
        FLEX_QUERY_ID="1234567"

NOTES
  • Flex data is END-OF-DAY (official settlement marks), not intraday/live —
    ideal for a daily VaR/PnL run. For live intraday positions use TWS/ib_insync.
  • IBKR rate-limits how often a query can be run; a daily pull is fine.
"""
from __future__ import annotations

import io
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from flex_parser import parse_flex_csv
from pnl_db import upsert_sections

_BASE = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"
_SEND = _BASE + "/SendRequest"
_GET = _BASE + "/GetStatement"


class FlexError(RuntimeError):
    """Any failure talking to the Flex Web Service or with credentials."""


def _get_creds(token: str | None, query_id: str | None) -> tuple[str, str]:
    token = token or os.environ.get("FLEX_TOKEN")
    query_id = query_id or os.environ.get("FLEX_QUERY_ID")
    if not token or not query_id:
        try:
            import streamlit as st
            token = token or st.secrets.get("FLEX_TOKEN")
            query_id = query_id or st.secrets.get("FLEX_QUERY_ID")
        except Exception:
            pass
    if not token or not query_id:
        raise FlexError(
            "Missing FLEX_TOKEN / FLEX_QUERY_ID — set them in .streamlit/secrets.toml "
            "or as environment variables."
        )
    return str(token), str(query_id)


def _http_get(url: str, params: dict, retries: int = 3) -> str:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": "market-dashboard-flex/1.0"})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8-sig", errors="replace")
        except (urllib.error.URLError, ssl.SSLError, ConnectionError, TimeoutError) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise FlexError(f"Flex Web Service connection failed after {retries} attempts: {last_err}")


def _parse_status_xml(text: str):
    """Return (status, code, reference, url, message) if `text` is a Flex status XML, else None."""
    t = text.lstrip()
    if not t.startswith("<"):
        return None
    try:
        root = ET.fromstring(t)
    except ET.ParseError:
        return None
    return (
        (root.findtext("Status") or "").strip(),
        (root.findtext("ErrorCode") or root.findtext("Code") or "").strip(),
        (root.findtext("ReferenceCode") or "").strip(),
        (root.findtext("Url") or "").strip(),
        (root.findtext("ErrorMessage") or "").strip(),
    )


def fetch_statement(token: str | None = None, query_id: str | None = None,
                    max_wait: int = 180) -> str:
    """Run the two-step Flex Web Service flow and return the raw statement text."""
    token, query_id = _get_creds(token, query_id)

    # Step 1 — SendRequest → reference code
    resp = _http_get(_SEND, {"t": token, "q": query_id, "v": "3"})
    parsed = _parse_status_xml(resp)
    if not parsed:
        raise FlexError(f"Unexpected SendRequest response: {resp[:200]}")
    status, code, ref, url, msg = parsed
    if status != "Success" or not ref:
        raise FlexError(f"Flex SendRequest failed: status={status} code={code} {msg}".strip())
    get_url = url or _GET

    # Step 2 — GetStatement, polling while IBKR generates it (warn code 1019)
    deadline = time.time() + max_wait
    while True:
        stmt = _http_get(get_url, {"t": token, "q": ref, "v": "3"})
        p = _parse_status_xml(stmt)
        if p:
            status, code, _, _, msg = p
            # Any Warn = "not ready yet", keep polling: 1019 is the documented
            # generation-in-progress code, but IBKR also emits transient Warns
            # with literal "null" code/message while the statement builds.
            if (status == "Warn" or code == "1019"
                    or "generation in progress" in msg.lower()):
                if time.time() > deadline:
                    raise FlexError(
                        f"Flex statement not ready after {max_wait}s (last "
                        f"response: status={status} code={code} {msg}) — "
                        "IBKR's Flex server is slow/busy right now. This is "
                        "usually transient; wait a minute and click Update "
                        "again.")
                time.sleep(5)
                continue
            if status and status != "Success":
                raise FlexError(f"Flex GetStatement error: status={status} code={code} {msg}".strip())
        return stmt  # actual statement (CSV) — not a status XML


def update_portfolio(token: str | None = None, query_id: str | None = None) -> dict:
    """Fetch the latest Flex statement and upsert it into pnl_data.db.
    Returns {section_name: rows_added_or_replaced}."""
    stmt = fetch_statement(token, query_id)
    if stmt.lstrip().startswith("<"):
        raise FlexError(
            "Flex query is returning XML. Set the query's output format to CSV/Text "
            "(the parser expects the sectioned CSV layout), then retry."
        )
    sections = parse_flex_csv(io.StringIO(stmt))
    return upsert_sections(sections)


if __name__ == "__main__":
    try:
        result = update_portfolio()
        print("Updated sections:", result)
    except FlexError as e:
        print("FlexError:", e)

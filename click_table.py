"""Generic clickable HTML table — a tiny bidirectional Streamlit component.

Why: Streamlit widgets cannot live inside (or stay row-aligned with) an
st.markdown HTML table, so an in-row "click to select" column is impossible
with native widgets (see the Vol-Monitor alignment lesson). This component
renders the table HTML inside an iframe and attaches click handlers to any
element carrying a data-key attribute; a click posts that key back to
Python as the component's value (clicking the selected key again clears
the selection). Selection highlight is applied inside the iframe.

Usage:
    from click_table import click_table
    sel = click_table(html, selected=st.session_state.get("my_key"),
                      key="my_key")     # sel = clicked data-key or None

Built 2026-08-17 for Core Markets' chart selector; deliberately generic —
any tab that wants clickable rows in a compact HTML table can reuse it.
"""
import os

import streamlit.components.v1 as components

_DIR = os.path.join(os.path.dirname(__file__), "click_table_frontend")

try:
    _component = components.declare_component("click_table", path=_DIR)
except Exception:                                   # pragma: no cover
    _component = None


def click_table(html: str, selected=None, key=None):
    """Render clickable-table HTML; returns the clicked data-key (or None).
    Falls back to a plain (non-clickable) markdown render if the component
    machinery is unavailable."""
    if _component is None:
        import streamlit as st
        st.markdown(html, unsafe_allow_html=True)
        return None
    return _component(html=html, selected=selected, key=key, default=None)

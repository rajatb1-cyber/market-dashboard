"""
🧪 Custom Series — define computed series (spreads, ratios, transforms) as
functions of ANY series the dashboard knows.

2026-08-14 overhaul (Rajat): the A/B/C/D pickers now search the SAME
comprehensive registry as the 📈 Chart tab (type-ahead over watchlist +
curated majors + econ catalog + sovereign curves + vol complex + STIR/CMD
db contracts + saved customs) instead of the old four-source checkbox
lists. Saved entries store the full registry spec per variable (legacy
'key' entries still evaluate — see analyzer._fetch_var_series). Because
saved series re-enter the registry as `CUST · …` on the next rerun, custom
series can be built FROM custom series (cycle-guarded in the evaluator).
"""
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import (
    _TF_MAP,
    _TF_DAYS,
    _load_custom_series,
    _write_custom_series,
    _append_custom_series,
)
from charting import _registry, get_series

_SAFE_FUNCS = {
    "abs": abs, "log": np.log, "log10": np.log10,
    "exp": np.exp, "sqrt": np.sqrt,
}

_VAR_LABELS = ["A", "B", "C", "D"]

# analyzer's _TF_MAP tops out at 5Y — extend locally with 10Y/Max
# (_TF_DAYS already knows both, so the trim just works)
_CS_TF = {**_TF_MAP, "10Y": "10y", "Max": "max"}


def _eval_formula(formula: str, local_ns: dict):
    """Returns a pd.Series on success, an Exception on error, or None if result is scalar."""
    try:
        result = eval(formula.strip(), {"__builtins__": None}, {**_SAFE_FUNCS, **local_ns})
        return result if isinstance(result, pd.Series) else None
    except Exception as exc:
        return exc


@st.fragment
def render_custom_series():
    st.markdown("### Custom Series")
    st.caption(
        "Build new series as functions of existing data — spreads, ratios, "
        "transforms. Each variable searches the full Charting registry "
        "(type 'gold', 'us 2y', 'cpi', a CUST name…). Saved series appear "
        "in the 📈 **Charting** picker as `CUST · …`, join this registry as "
        "inputs for further custom series, and remain a **Custom** source "
        "in the Analyzer."
    )

    reg = _registry()
    labels = list(reg)
    options_none = ["— none —"] + labels

    # ── Timeframe ──────────────────────────────────────────────────────────────
    tf = st.segmented_control(
        "Preview timeframe", list(_CS_TF.keys()),
        default="1Y", key="cs_tf",
    )
    period = _CS_TF.get(tf, "1y")

    # ── Variable pickers (registry type-ahead, Chart-tab style) ────────────────
    def _clear_builder():
        for v in _VAR_LABELS:
            st.session_state[f"cs_v2_{v}"] = "— none —"
        st.session_state["cs_formula"] = ""

    hd_col, clr_col = st.columns([5, 1])
    hd_col.markdown(f"**Assign series to variables** — type to search "
                    f"({len(labels)} available)")
    clr_col.button("🧹 Clear", key="cs_clear", use_container_width=True,
                   on_click=_clear_builder,
                   help="reset A/B/C/D and the formula")
    var_cols = st.columns(4)
    var_sel: dict[str, str | None] = {}
    for col, var in zip(var_cols, _VAR_LABELS):
        with col:
            # key v2: old-format keys held names from the retired source
            # lists, which are not valid options here
            sel = st.selectbox(f"{var}", options_none, index=0,
                               key=f"cs_v2_{var}")
            var_sel[var] = sel if sel != "— none —" else None

    active_vars = {v: n for v, n in var_sel.items() if n is not None}
    if not active_vars:
        st.info("Assign at least one variable above.")
        return

    var_hint = "  ·  ".join(f"**{v}** = {n}" for v, n in active_vars.items())
    st.markdown(
        f"<p style='font-size:12px;color:#64748B;margin-bottom:4px'>{var_hint}</p>",
        unsafe_allow_html=True,
    )

    # ── Formula + name inputs ──────────────────────────────────────────────────
    fi_col, ni_col = st.columns([3, 2])
    with fi_col:
        formula = st.text_input(
            "Formula",
            placeholder="e.g.   B - A   |   A / B * 100   |   (A - B) * 10000   |   log(A / B)",
            key="cs_formula",
        )
        st.caption("Supported: +  −  ×  ÷  and functions abs, log, log10, exp, sqrt")
    with ni_col:
        series_name = st.text_input(
            "Series name",
            placeholder="e.g. EUR–US 2Y Spread",
            key="cs_series_name",
        )

    # ── Fetch + preview (auto, no button needed) ───────────────────────────────
    if formula.strip():
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=_TF_DAYS.get(period, 365))
        raw: dict[str, pd.Series] = {}
        failed: list[str] = []
        for var, lbl in active_vars.items():
            try:
                with st.spinner(f"Loading {var} = {lbl}…"):
                    s, _u = get_series(reg[lbl])
            except Exception as ex:
                failed.append(f"{var} ({lbl}): {type(ex).__name__}")
                continue
            s = s[s.index >= cutoff]
            if not s.empty:
                s = s.copy()
                s.index = pd.to_datetime(s.index).normalize()
                raw[var] = s
        if failed:
            st.caption("⚠ " + " · ".join(failed))

        if not raw:
            st.warning("No data returned for the selected series.")
        else:
            combined = pd.concat(raw, axis=1).dropna()
            if combined.empty:
                st.warning("No overlapping dates across the selected series "
                           "(mixed frequencies? try a longer timeframe).")
            else:
                local_ns = {k: combined[k] for k in raw if k in combined.columns}
                result   = _eval_formula(formula, local_ns)

                if isinstance(result, Exception):
                    st.error(f"Formula error: {result}")
                elif result is None:
                    st.error(
                        "Formula must return a series — use variables A, B, C, D "
                        "with arithmetic operators."
                    )
                else:
                    # ── Preview chart ──────────────────────────────────────────
                    display_name = series_name.strip() if series_name.strip() else formula.strip()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result.index, y=result.values,
                        name=display_name,
                        line=dict(color="#1D4ED8", width=6),
                        hovertemplate=f"<b>{display_name}</b>: %{{y:.4g}}<extra></extra>",
                    ))
                    # zero line only when the data spans it — an always-on
                    # y=0 shape gets included in plotly's autorange and pins
                    # the axis at 0, so window switches barely rescale
                    _lo = float(np.nanmin(result.values))
                    _hi = float(np.nanmax(result.values))
                    if _lo <= 0 <= _hi:
                        fig.add_hline(y=0, line_dash="dot",
                                      line_color="#94A3B8", line_width=1)
                    fig.update_layout(
                        height=380,
                        margin=dict(l=10, r=10, t=40, b=10),
                        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                        xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
                        yaxis=dict(gridcolor="#E8EDF5", zeroline=False),
                        hovermode="x unified",
                        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
                        title=dict(text=f"Preview: {display_name}", font=dict(size=13)),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Save ──────────────────────────────────────────────────
                    if st.button("💾 Save Series", key="cs_save_btn"):
                        if not series_name.strip():
                            st.warning("Enter a series name before saving.")
                        else:
                            entry = {
                                "id":      str(uuid.uuid4())[:8],
                                "name":    series_name.strip(),
                                "formula": formula.strip(),
                                # full registry spec per variable — evaluated by
                                # analyzer._fetch_var_series via charting.get_series
                                "series":  {
                                    v: {"name": lbl, "spec": reg[lbl]}
                                    for v, lbl in active_vars.items()
                                },
                            }
                            _append_custom_series(entry)
                            st.success(
                                f"Saved **{series_name.strip()}**. It's now in the "
                                "comprehensive registry — searchable in 📈 Charting "
                                f"as `CUST · {series_name.strip()}`, usable as an "
                                "input to NEW custom series here, and available as "
                                "a **Custom** source in the Analyzer."
                            )
                            st.rerun()
    else:
        st.info("Enter a formula above — the preview chart will appear here.")

    # ── Saved custom series table ──────────────────────────────────────────────
    items = _load_custom_series()
    if not items:
        return

    st.markdown("---")
    st.markdown("#### Saved Custom Series")

    _TD  = "padding:4px 10px; font-size:13px; border-bottom:1px solid #E8EDF5; white-space:nowrap"
    _TH  = f"{_TD}; background:#F8FAFC; font-weight:600; color:#475569"
    _HDR = ("background:#1B3A6B; color:#FFFFFF; font-size:13px; "
            "font-weight:700; padding:5px 10px; white-space:nowrap")

    tbl  = "<table style='border-collapse:collapse;width:100%'>"
    tbl += "<tr>" + "".join(
        f"<th style='{_HDR}'>{h}</th>"
        for h in ["#", "Name", "Formula", "A", "B", "C", "D"]
    ) + "</tr>"

    for i, cs in enumerate(items, 1):
        srs = cs.get("series", {})
        def _sv(v, _s=srs):
            return _s[v]["name"] if v in _s else "—"
        tbl += (
            f"<tr>"
            f"<td style='{_TD};color:#94A3B8'>{i}</td>"
            f"<td style='{_TH}'>{cs['name']}</td>"
            f"<td style='{_TD};font-family:monospace'>{cs['formula']}</td>"
            f"<td style='{_TD}'>{_sv('A')}</td>"
            f"<td style='{_TD}'>{_sv('B')}</td>"
            f"<td style='{_TD}'>{_sv('C')}</td>"
            f"<td style='{_TD}'>{_sv('D')}</td>"
            f"</tr>"
        )
    tbl += "</table>"
    st.markdown(tbl, unsafe_allow_html=True)

    # ── Edit / Delete ──────────────────────────────────────────────────────────
    st.markdown("")
    with st.expander("Edit / Delete a custom series"):
        n_items = len(items)

        def _fmt_cs(i):
            return f"#{i+1}  {items[i]['name']}"

        sel_idx = st.selectbox("Select series", range(n_items),
                               format_func=_fmt_cs, key="cs_manage_sel")
        sel_cs  = items[sel_idx]

        with st.form("cs_edit_form", clear_on_submit=False):
            new_name = st.text_input("Name", value=sel_cs.get("name", ""),
                                     key="cs_edit_name")
            bc1, bc2, _ = st.columns([1, 1, 4])
            with bc1:
                do_edit = st.form_submit_button("💾 Save", use_container_width=True)
            with bc2:
                do_delete = st.form_submit_button("🗑 Delete", use_container_width=True,
                                                   type="secondary")

        if do_edit:
            items[sel_idx]["name"] = new_name.strip() or sel_cs["name"]
            _write_custom_series(items)
            st.success("Updated.")
            st.rerun()

        if do_delete:
            items.pop(sel_idx)
            _write_custom_series(items)
            st.success("Deleted.")
            st.rerun()

import sqlite3
import pathlib
import json
import uuid
from datetime import date, timedelta, datetime

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from watchlist import fetch_chart_data, load_config

# ── DB paths ───────────────────────────────────────────────────────────────────
_HERE         = pathlib.Path(__file__).parent
_STIR_DB      = str(_HERE / "stir_bars.db")
_COMMOD_DB    = str(_HERE / "commodities_bars.db")
_MODELS_FILE  = str(_HERE / "analyzer_models.json")
_CUSTOM_FILE  = str(_HERE / "custom_series.json")


# ── Saved model persistence ───────────────────────────────────────────────────

def _load_models() -> list[dict]:
    try:
        with open(_MODELS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_model(model: dict) -> None:
    models = _load_models()
    models.append(model)
    with open(_MODELS_FILE, "w") as f:
        json.dump(models, f, indent=2)


def _write_models(models: list[dict]) -> None:
    with open(_MODELS_FILE, "w") as f:
        json.dump(models, f, indent=2)


def _registry_lookup_maps() -> tuple[dict, dict]:
    """(key -> label, display-name -> label) over charting's registry, used to
    map a saved model's stored series back onto the search picker's labels.
    The name map is the fallback for keys that changed shape when the picker
    moved to the registry (e.g. watchlist pseudo-tickers, now 'econ:<label>')."""
    from charting import registry_options      # lazy — charting imports analyzer
    reg = registry_options()
    by_key: dict = {}
    by_name: dict = {}
    for lbl, key in reg.items():
        by_key.setdefault(key, lbl)
        disp = lbl.split(" · ", 1)[-1].strip()
        by_name.setdefault(disp, lbl)
        if disp.endswith(")") and "(" in disp:
            by_name.setdefault(disp[:disp.rfind("(")].strip(), lbl)
    return by_key, by_name


def _load_model_to_state(m: dict) -> None:
    """Populate session state so the Analyzer widgets load this saved model.
    Stored series are {"name", "key", "invert"}; both are mapped back to a
    registry label (key first, display name as fallback)."""
    srs_specs = m.get("series", [])
    try:
        by_key, by_name = _registry_lookup_maps()
    except Exception:
        by_key, by_name = {}, {}

    labels, missing = [], []
    for spec in srs_specs:
        key = spec.get("key", "")
        nm  = spec.get("name", "")
        lbl = by_key.get(key) or by_name.get(nm)
        if lbl is None or lbl in labels:
            missing.append(nm or key)
            continue
        labels.append(lbl)
        st.session_state[f"az_inv_{lbl}"] = bool(spec.get("invert", False))

    st.session_state["az_y_sel"] = labels[0] if labels else None
    st.session_state["az_x_sel"] = labels[1:4]
    st.session_state["az_load_missing"] = missing

    if m.get("timeframe"):
        st.session_state["az_tf"] = m["timeframe"]

    if m.get("timeframe") == "Custom":
        if m.get("start_date"):
            try:
                st.session_state["az_custom_start"] = date.fromisoformat(m["start_date"])
            except Exception:
                pass
        if m.get("end_date"):
            try:
                st.session_state["az_custom_end"] = date.fromisoformat(m["end_date"])
            except Exception:
                pass


def _refresh_saved_models() -> int:
    """Re-fetch data and recompute stats for every saved model. Returns count updated."""
    models = _load_models()
    updated = 0
    for m in models:
        try:
            srs_specs = m.get("series", [])
            if not srs_specs:
                continue
            tf_m = m.get("timeframe", "1Y")
            if tf_m == "Custom":
                period    = "1y"
                start_m   = m.get("start_date")
                end_m     = m.get("end_date")
            else:
                period    = _TF_MAP.get(tf_m, "1y")
                start_m   = end_m = None

            # Fetch + align all series
            raw: dict[str, pd.Series] = {}
            for spec in srs_specs:
                s = _fetch_series(spec["key"], period, start=start_m, end=end_m)
                if not s.empty:
                    s = s.copy()
                    s.index = pd.to_datetime(s.index).normalize()
                    if spec.get("invert"):
                        s = s * -1
                    raw[spec["name"]] = s

            if not raw:
                continue

            # Actual last = latest Y1 value (unaligned, most recent available)
            y_name = srs_specs[0]["name"]
            if y_name in raw:
                m["actual_last"] = round(float(raw[y_name].iloc[-1]), 2)

            # Regression stats (need ≥2 series with overlapping dates)
            if len(raw) >= 2:
                aligned = pd.concat(raw, axis=1).dropna()
                min_obs = len(raw) + 2
                if len(aligned) >= min_obs:
                    y_arr = aligned.iloc[:, 0].values.astype(float)
                    X_raw = aligned.iloc[:, 1:].values.astype(float)
                    X_mat = np.column_stack([np.ones(len(y_arr)), X_raw])
                    res = _ols(y_arr, X_mat)
                    m["r2"]          = round(res["r2"],           4)
                    m["fitted_last"] = round(float(res["fitted"][-1]), 2)
                    m["resid_last"]  = round(float(res["resid"][-1]),  2)

            updated += 1
        except Exception:
            pass

    _write_models(models)
    return updated


# ── Custom series persistence ──────────────────────────────────────────────────

def _load_custom_series() -> list[dict]:
    try:
        with open(_CUSTOM_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _write_custom_series(items: list[dict]) -> None:
    with open(_CUSTOM_FILE, "w") as f:
        json.dump(items, f, indent=2)


def _append_custom_series(item: dict) -> None:
    items = _load_custom_series()
    items.append(item)
    _write_custom_series(items)


def get_custom_series_list() -> list[tuple[str, str]]:
    """Returns [(display_name, key), ...] for all saved custom series."""
    return [(cs["name"], f"custom:{cs['id']}") for cs in _load_custom_series()]


# ── Display-name maps for DB contracts ────────────────────────────────────────
_STIR_NAMES = {"I": "Euribor", "SO3": "SONIA", "SR3": "SOFR"}
_COMMOD_NAMES = {
    "COIL": "Brent", "WTI": "WTI", "NG": "Nat Gas",
    "NGF": "UK Gas", "TFM": "TTF",
}
_MONTH_ABBR = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}

def _exp6_label(exp6: str) -> str:
    """'202612' → 'Dec26'"""
    return f"{_MONTH_ABBR.get(exp6[4:6], exp6[4:6])}{exp6[2:4]}"


# ── DB contract discovery ──────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _stir_contracts() -> list[tuple[str, str]]:
    """Returns list of (display_name, key) for every contract in stir_bars.db."""
    try:
        conn = sqlite3.connect(_STIR_DB)
        rows = conn.execute(
            "SELECT DISTINCT symbol, exchange, exp6 FROM stir_bars ORDER BY symbol, exp6"
        ).fetchall()
        conn.close()
    except Exception:
        return []
    out = []
    for sym, exch, exp6 in rows:
        label = f"{_STIR_NAMES.get(sym, sym)} {_exp6_label(exp6)}"
        key   = f"stir:{sym}:{exch}:{exp6}"
        out.append((label, key))
    return out


@st.cache_data(ttl=60)
def _commod_contracts() -> list[tuple[str, str]]:
    """Returns list of (display_name, key) for every contract in commodities_bars.db."""
    try:
        conn = sqlite3.connect(_COMMOD_DB)
        rows = conn.execute(
            "SELECT DISTINCT symbol, exchange, exp6 FROM bars ORDER BY symbol, exp6"
        ).fetchall()
        conn.close()
    except Exception:
        return []
    out = []
    for sym, exch, exp6 in rows:
        label = f"{_COMMOD_NAMES.get(sym, sym)} {_exp6_label(exp6)}"
        key   = f"commod:{sym}:{exch}:{exp6}"
        out.append((label, key))
    return out


# ── DB series fetch ────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _fetch_db_series(db_path: str, table: str, symbol: str,
                     exchange: str, exp6: str) -> pd.Series:
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            f"SELECT bar_date, close FROM {table} "
            "WHERE symbol=? AND exchange=? AND exp6=? ORDER BY bar_date",
            (symbol, exchange, exp6),
        ).fetchall()
        conn.close()
    except Exception:
        return pd.Series(dtype=float)
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows, columns=["bar_date", "close"])
    df["bar_date"] = pd.to_datetime(df["bar_date"])
    df.set_index("bar_date", inplace=True)
    return df["close"].astype(float)


# ── Ticker registries (yfinance-based) ────────────────────────────────────────

_VOL_TICKERS = [
    {"name": "VIX",   "ticker": "^VIX"},
    {"name": "VVIX",  "ticker": "^VVIX"},
    {"name": "VXN",   "ticker": "^VXN"},
    {"name": "VXD",   "ticker": "^VXD"},
    {"name": "VIX3M", "ticker": "^VIX3M"},
    {"name": "SKEW",  "ticker": "^SKEW"},
    {"name": "MOVE",  "ticker": "^MOVE"},
    {"name": "OVX",   "ticker": "^OVX"},
    {"name": "GVZ",   "ticker": "^GVZ"},
]

_TRACE_COLORS = ["#0EA5E9", "#F59E0B", "#10B981", "#F43F5E"]

_TF_MAP      = {"3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y",
                "10Y": "10y", "Max": "max"}
_TF_DAYS     = {"3mo": 92, "6mo": 183, "1y": 365, "2y": 730, "5y": 1825,
                "10y": 3653, "20y": 7305, "max": 36500}


def _fetch_series(key: str, period: str,
                  start: str | None = None, end: str | None = None) -> pd.Series:
    """Dispatch to the right data source based on the key prefix."""
    if key.startswith("stir:"):
        _, sym, exch, exp6 = key.split(":")
        s = _fetch_db_series(_STIR_DB, "stir_bars", sym, exch, exp6)
    elif key.startswith("commod:"):
        _, sym, exch, exp6 = key.split(":")
        s = _fetch_db_series(_COMMOD_DB, "bars", sym, exch, exp6)
    elif key.startswith("custom:"):
        return _fetch_custom_series(key[7:], period, start=start, end=end)
    elif key.startswith("econ:"):
        # lazy import — charting imports analyzer at module level, so a
        # top-level import here would be circular
        from charting import fetch_econ_series
        return fetch_econ_series(key[5:], period, start=start, end=end)
    else:
        yf_period = None if start else period
        df = fetch_chart_data(key, period=yf_period, interval="1d", start=start, end=end)
        s  = df["Close"].squeeze().astype(float) if not df.empty else pd.Series(dtype=float)

    # Trim DB series to requested period or custom date range
    if (key.startswith("stir:") or key.startswith("commod:")) and not s.empty:
        if start:
            s = s[s.index >= pd.Timestamp(start)]
        else:
            cutoff = pd.Timestamp(date.today() - timedelta(days=_TF_DAYS.get(period, 365)))
            s = s[s.index >= cutoff]
        if end:
            s = s[s.index <= pd.Timestamp(end)]

    return s


_CUSTOM_SAFE = {
    "abs": abs, "log": np.log, "log10": np.log10,
    "exp": np.exp, "sqrt": np.sqrt,
}


def _fetch_var_series(vspec: dict, period: str,
                      start: str | None = None, end: str | None = None) -> pd.Series:
    """One custom-series variable → Series. New-format entries (2026-08-14
    Custom Series overhaul) store a full charting-registry spec under
    'spec' (fetched via charting.get_series — econ/curve/wl sources
    included); legacy entries store an analyzer key under 'key'."""
    if vspec.get("spec"):
        import charting as _ch      # lazy — charting imports analyzer
        try:
            s, _u = _ch.get_series(vspec["spec"])
        except Exception:
            return pd.Series(dtype=float)
        if start:
            s = s[s.index >= pd.Timestamp(start)]
        else:
            cutoff = pd.Timestamp(date.today()
                                  - timedelta(days=_TF_DAYS.get(period, 365)))
            s = s[s.index >= cutoff]
        if end:
            s = s[s.index <= pd.Timestamp(end)]
        return s
    return _fetch_series(vspec["key"], period, start=start, end=end)


# customs may reference other customs (registry CUST entries are pickable
# since the overhaul) — guard against self/cyclic references
_CS_EVAL_STACK: set = set()


def _fetch_custom_series(cs_id: str, period: str,
                         start: str | None = None, end: str | None = None) -> pd.Series:
    """Evaluate a saved custom series formula and return the result as a Series."""
    cs = next((c for c in _load_custom_series() if c["id"] == cs_id), None)
    if cs is None or cs_id in _CS_EVAL_STACK:
        return pd.Series(dtype=float)

    _CS_EVAL_STACK.add(cs_id)
    try:
        raw: dict[str, pd.Series] = {}
        for var, spec in cs.get("series", {}).items():
            s = _fetch_var_series(spec, period, start=start, end=end)
            if not s.empty:
                s = s.copy()
                s.index = pd.to_datetime(s.index).normalize()
                raw[var] = s
    finally:
        _CS_EVAL_STACK.discard(cs_id)

    if not raw:
        return pd.Series(dtype=float)

    combined = pd.concat(raw, axis=1).dropna()
    if combined.empty:
        return pd.Series(dtype=float)

    local_ns = {k: combined[k] for k in raw if k in combined.columns}
    try:
        result = eval(cs["formula"], {"__builtins__": None}, {**_CUSTOM_SAFE, **local_ns})
        if isinstance(result, pd.Series):
            result.name = cs["name"]
            return result
    except Exception:
        pass
    return pd.Series(dtype=float)


# ── OLS regression helper ─────────────────────────────────────────────────────

def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """
    OLS with full statistics. X must already include a ones column for intercept.
    Returns dict: betas, se, t, p, r2, adj_r2, f_stat, f_pval, fitted, resid, n, k.
    """
    try:
        from scipy import stats as sp
        _sp = sp
    except ImportError:
        _sp = None

    n, k = X.shape
    betas = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ betas
    resid  = y - fitted
    rss    = float(np.dot(resid, resid))
    tss    = float(np.dot(y - y.mean(), y - y.mean()))
    r2     = 1.0 - rss / tss if tss > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else np.nan
    sigma2 = rss / max(n - k, 1)

    try:
        cov_b = sigma2 * np.linalg.inv(X.T @ X)
        se    = np.sqrt(np.maximum(np.diag(cov_b), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)

    t_vals = np.where(se > 0, betas / se, np.nan)

    if _sp is not None:
        p_vals = 2.0 * _sp.t.sf(np.abs(t_vals), df=max(n - k, 1))
        p_vars = k - 1
        if p_vars > 0 and 0 < r2 < 1:
            f_stat = (r2 / p_vars) / ((1.0 - r2) / max(n - k, 1))
            f_pval = float(_sp.f.sf(f_stat, p_vars, max(n - k, 1)))
        else:
            f_stat = f_pval = np.nan
    else:
        p_vals = np.full(k, np.nan)
        f_stat = f_pval = np.nan

    return {
        "betas": betas, "se": se, "t": t_vals, "p": p_vals,
        "r2": r2, "adj_r2": adj_r2,
        "f_stat": f_stat, "f_pval": f_pval,
        "fitted": fitted, "resid": resid,
        "n": n, "k": k,
    }


def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if not np.isnan(v) else "—"


def _az_short(lbl: str, cap: int = 16) -> str:
    """Registry label → compact caption (drop the 'PFX · ' and '(ticker)')."""
    t = lbl.split(" · ", 1)[-1].strip()
    if t.endswith(")") and "(" in t:
        t = t[:t.rfind("(")].strip()
    return t if len(t) <= cap else t[:cap - 1] + "…"


def _sig(p):
    if np.isnan(p): return ""
    if p < 0.01:    return "***"
    if p < 0.05:    return "**"
    if p < 0.10:    return "*"
    return ""


# ── Main render ────────────────────────────────────────────────────────────────

@st.fragment
def render_analyzer():
    st.markdown("### Analyzer")

    # ── Saved models table ────────────────────────────────────────────────────
    models = _load_models()
    if models:
        models_display = list(reversed(models))

        # ── Heading + Load row + Update ────────────────────────────────────────
        hd_col, sel_col, load_col, upd_col = st.columns([2, 2.5, 1.2, 1])
        with hd_col:
            st.markdown("#### Saved Models")
        with sel_col:
            load_idx = st.selectbox(
                "load_sel",
                range(len(models_display)),
                format_func=lambda i: f"#{i+1}  {models_display[i].get('model_name', '')}",
                key="az_load_sel",
                label_visibility="collapsed",
            )
        with load_col:
            st.markdown("<div style='margin-top:4px'>", unsafe_allow_html=True)
            if st.button("📊 Load to Chart", use_container_width=True, key="az_load_btn"):
                _load_model_to_state(models_display[load_idx])
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with upd_col:
            st.markdown("<div style='margin-top:4px'>", unsafe_allow_html=True)
            if st.button("🔄 Update", use_container_width=True, key="az_update_btn"):
                # Targeted refresh: only the data caches _refresh_saved_models reads.
                _fetch_db_series.clear()        # local: STIR/commod DB bars
                fetch_chart_data.clear()        # watchlist: yfinance daily bars
                n_updated = _refresh_saved_models()
                st.success(f"Updated {n_updated} model{'s' if n_updated != 1 else ''}.")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        _TD_m  = "padding:4px 10px; font-size:13px; border-bottom:1px solid #E8EDF5; white-space:nowrap"
        _TH_m  = f"{_TD_m}; background:#F8FAFC; font-weight:600; color:#475569"
        _HDR_m = "background:#1B3A6B; color:#FFFFFF; font-size:13px; font-weight:700; padding:5px 10px; white-space:nowrap"

        def _sname(srs, i):
            if i >= len(srs): return "—"
            nm = srs[i]["name"]
            return f"{nm} (inv)" if srs[i].get("invert") else nm

        tbl  = "<table style='border-collapse:collapse;width:100%'>"
        tbl += "<tr>" + "".join(
            f"<th style='{_HDR_m}'>{h}</th>"
            for h in ["#", "Model", "Saved", "Y", "X1", "X2", "X3", "TF", "R²", "Actual Val", "Modeled Val", "Residual", "Notes"]
        ) + "</tr>"

        for idx, m in enumerate(models_display, 1):
            srs    = m.get("series", [])
            r2_str = f"{m['r2']:.4f}"           if m.get("r2")          is not None else "—"
            av_str = f"{m['actual_last']:.2f}"  if m.get("actual_last") is not None else "—"
            fv_str = f"{m['fitted_last']:.2f}"  if m.get("fitted_last") is not None else "—"
            rv_str = f"{m['resid_last']:.2f}"   if m.get("resid_last")  is not None else "—"
            tbl += (
                f"<tr>"
                f"<td style='{_TD_m};color:#94A3B8'>{idx}</td>"
                f"<td style='{_TH_m}'>{m.get('model_name', '—')}</td>"
                f"<td style='{_TD_m};color:#94A3B8'>{m.get('saved_at', '')}</td>"
                f"<td style='{_TH_m}'>{_sname(srs, 0)}</td>"
                f"<td style='{_TD_m}'>{_sname(srs, 1)}</td>"
                f"<td style='{_TD_m}'>{_sname(srs, 2)}</td>"
                f"<td style='{_TD_m}'>{_sname(srs, 3)}</td>"
                f"<td style='{_TD_m}'>{m.get('timeframe', '')}</td>"
                f"<td style='{_TD_m}'>{r2_str}</td>"
                f"<td style='{_TD_m}'>{av_str}</td>"
                f"<td style='{_TD_m}'>{fv_str}</td>"
                f"<td style='{_TD_m}'>{rv_str}</td>"
                f"<td style='{_TD_m};max-width:220px;white-space:normal'>{m.get('notes', '')}</td>"
                f"</tr>"
            )
        tbl += "</table>"
        st.markdown(tbl, unsafe_allow_html=True)

        # ── Edit / Delete ─────────────────────────────────────────────────────
        st.markdown("")
        with st.expander("Edit / Delete a saved model"):
            n_models = len(models)
            actual_indices = list(reversed(range(n_models)))

            def _fmt_model(actual_idx):
                mm = models[actual_idx]
                display_num = n_models - actual_idx
                return f"#{display_num}  {mm.get('model_name', '(unnamed)')}  ·  {mm.get('saved_at', '')}"

            sel_actual = st.selectbox(
                "Select model", actual_indices,
                format_func=_fmt_model, key="az_manage_sel",
            )
            sel_m = models[sel_actual]

            with st.form("az_edit_form", clear_on_submit=False):
                ec1, ec2 = st.columns([2, 3])
                with ec1:
                    edit_name = st.text_input("Model name",
                                              value=sel_m.get("model_name", ""),
                                              key="az_edit_name")
                with ec2:
                    edit_notes = st.text_input("Notes",
                                               value=sel_m.get("notes", ""),
                                               key="az_edit_notes")
                bc1, bc2, _ = st.columns([1, 1, 4])
                with bc1:
                    do_save = st.form_submit_button("💾 Save Changes",
                                                    use_container_width=True)
                with bc2:
                    do_delete = st.form_submit_button("🗑 Delete",
                                                      use_container_width=True,
                                                      type="secondary")

            if do_save:
                models[sel_actual]["model_name"] = edit_name.strip() or "(unnamed)"
                models[sel_actual]["notes"]       = edit_notes
                _write_models(models)
                st.success("Changes saved.")
                st.rerun()

            if do_delete:
                models.pop(sel_actual)
                _write_models(models)
                st.success("Model deleted.")
                st.rerun()

    st.markdown("---")

    # ── Series pickers — type-to-search over the full Charting registry ───────
    from charting import registry_options     # lazy — charting imports analyzer
    reg    = registry_options()
    labels = list(reg)

    _missing = st.session_state.pop("az_load_missing", None)
    if _missing:
        st.warning("Could not map these saved series back to the registry "
                   "(loaded the rest): " + ", ".join(str(x) for x in _missing))

    c1, c2 = st.columns([1.6, 2.4])
    y_sel = c1.selectbox(f"Y — dependent (type to search, {len(labels)} series)",
                         labels, index=None, placeholder="e.g. 'gold', 'cpi', '2s10s'…",
                         key="az_y_sel")
    x_sel = c2.multiselect("X — regressors (up to 3, type to search)", labels,
                           max_selections=3, key="az_x_sel",
                           placeholder="pick up to 3 explanatory series")

    if y_sel is None:
        st.info("Pick a Y series above — the search box matches anywhere in the "
                "name (ticker, class prefix, region).")
        return

    # ── Timeframe ──────────────────────────────────────────────────────────────
    tf = st.segmented_control(
        "Timeframe", list(_TF_MAP.keys()) + ["Custom"],
        default="1Y", key="az_tf",
    )

    start_str = end_str = None
    if tf == "Custom":
        dc1, dc2, _ = st.columns([1, 1, 2])
        with dc1:
            az_start = st.date_input(
                "From",
                value=st.session_state.get("az_custom_start",
                                           date.today() - timedelta(days=365)),
                max_value=date.today() - timedelta(days=2),
                key="az_custom_start",
            )
        with dc2:
            az_end = st.date_input(
                "To",
                value=st.session_state.get("az_custom_end", date.today()),
                min_value=az_start + timedelta(days=1),
                max_value=date.today(),
                key="az_custom_end",
            )
        start_str = str(az_start)
        end_str   = str(az_end)
        period    = "1y"  # fallback for DB cutoff logic; actual range is via start/end
    else:
        period = _TF_MAP.get(tf, "1y")

    # ── Selection order: Y first (index 0 = thick black line / regression
    # dependent), then the X's; an X that repeats Y is dropped ────────────────
    chosen  = [y_sel]
    dropped = []
    for lbl in x_sel:
        if lbl == y_sel or lbl in chosen:
            dropped.append(lbl)
            continue
        chosen.append(lbl)
    if dropped:
        st.caption("⚠ ignored (same as Y / duplicate): " + ", ".join(dropped))

    # ── Invert toggles — keyed by LABEL so they survive re-ordering ───────────
    inv_cols = st.columns(max(len(chosen), 1))
    inverts  = {}
    for col, lbl in zip(inv_cols, chosen):
        with col:
            inverts[lbl] = st.checkbox(f"Invert ×−1 · {_az_short(lbl)}",
                                       key=f"az_inv_{lbl}")

    raw_selections = [(lbl, reg[lbl], _TRACE_COLORS[i], inverts[lbl])
                      for i, lbl in enumerate(chosen)]

    # ── Fetch data ─────────────────────────────────────────────────────────────
    series: dict[str, pd.Series] = {}
    for name, key, _color, _inv in raw_selections:
        with st.spinner(f"Loading {name}…"):
            s = _fetch_series(key, period, start=start_str, end=end_str)
        if not s.empty:
            series[name] = s

    if not series:
        st.warning("No data returned for the selected tickers.")
        return

    active = [(name, key, color, inv)
              for name, key, color, inv in raw_selections if name in series]
    n = len(active)

    # ── Build figure ───────────────────────────────────────────────────────────
    fig = go.Figure()

    yref_map = {0: "y", 1: "y2", 2: "y3", 3: "y4"}
    ykey_map = {0: "yaxis", 1: "yaxis2", 2: "yaxis3", 3: "yaxis4"}

    for i, (name, _key, color, inv) in enumerate(active):
        s = series[name]
        vals  = s.values * (-1 if inv else 1)
        label = f"{name} (inv)" if inv else name
        line_style = dict(color="#000000", width=4.5) if i == 0 else dict(color=color, width=2.5, dash="dot")
        fig.add_trace(go.Scatter(
            x=s.index, y=vals,
            name=label,
            yaxis=yref_map[i],
            line=line_style,
            hovertemplate=f"<b>{label}</b>: %{{y:.4g}}<extra></extra>",
        ))

    if n == 1:
        domain = [0.0, 1.0]
        margin = dict(l=10, r=10, t=40, b=10)
    elif n == 2:
        domain = [0.0, 0.92]
        margin = dict(l=10, r=60, t=40, b=10)
    else:
        domain = [0.07, 0.93]
        margin = dict(l=65, r=65, t=40, b=10)

    _ax_positions = [
        dict(side="left"),
        dict(side="right", overlaying="y"),
        dict(side="left",  overlaying="y", anchor="free", position=0.0),
        dict(side="right", overlaying="y", anchor="free", position=1.0),
    ]

    layout: dict = dict(
        height=460,
        margin=margin,
        paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        xaxis=dict(gridcolor="#E8EDF5", zeroline=False, domain=domain),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E2E8F0", borderwidth=1,
        ),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
    )

    for i, (name, _key, color, inv) in enumerate(active):
        ax_label = f"{name} (inv)" if inv else name
        ax: dict = dict(
            title=dict(text=ax_label, font=dict(color=color, size=11)),
            tickfont=dict(color=color),
            gridcolor="#E8EDF5",
            zeroline=False,
            showgrid=(i == 0),
        )
        ax.update(_ax_positions[i])
        layout[ykey_map[i]] = ax

    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # ── Regression ────────────────────────────────────────────────────────────
    reg_r2 = reg_actual_last = reg_fitted_last = reg_resid_last = None

    if len(active) >= 2:
        y_name  = active[0][0]
        x_names = [a[0] for a in active[1:]]
        _y_inv  = active[0][3]
        reg_actual_last = float(series[y_name].iloc[-1]) * (-1 if _y_inv else 1)

        st.markdown("---")
        formula = " + ".join(x_names)
        st.markdown(f"#### Regression  ·  {y_name}  =  f({formula})")

        try:
            # Align all series on common dates — normalise index to strip time components
            aligned = {}
            for a in active:
                name_a, _key_a, _col_a, inv_a = a
                s = series[name_a].copy()
                s.index = pd.to_datetime(s.index).normalize()
                if inv_a:
                    s = s * -1
                aligned[name_a] = s
            df_reg = pd.concat(aligned, axis=1).dropna()

            min_obs = len(active) + 2
            if len(df_reg) < min_obs:
                st.warning(f"Only {len(df_reg)} overlapping observations — need at least {min_obs} for regression.")
            else:
                y_arr = df_reg.iloc[:, 0].values.astype(float)
                X_raw = df_reg.iloc[:, 1:].values.astype(float)
                X_mat = np.column_stack([np.ones(len(y_arr)), X_raw])

                res = _ols(y_arr, X_mat)
                reg_r2          = res["r2"]
                reg_fitted_last = float(res["fitted"][-1])
                reg_resid_last  = float(res["resid"][-1])

                # ── Actual vs Fitted chart ─────────────────────────────────────
                fig_fit = go.Figure()
                fig_fit.add_trace(go.Scatter(
                    x=df_reg.index, y=y_arr,
                    name=f"{y_name} (actual)",
                    line=dict(color="#000000", width=3),
                    hovertemplate=f"<b>{y_name}</b>: %{{y:.1f}}<extra></extra>",
                ))
                fig_fit.add_trace(go.Scatter(
                    x=df_reg.index, y=res["fitted"],
                    name="Fitted",
                    line=dict(color="#0EA5E9", width=2, dash="dash"),
                    hovertemplate="<b>Fitted</b>: %{y:.1f}<extra></extra>",
                ))
                fig_fit.add_trace(go.Scatter(
                    x=df_reg.index, y=res["resid"],
                    name="Residual",
                    yaxis="y2",
                    line=dict(color="#DC2626", width=1.5),
                    hovertemplate="<b>Residual</b>: %{y:.1f}<extra></extra>",
                ))
                fig_fit.update_layout(
                    title=dict(text=f"Actual vs Fitted — {y_name}", font=dict(size=13, color="#1A202C")),
                    height=550,
                    margin=dict(l=10, r=60, t=40, b=50),
                    paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                    xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
                    yaxis=dict(gridcolor="#E8EDF5", zeroline=False, title=dict(text=y_name, font=dict(size=11))),
                    yaxis2=dict(
                        overlaying="y", side="right", zeroline=True, zerolinecolor="#94A3B8",
                        gridcolor="#E8EDF5", showgrid=False,
                        title=dict(text="Residual", font=dict(color="#94A3B8", size=11)),
                        tickfont=dict(color="#94A3B8"),
                    ),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                                font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
                                bordercolor="#E2E8F0", borderwidth=1),
                    font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
                )
                st.plotly_chart(fig_fit, use_container_width=True)

                # ── Summary stats + coefficient table (half-width, below chart) ─
                _TD  = "padding:3px 8px; font-size:11px; border-bottom:1px solid #E8EDF5"
                _TH  = f"{_TD}; background:#F8FAFC; font-weight:600; color:#475569"
                _HDR = f"background:#1B3A6B; color:#FFFFFF; font-size:11px; font-weight:700; padding:4px 8px"

                s_rows = [
                    ("Obs",     f"{res['n']:,}"),
                    ("R²",      f"{res['r2']:.4f}"),
                    ("Adj R²",  f"{res['adj_r2']:.4f}" if not np.isnan(res['adj_r2']) else "—"),
                    ("F-stat",  _fmt(res['f_stat'], ".2f")),
                    ("Prob(F)", _fmt(res['f_pval'], ".4f")),
                ]
                s_html  = f"<table style='border-collapse:collapse;width:100%'>"
                s_html += f"<tr><th style='{_HDR}'>Metric</th><th style='{_HDR}'>Value</th></tr>"
                for label, val in s_rows:
                    s_html += f"<tr><td style='{_TH}'>{label}</td><td style='{_TD}'>{val}</td></tr>"
                s_html += "</table>"

                def _p_color(p_val):
                    if p_val is None: return "#1A202C"
                    if p_val < 0.01:  return "#059669"
                    if p_val < 0.05:  return "#10B981"
                    if p_val < 0.10:  return "#D97706"
                    return "#DC2626"

                var_names = ["const"] + x_names
                c_html  = f"<table style='border-collapse:collapse;width:100%;margin-top:10px'>"
                c_html += f"<tr>{''.join(f'<th style={chr(39)}{_HDR}{chr(39)}>{h}</th>' for h in ['Var','β','SE','t','p',''])}</tr>"
                for j, vname in enumerate(var_names):
                    p_raw = res["p"][j]
                    p_val = float(p_raw) if not np.isnan(p_raw) else None
                    p_str = f"{p_val:.4f}" if p_val is not None else "—"
                    color = _p_color(p_val)
                    sig   = _sig(p_raw)
                    c_html += (
                        f"<tr>"
                        f"<td style='{_TH}'>{vname}</td>"
                        f"<td style='{_TD}'>{_fmt(res['betas'][j])}</td>"
                        f"<td style='{_TD}'>{_fmt(res['se'][j])}</td>"
                        f"<td style='{_TD}'>{_fmt(res['t'][j], '.3f')}</td>"
                        f"<td style='{_TD};color:{color};font-weight:600'>{p_str}</td>"
                        f"<td style='{_TD};color:{color}'>{sig}</td>"
                        f"</tr>"
                    )
                c_html += "</table>"
                c_html += "<p style='font-size:10px;color:#94A3B8;margin-top:4px'>*** p&lt;0.01 &nbsp; ** p&lt;0.05 &nbsp; * p&lt;0.10</p>"

                tbl_col, _ = st.columns([1, 1])
                with tbl_col:
                    st.markdown(s_html + c_html, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Regression error: {e}")
    else:
        st.caption("Add at least one X regressor to run the regression model.")

    # ── Save model ────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.form("az_save_form", clear_on_submit=True):
        save_c1, save_c2, save_c3 = st.columns([2, 3, 1])
        with save_c1:
            model_name = st.text_input("Model name",
                                       placeholder="e.g. SPX vs VIX hedge")
        with save_c2:
            notes = st.text_input("Notes (optional)",
                                  placeholder="e.g. VIX as hedge proxy for SPX drawdown")
        with save_c3:
            st.markdown("<div style='margin-top:26px'>", unsafe_allow_html=True)
            save_clicked = st.form_submit_button("💾 Save Model", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if save_clicked:
        model = {
            "saved_at":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            "model_name":  model_name.strip() or "(unnamed)",
            "series":      [{"name": nm, "key": ky, "invert": iv}
                            for nm, ky, _cl, iv in active],
            "timeframe":   tf,
            "start_date":  start_str,
            "end_date":    end_str,
            "r2":           round(reg_r2,           4) if reg_r2           is not None else None,
            "actual_last":  round(reg_actual_last,  2) if reg_actual_last  is not None else None,
            "fitted_last":  round(reg_fitted_last,  2) if reg_fitted_last  is not None else None,
            "resid_last":   round(reg_resid_last,   2) if reg_resid_last   is not None else None,
            "notes":       notes,
        }
        _save_model(model)
        st.success("Model saved.")


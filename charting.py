"""📈 Charting — ONE searchable home for every series the dashboard knows
(Rajat 2026-08-06: retires the old Charts & Indicators tab).

Registry = watchlist instruments (load_config) + curated market majors +
the whole Economic Data catalog (macro_econ) + the vol complex
(VIX/VVIX/VXN/SKEW/OVX/GVZ…) + every STIR contract in stir_bars.db +
every Commodities contract in commodities_bars.db + saved Custom Series.
The picker is a multiselect with built-in type-ahead search — type "cpi",
"gold", "sofr" or the name of a custom series and matches surface,
Bloomberg-style, no giant dropdowns as the registry grows.

Chart: up to 4 series, EACH ON ITS OWN AXIS (analyzer-style), window /
transform / rebase / log controls. Plotted series are stashed in session
state ready for the coming analysis tools (regression etc.).

The 🧪 Custom Series builder lives as a sibling sub-tab under 📈 Charting;
anything saved there shows up here as a "CUST · …" entry on the next rerun."""

import json
import math
import os
import pickle
import uuid
from datetime import date

import pandas as pd
import streamlit as st

import macro_econ as _me
from analyzer import (
    _VOL_TICKERS,
    _commod_contracts,
    _fetch_custom_series,
    _fetch_series,
    _stir_contracts,
    get_custom_series_list,
)
from watchlist import (ALPHAVANTAGE_FX_MAP, ECB_MAP, FRED_MAP, JGB_MAP,
                       load_config)

# ── Curated market majors (merged with the watchlist, deduped by ticker) ─────
_CURATED = [
    ("^GSPC", "S&P 500", "Equity"), ("^NDX", "Nasdaq 100", "Equity"),
    ("ES=F", "S&P 500 fut", "Equity"), ("NQ=F", "Nasdaq fut", "Equity"),
    ("RTY=F", "Russell 2000 fut", "Equity"), ("YM=F", "Dow fut", "Equity"),
    ("^STOXX50E", "EuroStoxx 50", "Equity"), ("^GDAXI", "DAX", "Equity"),
    ("^FTSE", "FTSE 100", "Equity"), ("^N225", "Nikkei 225", "Equity"),
    ("^SOX", "PHLX Semiconductor idx", "Equity"),
    ("SMH", "Semiconductor ETF (SMH)", "Equity"),
    ("^VIX", "VIX", "Vol"), ("^MOVE", "MOVE index", "Vol"),
    ("ZT=F", "UST 2Y fut", "Rate"), ("ZF=F", "UST 5Y fut", "Rate"),
    ("ZN=F", "UST 10Y fut", "Rate"), ("ZB=F", "UST Bond fut", "Rate"),
    ("UB=F", "UST Ultra fut", "Rate"),
    ("TLT", "20+yr Treasury ETF (TLT)", "Rate"),
    ("IEF", "7-10yr Treasury ETF (IEF)", "Rate"),
    ("SHY", "1-3yr Treasury ETF (SHY)", "Rate"),
    ("GC=F", "Gold", "Commodity"), ("SI=F", "Silver", "Commodity"),
    ("HG=F", "Copper", "Commodity"), ("CL=F", "WTI", "Commodity"),
    ("BZ=F", "Brent", "Commodity"), ("NG=F", "NatGas", "Commodity"),
    ("EURUSD=X", "EURUSD", "FX"), ("GBPUSD=X", "GBPUSD", "FX"),
    ("USDJPY=X", "USDJPY", "FX"), ("AUDUSD=X", "AUDUSD", "FX"),
    ("USDCHF=X", "USDCHF", "FX"), ("USDCAD=X", "USDCAD", "FX"),
    ("EURGBP=X", "EURGBP", "FX"), ("EURJPY=X", "EURJPY", "FX"),
    ("DX-Y.NYB", "Dollar index (DXY)", "FX"),
    ("BTC-USD", "Bitcoin", "Crypto"), ("ETH-USD", "Ethereum", "Crypto"),
]

_CLASS_PREFIX = {"Equity": "EQ", "Rate": "RATES", "Rates": "RATES",
                 "Bond": "RATES", "Commodity": "CMD", "FX": "FX",
                 "Currency": "FX", "Crypto": "CRY", "Vol": "VOL"}

# non-US sovereign curve tenors served by the Rates tab fetchers.
# EA (ECB AAA curve) was REPLACED as a display entry 2026-08-18 (Rajat:
# "instead add German, French and Italian yields") by the _CNBC_COUNTRY
# entries below — _curve_series("EA", …) still works as the DE splice base.
_CURVE_MATS = {
    "UK": ("UK", "gilt yield (BoE)", ["1Y", "2Y", "5Y", "10Y", "20Y", "30Y"]),
    "JP": ("JP", "JGB yield (MOF)",
           ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]),
    "AU": ("AU", "ACGB yield (RBA)", ["2Y", "3Y", "5Y", "10Y"]),
}

# country benchmark yields off the CNBC daily feed (same source as Core
# Markets). deep=True (Germany) back-extends with the ECB euro-area AAA
# curve — Bund-dominated, offset-anchored at the junction — for long
# windows; FR/IT stay CNBC-native (~2y): anchoring their credit spread
# onto an AAA curve would fabricate history.
_CNBC_COUNTRY = {
    "DE": ("German yield",  ["2Y", "5Y", "10Y", "30Y"], "-DE", True),
    "FR": ("French yield",  ["2Y", "5Y", "10Y", "30Y"], "-FR", False),
    "IT": ("Italian yield", ["2Y", "5Y", "10Y", "30Y"], "-IT", False),
}


def _wl_special(t: str) -> bool:
    """Watchlist tickers that are NOT plain Yahoo symbols — pseudo-tickers
    (^US2YT, ^ECB2Y…, ^JPY10Y, real yields, breakevens), the synthetic
    BBDXY basket, FRED-served oil and AlphaVantage FX. watchlist.
    fetch_chart_data resolves all of them via its own source maps, so
    Charting routes them through it instead of yfinance."""
    return (t == "BBDXY_SYNTH" or t in FRED_MAP or t in ECB_MAP
            or t in JGB_MAP or t in ALPHAVANTAGE_FX_MAP)


def _registry() -> dict:
    """{display_label: spec} — spec is {'src': 'yf', 'code': ticker},
    {'src': 'econ', 'espec': catalog_entry} or {'src': 'akey', 'code': key}
    (an analyzer key: stir:… / commod:… / custom:… fetched via
    analyzer._fetch_series). Built fresh each render (cheap), so a series
    saved in the Custom Series sub-tab appears here on the next rerun."""
    out = {}
    seen = set()
    try:
        for inst in load_config().get("instruments", []):
            t, n = inst.get("ticker"), inst.get("name") or inst.get("ticker")
            if not t or t in seen:
                continue
            seen.add(t)
            pfx = _CLASS_PREFIX.get(inst.get("class", ""), "MKT")
            if _wl_special(t):
                # pseudo-ticker → keep the name, drop the internal symbol
                out[f"{pfx} · {n}"] = {"src": "wl", "code": t}
            else:
                out[f"{pfx} · {n} ({t})"] = {"src": "yf", "code": t}
    except Exception:
        pass
    for t, n, cls in _CURATED:
        if t in seen:
            continue
        seen.add(t)
        out[f"{_CLASS_PREFIX.get(cls, 'MKT')} · {n} ({t})"] = \
            {"src": "yf", "code": t}
    _short = {"US": "US", "Euro Area": "EA", "UK": "UK", "Japan": "JP"}
    for rg, specs in _me._CATALOG.items():
        for s_ in specs:
            out[f"{_short.get(rg, rg)} · {s_['label']}"] = \
                {"src": "econ", "espec": s_}
    # vol complex (plain yfinance tickers; ^VIX / ^MOVE are already curated)
    for v in _VOL_TICKERS:
        t, n = v["ticker"], v["name"]
        if t in seen:
            continue
        seen.add(t)
        out[f"VOL · {n} ({t})"] = {"src": "yf", "code": t}
    # sovereign yield curves — US from FRED constant-maturity series (daily,
    # history to the 60s); EA/UK/JP/AU reuse the Rates tab fetchers
    # (ECB SDMX / BoE spot curve / MOF JGB / RBA F2)
    for mat in ["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y",
                "30Y"]:
        code = "DGS" + (mat[:-1] + "MO" if mat.endswith("M") else mat[:-1])
        out[f"RATES · US {mat} Treasury yield ({code})"] = {
            "src": "econ",
            "espec": {"label": f"US {mat} Treasury yield", "src": "fred",
                      "code": code, "transform": "level", "unit": "%"}}
    for cty, (disp, tag, mats) in _CURVE_MATS.items():
        for mat in mats:
            out[f"RATES · {disp} {mat} {tag}"] = \
                {"src": "curve", "cty": cty, "mat": mat}
    for cc, (tag, mats, sfx, deep) in _CNBC_COUNTRY.items():
        for mat in mats:
            out[f"RATES · {cc} {mat} {tag}"] = \
                {"src": "cnbcy", "sym": f"{cc}{mat}{sfx}", "mat": mat,
                 "deep": deep}
    # STIR / Commodities DB contracts + saved custom series (analyzer keys)
    try:
        for label, key in _stir_contracts():
            out[f"STIR · {label}"] = {"src": "akey", "code": key}
    except Exception:
        pass
    try:
        for label, key in _commod_contracts():
            out[f"CMD · {label}"] = {"src": "akey", "code": key}
    except Exception:
        pass
    try:
        for name, key in get_custom_series_list():
            out[f"CUST · {name}"] = {"src": "akey", "code": key}
    except Exception:
        pass
    return out


# ── Fetch (daily disk cache, reusing macro_econ's cache dir) ─────────────────
def _yf_series(tkr: str) -> pd.Series:
    ck = "yfd_" + tkr.replace("=", "_").replace("^", "_").replace("-", "_") \
        .replace(".", "_")
    hit = _me._cache_get(ck)
    if hit is not None:
        return hit
    import yfinance as yf
    h = yf.Ticker(tkr).history(period="max", auto_adjust=True)["Close"]
    if h.empty:
        raise ValueError(f"no history for {tkr}")
    s = pd.Series(h.values, index=pd.DatetimeIndex(
        [ts.tz_localize(None) if ts.tzinfo else ts for ts in h.index]))
    _me._cache_put(ck, s)
    return s


def _wl_series(tkr: str) -> pd.Series:
    """Special watchlist instruments via watchlist.fetch_chart_data (which
    resolves FRED / ECB / JGB / BBDXY / AlphaVantage), daily disk cache."""
    ck = "wld_" + "".join(c if c.isalnum() else "_" for c in tkr)
    hit = _me._cache_get(ck)
    if hit is not None:
        return hit
    from watchlist import fetch_chart_data
    df = fetch_chart_data(tkr, "max", "1d")
    if df is None or df.empty:
        raise ValueError(f"no history for {tkr}")
    c = df["Close"]
    s = (c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c) \
        .astype(float).dropna()
    s.index = pd.DatetimeIndex(s.index)
    _me._cache_put(ck, s)
    return s


def _curve_series(cty: str, mat: str) -> pd.Series:
    """One tenor of a non-US sovereign curve, via the Rates tab fetchers
    (their st.cache_data gives 1-2h in-session caching; a daily disk cache
    on top keeps cross-session loads instant). The whole curve is disk-
    cached in one shot — sibling tenors then load without refetching."""
    ck = f"curve_{cty}_{mat}"
    hit = _me._cache_get(ck)
    if hit is not None:
        return hit
    import rates as _rt
    if cty == "EA":
        df = _rt.fetch_ecb_curve("2004-09-06")     # ECB YC inception
    elif cty == "UK":
        df = _rt.fetch_uk_gilts("2016-01-01")      # pulls both BoE files
    elif cty == "JP":
        df = _rt.fetch_japan_jgb()
    else:
        df = _rt.fetch_australia_bonds()
    if df.empty:
        raise ValueError(f"no {cty} curve data")
    for c in df.columns:
        sc = pd.Series(pd.to_numeric(df[c].values, errors="coerce"),
                       index=pd.DatetimeIndex(df.index)).dropna()
        if not sc.empty:
            _me._cache_put(f"curve_{cty}_{c}", sc)
    hit = _me._cache_get(ck)
    if hit is None:
        raise ValueError(f"no {cty} {mat} curve data")
    return hit


def get_series(spec: dict) -> tuple:
    """(raw_series, natural_unit) for a registry spec."""
    if spec["src"] == "yf":
        return _yf_series(spec["code"]), "px"
    if spec["src"] == "wl":
        t = spec["code"]
        pct = (t in ECB_MAP or t in JGB_MAP
               or (t in FRED_MAP and not FRED_MAP[t].startswith("DCOIL")))
        return _wl_series(t), "%" if pct else "px"
    if spec["src"] == "curve":
        s = _curve_series(spec["cty"], spec["mat"])
        if spec["cty"] == "UK":
            s = _uk_recent_splice(s, spec["mat"])
        return s, "%"
    if spec["src"] == "cnbcy":
        return _cnbc_country_series(spec["sym"], spec["mat"],
                                    spec.get("deep", False)), "%"
    if spec["src"] == "akey":
        # STIR / Commodities DB contracts and custom series, via the analyzer.
        # "max" maps to _TF_DAYS["max"] so the DB trim never clips the history;
        # Charting applies its own window cutoff afterwards.
        return _fetch_series(spec["code"], "max"), "px"
    e = spec["espec"]
    return _me.fetch_series(e["src"], e["code"]), e.get("unit", "")


# ── Registry as an Analyzer picker (Rajat 2026-08-19: the Analyzer moved
# under 📈 Charting and now searches this same registry) ────────────────────
def registry_options() -> dict:
    """{display_label: analyzer_key} for every registry entry — the shared
    'comprehensive database' picker. yf specs -> raw ticker; akey specs pass
    through (stir:/commod:/custom:); every other spec (econ / wl / curve /
    cnbcy) -> 'econ:<label>' resolved back through this registry by
    fetch_econ_series()."""
    out = {}
    for lbl, spec in _registry().items():
        if spec["src"] in ("yf", "akey"):
            out[lbl] = spec["code"]
        else:
            out[lbl] = "econ:" + lbl
    return out


def fetch_econ_series(label, period="max", start=None, end=None):
    """Fetch a non-yf/non-akey registry series by its label (econ catalog,
    watchlist pseudo-tickers, sovereign curves), trimmed to start/end or the
    requested period. Returns an empty Series if the label is unknown."""
    spec = _registry().get(label)
    if spec is None or spec["src"] in ("yf", "akey"):
        return pd.Series(dtype=float)
    try:
        s, _u = get_series(spec)
    except Exception:
        return pd.Series(dtype=float)
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s = s.dropna()
    s.index = pd.DatetimeIndex(s.index)
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    elif period not in (None, "max"):
        from analyzer import _TF_DAYS
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(
            days=_TF_DAYS.get(period, 365))
        s = s[s.index >= cutoff]
    if end:
        s = s[s.index <= pd.Timestamp(end)]
    return s


# BoE publishes its fitted gilt curve with a ~3-week lag, so UK curve plots
# stopped weeks short of today (Rajat 2026-08-14). Extend the tail with the
# near-real-time CNBC benchmark gilt yields (same feed as Core Markets),
# level-anchored at the junction — fitted-spot vs benchmark constructs sit
# a few bp apart, so the CNBC tail is shifted by the junction-day offset.
_UK_CNBC = {"2Y": "UK2Y-GB", "5Y": "UK5Y-GB",
            "10Y": "UK10Y-GB", "30Y": "UK30Y-GB"}


def _uk_recent_splice(s: pd.Series, mat: str) -> pd.Series:
    sym = _UK_CNBC.get(mat)
    if not sym or s.empty:
        return s
    try:
        from core_markets import _yld_hist
        h = _yld_hist(sym)
    except Exception:
        return s
    if not h:
        return s
    last_d = s.index.max()
    tail = [(d, v) for d, v in h if pd.Timestamp(d) > last_d]
    if not tail:
        return s
    overlap = [v for d, v in h if pd.Timestamp(d) <= last_d]
    off = (float(s.iloc[-1]) - overlap[-1]) if overlap else 0.0
    ext = pd.Series([v + off for _d, v in tail],
                    index=pd.DatetimeIndex([pd.Timestamp(d)
                                            for d, _v in tail]))
    return pd.concat([s, ext]).sort_index()


@st.cache_data(ttl=86400, show_spinner=False)
def _cnbc_range_hist_c(sym: str, rng: str):
    """[(date, value)] from a CNBC chart range. Undocumented depths (probed
    2026-08-18): '1Y'≈2y daily, '5Y'≈10y WEEKLY, 'ALL'≈1994→ quarterly.
    Raises on failure (never caches empties)."""
    from core_markets import _cnbc_bars
    rows = _cnbc_bars(sym, rng)
    if not rows:
        raise ValueError(f"no {rng} bars for {sym}")
    out = {}
    for t, v in rows:
        out[t.date()] = v
    return sorted(out.items())


def _cnbc_range_hist(sym: str, rng: str):
    try:
        return _cnbc_range_hist_c(sym, rng)
    except Exception:
        return None


def _cnbc_country_series(sym: str, mat: str, deep: bool = False) -> pd.Series:
    """Country benchmark yields, layered by resolution — all GENUINE
    country data from CNBC's own ranges: quarterly from ~1994 ('ALL'),
    weekly from ~2016 ('5Y'), daily last ~2y ('1Y'); finer layers override
    coarser on shared dates. deep=True (Germany) additionally overlays the
    ECB euro-area AAA curve's DAILY history for 2004→ (Bund-dominated,
    offset-anchored at the daily junction) since daily proxy beats weekly
    genuine for charting resolution there."""
    from core_markets import _yld_hist
    daily = _yld_hist(sym)
    if not daily:
        raise ValueError(f"no CNBC history for {sym}")
    merged: dict = {}
    for rng in ("ALL", "5Y"):
        layer = _cnbc_range_hist(sym, rng)
        if layer:
            merged.update(dict(layer))
    if deep:
        try:
            base = _curve_series("EA", mat)
        except Exception:
            base = None
        if base is not None and len(base):
            first = pd.Timestamp(daily[0][0])
            ref = base[base.index <= first]
            if len(ref):
                off = float(daily[0][1]) - float(ref.iloc[-1])
                for d, v in base.items():
                    if d < first:
                        merged[d.date()] = float(v) + off
    merged.update(dict(daily))             # native daily always wins
    items = sorted(merged.items())
    return pd.Series([v for _d, v in items],
                     index=pd.DatetimeIndex([pd.Timestamp(d)
                                             for d, _v in items]))


def _transform(s: pd.Series, spec: dict, mode: str, nat: str = "px") -> tuple:
    """Frequency-aware transform → (series, unit_label). `nat` is the
    series' natural unit, shown when mode leaves levels untouched."""
    if spec["src"] == "econ":
        return _me._apply_transform(s, spec["espec"],
                                    "auto" if mode == "auto" else mode)
    # market series: daily
    gap = s.index.to_series().diff().dt.days.median() or 1
    py = 252 if gap <= 4 else (12 if gap <= 45 else 4)
    pm = 21 if gap <= 4 else 1
    if mode == "yoy":
        return ((s / s.shift(py) - 1) * 100).dropna(), "% YoY"
    if mode == "mom":
        return ((s / s.shift(pm) - 1) * 100).dropna(), "% 1m"
    if mode == "diff":
        return s.diff().dropna(), "Δ"
    return s.dropna(), nat


_PALETTE = ["#2563EB", "#0D9488", "#B45309", "#7C3AED", "#DC2626", "#64748B"]


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI (EWM with alpha=1/n)."""
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + up / dn)).dropna()


# ── Chart templates (Rajat 2026-08-14): a template = selected series +
# global controls (transform/window/rebase/log) + every per-series ⚙ setting
# (colour/width/style/invert/log/multiplier/MVA config/RSI panel). Stored in
# chart_templates.json; loadable from the picker at the bottom of the tab. ──
_TPL_FP = os.path.join(os.path.dirname(__file__), "chart_templates.json")
_CHS_KEYS = ("inv", "log", "mult", "col", "w", "ls", "op",
             "mva", "mvam", "rsi", "rsip")


def _load_templates() -> list:
    try:
        with open(_TPL_FP) as fh:
            return json.load(fh)
    except Exception:
        return []


def _save_templates(items: list) -> None:
    with open(_TPL_FP, "w") as fh:
        json.dump(items, fh, indent=1)


def _capture_template(name: str, sel: list) -> dict:
    ss = st.session_state
    series = {}
    for lbl in sel:
        cfg = {}
        for k in _CHS_KEYS:
            v = ss.get(f"_chs_{k}_{lbl}")
            if v is not None:
                cfg[k] = v
        series[lbl] = cfg
    return {"id": uuid.uuid4().hex[:8], "name": name, "sel": list(sel),
            "tf": ss.get("_ch_tf", "auto"), "yrs": ss.get("_ch_yrs", "5y"),
            "rebase": bool(ss.get("_ch_rebase")),
            "logy": bool(ss.get("_ch_log")), "series": series}


def _apply_template(tpl: dict, valid_labels: set) -> None:
    """on_click callback — runs BEFORE the rerun renders widgets, so the
    state lands cleanly. Labels no longer in the registry are dropped."""
    ss = st.session_state
    sel = [l for l in tpl.get("sel", []) if l in valid_labels]
    missing = [l for l in tpl.get("sel", []) if l not in valid_labels]
    ss["_ch_sel"] = sel
    ss["_ch_tf"] = tpl.get("tf", "auto")
    ss["_ch_yrs"] = tpl.get("yrs", "5y")
    ss["_ch_rebase"] = bool(tpl.get("rebase"))
    ss["_ch_log"] = bool(tpl.get("logy"))
    for lbl, cfg in tpl.get("series", {}).items():
        if lbl not in valid_labels:
            continue
        for k in _CHS_KEYS:
            if k in cfg:
                ss[f"_chs_{k}_{lbl}"] = cfg[k]
    ss["_ch_tpl_missing"] = missing


def _templates_ui(reg: dict, sel: list) -> None:
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns([1.7, 1.1, 1.8, 0.7, 0.7])
    nm = c1.text_input("Template name", key="_ch_tpl_name",
                       placeholder="name this chart setup…")
    c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c2.button("💾 Save template", key="_ch_tpl_save", disabled=not sel,
                 use_container_width=True,
                 help="saves the current series, colours, transforms, MVA/RSI "
                      "settings and window as a reusable template"):
        name = (nm or "").strip() or f"template {len(_load_templates()) + 1}"
        items = _load_templates()
        items.append(_capture_template(name, sel))
        _save_templates(items)
        st.success(f"Saved template **{name}**.")
    tpls = _load_templates()
    if tpls:
        idx = c3.selectbox("Saved templates", range(len(tpls)),
                           format_func=lambda i: tpls[i]["name"],
                           key="_ch_tpl_sel")
        c4.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        c4.button("Load", key="_ch_tpl_load", use_container_width=True,
                  on_click=_apply_template, args=(tpls[idx], set(reg)))
        c5.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if c5.button("🗑", key="_ch_tpl_del", use_container_width=True,
                     help="delete the selected template"):
            tpls.pop(idx)
            _save_templates(tpls)
            st.rerun()
    miss = st.session_state.pop("_ch_tpl_missing", None)
    if miss:
        st.caption("⚠ dropped from template (no longer in registry): "
                   + ", ".join(miss))


def _short_lbl(lbl: str, cap: int = 18) -> str:
    """Registry label → compact button caption (drop 'PFX · ' and '(ticker)')."""
    t = lbl.split(" · ", 1)[-1].strip()
    if t.endswith(")") and "(" in t:
        t = t[:t.rfind("(")].strip()
    return t if len(t) <= cap else t[:cap - 1] + "…"


def render_charting():
    st.markdown(
        "<div style='background:#1E293B;color:#F8FAFC;padding:6px 12px;"
        "font-size:13px;font-weight:700;border-radius:6px;display:inline-block;"
        "margin-bottom:6px'>📈 Charting"
        "&nbsp;&nbsp;<span style='font-weight:400;font-size:11px;color:#94A3B8'>"
        "every dashboard series in one place — type to search "
        "(EQ/RATES/FX/CMD/VOL/CRY + sovereign yield curves US/EA/UK/JP/AU + "
        "US/EA/UK/JP econ + STIR &amp; CMD contracts + CUST custom series)"
        "</span></div>",
        unsafe_allow_html=True)
    reg = _registry()
    c1, c2, c3 = st.columns([3.4, 0.8, 0.7])
    sel = c1.multiselect(
        f"Series — type to search ({len(reg)} available)", list(reg),
        default=[], key="_ch_sel", max_selections=4,
        placeholder="e.g. 'gold', 'cpi', '10y', 'eurusd'…")
    tf = c2.selectbox("Transform", ["auto", "level", "yoy", "mom", "diff"],
                      key="_ch_tf")
    yrs = c3.selectbox("Window", ["6m", "1y", "2y", "3y", "5y", "10y", "20y",
                                  "max"], index=4, key="_ch_yrs")
    def _clear_chart():
        ss = st.session_state
        for lbl in ss.get("_ch_sel", []):
            for k in _CHS_KEYS:
                ss.pop(f"_chs_{k}_{lbl}", None)
        ss["_ch_sel"] = []

    cc1, cc2, cc3, _sp = st.columns([1.2, 0.8, 0.8, 2.2])
    rebase = cc1.checkbox("rebase to 100", key="_ch_rebase")
    logy = cc2.checkbox("log scale", key="_ch_log")
    cc3.button("🧹 Clear", key="_ch_clear", on_click=_clear_chart,
               use_container_width=True,
               help="clear the selected series and their ⚙ settings")

    if not sel:
        st.info("Pick up to 4 series — the search box matches anywhere in the "
                "name (ticker, class prefix, region) — or load a saved "
                "template below.")
        _templates_ui(reg, sel)
        return

    # ⚙ per-series styling & transforms (session-state only — nothing
    # persisted to disk)
    for i, (cx, lbl) in enumerate(zip(st.columns(len(sel)), sel)):
        with cx.popover("⚙ " + _short_lbl(lbl), use_container_width=True):
            p1, p2 = st.columns(2)
            p1.checkbox("Invert", key=f"_chs_inv_{lbl}")
            p2.checkbox("Log axis", key=f"_chs_log_{lbl}",
                        help="applies when this series has its own axis "
                             "(multi-axis chart, or a single series); on a "
                             "shared/rebased axis use the global log toggle")
            st.number_input("Multiplier", value=1.0, step=0.5, format="%g",
                            key=f"_chs_mult_{lbl}",
                            help="scale the series, e.g. 100, 0.01 or -1")
            st.color_picker("Colour", _PALETTE[i % len(_PALETTE)],
                            key=f"_chs_col_{lbl}")
            st.slider("Width", 0.5, 5.0, 1.8, 0.5, key=f"_chs_w_{lbl}")
            st.slider("Opacity", 0.2, 1.0, 1.0, 0.1, key=f"_chs_op_{lbl}")
            st.selectbox("Line style", ["solid", "dash", "dot", "dashdot"],
                         key=f"_chs_ls_{lbl}")
            m1, m2 = st.columns(2)
            m1.number_input("MVA (bars, 0=off)", 0, 500, 0, 5,
                            key=f"_chs_mva_{lbl}")
            m2.selectbox("MVA display",
                         ["series + MVA", "MVA only", "diff from MVA",
                          "% diff from MVA"], key=f"_chs_mvam_{lbl}")
            r1, r2 = st.columns(2)
            r1.checkbox("RSI panel", key=f"_chs_rsi_{lbl}",
                        help="adds an RSI sub-chart below the main chart")
            r2.number_input("RSI period", 2, 100, 14, 1,
                            key=f"_chs_rsip_{lbl}")

    import numpy as _np
    import plotly.graph_objects as go
    fig = go.Figure()
    if yrs == "max":
        cutoff = None
    elif yrs.endswith("m"):
        cutoff = pd.Timestamp.today() - pd.DateOffset(months=int(yrs[:-1]))
    else:
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=int(yrs[:-1]))
    rows = []
    rsi_panel = []   # (lbl, period, rsi_series) → sub-chart below the main one
    for lbl in sel:
        spec = reg[lbl]
        try:
            with st.spinner(f"Loading {lbl}…"):
                raw, nat_u = get_series(spec)
        except Exception as ex:
            st.caption(f"⚠ {lbl}: {type(ex).__name__} {str(ex)[:90]}")
            continue
        s, unit = _transform(raw, spec, tf, nat_u)
        if cutoff is not None:
            s = s[s.index >= cutoff]
        mult = float(st.session_state.get(f"_chs_mult_{lbl}") or 1.0)
        if mult != 1.0:
            s = s * mult
            unit = f"{unit} ×{mult:g}" if unit else f"×{mult:g}"
        s = s.dropna()
        if s.empty:
            continue
        if st.session_state.get(f"_chs_rsi_{lbl}"):
            p = int(st.session_state.get(f"_chs_rsip_{lbl}", 14) or 14)
            rsi_panel.append((lbl, p, _rsi(s, p)))
        # MVA — overlay mode keeps the series and adds a dotted MVA later
        # (computed on the *displayed* values, i.e. after invert/rebase);
        # the other modes replace the series here
        mvw = int(st.session_state.get(f"_chs_mva_{lbl}") or 0)
        mmode = st.session_state.get(f"_chs_mvam_{lbl}", "series + MVA")
        sfx = ""
        if mvw >= 2:
            ma = s.rolling(mvw).mean()
            if mmode == "MVA only":
                s, sfx = ma.dropna(), f" MVA{mvw}"
            elif mmode == "diff from MVA":
                s, unit, sfx = (s - ma).dropna(), f"Δ vs MVA{mvw}", \
                    f" ΔMVA{mvw}"
            elif mmode == "% diff from MVA":
                s, unit, sfx = ((s / ma - 1) * 100).dropna(), \
                    f"% vs MVA{mvw}", f" %MVA{mvw}"
        if s.empty:
            continue
        overlay = mvw if (mvw >= 2 and mmode == "series + MVA") else 0
        rows.append((lbl, s, unit, overlay, sfx))

    # ── zoom slider (both ends draggable) — trims the data itself, so every
    # axis re-autoranges and rebase/MVA/RSI recompute on the zoomed window.
    # Key includes window+selection so the range resets when either changes.
    if rows:
        dmin = min(r[1].index.min() for r in rows).date()
        dmax = max(r[1].index.max() for r in rows).date()
        if dmin < dmax:
            zkey = f"_ch_zoom_{yrs}_{abs(hash(tuple(sel))) % 100000}"
            z0, z1 = st.slider("Zoom", min_value=dmin, max_value=dmax,
                               value=(dmin, dmax), key=zkey,
                               format="DD MMM YY",
                               label_visibility="collapsed")
            if (z0, z1) != (dmin, dmax):
                t0, t1 = pd.Timestamp(z0), pd.Timestamp(z1)
                rows = [(lbl, s[(s.index >= t0) & (s.index <= t1)],
                         unit, mvw, sfx) for lbl, s, unit, mvw, sfx in rows]
                rows = [r for r in rows if not r[1].empty]
                rsi_panel = [(lbl, p, r[(r.index >= t0) & (r.index <= t1)])
                             for lbl, p, r in rsi_panel]

    _yref = {0: "y", 1: "y2", 2: "y3", 3: "y4"}
    _ykey = {0: "yaxis", 1: "yaxis2", 2: "yaxis3", 3: "yaxis4"}
    _axpos = [dict(side="left"),
              dict(side="right", overlaying="y"),
              dict(side="left", overlaying="y", anchor="free", position=0.0),
              dict(side="right", overlaying="y", anchor="free", position=1.0)]
    n = len(rows)
    multi_ax = n > 1 and not rebase
    # apply the ⚙ styles — invert flips the axis when the series owns one,
    # otherwise (single axis / rebase) it flips the data instead
    plotted = []
    for i, (lbl, s, unit, mvw, sfx) in enumerate(rows):
        inv = bool(st.session_state.get(f"_chs_inv_{lbl}"))
        if inv and not multi_ax:
            s = -s
        if rebase:
            base = s.iloc[0]
            s = (s / base * 100) if base else s
            if inv:
                s = 200 - s          # mirror about 100 (rebase undoes -s)
            unit = "idx=100"
        ls = st.session_state.get(f"_chs_ls_{lbl}", "solid")
        ma = s.rolling(mvw).mean().dropna() if mvw else None
        plotted.append((
            lbl, s, unit,
            st.session_state.get(f"_chs_col_{lbl}", _PALETTE[i % len(_PALETTE)]),
            float(st.session_state.get(f"_chs_w_{lbl}", 1.8)),
            None if ls == "solid" else ls, inv, ma, mvw, sfx))
    for i, (lbl, s, unit, col, wid, dash, inv, ma, mvw, sfx) in \
            enumerate(plotted):
        nm = lbl + sfx + (" (inv)" if inv else "")
        op = float(st.session_state.get(f"_chs_op_{lbl}", 1.0))
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=nm, opacity=op,
            yaxis=_yref[i] if multi_ax else "y",
            line=dict(color=col, width=wid, dash=dash),
            hovertemplate="%{x|%d %b %y} · %{y:,.4g}<extra>" + nm + "</extra>"))
        if ma is not None and not ma.empty:
            mnm = f"{_short_lbl(lbl)} MVA{mvw}"
            fig.add_trace(go.Scatter(
                x=ma.index, y=ma.values, mode="lines", name=mnm, opacity=op,
                yaxis=_yref[i] if multi_ax else "y",
                line=dict(color=col, width=max(0.8, wid * 0.7), dash="dot"),
                hovertemplate="%{x|%d %b %y} · %{y:,.4g}<extra>" + mnm
                              + "</extra>"))
    if not fig.data:
        st.warning("nothing plottable for that selection/window")
        return
    if n <= 1 or rebase:
        domain, margin = [0.0, 1.0], dict(l=30, r=30, t=30, b=30)
    elif n == 2:
        domain, margin = [0.0, 0.94], dict(l=30, r=55, t=30, b=30)
    else:
        domain, margin = [0.07, 0.93], dict(l=60, r=60, t=30, b=30)
    layout = dict(
        height=520, margin=margin, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1,
                    xanchor="right", font_size=11),
        xaxis=dict(gridcolor="#F1F5F9", domain=domain),
        hovermode="x unified", plot_bgcolor="#FFFFFF")
    if multi_ax:
        for i, (lbl, _s, unit, col, _w, _d, inv, _ma, _mw, _sx) in \
                enumerate(plotted):
            ttl = (unit or lbl) + (" (inv)" if inv else "")
            at = "log" if (logy or st.session_state.get(f"_chs_log_{lbl}")) \
                else "linear"
            ax = dict(title=dict(text=ttl, font=dict(color=col, size=11)),
                      tickfont=dict(color=col), gridcolor="#F1F5F9",
                      zeroline=False, showgrid=(i == 0), type=at)
            if inv:
                ax["autorange"] = "reversed"
            ax.update(_axpos[i])
            layout[_ykey[i]] = ax
    else:
        # single shared axis: the per-series log toggle still counts when
        # there is only one series (it owns the axis)
        _logy = logy or (len(plotted) == 1 and
                         st.session_state.get(f"_chs_log_{plotted[0][0]}"))
        layout["yaxis"] = dict(gridcolor="#F1F5F9",
                               type="log" if _logy else "linear",
                               title=plotted[0][2] if plotted else "")
        if not _logy:
            _allv = _np.concatenate([_np.asarray(t.y, dtype=float)
                                     for t in fig.data])
            _ref = 100 if rebase else 0
            if _np.nanmin(_allv) <= _ref <= _np.nanmax(_allv):
                fig.add_hline(y=_ref, line_color="#CBD5E1", line_width=1)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # RSI sub-chart — one shared panel, a line per series that asked for it
    if rsi_panel:
        colmap = {lbl: col for lbl, _s, _u, col, *_r in plotted}
        rfig = go.Figure()
        for lbl, p, r in rsi_panel:
            if r.empty:
                continue
            rnm = f"{_short_lbl(lbl)} RSI{p}"
            rfig.add_trace(go.Scatter(
                x=r.index, y=r.values, mode="lines", name=rnm,
                line=dict(color=colmap.get(lbl, "#64748B"), width=1.4),
                hovertemplate="%{x|%d %b %y} · %{y:.1f}<extra>" + rnm
                              + "</extra>"))
        if rfig.data:
            rfig.add_hline(y=70, line_color="#DC2626", line_width=1,
                           line_dash="dot")
            rfig.add_hline(y=30, line_color="#0D9488", line_width=1,
                           line_dash="dot")
            rfig.add_hline(y=50, line_color="#CBD5E1", line_width=1)
            rfig.update_layout(
                height=210, margin=dict(l=30, r=30, t=10, b=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1,
                            xanchor="right", font_size=10),
                xaxis=dict(gridcolor="#F1F5F9", domain=domain),
                yaxis=dict(range=[0, 100], gridcolor="#F1F5F9", title="RSI",
                           tickvals=[0, 30, 50, 70, 100]),
                hovermode="x unified", plot_bgcolor="#FFFFFF")
            st.plotly_chart(rfig, use_container_width=True)

    # keep the plotted set for the coming analysis tools (regression etc.)
    st.session_state["_ch_plotted"] = [(lbl, s) for lbl, s, *_r in plotted]
    st.caption(
        "Market series (incl. the VOL complex) = yfinance daily closes "
        "(auto-adjusted, cached daily); economic series = the Economic Data "
        "engine (FRED / DBnomics); RATES · sovereign yields = FRED DGS (US), "
        "CNBC benchmark bonds for DE/FR/IT (Bund/OAT/BTP — DE back-extended "
        "with the ECB euro-area AAA curve, offset-anchored; FR/IT ~2y "
        "native), BoE spot curve (UK, ~3wk lag + CNBC splice), MOF (JP), "
        "RBA F2 (AU, ~5d lag); STIR · / CMD · "
        "dated contracts come from stir_bars.db / commodities_bars.db; "
        "CUST · series are the formulas saved in the 🧪 Custom Series "
        "sub-tab. Each series gets its own axis "
        "unless rebased. The ⚙ row sets colour / width / line style / invert / "
        "multiplier / log / MVA / RSI per series (invert & log act on that "
        "series' axis when it owns one, so hover still shows true values; on "
        "a shared or rebased axis invert flips the line and log needs the "
        "global toggle). MVA display: overlay the dotted MVA on the series, "
        "show the MVA alone, or show the gap to the MVA in units or %. RSI "
        "(Wilder) draws in a shared panel below the chart. Analysis tools "
        "(regression, spreads, correlations) plug in here next — the plotted "
        "set is already kept for them.")

    _templates_ui(reg, sel)

"""🌍 Economic Data — macro series catalog + charting (Macro tab, Rajat 2026-08-06).

Source strategy ("cheap long-term histories"):
- **US: FRED** (free key in secrets, 50-80y histories). NB: true ISM PMI is NOT
  freely available anywhere (ISM licensing — pulled from FRED in 2016; the
  DBnomics ISM mirror is stale garbage) → Philly Fed / Empire State surveys
  serve as free PMI-proxies.
- **Euro area: DBnomics → Eurostat** (free, no key, ~25y histories).
- UK / Japan: planned (DBnomics ONS/BOJ/OECD — same fetcher will serve).

Per-series disk cache (macro_cache/, per calendar day). Chart: house-style
plotly, multi-select with optional rebase-to-100; latest-values table per region."""

import math
import os
import pickle
from datetime import date

import pandas as pd
import streamlit as st

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "macro_cache")

# transform: how the RAW series becomes the headline number
#   level | yoy (% vs 12m ago) | diff (Δ vs prior obs) | qoq (already % in source)
_CATALOG = {
    "US": [
        {"label": "Nonfarm Payrolls (Δ, k)", "src": "fred", "code": "PAYEMS",
         "transform": "diff", "unit": "k jobs"},
        {"label": "Unemployment Rate", "src": "fred", "code": "UNRATE",
         "transform": "level", "unit": "%"},
        {"label": "Initial Claims 4wk avg (k)", "src": "fred", "code": "IC4WSA",
         "transform": "level", "unit": "k", "scale": 0.001},
        {"label": "JOLTS Job Openings (m)", "src": "fred", "code": "JTSJOL",
         "transform": "level", "unit": "m", "scale": 0.001},
        {"label": "CPI YoY", "src": "fred", "code": "CPIAUCSL",
         "transform": "yoy", "unit": "%"},
        {"label": "Core CPI YoY", "src": "fred", "code": "CPILFESL",
         "transform": "yoy", "unit": "%"},
        {"label": "Core PCE YoY", "src": "fred", "code": "PCEPILFE",
         "transform": "yoy", "unit": "%"},
        {"label": "Retail Sales YoY", "src": "fred", "code": "RSAFS",
         "transform": "yoy", "unit": "%"},
        {"label": "Industrial Production YoY", "src": "fred", "code": "INDPRO",
         "transform": "yoy", "unit": "%"},
        {"label": "GDP QoQ ann.", "src": "fred", "code": "A191RL1Q225SBEA",
         "transform": "level", "unit": "%"},
        {"label": "Housing Starts (m, saar)", "src": "fred", "code": "HOUST",
         "transform": "level", "unit": "m", "scale": 0.001},
        {"label": "UMich Consumer Sentiment", "src": "fred", "code": "UMCSENT",
         "transform": "level", "unit": "idx"},
        {"label": "Philly Fed Mfg (PMI-proxy)", "src": "fred",
         "code": "GACDFSA066MSFRBPHI", "transform": "level", "unit": "idx"},
        {"label": "Empire State Mfg (PMI-proxy)", "src": "fred",
         "code": "GACDISA066MSFRBNY", "transform": "level", "unit": "idx"},
    ],
    "Euro Area": [
        {"label": "Unemployment Rate", "src": "dbn",
         "code": "Eurostat/une_rt_m/M.SA.TOTAL.PC_ACT.T.EA20",
         "transform": "level", "unit": "%"},
        {"label": "HICP YoY", "src": "dbn",
         "code": "Eurostat/prc_hicp_manr/M.RCH_A.CP00.EA20",
         "transform": "level", "unit": "%"},
        {"label": "Core HICP YoY", "src": "dbn",
         "code": "Eurostat/prc_hicp_manr/M.RCH_A.TOT_X_NRG_FOOD.EA20",
         "transform": "level", "unit": "%"},
        {"label": "Industrial Production YoY", "src": "dbn",
         "code": "Eurostat/STS_INPR_M/M.PRD.B-D.CA.PCH_SM.EA20",
         "transform": "level", "unit": "%"},
        {"label": "Retail Trade YoY", "src": "dbn",
         "code": "Eurostat/sts_trtu_m/M.VOL_SLS.G47.CA.PCH_SM.EA20",
         "transform": "level", "unit": "%"},
        {"label": "GDP QoQ", "src": "dbn",
         "code": "Eurostat/namq_10_gdp/Q.CLV_PCH_PRE.SCA.B1GQ.EA20",
         "transform": "level", "unit": "%"},
        {"label": "Economic Sentiment ESI (PMI-proxy)", "src": "dbn",
         "code": "Eurostat/ei_bssi_m_r2/M.BS-ESI-I.SA.EA20",
         "transform": "level", "unit": "idx"},
    ],
    # UK: ONS via DBnomics — NB series codes need the frequency SUFFIX
    # (MGSX.M etc.; the bare code silently returns an annual mangle)
    "UK": [
        {"label": "Unemployment Rate", "src": "dbn",
         "code": "ONS/LMS/MGSX.M", "transform": "level", "unit": "%"},
        {"label": "CPI YoY", "src": "dbn",
         "code": "ONS/MM23/D7G7.M", "transform": "level", "unit": "%"},
        {"label": "Retail Sales YoY", "src": "dbn",
         "code": "ONS/DRSI/J5EK.M", "transform": "yoy", "unit": "%"},
        {"label": "GDP QoQ", "src": "dbn",
         "code": "ONS/PN2/IHYQ.Q", "transform": "level", "unit": "%"},
    ],
    # Japan: IMF via DBnomics (BOJ Tankan not mirrored; IFS industrial
    # production discontinued 2023 — skipped). IMF mirror lags ~6-12m.
    "Japan": [
        {"label": "Unemployment Rate", "src": "dbn",
         "code": "IMF/IFS/M.JP.LUR_PT", "transform": "level", "unit": "%"},
        {"label": "CPI YoY", "src": "dbn",
         "code": "IMF/CPI/M.JP.PCPI_IX", "transform": "yoy", "unit": "%"},
        {"label": "GDP QoQ (real)", "src": "dbn",
         "code": "IMF/IFS/Q.JP.NGDP_R_SA_XDC", "transform": "mom", "unit": "%"},
    ],
}


# ── Fetchers (disk-cached per day) ───────────────────────────────────────────
def _cache_get(key: str):
    fp = os.path.join(_CACHE_DIR, f"{key}_{date.today().isoformat()}.pkl")
    if os.path.exists(fp):
        try:
            with open(fp, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    return None


def _cache_put(key: str, s: pd.Series):
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        fp = os.path.join(_CACHE_DIR, f"{key}_{date.today().isoformat()}.pkl")
        with open(fp, "wb") as fh:
            pickle.dump(s, fh)
    except Exception:
        pass


def _fred_key():
    try:
        return st.secrets["FRED_KEY"]
    except Exception:
        try:
            import toml
            p = os.path.join(os.path.dirname(__file__), ".streamlit",
                             "secrets.toml")
            if os.path.exists(p):
                return toml.load(p).get("FRED_KEY")
        except Exception:
            pass
    return None


def fetch_series(src: str, code: str) -> pd.Series:
    """Full history as a pd.Series indexed by date. Raises on failure."""
    ck = (src + "_" + code).replace("/", "_").replace(".", "_")
    hit = _cache_get(ck)
    if hit is not None:
        return hit
    import requests
    if src == "fred":
        r = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": code, "api_key": _fred_key(),
                    "file_type": "json"}, timeout=30)
        obs = r.json().get("observations", [])
        s = pd.Series(
            {pd.Timestamp(o["date"]): float(o["value"])
             for o in obs if o["value"] not in (".", "", None)}).sort_index()
    elif src == "dbn":
        r = requests.get(
            f"https://api.db.nomics.world/v22/series/{code}?observations=1",
            timeout=30)
        doc = r.json()["series"]["docs"][0]
        pairs = [(p, v) for p, v in zip(doc["period"], doc["value"])
                 if isinstance(v, (int, float))]   # ONS mixes in "NA" strings
        s = pd.Series({pd.Period(p).to_timestamp(): float(v)
                       for p, v in pairs}).sort_index()
    else:
        raise ValueError(f"unknown source {src}")
    if s.empty:
        raise ValueError(f"no observations for {code}")
    _cache_put(ck, s)
    return s


def _apply_transform(s: pd.Series, spec: dict, override: str = "auto") -> tuple:
    """(series, unit_label) after the spec/override transform."""
    t = spec.get("transform", "level") if override == "auto" else override
    if spec.get("scale"):
        s = s * spec["scale"]
    if t == "yoy":
        # robust for monthly/quarterly: compare to the obs ~1 year earlier
        per_year = 4 if (s.index.to_series().diff().dt.days.median() or 30) > 45 \
            else 12
        s = (s / s.shift(per_year) - 1) * 100
        return s.dropna(), "% YoY"
    if t == "mom":
        s = (s / s.shift(1) - 1) * 100
        return s.dropna(), "% chg"
    if t == "diff":
        return s.diff().dropna(), spec.get("unit", "Δ")
    return s.dropna(), spec.get("unit", "")


_PALETTE = ["#2563EB", "#0D9488", "#B45309", "#7C3AED", "#DC2626", "#64748B"]
_TH = ("padding:3px 9px;font-size:11px;color:#64748B;text-align:right;"
       "border-bottom:1px solid #E2E8F0;white-space:nowrap")
_TD = ("padding:3px 9px;font-size:12px;text-align:right;font-family:monospace;"
       "border-bottom:1px solid #F1F5F9;white-space:nowrap")


def render_econ():
    st.markdown(
        "<div style='background:#1E293B;color:#F8FAFC;padding:6px 12px;"
        "font-size:13px;font-weight:700;border-radius:6px;display:inline-block;"
        "margin-bottom:6px'>🌍 Economic Data"
        "&nbsp;&nbsp;<span style='font-weight:400;font-size:11px;color:#94A3B8'>"
        "US · FRED &nbsp;|&nbsp; Euro Area · Eurostat (DBnomics) · cached daily"
        "</span></div>", unsafe_allow_html=True)
    # ONE flat cross-region picker (Rajat 2026-08-06: "add US and EUR series
    # together") — labels prefixed "US ·" / "EA ·"; region dropdown now only
    # scopes the latest-values table below.
    _SHORT = {"US": "US", "Euro Area": "EA", "UK": "UK", "Japan": "JP"}
    flat = {f"{_SHORT.get(rg, rg)} · {s['label']}": s
            for rg, specs in _CATALOG.items() for s in specs}
    c2, c3, c4 = st.columns([3.2, 0.9, 0.7])
    sel = c2.multiselect("Series (chart — mix regions freely)", list(flat),
                         default=list(flat)[:1], key="_ec_sel_all",
                         max_selections=4)
    tf = c3.selectbox("Transform", ["auto", "level", "yoy", "mom", "diff"],
                      key="_ec_tf",
                      help="auto = each series' natural headline form "
                           "(NFP=Δ, CPI=YoY, rates=level…)")
    yrs = c4.selectbox("Window", ["6m", "1y", "2y", "3y", "5y", "10y", "20y",
                                  "max"], index=5, key="_ec_yrs")
    rebase = st.checkbox("rebase to 100 (compare shapes across units)",
                         key="_ec_rebase")

    # ── chart ────────────────────────────────────────────────────────────────
    if sel:
        import numpy as _np
        import plotly.graph_objects as go
        fig = go.Figure()
        if yrs == "max":
            cutoff = None
        elif yrs.endswith("m"):
            cutoff = pd.Timestamp.today() - pd.DateOffset(months=int(yrs[:-1]))
        else:
            cutoff = pd.Timestamp.today() - pd.DateOffset(years=int(yrs[:-1]))
        plotted = []      # (label, series, unit, colour)
        for i, lbl in enumerate(sel):
            spec = flat[lbl]
            try:
                with st.spinner(f"Loading {lbl}…"):
                    raw = fetch_series(spec["src"], spec["code"])
            except Exception as ex:
                st.caption(f"⚠ {lbl}: {type(ex).__name__} {str(ex)[:90]}")
                continue
            s, unit = _apply_transform(raw, spec, tf)
            if cutoff is not None:
                s = s[s.index >= cutoff]
            if s.empty:
                continue
            if rebase:
                base = s.iloc[0]
                s = (s / base * 100) if base not in (0, None) else s
                unit = "idx=100"
            plotted.append((lbl, s, unit, _PALETTE[len(plotted) % len(_PALETTE)]))
        # each series gets its OWN axis (Rajat 2026-08-06) — analyzer-style
        # left / right / outer-left / outer-right; rebase collapses to one axis
        _yref = {0: "y", 1: "y2", 2: "y3", 3: "y4"}
        _ykey = {0: "yaxis", 1: "yaxis2", 2: "yaxis3", 3: "yaxis4"}
        _axpos = [dict(side="left"),
                  dict(side="right", overlaying="y"),
                  dict(side="left", overlaying="y", anchor="free", position=0.0),
                  dict(side="right", overlaying="y", anchor="free", position=1.0)]
        n = len(plotted)
        multi_ax = n > 1 and not rebase
        for i, (lbl, s, unit, col) in enumerate(plotted):
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines", name=lbl,
                yaxis=_yref[i] if multi_ax else "y",
                line=dict(color=col, width=1.8),
                hovertemplate="%{x|%b %Y} · %{y:,.2f}<extra>" + lbl + "</extra>"))
        if fig.data:
            if n <= 1 or rebase:
                domain, margin = [0.0, 1.0], dict(l=30, r=30, t=30, b=30)
            elif n == 2:
                domain, margin = [0.0, 0.94], dict(l=30, r=55, t=30, b=30)
            else:
                domain, margin = [0.07, 0.93], dict(l=60, r=60, t=30, b=30)
            layout = dict(
                height=430, margin=margin, showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1,
                            xanchor="right", font_size=11),
                xaxis=dict(gridcolor="#F1F5F9", domain=domain),
                hovermode="x unified",
                plot_bgcolor="#FFFFFF")
            if multi_ax:
                for i, (lbl, _s, unit, col) in enumerate(plotted):
                    ax = dict(title=dict(text=unit or lbl,
                                         font=dict(color=col, size=11)),
                              tickfont=dict(color=col),
                              gridcolor="#F1F5F9", zeroline=False,
                              showgrid=(i == 0))
                    ax.update(_axpos[i])
                    layout[_ykey[i]] = ax
            else:
                layout["yaxis"] = dict(
                    gridcolor="#F1F5F9",
                    title=plotted[0][2] if plotted else "")
                _allv = _np.concatenate([_np.asarray(t.y, dtype=float)
                                         for t in fig.data])
                _ref = 100 if rebase else 0
                if _np.nanmin(_allv) <= _ref <= _np.nanmax(_allv):
                    fig.add_hline(y=_ref, line_color="#CBD5E1", line_width=1)
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    # ── latest-values table (region-scoped) ──────────────────────────────────
    region = st.selectbox("Latest values — region", list(_CATALOG),
                          key="_ec_region")
    cat = _CATALOG[region]
    with st.spinner("Latest values…"):
        rows = ""
        for spec in cat:
            try:
                raw = fetch_series(spec["src"], spec["code"])
                s, unit = _apply_transform(raw, spec, "auto")
                lastv, prevv = s.iloc[-1], (s.iloc[-2] if len(s) > 1 else None)
                dstr = s.index[-1].strftime("%b %y")
                arrow = ("<span style='color:#16A34A'>▲</span>"
                         if prevv is not None and lastv > prevv else
                         "<span style='color:#DC2626'>▼</span>"
                         if prevv is not None and lastv < prevv else "·")
                rows += (f"<tr><td style='{_TD};text-align:left'>"
                         f"<b>{spec['label']}</b></td>"
                         f"<td style='{_TD}'>{lastv:,.2f}</td>"
                         f"<td style='{_TD}'>{arrow}</td>"
                         f"<td style='{_TD}'>"
                         f"{prevv:,.2f}" if prevv is not None else "—")
                rows += (f"</td><td style='{_TD}'>{unit}</td>"
                         f"<td style='{_TD}'>{dstr}</td></tr>")
            except Exception:
                rows += (f"<tr><td style='{_TD};text-align:left'>"
                         f"<b>{spec['label']}</b></td>"
                         + f"<td style='{_TD}'>—</td>" * 5 + "</tr>")
        hdr = "".join(f"<th style='{_TH}'>{h}</th>" for h in
                      ("SERIES", "LATEST", "", "PREV", "UNIT", "AS OF"))
        st.markdown(f"<div style='overflow-x:auto'><table style='border-"
                    f"collapse:collapse'><thead><tr>{hdr}</tr></thead>"
                    f"<tbody>{rows}</tbody></table></div>",
                    unsafe_allow_html=True)
    st.caption(
        "US = FRED (headline transforms applied: NFP monthly change, inflation "
        "YoY, etc.). Euro Area = Eurostat via DBnomics. True ISM / S&P Global "
        "PMIs are licensed and not freely available — Philly Fed, Empire State "
        "and the EA Economic Sentiment Index serve as free survey proxies. "
        "NB the DBnomics Eurostat mirror can lag the primary source by weeks-to-months (check AS-OF); an ECB-SDW direct feed is the upgrade path if EA freshness matters. Series cached once per day. UK and Japan next.")

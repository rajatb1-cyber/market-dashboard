"""
Global Government Bond Rates tab.
Data sources (all free, no API key required):
  US         — US Treasury XML  (treasury.gov)            daily
  Euro Area  — ECB yield curve  (data-api.ecb.europa.eu)  daily
  Japan      — Japan MOF JGBs   (mof.go.jp)               ~1-day lag
  UK         — Bank of England  (bankofengland.co.uk)      ~1-day lag
  Australia  — RBA F2 benchmarks (rba.gov.au)             ~5-day lag
"""

import io
import re
import ssl
import urllib.request
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

COLORS = {
    "US":        "#2563EB",
    "Euro Area": "#DC2626",
    "Japan":     "#D97706",
    "UK":        "#7C3AED",
    "Australia": "#059669",
}

DISPLAY_MATS = ["1Y", "2Y", "5Y", "10Y", "20Y", "30Y"]

_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}


def _ssl():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ── US Treasury ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _us_month(year: int, month: int) -> pd.DataFrame:
    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/"
        "interest-rates/pages/xml?data=daily_treasury_yield_curve"
        f"&field_tdr_date_value_month={year}{month:02d}"
    )
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=_HDR), timeout=10
        ) as r:
            raw = r.read().decode()
    except Exception:
        return pd.DataFrame()

    dates = re.findall(r'<d:NEW_DATE[^>]*>([^<]+)</d:NEW_DATE>', raw)
    fields = {
        "1Y": "BC_1YEAR", "2Y": "BC_2YEAR", "3Y": "BC_3YEAR",
        "5Y": "BC_5YEAR", "7Y": "BC_7YEAR", "10Y": "BC_10YEAR",
        "20Y": "BC_20YEAR", "30Y": "BC_30YEAR",
    }
    vals = {k: re.findall(f'<d:{v}[^>]*>([^<]+)</', raw) for k, v in fields.items()}

    rows = []
    for i, dt in enumerate(dates):
        row = {"Date": pd.Timestamp(dt[:10])}
        for k, v in vals.items():
            try:
                row[k] = float(v[i]) if i < len(v) else None
            except (ValueError, IndexError):
                row[k] = None
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Date")


@st.cache_data(ttl=3600)
def fetch_us_treasury(months_back: int = 3) -> pd.DataFrame:
    today = date.today()
    dfs = []
    for i in range(months_back):
        y, m = today.year, today.month - i
        while m <= 0:
            m += 12
            y -= 1
        df = _us_month(y, m)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).sort_index().drop_duplicates()


# ── ECB Euro Area ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_ecb_curve(start: str | None = None) -> pd.DataFrame:
    if start is None:
        start = (date.today() - timedelta(days=14)).isoformat()
    mat_codes = {
        "1Y": "SR_1Y", "2Y": "SR_2Y", "3Y": "SR_3Y",
        "5Y": "SR_5Y", "7Y": "SR_7Y", "10Y": "SR_10Y",
        "20Y": "SR_20Y", "30Y": "SR_30Y",
    }
    dfs = []
    for mat, code in mat_codes.items():
        url = (
            "https://data-api.ecb.europa.eu/service/data/"
            f"YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{code}"
            f"?format=csvdata&startPeriod={start}"
        )
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=_HDR), timeout=10
            ) as r:
                tmp = pd.read_csv(io.StringIO(r.read().decode()))
            tmp = tmp[["TIME_PERIOD", "OBS_VALUE"]].copy()
            tmp.columns = ["Date", mat]
            tmp["Date"] = pd.to_datetime(tmp["Date"])
            tmp[mat] = pd.to_numeric(tmp[mat], errors="coerce")
            dfs.append(tmp.set_index("Date"))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, axis=1).sort_index()


# ── Japan MOF ─────────────────────────────────────────────────────────────────

def _parse_mof_csv(raw: str) -> pd.DataFrame:
    lines = raw.strip().split("\n")
    hdr_i = next((i for i, l in enumerate(lines) if l.strip().startswith("Date,")), None)
    if hdr_i is None:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr_i:])))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"].str.strip(), format="%Y/%m/%d", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    keep = [c for c in ["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"] if c in df.columns]
    return df[keep] if keep else pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_japan_jgb() -> pd.DataFrame:
    """JGB yields from MOF — historical archive + current-month CSV merged for ~1-day lag."""
    hist_url = "https://www.mof.go.jp/english/jgbs/reference/interest_rate/historical/jgbcme_all.csv"
    curr_url = "https://www.mof.go.jp/english/jgbs/reference/interest_rate/jgbcme.csv"

    hist_df = pd.DataFrame()
    curr_df = pd.DataFrame()

    try:
        with urllib.request.urlopen(
            urllib.request.Request(hist_url, headers=_HDR), timeout=20, context=_ssl()
        ) as r:
            hist_df = _parse_mof_csv(r.read().decode("utf-8", errors="ignore"))
    except Exception:
        pass

    try:
        with urllib.request.urlopen(
            urllib.request.Request(curr_url, headers=_HDR), timeout=15, context=_ssl()
        ) as r:
            curr_df = _parse_mof_csv(r.read().decode("utf-8", errors="ignore"))
    except Exception:
        pass

    if hist_df.empty and curr_df.empty:
        return pd.DataFrame()
    if hist_df.empty:
        return curr_df
    if curr_df.empty:
        return hist_df

    combined = pd.concat([hist_df, curr_df])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


# ── RBA Australia ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=7200)
def fetch_australia_bonds() -> pd.DataFrame:
    """Australian Government bond benchmark yields from RBA F2 table — daily, ~5-day lag."""
    url = "https://www.rba.gov.au/statistics/tables/csv/f2-data.csv"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=_HDR), timeout=25, context=_ssl()
        ) as r:
            raw = r.read().decode("utf-8-sig", errors="ignore")
    except Exception:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(raw), skiprows=10, header=0)
    df = df.rename(columns={df.columns[0]: "Date"})
    df = df[df["Date"].notna() & df["Date"].astype(str).str.match(r"\d")]
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    rename = {
        "FCMYGBAG2D":  "2Y",
        "FCMYGBAG3D":  "3Y",
        "FCMYGBAG5D":  "5Y",
        "FCMYGBAG10D": "10Y",
    }
    df = df.rename(columns=rename)
    keep = [c for c in ["2Y", "3Y", "5Y", "10Y"] if c in df.columns]
    result = df[keep].copy()
    for c in keep:
        result[c] = pd.to_numeric(result[c], errors="coerce")
    return result


# ── UK Bank of England ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_uk_gilts(start: str | None = None) -> pd.DataFrame:
    """UK Gilt nominal par yields from Bank of England — daily, ~1-day lag.
    Returns 5Y, 10Y, 20Y maturities."""
    if start is None:
        start = (date.today() - timedelta(days=30)).isoformat()
    try:
        datefrom = date.fromisoformat(start).strftime("%d/%b/%Y")
    except Exception:
        datefrom = "01/Jan/2020"

    url = (
        "https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp"
        f"?csv.x=yes&Datefrom={datefrom}&Dateto=now"
        "&SeriesCodes=IUDSNPY,IUDMNPY,IUDLNPY&CSVF=TT&UsingCodes=Y"
    )
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=_HDR), timeout=15
        ) as r:
            raw = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return pd.DataFrame()

    lines = raw.strip().split("\n")
    hdr_i = next((i for i, l in enumerate(lines) if l.strip().startswith("DATE,")), None)
    if hdr_i is None:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO("\n".join(lines[hdr_i:])))
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df = df.rename(columns={"IUDSNPY": "5Y", "IUDMNPY": "10Y", "IUDLNPY": "20Y"})
    keep = [c for c in ["5Y", "10Y", "20Y"] if c in df.columns]
    result = df[keep].copy()
    for c in keep:
        result[c] = pd.to_numeric(result[c], errors="coerce")
    return result


# ── Snapshot helpers ───────────────────────────────────────────────────────────

def _latest_two(df: pd.DataFrame, mat: str):
    if df.empty or mat not in df.columns:
        return None, None
    s = df[mat].dropna()
    if s.empty:
        return None, None
    latest = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) >= 2 else None
    return latest, prev


def _fmt_cell(val, chg):
    if val is None:
        return "—"
    chg_str = ""
    if chg is not None:
        sign = "+" if chg >= 0 else ""
        chg_str = f"  {sign}{chg:.2f}"
    return f"{val:.3f}{chg_str}"


# ── Charts ─────────────────────────────────────────────────────────────────────

def _base_layout(title: str, height: int) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=15, color="#1A202C")),
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F7F9FC",
        font=dict(color="#1A202C", family="Inter, Segoe UI, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11), bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#E2E8F0", borderwidth=1),
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E8F0",
                        font=dict(color="#1A202C", size=12)),
    )


def chart_yield_curve(dfs: dict, selected: list) -> go.Figure:
    fig = go.Figure()
    mat_years = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7,
                 "10Y": 10, "20Y": 20, "30Y": 30}

    for country in selected:
        df = dfs.get(country)
        if df is None or df.empty:
            continue
        last_row = df.iloc[-1]
        last_date = df.index[-1].strftime("%d %b %Y")
        xs, ys = [], []
        for mat, yrs in mat_years.items():
            if mat in last_row and pd.notna(last_row[mat]):
                xs.append(yrs)
                ys.append(float(last_row[mat]))
        if xs:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=f"{country} ({last_date})",
                line=dict(color=COLORS.get(country, "#888"), width=2.5),
                marker=dict(size=7),
                hovertemplate="%{y:.3f}%<extra>%{fullData.name}</extra>",
            ))

    fig.update_layout(**_base_layout("Yield Curves — Current", 400))
    fig.update_xaxes(
        title="Maturity (years)",
        tickvals=[1, 2, 3, 5, 7, 10, 20, 30],
        ticktext=["1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"],
        gridcolor="#E8EDF5", zeroline=False, linecolor="#E2E8F0",
    )
    fig.update_yaxes(title="Yield (%)", gridcolor="#E8EDF5", zeroline=False, linecolor="#E2E8F0")
    return fig


def chart_historical(dfs: dict, mat: str, selected: list, period: str) -> go.Figure:
    days = {"3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days.get(period, 365))

    fig = go.Figure()
    for country in selected:
        df = dfs.get(country)
        if df is None or df.empty or mat not in df.columns:
            continue
        s = df[mat].dropna()
        s = s[s.index >= cutoff]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=country,
            line=dict(color=COLORS.get(country, "#888"), width=2),
            hovertemplate="%{y:.3f}%<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(**_base_layout(f"{mat} Yields — Historical Comparison", 380))
    fig.update_xaxes(gridcolor="#E8EDF5", zeroline=False, linecolor="#E2E8F0",
                     showspikes=True, spikecolor="#94A3B8", spikethickness=1)
    fig.update_yaxes(title="Yield (%)", gridcolor="#E8EDF5", zeroline=False, linecolor="#E2E8F0",
                     showspikes=True, spikecolor="#94A3B8", spikethickness=1)
    return fig


# ── Main render ────────────────────────────────────────────────────────────────

def render_rates():
    st.markdown("### Global Government Bond Rates")

    all_countries = list(COLORS.keys())
    selected = st.pills(
        "Countries", all_countries, selection_mode="multi", default=all_countries,
    )
    if not selected:
        st.info("Select at least one country above.")
        return

    # ── Snapshot data ──────────────────────────────────────────────────────────
    with st.spinner("Loading rates data…"):
        snap_dfs = {
            "US":        fetch_us_treasury(months_back=2),
            "Euro Area": fetch_ecb_curve(),
            "Japan":     fetch_japan_jgb(),
            "UK":        fetch_uk_gilts(),
            "Australia": fetch_australia_bonds(),
        }

    # ── Yield curve + snapshot table ───────────────────────────────────────────
    col_chart, col_table = st.columns([1.4, 1])

    with col_chart:
        fig_curve = chart_yield_curve(snap_dfs, selected)
        st.plotly_chart(fig_curve, use_container_width=True)

    with col_table:
        st.markdown("**Rates Snapshot**")

        # Build table: rows = maturities, cols = countries
        rows = []
        for mat in DISPLAY_MATS:
            row = {"": mat}
            for country in selected:
                val, prev = _latest_two(snap_dfs.get(country, pd.DataFrame()), mat)
                chg = (val - prev) if (val is not None and prev is not None) else None
                row[country] = _fmt_cell(val, chg)
            rows.append(row)

        df_snap = pd.DataFrame(rows).set_index("")
        st.dataframe(df_snap, use_container_width=True, height=260)

        # Last-updated info
        notes = []
        for country in selected:
            df = snap_dfs.get(country)
            if df is not None and not df.empty:
                last = df.index[-1].strftime("%d %b")
                notes.append(f"{country}: {last}")
        if notes:
            st.caption("Last data point — " + " · ".join(notes))

    st.markdown("---")

    # ── Historical chart ───────────────────────────────────────────────────────
    st.markdown("**Historical Comparison**")

    hcol1, hcol2 = st.columns(2)
    with hcol1:
        mat_sel = st.selectbox("Maturity", DISPLAY_MATS, index=3)
    with hcol2:
        period_sel = st.segmented_control(
            "Period", ["3M", "6M", "1Y", "2Y"], default="1Y", key="rates_period"
        )
        if period_sel is None:
            period_sel = "1Y"

    months_map = {"3M": 4, "6M": 7, "1Y": 14, "2Y": 26}
    start_map  = {"3M": 90, "6M": 185, "1Y": 370, "2Y": 740}
    start_date = (date.today() - timedelta(days=start_map[period_sel])).isoformat()

    with st.spinner("Loading historical data…"):
        hist_dfs = {
            "US":        fetch_us_treasury(months_back=months_map[period_sel]),
            "Euro Area": fetch_ecb_curve(start=start_date),
            "Japan":     fetch_japan_jgb(),
            "UK":        fetch_uk_gilts(start=start_date),
            "Australia": fetch_australia_bonds(),
        }

    fig_hist = chart_historical(hist_dfs, mat_sel, selected, period_sel)
    st.plotly_chart(fig_hist, use_container_width=True)

    notes_lag = []
    if "UK" in selected:
        notes_lag.append("UK: BoE gilt par yields (5Y/10Y/20Y), ~1-day lag.")
    if "Japan" in selected:
        notes_lag.append("Japan: MOF JGBs, ~1-day lag.")
    if "Australia" in selected:
        notes_lag.append("Australia: RBA F2 benchmark bonds (2Y/3Y/5Y/10Y), ~5-day lag.")
    if notes_lag:
        st.caption("  ·  ".join(notes_lag))

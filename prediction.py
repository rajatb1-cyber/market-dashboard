import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import urllib.request
from urllib.parse import urlparse
from datetime import datetime

from watchlist import load_config, save_config

_GAMMA = "https://gamma-api.polymarket.com"
_CLOB  = "https://clob.polymarket.com"


# ── API helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _fetch_event_markets(event_slug: str) -> list:
    """Return all markets for a Polymarket event slug."""
    try:
        url = f"{_GAMMA}/events?slug={event_slug}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        if isinstance(data, list) and data:
            return data[0].get("markets", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=60)
def _fetch_market_by_slug(market_slug: str) -> dict:
    """Return a single market by its market slug."""
    try:
        url = f"{_GAMMA}/markets?slug={market_slug}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        if isinstance(data, list) and data:
            return data[0]
    except Exception:
        pass
    return {}


@st.cache_data(ttl=60)
def _fetch_market_by_condition(condition_id: str) -> dict:
    """Return a market by condition_id (for when we already have stored market data)."""
    try:
        url = f"{_GAMMA}/markets/{condition_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return {}


@st.cache_data(ttl=300)
def _fetch_history(token_id: str, days: int = 30) -> pd.DataFrame:
    interval = {7: "1w", 30: "1m", 90: "3m", 180: "6m", 365: "1y"}.get(days, "1m")
    try:
        url = f"{_CLOB}/prices-history?market={token_id}&interval={interval}&fidelity=60"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        history = data.get("history", [])
        if not history:
            return pd.DataFrame()
        df = pd.DataFrame(history)
        df["t"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_localize(None)
        df = df.rename(columns={"t": "date", "p": "probability"})
        df["probability"] = df["probability"] * 100
        return df.set_index("date")
    except Exception:
        return pd.DataFrame()


# ── URL parsing ────────────────────────────────────────────────────────────────

def _parse_polymarket_url(raw: str) -> tuple:
    """Return (event_slug, market_slug_or_None) from a URL or bare slug."""
    raw = raw.strip()
    if "#" in raw:
        raw = raw.split("#")[0]
    raw = raw.rstrip("/")

    if "polymarket.com" in raw:
        path  = urlparse(raw).path          # e.g. /event/event-slug/market-slug
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 3:
            return parts[1], parts[2]       # (event_slug, market_slug)
        elif len(parts) >= 2:
            return parts[1], None           # event URL only
        return raw.split("/")[-1], None

    return raw, None                        # bare slug treated as event slug


# ── Market parsing ────────────────────────────────────────────────────────────

def _parse_market(data: dict) -> dict:
    try:
        outcomes   = json.loads(data.get("outcomes", '["Yes","No"]'))
        prices_raw = json.loads(data.get("outcomePrices", '["0.5","0.5"]'))
        token_ids  = json.loads(data.get("clobTokenIds", "[]"))
        prices     = [float(p) * 100 for p in prices_raw]
    except Exception:
        outcomes  = ["Yes", "No"]
        prices    = [50.0, 50.0]
        token_ids = []

    end_date = data.get("endDate", "")
    if end_date:
        try:
            end_date = datetime.fromisoformat(
                end_date.replace("Z", "+00:00")
            ).strftime("%d %b %Y")
        except Exception:
            pass

    volume = 0.0
    try:
        volume = float(data.get("volume", 0))
    except Exception:
        pass

    return {
        "question":  data.get("question", ""),
        "outcomes":  outcomes,
        "prices":    prices,
        "token_ids": token_ids,
        "volume":    volume,
        "end_date":  end_date,
        "active":    data.get("active", True),
        "closed":    data.get("closed", False),
        "slug":      data.get("slug", ""),
    }


# ── Render ─────────────────────────────────────────────────────────────────────

def render_prediction():
    cfg     = st.session_state.get("wl_config") or load_config()
    markets = cfg.get("prediction_markets", [])

    st.markdown("### Prediction Markets")
    st.caption("Live probabilities from Polymarket · prices refresh every 60 s")

    # ── Add market ──────────────────────────────────────────────────────────
    with st.expander("➕  Add market"):
        c1, c2 = st.columns([5, 1])
        with c1:
            new_url = st.text_input(
                "URL", placeholder="https://polymarket.com/event/event-slug",
                key="pm_add_url", label_visibility="collapsed",
            )
        with c2:
            search_btn = st.button("Search", type="primary",
                                   use_container_width=True, key="pm_search_btn")

        if search_btn and new_url.strip():
            event_slug, market_slug = _parse_polymarket_url(new_url)
            found_markets = []

            with st.spinner("Fetching market…"):
                if market_slug:
                    raw = _fetch_market_by_slug(market_slug)
                    if raw:
                        found_markets = [raw]
                if not found_markets:
                    found_markets = _fetch_event_markets(event_slug)

            if found_markets:
                st.session_state["pm_found"] = found_markets
            else:
                st.error(f"Could not find any markets for `{event_slug}`. Check the URL.")
                st.session_state.pop("pm_found", None)

        # ── Market picker (shown after a successful search) ─────────────────
        if "pm_found" in st.session_state:
            found = st.session_state["pm_found"]
            existing_slugs = {m["slug"] for m in markets}

            if len(found) == 1:
                m = found[0]
                st.success(f"Found: **{m.get('question', m.get('slug'))}**")
                if st.button("Add this market", key="pm_add_single"):
                    slug = m.get("slug", "")
                    if slug and slug not in existing_slugs:
                        markets.append({"slug": slug, "name": m.get("question", slug)})
                        cfg["prediction_markets"] = markets
                        save_config(cfg)
                        st.session_state.wl_config = cfg
                    st.session_state.pop("pm_found", None)
                    st.rerun()
            else:
                questions = [m.get("question", m.get("slug", "")) for m in found]
                new_qs    = [q for q, m in zip(questions, found)
                             if m.get("slug", "") not in existing_slugs]
                if not new_qs:
                    st.info("All markets in this event are already tracked.")
                    st.session_state.pop("pm_found", None)
                else:
                    selected = st.multiselect(
                        f"This event has {len(found)} markets — select which to add:",
                        new_qs, default=new_qs, key="pm_multi_sel",
                    )
                    if st.button("Add selected", type="primary", key="pm_add_multi"):
                        sel_set = set(selected)
                        for m, q in zip(found, questions):
                            if q in sel_set:
                                slug = m.get("slug", "")
                                if slug and slug not in existing_slugs:
                                    markets.append({"slug": slug, "name": q})
                        cfg["prediction_markets"] = markets
                        save_config(cfg)
                        st.session_state.wl_config = cfg
                        st.session_state.pop("pm_found", None)
                        st.rerun()

    if not markets:
        st.info("No markets tracked yet — paste a Polymarket URL above to add one.")
        return

    # ── Fetch all tracked markets ───────────────────────────────────────────
    parsed = {}
    with st.spinner("Loading probabilities…"):
        for m in markets:
            raw = _fetch_market_by_slug(m["slug"])
            if raw:
                parsed[m["slug"]] = _parse_market(raw)

    if not parsed:
        st.warning("Could not load any market data.")
        return

    # ── Market cards ────────────────────────────────────────────────────────
    for m in markets:
        slug = m["slug"]
        p    = parsed.get(slug)
        if not p:
            st.warning(f"No data for `{slug}`")
            continue

        is_binary = len(p["outcomes"]) == 2 and any(
            o.lower() in ("yes", "no") for o in p["outcomes"]
        )
        status_badge = (
            "🔴 Resolved" if p["closed"] else
            "🟢 Live"     if p["active"] else
            "⚪ Inactive"
        )
        vol_str = f"${p['volume']:,.0f}" if p["volume"] else "—"

        hdr1, hdr2, hdr3 = st.columns([6, 2, 1])
        with hdr1:
            st.markdown(f"**{p['question']}**")
        with hdr2:
            st.caption(f"Vol: {vol_str}  ·  Exp: {p['end_date'] or '—'}")
        with hdr3:
            st.caption(status_badge)

        if is_binary:
            yes_idx  = next((i for i, o in enumerate(p["outcomes"]) if "yes" in o.lower()), 0)
            yes_prob = p["prices"][yes_idx]
            bar_color = (
                "#059669" if yes_prob >= 65 else
                "#D97706" if yes_prob >= 40 else
                "#DC2626"
            )
            pc1, pc2 = st.columns([1, 6])
            with pc1:
                st.markdown(
                    f"<div style='font-size:1.6rem;font-weight:700;color:{bar_color}'>"
                    f"{yes_prob:.1f}%</div>",
                    unsafe_allow_html=True,
                )
            with pc2:
                st.markdown(
                    f"<div style='margin-top:10px;background:#E2E8F0;border-radius:6px;height:10px'>"
                    f"<div style='width:{yes_prob:.1f}%;background:{bar_color};"
                    f"height:10px;border-radius:6px'></div></div>",
                    unsafe_allow_html=True,
                )
                st.caption("YES probability")
        else:
            cols = st.columns(min(len(p["outcomes"]), 6))
            for col, outcome, prob in zip(cols, p["outcomes"], p["prices"]):
                bar_color = "#059669" if prob >= 50 else "#0EA5E9"
                col.markdown(
                    f"<div style='font-size:1.1rem;font-weight:700;color:{bar_color}'>{prob:.1f}%</div>"
                    f"<div style='background:#E2E8F0;border-radius:4px;height:6px;margin:3px 0'>"
                    f"<div style='width:{prob:.1f}%;background:{bar_color};height:6px;border-radius:4px'>"
                    f"</div></div>"
                    f"<div style='font-size:0.78rem;color:#64748B'>{outcome}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

    # ── Remove ──────────────────────────────────────────────────────────────
    with st.expander("➖  Remove market"):
        name_to_slug = {m["name"]: m["slug"] for m in markets}
        to_remove    = st.multiselect("Select to remove", list(name_to_slug.keys()), key="pm_remove")
        if st.button("Remove selected", key="pm_rm_btn") and to_remove:
            remove_slugs = {name_to_slug[n] for n in to_remove}
            cfg["prediction_markets"] = [m for m in markets if m["slug"] not in remove_slugs]
            save_config(cfg)
            st.session_state.wl_config = cfg
            st.rerun()

    # ── History chart ───────────────────────────────────────────────────────
    st.markdown("#### Probability history")

    chartable = [m for m in markets if parsed.get(m["slug"], {}).get("token_ids")]
    if not chartable:
        st.caption("History not available for any tracked market.")
        return

    hc1, hc2, hc3 = st.columns([4, 2, 2])
    with hc1:
        sel_name = st.selectbox("Market", [m["name"] for m in chartable], key="pm_hist_sel")
    with hc2:
        hist_days = st.selectbox(
            "Period",
            [7, 30, 90, 180, 365],
            format_func=lambda d: {7: "1W", 30: "1M", 90: "3M", 180: "6M", 365: "1Y"}[d],
            index=1, key="pm_hist_days",
        )

    sel_slug = next(m["slug"] for m in chartable if m["name"] == sel_name)
    sel_p    = parsed[sel_slug]

    with hc3:
        if len(sel_p["outcomes"]) > 2:
            outcome_label = st.selectbox("Outcome", sel_p["outcomes"], key="pm_outcome_sel")
            outcome_idx   = sel_p["outcomes"].index(outcome_label)
        else:
            outcome_idx   = next(
                (i for i, o in enumerate(sel_p["outcomes"]) if "yes" in o.lower()), 0
            )
            outcome_label = sel_p["outcomes"][outcome_idx]
        st.caption(f"Showing: {outcome_label}")

    if outcome_idx < len(sel_p["token_ids"]):
        token_id = sel_p["token_ids"][outcome_idx]
        with st.spinner("Loading history…"):
            hist = _fetch_history(token_id, hist_days)

        if not hist.empty:
            last_prob  = float(hist["probability"].iloc[-1])
            line_color = "#059669" if last_prob >= 50 else "#DC2626"
            fill_color = (
                "rgba(5,150,105,0.08)" if last_prob >= 50 else "rgba(220,38,38,0.08)"
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[hist.index[0], hist.index[-1]], y=[50, 50],
                mode="lines",
                line=dict(color="#94A3B8", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist["probability"],
                mode="lines",
                line=dict(color=line_color, width=2),
                fill="tozeroy", fillcolor=fill_color,
                name=outcome_label,
                hovertemplate="%{x|%d %b %Y %H:%M}<br><b>%{y:.1f}%</b><extra></extra>",
            ))
            fig.update_layout(
                title=dict(
                    text=(
                        f"{sel_p['question']}  "
                        f"<span style='font-size:12px;color:#64748B'>"
                        f"current: {last_prob:.1f}%</span>"
                    ),
                    font=dict(size=13, color="#1A202C"),
                ),
                height=340,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
                yaxis=dict(range=[0, 100], ticksuffix="%",
                           gridcolor="#E8EDF5", zeroline=False),
                xaxis=dict(gridcolor="#E8EDF5", zeroline=False),
                showlegend=False,
                hovermode="x unified",
                font=dict(family="Inter, Segoe UI, sans-serif", color="#1A202C"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No history data available for this market / period.")
    else:
        st.info("History not available for this outcome.")

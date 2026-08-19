"""
📰 News tab — adaptive markets news triage.

Phase 1: ingest free RSS feeds + Finnhub headlines → dedup/cluster → rank by
book-relevance (your watchlist), recency, and an online Thompson-sampling bandit
over sources/topics that learns your taste from 👍/👎/save/click/hide feedback
(the bandit's Beta posteriors give exploration for free, so no filter bubble).

Phase 1.5 (todo): add a local embedding taste-vector (fastembed) + logistic head
for item-level content similarity — layered on top of this same feature set.

No scraping, no paywall bypass: we only use the headline+summary each feed offers;
you read the full piece via the link in your own browser/subscriptions.

Requires: feedparser (pip install feedparser). Finnhub reuses your FINNHUB_KEY secret.
"""
from __future__ import annotations
import os, re, json, time, math, sqlite3, hashlib, html
from datetime import datetime, timezone

import numpy as np
import streamlit as st

from watchlist import load_config

try:
    import feedparser  # type: ignore
    _HAS_FEEDPARSER = True
except Exception:
    _HAS_FEEDPARSER = False

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news.db")
FEEDS_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news_feeds.yaml")
LOOKBACK_DAYS = 2          # rank items from the last N days (markets triage = keep it fresh)
CLICK_W, UP_W, DOWN_W = 0.4, 1.0, 1.0   # feedback strengths


# ── storage ────────────────────────────────────────────────────────────────────
def _conn():
    c = sqlite3.connect(DB, timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    return c

def _init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS items(
            uid TEXT PRIMARY KEY, source TEXT, title TEXT, summary TEXT, url TEXT,
            published REAL, author TEXT, cluster TEXT, tickers TEXT, topics TEXT,
            first_seen REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT, uid TEXT, source TEXT,
            action TEXT, ts REAL)""")
        c.execute("CREATE TABLE IF NOT EXISTS model(k TEXT PRIMARY KEY, v TEXT)")

def _model_get(k, default):
    with _conn() as c:
        r = c.execute("SELECT v FROM model WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else default

def _model_set(k, v):
    with _conn() as c:
        c.execute("INSERT INTO model(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=?",
                  (k, json.dumps(v), json.dumps(v)))


# ── book-relevance: map news text → your watchlist instruments ───────────────────
# Hand aliases for how instruments actually get referred to in headlines.
_ALIASES = {
    "S&P 500": ["s&p 500", "s&p500", "s&p", "spx", "wall street", "u.s. stocks", "us stocks"],
    "NASDAQ": ["nasdaq", "tech stocks", "big tech"],
    "FTSE 100": ["ftse", "ftse 100", "london stocks", "uk stocks"],
    "DAX": ["dax", "german stocks"],
    "Euro Stoxx 50": ["euro stoxx", "european stocks", "stoxx"],
    "Russell 2000": ["russell 2000", "russell", "small caps", "small-caps"],
    "Nikkei 225": ["nikkei", "japanese stocks", "japan stocks"],
    "Hang Seng": ["hang seng", "hong kong stocks"],
    "CSI 300": ["csi 300", "china stocks", "chinese stocks", "mainland stocks"],
    "BBDXY": ["dollar index", "dxy", "greenback", "the dollar", "u.s. dollar", "us dollar"],
    "EUR/USD": ["eur/usd", "euro", "single currency"],
    "GBP/USD": ["gbp/usd", "sterling", "pound", "cable"],
    "USD/JPY": ["usd/jpy", "yen", "japanese yen"],
    "USD/CNH": ["yuan", "renminbi", "usd/cnh", "usd/cny"],
    "USD/INR": ["rupee", "usd/inr"],
    "US 2Y": ["2-year", "2 year", "two-year", "front end", "2y treasury"],
    "US 5Y": ["5-year", "5 year", "5y treasury"],
    "US 10Y": ["10-year", "10 year", "ten-year", "treasury yield", "10y", "u.s. treasuries", "us treasuries", "bond yields"],
    "US 30Y": ["30-year", "30 year", "long bond", "30y"],
    "EUR 10Y": ["bund", "german 10-year", "bunds", "german bonds"],
    "JPY 10Y": ["jgb", "japanese government bond", "boj yield"],
    "Gold": ["gold", "bullion", "xau"],
    "WTI Oil": ["wti", "u.s. crude", "us crude", "west texas"],
    "Brent Crude": ["brent", "crude oil", "oil prices", "opec"],
    "Silver": ["silver", "xag"],
    "Copper": ["copper", "dr copper"],
    "Bitcoin": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "Ethereum": ["ethereum", "ether", "eth"],
}

# topic tagging (keyword → topic; first hit adds the tag, multiple allowed)
_TOPIC_RX = [
    (re.compile(r"\b(fed|fomc|powell|rate cut|rate hike|ecb|lagarde|boe|boj|central bank|interest rate|hawkish|dovish|yield|treasur|bund|gilt|jgb|bond)", re.I), "rates"),
    (re.compile(r"\b(dollar|euro|yen|sterling|pound|yuan|renminbi|rupee|currenc|fx\b|forex|devalu)", re.I), "fx"),
    (re.compile(r"\b(oil|crude|opec|gold|copper|silver|natural gas|commodit|metal|wti|brent)", re.I), "commodities"),
    (re.compile(r"\b(bitcoin|crypto|ethereum|token|blockchain|stablecoin)", re.I), "crypto"),
    (re.compile(r"\b(earnings|profit|revenue|guidance|results|eps|buyback|dividend)", re.I), "earnings"),
    (re.compile(r"\b(cpi|inflation|payroll|jobs report|gdp|pmi|retail sales|unemployment|recession|economic data)", re.I), "macro-data"),
    (re.compile(r"\b(stocks|equities|s&p|nasdaq|dow|shares|index|rally|sell-?off|selloff)", re.I), "equities"),
    (re.compile(r"\b(war|sanction|tariff|geopolit|election|conflict|strike|coup|missile)", re.I), "geopolitics"),
]

@st.cache_data(ttl=3600)
def _alias_index():
    """Build {phrase -> (instrument_name, asset_class)} from watchlist + hand aliases."""
    cfg = load_config()
    idx = {}
    for inst in cfg.get("instruments", []):
        nm, cls = inst["name"], inst.get("class", "")
        phrases = set(_ALIASES.get(nm, []))
        phrases.add(nm.lower())
        for p in phrases:
            if len(p) >= 3:
                idx[p] = (nm, cls)
    return idx

def _match_book(text):
    """Return (matched_instrument_names, matched_classes) found in text."""
    t = " " + text.lower() + " "
    names, classes = [], set()
    for phrase, (nm, cls) in _alias_index().items():
        # word-ish boundary match
        if re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", t):
            if nm not in names:
                names.append(nm); classes.add(cls)
    return names, sorted(classes)

def _tag_topics(text, class_hint=""):
    tags = [lab for rx, lab in _TOPIC_RX if rx.search(text)]
    if class_hint and class_hint not in tags:
        tags.insert(0, class_hint)
    return tags or ["general"]


# ── ingest ───────────────────────────────────────────────────────────────────
def _uid(url, title):
    return hashlib.sha1((url or title).encode("utf-8", "replace")).hexdigest()[:16]

def _clean(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return html.unescape(re.sub(r"\s+", " ", s)).strip()

def _norm_title(t):
    return re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).strip()

# ── paywall handling ─────────────────────────────────────────────────────────
# Can't detect a paywall per-article without fetching the page (we don't scrape),
# and RSS rarely flags it. Paywalls are ~source-level, so we flag by known
# subscriber-only sources + URL/section patterns; you mark what you subscribe to.
PAYWALL_SOURCES = {
    "WSJ Markets", "WSJ World", "FT Home", "Barron's Real-Time",
    "Bloomberg Markets", "Bloomberg Economics", "Seeking Alpha",
}
_PAYWALL_URL = re.compile(r"/pro/|/investingclub/|/select/|(wsj|ft|barrons|bloomberg|seekingalpha)\.com/", re.I)
_PAYWALL_TITLE = re.compile(r"^(here'?s my|jim cramer\b|cramer:)|investing club", re.I)  # CNBC Pro/Club tells

def _is_paywalled(source, url, title, subscribed):
    if source in subscribed:
        return False
    if source in PAYWALL_SOURCES:
        return True
    if url and _PAYWALL_URL.search(url):
        return True
    if "cnbc" in source.lower() and title and _PAYWALL_TITLE.search(title):
        return True
    return False

def _load_feeds_cfg():
    """Curated feeds from YAML + your own sources added in-app (stored in news.db)."""
    import yaml
    with open(FEEDS_YAML, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("feeds", []).extend(_model_get("user_feeds", []))
    return cfg

def _finnhub():
    try:
        return __import__("finnhub").Client(api_key=st.secrets["FINNHUB_KEY"])
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_raw():
    """Poll all enabled feeds + Finnhub. Cached 5 min. Returns list of raw dicts."""
    cfg = _load_feeds_cfg()
    out, errors = [], []
    if _HAS_FEEDPARSER:
        for f in cfg.get("feeds", []):
            if not f.get("enabled", True):
                continue
            try:
                d = feedparser.parse(f["url"])
                for e in d.entries[:60]:
                    ts = None
                    for key in ("published_parsed", "updated_parsed"):
                        if e.get(key):
                            ts = time.mktime(e[key]); break
                    out.append({
                        "source": f["name"],
                        "title": _clean(e.get("title", "")),
                        "summary": _clean(e.get("summary", e.get("description", "")))[:600],
                        "url": e.get("link", ""),
                        "published": ts or time.time(),
                        "author": _clean(e.get("author", "")),
                        "class_hint": f.get("class_hint", ""),
                    })
            except Exception as ex:
                errors.append(f"{f['name']}: {ex}")
    fh_cfg = cfg.get("finnhub", {})
    if fh_cfg.get("enabled"):
        fh = _finnhub()
        if fh:
            try:
                for a in fh.general_news(fh_cfg.get("category", "general"))[:60]:
                    out.append({
                        "source": "Finnhub·" + (a.get("source") or "news"),
                        "title": _clean(a.get("headline", "")),
                        "summary": _clean(a.get("summary", ""))[:600],
                        "url": a.get("url", ""),
                        "published": float(a.get("datetime", time.time())),
                        "author": "", "class_hint": "",
                    })
            except Exception as ex:
                errors.append(f"Finnhub: {ex}")
    return out, errors

def _ingest():
    """Fetch → enrich → dedup → store new items in news.db. Returns count added."""
    raw, errors = _fetch_raw()
    now = time.time()
    with _conn() as c:
        seen_titles = {}  # norm_title -> cluster uid  (simple same-story clustering)
        for r in raw:
            if not r["title"]:
                continue
            uid = _uid(r["url"], r["title"])
            text = r["title"] + ". " + r["summary"]
            names, _cls = _match_book(text)
            topics = _tag_topics(text, r.get("class_hint", ""))
            nt = _norm_title(r["title"])
            cluster = seen_titles.get(nt)
            if not cluster:
                cluster = uid; seen_titles[nt] = cluster
            c.execute("""INSERT INTO items(uid,source,title,summary,url,published,author,
                cluster,tickers,topics,first_seen) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(uid) DO NOTHING""",
                (uid, r["source"], r["title"], r["summary"], r["url"], r["published"],
                 r["author"], cluster, json.dumps(names), json.dumps(topics), now))
        c.commit()
    return len(raw), errors


# ── ranking: book + recency + Thompson-sampling bandit over sources/topics ───────
def _bandit():
    """Beta posteriors per source & topic: {arm: [alpha, beta]}. α=1+likes, β=1+dislikes."""
    return _model_get("bandit", {"source": {}, "topic": {}})

def _bandit_update(source, topics, reward):
    """reward in [-1,1] scaled by weight; update Beta counts."""
    b = _bandit()
    a_inc, b_inc = (reward, 0) if reward >= 0 else (0, -reward)
    s = b["source"].setdefault(source, [1.0, 1.0]); s[0] += a_inc; s[1] += b_inc
    for t in topics:
        tt = b["topic"].setdefault(t, [1.0, 1.0]); tt[0] += a_inc; tt[1] += b_inc
    _model_set("bandit", b)

def _ucb(arm_stats, key, c=0.6):
    """Deterministic upper-confidence taste score: posterior mean + an exploration
    bonus for under-observed arms. Stable across reruns (no jitter), still explores."""
    a, bb = arm_stats.get(key, [1.0, 1.0]); n = a + bb
    return a / n + c * math.sqrt(1.0 / n)

def _hidden_sources():
    return set(_model_get("hidden_sources", []))

def _acted_uids():
    """Items you downvoted or muted — they disappear from the feed."""
    with _conn() as c:
        rows = c.execute(
            "SELECT DISTINCT uid FROM feedback WHERE action IN ('down','hide')").fetchall()
    return {r[0] for r in rows}

def _reviewed_uids():
    """Any item with feedback (up/down/save/skip/hide) — used to advance Build-mode batches."""
    with _conn() as c:
        return {r[0] for r in c.execute("SELECT DISTINCT uid FROM feedback").fetchall()}

def _source_ratings():
    return _model_get("source_ratings", {})            # {source_name: 1..10}

def _rating_for(source, ratings):
    if source in ratings:
        return ratings[source]
    if source.startswith("Finnhub"):                   # all Finnhub sub-sources share one rating
        return ratings.get("Finnhub")
    return None

# ── user tags: personalized keyword-profile themes with learned weights ──────────
# A tag (e.g. "large cap") learns the vocabulary of the stories you apply it to
# (Rocchio-style centroid) and gains preference weight; future stories whose words
# match that vocabulary get a boost scaled by how much you use & like the tag.
_STOP = set((
    "the a an and or of to in on for with at by from as is are was were be been this that "
    "these those it its into over under new says said will has have had not but you your they "
    "them we our out up down more less than then amid after market markets report reports year "
    "years week day week could would may about which who what when how his her their there here").split())

def _tokenize(text):
    return [w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", (text or "").lower()) if w not in _STOP]

def _apply_tag(tag, text, positive=True):
    tag = (tag or "").strip().lower()
    if not tag:
        return
    tags = _model_get("user_tags", {})
    e = tags.setdefault(tag, {"a": 1.0, "b": 1.0, "tok": {}})
    if positive:
        e["a"] += 1
        for k in list(e["tok"]):
            e["tok"][k] *= 0.9                          # gentle decay keeps the profile current
        for w in set(_tokenize(text)):
            e["tok"][w] = e["tok"].get(w, 0.0) + 1.0
        if len(e["tok"]) > 60:                          # keep it lean — top tokens only
            e["tok"] = dict(sorted(e["tok"].items(), key=lambda x: -x[1])[:60])
    else:
        e["b"] += 1                                     # negative example lowers the tag's pref
    _model_set("user_tags", tags)

def _tag_boost(text, tags):
    """Return (boost, [matched_tag_names]) for a story vs your learned tags."""
    toks = set(_tokenize(text))
    total, matched = 0.0, []
    for name, e in tags.items():
        tok = e.get("tok", {})
        denom = sum(tok.values())
        if denom <= 0:
            continue
        sim = sum(tok.get(w, 0.0) for w in toks) / denom     # fraction of tag's mass present
        if sim <= 0.03:
            continue
        pref = e["a"] / (e["a"] + e["b"])                    # how much you like the tag
        evidence = min(e["a"], 6.0) / 6.0                    # trust grows with examples
        contrib = sim * pref * evidence
        if contrib > 0.01:
            total += contrib
            matched.append(name)
    return total, matched

def _score_items(rows):
    """rows: list of dicts from db. Returns sorted list with score + breakdown."""
    b = _bandit()
    now = time.time()
    hidden = _hidden_sources()
    acted = _acted_uids()            # downvoted/muted items are removed from the feed
    ratings = _source_ratings()      # your explicit 1–10 source ranking (strong prior)
    utags = _model_get("user_tags", {})   # your learned personal tags
    scored = []
    # corroboration: cluster sizes across the candidate set
    clusters = {}
    for r in rows:
        clusters[r["cluster"]] = clusters.get(r["cluster"], 0) + 1
    for r in rows:
        if r["source"] in hidden or r["uid"] in acted:
            continue
        names = json.loads(r["tickers"] or "[]")
        topics = json.loads(r["topics"] or "[]")
        book = min(len(names), 3)                                  # 0..3
        age_h = max(0.0, (now - r["published"]) / 3600.0)
        recency = math.exp(-age_h / 18.0)                          # ~18h half-ish life
        corr = math.log1p(clusters.get(r["cluster"], 1) - 1)       # >0 if duplicated
        rt = _rating_for(r["source"], ratings)
        if rt == 1:                                    # rated 1 = "skip" → muted entirely (reversible)
            continue
        sa, sb = b["source"].get(r["source"], [1.0, 1.0]); sn = sa + sb
        th_top = max((_ucb(b["topic"], t) for t in topics), default=0.5)
        tag_b, tag_names = _tag_boost(r["title"] + " " + r["summary"], utags)
        # relevance/quality base — then the SOURCE RATING MULTIPLIES it (a gate), so a
        # distrusted source can't be rescued into the top by book-relevance alone.
        base = 1.4 * book + 0.35 * corr + 0.7 * th_top + 1.5 * tag_b   # content relevance
        if rt is not None:
            src_mult = 0.25 + (rt - 2) / 8.0 * 1.05    # rt2→0.25 … rt10→1.30
        else:
            src_mult = 0.7 + 0.5 * (sa / sn)           # unrated → learned click-rate (0.7…1.2)
        rec_mult = 0.35 + 0.65 * recency               # freshness GATE: fresh→1.0, ~2d old→~0.4
        score = base * src_mult * rec_mult + 0.4 * math.sqrt(1.0 / sn)
        scored.append({**r, "names": names, "topics": topics, "score": score,
                       "tag_names": tag_names,
                       "brk": {"book": book, "recency": round(recency, 2),
                               "corr": round(corr, 2),
                               "source": (f"★{rt}/10 ×{src_mult:.2f}" if rt else f"learn ×{src_mult:.2f}"),
                               "topic": round(th_top, 2),
                               "tags": (",".join(tag_names) or "-"),
                               "tag_boost": round(tag_b, 2),
                               "cluster_n": clusters.get(r["cluster"], 1)}})
    scored.sort(key=lambda x: -x["score"])
    return scored

def _feedback(uid, source, topics, action):
    with _conn() as c:
        c.execute("INSERT INTO feedback(uid,source,action,ts) VALUES(?,?,?,?)",
                  (uid, source, action, time.time()))
    if action == "hide":
        hs = set(_model_get("hidden_sources", [])); hs.add(source)
        _model_set("hidden_sources", sorted(hs))
    elif action == "skip":
        pass                                            # marks reviewed, no weight change
    else:
        reward = {"up": UP_W, "save": UP_W, "click": CLICK_W, "down": -DOWN_W}[action]
        _bandit_update(source, topics, reward)


# ── UI ───────────────────────────────────────────────────────────────────────
def _ago(ts):
    m = (time.time() - ts) / 60.0
    if m < 60: return f"{int(m)}m"
    if m < 1440: return f"{int(m/60)}h"
    return f"{int(m/1440)}d"

def render_news():
    _init_db()
    st.subheader("📰 News — adaptive triage")

    if not _HAS_FEEDPARSER:
        st.warning("RSS needs `feedparser`. Run: `pip install feedparser`  "
                   "(Finnhub headlines still work without it.)")

    top = st.columns([1, 1, 2, 2])
    with top[0]:
        if st.button("🔄 Refresh", use_container_width=True):
            _fetch_raw.clear(); st.session_state.pop("_news_batch", None); st.rerun()
    with top[1]:
        book_only = st.toggle("On-book only", value=False,
                              help="Only stories that touch your watchlist instruments")
        build = st.toggle("🛠️ Build mode", value=True,
                          help="Review the top 5, mark each, then Run to update weights and load the next 5")
        hide_pw = st.toggle("🔒 Hide paywalled", value=False,
                            help="Remove stories from subscriber-only sources you can't read "
                                 "(WSJ/FT/Bloomberg/Barron's/Seeking Alpha/CNBC-Pro). Off = show with a 🔒 badge.")

    # ingest (cached fetch under the hood)
    n_raw, errors = _ingest()

    # candidate pool: last N days
    cutoff = time.time() - LOOKBACK_DAYS * 86400
    with _conn() as c:
        c.row_factory = sqlite3.Row
        rows = [dict(r) for r in c.execute(
            "SELECT * FROM items WHERE published>=? ORDER BY published DESC LIMIT 800",
            (cutoff,)).fetchall()]

    all_sources = sorted({r["source"] for r in rows})
    all_classes = sorted({cls for _n, cls in [(i["name"], i.get("class", "")) for i in load_config()["instruments"]]})
    with top[2]:
        src_sel = st.multiselect("Sources", all_sources, default=[], placeholder="All sources")
    with top[3]:
        cls_sel = st.multiselect("Asset class", all_classes, default=[], placeholder="All classes")

    scored = _score_items(rows)
    subscribed = _model_get("subscribed", [])
    for it in scored:
        it["paywall"] = _is_paywalled(it["source"], it.get("url", ""), it["title"], subscribed)

    # filters
    def _keep(it):
        if hide_pw and it.get("paywall"): return False
        if src_sel and it["source"] not in src_sel: return False
        if book_only and not it["names"]: return False
        if cls_sel:
            inst_cls = {c for c in _match_book(it["title"] + " " + it["summary"])[1]}
            if not (inst_cls & set(cls_sel)): return False
        return True
    feed = [it for it in scored if _keep(it)][:80]

    hidden = _hidden_sources()
    cap = st.columns([3, 1])
    cap[0].caption(f"{len(feed)} ranked · {len(rows)} in last {LOOKBACK_DAYS}d · "
                   f"{n_raw} polled" + (f" · {len(hidden)} source(s) muted" if hidden else ""))
    if hidden and cap[1].button("Un-mute all"):
        _model_set("hidden_sources", []); st.rerun()
    if errors:
        with st.expander(f"⚠️ {len(errors)} feed error(s)"):
            st.write(errors)

    # ── source-rating survey (persisted; strong prior on the feed) ──
    with st.expander("⭐ Rate your sources — 1 = skip, 10 = must-read",
                     expanded=st.session_state.get("_news_rate_open", False)):
        fcfg = _load_feeds_cfg()
        names = sorted({f["name"] for f in fcfg.get("feeds", []) if f.get("enabled", True)})
        if fcfg.get("finnhub", {}).get("enabled"):
            names.append("Finnhub")
        cur = _source_ratings()
        if st.session_state.pop("_news_rated", False):
            st.success(f"✓ Saved {len(cur)} source ratings — feed re-ranked below.")
        st.caption("Baseline priority per source. Your 👍/👎 still refine topics & dismiss stories on top.")
        cols = st.columns(3)
        new = {}
        for i, nm in enumerate(names):
            new[nm] = cols[i % 3].slider(nm, 1, 10, int(cur.get(nm, 5)), key=f"rate_{nm}")
        if st.button("💾 Save ratings & re-rank", type="primary", key="save_ratings"):
            _model_set("source_ratings", {k: int(v) for k, v in new.items()})
            st.session_state["_news_rated"] = True
            st.session_state["_news_rate_open"] = True
            st.rerun()
        if cur:
            st.caption("Currently saved → " + " · ".join(
                f"{k} ★{v}" for k, v in sorted(cur.items(), key=lambda x: -x[1])))

    # ── add your own sources (stored in news.db, non-destructive) ──
    with st.expander("➕ Add / manage your own sources", expanded=False):
        uf = _model_get("user_feeds", [])
        with st.form("add_src", clear_on_submit=True):
            fc = st.columns([2, 4, 1])
            nnm = fc[0].text_input("Name", placeholder="FT Alphaville")
            nurl = fc[1].text_input("RSS / Atom URL", placeholder="https://…/rss.xml")
            submitted = fc[2].form_submit_button("Add")
        if submitted and nnm and nurl:
            ok, note = True, ""
            if _HAS_FEEDPARSER:
                try:
                    d = feedparser.parse(nurl)
                    ok = bool(d.entries)
                    note = f"{len(d.entries)} items" if ok else "no items parsed"
                except Exception as ex:
                    ok, note = False, str(ex)[:80]
            if ok:
                uf = [f for f in uf if f.get("name") != nnm] + \
                     [{"name": nnm, "url": nurl, "enabled": True}]
                _model_set("user_feeds", uf)
                _fetch_raw.clear()
                st.success(f"Added “{nnm}” ({note}). It's now in the feed and the rating survey.")
                st.rerun()
            else:
                st.warning(f"Couldn't validate that feed ({note}). Make sure it's an RSS/Atom URL.")
        if uf:
            st.caption("Your added sources:")
            for f in uf:
                rc = st.columns([6, 1])
                rc[0].write(f"• **{f['name']}** — {f['url']}")
                if rc[1].button("Remove", key=f"rmfeed_{f['name']}"):
                    _model_set("user_feeds", [x for x in uf if x["name"] != f["name"]])
                    _fetch_raw.clear()
                    st.rerun()

    # ── paywall & subscriptions ──
    with st.expander("🔒 Paywall & subscriptions", expanded=False):
        st.caption("Subscriber-only sources are flagged 🔒 (or removed via the ‘Hide paywalled’ "
                   "toggle up top). Mark any you DO subscribe to so they're treated as readable.")
        subs = st.multiselect("I subscribe to:", sorted(PAYWALL_SOURCES),
                              default=_model_get("subscribed", []), key="subs_sel")
        if st.button("💾 Save subscriptions", key="save_subs"):
            _model_set("subscribed", subs)
            st.toast("Saved")
            st.rerun()

    st.divider()

    # ── BUILD MODE: batch-of-5 active tuning ──
    utags = _model_get("user_tags", {})
    if build:
        reviewed = _reviewed_uids()
        # draw from the FULL scored pool (not the 80-capped display feed) so the queue
        # doesn't run dry while you still have unreviewed stories further down.
        queue = [it for it in scored if _keep(it) and it["uid"] not in reviewed]
        # PIN the current 5 so they stay fixed while you vote/tag — reruns (adding a tag
        # triggers one) won't swap them out. Only Run / Refresh / Re-show loads a new 5.
        if "_news_batch" not in st.session_state:
            st.session_state["_news_batch"] = queue[:5]
        batch = st.session_state["_news_batch"]
        with _conn() as cc:
            n_skip = cc.execute("SELECT COUNT(*) FROM feedback WHERE action='skip'").fetchone()[0]
        hd = st.columns([3, 1])
        hd[0].caption(f"🛠️ Reviewing top 5 · {len(reviewed)} reviewed · {len(queue)} in queue · "
                      "vote + tag, then Run")
        if n_skip and hd[1].button(
                f"↺ Re-show {n_skip} skipped",
                help="Brings the stories you marked ‘— skip’ back into the queue for another look. "
                     "Does NOT touch your 👍/👎, ratings, tags or learned weights."):
            with _conn() as cc:
                cc.execute("DELETE FROM feedback WHERE action='skip'")
            st.session_state.pop("_news_batch", None)
            st.rerun()
        if utags:
            st.caption("🏷️ Your tags → " + " · ".join(
                f"{n} ({int(e['a']-1)}👍" + (f"/{int(e['b']-1)}👎" if e['b'] > 1 else "") + ")"
                for n, e in sorted(utags.items(), key=lambda x: -x[1]['a'])))
        if not batch:
            st.info(f"You're caught up on the last {LOOKBACK_DAYS} days. New stories publish "
                    "through the day (thinner on weekends).")
            if st.button("🔄 Poll for new stories now", type="primary", use_container_width=True):
                _fetch_raw.clear(); st.session_state.pop("_news_batch", None); st.rerun()
        existing_tags = sorted(utags.keys())
        OPTS = ["👎 less", "— skip", "👍 more"]
        for it in batch:
            chips = "".join(
                f"<span style='background:rgba(58,107,140,.16);color:#3A6B8C;padding:1px 6px;"
                f"border-radius:5px;font-size:11px;margin-right:4px'>{html.escape(nm)}</span>"
                for nm in it["names"][:4])
            st.markdown(
                f"**[{html.escape(it['title'])}]({it['url']})**  \n"
                f"<span style='color:#8593A0;font-size:12px'>{'🔒 ' if it.get('paywall') else ''}{html.escape(it['source'])} · "
                f"{_ago(it['published'])} ago · {' · '.join(it['topics'][:3])} · "
                f"score {it['score']:.2f} · {it['brk']['source']}"
                + (f" · 🏷️{','.join(it['tag_names'])}" if it['tag_names'] else "")
                + f"</span> {chips}",
                unsafe_allow_html=True)
            rc = st.columns([3, 4])
            rc[0].radio("fb", OPTS, index=1, horizontal=True,
                        key=f"b5_{it['uid']}", label_visibility="collapsed")
            rc[1].multiselect("tags", existing_tags, key=f"tg_{it['uid']}",
                              placeholder="tags → applied on 👍/👎 (pick or type new)",
                              accept_new_options=True, label_visibility="collapsed")
        if batch and st.button("▶ Run — update weights & load next 5", type="primary",
                               use_container_width=True):
            acts = {"👍 more": "up", "👎 less": "down", "— skip": "skip"}
            for it in batch:
                action = acts[st.session_state.get(f"b5_{it['uid']}", "— skip")]
                _feedback(it["uid"], it["source"], it["topics"], action)
                tgs = st.session_state.get(f"tg_{it['uid']}", [])
                if tgs and action in ("up", "down"):
                    for t in tgs:
                        _apply_tag(t, it["title"] + " " + it["summary"], positive=(action == "up"))
            st.session_state.pop("_news_batch", None)
            st.toast("Weights + tags updated — next 5 loaded")
            st.rerun()
        return

    for it in feed:
        c1, c2 = st.columns([10, 3])
        with c1:
            chips = ""
            for nm in it["names"][:4]:
                chips += f"<span style='background:rgba(58,107,140,.16);color:#3A6B8C;padding:1px 6px;border-radius:5px;font-size:11px;margin-right:4px'>{html.escape(nm)}</span>"
            ttags = " · ".join(it["topics"][:3])
            st.markdown(
                f"**[{html.escape(it['title'])}]({it['url']})**  \n"
                f"<span style='color:#8593A0;font-size:12px'>{'🔒 ' if it.get('paywall') else ''}{html.escape(it['source'])} · "
                f"{_ago(it['published'])} ago"
                + (f" · {it['brk']['cluster_n']}× sources" if it['brk']['cluster_n'] > 1 else "")
                + f" · {ttags}</span> {chips}",
                unsafe_allow_html=True)
        with c2:
            b = st.columns(4)
            uid, src, tps = it["uid"], it["source"], it["topics"]
            if b[0].button("👍", key=f"up_{uid}", help="More like this"):
                _feedback(uid, src, tps, "up"); st.rerun()
            if b[1].button("👎", key=f"dn_{uid}", help="Less like this"):
                _feedback(uid, src, tps, "down"); st.rerun()
            if b[2].button("💾", key=f"sv_{uid}", help="Save"):
                _feedback(uid, src, tps, "save"); st.toast("Saved & learned")
            if b[3].button("🙈", key=f"hd_{uid}", help=f"Mute {src}"):
                _feedback(uid, src, tps, "hide"); st.rerun()
        with st.expander("why", expanded=False):
            st.caption(f"score {it['score']:.2f} — " + " · ".join(
                f"{k}:{v}" for k, v in it["brk"].items()))

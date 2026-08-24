"""🧮 Pricer — price arbitrary option structures off the dashboard's own synthetic
vol markets (the cached settlement smiles behind Vol Dash / Options / Rates-Vol).

Rajat's spec 2026-08-04: pick asset class → market → expiry (standard list, or every
listed expiry via checkbox) → lots → natural-language description ("7500/7600 cs",
"96.25 straddle", "112/110 ps", "1.16/1.18 rr", "7400/7500/7600 c fly"...). Output:
premium in native points + $, BS delta %, $ delta, $ theta, $ vega. Structures
accumulate in a session blotter with totals — the base for the coming scenario tool.

Pricing: v2 markets Black-76, rates markets Bachelier — the SAME pricers/surfaces
the rest of the dashboard uses (options_v2._black76_greeks / rates_options.
_bachelier_greeks + the per-expiry spline smiles, flat-extrapolated beyond listed
strikes). NO new data fetches — everything reads the daily settlement caches."""

import math
import re
from datetime import date

import numpy as np
import streamlit as st

import options_v2 as _ov2
import rates_options as _ro
import vol_dashboard as _vd

# ── Contract multipliers ($ per 1.0 move in quoted price, per contract) ───────
# European contracts (€/£) shown as-is — labelled in the caption, not converted.
_MULT = {
    # equities
    "ES": 50, "NQ": 20, "RTY": 50, "ESTX": 10, "DAX": 5,
    # commodities
    "GC": 100, "SI": 5000, "HG": 25000, "CL": 1000, "BRN": 1000, "BTC": 0.1,
    # FX (per 1.0 move in USD-per-unit quote)
    "EUR": 125000, "GBP": 62500, "JPY": 12500000, "AUD": 100000,
    "CAD": 100000, "CHF": 125000, "MXN": 500000, "NZD": 100000,
    # bonds ($/pt per 100k face; Eurex €/pt) — NB Buxl = "UX", Schatz = "DU"
    "FV": 1000, "TY": 1000, "US": 1000, "UB": 1000,
    "DU": 1000, "OE": 1000, "RX": 1000, "UX": 1000,
    "TU": 2000,   # 2Y note (ZT): $200k face → $2,000/pt, unlike the rest
    # STIRs ($/€/£ 2500 per 1.00 price pt = 25/bp)
    "SOFR": 2500, "SOFR_1Y": 2500, "SOFR_2Y": 2500,
    "ER": 2500, "ER_1Y": 2500, "ER_2Y": 2500,
    "SONIA": 2500, "SONIA_1Y": 2500, "SONIA_2Y": 2500,
}
_EUR_CCY = {"ESTX", "DAX", "DU", "OE", "RX", "UX", "ER", "ER_1Y", "ER_2Y"}
_GBP_CCY = {"SONIA", "SONIA_1Y", "SONIA_2Y"}


# ── Live underlying (Rajat 2026-08-04: "1d-ago vol but LIVE future for pricing")
# yfinance ~15-min-delayed quotes; shift = the ticker's own last/previous_close
# applied to the surface forward (parallel front→deferred), smile floats with the
# forward (sticky-moneyness: IV read at the same moneyness vs the settlement F).
_YF_LIVE = {
    "ES": "ES=F", "NQ": "NQ=F", "RTY": "RTY=F",
    "ESTX": "^STOXX50E", "DAX": "^GDAXI",          # cash index return → future
    "GC": "GC=F", "SI": "SI=F", "HG": "HG=F", "CL": "CL=F", "BRN": "BZ=F",
    "BTC": "BTC-USD",
    "EUR": "6E=F", "JPY": "6J=F", "GBP": "6B=F", "AUD": "6A=F",
    "CAD": "6C=F", "CHF": "6S=F", "MXN": "6M=F", "NZD": "6N=F",
    "TU": "ZT=F", "FV": "ZF=F", "TY": "ZN=F", "US": "ZB=F", "UB": "UB=F",
    # OE/RX (Eurex) and STIRs: no yahoo futures — settlement F stands.
}


@st.cache_data(ttl=120, show_spinner=False)
def _live_shift(mkt: str):
    """(last, prev_close) from yahoo for the market's live ticker, or None."""
    tkr = _YF_LIVE.get(mkt)
    if not tkr:
        return None
    try:
        import yfinance as yf
        fi = yf.Ticker(tkr).fast_info
        last = float(fi["last_price"])
        prev = float(fi["previous_close"])
        if not (math.isfinite(last) and math.isfinite(prev) and last > 0 and prev > 0):
            return None
        return (last, prev)
    except Exception:
        return None


# ── Market universe (mirrors the Vol Dash panels; crosses excluded — synthetic) ──
def _universe():
    """{asset_class: [(src, mkt, label), ...]} from vol_dashboard._PANELS."""
    out, seen = {}, set()
    for title, _u, src, _meas, mkts in _vd._PANELS:
        cls = "Rates" if title.startswith("Rates") else title
        for entry in mkts:
            mkt, lbl = entry[0], entry[1]
            if (src, mkt) in seen:
                continue
            seen.add((src, mkt))
            out.setdefault(cls, []).append((src, mkt, lbl))
    return out


# ── Yield-terms display for rates markets (Rajat 2026-08-21) ─────────────────
# Money-market: exact, price = 100 − rate. Bond futures: model-based (≈) —
# fwd yld = CTD-maturity par yield (same source as the DV01 estimate),
# strike yld = fwd yld + (F − K)/DV01 bp. Respects the per-market CTD-box
# session state from the Rates Options tab, like _get_market_dv01.
def _yield_disp(src: str, mkt: str, F: float, legs: list, fut: bool):
    """(fwd_str, strikes_str) in yield terms, or ("—", "—") if not a rates
    market / no conversion available."""
    if src != "rates":
        return "—", "—"
    m = _ro._MARKETS_RATES.get(mkt) or {}
    ks = []
    if not fut:
        for _q, _cp, k in legs:
            if k not in ks:
                ks.append(k)
    if mkt in _ro._MONEY_MKT_MARKETS:
        fs = f"{round(100 - F, 4):g}%"
        return fs, ("/".join(f"{round(100 - k, 4):g}" for k in ks) + "%"
                    if ks else "—")
    if not m.get("needs_ctd"):
        return "—", "—"
    try:
        dv01, ok = _vd._market_dv01(mkt, m)
        if not (ok and math.isfinite(F)):
            return "—", "—"
        cy = st.session_state.get(f"_ro_ctdyrs_{mkt}", m["ctd_years"])
        dy, _dc = _ro._default_ctd_inputs(cy, curve=m["curve"])
        fs = f"≈{dy:.2f}%"
        return fs, ("/".join(f"{dy + (F - k) / dv01 * 0.01:.2f}"
                             for k in ks) + "%" if ks else "—")
    except Exception:
        return "—", "—"


# ── Duration display for rates markets (Rajat 2026-08-21) ────────────────────
# Bond futures: CTD modified duration from the same _estimate_dv01 machinery
# (respects the Rates-tab CTD-maturity box; NB if a DV01 was typed directly
# in that box, this still shows the estimated-CTD duration). Money-market:
# 0.25y — the underlying is always a 3M rate future, midcurves included.
def _dur_disp(src: str, mkt: str) -> str:
    if src != "rates":
        return "—"
    m = _ro._MARKETS_RATES.get(mkt) or {}
    if "fixed_dv01" in m:
        return "0.25y"
    if not m.get("needs_ctd"):
        return "—"
    try:
        cy = st.session_state.get(f"_ro_ctdyrs_{mkt}", m["ctd_years"])
        dy, dc = _ro._default_ctd_inputs(cy, curve=m["curve"])
        _p, mod_dur, _cf, _dv = _ro._estimate_dv01(dc / 100, dy / 100, cy,
                                                   freq=m["freq"])
        return f"{mod_dur:.1f}y" if math.isfinite(mod_dur) else "—"
    except Exception:
        return "—"


# ── NLP description parser ────────────────────────────────────────────────────
# Returns (legs, note) — legs = [(qty, "C"|"P", strike)] for ONE lot of the
# structure — or (None, error_message).
_STRUCTS_HELP = ("**Syntax**: strikes separated by `/`, then the structure — "
                 "`7500 c` · `7500 p` · `7500/7600 cs` · `7400/7300 ps` · "
                 "`7500 straddle` · `7300/7700 strangle` · `7300/7700 rr` "
                 "(short put / long call) · `7400/7500/7600 c fly` · "
                 "`7200/7300/7700/7800 condor` · `110/112/114 c ladder` "
                 "(+1/−1/−1) · ratio `7500/7600 1x2 cs` · "
                 "prefix `sell` to flip signs.")


def parse_structure(desc: str):
    if not desc or not desc.strip():
        return None, "empty description"
    s = desc.strip().lower()
    sell = bool(re.match(r"^\s*(sell|short)\b", s))
    s = re.sub(r"^\s*(sell|short|buy|long)\b", "", s).strip()

    # ratio like 1x2 / 2x3
    ratio = None
    mrat = re.search(r"\b(\d+)\s*x\s*(\d+)\b", s)
    if mrat:
        ratio = (int(mrat.group(1)), int(mrat.group(2)))
        s = s.replace(mrat.group(0), " ")

    # strikes: numbers (int/decimal), typically K1/K2/...
    ks = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not ks:
        return None, "no strikes found"

    def _cp_flag(default="C"):
        # a lone p/put token (not part of 'ps'/'spread') flips the fly/condor type
        if re.search(r"\bp(?:ut)?\b", s):
            return "P"
        if re.search(r"\bc(?:all)?\b", s):
            return "C"
        return default

    # strike order is LITERAL: first strike = the leg you BUY (Rajat 2026-08-04
    # confirmed after initial surprise — "7300/7200 cs" = long 7300C short 7200C,
    # a credit spread with negative prem = premium RECEIVED). `sell` flips signs.
    legs = None
    if re.search(r"\b(cs|call\s*spread)\b", s):
        if len(ks) < 2: return None, "call spread needs 2 strikes"
        q1, q2 = ratio if ratio else (1, 1)
        legs = [(q1, "C", ks[0]), (-q2, "C", ks[1])]
    elif re.search(r"\b(ps|put\s*spread)\b", s):
        if len(ks) < 2: return None, "put spread needs 2 strikes"
        q1, q2 = ratio if ratio else (1, 1)
        legs = [(q1, "P", ks[0]), (-q2, "P", ks[1])]
    elif re.search(r"\b(rr|risk\s*rev(?:ersal)?)\b", s):
        if len(ks) < 2: return None, "risk reversal needs 2 strikes (put/call)"
        legs = [(-1, "P", min(ks[:2])), (1, "C", max(ks[:2]))]
    elif re.search(r"\b(strad(?:dle)?)\b", s):
        legs = [(1, "C", ks[0]), (1, "P", ks[0])]
    elif re.search(r"\b(strangle|strang|stg)\b", s):
        if len(ks) < 2: return None, "strangle needs 2 strikes"
        legs = [(1, "P", min(ks[:2])), (1, "C", max(ks[:2]))]
    elif re.search(r"\b(fly|butterfly)\b", s):
        if len(ks) < 3: return None, "fly needs 3 strikes"
        cp = _cp_flag("C")
        legs = [(1, cp, ks[0]), (-2, cp, ks[1]), (1, cp, ks[2])]
    elif re.search(r"\b(condor)\b", s):
        if len(ks) < 4: return None, "condor needs 4 strikes"
        cp = _cp_flag("C")
        legs = [(1, cp, ks[0]), (-1, cp, ks[1]), (-1, cp, ks[2]), (1, cp, ks[3])]
    elif re.search(r"\b(ladder)\b", s):
        if len(ks) < 3: return None, "ladder needs 3 strikes"
        cp = _cp_flag("C")
        legs = [(1, cp, ks[0]), (-1, cp, ks[1]), (-1, cp, ks[2])]
    elif re.search(r"\bp(?:ut)?\b", s):
        # >1 strike with only a bare p/c word = an UNRECOGNIZED structure term —
        # error instead of silently pricing a single option (the "call ladder"
        # lesson, 2026-08-04: it priced as one 110 call). New terms get added to
        # the dictionary as Rajat uses them.
        if len(ks) > 1:
            return None, ("multiple strikes but I don't know this structure "
                          "term yet — " + _STRUCTS_HELP)
        legs = [(1, "P", ks[0])]
    elif re.search(r"\bc(?:all)?\b", s):
        if len(ks) > 1:
            return None, ("multiple strikes but I don't know this structure "
                          "term yet — " + _STRUCTS_HELP)
        legs = [(1, "C", ks[0])]
    else:
        return None, ("could not infer the structure — " + _STRUCTS_HELP)

    if sell:
        legs = [(-q, cp, k) for q, cp, k in legs]
    return legs, None


# ── Weekly expiry DATES (definitions only — Rajat 2026-08-04: "I do need the
# dates but we don't need to download all the options related to that"). CME
# weekly treasury option roots: Friday ZN1-5/ZF1-5/ZB1-5/UB1-5, Monday VY/VF/
# VB/VU 1-5, Wednesday WY/WF/WB/WU 1-5. Definitions are pennies; settlements
# for these roots are NEVER fetched — weekly dates price via the surface
# time-interpolation. Unresolvable roots degrade gracefully (Friday-only, then
# empty).
_WEEKLY_DEF_ROOTS = {
    # treasuries: Friday ZN…/Monday VY…/Wednesday WY… families ×1-5
    "TY": ["ZN1", "ZN2", "ZN3", "ZN4", "ZN5", "VY1", "VY2", "VY3", "VY4", "VY5",
           "WY1", "WY2", "WY3", "WY4", "WY5"],
    "FV": ["ZF1", "ZF2", "ZF3", "ZF4", "ZF5", "VF1", "VF2", "VF3", "VF4", "VF5",
           "WF1", "WF2", "WF3", "WF4", "WF5"],
    "US": ["ZB1", "ZB2", "ZB3", "ZB4", "ZB5", "VB1", "VB2", "VB3", "VB4", "VB5",
           "WB1", "WB2", "WB3", "WB4", "WB5"],
    "UB": ["UB1", "UB2", "UB3", "UB4", "UB5", "VU1", "VU2", "VU3", "VU4", "VU5",
           "WU1", "WU2", "WU3", "WU4", "WU5"],
    # E-mini S&P dailies/weeklies: Mon E#A, Tue E#B, Wed E#C, Thu E#D, Fri EW#,
    # EOM EW (EW3 = 3rd-Friday, already in the daily load — harmless dup)
    "ES": (["EW", "EW1", "EW2", "EW3", "EW4"]
           + [f"E{i}{s}" for i in range(1, 6) for s in "ABCD"]),
    # Nasdaq same scheme with Q roots
    "NQ": (["QNE", "QN1", "QN2", "QN3", "QN4"]
           + [f"Q{i}{s}" for i in range(1, 6) for s in "ABCD"]),
    # FX weeklies = daily coverage where listed (every root verified against
    # its 6X underlying via a defs probe 2026-08-24 — several lookalike roots
    # are NOT FX: MC=micro-crude, WB/WF=treasury, 3CA=corn, SN=soybeans).
    # Scheme: Mon/Tue/Wed/Thu prefix+pair-letter ×1-5, Fri = week#+pair code.
    # CAD lists no Tuesday; CHF is Friday-only; MXN/NZD have no weeklies.
    "EUR": ([f"{p}{i}" for p in ("MO", "TU", "WE", "SU") for i in range(1, 6)]
            + [f"{i}EU" for i in range(1, 6)]),
    "JPY": ([f"{p}{i}" for p in ("MJ", "TJ", "WJ", "SJ") for i in range(1, 6)]
            + [f"{i}JY" for i in range(1, 6)]),
    "GBP": ([f"{p}{i}" for p in ("MB", "TG", "WG", "SB") for i in range(1, 6)]
            + [f"{i}BP" for i in range(1, 6)]),
    "AUD": ([f"{p}{i}" for p in ("MA", "TA", "WA", "SA") for i in range(1, 6)]
            + [f"{i}AD" for i in range(1, 6)]),
    "CAD": ([f"{p}{i}" for p in ("MD", "WD", "SD") for i in range(1, 6)]
            + [f"{i}CD" for i in range(1, 6)]),
    "CHF": [f"{i}SF" for i in range(1, 6)],
}


def _wk_path(mkt: str, tdate: str) -> str:
    import os
    return os.path.join(os.path.dirname(__file__), "vol_dash_cache",
                        f"WKEXP_{mkt}_{tdate}.pkl")


def _wk_fetch(mkt: str, roots: list, tdate: str, fp: str) -> None:
    """Background worker: definitions-only fetch of the weekly roots' expiry
    dates → disk pickle. NO streamlit calls (daemon thread). Symbols go up in
    SMALL CHUNKS retried independently — Databento's gateway 504s randomly under
    evening congestion (2026-08-04: single-root probes half-failed half-worked),
    so one big batch = one bad lottery ticket; partial results still get saved."""
    import os
    import pickle as _pkl
    import time as _t
    from datetime import timedelta
    try:
        import databento as db
        client = db.Historical(key=_vd._api_key())
        end = (date.fromisoformat(tdate) + timedelta(days=1)).isoformat()
        today = date.today()
        exps = set()
        syms = [f"{r}.OPT" for r in roots]
        for i in range(0, len(syms), 5):
            chunk = syms[i:i + 5]
            for att in range(3):
                try:
                    raw = _ov2._get_range(client, "GLBX.MDP3", chunk, "definition",
                                          tdate, end, stype_in="parent")
                    df = raw.to_df()
                    if not df.empty and "expiration" in df.columns:
                        exps |= {ex.date() for ex in df["expiration"].dropna()
                                 if (ex.date() - today).days >= 1}
                    break
                except Exception:
                    _t.sleep(10 * (att + 1))
        if exps:
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "wb") as fh:
                _pkl.dump(sorted(exps), fh)
    except Exception:
        pass


# Rajat 2026-08-04: "build a library for each market — for now focus on S&P,
# Nasdaq, Rates and STIR". STIRs need no extra roots (SR3/ER/SO3 defs already
# carry every serial/quarterly/midcurve expiry); the library below warms the
# extra-roots markets in ONE background thread when the Pricer opens.
_WK_WARM_MKTS = [("v2", "ES"), ("v2", "NQ"), ("v2", "EUR"),
                 ("rates", "TY"), ("rates", "FV"), ("rates", "US"), ("rates", "UB")]


def _wk_warm_all() -> None:
    """Daemon: sequentially fetch missing weekly-date pickles for the focus
    markets (definitions only, chunked+retried inside _wk_fetch)."""
    import os
    for src, mkt in _WK_WARM_MKTS:
        try:
            roots = _WEEKLY_DEF_ROOTS.get(mkt)
            if not roots:
                continue
            cfg = (_ro._MARKETS_RATES if src == "rates" else _ov2._MARKETS_V2)[mkt]
            td = str(_vd._trade_date(cfg["ds"]))
            fp = _wk_path(mkt, td)
            if not os.path.exists(fp):
                _wk_fetch(mkt, roots, td, fp)
        except Exception:
            continue


def _weekly_expiries(src: str, mkt: str, tdate: str) -> list:
    """Weekly option expiry DATES (definitions only). NON-BLOCKING: serves the
    disk pickle when present; otherwise kicks a daemon fetch and returns [] —
    Databento gateway hangs must never spin the render (Rajat hit it live,
    2026-08-04). Dates appear on a later rerun once the worker lands."""
    import os
    import pickle as _pkl
    roots = _WEEKLY_DEF_ROOTS.get(mkt)
    if not roots:
        return []
    fp = _wk_path(mkt, tdate)
    if os.path.exists(fp):
        try:
            with open(fp, "rb") as fh:
                return _pkl.load(fh)
        except Exception:
            pass
    flag = f"_pr_wkfetch_{mkt}_{tdate}"
    if not st.session_state.get(flag):
        st.session_state[flag] = True
        import threading
        threading.Thread(target=_wk_fetch, args=(mkt, roots, tdate, fp),
                         daemon=True).start()
    return []


# ── Surfaces / expiries (cached; zero new fetches — settlement caches only) ───
@st.cache_data(ttl=1800, show_spinner=False)
def _surface_for(src: str, mkt: str):
    """(surface, tdate, all_expiries, err). Standard expiries = surface keys."""
    curve, tdate, err = _vd._load_market(src, mkt)
    if not tdate:
        return None, None, [], (err or "market data unavailable")
    try:
        if src == "rates":
            m = _ro._MARKETS_RATES[mkt]
            dv01, _ok = _vd._market_dv01(mkt, m)
            surf = _ro._build_surface_data_rates(mkt, tdate, m, dv01)
            data = _ro._load_data(mkt, tdate)
        else:
            surf = _ov2._build_surface_data(mkt, tdate)
            data = _vd._sanitize_v2_defs(_ov2._load_data(mkt, tdate))
        defs = data.get("opt_defs")
        today = date.today()
        all_exps = sorted({ex.date() for ex in defs["expiration"].dropna()
                           if (ex.date() - today).days >= 1}) if defs is not None else []
        if not surf:
            return None, tdate, all_exps, "no smile surface for this market"
        return surf, tdate, all_exps, None
    except Exception as ex:
        return None, tdate, [], f"{type(ex).__name__}: {str(ex)[:120]}"


def _chain_smile_for_expiry(src: str, mkt: str, tdate: str, expiry: date):
    """For a NON-standard expiry (not in the fitted surface): build its chain and
    return a minimal surface-entry-alike for IV readout. Chains are disk/session
    cached by the source modules, so this stays cheap."""
    if src == "rates":
        m = _ro._MARKETS_RATES[mkt]
        dv01, _ok = _vd._market_dv01(mkt, m)
        data = _ro._load_data(mkt, tdate)
        chain = _ro._build_chain(mkt, expiry, 15, 3.0, data, m["r"], dv01,
                                 cache_date=tdate)
    else:
        data = _vd._sanitize_v2_defs(_ov2._load_data(mkt, tdate))
        chain = _ov2._build_chain(mkt, expiry, 15, 3.0, data, cache_date=tdate)
    if chain is None or chain.empty:
        return None
    return chain


def _leg_iv(src, mkt, surf, tdate, expiry, K):
    """IV for one strike at one expiry — surface spline when the expiry is in the
    fitted surface, else a fresh chain's nearest-strike OTM IV."""
    if surf and expiry in surf:
        return (_ro._iv_from_surface_rates(surf, expiry, K) if src == "rates"
                else _ov2._iv_from_surface(surf, expiry, K))
    chain = _chain_smile_for_expiry(src, mkt, tdate, expiry)
    if chain is None:
        return None
    F = float(chain["F"].iloc[0])
    ch = chain.dropna(subset=["strike"]).sort_values("strike")
    # nearest listed strike's OTM IV (flat between strikes — good enough off-surface)
    ch = ch.iloc[(ch["strike"] - K).abs().argsort()[:1]]
    row = ch.iloc[0]
    iv = row.get("put_iv") if K <= F else row.get("call_iv")
    if iv is None or not (isinstance(iv, float) and math.isfinite(iv) and iv > 0):
        iv = row.get("call_iv") if K <= F else row.get("put_iv")
    return float(iv) if iv and math.isfinite(iv) and iv > 0 else None


# ── Arbitrary-date pricing off the fitted surface (Rajat 2026-08-04: "once you
# have the surface you can price a 1w or 6w using that surface" — no weekly-
# option downloads). F* linear in days between bracketing expiries' F; vol read
# at MATCHED moneyness on each bracketing expiry (strike shifted so it sits at
# the same moneyness vs that expiry's own F), then TOTAL VARIANCE σ²T linear in
# T; flat vol before the first expiry, flat-forward-vol beyond the last.
def _interp_meta(src, mkt, surf, target: date):
    exps = sorted(surf.keys())
    if not exps:
        return None, None, None
    td = (target - date.today()).days
    if td <= 0:
        return None, None, None
    T = td / 365.0
    r = float(surf[exps[0]]["r"])
    ds_ = [(e - date.today()).days for e in exps]
    Fs = [float(surf[e]["F"]) for e in exps]
    if td <= ds_[0]:
        F = Fs[0]
    elif td >= ds_[-1]:
        F = Fs[-1]
    else:
        import numpy as _np
        F = float(_np.interp(td, ds_, Fs))
    return F, T, r


def _leg_iv_interp(src, surf, target: date, K: float, Fstar: float):
    exps = sorted(surf.keys())
    td = (target - date.today()).days

    def _iv_at(e):
        Fi = float(surf[e]["F"])
        Ki = (Fi + (K - Fstar)) if src == "rates" else Fi * (K / Fstar)
        return (_ro._iv_from_surface_rates(surf, e, Ki) if src == "rates"
                else _ov2._iv_from_surface(surf, e, Ki))

    lo = [e for e in exps if (e - date.today()).days <= td]
    hi = [e for e in exps if (e - date.today()).days >= td]
    T = td / 365.0
    if not lo:                                   # before first listed expiry: flat vol
        return _iv_at(hi[0])
    if not hi:                                   # beyond last: flat FORWARD vol
        e1 = lo[-1]
        iv1 = _iv_at(e1)
        if iv1 is None:
            return None
        T1 = (e1 - date.today()).days / 365.0
        w = iv1 * iv1 * T1 + iv1 * iv1 * (T - T1)
        return math.sqrt(w / T)
    e1, e2 = lo[-1], hi[0]
    if e1 == e2:
        return _iv_at(e1)
    iv1, iv2 = _iv_at(e1), _iv_at(e2)
    if iv1 is None or iv2 is None:
        return iv1 or iv2
    T1 = (e1 - date.today()).days / 365.0
    T2 = (e2 - date.today()).days / 365.0
    w1, w2 = iv1 * iv1 * T1, iv2 * iv2 * T2
    w = w1 + (w2 - w1) * (T - T1) / max(T2 - T1, 1e-9)
    return math.sqrt(max(w, 1e-10) / T)


def _expiry_meta(src, mkt, surf, tdate, expiry):
    """(F, T, r) for the expiry — surface entry, else from its chain."""
    if surf and expiry in surf:
        s = surf[expiry]
        return float(s["F"]), float(s["T"]), float(s["r"])
    chain = _chain_smile_for_expiry(src, mkt, tdate, expiry)
    if chain is None or chain.empty:
        return None, None, None
    m = (_ro._MARKETS_RATES if src == "rates" else _ov2._MARKETS_V2)[mkt]
    return (float(chain["F"].iloc[0]), float(chain["T"].iloc[0]),
            float(m.get("r", 0.045)))


def price_structure(src: str, mkt: str, expiry: date, legs: list, lots: int,
                    interp: bool = False, live: bool = True):
    """Price one structure. `interp=True` prices an ARBITRARY date off the fitted
    surface (time-interpolated). `live=True` shifts the forward to yahoo's ~15min
    quote (vols stay settlement — sticky-moneyness). Returns dict with 'err'."""
    surf, tdate, _alle, err = _surface_for(src, mkt)
    if err and surf is None:
        return {"err": err}
    if interp:
        F, T, r = _interp_meta(src, mkt, surf, expiry)
    else:
        F, T, r = _expiry_meta(src, mkt, surf, tdate, expiry)
        if F is None and surf:
            # listed date with no loaded chain data (e.g. a weekly whose
            # settlements we deliberately don't download) → surface interp
            interp = True
            F, T, r = _interp_meta(src, mkt, surf, expiry)
    if F is None or not T or T <= 0:
        return {"err": "no F/T for this expiry"}
    # live-forward shift: settlement smile, live F (sticky-moneyness — the strike
    # is read on the settlement surface at its moneyness vs the SETTLEMENT F)
    F_set, is_live = F, False
    if live:
        q = _live_shift(mkt)
        if q:
            last, prev = q
            F = F + (last - prev) if src == "rates" else F * (last / prev)
            is_live = abs(F - F_set) > 1e-12

    def _k_read(K):
        if not is_live:
            return K
        return K - (F - F_set) if src == "rates" else K * (F_set / F)

    greeks_fn = _ro._bachelier_greeks if src == "rates" else _ov2._black76_greeks
    leg_rows, tot = [], {k: 0.0 for k in ("price", "delta", "vega", "theta")}
    for qty, cp, K in legs:
        Kr = _k_read(K)
        iv = (_leg_iv_interp(src, surf, expiry, Kr, F_set) if interp
              else _leg_iv(src, mkt, surf, tdate, expiry, Kr))
        if iv is None and not interp and surf:
            iv = _leg_iv_interp(src, surf, expiry, Kr, F_set)
        if iv is None:
            return {"err": f"no IV for strike {K:g} @ {expiry}"}
        g = greeks_fn(F, K, T, r, iv, cp)
        for k in tot:
            tot[k] += qty * g[k]
        leg_rows.append(dict(qty=qty, cp=cp, K=K, iv=iv, prem=g["price"],
                             delta=g["delta"]))
    mult = _MULT.get(mkt, 1.0)
    # ATM IV at this expiry (read at the SETTLEMENT forward point — under a live
    # shift the ATM strike F_live maps back to F_set on the settlement smile)
    atm_iv = (_leg_iv_interp(src, surf, expiry, F_set, F_set) if interp
              else _leg_iv(src, mkt, surf, tdate, expiry, F_set))
    dv01 = None
    if src == "rates":
        try:
            dv01, _ok = _vd._market_dv01(mkt, _ro._MARKETS_RATES[mkt])
            if not (_ok and dv01 and math.isfinite(dv01) and dv01 > 0):
                dv01 = None
        except Exception:
            dv01 = None
    if atm_iv is None:
        atm_disp = "—"
    elif src == "rates":
        atm_disp = f"{atm_iv / dv01:.1f}bp" if dv01 else f"{atm_iv:.2f}pt"
    else:
        atm_disp = f"{atm_iv * 100:.2f}%"
    # $ delta: rates per 1bp YIELD move (Δ/pt × dv01 pts/bp — Rajat 2026-08-04),
    # everything else per 1.0 price point
    if src == "rates" and dv01:
        delta_usd = tot["delta"] * mult * lots * dv01
        delta_unit = "/bp"
    else:
        delta_usd = tot["delta"] * mult * lots
        delta_unit = "/pt"
    return {
        "err": None, "F": F, "T": T, "r": r, "tdate": tdate, "mult": mult,
        "dv01": dv01,                     # underlying future, pts per bp
        "interp": bool(interp), "live": is_live, "f_settle": F_set,
        "legs": leg_rows,
        "atm_disp": atm_disp,
        "prem_pts": tot["price"],
        "prem_usd": tot["price"] * mult * lots,
        "delta_pct": tot["delta"] * 100.0,
        "delta_usd": delta_usd, "delta_unit": delta_unit,
        "theta_usd": tot["theta"] * mult * lots,        # $ per calendar day
        "vega_usd": tot["vega"] * mult * lots,          # $ per 1pp IV move
    }


# ── Futures lines (Rajat 2026-08-18: "simply add a futures — select Future,
# input lots, see the delta; the rest of the columns stay blank"). Delta-only
# rows: $Δ = lots × mult (×dv01 → $/bp for rates); prem/θ/vega/ATM blank. ────
def price_future(src: str, mkt: str, lots: int, live: bool = True):
    surf, tdate, _alle, err = _surface_for(src, mkt)
    if err and surf is None:
        return {"err": err}
    if not surf:
        return {"err": "no surface forward for this market"}
    F = float(surf[sorted(surf.keys())[0]]["F"])
    F_set, is_live = F, False
    if live:
        q = _live_shift(mkt)
        if q:
            last, prev = q
            F = F + (last - prev) if src == "rates" else F * (last / prev)
            is_live = abs(F - F_set) > 1e-12
    mult = _MULT.get(mkt, 1.0)
    dv01 = None
    if src == "rates":
        try:
            dv01, _ok = _vd._market_dv01(mkt, _ro._MARKETS_RATES[mkt])
            if not (_ok and dv01 and math.isfinite(dv01) and dv01 > 0):
                dv01 = None
        except Exception:
            dv01 = None
    if src == "rates" and dv01:
        delta_usd, unit = mult * lots * dv01, "/bp"
    else:
        delta_usd, unit = mult * lots, "/pt"
    return {"err": None, "F": F, "T": None, "r": None, "tdate": tdate,
            "mult": mult, "dv01": dv01, "interp": False, "live": is_live,
            "f_settle": F_set,
            "legs": [], "atm_disp": "—", "prem_pts": 0.0, "prem_usd": 0.0,
            "delta_pct": None, "delta_usd": delta_usd, "delta_unit": unit,
            "theta_usd": 0.0, "vega_usd": 0.0}


# ── Scenario engine (phase 2, Rajat 2026-08-04): reprice ONE structure's stored
# legs under shifted F / vol / time. Sticky-strike: each leg keeps ITS OWN fitted
# IV (scaled by the vol bump) — standard quick-scenario mechanics, disclosed in
# the caption. Pure pricers (no greeks) for speed.
def _scn_value(src: str, legs: list, F: float, T: float, r: float,
               vol_mult: float = 1.0) -> float:
    """Structure value (pts) at (F, T, iv×vol_mult). legs = res['legs'] dicts."""
    pricer = _ro._bachelier if src == "rates" else _ov2._b76
    T = max(T, 1e-6)
    v = 0.0
    for lg in legs:
        v += lg["qty"] * pricer(F, lg["K"], T, r,
                                max(lg["iv"] * vol_mult, 1e-6), lg["cp"])
    return v


def _render_scenario(b: dict) -> None:
    import numpy as np
    import plotly.graph_objects as go
    r_ = b["res"]
    src, mkt = b["src"], b["mkt"]
    legs = [dict(qty=lg["qty"], cp=lg["cp"], K=lg["K"], iv=lg["iv"])
            for lg in r_["legs"]]
    F0, T0 = float(r_["F"]), float(r_["T"])
    rr = float(r_.get("r", 0.045))
    mult, lots = float(r_["mult"]), b["lots"]
    v0 = _scn_value(src, legs, F0, T0, rr)
    ivs = [lg["iv"] for lg in legs]
    iv_ref = ivs[0]
    sig_move = (iv_ref * math.sqrt(T0) if src == "rates"
                else F0 * iv_ref * math.sqrt(T0))    # 1σ move to expiry, pts

    st.markdown(
        "<div style='background:#5B4FC7;color:#FFFFFF;padding:6px 12px;"
        "font-size:13px;font-weight:700;border-radius:9px;display:inline-block;"
        f"margin-top:10px'>🎬 Scenario — {b['mlbl']} · {b['exp']} · "
        f"{lots} lots · {b['desc']}</div>", unsafe_allow_html=True)

    def _pl(F, T, vm=1.0):
        return (_scn_value(src, legs, F, T, rr, vm) - v0) * mult * lots

    # ── 1. P&L vs underlying (now / half-time / expiry) ──────────────────────
    # STRUCTURE-AWARE range (Rajat 2026-08-05: a 0.0625-spaced SR3 condor was a
    # blip inside a ±2.5σ window and looked mispriced): multi-strike structures
    # get a window around their strikes; single strikes keep the ±2.5σ view.
    _kmin = min(lg["K"] for lg in legs)
    _kmax = max(lg["K"] for lg in legs)
    _span = _kmax - _kmin
    if _span > 0:
        _pad = 0.6 * max(_span, sig_move)
        _lo = min(_kmin, F0) - _pad
        _hi = max(_kmax, F0) + _pad
    else:
        _lo, _hi = F0 - 2.5 * sig_move, F0 + 2.5 * sig_move
    Fs = np.linspace(_lo, _hi, 161)
    fig = go.Figure()
    for T_, lbl, col, dash in ((T0, "today", "#2563EB", "solid"),
                               (T0 / 2, "half-time", "#64748B", "dash"),
                               (1e-6, "expiry", "#0D9488", "solid")):
        fig.add_trace(go.Scatter(
            x=Fs, y=[_pl(float(f), T_) for f in Fs], mode="lines", name=lbl,
            line=dict(color=col, width=2 if lbl != "half-time" else 1.5,
                      dash=dash),
            hovertemplate="F %{x:,.6~f} · $%{y:,.0f}<extra>" + lbl + "</extra>"))
    fig.add_hline(y=0, line_color="#94A3B8", line_width=1)
    fig.add_vline(x=F0, line_color="#1E293B", line_width=1.2,
                  annotation_text=f"F {_vd._strike_fmt(F0)}",
                  annotation_font_size=10)
    for lg in legs:
        fig.add_vline(x=lg["K"], line_color="#E2E8F0", line_width=1,
                      line_dash="dot")
    fig.update_layout(
        title=dict(text="P&L vs underlying (vols unchanged, sticky-strike)",
                   font_size=13),
        height=360, margin=dict(l=30, r=30, t=44, b=30), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1,
                    xanchor="right", font_size=10),
        xaxis=dict(gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9", tickprefix="$", tickformat=",.0f"),
        plot_bgcolor="#FFFFFF")
    st.plotly_chart(fig, use_container_width=True)

    # ── 2. spot × vol P&L matrix at a CHOSEN valuation date (Rajat 2026-08-04:
    # "today" wasn't enough — any date from tomorrow to expiry) ──────────────
    from datetime import timedelta as _td_
    _exp_d = b["exp"] if isinstance(b["exp"], date) else date.fromisoformat(str(b["exp"]))
    _min_d = date.today() + _td_(days=1)
    _max_d = max(_exp_d, _min_d)
    _mtx_d = st.date_input(
        "Matrix valuation date", key="_pr_scn_mtxd",
        value=min(date.today() + _td_(days=7), _max_d),
        min_value=_min_d, max_value=_max_d,
        help="Value the spot×vol matrix as of this date (up to and including "
             "expiry — at expiry it's the intrinsic payoff).")
    _dfwd = (_mtx_d - date.today()).days
    T_mtx = max(T0 - _dfwd / 365.0, 1e-6)
    spot_levels = np.linspace(_lo, _hi, 9)               # structure-aware window
    vol_shifts = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    z = [[_pl(float(sl), T_mtx, 1.0 + vs) for sl in spot_levels]
         for vs in vol_shifts]
    fig2 = go.Figure(go.Heatmap(
        z=z,
        x=[_vd._strike_fmt(float(sl)) for sl in spot_levels],
        y=[f"{v:+.0%} vol" for v in vol_shifts],
        colorscale=[[0.0, "#DC2626"], [0.5, "#FFFFFF"], [1.0, "#16A34A"]],
        zmid=0.0, showscale=False,
        text=[[f"{val / 1000:,.1f}k" if abs(val) >= 1000 else f"{val:,.0f}"
               for val in row] for row in z],
        texttemplate="%{text}", textfont_size=10,
        hovertemplate="F %{x} · %{y} · $%{z:,.0f}<extra></extra>"))
    fig2.update_layout(
        title=dict(text=f"P&L matrix — underlying level (structure-scaled) × parallel vol "
                        f"shift, valued {_mtx_d} (+{_dfwd}d"
                        + (", expiry — intrinsic" if _mtx_d >= _exp_d else "")
                        + ") ($)",
                   font_size=13),
        height=300, margin=dict(l=30, r=30, t=44, b=30),
        xaxis=dict(side="bottom"), plot_bgcolor="#FFFFFF")
    st.plotly_chart(fig2, use_container_width=True)

    # ── 3. time decay at constant F ──────────────────────────────────────────
    days = np.linspace(0, T0 * 365.0, 60)
    fig3 = go.Figure()
    for vm, lbl, col in ((1.0, "flat vol", "#B45309"),
                         (1.1, "vol +10%", "#94A3B8")):
        fig3.add_trace(go.Scatter(
            x=days, y=[_pl(F0, T0 - d / 365.0, vm) for d in days],
            mode="lines", name=lbl,
            line=dict(color=col, width=2 if vm == 1.0 else 1.2,
                      dash="solid" if vm == 1.0 else "dot"),
            hovertemplate="+%{x:.0f}d · $%{y:,.0f}<extra>" + lbl + "</extra>"))
    fig3.add_hline(y=0, line_color="#94A3B8", line_width=1)
    fig3.update_layout(
        title=dict(text="Time decay — P&L at unchanged underlying", font_size=13),
        height=300, margin=dict(l=30, r=30, t=44, b=30), showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1,
                    xanchor="right", font_size=10),
        xaxis=dict(title="calendar days forward", gridcolor="#F1F5F9"),
        yaxis=dict(gridcolor="#F1F5F9", tickprefix="$", tickformat=",.0f"),
        plot_bgcolor="#FFFFFF")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Sticky-strike mechanics: each leg keeps its fitted settlement IV "
        "(scaled by the vol shift); no smile re-read as spot moves, no vol-path "
        "assumptions in the decay chart. Spot axis spans ±2.5σ of the "
        "move-to-expiry; matrix vol shifts are parallel and relative.")
    if st.button("✕ Close scenario", key="_pr_scn_close"):
        st.session_state.pop("_pr_scn", None)
        st.rerun()


# ── UI — line-pricer rebuild (Rajat 2026-08-18: "getting complicated to add
# and track — full rebuild. Line pricer: select Option and the next columns
# populate (asset class, market, …); on top an option to add more lines.").
# Saved structures + portfolios REMOVED (pricer_saved.json /
# pricer_portfolios.json left on disk, unused). Every line reprices off the
# cached surfaces + live forwards on each rerun; scenario kept (one option
# line at a time, picked below the table). ───────────────────────────────────


def _fmt_money(v):
    if v is None or not math.isfinite(v):
        return "—"
    return f"−${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


_LINE_COLS = [0.72, 0.9, 1.35, 1.3, 0.52, 1.9, 0.26]


# ── Saved line sets ("portfolios", Rajat 2026-08-18: "save one or more lines
# like a portfolio"). The CURRENT lines are the portfolio — 💾 saves them all
# under a name (same name = overwrite), ↺ Load replaces the lines by
# pre-seeding the widget keys. pricer_linesets.json. ─────────────────────────
def _store_fp(name: str) -> str:
    import os
    return os.path.join(os.path.dirname(__file__), name)


def _store_load(fname: str) -> list:
    import json
    import os
    fp = _store_fp(fname)
    try:
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return []


def _store_write(fname: str, items: list) -> None:
    import json
    try:
        with open(_store_fp(fname), "w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=1)
    except Exception:
        pass


# TWO stores (Rajat 2026-08-19: "there will be 2 databases"):
#   portfolios — curated, generally same-asset-class views (Add ticked)
#   blotters   — ad-hoc dated snapshots of the whole screen (Save blotter)
def _ls_load() -> list:
    return _store_load("pricer_linesets.json")


def _ls_write(items: list) -> None:
    _store_write("pricer_linesets.json", items)


def _bl_load() -> list:
    # one-time migration: auto-named "blotter …" entries saved into the
    # portfolios store before the split move over here
    bl = _store_load("pricer_blotters.json")
    ls = _ls_load()
    strays = [s for s in ls if s.get("name", "").startswith("blotter ")]
    if strays:
        bl.extend(strays)
        _store_write("pricer_blotters.json", bl)
        _ls_write([s for s in ls if not s.get("name", "").startswith("blotter ")])
    return bl


def _bl_write(items: list) -> None:
    _store_write("pricer_blotters.json", items)


def _ls_apply(ls: dict, uni: dict):
    """Replace the current lines with a saved set by pre-seeding the line
    widgets' session keys. Returns (stubs, notes)."""
    import uuid as _uuid
    from datetime import timedelta
    ss = st.session_state
    # restore the daily/weekly-expiries toggle the set was saved with
    # (pending-key deferral — the checkbox is instantiated above the Load
    # buttons, so its key can't be written directly here)
    if "show_all" in ls:
        ss["_pr_showall_pend"] = bool(ls["show_all"])
    stubs, notes = [], []
    for ln in ls.get("lines", []):
        cls = ln.get("cls")
        if cls not in uni:
            notes.append(f"{ln.get('mkt')}: asset class gone — skipped")
            continue
        j = next((k for k, o in enumerate(uni[cls]) if o[1] == ln["mkt"]), None)
        if j is None:
            notes.append(f"{ln.get('mkt')}: market gone — skipped")
            continue
        lid = _uuid.uuid4().hex[:6]
        ss[f"_prl_t_{lid}"] = "Future" if ln.get("kind") == "fut" else "Option"
        ss[f"_prl_c_{lid}"] = cls
        ss[f"_prl_m_{lid}_{cls}"] = j
        ss[f"_prl_l_{lid}"] = int(ln.get("lots", 1))
        stub = {"id": lid}
        if ln.get("kind") != "fut":
            ss[f"_prl_d_{lid}"] = ln.get("desc", "")
            if ln.get("exp"):
                d = date.fromisoformat(ln["exp"])
                if date.today() < d <= date.today() + timedelta(days=400):
                    # hand the date to _render_line, which picks it in the
                    # expiry DROPDOWN when it's a listed expiry and only
                    # falls back to the custom-date path for non-listed
                    # dates (Rajat 2026-08-19: a listed Bund expiry showed
                    # as "custom date" — confusing)
                    stub["seed_exp"] = ln["exp"]
                else:
                    notes.append(f"{ln['mkt']} {ln.get('desc', '')}: expiry "
                                 f"{ln['exp']} passed — pick a new one")
        stubs.append(stub)
    return stubs, notes


def _render_line(i: int, ln: dict, uni: dict, use_live: bool,
                 show_all: bool = False):
    """One pricer line: cascading widgets → priced entry dict.
    Returns (entry, remove_clicked)."""
    from datetime import timedelta
    lid = ln["id"]
    c = st.columns(_LINE_COLS)
    lv = "visible" if i == 0 else "collapsed"     # first row shows the labels
    typ = c[0].selectbox("Type", ["Option", "Future"],
                         key=f"_prl_t_{lid}", label_visibility=lv)
    cls = c[1].selectbox("Class", list(uni),
                         key=f"_prl_c_{lid}", label_visibility=lv)
    opts = uni[cls]
    mi = c[2].selectbox("Market", range(len(opts)),
                        format_func=lambda j, o=opts: o[j][2],
                        key=f"_prl_m_{lid}_{cls}", label_visibility=lv)
    src, mkt, mlbl = opts[mi]
    entry = dict(line=i + 1, lid=lid, src=src, mkt=mkt, mlbl=mlbl, cls=cls)

    surf, tdate, all_exps, serr = _surface_for(src, mkt)
    no_surface = serr and surf is None

    if typ == "Future":
        c[3].selectbox("Expiry", ["— front"], key=f"_prl_ef_{lid}",
                       disabled=True, label_visibility=lv)
        lots = c[4].number_input("Lots", value=1, step=1,
                                 key=f"_prl_l_{lid}", label_visibility=lv,
                                 help="negative = short")
        c[5].text_input("Structure", value="future — lots only",
                        key=f"_prl_df_{lid}", disabled=True,
                        label_visibility=lv)
        if no_surface:
            entry["err"] = serr
        elif int(lots) == 0:
            entry["err"] = "lots = 0"
        else:
            res = price_future(src, mkt, int(lots), live=use_live)
            entry.update(res=res, err=res.get("err"), lots=int(lots),
                         kind="fut", exp=None,
                         desc=f"future ({'long' if lots > 0 else 'short'})")
    else:
        if no_surface:
            c[3].selectbox("Expiry", ["—"], key=f"_prl_e0_{lid}",
                           disabled=True, label_visibility=lv)
            c[4].number_input("Lots", value=1, step=1, key=f"_prl_l_{lid}",
                              label_visibility=lv)
            c[5].text_input("Structure", key=f"_prl_d_{lid}", disabled=True,
                            label_visibility=lv)
            entry["err"] = f"{serr} — load Vol Dash once today"
        else:
            std = sorted(surf.keys()) if surf else []
            has_data = set(all_exps) | set(std)
            if show_all:      # dailies/weeklies + every listed expiry
                wk = _weekly_expiries(src, mkt, tdate)
                exps = sorted(set(all_exps) | set(std) | set(wk))
            else:             # default: standard fitted-surface expiries only
                exps = std or sorted(all_exps)

            def _efmt(x):
                if isinstance(x, str):
                    return x
                return (f"{x} ({(x - date.today()).days}d)"
                        + ("" if x in has_data else " ≈"))

            # one-shot expiry seed from a portfolio/blotter load: pick the
            # saved date in the dropdown when listed, custom path otherwise
            _seed = ln.pop("seed_exp", None)
            if _seed:
                _sd = date.fromisoformat(_seed)
                _ekey = f"_prl_e_{lid}_{mkt}"
                _wk_seed = ([] if show_all else
                            _weekly_expiries(src, mkt, tdate))
                if _sd in exps:
                    st.session_state[_ekey] = _sd
                elif _sd in all_exps or _sd in _wk_seed:
                    # listed (or a known daily/weekly) but hidden by the
                    # toggle → keep it a dropdown date, never "custom"
                    st.session_state[_ekey] = _sd
                    exps = sorted(set(exps) | {_sd})
                else:
                    st.session_state[_ekey] = "custom date ≈"
                    st.session_state[f"_prl_ecd_{lid}"] = _sd

            exp_sel = c[3].selectbox("Expiry", exps + ["custom date ≈"],
                                     key=f"_prl_e_{lid}_{mkt}",
                                     format_func=_efmt, label_visibility=lv)
            if isinstance(exp_sel, str):
                exp = c[3].date_input(
                    "date", value=date.today() + timedelta(days=14),
                    min_value=date.today() + timedelta(days=1),
                    max_value=date.today() + timedelta(days=400),
                    key=f"_prl_ecd_{lid}", label_visibility="collapsed")
                interp = True
            else:
                exp, interp = exp_sel, exp_sel not in has_data
            lots = c[4].number_input("Lots", value=1, step=1,
                                     key=f"_prl_l_{lid}", label_visibility=lv,
                                     help="negative = short")
            desc = c[5].text_input(
                "Structure", key=f"_prl_d_{lid}", label_visibility=lv,
                placeholder="7500/7600 cs · 96.25 straddle · sell 112 p")
            if not (desc or "").strip():
                entry["pending"] = True
            else:
                legs, perr = parse_structure(desc)
                if perr:
                    entry["err"] = perr
                else:
                    res = price_structure(src, mkt, exp, legs, int(lots),
                                          interp=interp, live=use_live)
                    entry.update(res=res, err=res.get("err"), lots=int(lots),
                                 desc=desc.strip(), exp=exp, legs=legs,
                                 kind="opt")
    c[6].markdown("<div style='height:28px'></div>" if i == 0 else "",
                  unsafe_allow_html=True)
    rm = c[6].button("✕", key=f"_prl_x_{lid}", help="remove this line")
    return entry, rm


def render_pricer():
    import uuid as _uuid
    # Indigo-Studio identity (Rajat 2026-08-19: E+F blend from the design board)
    st.markdown(
        "<div style='display:inline-flex;align-items:center;gap:10px;"
        "margin-bottom:6px'>"
        "<span style='background:#5B4FC7;color:#FFFFFF;border-radius:9px;"
        "width:32px;height:32px;display:inline-flex;align-items:center;"
        "justify-content:center;font-size:15px'>💲</span>"
        "<span style='font-size:16.5px;font-weight:700;color:#191731'>Pricer</span>"
        "<span style='font-family:ui-monospace,Consolas,monospace;font-size:11px;"
        "color:#6E6A8F'>line pricer · settlement vol surfaces · live forwards"
        "</span></div>", unsafe_allow_html=True)
    # warm the weekly-dates library once per session-day (background, defs-only)
    _warm_key = f"_pr_wk_warm_{date.today().isoformat()}"
    if not st.session_state.get(_warm_key):
        st.session_state[_warm_key] = True
        import threading
        threading.Thread(target=_wk_warm_all, daemon=True).start()

    uni = _universe()
    lines = st.session_state.setdefault("_pr_lines", [])
    if not lines:
        lines.append({"id": _uuid.uuid4().hex[:6]})

    t1, t2, t3, t5, t4 = st.columns([0.9, 1.0, 0.9, 1.1, 2.5])
    # NB no st.rerun() in these toolbar handlers: an early rerun aborts the
    # script BEFORE the line widgets below render, and Streamlit culls the
    # state of un-rendered widgets — every line was resetting to defaults
    # ("lines become S&P 500", Rajat 2026-08-19). The click itself is
    # already a fresh run; mutations here are picked up by the loop below.
    if t1.button("➕ Add line", type="primary", key="_pr_addline"):
        lines.append({"id": _uuid.uuid4().hex[:6]})
    if t2.button("🔄 Refresh live", key="_pr_refresh",
                 help="clears the live-quote cache; all lines reprice"):
        _live_shift.clear()
    if t3.button("🗑 Clear lines", key="_pr_clearlines"):
        st.session_state["_pr_lines"] = []
        st.session_state.pop("_pr_scn", None)
        st.rerun()
    with t5.popover("💾 Save blotter", use_container_width=True):
        st.caption("saves ALL currently priced lines, as-is, under a name "
                   "(same name = overwrite); load it back from the "
                   "Portfolios row below")
        _blot_nm = st.text_input("Name", key="_pr_blot_nm",
                                 placeholder="blank = auto-dated name")
        _blot_go = st.button("Save", key="_pr_blot_go", type="primary")
    use_live = t4.checkbox(
        "live underlying (yahoo, ~15min delay) — settlement vols, live forward",
        value=True, key="_pr_live")
    _sa_pend = st.session_state.pop("_pr_showall_pend", None)
    if _sa_pend is not None:
        st.session_state["_pr_showall"] = bool(_sa_pend)
    show_all = t4.checkbox(
        "include daily/weekly expiries", value=False, key="_pr_showall",
        help="adds every listed expiry incl. dailies/weeklies to the expiry "
             "dropdowns (marked ≈ = surface-interpolated). Off = the "
             "standard fitted-surface expiries only.")
    with st.expander("ℹ structure syntax", expanded=False):
        st.markdown(_STRUCTS_HELP)

    # ── the lines ─────────────────────────────────────────────────────────────
    # scoped restyle (Rajat 2026-08-19: line dropdowns' font clashed with the
    # table) — st.container(key=) stamps .st-key-pr_lines on this block only,
    # so the CSS can't leak into other tabs' widgets
    st.markdown(
        "<style>"
        ".st-key-pr_lines div[data-baseweb='select'] > div"
        "{font-size:12.5px;min-height:34px;}"
        ".st-key-pr_lines .stTextInput input,"
        ".st-key-pr_lines .stNumberInput input,"
        ".st-key-pr_lines .stDateInput input{font-size:12.5px;}"
        ".st-key-pr_lines [data-testid='stWidgetLabel'] p"
        "{font-size:10px;font-family:ui-monospace,Consolas,monospace;"
        "text-transform:uppercase;letter-spacing:.07em;color:#5B4FC7;"
        "font-weight:600;}"
        "</style>", unsafe_allow_html=True)
    entries, rm_id = [], None
    with st.container(key="pr_lines"):
        for i, ln in enumerate(lines):
            entry, rm = _render_line(i, ln, uni, use_live, show_all)
            if rm:
                rm_id = ln["id"]
            entries.append(entry)
    if rm_id is not None:
        st.session_state["_pr_lines"] = [l for l in lines if l["id"] != rm_id]
        st.rerun()

    # ── results table ─────────────────────────────────────────────────────────
    rows = []
    for e in entries:
        r_ = e.get("res")
        if e.get("pending"):
            rows.append({"#": e["line"], "Market": e["mlbl"],
                         "Structure": "· enter a description ·",
                         "Expiry": "", "Days": "", "Lots": "", "Fwd": "", "ATM": "",
                         "Prem pts": "", "Prem $": "", "Δ %": "", "Δ $": "",
                         "θ $/d": "", "Vega $": ""})
            continue
        if e.get("err") or not r_:
            rows.append({"#": e["line"], "Market": e["mlbl"],
                         "Structure": f"⚠ {e.get('err', 'no result')}",
                         "Expiry": "", "Days": "", "Lots": "", "Fwd": "", "ATM": "",
                         "Prem pts": "", "Prem $": "", "Δ %": "", "Δ $": "",
                         "θ $/d": "", "Vega $": ""})
            continue
        fut = e.get("kind") == "fut"
        if fut:
            _stru = e["desc"]
        else:
            _lg = " · ".join(
                f"{'+' if q > 0 else ''}{q} {k:g}{cp.lower()} "
                f"@{r_['legs'][j]['iv'] * (1 if e['src'] == 'rates' else 100):.2f}"
                for j, (q, cp, k) in enumerate(e["legs"]))
            _stru = f"{e['desc']}  [{_lg}]"
        def _wcls(v):
            # signal-wash class (design board variant E): teal = positive,
            # terracotta = negative risk; grey for zero/absent
            if v is None or not math.isfinite(v) or v == 0:
                return "w0"
            return "wp" if v > 0 else "wn"
        _fyld, _kyld = _yield_disp(e["src"], e["mkt"], r_["F"],
                                   e.get("legs") or [], fut)
        rows.append({
            "#": e["line"], "Market": e["mlbl"],
            "Expiry": ("—" if fut else
                       f"{e['exp']}{' ≈' if r_.get('interp') else ''}"),
            "Days": "—" if fut else (e["exp"] - date.today()).days,
            "Lots": e["lots"], "Structure": _stru,
            "Fwd": (f"{_vd._strike_fmt(r_['F'])}"
                    + (" (live)" if r_.get("live") else " (settle)")),
            "Fwd yld": _fyld,
            "K yld": _kyld,
            # underlying future's DV01 in ccy/bp PER CONTRACT (pts/bp ×
            # mult) — position-level $/bp is already the Δ $ column
            "DV01": (_fmt_money(r_["dv01"] * r_["mult"])
                     if r_.get("dv01") else "—"),
            "Dur": _dur_disp(e["src"], e["mkt"]),
            "ATM": "—" if fut else r_.get("atm_disp", "—"),
            # FX premium in cents (price pts × 100, e.g. 6E 0.0085 → 0.85¢)
            # — points are unreadable at FX quote scale (Rajat 2026-08-21)
            "Prem pts": ("—" if fut else
                         (f"{r_['prem_pts'] * 100:.4g}¢"
                          if e.get("cls") == "FX"
                          else f"{r_['prem_pts']:.4g}")),
            "Prem $": "—" if fut else _fmt_money(r_["prem_usd"]),
            # tenor-normalized premium: total $ prem / √(trading DAYS to
            # expiry) — e.g. 20 busdays → prem/√20 — for comparing
            # structures across expiries on the Risk/VaR daily-vol clock
            # (daily = annual/√256). NOT the BS calendar/365 T.
            "Prem $/√T": ("—" if fut else _fmt_money(
                r_["prem_usd"] / math.sqrt(
                    max(int(np.busday_count(date.today(), e["exp"])), 1)))),
            "Δ %": "—" if fut else f"{r_['delta_pct']:+.1f}",
            "Δ $": f"{_fmt_money(r_['delta_usd'])}{r_.get('delta_unit', '')}",
            "θ $/d": "—" if fut else _fmt_money(r_["theta_usd"]),
            "Vega $": "—" if fut else _fmt_money(r_["vega_usd"]),
            "_wash": {
                "Prem $": "w0" if fut else _wcls(r_["prem_usd"]),
                "Prem $/√T": "w0" if fut else _wcls(r_["prem_usd"]),
                "Δ $": _wcls(r_["delta_usd"]),
                "θ $/d": "w0" if fut else _wcls(r_["theta_usd"]),
                "Vega $": "w0" if fut else _wcls(r_["vega_usd"]),
            },
        })
    # ── results table + risk totals: one styled block (E+F blend, Rajat
    # 2026-08-19 design board — signal-washed greek cells, indigo identity,
    # segmented stat-bar totals). Rendered via click_table multi-mode so the
    # ✓ column stays row-aligned AND fully styleable (data_editor can't
    # colour cells; native checkboxes can't sit inside an HTML table). ───────
    ok = [e["res"] for e in entries if e.get("res") and not e.get("err")]
    ticked = []
    if rows:
        import html as _hesc
        _COLS = ["#", "Market", "Expiry", "Days", "Lots", "Structure",
                 "Fwd", "Fwd yld", "K yld", "DV01", "Dur", "ATM",
                 "Prem pts", "Prem $", "Prem $/√T", "Δ %", "Δ $",
                 "θ $/d", "Vega $"]
        _LEFT = {"Market", "Structure"}
        _css = (
            "<style>"
            "body{background:#F4F3FB;margin:0;}"
            ".card{background:#FFFFFF;border:1px solid #E3E0F2;"
            "border-radius:12px;padding:10px 12px;overflow-x:auto;"
            "box-shadow:0 1px 2px rgba(45,40,90,.05),"
            "0 6px 18px rgba(45,40,90,.06);}"
            "table{border-collapse:collapse;font-size:12px;width:100%;}"
            "th{color:#5B4FC7;font-size:10px;text-transform:uppercase;"
            "letter-spacing:.07em;padding:6px 8px;text-align:right;"
            "border-bottom:2px solid #5B4FC7;white-space:nowrap;"
            "font-family:ui-monospace,Consolas,monospace;}"
            "td{padding:6.5px 8px;border-bottom:1px solid #EFEDF8;"
            "text-align:right;white-space:nowrap;color:#191731;"
            "font-family:ui-monospace,Consolas,monospace;"
            "font-variant-numeric:tabular-nums;}"
            "th.al,td.al{text-align:left;}"
            "td.mkt{font-family:system-ui,'Segoe UI',sans-serif;"
            "font-weight:600;}"
            "tr:nth-child(even) td{background:#FAF9FE;}"
            "tr.ct-on td{background:#EFEBFF;}"
            "tr.ct-on td:first-child{box-shadow:inset 3px 0 0 #5B4FC7;}"
            "td.sel{cursor:pointer;color:#B9B4D8;font-size:17px;"
            "line-height:1;text-align:center;width:34px;}"
            "td.sel.ct-sel{color:#5B4FC7;}"
            "td.wp{background:#E9F7F5 !important;color:#0F766E;"
            "font-weight:600;}"
            "td.wn{background:#FDEFE9 !important;color:#C2410C;"
            "font-weight:600;}"
            "td.w0{color:#A6A3C0;}"
            ".tot{margin-top:10px;display:flex;flex-wrap:wrap;gap:0;"
            "border:1px solid #E3E0F2;border-radius:10px;overflow:hidden;"
            "background:#FFFFFF;width:max-content;max-width:100%;}"
            ".tcell{padding:7px 16px;border-right:1px solid #EFEDF8;}"
            ".tcell:last-child{border-right:none;}"
            ".tcell span{display:block;font-size:9px;text-transform:"
            "uppercase;letter-spacing:.08em;color:#8C88AC;"
            "font-family:ui-monospace,Consolas,monospace;}"
            ".tcell b{font-family:ui-monospace,Consolas,monospace;"
            "font-size:14px;color:#191731;}"
            ".tcell.p{background:#E9F7F5;}.tcell.p b{color:#0F766E;}"
            ".tcell.n{background:#FDEFE9;}.tcell.n b{color:#C2410C;}"
            ".note{margin-top:7px;font-size:10.5px;color:#8C88AC;"
            "font-family:ui-monospace,Consolas,monospace;}"
            "</style>")
        _head = ("<tr><th title='tick lines to add to a portfolio / run "
                 "the scenario'>✓</th>"
                 + "".join(f"<th class='al'>{c}</th>" if c in _LEFT
                           else f"<th>{c}</th>" for c in _COLS) + "</tr>")
        _trs = []
        for rw, e in zip(rows, entries):
            w = rw.get("_wash", {})
            tds = [f"<td class='sel' data-key='{e['lid']}' data-on='■'>□"
                   "</td>"]
            for c in _COLS:
                v = str(rw.get(c, ""))
                cls = []
                if c in _LEFT:
                    cls.append("al")
                if c == "Market":
                    cls.append("mkt")
                if c in w:
                    cls.append(w[c])
                elif v in ("—", ""):
                    cls.append("w0")
                tds.append(f"<td class='{' '.join(cls)}'>"
                           f"{_hesc.escape(v)}</td>")
            _trs.append("<tr>" + "".join(tds) + "</tr>")
        _tot = ""
        if ok:
            d_pt = sum(r["delta_usd"] for r in ok
                       if r.get("delta_unit", "/pt") == "/pt")
            d_bp = sum(r["delta_usd"] for r in ok
                       if r.get("delta_unit") == "/bp")

            def _tc(lbl, v, sfx=""):
                cls = " p" if v > 0 else (" n" if v < 0 else "")
                return (f"<div class='tcell{cls}'><span>{lbl}</span>"
                        f"<b>{_fmt_money(v)}{sfx}</b></div>")
            _tot = "<div class='tot'>" + _tc(
                "net prem", sum(r["prem_usd"] for r in ok))
            if any(r.get("delta_unit", "/pt") == "/pt" for r in ok):
                _tot += _tc("Δ eq/fx/cmd", d_pt, "/pt")
            if any(r.get("delta_unit") == "/bp" for r in ok):
                _tot += _tc("Δ rates", d_bp, "/bp")
            _tot += (_tc("θ /day", sum(r["theta_usd"] for r in ok))
                     + _tc("vega /1pp", sum(r["vega_usd"] for r in ok))
                     + "</div>")
            _tot += (f"<div class='note'>{len(ok)}/{len(entries)} "
                     "line(s) priced</div>")
        _tbl_html = (_css + "<div class='card'><table>" + _head
                     + "".join(_trs) + "</table></div>" + _tot)
        _cur_lids = {e["lid"] for e in entries}
        _prev_sel = [k for k in (st.session_state.get("_pr_tbl_sel") or [])
                     if k in _cur_lids]
        from click_table import click_table as _ctbl
        _sel = _ctbl(_tbl_html, selected=_prev_sel, key="_pr_tbl_sel",
                     multi=True)
        _lid_ix = {e["lid"]: i for i, e in enumerate(entries)}
        ticked = sorted(_lid_ix[k] for k in (_sel or []) if k in _lid_ix)

    if ok:
        st.caption(
            "Settlement smiles (spline per expiry, flat-extrapolated); v2 "
            "Black-76, rates Bachelier — per-leg IVs in [brackets]. F (live) "
            "= settlement forward shifted to the yahoo quote. Δ$ per 1.0pt "
            "(rates per 1bp yield; futures lines = lots × multiplier); θ per "
            "calendar day; vega per 1pp IV; Δ totals kept in native units. "
            "European contracts (€/£) unconverted. ≈ = surface-interpolated "
            "date. Lines reprice on every rerun — this is a live view.")

    # ── portfolios (saved line sets) ──────────────────────────────────────────
    def _line_payload(e):
        return dict(kind=e.get("kind", "opt"), cls=e.get("cls", ""),
                    src=e["src"], mkt=e["mkt"], lots=e.get("lots", 1),
                    desc=e.get("desc", ""),
                    exp=(e["exp"].isoformat() if e.get("exp") else None))

    # deferred top-toolbar Save-blotter action (entries only exist now) —
    # writes to the BLOTTERS store, separate from portfolios
    if _blot_go:
        payload = [_line_payload(e) for e in entries
                   if e.get("res") and not e.get("err")]
        if not payload:
            st.toast("Save blotter: no priced lines", icon="⚠️")
        else:
            from datetime import datetime as _dt_
            name = (_blot_nm or "").strip() or \
                "blotter " + _dt_.now().strftime("%Y-%m-%d %H:%M")
            bls = [s for s in _bl_load() if s["name"] != name]
            bls.append({"name": name, "lines": payload,
                        "show_all": bool(show_all),
                        "ts": _dt_.now().isoformat(timespec="minutes")})
            _bl_write(bls)
            st.toast(f"blotter saved as “{name}” ({len(payload)} lines)",
                     icon="💾")

    _msg = st.session_state.pop("_pr_ls_msg", None)
    if _msg:
        st.info(_msg)
    # Semantics (Rajat 2026-08-19): the NAME BOX is for CREATING a new
    # portfolio only — always empty by default, never auto-filled. Every
    # action on an EXISTING portfolio (Replace / Load / Delete, and Add
    # ticked when the box is empty) targets the DROPDOWN selection.
    l1, l2, l3, l4, l5, l6 = st.columns([1.25, 0.85, 1.5, 0.8, 0.55, 0.4])
    nm = l1.text_input("New portfolio name", key="_pr_ls_name",
                       placeholder="only when creating a new one")
    _sets = _ls_load()
    pick = None
    if _sets:
        pick = l3.selectbox(
            "Portfolios", range(len(_sets)), key="_pr_ls_pick",
            format_func=lambda i: (f"{_sets[i]['name']} "
                                   f"({len(_sets[i]['lines'])} lines · "
                                   f"{_sets[i].get('ts', '')[:10]})"))
    l2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if l2.button("➕ Add ticked", key="_pr_ls_addtk",
                 help="adds the ✓-ticked priced lines to a portfolio: to the "
                      "NEW name on the left if you typed one, else to the "
                      "portfolio selected in the dropdown"):
        sel = [entries[i] for i in ticked
               if entries[i].get("res") and not entries[i].get("err")]
        skipped = len(ticked) - len(sel)
        name = (nm or "").strip() or \
            (_sets[pick]["name"] if pick is not None else "")
        if not name:
            st.warning("type a new portfolio name (left) — none exist yet")
        elif not sel:
            st.warning("tick ✓ on priced lines in the table first")
        else:
            from datetime import datetime as _dt_
            sets = _ls_load()
            tgt = next((s for s in sets if s["name"] == name), None)
            if tgt is None:
                tgt = {"name": name, "lines": [],
                       "ts": _dt_.now().isoformat(timespec="minutes")}
                sets.append(tgt)
            tgt["lines"].extend(_line_payload(e) for e in sel)
            tgt["show_all"] = bool(show_all)
            tgt["ts"] = _dt_.now().isoformat(timespec="minutes")
            _ls_write(sets)
            # rerun so the Portfolios dropdown (rendered above) picks up a
            # newly created set — opening a selectbox alone never reruns
            st.session_state["_pr_ls_msg"] = (
                f"added {len(sel)} line(s) to **{name}** "
                f"(now {len(tgt['lines'])} lines)"
                + (f" · {skipped} unpriced skipped" if skipped else ""))
            st.session_state.pop("_pr_ls_name", None)  # name box: create-only
            st.rerun()
    if _sets:
        l4.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if l4.button("↺ Load", key="_pr_ls_load",
                     help="replaces the current lines with this portfolio"):
            stubs, notes = _ls_apply(_sets[pick], uni)
            st.session_state["_pr_lines"] = stubs
            st.session_state.pop("_pr_scn", None)
            if notes:
                st.session_state["_pr_ls_msg"] = " · ".join(notes[:4])
            st.rerun()
        l5.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if l5.button("⟳ Replace", key="_pr_ls_repl",
                     help="OVERWRITES the dropdown-selected portfolio with "
                          "ALL currently priced lines (asks to confirm) — "
                          "the edit loop: ↺ Load, change lines, ⟳ Replace"):
            st.session_state["_pr_repl_confirm"] = _sets[pick]["name"]
        l6.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if l6.button("🗑", key="_pr_ls_del", help="delete this portfolio"):
            _ls_write([s for i, s in enumerate(_sets) if i != pick])
            st.rerun()

    # ── replace confirmation (destructive → explicit yes) ────────────────────
    _rc = st.session_state.get("_pr_repl_confirm")
    if _rc:
        _n_priced = sum(1 for e in entries
                        if e.get("res") and not e.get("err"))
        st.warning(f"Replace portfolio **{_rc}** with the {_n_priced} "
                   f"currently priced line(s)? This overwrites its contents.")
        cc1, cc2, _ccsp = st.columns([0.9, 0.6, 3.7])
        if cc1.button("✓ Yes, replace", type="primary", key="_pr_repl_yes"):
            payload = [_line_payload(e) for e in entries
                       if e.get("res") and not e.get("err")]
            st.session_state.pop("_pr_repl_confirm", None)
            sets = _ls_load()
            tgt = next((s for s in sets if s["name"] == _rc), None)
            if not payload:
                st.error("no priced lines to save — not replaced")
            elif tgt is None:
                st.error(f"portfolio “{_rc}” no longer exists")
            else:
                from datetime import datetime as _dt_
                tgt["lines"] = payload
                tgt["show_all"] = bool(show_all)
                tgt["ts"] = _dt_.now().isoformat(timespec="minutes")
                _ls_write(sets)
                st.toast(f"“{_rc}” replaced ({len(payload)} lines)", icon="🔄")
                st.rerun()
        if cc2.button("✕ Cancel", key="_pr_repl_no"):
            st.session_state.pop("_pr_repl_confirm", None)
            st.rerun()
    # ── blotters (separate store: ad-hoc dated snapshots) ────────────────────
    _bls = _bl_load()
    if _bls:
        b1, b2, b3, _bsp = st.columns([1.6, 0.6, 0.45, 2.7])
        bpick = b1.selectbox(
            "Blotters", range(len(_bls)), key="_pr_bl_pick",
            format_func=lambda i: (f"{_bls[i]['name']} "
                                   f"({len(_bls[i]['lines'])} lines)"))
        b2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if b2.button("↺ Load", key="_pr_bl_load",
                     help="replaces the current lines with this blotter"):
            stubs, notes = _ls_apply(_bls[bpick], uni)
            st.session_state["_pr_lines"] = stubs
            st.session_state.pop("_pr_scn", None)
            if notes:
                st.session_state["_pr_ls_msg"] = " · ".join(notes[:4])
            st.rerun()
        b3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if b3.button("🗑", key="_pr_bl_del", help="delete this blotter"):
            _bl_write([s for i, s in enumerate(_bls) if i != bpick])
            st.rerun()

    # ── scenario — runs on the ✓-ticked line (Rajat 2026-08-19: dropdown
    # removed; exactly ONE ticked line, must be a priced option) ─────────────
    if any(e.get("res") and not e.get("err") for e in entries):
        sc1, _scsp = st.columns([1.1, 4.3])
        if sc1.button("🎬 Run Scenario", key="_pr_scn_btn",
                      help="tick ✓ exactly ONE option line in the table, "
                           "then run — P&L vs underlying, spot×vol matrix "
                           "and time decay render below"):
            _tk = [entries[i] for i in ticked]
            if len(_tk) != 1:
                st.error(f"Run Scenario: tick exactly ONE line in the table "
                         f"({len(_tk)} ticked).")
            elif _tk[0].get("err") or not _tk[0].get("res"):
                st.error("Run Scenario: the ticked line isn't priced.")
            elif _tk[0].get("kind") == "fut":
                st.error("Scenario needs an option structure — a future is "
                         "linear (P&L = Δ$ × move).")
            else:
                import copy
                e = _tk[0]
                st.session_state["_pr_scn"] = copy.deepcopy(
                    dict(src=e["src"], mkt=e["mkt"], mlbl=e["mlbl"],
                         exp=e["exp"], lots=e["lots"], desc=e["desc"],
                         res=e["res"]))
                st.rerun()

    _scn = st.session_state.get("_pr_scn")
    if _scn:
        try:
            _render_scenario(_scn)
        except Exception as _sx:
            st.error(f"scenario failed: {type(_sx).__name__}: {str(_sx)[:140]}")

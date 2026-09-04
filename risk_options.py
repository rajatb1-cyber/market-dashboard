"""Options VaR for the Risk/VaR tab (Rajat 2026-08-24).

Two methods, selected in the dropdown beside 🎲 Run VaR Risk:

- **Delta-equivalent mapping** — each option becomes delta × lots × mult of
  its underlying future (× DV01 → $/bp for rates) and enters risk_div's
  parametric √(vᵀRv) as an extra signed row: the underlying's proxy carries
  the correlation, magnitude = |delta-equiv $risk| × underlying implied vol
  / √256. Fast, linear — understates gamma near strikes/expiry.

- **Full-revaluation historical** — each structure is repriced under the
  last ~250 daily underlying moves applied to today's forward (per-leg
  fitted IVs held sticky, T fixed at today's — a 1d horizon), giving a P&L
  distribution per position: VaR95/99 from percentiles (gamma-exact,
  condor kinks included), 1σ = std feeds √(vᵀRv) via the same proxy row.
  Vol risk is NOT captured (price risk, vol held constant).

All pricing runs off pricer.price_structure / pricer._scn_value — the same
settlement surfaces the Pricer tab shows, no new data fetches beyond the
underlying history (yfinance for equity/commod/FX, stir_bars.db for STIR
contracts, the position's saved yield proxy × DV01 for bond futures).
"""
from __future__ import annotations

_BUILD = "2026-09-04.1"   # shown in the tab — bump when this module changes

import math
import re
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
_STIR_DB = str(_HERE / "stir_bars.db")

# IBKR underlying root → (src, mkt) in the pricer universe
_ROOT_MAP = {
    "ES": ("v2", "ES"), "NQ": ("v2", "NQ"), "RTY": ("v2", "RTY"),
    "MES": ("v2", "ES"), "MNQ": ("v2", "NQ"), "M2K": ("v2", "RTY"),
    "MGC": ("v2", "GC"), "MCL": ("v2", "CL"), "M6E": ("v2", "EUR"),
    "GC": ("v2", "GC"), "SI": ("v2", "SI"), "HG": ("v2", "HG"),
    "CL": ("v2", "CL"), "COIL": ("v2", "BRN"), "MBT": ("v2", "BTC"),
    "EUR": ("v2", "EUR"), "GBP": ("v2", "GBP"), "JPY": ("v2", "JPY"),
    "AUD": ("v2", "AUD"), "CAD": ("v2", "CAD"), "CHF": ("v2", "CHF"),
    "ZT": ("rates", "TU"), "ZF": ("rates", "FV"), "ZN": ("rates", "TY"),
    "ZB": ("rates", "US"), "UB": ("rates", "UB"),
    "GBS": ("rates", "DU"), "GBM": ("rates", "OE"),
    "GBL": ("rates", "RX"), "GBX": ("rates", "UX"),
    # Eurex Flex style: Underlying = "FGBL 20261208 M" (root token + date)
    "FGBS": ("rates", "DU"), "FGBM": ("rates", "OE"),
    "FGBL": ("rates", "RX"), "FGBX": ("rates", "UX"),
    "SR3": ("rates", "SOFR"), "SO3": ("rates", "SONIA"),
    "I": ("rates", "ER"), "ER": ("rates", "ER"),
    # IB-symbol style underlyings (live Flex pulls report e.g. "SOFR3"
    # instead of the contract "SR3U6" — seen 2026-08-24)
    "SOFR3": ("rates", "SOFR"), "SONIA3": ("rates", "SONIA"),
    "EUU": ("v2", "EUR"),
    # FX FOP roots ≠ future roots (EUU/JPU/GBU — see reference_fx_fop_ibkr);
    # JPUV6 puts hit the unmapped-root skip 2026-09-04
    "JPU": ("v2", "JPY"), "GBU": ("v2", "GBP"),
}
# stir_bars.db symbol for each STIR pricer market
_STIR_DB_SYM = {"SOFR": "SR3", "SONIA": "SO3", "ER": "I"}
_MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


def _split_contract(sym: str):
    """'SR3U6' → ('SR3', '202609'); Eurex Flex style 'FGBL 20261208 M' →
    ('FGBL', '202612'). Year digit resolved to the nearest present-or-future
    decade year."""
    s = (sym or "").strip()
    me = re.match(r"^([A-Z0-9]+)\s+(\d{8})\b", s)     # Eurex: root + yyyymmdd
    if me:
        return me.group(1), me.group(2)[:6]
    m = re.match(r"^(.+?)([FGHJKMNQUVXZ])(\d{1,2})$", s)
    if not m:
        return s, None
    root, mc, yd = m.group(1), m.group(2), m.group(3)
    mon = _MONTH_CODE[mc]
    if len(yd) == 2:
        yr = 2000 + int(yd)
    else:
        yr = (date.today().year // 10) * 10 + int(yd)
        if yr < date.today().year - 1:
            yr += 10
    return root, f"{yr:04d}{mon:02d}"


def _midcurve(mkt: str, und_exp6: str | None, expiry: date) -> str:
    """STIR options on a deferred quarterly are midcurves → _1Y/_2Y market."""
    if mkt not in _STIR_DB_SYM or not und_exp6:
        return mkt
    off_m = (int(und_exp6[:4]) - expiry.year) * 12 + int(und_exp6[4:6]) - expiry.month
    if off_m >= 21:
        return f"{mkt}_2Y"
    if off_m >= 9:
        return f"{mkt}_1Y"
    return mkt


def underlying_key(sym: str) -> str:
    """Normalize a futures symbol to its underlying complex for the net-risk
    split ('SR3M6'/'SR3U6' → 'SOFR', 'MESU6' → 'ES'); unmapped → the root."""
    root, _ = _split_contract(sym)
    pair = _ROOT_MAP.get(root)
    return pair[1] if pair else root


# IB-symbol / Eurex underlying spellings → the contract root style the book's
# own futures rows use, so option deltas land on the SAME split row
_CANON_ROOT = {"SOFR3": "SR3", "SONIA3": "SO3", "EUU": "EUR", "EUUU": "EUR",
               "JPU": "JPY", "GBU": "GBP"}
_CODE_MONTH = {v: k for k, v in _MONTH_CODE.items()}


def underlying_contract(sym: str, exp6: str | None = None) -> str:
    """Canonical per-CONTRACT key for the net-risk split (Rajat 2026-08-24:
    'not SOFR but SR3M6, SR3U6'): 'SR3M6' → 'SR3M6'; 'SOFR3' + 202609 →
    'SR3U6'; 'FGBL 20261208 M' → 'FGBLZ6'. Root-only with no month → root."""
    root, e = _split_contract(sym)
    root = _CANON_ROOT.get(root, root)
    e = e or exp6
    if e:
        try:
            return f"{root}{_CODE_MONTH[int(e[4:6])]}{int(e[:4]) % 10}"
        except Exception:
            pass
    return root


def option_book(book: pd.DataFrame, sel: set | None = None):
    """Parse is_option rows → position dicts + skip notes. `sel` (the saved
    ✓ selection) filters like the futures table does; None = all options."""
    opts, notes = [], []
    if book is None or book.empty or "is_option" not in book.columns:
        return opts, notes
    for _, r in book[book["is_option"]].iterrows():
        sym = str(r["Symbol"])
        if sel is not None and sym not in sel:
            continue
        und = str(r.get("Underlying") or "")
        root, und_exp6 = _split_contract(und)
        pair = _ROOT_MAP.get(root)
        if pair is None:
            notes.append(f"{sym}: underlying root “{root}” not mapped — skipped")
            continue
        # Put/Call: Flex field → SubCategory → parse the symbol itself
        # ("ESU6 P7600" / "P OGBL 20261023 119 M") — live Web-Service pulls
        # ship the rows with the field empty (seen 2026-08-24)
        right = str(r.get("PutCall") or "").strip().upper()[:1]
        if right not in ("P", "C"):
            sc = str(r.get("SubCategory") or "").strip().upper()
            if sc in ("P", "C"):
                right = sc
        if right not in ("P", "C"):
            m = (re.search(r"(?:^|\s)([PC])\s*\d", sym.upper())
                 or re.match(r"^\s*([PC])\s", sym.upper()))
            if m:
                right = m.group(1)
        if right not in ("P", "C"):
            notes.append(f"{sym}: no Put/Call flag in Flex — skipped")
            continue
        try:
            expiry = pd.Timestamp(str(r.get("Expiry"))).date()
        except Exception:
            notes.append(f"{sym}: bad expiry “{r.get('Expiry')}” — skipped")
            continue
        # strictly-past only: an option expiring TODAY still has today's move
        # in it (Rajat 2026-09-04: EUU put spread expiring NFP day was
        # silently dropped from the scenario tool) — it prices with tiny T
        if expiry < date.today():
            notes.append(f"{sym}: expired {expiry} — skipped")
            continue
        src, mkt = pair
        # IB-symbol underlyings ("SOFR3", "ES") carry no contract month —
        # derive the quarterly from the option expiry (its own quarter,
        # rounded up; right for STIR/index/FX quarterlies+serials, approx
        # for monthly commodity cycles). NB midcurves can't be detected
        # without a contract month → treated as front.
        if not und_exp6:
            qm = ((expiry.month + 2) // 3) * 3
            und_exp6 = f"{expiry.year:04d}{qm:02d}"
        mkt = _midcurve(mkt, und_exp6, expiry)
        opts.append({
            "sym": sym, "src": src, "mkt": mkt, "root": root,
            "und": und, "und_exp6": und_exp6,
            "right": right, "K": float(r.get("Strike") or 0.0),
            "qty": float(r.get("Quantity") or 0.0),
            "mult": float(r.get("Multiplier") or 0.0),
            "fxr": float(r.get("FXRateToBase") or 1.0),
            "expiry": expiry,
        })
    return opts, notes


def _greeks(o: dict, live: bool):
    """Price the single-leg position via the pricer (surface expiry first,
    time-interpolated fallback for unlisted dates)."""
    import pricer
    legs = [(1, o["right"], o["K"])]
    res = pricer.price_structure(o["src"], o["mkt"], o["expiry"], legs,
                                 int(o["qty"]), interp=False, live=live)
    if res.get("err"):
        res = pricer.price_structure(o["src"], o["mkt"], o["expiry"], legs,
                                     int(o["qty"]), interp=True, live=live)
    return res


def _hist_dF(o: dict, F0: float, dv01, proxy, fred_key, n: int = 260):
    """Daily ΔF history in PRICE POINTS of the underlying future (newest last).
    STIR → exact contract closes from stir_bars.db; v2 markets → yfinance
    continuous ×F0; bond futures → saved yield proxy (−Δy, pp) × DV01×100."""
    base = o["mkt"].split("_")[0]
    if base in _STIR_DB_SYM:
        try:
            conn = sqlite3.connect(_STIR_DB)
            rows = conn.execute(
                "SELECT bar_date, close FROM stir_bars WHERE symbol=? AND exp6=? "
                "ORDER BY bar_date", (_STIR_DB_SYM[base], o["und_exp6"])).fetchall()
            conn.close()
            s = pd.Series({pd.Timestamp(d): c for d, c in rows}).astype(float)
            return s.diff().dropna().tail(n)
        except Exception:
            return pd.Series(dtype=float)
    if o["src"] == "v2":
        import pricer as _pr
        tkr = _pr._YF_LIVE.get(o["mkt"])
        if not tkr:
            return pd.Series(dtype=float)
        try:
            import yfinance as yf
            h = yf.Ticker(tkr).history(period="2y", auto_adjust=True)["Close"]
            h.index = pd.DatetimeIndex(
                [t.tz_localize(None) if t.tzinfo else t for t in h.index])
            return (h.pct_change().dropna() * F0).tail(n)
        except Exception:
            return pd.Series(dtype=float)
    # bond futures: proxy yields via risk_div (price-like: −Δy in pp)
    if not (dv01 and proxy):
        return pd.Series(dtype=float)
    try:
        import risk_div
        start = (pd.Timestamp.today() - pd.Timedelta(days=750)).date().isoformat()
        risk_div._prime_proxy_batch([proxy], start)
        r = risk_div._proxy_returns(proxy, start, fred_key)
        return (r * dv01 * 100.0).tail(n)     # pp × (pts/bp × 100 bp/pp)
    except Exception:
        return pd.Series(dtype=float)


_PNL_MEMO: dict = {}
_PNL_TTL_S = 120


def _und_moves(o: dict):
    """Underlying price moves in POINTS vs the close 1/3/5 business days back:
    v2 markets from yfinance daily closes (incl. today's partial bar = the
    rough live mark), STIR from the exact contract's settles in stir_bars.db.
    Returns {1: Δ, 3: Δ, 5: Δ} or None (bond futures → no easy history)."""
    base = o["mkt"].split("_")[0]
    closes = None
    if base in _STIR_DB_SYM:
        try:
            conn = sqlite3.connect(_STIR_DB)
            rows = conn.execute(
                "SELECT bar_date, close FROM stir_bars WHERE symbol=? AND exp6=? "
                "ORDER BY bar_date", (_STIR_DB_SYM[base], o["und_exp6"])).fetchall()
            conn.close()
            closes = [float(c) for _, c in rows]
        except Exception:
            return None
    elif o["src"] == "v2":
        import pricer as _pr
        tkr = _pr._YF_LIVE.get(o["mkt"])
        if not tkr:
            return None
        try:
            import yfinance as yf
            h = yf.Ticker(tkr).history(period="10d", auto_adjust=True)["Close"]
            closes = [float(v) for v in h.values]
        except Exception:
            return None
    if not closes or len(closes) < 2:
        return None
    last = closes[-1]
    out = {}
    for hz in (1, 3, 5):
        if len(closes) > hz:
            out[hz] = last - closes[-1 - hz]
    return out or None


def est_pnl(book: pd.DataFrame, sel: set | None = None, live: bool = True) -> dict:
    """Rough option PnL from delta × underlying move (Rajat 2026-08-26:
    'better than blank — keep it grey'). {symbol: {1:$, 3:$, 5:$}} for the
    options whose underlying has a usable history; memoized ~2 min. First
    order only — gamma/vega/theta ignored, hence the muted display."""
    import time as _t
    now = _t.time()
    key = "est_pnl"
    ent = _PNL_MEMO.get(key)
    if ent and now - ent[0] < _PNL_TTL_S:
        return ent[1]
    out = {}
    opts, _notes = option_book(book, sel)
    moves_memo: dict = {}
    for o in opts:
        try:
            res = _greeks(o, live)
            if res.get("err"):
                continue
            dv01 = res.get("dv01")
            pdelta = float(res.get("delta_usd") or 0.0)     # $/pt (rates: $/bp)
            if o["src"] == "rates":
                if not dv01:
                    continue
                pdelta = pdelta / float(dv01)               # back to $/pt
            mk = (o["src"], o["mkt"], o["und_exp6"])
            if mk not in moves_memo:
                moves_memo[mk] = _und_moves(o)
            mv = moves_memo[mk]
            if not mv:
                continue
            out[o["sym"]] = {hz: pdelta * d * o["fxr"] for hz, d in mv.items()}
        except Exception:
            continue
    _PNL_MEMO[key] = (now, out)
    return out


def compute(book: pd.DataFrame, mode: str, products: dict, ivols: dict,
            proxies: dict, fred_key=None, live: bool = True,
            sel: set | None = None) -> dict:
    """mode ∈ {'delta', 'reval'} → {rows, extra_pos, total, notes, mode}.
    rows    — per-position display tuples for the report box
    extra_pos — [name, product, proxy, sign, var(1σ$)] rows for risk_div
    total   — (reval only) options-book VaR from the summed P&L vectors."""
    import risk as _risk

    opts, notes = option_book(book, sel)
    rows, extra, vecs, exposures = [], [], [], []
    for o in opts:
        res = _greeks(o, live)
        if res.get("err"):
            notes.append(f"{o['sym']}: {res['err']} — skipped")
            continue
        d_usd = float(res.get("delta_usd") or 0.0) * o["fxr"]
        dv01 = res.get("dv01")
        prod = ("Rates" if o["src"] == "rates" else
                products.get(o["und"]) or _risk._guess_product(o["und"], o["und"]))
        proxy = (proxies.get(o["und"]) or proxies.get(o["root"])
                 or _risk._guess_proxy(prod, o["root"], o["und"]))
        sign = 1 if d_usd > 0 else -1
        F0 = float(res["F"])
        if prod == "Rates":
            risk_de = abs(d_usd)                       # already $/bp
            de_txt = f"${abs(d_usd):,.0f}/bp"
            # rates split rows are PER CONTRACT (Rajat: "SR3M6, SR3U6 …");
            # equities/FX/commod stay at the complex level (ES, EUR, …)
            exposures.append((prod,
                              underlying_contract(o["und"], o["und_exp6"]),
                              d_usd))                          # signed $/bp
        else:
            risk_de = abs(d_usd) * F0                  # $ notional equiv
            de_txt = f"${risk_de:,.0f}"
            exposures.append((prod, o["mkt"], d_usd * F0))     # signed $ ntl
        # underlying vol: manual ⚙ ivol if saved for this underlying, else the
        # leg's own fitted surface IV (so delta mode works with zero setup)
        iv_u = ivols.get(o["und"]) or ivols.get(o["root"])
        if not iv_u:
            try:
                leg_iv = float(res["legs"][0]["iv"])
                if prod == "Rates":            # Bachelier price-vol pts → bp/yr
                    iv_u = leg_iv / dv01 if dv01 else None
                else:                          # Black vol fraction → % ann
                    iv_u = leg_iv * 100.0
            except Exception:
                iv_u = None

        var1s = v95 = v99 = None
        n_obs = 0
        if mode == "delta":
            if iv_u:
                var1s = (risk_de * float(iv_u) / math.sqrt(256)
                         / (1.0 if prod == "Rates" else 100.0))
                v95, v99 = var1s * 1.645, var1s * 2.326
            else:
                notes.append(f"{o['sym']}: no vol available (no manual ivol, "
                             f"no surface fit) — skipped")
        else:                                          # full reval
            dF = _hist_dF(o, F0, dv01, proxy, fred_key)
            n_obs = len(dF)
            if n_obs >= 60:
                v0 = _reval(o, res, F0)
                pnl = pd.Series(
                    [(_reval(o, res, F0 + float(x)) - v0)
                     * res["mult"] * o["qty"] * o["fxr"] for x in dF.values],
                    index=dF.index)
                vecs.append(pnl)
                var1s = float(pnl.std())
                v95 = float(-np.percentile(pnl.values, 5))
                v99 = float(-np.percentile(pnl.values, 1))
            else:
                notes.append(f"{o['sym']}: only {n_obs} days of underlying "
                             "history — skipped")
        if var1s:
            extra.append([o["sym"], prod, proxy, sign, float(var1s)])
        rows.append((o["sym"], o["mkt"], o["right"], o["K"], o["expiry"],
                     o["qty"], de_txt, sign, var1s, v95, v99, n_obs))

    total = None
    if mode == "reval" and vecs:
        allv = pd.concat(vecs, axis=1).dropna()
        if len(allv) >= 60:
            s = allv.sum(axis=1)
            total = {"n": len(s), "sigma": float(s.std()),
                     "v95": float(-np.percentile(s.values, 5)),
                     "v99": float(-np.percentile(s.values, 1)),
                     "undiv": float(sum((r[9] or 0.0) for r in rows))}
    return {"rows": rows, "extra_pos": extra, "total": total,
            "notes": notes, "mode": mode, "exposures": exposures}


def _reval(o: dict, res: dict, F: float) -> float:
    """Structure value (pts, 1 lot) at forward F — per-leg IVs sticky, T fixed."""
    import pricer
    return pricer._scn_value(o["src"], res["legs"], F,
                             max(float(res["T"] or 0.0), 1e-6),
                             float(res.get("r") or 0.0))

"""Streamlit dashboard: Sector Breakout Scanner.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core.data import fetch_ohlcv, truncate_frames, get_fund_info
import core.etfs as etfs
import core.indicators as ind
import core.stocks as stocks
from core.etfs import MARKETS, LOAD_ERRORS
from core.scoring import score_sector, breakout_snapshot, imminence_snapshot, score_snapshot, lifecycle_stage, PRIMARY_ORDER, mtf_supertrend_all, ST_COMBOS, supertrend_reversal, _frames_back, ST_LOOKBACK
from core.patterns import detect_breakout_retest
import core.ipos as ipos
import core.alerts as alertmod
import core.flows as flows

st.set_page_config(page_title="Sector Breakout Scanner", layout="wide", page_icon="📈")


@st.cache_data(ttl=21600, show_spinner=False)
def cached_fund_info(ticker: str) -> dict:
    return get_fund_info(ticker)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_recent_ipos(within_days: int) -> list[dict]:
    """Fetch recent NSE IPOs from the official NSE API (cached 6h)."""
    return ipos.fetch_recent_ipos(within_days)


@st.cache_data(ttl=900, show_spinner=False)
def cached_last_price(ticker: str) -> float | None:
    """Latest daily close for a ticker (cached 15 min), or None."""
    try:
        d = fetch_ohlcv(ticker).get("1d", pd.DataFrame())
        return float(d["Close"].iloc[-1]) if not d.empty else None
    except Exception:
        return None


def tradingview_url(ticker: str) -> str:
    """Build a TradingView chart URL. NSE tickers (.NS) map to the NSE exchange."""
    if ticker.upper().endswith(".NS"):
        return f"https://www.tradingview.com/chart/?symbol=NSE:{ticker[:-3].upper()}"
    return f"https://www.tradingview.com/chart/?symbol={ticker.upper()}"


def ipo_closes_since(ticker: str, listing_date) -> "pd.Series | None":
    """Daily closes for an IPO since its listing date (cached), or None."""
    d = cached_ohlcv(ticker).get("1d", pd.DataFrame())
    if d is None or d.empty or "Close" not in d:
        return None
    s = d["Close"].dropna()
    try:
        s = s[s.index >= pd.Timestamp(listing_date)]
    except Exception:
        pass
    return s if len(s) else None


def format_aum(aum, currency) -> str | None:
    if not aum:
        return None
    sym = {"USD": "$", "INR": "₹"}.get(currency, "")
    if aum >= 1e9:
        return f"{sym}{aum / 1e9:.2f}B"
    if aum >= 1e6:
        return f"{sym}{aum / 1e6:.0f}M"
    return f"{sym}{aum:,.0f}"


@st.cache_data(ttl=900, show_spinner=False)
def cached_ohlcv(ticker: str) -> dict:
    return fetch_ohlcv(ticker)


def load_frames(ticker: str) -> dict:
    return cached_ohlcv(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def st_reversal_cached(ticker: str) -> dict:
    """MTF Supertrend reversal read for any ticker (all combos), cached 15 min.
    Same data the Supertrend Reversal tab uses (``mtf_supertrend_all``)."""
    try:
        return mtf_supertrend_all(load_frames(ticker)) or {}
    except Exception:
        return {}


@st.cache_data(ttl=900, show_spinner=False)
def st_reversal_tf_cached(ticker: str, tfs: tuple) -> dict | None:
    """Supertrend reversal for the **exact** timeframe stack selected in the
    sidebar (not just the predefined MTF combos), including the score 1w/2w ago.
    Anchor = the last timeframe in ``tfs``. Returns the reversal dict or None."""
    try:
        frames = load_frames(ticker)
        cur = supertrend_reversal(frames, list(tfs))
        if cur is None:
            return None
        p1 = supertrend_reversal(_frames_back(frames, ST_LOOKBACK["1w"]), list(tfs))
        p2 = supertrend_reversal(_frames_back(frames, ST_LOOKBACK["2w"]), list(tfs))
        cur["score_1w"] = p1["score"] if p1 else None
        cur["score_2w"] = p2["score"] if p2 else None
        return cur
    except Exception:
        return None


def _tf_stack_label(tfs) -> str:
    """Human label for a timeframe stack, e.g. ('1h','2h','4h','1d') → '1h+2h+4h+1d',
    ('1d','1wk') → '1d+1w'."""
    return "+".join("1w" if t == "1wk" else str(t) for t in tfs)


@st.cache_data(ttl=3600, show_spinner=False)
def ticker_is_valid(ticker: str) -> bool:
    """Quick check that yfinance returns any recent data for a ticker."""
    try:
        frames = fetch_ohlcv(ticker)
        return not frames.get("1d", pd.DataFrame()).empty
    except Exception:
        return False


# ----------------------------- Sidebar -----------------------------
st.sidebar.title("⚙️ Scanner Settings")

if LOAD_ERRORS:
    for mkt_err, msg in LOAD_ERRORS.items():
        st.sidebar.error(f"⚠️ {mkt_err} config not loaded (using defaults):\n\n{msg}")

market_choice = st.sidebar.radio("Market", ["US", "India", "Both"], index=0)

selected_markets = ["US", "India"] if market_choice == "Both" else [market_choice]

tf_choice = st.sidebar.radio(
    "Timeframe",
    ["Blend (4h + 1D)", "Blend (4h + 1D + 1W)", "Blend (1D + 1W)",
     "1D only", "4h only", "1W only",
     "MTF 1h+2h+4h", "MTF 1h+2h+4h+1D", "MTF 1h+2h+4h+1D+1W"],
    index=0,
    help="Which timeframe(s) the breakout/exit scores are based on. "
         "1W (weekly) captures the higher-timeframe trend. The **MTF** stacks add "
         "intraday 1h/2h bars — they re-score *every* tab on those timeframes and "
         "drive the Supertrend-reversal anchor (the largest timeframe in the stack).",
)
TF_MAP = {
    "Blend (4h + 1D)": ("4h", "1d"),
    "Blend (4h + 1D + 1W)": ("4h", "1d", "1wk"),
    "Blend (1D + 1W)": ("1d", "1wk"),
    "1D only": ("1d",),
    "4h only": ("4h",),
    "1W only": ("1wk",),
    "MTF 1h+2h+4h": ("1h", "2h", "4h"),
    "MTF 1h+2h+4h+1D": ("1h", "2h", "4h", "1d"),
    "MTF 1h+2h+4h+1D+1W": ("1h", "2h", "4h", "1d", "1wk"),
}
selected_tf = TF_MAP[tf_choice]

# Which Supertrend-reversal combo (ST_COMBOS key) a selected timeframe maps to.
# MTF stacks map 1:1; the classic blends fall back to a sensible default so the
# Supertrend tab always has an anchor.
ST_STACK_MAP = {
    ("1h", "2h", "4h"): "1h+2h+4h",
    ("1h", "2h", "4h", "1d"): "1h+2h+4h+1d",
    ("1h", "2h", "4h", "1d", "1wk"): "1h+2h+4h+1d+1w",
}

# Optional 'as-of' date for backtesting / validating a past setup.
import datetime as _dt
backtest = st.sidebar.checkbox(
    "🕐 Score as of a past date (validate)", value=False,
    help="Truncates data to the chosen day so you can check a historical setup, "
         "then compare it to what actually happened afterward.",
)
as_of_date = None
if backtest:
    as_of_date = st.sidebar.date_input(
        "As-of date",
        value=_dt.date.today() - _dt.timedelta(days=30),
        max_value=_dt.date.today(),
    )

# Build an editable ticker list per market.
if st.sidebar.button("🔄 Load fresh lists from config files (US & India)"):
    etfs.reload()
    for _m in ("US", "India"):
        st.session_state.pop(f"list_{_m}", None)
    if etfs.LOAD_ERRORS:
        st.sidebar.error("Reloaded, but some config files had errors (see above).")
    else:
        st.sidebar.success("Reloaded US & India lists from config files.")
    st.rerun()

custom_lists: dict[str, dict] = {}
for mkt in selected_markets:
    default_map = etfs.MARKETS[mkt]["etfs"]
    default_text = "\n".join(f"{t} = {n}" for t, n in default_map.items())
    with st.sidebar.expander(f"{mkt} ETFs ({len(default_map)})", expanded=False):
        if st.button(f"↺ Reset {mkt} list to defaults", key=f"reset_{mkt}"):
            st.session_state[f"list_{mkt}"] = default_text

        # ---- Add an ETF via a simple form (processed before the text box below) ----
        with st.form(f"add_form_{mkt}", clear_on_submit=True):
            st.caption("➕ Add an ETF")
            suffix_hint = " (use .NS suffix for NSE, e.g. NIFTYBEES.NS)" if mkt == "India" else ""
            new_ticker = st.text_input(f"Ticker{suffix_hint}", key=f"add_tkr_{mkt}")
            new_name = st.text_input("Name (optional)", key=f"add_name_{mkt}")
            validate = st.checkbox("Validate ticker with yfinance", value=True, key=f"add_val_{mkt}")
            submitted = st.form_submit_button("Add ETF")

        if submitted and new_ticker.strip():
            tkr = new_ticker.strip().upper()
            name = new_name.strip() or tkr
            current = st.session_state.get(f"list_{mkt}", default_text)
            existing = {
                ln.split("=")[0].strip().upper()
                for ln in current.splitlines() if ln.strip()
            }
            if tkr in existing:
                st.warning(f"{tkr} is already in the {mkt} list.")
            elif validate and not ticker_is_valid(tkr):
                st.error(f"'{tkr}' returned no data from yfinance. Check the symbol/suffix.")
            else:
                st.session_state[f"list_{mkt}"] = current.rstrip() + f"\n{tkr} = {name}"
                st.success(f"Added {tkr} — {name}. Click 'Scan / Refresh' to include it.")

        text = st.text_area(
            f"{mkt} tickers (one per line, TICKER = Name)",
            value=default_text,
            height=180,
            key=f"list_{mkt}",
        )
    parsed = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line:
            t, n = line.split("=", 1)
            parsed[t.strip()] = n.strip()
        else:
            parsed[line] = line
    custom_lists[mkt] = parsed

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Display")
show_aum = st.sidebar.checkbox(
    "Show fund size / AUM (slower)", value=False,
    help="Fetches each fund's net assets from yfinance. Adds a second or two per ETF.",
)
fast_move = st.sidebar.slider(
    "Fast-move threshold (2-wk score Δ)", 3, 15, 6,
    help="Rows are colored GREEN when the breakout score is rising faster than this "
         "(demand building / consolidating tighter) and RED when it is falling that fast.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Investment Allocation")
default_ccy = "₹" if market_choice == "India" else "$"
ccy = st.sidebar.selectbox(
    "Currency", ["$", "₹"], index=(1 if default_ccy == "₹" else 0)
)
invest_amount = st.sidebar.number_input(
    "Amount to invest", min_value=0.0, value=100000.0, step=1000.0, format="%.2f"
)
top_n = st.sidebar.slider("Split across top N candidates", 2, 3, 3)
min_breakout = st.sidebar.slider("Min breakout score to qualify", 40, 80, 55)
breakout_patience = st.sidebar.slider(
    "Trim if no breakout within N days", 5, 120, 30,
    help="For holdings that are still coiling below their breakout trigger: if "
         "the breakout hasn't fired within this many days of holding (or of the "
         "consolidation), the exit plan flags the position as stale and suggests "
         "a light trim to free up dead money.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Scoring timeframe: **{tf_choice}**.\n\n"
    "Blend weights 4h = 0.4, 1D = 0.6, 1W = 0.5 (normalised across the "
    "selected timeframes). Data via yfinance, cached 15 min. Not investment advice."
)
run = st.sidebar.button("🔄 Scan / Refresh", type="primary", use_container_width=True)


# ----------------------------- Scan logic -----------------------------
def run_scan(markets: list[str], lists: dict[str, dict], timeframes: tuple,
             as_of=None) -> list:
    results = []
    tasks = []
    for mkt in markets:
        for ticker, name in lists[mkt].items():
            tasks.append((mkt, ticker, name))

    bench_cache: dict[str, dict] = {}
    progress = st.progress(0.0, text="Starting scan…")
    total = len(tasks)

    for i, (mkt, ticker, name) in enumerate(tasks, start=1):
        bench_ticker = MARKETS[mkt]["benchmark"]
        if bench_ticker not in bench_cache:
            bench_cache[bench_ticker] = truncate_frames(load_frames(bench_ticker), as_of)
        bench_frames = bench_cache[bench_ticker]

        progress.progress(i / total, text=f"Scanning {ticker} — {name} ({i}/{total})")
        frames = truncate_frames(load_frames(ticker), as_of)
        try:
            score = score_sector(ticker, name, frames, bench_frames, timeframes)
            score.breakout_1w_ago = breakout_snapshot(
                ticker, name, frames, bench_frames, timeframes, "1w")
            score.breakout_2w_ago = breakout_snapshot(
                ticker, name, frames, bench_frames, timeframes, "2w")
            score.imminence_1w_ago = imminence_snapshot(
                ticker, name, frames, bench_frames, timeframes, "1w")
            score.st_mtf = mtf_supertrend_all(frames)
            results.append((mkt, score))
        except Exception as exc:  # keep scanning even if one ticker fails
            st.warning(f"Failed to score {ticker}: {exc}")

    progress.empty()
    return results


def run_retest_scan(markets: list[str], min_conf: float) -> list[dict]:
    """Scan the large-cap stock universes for a 52-week-high breakout that is
    now retesting the breakout level. Returns rows sorted by confidence."""
    tasks = []
    for mkt in markets:
        for ticker, name in stocks.STOCK_MARKETS[mkt].items():
            tasks.append((mkt, ticker, name))

    hits: list[dict] = []
    progress = st.progress(0.0, text="Starting retest scan…")
    total = max(1, len(tasks))
    for i, (mkt, ticker, name) in enumerate(tasks, start=1):
        progress.progress(i / total, text=f"Checking {ticker} — {name} ({i}/{total})")
        try:
            frames = load_frames(ticker)
            res = detect_breakout_retest(frames.get("1d", pd.DataFrame()))
        except Exception:
            res = {}
        if res and res.get("confidence", 0) >= min_conf:
            hits.append(
                {
                    "Market": mkt,
                    "Ticker": tradingview_url(ticker),
                    "Symbol": ticker,
                    "Name": name,
                    "Price": res["price"],
                    "52W High": res["high_52w"],
                    "% from High": res["pct_from_high"],
                    "Breakout Level": res["breakout_level"],
                    "Retest (% vs level)": res["dist_to_level_pct"],
                    "Pullback %": res["retest_depth_pct"],
                    "Vol light": "✓" if res["vol_light"] else "",
                    "Confidence": res["confidence"],
                    "Detail": res["note"],
                }
            )
    progress.empty()
    hits.sort(key=lambda h: h["Confidence"], reverse=True)
    return hits


st.title("📈 Sector Breakout Scanner")
st.caption(
    "Finds sectors **consolidating and ready for a bull breakout** (MACD near 0, "
    "tight range, neutral-bullish RSI, positive relative strength, low-rising ADX, "
    "volume pickup) — and flags sectors that are **over-extended and due for an exit**."
)

if backtest and as_of_date:
    st.warning(
        f"🕐 **Backtest mode** — scores computed **as of {as_of_date:%d %b %Y}** "
        "(data after this date is ignored). Compare against later price action to "
        "validate the signal. Uncheck the sidebar option for live scores."
    )

if run:
    st.session_state["results"] = run_scan(selected_markets, custom_lists, selected_tf, as_of_date)

results = st.session_state.get("results", [])
scan_ready = bool(results)
# Note: we no longer st.stop() here when there is no scan yet — the Exit-plan
# tab is designed to work standalone (it scores only your uploaded holdings).
# The scanner-dependent tabs are gated later with `if not scan_ready: st.stop()`
# rendered *after* the Exit-plan tab, so that tab always works without a scan.


# ----------------------------- Build table -----------------------------
def build_score_df(scored: list, with_aum: bool) -> pd.DataFrame:
    """Turn a list of (market, SectorScore) into the breakout table DataFrame."""
    rows = []
    for mkt, s in scored:
        aum_str = None
        if with_aum:
            fi = cached_fund_info(s.ticker)
            aum_str = format_aum(fi.get("aum"), fi.get("currency"))
        rows.append(
            {
                "Market": mkt,
                "Ticker": tradingview_url(s.ticker),
                "Symbol": s.ticker,
                "Sector": s.name,
                "AUM": aum_str,
                "Score": round(0.6 * s.breakout_score + 0.4 * s.imminence, 1),
                "Breakout": s.breakout_score,
                "Imminence": s.imminence,
                "Trigger": s.imminence_label,
                "HNI flow": s.accumulation_score,
                "Vol score": s.volume_score,
                "1W ago": s.breakout_1w_ago,
                "2W ago": s.breakout_2w_ago,
                "Trend": f"{s.breakout_trend} ({s.breakout_delta:+.1f})",
                "_delta": s.breakout_delta,
                "Consol. since": s.consolidation.get("start"),
                "Consol. (bars)": (
                    f"{s.consolidation.get('bars')} {s.consolidation.get('unit')}"
                    if s.consolidation else None
                ),
                "Range%": s.consolidation.get("range_pct"),
                "Maturity": s.maturity,
                "Pattern": s.patterns.get("labels") or "—",
                "Exit": s.exit_score,
                "Signal": s.signal,
                "Action": s.action,
                "T1": s.targets.get("target1"),
                "T1%": s.targets.get("target1_pct"),
                "T2": s.targets.get("target"),
                "Upside%": s.targets.get("upside_pct"),
                "R:R": s.targets.get("risk_reward"),
                "RSI(1d)": s.tf["1d"].raw.get("rsi") if s.tf["1d"].ok else None,
                "ADX(1d)": s.tf["1d"].raw.get("adx") if s.tf["1d"].ok else None,
                "RelStr%(1d)": s.tf["1d"].raw.get("rel_strength_pct") if s.tf["1d"].ok else None,
                "MACD%(1d)": s.tf["1d"].raw.get("macd_pct") if s.tf["1d"].ok else None,
            }
        )
    frame = pd.DataFrame(rows)
    if not with_aum and "AUM" in frame.columns:
        frame = frame.drop(columns=["AUM"])
    return frame


df = build_score_df(results, show_aum)


def parse_watchlist(uploaded) -> list[dict]:
    """Parse a TradingView watchlist export (.xlsx/.csv) into [{yf, name}].
    Accepts a 'Symbol' column, or falls back to 'TV Code' (e.g. 'NSE:AEGISLOG,')."""
    import io
    fname = uploaded.name.lower()
    data = uploaded.getvalue()
    try:
        if fname.endswith((".xlsx", ".xls")):
            frame = pd.read_excel(io.BytesIO(data))
        else:
            frame = pd.read_csv(io.BytesIO(data))
    except Exception:
        return []
    frame.columns = [str(c).strip() for c in frame.columns]

    def _find(*names):
        for c in frame.columns:
            if c.lower() in names:
                return c
        return None

    sym_col = _find("symbol", "ticker")
    tv_col = next((c for c in frame.columns if "tv code" in c.lower()), None)
    desc_col = _find("description", "name", "company")
    sec_col = _find("sector")

    out, seen = [], set()
    for _, r in frame.iterrows():
        sym = None
        if sym_col and pd.notna(r.get(sym_col)):
            sym = str(r[sym_col])
        elif tv_col and pd.notna(r.get(tv_col)):
            sym = str(r[tv_col])
        if not sym:
            continue
        sym = sym.split(":")[-1].strip().strip(",").strip().upper()
        if not sym or sym in ("SYMBOL", "NAN"):
            continue
        yf = sym if sym.endswith(".NS") else f"{sym}.NS"
        if yf in seen:
            continue
        seen.add(yf)
        name = None
        if desc_col and pd.notna(r.get(desc_col)):
            name = str(r[desc_col]).strip()
        elif sec_col and pd.notna(r.get(sec_col)):
            name = str(r[sec_col]).strip()
        out.append({"yf": yf, "name": name or sym})
    return out


def run_list_scan(pairs: list[dict], market: str, timeframes: tuple) -> list:
    """Score an arbitrary list of tickers against a market's benchmark."""
    bench = truncate_frames(load_frames(MARKETS[market]["benchmark"]), None)
    out = []
    prog = st.progress(0.0, text="Scoring watchlist…")
    n = max(1, len(pairs))
    for i, p in enumerate(pairs, start=1):
        prog.progress(i / n, text=f"Scoring {p['yf']} ({i}/{n})")
        try:
            frames = load_frames(p["yf"])
            s = score_sector(p["yf"], p["name"], frames, bench, timeframes)
            s.breakout_1w_ago = breakout_snapshot(
                p["yf"], p["name"], frames, bench, timeframes, "1w")
            s.breakout_2w_ago = breakout_snapshot(
                p["yf"], p["name"], frames, bench, timeframes, "2w")
            s.imminence_1w_ago = imminence_snapshot(
                p["yf"], p["name"], frames, bench, timeframes, "1w")
            out.append((market, s))
        except Exception as exc:
            st.warning(f"Failed to score {p['yf']}: {exc}")
    prog.empty()
    return out


def _style_by_trend(row):
    """Green when the 2-week score is rising fast, red when falling fast.

    Uses solid mid-tone backgrounds with explicit white text so it stays legible
    on both light and dark themes.
    """
    d = row.get("_delta", 0)
    if d >= fast_move:
        style = "background-color: #1e7d46; color: #ffffff; font-weight: 600"  # green
    elif d <= -fast_move:
        style = "background-color: #b23b3b; color: #ffffff; font-weight: 600"  # red
    else:
        style = ""
    return [style] * len(row)


TICKER_LINK = st.column_config.LinkColumn(
    "Ticker", help="Opens the TradingView chart", display_text=r"symbol=(.+)$"
)


def render_table(frame):
    styler = frame.style.apply(_style_by_trend, axis=1)
    cfg = {
        "Ticker": TICKER_LINK,
        "_delta": None,  # hide helper column
        "Score": st.column_config.ProgressColumn(
            "Score", help="Combined breakout score = 60% readiness (base quality) + "
            "40% imminence (is it firing NOW). This is the ranking column — it lifts "
            "candidates that are actually triggering above pretty-but-dormant bases",
            min_value=0, max_value=100, format="%d"
        ),
        "Breakout": st.column_config.ProgressColumn(
            "Breakout", help="Readiness — how well-formed the base is (setup quality)",
            min_value=0, max_value=100, format="%d"
        ),
        "Imminence": st.column_config.ProgressColumn(
            "Imminence", help="Is the breakout firing NOW — price at the line, volume "
            "expanding, ADX rising, squeeze firing, and price above the base's "
            "anchored VWAP. A high readiness with low imminence is still just coiling",
            min_value=0, max_value=100, format="%d"
        ),
        "HNI flow": st.column_config.ProgressColumn(
            "HNI flow", help="Smart-money / HNI accumulation proxy — how strongly "
            "large investors appear to be buying, from Chaikin Money Flow, up-day "
            "volume dominance, OBV confirmation, Money-Flow Index and relative "
            "strength (0 = distribution, 100 = heavy accumulation)",
            min_value=0, max_value=100, format="%d"
        ),
        "Vol score": st.column_config.ProgressColumn(
            "Vol score", help="Volume-confirmation score — is recent volume expanding "
            "to back the move (ideal ≈1.3–1.6× baseline). Low = volume dry-up, "
            "100 = strong volume thrust",
            min_value=0, max_value=100, format="%d"
        ),
        "1W ago": st.column_config.NumberColumn(
            "1W ago", help="Breakout score as of ~1 week ago", format="%d"
        ),
        "2W ago": st.column_config.NumberColumn(
            "2W ago", help="Breakout score as of ~2 weeks ago", format="%d"
        ),
        "Exit": st.column_config.ProgressColumn(
            "Exit", min_value=0, max_value=100, format="%d"
        ),
    }
    st.dataframe(styler, use_container_width=True, hide_index=True, column_config=cfg)


# ------------------- SIP & Exit Plan (tab8) --------------------------
# A weekly-primary rotation helper aimed at ~5%/month: accumulate (SIP) into
# the top-3 coiling leaders of the selected market, and scale out of held
# positions (uploaded CSV) as they get over-extended.
CCY_SYM = {"US": "$", "India": "₹"}

# Dip-buying ladder: (price offset from today's close, share of the daily chunk).
# More capital is queued at lower prices so a falling ETF lowers your average
# cost. Shares sum to 1.0; the blended fill price is ~2% below market.
SIP_LADDER = [(0.00, 0.40), (-0.02, 0.30), (-0.04, 0.20), (-0.06, 0.10)]


def todays_sip_entry(sc, price, held_qty, held_pct, recently_bought,
                     cash=0.0, sym="$"):
    """Decide whether — and at what laddered prices — to SIP into one candidate
    *today*, accounting for the breakout trigger (imminence), how much you
    already hold, and whether you just bought it.

    ``sc`` is a SectorScore; ``held_qty`` / ``held_pct`` come from the uploaded
    positions; ``recently_bought`` is True when the order history shows a buy
    inside the current bar. Returns a row dict for the SIP-entry table.
    """
    b = sc.breakout_score or 0
    e = sc.exit_score or 0
    imm = sc.imminence if sc.imminence is not None else 0

    # How much of today's chunk to actually place, and the action label.
    if not price or price <= 0:
        action, size = "⬜ No price", 0.0
    elif e >= 50:
        action, size = "⚠️ Extended — book, don't add", 0.0
    elif recently_bought:
        action, size = "✋ Just bought — skip today", 0.0
    elif held_pct is not None and held_pct >= 0.20:
        # Already a core-sized position — only top up on the lower dip rungs.
        action, size = "🟢 Core holding — add on dips only", 0.5
    elif imm >= 55 and b >= 45:
        action, size = "🟢 BUY today — trigger firing", 1.0
    elif imm >= 40 and b >= 45:
        action, size = "🟢 BUY small — trigger building", 0.5
    elif b >= 45:
        action, size = "⏳ WAIT — coiling, no trigger", 0.0
    else:
        action, size = "⬜ Not a SIP candidate", 0.0

    # Build the laddered limit prices for the rungs we intend to place. When
    # sizing at 0.5 we skip the market rung and queue only the dip rungs.
    rungs = SIP_LADDER if size >= 1.0 else SIP_LADDER[1:] if size > 0 else []
    rung_prices = [round(price * (1 + off), 2) for off, _ in rungs] if price else []
    entries = " / ".join(f"{sym}{p:,.2f}" for p in rung_prices) if rung_prices else "—"

    chunk = round(cash * size, 2) if cash else None
    units = None
    if chunk and price and rung_prices:
        # Rough total units if every intended rung fills, weighted by the ladder.
        wsum = sum(w for _, w in rungs) or 1.0
        units = int(sum((chunk * (w / wsum)) // rp
                        for (_, w), rp in zip(rungs, rung_prices)))

    return {
        "action": action,
        "entries": entries,
        "chunk": chunk,
        "units": units,
    }


def _pos_num(x):
    """Parse a messy CSV cell ('$119.37', '1,234', '-5.08%', '--') to float."""
    if x is None:
        return None
    s = str(x).replace("$", "").replace(",", "").replace("%", "").strip()
    if s in ("", "--", "N/A", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_positions_text(text: str):
    """Auto-detect a Schwab (US) or Zerodha (India) positions CSV and normalise
    it to a list of dicts: {market, yf, symbol, qty, avg, ltp, gain_pct}."""
    import io
    lines = text.splitlines()
    header_idx, fmt = None, None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if "symbol" in low and "asset type" in low:
            header_idx, fmt = i, "us"
            break
        if "instrument" in low and "ltp" in low:
            header_idx, fmt = i, "india"
            break
    if fmt is None:
        return [], None
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    except Exception:
        return [], None
    df.columns = [str(c).strip() for c in df.columns]
    out = []
    if fmt == "us":
        for _, r in df.iterrows():
            sym = str(r.get("Symbol", "")).strip()
            atype = str(r.get("Asset Type", "")).strip().lower()
            if not sym or sym.lower() in ("cash & cash investments", "positions total"):
                continue
            if "cash" in atype or "money market" in atype:
                continue
            qty = _pos_num(r.get("Qty (Quantity)", r.get("Qty")))
            if qty is None:
                continue
            out.append({
                "market": "US", "yf": sym.upper(), "symbol": sym.upper(),
                "qty": qty, "avg": _pos_num(r.get("Cost/Share")),
                "ltp": _pos_num(r.get("Price")),
                "gain_pct": _pos_num(r.get("Gain % (Gain/Loss %)", r.get("Gain %"))),
            })
    else:
        for _, r in df.iterrows():
            sym = str(r.get("Instrument", "")).strip()
            if not sym or sym.lower() == "instrument":
                continue
            qty = _pos_num(r.get("Qty.", r.get("Qty")))
            out.append({
                "market": "India", "yf": f"{sym.upper()}.NS", "symbol": sym.upper(),
                "qty": qty, "avg": _pos_num(r.get("Avg. cost")),
                "ltp": _pos_num(r.get("LTP")),
                "gain_pct": _pos_num(r.get("Net chg.")),
            })
    return out, fmt


@st.cache_data(ttl=900, show_spinner=False)
def score_holding_cached(ticker: str, market: str, timeframes: tuple):
    """Score any held ticker (stock or ETF) and return the fields the exit plan
    needs. Cached 15 min. Returns a plain dict (picklable)."""
    bench = load_frames(MARKETS[market]["benchmark"])
    frames = load_frames(ticker)
    s = score_sector(ticker, ticker, frames, bench, timeframes)
    snap_1w = score_snapshot(ticker, ticker, frames, bench, timeframes, "1w")
    imm_1w = snap_1w.imminence if snap_1w else None
    exit_imm_1w = snap_1w.exit_imminence if snap_1w else None
    # ---- MTF Supertrend reversal read (same model as the Supertrend tab) ----
    # The selected timeframe picks the ST combo; classic blends fall back to the
    # 1D-anchored stack (the more reliable reversal anchor per backtest).
    st_combo = ST_STACK_MAP.get(tuple(timeframes), "1h+2h+4h+1d")
    st_cur = (mtf_supertrend_all(frames) or {}).get(st_combo) or {}
    # ---- Chandelier Exit on the selected stack's anchor timeframe ----
    # This intentionally follows the sidebar selection exactly, unlike the
    # legacy Supertrend holding read that maps classic blends to an MTF combo.
    ce_stop = ce_dist_atr = None
    ce_anchor = timeframes[-1] if timeframes else None
    ce_df = frames.get(ce_anchor) if ce_anchor else None
    if (ce_df is not None and len(ce_df) >= 22
            and {"High", "Low", "Close"}.issubset(ce_df.columns)):
        ce_long, _ = ind.chandelier_exit(
            ce_df["High"], ce_df["Low"], ce_df["Close"], period=22, mult=3.0)
        ce_last = ce_long.iloc[-1]
        ce_atr = ind.atr(ce_df["High"], ce_df["Low"], ce_df["Close"], 22).iloc[-1]
        ce_price = float(ce_df["Close"].iloc[-1])
        if pd.notna(ce_last):
            ce_stop = float(ce_last)
            if pd.notna(ce_atr) and ce_atr > 0:
                ce_dist_atr = (ce_price - ce_stop) / float(ce_atr)
    # ---- 10-day EMA proximity (for SIP entry sizing) ----
    ema10 = ema10_dist = None
    d1 = frames.get("1d")
    if d1 is not None and not d1.empty and "Close" in d1.columns and len(d1) >= 10:
        e10 = d1["Close"].ewm(span=10, adjust=False).mean()
        if not e10.empty:
            ema10 = float(e10.iloc[-1])
            px = float(d1["Close"].iloc[-1])
            if ema10:
                ema10_dist = round((px / ema10 - 1) * 100, 2)
    return {
        "exit_score": s.exit_score,
        "breakout": s.breakout_score,
        "signal": s.signal,
        "price": (s.targets or {}).get("entry"),
        "target1": (s.targets or {}).get("target1"),
        "target1_pct": (s.targets or {}).get("target1_pct"),
        "scale_out_pct": (s.targets or {}).get("scale_out_pct"),
        "target": (s.targets or {}).get("target"),
        "stop": (s.targets or {}).get("stop"),
        "breakout_level": (s.targets or {}).get("breakout_level"),
        "cons_days": (s.consolidation or {}).get("days"),
        "rsi": s.tf["1d"].raw.get("rsi") if s.tf["1d"].ok else None,
        # ---- Breakout-trigger (imminence) algorithm fields ----
        "imminence": s.imminence,
        "imminence_label": s.imminence_label,
        "imminence_note": s.imminence_note,
        "imminence_1w_ago": imm_1w,
        "dist_to_breakout_pct": s.dist_to_breakout_pct,
        "range_position": s.range_position,
        "accumulation": s.accumulation_score,
        "volume_score": s.volume_score,
        # ---- Breakdown-trigger (exit-imminence) algorithm fields ----
        "exit_imminence": s.exit_imminence,
        "exit_imminence_label": s.exit_imminence_label,
        "exit_imminence_note": s.exit_imminence_note,
        "exit_imminence_1w_ago": exit_imm_1w,
        # ---- MTF Supertrend reversal fields ----
        "st_combo": st_combo,
        "st_trend": st_cur.get("trend"),
        "st_reversal": st_cur.get("score"),
        "st_reversal_to": st_cur.get("reversal_to"),
        "st_stack": st_cur.get("stack"),
        "st_flip_price": st_cur.get("flip_price"),
        "st_dist_atr": st_cur.get("dist_atr"),
        "st_reversal_1w": st_cur.get("score_1w"),
        "st_anchor": st_cur.get("anchor"),
        # ---- Chandelier Exit fields (22-bar highest-high, 3x ATR) ----
        "ce_stop": ce_stop,
        "ce_dist_atr": ce_dist_atr,
        "ce_anchor": ce_anchor,
        # ---- 10-day EMA (SIP entry proximity) ----
        "ema10": ema10,
        "ema10_dist_pct": ema10_dist,
    }


def exit_tier(exit_score):
    """Map an exit score to a scale-out % and label."""
    if exit_score is None:
        return 0, "❔ No data"
    if exit_score >= 65:
        return 100, "🔴 Exit fully — over-extended"
    if exit_score >= 50:
        return 50, "🟠 Trim half — getting extended"
    if exit_score >= 40:
        return 25, "🟡 Trim a quarter — watch"
    return 0, "🟢 Hold — trend intact"


# Local exchange trading hours: tz name, (open h, m), (close h, m).
MARKET_HOURS = {
    "US": ("America/New_York", (9, 30), (16, 0)),
    "India": ("Asia/Kolkata", (9, 15), (15, 30)),
}
ACTION_WINDOW_MIN = 30  # act within this many minutes of a bar close


def _primary_tf(timeframes) -> str:
    """The highest (longest) timeframe drives the action cadence."""
    for tf in ("1wk", "1d", "4h"):
        if tf in timeframes:
            return tf
    return "1d"


def bar_close_status(market, timeframes):
    """(message, in_window) describing when to act, based on the selected
    timeframe's bar cadence at this market rather than a fixed daily close.

    * 4h  → next 4-hour bar close within the session
    * 1d  → the daily close
    * 1wk → the weekly close (Friday)
    """
    tf = _primary_tf(timeframes)
    label = {"4h": "4-hour", "1d": "daily", "1wk": "weekly"}[tf]
    spec = MARKET_HOURS.get(market)
    if not spec:
        return (f"{market} ({label} bar): schedule unknown", False)
    tzname, (oh, om), (ch, cm) = spec
    try:
        from zoneinfo import ZoneInfo
        import datetime as _d
        now = _d.datetime.now(ZoneInfo(tzname))
    except Exception:
        return (f"{market} ({label} bar): timezone unavailable", False)

    dow = now.weekday()  # 0=Mon … 6=Sun
    if dow >= 5:
        return (f"{market} ({label} bar): market closed (weekend)", False)

    close_today = now.replace(hour=ch, minute=cm, second=0, microsecond=0)
    open_today = now.replace(hour=oh, minute=om, second=0, microsecond=0)

    if tf == "1wk":
        if dow < 4:  # Mon–Thu
            days = 4 - dow
            return (f"{market} (weekly bar): act Friday near close (~{days}d away)", False)
        mins = int((close_today - now).total_seconds() // 60)  # Friday
        if mins < 0:
            return (f"{market} (weekly bar): Friday session closed", False)
        return (f"{market} (weekly bar): {mins} min to Friday close",
                mins <= ACTION_WINDOW_MIN)

    # Build today's bar-close times for 4h / 1d.
    if tf == "4h":
        closes, t = [], open_today
        while t < close_today:
            t = t + _d.timedelta(hours=4)
            closes.append(min(t, close_today))
        closes = sorted(set(closes))
    else:  # 1d
        closes = [close_today]

    upcoming = [c for c in closes if (c - now).total_seconds() > -60]
    if not upcoming:
        return (f"{market} ({label} bar): closed for today", False)
    mins = int((upcoming[0] - now).total_seconds() // 60)
    return (f"{market} ({label} bar): {mins} min to next bar close",
            mins <= ACTION_WINDOW_MIN)


def daily_action(sc):
    """End-of-day two-sided decision for one holding.

    Returns dict(side, label, pct, act_price, day_target) where side is one of
    add / trim / exit / hold. ADD when the setup is still building (strong
    breakout, low exit) and price is above its stop but below the first target;
    TRIM at the first target or a rising exit score; EXIT when over-extended.

    The **ADD** side uses the Breakout-Trigger algorithm: a strong base
    (readiness) is not enough — the breakout must actually be *firing* now
    (imminence). A ready-but-dormant base is held, not added to, which avoids
    piling into names that quietly coil for weeks (the ITB problem).
    """
    b = sc.get("breakout") or 0
    e = sc.get("exit_score") or 0
    rsi = sc.get("rsi") or 0
    price = sc.get("price")
    t1 = sc.get("target1")
    t2 = sc.get("target")
    stop = sc.get("stop")
    imm = sc.get("imminence")
    imm = imm if imm is not None else 50  # neutral if the trigger can't be scored
    xi = sc.get("exit_imminence")
    xi = xi if xi is not None else 50  # neutral if the breakdown trigger is n/a

    # ---- Hard exits: non-negotiable regardless of the breakdown trigger ----
    if e >= 65 or rsi >= 80 or (t2 and price and price >= t2):
        return {"side": "exit", "label": "🔴 EXIT — over-extended",
                "pct": 100, "act_price": price, "day_target": t2 or price}
    # Hitting the first target is a prudent book-partial on its own.
    if t1 and price and price >= t1:
        return {"side": "trim", "label": "🟠 TRIM — booked at target",
                "pct": 40, "act_price": price, "day_target": t1}
    # ---- TRIM side, gated by the breakdown trigger (exit-imminence) ----
    # Symmetric to the ADD gating: being *extended* isn't enough — trim harder
    # only when a top is actually *firing* now (price losing support, selling
    # volume expanding, momentum rolling down). Extended-but-still-trending
    # winners are trimmed lightly / left to run instead of dumped early.
    if e >= 50:
        if xi >= 52:
            return {"side": "trim", "label": "🔴 TRIM — extended + rolling over",
                    "pct": 40, "act_price": price, "day_target": t1 or price}
        if xi >= 32:
            return {"side": "trim", "label": "🟠 TRIM — extended, wobbling",
                    "pct": 25, "act_price": price, "day_target": t1 or price}
        return {"side": "trim", "label": "🟡 TRIM light — extended, still holding",
                "pct": 15, "act_price": price, "day_target": t1 or price}
    if e >= 40:
        if xi >= 52:
            return {"side": "trim", "label": "🟠 TRIM — breakdown firing",
                    "pct": 25, "act_price": price, "day_target": t1 or price}
        return {"side": "hold", "label": "🟢 HOLD — extended, no breakdown yet",
                "pct": 0, "act_price": price, "day_target": t1 or price}
    # ---- ADD side, gated by the breakout trigger (imminence) ----
    base_ok = b >= 58 and e < 45 and price and stop and price > stop and (not t1 or price < t1)
    if base_ok and imm >= 55:
        # Ready AND firing — scale add size by how strong the trigger is.
        add_pct = 30 if imm >= 70 else 20 if imm >= 62 else 15
        return {"side": "add", "label": "🟢 ADD — ready + firing",
                "pct": add_pct, "act_price": round(price * 0.99, 2),
                "day_target": t1 or price}
    if base_ok and imm >= 40:
        # Ready, trigger building — small starter add only.
        return {"side": "add", "label": "🟢 ADD small — trigger building",
                "pct": 10, "act_price": round(price * 0.99, 2),
                "day_target": t1 or price}
    if base_ok:
        # Ready but no trigger (coiling) — do NOT add; wait for it to fire.
        return {"side": "hold", "label": "⏳ HOLD — ready, no trigger yet",
                "pct": 0, "act_price": price, "day_target": t1 or price}
    return {"side": "hold", "label": "⚪ HOLD — do nothing",
            "pct": 0, "act_price": price, "day_target": t1 or price}


def apply_supertrend(act, sc):
    """Overlay the MTF Supertrend reversal read on a two-sided action.

    The Supertrend anchor is the largest timeframe in the selected stack. Its
    direction plus the reversal score (0-100, higher = a flip is more likely
    soon) refine the add/trim/exit call:

      * **Bull anchor + high reversal score** → topping risk → escalate to a
        TRIM (or withhold an ADD) even if the breakout metrics still look fine.
      * **Bear anchor already flipped down** → the long-trend stop is hit →
        EXIT (the anchor Supertrend line is the trailing stop).
      * **Bear anchor + very high reversal score** → a bull flip is imminent
        (the backtest's highest-edge long trigger) → convert a would-be
        exit/hold into a small anticipatory starter ADD, unless a breakdown is
        actively firing.

    Returns (act, note). ``act`` is unchanged when Supertrend adds nothing.
    """
    trend = sc.get("st_trend")
    rev = sc.get("st_reversal")
    if trend is None or rev is None:
        return act, None
    is_bull = "Bull" in trend
    price = sc.get("price")
    flip = sc.get("st_flip_price")
    side = act["side"]

    if is_bull:
        # Uptrend intact — only a strongly firing reversal should override.
        if rev >= 75 and side in ("hold", "add"):
            return ({"side": "trim",
                     "label": "🟠 TRIM — ST topping (reversal imminent)",
                     "pct": 33, "act_price": price,
                     "day_target": sc.get("target1")},
                    f"MTF Supertrend still Bull but reversal score {rev:.0f} — high "
                    "flip risk; trim into strength.")
        if rev >= 75 and side == "trim":
            return ({**act, "pct": max(act.get("pct", 0), 50),
                     "label": "🔴 TRIM more — ST topping"},
                    f"Supertrend reversal score {rev:.0f} reinforces the trim.")
        if rev >= 60 and side == "add":
            return ({"side": "hold",
                     "label": "⏳ HOLD — ST reversal pressure building",
                     "pct": 0, "act_price": price,
                     "day_target": sc.get("target1")},
                    f"Add withheld — Supertrend reversal score {rev:.0f} is rising.")
        return act, None

    # ---- Bearish anchor: the long-trend Supertrend has flipped down ----
    xi = sc.get("exit_imminence") or 0
    if rev >= 65 and side in ("hold", "exit") and xi < 55:
        return ({"side": "add",
                 "label": "🟢 ADD starter — ST bull flip imminent",
                 "pct": 8,
                 "act_price": round(price * 0.99, 2) if price else price,
                 "day_target": sc.get("target1")},
                f"Supertrend Bear but reversal score {rev:.0f} — bull flip imminent; "
                "anticipatory starter only.")
    if side in ("hold", "add"):
        return ({"side": "exit",
                 "label": "🔴 EXIT — ST flipped bearish (trend stop)",
                 "pct": 100, "act_price": price, "day_target": flip or price},
                "MTF Supertrend anchor is Bearish — trend stop hit"
                + (f" (flip line {flip:.2f})." if flip else "."))
    return act, None


def breakout_watch(sc, days_waited, patience_days):
    """Track whether a held setup has actually broken out yet, and flag it as
    stale if the breakout hasn't fired within `patience_days`.

    Returns dict(trigger, status, stale, waited, to_trigger_pct):
      * trigger        — the price that must be crossed for a breakout
      * to_trigger_pct — how far (%) price still is from that trigger
      * stale          — True when it's still coiling below the trigger past the
                         patience window (dead money → consider trimming/exiting)
    """
    price = sc.get("price")
    trig = sc.get("breakout_level")
    if not trig or not price:
        return {"trigger": trig, "status": "—", "stale": False,
                "waited": days_waited, "to_trigger_pct": None}
    if price >= trig:
        return {"trigger": trig, "status": "✅ broken out", "stale": False,
                "waited": days_waited, "to_trigger_pct": 0.0}
    to_pct = round((trig / price - 1) * 100, 1)
    w = days_waited
    if w is not None and patience_days and w >= patience_days:
        return {"trigger": trig, "status": f"⌛ stale {w}d — no breakout",
                "stale": True, "waited": w, "to_trigger_pct": to_pct}
    wtxt = f"{w}/{patience_days}d" if (w is not None and patience_days) else "coiling"
    return {"trigger": trig, "status": f"⏳ {wtxt} · +{to_pct}% to trigger",
            "stale": False, "waited": w, "to_trigger_pct": to_pct}


# ---- Order-history (tradebook / transactions) enrichment ------------------
# The positions file is the source of truth for what you *hold today*; an
# order-history export (Zerodha tradebook .xlsx/.csv or Schwab transactions
# .csv) is layered on top to make the add/trim/exit call smarter: it tells us
# your true average cost, how long you've held, and — crucially — whether you
# already traded this name in the current bar, so the plan doesn't tell you to
# keep adding to something you just bought (or re-trim what you just sold).

def _orders_from_frame(frame):
    """Normalise a tradebook/transactions frame to a list of order dicts:
    {market, symbol, yf, side, qty, price, date}. Auto-detects the broker."""
    frame.columns = [str(c).strip() for c in frame.columns]
    cols = {c.lower(): c for c in frame.columns}
    out = []
    # Zerodha tradebook (India): Symbol, Trade Type (buy/sell), Quantity, Price.
    if "trade type" in cols and "symbol" in cols and "quantity" in cols:
        for _, r in frame.iterrows():
            sym = str(r[cols["symbol"]]).strip().upper()
            side = str(r[cols["trade type"]]).strip().lower()
            if side not in ("buy", "sell") or not sym or sym in ("NAN", ""):
                continue
            date = r.get(cols.get("trade date")) or r.get(cols.get("order execution time"))
            out.append({
                "market": "India", "symbol": sym, "yf": f"{sym}.NS", "side": side,
                "qty": _pos_num(r[cols["quantity"]]), "price": _pos_num(r[cols["price"]]),
                "date": pd.to_datetime(date, errors="coerce"),
            })
        return out, "zerodha_tradebook"
    # Schwab transactions (US): Date, Action (Buy/Sell…), Symbol, Quantity, Price.
    if "action" in cols and "symbol" in cols and "quantity" in cols:
        for _, r in frame.iterrows():
            act = str(r[cols["action"]]).strip().lower()
            side = "buy" if "buy" in act else "sell" if "sell" in act else None
            sym = str(r[cols["symbol"]]).strip().upper()
            if side is None or not sym or sym in ("NAN", ""):
                continue
            out.append({
                "market": "US", "symbol": sym, "yf": sym, "side": side,
                "qty": _pos_num(r[cols["quantity"]]), "price": _pos_num(r[cols["price"]]),
                "date": pd.to_datetime(r.get(cols.get("date")), errors="coerce"),
            })
        return out, "schwab_txn"
    return [], None


def parse_orders_bytes(data: bytes, filename: str):
    """Parse an uploaded order-history file (.xlsx/.xls/.csv) into order dicts.
    Header rows are auto-located (broker exports carry title/metadata rows on
    top). Returns (orders, fmt)."""
    import io
    low = filename.lower()
    try:
        if low.endswith((".xlsx", ".xls")):
            raw = pd.read_excel(io.BytesIO(data), header=None)
        else:
            raw = pd.read_csv(io.BytesIO(data), header=None, dtype=str,
                              on_bad_lines="skip")
    except Exception:
        return [], None
    hidx = None
    for i in range(min(40, len(raw))):
        vals = [str(v).strip().lower() for v in raw.iloc[i].values]
        if "symbol" in vals and ("trade type" in vals or "action" in vals):
            hidx = i
            break
    if hidx is None:
        return [], None
    frame = raw.iloc[hidx + 1:].copy()
    frame.columns = list(raw.iloc[hidx].values)
    frame = frame.dropna(how="all")
    return _orders_from_frame(frame)


def summarize_orders(orders: list) -> dict:
    """Aggregate a flat order list into per-holding stats, keyed by
    'MARKET:SYMBOL'. Returns {net_qty, buy_qty, sell_qty, avg_buy, first_buy,
    last_date, last_side, n_trades}."""
    from collections import defaultdict
    agg = defaultdict(lambda: {
        "buy_qty": 0.0, "sell_qty": 0.0, "cost": 0.0, "trades": [],
        "first_buy": None, "last_date": None, "last_side": None, "n_trades": 0})
    for o in orders:
        key = f"{o['market']}:{o['symbol']}"
        a = agg[key]
        a["n_trades"] += 1
        q = o["qty"] or 0
        a["trades"].append({"date": o["date"], "side": o["side"], "qty": q})
        if o["side"] == "buy":
            a["buy_qty"] += q
            a["cost"] += q * (o["price"] or 0)
            d = o["date"]
            if pd.notna(d) and (a["first_buy"] is None or d < a["first_buy"]):
                a["first_buy"] = d
        else:
            a["sell_qty"] += q
        d = o["date"]
        if pd.notna(d) and (a["last_date"] is None or d >= a["last_date"]):
            a["last_date"] = d
            a["last_side"] = o["side"]
    out = {}
    for key, a in agg.items():
        out[key] = {
            "net_qty": a["buy_qty"] - a["sell_qty"],
            "buy_qty": a["buy_qty"], "sell_qty": a["sell_qty"],
            "avg_buy": (a["cost"] / a["buy_qty"]) if a["buy_qty"] else None,
            "first_buy": a["first_buy"], "last_date": a["last_date"],
            "last_side": a["last_side"], "n_trades": a["n_trades"],
            "trades": a["trades"],
        }
    return out


def _bar_window_days(timeframes) -> int:
    """How many calendar days count as 'within the current bar' for the chosen
    cadence — used to detect a trade you already made this bar."""
    return {"4h": 1, "1d": 1, "1wk": 7}.get(_primary_tf(timeframes), 1)


def apply_order_history(act: dict, hist: dict, timeframes, pos_qty=None) -> tuple:
    """Refine a daily_action() result using this holding's order history.
    Returns (act, note). Guards against over-trading within the current bar,
    but is **quantity-aware**: a recent trade only suppresses a repeat call if
    you already traded *at least* as many shares as the tool is now suggesting.
    A smaller prior trade lets the call stand, netted down to the remaining size.

      • recent BUY  vs an ADD  → if you already added ≥ the suggested shares,
        downgrade to HOLD; else keep ADD for the remaining shares only.
      • recent SELL vs a TRIM  → if you already trimmed ≥ the suggested shares,
        downgrade to HOLD; else keep TRIM for the remaining shares only.
      • a genuine EXIT (over-extended) is always allowed through in full.
    """
    if not hist:
        return act, ""
    note_bits = []
    last = hist.get("last_date")
    days_since = None
    if last is not None and pd.notna(last):
        days_since = (pd.Timestamp.now(tz=None).normalize()
                      - pd.Timestamp(last).normalize()).days
    win = _bar_window_days(timeframes)
    recent = days_since is not None and days_since <= win

    # Shares traded on the guarding side *within the current bar window*.
    def _recent_qty(side: str) -> float:
        tot = 0.0
        for t in (hist.get("trades") or []):
            d = t.get("date")
            if pd.isna(d) or t.get("side") != side:
                continue
            dd = (pd.Timestamp.now(tz=None).normalize()
                  - pd.Timestamp(d).normalize()).days
            if 0 <= dd <= win:
                tot += (t.get("qty") or 0)
        return tot

    guard_side = {"add": "buy", "trim": "sell"}.get(act["side"])
    if recent and guard_side and hist.get("last_side") == guard_side:
        orig_pct = act.get("pct") or 0
        suggested_shares = (pos_qty or 0) * orig_pct / 100.0
        traded = _recent_qty(guard_side)
        verb = "added" if guard_side == "buy" else "trimmed"
        if suggested_shares <= 0 or not pos_qty:
            # No position size to compare against — fall back to recency guard.
            act = dict(act, side="hold",
                       label=f"✋ HOLD — {verb} recently", pct=0)
            note_bits.append(f"{verb} {days_since}d ago — skip repeating")
        elif traded >= suggested_shares - 1e-9:
            # Already traded at least as much as suggested → stand down.
            already_pct = round(traded / pos_qty * 100)
            act = dict(act, side="hold",
                       label=f"✋ HOLD — already {verb} {already_pct}%", pct=0)
            note_bits.append(
                f"{verb} ~{already_pct}% {days_since}d ago "
                f"(≥ suggested {round(orig_pct)}%) — enough for now")
        else:
            # Partial: recommend only the *remaining* size.
            remaining = suggested_shares - traded
            new_pct = max(1, round(remaining / pos_qty * 100))
            already_pct = round(traded / pos_qty * 100)
            act = dict(act, pct=new_pct)
            note_bits.append(
                f"already {verb} ~{already_pct}% {days_since}d ago — "
                f"suggesting the remaining ~{new_pct}%")
    elif recent and hist.get("last_side") == "sell" and act["side"] == "exit":
        note_bits.append(f"sold {days_since}d ago — but over-extended, exit stands")
    elif hist.get("last_side"):
        note_bits.append(f"last {hist['last_side']} {days_since}d ago"
                         if days_since is not None else f"last {hist['last_side']}")
    return act, "; ".join(note_bits)


def _days_held(hist) -> int:
    """Calendar days since the first recorded buy (holding age)."""
    if not hist or hist.get("first_buy") is None or pd.isna(hist.get("first_buy")):
        return None
    return (pd.Timestamp.now(tz=None).normalize()
            - pd.Timestamp(hist["first_buy"]).normalize()).days



SHOW_SIP_TAB = False  # SIP & Exit Plan tab hidden; flip to True to restore it
tab_swp, tab_st, tab_dump, tab_watch, tab_ipo_ath, tab_exit, tab_trigger, tab1, tab2, tab3, tab_rate, tab_life, tab_flows, tab4, tab5, tab6, tab7, tab_alerts = st.tabs(
    ["🏧 SWP Exit — Supertrend", "🔀 Supertrend Reversal",
     "📥 Dump Screen — Supertrend", "⭐ Watchlist — Supertrend",
     "🏆 IPO near ATH — Supertrend",
     "🎯 Exit plan — your holdings", "⚡ Breakout Trigger",
     "🚀 Breakout Candidates", "💰 Allocation", "🔴 Exit Watch", "⭐ Rate My List",
     "🔄 Sector Lifecycle", "🏦 Institutional Flows", "🔎 Details", "🔁 52W-High Retest",
     "🆕 NSE IPOs near launch", "🎯 Near-Zero MACD Coil", "🔔 Alerts"]
)

with tab_swp:
    # ===== SWP Exit — mirror of the Supertrend SIP-entry logic =====
    st.markdown("### 🏧 SWP Exit — systematic withdrawal on Supertrend reversal")
    st.info(
        f"⏱️ **Active stack: {tf_choice}** — the exit is scored on the MTF Supertrend "
        "reversal of the timeframe stack selected in the sidebar (anchor = the largest "
        "timeframe in the stack). Switch to an **MTF** stack for a true multi-timeframe "
        "read."
    )
    st.caption(
        "The **mirror of the Supertrend SIP entry**: on the entry side you SIP most "
        "when the trend is **Bullish and the reversal score is low** (uptrend firmly "
        "intact). Here, on the **exit** side, you run a **SWP (systematic withdrawal)** "
        "when the trend is **Bearish and the reversal score is low** — a downtrend "
        "firmly locked in with little chance of a near-term bounce. The withdrawal % "
        "is largest at the lowest scores and tapers to *Hold* as the score rises (a "
        "bull flip becomes more likely). A **Bullish** holding with a **very high** "
        "score (a top building) also starts a smaller, anticipatory SWP. Upload the "
        "**same** positions / order-history files as the Exit-plan tab."
    )

    swp_downloads_dir = Path(os.environ.get("SCANNER_DOWNLOADS_DIR", Path.home() / "Downloads"))

    swp_up = st.file_uploader("Upload positions CSV", type=["csv"], key="swp_pos_upload")
    swp_sample_files = {"— none —": None}
    if swp_downloads_dir.is_dir():
        for fp in sorted(swp_downloads_dir.glob("*.csv")):
            swp_sample_files[fp.name] = str(fp)
    swp_sample_choice = st.selectbox(
        f"…or load a CSV from your Downloads folder ({swp_downloads_dir})",
        list(swp_sample_files.keys()), index=0, key="swp_pos_sample")

    st.markdown("**➕ Order history (optional)** — avoids re-selling what you just sold")
    swp_order_ups = st.file_uploader(
        "Upload order history (tradebook / transactions)",
        type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="swp_orders_upload")
    swp_hist_pick_files = {}
    if swp_downloads_dir.is_dir():
        for fp in sorted(list(swp_downloads_dir.glob("*.csv")) + list(swp_downloads_dir.glob("*.xlsx"))):
            swp_hist_pick_files[fp.name] = str(fp)
    swp_hist_picks = st.multiselect(
        "…or pick order-history files from Downloads",
        list(swp_hist_pick_files.keys()), key="swp_orders_pick")

    def _st_swp_exit(trend, score):
        """Exit / SWP guidance — the mirror of the Supertrend SIP entry.

        * **Bearish** anchor trend + **low** reversal score = downtrend firmly
          intact, little near-term bounce → withdraw the largest tranche; the %
          tapers as the score rises (a bull flip grows more likely), then Hold.
        * **Bullish** anchor trend + **very high** reversal score = a top is
          building → start a smaller, anticipatory SWP that scales with the score.

        Returns ``{'pct': int, 'label': str}``."""
        if score is None:
            return {"pct": 0, "label": "—"}
        is_bull = isinstance(trend, str) and "Bull" in trend
        if not is_bull:
            if score < 20:
                return {"pct": 30, "label": "🔴 Strong SWP — downtrend locked"}
            if score < 40:
                return {"pct": 20, "label": "🔴 SWP — downtrend"}
            if score < 55:
                return {"pct": 10, "label": "🟠 Light SWP — weak trend"}
            return {"pct": 0, "label": "⏸️ Hold — bull flip building"}
        # Bullish anchor — a high reversal score = top building → begin exiting.
        if score >= 80:
            return {"pct": 20, "label": "🔴 Reversal SWP — top imminent"}
        if score >= 65:
            return {"pct": 12, "label": "🟠 Starter SWP — top building"}
        if score >= 55:
            return {"pct": 6, "label": "🟡 Early trim — topping early"}
        return {"pct": 0, "label": "🟢 Hold — uptrend intact"}

    swp_raw = None
    if swp_up is not None:
        swp_raw = swp_up.getvalue().decode("utf-8", errors="ignore")
    elif swp_sample_files.get(swp_sample_choice):
        try:
            with open(swp_sample_files[swp_sample_choice], "r", encoding="utf-8", errors="ignore") as fh:
                swp_raw = fh.read()
        except Exception as exc:
            st.error(f"Couldn't read the file: {exc}")

    if not swp_raw:
        st.info("👆 Upload a positions CSV (or pick a Downloads sample) to see the SWP exit plan.")
    else:
        # ---- Order-history enrichment (optional) ----
        swp_orders_all, swp_srcs = [], []
        for f in (swp_order_ups or []):
            try:
                o, ofmt = parse_orders_bytes(f.getvalue(), f.name)
            except Exception:
                o, ofmt = [], None
            if o:
                swp_orders_all += o
                swp_srcs.append(f"{f.name} ({ofmt}, {len(o)})")
        for nm in (swp_hist_picks or []):
            path = swp_hist_pick_files.get(nm)
            if not path:
                continue
            try:
                with open(path, "rb") as fh:
                    o, ofmt = parse_orders_bytes(fh.read(), nm)
            except Exception:
                o, ofmt = [], None
            if o:
                swp_orders_all += o
                swp_srcs.append(f"{nm} ({ofmt}, {len(o)})")
        swp_orders_summary = summarize_orders(swp_orders_all) if swp_orders_all else {}
        if swp_orders_summary:
            st.caption(
                f"📗 Order history loaded: {len(swp_orders_all)} trades across "
                f"{len(swp_orders_summary)} symbols — {', '.join(swp_srcs)}.")

        swp_positions, swp_fmt = parse_positions_text(swp_raw)
        if not swp_positions:
            st.error(
                "Couldn't recognise this CSV. Expected a Schwab positions export "
                "(has 'Symbol' + 'Asset Type') or a Zerodha holdings export "
                "(has 'Instrument' + 'LTP')."
            )
        else:
            st.caption(f"Detected **{swp_fmt.upper()}** format — {len(swp_positions)} holdings.")

            # ---- Investable cash per market (for the SIP-entry side) ----
            swp_markets = sorted({p["market"] for p in swp_positions})
            swp_cash = {}
            _ccols = st.columns(max(len(swp_markets), 1))
            for _i, _mk in enumerate(swp_markets):
                with _ccols[_i]:
                    swp_cash[_mk] = st.number_input(
                        f"💵 Investable cash to SIP ({_mk}, {CCY_SYM.get(_mk, '')})",
                        min_value=0.0, value=0.0, step=1000.0,
                        key=f"swp_sip_cash_{_mk}",
                        help="Cash to deploy into your BULLISH holdings that have a LOW "
                        "reversal score. It is spread across sectors (diversified), "
                        "tilted toward names trading near/below their 10-day EMA, and "
                        "trimmed for names you already bought this bar.")

            swp_rows = []
            ce_rows = []
            sip_candidates = []
            # Bearish holdings whose reversal-to-Bull score is at/above this get a
            # small "reversal starter" SIP ahead of a likely bear→bull flip.
            ST_REV_SIP_MIN = 90
            with st.spinner("Scoring your holdings on the Supertrend reversal…"):
                for p in swp_positions:
                    try:
                        sc = score_holding_cached(p["yf"], p["market"], tuple(selected_tf))
                    except Exception:
                        sc = None
                    sym = CCY_SYM.get(p["market"], "")
                    if sc is None:
                        swp_rows.append({
                            "_amt": -1.0, "_color": "",
                            "Ticker": tradingview_url(p["yf"]), "Symbol": p["symbol"],
                            "Market": p["market"], "Qty": p["qty"],
                            "Trend": "❔", "Reversal Score": None,
                            "SWP action": "❔ No data", "SWP %": 0,
                            "Shares to sell": None, f"Value freed {sym}": None,
                        })
                        ce_rows.append({
                            "_amt": -1.0, "_color": "",
                            "Ticker": tradingview_url(p["yf"]), "Symbol": p["symbol"],
                            "Market": p["market"], "Qty": p["qty"],
                            "CE action": "❔ No data", "SWP %": 0,
                            "Shares to sell": None, f"Value freed {sym}": None,
                        })
                        continue
                    trend = sc.get("st_trend")
                    score = sc.get("st_reversal")
                    price = sc.get("price") or p.get("ltp")
                    qty = p.get("qty") or 0
                    swp = _st_swp_exit(trend, score)

                    # Order-history guard: don't re-sell what you already sold this
                    # bar. Reuse the quantity-aware dedup from the Exit-plan tab.
                    hist = swp_orders_summary.get(f"{p['market']}:{p['symbol']}")
                    held_days = _days_held(hist)
                    act = {"side": "trim" if swp["pct"] > 0 else "hold",
                           "label": swp["label"], "pct": swp["pct"],
                           "act_price": price, "day_target": None}
                    act, hist_note = apply_order_history(act, hist, selected_tf, qty)
                    final_pct = act["pct"] if act["side"] != "hold" else 0
                    action_label = act["label"] if act["side"] != "hold" else swp["label"]
                    if act["side"] == "hold" and swp["pct"] > 0:
                        action_label = act["label"]  # downgraded by history

                    shares = int(round(qty * final_pct / 100)) if qty else None
                    value = round(shares * price, 2) if (shares and price) else None
                    # Deeper red for a bigger withdrawal.
                    color = ("#7f1d1d" if final_pct >= 25 else
                             "#b91c1c" if final_pct >= 15 else
                             "#c2410c" if final_pct >= 6 else "")

                    # ---- SIP-entry candidacy ----
                    # (a) Bullish + LOW reversal score → uptrend firmly intact, or
                    # (b) Bearish + VERY HIGH reversal score → a bear→bull flip is
                    #     imminent, so take a small "reversal starter" SIP.
                    is_bull = isinstance(trend, str) and "Bull" in trend
                    is_bear = isinstance(trend, str) and "Bear" in trend
                    bull_sip = is_bull and score is not None and score < 55
                    rev_sip = (is_bear and score is not None
                               and score >= ST_REV_SIP_MIN)
                    if (bull_sip or rev_sip) and price:
                        if bull_sip:
                            tier_w, tier_lbl = ((1.0, "🟢 Strong") if score < 20 else
                                                (0.66, "🟢 SIP") if score < 40 else
                                                (0.33, "🟡 Light"))
                        else:
                            tier_w, tier_lbl = (0.25, "🔵 Reversal starter")
                        ema_dist = sc.get("ema10_dist_pct")
                        # Reward pullbacks toward/below the 10-day EMA; fade extension.
                        if ema_dist is None:
                            ema_f = 1.0
                        elif ema_dist <= 0:
                            ema_f = 1.25
                        elif ema_dist <= 2:
                            ema_f = 1.0
                        elif ema_dist <= 5:
                            ema_f = 0.7
                        else:
                            ema_f = 0.4
                        # Freshness — down-weight a name you already bought this bar.
                        recent_buy = False
                        _ld = hist.get("last_date") if hist else None
                        if (hist and hist.get("last_side") == "buy"
                                and _ld is not None and pd.notna(_ld)):
                            _ds = (pd.Timestamp.now(tz=None).normalize()
                                   - pd.Timestamp(_ld).normalize()).days
                            recent_buy = _ds <= _bar_window_days(selected_tf)
                        fresh_f = 0.5 if recent_buy else 1.0
                        # ---- Protective stop-loss (capital protection) ----
                        # Derive the anchor-timeframe ATR from the Supertrend fields:
                        # dist_atr = |price − flip| in ATR multiples ⇒ ATR = that ÷ dist.
                        flip = sc.get("st_flip_price")
                        dist_atr = sc.get("st_dist_atr")
                        atr = (abs(price - flip) / dist_atr
                               if (flip and dist_atr and dist_atr > 0) else None)
                        if bull_sip:
                            # Bullish → trail the Supertrend flip line (trend stop);
                            # fall back to a 1.5×ATR / structure stop if it sits above.
                            if flip and flip < price:
                                stop_px, stop_basis = flip, "ST flip"
                            elif atr:
                                stop_px, stop_basis = price - 1.5 * atr, "1.5×ATR"
                            else:
                                stop_px, stop_basis = sc.get("stop"), "structure"
                        else:
                            # Reversal starter (bearish) → tighter 1.5×ATR invalidation
                            # (catching a falling knife: cut it if it keeps dropping).
                            if atr:
                                stop_px, stop_basis = price - 1.5 * atr, "1.5×ATR"
                            else:
                                stop_px, stop_basis = sc.get("stop"), "structure"
                        risk_pct = (round((price - stop_px) / price * 100, 1)
                                    if (stop_px and price and stop_px < price) else None)
                        sip_candidates.append({
                            "yf": p["yf"], "Symbol": p["symbol"], "Market": p["market"],
                            "price": price, "score": score, "ema_dist": ema_dist,
                            "trend": trend, "reversal_starter": rev_sip,
                            "existing_value": round((qty or 0) * price, 2),
                            "weight": tier_w * ema_f * fresh_f,
                            "tier": tier_lbl, "fresh": not recent_buy,
                            "stop_px": stop_px, "stop_basis": stop_basis,
                            "risk_pct": risk_pct,
                        })

                    swp_rows.append({
                        "_amt": float(value) if value else 0.0,
                        "_color": color,
                        "Ticker": tradingview_url(p["yf"]),
                        "Symbol": p["symbol"],
                        "Market": p["market"],
                        "Qty": qty,
                        "Held days": held_days,
                        "Gain %": p.get("gain_pct"),
                        "Trend": trend,
                        "Reversal Score": score,
                        "Score 1w ago": sc.get("st_reversal_1w"),
                        "SWP action": action_label,
                        "SWP %": final_pct,
                        "Shares to sell": shares,
                        f"Value freed {sym}": value,
                        "Price": price,
                        "ST flip/stop": sc.get("st_flip_price"),
                        "History note": hist_note or None,
                    })

                    # ---- Chandelier Exit SWP (independent trailing-stop read) ----
                    # A completed close below the 22-bar / 3x ATR long stop is the
                    # exit trigger. Being within one ATR is only a warning, not a sale.
                    ce_stop = sc.get("ce_stop")
                    ce_dist_atr = sc.get("ce_dist_atr")
                    if ce_stop is None or price is None:
                        ce_swp = {"pct": 0, "label": "❔ No Chandelier data"}
                    elif price < ce_stop:
                        ce_swp = {"pct": 30, "label": "🔴 Strong SWP — CE stop breached"}
                    elif ce_dist_atr is not None and ce_dist_atr <= 1:
                        ce_swp = {"pct": 0, "label": "🟡 Watch — within 1 ATR of CE stop"}
                    else:
                        ce_swp = {"pct": 0, "label": "🟢 Hold — CE stop intact"}

                    ce_act = {
                        "side": "trim" if ce_swp["pct"] > 0 else "hold",
                        "label": ce_swp["label"], "pct": ce_swp["pct"],
                        "act_price": price, "day_target": None,
                    }
                    ce_act, ce_hist_note = apply_order_history(
                        ce_act, hist, selected_tf, qty)
                    ce_final_pct = ce_act["pct"] if ce_act["side"] != "hold" else 0
                    ce_label = ce_act["label"] if ce_act["side"] != "hold" else ce_swp["label"]
                    if ce_act["side"] == "hold" and ce_swp["pct"] > 0:
                        ce_label = ce_act["label"]
                    ce_shares = int(round(qty * ce_final_pct / 100)) if qty else None
                    ce_value = (round(ce_shares * price, 2)
                                if (ce_shares and price) else None)
                    ce_color = "#7f1d1d" if ce_final_pct else (
                        "#b45309" if "Watch" in ce_label else "")
                    ce_dist_pct = (round((price / ce_stop - 1) * 100, 1)
                                   if (price and ce_stop) else None)
                    ce_rows.append({
                        "_amt": float(ce_value) if ce_value else 0.0,
                        "_color": ce_color,
                        "Ticker": tradingview_url(p["yf"]),
                        "Symbol": p["symbol"],
                        "Market": p["market"],
                        "Qty": qty,
                        "Held days": held_days,
                        "Gain %": p.get("gain_pct"),
                        "CE action": ce_label,
                        "SWP %": ce_final_pct,
                        "Shares to sell": ce_shares,
                        f"Value freed {sym}": ce_value,
                        "Price": price,
                        "CE long stop": ce_stop,
                        "Distance to CE %": ce_dist_pct,
                        "Distance to CE (ATR)": ce_dist_atr,
                        "CE anchor": sc.get("ce_anchor"),
                        "History note": ce_hist_note or None,
                    })

            if not swp_rows:
                st.info("No holdings could be scored yet.")
            else:
                # Currency of the majority market (for the headline banner).
                mkts = [r["Market"] for r in swp_rows]
                hsym = CCY_SYM.get(max(set(mkts), key=mkts.count), "")
                total = sum(r["_amt"] for r in swp_rows if r["_amt"] > 0)
                n_exit = sum(1 for r in swp_rows if r["SWP %"] > 0)
                if n_exit:
                    top_bits = " · ".join(
                        f"**{r['Symbol']}** {r['SWP action']} "
                        f"({CCY_SYM.get(r['Market'],'')}{r['_amt']:,.0f})"
                        for r in sorted(swp_rows, key=lambda x: x["_amt"], reverse=True)[:6]
                        if r["_amt"] > 0)
                    st.error(
                        f"🔻 **SWP now — withdraw ~{hsym}{total:,.0f} across "
                        f"{n_exit} holding(s):** {top_bits}")
                else:
                    st.success(
                        "✅ No SWP exits right now — no holding is in a locked "
                        "downtrend or topping on this stack.")

                swp_df = (pd.DataFrame(swp_rows)
                          .sort_values(["_amt", "Reversal Score"],
                                       ascending=[False, True])
                          .reset_index(drop=True))

                def _swp_row_color(row):
                    c = row["_color"]
                    style = (f"background-color: {c}; color: #ffffff; font-weight: 600"
                             if c else "")
                    return [style] * len(row)

                styler = swp_df.style.apply(_swp_row_color, axis=1)
                st.dataframe(
                    styler, use_container_width=True, hide_index=True,
                    column_config={
                        "_amt": None, "_color": None,
                        "Ticker": st.column_config.LinkColumn(
                            "Ticker", display_text=r"symbol=(.+)$"),
                        "Gain %": st.column_config.NumberColumn("Gain %", format="%.1f%%"),
                        "Trend": st.column_config.TextColumn(
                            "Trend", help="Current Supertrend direction of the anchor "
                            "(largest) timeframe in the selected stack"),
                        "Reversal Score": st.column_config.ProgressColumn(
                            "Reversal Score", help="0–100. On the exit side a Bearish "
                            "trend with a LOW score = downtrend locked (SWP hardest); "
                            "a Bullish trend with a HIGH score = topping (start SWP)",
                            min_value=0, max_value=100, format="%d"),
                        "Score 1w ago": st.column_config.NumberColumn(
                            "Score 1w ago", help="Reversal score ~1 week ago", format="%d"),
                        "SWP %": st.column_config.NumberColumn(
                            "SWP %", help="% of this holding to withdraw now", format="%d%%"),
                        "ST flip/stop": st.column_config.NumberColumn(
                            "ST flip/stop", help="Anchor Supertrend line — the level "
                            "price must reclaim to flip bullish (a close back above it "
                            "stops the SWP)", format="%.2f"),
                        "Price": st.column_config.NumberColumn("Price", format="%.2f"),
                    },
                )
                st.download_button(
                    "⬇️ Download SWP exit plan",
                    swp_df.drop(columns=["_amt", "_color"]).to_csv(index=False).encode(),
                    file_name=f"swp_exit_{'-'.join(selected_tf)}.csv", mime="text/csv",
                    key="swp_dl")
                st.caption(
                    "**SWP %** scales with how *locked* the reversal is: a Bearish "
                    "holding withdraws 30/20/10% as the score sits below 20/40/55, then "
                    "Holds once ≥55 (a bull flip is building). A Bullish holding starts "
                    "a 6/12/20% SWP as the score climbs through 55/65/80 (a top "
                    "building). **Shares to sell** = SWP % × quantity; rows are ordered "
                    "by **Value freed**, largest exit first. Order history nets down or "
                    "cancels a tranche you've already sold this bar. Educational "
                    "info, not investment advice."
                )

                # ============== Chandelier Exit SWP (independent) ===============
                st.divider()
                st.markdown("#### 🕯️ Chandelier Exit SWP — independent trailing-stop plan")
                st.caption(
                    "Uses the **selected timeframe's anchor** (largest timeframe) with "
                    "a standard **22-bar highest high − 3×ATR(22)** long stop. This is "
                    "an independent profit-protection read: a completed close below the "
                    "stop triggers a 30% SWP; being within 1 ATR is a warning only."
                )
                ce_total = sum(r["_amt"] for r in ce_rows if r["_amt"] > 0)
                ce_n_exit = sum(1 for r in ce_rows if r["SWP %"] > 0)
                if ce_n_exit:
                    ce_top = " · ".join(
                        f"**{r['Symbol']}** {r['CE action']} "
                        f"({CCY_SYM.get(r['Market'], '')}{r['_amt']:,.0f})"
                        for r in sorted(ce_rows, key=lambda x: x["_amt"], reverse=True)[:6]
                        if r["_amt"] > 0)
                    st.error(
                        f"🔻 **Chandelier SWP now — withdraw ~{hsym}{ce_total:,.0f} "
                        f"across {ce_n_exit} holding(s):** {ce_top}")
                else:
                    st.success("✅ No Chandelier stop breaches right now — hold, while "
                               "monitoring any yellow watch rows.")

                ce_df = (pd.DataFrame(ce_rows)
                         .sort_values(["_amt", "Distance to CE %"],
                                      ascending=[False, True])
                         .reset_index(drop=True))

                def _ce_row_color(row):
                    c = row["_color"]
                    style = (f"background-color: {c}; color: #ffffff; font-weight: 600"
                             if c else "")
                    return [style] * len(row)

                st.dataframe(
                    ce_df.style.apply(_ce_row_color, axis=1),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "_amt": None, "_color": None,
                        "Ticker": st.column_config.LinkColumn(
                            "Ticker", display_text=r"symbol=(.+)$"),
                        "Gain %": st.column_config.NumberColumn("Gain %", format="%.1f%%"),
                        "SWP %": st.column_config.NumberColumn(
                            "SWP %", help="30% only after a completed close below the "
                            "Chandelier long stop.", format="%d%%"),
                        "CE long stop": st.column_config.NumberColumn(
                            "CE long stop", help="Highest high over 22 anchor bars minus "
                            "3×ATR(22). A completed close below this level breaches the "
                            "long trailing stop.", format="%.2f"),
                        "Distance to CE %": st.column_config.NumberColumn(
                            "Distance to CE %", help="Price above/below the Chandelier "
                            "long stop. Negative = breached.", format="%.1f%%"),
                        "Distance to CE (ATR)": st.column_config.NumberColumn(
                            "Distance to CE (ATR)", help="Price distance above/below the "
                            "long stop in 22-period ATRs. ≤1 = warning zone.",
                            format="%.2f"),
                        "Price": st.column_config.NumberColumn("Price", format="%.2f"),
                    },
                )
                st.download_button(
                    "⬇️ Download Chandelier SWP plan",
                    ce_df.drop(columns=["_amt", "_color"]).to_csv(index=False).encode(),
                    file_name=f"chandelier_swp_{'-'.join(selected_tf)}.csv",
                    mime="text/csv", key="ce_swp_dl")
                st.caption(
                    "**CE long stop** trails below the highest high of the last 22 "
                    "anchor bars by 3×ATR(22). **Distance to CE %** below zero means "
                    "the exit level has been breached. Order history nets down or "
                    "cancels a tranche already sold this bar. Educational info, not "
                    "investment advice.")

                # ================= SIP deployment (entry side) =================
                st.divider()
                st.markdown(
                    "#### 💧 SIP deployment — bullish (low score) & reversal-imminent "
                    "(bearish, very high score) sectors")
                if not any((swp_cash.get(m) or 0) > 0 for m in swp_markets):
                    st.caption(
                        "Enter investable cash above to get a **diversified** SIP plan: "
                        "cash is split across your bullish, low-reversal-score holdings "
                        "(different sectors), tilted toward names near/below their "
                        "10-day EMA, and reduced for names you already bought this bar. "
                        "A small **starter** slice also goes to bearish names whose "
                        f"reversal-to-Bull score is ≥ 90 (flip looks imminent).")
                else:
                    for _mk in swp_markets:
                        cash = swp_cash.get(_mk) or 0.0
                        if cash <= 0:
                            continue
                        msym = CCY_SYM.get(_mk, "")
                        cands = [c for c in sip_candidates
                                 if c["Market"] == _mk and c["price"]]
                        if not cands:
                            st.info(
                                f"**{_mk}:** no bullish (low-score) or reversal-"
                                "imminent holding to SIP into right now.")
                            continue
                        bull_cands = [c for c in cands
                                      if not c.get("reversal_starter")]
                        rev_cands = [c for c in cands if c.get("reversal_starter")]

                        alloc_by = {}
                        # (1) Reversal starters (bearish + very-high score) share a
                        #     small bounded bucket so each stays "starter" size.
                        if rev_cands:
                            rev_pool = cash * 0.15
                            twr = sum(c["weight"] for c in rev_cands) or 1.0
                            for c in rev_cands:
                                a = rev_pool * c["weight"] / twr
                                alloc_by[id(c)] = min(a, cash * 0.05)  # ≤5% each
                        remaining = cash - sum(alloc_by.values())

                        # (2) Bullish low-score names split the remaining cash with a
                        #     diversified (equal-weight) + conviction water-fill under
                        #     a per-name cap so cash spreads across sectors.
                        if bull_cands:
                            n = len(bull_cands)
                            tot_w = sum(c["weight"] for c in bull_cands) or 1.0
                            fracs = {id(c): 0.5 / n + 0.5 * c["weight"] / tot_w
                                     for c in bull_cands}
                            cap = max(0.40, 1.0 / n)
                            for _ in range(12):
                                excess = 0.0
                                under = []
                                for c in bull_cands:
                                    k = id(c)
                                    if fracs[k] > cap + 1e-9:
                                        excess += fracs[k] - cap
                                        fracs[k] = cap
                                    else:
                                        under.append(c)
                                if excess <= 1e-9 or not under:
                                    break
                                tu = sum(fracs[id(c)] for c in under) or 1.0
                                for c in under:
                                    fracs[id(c)] += excess * fracs[id(c)] / tu
                            for c in bull_cands:
                                alloc_by[id(c)] = remaining * fracs[id(c)]

                        sip_rows = []
                        for c in cands:
                            alloc = alloc_by.get(id(c), 0.0)
                            sh = int(alloc // c["price"]) if c["price"] else 0
                            spend = round(sh * c["price"], 2)
                            add_pct = (round(spend / c["existing_value"] * 100, 1)
                                       if c.get("existing_value") else None)
                            stop_px = c.get("stop_px")
                            # Capital at risk on this entry = shares × (entry − stop).
                            risk_amt = (round(sh * (c["price"] - stop_px), 2)
                                        if (sh and stop_px and stop_px < c["price"])
                                        else None)
                            stop_lbl = (round(stop_px, 2) if stop_px else None)
                            sip_rows.append({
                                "_spend": spend,
                                "_risk": risk_amt or 0.0,
                                "Ticker": tradingview_url(c["yf"]),
                                "Symbol": c["Symbol"],
                                "Trend": c.get("trend"),
                                "Reversal Score": c["score"],
                                "10-EMA dist %": c["ema_dist"],
                                "SIP tier": c["tier"] + ("" if c["fresh"]
                                                         else " · recent buy"),
                                f"Existing {msym}": c.get("existing_value"),
                                "Add %": add_pct,
                                "Buy shares": sh,
                                f"Deploy {msym}": spend,
                                "Price": c["price"],
                                "Stop @": stop_lbl,
                                "Stop basis": c.get("stop_basis"),
                                "Risk %": c.get("risk_pct"),
                                f"Risk {msym}": risk_amt,
                            })
                        deployed = sum(r["_spend"] for r in sip_rows)
                        risk_tot = sum(r["_risk"] for r in sip_rows)
                        n_dep = sum(1 for r in sip_rows if r["_spend"] > 0)
                        top = " · ".join(
                            f"**{r['Symbol']}** {msym}{r['_spend']:,.0f}"
                            for r in sorted(sip_rows, key=lambda x: x["_spend"],
                                            reverse=True)[:6] if r["_spend"] > 0)
                        if n_dep:
                            rk = (f" · max risk to stops **{msym}{risk_tot:,.0f}** "
                                  f"({risk_tot / deployed * 100:.0f}% of deployed)"
                                  if deployed else "")
                            st.success(
                                f"💧 **{_mk} — SIP {msym}{deployed:,.0f} of "
                                f"{msym}{cash:,.0f} ({deployed / cash * 100:.0f}%) "
                                f"across {n_dep} sector(s):** {top}{rk}")
                        else:
                            st.info(
                                f"**{_mk}:** cash too small to buy a full share of any "
                                "candidate at current prices.")
                        sip_df = (pd.DataFrame(sip_rows)
                                  .sort_values("_spend", ascending=False)
                                  .drop(columns=["_spend", "_risk"])
                                  .reset_index(drop=True))
                        st.dataframe(
                            sip_df, use_container_width=True, hide_index=True,
                            column_config={
                                "Ticker": st.column_config.LinkColumn(
                                    "Ticker", display_text=r"symbol=(.+)$"),
                                "Reversal Score": st.column_config.ProgressColumn(
                                    "Reversal Score", help="Lower = uptrend more firmly "
                                    "intact → larger SIP tilt", min_value=0,
                                    max_value=100, format="%d"),
                                "10-EMA dist %": st.column_config.NumberColumn(
                                    "10-EMA dist %", help="Price vs the 10-day EMA. "
                                    "≤ 0 = at/below the EMA (best pullback entry, "
                                    "boosted); the more extended above it, the smaller "
                                    "the SIP.", format="%.2f%%"),
                                "Add %": st.column_config.NumberColumn(
                                    "Add %", help="Deploy amount as a % of your existing "
                                    "position value in this name", format="%.1f%%"),
                                f"Deploy {msym}": st.column_config.NumberColumn(
                                    f"Deploy {msym}", help="Cash to deploy into this "
                                    "name now", format="%.0f"),
                                f"Existing {msym}": st.column_config.NumberColumn(
                                    f"Existing {msym}", format="%.0f"),
                                "Price": st.column_config.NumberColumn("Price", format="%.2f"),
                                "Stop @": st.column_config.NumberColumn(
                                    "Stop @", help="Protective stop-loss for this entry. "
                                    "Bullish → the Supertrend flip line (trailing trend "
                                    "stop); reversal starter → entry − 1.5×ATR.",
                                    format="%.2f"),
                                "Risk %": st.column_config.NumberColumn(
                                    "Risk %", help="Downside from entry to the protective "
                                    "stop (bullish: Supertrend flip / trend stop; "
                                    "reversal starter: entry − 1.5×ATR).",
                                    format="%.1f%%"),
                                f"Risk {msym}": st.column_config.NumberColumn(
                                    f"Risk {msym}", help="Capital at risk on this entry "
                                    "if the stop is hit = shares × (entry − stop).",
                                    format="%.0f"),
                            },
                        )
                        st.download_button(
                            f"⬇️ Download {_mk} SIP plan",
                            sip_df.to_csv(index=False).encode(),
                            file_name=f"swp_sip_{_mk}_{'-'.join(selected_tf)}.csv",
                            mime="text/csv", key=f"swp_sip_dl_{_mk}")
                    st.caption(
                        "**How the SIP is sized:** each bullish holding with a reversal "
                        "score < 55 gets a base weight (Strong < 20, SIP < 40, Light "
                        "< 55), multiplied by a **10-day EMA** factor (×1.25 at/below "
                        "the EMA, fading to ×0.4 when > 5% extended) and a **freshness** "
                        "factor (×0.5 if already bought this bar). Bullish cash is then "
                        "split **50% equally across sectors** (diversification) and "
                        "**50% by conviction**, capped per name so no single sector hogs "
                        "the deployment. A **reversal starter** bucket (max **15%** of "
                        "cash, **≤ 5%** per name) is set aside first for **bearish** "
                        "names whose reversal-to-Bull score is **≥ 90** — a small "
                        "position ahead of a likely bear→bull flip. **Add %** = deploy ÷ "
                        "your existing position. **Capital protection:** every entry "
                        "carries a **stop-loss** — bullish names trail the **Supertrend "
                        "flip line** (trend stop), reversal starters use **entry − "
                        "1.5×ATR**; the **Risk %** and **Risk** columns show the downside "
                        "and cash at risk if the stop is hit. Educational info, not "
                        "investment advice."
                    )


with tab_exit:
    # ============ SECTION B — Exit plan for uploaded positions ============
    st.markdown("### 🎯 Exit plan — your holdings")
    st.info(
        f"⏱️ **Active timeframe: {tf_choice}** — every add/trim/exit call, target, "
        "stop and breakout level below is computed on this timeframe. Change it from "
        "the sidebar (weekly-inclusive blends give slower, higher-timeframe signals)."
    )
    st.caption(
        "Upload your positions CSV (Schwab US *Individual-Positions…* or Zerodha "
        "India *holdings…*) and act **near each bar close of the selected timeframe** "
        "(4h / daily / weekly — set it in the sidebar). For each holding you get a "
        "**two-sided action** — 🟢 add more (with %) while the setup is still building, "
        "or 🟠/🔴 trim/exit (with %) at that bar's target — plus a detailed targets & "
        "stops reference below."
    )

    up = st.file_uploader("Upload positions CSV", type=["csv"], key="pos_upload")

    # Cross-platform (Windows / macOS / Linux): offer any CSV files found in the
    # current user's Downloads folder instead of hardcoding a path.
    downloads_dir = Path(os.environ.get("SCANNER_DOWNLOADS_DIR", Path.home() / "Downloads"))
    sample_files = {"— none —": None}
    if downloads_dir.is_dir():
        for fp in sorted(downloads_dir.glob("*.csv")):
            sample_files[fp.name] = str(fp)
    sample_choice = st.selectbox(
        f"…or load a CSV from your Downloads folder ({downloads_dir})",
        list(sample_files.keys()), index=0)

    # ---- Optional order history (tradebook / transactions) ----
    st.markdown("**➕ Order history (optional)** — refines add/trim/exit calls")
    st.caption(
        "Add your **Zerodha tradebook** (India, `.xlsx`/`.csv`) and/or **Schwab "
        "transactions** (US, `.csv`). Used for true average cost, holding age, and "
        "to avoid telling you to re-add / re-trim a name you already traded this bar."
    )
    order_ups = st.file_uploader(
        "Upload order history (tradebook / transactions)",
        type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="orders_upload")
    hist_pick_files = {}
    if downloads_dir.is_dir():
        for fp in sorted(list(downloads_dir.glob("*.csv")) + list(downloads_dir.glob("*.xlsx"))):
            hist_pick_files[fp.name] = str(fp)
    hist_picks = st.multiselect(
        "…or pick order-history files from Downloads",
        list(hist_pick_files.keys()), key="orders_pick")

    raw_text = None
    if up is not None:
        raw_text = up.getvalue().decode("utf-8", errors="ignore")
    elif sample_files.get(sample_choice):
        try:
            with open(sample_files[sample_choice], "r", encoding="utf-8", errors="ignore") as fh:
                raw_text = fh.read()
        except Exception as exc:
            st.error(f"Couldn't read the file: {exc}")

    if not raw_text:
        st.info("👆 Upload a positions CSV (or pick a Downloads sample) to see an exit plan.")
    else:
        # ---- Build order-history summary (optional enrichment layer) ----
        orders_all = []
        order_srcs = []
        for f in (order_ups or []):
            try:
                o, ofmt = parse_orders_bytes(f.getvalue(), f.name)
            except Exception:
                o, ofmt = [], None
            if o:
                orders_all += o
                order_srcs.append(f"{f.name} ({ofmt}, {len(o)})")
        for nm in (hist_picks or []):
            path = hist_pick_files.get(nm)
            if not path:
                continue
            try:
                with open(path, "rb") as fh:
                    o, ofmt = parse_orders_bytes(fh.read(), nm)
            except Exception:
                o, ofmt = [], None
            if o:
                orders_all += o
                order_srcs.append(f"{nm} ({ofmt}, {len(o)})")
        orders_summary = summarize_orders(orders_all) if orders_all else {}
        if orders_summary:
            st.caption(
                f"📗 Order history loaded: {len(orders_all)} trades across "
                f"{len(orders_summary)} symbols — {', '.join(order_srcs)}.")

        positions, fmt = parse_positions_text(raw_text)
        if not positions:
            st.error(
                "Couldn't recognise this CSV. Expected a Schwab positions export "
                "(has 'Symbol' + 'Asset Type') or a Zerodha holdings export "
                "(has 'Instrument' + 'LTP')."
            )
        else:
            st.caption(f"Detected **{fmt.upper()}** format — {len(positions)} holdings.")
            exit_rows = []
            action_rows = []
            with st.spinner("Scoring your holdings…"):
                for p in positions:
                    try:
                        sc = score_holding_cached(p["yf"], p["market"], tuple(selected_tf))
                    except Exception:
                        sc = None
                    if sc is None:
                        exit_rows.append({
                            "Ticker": tradingview_url(p["yf"]), "Symbol": p["symbol"],
                            "Market": p["market"], "Qty": p["qty"],
                            "Exit score": None, "Action": "❔ No data",
                            "Exit %": None, "Sell @": p.get("ltp"),
                        })
                        action_rows.append({
                            "Symbol": p["symbol"], "Market": p["market"],
                            "Action": "❔ No data", "_side": "hold",
                        })
                        continue
                    pct, label = exit_tier(sc["exit_score"])
                    price = sc["price"] or p.get("ltp")
                    qty = p["qty"] or 0
                    exit_qty = int(round(qty * pct / 100)) if qty else None
                    sym = CCY_SYM.get(p["market"], "")
                    free_val = round(exit_qty * price, 2) if (exit_qty and price) else None

                    # ---- Order history + breakout watch (feeds the decision) ----
                    hist = orders_summary.get(f"{p['market']}:{p['symbol']}")
                    held_days = _days_held(hist)
                    hist_avg = hist.get("avg_buy") if hist else None
                    n_trades = hist.get("n_trades") if hist else None
                    # Days waited for the breakout: how long you've held it (from
                    # order history) if known, else how long it has been coiling.
                    days_waited = held_days if held_days is not None else sc.get("cons_days")
                    bw = breakout_watch(sc, days_waited, breakout_patience)

                    # ---- Two-sided action (add / trim / exit / hold) ----
                    act = daily_action(sc)
                    # Stale-breakout rule: still coiling below the trigger past the
                    # patience window → trim dead money instead of holding/adding.
                    if bw["stale"] and act["side"] in ("hold", "add"):
                        act = {"side": "trim",
                               "label": "⌛ TRIM — breakout not happening",
                               "pct": 25, "act_price": price,
                               "day_target": sc.get("target1")}
                    act, hist_note = apply_order_history(act, hist, selected_tf, qty)
                    # ---- MTF Supertrend overlay (trend stop + reversal timing) ----
                    act, st_note = apply_supertrend(act, sc)
                    a_price = act["act_price"] or price
                    shares = int(round(qty * act["pct"] / 100)) if qty else None
                    if act["side"] == "add":
                        shares_delta = shares                       # buy more
                        cash_delta = -round(shares * a_price, 2) if (shares and a_price) else None
                    elif act["side"] in ("trim", "exit"):
                        shares_delta = -shares if shares else None   # sell
                        cash_delta = round(shares * a_price, 2) if (shares and a_price) else None
                    else:
                        shares_delta, cash_delta = 0, 0
                    action_rows.append({
                        "Ticker": tradingview_url(p["yf"]),
                        "Symbol": p["symbol"],
                        "Market": p["market"],
                        "Qty held": qty,
                        "Gain %": p.get("gain_pct"),
                        "Action": act["label"],
                        "Adjust %": act["pct"] if act["side"] != "hold" else 0,
                        "Shares Δ": shares_delta,
                        "At price": a_price,
                        "Bar target": act["day_target"],
                        "Breakout above": bw["trigger"],
                        "Breakout watch": bw["status"],
                        "Trigger": sc.get("imminence_label"),
                        "Imminence": sc.get("imminence"),
                        "Imm 1w ago": sc.get("imminence_1w_ago"),
                        "% to line": sc.get("dist_to_breakout_pct"),
                        "Breakdown": sc.get("exit_imminence_label"),
                        "Breakdown score": sc.get("exit_imminence"),
                        "Bkdn 1w ago": sc.get("exit_imminence_1w_ago"),
                        "ST trend": sc.get("st_trend"),
                        "ST reversal": sc.get("st_reversal"),
                        "ST rev 1w": sc.get("st_reversal_1w"),
                        "ST flip/stop": sc.get("st_flip_price"),
                        "ST signal": st_note,
                        f"Cash Δ {sym}": cash_delta,
                        "Held days": held_days,
                        "Trades": n_trades,
                        "History note": hist_note or None,
                        "Breakout": sc.get("breakout"),
                        "Exit score": sc.get("exit_score"),
                        "RSI": sc.get("rsi"),
                        "_side": act["side"],
                    })
                    exit_rows.append({
                        "Ticker": tradingview_url(p["yf"]),
                        "Symbol": p["symbol"],
                        "Market": p["market"],
                        "Qty": qty,
                        "Avg cost": p.get("avg"),
                        "Book avg (hist)": round(hist_avg, 2) if hist_avg else None,
                        "Held days": held_days,
                        "Gain %": p.get("gain_pct"),
                        "Exit score": sc["exit_score"],
                        "RSI": sc.get("rsi"),
                        "Action": label,
                        "Exit %": pct,
                        "Exit qty": exit_qty,
                        "Sell @": price,
                        "Breakout above": bw["trigger"],
                        "Breakout watch": bw["status"],
                        f"Frees {sym}": free_val,
                        "ST trend": sc.get("st_trend"),
                        "ST reversal": sc.get("st_reversal"),
                        "ST stop (flip)": sc.get("st_flip_price"),
                        "T1 (scale ~40%)": sc.get("target1"),
                        "T1 %": sc.get("target1_pct"),
                        "T2 (runner)": sc.get("target"),
                        "Trail stop @": sc.get("stop"),
                    })

            # ===== Timeframe-based action plan (cadence = selected timeframe) ===
            _tf_lbl = {"4h": "4-hour", "1d": "daily", "1wk": "weekly"}[_primary_tf(selected_tf)]
            st.markdown(f"#### 🕒 {_tf_lbl.capitalize()} action plan — add / trim / exit")
            markets_held = sorted({r["Market"] for r in action_rows})
            close_bits = []
            in_window = False
            for mk in markets_held:
                msg, win = bar_close_status(mk, selected_tf)
                close_bits.append(msg)
                in_window = in_window or win
            if close_bits:
                (st.success if in_window else st.info)(
                    " · ".join(close_bits)
                    + (f"  — ✅ you're in the last-{ACTION_WINDOW_MIN}-min action window."
                       if in_window else
                       f"  — act within {ACTION_WINDOW_MIN} min of the {_tf_lbl} bar close.")
                )

            act_df = pd.DataFrame(action_rows)
            side_order = {"exit": 0, "trim": 1, "add": 2, "hold": 3}
            act_df["_o"] = act_df["_side"].map(side_order).fillna(3)
            act_df = act_df.sort_values("_o").drop(columns=["_o", "_side"]).reset_index(drop=True)
            todo = [r for r in action_rows if r["_side"] in ("add", "trim", "exit")]
            n_add = sum(1 for r in todo if r["_side"] == "add")
            n_out = sum(1 for r in todo if r["_side"] in ("trim", "exit"))
            st.caption(
                f"**{len(todo)} action(s)** — 🟢 {n_add} to add, 🔴/🟠 {n_out} to "
                f"trim/exit; the rest are HOLD. **Adjust %** = how much of the position "
                "to add (buy a ~1% dip) or trim (sell at price). **Bar target** = the "
                f"level you're playing for over this {_tf_lbl} bar. **Breakout "
                "above** = the price that must be crossed to confirm the breakout; "
                "**Breakout watch** shows ✅ once it clears, or ⏳/⌛ how long it's "
                f"been coiling (a name still stuck after {breakout_patience}d is "
                "flagged stale and trimmed). **Trigger / Imminence** apply the "
                "Breakout-Trigger algorithm to your holdings: an **ADD** only fires "
                "when the base is *ready AND the trigger is firing* — a ready-but-"
                "dormant name shows *⏳ HOLD — ready, no trigger yet* instead of "
                "adding, so you don't pile into something quietly coiling. "
                "**Breakdown / Breakdown score** are the downside mirror: a **TRIM/"
                "EXIT** scales with whether a *top is actually firing now* — an "
                "extended-but-still-trending winner is trimmed lightly (or held to "
                "let it run), and only a real roll-over is cut hard. **Cash Δ** "
                "is negative when you deploy cash, positive when you free it. "
                "**ST trend / ST reversal / ST flip-stop** apply the MTF Supertrend "
                "model: the anchor is the largest timeframe in the selected stack. A "
                "**Bull** holding with a high **ST reversal** score is topping — the "
                "overlay escalates a hold/add into a TRIM; once the anchor actually "
                "flips **Bear**, the **ST flip/stop** line is the trailing-stop EXIT. "
                "A **Bear** holding with a very high reversal score (imminent bull "
                "flip — the backtest's best long trigger) becomes a small anticipatory "
                "starter ADD. **ST signal** explains any such override."
                + (" **Held days / Trades / History note** come from your order "
                   "history — a repeat ADD (just bought) or repeat TRIM (just sold) "
                   "is held back to HOLD this bar; a true over-extended EXIT still "
                   "comes through." if orders_summary else "")
            )
            st.dataframe(
                act_df, use_container_width=True, hide_index=True,
                column_config={
                    "Ticker": st.column_config.LinkColumn(
                        "Ticker", display_text=r"symbol=(.+)$"),
                    "Adjust %": st.column_config.NumberColumn("Adjust %", format="%d%%"),
                    "Gain %": st.column_config.NumberColumn("Gain %", format="%.1f%%"),
                    "Breakout": st.column_config.ProgressColumn(
                        "Breakout", min_value=0, max_value=100, format="%d"),
                    "Imminence": st.column_config.ProgressColumn(
                        "Imminence", help="Breakout-trigger score — is the breakout "
                        "firing NOW (price at the line, volume expanding, ADX rising, "
                        "squeeze firing, MACD growing). ADD calls require this, not "
                        "just a strong base", min_value=0, max_value=100, format="%d"),
                    "Imm 1w ago": st.column_config.NumberColumn(
                        "Imm 1w ago", help="Trigger score ~1 week ago — compare with "
                        "Imminence to see if the trigger is building or fading",
                        format="%d"),
                    "% to line": st.column_config.NumberColumn(
                        "% to line", help="Distance from price to the breakout line "
                        "(+ = still below, − = already above)", format="%.1f%%"),
                    "Breakdown score": st.column_config.ProgressColumn(
                        "Breakdown score", help="Breakdown-trigger score — is a TOP "
                        "firing NOW (price losing support, selling volume expanding, "
                        "momentum rolling over, %B falling, money flowing out). TRIM/"
                        "EXIT size scales with this — extended-but-holding winners are "
                        "trimmed lightly, only real roll-overs are cut hard",
                        min_value=0, max_value=100, format="%d"),
                    "Bkdn 1w ago": st.column_config.NumberColumn(
                        "Bkdn 1w ago", help="Breakdown-trigger score ~1 week ago — "
                        "compare with Breakdown score to see if the roll-over is "
                        "building or easing", format="%d"),
                    "ST reversal": st.column_config.ProgressColumn(
                        "ST reversal", help="MTF Supertrend reversal score (0–100). "
                        "Higher = the anchor trend (largest TF in the selected stack) "
                        "is more likely to flip soon. A BULL holding with a high score "
                        "is topping (trim); a BEAR holding with a high score is a "
                        "bull flip imminent (starter add).", min_value=0, max_value=100,
                        format="%d"),
                    "ST rev 1w": st.column_config.NumberColumn(
                        "ST rev 1w", help="Supertrend reversal score ~1 week ago — "
                        "compare with ST reversal to see if flip pressure is building",
                        format="%d"),
                    "ST flip/stop": st.column_config.NumberColumn(
                        "ST flip/stop", help="The anchor Supertrend line. In a BULL "
                        "trend it is the trailing stop (exit on a close below it); in "
                        "a BEAR trend it is the level price must reclaim to flip "
                        "bullish.", format="%.2f"),
                    "ST signal": st.column_config.TextColumn(
                        "ST signal", help="How the Supertrend overlay changed the call "
                        "(topping trim, trend-stop exit, or bull-flip starter add)"),
                    "Exit score": st.column_config.ProgressColumn(
                        "Exit score", min_value=0, max_value=100, format="%d"),
                },
            )
            ac_dl, ac_email = st.columns([1, 1])
            ac_dl.download_button(
                "⬇️ Download this action plan",
                act_df.to_csv(index=False).encode(),
                file_name=f"{_primary_tf(selected_tf)}_action_plan.csv", mime="text/csv",
            )
            _cfg = alertmod.load_config()
            if ac_email.button("📧 Email me this plan", disabled=not alertmod.config_ready(_cfg)):
                if not todo:
                    lines = ["No actions — everything is HOLD."]
                else:
                    lines = []
                    for r in todo:
                        sym = CCY_SYM.get(r["Market"], "")
                        cash = r.get(f"Cash Δ {sym}")
                        sd = r.get("Shares Δ")
                        sd_str = f"{sd:+}" if isinstance(sd, (int, float)) and sd is not None else "?"
                        lines.append(
                            f"{r['Action']}  {r['Symbol']} ({r['Market']}): "
                            f"{r['Adjust %']}%  {sd_str} sh @ {r['At price']} "
                            f"→ target {r['Bar target']}"
                            + (f"  |  breakout above {r.get('Breakout above')}"
                               if r.get("Breakout above") else "")
                            + (f"  cash {sym}{cash:+}" if isinstance(cash, (int, float)) else "")
                        )
                body = (
                    f"{_tf_lbl.capitalize()} action plan "
                    f"(act near the {_tf_lbl} bar close)\n\n"
                    + "\n".join(lines)
                    + "\n\nEducational info, not investment advice."
                )
                try:
                    alertmod.send_email(
                        _cfg, f"📈 {_tf_lbl.capitalize()} action plan — {len(todo)} action(s)", body)
                    st.success("Emailed your action plan.")
                except Exception as exc:
                    st.error(f"Couldn't send email: {exc}")
            if not alertmod.config_ready(_cfg):
                st.caption("ℹ️ Set up email in the 🔔 Alerts tab to enable emailing this plan.")

            st.divider()
            st.markdown("#### 📋 Detailed exit reference (targets & stops)")
            exit_df = pd.DataFrame(exit_rows).sort_values(
                "Exit score", ascending=False, na_position="last").reset_index(drop=True)
            st.dataframe(
                exit_df, use_container_width=True, hide_index=True,
                column_config={
                    "Ticker": st.column_config.LinkColumn(
                        "Ticker", display_text=r"symbol=(.+)$"),
                    "Exit score": st.column_config.ProgressColumn(
                        "Exit score", min_value=0, max_value=100, format="%d"),
                    "Exit %": st.column_config.NumberColumn("Exit %", format="%d%%"),
                    "T1 %": st.column_config.NumberColumn("T1 %", format="%.1f%%"),
                    "Gain %": st.column_config.NumberColumn("Gain %", format="%.1f%%"),
                    "ST reversal": st.column_config.ProgressColumn(
                        "ST reversal", help="MTF Supertrend reversal score (0–100). "
                        "Higher = the anchor trend is closer to flipping.",
                        min_value=0, max_value=100, format="%d"),
                    "ST stop (flip)": st.column_config.NumberColumn(
                        "ST stop (flip)", help="Anchor Supertrend line — the trailing "
                        "stop while the trend is Bull (exit on a close below it).",
                        format="%.2f"),
                },
            )
            st.download_button(
                "⬇️ Download exit plan CSV",
                exit_df.to_csv(index=False).encode(),
                file_name="exit_plan.csv", mime="text/csv",
            )
            st.caption(
                "**Exit %** scales with the exit score: ≥65 exit fully, ≥50 trim half, "
                "≥40 trim a quarter, else hold. **Sell @** = current price (place the "
                "scale-out limit at/above it). **Breakout above** = the price that must "
                "cross for the breakout to trigger; **Breakout watch** flags a position "
                f"stale (⌛) if it's still coiling below that trigger after "
                f"{breakout_patience} days. **T1 (scale ~40%)** = first resistance — "
                "book ~40% here to lock in a month's worth of gains; **T2 (runner)** = "
                "full measured-move target for the remainder; **Trail stop @** = exit the "
                "rest if it breaks below. Educational info, not investment advice."
            )



# ============ Stock-list Supertrend screeners (Dump / Watchlist tabs) ============
# These two tabs screen an arbitrary NSE stock list from an Excel/CSV export and
# rank every name by the SAME MTF Supertrend reversal-score algorithm the
# "Supertrend Reversal" tab uses. They are independent of the ETF scan, so they
# live before the scan-ready gate below.

def _st_score_color(score):
    if score is None:
        return ""
    if score >= 70:
        return "#1e7d46"
    if score >= 50:
        return "#3f7d46"
    if score >= 30:
        return "#b3952f"
    return "#5a5f64"


def _st_entry_guidance(trend, score):
    """Entry/SIP guidance from anchor trend + reversal score (mirrors the
    Supertrend Reversal tab). Returns ``{'pct': int, 'label': str}``."""
    if score is None:
        return {"pct": 0, "label": "—"}
    is_bull = isinstance(trend, str) and "Bull" in trend
    if is_bull:
        if score < 20:
            return {"pct": 30, "label": "🟢 Strong SIP"}
        if score < 40:
            return {"pct": 20, "label": "🟢 SIP"}
        if score < 55:
            return {"pct": 10, "label": "🟡 Light SIP"}
        return {"pct": 0, "label": "⏸️ Hold — reversal risk"}
    if score >= 80:
        return {"pct": 20, "label": "🟢 Reversal SIP — bull flip imminent"}
    if score >= 65:
        return {"pct": 12, "label": "🟡 Starter SIP — reversal building"}
    if score >= 55:
        return {"pct": 6, "label": "🟠 Early nibble — reversal early"}
    return {"pct": 0, "label": "🚫 Avoid — downtrend"}


@st.cache_data(ttl=900, show_spinner=False)
def _ipo_ath_stats(ticker: str):
    """All-time-high stats from full daily history: ``(ath, current, pct_from_ath)``
    where ``pct_from_ath`` is (current/ATH − 1)·100 (≤0 = below the ATH). ATH uses
    the daily **High** series when present, else Close. Returns ``None`` on no data."""
    try:
        d = cached_ohlcv(ticker).get("1d", pd.DataFrame())
    except Exception:
        return None
    if d is None or d.empty or "Close" not in d:
        return None
    close = d["Close"].dropna()
    if close.empty:
        return None
    high = d["High"].dropna() if "High" in d.columns else close
    ath = float(max(high.max(), close.max()))
    current = float(close.iloc[-1])
    if ath <= 0:
        return None
    return ath, current, (current / ath - 1.0) * 100.0


def _resolve_default_file(path, folder_glob):
    """Use ``path`` if it exists; otherwise fall back to the newest file matching
    ``folder_glob`` in the same folder (handles weekly-dated exports)."""
    if path and os.path.exists(path):
        return path
    try:
        import glob
        folder = os.path.dirname(path) if path else ""
        if folder and os.path.isdir(folder) and folder_glob:
            cands = sorted(glob.glob(os.path.join(folder, folder_glob)),
                           key=os.path.getmtime, reverse=True)
            if cands:
                return cands[0]
    except Exception:
        pass
    return path


def _extract_stock_symbols(src):
    """Read an Excel/CSV export and return the NSE symbols from the sheet whose
    'Symbol' column has the most entries. Returns ``(symbols, error)``."""
    try:
        name = src if isinstance(src, str) else getattr(src, "name", "")
        if str(name).lower().endswith(".csv"):
            book = {"_": pd.read_csv(src)}
        else:
            book = pd.read_excel(src, sheet_name=None)
    except Exception as exc:
        return [], str(exc)
    best = []
    best_pri = -1
    for _sh, df in book.items():
        if df is None or df.empty:
            continue
        symcol = None
        for c in df.columns:
            k = str(c).strip().lower().replace("\n", "")
            if k == "symbol" or k.startswith("symbol"):
                symcol = c
                break
        if symcol is None:
            continue
        vals = []
        for v in df[symcol].dropna().astype(str):
            v = v.strip().upper().replace("\n", "")
            if v and v not in ("SYMBOL", "NAN"):
                vals.append(v)
        # Prefer the main TradingView export sheet ("Dump of stocks_<date>") over
        # the auxiliary corporate-action sheets; break ties by symbol count.
        pri = 1 if str(_sh).strip().lower().startswith("dump of stocks") else 0
        if (pri, len(vals)) > (best_pri, len(best)):
            best, best_pri = vals, pri
    seen, out = set(), []
    for v in best:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out, None


def render_stock_screen(default_path, folder_glob, key_prefix, heading, blurb):
    st.subheader(heading)
    st.caption(blurb)
    tf_stack = tuple(selected_tf)
    st_combo = _tf_stack_label(tf_stack)
    anchor_tf = _tf_stack_label((tf_stack[-1],))
    st.info(
        f"⏱️ Ranking on the sidebar timeframe stack **{st_combo}** (anchor "
        f"**{anchor_tf}**) — the same Supertrend reversal-score algorithm as the "
        "**Supertrend Reversal** tab. Change the stack from the sidebar **Timeframe** "
        "control. **Higher score = the anchor trend is more likely to reverse soon.**"
    )
    resolved = _resolve_default_file(default_path, folder_glob)
    up = st.file_uploader("Upload the stock list (Excel / CSV)",
                          type=["xlsx", "xls", "csv"], key=f"{key_prefix}_upl")
    path = st.text_input("…or read from this path", value=resolved,
                         key=f"{key_prefix}_path")
    src = up if up is not None else (path if (path and os.path.exists(path)) else None)
    if src is None:
        st.warning("Upload a file, or enter a valid path above, to screen the list.")
        return
    symbols, err = _extract_stock_symbols(src)
    if err:
        st.error(f"Could not read the file: {err}")
        return
    if not symbols:
        st.error("No **Symbol** column found in the file.")
        return
    src_label = getattr(up, "name", None) or os.path.basename(path)
    st.caption(f"📄 **{src_label}** — found **{len(symbols)}** NSE symbols.")
    c1, c2, c3 = st.columns([3, 1, 1])
    n = c1.slider("Symbols to screen (from the top of the list)", 5,
                  int(min(len(symbols), 200)), int(min(30, len(symbols))),
                  key=f"{key_prefix}_n",
                  help="Each symbol needs multi-timeframe data (~1–2s each). Start "
                  "small, then increase. Results are cached 15 min.")
    only_actionable = c2.checkbox("Actionable only", value=False,
                                  key=f"{key_prefix}_act",
                                  help="Hide Hold / Avoid rows (show only SIP names).")
    run = c3.button("▶️ Run screen", key=f"{key_prefix}_run", type="primary")

    rows_key, sig_key, fails_key = (f"{key_prefix}_rows", f"{key_prefix}_sig",
                                    f"{key_prefix}_fails")
    sig = f"{src_label}|{st_combo}|{n}"
    if run:
        rows, fails = [], 0
        subset = symbols[:n]
        prog = st.progress(0.0, text="Scoring…")
        for i, sy in enumerate(subset):
            tk = sy if sy.upper().endswith(".NS") else f"{sy}.NS"
            data = st_reversal_tf_cached(tk, tf_stack)
            if data and data.get("score") is not None:
                score = data.get("score")
                da = data.get("dist_atr")
                entry = _st_entry_guidance(data.get("trend"), score)
                rows.append({
                    "_score": score, "_color": _st_score_color(score),
                    "_pct": entry["pct"],
                    "Ticker": tradingview_url(tk), "Symbol": sy,
                    "Trend": data.get("trend"),
                    "SIP action": entry["label"], "SIP %": entry["pct"],
                    "Reversal Score": score,
                    "Reversal to": data.get("reversal_to"),
                    "Score 1w ago": data.get("score_1w"),
                    "Score 2w ago": data.get("score_2w"),
                    "Stack": data.get("stack"),
                    "Dist-to-flip (ATR)": round(da, 2) if da is not None else None,
                })
            else:
                fails += 1
            prog.progress((i + 1) / len(subset),
                          text=f"Scored {i + 1}/{len(subset)}  ·  {sy}")
        prog.empty()
        st.session_state[rows_key] = rows
        st.session_state[sig_key] = sig
        st.session_state[fails_key] = fails

    rows = st.session_state.get(rows_key)
    if rows is None:
        st.info("Click **▶️ Run screen** to score and rank the list.")
        return
    if st.session_state.get(sig_key) != sig:
        st.warning("Settings changed since the last run — click **▶️ Run screen** "
                   "to refresh.")
    fails = st.session_state.get(fails_key, 0)
    if not rows:
        st.error(f"No Supertrend data returned for the screened symbols "
                 f"({fails} had no data — check the symbols are valid NSE tickers).")
        return
    if only_actionable:
        rows = [r for r in rows if r["_pct"] > 0]
    st.success(
        f"✅ Ranked **{len(rows)}** symbols on the {st_combo} Supertrend reversal "
        f"score" + (f" · {fails} skipped (no data)" if fails else ""))

    colcfg = {
        "_score": None, "_color": None, "_pct": None,
        "Ticker": st.column_config.LinkColumn(
            "Ticker", display_text=r"symbol=(.+)$"),
        "Reversal Score": st.column_config.ProgressColumn(
            "Reversal Score", help="0–100. Higher = the anchor trend is more "
            "likely to reverse soon (weighted lower-TF flip cascade + ATR "
            "proximity).", min_value=0, max_value=100, format="%d"),
        "SIP action": st.column_config.TextColumn(
            "SIP action", help="Entry guidance: bullish + low score (uptrend "
            "intact) or bearish + very-high score (bull flip imminent)."),
        "SIP %": st.column_config.NumberColumn("SIP %", format="%d%%"),
        "Score 1w ago": st.column_config.NumberColumn("Score 1w ago", format="%d"),
        "Score 2w ago": st.column_config.NumberColumn("Score 2w ago", format="%d"),
        "Dist-to-flip (ATR)": st.column_config.NumberColumn(
            "Dist-to-flip (ATR)", help="How far the anchor price is from its "
            "Supertrend flip line, in ATRs. Smaller = riper for a flip.",
            format="%.2f"),
    }

    def _color_rows(row):
        c = row["_color"]
        s = (f"background-color: {c}; color: #ffffff; font-weight: 600"
             if c else "")
        return [s] * len(row)

    def _tbl(group, title, cap, ascending, suffix):
        st.markdown(f"#### {title}")
        if not group:
            st.caption("_None in this group._")
            return
        st.caption(cap)
        g = (pd.DataFrame(group).sort_values("_score", ascending=ascending)
             .reset_index(drop=True))
        g.insert(0, "Rank", range(1, len(g) + 1))
        st.dataframe(g.style.apply(_color_rows, axis=1), use_container_width=True,
                     hide_index=True, column_config=colcfg)
        st.download_button(
            "⬇️ Download CSV",
            g.drop(columns=["_score", "_color", "_pct"]).to_csv(index=False).encode(),
            file_name=f"{key_prefix}_{suffix}_{st_combo.replace('+', '-')}.csv",
            mime="text/csv", key=f"{key_prefix}_dl_{suffix}")

    bull = [r for r in rows if "Bull" in (r["Trend"] or "")]
    bear = [r for r in rows if "Bull" not in (r["Trend"] or "")]
    _tbl(bull, f"🟢 Bullish trend ({len(bull)})",
         "Uptrend intact — **ascending** by reversal score: lowest reversal risk "
         "(strongest SIP) first.", True, "bull")
    _tbl(bear, f"🔴 Bearish trend ({len(bear)})",
         "Downtrend — **descending** by reversal score: closest to a bull flip "
         "(reversal-starter candidates) first.", False, "bear")
    st.caption("Educational info, not investment advice.")


with tab_dump:
    render_stock_screen(
        r"C:\Users\ajaysingla\OneDrive\msft_backup\AJAY\stocks\InHouse_Stock_Screener\Dump of stocks_8th August 2026.xlsx",
        "Dump of stocks_*.xlsx", "dump",
        "📥 Dump Screen — Supertrend ranking",
        "Screen a full **stock dump** (e.g. a TradingView export) and rank every "
        "name by the MTF Supertrend reversal score — the same algorithm as the "
        "**Supertrend Reversal** tab.")

with tab_watch:
    render_stock_screen(
        r"C:\Users\ajaysingla\OneDrive\msft_backup\AJAY\stocks\InHouse_Stock_Screener\Watchlist Friendship Day 2026.xlsx",
        "Watchlist*.xlsx", "watch",
        "⭐ Watchlist — Supertrend ranking",
        "Screen your **curated watchlist** and rank names by the MTF Supertrend "
        "reversal score — the same algorithm as the **Supertrend Reversal** tab.")


with tab_ipo_ath:
    st.subheader("🏆 Recent IPOs trading near their all-time high — Supertrend rank")
    st.caption(
        "Screens **NSE IPOs listed within your chosen window** (fetched live from the "
        "official NSE past-issues API), keeps only those trading **close to their "
        "all-time high** (within the ± band you set), and rates each by the **MTF "
        "Supertrend reversal score** — the same algorithm as the **Supertrend "
        "Reversal** tab. A young stock riding near its ATH with a **low** reversal "
        "score is a fresh leader whose uptrend is still intact; the **% from ATH** "
        "column shows exactly how far below its peak each name is trading."
    )
    tf_stack = tuple(selected_tf)
    st_combo = _tf_stack_label(tf_stack)
    anchor_tf = _tf_stack_label((tf_stack[-1],))
    st.info(
        f"⏱️ Ranking on the sidebar timeframe stack **{st_combo}** (anchor "
        f"**{anchor_tf}**). Change it from the sidebar **Timeframe** control. "
        "**Higher reversal score = the anchor trend is more likely to reverse soon** "
        "(i.e. a near-ATH name with a **high** score may be topping)."
    )

    ia = st.columns([1.2, 1.5, 1.1])
    _ipo_ath_ranges = {"6 months": 180, "1 year": 365, "2 years": 730}
    with ia[0]:
        ipoath_win = st.selectbox("IPO listed within", list(_ipo_ath_ranges.keys()),
                                  index=1, key="ipoath_win")
    with ia[1]:
        ipoath_band = st.slider("Near ATH — within ± % of all-time high", 1, 30, 10, 1,
                                key="ipoath_band",
                                help="Keep names whose current price is within this "
                                "percent of their all-time high.")
    with ia[2]:
        ipoath_board = st.selectbox("Board", ["All", "Mainboard", "SME"],
                                    key="ipoath_board")

    run_ipoath = st.button("▶️ Fetch & rank near-ATH IPOs", type="primary",
                           key="ipoath_run")

    if run_ipoath:
        within_days = _ipo_ath_ranges[ipoath_win]
        try:
            with st.spinner("Fetching recent IPOs from NSE…"):
                ipo_list = cached_recent_ipos(within_days)
        except Exception as exc:
            st.error(
                "Couldn't fetch the IPO list from NSE right now "
                f"({type(exc).__name__}). NSE may be rate-limiting — try again shortly."
            )
            ipo_list = None

        if ipo_list is not None:
            if ipoath_board != "All":
                ipo_list = [r for r in ipo_list if r.get("board") == ipoath_board]
            band = float(ipoath_band)
            rows, near, no_data = [], 0, 0
            total = max(1, len(ipo_list))
            prog = st.progress(0.0, text="Screening near-ATH IPOs…")
            for i, r in enumerate(ipo_list, start=1):
                sym = r["symbol"]
                prog.progress(i / total, text=f"Checking {sym} ({i}/{total})")
                tk = f"{sym}.NS"
                stats = _ipo_ath_stats(tk)
                if stats is None:
                    no_data += 1
                    continue
                ath, current, pct_from_ath = stats
                if abs(pct_from_ath) > band:
                    continue
                near += 1
                data = st_reversal_tf_cached(tk, tf_stack)
                if not data or data.get("score") is None:
                    no_data += 1
                    continue
                score = data.get("score")
                da = data.get("dist_atr")
                entry = _st_entry_guidance(data.get("trend"), score)
                rows.append({
                    "_score": score, "_color": _st_score_color(score),
                    "_pct": entry["pct"],
                    "Ticker": tradingview_url(tk), "Symbol": sym,
                    "Company": r.get("company", ""),
                    "Board": r.get("board", ""),
                    "Listed": str(r["listing_date"]), "Days": r["days_since"],
                    "Trend": data.get("trend"),
                    "SIP action": entry["label"], "SIP %": entry["pct"],
                    "Reversal Score": score,
                    "Reversal to": data.get("reversal_to"),
                    "ATH ₹": round(ath, 2), "Now ₹": round(current, 2),
                    "% from ATH": round(pct_from_ath, 1),
                    "Score 1w ago": data.get("score_1w"),
                    "Score 2w ago": data.get("score_2w"),
                    "Stack": data.get("stack"),
                    "Dist-to-flip (ATR)": round(da, 2) if da is not None else None,
                })
            prog.empty()
            st.session_state["ipoath_rows"] = rows
            st.session_state["ipoath_sig"] = f"{ipoath_win}|{band}|{ipoath_board}|{st_combo}"
            st.session_state["ipoath_meta"] = {
                "total": len(ipo_list), "near": near, "no_data": no_data,
                "win": ipoath_win, "band": band, "combo": st_combo,
            }

    ath_rows = st.session_state.get("ipoath_rows")
    if ath_rows is None:
        st.info("👆 Click **Fetch & rank near-ATH IPOs** to pull the NSE IPO list, "
                "keep the ones trading near their all-time high, and score them.")
    elif not ath_rows:
        m = st.session_state.get("ipoath_meta", {})
        st.warning(
            f"Scanned **{m.get('total', 0)}** IPOs from the last **{m.get('win', '')}** — "
            f"none are trading within **±{int(m.get('band', ipoath_band))}%** of their "
            "all-time high with Supertrend data. Widen the band or the listing window."
        )
    else:
        m = st.session_state.get("ipoath_meta", {})
        cur_sig = f"{ipoath_win}|{float(ipoath_band)}|{ipoath_board}|{st_combo}"
        if st.session_state.get("ipoath_sig") != cur_sig:
            st.warning("Settings changed since the last run — click **Fetch & rank "
                       "near-ATH IPOs** to refresh.")
        st.success(
            f"✅ **{len(ath_rows)}** IPOs from the last **{m.get('win', '')}** are within "
            f"**±{int(m.get('band', ipoath_band))}%** of their all-time high, ranked on "
            f"the **{m.get('combo', st_combo)}** Supertrend reversal score "
            f"(scanned {m.get('total', 0)}, {m.get('near', 0)} near ATH)."
        )

        ath_colcfg = {
            "_score": None, "_color": None, "_pct": None,
            "Ticker": st.column_config.LinkColumn(
                "Ticker", display_text=r"symbol=(.+)$"),
            "Reversal Score": st.column_config.ProgressColumn(
                "Reversal Score", help="0–100. Higher = the anchor trend is more "
                "likely to reverse soon.", min_value=0, max_value=100, format="%d"),
            "% from ATH": st.column_config.NumberColumn(
                "% from ATH", help="Current price vs all-time high "
                "(0% = at the high, −5% = 5% below it).", format="%.1f%%"),
            "ATH ₹": st.column_config.NumberColumn("ATH ₹", format="%.2f"),
            "Now ₹": st.column_config.NumberColumn("Now ₹", format="%.2f"),
            "SIP action": st.column_config.TextColumn(
                "SIP action", help="Entry guidance from anchor trend + reversal score."),
            "SIP %": st.column_config.NumberColumn("SIP %", format="%d%%"),
            "Score 1w ago": st.column_config.NumberColumn("Score 1w ago", format="%d"),
            "Score 2w ago": st.column_config.NumberColumn("Score 2w ago", format="%d"),
            "Dist-to-flip (ATR)": st.column_config.NumberColumn(
                "Dist-to-flip (ATR)", help="Distance of the anchor price from its "
                "Supertrend flip line, in ATRs. Smaller = riper for a flip.",
                format="%.2f"),
        }

        def _ath_color_rows(row):
            c = row["_color"]
            s = (f"background-color: {c}; color: #ffffff; font-weight: 600"
                 if c else "")
            return [s] * len(row)

        def _ath_tbl(group, title, cap, ascending, suffix):
            st.markdown(f"#### {title}")
            if not group:
                st.caption("_None in this group._")
                return
            st.caption(cap)
            g = (pd.DataFrame(group).sort_values("_score", ascending=ascending)
                 .reset_index(drop=True))
            g.insert(0, "Rank", range(1, len(g) + 1))
            st.dataframe(g.style.apply(_ath_color_rows, axis=1),
                         use_container_width=True, hide_index=True,
                         column_config=ath_colcfg)
            st.download_button(
                "⬇️ Download CSV",
                g.drop(columns=["_score", "_color", "_pct"]).to_csv(index=False).encode(),
                file_name=f"ipo_near_ath_{suffix}_{st_combo.replace('+', '-')}.csv",
                mime="text/csv", key=f"ipoath_dl_{suffix}")

        bull = [r for r in ath_rows if "Bull" in (r["Trend"] or "")]
        bear = [r for r in ath_rows if "Bull" not in (r["Trend"] or "")]
        _ath_tbl(bull, f"🟢 Bullish trend ({len(bull)})",
                 "Uptrend intact near the ATH — **ascending** by reversal score: "
                 "lowest reversal risk (strongest fresh leader) first.", True, "bull")
        _ath_tbl(bear, f"🔴 Bearish trend ({len(bear)})",
                 "Rolling over near the highs — **descending** by reversal score: "
                 "closest to a bull flip first.", False, "bear")
        st.caption(
            "ATH = highest daily high in the available history (recent IPOs, so this "
            "is effectively the post-listing peak). Very new IPOs without yfinance "
            "history are skipped. Educational info, not investment advice.")


# --- Gate: scanner tabs need a scan; the tabs above do not ---
if not scan_ready:
    st.stop()


with tab_st:
    st.subheader("🔀 Supertrend Reversal — MTF trend-reversal probability")
    st.caption(
        "Lower timeframes flip **before** the higher one, so a pending reversal of "
        "the anchor trend shows up as a **bottom-up cascade** of Supertrend flips. "
        "The **Reversal Score** (0–100) blends that weighted cascade with how close "
        "the anchor's price sits to its own Supertrend flip line (in ATRs). "
        "**Higher = the current trend is more likely to reverse soon.** The **Trend** "
        "column is the anchor timeframe's *current* Supertrend direction; **Stack** "
        "shows each timeframe's direction (🟢 up / 🔴 down), smallest→largest. "
        "**SIP action / % / amount**: deploy when the setup favours an uptrend — a "
        "**bullish** anchor trend with a **low** reversal score (uptrend intact), *or* "
        "a **bearish** trend with a **very high** reversal score (bull flip imminent → "
        "smaller anticipatory starter SIP). Set your **investable cash** below each "
        "market — rows are ordered by entry size (largest SIP first)."
    )

    st_combo = ST_STACK_MAP.get(tuple(selected_tf), "1h+2h+4h+1d")
    anchor_tf = ST_COMBOS[st_combo][-1]
    st.info(
        f"⏱️ Timeframe stack **{st_combo}** — set it from the sidebar **Timeframe** "
        f"control (pick an **MTF …** option). Predicting a reversal of the "
        f"**{anchor_tf}** Supertrend (the largest timeframe in the stack) using the "
        "lower-timeframe flip cascade + ATR proximity. **Score 1w / 2w ago** let you "
        "see whether the reversal pressure is **building** (rising) or **fading** "
        "(falling)."
    )

    def _st_color(score):
        if score is None:
            return ""
        if score >= 70:
            return "#1e7d46"
        if score >= 50:
            return "#3f7d46"
        if score >= 30:
            return "#b3952f"
        return "#5a5f64"

    def _st_sip_entry(trend, score):
        """Entry / SIP guidance from the anchor trend + reversal score.

        Two entry cases:
        * **Bullish** anchor trend with a **low** reversal score = uptrend intact,
          little near-term top risk → deploy the most cash; the % tapers as the
          reversal score climbs, then stops (Hold).
        * **Bearish** anchor trend with a **very high** reversal score = a bull
          flip is imminent → start a (more conservative, anticipatory) SIP that
          scales up with the score.

        Returns ``{'pct': int, 'label': str}``."""
        if score is None:
            return {"pct": 0, "label": "—"}
        is_bull = isinstance(trend, str) and "Bull" in trend
        if is_bull:
            if score < 20:
                return {"pct": 30, "label": "🟢 Strong SIP"}
            if score < 40:
                return {"pct": 20, "label": "🟢 SIP"}
            if score < 55:
                return {"pct": 10, "label": "🟡 Light SIP"}
            return {"pct": 0, "label": "⏸️ Hold — reversal risk"}
        # Bearish anchor trend — a high reversal score = bull flip building.
        if score >= 80:
            return {"pct": 20, "label": "🟢 Reversal SIP — bull flip imminent"}
        if score >= 65:
            return {"pct": 12, "label": "🟡 Starter SIP — reversal building"}
        if score >= 55:
            return {"pct": 6, "label": "🟠 Early nibble — reversal early"}
        return {"pct": 0, "label": "🚫 Avoid — downtrend"}

    def _st_row_color(row):
        c = row["_color"]
        style = (f"background-color: {c}; color: #ffffff; font-weight: 600"
                 if c else "")
        return [style] * len(row)

    st_colcfg = {
        "_score": None, "_color": None, "_amt": None,
        "Ticker": st.column_config.LinkColumn(
            "Ticker", display_text=r"symbol=(.+)$"),
        "Trend": st.column_config.TextColumn(
            "Trend", help=f"Current Supertrend direction of the anchor "
            f"({anchor_tf}) timeframe"),
        "SIP action": st.column_config.TextColumn(
            "SIP action", help="Entry guidance. Deploy when the setup favours "
            "an uptrend: (a) a BULLISH anchor trend with a LOW reversal score "
            "(uptrend intact) — size tapers as reversal risk rises, then Hold; "
            "or (b) a BEARISH anchor trend with a VERY HIGH reversal score "
            "(bull flip imminent) — a smaller, anticipatory starter SIP that "
            "scales up with the score. Otherwise avoid."),
        "SIP %": st.column_config.NumberColumn(
            "SIP %", help="Suggested % of this market's investable cash to "
            "deploy now", format="%d%%"),
        "SIP amount": st.column_config.TextColumn(
            "SIP amount", help="SIP % × investable cash."),
        "Reversal Score": st.column_config.ProgressColumn(
            "Reversal Score", help="0–100. Higher = the anchor trend is more "
            "likely to reverse soon (weighted lower-TF flip cascade + ATR "
            "proximity of the anchor to its own flip line)",
            min_value=0, max_value=100, format="%d"),
        "Score 1w ago": st.column_config.NumberColumn(
            "Score 1w ago", help="Reversal score ~1 week ago — compare with "
            "today to see if reversal pressure is building or fading",
            format="%d"),
        "Score 2w ago": st.column_config.NumberColumn(
            "Score 2w ago", help="Reversal score ~2 weeks ago", format="%d"),
        "Reversal to": st.column_config.TextColumn(
            "Reversal to", help="Direction the trend would flip TO if it "
            "reverses"),
        "Stack": st.column_config.TextColumn(
            "Stack", help="Per-timeframe Supertrend direction "
            "(🟢 up / 🔴 down), ordered smallest→largest timeframe"),
        "Dist-to-flip (ATR)": st.column_config.NumberColumn(
            "Dist-to-flip (ATR)", help="How far the anchor's price is from its "
            "Supertrend flip line, in ATRs. Smaller = riper for a flip",
            format="%.2f"),
    }

    def _render_st_group(group_rows, title, caption, ascending, mkt, suffix):
        st.markdown(f"#### {title}")
        if not group_rows:
            st.caption("_None in this group right now._")
            return
        if caption:
            st.caption(caption)
        gdf = (pd.DataFrame(group_rows)
               .sort_values("_score", ascending=ascending)
               .reset_index(drop=True))
        styler = gdf.style.apply(_st_row_color, axis=1)
        st.dataframe(styler, use_container_width=True, hide_index=True,
                     column_config=st_colcfg)
        st.download_button(
            "⬇️ Download CSV",
            gdf.drop(columns=["_score", "_color", "_amt"]).to_csv(index=False).encode(),
            file_name=f"supertrend_reversal_{mkt}_{suffix}_{st_combo.replace('+', '-')}.csv",
            mime="text/csv", key=f"st_dl_{mkt}_{suffix}",
        )

    for mkt in selected_markets:
        pool = [s for m, s in results if m == mkt]
        if not pool:
            st.info(f"No {mkt} sectors scanned yet.")
            continue

        sym = CCY_SYM.get(mkt, "$")
        st.markdown(f"### {mkt} market")
        cash = st.number_input(
            f"💵 Investable cash to deploy ({mkt}, {sym})",
            min_value=0.0, value=100000.0, step=1000.0,
            key=f"st_cash_{mkt}",
            help="Cash available for this market. Each row's SIP % is applied to "
            "this to size the entry, and rows are ordered by entry size (largest "
            "first).",
        )

        rows, ripe = [], []
        for s in pool:
            data = (getattr(s, "st_mtf", {}) or {}).get(st_combo)
            if not data:
                continue
            score = data.get("score")
            da = data.get("dist_atr")
            entry = _st_sip_entry(data.get("trend"), score)
            amt = round(cash * entry["pct"] / 100.0)
            rows.append({
                "_score": score if score is not None else -1,
                "_amt": amt,
                "_color": _st_color(score),
                "Ticker": tradingview_url(s.ticker),
                "Symbol": s.ticker,
                "Sector": s.name,
                "Trend": data.get("trend"),
                "SIP action": entry["label"],
                "SIP %": entry["pct"],
                "SIP amount": f"{sym}{amt:,.0f}" if entry["pct"] > 0 else "—",
                "Reversal Score": score,
                "Score 1w ago": data.get("score_1w"),
                "Score 2w ago": data.get("score_2w"),
                "Reversal to": data.get("reversal_to"),
                "Stack": data.get("stack"),
                "Dist-to-flip (ATR)": round(da, 2) if da is not None else None,
            })
            if score is not None and score >= 60 and "Bull" in (data.get("trend") or ""):
                ripe.append((s, data))

        if not rows:
            st.info(f"No Supertrend data available for {mkt} yet.")
            continue

        deploy_total = sum(r["_amt"] for r in rows)
        n_entries = sum(1 for r in rows if r["_amt"] > 0)
        if n_entries:
            top_bits = " · ".join(
                f"**{r['Symbol']}** {r['SIP action']} {r['SIP amount']}"
                for r in sorted(rows, key=lambda x: x["_amt"], reverse=True)[:6]
                if r["_amt"] > 0
            )
            st.success(
                f"💧 **SIP now — {sym}{deploy_total:,.0f} of {sym}{cash:,.0f} "
                f"({(deploy_total / cash * 100) if cash else 0:.0f}% of cash) across "
                f"{n_entries} name(s):** {top_bits}"
            )
        else:
            st.caption("No names to SIP into right now on this stack.")

        if ripe:
            ripe.sort(key=lambda x: x[1]["score"], reverse=True)
            picks = " · ".join(
                f"**{s.ticker}** ({s.name}, score {d['score']:.0f}, {d['reversal_to']})"
                for s, d in ripe[:8]
            )
            st.warning(
                f"🔺 **Bullish names with high reversal pressure (topping — caution "
                f"on adds):** {picks}"
            )

        # Split into two tables by current anchor trend.
        bull_rows = [r for r in rows if "Bull" in (r["Trend"] or "")]
        bear_rows = [r for r in rows if "Bull" not in (r["Trend"] or "")]
        # Bullish → ascending by reversal score (lowest reversal risk / strongest
        # SIP first). Bearish → descending (closest to a bull flip first).
        _render_st_group(
            bull_rows, f"🟢 Bullish trend ({len(bull_rows)})",
            "Uptrend intact — sorted **ascending** by reversal score: lowest "
            "reversal risk (strongest SIP) at the top.", True, mkt, "bull")
        _render_st_group(
            bear_rows, f"🔴 Bearish trend ({len(bear_rows)})",
            "Downtrend — sorted **descending** by reversal score: closest to a bull "
            "flip (best reversal SIP) at the top.", False, mkt, "bear")

    st.caption(
        "**How to read it:** a rising score across *Score 2w ago → 1w ago → now* means "
        "the lower timeframes are flipping one-by-one against the anchor trend and the "
        "anchor price is closing in on its Supertrend line — a reversal is building. "
        "Use it as an **early lead** on daily/weekly Supertrend flips, then confirm on "
        "the anchor timeframe itself. Educational info, not investment advice."
    )


with tab1:
    st.subheader("Ranked by breakout readiness + imminence")
    st.caption(
        "🟢 green = score rising fast (demand building / tightening) · "
        "🔴 red = score falling fast · **Score** = 60% readiness + 40% imminence "
        "(is it firing now) · **Action** column suggests staged SIP / "
        "profit-booking · click a **Ticker** to open its TradingView chart."
    )
    ranked = (df.sort_values("Score", ascending=False).reset_index(drop=True)
              if "Score" in df.columns else df)
    render_table(ranked)
    st.download_button(
        "⬇️ Download CSV",
        ranked.drop(columns=["_delta"], errors="ignore").to_csv(index=False).encode(),
        file_name="breakout_scan.csv", mime="text/csv",
    )

    # -------- Today's SIP entry plan (position- & history-aware) --------
    st.divider()
    st.markdown("#### 📅 Today's SIP entry plan (position- & history-aware)")
    st.caption(
        "Clear per-candidate **buy-today** calls: the laddered entry prices for "
        "today, gated by the **breakout trigger** (imminence) and adjusted for what "
        "you **already hold** and what you **just bought**. Upload your broker "
        "**positions** and **transactions** CSVs (Schwab / Zerodha) — a name you "
        "already added this bar shows *✋ Just bought — skip*, an already-core "
        "position adds *on dips only*, and only names whose trigger is firing get a "
        "full *🟢 BUY today* ladder."
    )
    sc1, sc2 = st.columns(2)
    sip_pos_up = sc1.file_uploader(
        "Positions CSV", type=["csv"], key="sip_pos_upload",
        help="Your current holdings export — used to avoid over-adding to names you "
        "already own a lot of.")
    sip_ord_up = sc2.file_uploader(
        "Transactions CSV", type=["csv", "xlsx", "xls"], key="sip_ord_upload",
        help="Your order history — used to skip names you already bought this bar.")
    sip_cash = st.number_input(
        f"Cash to deploy per BUY name today ({CCY_SYM.get(selected_markets[0], '$') if selected_markets else '$'})",
        min_value=0.0, value=0.0, step=500.0, key="sip_entry_cash",
        help="Optional — sizes the ladder into approximate units per rung.")

    sip_positions, sip_pos_lookup, sip_port_total = [], {}, 0.0
    if sip_pos_up is not None:
        try:
            sip_positions, _ = parse_positions_text(sip_pos_up.getvalue().decode("utf-8", "ignore"))
        except Exception:
            sip_positions = []
        for p in sip_positions:
            val = (p.get("qty") or 0) * (p.get("ltp") or p.get("avg") or 0)
            sip_port_total += val
            sip_pos_lookup[(p["market"], p["symbol"])] = {"qty": p.get("qty"), "val": val}

    sip_orders_sum = {}
    if sip_ord_up is not None:
        try:
            _o, _f = parse_orders_bytes(sip_ord_up.getvalue(), sip_ord_up.name)
            sip_orders_sum = summarize_orders(_o) if _o else {}
        except Exception:
            sip_orders_sum = {}

    bar_days = _bar_window_days(selected_tf)
    now_ts = pd.Timestamp.today().normalize()
    sip_entry_rows = []
    for mkt, s in sorted(results, key=lambda ms: 0.6 * ms[1].breakout_score + 0.4 * ms[1].imminence, reverse=True):
        base_sym = s.ticker.replace(".NS", "").upper()
        price = (s.targets or {}).get("entry")
        pos = sip_pos_lookup.get((mkt, base_sym))
        held_qty = pos["qty"] if pos else None
        held_pct = (pos["val"] / sip_port_total) if (pos and sip_port_total) else None
        osum = sip_orders_sum.get(f"{mkt}:{base_sym}")
        recently_bought = False
        if osum and osum.get("last_side") == "buy" and osum.get("last_date") is not None:
            try:
                recently_bought = (now_ts - pd.Timestamp(osum["last_date"]).normalize()).days <= bar_days
            except Exception:
                recently_bought = False
        plan = todays_sip_entry(
            s, price, held_qty, held_pct, recently_bought,
            cash=sip_cash, sym=CCY_SYM.get(mkt, "$"))
        sip_entry_rows.append({
            "Ticker": tradingview_url(s.ticker),
            "Symbol": s.ticker,
            "Market": mkt,
            "Score": round(0.6 * s.breakout_score + 0.4 * s.imminence, 1),
            "Imminence": s.imminence,
            "Trigger": s.imminence_label,
            "Price": price,
            "Held qty": held_qty,
            "Held %": round(held_pct * 100, 1) if held_pct is not None else None,
            "SIP action": plan["action"],
            "Buy today @": plan["entries"],
            f"Chunk": plan["chunk"],
            "≈ Units": plan["units"],
        })

    sip_entry_df = pd.DataFrame(sip_entry_rows)
    if sip_entry_df.empty:
        st.info("No scanned candidates yet — run a scan from the sidebar to build "
                "today's SIP entry plan.")
    else:
        # Surface the actionable BUY names first.
        _buy = sip_entry_df[sip_entry_df["SIP action"].str.startswith("🟢")]
        _other = sip_entry_df[~sip_entry_df["SIP action"].str.startswith("🟢")]
        sip_entry_df = pd.concat([_buy, _other]).reset_index(drop=True)
        if not sip_cash:
            sip_entry_df = sip_entry_df.drop(columns=["Chunk", "≈ Units"])
        st.dataframe(
            sip_entry_df, use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.LinkColumn("Ticker", display_text=r"symbol=(.+)$"),
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=100, format="%d"),
                "Imminence": st.column_config.ProgressColumn(
                    "Imminence", min_value=0, max_value=100, format="%d"),
                "Held %": st.column_config.NumberColumn(
                    "Held %", help="This holding's share of your uploaded portfolio value",
                    format="%.1f%%"),
                "Buy today @": st.column_config.TextColumn(
                    "Buy today @", help="Laddered limit prices to place today (market + dip "
                    "rungs, or dip-only rungs when adding to a core position)"),
            },
        )
    if not sip_positions and sip_pos_up is not None:
        st.warning("Couldn't parse that positions CSV — expected a Schwab or Zerodha export.")
    st.caption(
        "**How to read it:** 🟢 = place the laddered limit orders shown under *Buy "
        "today @* · ⏳ = base is coiling, wait for the trigger · ✋ = you already added "
        "this bar · ⚠️ = extended, book don't add. Upload the **transactions** CSV to "
        "enable the *just-bought* guard and the **positions** CSV for the *core "
        "holding* / *Held %* awareness."
    )

    # -------- Score an uploaded watchlist (India stocks) --------
    st.divider()
    st.markdown("#### 📄 Score an uploaded watchlist (India)")
    st.caption(
        "Upload a TradingView watchlist export (`.xlsx`/`.csv`) with a **Symbol** or "
        "**TV Code** column. Each symbol is treated as an NSE stock (`.NS`), scored "
        "against the India benchmark using the sidebar timeframe."
    )
    wl = st.file_uploader(
        "Upload watchlist", type=["xlsx", "xls", "csv"], key="wl_upload")
    if wl is not None and st.button("▶️ Score watchlist"):
        pairs = parse_watchlist(wl)
        if not pairs:
            st.error(
                "Couldn't find any symbols. Expected a 'Symbol' or 'TV Code' column.")
        else:
            st.session_state["wl_results"] = run_list_scan(pairs, "India", selected_tf)
            st.session_state["wl_count"] = len(pairs)

    wl_results = st.session_state.get("wl_results")
    if wl_results:
        st.caption(
            f"Scored **{len(wl_results)}** of "
            f"{st.session_state.get('wl_count', len(wl_results))} watchlist symbols "
            "— ranked by breakout readiness.")
        wdf = build_score_df(wl_results, False).sort_values(
            "Score", ascending=False).reset_index(drop=True)
        render_table(wdf)
        st.download_button(
            "⬇️ Download watchlist scores CSV",
            wdf.drop(columns=["_delta"]).to_csv(index=False).encode(),
            file_name="watchlist_scores.csv", mime="text/csv", key="wl_dl",
        )

with tab2:
    st.subheader("Suggested allocation across top consolidating ETFs")
    st.caption(
        f"Distributes **{ccy}{invest_amount:,.0f}** across the top {top_n} breakout "
        f"candidates (breakout ≥ {min_breakout}, and not over-extended), weighted by "
        "each ETF's breakout score."
    )

    # Qualify: strong breakout readiness and not already extended (exit < 50).
    qualified = [
        (mkt, s) for mkt, s in results
        if s.breakout_score >= min_breakout and s.exit_score < 50
    ]
    qualified.sort(key=lambda x: x[1].breakout_score, reverse=True)
    picks = qualified[:top_n]

    if invest_amount <= 0:
        st.info("Enter an amount to invest in the sidebar to see a suggested split.")
    elif not picks:
        st.warning(
            "No ETFs currently qualify (none are both breakout-ready and not "
            "over-extended). Lower the 'Min breakout score' or wait for a better setup."
        )
    else:
        weight_total = sum(s.breakout_score for _, s in picks)
        alloc_rows = []
        for mkt, s in picks:
            weight = s.breakout_score / weight_total
            amount = invest_amount * weight
            price = s.tf["1d"].raw.get("price") if s.tf["1d"].ok else None
            valid_price = price is not None and pd.notna(price) and price > 0
            units = int(amount // price) if valid_price else None
            alloc_rows.append(
                {
                    "Market": mkt,
                    "Ticker": s.ticker,
                    "Sector": s.name,
                    "Breakout": s.breakout_score,
                    "Weight %": round(weight * 100, 1),
                    f"Allocation ({ccy})": round(amount, 2),
                    "Price": round(price, 2) if valid_price else None,
                    "Approx Units": units,
                    "Signal": s.signal,
                }
            )
        alloc_df = pd.DataFrame(alloc_rows)
        st.dataframe(
            alloc_df, use_container_width=True, hide_index=True,
            column_config={
                "Weight %": st.column_config.ProgressColumn(
                    "Weight %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Breakout": st.column_config.ProgressColumn(
                    "Breakout", min_value=0, max_value=100, format="%d"
                ),
            },
        )
        invested = sum(r[f"Allocation ({ccy})"] for r in alloc_rows)
        m1, m2 = st.columns(2)
        m1.metric("Total to invest", f"{ccy}{invest_amount:,.0f}")
        m2.metric("Across", f"{len(picks)} ETFs")
        st.caption(
            "Units are whole-share estimates using the latest daily close; residual "
            "cash from rounding is not reinvested. Weights are proportional to "
            "breakout score — higher-conviction setups get more capital."
        )
        st.download_button(
            "⬇️ Download allocation CSV", alloc_df.to_csv(index=False).encode(),
            file_name="allocation.csv", mime="text/csv",
        )

with tab3:
    st.subheader("Sectors getting over-extended (consider trimming / exiting)")
    exit_df = df[df["Exit"] >= 50].sort_values("Exit", ascending=False).reset_index(drop=True)
    if exit_df.empty:
        st.success("No sectors are over-extended right now. ✅")
    else:
        render_table(exit_df)

with tab_rate:
    st.subheader("⭐ Rate my list — score an uploaded file")
    st.caption(
        "Upload an Excel/CSV file (e.g. a TradingView screener dump or watchlist) "
        "with a **Symbol** or **TV Code** column. Every symbol is scored with the "
        "same screener algorithm (MACD@0, RSI, ADX, relative strength, volume, "
        "consolidation, patterns) across the sidebar timeframe and ranked by "
        "breakout readiness."
    )

    rate_market = st.radio(
        "Treat symbols as", ["India", "US"], horizontal=True, key="rate_market",
        help="India symbols are scored as NSE (.NS) against ^NSEI; US symbols "
             "against SPY.",
    )
    rate_file = st.file_uploader(
        "Upload file to rate", type=["xlsx", "xls", "csv"], key="rate_upload")

    if rate_file is not None and st.button("▶️ Rate these stocks", key="rate_btn"):
        pairs = parse_watchlist(rate_file)
        if rate_market == "US":
            # parse_watchlist forces a .NS suffix; strip it back for US symbols.
            for p in pairs:
                if p["yf"].endswith(".NS"):
                    p["yf"] = p["yf"][:-3]
        if not pairs:
            st.error(
                "Couldn't find any symbols. Expected a 'Symbol' or 'TV Code' column.")
        else:
            st.session_state["rate_results"] = run_list_scan(
                pairs, rate_market, selected_tf)
            st.session_state["rate_count"] = len(pairs)

    rate_results = st.session_state.get("rate_results")
    if rate_results:
        st.caption(
            f"Scored **{len(rate_results)}** of "
            f"{st.session_state.get('rate_count', len(rate_results))} symbols "
            "— ranked by breakout readiness. 🟢 rising fast · 🔴 falling fast · "
            "click a **Ticker** for its TradingView chart.")
        rdf = build_score_df(rate_results, False).sort_values(
            "Breakout", ascending=False).reset_index(drop=True)
        render_table(rdf)
        st.download_button(
            "⬇️ Download rated list CSV",
            rdf.drop(columns=["_delta"]).to_csv(index=False).encode(),
            file_name="rated_list.csv", mime="text/csv", key="rate_dl",
        )
    else:
        st.info("Upload a file and press **Rate these stocks** to see scores.")

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fii_dii():
    return flows.fii_dii()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_large_deals():
    return flows.large_deals()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_delivery():
    return flows.delivery_data()


with tab_flows:
    st.subheader("🏦 Institutional Flows — FII/DII, big-money deals & delivery (NSE)")
    st.caption(
        "See whether **big money is adding or pulling out** — sourced live from "
        "NSE. **FII/DII** shows market-wide net cash flow; **bulk & block deals** "
        "name the *fund house / HNI / broker* on each large trade with an "
        "approximate **₹ value**; **delivery %** flags genuine accumulation vs "
        "intraday churn. 🇮🇳 India only — NSE publishes these; the US equivalent "
        "(quarterly 13F / ETF creation-redemption) isn't a daily feed."
    )
    if st.button("🔄 Refresh institutional data (clears cache)", key="flows_refresh"):
        fetch_fii_dii.clear()
        fetch_large_deals.clear()
        fetch_delivery.clear()
        st.rerun()
    st.caption("Data is cached ~30 min. Figures are provisional and update after "
               "market close on trading days.")

    # -------------------- 1) FII / DII net cash flow --------------------
    st.markdown("### 1️⃣ FII/DII net cash flow (market-wide)")
    try:
        fd = fetch_fii_dii()
        date_str = fd["Date"].iloc[0] if not fd.empty else ""
        cols = st.columns(len(fd))
        for col, (_, r) in zip(cols, fd.iterrows()):
            net = r["Net ₹cr"] or 0
            col.metric(
                f"{r['Category']} net",
                f"₹{net:,.0f} cr",
                delta=("Buying" if net >= 0 else "Selling"),
                delta_color=("normal" if net >= 0 else "inverse"),
            )
        st.caption(f"Provisional cash-market figures for **{date_str}**. "
                   "Positive net = institutions are net buyers that day.")
        st.dataframe(
            fd, use_container_width=True, hide_index=True,
            column_config={
                "Buy ₹cr": st.column_config.NumberColumn("Buy ₹cr", format="%.0f"),
                "Sell ₹cr": st.column_config.NumberColumn("Sell ₹cr", format="%.0f"),
                "Net ₹cr": st.column_config.NumberColumn("Net ₹cr", format="%.0f"),
            },
        )
    except Exception as exc:
        st.warning(f"⚠️ Couldn't load FII/DII data right now: {exc}")

    # -------------------- 2) Bulk & block deals --------------------
    st.markdown("### 2️⃣ Bulk & block deals — who added / trimmed money")
    st.caption(
        "Each row is a **large trade reported to NSE** with the counterparty "
        "named. **Value ₹cr ≈ Qty × avg price.** Bulk = ≥0.5% of shares on the "
        "exchange; block = negotiated large trades. Use *Net by client* to see "
        "which fund house/HNI added the most money today."
    )
    try:
        ld = fetch_large_deals()
        st.caption(f"As on **{ld['as_on']}** · {len(ld['bulk'])} bulk · "
                   f"{len(ld['block'])} block deals.")
        which = st.radio("Deal type", ["Bulk deals", "Block deals"],
                         horizontal=True, key="deal_type")
        deals = ld["bulk"] if which == "Bulk deals" else ld["block"]

        if deals is None or deals.empty:
            st.info(f"No {which.lower()} reported for this session.")
        else:
            side = st.radio("Show", ["All", "Buys only", "Sells only"],
                            horizontal=True, key="deal_side")
            view = deals
            if side == "Buys only":
                view = deals[deals["Side"].str.upper() == "BUY"]
            elif side == "Sells only":
                view = deals[deals["Side"].str.upper() == "SELL"]

            a, b = st.columns(2)
            with a:
                st.markdown("**💰 Net by client (fund/HNI) — ₹cr**")
                nbc = flows.net_by_client(deals, 15)
                st.dataframe(
                    nbc, use_container_width=True, hide_index=True,
                    column_config={"Net ₹cr": st.column_config.NumberColumn(
                        "Net ₹cr", format="%.2f")},
                )
            with b:
                st.markdown("**🏷️ Net by stock — ₹cr**")
                nbs = flows.net_by_symbol(deals, 15)
                st.dataframe(
                    nbs, use_container_width=True, hide_index=True,
                    column_config={"Net ₹cr": st.column_config.NumberColumn(
                        "Net ₹cr", format="%.2f")},
                )

            st.markdown("**📋 All deals**")
            st.dataframe(
                view.sort_values("Value ₹cr", ascending=False),
                use_container_width=True, hide_index=True,
                column_config={
                    "Qty": st.column_config.NumberColumn("Qty", format="%d"),
                    "Avg price": st.column_config.NumberColumn(
                        "Avg price", format="%.2f"),
                    "Value ₹cr": st.column_config.NumberColumn(
                        "Value ₹cr", format="%.2f"),
                },
            )
            st.download_button(
                "⬇️ Download deals CSV",
                view.to_csv(index=False).encode(),
                file_name=f"{which.replace(' ', '_').lower()}.csv",
                mime="text/csv", key="deals_dl",
            )
    except Exception as exc:
        st.warning(f"⚠️ Couldn't load bulk/block deals right now: {exc}")

    # -------------------- 3) Delivery % --------------------
    st.markdown("### 3️⃣ Delivery % — accumulation conviction")
    st.caption(
        "**Delivery %** = shares actually taken into demat vs total traded. "
        "A high and rising delivery % on an up day means buyers are *holding*, "
        "not day-trading — a sign of real accumulation. Enter NSE symbols "
        "(comma-separated) to check."
    )
    try:
        dv_all, dv_date = fetch_delivery()
        default_syms = "RELIANCE, HDFCBANK, NIFTYBEES, TCS, INFY"
        syms_txt = st.text_input(
            "NSE symbols", value=default_syms, key="deliv_syms",
            help="Equity or ETF symbols, e.g. RELIANCE, NIFTYBEES.",
        )
        wanted = [t.strip().upper().replace(".NS", "")
                  for t in syms_txt.split(",") if t.strip()]
        sub = dv_all[dv_all["SYMBOL"].str.upper().isin(wanted)].copy()
        st.caption(f"Bhavcopy dated **{dv_date}** (latest available).")
        if sub.empty:
            st.info("None of those symbols found in the latest bhavcopy "
                    "(check spelling; ETFs like NIFTYBEES work).")
        else:
            sub = sub.rename(columns={
                "SYMBOL": "Symbol", "CLOSE_PRICE": "Close",
                "TTL_TRD_QNTY": "Traded qty", "DELIV_QTY": "Delivered qty",
                "DELIV_PER": "Delivery %"})
            st.dataframe(
                sub.sort_values("Delivery %", ascending=False),
                use_container_width=True, hide_index=True,
                column_config={
                    "Close": st.column_config.NumberColumn("Close", format="%.2f"),
                    "Traded qty": st.column_config.NumberColumn(
                        "Traded qty", format="%d"),
                    "Delivered qty": st.column_config.NumberColumn(
                        "Delivered qty", format="%d"),
                    "Delivery %": st.column_config.ProgressColumn(
                        "Delivery %", min_value=0, max_value=100, format="%.1f%%"),
                },
            )
        with st.expander("🏆 Top delivery % across NSE (liquid names)"):
            liquid = dv_all[dv_all["TTL_TRD_QNTY"] > 100000].copy()
            top = liquid.sort_values("DELIV_PER", ascending=False).head(25).rename(
                columns={"SYMBOL": "Symbol", "CLOSE_PRICE": "Close",
                         "DELIV_PER": "Delivery %"})
            st.dataframe(
                top[["Symbol", "Close", "Delivery %"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "Close": st.column_config.NumberColumn("Close", format="%.2f"),
                    "Delivery %": st.column_config.ProgressColumn(
                        "Delivery %", min_value=0, max_value=100, format="%.1f%%"),
                },
            )
    except Exception as exc:
        st.warning(f"⚠️ Couldn't load delivery data right now: {exc}")

    st.caption(
        "**How to read it:** FII/DII net positive + bulk/block *buys* from fund "
        "houses + high delivery % on the same names = strong institutional "
        "accumulation. Persistent net selling + low delivery = money leaving "
        "(pair with the Distribution score on the Sector Lifecycle tab). "
        "Educational info, not investment advice."
    )

with tab_trigger:
    st.subheader("⚡ Breakout Trigger — readiness vs imminence")
    st.caption(
        "The **Breakout Candidates** tab ranks by **readiness** — how *well-formed* "
        "the coiled base is (MACD@0, tight squeeze, positive RS, above the 200-DMA). "
        "But a textbook base can sit coiled for **weeks**. This tab adds a separate "
        "**Imminence / trigger score** that measures whether the spring is actually "
        "*releasing right now*: price pushing the **breakout line**, **volume "
        "expanding**, **ADX ticking up**, the **squeeze firing** (bandwidth expanding) "
        "and the **MACD histogram** growing off zero. A name is only **actionable now** "
        "when it is **ready AND firing** — a high readiness score with low imminence "
        "means *'good base, no trigger — watchlist'*, not *'buy today'*."
    )
    st.info(
        f"⏱️ Scores use the active timeframe **{tf_choice}** (set in the sidebar). "
        "**Readiness** = setup quality (the old breakout score). **Imminence** = is it "
        "firing now. Rows are sorted by **Imminence** (most imminent first)."
    )

    if not scan_ready:
        st.info("Run a scan from the sidebar to populate the trigger view.")
    else:
        def _verdict(readiness, imm):
            if readiness < 45:
                return "⚪ No setup"
            if readiness >= 55 and imm >= 55:
                return "🟢 BUY — ready + firing"
            if readiness >= 55 and imm >= 35:
                return "🟡 Almost — trigger building"
            if readiness >= 55:
                return "⏳ Watchlist — ready, no trigger"
            if imm >= 55:
                return "🔶 Moving — base still thin"
            return "😴 Coiling — wait"

        def _imm_color(imm):
            if imm >= 75:
                return "#1e7d46"
            if imm >= 55:
                return "#3f7d46"
            if imm >= 35:
                return "#b3952f"
            return "#5a5f64"

        for mkt in selected_markets:
            pool = [s for m, s in results if m == mkt]
            if not pool:
                st.info(f"No {mkt} sectors scanned yet.")
                continue

            rows, firing = [], []
            for s in pool:
                parts = s.imminence_parts or {}
                verdict = _verdict(s.breakout_score, s.imminence)
                rows.append({
                    "_imm": s.imminence,
                    "_color": _imm_color(s.imminence),
                    "Ticker": tradingview_url(s.ticker),
                    "Symbol": s.ticker,
                    "Sector": s.name,
                    "Readiness": s.breakout_score,
                    "Imminence": s.imminence,
                    "Imm 1w ago": s.imminence_1w_ago,
                    "Trigger": s.imminence_label,
                    "% to line": s.dist_to_breakout_pct,
                    "Box pos": s.range_position,
                    "Range": parts.get("range_position"),
                    "Volume": parts.get("volume_thrust"),
                    "ADX↑": parts.get("adx_rising"),
                    "Squeeze": parts.get("squeeze_firing"),
                    "AVWAP": parts.get("avwap"),
                    "Verdict": verdict,
                })
                if s.breakout_score >= 55 and s.imminence >= 55:
                    firing.append(s)

            trig_df = (pd.DataFrame(rows)
                       .sort_values(["_imm", "Readiness"], ascending=[False, False])
                       .reset_index(drop=True))

            st.markdown(f"### {mkt} market")

            if firing:
                firing.sort(key=lambda x: x.imminence, reverse=True)
                picks = " · ".join(
                    f"**{s.ticker}** ({s.name}, readiness {s.breakout_score:.0f}, "
                    f"imminence {s.imminence:.0f})"
                    for s in firing[:8]
                )
                st.success(f"🟢 **Ready AND firing — actionable now:** {picks}")
            else:
                # Surface the ITB-type case: well-formed bases with no trigger.
                waiting = sorted(
                    [s for s in pool if s.breakout_score >= 58 and s.imminence < 35],
                    key=lambda x: x.breakout_score, reverse=True)
                if waiting:
                    wp = " · ".join(
                        f"**{s.ticker}** (readiness {s.breakout_score:.0f}, "
                        f"{s.dist_to_breakout_pct:+.1f}% to line)"
                        for s in waiting[:8] if s.dist_to_breakout_pct is not None)
                    st.warning(
                        "⏳ **Ready but no trigger yet — high-quality bases that are "
                        f"*coiling*, not firing (watchlist):** {wp}"
                    )
                else:
                    st.caption("No breakout firing right now — wait for a trigger.")

            def _trig_color(row):
                c = row["_color"]
                style = (f"background-color: {c}; color: #ffffff; font-weight: 600"
                         if c else "")
                return [style] * len(row)

            styler = trig_df.style.apply(_trig_color, axis=1)
            st.dataframe(
                styler, use_container_width=True, hide_index=True,
                column_config={
                    "_imm": None, "_color": None,
                    "Ticker": st.column_config.LinkColumn(
                        "Ticker", display_text=r"symbol=(.+)$"),
                    "Readiness": st.column_config.ProgressColumn(
                        "Readiness", help="Setup quality — how well-formed the base is "
                        "(the original breakout score)", min_value=0, max_value=100,
                        format="%d"),
                    "Imminence": st.column_config.ProgressColumn(
                        "Imminence", help="Is the breakout firing NOW — price at line, "
                        "volume expanding, ADX rising, squeeze firing, MACD growing",
                        min_value=0, max_value=100, format="%d"),
                    "Imm 1w ago": st.column_config.NumberColumn(
                        "Imm 1w ago", help="Imminence/trigger score as of ~1 week ago "
                        "— compare with today's Imminence to see if the trigger is "
                        "building (rising) or fading (falling)", format="%d"),
                    "% to line": st.column_config.NumberColumn(
                        "% to line", help="Distance from price to the breakout line "
                        "(20-bar high). + = still below it, − = already above",
                        format="%.1f%%"),
                    "Box pos": st.column_config.NumberColumn(
                        "Box pos", help="Where price sits in its range (0 = bottom, "
                        "1 = top of the box)", format="%.2f"),
                    "Range": st.column_config.NumberColumn(
                        "Range", help="Range-position sub-score (near line = high)",
                        format="%d"),
                    "Volume": st.column_config.NumberColumn(
                        "Volume", help="Volume-thrust sub-score (expanding = high)",
                        format="%d"),
                    "ADX↑": st.column_config.NumberColumn(
                        "ADX↑", help="ADX-rising sub-score (trend waking = high)",
                        format="%d"),
                    "Squeeze": st.column_config.NumberColumn(
                        "Squeeze", help="Squeeze-firing sub-score (bandwidth expanding "
                        "= high; still contracting = 0)", format="%d"),
                    "AVWAP": st.column_config.NumberColumn(
                        "AVWAP", help="Anchored-VWAP sub-score — price above the base's "
                        "anchored VWAP (buyers since the base low in control) = high; "
                        "below it = 0", format="%d"),
                },
            )
            st.download_button(
                "⬇️ Download trigger CSV",
                trig_df.drop(columns=["_imm", "_color"]).to_csv(index=False).encode(),
                file_name=f"breakout_trigger_{mkt}.csv", mime="text/csv",
                key=f"trig_dl_{mkt}",
            )
        st.caption(
            "**Why this tab exists:** the top *readiness* name can stay #1 for weeks "
            "while it quietly coils (e.g. price sitting mid-range, ADX falling, squeeze "
            "still tightening). **Imminence** separates *'well-formed base'* from "
            "*'breaking out now'* so the actionable pick is the one actually firing — "
                "not merely the one with the prettiest base. **Imm 1w ago** shows the "
                "trigger score a week back — compare it with today's Imminence to see if "
                "the trigger is **building** (rising) or **fading** (falling). Educational "
                "info, not investment advice."
        )

with tab_life:
    st.subheader("🔄 Sector Lifecycle — basing → breakout → stretched → distribution")
    st.caption(
        "Every scanned sector/ETF placed on the **market lifecycle**: 🟡 Stage 1 "
        "*basing/consolidating* → 🟢 Stage 2 *breakout/markup* → 🟠 Stage 3 "
        "*stretched* → 🔴 Stage 4 *distribution (money leaving)* → ⚫ Stage 5 "
        "*decline*. Distribution is detected from **OBV/price divergence, Chaikin "
        "Money Flow, down-day volume, relative-strength roll-over and high-volume "
        "down days** — the footprint of investors pulling money out even while "
        "price still holds. Rows are colour-coded and sorted Stage 1→5."
    )
    st.info(
        f"⏱️ Scores use the active timeframe **{tf_choice}** (set in the sidebar). "
        "Pair **Blend (1D + 1W)** for steadier stage reads."
    )

    if not scan_ready:
        st.info("Run a scan from the sidebar to populate the lifecycle view.")
    else:
        def _primary_raw(s):
            """Raw indicators from the primary timeframe of the active blend."""
            key = next((k for k in PRIMARY_ORDER
                        if k in selected_tf and s.tf.get(k) and s.tf[k].ok),
                       next((k for k in PRIMARY_ORDER
                             if s.tf.get(k) and s.tf[k].ok), None))
            return s.tf[key].raw if key else {}

        def _fmt_vol(v):
            if not v:
                return "—"
            v = float(v)
            for unit, div in (("M", 1e6), ("K", 1e3)):
                if v >= div:
                    return f"{v/div:.1f}{unit}"
            return f"{v:.0f}"

        VOL_SIGNIF = 1.3   # recent volume ≥1.3× its baseline = "significant"

        for mkt in selected_markets:
            pool = [s for m, s in results if m == mkt]
            if not pool:
                st.info(f"No {mkt} sectors scanned yet.")
                continue

            rows, actionable, dist_warn = [], [], []
            for s in pool:
                raw = _primary_raw(s)
                dist_200 = raw.get("dist_200dma_pct")
                dma_falling = dist_200 is not None and dist_200 < 0
                stg = lifecycle_stage(
                    s.breakout_score, s.exit_score, s.distribution_score, dma_falling)
                vol_ratio = raw.get("vol_ratio")
                signif = bool(vol_ratio and vol_ratio >= VOL_SIGNIF)
                rows.append({
                    "_order": stg["order"],
                    "_break": s.breakout_score,
                    "_color": stg["color"],
                    "Stage": stg["stage"],
                    "Ticker": tradingview_url(s.ticker),
                    "Symbol": s.ticker,
                    "Sector": s.name,
                    "Breakout": s.breakout_score,
                    "Exit": s.exit_score,
                    "Distribution": s.distribution_score,
                    "RS %": raw.get("rel_strength_pct"),
                    "CMF": raw.get("cmf"),
                    "Vol ×": round(vol_ratio, 2) if vol_ratio else None,
                    "Avg vol": _fmt_vol(raw.get("avg_vol")),
                    "Big vol?": "✅" if signif else "",
                    "What to do": stg["action"],
                })
                if stg["order"] in (1, 2) and signif:
                    actionable.append((s, stg, vol_ratio))
                if stg["order"] == 4 and signif:
                    dist_warn.append((s, vol_ratio))

            life_df = (pd.DataFrame(rows)
                       .sort_values(["_order", "_break"], ascending=[True, False])
                       .reset_index(drop=True))

            def _stage_color(row):
                c = row["_color"]
                style = (f"background-color: {c}; color: #ffffff; font-weight: 600"
                         if c else "")
                return [style] * len(row)

            st.markdown(f"### {mkt} market")

            # -------- Suggested ETFs with significant volume --------
            if actionable:
                actionable.sort(key=lambda x: x[0].breakout_score, reverse=True)
                picks = " · ".join(
                    f"**{s.ticker}** ({s.name}, {stg['stage'].split(' · ')[1]}, "
                    f"{vr:.1f}× vol)"
                    for s, stg, vr in actionable[:8]
                )
                st.success(
                    "✅ **Actionable now — basing/breakout sectors backed by "
                    f"significant volume:** {picks}"
                )
            else:
                st.caption(
                    "No basing/breakout sectors with significant volume right now — "
                    "wait for a volume-backed setup."
                )
            if dist_warn:
                dist_warn.sort(key=lambda x: x[0].distribution_score, reverse=True)
                dpicks = " · ".join(
                    f"**{s.ticker}** ({s.name}, {vr:.1f}× vol)" for s, vr in dist_warn[:8])
                st.error(
                    "🔴 **Distribution on heavy volume — investors exiting, "
                    f"reduce/avoid:** {dpicks}"
                )

            styler = life_df.style.apply(_stage_color, axis=1)
            st.dataframe(
                styler, use_container_width=True, hide_index=True,
                column_config={
                    "_order": None, "_break": None, "_color": None,
                    "Ticker": st.column_config.LinkColumn(
                        "Ticker", display_text=r"symbol=(.+)$"),
                    "Breakout": st.column_config.ProgressColumn(
                        "Breakout", min_value=0, max_value=100, format="%d"),
                    "Exit": st.column_config.ProgressColumn(
                        "Exit", min_value=0, max_value=100, format="%d"),
                    "Distribution": st.column_config.ProgressColumn(
                        "Distribution", help="Money-leaving pressure (OBV/CMF/"
                        "down-volume/RS)", min_value=0, max_value=100, format="%d"),
                    "RS %": st.column_config.NumberColumn(
                        "RS %", help="Relative strength vs benchmark", format="%.1f%%"),
                    "CMF": st.column_config.NumberColumn(
                        "CMF", help="Chaikin Money Flow (+ in / − out)", format="%.2f"),
                    "Vol ×": st.column_config.NumberColumn(
                        "Vol ×", help="Recent vs baseline volume", format="%.2f×"),
                    "Big vol?": st.column_config.TextColumn(
                        "Big vol?", help=f"Recent volume ≥ {VOL_SIGNIF}× baseline"),
                },
            )
            st.download_button(
                "⬇️ Download lifecycle CSV",
                life_df.drop(columns=["_order", "_break", "_color"]).to_csv(
                    index=False).encode(),
                file_name=f"sector_lifecycle_{mkt}.csv", mime="text/csv",
                key=f"life_dl_{mkt}",
            )
        st.caption(
            "**Distribution score** rises when smart money is leaving: OBV falling "
            "while price holds, negative/declining Chaikin Money Flow, down-day "
            "volume dominance, relative strength rolling over, and clusters of "
            "high-volume down days. **Big vol? ✅** marks ETFs whose recent volume "
            f"is ≥ {VOL_SIGNIF}× their baseline — the liquid, tradeable names. "
            "Educational info, not investment advice."
        )

with tab4:
    st.subheader("Per-sector breakdown")
    labels = {f"{s.ticker} — {s.name} [{mkt}]": (mkt, s) for mkt, s in results}
    pick = st.selectbox("Select a sector", list(labels.keys()))
    mkt, s = labels[pick]

    st.markdown(f"🔗 [Open {s.ticker} on TradingView]({tradingview_url(s.ticker)})")
    if show_aum:
        fi = cached_fund_info(s.ticker)
        aum_str = format_aum(fi.get("aum"), fi.get("currency"))
        if aum_str:
            st.markdown(f"**Fund size (AUM):** {aum_str}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Breakout Score", f"{s.breakout_score:.0f}/100",
              delta=f"{s.breakout_delta:+.1f} ({s.breakout_trend})")
    c2.metric("Exit Score", f"{s.exit_score:.0f}/100", delta=f"{s.exit_delta:+.1f}")
    if s.targets:
        c3.metric("T2 target", f"{s.targets['target']:.2f}",
                  delta=f"{s.targets['upside_pct']:+.1f}% upside")
    st.markdown(f"**Signal:** {s.signal}")
    st.markdown(f"### 👉 Action: {s.action}")

    if s.patterns and s.patterns.get("all"):
        st.markdown("#### 📐 Chart patterns detected")
        for p in s.patterns["all"]:
            st.write(f"**{p['name']}** — confidence {p['confidence']:.0f}/100 · {p['note']}")
        st.caption("Heuristic pattern flags — confirm visually on the TradingView chart.")

    if s.consolidation:
        cons = s.consolidation
        st.info(
            f"📦 **Consolidating since {cons['start']}** — "
            f"{cons['bars']} {cons['unit']} (~{cons['days']} calendar days) on the "
            f"**{cons['interval']}** interval. Range {cons['range_low']}–{cons['range_high']} "
            f"({cons['range_pct']}% wide)."
        )
        if s.maturity and s.maturity != "—":
            mat_msg = (
                f"⏳ **Base maturity: {s.maturity}** — {s.maturity_note} "
                f"(breakout score ×{s.readiness_factor:.2f}; raw setup quality {s.setup_quality:.0f})."
            )
            if s.readiness_factor < 0.85:
                st.warning(mat_msg)
            else:
                st.caption(mat_msg)

        st.markdown("#### 🎯 Realistic trade plan (measured move)")
        t = s.targets
        tcols = st.columns(6)
        tcols[0].metric("Entry", f"{t['entry']:.2f}")
        tcols[1].metric("Breakout level", f"{t['breakout_level']:.2f}")
        tcols[2].metric(
            f"T1 · scale ~{t.get('scale_out_pct', 40)}%", f"{t['target1']:.2f}",
            delta=f"{t['target1_pct']:+.1f}%")
        tcols[3].metric("T2 · runner", f"{t['target']:.2f}", delta=f"{t['upside_pct']:+.1f}%")
        tcols[4].metric("Stop", f"{t['stop']:.2f}", delta=f"{t['downside_pct']:+.1f}%")
        tcols[5].metric("Risk : Reward", f"1 : {t['risk_reward']}" if t['risk_reward'] else "—")
        st.caption(
            "**T1** = first resistance — book ~40% to lock in a month's worth of gains. "
            "**T2** = 20-bar resistance + consolidation range height (full measured move) "
            "for the runner. **Stop** = below the range low (or 1.5×ATR). "
            "Educational estimate, not advice."
        )

    for key in ("1wk", "1d", "4h", "2h", "1h"):
        if key not in s.tf:
            continue
        tfr = s.tf[key]
        st.markdown(f"#### {key} timeframe")
        if not tfr.ok:
            st.warning(f"Not enough {key} data to score.")
            continue
        cc = st.columns(2)
        with cc[0]:
            st.markdown("**Breakout sub-scores**")
            st.dataframe(
                pd.DataFrame(tfr.breakout_parts.items(), columns=["Component", "Score"]),
                hide_index=True, use_container_width=True,
            )
        with cc[1]:
            st.markdown("**Exit sub-scores**")
            st.dataframe(
                pd.DataFrame(tfr.exit_parts.items(), columns=["Component", "Score"]),
                hide_index=True, use_container_width=True,
            )
        st.markdown("**Raw indicators**")
        st.dataframe(
            pd.DataFrame(tfr.raw.items(), columns=["Indicator", "Value"]),
            hide_index=True, use_container_width=True,
        )


with tab5:
    st.subheader("52-week-high breakout → retest")
    st.caption(
        "Scans the **large-cap stock universes** (`config/stocks_us.json` & "
        "`config/stocks_india.json`) for names that broke out to a fresh **52-week "
        "high** and are now **retesting** the breakout level (old resistance turning "
        "into support) — the classic continuation entry. Click a **Ticker** for the chart."
    )

    if stocks.STOCK_LOAD_ERRORS:
        for mkt_err, msg in stocks.STOCK_LOAD_ERRORS.items():
            st.error(f"⚠️ {mkt_err} stock config not loaded (using defaults): {msg}")

    ctrl = st.columns([1.2, 1.2, 1, 1])
    with ctrl[0]:
        rt_markets = st.multiselect(
            "Markets", ["US", "India"], default=["US", "India"], key="rt_markets"
        )
    with ctrl[1]:
        rt_min_conf = st.slider("Min confidence", 40, 90, 55, 5, key="rt_conf")
    with ctrl[2]:
        n_us = len(stocks.STOCK_MARKETS.get("US", {}))
        n_in = len(stocks.STOCK_MARKETS.get("India", {}))
        st.metric("Universe", f"{n_us + n_in} stocks")
    with ctrl[3]:
        if st.button("🔄 Reload stock lists"):
            stocks.reload()
            st.rerun()

    total_sel = sum(len(stocks.STOCK_MARKETS.get(m, {})) for m in rt_markets)
    st.caption(
        f"⏳ Scans **{total_sel}** stocks live from yfinance — the first run may take "
        "a minute; results are cached for 15 min."
    )

    if st.button("▶️ Run 52W-High Retest scan", type="primary"):
        if not rt_markets:
            st.warning("Select at least one market.")
        else:
            with st.spinner("Scanning for breakout retests…"):
                st.session_state["retest_hits"] = run_retest_scan(rt_markets, rt_min_conf)

    hits = st.session_state.get("retest_hits")
    if hits is None:
        st.info("👆 Click **Run 52W-High Retest scan** to search the stock universe.")
    elif not hits:
        st.warning("No breakout-retest setups found right now. Try lowering the min confidence.")
    else:
        rt_df = pd.DataFrame(hits)
        st.success(f"Found **{len(rt_df)}** breakout-retest setup(s).")
        st.dataframe(
            rt_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.LinkColumn(
                    "Ticker", help="Opens the TradingView chart",
                    display_text=r"symbol=(.+)$",
                ),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%d"
                ),
            },
        )
        st.download_button(
            "⬇️ Download CSV",
            rt_df.to_csv(index=False).encode(),
            file_name="breakout_retest.csv", mime="text/csv",
        )
        st.caption(
            "Heuristic setup: price cleared a prior base top (breakout) to a new 52w "
            "high, then pulled back to within a few % of that level. **Confirm visually** "
            "— a valid retest should *hold* the level on light volume, not slice through it."
        )


with tab6:
    st.subheader("NSE IPOs forming a post-launch CUP (dip → recovery to launch)")
    st.caption(
        "IPO list is fetched live from the **official NSE India** past-issues API. This "
        "scan looks for the **first cup / cup-with-handle in the timeline since listing**: "
        "the price fell meaningfully below its launch (issue) price — a **20-40% dip** — "
        "and has now climbed **back to (±band of) the launch value**. That U-shape signals "
        "the market re-rating a fundamentally-growing company back to its debut price, a "
        "classic base from which the next leg can begin. Click a **Ticker** for the chart."
    )

    ic = st.columns([1.15, 1.35, 1, 1])
    with ic[0]:
        ipo_days = st.slider("Listed within (days)", 90, 500, 365, 15, key="ipo_days")
    with ic[1]:
        ipo_dip = st.slider(
            "Cup depth — dip below launch (%)", 10, 70, (20, 40), 5, key="ipo_dip",
            help="How far the price fell below its launch price at the bottom of the cup.",
        )
    with ic[2]:
        ipo_band = st.slider("Back-to-launch ± (%)", 1, 20, 5, 1, key="ipo_band")
    with ic[3]:
        ipo_board = st.selectbox("Board", ["All", "Mainboard", "SME"], key="ipo_board")

    run_ipo = st.button("▶️ Fetch & scan IPO cups", type="primary")

    if run_ipo:
        try:
            with st.spinner("Fetching recent IPOs from NSE…"):
                ipo_list = cached_recent_ipos(ipo_days)
        except Exception as exc:
            st.error(
                "Couldn't fetch the IPO list from NSE right now "
                f"({type(exc).__name__}). NSE may be rate-limiting — try again shortly."
            )
            ipo_list = None

        if ipo_list is not None:
            if ipo_board != "All":
                ipo_list = [r for r in ipo_list if r.get("board") == ipo_board]
            dip_min, dip_max = float(ipo_dip[0]), float(ipo_dip[1])
            rows = []
            prog = st.progress(0.0, text="Analysing IPO cups…")
            total = max(1, len(ipo_list))
            for i, r in enumerate(ipo_list, start=1):
                prog.progress(i / total, text=f"Analysing {r['symbol']} ({i}/{total})")
                ticker = f"{r['symbol']}.NS"
                closes = ipo_closes_since(ticker, r["listing_date"])
                if closes is None or r["issue_price"] <= 0:
                    continue
                cup = ipos.analyze_cup(
                    closes, r["issue_price"],
                    dip_min=dip_min, dip_max=dip_max, recover_band=float(ipo_band),
                )
                if cup is None:
                    continue
                drop = -cup["depth_pct"]  # positive dip magnitude
                # Cup qualifies: it dipped into the chosen depth band AND has climbed
                # back to within ±band of the launch price (the rim).
                if not (dip_min <= drop <= dip_max):
                    continue
                if abs(cup["now_vs_launch"]) > ipo_band:
                    continue
                rows.append(
                    {
                        "Ticker": tradingview_url(ticker),
                        "Symbol": r["symbol"],
                        "Company": r["company"],
                        "Shape": cup["shape"],
                        "Board": r.get("board", ""),
                        "Listed": str(r["listing_date"]),
                        "Days": r["days_since"],
                        "Issue ₹": r["issue_price"],
                        "Low ₹": cup["trough_price"],
                        "Dip %": cup["depth_pct"],
                        "Now ₹": cup["current"],
                        "% vs launch": cup["now_vs_launch"],
                        "Rebound %": cup["rebound_pct"],
                        "Confidence": cup["confidence"],
                    }
                )
            prog.empty()
            st.session_state["ipo_rows"] = rows
            st.session_state["ipo_scanned_total"] = len(ipo_list)
            st.session_state["ipo_dip_used"] = (dip_min, dip_max)
            st.session_state["ipo_band_used"] = ipo_band

    ipo_rows = st.session_state.get("ipo_rows")
    if ipo_rows is None:
        st.info("👆 Click **Fetch & scan IPO cups** to pull the latest NSE IPO list.")
    elif not ipo_rows:
        dmn, dmx = st.session_state.get("ipo_dip_used", (20, 40))
        st.warning(
            f"Scanned {st.session_state.get('ipo_scanned_total', 0)} IPOs — none have "
            f"formed a {int(dmn)}-{int(dmx)}% cup that's back within "
            f"±{st.session_state.get('ipo_band_used', ipo_band)}% of launch. "
            "Widen the depth range, the band, or the listing window."
        )
    else:
        ipo_df = (
            pd.DataFrame(ipo_rows)
            .sort_values("Confidence", ascending=False)
            .reset_index(drop=True)
        )
        dmn, dmx = st.session_state.get("ipo_dip_used", (20, 40))
        st.success(
            f"**{len(ipo_df)}** of {st.session_state.get('ipo_scanned_total', 0)} recent "
            f"NSE IPOs dipped {int(dmn)}-{int(dmx)}% and have recovered back to their "
            "launch price — a post-IPO cup."
        )
        st.dataframe(
            ipo_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.LinkColumn(
                    "Ticker", help="Opens the TradingView chart",
                    display_text=r"symbol=(.+)$",
                ),
                "Dip %": st.column_config.NumberColumn("Dip %", format="%.1f%%"),
                "% vs launch": st.column_config.NumberColumn("% vs launch", format="%.1f%%"),
                "Rebound %": st.column_config.NumberColumn("Rebound %", format="%.1f%%"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%d"
                ),
            },
        )
        st.download_button(
            "⬇️ Download CSV",
            ipo_df.to_csv(index=False).encode(),
            file_name="nse_ipo_cups.csv", mime="text/csv",
        )
        st.caption(
            "**Cup** = fell into the chosen dip range then recovered to the launch rim; "
            "**Cup w/ Handle** = a small pullback is now forming just under the rim. "
            "Launch price = NSE **issue price**; **Low ₹** is the trough since listing. "
            "Some very recent IPOs lack yfinance history and are skipped. "
            "Educational info, not investment advice."
        )

# ---------------------- Near-Zero MACD Coil (tab7) ---------------------
# Model basket: ETFs whose MACD *and* signal line have reset back to the zero
# line and are hovering there, while RSI cools into the 40-55 zone and the
# Bollinger bands squeeze — the classic "coil before the bull leg" fingerprint.
COIL_MODELS = ["VB", "XLRE", "XLI", "VGT", "XLU", "VUG"]


def _coil_gauss(x, center, width):
    return math.exp(-((x - center) ** 2) / (2 * width ** 2))


def _coil_primary(s):
    """Raw indicators + breakout parts from the first usable primary timeframe."""
    for k in ("1d", "1wk", "4h"):
        r = s.tf.get(k)
        if r is not None and r.ok:
            return r.raw, r.breakout_parts, k
    return {}, {}, None


def _rsi_band(rsi, lo=40.0, hi=55.0, taper=8.0):
    """100 inside the 40-55 band, tapering to 0 within ``taper`` points either side."""
    if lo <= rsi <= hi:
        return 100.0
    if rsi < lo:
        return max(0.0, 100.0 * (1 - (lo - rsi) / taper))
    return max(0.0, 100.0 * (1 - (rsi - hi) / taper))


def coil_match(s):
    """0-100 score built on exactly three criteria:
      1. MACD *and* its signal line hugging the zero line (momentum reset).
      2. RSI in the 40-55 band (cooling off a low, room to run).
      3. Bollinger squeeze (tight range).
    """
    raw, _parts, _tf = _coil_primary(s)
    if not raw:
        return None
    macd_pct = raw.get("macd_pct", 9.0)
    signal_pct = raw.get("signal_pct", 9.0)
    rsi = raw.get("rsi", 0.0)
    bw_rank = raw.get("bandwidth_pct_rank", 100.0)

    # 1) MACD + signal both hugging zero (average of the two zero-line fits).
    macd_zero_c = (_coil_gauss(macd_pct, 0.0, 0.7)
                   + _coil_gauss(signal_pct, 0.0, 0.8)) / 2 * 100
    # 2) RSI parked in the 40-55 band.
    rsi_c = _rsi_band(rsi)
    # 3) Bollinger squeeze — tighter band vs history = higher.
    squeeze_c = max(0.0, 100.0 - bw_rank)

    score = 0.40 * macd_zero_c + 0.30 * rsi_c + 0.30 * squeeze_c
    return round(score, 1)


def coil_stage(s):
    """Where price sits vs the breakout trigger: coiling / breaking out / gone."""
    t = s.targets or {}
    price = t.get("entry")
    lvl = t.get("breakout_level")
    if not price or not lvl:
        return "—", None
    gap = (lvl / price - 1) * 100          # +ve => price still below the trigger
    if gap > 2:
        return "🔩 Coiling", gap
    if gap >= -2:
        return "🚀 Breakout starting", gap
    return "📈 Broke out", gap


with tab7:
    st.subheader("Near-zero MACD coil — models: VB · XLRE · XLI · VGT · XLU · VUG")
    st.caption(
        "Screens the scanned ETFs on **three criteria** (same setup as the model "
        "basket): **① MACD and its signal line hugging the zero line** (momentum "
        "reset, breakout loading), **② RSI in the 40-55 band** (cooling off a low, "
        "room to run) and **③ a Bollinger squeeze** (tight range). The **Entry** is "
        "the breakout trigger (top of the range); **Entry %** is how far price must "
        "still travel to it. **T1/T2** are the measured-move and extension exit targets."
    )

    min_match = st.slider("Minimum coil match %", 0, 100, 55, 5, key="coil_min")

    coil_rows = []
    for mkt, s in results:
        m = coil_match(s)
        if m is None:
            continue
        raw, _parts, _tf = _coil_primary(s)
        stage, gap = coil_stage(s)
        t = s.targets or {}
        price = t.get("entry")
        entry = t.get("breakout_level")
        target1 = t.get("target")
        stop = t.get("stop")

        # T2 = 1.618 extension of the measured move beyond the trigger.
        target2 = None
        if entry is not None and target1 is not None:
            target2 = round(entry + 1.618 * (target1 - entry), 2)

        def _pct(to, frm):
            return round((to / frm - 1) * 100, 1) if (to and frm) else None

        coil_rows.append(
            {
                "Market": mkt,
                "Ticker": tradingview_url(s.ticker),
                "Symbol": s.ticker,
                "Sector": s.name,
                "Model?": "★" if s.ticker in COIL_MODELS else "",
                "Match": m,
                "MACD%": raw.get("macd_pct"),
                "Signal%": raw.get("signal_pct"),
                "Near-0 bars": raw.get("near_zero_bars"),
                "RSI": raw.get("rsi"),
                "BB squeeze": (
                    round(100 - raw.get("bandwidth_pct_rank"), 0)
                    if raw.get("bandwidth_pct_rank") is not None else None
                ),
                "Stage": stage,
                "Price": price,
                "Entry (trigger)": entry,
                "Entry %": (round(gap, 1) if gap is not None else None),
                "T1": target1,
                "T1 %": _pct(target1, price),
                "T2": target2,
                "T2 %": _pct(target2, price),
                "Stop": stop,
                "Stop %": _pct(stop, price),
                "R:R": t.get("risk_reward"),
                "Signal": s.signal,
            }
        )

    if not coil_rows:
        st.info("No scanned ETFs yet — run a scan from the sidebar first.")
    else:
        coil_df = pd.DataFrame(coil_rows)
        shown = coil_df[coil_df["Match"] >= min_match].sort_values(
            "Match", ascending=False
        ).reset_index(drop=True)

        # Model-basket fingerprint reference.
        models_in = coil_df[coil_df["Model?"] == "★"]
        if not models_in.empty:
            avg = models_in["Match"].mean()
            st.caption(
                f"📌 Model basket in this scan: "
                + " · ".join(
                    f"{r.Symbol} {r.Match:.0f}" for r in models_in.itertuples()
                )
                + f"  (avg match {avg:.0f})"
            )
        else:
            st.caption(
                "📌 Model ETFs (VB/XLRE/XLI/VGT/XLU/VUG) aren't in the current scan — "
                "enable the US market and include them to see their reference scores."
            )

        if shown.empty:
            st.warning(
                f"No ETFs currently match the coil model at ≥ {min_match}%. "
                "Lower the threshold or wait for setups to tighten."
            )
        else:
            st.dataframe(
                shown,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.LinkColumn(
                        "Ticker", help="Opens the TradingView chart",
                        display_text=r"symbol=(.+)$",
                    ),
                    "Match": st.column_config.ProgressColumn(
                        "Match", help="How closely it matches the coil model",
                        min_value=0, max_value=100, format="%d",
                    ),
                    "MACD%": st.column_config.NumberColumn("MACD%", format="%.2f"),
                    "Signal%": st.column_config.NumberColumn("Signal%", format="%.2f"),
                    "RSI": st.column_config.NumberColumn("RSI", format="%.0f"),
                    "BB squeeze": st.column_config.NumberColumn(
                        "BB squeeze", help="100 = tightest band (max squeeze)",
                        format="%d",
                    ),
                    "Entry %": st.column_config.NumberColumn(
                        "Entry %", help="% price must rise to hit the breakout trigger",
                        format="%.1f%%",
                    ),
                    "T1 %": st.column_config.NumberColumn("T1 %", format="%.1f%%"),
                    "T2 %": st.column_config.NumberColumn("T2 %", format="%.1f%%"),
                    "Stop %": st.column_config.NumberColumn("Stop %", format="%.1f%%"),
                },
            )
            st.download_button(
                "⬇️ Download CSV",
                shown.to_csv(index=False).encode(),
                file_name="near_zero_macd_coil.csv", mime="text/csv",
            )
            st.caption(
                "**Entry (trigger)** = top of the consolidation; buy the break above it. "
                "**T1** = measured-move target, **T2** = 1.618 extension, **Stop** = "
                "range low / 1.5·ATR. Educational info, not investment advice."
            )

if SHOW_SIP_TAB:
  with tab_sip:
    st.subheader("📅 SIP & Exit Plan — weekly-primary rotation (~5%/month)")
    st.caption(
        "Built for a steady ~5%/month rotation: **accumulate (SIP)** into the top-3 "
        "coiling leaders of the selected market, and **scale out** of holdings as they "
        "get over-extended. Best paired with the **Blend (1D + 1W)** timeframe "
        "(weekly decides *what* to own, daily times the buy)."
    )

    # ============ SECTION A — Staggered daily deployment (SIP) ============
    st.markdown("### 1️⃣ Deploy your cash in daily chunks — top 3 ETFs")
    st.caption(
        "Enter your **total cash**; instead of a lump sum, it's split into **equal "
        "daily chunks** across the top-3 leaders, and each chunk is placed as a "
        "**dip-buying ladder** (part at market, more queued lower). If the price "
        "falls, the lower rungs fill and your **average cost drops** — so a modest "
        "recovery puts you back in profit. Every scanned ETF is listed below with "
        "the same row colours as the Breakout tab (green = momentum rising, red = "
        "falling), but **only the top-3 leaders receive a cash allocation**."
    )
    for mkt in selected_markets:
        sym = CCY_SYM.get(mkt, "")
        # Qualify: breakout-ready and not already extended; else best available.
        pool = [s for m, s in results if m == mkt]
        qualified = [s for s in pool if s.breakout_score >= 45 and s.exit_score < 50]
        qualified.sort(key=lambda s: s.breakout_score, reverse=True)
        top3 = (qualified or sorted(pool, key=lambda s: s.breakout_score, reverse=True))[:3]

        st.markdown(f"**{mkt} market**")
        if not top3:
            st.info(f"No {mkt} ETFs scanned yet — run a scan from the sidebar.")
            continue

        c1, c2, c3 = st.columns(3)
        total_cash = c1.number_input(
            f"Total cash to deploy ({sym})", min_value=0.0, value=0.0, step=1000.0,
            key=f"sip_total_{mkt}",
            help="Your entire amount for this market — deployed gradually, not at once.",
        )
        deploy_days = c2.number_input(
            "Spread over N trading days", min_value=1, max_value=120, value=20, step=1,
            key=f"sip_days_{mkt}",
            help="e.g. 20 ≈ one month of trading days. More days = smaller, safer chunks.",
        )
        style = c3.radio(
            "Deployment style", ["Dip ladder (recommended)", "Equal daily"],
            key=f"sip_style_{mkt}",
            help="Dip ladder queues more cash at lower prices to average down.",
        )
        use_ladder = style.startswith("Dip")
        # Blended fill factor if every ladder rung fills (≈0.98 => ~2% below market).
        ladder_factor = sum(w * (1 + off) for off, w in SIP_LADDER)

        daily_total = round(total_cash / deploy_days, 2) if total_cash else 0.0
        wsum = sum(s.breakout_score for s in top3) or 1.0
        top3_syms = {s.ticker for s in top3}

        # Show ALL scanned stocks/ETFs (sorted by breakout score), but only the
        # top-3 leaders receive a cash allocation.
        display_pool = sorted(pool, key=lambda s: s.breakout_score, reverse=True)
        sip_rows, ladder_rows = [], []
        for s in display_pool:
            t = s.targets or {}
            price = t.get("entry")
            in_top3 = s.ticker in top3_syms
            if in_top3:
                pct = round(s.breakout_score / wsum * 100, 1)
                alloc = round(total_cash * pct / 100, 2) if total_cash else None
                daily_chunk = round(daily_total * pct / 100, 2) if daily_total else None
                units_day = int(daily_chunk // price) if (daily_chunk and price and price > 0) else None
                avg_ladder = round(price * ladder_factor, 2) if (price and use_ladder) else None
            else:
                pct = alloc = daily_chunk = units_day = avg_ladder = None
            sip_rows.append({
                "Ticker": tradingview_url(s.ticker),
                "Symbol": s.ticker,
                "Sector": s.name,
                "Breakout": s.breakout_score,
                "1W ago": s.breakout_1w_ago,
                "2W ago": s.breakout_2w_ago,
                "Signal": s.signal,
                "SIP %": pct,
                f"Total {sym}": alloc,
                f"Daily {sym}": daily_chunk,
                "≈ Units/day": units_day,
                "Buy @ (now)": price,
                "Avg if laddered": avg_ladder,
                "T1 (scale ~40%)": t.get("target1"),
                "T2 (runner)": t.get("target"),
                "Stop": t.get("stop"),
                "_delta": s.breakout_delta,
            })
            if in_top3 and use_ladder and daily_chunk and price:
                for off, w in SIP_LADDER:
                    rung_amt = round(daily_chunk * w, 2)
                    rung_px = round(price * (1 + off), 2)
                    ladder_rows.append({
                        "Symbol": s.ticker,
                        "Rung": "Market" if off == 0 else f"{off*100:.0f}%",
                        "Limit price": rung_px,
                        f"Amount {sym}": rung_amt,
                        "≈ Units": int(rung_amt // rung_px) if rung_px > 0 else None,
                    })

        sip_df = pd.DataFrame(sip_rows)
        if not use_ladder:
            sip_df = sip_df.drop(columns=["Avg if laddered"])
        styler = sip_df.style.apply(_style_by_trend, axis=1)
        st.dataframe(
            styler, use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.LinkColumn(
                    "Ticker", display_text=r"symbol=(.+)$"),
                "_delta": None,  # hide helper column used for row colouring
                "Breakout": st.column_config.ProgressColumn(
                    "Breakout", min_value=0, max_value=100, format="%d"),
                "1W ago": st.column_config.NumberColumn(
                    "1W ago", help="Breakout score ~1 week ago", format="%d"),
                "2W ago": st.column_config.NumberColumn(
                    "2W ago", help="Breakout score ~2 weeks ago", format="%d"),
                "SIP %": st.column_config.NumberColumn("SIP %", format="%.1f%%"),
            },
        )

        if total_cash:
            st.caption(
                f"Plan: deploy **{sym}{daily_total:,.0f}/day for {int(deploy_days)} "
                f"trading days** (≈ {int(deploy_days)//5 or 1} weeks) — total "
                f"**{sym}{total_cash:,.0f}**."
                + (f" With the dip ladder, if all rungs fill your average cost is "
                   f"≈ **{(1-ladder_factor)*100:.1f}% below** today's price."
                   if use_ladder else "")
            )
            if use_ladder and ladder_rows:
                with st.expander("📉 Today's dip-ladder limit orders (per ETF)"):
                    st.caption(
                        "Place these as **limit / GTC orders** for today's chunk. The "
                        "**Market** rung buys now; lower rungs fill only if price dips, "
                        "pulling your average cost down. Any rung that doesn't fill "
                        "today simply rolls into tomorrow's chunk."
                    )
                    st.dataframe(
                        pd.DataFrame(ladder_rows), use_container_width=True,
                        hide_index=True,
                        column_config={
                            f"Amount {sym}": st.column_config.NumberColumn(
                                f"Amount {sym}", format="%.0f"),
                        },
                    )
    st.caption(
        "SIP % is weighted by each ETF's breakout score. **Breakout / 1W ago / 2W ago** "
        "show the score trend — rising left-to-right means demand is building. "
        "**Buy @ (now)** = today's close; **Avg if laddered** = your cost basis if the "
        "whole dip ladder fills (lower = safer). Skip a day if a name is already "
        "extended (see the Exit plan below or the Exit Watch tab)."
    )

# ---------------------------- Alerts (tab_alerts) ----------------------------
# TradingView has no public API to create alerts programmatically, so we hand
# the user everything to make them in a couple of clicks: a levels table + a
# generated Pine study whose alertcondition()s become ready-made alerts.
def tv_symbol(ticker: str) -> str:
    """TradingView symbol notation, e.g. VGT or NSE:BANKBEES."""
    t = ticker.upper()
    return f"NSE:{t[:-3]}" if t.endswith(".NS") else t


def build_pine_alert_script(ticker, buy, target, stop, want_buy, want_target, want_stop):
    """Pine v5 study: plots the chosen levels and defines an alertcondition for
    each, so TradingView 'Create Alert' can fire on a Buy/Target/Stop cross."""
    tv = tv_symbol(ticker)
    lines = [
        "//@version=5",
        f'indicator("Breakout Alerts — {tv}", overlay=true)',
    ]
    conds = []
    if want_buy and buy is not None:
        lines.append(f'buy = input.float({buy}, "Buy trigger")')
        lines.append('plot(buy, "Buy trigger", color=color.green, linewidth=2)')
        conds.append(('ta.crossover(close, buy)', "Buy trigger hit",
                      f"{tv} crossed BUY trigger"))
    if want_target and target is not None:
        lines.append(f'target = input.float({target}, "Target")')
        lines.append('plot(target, "Target", color=color.blue, linewidth=2)')
        conds.append(('ta.crossover(close, target)', "Target reached",
                      f"{tv} reached TARGET"))
    if want_stop and stop is not None:
        lines.append(f'stop = input.float({stop}, "Stop")')
        lines.append('plot(stop, "Stop", color=color.red, linewidth=2)')
        conds.append(('ta.crossunder(close, stop)', "Stop hit",
                      f"{tv} broke STOP"))
    for expr, title, msg in conds:
        lines.append(f'alertcondition({expr}, "{title}", "{msg}")')
    return "\n".join(lines)


with tab_alerts:
    st.subheader("🔔 Alerts — buy-trigger & target levels for TradingView")
    st.info(
        "⚠️ TradingView has **no public API** to add alerts automatically, so this "
        "app can't push them into your account directly. Use the **built-in email "
        "alerts** below (recommended), or the Pine script for TradingView."
    )

    # ---------------- Built-in email alert settings ----------------
    with st.expander("📧 Email alert settings (one-time setup)", expanded=False):
        cfg = alertmod.load_config()
        st.caption(
            "For Gmail, use an **App Password** (Google Account → Security → "
            "2-Step Verification → App passwords), not your normal password. "
            "Settings are saved locally to `alert_config.json` (git-ignored)."
        )
        e1, e2, e3 = st.columns(3)
        host = e1.text_input("SMTP host", value=str(cfg.get("host", "smtp.gmail.com")))
        port = e2.number_input("Port", value=int(cfg.get("port", 465)), step=1)
        use_ssl = e3.checkbox("Use SSL (465)", value=bool(cfg.get("use_ssl", True)))
        username = st.text_input("SMTP username (your email)",
                                 value=str(cfg.get("username", "")))
        password = st.text_input("SMTP password / app password", type="password",
                                  value=str(cfg.get("password", "")))
        sender = st.text_input("From (optional, defaults to username)",
                               value=str(cfg.get("sender", "")))
        to_addr = st.text_input("Send alerts to", value=str(cfg.get("to", "")))
        s1, s2 = st.columns(2)
        if s1.button("💾 Save email settings"):
            alertmod.save_config({
                "host": host, "port": int(port), "use_ssl": bool(use_ssl),
                "username": username, "password": password,
                "sender": sender, "to": to_addr,
            })
            st.success("Saved. Send a test email to confirm it works.")
        if s2.button("✉️ Send test email"):
            ok, msg = alertmod.send_test_email({
                "host": host, "port": int(port), "use_ssl": bool(use_ssl),
                "username": username, "password": password,
                "sender": sender, "to": to_addr,
            })
            (st.success if ok else st.error)(msg)

    if not results:
        st.info("Run a scan from the sidebar first to get alert levels.")
    else:
        by_sym = {s.ticker: (mkt, s) for mkt, s in results}
        all_syms = list(by_sym.keys())
        default = [t for t in sorted(
            all_syms, key=lambda x: by_sym[x][1].breakout_score, reverse=True)][:3]

        picks = st.multiselect(
            "ETFs to build alerts for", all_syms, default=default, key="alert_syms")

        c1, c2, c3 = st.columns(3)
        want_buy = c1.checkbox("Buy trigger (breakout)", value=True, key="al_buy")
        want_target = c2.checkbox("Target reached", value=True, key="al_tgt")
        want_stop = c3.checkbox("Stop hit", value=False, key="al_stop")

        if not picks:
            st.info("Pick at least one ETF above.")
        else:
            alert_rows = []
            for tk in picks:
                mkt, s = by_sym[tk]
                t = s.targets or {}
                sym = CCY_SYM.get(mkt, "")
                row = {
                    "Ticker": tradingview_url(tk),
                    "Symbol": tk,
                    "TV symbol": tv_symbol(tk),
                    "Price": t.get("entry"),
                }
                if want_buy:
                    lvl = t.get("breakout_level")
                    row["🟢 Buy @"] = lvl
                    row["Buy Δ%"] = (
                        round((lvl / t["entry"] - 1) * 100, 1)
                        if (lvl and t.get("entry")) else None)
                if want_target:
                    tg = t.get("target")
                    row["🎯 Target @"] = tg
                    row["Target Δ%"] = (
                        round((tg / t["entry"] - 1) * 100, 1)
                        if (tg and t.get("entry")) else None)
                if want_stop:
                    row["🛑 Stop @"] = t.get("stop")
                alert_rows.append(row)

            adf = pd.DataFrame(alert_rows)
            st.dataframe(
                adf, use_container_width=True, hide_index=True,
                column_config={
                    "Ticker": st.column_config.LinkColumn(
                        "Ticker", display_text=r"symbol=(.+)$"),
                    "Buy Δ%": st.column_config.NumberColumn("Buy Δ%", format="%.1f%%"),
                    "Target Δ%": st.column_config.NumberColumn(
                        "Target Δ%", format="%.1f%%"),
                },
            )
            st.download_button(
                "⬇️ Download alert levels CSV",
                adf.to_csv(index=False).encode(),
                file_name="alert_levels.csv", mime="text/csv",
            )
            st.caption(
                "**Buy Δ% / Target Δ%** = distance from today's price to that level. "
                "**Manual route:** open the chart, click **Create Alert**, set "
                "*Condition → Price → Crossing Up* and type the level."
            )

            # -------- Create built-in email alerts from these levels --------
            st.markdown("#### 📧 Create email alerts from the levels above")
            if st.button("➕ Create email alerts for selected ETFs"):
                new = []
                for tk in picks:
                    mkt, s = by_sym[tk]
                    t = s.targets or {}
                    tv = tv_symbol(tk)
                    if want_buy and t.get("breakout_level"):
                        new.append(alertmod.make_alert(
                            tk, "buy", t["breakout_level"], tv=tv,
                            note=f"{tv} breakout BUY trigger"))
                    if want_target and t.get("target"):
                        new.append(alertmod.make_alert(
                            tk, "target", t["target"], tv=tv,
                            note=f"{tv} TARGET reached"))
                    if want_stop and t.get("stop"):
                        new.append(alertmod.make_alert(
                            tk, "stop", t["stop"], tv=tv, note=f"{tv} STOP hit"))
                added = alertmod.add_alerts(new)
                if added:
                    st.success(
                        f"Created {len(added)} alert(s). Manage them in "
                        "**🔔 Active alerts** below and run the watcher to get emails."
                    )
                else:
                    st.info("No new alerts (they may already exist).")

            st.markdown("#### 📜 Pine scripts (copy → Pine Editor → Add to chart)")
            for tk in picks:
                _, s = by_sym[tk]
                t = s.targets or {}
                code = build_pine_alert_script(
                    tk, t.get("breakout_level"), t.get("target"), t.get("stop"),
                    want_buy, want_target, want_stop)
                with st.expander(f"{tv_symbol(tk)} — Pine alert script"):
                    st.code(code, language="python")
                    st.caption(
                        f"Open the **{tv_symbol(tk)}** chart → Pine Editor → paste → "
                        "**Add to chart** → **Create Alert** → Condition = this "
                        "indicator → choose the trigger. Levels are pre-filled as "
                        "inputs you can fine-tune."
                    )

    # ---------------- Active alerts management (always shown) ----------------
    st.divider()
    st.markdown("### 🔔 Active email alerts")
    all_alerts = alertmod.load_alerts()
    if not all_alerts:
        st.caption("No alerts yet — create some from the levels above.")
    else:
        adf = pd.DataFrame([
            {
                "id": a["id"],
                "Symbol": a["tv"],
                "Type": a["kind"],
                "Fires when": f"price {'▲ ≥' if a['direction'] == 'above' else '▼ ≤'} {a['level']}",
                "Level": a["level"],
                "Status": a["status"],
                "Triggered @": a.get("triggered_price"),
                "When": a.get("triggered_at"),
            }
            for a in all_alerts
        ])
        st.dataframe(
            adf.drop(columns=["id"]), use_container_width=True, hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn("Status"),
            },
        )
        m1, m2, m3 = st.columns(3)
        if m1.button("🔍 Check alerts now"):
            fired = alertmod.check_alerts(price_lookup=cached_last_price)
            if fired:
                st.success(
                    "Fired: " + ", ".join(
                        f"{a['tv']} {a['kind']} @ {a['triggered_price']}" for a in fired)
                    + ("" if alertmod.config_ready() else
                       " (email not configured — set it up above to receive mail)")
                )
            else:
                st.info("Checked — nothing triggered yet.")
        to_del = m2.multiselect(
            "Delete alerts", options=adf["id"].tolist(),
            format_func=lambda i: next(
                (f"{a['tv']} {a['kind']} @ {a['level']}" for a in all_alerts
                 if a["id"] == i), i))
        if m2.button("🗑️ Delete selected") and to_del:
            alertmod.remove_alerts(to_del)
            st.rerun()
        if m3.button("🧹 Clear triggered"):
            alertmod.clear_triggered()
            st.rerun()

    st.caption(
        "**Get emails in the background** — run the watcher so alerts fire even with "
        "the browser closed:\n\n"
        "```\ncd sector-breakout-scanner\n"
        "python alert_watcher.py --loop 900   # checks every 15 min\n```\n"
        "Or schedule `python alert_watcher.py` (single check) via Windows Task "
        "Scheduler. Educational info, not investment advice."
    )

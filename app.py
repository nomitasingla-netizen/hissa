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
import core.stocks as stocks
from core.etfs import MARKETS, LOAD_ERRORS
from core.scoring import score_sector, breakout_snapshot
from core.patterns import detect_breakout_retest
import core.ipos as ipos
import core.alerts as alertmod

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
     "1D only", "4h only", "1W only"],
    index=0,
    help="Which timeframe(s) the breakout/exit scores are based on. "
         "1W (weekly) captures the higher-timeframe trend.",
)
TF_MAP = {
    "Blend (4h + 1D)": ("4h", "1d"),
    "Blend (4h + 1D + 1W)": ("4h", "1d", "1wk"),
    "Blend (1D + 1W)": ("1d", "1wk"),
    "1D only": ("1d",),
    "4h only": ("4h",),
    "1W only": ("1wk",),
}
selected_tf = TF_MAP[tf_choice]

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

if run or "results" not in st.session_state:
    if run:
        st.session_state["results"] = run_scan(selected_markets, custom_lists, selected_tf, as_of_date)
    elif "results" not in st.session_state:
        st.info("👈 Configure your markets/ETFs and click **Scan / Refresh** to begin.")
        st.stop()

results = st.session_state.get("results", [])
if not results:
    st.stop()


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
                "Breakout": s.breakout_score,
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
        "Breakout": st.column_config.ProgressColumn(
            "Breakout", min_value=0, max_value=100, format="%d"
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

tab_sip, tab1, tab2, tab3, tab_rate, tab4, tab5, tab6, tab7, tab_alerts = st.tabs(
    ["📅 SIP & Exit Plan", "🚀 Breakout Candidates", "💰 Allocation", "🔴 Exit Watch",
     "⭐ Rate My List", "🔎 Details", "🔁 52W-High Retest", "🆕 NSE IPOs near launch",
     "🎯 Near-Zero MACD Coil", "🔔 Alerts"]
)

with tab1:
    st.subheader("Ranked by breakout readiness")
    st.caption(
        "🟢 green = score rising fast (demand building / tightening) · "
        "🔴 red = score falling fast · **Action** column suggests staged SIP / "
        "profit-booking · click a **Ticker** to open its TradingView chart."
    )
    ranked = df.sort_values("Breakout", ascending=False).reset_index(drop=True)
    render_table(ranked)
    st.download_button(
        "⬇️ Download CSV",
        ranked.drop(columns=["_delta"]).to_csv(index=False).encode(),
        file_name="breakout_scan.csv", mime="text/csv",
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
            "Breakout", ascending=False).reset_index(drop=True)
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

    for key in ("1wk", "1d", "4h"):
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

# ------------------- SIP & Exit Plan (tab8) --------------------------
# A weekly-primary rotation helper aimed at ~5%/month: accumulate (SIP) into
# the top-3 coiling leaders of the selected market, and scale out of held
# positions (uploaded CSV) as they get over-extended.
CCY_SYM = {"US": "$", "India": "₹"}

# Dip-buying ladder: (price offset from today's close, share of the daily chunk).
# More capital is queued at lower prices so a falling ETF lowers your average
# cost. Shares sum to 1.0; the blended fill price is ~2% below market.
SIP_LADDER = [(0.00, 0.40), (-0.02, 0.30), (-0.04, 0.20), (-0.06, 0.10)]


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
        "rsi": s.tf["1d"].raw.get("rsi") if s.tf["1d"].ok else None,
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
    """
    b = sc.get("breakout") or 0
    e = sc.get("exit_score") or 0
    rsi = sc.get("rsi") or 0
    price = sc.get("price")
    t1 = sc.get("target1")
    t2 = sc.get("target")
    stop = sc.get("stop")

    if e >= 65 or rsi >= 80 or (t2 and price and price >= t2):
        return {"side": "exit", "label": "🔴 EXIT — over-extended",
                "pct": 100, "act_price": price, "day_target": t2 or price}
    if e >= 50 or (t1 and price and price >= t1):
        return {"side": "trim", "label": "🟠 TRIM — book partial",
                "pct": 40, "act_price": price, "day_target": t1 or price}
    if e >= 40:
        return {"side": "trim", "label": "🟡 TRIM light — watch",
                "pct": 25, "act_price": price, "day_target": t1 or price}
    if b >= 58 and e < 45 and price and stop and price > stop and (not t1 or price < t1):
        add_pct = 30 if b >= 70 else 20 if b >= 64 else 10
        return {"side": "add", "label": "🟢 ADD — accumulate",
                "pct": add_pct, "act_price": round(price * 0.99, 2),
                "day_target": t1 or price}
    return {"side": "hold", "label": "⚪ HOLD — do nothing",
            "pct": 0, "act_price": price, "day_target": t1 or price}


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
        "buy_qty": 0.0, "sell_qty": 0.0, "cost": 0.0,
        "first_buy": None, "last_date": None, "last_side": None, "n_trades": 0})
    for o in orders:
        key = f"{o['market']}:{o['symbol']}"
        a = agg[key]
        a["n_trades"] += 1
        q = o["qty"] or 0
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
        }
    return out


def _bar_window_days(timeframes) -> int:
    """How many calendar days count as 'within the current bar' for the chosen
    cadence — used to detect a trade you already made this bar."""
    return {"4h": 1, "1d": 1, "1wk": 7}.get(_primary_tf(timeframes), 1)


def apply_order_history(act: dict, hist: dict, timeframes) -> tuple:
    """Refine a daily_action() result using this holding's order history.
    Returns (act, note). Guards against over-trading within the current bar:
    a recent BUY downgrades an ADD to HOLD; a recent SELL downgrades a (mild)
    TRIM to HOLD. A genuine EXIT (over-extended) is always allowed through."""
    if not hist:
        return act, ""
    import datetime as _d
    note_bits = []
    last = hist.get("last_date")
    days_since = None
    if last is not None and pd.notna(last):
        days_since = (pd.Timestamp.now(tz=None).normalize()
                      - pd.Timestamp(last).normalize()).days
    win = _bar_window_days(timeframes)
    recent = days_since is not None and days_since <= win
    if recent and hist.get("last_side") == "buy" and act["side"] == "add":
        act = dict(act, side="hold", label="✋ HOLD — added recently", pct=0)
        note_bits.append(f"bought {days_since}d ago — skip adding again")
    elif recent and hist.get("last_side") == "sell" and act["side"] == "trim":
        act = dict(act, side="hold", label="✋ HOLD — trimmed recently", pct=0)
        note_bits.append(f"sold {days_since}d ago — skip trimming again")
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
        "recovery puts you back in profit."
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

        sip_rows, ladder_rows = [], []
        for s in top3:
            t = s.targets or {}
            price = t.get("entry")
            pct = round(s.breakout_score / wsum * 100, 1)
            alloc = round(total_cash * pct / 100, 2) if total_cash else None
            daily_chunk = round(daily_total * pct / 100, 2) if daily_total else None
            units_day = int(daily_chunk // price) if (daily_chunk and price and price > 0) else None
            avg_ladder = round(price * ladder_factor, 2) if (price and use_ladder) else None
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
            })
            if use_ladder and daily_chunk and price:
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
        st.dataframe(
            sip_df, use_container_width=True, hide_index=True,
            column_config={
                "Ticker": st.column_config.LinkColumn(
                    "Ticker", display_text=r"symbol=(.+)$"),
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

    st.divider()

    # ============ SECTION B — Exit plan for uploaded positions ============
    st.markdown("### 2️⃣ Exit plan — your holdings")
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

                    # ---- Daily two-sided action (add / trim / exit / hold) ----
                    act = daily_action(sc)
                    hist = orders_summary.get(f"{p['market']}:{p['symbol']}")
                    act, hist_note = apply_order_history(act, hist, selected_tf)
                    held_days = _days_held(hist)
                    hist_avg = hist.get("avg_buy") if hist else None
                    n_trades = hist.get("n_trades") if hist else None
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
                        f"Frees {sym}": free_val,
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
                f"level you're playing for over this {_tf_lbl} bar. **Cash Δ** is "
                "negative when you deploy cash, positive when you free it."
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
                "scale-out limit at/above it). **T1 (scale ~40%)** = first resistance — "
                "book ~40% here to lock in a month's worth of gains; **T2 (runner)** = "
                "full measured-move target for the remainder; **Trail stop @** = exit the "
                "rest if it breaks below. Educational info, not investment advice."
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


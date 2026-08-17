"""Scoring engine: breakout-readiness and over-extended (exit) scores.

Design goals
------------
* Transparent: every score is a weighted blend of named 0-100 sub-scores.
* Two timeframes (4h, 1d) are scored independently then blended (1d weighted
  higher by default because it drives the primary trend).

Breakout readiness (0-100, higher = better)
    A sector that is *consolidating and coiling* for a bull breakout tends to:
      - have MACD hovering near the zero line (momentum reset), turning up
      - be in a tight range (low Bollinger bandwidth = a 'squeeze')
      - have RSI in a neutral-bullish 45-60 zone with room to run
      - show positive relative strength vs its benchmark
      - have low-but-rising ADX with +DI above -DI (trend about to start)
      - show volume dry-up followed by a mild pickup

Exit / over-extended (0-100, higher = time to reduce / exit)
    A sector that has run too far tends to:
      - have RSI overbought (>70)
      - trade far above its 20/50 EMA in ATR terms
      - have MACD stretched well above zero
      - close above the upper Bollinger band (%B > 1)
      - have a very high ADX with momentum (histogram) starting to fade
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ind
from . import patterns as pat

# Canonical timeframes the engine can score (short → long).
TIMEFRAMES = ("4h", "1d", "1wk")
# Preference order for the "primary" timeframe used for targets/consolidation.
PRIMARY_ORDER = ("1d", "1wk", "4h")

# Blend weights between timeframes (normalised across whichever are selected).
TIMEFRAME_WEIGHTS = {"4h": 0.4, "1d": 0.6, "1wk": 0.5}

# Weights for breakout sub-scores (sum need not be 1; normalised internally).
# Breakout sub-scores as a 100-point checklist (weights == points / 100):
#   MACD near zero .......... 30
#   Bollinger squeeze ....... 20
#   Relative strength up .... 20
#   ADX < 20 (pre-trend) .... 10
#   Above the 200-DMA ....... 10
#   Volume confirmation ..... 10
BREAKOUT_WEIGHTS = {
    "macd_zero": 0.30,
    "consolidation": 0.20,
    "rel_strength": 0.20,
    "adx": 0.10,
    "above_200dma": 0.10,
    "volume": 0.10,
}

EXIT_WEIGHTS = {
    "rsi_ob": 0.28,
    "ema_extension": 0.24,
    "macd_stretch": 0.18,
    "bollinger": 0.18,
    "adx_fade": 0.12,
}

# Distribution / "money leaving" sub-scores (0-100, higher = heavier outflow).
# Detects smart-money exiting even while price still looks OK: OBV/price
# divergence and negative money-flow are weighted highest.
DISTRIBUTION_WEIGHTS = {
    "obv_div": 0.30,     # price holding highs while OBV rolls over
    "cmf": 0.22,         # Chaikin Money Flow negative / falling
    "downvol": 0.18,     # down-day volume dominating up-day volume
    "rs_roll": 0.12,     # relative strength vs benchmark rolling over
    "mfi": 0.10,         # Money Flow Index falling / below 50
    "dist_days": 0.08,   # count of high-volume down days (O'Neil)
}


def _gaussian(x: float, center: float, width: float) -> float:
    """Bell curve peaking at 1.0 when x == center."""
    return math.exp(-((x - center) ** 2) / (2 * width ** 2))


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class TimeframeResult:
    ok: bool
    breakout: float = 0.0
    exit: float = 0.0
    distribution: float = 0.0
    breakout_parts: dict = field(default_factory=dict)
    exit_parts: dict = field(default_factory=dict)
    distribution_parts: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def _weighted(parts: dict, weights: dict) -> float:
    total_w = sum(weights.values())
    return sum(parts[k] * weights[k] for k in weights) / total_w


def score_timeframe(df: pd.DataFrame, bench_df: pd.DataFrame | None) -> TimeframeResult:
    if df is None or df.empty or len(df) < 40:
        return TimeframeResult(ok=False)

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    macd_line, signal_line, hist = ind.macd(close)
    rsi = ind.rsi(close)
    adx, plus_di, minus_di = ind.adx(high, low, close)
    _, bb_up, _, bandwidth = ind.bollinger(close)
    bw_pct = ind.bandwidth_percentile(bandwidth)
    atr = ind.atr(high, low, close)
    ema20 = ind.ema(close, 20)
    ema50 = ind.ema(close, 50)
    sma200 = ind.sma(close, 200)

    last = -1
    price = close.iloc[last]
    atr_v = atr.iloc[last] or (price * 0.01)

    macd_v = macd_line.iloc[last]
    hist_v = hist.iloc[last]
    hist_prev = hist.iloc[last - 1]
    rsi_v = rsi.iloc[last]
    rsi_prev = rsi.iloc[last - 3] if len(rsi) > 3 else rsi_v
    adx_v = adx.iloc[last]
    plus_v = plus_di.iloc[last]
    minus_v = minus_di.iloc[last]

    # MACD measured as a percentage of price so it is comparable across tickers.
    macd_pct = macd_v / price * 100

    rel_str = 0.0
    if bench_df is not None and not bench_df.empty:
        rel_str = ind.relative_strength(close, bench_df["Close"].astype(float))

    # Volume: recent 5-bar average vs prior 20-bar average.
    vol_recent = volume.tail(5).mean()
    vol_base = volume.tail(25).head(20).mean()
    vol_ratio = (vol_recent / vol_base) if vol_base else 1.0

    # ---------------- Breakout sub-scores (100-point checklist) ----------------
    # MACD near zero and turning up  (30 pts).
    zero_score = _gaussian(macd_pct, center=0.0, width=0.6) * 100
    if hist_v > hist_prev:            # momentum improving
        zero_score = _clamp(zero_score + 12)
    if macd_v > signal_line.iloc[last] and abs(macd_pct) < 1.0:  # fresh cross near zero
        zero_score = _clamp(zero_score + 12)

    # Bollinger squeeze — tighter bandwidth vs history = higher  (20 pts).
    consolidation_score = _clamp(100 - bw_pct)

    # Relative strength improving vs benchmark  (20 pts).
    rel_score = _clamp(50 + rel_str * 5)   # +10% RS -> 100, -10% -> 0

    # ADX < 20 (pre-trend coil): full marks below 20, taper to 0 by 40  (10 pts).
    if adx_v <= 20:
        adx_score = 100.0
    else:
        adx_score = _clamp(100 - (adx_v - 20) / (40 - 20) * 100)
    if plus_v > minus_v:               # bullish DI alignment nudges it up
        adx_score = _clamp(adx_score + 10)
    else:
        adx_score = _clamp(adx_score - 10)

    # Above the 200-DMA — long-term trend filter  (10 pts).
    sma200_v = float(sma200.iloc[last])
    if math.isnan(sma200_v) or sma200_v <= 0:
        dma_score = 50.0               # not enough history — neutral
        dist_200 = None
    else:
        dist_200 = (price / sma200_v - 1) * 100
        dma_score = 100.0 if price >= sma200_v else _clamp(100 + dist_200 * 6)

    # Volume confirmation — a mild pickup (≈1.0-1.6x) is ideal  (10 pts).
    vol_score = _gaussian(vol_ratio, center=1.3, width=0.5) * 100

    breakout_parts = {
        "macd_zero": round(zero_score, 1),
        "consolidation": round(consolidation_score, 1),
        "rel_strength": round(rel_score, 1),
        "adx": round(adx_score, 1),
        "above_200dma": round(dma_score, 1),
        "volume": round(vol_score, 1),
    }
    breakout = _weighted(breakout_parts, BREAKOUT_WEIGHTS)

    # ---------------- Exit sub-scores ----------------
    rsi_ob = _clamp((rsi_v - 60) / (82 - 60) * 100)

    ema_ext_atr = (price - ema20.iloc[last]) / atr_v          # ATRs above EMA20
    ema_score = _clamp((ema_ext_atr - 1.5) / (5.0 - 1.5) * 100)

    macd_stretch = _clamp((macd_pct - 1.0) / (4.0 - 1.0) * 100)

    pct_b = (price - bb_up.iloc[last]) / atr_v                # >0 means above upper band
    bb_score = _clamp((pct_b + 0.5) / 2.0 * 100)

    adx_fade = 0.0
    if adx_v > 35 and hist_v < hist_prev:                     # strong trend but fading
        adx_fade = _clamp((adx_v - 35) / (55 - 35) * 100)

    exit_parts = {
        "rsi_ob": round(rsi_ob, 1),
        "ema_extension": round(ema_score, 1),
        "macd_stretch": round(macd_stretch, 1),
        "bollinger": round(bb_score, 1),
        "adx_fade": round(adx_fade, 1),
    }
    exit_score = _weighted(exit_parts, EXIT_WEIGHTS)

    # ---------------- Distribution sub-scores ("money leaving") ----------------
    # These fire when volume/flow deteriorates even while price still holds up —
    # the classic footprint of smart money selling into strength.
    obv_line = ind.obv(close, volume)
    cmf_v = float(ind.cmf(high, low, close, volume).iloc[last])
    mfi_series = ind.mfi(high, low, close, volume)
    mfi_v = float(mfi_series.iloc[last])
    mfi_prev = float(mfi_series.iloc[last - 5]) if len(mfi_series) > 5 else mfi_v

    # OBV/price divergence over ~15 bars: price near its recent high but OBV well
    # off its own high => accumulation is quietly reversing.
    div_score = 0.0
    n = 15
    if len(close) > n and len(obv_line) > n:
        pr = close.tail(n)
        ov = obv_line.tail(n)
        pr_rank = float((pr.iloc[-1] - pr.min()) / ((pr.max() - pr.min()) or np.nan))
        ov_rank = float((ov.iloc[-1] - ov.min()) / ((ov.max() - ov.min()) or np.nan))
        if not (math.isnan(pr_rank) or math.isnan(ov_rank)):
            # High price-rank with low OBV-rank => bearish divergence.
            div_score = _clamp((pr_rank - ov_rank) * 150) if pr_rank >= 0.5 else 0.0

    # Chaikin Money Flow: -0.10 or lower => full marks, +0.10 => zero.
    cmf_score = _clamp((0.10 - cmf_v) / 0.20 * 100)

    # Down-day volume dominance over the last 20 bars.
    ret = close.diff().tail(20)
    vol20 = volume.tail(20)
    up_vol = float(vol20[ret > 0].sum())
    dn_vol = float(vol20[ret < 0].sum())
    dvr = dn_vol / (up_vol + dn_vol) if (up_vol + dn_vol) else 0.5
    downvol_score = _clamp((dvr - 0.5) / 0.25 * 100)

    # Relative strength rolling over (negative RS vs benchmark).
    rs_roll_score = _clamp(-rel_str / 10 * 100)

    # MFI falling and below the 50 midline.
    mfi_score = _clamp((55 - mfi_v) / 25 * 100) if mfi_v < mfi_prev else 0.0

    # Distribution days: down >0.7% on >1.3x average volume in last 25 bars.
    vbase = volume.tail(45).head(20).mean() or 1.0
    r25 = close.pct_change().tail(25)
    v25 = volume.tail(25)
    dist_days = int(((r25 < -0.007) & (v25 > 1.3 * vbase)).sum())
    dist_days_score = _clamp(dist_days / 5 * 100)

    distribution_parts = {
        "obv_div": round(div_score, 1),
        "cmf": round(cmf_score, 1),
        "downvol": round(downvol_score, 1),
        "rs_roll": round(rs_roll_score, 1),
        "mfi": round(mfi_score, 1),
        "dist_days": round(dist_days_score, 1),
    }
    distribution_score = _weighted(distribution_parts, DISTRIBUTION_WEIGHTS)

    # Signal line as % of price + how long MACD has hugged the zero line.
    signal_v = float(signal_line.iloc[last])
    signal_pct = signal_v / price * 100
    macd_abs_pct = (macd_line / close * 100).abs()
    near_zero_bars = 0
    for v in macd_abs_pct.iloc[::-1]:
        if pd.notna(v) and v < 0.8:
            near_zero_bars += 1
        else:
            break

    raw = {
        "price": round(float(price), 2),
        "macd_pct": round(float(macd_pct), 3),
        "signal_pct": round(float(signal_pct), 3),
        "near_zero_bars": int(near_zero_bars),
        "rsi": round(float(rsi_v), 1),
        "adx": round(float(adx_v), 1),
        "plus_di": round(float(plus_v), 1),
        "minus_di": round(float(minus_v), 1),
        "bandwidth_pct_rank": round(float(bw_pct), 0),
        "rel_strength_pct": round(float(rel_str), 2),
        "vol_ratio": round(float(vol_ratio), 2),
        "avg_vol": round(float(vol_base), 0) if vol_base else None,
        "cmf": round(cmf_v, 3),
        "mfi": round(mfi_v, 1),
        "dist_days": dist_days,
        "ema20_atr_ext": round(float(ema_ext_atr), 2),
        "dist_200dma_pct": round(float(dist_200), 2) if dist_200 is not None else None,
    }

    return TimeframeResult(
        ok=True,
        breakout=round(breakout, 1),
        exit=round(exit_score, 1),
        distribution=round(distribution_score, 1),
        breakout_parts=breakout_parts,
        exit_parts=exit_parts,
        distribution_parts=distribution_parts,
        raw=raw,
    )


@dataclass
class SectorScore:
    ticker: str
    name: str
    breakout_score: float
    exit_score: float
    signal: str
    tf: dict  # {'4h': TimeframeResult, '1d': TimeframeResult, '1wk': TimeframeResult}
    breakout_delta: float = 0.0
    breakout_trend: str = "→ Flat"
    exit_delta: float = 0.0
    distribution_score: float = 0.0
    targets: dict = field(default_factory=dict)
    consolidation: dict = field(default_factory=dict)
    action: str = ""
    setup_quality: float = 0.0
    maturity: str = "—"
    maturity_note: str = ""
    readiness_factor: float = 1.0
    patterns: dict = field(default_factory=dict)
    breakout_1w_ago: float | None = None
    breakout_2w_ago: float | None = None


# Bars ≈ two weeks per timeframe for the score trend (10 trading days on 1D;
# ~2 four-hour bars per session × 10 sessions on 4h).
TREND_LOOKBACK = {"1d": 10, "4h": 20, "1wk": 4}

# Per-timeframe bar counts for looking the score back 1 and 2 weeks.
HORIZON_LOOKBACK = {
    "1w": {"1d": 5, "4h": 10, "1wk": 1},
    "2w": {"1d": 10, "4h": 20, "1wk": 2},
}


def _frames_back(frames: dict, lookback: dict) -> dict:
    """Return copies of ``frames`` trimmed by ``lookback`` bars per timeframe,
    so the engine can be re-run 'as of' that many bars ago."""
    out = {}
    for key, df in (frames or {}).items():
        n = lookback.get(key, 0)
        if df is None or df.empty or n <= 0:
            out[key] = df
        else:
            out[key] = df.iloc[:-n] if len(df) > n else df.iloc[:0]
    return out


# ---------------------------------------------------------------------------
# Consolidation maturity — a base that has coiled for a *long* time is less
# likely to break out in the near term ("dead money"). Bands are in CALENDAR
# DAYS of the detected consolidation and map to a readiness multiplier applied
# to the breakout score. A fresh volume/MACD trigger largely cancels the
# penalty — the longer the base, the bigger the move once it finally resolves.
# ---------------------------------------------------------------------------
MATURITY_BANDS = [
    # min_days, max_days, label,        factor, note
    (0,    20,      "🌱 Forming",  0.92, "Base still young — the box may not hold yet."),
    (20,   60,      "✅ Prime",    1.00, "Ideal breakout window (about 3–8 weeks)."),
    (60,   90,      "🟡 Maturing", 0.88, "Getting long — needs a trigger soon or it goes stale."),
    (90,   150,     "🟠 Extended", 0.76, "Long base — breakout odds fade for the next ~30 days without a catalyst."),
    (150,  10 ** 6, "🔴 Stale",    0.64, "Dead-money base — hold off until a strong volume/MACD trigger fires."),
]


def consolidation_maturity(days: int | None, has_trigger: bool) -> dict:
    """Classify how mature/stale the consolidation is and how much to discount
    the breakout score. A fresh breakout trigger largely removes the penalty."""
    if not days or days <= 0:
        return {"label": "—", "factor": 1.0, "note": "", "days": days or 0}
    label, factor, note = MATURITY_BANDS[-1][2:]
    for lo, hi, lbl, fac, nte in MATURITY_BANDS:
        if lo <= days < hi:
            label, factor, note = lbl, fac, nte
            break
    if has_trigger and factor < 1.0:
        factor = min(1.0, factor + 0.25)
        note = "⚡ Trigger firing — long base may finally be resolving; watch closely."
        label = f"{label} + trigger"
    return {"label": label, "factor": round(factor, 2), "note": note, "days": days}


def _trend_label(delta: float) -> str:
    if delta >= 2.0:
        return "↑ Rising"
    if delta <= -2.0:
        return "↓ Falling"
    return "→ Flat"


def action_plan(breakout: float, exit_score: float, trend_delta: float,
                consolidating: bool, stale_base: bool = False) -> str:
    """Staged capital-deployment guidance: accumulate small during consolidation,
    scale in on confirmation, and book partial profits when extended.
    """
    # --- Profit-booking / exit side takes priority ---
    if exit_score >= 65:
        return "🔴 EXIT — book full profits (over-extended)"
    if exit_score >= 50:
        return "🟠 BOOK PARTIAL PROFITS (~25–50%) + trail stop"
    # --- Stale/dormant base without a trigger: don't pre-position capital ---
    if stale_base and breakout >= 45:
        return "⌛ STALE BASE — accumulation on hold; wait for a volume/MACD breakout trigger"
    # --- Accumulation side ---
    if breakout >= 70 and trend_delta >= 0:
        return "🟢 CONFIRMED — scale in (deploy 50–75%)"
    if breakout >= 58:
        return "🟢 START SIP — deploy ~25% now, add on breakout"
    if breakout >= 45 and consolidating:
        return "🟡 EARLY SIP — start small (~10%), add on dips"
    if breakout >= 45:
        return "🟡 Watchlist — wait for tighter setup"
    return "⚪ Avoid — no position yet"


def detect_consolidation(df: pd.DataFrame, interval: str, cap_atr: float = 5.0,
                         max_bars: int = 200) -> dict:
    """Detect how long the current sideways consolidation has lasted.

    Walks backward from the last bar, expanding a high/low box while its total
    height stays within ``cap_atr`` * ATR. The first bar that would blow the box
    out marks the start of the consolidation.

    Returns start date, bar count, calendar days and the range width (%).
    """
    if df is None or df.empty or len(df) < 15:
        return {}
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    price = float(close.iloc[-1])
    atr_v = float(ind.atr(high, low, close).iloc[-1])
    if atr_v <= 0:
        atr_v = price * 0.01
    cap = cap_atr * atr_v

    n = len(df)
    hi = float(high.iloc[-1])
    lo = float(low.iloc[-1])
    start_idx = n - 1
    for i in range(n - 2, max(-1, n - 1 - max_bars), -1):
        new_hi = max(hi, float(high.iloc[i]))
        new_lo = min(lo, float(low.iloc[i]))
        if (new_hi - new_lo) <= cap:
            hi, lo = new_hi, new_lo
            start_idx = i
        else:
            break

    start_ts = df.index[start_idx]
    bars = n - start_idx
    days = int((df.index[-1] - start_ts).days)
    unit = {"1d": "days", "4h": "4h-bars", "1wk": "weeks"}.get(interval, "bars")
    return {
        "interval": interval,
        "start": start_ts.strftime("%Y-%m-%d"),
        "bars": bars,
        "unit": unit,
        "days": days,
        "range_pct": round((hi - lo) / price * 100, 1),
        "range_low": round(lo, 2),
        "range_high": round(hi, 2),
    }


# Suggested % of a position to scale out at the interim (T1) target.
SCALE_OUT_T1_PCT = 40


def project_targets(df: pd.DataFrame) -> dict:
    """A realistic measured-move target, stop and risk:reward from consolidation.

    * breakout level = recent 20-bar high (resistance)
    * measured move  = the consolidation range height (min 2.5*ATR)
    * target1 (T1)   = first resistance — interim scale-out (~monthly cadence)
    * target  (T2)   = breakout level + measured move — full runner
    * stop           = min(range low, price - 1.5*ATR)
    """
    if df is None or df.empty or len(df) < 25:
        return {}
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    atr_v = float(ind.atr(high, low, close).iloc[-1])
    price = float(close.iloc[-1])
    if atr_v <= 0:
        atr_v = price * 0.01

    range_high = float(high.tail(20).max())
    range_low = float(low.tail(20).min())
    measured = max(range_high - range_low, 2.5 * atr_v)

    target = range_high + measured          # full measured-move (T2 / runner)
    stop = min(range_low, price - 1.5 * atr_v)

    # Interim first target (T1) for a partial scale-out. First resistance is the
    # natural place to book part of the position; once price is already above it
    # we fall back to the midpoint of the remaining move to the full target.
    if price < range_high:
        target1 = range_high
    else:
        target1 = price + 0.5 * (target - price)
    target1 = max(target1, price * 1.01)    # keep it meaningfully above price
    target1 = min(target1, target)          # never above the full target

    upside_pct = (target / price - 1) * 100
    target1_pct = (target1 / price - 1) * 100
    downside_pct = (stop / price - 1) * 100
    risk = price - stop
    reward = target - price
    rr = (reward / risk) if risk > 0 else None

    return {
        "entry": round(price, 2),
        "breakout_level": round(range_high, 2),
        "target1": round(target1, 2),
        "target1_pct": round(target1_pct, 1),
        "scale_out_pct": SCALE_OUT_T1_PCT,
        "target": round(target, 2),
        "stop": round(stop, 2),
        "upside_pct": round(upside_pct, 1),
        "downside_pct": round(downside_pct, 1),
        "risk_reward": round(rr, 2) if rr else None,
    }


def _blend(values: dict[str, float | None], timeframes: tuple[str, ...]) -> float:
    """Weighted blend of the selected timeframes, skipping unavailable ones."""
    vals, wts = [], []
    for key in timeframes:
        v = values.get(key)
        if v is not None:
            vals.append(v)
            wts.append(TIMEFRAME_WEIGHTS[key])
    if not vals:
        return 0.0
    return sum(v * w for v, w in zip(vals, wts)) / sum(wts)


def classify(breakout: float, exit_score: float) -> str:
    """Turn the two scores into a plain-English action signal."""
    if exit_score >= 65:
        return "🔴 EXIT / TAKE PROFITS — over-extended"
    if exit_score >= 50:
        return "🟠 TRIM — getting extended"
    if breakout >= 70 and exit_score < 45:
        return "🟢 STRONG BREAKOUT SETUP"
    if breakout >= 58 and exit_score < 45:
        return "🟢 BUILDING — watch for trigger"
    if breakout >= 45:
        return "🟡 CONSOLIDATING — not ready yet"
    return "⚪ NEUTRAL / AVOID"


def lifecycle_stage(breakout: float, exit_score: float, distribution: float,
                    dma200_falling: bool = False) -> dict:
    """Map the breakout / exit / distribution scores onto a 5-stage market
    lifecycle (Wyckoff / Stan-Weinstein style). Priority is top-down: an active
    distribution/decline read overrides a still-bullish breakout score, because
    money leaving is the most actionable warning.

    Returns a dict: {order, stage, label, action, color} where ``order`` sorts
    the table 1→5 (accumulate → decline) and ``color`` is a hex background.
    """
    # Stage 5 — Decline / Markdown: trend broken, price under a falling 200-DMA.
    if dma200_falling and breakout < 40 and distribution >= 45:
        return {"order": 5, "stage": "5 · Decline",
                "label": "⚫ Stage 5 — Decline / Markdown",
                "action": "Avoid — downtrend; wait for a new base to form",
                "color": "#3a3f44"}
    # Stage 4 — Distribution / Topping: money leaving while price still holds.
    if distribution >= 60 or (distribution >= 50 and exit_score >= 45):
        return {"order": 4, "stage": "4 · Distribution",
                "label": "🔴 Stage 4 — Distribution / Topping",
                "action": "Exit / take profits — smart money is selling",
                "color": "#b23b3b"}
    # Stage 3 — Stretched / Extended: still up but overheated.
    if exit_score >= 50:
        return {"order": 3, "stage": "3 · Stretched",
                "label": "🟠 Stage 3 — Stretched / Extended",
                "action": "Trim & trail stop — overheated, tighten risk",
                "color": "#c77d3a"}
    # Stage 2 — Breakout / Markup: trend firing.
    if breakout >= 58 and exit_score < 50:
        return {"order": 2, "stage": "2 · Breakout",
                "label": "🟢 Stage 2 — Breakout / Markup",
                "action": "Hold / add on strength — trend is running",
                "color": "#1e7d46"}
    # Stage 1 — Basing / Consolidating: coiling, about to break.
    if breakout >= 45:
        return {"order": 1, "stage": "1 · Consolidating",
                "label": "🟡 Stage 1 — Basing / Consolidating",
                "action": "Accumulate small — coiling, watch for the trigger",
                "color": "#b3952f"}
    return {"order": 6, "stage": "0 · Neutral",
            "label": "⚪ Neutral — no edge",
            "action": "No position — no clear stage yet",
            "color": ""}


def score_sector(
    ticker: str, name: str, frames: dict, bench_frames: dict,
    timeframes: tuple[str, ...] = ("4h", "1d"),
) -> SectorScore:
    tf = {}
    for key in TIMEFRAMES:
        bench = bench_frames.get(key) if bench_frames else None
        tf[key] = score_timeframe(frames.get(key, pd.DataFrame()), bench)

    breakout_vals = {k: (tf[k].breakout if tf[k].ok else None) for k in TIMEFRAMES}
    exit_vals = {k: (tf[k].exit if tf[k].ok else None) for k in TIMEFRAMES}
    dist_vals = {k: (tf[k].distribution if tf[k].ok else None) for k in TIMEFRAMES}

    setup_quality = round(_blend(breakout_vals, timeframes), 1)
    exit_score = round(_blend(exit_vals, timeframes), 1)
    distribution_score = round(_blend(dist_vals, timeframes), 1)

    # ---- Score trend: re-score a few bars back and compare ----
    prev_breakout_vals: dict[str, float | None] = {}
    prev_exit_vals: dict[str, float | None] = {}
    for key in TIMEFRAMES:
        df = frames.get(key, pd.DataFrame())
        back = TREND_LOOKBACK[key]
        if df is not None and len(df) > 40 + back:
            prev = score_timeframe(df.iloc[:-back], bench_frames.get(key) if bench_frames else None)
            prev_breakout_vals[key] = prev.breakout if prev.ok else None
            prev_exit_vals[key] = prev.exit if prev.ok else None
        else:
            prev_breakout_vals[key] = None
            prev_exit_vals[key] = None

    prev_breakout = _blend(prev_breakout_vals, timeframes)
    prev_exit = _blend(prev_exit_vals, timeframes)
    raw_breakout_delta = setup_quality - prev_breakout
    exit_delta = round(exit_score - prev_exit, 1)

    # ---- Realistic target + consolidation from the primary timeframe ----
    # Prefer a selected timeframe (1d > 1wk > 4h); otherwise fall back to any
    # timeframe that scored successfully.
    target_key = next(
        (k for k in PRIMARY_ORDER if k in timeframes and tf[k].ok),
        next((k for k in PRIMARY_ORDER if tf[k].ok), None),
    )
    targets = project_targets(frames.get(target_key, pd.DataFrame())) if target_key else {}
    consolidation = (
        detect_consolidation(frames.get(target_key, pd.DataFrame()), target_key)
        if target_key else {}
    )

    # ---- Consolidation maturity: discount stale, long-dormant bases ----
    target_raw = tf[target_key].raw if (target_key and tf[target_key].ok) else {}
    has_trigger = (
        target_raw.get("vol_ratio", 0) >= 1.5
        and abs(target_raw.get("macd_pct", 9)) < 1.2
        and raw_breakout_delta >= 2.0
    )
    maturity = consolidation_maturity(consolidation.get("days"), has_trigger)
    factor = maturity["factor"]

    breakout = round(setup_quality * factor, 1)
    breakout_delta = round((setup_quality - prev_breakout) * factor, 1)
    signal = classify(breakout, exit_score)

    is_consolidating = bool(consolidation) and consolidation.get("range_pct", 100) <= 12
    stale_base = factor < 0.85
    action = action_plan(breakout, exit_score, breakout_delta, is_consolidating, stale_base)

    if consolidation:
        consolidation["maturity"] = maturity["label"]

    # ---- Chart patterns on the primary timeframe frame ----
    patterns = (
        pat.detect_patterns(frames.get(target_key, pd.DataFrame()))
        if target_key else {"primary": "", "labels": "", "all": []}
    )

    return SectorScore(
        ticker=ticker, name=name,
        breakout_score=breakout, exit_score=exit_score,
        signal=signal, tf=tf,
        breakout_delta=breakout_delta,
        breakout_trend=_trend_label(breakout_delta),
        exit_delta=exit_delta,
        distribution_score=distribution_score,
        targets=targets,
        consolidation=consolidation,
        action=action,
        setup_quality=setup_quality,
        maturity=maturity["label"],
        maturity_note=maturity["note"],
        readiness_factor=factor,
        patterns=patterns,
    )


def breakout_snapshot(
    ticker: str, name: str, frames: dict, bench_frames: dict,
    timeframes: tuple[str, ...], horizon: str,
) -> float | None:
    """Re-run the full breakout score 'as of' 1 or 2 weeks ago (``horizon`` in
    {'1w','2w'}) by trimming each timeframe by the matching number of bars.
    Returns the historical breakout score, or None if it can't be computed."""
    lookback = HORIZON_LOOKBACK.get(horizon)
    if not lookback:
        return None
    past = _frames_back(frames, lookback)
    try:
        return score_sector(ticker, name, past, bench_frames, timeframes).breakout_score
    except Exception:
        return None

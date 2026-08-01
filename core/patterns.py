"""Classical chart-pattern detection (heuristic, dependency-light).

Detects a handful of *bullish* setups on an OHLCV frame:

  * VCP            – Volatility Contraction Pattern (Minervini-style tightening)
  * Cup with Handle
  * Double Bottom
  * Bull Flag

These are rule-based approximations meant to *flag candidates* for a human to
confirm visually on the chart — not exact/ML pattern recognisers. Each detector
returns a confidence in 0-100; only patterns above ``MIN_CONFIDENCE`` are
reported by :func:`detect_patterns`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CONFIDENCE = 55.0


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def _cols(df: pd.DataFrame):
    return (
        df["High"].astype(float),
        df["Low"].astype(float),
        df["Close"].astype(float),
        df["Volume"].astype(float) if "Volume" in df else pd.Series(1.0, index=df.index),
    )


def _swings(series: pd.Series, order: int = 3, kind: str = "high") -> list[int]:
    """Indices of local extrema via a simple k-neighbour test."""
    v = series.values.astype(float)
    n = len(v)
    out: list[int] = []
    for i in range(order, n - order):
        seg = v[i - order:i + order + 1]
        if kind == "high" and int(np.argmax(seg)) == order:
            out.append(i)
        elif kind == "low" and int(np.argmin(seg)) == order:
            out.append(i)
    return out


# --------------------------------------------------------------------------
# VCP — successive volatility contractions into the highs on drying volume.
# --------------------------------------------------------------------------
def detect_vcp(df: pd.DataFrame, lookback: int = 72) -> tuple[float, str]:
    if df is None or len(df) < 45:
        return 0.0, ""
    seg = df.tail(lookback)
    high, low, close, vol = _cols(seg)
    n = len(seg)
    third = n // 3
    if third < 6:
        return 0.0, ""

    def rng_pct(h, l):
        top, bot = float(h.max()), float(l.min())
        return (top - bot) / top * 100 if top else 0.0

    r1 = rng_pct(high.iloc[:third], low.iloc[:third])              # oldest
    r2 = rng_pct(high.iloc[third:2 * third], low.iloc[third:2 * third])
    r3 = rng_pct(high.iloc[2 * third:], low.iloc[2 * third:])      # newest / tightest
    if not (r1 > r2 > r3) or r1 <= 0:
        return 0.0, ""

    contraction = 1 - (r3 / r1)                # 0..1, higher = tighter finish
    if r3 > 0.7 * r1:                          # must be a real tightening
        return 0.0, ""

    # Price near the top of the whole base (holding the highs).
    top = float(high.max())
    bot = float(low.min())
    pos = (float(close.iloc[-1]) - bot) / (top - bot) if top > bot else 0.0
    if pos < 0.55:
        return 0.0, ""

    # Volume drying up (newest third vs oldest third).
    v1 = float(vol.iloc[:third].mean())
    v3 = float(vol.iloc[2 * third:].mean())
    vol_dry = v3 < v1 if v1 else False

    conf = 40 + contraction * 45 + (pos - 0.55) * 40 + (10 if vol_dry else 0)
    note = (
        f"3 contractions {r1:.1f}%→{r2:.1f}%→{r3:.1f}%, "
        f"price {pos*100:.0f}% up the base"
        + (", volume drying up" if vol_dry else "")
    )
    return _clamp(conf), note


# --------------------------------------------------------------------------
# Cup with Handle — rounded recovery to the old high, then a shallow handle.
# --------------------------------------------------------------------------
def detect_cup_handle(df: pd.DataFrame, lookback: int = 130) -> tuple[float, str]:
    if df is None or len(df) < 60:
        return 0.0, ""
    seg = df.tail(lookback)
    high, low, close, _ = _cols(seg)
    n = len(seg)
    handle_room = max(5, n // 12)

    # Left rim: highest high in the first ~45% of the window.
    left_end = max(3, int(n * 0.45))
    left_idx = int(high.iloc[:left_end].argmax())
    left_peak = float(high.iloc[:left_end].max())

    # Cup bottom: lowest low between the left rim and the handle zone.
    mid = low.iloc[left_idx:n - handle_room]
    if len(mid) < 8:
        return 0.0, ""
    bottom_idx = int(mid.argmin()) + left_idx
    bottom = float(low.iloc[left_idx:n - handle_room].min())

    # Right rim: highest high after the bottom (before the handle).
    right = high.iloc[bottom_idx:n - handle_room]
    if len(right) < 3:
        return 0.0, ""
    right_peak = float(right.max())

    if left_peak <= 0:
        return 0.0, ""
    depth = (left_peak - bottom) / left_peak
    if not (0.12 <= depth <= 0.45):            # a real, not-too-deep cup
        return 0.0, ""
    if right_peak < 0.90 * left_peak:          # must recover near the old high
        return 0.0, ""

    # Roundedness: bottom sits near the middle of the left→right span.
    span = (n - handle_room) - left_idx
    centering = (bottom_idx - left_idx) / span if span else 0.5
    if not (0.30 <= centering <= 0.75):
        return 0.0, ""

    # Handle: shallow late pullback that holds in the upper part of the cup.
    handle = seg.iloc[n - handle_room:]
    h_high = float(handle["High"].astype(float).max())
    h_low = float(handle["Low"].astype(float).min())
    handle_depth = (h_high - h_low) / h_high if h_high else 1.0
    if handle_depth > depth / 2 + 0.02:        # handle must be shallow vs cup
        return 0.0, ""
    if h_low < bottom + 0.5 * (right_peak - bottom):  # handle stays high in the cup
        return 0.0, ""

    round_score = 1 - abs(centering - 0.5) * 2
    conf = 45 + round_score * 25 + (right_peak / left_peak - 0.90) * 120
    conf += 10 if handle_depth < depth / 3 else 0
    note = (
        f"cup depth {depth*100:.0f}%, recovered to {right_peak/left_peak*100:.0f}% "
        f"of the left rim, handle {handle_depth*100:.0f}% deep"
    )
    return _clamp(conf), note


# --------------------------------------------------------------------------
# Double Bottom — two lows near the same level, price reclaiming the neckline.
# --------------------------------------------------------------------------
def detect_double_bottom(df: pd.DataFrame, lookback: int = 90) -> tuple[float, str]:
    if df is None or len(df) < 40:
        return 0.0, ""
    seg = df.tail(lookback)
    high, low, close, _ = _cols(seg)
    lows = _swings(low, order=3, kind="low")
    if len(lows) < 2:
        return 0.0, ""

    # Consider the two most-recent significant swing lows.
    best = None
    for a in range(len(lows)):
        for b in range(a + 1, len(lows)):
            i, j = lows[a], lows[b]
            if j - i < 8:                       # need separation
                continue
            la, lb = float(low.iloc[i]), float(low.iloc[j])
            base = min(la, lb)
            if base <= 0:
                continue
            diff = abs(la - lb) / base
            if diff > 0.04:                     # bottoms roughly equal (±4%)
                continue
            peak = float(high.iloc[i:j].max())  # neckline between the bottoms
            rise = (peak - base) / base
            if rise < 0.05:                     # a real hump between them
                continue
            best = (i, j, base, peak, diff, rise)
    if not best:
        return 0.0, ""

    i, j, base, peak, diff, rise = best
    price = float(close.iloc[-1])
    # How far price has reclaimed toward the neckline (breakout trigger).
    reclaim = (price - base) / (peak - base) if peak > base else 0.0
    if reclaim < 0.4:                           # still near the lows, not confirming
        return 0.0, ""

    conf = 45 + (1 - diff / 0.04) * 20 + _clamp(reclaim, 0, 1.2) * 30
    note = (
        f"two lows ~{base:.2f} ({diff*100:.1f}% apart), neckline {peak:.2f}, "
        f"price has reclaimed {reclaim*100:.0f}% toward breakout"
    )
    return _clamp(conf), note


# --------------------------------------------------------------------------
# Bull Flag — a sharp flagpole then a short, shallow drift on lighter volume.
# --------------------------------------------------------------------------
def detect_bull_flag(df: pd.DataFrame, lookback: int = 30) -> tuple[float, str]:
    if df is None or len(df) < 25:
        return 0.0, ""
    seg = df.tail(lookback)
    high, low, close, vol = _cols(seg)
    n = len(seg)
    flag_len = max(4, n // 4)
    pole = seg.iloc[: n - flag_len]
    flag = seg.iloc[n - flag_len:]
    if len(pole) < 6 or len(flag) < 4:
        return 0.0, ""

    p_low = float(pole["Low"].astype(float).min())
    p_high = float(pole["High"].astype(float).max())
    pole_gain = (p_high - p_low) / p_low if p_low else 0.0
    if pole_gain < 0.10:                        # need a real flagpole (>=10%)
        return 0.0, ""

    f_high = float(flag["High"].astype(float).max())
    f_low = float(flag["Low"].astype(float).min())
    flag_range = (f_high - f_low) / f_high if f_high else 1.0
    if flag_range > 0.09:                       # flag must be tight
        return 0.0, ""

    # Flag should drift sideways/down and hold above the mid of the pole.
    flag_close = flag["Close"].astype(float).values
    slope = float(np.polyfit(np.arange(len(flag_close)), flag_close, 1)[0])
    if f_low < p_low + 0.5 * (p_high - p_low):
        return 0.0, ""

    v_pole = float(pole["Volume"].astype(float).mean()) if "Volume" in seg else 0.0
    v_flag = float(flag["Volume"].astype(float).mean()) if "Volume" in seg else 0.0
    vol_dry = v_flag < v_pole if v_pole else False

    conf = 45 + _clamp(pole_gain * 120, 0, 30) + (1 - flag_range / 0.09) * 15
    conf += 8 if vol_dry else 0
    conf -= 8 if slope > 0.0 else 0  # a strong up-drift is less flag-like
    note = (
        f"flagpole +{pole_gain*100:.0f}%, tight flag ({flag_range*100:.0f}% range)"
        + (", volume contracting" if vol_dry else "")
    )
    return _clamp(conf), note


_DETECTORS = {
    "🟩 VCP": detect_vcp,
    "🏆 Cup & Handle": detect_cup_handle,
    "⚏ Double Bottom": detect_double_bottom,
    "🚩 Bull Flag": detect_bull_flag,
}


# --------------------------------------------------------------------------
# 52-week-high breakout → retest.
#
# The classic continuation setup: a stock clears a prior resistance/base to
# print a fresh 52-week high (the breakout), then pulls back to *retest* that
# breakout level (old resistance becoming new support) — ideally on lighter
# volume — before (hopefully) resuming higher.
# --------------------------------------------------------------------------
def detect_breakout_retest(
    df: pd.DataFrame,
    prox_pct: float = 6.0,       # how close to the 52w high still counts
    base_start: int = 60,        # base window ends `base_end` bars ago
    base_end: int = 15,
    thrust_pct: float = 3.0,     # breakout must clear the base top by this %
    retest_band: float = 5.0,    # close must be back within this % above the level
    undercut: float = 1.5,       # allow this % dip below the level (support test)
) -> dict:
    """Return a dict describing a 52w-high breakout-and-retest, or ``{}``.

    Keys: confidence, price, high_52w, pct_from_high, breakout_level,
    dist_to_level_pct, retest_depth_pct, vol_light, note.
    """
    empty: dict = {}
    if df is None or len(df) < 130:
        return empty
    high, low, close, vol = _cols(df)
    price = float(close.iloc[-1])

    # --- Fresh 52-week high territory ---
    window52 = min(252, len(df))
    high_52w = float(high.iloc[-window52:].max())
    if high_52w <= 0:
        return empty
    pct_from_high = (price / high_52w - 1) * 100
    if pct_from_high < -prox_pct:          # too far below the highs
        return empty

    # --- Prior base / resistance that was broken ---
    base = high.iloc[-base_start:-base_end]
    if len(base) < 10:
        return empty
    resistance = float(base.max())
    if resistance <= 0:
        return empty

    recent_high = float(high.iloc[-base_end:].max())   # the breakout thrust
    broke = recent_high > resistance * (1 + thrust_pct / 100)
    made_new_high = recent_high >= high_52w * 0.995     # breakout printed the 52w high
    if not (broke and made_new_high):
        return empty

    # --- Pullback that is now retesting the breakout level ---
    pulled_back = price <= recent_high * 0.98           # off the breakout peak
    dist_to_level = (price - resistance) / resistance * 100
    in_zone = -undercut <= dist_to_level <= retest_band  # back near old resistance
    if not (pulled_back and in_zone):
        return empty

    retest_depth = (recent_high - price) / recent_high * 100

    # Volume: breakout day heavier than the recent pullback (support test light).
    v_break = float(vol.iloc[-base_end:-3].mean()) if len(vol) > base_end else 0.0
    v_now = float(vol.iloc[-3:].mean())
    vol_light = (v_now < v_break) if v_break else False

    # --- Confidence ---
    tightness = 1 - _clamp(abs(dist_to_level) / retest_band, 0, 1)  # hugging the level
    conf = 45 + tightness * 30
    conf += _clamp((prox_pct + pct_from_high) / prox_pct, 0, 1) * 12  # near the highs
    conf += 8 if vol_light else 0
    conf += 5 if 2.0 <= retest_depth <= 9.0 else 0                    # healthy pullback

    note = (
        f"broke ${resistance:.2f} base top → 52w high ${high_52w:.2f}; "
        f"retesting at ${price:.2f} ({dist_to_level:+.1f}% vs level, "
        f"{retest_depth:.1f}% off high)"
        + (", volume light" if vol_light else "")
    )
    return {
        "confidence": round(_clamp(conf), 1),
        "price": round(price, 2),
        "high_52w": round(high_52w, 2),
        "pct_from_high": round(pct_from_high, 1),
        "breakout_level": round(resistance, 2),
        "dist_to_level_pct": round(dist_to_level, 1),
        "retest_depth_pct": round(retest_depth, 1),
        "vol_light": vol_light,
        "note": note,
    }


def detect_patterns(df: pd.DataFrame) -> dict:
    """Run all detectors on an OHLCV frame and return the ones that fire.

    Returns::

        {
          "primary": "🏆 Cup & Handle",      # highest-confidence (or "")
          "labels": "Cup & Handle, VCP",      # comma list for tables
          "all": [{"name", "confidence", "note"}, ...],  # sorted desc
        }
    """
    if df is None or df.empty:
        return {"primary": "", "labels": "", "all": []}
    hits = []
    for name, fn in _DETECTORS.items():
        try:
            conf, note = fn(df)
        except Exception:
            conf, note = 0.0, ""
        if conf >= MIN_CONFIDENCE:
            hits.append({"name": name, "confidence": round(conf, 1), "note": note})
    hits.sort(key=lambda h: h["confidence"], reverse=True)
    return {
        "primary": hits[0]["name"] if hits else "",
        "labels": ", ".join(h["name"] for h in hits),
        "all": hits,
    }

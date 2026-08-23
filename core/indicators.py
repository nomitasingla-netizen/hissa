"""Technical indicator implementations using pure pandas / numpy.

These are dependency-light (no TA-Lib) so the app installs cleanly on Windows.
All functions take a pandas Series/DataFrame of OHLCV data and return Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line, histogram)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, mult: float = 3.0):
    """ATR Supertrend trailing stop.

    Returns ``(line, direction)`` as Series:
      * ``direction``  +1 = uptrend (line is support *below* price),
                       -1 = downtrend (line is resistance *above* price).
      * ``line``       the Supertrend value — in an uptrend it is the final
                       lower band, in a downtrend the final upper band. This is
                       exactly the close price that would trigger a flip, so
                       ``|close - line| / ATR`` is the distance-to-reversal in
                       ATRs.

    Standard algorithm: basic bands ``hl2 ± mult*ATR`` are carried forward into
    'final' bands that only ratchet in the trend's favour; direction flips when
    close crosses the opposing final band.
    """
    atr_ = atr(high, low, close, period)
    hl2 = (high + low) / 2.0
    upper = (hl2 + mult * atr_).to_numpy()
    lower = (hl2 - mult * atr_).to_numpy()
    c = close.to_numpy()
    n = len(c)

    fu = np.full(n, np.nan)   # final upper band
    fl = np.full(n, np.nan)   # final lower band
    dir_ = np.full(n, 1.0)

    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if i == 0 or np.isnan(fu[i - 1]) or np.isnan(fl[i - 1]):
            fu[i], fl[i], dir_[i] = upper[i], lower[i], 1.0
            continue
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if dir_[i - 1] > 0:
            dir_[i] = -1.0 if c[i] < fl[i] else 1.0
        else:
            dir_[i] = 1.0 if c[i] > fu[i] else -1.0

    line = np.where(dir_ > 0, fl, fu)
    return (pd.Series(line, index=close.index),
            pd.Series(dir_, index=close.index))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """Return (adx, plus_di, minus_di) using Wilder's smoothing."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """Return (mid, upper, lower, bandwidth_pct)."""
    mid = sma(close, window)
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower) / mid * 100
    return mid, upper, lower, bandwidth


def bandwidth_percentile(bandwidth: pd.Series, lookback: int = 120) -> float:
    """How tight is current Bollinger bandwidth vs recent history (0-100).

    Low percentile => tight consolidation (a 'squeeze').
    """
    recent = bandwidth.dropna().tail(lookback)
    if len(recent) < 10:
        return 50.0
    current = recent.iloc[-1]
    return float((recent < current).mean() * 100)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative volume added on up days, subtracted on
    down days. A rising OBV means volume is flowing in (accumulation); a falling
    OBV while price holds up warns of distribution."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def anchored_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series, anchor_idx: int) -> pd.Series:
    """Anchored VWAP — the volume-weighted average price accumulated from a
    fixed *anchor* bar (``anchor_idx``, a positional index) to the end of the
    series, using the typical price (H+L+C)/3.

    Anchored at the start / swing-low of a base, it is the average price every
    buyer since that anchor has paid: while price holds **above** the anchored
    VWAP those base-buyers are in profit and in control (bullish), and losing it
    means the base is failing. The last value is the current AVWAP.
    """
    anchor_idx = max(0, min(anchor_idx, len(close) - 1))
    tp = (high + low + close) / 3.0
    tp = tp.iloc[anchor_idx:]
    vol = volume.iloc[anchor_idx:]
    cum_vol = vol.cumsum().replace(0, np.nan)
    cum_pv = (tp * vol).cumsum()
    return cum_pv / cum_vol


def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
        window: int = 20) -> pd.Series:
    """Chaikin Money Flow (−1..+1). Positive = buying pressure (money in),
    negative = selling pressure (money leaving)."""
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng          # money-flow multiplier
    mfv = mfm.fillna(0.0) * volume                         # money-flow volume
    denom = volume.rolling(window).sum().replace(0, np.nan)
    return (mfv.rolling(window).sum() / denom).fillna(0.0)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
        period: int = 14) -> pd.Series:
    """Money Flow Index (0..100) — a volume-weighted RSI. Falling MFI shows
    money rotating out even if price is still holding."""
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    pos = rmf.where(tp.diff() > 0, 0.0)
    neg = rmf.where(tp.diff() < 0, 0.0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum().replace(0, np.nan)
    ratio = pos_sum / neg_sum
    return (100 - 100 / (1 + ratio)).fillna(50.0)


def relative_strength(close: pd.Series, bench_close: pd.Series, lookback: int = 60) -> float:
    """Ratio-line slope of ETF vs benchmark over lookback (percent).

    Positive => the ETF is outperforming its benchmark.
    """
    df = pd.concat([close, bench_close], axis=1).dropna()
    if len(df) < lookback + 1:
        lookback = max(5, len(df) - 1)
    if len(df) < 6:
        return 0.0
    ratio = df.iloc[:, 0] / df.iloc[:, 1]
    past = ratio.iloc[-lookback - 1]
    now = ratio.iloc[-1]
    if past == 0 or np.isnan(past):
        return 0.0
    return float((now / past - 1) * 100)

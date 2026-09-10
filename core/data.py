"""Market data fetching via yfinance with 4h and 1d timeframes.

yfinance does not offer a native 4h interval, so we download 1h bars and
resample to 4h. Results are cached by Streamlit at the call site.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

# yfinance limits intraday (<=1h) history to ~730 days.
_INTRADAY_PERIOD = "180d"
_DAILY_PERIOD = "2y"
# Weekly uses a long window so the 200-week moving average has enough history.
_WEEKLY_PERIOD = "10y"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance can return MultiIndex columns for a single ticker; flatten them."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a missing price. yfinance often appends a placeholder row
    for the current/most-recent session (common on NSE ``.NS`` tickers) that has
    a Volume but NaN OHLC — leaving it in poisons every indicator downstream."""
    if df is None or df.empty:
        return df
    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    if not price_cols:
        return df
    return df.dropna(subset=price_cols)


def _resample_intraday(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample flattened 1h OHLCV bars to a coarser intraday bar (e.g. 2h/4h)."""
    if df_1h is None or df_1h.empty:
        return df_1h
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    cols = {c: agg[c] for c in df_1h.columns if c in agg}
    return df_1h.resample(rule).agg(cols).dropna(how="any")


def _resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    return _resample_intraday(df_1h, "4h")


def fetch_ohlcv(ticker: str) -> dict[str, pd.DataFrame]:
    """Return {'1h','2h','4h','1d','1wk'} of OHLCV data for a ticker.

    yfinance has no native 2h/4h interval, so 1h bars are downloaded once and
    resampled to 2h and 4h. Empty DataFrames are returned for any timeframe
    that fails to download.
    """
    result: dict[str, pd.DataFrame] = {
        "1h": pd.DataFrame(), "2h": pd.DataFrame(), "4h": pd.DataFrame(),
        "1d": pd.DataFrame(), "1wk": pd.DataFrame(),
    }

    try:
        daily = yf.download(
            ticker, period=_DAILY_PERIOD, interval="1d",
            auto_adjust=True, progress=False, threads=False,
        )
        result["1d"] = _drop_incomplete(_flatten(daily))
    except Exception:
        pass

    try:
        hourly = yf.download(
            ticker, period=_INTRADAY_PERIOD, interval="1h",
            auto_adjust=True, progress=False, threads=False,
        )
        hourly = _drop_incomplete(_flatten(hourly))
        result["1h"] = hourly
        result["2h"] = _resample_intraday(hourly, "2h")
        result["4h"] = _resample_intraday(hourly, "4h")
    except Exception:
        pass

    try:
        weekly = yf.download(
            ticker, period=_WEEKLY_PERIOD, interval="1wk",
            auto_adjust=True, progress=False, threads=False,
        )
        result["1wk"] = _drop_incomplete(_flatten(weekly))
    except Exception:
        pass

    return result


def fetch_daily_ohlcv(ticker: str) -> pd.DataFrame:
    """Fetch daily OHLCV only for lightweight universe-level screens."""
    try:
        daily = yf.download(
            ticker, period=_DAILY_PERIOD, interval="1d",
            auto_adjust=True, progress=False, threads=False,
        )
        return _drop_incomplete(_flatten(daily))
    except Exception:
        return pd.DataFrame()


def get_fund_info(ticker: str) -> dict:
    """Return fund 'wealth' metrics: AUM (net/total assets) and currency.

    Uses yfinance ``.info`` which is slower and best-effort — missing values
    return None. Callers should cache this heavily.
    """
    info = {}
    try:
        raw = yf.Ticker(ticker).info or {}
        aum = raw.get("totalAssets") or raw.get("netAssets")
        info = {
            "aum": float(aum) if aum else None,
            "currency": raw.get("currency"),
            "name": raw.get("longName") or raw.get("shortName"),
        }
    except Exception:
        info = {"aum": None, "currency": None, "name": None}
    return info


def truncate_frames(frames: dict, as_of) -> dict:
    """Return copies of the frames keeping only bars on/before ``as_of``.

    ``as_of`` is a ``datetime.date``. Used for 'as-of' backtesting so scores
    reflect only information available up to that day. Timezone-aware indexes
    are compared naively (date only).
    """
    if as_of is None:
        return frames
    cutoff = pd.Timestamp(as_of) + pd.Timedelta(hours=23, minutes=59)
    out: dict = {}
    for key, df in frames.items():
        if df is None or df.empty:
            out[key] = df
            continue
        idx = df.index
        try:
            naive = idx.tz_localize(None) if idx.tz is not None else idx
        except (TypeError, AttributeError):
            naive = idx
        out[key] = df[naive <= cutoff]
    return out

"""Market data fetching via yfinance with 4h and 1d timeframes.

yfinance does not offer a native 4h interval, so we download 1h bars and
resample to 4h. Results are cached by Streamlit at the call site.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

# yfinance limits intraday (<=1h) history to ~730 days.
_INTRADAY_PERIOD = "180d"
_ENTRY_INTRADAY_PERIOD = "60d"
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


def _latest_session_date(df: pd.DataFrame):
    """Return the date of the latest intraday session, preserving local exchange time."""
    if df is None or df.empty:
        return None
    stamp = pd.Timestamp(df.index[-1])
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _refresh_intraday_quote(hourly: pd.DataFrame, quote: dict) -> pd.DataFrame:
    """Overlay a latest quote onto the final hourly bar when available."""
    if hourly is None or hourly.empty or not quote.get("last_price"):
        return hourly
    refreshed = hourly.copy()
    idx = refreshed.index[-1]
    last = quote["last_price"]
    refreshed.loc[idx, "Close"] = last
    if quote.get("day_high") is not None:
        refreshed.loc[idx, "High"] = max(
            float(refreshed.loc[idx, "High"]), quote["day_high"], last)
    if quote.get("day_low") is not None:
        refreshed.loc[idx, "Low"] = min(
            float(refreshed.loc[idx, "Low"]), quote["day_low"], last)
    refreshed.attrs["latest_quote_applied"] = True
    return refreshed


def _merge_intraday_daily(daily: pd.DataFrame, hourly: pd.DataFrame,
                          quote: dict) -> pd.DataFrame:
    """Upsert the latest intraday session into daily OHLCV data."""
    if hourly is None or hourly.empty:
        return daily
    session_day = _latest_session_date(hourly)
    if session_day is None:
        return daily

    intraday_dates = pd.DatetimeIndex(hourly.index)
    if intraday_dates.tz is not None:
        mask = intraday_dates.tz_localize(None).normalize() == session_day
    else:
        mask = intraday_dates.normalize() == session_day
    session = hourly.loc[mask]
    if session.empty:
        return daily

    bar = {
        "Open": float(session["Open"].iloc[0]),
        "High": float(session["High"].max()),
        "Low": float(session["Low"].min()),
        "Close": float(quote.get("last_price") or session["Close"].iloc[-1]),
        "Volume": float(quote.get("last_volume") or session["Volume"].sum()),
    }
    if quote.get("day_high") is not None:
        bar["High"] = max(bar["High"], quote["day_high"], bar["Close"])
    if quote.get("day_low") is not None:
        bar["Low"] = min(bar["Low"], quote["day_low"], bar["Close"])
    if quote.get("open") is not None:
        bar["Open"] = quote["open"]

    merged = daily.copy()
    for column in merged.columns:
        if column not in bar:
            bar[column] = pd.NA
    merged.loc[session_day, list(bar)] = pd.Series(bar)
    merged = merged.sort_index()
    merged.attrs["latest_session_date"] = session_day.strftime("%Y-%m-%d")
    merged.attrs["latest_quote_applied"] = bool(quote.get("last_price"))
    return merged


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
    quote = fetch_latest_quote(ticker)

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
        hourly = _refresh_intraday_quote(hourly, quote)
        result["1h"] = hourly
        result["2h"] = _resample_intraday(hourly, "2h")
        result["4h"] = _resample_intraday(hourly, "4h")
        result["1d"] = _merge_intraday_daily(result["1d"], hourly, quote)
        session_day = _latest_session_date(hourly)
        for key in ("1h", "2h", "4h", "1d"):
            frame = result[key]
            if frame is None or frame.empty or session_day is None:
                continue
            latest_day = _latest_session_date(frame)
            frame.attrs["is_current_session"] = latest_day == session_day
    except Exception:
        pass

    try:
        weekly = yf.download(
            ticker, period=_WEEKLY_PERIOD, interval="1wk",
            auto_adjust=True, progress=False, threads=False,
        )
        weekly = _drop_incomplete(_flatten(weekly))
        if not weekly.empty:
            quote_price = quote.get("last_price")
            weekly_close = float(weekly["Close"].iloc[-1])
            weekly.attrs["is_current_session"] = bool(
                quote_price and abs(weekly_close / quote_price - 1) < 0.002)
        result["1wk"] = weekly
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


def fetch_30m_ohlcv(ticker: str) -> pd.DataFrame:
    """Fetch 30-minute OHLCV for short-term Bollinger entry levels."""
    try:
        intraday = yf.download(
            ticker, period=_ENTRY_INTRADAY_PERIOD, interval="30m",
            auto_adjust=True, progress=False, threads=False,
        )
        return _drop_incomplete(_flatten(intraday))
    except Exception:
        return pd.DataFrame()


def fetch_latest_quote(ticker: str) -> dict:
    """Return Yahoo's latest quote and prior regular-session close, if available."""
    try:
        info = yf.Ticker(ticker).fast_info
        last = info.get("lastPrice")
        previous = (info.get("regularMarketPreviousClose")
                    or info.get("previousClose"))
        return {
            "last_price": float(last) if last is not None else None,
            "previous_close": float(previous) if previous is not None else None,
            "open": float(info["open"]) if info.get("open") is not None else None,
            "day_high": float(info["dayHigh"]) if info.get("dayHigh") is not None else None,
            "day_low": float(info["dayLow"]) if info.get("dayLow") is not None else None,
            "last_volume": (
                float(info["lastVolume"]) if info.get("lastVolume") is not None else None),
        }
    except Exception:
        return {
            "last_price": None, "previous_close": None, "open": None,
            "day_high": None, "day_low": None, "last_volume": None,
        }


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

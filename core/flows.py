"""Institutional money-flow data from NSE (India).

Three public NSE sources are used, none of which need an API key:

* **FII/DII cash-market net** — daily provisional buy/sell/net (₹ cr), market-wide.
* **Bulk & Block deals** — every large trade with the *client name* (fund house /
  HNI / broker), buy/sell side, quantity and weighted-avg price → shows *who*
  added or trimmed money and roughly how much (₹).
* **Delivery %** — from the daily full bhavcopy; a high delivery ratio on an up
  move signals genuine accumulation rather than intraday churn.

The engine primes an ``nseindia.com`` session (the API rejects cookie-less
requests) and is dependency-light (requests + pandas). All functions raise
``FlowError`` on failure so the UI can degrade gracefully.
"""
from __future__ import annotations

import datetime as _dt
import io

import pandas as pd
import requests

NSE_HOME = "https://www.nseindia.com"
ARCHIVES = "https://archives.nseindia.com"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{NSE_HOME}/",
}


class FlowError(RuntimeError):
    """Raised when NSE data can't be fetched (blocked, offline, holiday, etc.)."""


def _session() -> requests.Session:
    """A requests session pre-loaded with NSE cookies. NSE returns 401/403 for
    cookie-less API calls, so we warm up on the homepage first."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(NSE_HOME, timeout=12)
        s.get(f"{NSE_HOME}/market-data/live-equity-market", timeout=12)
    except requests.RequestException as exc:  # pragma: no cover - network
        raise FlowError(f"Couldn't reach nseindia.com: {exc}") from exc
    return s


def _get_json(s: requests.Session, url: str) -> dict | list:
    try:
        r = s.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as exc:
        raise FlowError(f"NSE request failed ({url}): {exc}") from exc


def _to_num(x) -> float | None:
    if x is None:
        return None
    try:
        return float(str(x).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


# --------------------------------------------------------------------------- #
# 1. FII / DII cash-market net (market-wide, daily provisional)
# --------------------------------------------------------------------------- #
def fii_dii() -> pd.DataFrame:
    """Latest FII/FPI and DII cash-market figures (₹ crore).

    Columns: Category, Date, Buy ₹cr, Sell ₹cr, Net ₹cr.
    """
    s = _session()
    data = _get_json(s, f"{NSE_HOME}/api/fiidiiTradeReact")
    if not isinstance(data, list) or not data:
        raise FlowError("NSE returned no FII/DII data (market holiday?).")
    rows = []
    for d in data:
        rows.append({
            "Category": d.get("category", "").replace("/FPI", " / FPI"),
            "Date": d.get("date"),
            "Buy ₹cr": _to_num(d.get("buyValue")),
            "Sell ₹cr": _to_num(d.get("sellValue")),
            "Net ₹cr": _to_num(d.get("netValue")),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Bulk & Block deals (who bought/sold, and ~how much in ₹)
# --------------------------------------------------------------------------- #
def _deals_frame(records: list) -> pd.DataFrame:
    rows = []
    for d in records or []:
        qty = _to_num(d.get("qty"))
        watp = _to_num(d.get("watp"))
        value_cr = round(qty * watp / 1e7, 2) if (qty and watp) else None
        rows.append({
            "Date": d.get("date"),
            "Symbol": d.get("symbol"),
            "Name": d.get("name"),
            "Client (fund/HNI/broker)": (d.get("clientName") or "").strip(),
            "Side": d.get("buySell"),
            "Qty": int(qty) if qty else None,
            "Avg price": watp,
            "Value ₹cr": value_cr,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["Side"].notna()]
    return df


def large_deals() -> dict[str, pd.DataFrame]:
    """Today's bulk and block deals as {'bulk': df, 'block': df, 'as_on': str}."""
    s = _session()
    # Prime the large-deals page so the API accepts the follow-up call.
    try:
        s.get(f"{NSE_HOME}/market-data/large-deals", timeout=12)
    except requests.RequestException:
        pass
    data = _get_json(s, f"{NSE_HOME}/api/snapshot-capital-market-largedeal")
    if not isinstance(data, dict):
        raise FlowError("NSE returned no large-deals data.")
    return {
        "bulk": _deals_frame(data.get("BULK_DEALS_DATA")),
        "block": _deals_frame(data.get("BLOCK_DEALS_DATA")),
        "as_on": data.get("as_on_date", ""),
    }


def net_by_client(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Aggregate deals into net ₹cr per client (BUY positive, SELL negative)."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["Signed ₹cr"] = d.apply(
        lambda r: (r["Value ₹cr"] or 0) * (1 if str(r["Side"]).upper() == "BUY" else -1),
        axis=1,
    )
    agg = (d.groupby("Client (fund/HNI/broker)")
             .agg(**{"Net ₹cr": ("Signed ₹cr", "sum"),
                     "Deals": ("Symbol", "count")})
             .reset_index()
             .sort_values("Net ₹cr", key=lambda s: s.abs(), ascending=False)
             .head(top))
    agg["Net ₹cr"] = agg["Net ₹cr"].round(2)
    return agg.reset_index(drop=True)


def net_by_symbol(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Aggregate deals into net ₹cr per stock (BUY positive, SELL negative)."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d["Signed ₹cr"] = d.apply(
        lambda r: (r["Value ₹cr"] or 0) * (1 if str(r["Side"]).upper() == "BUY" else -1),
        axis=1,
    )
    agg = (d.groupby(["Symbol", "Name"])
             .agg(**{"Net ₹cr": ("Signed ₹cr", "sum"),
                     "Deals": ("Client (fund/HNI/broker)", "count")})
             .reset_index()
             .sort_values("Net ₹cr", key=lambda s: s.abs(), ascending=False)
             .head(top))
    agg["Net ₹cr"] = agg["Net ₹cr"].round(2)
    return agg.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Delivery % (accumulation-conviction proxy) — daily full bhavcopy
# --------------------------------------------------------------------------- #
def delivery_data(max_lookback: int = 7) -> tuple[pd.DataFrame, str]:
    """Latest available full bhavcopy with delivery %. Walks back up to
    ``max_lookback`` days (weekends/holidays) until a file is found.

    Returns (DataFrame[SYMBOL, SERIES, CLOSE, TTL_TRD_QNTY, DELIV_QTY, DELIV_PER],
    date_str). Only EQ-series rows are kept.
    """
    s = _session()
    today = _dt.date.today()
    last_err = None
    for i in range(max_lookback + 1):
        d = today - _dt.timedelta(days=i)
        ds = d.strftime("%d%m%Y")
        url = f"{ARCHIVES}/products/content/sec_bhavdata_full_{ds}.csv"
        try:
            r = s.get(url, timeout=15)
            if r.status_code != 200 or not r.text.startswith("SYMBOL"):
                continue
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
            df["SERIES"] = df["SERIES"].astype(str).str.strip()
            df = df[df["SERIES"] == "EQ"].copy()
            for col in ("DELIV_PER", "DELIV_QTY", "TTL_TRD_QNTY", "CLOSE_PRICE"):
                if col in df:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            keep = ["SYMBOL", "SERIES", "CLOSE_PRICE", "TTL_TRD_QNTY",
                    "DELIV_QTY", "DELIV_PER"]
            out = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
            return out, d.strftime("%d-%b-%Y")
        except requests.RequestException as exc:  # pragma: no cover - network
            last_err = exc
            continue
    raise FlowError(
        f"No NSE bhavcopy found in the last {max_lookback} days"
        + (f" ({last_err})" if last_err else "")
    )


def delivery_for(symbols, max_lookback: int = 7) -> pd.DataFrame:
    """Delivery rows for the given NSE symbols (case-insensitive)."""
    df, date_str = delivery_data(max_lookback)
    wanted = {str(x).upper().replace(".NS", "").strip() for x in symbols}
    out = df[df["SYMBOL"].str.upper().isin(wanted)].copy()
    out.insert(0, "Date", date_str)
    return out.reset_index(drop=True)

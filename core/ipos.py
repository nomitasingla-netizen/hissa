"""Recent NSE IPOs, fetched from a reliable source (NSE official website).

Source: NSE India public API ``/api/public-past-issues`` — the exchange's own
list of past public issues, including the trading ``symbol``, ``company`` name,
``issuePrice`` (the IPO launch/issue price) and ``listingDate``. Accessing NSE
APIs requires priming a browser-like session (a homepage GET to obtain cookies),
which this module handles.

The tab uses this to find IPOs listed in the **last ~1 year** whose current
market price is close (±band%) to the launch price. Current prices come from
yfinance (``SYMBOL.NS``).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import requests

NSE_HOME = "https://www.nseindia.com"
NSE_PAST_ISSUES = "https://www.nseindia.com/api/public-past-issues"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/all-upcoming-issues-ipo",
}


def _parse_date(text: str):
    if not text:
        return None
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(text, dayfirst=True).to_pydatetime()
    except Exception:
        return None


def _parse_price(text) -> float:
    """Parse an issue price that may contain spaces or a 'lo-hi' range."""
    if text is None:
        return 0.0
    s = str(text).strip().replace(",", "")
    if not s:
        return 0.0
    # Range like "100 - 110" or "100to110" -> take the upper (cut-off) bound.
    for sep in ("-", "to", "–"):
        if sep in s:
            parts = [p.strip() for p in s.replace("to", "-").split("-") if p.strip()]
            nums = []
            for p in parts:
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            if nums:
                return max(nums)
    try:
        return float(s)
    except ValueError:
        return 0.0


def fetch_recent_ipos(within_days: int = 365) -> list[dict]:
    """Fetch NSE past issues and return equity IPOs listed within ``within_days``.

    Each item: {symbol, company, issue_price, listing_date (date), days_since}.
    Raises on network/parse failure so the caller can surface the error.
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    # Prime cookies from the homepage, then hit the JSON API.
    session.get(NSE_HOME, timeout=20)
    resp = session.get(NSE_PAST_ISSUES, timeout=20)
    resp.raise_for_status()
    rows = resp.json()

    cutoff = datetime.now() - timedelta(days=within_days)
    out: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        symbol = (row.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        # Keep only equity IPOs (mainboard EQ/BE and SME); skip bonds/NCDs/other.
        sec_type = (row.get("securityType") or "").strip().upper()
        if sec_type not in ("EQ", "BE", "SME"):
            continue
        listing = _parse_date(row.get("listingDate"))
        if listing is None or listing < cutoff or listing > datetime.now():
            continue
        price = _parse_price(row.get("issuePrice") or row.get("priceRange"))
        if price <= 0:
            continue
        seen.add(symbol)
        out.append(
            {
                "symbol": symbol,
                "company": (row.get("company") or symbol).strip(),
                "issue_price": round(price, 2),
                "listing_date": listing.date(),
                "days_since": (datetime.now() - listing).days,
                "board": "SME" if sec_type == "SME" else "Mainboard",
            }
        )
    out.sort(key=lambda r: r["listing_date"], reverse=True)
    return out


def analyze_cup(
    closes: "pd.Series",
    launch_price: float,
    dip_min: float = 20.0,
    dip_max: float = 40.0,
    recover_band: float = 5.0,
) -> dict | None:
    """Describe the **post-IPO cup**: after listing the price fell to a trough and
    then recovered back toward the launch (issue) price — the first cup / cup-with-
    handle in the timeline that signals the market re-rating a fundamentally-growing
    company back to (or above) its debut value.

    ``closes`` are daily closing prices *since listing*; ``launch_price`` is the NSE
    issue price. ``dip_min``/``dip_max`` bound the ideal drawdown depth used for
    confidence shaping; ``recover_band`` is how close (±%) to launch counts as "back
    at the rim". Returns a descriptor dict, or ``None`` if there isn't enough data.
    """
    if launch_price <= 0:
        return None
    closes = closes.dropna()
    if len(closes) < 20:
        return None

    n = len(closes)
    current = float(closes.iloc[-1])
    trough_date = closes.idxmin()
    trough_price = float(closes.min())
    trough_pos = int(closes.index.get_loc(trough_date))
    trough_frac = trough_pos / max(1, n - 1)

    depth_pct = (trough_price / launch_price - 1) * 100          # negative = drawdown
    now_vs_launch = (current / launch_price - 1) * 100
    rebound_pct = (current / trough_price - 1) * 100 if trough_price > 0 else 0.0

    # Right side of the cup: highest close after the trough is the "rim".
    post = closes.iloc[trough_pos:]
    rim_price = float(post.max())
    rim_date = post.idxmax()
    rim_pos = int(closes.index.get_loc(rim_date))
    handle_depth = (current / rim_price - 1) * 100 if rim_price > 0 else 0.0
    has_handle = (
        rim_pos < n - 2                       # rim formed a few bars back, not today
        and -12.0 <= handle_depth <= -1.5     # shallow pullback off the rim
        and rim_price >= launch_price * 0.90  # rim recovered near/above launch
    )
    shape = "Cup w/ Handle" if has_handle else "Cup"

    # ---- confidence (0-100) ----
    drop = -depth_pct  # positive magnitude of the dip
    if dip_min <= drop <= dip_max:
        depth_s = 1.0
    elif drop < dip_min:
        depth_s = max(0.0, drop / dip_min) if dip_min else 0.0
    else:
        depth_s = max(0.0, 1 - (drop - dip_max) / 40.0)
    recov_s = max(0.0, 1 - abs(now_vs_launch) / max(recover_band * 3, 1.0))
    sym_s = max(0.0, 1 - abs(trough_frac - 0.5) / 0.5)  # trough near mid = symmetric
    reb_s = min(1.0, max(0.0, rebound_pct / drop)) if drop > 0 else 0.0
    conf = 100 * (0.34 * depth_s + 0.30 * recov_s + 0.18 * reb_s + 0.18 * sym_s)

    return {
        "current": round(current, 2),
        "now_vs_launch": round(now_vs_launch, 1),
        "trough_price": round(trough_price, 2),
        "trough_date": trough_date.date() if hasattr(trough_date, "date") else trough_date,
        "depth_pct": round(depth_pct, 1),
        "rebound_pct": round(rebound_pct, 1),
        "rim_price": round(rim_price, 2),
        "handle_depth": round(handle_depth, 1),
        "shape": shape,
        "trough_frac": round(trough_frac, 2),
        "confidence": round(conf, 1),
    }

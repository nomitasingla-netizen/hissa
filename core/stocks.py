"""Large-cap stock universes for the 52-week-high breakout-retest scan.

Loaded from editable JSON config files (auto-seeded from the defaults below on
first run), exactly like ``core/etfs.py``:

    config/stocks_us.json      -> {"stocks": {"AAPL": "Apple", ...}}
    config/stocks_india.json   -> {"stocks": {"RELIANCE.NS": "Reliance", ...}}

Edit those files (or use the app's reload button) to add/remove tickers — up to
the "top 250" of each market. Indian symbols use the ``.NS`` (NSE) suffix.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

STOCK_LOAD_ERRORS: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Built-in defaults (seed the config files; editable afterward). These are
# well-known large caps — expand each toward the top 250 in the JSON files.
# ---------------------------------------------------------------------------
DEFAULT_US_STOCKS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet A", "AMZN": "Amazon",
    "NVDA": "NVIDIA", "META": "Meta", "TSLA": "Tesla", "AVGO": "Broadcom",
    "BRK-B": "Berkshire B", "JPM": "JPMorgan", "V": "Visa", "MA": "Mastercard",
    "UNH": "UnitedHealth", "XOM": "Exxon", "LLY": "Eli Lilly", "JNJ": "J&J",
    "PG": "Procter & Gamble", "HD": "Home Depot", "MRK": "Merck", "ABBV": "AbbVie",
    "COST": "Costco", "PEP": "PepsiCo", "KO": "Coca-Cola", "ADBE": "Adobe",
    "CRM": "Salesforce", "WMT": "Walmart", "BAC": "Bank of America", "NFLX": "Netflix",
    "AMD": "AMD", "CVX": "Chevron", "ACN": "Accenture", "TMO": "Thermo Fisher",
    "MCD": "McDonald's", "ABT": "Abbott", "DHR": "Danaher", "LIN": "Linde",
    "TXN": "Texas Instruments", "INTC": "Intel", "QCOM": "Qualcomm", "ORCL": "Oracle",
    "CSCO": "Cisco", "WFC": "Wells Fargo", "DIS": "Disney", "INTU": "Intuit",
    "IBM": "IBM", "GE": "GE", "CAT": "Caterpillar", "AMAT": "Applied Materials",
    "NOW": "ServiceNow", "PFE": "Pfizer", "GS": "Goldman Sachs", "HON": "Honeywell",
    "UNP": "Union Pacific", "MS": "Morgan Stanley", "RTX": "RTX", "LOW": "Lowe's",
    "SPGI": "S&P Global", "BKNG": "Booking", "PLD": "Prologis", "ISRG": "Intuitive Surgical",
    "T": "AT&T", "VZ": "Verizon", "BLK": "BlackRock", "ELV": "Elevance",
    "SBUX": "Starbucks", "MDT": "Medtronic", "DE": "Deere", "ADI": "Analog Devices",
    "LRCX": "Lam Research", "MU": "Micron", "PANW": "Palo Alto", "KLAC": "KLA",
    "SNPS": "Synopsys", "CDNS": "Cadence", "REGN": "Regeneron", "VRTX": "Vertex",
    "UBER": "Uber", "BA": "Boeing", "NKE": "Nike", "PM": "Philip Morris",
}

DEFAULT_INDIA_STOCKS = {
    "RELIANCE.NS": "Reliance", "TCS.NS": "TCS", "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank", "INFY.NS": "Infosys", "HINDUNILVR.NS": "HUL",
    "ITC.NS": "ITC", "SBIN.NS": "SBI", "BHARTIARTL.NS": "Bharti Airtel",
    "BAJFINANCE.NS": "Bajaj Finance", "KOTAKBANK.NS": "Kotak Bank", "LT.NS": "L&T",
    "HCLTECH.NS": "HCL Tech", "ASIANPAINT.NS": "Asian Paints", "AXISBANK.NS": "Axis Bank",
    "MARUTI.NS": "Maruti", "SUNPHARMA.NS": "Sun Pharma", "TITAN.NS": "Titan",
    "ULTRACEMCO.NS": "UltraTech", "WIPRO.NS": "Wipro", "NESTLEIND.NS": "Nestle India",
    "ONGC.NS": "ONGC", "NTPC.NS": "NTPC", "POWERGRID.NS": "Power Grid",
    "TATAMOTORS.NS": "Tata Motors", "TATASTEEL.NS": "Tata Steel", "JSWSTEEL.NS": "JSW Steel",
    "ADANIENT.NS": "Adani Ent", "ADANIPORTS.NS": "Adani Ports", "COALINDIA.NS": "Coal India",
    "BAJAJFINSV.NS": "Bajaj Finserv", "HDFCLIFE.NS": "HDFC Life", "SBILIFE.NS": "SBI Life",
    "TECHM.NS": "Tech Mahindra", "GRASIM.NS": "Grasim", "DIVISLAB.NS": "Divi's Labs",
    "DRREDDY.NS": "Dr Reddy's", "CIPLA.NS": "Cipla", "EICHERMOT.NS": "Eicher",
    "HEROMOTOCO.NS": "Hero MotoCorp", "BAJAJ-AUTO.NS": "Bajaj Auto", "BRITANNIA.NS": "Britannia",
    "APOLLOHOSP.NS": "Apollo Hospitals", "INDUSINDBK.NS": "IndusInd Bank", "HINDALCO.NS": "Hindalco",
    "BPCL.NS": "BPCL", "IOC.NS": "IOC", "GAIL.NS": "GAIL", "DABUR.NS": "Dabur",
    "GODREJCP.NS": "Godrej Consumer", "PIDILITIND.NS": "Pidilite", "DMART.NS": "Avenue Supermarts",
    "SIEMENS.NS": "Siemens", "HAVELLS.NS": "Havells", "AMBUJACEM.NS": "Ambuja Cement",
    "SHREECEM.NS": "Shree Cement", "BANKBARODA.NS": "Bank of Baroda", "PNB.NS": "PNB",
    "VEDL.NS": "Vedanta", "TATAPOWER.NS": "Tata Power", "DLF.NS": "DLF",
    "LTIM.NS": "LTIMindtree", "TRENT.NS": "Trent", "BEL.NS": "Bharat Electronics",
    "COLPAL.NS": "Colgate India", "MARICO.NS": "Marico", "BERGEPAINT.NS": "Berger Paints",
    "MOTHERSON.NS": "Samvardhana Motherson", "TVSMOTOR.NS": "TVS Motor", "CANBK.NS": "Canara Bank",
    "NAUKRI.NS": "Info Edge", "BOSCHLTD.NS": "Bosch", "SRF.NS": "SRF", "CHOLAFIN.NS": "Cholamandalam",
}

_STOCK_DEFAULTS = {"US": DEFAULT_US_STOCKS, "India": DEFAULT_INDIA_STOCKS}
_STOCK_FILES = {"US": "stocks_us.json", "India": "stocks_india.json"}


def _path(market: str) -> Path:
    return CONFIG_DIR / _STOCK_FILES[market]


def _seed(market: str) -> None:
    path = _path(market)
    if path.exists():
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stocks": _STOCK_DEFAULTS[market]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _loads_lenient(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", text))


def load_stocks(market: str) -> dict:
    _seed(market)
    path = _path(market)
    STOCK_LOAD_ERRORS.pop(market, None)
    try:
        data = _loads_lenient(path.read_text(encoding="utf-8"))
        stocks = data.get("stocks") or {}
        if not isinstance(stocks, dict) or not stocks:
            stocks = dict(_STOCK_DEFAULTS[market])
        return stocks
    except (json.JSONDecodeError, OSError) as exc:
        STOCK_LOAD_ERRORS[market] = f"{_STOCK_FILES[market]}: {exc}"
        return dict(_STOCK_DEFAULTS[market])


def load_all_stocks() -> dict:
    return {market: load_stocks(market) for market in _STOCK_FILES}


def reload() -> dict:
    global STOCK_MARKETS
    STOCK_MARKETS = load_all_stocks()
    return STOCK_MARKETS


STOCK_MARKETS = load_all_stocks()

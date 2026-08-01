"""Sector ETF/stock universes for the US and Indian markets.

The universes are loaded from editable JSON config files under ``config/``:

    config/us.json      -> {"benchmark": "SPY",   "etfs": {"XLK": "Technology", ...}}
    config/india.json   -> {"benchmark": "^NSEI", "etfs": {"NIFTYBEES.NS": "...", ...}}

On first run these files are auto-created from the built-in defaults below, so you
can simply edit them afterward to add/remove tickers. Add stocks the same way
(e.g. "AAPL": "Apple" or "RELIANCE.NS": "Reliance"). Restart the app to pick up
changes. Indian symbols use the ``.NS`` (NSE) suffix required by yfinance.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# Config lives at <project root>/config
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Populated with a human-readable message if a config file fails to parse, so
# the UI can warn the user instead of silently falling back to defaults.
LOAD_ERRORS: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Built-in defaults (used to seed the config files on first run / as fallback).
# ---------------------------------------------------------------------------
DEFAULT_BENCHMARKS = {
    "US": "SPY",       # S&P 500
    "India": "^NSEI",  # NIFTY 50 index
}

DEFAULT_US_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
    "SMH": "Semiconductors",
    "XBI": "Biotech",
    "KRE": "Regional Banks",
    "ITB": "Home Builders",
    "GDX": "Gold Miners",
    "QQQ": "Nasdaq 100",
    "QQQM": "Nasdaq 100 (Mini)",
    "VGT": "Vanguard Info Tech",
    "VUG": "Vanguard Growth",
    "VO": "Vanguard Mid-Cap",
    "VB": "Vanguard Small-Cap",
}

DEFAULT_INDIA_ETFS = {
    "NIFTYBEES.NS": "Nifty 50 (Broad)",
    "JUNIORBEES.NS": "Nifty Next 50",
    "BANKBEES.NS": "Banking",
    "ITBEES.NS": "IT",
    "PHARMABEES.NS": "Pharma",
    "PSUBNKBEES.NS": "PSU Banks",
    "AUTOBEES.NS": "Auto",
    "FMCGIETF.NS": "FMCG",
    "METALIETF.NS": "Metals",
    "INFRABEES.NS": "Infrastructure",
    "CONSUMBEES.NS": "Consumption",
    "HEALTHIETF.NS": "Healthcare",
    "MOM100.NS": "Midcap 100",
    "MID150BEES.NS": "Midcap 150",
    "MOSMALL250.NS": "Smallcap 250",
    "CPSEETF.NS": "CPSE (PSU)",
    "MOREALTY.NS": "Realty",
}

_DEFAULTS = {
    "US": {"benchmark": DEFAULT_BENCHMARKS["US"], "etfs": DEFAULT_US_ETFS},
    "India": {"benchmark": DEFAULT_BENCHMARKS["India"], "etfs": DEFAULT_INDIA_ETFS},
}

# Config file name per market.
_CONFIG_FILES = {"US": "us.json", "India": "india.json"}


def _config_path(market: str) -> Path:
    return CONFIG_DIR / _CONFIG_FILES[market]


def _seed_config(market: str) -> None:
    """Write the default universe to disk if the config file doesn't exist."""
    path = _config_path(market)
    if path.exists():
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": _DEFAULTS[market]["benchmark"],
        "etfs": _DEFAULTS[market]["etfs"],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _loads_lenient(text: str) -> dict:
    """Parse JSON, tolerating trailing commas (a common hand-edit mistake)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", text)  # strip trailing commas
        return json.loads(cleaned)


def load_market(market: str) -> dict:
    """Load a market's universe from its config file, seeding/falling back to defaults."""
    _seed_config(market)
    path = _config_path(market)
    LOAD_ERRORS.pop(market, None)
    try:
        data = _loads_lenient(path.read_text(encoding="utf-8"))
        benchmark = data.get("benchmark") or _DEFAULTS[market]["benchmark"]
        etfs = data.get("etfs") or {}
        if not isinstance(etfs, dict) or not etfs:
            etfs = dict(_DEFAULTS[market]["etfs"])
        return {"benchmark": benchmark, "etfs": etfs}
    except (json.JSONDecodeError, OSError) as exc:
        # Record the error so the UI can warn, then use safe defaults.
        LOAD_ERRORS[market] = f"{_CONFIG_FILES[market]}: {exc}"
        return {
            "benchmark": _DEFAULTS[market]["benchmark"],
            "etfs": dict(_DEFAULTS[market]["etfs"]),
        }


def load_all() -> dict:
    return {market: load_market(market) for market in _CONFIG_FILES}


def reload() -> dict:
    """Re-read all config files from disk and refresh the module-level globals.

    Use this to pick up on-disk edits to config/us.json / config/india.json
    without restarting the Python process.
    """
    global MARKETS, BENCHMARKS, US_ETFS, INDIA_ETFS
    MARKETS = load_all()
    BENCHMARKS = {m: MARKETS[m]["benchmark"] for m in MARKETS}
    US_ETFS = MARKETS["US"]["etfs"]
    INDIA_ETFS = MARKETS["India"]["etfs"]
    return MARKETS


# Public API (loaded at import; call reload() after editing config files).
MARKETS = load_all()
BENCHMARKS = {m: MARKETS[m]["benchmark"] for m in MARKETS}
US_ETFS = MARKETS["US"]["etfs"]
INDIA_ETFS = MARKETS["India"]["etfs"]

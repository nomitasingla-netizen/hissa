"""Native price alerts with email delivery — a TradingView-free alternative.

Alerts and SMTP settings are persisted as small JSON files in the project root
(both are git-ignored). A price alert fires when the latest price crosses a
level in the chosen direction:

    * Buy trigger / Target  -> direction "above" (fires when price >= level)
    * Stop                  -> direction "below" (fires when price <= level)

The same functions power both the in-app UI (create / list / delete / check)
and the standalone ``alert_watcher.py`` background process, so alerts keep
working even when the dashboard/browser is closed.
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALERTS_FILE = os.path.join(_ROOT, "alerts.json")
CONFIG_FILE = os.path.join(_ROOT, "alert_config.json")


# --------------------------- persistence ---------------------------
def _read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def load_alerts() -> list[dict]:
    data = _read_json(ALERTS_FILE, [])
    return data if isinstance(data, list) else []


def save_alerts(alerts: list[dict]) -> None:
    _write_json(ALERTS_FILE, alerts)


def load_config() -> dict:
    """SMTP/email settings. Environment variables override the JSON file so
    secrets can be kept out of disk if preferred."""
    cfg = _read_json(CONFIG_FILE, {})
    if not isinstance(cfg, dict):
        cfg = {}
    env = {
        "host": os.getenv("ALERT_SMTP_HOST"),
        "port": os.getenv("ALERT_SMTP_PORT"),
        "username": os.getenv("ALERT_SMTP_USER"),
        "password": os.getenv("ALERT_SMTP_PASS"),
        "sender": os.getenv("ALERT_SMTP_FROM"),
        "to": os.getenv("ALERT_EMAIL_TO"),
        "use_ssl": os.getenv("ALERT_SMTP_SSL"),
    }
    for k, v in env.items():
        if v:
            cfg[k] = v
    cfg.setdefault("host", "smtp.gmail.com")
    cfg.setdefault("port", 465)
    cfg.setdefault("use_ssl", True)
    return cfg


def save_config(cfg: dict) -> None:
    _write_json(CONFIG_FILE, cfg)


def config_ready(cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return bool(cfg.get("host") and cfg.get("username")
               and cfg.get("password") and cfg.get("to"))


# --------------------------- alert CRUD ---------------------------
def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def make_alert(symbol: str, kind: str, level: float, tv: str = "",
               email: str = "", note: str = "") -> dict:
    kind = kind.lower()
    direction = "below" if kind == "stop" else "above"
    return {
        "id": f"{symbol}-{kind}-{int(time.time() * 1000)}",
        "symbol": symbol,
        "tv": tv or symbol,
        "kind": kind,
        "level": round(float(level), 4),
        "direction": direction,
        "email": email,
        "note": note or f"{symbol} {kind}",
        "status": "active",
        "created": _now(),
        "triggered_at": None,
        "triggered_price": None,
    }


def add_alerts(new_alerts: list[dict]) -> list[dict]:
    """Append alerts, skipping exact duplicates (same symbol/kind/level active)."""
    alerts = load_alerts()
    existing = {
        (a["symbol"], a["kind"], a["level"]) for a in alerts
        if a.get("status") == "active"
    }
    added = []
    for a in new_alerts:
        key = (a["symbol"], a["kind"], a["level"])
        if key in existing:
            continue
        alerts.append(a)
        existing.add(key)
        added.append(a)
    save_alerts(alerts)
    return added


def remove_alerts(ids: list[str]) -> None:
    ids = set(ids)
    save_alerts([a for a in load_alerts() if a.get("id") not in ids])


def clear_triggered() -> None:
    save_alerts([a for a in load_alerts() if a.get("status") != "triggered"])


# --------------------------- email ---------------------------
def send_email(cfg: dict, subject: str, body: str, to: str | None = None) -> None:
    to = to or cfg.get("to")
    sender = cfg.get("sender") or cfg.get("username")
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    host = cfg["host"]
    port = int(cfg.get("port", 465))
    use_ssl = str(cfg.get("use_ssl", True)).lower() in ("1", "true", "yes", "on") \
        if not isinstance(cfg.get("use_ssl"), bool) else cfg.get("use_ssl")

    if use_ssl:
        with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context()) as s:
            s.login(cfg["username"], cfg["password"])
            s.sendmail(sender, [to], msg.as_string())
    else:
        with smtplib.SMTP(host, port) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(cfg["username"], cfg["password"])
            s.sendmail(sender, [to], msg.as_string())


def send_test_email(cfg: dict) -> tuple[bool, str]:
    try:
        send_email(cfg, "✅ Sector Scanner alert test",
                   "This is a test email from your Sector Breakout Scanner. "
                   "If you received it, email alerts are configured correctly.")
        return True, f"Test email sent to {cfg.get('to')}."
    except Exception as exc:
        return False, f"Failed to send: {exc}"


# --------------------------- checking ---------------------------
def _default_price(symbol: str) -> float | None:
    from .data import fetch_ohlcv  # lazy import to keep this module light
    try:
        d = fetch_ohlcv(symbol).get("1d")
        if d is None or d.empty:
            return None
        return float(d["Close"].iloc[-1])
    except Exception:
        return None


def _crossed(direction: str, price: float, level: float) -> bool:
    return price >= level if direction == "above" else price <= level


def check_alerts(price_lookup=None, send: bool = True) -> list[dict]:
    """Check all active alerts against the latest price. Fires (and optionally
    emails) any that crossed, marks them triggered, and returns the fired list.
    ``price_lookup`` is a callable(symbol) -> price|None (defaults to yfinance).
    """
    price_lookup = price_lookup or _default_price
    alerts = load_alerts()
    cfg = load_config()
    can_email = send and config_ready(cfg)
    fired: list[dict] = []
    prices: dict[str, float | None] = {}

    for a in alerts:
        if a.get("status") != "active":
            continue
        sym = a["symbol"]
        if sym not in prices:
            prices[sym] = price_lookup(sym)
        price = prices[sym]
        if price is None:
            continue
        if _crossed(a["direction"], price, a["level"]):
            a["status"] = "triggered"
            a["triggered_at"] = _now()
            a["triggered_price"] = round(price, 4)
            fired.append(a)
            if can_email:
                arrow = "▲" if a["direction"] == "above" else "▼"
                subject = f"🔔 {a['tv']} {a['kind'].upper()} hit @ {price:.2f}"
                body = (
                    f"{a['note']}\n\n"
                    f"Symbol : {a['tv']} ({sym})\n"
                    f"Trigger: price {arrow} {a['level']}  ({a['kind']})\n"
                    f"Price  : {price:.2f}\n"
                    f"Time   : {a['triggered_at']}\n"
                )
                try:
                    send_email(cfg, subject, body, to=a.get("email") or cfg.get("to"))
                except Exception as exc:
                    a["email_error"] = str(exc)

    if fired:
        save_alerts(alerts)
    return fired

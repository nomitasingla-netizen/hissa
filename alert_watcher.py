"""Standalone background watcher for price alerts.

Runs independently of the Streamlit dashboard, so email alerts fire even when
the browser is closed. Create alerts and set SMTP settings in the app's
"🔔 Alerts" tab, then run this watcher.

Usage
-----
    # check once and exit (ideal for Windows Task Scheduler every 15 min):
    python alert_watcher.py

    # check continuously, every 15 minutes:
    python alert_watcher.py --loop 900
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from core import alerts as al


def _run_once() -> int:
    active = [a for a in al.load_alerts() if a.get("status") == "active"]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not active:
        print(f"[{stamp}] No active alerts.")
        return 0
    if not al.config_ready():
        print(f"[{stamp}] {len(active)} active alerts but email is NOT configured "
              "— set SMTP settings in the app's Alerts tab (or ALERT_SMTP_* env vars).")
    fired = al.check_alerts()
    if fired:
        for a in fired:
            print(f"[{stamp}] 🔔 FIRED: {a['tv']} {a['kind'].upper()} @ "
                  f"{a['triggered_price']} (level {a['level']})"
                  + (f"  EMAIL ERROR: {a['email_error']}" if a.get("email_error") else ""))
    else:
        print(f"[{stamp}] Checked {len(active)} alerts — none triggered.")
    return len(fired)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sector Scanner price-alert watcher")
    ap.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                    help="Keep running, checking every SECONDS (e.g. 900 = 15 min). "
                         "Omit to check once and exit.")
    args = ap.parse_args()

    if args.loop <= 0:
        _run_once()
        return

    print(f"Watcher started — checking every {args.loop}s. Ctrl+C to stop.")
    try:
        while True:
            _run_once()
            time.sleep(args.loop)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()

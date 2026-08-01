#!/usr/bin/env bash
# ==== Stop the Sector Breakout Scanner dashboard (macOS / Linux) ====
# Kills whatever process is listening on port 8501.
PORT=8501

if ! command -v lsof >/dev/null 2>&1; then
  echo "lsof not found; cannot locate the process. Stop it manually (Ctrl+C in its terminal)."
  exit 1
fi

PIDS=$(lsof -ti "tcp:$PORT" 2>/dev/null)
if [ -z "$PIDS" ]; then
  echo "Scanner is not running."
else
  echo "Stopping Scanner (PID $PIDS)..."
  kill $PIDS 2>/dev/null || kill -9 $PIDS 2>/dev/null
  echo "Scanner stopped."
fi

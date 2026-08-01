#!/usr/bin/env bash
# ==== Start the Sector Breakout Scanner dashboard (macOS / Linux) ====
cd "$(dirname "$0")" || exit 1
PORT=8501

open_url() {
  if command -v open >/dev/null 2>&1; then
    open "http://localhost:$PORT"          # macOS
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:$PORT"      # Linux
  fi
}

# If it's already running on the port, just open the browser.
if command -v lsof >/dev/null 2>&1 && lsof -ti "tcp:$PORT" >/dev/null 2>&1; then
  echo "Scanner already running. Opening browser..."
  open_url
  exit 0
fi

echo "Starting Sector Breakout Scanner..."
open_url
exec python3 -m streamlit run app.py --server.headless true --server.port "$PORT"

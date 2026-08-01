#!/usr/bin/env bash
# Launch the Sector Breakout Scanner dashboard (macOS / Linux).
cd "$(dirname "$0")" || exit 1
exec python3 -m streamlit run app.py

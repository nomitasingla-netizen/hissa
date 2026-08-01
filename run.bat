@echo off
REM Launch the Sector Breakout Scanner dashboard.
cd /d "%~dp0"
python -m streamlit run app.py

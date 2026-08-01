@echo off
REM ==== Start the Sector Breakout Scanner dashboard ====
cd /d "%~dp0"

REM If it's already running on port 8501, just open the browser.
netstat -ano | findstr /R /C:"LISTENING" | findstr ":8501" >nul
if %errorlevel%==0 (
    echo Scanner already running. Opening browser...
    start "" "http://localhost:8501"
    goto :eof
)

echo Starting Sector Breakout Scanner...
start "" "http://localhost:8501"
python -m streamlit run app.py --server.headless true --server.port 8501

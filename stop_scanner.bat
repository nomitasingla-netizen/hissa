@echo off
REM ==== Stop the Sector Breakout Scanner dashboard ====
REM Kills whatever process is listening on port 8501.
setlocal enabledelayedexpansion
set FOUND=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo Stopping Scanner (PID %%p)...
    taskkill /PID %%p /F >nul 2>&1
    set FOUND=1
)
if "!FOUND!"=="0" (
    echo Scanner is not running.
) else (
    echo Scanner stopped.
)
timeout /t 2 >nul

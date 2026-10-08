@echo off
title Smart Assistive Vision - Live Sharable Tunnel
echo ============================================================
echo   Smart Vision Assistant - Live Sharable Simulation Link
echo ============================================================
echo.
echo [1/2] Verifying Python simulation server is active on port 8000...
netstat -ano | findstr :8000 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Starting Python simulation server...
    set PYTHON_BIN=portable_python\python.exe
    if not exist "%PYTHON_BIN%" set PYTHON_BIN=python
    start /min "Python Sim Server" %PYTHON_BIN% run_simulation_server.py
    timeout /t 3 >nul
)

echo [2/2] Generating instant public HTTPS link for your friend...
echo.
echo ============================================================
echo   COPY THE HTTPS URL BELOW AND SEND IT TO YOUR FRIEND:
echo   (Add /simulation.html to the link if opening in browser)
echo   (Press Ctrl+C when you want to stop sharing)
echo ============================================================
echo.

ssh -o StrictHostKeyChecking=no -p 443 -R0:localhost:8000 a.pinggy.io
pause

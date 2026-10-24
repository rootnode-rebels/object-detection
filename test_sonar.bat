@echo off
title HC-SR04 Sonar Diagnostic Suite
echo ============================================================
echo   HC-SR04 Ultrasonic Sonar Diagnostic Suite
echo   Smart Vision Assistant for Visually Impaired
echo ============================================================
echo.

set PYTHON_BIN=portable_python\python.exe
if not exist "%PYTHON_BIN%" (
    set PYTHON_BIN=python
)

echo [MENU] Select Sonar Diagnostic Mode:
echo   1. Interactive Terminal Sonar Radar Simulation (No hardware needed)
echo   2. Hardware USB Serial Sonar Bridge (e.g., Arduino COM3 / COM4)
echo.

set /p CHOICE="Enter choice [1 or 2, default=1]: "
if "%CHOICE%"=="2" (
    set /p PORT_NAME="Enter COM Port (e.g. COM3): "
    echo [INFO] Connecting to %PORT_NAME%...
    %PYTHON_BIN% test_sonar.py --port %PORT_NAME%
) else (
    echo [INFO] Launching Terminal Sonar Radar Simulation...
    %PYTHON_BIN% test_sonar.py --sim
)

pause

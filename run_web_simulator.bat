@echo off
title Smart Assistive Vision - Real Python Simulation Server
echo ============================================================
echo   Smart Vision Assistant - Python AI Simulation Server
echo   Running ACTUAL Python Codebase:
echo   - yolo_detector.py with ONNX 92-Class Model
echo   - 9-Part Spatial Grid Navigation Engine
echo   - Live Camera / Corridor Feed with MJPEG Streaming
echo   - Native Windows Speech Audio
echo ============================================================
echo.

set PYTHON_BIN=portable_python\python.exe
if not exist "%PYTHON_BIN%" (
    set PYTHON_BIN=python
)

echo [INFO] Starting Python Simulation Server on http://localhost:8000...
echo [INFO] Opening http://localhost:8000/simulation.html in your browser...
echo.

start "" "http://localhost:8000/simulation.html"
%PYTHON_BIN% run_simulation_server.py

pause


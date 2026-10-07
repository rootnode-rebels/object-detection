@echo off
title Smart Assistive Vision - CPU Hardware Benchmark
echo ============================================================
echo   Smart Vision Assistant - Edge AI CPU Benchmark Tool
echo   Raspberry Pi 5 Architecture (ARM NEON / OpenCV DNN)
echo ============================================================
echo.

set PYTHON_BIN=portable_python\python.exe

if not exist "%PYTHON_BIN%" (
    set PYTHON_BIN=python
)

echo [INFO] Running 100-frame hardware & AI pipeline benchmark...
echo.

%PYTHON_BIN% benchmark_performance.py --frames 100 --resolution 320

echo.
pause

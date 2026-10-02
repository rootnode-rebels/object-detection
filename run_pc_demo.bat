@echo off
title Smart Assistive Vision - PC Simulation Demo
echo ============================================================
echo   Smart Vision Assistant for Visually Impaired
echo   Running in PC Simulation Mode (Raspberry Pi 5 Specs)
echo   YOLOv8 92-Class Custom Assistive Model
echo ============================================================
echo.

set PYTHON_BIN=portable_python\python.exe

if not exist %PYTHON_BIN% (
    set PYTHON_BIN=python
)

echo [INFO] Launching PC Simulator with:
echo        Model:   model_files\yolov8_assistive_92.onnx
echo        Classes: classes.txt (92 categories)
echo.
echo Controls during demo:
echo   [SPACE]     : Press virtual hardware tactile button (scan query)
echo   [W] / [UP]  : Move closer to obstacle (decreases distance)
echo   [S] / [DOWN]: Move farther from obstacle (increases distance)
echo   [A]         : Toggle Auto-walk radar simulation
echo   [Q] / [ESC] : Quit simulation
echo.

%PYTHON_BIN% simulate_pc.py --model model_files\yolov8_assistive_92.onnx --classes classes.txt --resolution 320

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Simulation exited with error code %ERRORLEVEL%.
    pause
)

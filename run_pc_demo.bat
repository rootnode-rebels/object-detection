@echo off
title Smart Assistive Vision - PC Simulation Demo
echo ============================================================
echo   Smart Vision Assistant for Visually Impaired
echo   Running in PC Simulation Mode (Raspberry Pi 5 Specs)
echo   YOLOv8 92-Class Custom Assistive Model
echo ============================================================
echo.

set PYTHON_BIN=portable_python\python.exe

if not exist "%PYTHON_BIN%" (
    where python >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Python is not installed or not in system PATH.
        if exist "dist\SmartVisionAssistant\SmartVisionAssistant.exe" (
            echo [INFO] Falling back to pre-compiled Standalone Executable...
            echo.
            start "" "dist\SmartVisionAssistant\SmartVisionAssistant.exe"
            exit /b 0
        ) else (
            echo [ERROR] Please install Python 3.10+ from python.org or run dist\SmartVisionAssistant\SmartVisionAssistant.exe
            pause
            exit /b 1
        )
    )
    set PYTHON_BIN=python
)

:: Check if required dependencies are installed, auto-install if missing
%PYTHON_BIN% -c "import cv2, numpy, win32com.client" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Required Python modules (OpenCV, NumPy, PyWin32) not found.
    echo [INFO] Auto-downloading and installing dependencies into your environment...
    echo.
    %PYTHON_BIN% -m pip install --upgrade pip
    %PYTHON_BIN% -m pip install -r "%~dp0requirements.txt"
    echo.
)


echo [INFO] Launching PC Simulator with:
echo        Model:   model_files\yolov8_assistive_92.onnx
echo        Classes: classes.txt (92 categories)
echo.
echo [INFO] Automatic Voice Guidance is ACTIVE:
echo        - Obstacle on left   --^> Voice announces: "Obstacle on the left. Go right."
echo        - Obstacle on right  --^> Voice announces: "Obstacle on the right. Go left."
echo        - Obstacle ahead     --^> Voice announces: "Obstacle directly ahead. Go right or go left."
echo        No button press required! System speaks automatically in real-time.
echo.
echo Interactive Controls:
echo   [J] / [LEFT]  : Shift obstacle to LEFT (Triggers "Go right")
echo   [K] / [RIGHT] : Shift obstacle to RIGHT (Triggers "Go left")
echo   [C]           : Center obstacle
echo   [W] / [UP]    : Move closer to obstacle (decreases distance)
echo   [S] / [DOWN]  : Move farther from obstacle (increases distance)
echo   [A]           : Toggle Auto-walk radar simulation
echo   [SPACE]       : Optional manual instant scan
echo   [L] / [M] / [R]: Query specific sector (Left / Middle / Right)
echo   [Q] / [ESC]   : Quit simulation
echo.

%PYTHON_BIN% simulate_pc.py --model model_files\yolov8_assistive_92.onnx --classes classes.txt --resolution 320

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Simulation exited with error code %ERRORLEVEL%.
    pause
)

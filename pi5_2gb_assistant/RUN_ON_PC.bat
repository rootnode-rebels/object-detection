@echo off
title Smart Vision Assistant - PC Simulation
echo ============================================================
echo   Smart Vision Assistant for Visually Impaired
echo   Running YOLOv8 92-Class Custom Assistive Model
echo ============================================================
echo.

set PY_CMD=
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PY_CMD=python
) else (
    where py >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set PY_CMD=py
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python was not found in your system PATH!
    echo Please run INSTALL_MODULES.bat or install Python from python.org.
    pause
    exit /b 1
)

:: Verify OpenCV is available, auto-install if missing
%PY_CMD% -c "import cv2, numpy, psutil, win32com.client" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] First-time setup: Installing required modules...
    %PY_CMD% -m pip install opencv-python numpy psutil pywin32
    echo.
)

echo [INFO] Launching Smart Vision Assistant with:
echo        - AI Model: model_files\yolov8_assistive_92.onnx (92 custom blind-safety classes)
echo        - Spatial Voice Announcements: Active
echo        - Directional Guidance: Active
echo.
echo [CONTROLS]:
echo   - Press 'q' inside the webcam window to exit.
echo   - Press 's' to simulate a close obstacle trigger.
echo.

%PY_CMD% "%~dp0simulate_pc.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [NOTE] If webcam didn't open, make sure your webcam is plugged in and allowed in Windows Settings.
    pause
)

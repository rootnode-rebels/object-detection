@echo off
title Smart Vision Assistant - 1-Click Module Installer
echo ============================================================
echo   Smart Vision Assistant - Automated Module Installer
echo   Target: Windows PC (Webcam, AI Model, Voice Guidance)
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
    echo.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo CRITICAL: During installation, check the box "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo [OK] Python detected:
%PY_CMD% --version
echo [INFO] Python executable:
%PY_CMD% -c "import sys; print(' ', sys.executable)"
echo.

echo [INFO] Installing required AI and vision modules...
echo (opencv-python, numpy, psutil, pywin32)
echo.
%PY_CMD% -m pip install --upgrade pip
%PY_CMD% -m pip install opencv-python numpy psutil pywin32

echo.
echo ============================================================
echo   Verifying Installation...
echo ============================================================
%PY_CMD% -c "import cv2; print('  [OK] cv2 (OpenCV):', cv2.__version__)"
%PY_CMD% -c "import numpy; print('  [OK] numpy:', numpy.__version__)"
%PY_CMD% -c "import psutil; print('  [OK] psutil:', psutil.__version__)"
%PY_CMD% -c "import win32com.client; print('  [OK] win32com (Windows Speech): Available')"

echo.
if %ERRORLEVEL% EQU 0 (
    echo ============================================================
    echo   SUCCESS! All required modules are installed correctly.
    echo   You can now double-click: RUN_ON_PC.bat
    echo ============================================================
) else (
    echo [ERROR] Verification failed. Please check the error message above.
)
echo.
pause

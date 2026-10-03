@echo off
title Smart Assistive Vision - Client Module Installer
echo ============================================================
echo   Smart Vision Assistant - Automated Module Installer
echo   Installs all required Python modules for your PC
echo ============================================================
echo.

:: 1. Check Python installation
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in your system PATH.
    echo.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    echo NOTE: Make sure to check "Add Python to PATH" during installation.
    echo.
    echo Alternatively, you can run the standalone app without Python:
    echo   run_exe.bat
    echo.
    pause
    exit /b 1
)

echo [OK] Python detected:
python --version
echo.

:: 2. Upgrade pip and install required packages
echo [INFO] Installing required Python modules (OpenCV, NumPy, PyWin32, etc.)...
echo This may take a few moments on the first run.
echo.
python -m pip install --upgrade pip
python -m pip install -r "%~dp0requirements.txt"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Some modules may have failed to install. Retrying core dependencies individually...
    python -m pip install opencv-python numpy psutil pywin32
)

echo.
echo ============================================================
echo   Verifying Installed Modules...
echo ============================================================
python -c "import cv2; print('  [OK] OpenCV version:', cv2.__version__)"
python -c "import numpy; print('  [OK] NumPy version:', numpy.__version__)"
python -c "import win32com.client; print('  [OK] Windows SAPI Voice Speech: Available')"
python -c "import psutil; print('  [OK] psutil: Available')"

echo.
echo ============================================================
echo   SUCCESS! All required modules are installed and ready.
echo   You can now launch the application using:
echo     - run_pc_demo.bat  (Interactive PC Simulation)
echo     - run_exe.bat      (Standalone Executable)
echo ============================================================
echo.
pause

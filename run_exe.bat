@echo off
title Smart Assistive Vision - Standalone Windows App
echo ============================================================
echo   Smart Vision Assistant (Standalone Windows Executable)
echo   YOLOv8 92-Class Custom Assistive Model + Voice Guidance
echo   No Python Installation Required
echo ============================================================
echo.
cd /d "%~dp0"

if not exist "dist\SmartVisionAssistant\SmartVisionAssistant.exe" (
    echo [ERROR] Could not find dist\SmartVisionAssistant\SmartVisionAssistant.exe
    echo Please ensure the executable distribution is downloaded properly.
    echo.
    pause
    exit /b 1
)

echo [INFO] Launching SmartVisionAssistant.exe...
echo [INFO] Please wait while the AI neural network model loads...
echo.

"dist\SmartVisionAssistant\SmartVisionAssistant.exe"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
) else (
    echo.
    echo [INFO] Application closed cleanly.
)

pause

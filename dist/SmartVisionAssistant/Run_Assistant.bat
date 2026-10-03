@echo off
title Smart Assistive Vision - Standalone Windows Application
cd /d "%~dp0"
echo ============================================================
echo   Smart Vision Assistant (Standalone Windows Application)
echo   YOLOv8 92-Class Spatial Vision + Spoken Navigation Guidance
echo   No Python Installation Required
echo ============================================================
echo.
echo [INFO] Loading YOLOv8 AI Model and initializing camera...
echo.

"%~dp0SmartVisionAssistant.exe"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with error code %ERRORLEVEL%.
) else (
    echo.
    echo [INFO] Application closed cleanly.
)

pause

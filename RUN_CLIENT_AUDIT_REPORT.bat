@echo off
title Smart Vision Assistant - Client Audit and Dataset Explainer
echo ============================================================
echo   Smart Vision Assistant - Technical Audit Report
echo   Running Comprehensive System, Dataset and Architecture Audit...
echo ============================================================
echo.

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in system PATH.
    pause
    exit /b 1
)

python "%~dp0show_model_metrics.py"

echo.
pause

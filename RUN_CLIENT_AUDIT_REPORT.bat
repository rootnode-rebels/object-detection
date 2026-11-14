@echo off
title Smart Vision Assistant - Client Audit & Dataset Explainer
cls
echo ============================================================================
echo   Running Comprehensive System, Dataset & Architecture Audit...
echo ============================================================================
echo.
python "%~dp0show_model_metrics.py" 30
if %ERRORLEVEL% NEQ 0 (
    python show_model_metrics.py 30
)
echo.
echo ============================================================================
echo   Audit Complete. Press any key to close this window.
echo ============================================================================
pause >nul

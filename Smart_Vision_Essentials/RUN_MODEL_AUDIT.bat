@echo off
title Smart Vision Assistant - Model Audit and Benchmark Report
echo ============================================================
echo   Running AI Model Audit and Performance Benchmark...
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
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

%PY_CMD% "%~dp0show_model_metrics.py"
echo.
pause

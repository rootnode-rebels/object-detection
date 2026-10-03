@echo off
title Smart Assistive Vision - Standalone Windows App
echo ============================================================
echo   Launching Smart Vision Assistant (Standalone EXE)
echo   YOLOv8 92-Class Custom Assistive Model + Voice Guidance
echo ============================================================
echo.
cd /d "%~dp0"
start "" "dist\SmartVisionAssistant\SmartVisionAssistant.exe"

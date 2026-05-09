@echo off
setlocal enabledelayedexpansion
cd /d %~dp0

where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed or not available on PATH.
    pause
    exit /b 1
)

python run_app.py
pause

@echo off
rem Double-click this file to start Ollama, the API and the Streamlit UI.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
    echo.
    echo Startup failed. See the message above.
    pause
)

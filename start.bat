@echo off
rem Double-click: update to the latest version (when git is installed), serve the ocean and open the browser.
cd /d "%~dp0"
where git >nul 2>nul && git pull
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve.ps1"
pause

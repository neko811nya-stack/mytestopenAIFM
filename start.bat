@echo off
rem Double-click: update to the latest version, start a local server and open the ocean.
cd /d "%~dp0"
git pull
start "" cmd /c "timeout /t 2 >nul & start http://localhost:8000/"
where python >nul 2>nul && (python -m http.server 8000) || (py -m http.server 8000)
pause

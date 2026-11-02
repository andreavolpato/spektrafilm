@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Spektrafilm.ps1"
if errorlevel 1 (
    echo.
    echo Installation did not finish. Review the error above and try again.
    pause
    exit /b 1
)
pause

@echo off
setlocal
set "PYTHON=%~dp0.venv\Scripts\python.exe"
set "SOURCE=%~dp0Source"
if not exist "%PYTHON%" (
    echo Spektrafilm is not installed yet. Run Install-Spektrafilm.bat first.
    pause
    exit /b 1
)
cd /d "%SOURCE%" || exit /b 1
"%PYTHON%" -m spektrafilm_gui.app

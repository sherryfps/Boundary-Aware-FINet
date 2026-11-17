@echo off
setlocal
cd /d "%~dp0"
echo Setting up the local FINet CPU environment in this project folder.
if exist ".venv\Scripts\python.exe" goto validate_environment

py -3.12 -c "import sys,struct; assert sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8" >nul 2>&1
if not errorlevel 1 (
    py -3.12 -m venv .venv
    if errorlevel 1 goto failed
    goto validate_environment
)

rem Codex includes Python 3.12 on laptops without the system Python launcher.
set "FINET_BASE_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%FINET_BASE_PYTHON%" goto missing_python
"%FINET_BASE_PYTHON%" -c "import sys,struct; assert sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 goto missing_python
"%FINET_BASE_PYTHON%" -m venv .venv
if errorlevel 1 goto failed

:validate_environment
".venv\Scripts\python.exe" -c "import sys,struct; assert sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 (
    echo The existing .venv cannot run here. Preserve it under another name and rerun this setup.
    goto failed
)
set "TORCH_HOME=%~dp0model_cache\torch"
".venv\Scripts\python.exe" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-transfer-cpu.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\verify_portable_install.py
if errorlevel 1 goto failed
echo Setup and real trained-model inference checks passed.
echo Run START_FINET.bat to open the local application.
if /i not "%~1"=="--no-pause" pause
exit /b 0
:missing_python
echo Install 64-bit Python 3.12 from https://www.python.org/downloads/ first.
echo Then run this setup again. No environment was changed.
:failed
echo Setup stopped. Give the displayed error to Codex; do not retrain completed models.
if /i not "%~1"=="--no-pause" pause
exit /b 1

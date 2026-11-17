@echo off
setlocal
title FINet Vision AI
pushd "%~dp0"
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo FINet needs its Python environment. Run SETUP_NEW_LAPTOP.cmd first.
    popd
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\launch_app.py %*
set "FINET_EXIT_CODE=%ERRORLEVEL%"
popd
if not "%FINET_EXIT_CODE%"=="0" pause
exit /b %FINET_EXIT_CODE%

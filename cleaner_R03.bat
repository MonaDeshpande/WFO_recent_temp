@echo off
REM ===========================================================================
REM == SCADA Data Cleaning Launcher (WFO_VENV) - Corrected Path
REM == This file runs the Python script inside the 'wfo_venv' virtual environment.
REM == ASSUMES: The 'wfo_venv' folder exists and has all required libraries
REM ==          (pandas, numpy, psycopg2-binary, scipy, xlsxwriter) installed.
REM ===========================================================================

set SCRIPT_NAME="cleaner_R03.py"
set VENV_NAME="wfo_venv"
REM %~dp0 resolves to the current directory path (e.g., E:\GENERATING_DATA\)
set VENV_PATH="%~dp0%VENV_NAME%"
set PYTHON_EXE="%VENV_PATH%\Scripts\python.exe"

echo.
echo === Starting SCADA Data Cleaning Process in %VENV_NAME% ===
echo.
echo Checking for VENV Python executable...

REM --- 1. Check for the VENV Python Executable ---
if not exist %PYTHON_EXE% (
    echo ❌ ERROR: The VENV Python executable was not found.
    echo ❌ Expected Path: %PYTHON_EXE%
    echo ❌ Please ensure the '%VENV_NAME%' folder is correctly set up in this directory.
    pause
    exit /b 1
)
echo ✅ VENV Python executable found.

echo.
echo === Launching the Data Cleaning Script... ===
echo.

REM *** CRITICAL LINE: Executes the script using the VENV's Python in a new, persistent window ***
start "SCADA Data Cleaning (wfo_venv)" cmd /k "%PYTHON_EXE% %SCRIPT_NAME%"

echo.
echo The data cleaning process has been launched in a new window.
echo Please monitor the new window for progress or errors.
echo.
pause >nul
exit /b 0
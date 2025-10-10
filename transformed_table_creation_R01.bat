@echo off
REM ===========================================================================
REM == Direct Execution in 'venv' - FIXED QUOTATION MARKS
REM == ASSUMES: A folder named 'venv' exists in this directory
REM ==          and contains all required libraries (psycopg2, etc.).
REM ===========================================================================

set SCRIPT_NAME="transformed_table_creation_R01.py"
set VENV_NAME="venv"
REM %~dp0 is the path to the directory where this batch file is located (e.g., E:\GENERATING_DATA\)
set VENV_PATH="%~dp0%VENV_NAME%"
set PYTHON_EXE="%VENV_PATH%\Scripts\python.exe"

echo.
echo === Checking for Virtual Environment and Python Executable... ===
if not exist %PYTHON_EXE% (
    echo ❌ ERROR: The expected Python executable was not found.
    echo ❌ Path: %PYTHON_EXE%
    echo ❌ Please ensure the '%VENV_NAME%' folder exists in the same directory.
    pause
    exit /b 1
)
echo ✅ Virtual Environment executable found.

echo.
echo === Launching the Data Transformation Script in %VENV_NAME%... ===
echo.

REM *** CRITICAL LINE: Uses one set of quotes around the executable path for CMD /k ***
start "SCADA Transformation (venv)" cmd /k "%PYTHON_EXE% %SCRIPT_NAME%"

echo.
echo The data transformation process is running in a new window.
echo You can close this window now.
echo.
pause >nul
exit /b 0
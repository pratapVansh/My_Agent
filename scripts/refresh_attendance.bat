@echo off
:: Refreshes ERP attendance on a schedule (Windows Task Scheduler).
:: Logs to scripts\refresh_attendance.log
::
:: Credentials are read from the environment, never stored in this file.
:: Set them once for the account that runs the scheduled task:
::
::     setx ERP_ROLLNO   "your_roll_number"
::     setx ERP_PASSWORD "your_erp_password"
::     setx AGENT_USER_ID "your_agent_user_id"
::
:: Paths are derived from this script's own location, so the checkout can live
:: anywhere and can be moved without editing anything here.

setlocal

set "PROJECT=%~dp0.."
set "LOG=%~dp0refresh_attendance.log"

if exist "%PROJECT%\venv\Scripts\python.exe" (
    set "PYTHON=%PROJECT%\venv\Scripts\python.exe"
) else if exist "%PROJECT%\.venv\Scripts\python.exe" (
    set "PYTHON=%PROJECT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

if "%AGENT_USER_ID%"=="" (
    echo %DATE% %TIME% - ABORT: AGENT_USER_ID is not set >> "%LOG%"
    exit /b 2
)
if "%ERP_ROLLNO%"=="" (
    echo %DATE% %TIME% - ABORT: ERP_ROLLNO is not set >> "%LOG%"
    exit /b 2
)
if "%ERP_PASSWORD%"=="" (
    echo %DATE% %TIME% - ABORT: ERP_PASSWORD is not set >> "%LOG%"
    exit /b 2
)

echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo %DATE% %TIME% - Starting ERP attendance refresh >> "%LOG%"
echo ============================================================ >> "%LOG%"

cd /d "%PROJECT%"

:: Roll number and password are passed via the environment the child inherits,
:: not as arguments — a command line is readable by any process listing.
"%PYTHON%" scripts\scrape_erp_attendance.py ^
    --user-id "%AGENT_USER_ID%" ^
    --headless >> "%LOG%" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo %DATE% %TIME% - SUCCESS >> "%LOG%"
) else (
    echo %DATE% %TIME% - FAILED ^(exit code %ERRORLEVEL%^) >> "%LOG%"
)

endlocal

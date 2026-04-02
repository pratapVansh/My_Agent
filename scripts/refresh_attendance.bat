@echo off
:: Auto-refreshes ERP attendance every day via Task Scheduler
:: Logs to scripts\refresh_attendance.log

set PROJECT=C:\Users\iamch\OneDrive\Desktop\My_Agent
set PYTHON=C:\Users\iamch\AppData\Local\Programs\Python\Python311\python.exe
set LOG=%PROJECT%\scripts\refresh_attendance.log

echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo %DATE% %TIME% — Starting ERP attendance refresh >> "%LOG%"
echo ============================================================ >> "%LOG%"

cd /d "%PROJECT%"
"%PYTHON%" scripts\scrape_erp_attendance.py ^
    --user-id vansh ^
    --rollno 23IT3048 ^
    --password 14735 ^
    --headless >> "%LOG%" 2>&1

if %ERRORLEVEL% EQU 0 (
    echo %DATE% %TIME% — SUCCESS >> "%LOG%"
) else (
    echo %DATE% %TIME% — FAILED (exit code %ERRORLEVEL%^) >> "%LOG%"
)

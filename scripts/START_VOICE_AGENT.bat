@echo off
REM Quick Start Script for Voice Agent (Windows)
REM Run this to start both backend and frontend

echo ======================================
echo 🎙️ VOICE AGENT QUICK START
echo ======================================
echo.

cd /d "%~dp0.."
set PROJECT_ROOT=%CD%

echo 📁 Project root: %PROJECT_ROOT%
echo.

REM Check if virtual environment exists
if not exist "%PROJECT_ROOT%\venv" if not exist "%PROJECT_ROOT%\.venv" (
    echo ⚠️  Python virtual environment not found!
    echo    Run: python -m venv venv
    echo    Then: venv\Scripts\activate
    echo    Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if frontend dependencies are installed
if not exist "%PROJECT_ROOT%\frontend\node_modules" (
    echo ⚠️  Frontend dependencies not installed!
    echo    Run: cd frontend
    echo    Then: npm install
    pause
    exit /b 1
)

echo ✅ Dependencies found
echo.

REM Activate virtual environment
if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
) else if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
)

echo 🚀 Starting Backend (Port 10000)...
start "Backend-My_Agent" cmd /k "cd /d %PROJECT_ROOT% && uvicorn app.main:app --reload --host 0.0.0.0 --port 10000"

echo    Waiting for backend to start...
timeout /t 5 /nobreak > nul

REM Test backend connection
curl -s http://localhost:10000/docs > nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend started successfully!
    echo    API Docs: http://localhost:10000/docs
) else (
    echo ⚠️  Backend may still be starting...
    echo    Check the Backend window for errors
)

echo.

echo 🚀 Starting Frontend (Port 3000)...
start "Frontend-My_Agent" cmd /k "cd /d %PROJECT_ROOT%\frontend && npm run dev"

echo    Waiting for frontend to start...
timeout /t 5 /nobreak > nul

echo.
echo ======================================
echo ✅ VOICE AGENT READY!
echo ======================================
echo.
echo 🌐 Open in browser:
echo    → User Mode: http://localhost:3000/user
echo    → Recruiter Mode: http://localhost:3000/recruiter
echo.
echo 📚 API Documentation:
echo    → http://localhost:10000/docs
echo.
echo 🎤 How to Use Voice:
echo    1. Click microphone button
echo    2. Allow browser microphone access
echo    3. Speak your query
echo    4. Wait 2 seconds of silence
echo    5. Response plays automatically!
echo.
echo 🛑 Close the Backend and Frontend windows to stop
echo ======================================
echo.

REM Open browser automatically
timeout /t 3 /nobreak > nul
start http://localhost:3000/user

echo.
echo Press any key to exit this window...
pause > nul

#!/bin/bash
# Quick Start Script for Voice Agent
# Run this to start both backend and frontend

echo "======================================"
echo "🎙️ VOICE AGENT QUICK START"
echo "======================================"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Check if backend dependencies are installed
if [ ! -d "$PROJECT_ROOT/venv" ] && [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "⚠️  Python virtual environment not found!"
    echo "   Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "$PROJECT_ROOT/frontend/node_modules" ]; then
    echo "⚠️  Frontend dependencies not installed!"
    echo "   Run: cd frontend && npm install"
    exit 1
fi

echo "✅ Dependencies found"
echo ""

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "🚀 Starting Backend (Port 10000)..."
cd "$PROJECT_ROOT"

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

uvicorn app.main:app --reload --host 0.0.0.0 --port 10000 > backend.log 2>&1 &
BACKEND_PID=$!

echo "   Backend PID: $BACKEND_PID"
echo "   Logs: backend.log"
echo ""

# Wait for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:10000/docs > /dev/null 2>&1; then
    echo "✅ Backend started successfully!"
    echo "   API Docs: http://localhost:10000/docs"
else
    echo "❌ Backend failed to start. Check backend.log"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""

# Start frontend
echo "🚀 Starting Frontend (Port 3000)..."
cd "$PROJECT_ROOT/frontend"

npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!

echo "   Frontend PID: $FRONTEND_PID"
echo "   Logs: frontend.log"
echo ""

# Wait for frontend to start
sleep 5

echo "======================================"
echo "✅ VOICE AGENT READY!"
echo "======================================"
echo ""
echo "🌐 Open in browser:"
echo "   → User Mode: http://localhost:3000/user"
echo "   → Recruiter Mode: http://localhost:3000/recruiter"
echo ""
echo "📚 API Documentation:"
echo "   → http://localhost:10000/docs"
echo ""
echo "🎤 How to Use Voice:"
echo "   1. Click microphone button"
echo "   2. Allow browser microphone access"
echo "   3. Speak your query"
echo "   4. Wait 2 seconds of silence"
echo "   5. Response plays automatically!"
echo ""
echo "🛑 Press Ctrl+C to stop both servers"
echo "======================================"
echo ""

# Keep script running
wait

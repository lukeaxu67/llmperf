#!/bin/bash
# LLMPerf Development Startup Script
# Configured for China mirror sources

echo "========================================"
echo "LLMPerf Development Server"
echo "========================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "WARNING: Node.js not found. Frontend will not be available."
    echo "Only backend API will be started."
    echo
    echo "Backend:  http://localhost:8000"
    echo "API Docs: http://localhost:8000/api/docs"
    echo
    python3 -m llmperf.cli.web --host 0.0.0.0 --port 8000 --reload
    exit 0
fi

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    echo "Using Taobao mirror for npm..."
    cd frontend && npm install --registry=https://registry.npmmirror.com && cd ..
    echo
fi

# Start development servers
echo "Starting development servers..."
echo
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/api/docs"
echo
echo "Tip: Using China mirror sources for faster downloads"
echo

# Start frontend in background
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

# Start backend
python3 -m llmperf.cli.web --host 0.0.0.0 --port 8000 --reload

# Cleanup on exit
trap "kill $FRONTEND_PID 2>/dev/null" EXIT

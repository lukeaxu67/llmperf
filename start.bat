@echo off
REM LLMPerf Development Startup Script for Windows
REM Configured for China mirror sources

echo ========================================
echo LLMPerf Development Server
echo ========================================
echo.

REM Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found in PATH
    exit /b 1
)

REM Check Node.js
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo WARNING: Node.js not found. Frontend will not be available.
    echo Only backend API will be started.
    echo.
    goto :backend_only
)

REM Install frontend dependencies if needed
if not exist "frontend\node_modules" (
    echo Installing frontend dependencies...
    echo Using Taobao mirror for npm...
    cd frontend
    npm install --registry=https://registry.npmmirror.com
    cd ..
    echo.
)

REM Start frontend and backend
echo Starting development servers...
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/api/docs
echo.
echo Tip: Using China mirror sources for faster downloads
echo.

start "LLMPerf Frontend" cmd /c "cd frontend && npm run dev"
goto :backend

:backend_only
echo Starting backend only...
echo.
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/api/docs
echo.

:backend
python -m llmperf.cli.web --host 0.0.0.0 --port 8000 --reload

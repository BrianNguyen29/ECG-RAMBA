@echo off
echo ==========================================
echo    ❤️  ECG-RAMBA Web App Launcher
echo ==========================================
echo.
echo [1/2] Launching Backend...
start "ECG-RAMBA Backend" cmd /k "cd backend && pip install -r requirements.txt && python -m uvicorn main:app --reload"

echo [2/2] Launching Frontend...
start "ECG-RAMBA Frontend" cmd /k "cd frontend && npm install && npm run dev"

echo.
echo ✅ Services initiated!
echo ------------------------------------------
echo 🌍 Backend API: http://localhost:8000
echo 🌍 Web Dashboard: http://localhost:5173
echo.
echo Press any key to exit this launcher...
pause >nul

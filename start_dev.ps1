# ECG-RAMBA Development Environment Startup Script
# Usage: .\start_dev.ps1

$BACKEND_PORT = 8003
$FRONTEND_PORT = 5173

Write-Host ""
Write-Host "===== ECG-RAMBA Development Environment =====" -ForegroundColor Cyan
Write-Host ""

# Kill existing processes
Write-Host "[1] Clearing ports..." -ForegroundColor Yellow
$ErrorActionPreference = "SilentlyContinue"
Get-NetTCPConnection -LocalPort $BACKEND_PORT | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
Get-NetTCPConnection -LocalPort $FRONTEND_PORT | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
$ErrorActionPreference = "Continue"
Start-Sleep -Seconds 1
Write-Host "    Ports cleared." -ForegroundColor Green

# Start Backend
Write-Host ""
Write-Host "[2] Starting Backend (Port $BACKEND_PORT)..." -ForegroundColor Yellow
$backendPath = Join-Path $PSScriptRoot "web_app\backend"
$backendCmd = "cd '$backendPath'; python -m uvicorn main:app --host 127.0.0.1 --port $BACKEND_PORT --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd
Start-Sleep -Seconds 3
Write-Host "    Backend: http://127.0.0.1:$BACKEND_PORT/docs" -ForegroundColor Green

# Start Frontend
Write-Host ""
Write-Host "[3] Starting Frontend (Port $FRONTEND_PORT)..." -ForegroundColor Yellow
$frontendPath = Join-Path $PSScriptRoot "web_app\frontend"
$frontendCmd = "cd '$frontendPath'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd
Start-Sleep -Seconds 2
Write-Host "    Frontend: http://localhost:$FRONTEND_PORT" -ForegroundColor Green

# Done
Write-Host ""
Write-Host "===== Servers Started =====" -ForegroundColor Cyan
Write-Host "Backend API:  http://127.0.0.1:$BACKEND_PORT/docs"
Write-Host "Frontend UI:  http://localhost:$FRONTEND_PORT"
Write-Host ""

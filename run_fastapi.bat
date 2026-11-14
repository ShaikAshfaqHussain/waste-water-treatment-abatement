@echo off
echo ========================================================
echo Starting EcoVision AI Wastewater FastAPI Server...
echo Web Dashboard URL: http://localhost:8000
echo REST API Docs:     http://localhost:8000/docs
echo ========================================================
cd /d "%~dp0"
python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --reload
pause

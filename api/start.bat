@echo off
echo Starting Textropy AI API...
cd /d "%~dp0.."
python -m uvicorn api.main:app --reload --port 8000
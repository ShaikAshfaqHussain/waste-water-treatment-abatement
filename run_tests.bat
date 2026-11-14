@echo off
echo ========================================================
echo Running Automated Pytest Suite for Wastewater AI...
echo ========================================================
cd /d "%~dp0"
python -m pytest tests/test_wastewater_pipeline.py -v
pause

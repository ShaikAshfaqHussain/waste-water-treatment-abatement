@echo off
echo ========================================================
echo Starting EcoVision AI Wastewater Streamlit Dashboard...
echo Dashboard URL: http://localhost:8501
echo ========================================================
cd /d "%~dp0"
python -m streamlit run app_streamlit.py
pause

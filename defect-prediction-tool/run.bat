@echo off
echo ========================================
echo   AI Defect Prediction Tool
echo ========================================
echo.

cd /d "%~dp0"

echo Dang khoi dong Streamlit...
start "" "C:\Users\Dell 7630\AppData\Local\Programs\Python\Python313\Scripts\streamlit.exe" run app.py

echo.
echo Mo trinh duyet tai: http://localhost:8501
echo.
pause

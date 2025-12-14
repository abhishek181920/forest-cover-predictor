@echo off
echo Forest Cover Type Prediction - Streamlit Application
echo ======================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
python -c "import streamlit, pandas, numpy, sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error: Failed to install required packages
        pause
        exit /b 1
    )
)

echo Starting Streamlit Application...
streamlit run streamlit_app.py

pause
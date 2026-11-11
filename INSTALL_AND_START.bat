@echo off
setlocal
cd /d "%~dp0"
echo ===============================================
echo ORBITAL - ESA Collision Avoidance Research Platform
echo ===============================================
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating Python virtual environment...
    py -3.11 -m venv .venv >nul 2>&1
    if errorlevel 1 py -3.13 -m venv .venv >nul 2>&1
    if errorlevel 1 py -3 -m venv .venv
    if errorlevel 1 (
        echo Could not create the environment. Install Python 3.11 or 3.13 and enable the Python launcher.
        pause
        exit /b 1
    )
)
call ".venv\Scripts\activate.bat"
echo [2/3] Installing / updating dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. Check your internet connection and Python version.
    pause
    exit /b 1
)
echo [3/3] Starting ORBITAL...
streamlit run app/app.py
pause

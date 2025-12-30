@echo off
cls
echo ========================================================================
echo ADAPTIVE MICRO-GRID SEGMENTATION - SETUP ONLY
echo ========================================================================
echo.
echo This script only sets up the environment without running the project.
echo.

REM Change to the script directory
cd /d "%~dp0"

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.7 or higher from https://www.python.org/
    pause
    exit /b 1
)
python --version
echo.

echo [2/3] Creating virtual environment...
if not exist "venv" (
    echo Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists.
)
echo.

echo [3/3] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
echo Installing required packages...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo.

echo ========================================================================
echo SETUP COMPLETE!
echo ========================================================================
echo.
echo Virtual environment is ready at: venv\
echo.
echo To run the project manually:
echo   1. Activate venv: venv\Scripts\activate
echo   2. Run training: python train.py
echo   3. Run evaluation: python evaluate.py
echo   4. Make predictions: python predict.py
echo.
echo Or simply run: run_project.bat
echo.
echo ========================================================================

pause

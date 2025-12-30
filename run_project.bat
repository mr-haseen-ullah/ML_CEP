@echo off
cls
echo ========================================================================
echo ADAPTIVE MICRO-GRID SEGMENTATION - AUTOMATED SETUP AND EXECUTION
echo ========================================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.7 or higher from https://www.python.org/
    pause
    exit /b 1
)
python --version
echo.

echo [2/5] Creating virtual environment...
if not exist "venv" (
    echo Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        echo Make sure Python venv module is available.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists.
)
echo.

echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

echo [4/5] Installing dependencies...
echo Installing required packages from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo Dependencies installed successfully!
echo.

echo ========================================================================
echo STARTING PROJECT EXECUTION
echo ========================================================================
echo.

echo [5/5] Running Training Pipeline...
echo -----------------------------------------------------------------------
python train.py
if errorlevel 1 (
    echo.
    echo WARNING: Training encountered an error!
    echo This might be due to missing dataset (will use synthetic data).
    echo.
)
echo.

echo ========================================================================
echo Training complete! Running Evaluation...
echo ========================================================================
echo.
python evaluate.py
if errorlevel 1 (
    echo.
    echo WARNING: Evaluation encountered an error!
    echo.
)
echo.

echo ========================================================================
echo PROJECT EXECUTION COMPLETE!
echo ========================================================================
echo.
echo Results saved to:
echo   - models/       (trained models and visualizations)
echo   - results/      (evaluation metrics and plots)
echo.
echo To make predictions, run:
echo   python predict.py
echo.
echo To run evaluation again:
echo   python evaluate.py
echo.
echo ========================================================================

REM Keep window open
echo.
echo Press any key to exit...
pause >nul

@echo off
cls
echo ========================================================================
echo DOWNLOADING UCI APPLIANCES ENERGY DATASET
echo ========================================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

echo Attempting to download energydata_complete.csv...
echo.

REM Check if file already exists
if exist "energydata_complete.csv" (
    echo energydata_complete.csv already exists!
    echo Do you want to re-download it? (Y/N)
    choice /C YN /N
    if errorlevel 2 goto :skip_download
    echo Deleting existing file...
    del energydata_complete.csv
)

REM Try to download using Python
echo Downloading from UCI Machine Learning Repository...
echo.

python -c "import urllib.request; urllib.request.urlretrieve('https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip', 'dataset.zip'); print('Download complete!')"

if errorlevel 1 (
    echo.
    echo Python download failed. Trying PowerShell...
    echo.
    
    REM Try PowerShell as fallback
    powershell -Command "Invoke-WebRequest -Uri 'https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip' -OutFile 'dataset.zip'"
    
    if errorlevel 1 (
        echo.
        echo ========================================================================
        echo AUTOMATIC DOWNLOAD FAILED
        echo ========================================================================
        echo.
        echo Please download manually:
        echo.
        echo Option 1 - UCI Repository (Official):
        echo   1. Open: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
        echo   2. Click "Download" button
        echo   3. Extract the ZIP file
        echo   4. Copy energydata_complete.csv to this folder:
        echo      %CD%
        echo.
        echo Option 2 - Kaggle (Alternative):
        echo   1. Open: https://www.kaggle.com/datasets/loveall/appliances-energy-prediction
        echo   2. Click "Download" button
        echo   3. Extract and copy energydata_complete.csv here
        echo.
        echo ========================================================================
        pause
        exit /b 1
    )
)

echo.
echo Extracting dataset.zip...
echo.

REM Extract using PowerShell
powershell -Command "Expand-Archive -Path 'dataset.zip' -DestinationPath '.' -Force"

if errorlevel 1 (
    echo Extraction failed. Please extract dataset.zip manually.
    pause
    exit /b 1
)

REM Find and copy the CSV file
if exist "energydata_complete.csv" (
    echo Dataset extracted successfully!
) else (
    echo Searching for CSV file in extracted folders...
    for /r %%f in (energydata_complete.csv) do (
        copy "%%f" "energydata_complete.csv"
        echo Found and copied: %%f
    )
)

REM Cleanup
if exist "dataset.zip" del dataset.zip
if exist "appliances energy prediction" rd /s /q "appliances energy prediction"

:skip_download

echo.
echo ========================================================================
echo VERIFYING DATASET
echo ========================================================================
echo.

if exist "energydata_complete.csv" (
    for %%A in (energydata_complete.csv) do (
        echo File: energydata_complete.csv
        echo Size: %%~zA bytes
        echo.
        echo Dataset is ready!
    )
    echo.
    echo ========================================================================
    echo SUCCESS!
    echo ========================================================================
    echo.
    echo The dataset is now available in:
    echo %CD%\energydata_complete.csv
    echo.
    echo You can now run: run_project.bat
    echo.
) else (
    echo ========================================================================
    echo DATASET NOT FOUND
    echo ========================================================================
    echo.
    echo Please download manually from:
    echo https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
    echo.
    echo Place energydata_complete.csv in:
    echo %CD%
    echo.
)

pause

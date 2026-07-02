@echo off
echo ============================================================
echo  Task 12 -- Binary Classification (Calibrated)
echo  PlaceMux Phase 1 Industry Immersion
echo ============================================================

cd /d "%~dp0"

echo.
echo [1/2] Training Calibrated Classifier (Task 12)...
python -m src.train_task12
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Training failed. Exiting.
    pause
    exit /b 1
)

echo.
echo [2/2] Launching Live Verification App (Gradio)...
python app.py

pause

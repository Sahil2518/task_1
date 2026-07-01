@echo off
echo ============================================================
echo  Task 11 -- Ensemble Learning
echo  PlaceMux Phase 1 Industry Immersion
echo ============================================================

cd /d "%~dp0"

echo.
echo [1/2] Training Ensemble Models...
python -m src.train_ensemble
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Training failed. Exiting.
    pause
    exit /b 1
)

echo.
echo [2/2] Launching Live Verification App (Gradio)...
python app.py

pause

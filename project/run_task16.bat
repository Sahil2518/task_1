@echo off
REM ─────────────────────────────────────────────────────────────────
REM  Task 16 — Model Validation & K-Fold
REM  PlaceMux Phase 1 Industry Immersion
REM  Run from the project root:  run_task16.bat
REM ─────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Task 16 ^| Model Validation ^& K-Fold
echo ============================================================
echo.

cd /d "%~dp0"

python -m src.train_task16

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Task 16 exited with error code %ERRORLEVEL%.
    echo         Check the output above for details.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   Task 16 complete. Results in logs\task16_kfold_results.json
echo   Plots saved to logs\task16_plots\
echo ============================================================
pause

@echo off
REM ─────────────────────────────────────────────────────────────────
REM  Task 17 — Hyperparameter Tuning (Advanced)
REM  PlaceMux Phase 1 Industry Immersion
REM  Run from the project root:  run_task17.bat
REM ─────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Task 17 ^| Hyperparameter Tuning (Advanced)
echo   Optuna TPE + MedianPruner, XGBoost + LightGBM
echo ============================================================
echo.

cd /d "%~dp0"

python -m src.train_task17

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Task 17 exited with error code %ERRORLEVEL%.
    echo         Check the output above for details.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   Task 17 complete.
echo   Results  -> logs\task17_results.json
echo   Plots    -> logs\task17_plots\
echo   DB log   -> logs\task17_optuna.db
echo   CSV log  -> logs\task17_all_trials.csv
echo ============================================================
pause

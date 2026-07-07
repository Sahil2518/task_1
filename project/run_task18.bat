@echo off
REM ─────────────────────────────────────────────────────────────────
REM  Task 18 — Natural Language Processing
REM  PlaceMux Phase 1 Industry Immersion
REM  Run from the project root:  run_task18.bat
REM ─────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Task 18 ^| Natural Language Processing
echo   Text Cleaning, TF-IDF, NLP Classification
echo ============================================================
echo.

cd /d "%~dp0"

python -m src.train_task18

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Task 18 exited with error code %ERRORLEVEL%.
    echo         Check the output above for details.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   Task 18 complete.
echo   Results  -> logs\task18_results.json
echo   Plots    -> logs\task18_plots\
echo   Model    -> models\nlp_pipeline.pkl
echo ============================================================
pause

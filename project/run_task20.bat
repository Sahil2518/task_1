@echo off
REM ─────────────────────────────────────────────────────────────────
REM  Task 20 — End-to-End Flask Deployment (Capstone)
REM  PlaceMux Phase 1 Industry Immersion
REM  Run from the project root:  run_task20.bat
REM ─────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Task 20 ^| End-to-End Flask Deployment
echo   Check Artifact · Start Flask · Run E2E Test
echo ============================================================
echo.

cd /d "%~dp0"

REM ── Step 1: Check for Task 19 Artifact ──
echo [Step 1/3] Checking for Task 19 artifact...
if not exist "models\task19\placemux_pipeline_v1.0.0.joblib" (
    echo   Artifact not found. Running train_task19.py to create it...
    python -m src.train_task19
    if %ERRORLEVEL% NEQ 0 (
        echo   [FATAL] Failed to create artifact. Exiting.
        pause
        exit /b %ERRORLEVEL%
    )
) else (
    echo   Artifact found. Proceeding.
)
echo.

REM ── Step 2: Start Flask App in Background ──
echo [Step 2/3] Starting Flask application...
start "Flask_Task20" cmd /c "python -m src.serve_task20"
echo   Flask server started in a new window on port 5020.
echo   Waiting a few seconds for startup...
timeout /t 5 /nobreak >nul
echo.

REM ── Step 3: Run End-to-End Tests and Latency Benchmark ──
echo [Step 3/3] Running End-to-End Test and Latency Benchmark...
python -m src.test_task20
set TEST_RESULT=%ERRORLEVEL%
echo.

if %TEST_RESULT% NEQ 0 (
    echo [ERROR] test_task20.py exited with error code %TEST_RESULT%.
    echo         The deployment failed verification.
    pause
    exit /b %TEST_RESULT%
)

REM ── Step 4: Package Deliverables ──
echo [Step 4/4] Packaging final deliverables into a ZIP archive...
python -m src.zip_task20
echo.

echo ============================================================
echo   Task 20 execution complete.
echo   See logs\task20_walkthrough.md for details.
echo   You can safely close the Flask server window now.
echo ============================================================
pause

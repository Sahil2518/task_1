@echo off
REM ─────────────────────────────────────────────────────────────────
REM  Task 19 — Application Model Serializing
REM  PlaceMux Phase 1 Industry Immersion
REM  Run from the project root:  run_task19.bat
REM ─────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   Task 19 ^| Application Model Serializing
echo   Bundle · Save · Metadata · Load-Test · FastAPI Stub
echo ============================================================
echo.

cd /d "%~dp0"

REM ── Step 1: Train and serialise the model artifact ──
echo [Step 1/2] Running train_task19.py ...
echo.
python -m src.train_task19

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] train_task19.py exited with error code %ERRORLEVEL%.
    echo         Check the output above for details.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   Serialization complete.
echo   Artifact  -^> models\task19\placemux_pipeline_v1.0.0.joblib
echo   Metadata  -^> models\task19\metadata.json
echo   Results   -^> logs\task19_results.json
echo ============================================================
echo.

REM ── Step 2: Launch FastAPI predict stub ──
echo [Step 2/2] Starting FastAPI prediction server on port 8019 ...
echo   Docs available at: http://127.0.0.1:8019/docs
echo   Press Ctrl+C to stop the server.
echo.
python -m uvicorn src.serve_task19:app --host 0.0.0.0 --port 8019 --reload

pause

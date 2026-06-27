@echo off
echo ============================================
echo  PlaceMux End-to-End ML Pipeline
echo  Task 8 - One Command Execution
echo ============================================
echo.

echo [1/3] Training pipeline (data -> features -> model -> evaluation)...
python -m src.train
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Training pipeline encountered an error.
    exit /b 1
)

echo.
echo [2/3] Running live verification (predict on unseen data)...
python -m src.predict
if %ERRORLEVEL% NEQ 0 (
    echo [FAILED] Live verification encountered an error.
    exit /b 1
)

echo.
echo [3/3] Pipeline complete. Artifacts saved:
echo   - models/pipeline.pkl   (unified sklearn Pipeline)
echo   - logs/metrics.json     (evaluation metrics)
echo   - logs/results.csv      (experiment log)
echo   - logs/confusion_matrix.png
echo   - logs/roc_curve.png
echo   - logs/pr_curve.png
echo ============================================
echo  ALL DONE - Pipeline finished successfully
echo ============================================

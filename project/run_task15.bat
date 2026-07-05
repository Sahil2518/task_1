@echo off
echo.
echo ===================================================
echo Running Task 15: K-Means Clustering
echo PlaceMux Phase 1 Industry Immersion
echo ===================================================
echo.

python -m src.train_task15

echo.
echo Done! Review the following outputs:
echo   logs/task15_cluster_analysis.json   -- Full results
echo   logs/task15_plots/                  -- Visualisations
echo.
pause

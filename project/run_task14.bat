@echo off
echo.
echo ===================================================
echo Running Task 14: Data Cluster Parameter Prep
echo ===================================================
python -m src.train_task14

echo.
echo Done! Please review logs/cluster_evaluation.png to verify the choice of k.
pause

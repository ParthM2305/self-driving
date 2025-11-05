@echo off
REM Full training script for CARLA dataset

echo ============================================================
echo Multimodal Self-Driving Model - Full Training
echo ============================================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Train with optimal settings for full dataset
echo Starting training with pointnet LiDAR mode...
echo.

python train.py ^
    --data-dir ./data ^
    --save-dir ./runs ^
    --lidar-mode pointnet ^
    --epochs 50 ^
    --batch-size 32 ^
    --lr 0.001 ^
    --weight-decay 0.0001 ^
    --scheduler plateau ^
    --workers 4 ^
    --pretrained ^
    --seed 42 ^
    --vis-every 5

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Training failed
    exit /b 1
)

echo.
echo ============================================================
echo Training completed successfully!
echo ============================================================
echo.
echo Check the runs/ directory for results, checkpoints, and visualizations.
echo.

pause

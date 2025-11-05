@echo off
REM Improved Training Script with Enhanced Techniques
REM This uses better hyperparameters and training strategies

echo ================================================
echo IMPROVED SELF-DRIVING MODEL TRAINING
echo ================================================
echo.
echo Improvements:
echo - Weighted loss (steering 2x, brake 1.5x)
echo - Gradient clipping (prevents explosions)
echo - Label smoothing (better generalization)
echo - Cosine annealing scheduler (better convergence)
echo - Early stopping (prevents overfitting)
echo - Data augmentation enabled
echo.
echo Starting training...
echo ================================================
echo.

call .venv\Scripts\activate.bat

python train_improved.py ^
    --data-dir ./data ^
    --save-dir ./runs ^
    --lidar-mode pointnet ^
    --epochs 100 ^
    --batch-size 32 ^
    --lr 0.001 ^
    --weight-decay 0.0001 ^
    --scheduler cosine ^
    --grad-clip 1.0 ^
    --label-smoothing 0.05 ^
    --early-stop-patience 15 ^
    --workers 4 ^
    --pretrained ^
    --seed 42

echo.
echo ================================================
echo Training Complete!
echo Check ./runs/improved_run_* for results
echo ================================================
pause

@echo off
REM Smoke test script for multimodal self-driving model
REM Tests the entire pipeline with a small dataset

echo ============================================================
echo Multimodal Self-Driving Model - Smoke Test
echo ============================================================
echo.

REM Check if virtual environment is activated
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please activate virtual environment.
    echo Run: .\venv\Scripts\Activate.ps1
    exit /b 1
)

echo [1/5] Preparing test dataset (100 samples)...
echo ============================================================
python prepare_data.py --data-dir ./CARLA_15GB/default --out ./data_smoke_test --max-samples 100 --debug
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Data preparation failed
    exit /b 1
)
echo.

echo [2/5] Verifying prepared data...
echo ============================================================
python prepare_data.py --verify --out ./data_smoke_test
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Data verification failed
    exit /b 1
)
echo.

echo [3/5] Running dataset sanity check...
echo ============================================================
python dataset.py --sanity --data ./data_smoke_test
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Dataset sanity check failed
    exit /b 1
)
echo.

echo [4/5] Training for 1 epoch (debug mode)...
echo ============================================================
python train.py --data-dir ./data_smoke_test --epochs 1 --batch-size 8 --workers 0 --debug --run-name smoke_test
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Training failed
    exit /b 1
)
echo.

echo [5/5] Running inference demo...
echo ============================================================
REM Find the latest checkpoint
for /f "delims=" %%i in ('dir /b /od /a-d "runs\smoke_test\checkpoints\*.pth" 2^>nul') do set CHECKPOINT=%%i
if not defined CHECKPOINT (
    echo ERROR: No checkpoint found
    exit /b 1
)

python inference_demo.py --checkpoint runs\smoke_test\checkpoints\%CHECKPOINT% --data-dir ./data_smoke_test --n-samples 3 --output-dir ./output_smoke_test
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Inference demo failed
    exit /b 1
)
echo.

echo ============================================================
echo Smoke test completed successfully!
echo ============================================================
echo.
echo Generated outputs:
echo - Prepared data: ./data_smoke_test/
echo - Training run: ./runs/smoke_test/
echo - Demo outputs: ./output_smoke_test/
echo.
echo Next steps:
echo 1. Check training logs: .\runs\smoke_test\training.log
echo 2. View demo outputs: .\output_smoke_test\
echo 3. Run full training: python train.py --data-dir ./data --epochs 25
echo.

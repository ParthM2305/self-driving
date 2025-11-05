@echo off
REM Quick launcher for common tasks
REM Usage: quick_run.bat [task]

if "%1"=="" goto show_help
if "%1"=="test" goto test_setup
if "%1"=="smoke" goto smoke_test
if "%1"=="prepare" goto prepare_data
if "%1"=="train" goto train
if "%1"=="eval" goto evaluate
if "%1"=="demo" goto demo
goto show_help

:show_help
echo ============================================================
echo Multimodal Self-Driving Model - Quick Launcher
echo ============================================================
echo.
echo Usage: quick_run.bat [task]
echo.
echo Available tasks:
echo   test      - Test setup and verify environment
echo   smoke     - Run smoke test (end-to-end validation)
echo   prepare   - Prepare dataset (with prompts)
echo   train     - Train model (with prompts)
echo   eval      - Evaluate model (with prompts)
echo   demo      - Run inference demo (with prompts)
echo.
echo Examples:
echo   quick_run.bat test
echo   quick_run.bat smoke
echo   quick_run.bat train
echo.
goto end

:test_setup
echo Running setup test...
python test_setup.py
goto end

:smoke_test
echo Running smoke test...
.\run_smoke_test.bat
goto end

:prepare_data
echo.
echo ============================================================
echo Data Preparation
echo ============================================================
echo.
set /p DATA_DIR="Enter data directory (default: ./CARLA_15GB/default): "
if "%DATA_DIR%"=="" set DATA_DIR=./CARLA_15GB/default

set /p OUT_DIR="Enter output directory (default: ./data): "
if "%OUT_DIR%"=="" set OUT_DIR=./data

set /p MAX_SAMPLES="Enter max samples per split (default: all, enter number to limit): "
if "%MAX_SAMPLES%"=="" (
    python prepare_data.py --data-dir %DATA_DIR% --out %OUT_DIR%
) else (
    python prepare_data.py --data-dir %DATA_DIR% --out %OUT_DIR% --max-samples %MAX_SAMPLES%
)
goto end

:train
echo.
echo ============================================================
echo Training
echo ============================================================
echo.
set /p DATA_DIR="Enter data directory (default: ./data): "
if "%DATA_DIR%"=="" set DATA_DIR=./data

set /p EPOCHS="Enter number of epochs (default: 25): "
if "%EPOCHS%"=="" set EPOCHS=25

set /p DEVICE="Enter device (cuda/cpu, default: auto): "
if "%DEVICE%"=="" set DEVICE=auto

set /p BATCH_SIZE="Enter batch size (default: auto): "
if "%BATCH_SIZE%"=="" (
    python train.py --data-dir %DATA_DIR% --epochs %EPOCHS% --device %DEVICE%
) else (
    python train.py --data-dir %DATA_DIR% --epochs %EPOCHS% --device %DEVICE% --batch-size %BATCH_SIZE%
)
goto end

:evaluate
echo.
echo ============================================================
echo Evaluation
echo ============================================================
echo.
set /p CHECKPOINT="Enter checkpoint path: "
if "%CHECKPOINT%"=="" (
    echo ERROR: Checkpoint path required
    goto end
)

set /p DATA_DIR="Enter data directory (default: ./data): "
if "%DATA_DIR%"=="" set DATA_DIR=./data

set /p SPLIT="Enter split to evaluate (train/validation/test, default: test): "
if "%SPLIT%"=="" set SPLIT=test

python evaluate.py --checkpoint %CHECKPOINT% --data-dir %DATA_DIR% --split %SPLIT%
goto end

:demo
echo.
echo ============================================================
echo Inference Demo
echo ============================================================
echo.
set /p CHECKPOINT="Enter checkpoint path: "
if "%CHECKPOINT%"=="" (
    echo ERROR: Checkpoint path required
    goto end
)

set /p DATA_DIR="Enter data directory (default: ./data): "
if "%DATA_DIR%"=="" set DATA_DIR=./data

set /p N_SAMPLES="Enter number of samples (default: 5): "
if "%N_SAMPLES%"=="" set N_SAMPLES=5

python inference_demo.py --checkpoint %CHECKPOINT% --data-dir %DATA_DIR% --n-samples %N_SAMPLES% --render-bev
goto end

:end
echo.
pause

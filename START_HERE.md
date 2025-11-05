# 🚀 READY TO RUN - Start Here!

## ⚡ Quick Start (5 Commands)

Copy and paste these commands one by one in PowerShell:

### 1️⃣ Install Dependencies (2-5 minutes)
```powershell
pip install -r requirements.txt
```

### 2️⃣ Test Your Setup (30 seconds)
```powershell
python test_setup.py
```
✅ **Expected**: All tests pass, shows GPU/CPU info

### 3️⃣ Run Smoke Test (5-10 minutes)
```powershell
.\run_smoke_test.bat
```
✅ **Expected**: Creates `data_smoke_test/`, `runs/smoke_test/`, `output_smoke_test/`

### 4️⃣ Prepare Full Dataset (10-30 minutes)
```powershell
python prepare_data.py --data-dir ./CARLA_15GB/default --out ./data --max-samples 10000
```
✅ **Expected**: Creates `data/` with images, lidar, and CSV files

### 5️⃣ Train Model (1-2 hours on GPU, 10+ hours on CPU)
```powershell
# GPU (recommended)
python train.py --data-dir ./data --epochs 25 --batch-size 64 --device cuda

# OR CPU (slower)
python train.py --data-dir ./data --epochs 10 --batch-size 8 --device cpu --num-points 1024
```
✅ **Expected**: Creates `runs/run_*/checkpoints/checkpoint_epoch025_best.pth`

---

## 📊 After Training

### Evaluate Model
```powershell
python evaluate.py --checkpoint runs\run_*\checkpoints\checkpoint_epoch025_best.pth --data-dir ./data --split test
```
✅ **Output**: `evaluation_results/metrics.json` and plots

### Run Demo
```powershell
python inference_demo.py --checkpoint runs\run_*\checkpoints\checkpoint_epoch025_best.pth --data-dir ./data --n-samples 10
```
✅ **Output**: `demo_outputs/sample_*.png` with predictions

---

## 🎯 Interactive Helper

Use the interactive launcher:
```powershell
.\quick_run.bat
```

Choose from:
- `quick_run.bat test` - Test setup
- `quick_run.bat smoke` - Smoke test
- `quick_run.bat prepare` - Prepare data (interactive prompts)
- `quick_run.bat train` - Train model (interactive prompts)
- `quick_run.bat eval` - Evaluate (interactive prompts)
- `quick_run.bat demo` - Run demo (interactive prompts)

---

## 📁 What You'll Get

After running the full pipeline:

```
Your Project/
├── data/                          # Prepared dataset
│   ├── images/
│   ├── lidar/
│   └── *_index.csv
│
├── runs/run_20231105_*/          # Training run
│   ├── checkpoints/
│   │   └── checkpoint_epoch025_best.pth  ← Your trained model
│   ├── visualizations/
│   ├── training.log
│   └── training_curves.png
│
├── evaluation_results/            # Evaluation outputs
│   ├── metrics.json              ← Performance metrics
│   ├── predictions.csv
│   ├── scatter_plots.png
│   └── error_distributions.png
│
└── demo_outputs/                  # Demo visualizations
    ├── sample_0000.png
    ├── sample_0001.png
    └── ...
```

---

## 💡 Tips for Success

### First Time Users
1. **Always start with smoke test**: `.\run_smoke_test.bat`
2. **Use small dataset first**: `--max-samples 5000`
3. **Check GPU**: Run `python test_setup.py` to see GPU info
4. **Monitor training**: Check `runs/*/training.log`

### GPU Users
- Use `--batch-size 64` or higher
- Enable mixed precision (on by default)
- Use PointNet mode for best accuracy
- Expected training time: 1-2 hours for 25 epochs

### CPU Users
- Use `--batch-size 8` 
- Use `--num-points 1024` (instead of 4096)
- Use `--lidar-mode bev` (faster than PointNet)
- Expected training time: 10-20 hours for 10 epochs
- Consider training on a subset: `--max-samples 2000`

### Troubleshooting
| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch-size` and `--num-points` |
| Slow training | Use GPU or reduce dataset size |
| Import errors | Run `pip install -r requirements.txt --force-reinstall` |
| No parquet files | Check `CARLA_15GB/default/partial-train/` exists |

---

## 📖 Documentation

- **README.md** - Full documentation
- **next_steps.txt** - Detailed guide with examples
- **data/README.md** - Data format specification
- **PROJECT_SUMMARY.md** - Technical details

---

## ✅ Success Indicators

You know it's working when:

✅ `test_setup.py` shows all tests passed
✅ Smoke test completes without errors
✅ Training log shows decreasing loss
✅ Validation loss improves over epochs
✅ Evaluation shows R² > 0.7 for steering
✅ Demo images show green/red arrows

---

## 🎓 Expected Performance

After 25 epochs on 10K samples:

| Control | Target MAE | Target R² |
|---------|------------|-----------|
| Steering | < 0.10 | > 0.70 |
| Throttle | < 0.08 | > 0.65 |
| Brake | < 0.06 | > 0.60 |

Better results with more data and longer training!

---

## 🆘 Need Help?

1. Check `README.md` for detailed instructions
2. Check `next_steps.txt` for common issues
3. Review `training.log` for error messages
4. Make sure all dependencies are installed
5. Verify data directory structure matches `data/README.md`

---

## 🎉 You're All Set!

**Start with Step 1 above and follow in order.**

The complete system is ready to:
- ✅ Process your CARLA dataset
- ✅ Train a multimodal deep learning model
- ✅ Evaluate performance with metrics
- ✅ Visualize predictions

**Good luck with your autonomous driving research!** 🚗💨

---

*For advanced usage and customization, see README.md*

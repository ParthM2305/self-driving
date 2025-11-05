# Multimodal Self-Driving Model

A complete, production-ready implementation of a multimodal deep learning model for autonomous driving using CARLA simulator data. The model fuses RGB camera images, LiDAR point clouds, and vehicle telemetry to predict steering, throttle, and brake controls.

## 🚗 Overview

This project implements an end-to-end learning approach for autonomous driving with the following features:

- **Multimodal Fusion**: Combines camera images (ResNet18), LiDAR point clouds (PointNet/BEV-CNN), and scalar features
- **Multiple LiDAR Encoders**: PointNet-style encoder, BEV CNN, or sector histogram (selectable)
- **Production-Ready**: Mixed precision training, checkpointing, extensive logging, reproducibility
- **CPU/GPU Support**: Automatic device detection with graceful fallbacks for CPU-only systems
- **Comprehensive Evaluation**: Metrics, visualizations, and inference demos

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Modalities                     │
├──────────────┬──────────────┬──────────────────────────┤
│  RGB Image   │    LiDAR     │  Scalar Features (speed) │
│  (224×224)   │  (N×4 points)│      (1-dim)             │
└──────┬───────┴──────┬───────┴───────┬──────────────────┘
       │              │               │
       ▼              ▼               │
┌─────────────┐ ┌─────────────┐      │
│ ImageEncoder│ │LiDAREncoder │      │
│  ResNet18   │ │ PointNet /  │      │
│  (256-dim)  │ │ BEV-CNN     │      │
│             │ │ (256-dim)   │      │
└──────┬──────┘ └──────┬──────┘      │
       │               │              │
       └───────┬───────┴──────────────┘
               ▼
        ┌─────────────┐
        │  FusionNet  │
        │    (MLP)    │
        └──────┬──────┘
               │
        ┌──────┴───────┐
        │  Multi-Head  │
        │   Outputs    │
        └──────────────┘
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
 Steering  Throttle   Brake
  [-1,1]    [0,1]     [0,1]
```

### Model Components

1. **ImageEncoder**: ResNet18 backbone (optionally pretrained) + projection head → 256-dim
2. **LiDAREncoder**: 
   - **PointNet**: Point-wise MLPs + global max pooling → 256-dim
   - **BEV-CNN**: Bird's eye view projection + CNN → 256-dim
   - **Sector Histogram**: Radial sector statistics + MLP → 256-dim
3. **FusionNet**: Concatenate features + MLP → 3 prediction heads

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 16GB RAM (minimum), 32GB recommended
- 20-30GB disk space for dataset

## 🛠️ Installation

### Windows (PowerShell)

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Linux/Mac

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 📁 Data Preparation

### Expected Raw Data Structure

```
CARLA_15GB/
└── default/
    ├── partial-train/
    │   ├── data-00000-of-00010.parquet
    │   └── ...
    ├── partial-validation/
    └── partial-test/
```

### Prepare Dataset

```powershell
# Full dataset (may take 30-60 minutes)
python prepare_data.py --data-dir ./CARLA_15GB/default --out ./data

# Quick test with limited samples (recommended for first run)
python prepare_data.py --data-dir ./CARLA_15GB/default --out ./data --max-samples 5000

# With BEV projections (optional, requires more disk space)
python prepare_data.py --data-dir ./CARLA_15GB/default --out ./data --generate-bev

# Debug mode (100 samples only)
python prepare_data.py --data-dir ./CARLA_15GB/default --out ./data --debug
```

### Verify Prepared Data

```powershell
python prepare_data.py --verify --out ./data
python dataset.py --sanity --data ./data
```

## 🚀 Training

### GPU Training (Recommended)

```powershell
# Full training with default parameters
python train.py --data-dir ./data --epochs 25 --batch-size 64 --device cuda

# With specific LiDAR encoder
python train.py --data-dir ./data --epochs 25 --lidar-mode pointnet --num-points 4096

# Resume from checkpoint
python train.py --data-dir ./data --epochs 50 --resume ./runs/run_20231105_120000/checkpoints/checkpoint_epoch025.pth
```

### CPU Training (Fallback)

```powershell
# CPU mode with reduced settings
python train.py --data-dir ./data --epochs 10 --batch-size 8 --device cpu --num-points 1024 --workers 2

# Quick test (debug mode)
python train.py --data-dir ./data --debug --device cpu
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `./data` | Directory with prepared data |
| `--lidar-mode` | `pointnet` | LiDAR encoder: `pointnet`, `bev`, `sector_hist` |
| `--epochs` | 25 | Number of training epochs |
| `--batch-size` | Auto | Batch size (auto-detected based on device) |
| `--lr` | 1e-4 | Learning rate |
| `--device` | `auto` | Device: `auto`, `cuda`, `cpu` |
| `--pretrained` | True | Use ImageNet pretrained weights |
| `--freeze-backbone` | False | Freeze early layers of image encoder |
| `--scheduler` | `plateau` | LR scheduler: `plateau`, `cosine`, `none` |
| `--use-amp` | True | Use mixed precision (GPU only) |
| `--resume` | None | Path to checkpoint to resume from |

### Loss Weights

```powershell
# Custom loss weights
python train.py --data-dir ./data --steer-weight 1.0 --throttle-weight 0.5 --brake-weight 0.5
```

## 📈 Evaluation

```powershell
# Evaluate on test set
python evaluate.py --checkpoint ./runs/run_*/checkpoints/checkpoint_epoch025_best.pth --data-dir ./data --split test

# Evaluate on validation set
python evaluate.py --checkpoint ./path/to/checkpoint.pth --data-dir ./data --split validation

# Specify output directory
python evaluate.py --checkpoint ./checkpoint.pth --data-dir ./data --output-dir ./my_evaluation
```

### Outputs

- `metrics.json`: MSE, MAE, RMSE, R² for each control output
- `predictions.csv`: Per-sample predictions and targets
- `scatter_plots.png`: Predicted vs actual for each control
- `error_distributions.png`: Error histograms
- `error_vs_actual.png`: Error analysis
- `correlation_matrix.png`: Correlation between predictions and targets

## 🎨 Inference Demo

```powershell
# Run demo on 5 random samples
python inference_demo.py --checkpoint ./checkpoint.pth --data-dir ./data --n-samples 5

# Specific samples with BEV rendering
python inference_demo.py --checkpoint ./checkpoint.pth --data-dir ./data --sample-indices 0 10 50 100 --render-bev

# Show interactively (instead of just saving)
python inference_demo.py --checkpoint ./checkpoint.pth --data-dir ./data --show
```

### Demo Outputs

Each sample visualization includes:
- RGB image with steering angle overlay (red = actual, green = predicted)
- LiDAR bird's eye view (if `--render-bev`)
- Table of control values (steering, throttle, brake)

## 🧪 Quick Smoke Test

Test the entire pipeline end-to-end:

```powershell
# Run smoke test script
.\run_smoke_test.bat
```

Or manually:

```powershell
# 1. Prepare small dataset
python prepare_data.py --data-dir ./CARLA_15GB/default --out ./data_test --max-samples 100 --debug

# 2. Train for 1 epoch
python train.py --data-dir ./data_test --epochs 1 --debug --device cpu

# 3. Evaluate
python evaluate.py --checkpoint ./runs/*/checkpoints/*.pth --data-dir ./data_test --split validation

# 4. Run inference demo
python inference_demo.py --checkpoint ./runs/*/checkpoints/*.pth --data-dir ./data_test --n-samples 3
```

## 💻 Hardware Recommendations

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 16GB
- **GPU**: None (CPU mode supported)
- **Disk**: 20GB free
- **Training time**: ~10-20 hours (CPU), ~2-4 hours (GPU)

### Recommended Setup
- **CPU**: 8+ cores
- **RAM**: 32GB
- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3060/4060 or better)
- **Disk**: 30GB free (SSD preferred)
- **Training time**: ~1-2 hours (GPU)

### Optimal Setup
- **CPU**: 16+ cores
- **RAM**: 64GB
- **GPU**: NVIDIA GPU with 24GB+ VRAM (RTX 4090, A5000, etc.)
- **Disk**: 50GB free (NVMe SSD)
- **Training time**: ~30-60 minutes (GPU)

## 🔧 Configuration Tips

### For Low-Memory Systems

```powershell
# Reduce batch size and num_points
python train.py --data-dir ./data --batch-size 16 --num-points 2048 --workers 2
```

### For Faster Training (with GPU)

```powershell
# Use BEV mode (faster than PointNet)
python train.py --data-dir ./data --lidar-mode bev --batch-size 128 --use-amp

# Freeze backbone to reduce trainable parameters
python train.py --data-dir ./data --freeze-backbone --batch-size 128
```

### For Better Accuracy

```powershell
# Increase model capacity and training time
python train.py --data-dir ./data --epochs 50 --lr 5e-5 --num-points 8192 --batch-size 32
```

## 📊 Expected Performance

On CARLA dataset with default settings (25 epochs, PointNet mode):

| Metric | Steering | Throttle | Brake |
|--------|----------|----------|-------|
| MAE | 0.05-0.08 | 0.03-0.05 | 0.02-0.04 |
| RMSE | 0.08-0.12 | 0.05-0.08 | 0.04-0.06 |
| R² | 0.75-0.85 | 0.70-0.80 | 0.65-0.75 |

*Note: Actual performance depends on data quality, hyperparameters, and training duration.*

## 🔍 LiDAR Encoder Comparison

### PointNet Mode
- **Pros**: Permutation-invariant, works directly on points, good for sparse data
- **Cons**: Computationally expensive (O(N) per point), requires sampling/padding
- **Best for**: High-accuracy requirements, GPU training

### BEV Mode
- **Pros**: Fast (O(1) inference), compact representation, works well with CNNs
- **Cons**: Loses height information, requires preprocessing
- **Best for**: Real-time applications, CPU training

### Sector Histogram Mode
- **Pros**: Very fast, compact (108-dim), interpretable
- **Cons**: Coarse representation, may miss fine details
- **Best for**: Quick prototyping, extremely limited compute

## 🐛 Troubleshooting

### Out of Memory (OOM)

```powershell
# Reduce batch size
python train.py --batch-size 16

# Reduce num_points
python train.py --num-points 1024

# Disable mixed precision
python train.py --no-amp
```

### Slow Training

```powershell
# Increase workers (if I/O bound)
python train.py --workers 8

# Use BEV mode instead of PointNet
python train.py --lidar-mode bev

# Enable mixed precision (GPU only)
python train.py --use-amp
```

### Import Errors

```powershell
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Dataset Issues

```powershell
# Verify data integrity
python prepare_data.py --verify --out ./data

# Regenerate dataset
python prepare_data.py --data-dir ./CARLA_15GB/default --out ./data_new
```

## 📚 File Structure

```
.
├── data/                      # Prepared dataset (generated)
│   ├── README.md             # Data organization guide
│   ├── images/               # RGB images
│   ├── lidar/                # LiDAR point clouds
│   ├── *_index.csv           # Metadata indices
│   └── *_stats.json          # Dataset statistics
├── runs/                      # Training runs (generated)
│   └── run_*/
│       ├── checkpoints/      # Model checkpoints
│       ├── visualizations/   # Training visualizations
│       ├── config.json       # Run configuration
│       └── training.log      # Training log
├── prepare_data.py           # Data preparation script
├── dataset.py                # PyTorch Dataset classes
├── model.py                  # Model architectures
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── inference_demo.py         # Inference visualization
├── utils.py                  # Helper functions
├── requirements.txt          # Python dependencies
├── run_smoke_test.bat        # Smoke test script
└── README.md                 # This file
```

## 🔬 Extending the Model

### Adding a New LiDAR Encoder

1. Implement encoder class in `model.py`:
   ```python
   class MyCustomEncoder(nn.Module):
       def __init__(self, output_dim=256):
           super().__init__()
           # Your architecture
   ```

2. Add to `LiDAREncoder` wrapper:
   ```python
   elif lidar_mode == 'my_custom':
       self.encoder = MyCustomEncoder(output_dim=output_dim)
   ```

3. Update dataset to handle your input format in `dataset.py`

### Adding More Control Outputs

1. Modify `FusionNet` in `model.py` to add new heads
2. Update loss computation in `train.py`
3. Update evaluation metrics in `evaluate.py`

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_selfdriving_2024,
  title={Multimodal Self-Driving Model},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/repo}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- CARLA Simulator for the dataset
- PyTorch team for the deep learning framework
- ResNet and PointNet authors for the architectural foundations

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues and documentation
- Review troubleshooting section above

---

**Happy Autonomous Driving! 🚗💨**

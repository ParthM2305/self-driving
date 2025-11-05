"""
Quick test script to verify model architecture and basic functionality.
Run this before training to ensure everything is set up correctly.
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("="*70)
print("Self-Driving Model - Quick Architecture Test")
print("="*70)

# Test 1: Import all modules
print("\n[1/5] Testing module imports...")
try:
    import model
    import dataset
    import utils
    from train import Trainer
    from evaluate import Evaluator
    print("  ✓ All modules imported successfully")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check PyTorch and CUDA
print("\n[2/5] Checking PyTorch setup...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  Running on CPU (training will be slower)")

# Test 3: Test model architectures
print("\n[3/5] Testing model architectures...")

batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # Test ImageEncoder
    from model import ImageEncoder
    img_encoder = ImageEncoder(pretrained=False, output_dim=256).to(device)
    dummy_img = torch.randn(batch_size, 3, 224, 224).to(device)
    img_feat = img_encoder(dummy_img)
    assert img_feat.shape == (batch_size, 256), f"ImageEncoder output shape mismatch: {img_feat.shape}"
    print(f"  ✓ ImageEncoder: {dummy_img.shape} → {img_feat.shape}")
    
    # Test PointNetEncoder
    from model import PointNetEncoder
    pointnet = PointNetEncoder(input_dim=4, output_dim=256).to(device)
    dummy_points = torch.randn(batch_size, 1024, 4).to(device)
    point_feat = pointnet(dummy_points)
    assert point_feat.shape == (batch_size, 256), f"PointNet output shape mismatch: {point_feat.shape}"
    print(f"  ✓ PointNetEncoder: {dummy_points.shape} → {point_feat.shape}")
    
    # Test BEVCNNEncoder
    from model import BEVCNNEncoder
    bev_encoder = BEVCNNEncoder(input_channels=1, output_dim=256).to(device)
    dummy_bev = torch.randn(batch_size, 1, 128, 128).to(device)
    bev_feat = bev_encoder(dummy_bev)
    assert bev_feat.shape == (batch_size, 256), f"BEV encoder output shape mismatch: {bev_feat.shape}"
    print(f"  ✓ BEVCNNEncoder: {dummy_bev.shape} → {bev_feat.shape}")
    
    # Test FusionNet
    from model import FusionNet
    fusion = FusionNet(image_dim=256, lidar_dim=256, scalar_dim=1).to(device)
    dummy_scalars = torch.randn(batch_size, 1).to(device)
    steer, throttle, brake = fusion(img_feat, point_feat, dummy_scalars)
    assert steer.shape == (batch_size, 1), f"Steer output shape mismatch: {steer.shape}"
    print(f"  ✓ FusionNet: ({img_feat.shape}, {point_feat.shape}, {dummy_scalars.shape}) → steer{steer.shape}")
    
    # Test complete model
    from model import MultimodalDrivingModel
    model = MultimodalDrivingModel(lidar_mode='pointnet', pretrained_image=False).to(device)
    outputs = model(dummy_img, dummy_points, dummy_scalars)
    assert 'steer' in outputs and 'throttle' in outputs and 'brake' in outputs
    print(f"  ✓ MultimodalDrivingModel: Full forward pass successful")
    print(f"    Output keys: {list(outputs.keys())}")
    print(f"    Steer range: [{outputs['steer'].min():.3f}, {outputs['steer'].max():.3f}]")
    print(f"    Throttle range: [{outputs['throttle'].min():.3f}, {outputs['throttle'].max():.3f}]")
    print(f"    Brake range: [{outputs['brake'].min():.3f}, {outputs['brake'].max():.3f}]")
    
except Exception as e:
    print(f"  ✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test utilities
print("\n[4/5] Testing utility functions...")
try:
    from utils import set_seed, get_device, AverageMeter, count_parameters
    
    # Test seed setting
    set_seed(42)
    r1 = torch.rand(1).item()
    set_seed(42)
    r2 = torch.rand(1).item()
    assert r1 == r2, "Seed setting not working"
    print(f"  ✓ Seed setting works (reproducible random: {r1:.6f})")
    
    # Test device detection
    device = get_device('auto')
    print(f"  ✓ Device detection: {device}")
    
    # Test parameter counting
    param_count = count_parameters(model)
    print(f"  ✓ Parameter counting: {param_count:,} trainable parameters")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    assert abs(meter.avg - 4.5) < 1e-6, "AverageMeter not working"
    print(f"  ✓ AverageMeter works (avg of 0-9 = {meter.avg})")
    
except Exception as e:
    print(f"  ✗ Utility test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Memory usage estimate
print("\n[5/5] Estimating memory requirements...")
try:
    import psutil
    
    # Current memory usage
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"  Current process memory: {mem_mb:.1f} MB")
    
    # Model size estimate
    param_size_mb = param_count * 4 / 1024 / 1024  # 4 bytes per float32
    print(f"  Model size (FP32): ~{param_size_mb:.1f} MB")
    
    # Recommended settings
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_mem_gb < 6:
            print(f"  ⚠ GPU Memory ({gpu_mem_gb:.1f} GB) is limited")
            print(f"    Recommended: --batch-size 16 --num-points 2048")
        elif gpu_mem_gb < 12:
            print(f"  ✓ GPU Memory ({gpu_mem_gb:.1f} GB) is adequate")
            print(f"    Recommended: --batch-size 32 --num-points 4096")
        else:
            print(f"  ✓ GPU Memory ({gpu_mem_gb:.1f} GB) is excellent")
            print(f"    Recommended: --batch-size 64 --num-points 4096")
    else:
        print(f"  ⚠ No GPU detected - training will be slow")
        print(f"    Recommended: --batch-size 8 --num-points 1024 --device cpu")
    
except ImportError:
    print("  ℹ psutil not installed, skipping memory check")
    print("    Install with: pip install psutil")
except Exception as e:
    print(f"  ℹ Memory check failed: {e}")

# Final summary
print("\n" + "="*70)
print("All tests passed! ✓")
print("="*70)
print("\nYour environment is ready for training!")
print("\nNext steps:")
print("  1. Prepare data: python prepare_data.py --data-dir ./CARLA_15GB/default --out ./data")
print("  2. Run smoke test: .\\run_smoke_test.bat")
print("  3. Train model: python train.py --data-dir ./data --epochs 25")
print("\nFor detailed instructions, see README.md and next_steps.txt")
print("="*70)

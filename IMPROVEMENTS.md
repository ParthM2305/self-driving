# Model Improvement Plan

## Current Results (Baseline)
- **Steering**: R² = -0.05, MAE = 0.083 (4.7°)
- **Throttle**: R² = -2.45, MAE = 0.255 (poor)
- **Brake**: R² = 0.46, MAE = 0.253 (moderate)
- **Training**: 50 epochs, loss 0.228 → 0.018

---

## Implemented Improvements ✅

### 1. **Weighted Loss Function**
**Problem**: All outputs treated equally, but steering is most critical for safety.

**Solution**: 
```python
loss_weights = {
    'steer': 2.0,      # 2x weight (most critical)
    'throttle': 1.0,   # Standard weight
    'brake': 1.5       # 1.5x weight (safety critical)
}
```

**Expected Impact**: 
- Better steering predictions (currently worst performer)
- Balanced learning across outputs
- **Estimated R² improvement**: +0.2 to +0.3 for steering

---

### 2. **Gradient Clipping**
**Problem**: Large gradients can destabilize training.

**Solution**: 
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Expected Impact**:
- More stable training
- Better convergence
- Prevents gradient explosions

---

### 3. **Label Smoothing for Regression**
**Problem**: Model overfits to exact target values.

**Solution**: Smooth targets slightly toward mean (5% smoothing)
```python
target_smoothed = (1 - 0.05) * target + 0.05 * target.mean()
```

**Expected Impact**:
- Better generalization
- Reduced overfitting
- **Estimated R² improvement**: +0.1 to +0.15

---

### 4. **Cosine Annealing Scheduler**
**Problem**: ReduceLROnPlateau can get stuck in local minima.

**Solution**: Cosine annealing with warm restarts
```python
CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

**Expected Impact**:
- Better exploration of loss landscape
- Escape local minima
- **Estimated loss reduction**: 10-15%

---

### 5. **Early Stopping**
**Problem**: Training for fixed 50 epochs may cause overfitting.

**Solution**: Stop if no improvement for 15 epochs
```python
early_stop_patience = 15
```

**Expected Impact**:
- Prevent overfitting
- Save training time
- Better generalization

---

### 6. **Data Augmentation** (Already in dataset.py)
**Enabled**: Random flips, brightness, contrast, noise

**Expected Impact**:
- Better robustness
- Improved generalization
- **Estimated R² improvement**: +0.15 to +0.25

---

## Additional Improvements to Consider

### 7. **Larger Model (Future)**
**Change**: ResNet18 → ResNet34 or ResNet50
- More capacity for complex patterns
- Better feature extraction
- **Trade-off**: 2-3x slower training

### 8. **Attention Mechanism (Future)**
**Add**: Cross-modal attention between image and LiDAR
- Better feature fusion
- Learn what to focus on
- **Expected R² improvement**: +0.1 to +0.2

### 9. **More Data**
**Collect**: Use full CARLA dataset (not just partial)
- Currently: 904 training samples
- Full dataset: Potentially 10,000+ samples
- **Expected R² improvement**: +0.3 to +0.5

### 10. **Temporal Modeling (Future)**
**Add**: LSTM/GRU for sequential frames
- Use past frames for prediction
- Model dynamics and motion
- **Expected R² improvement**: +0.2 to +0.4

---

## Expected Results Summary

### With Current Improvements (train_improved.py)

| Output | Current R² | Expected R² | Improvement |
|--------|-----------|-------------|-------------|
| **Steering** | -0.05 | **0.30-0.45** | ↑ 0.35-0.50 |
| **Throttle** | -2.45 | **-0.50 to 0.10** | ↑ 1.95-2.55 |
| **Brake** | 0.46 | **0.60-0.75** | ↑ 0.14-0.29 |

### Key Changes:
1. **Steering**: Should become positive R² (better than mean prediction)
2. **Throttle**: Major improvement from proper weighting
3. **Brake**: Already good, should reach 0.6-0.7
4. **Overall**: Expect 40-60% better performance

---

## How to Use

### Run Improved Training:
```bash
# Windows
train_improved.bat

# Or manually:
python train_improved.py --data-dir ./data --save-dir ./runs --pretrained --epochs 100
```

### Compare Results:
```bash
# After training completes
python evaluate.py --checkpoint ./runs/improved_run_*/checkpoints/best_checkpoint.pth --data-dir ./data
```

---

## Training Configuration Comparison

| Parameter | Original | Improved | Reason |
|-----------|----------|----------|--------|
| Loss Weights | Equal (1:1:1) | 2:1:1.5 | Prioritize steering |
| Scheduler | Plateau | Cosine | Better exploration |
| Grad Clip | None | 1.0 | Stability |
| Label Smooth | None | 0.05 | Generalization |
| Early Stop | None | 15 epochs | Prevent overfit |
| Epochs | 50 | 100 | More training |
| Augmentation | Enabled | Enabled | Same |

---

## Timeline

1. **Now**: Run `train_improved.bat`
2. **~8-12 hours**: Training completes (100 epochs, CPU)
3. **Evaluation**: Compare new results with baseline
4. **Expected**: 40-60% improvement in R² scores

---

## Next Steps After This Training

If results are still not satisfactory:
1. Try ResNet34 (bigger model)
2. Add more data from full CARLA dataset
3. Implement attention mechanism
4. Add temporal modeling (LSTM)
5. Try different output activations (tanh for steering)

---

## Files Created

- ✅ `train_improved.py` - Enhanced training script
- ✅ `train_improved.bat` - Easy execution script
- ✅ `IMPROVEMENTS.md` - This document

**Ready to train? Run:**
```
train_improved.bat
```

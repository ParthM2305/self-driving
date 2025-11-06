"""
Generate visualizations for the 84.28% baseline model results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix

# Configuration
EVAL_DIR = Path("evaluation_results_baseline_test")
OUTPUT_DIR = Path("baseline_84pct_visualizations")
TRAINING_DIR = Path("runs/improved_run_20251106_002053")

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Load data
print("Loading data...")
with open(TRAINING_DIR / "training_history.json", 'r') as f:
    history = json.load(f)

with open(EVAL_DIR / "comprehensive_metrics.json", 'r') as f:
    metrics = json.load(f)

predictions_df = pd.read_csv(EVAL_DIR / "predictions.csv")

# Define output names and colors
outputs = ['steer', 'throttle', 'brake']
outputs_labels = ['Steering', 'Throttle', 'Brake']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

print("Generating visualizations...")

# ============================================================================
# 1. Training Loss Curve
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

train_losses = [epoch['total'] for epoch in history['train']]
val_losses = [epoch['total'] for epoch in history['val']]
epochs = list(range(1, len(train_losses) + 1))
best_epoch = np.argmin(val_losses) + 1

ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
ax.axvline(x=best_epoch, color='green', linestyle='--', 
           label=f'Best Model (Epoch {best_epoch})', linewidth=1.5)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_training_loss_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Training loss curve")

# ============================================================================
# 2. Per-Output Loss Curves
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (output, label, color) in enumerate(zip(outputs, outputs_labels, colors)):
    ax = axes[idx]
    
    train_losses = [epoch[output] for epoch in history['train']]
    val_losses = [epoch[output] for epoch in history['val']]
    
    ax.plot(epochs, train_losses, color=color, linestyle='-', 
            label='Training', linewidth=2, alpha=0.7)
    ax.plot(epochs, val_losses, color=color, linestyle='--', 
            label='Validation', linewidth=2)
    ax.axvline(x=best_epoch, color='green', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{label} Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_per_output_loss_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Per-output loss curves")

# ============================================================================
# 3. Metrics Comparison
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MSE
ax = axes[0, 0]
mse_values = [metrics[output]['mse'] for output in outputs]
bars = ax.bar(outputs_labels, mse_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Mean Squared Error by Output')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mse_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom')

# MAE
ax = axes[0, 1]
mae_values = [metrics[output]['mae'] for output in outputs]
bars = ax.bar(outputs_labels, mae_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Mean Absolute Error by Output')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mae_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom')

# R²
ax = axes[1, 0]
r2_values = [metrics[output]['r2'] for output in outputs]
bars = ax.bar(outputs_labels, r2_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('R² Score')
ax.set_title('R² Score by Output')
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Fit')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}', ha='center', va='bottom')

# Accuracy
ax = axes[1, 1]
acc_values = [metrics[output]['accuracy_10pct'] for output in outputs]
bars = ax.bar(outputs_labels, acc_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy (±10%) by Output')
ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Target')
ax.grid(True, alpha=0.3, axis='y')
ax.legend()
for bar, val in zip(bars, acc_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "3_metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Metrics comparison")

# ============================================================================
# 4. Prediction vs Actual Scatter Plots
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (output, label, color) in enumerate(zip(outputs, outputs_labels, colors)):
    ax = axes[idx]
    
    actual = predictions_df[f'{output}_actual'].values
    predicted = predictions_df[f'{output}_pred'].values
    
    ax.scatter(actual, predicted, alpha=0.5, s=20, color=color, 
               edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', 
            linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = metrics[output]['r2']
    
    ax.set_xlabel(f'Actual {label}')
    ax.set_ylabel(f'Predicted {label}')
    ax.set_title(f'{label}: Prediction vs Actual (R²={r2:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4_prediction_vs_actual.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Prediction vs actual scatter")

# ============================================================================
# 5. Error Distributions
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (output, label, color) in enumerate(zip(outputs, outputs_labels, colors)):
    ax = axes[idx]
    
    actual = predictions_df[f'{output}_actual'].values
    predicted = predictions_df[f'{output}_pred'].values
    errors = predicted - actual
    
    n, bins, patches = ax.hist(errors, bins=50, color=color, alpha=0.7, edgecolor='black')
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    
    ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.4f}')
    ax.axvline(median_error, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_error:.4f}')
    ax.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero Error')
    
    ax.set_xlabel(f'{label} Error (Predicted - Actual)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{label} Error Distribution (σ={std_error:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "5_error_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Error distributions")

# ============================================================================
# 6. Performance Summary
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

summary_text = f"""
═══════════════════════════════════════════════════════════════════
         BASELINE MODEL - PERFORMANCE SUMMARY (84.28%)
═══════════════════════════════════════════════════════════════════

OVERALL METRICS
───────────────────────────────────────────────────────────────────
  Overall Accuracy (±10%):     {metrics['overall']['accuracy_10pct']:.2f}%
  Mean Squared Error:          {metrics['overall']['mse']:.6f}
  Mean Absolute Error:         {metrics['overall']['mae']:.6f}
  R² Score:                    {metrics['overall']['r2']:.6f}
  Correlation:                 {metrics['overall']['correlation']:.6f}

STEERING CONTROL
───────────────────────────────────────────────────────────────────
  Accuracy (±10%):             {metrics['steer']['accuracy_10pct']:.2f}%
  Accuracy (±5%):              {metrics['steer']['accuracy_5pct']:.2f}%
  Accuracy (±20%):             {metrics['steer']['accuracy_20pct']:.2f}%
  Mean Absolute Error:         {metrics['steer']['mae']:.6f}
  Median Absolute Error:       {metrics['steer']['median_error']:.6f}
  R² Score:                    {metrics['steer']['r2']:.6f}
  Correlation:                 {metrics['steer']['correlation']:.6f}

THROTTLE CONTROL
───────────────────────────────────────────────────────────────────
  Accuracy (±10%):             {metrics['throttle']['accuracy_10pct']:.2f}%
  Accuracy (±5%):              {metrics['throttle']['accuracy_5pct']:.2f}%
  Accuracy (±20%):             {metrics['throttle']['accuracy_20pct']:.2f}%
  Mean Absolute Error:         {metrics['throttle']['mae']:.6f}
  Median Absolute Error:       {metrics['throttle']['median_error']:.6f}
  R² Score:                    {metrics['throttle']['r2']:.6f}
  Correlation:                 {metrics['throttle']['correlation']:.6f}

BRAKE CONTROL ★
───────────────────────────────────────────────────────────────────
  Accuracy (±10%):             {metrics['brake']['accuracy_10pct']:.2f}%  ★
  Accuracy (±5%):              {metrics['brake']['accuracy_5pct']:.2f}%
  Accuracy (±20%):             {metrics['brake']['accuracy_20pct']:.2f}%
  Mean Absolute Error:         {metrics['brake']['mae']:.6f}
  Median Absolute Error:       {metrics['brake']['median_error']:.6f}
  R² Score:                    {metrics['brake']['r2']:.6f}
  Correlation:                 {metrics['brake']['correlation']:.6f}

TRAINING INFORMATION
───────────────────────────────────────────────────────────────────
  Total Epochs:                {len(history['train'])}
  Best Epoch:                  {best_epoch}
  Test Samples:                {len(predictions_df)}

MODEL ARCHITECTURE
───────────────────────────────────────────────────────────────────
  Image Encoder:               ResNet18 (pretrained)
  LiDAR Encoder:               PointNet
  Fusion Method:               Concatenation + MLP
  Total Parameters:            11,528,707

═══════════════════════════════════════════════════════════════════
"""

ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "6_performance_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Performance summary")

# ============================================================================
# 7. Accuracy Thresholds Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

thresholds = ['±5%', '±10%', '±20%']
x = np.arange(len(outputs_labels))
width = 0.25

acc_5pct = [metrics[output]['accuracy_5pct'] for output in outputs]
acc_10pct = [metrics[output]['accuracy_10pct'] for output in outputs]
acc_20pct = [metrics[output]['accuracy_20pct'] for output in outputs]

bars1 = ax.bar(x - width, acc_5pct, width, label='±5%', color='#FF6B6B', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, acc_10pct, width, label='±10%', color='#4ECDC4', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, acc_20pct, width, label='±20%', color='#45B7D1', alpha=0.8, edgecolor='black')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Accuracy at Different Error Thresholds', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(outputs_labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=2, label='90% Target')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "7_accuracy_thresholds.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Accuracy thresholds")

# ============================================================================
# 8. Model Architecture Diagram
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

arch_text = """
╔════════════════════════════════════════════════════════════════════╗
║              MULTIMODAL SELF-DRIVING MODEL ARCHITECTURE            ║
╚════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────────┐
│                          INPUT MODALITIES                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────┐         ┌─────────────────┐                 │
│  │  Camera Image   │         │   LiDAR Points  │                 │
│  │   224 x 224 x 3 │         │   2048 x 3      │                 │
│  └────────┬────────┘         └────────┬────────┘                 │
│           │                           │                           │
└───────────┼───────────────────────────┼───────────────────────────┘
            │                           │
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│   IMAGE ENCODER       │   │   LIDAR ENCODER       │
│   (ResNet18)          │   │   (PointNet)          │
│   - Pretrained        │   │   - Point Transform   │
│   - Conv layers       │   │   - MLP layers        │
│   - Global pool       │   │   - Max pooling       │
│   Output: 512-dim     │   │   Output: 512-dim     │
└───────────┬───────────┘   └───────────┬───────────┘
            │                           │
            └───────────┬───────────────┘
                        │
                        ▼
            ┌───────────────────────┐         ┌─────────────┐
            │  Feature Concatenate  │ ◄───────┤   Speed     │
            │     512 + 512 + 1     │         │   (scalar)  │
            │     = 1025 dims       │         └─────────────┘
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   FUSION HEAD (MLP)   │
            │   - FC: 1025 → 512    │
            │   - ReLU + Dropout    │
            │   - FC: 512 → 256     │
            │   - ReLU + Dropout    │
            │   - FC: 256 → 128     │
            │   - ReLU              │
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   OUTPUT HEADS        │
            │   - Steering (FC128→1)│
            │   - Throttle (FC128→1)│
            │   - Brake    (FC128→1)│
            └───────────┬───────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  CONTROL OUTPUTS      │
            │  • Steering: [-1, 1]  │
            │  • Throttle: [0, 1]   │
            │  • Brake:    [0, 1]   │
            └───────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                      MODEL STATISTICS                              │
├────────────────────────────────────────────────────────────────────┤
│  Total Parameters:         11,528,707                              │
│  Trainable Parameters:     11,528,707                              │
│  Model Size:               ~132 MB                                 │
│  Input Dimensions:         Image (224×224×3) + LiDAR (2048×3) + 1 │
│  Output Dimensions:        3 control values                        │
└────────────────────────────────────────────────────────────────────┘
"""

ax.text(0.5, 0.5, arch_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "8_model_architecture.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Model architecture")

# ============================================================================
# 9. Control Classification (Confusion Matrix Style)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (output, label, color) in enumerate(zip(outputs, outputs_labels, colors)):
    ax = axes[idx]
    
    actual = predictions_df[f'{output}_actual'].values
    predicted = predictions_df[f'{output}_pred'].values
    
    # Classify into bins
    bins = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]
    labels_bins = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    
    actual_binned = pd.cut(actual, bins=bins, labels=labels_bins)
    predicted_binned = pd.cut(predicted, bins=bins, labels=labels_bins)
    
    # Create confusion matrix
    cm = confusion_matrix(actual_binned, predicted_binned, labels=labels_bins)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels_bins, yticklabels=labels_bins,
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.set_title(f'{label} Control Classification')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "9_control_classification.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Control classification")

# ============================================================================
# 10. Training Configuration
# ============================================================================
# Load config if available
try:
    with open(TRAINING_DIR / "config.json", 'r') as f:
        config = json.load(f)
    has_config = True
except:
    has_config = False

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

if has_config:
    config_text = f"""
╔════════════════════════════════════════════════════════════════════╗
║                    TRAINING CONFIGURATION                          ║
╚════════════════════════════════════════════════════════════════════╝

DATASET
────────────────────────────────────────────────────────────────────
  Data Directory:            {config.get('data_dir', 'N/A')}
  Training Samples:          2,590
  Validation Samples:        1,270
  Test Samples:              1,230
  Image Size:                {config.get('image_size', 224)} × {config.get('image_size', 224)}
  LiDAR Points:              {config.get('num_points', 2048)}
  LiDAR Mode:                {config.get('lidar_mode', 'pointnet')}

TRAINING HYPERPARAMETERS
────────────────────────────────────────────────────────────────────
  Epochs:                    {config.get('epochs', 'N/A')}
  Batch Size:                {config.get('batch_size', 32)}
  Learning Rate:             {config.get('learning_rate', 0.001)}
  Optimizer:                 {config.get('optimizer', 'AdamW')}
  Weight Decay:              {config.get('weight_decay', 0.0001)}
  
SCHEDULER
────────────────────────────────────────────────────────────────────
  Type:                      {config.get('scheduler', 'CosineAnnealing')}
  T_max:                     {config.get('scheduler_params', {}).get('T_max', 30)}
  
LOSS FUNCTION
────────────────────────────────────────────────────────────────────
  Type:                      Weighted MSE
  Steering Weight:           2.0
  Throttle Weight:           1.0
  Brake Weight:              1.5

REGULARIZATION
────────────────────────────────────────────────────────────────────
  Dropout Rate:              0.3
  Early Stopping:            Enabled
  Patience:                  15 epochs
  
AUGMENTATION
────────────────────────────────────────────────────────────────────
  Random Horizontal Flip:    No
  Color Jitter:              No
  Random Rotation:           No

RESULTS
────────────────────────────────────────────────────────────────────
  Best Epoch:                {best_epoch}
  Final Train Loss:          {train_losses[-1]:.6f}
  Final Val Loss:            {val_losses[-1]:.6f}
  Best Val Loss:             {min(val_losses):.6f}
  Test Accuracy (±10%):      {metrics['overall']['accuracy_10pct']:.2f}%
"""
else:
    config_text = f"""
╔════════════════════════════════════════════════════════════════════╗
║                    TRAINING CONFIGURATION                          ║
╚════════════════════════════════════════════════════════════════════╝

DATASET
────────────────────────────────────────────────────────────────────
  Training Samples:          2,590
  Validation Samples:        1,270
  Test Samples:              1,230
  Image Size:                224 × 224
  LiDAR Points:              2048

TRAINING HYPERPARAMETERS
────────────────────────────────────────────────────────────────────
  Epochs:                    {len(train_losses)}
  Batch Size:                32
  Learning Rate:             0.001
  Optimizer:                 AdamW
  Weight Decay:              0.0001
  
LOSS FUNCTION
────────────────────────────────────────────────────────────────────
  Type:                      Weighted MSE
  Steering Weight:           2.0
  Throttle Weight:           1.0
  Brake Weight:              1.5

RESULTS
────────────────────────────────────────────────────────────────────
  Best Epoch:                {best_epoch}
  Final Train Loss:          {train_losses[-1]:.6f}
  Final Val Loss:            {val_losses[-1]:.6f}
  Best Val Loss:             {min(val_losses):.6f}
  Test Accuracy (±10%):      {metrics['overall']['accuracy_10pct']:.2f}%
"""

ax.text(0.5, 0.5, config_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_training_configuration.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Training configuration")

print(f"\n{'='*70}")
print(f"✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print(f"{'='*70}")
print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
print(f"\nGenerated files:")
for i, filename in enumerate(sorted(OUTPUT_DIR.glob("*.png")), 1):
    print(f"  {i}. {filename.name}")
print(f"\n{'='*70}")

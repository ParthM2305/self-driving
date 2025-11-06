# Baseline Model Visualizations (84.28% Accuracy)

This folder contains comprehensive visualizations for the **baseline production model** that achieves **84.28% overall accuracy**.

## Generated Visualizations

### 1. Training Loss Curve (`1_training_loss_curve.png`)
- Shows training and validation loss over all 28 epochs
- Highlights the best model checkpoint (Epoch 13)
- Demonstrates convergence and no overfitting

### 2. Per-Output Loss Curves (`2_per_output_loss_curves.png`)
- Individual loss curves for steering, throttle, and brake
- Compares training vs validation loss for each control output
- Shows which outputs converged best during training

### 3. Metrics Comparison (`3_metrics_comparison.png`)
- Side-by-side comparison of key metrics:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
  - R² Score
  - Accuracy (±10%)
- Visualizes performance differences between steering, throttle, and brake

### 4. Prediction vs Actual Scatter (`4_prediction_vs_actual.png`)
- Scatter plots showing predicted values vs ground truth
- Perfect prediction line for reference
- R² scores displayed for each output
- Demonstrates how well the model predictions align with actual values

### 5. Error Distributions (`5_error_distributions.png`)
- Histograms of prediction errors for each control output
- Shows error distribution centered around zero (good!)
- Includes mean, median, and standard deviation statistics
- Helps identify prediction bias

### 6. Performance Summary (`6_performance_summary.png`)
- Comprehensive text-based summary of all metrics
- Overall and individual control statistics
- Training information and model architecture details
- Quick reference for all key numbers

## Key Results Highlighted

- **Overall Accuracy**: 84.28% (±10% tolerance)
- **Brake Control**: 97.15% accuracy ⭐ (exceptional!)
- **Throttle Control**: 82.60% accuracy
- **Steering Control**: 73.09% accuracy
- **Best Model**: Epoch 13 (early stopping)
- **R² Score**: 0.8436 (strong predictive power)

## Usage

These visualizations are ready for:
- Project reports and presentations
- Documentation and GitHub README
- Performance analysis and comparison
- Stakeholder presentations

## Generation

Generated on: November 6, 2025
Script: `generate_baseline_visualizations.py`
Source data:
- Training history: `runs/improved_run_20251106_002053/training_history.json`
- Evaluation results: `evaluation_results_baseline_test/`

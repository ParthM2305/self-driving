# Report Visualizations

This folder contains comprehensive visualizations for the Self-Driving Model project report.

## Generated Visualizations

### 1. Training Loss Curve (`01_training_loss_curve.png`)
- Shows training and validation loss over 28 epochs
- Highlights best epoch (epoch 2) with lowest validation loss
- Demonstrates model convergence and early stopping effectiveness

### 2. Per-Output Loss Curves (`02_per_output_loss_curves.png`)
- Individual loss curves for steering, throttle, and brake
- Compares training vs validation loss for each output
- Shows which control outputs learned fastest

### 3. Metrics Comparison (`03_metrics_comparison.png`)
- Bar charts comparing MSE, MAE, R², and Correlation across all outputs
- Visual representation of model performance per control signal
- Helps identify strengths (brake/throttle) and areas for improvement (steering)

### 4. Accuracy Thresholds (`04_accuracy_thresholds.png`)
- Accuracy at different tolerance levels (±5%, ±10%, ±20%)
- Grouped bar chart for easy comparison
- Shows model performs better with more lenient thresholds

### 5. Prediction vs Actual Scatter Plots (`05_prediction_vs_actual.png`)
- Scatter plots for steering, throttle, and brake predictions
- Red dashed line shows perfect prediction
- Green line shows linear fit with equation
- R² score displayed for each output

### 6. Error Distributions (`06_error_distributions.png`)
- Histograms showing error distribution for each output
- Mean, median, and standard deviation marked
- Helps assess prediction bias and spread

### 7. Residual Plots (`07_residual_plots.png`)
- Residuals (prediction errors) vs predicted values
- Checks for systematic bias or heteroscedasticity
- ±2σ bounds shown to identify outliers

### 8. Model Architecture (`08_model_architecture.png`)
- Visual diagram of the multimodal architecture
- Shows data flow from inputs (camera + LiDAR + scalars) to outputs
- Includes layer dimensions and training configuration

### 9. Performance Summary (`09_performance_summary.png`)
- Comprehensive metrics table with all key statistics
- Overall and per-output performance breakdown
- Training details and dataset information
- Professional summary suitable for presentations

### 10. Training Configuration (`10_training_configuration.png`)
- Details of all 6 optimization techniques used
- Weighted loss, gradient clipping, label smoothing, etc.
- Shows improvement from baseline (58%) to final (83%)

## Usage in Report

These visualizations are designed for direct inclusion in:
- Project reports and documentation
- Presentation slides
- Research papers
- GitHub README

All images are high-resolution (300 DPI) PNG format, suitable for printing and digital use.

## Regenerating Visualizations

To regenerate these visualizations:

```bash
python generate_visualizations.py
```

This will create all 10 visualizations in the `report_visualizations/` folder.

## Statistics Summary

- **Overall Accuracy**: 83.22% (±10% threshold)
- **Steering**: 76.5% accuracy, MAE 0.0849
- **Throttle**: 87.32% accuracy, MAE 0.0816
- **Brake**: 85.85% accuracy, MAE 0.0965
- **Training Epochs**: 28 (early stopping at best epoch 2)
- **Model Parameters**: ~11.5M
- **Training Time**: ~140 minutes on CPU

## Author

Parth Malhotra  
College Project: Self-Driving Car (Minor 7th Sem)  
Date: November 6, 2025

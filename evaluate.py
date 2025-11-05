"""
Evaluation script for multimodal self-driving model.
Computes metrics, generates visualizations, and saves results.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model import MultimodalDrivingModel
from dataset import CarlaDataset
from utils import setup_logging, load_checkpoint, get_device


class Evaluator:
    """Model evaluator."""
    
    def __init__(self, args):
        self.args = args
        
        # Setup
        self.device = get_device(args.device)
        self.logger = setup_logging()
        
        # Output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.logger.info(f"Loading dataset from {args.data_dir}")
        self.dataset = CarlaDataset(
            index_file=Path(args.data_dir) / f'{args.split}_index.csv',
            lidar_mode=args.lidar_mode,
            image_size=(args.image_size, args.image_size),
            num_points=args.num_points,
            augment=False  # No augmentation for evaluation
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(self.device.type == 'cuda')
        )
        
        self.logger.info(f"Loaded {len(self.dataset)} samples")
        
        # Load model
        self.logger.info(f"Loading model from {args.checkpoint}")
        self.model = MultimodalDrivingModel(
            lidar_mode=args.lidar_mode,
            pretrained_image=False
        )
        
        load_checkpoint(args.checkpoint, self.model, device=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("Model loaded successfully")
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Run evaluation on the dataset.
        
        Returns:
            Tuple of (metrics_dict, predictions_df)
        """
        self.logger.info("Running evaluation...")
        
        all_predictions = {'steer': [], 'throttle': [], 'brake': []}
        all_targets = {'steer': [], 'throttle': [], 'brake': []}
        all_indices = []
        
        for batch in tqdm(self.dataloader, desc="Evaluating"):
            image = batch['image'].to(self.device)
            lidar = batch['lidar'].to(self.device)
            scalars = batch['scalars'].to(self.device)
            targets = batch['targets'].cpu().numpy()
            indices = batch['idx'].numpy()
            
            # Forward pass
            predictions = self.model(image, lidar, scalars)
            
            # Store predictions and targets
            for key in all_predictions:
                all_predictions[key].append(predictions[key].cpu().numpy())
            
            all_targets['steer'].append(targets[:, 0])
            all_targets['throttle'].append(targets[:, 1])
            all_targets['brake'].append(targets[:, 2])
            all_indices.append(indices)
        
        # Concatenate all batches
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key])
            all_targets[key] = np.concatenate(all_targets[key])
        all_indices = np.concatenate(all_indices)
        
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_targets)
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'idx': all_indices,
            'actual_steer': all_targets['steer'],
            'pred_steer': all_predictions['steer'],
            'actual_throttle': all_targets['throttle'],
            'pred_throttle': all_predictions['throttle'],
            'actual_brake': all_targets['brake'],
            'pred_brake': all_predictions['brake']
        })
        
        return metrics, predictions_df
    
    def compute_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        for key in ['steer', 'throttle', 'brake']:
            pred = predictions[key]
            target = targets[key]
            
            # MSE
            mse = mean_squared_error(target, pred)
            metrics[f'{key}_mse'] = float(mse)
            
            # RMSE
            rmse = np.sqrt(mse)
            metrics[f'{key}_rmse'] = float(rmse)
            
            # MAE
            mae = mean_absolute_error(target, pred)
            metrics[f'{key}_mae'] = float(mae)
            
            # R2 score
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - target.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            metrics[f'{key}_r2'] = float(r2)
            
            self.logger.info(
                f"{key.upper()}: MSE={mse:.6f}, RMSE={rmse:.6f}, "
                f"MAE={mae:.6f}, R2={r2:.4f}"
            )
        
        # Combined metrics
        metrics['combined_mse'] = np.mean([
            metrics['steer_mse'],
            metrics['throttle_mse'],
            metrics['brake_mse']
        ])
        metrics['combined_mae'] = np.mean([
            metrics['steer_mae'],
            metrics['throttle_mae'],
            metrics['brake_mae']
        ])
        
        return metrics
    
    def visualize_results(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ):
        """Create and save visualizations."""
        self.logger.info("Creating visualizations...")
        
        # 1. Scatter plots
        self.plot_scatter(predictions, targets)
        
        # 2. Error distributions
        self.plot_error_distributions(predictions, targets)
        
        # 3. Error vs actual value
        self.plot_error_vs_actual(predictions, targets)
        
        # 4. Correlation matrix
        self.plot_correlation_matrix(predictions, targets)
    
    def plot_scatter(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ):
        """Create scatter plots of predictions vs targets."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for ax, key in zip(axes, ['steer', 'throttle', 'brake']):
            pred = predictions[key]
            target = targets[key]
            
            # Scatter plot
            ax.scatter(target, pred, alpha=0.3, s=10)
            
            # Perfect prediction line
            min_val = min(target.min(), pred.min())
            max_val = max(target.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
            
            # Add R2 score to plot
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - target.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            ax.text(0.05, 0.95, f'$R^2$ = {r2:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(f'Actual {key}', fontsize=12)
            ax.set_ylabel(f'Predicted {key}', fontsize=12)
            ax.set_title(f'{key.capitalize()} Predictions', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'scatter_plots.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved scatter plots: {save_path}")
    
    def plot_error_distributions(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ):
        """Plot error distributions."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for ax, key in zip(axes, ['steer', 'throttle', 'brake']):
            errors = predictions[key] - targets[key]
            
            # Histogram
            ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            
            # Add mean and std
            mean_error = errors.mean()
            std_error = errors.std()
            
            ax.axvline(mean_error, color="r", linestyle="--", linewidth=2, label=f"Mean = {mean_error:.4f}")
            ax.axvline(0, color="g", linestyle="-", linewidth=1, label="Zero error")
            
            ax.text(0.05, 0.95, f'Std = {std_error:.4f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(f'{key.capitalize()} Error (Pred - Actual)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{key.capitalize()} Error Distribution', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / 'error_distributions.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved error distributions: {save_path}")
    
    def plot_error_vs_actual(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ):
        """Plot error vs actual values."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for ax, key in zip(axes, ['steer', 'throttle', 'brake']):
            errors = predictions[key] - targets[key]
            actual = targets[key]
            
            # Scatter plot
            ax.scatter(actual, errors, alpha=0.3, s=10)
            ax.axhline(0, color='r', linestyle='--', linewidth=2)
            
            ax.set_xlabel(f'Actual {key}', fontsize=12)
            ax.set_ylabel('Prediction Error', fontsize=12)
            ax.set_title(f'{key.capitalize()} Error vs Actual', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'error_vs_actual.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved error vs actual plots: {save_path}")
    
    def plot_correlation_matrix(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ):
        """Plot correlation matrix between predictions and targets."""
        # Create dataframe
        data = {}
        for key in ['steer', 'throttle', 'brake']:
            data[f'actual_{key}'] = targets[key]
            data[f'pred_{key}'] = predictions[key]
        
        df = pd.DataFrame(data)
        
        # Compute correlation
        corr = df.corr()
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'correlation_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved correlation matrix: {save_path}")
    
    def run(self):
        """Run complete evaluation pipeline."""
        # Evaluate
        metrics, predictions_df = self.evaluate()
        
        # Save metrics
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Saved metrics: {metrics_path}")
        
        # Save predictions
        predictions_path = self.output_dir / 'predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        self.logger.info(f"Saved predictions: {predictions_path}")
        
        # Create visualizations
        predictions = {
            'steer': predictions_df['pred_steer'].values,
            'throttle': predictions_df['pred_throttle'].values,
            'brake': predictions_df['pred_brake'].values
        }
        targets = {
            'steer': predictions_df['actual_steer'].values,
            'throttle': predictions_df['actual_throttle'].values,
            'brake': predictions_df['actual_brake'].values
        }
        
        self.visualize_results(predictions, targets)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Dataset: {self.args.split}")
        print(f"Samples: {len(predictions_df)}")
        print("\nMetrics:")
        print("-"*60)
        for key in ['steer', 'throttle', 'brake']:
            print(f"{key.upper()}:")
            print(f"  MSE:  {metrics[f'{key}_mse']:.6f}")
            print(f"  RMSE: {metrics[f'{key}_rmse']:.6f}")
            print(f"  MAE:  {metrics[f'{key}_mae']:.6f}")
            print(f"  R²:   {metrics[f'{key}_r2']:.6f}")
        print("\nCombined:")
        print(f"  MSE: {metrics['combined_mse']:.6f}")
        print(f"  MAE: {metrics['combined_mae']:.6f}")
        print("="*60)
        
        self.logger.info("Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description='Evaluate self-driving model')
    
    # Model & data
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing processed data')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='Data split to evaluate')
    parser.add_argument('--lidar-mode', type=str, default='pointnet',
                        choices=['pointnet', 'bev', 'sector_hist'],
                        help='LiDAR encoding mode')
    
    # Parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-points', type=int, default=4096,
                        help='Number of points for pointnet mode')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='Output directory for results')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # Auto-adjust for CPU
    device = get_device(args.device)
    if device.type == 'cpu':
        args.batch_size = min(args.batch_size, 16)
        args.workers = 2
        if args.num_points > 1024:
            args.num_points = 1024
    
    # Run evaluation
    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == '__main__':
    main()

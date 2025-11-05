"""
Comprehensive Evaluation with Detailed Statistics and Human-Readable Results

Provides:
1. Overall model accuracy and performance metrics
2. Individual statistics for steering, throttle, brake
3. Random test cases with human-readable interpretations
4. Detailed analysis and visualizations
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from model import MultimodalDrivingModel
from dataset import CarlaDataset
from utils import setup_logging, get_device, load_checkpoint


class ComprehensiveEvaluator:
    """Comprehensive model evaluation with detailed statistics."""
    
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.device)
        
        # Setup output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(self.output_dir / 'evaluation.log')
        self.logger.info(f"Comprehensive Evaluation")
        self.logger.info(f"Device: {self.device}")
        
        # Load dataset
        index_file = Path(args.data_dir) / f'{args.split}_index.csv'
        self.logger.info(f"Loading dataset from {index_file}")
        
        self.dataset = CarlaDataset(
            index_file=str(index_file),
            lidar_mode=args.lidar_mode,
            image_size=(args.image_size, args.image_size),
            num_points=args.num_points,
            augment=False,
            device=self.device.type
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False
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
    
    def interpret_steering(self, value: float) -> str:
        """Convert steering value to human-readable direction."""
        if abs(value) < 0.05:
            return "STRAIGHT"
        elif value > 0:
            direction = "RIGHT"
            if value > 0.5:
                intensity = "HARD"
            elif value > 0.2:
                intensity = "MODERATE"
            else:
                intensity = "SLIGHT"
        else:
            direction = "LEFT"
            value_abs = abs(value)
            if value_abs > 0.5:
                intensity = "HARD"
            elif value_abs > 0.2:
                intensity = "MODERATE"
            else:
                intensity = "SLIGHT"
        
        if abs(value) < 0.05:
            return "STRAIGHT"
        else:
            return f"{intensity} {direction} ({value:.3f})"
    
    def interpret_throttle(self, value: float) -> str:
        """Convert throttle value to human-readable action."""
        if value < 0.1:
            return "IDLE (coasting)"
        elif value < 0.3:
            return "LIGHT acceleration"
        elif value < 0.6:
            return "MODERATE acceleration"
        else:
            return "FULL acceleration"
    
    def interpret_brake(self, value: float) -> str:
        """Convert brake value to human-readable action."""
        if value < 0.1:
            return "NO BRAKE (released)"
        elif value < 0.3:
            return "LIGHT braking"
        elif value < 0.6:
            return "MODERATE braking"
        else:
            return "HARD braking"
    
    def get_driving_decision(self, steer: float, throttle: float, brake: float) -> str:
        """Get overall driving decision."""
        actions = []
        
        # Steering
        if abs(steer) >= 0.05:
            actions.append(self.interpret_steering(steer))
        
        # Speed control
        if brake > 0.2:
            actions.append(self.interpret_brake(brake))
        elif throttle > 0.2:
            actions.append(self.interpret_throttle(throttle))
        else:
            actions.append("Maintaining speed")
        
        if not actions or (len(actions) == 1 and "STRAIGHT" in actions[0]):
            return "Continue straight at current speed"
        
        return " + ".join(actions)
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[Dict, pd.DataFrame, List[Dict]]:
        """
        Run comprehensive evaluation.
        
        Returns:
            Tuple of (metrics_dict, predictions_df, sample_cases)
        """
        self.logger.info("Running evaluation...")
        
        all_predictions = {'steer': [], 'throttle': [], 'brake': []}
        all_targets = {'steer': [], 'throttle': [], 'brake': []}
        all_indices = []
        
        for batch in tqdm(self.dataloader, desc="Evaluating"):
            images = batch['image'].to(self.device)
            lidar = batch['lidar'].to(self.device)
            scalars = batch['scalars'].to(self.device)
            targets = batch['targets'].to(self.device)
            indices = batch['idx'].numpy()
            
            # Forward pass
            predictions = self.model(images, lidar, scalars)
            
            # Collect predictions and targets
            all_predictions['steer'].extend(predictions['steer'].cpu().numpy())
            all_predictions['throttle'].extend(predictions['throttle'].cpu().numpy())
            all_predictions['brake'].extend(predictions['brake'].cpu().numpy())
            
            all_targets['steer'].extend(targets[:, 0].cpu().numpy())
            all_targets['throttle'].extend(targets[:, 1].cpu().numpy())
            all_targets['brake'].extend(targets[:, 2].cpu().numpy())
            
            all_indices.extend(indices)
        
        # Convert to numpy arrays
        for key in all_predictions:
            all_predictions[key] = np.array(all_predictions[key])
            all_targets[key] = np.array(all_targets[key])
        
        # Compute comprehensive metrics
        metrics = self.compute_comprehensive_metrics(all_predictions, all_targets)
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'index': all_indices,
            'steer_pred': all_predictions['steer'],
            'steer_actual': all_targets['steer'],
            'throttle_pred': all_predictions['throttle'],
            'throttle_actual': all_targets['throttle'],
            'brake_pred': all_predictions['brake'],
            'brake_actual': all_targets['brake']
        })
        
        # Select random sample cases
        sample_cases = self.select_sample_cases(predictions_df, n_samples=3)
        
        return metrics, predictions_df, sample_cases
    
    def compute_comprehensive_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ) -> Dict:
        """Compute comprehensive metrics for all outputs."""
        metrics = {}
        
        # Individual metrics for each output
        for key in ['steer', 'throttle', 'brake']:
            pred = predictions[key]
            target = targets[key]
            
            # Basic metrics
            mse = mean_squared_error(target, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(target, pred)
            r2 = r2_score(target, target) if len(np.unique(target)) > 1 else 0
            
            # Additional metrics
            max_error = np.max(np.abs(target - pred))
            median_error = np.median(np.abs(target - pred))
            std_error = np.std(target - pred)
            
            # Percentage metrics
            mean_target = np.mean(np.abs(target))
            mape = np.mean(np.abs((target - pred) / (np.abs(target) + 1e-8))) * 100
            
            # Accuracy thresholds (within X% of target)
            acc_5_pct = np.mean(np.abs(target - pred) <= 0.05) * 100
            acc_10_pct = np.mean(np.abs(target - pred) <= 0.10) * 100
            acc_20_pct = np.mean(np.abs(target - pred) <= 0.20) * 100
            
            # Correlation
            correlation = np.corrcoef(target, pred)[0, 1] if len(target) > 1 else 0
            
            metrics[key] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'max_error': float(max_error),
                'median_error': float(median_error),
                'std_error': float(std_error),
                'mape': float(mape),
                'correlation': float(correlation),
                'accuracy_5pct': float(acc_5_pct),
                'accuracy_10pct': float(acc_10_pct),
                'accuracy_20pct': float(acc_20_pct),
                'mean_target': float(mean_target),
                'mean_pred': float(np.mean(pred)),
                'std_target': float(np.std(target)),
                'std_pred': float(np.std(pred))
            }
        
        # Overall/combined metrics
        all_pred = np.concatenate([predictions[k] for k in ['steer', 'throttle', 'brake']])
        all_target = np.concatenate([targets[k] for k in ['steer', 'throttle', 'brake']])
        
        metrics['overall'] = {
            'mse': float(mean_squared_error(all_target, all_pred)),
            'mae': float(mean_absolute_error(all_target, all_pred)),
            'r2': float(r2_score(all_target, all_pred)),
            'correlation': float(np.corrcoef(all_target, all_pred)[0, 1]),
            'accuracy_10pct': float(np.mean(np.abs(all_target - all_pred) <= 0.10) * 100)
        }
        
        return metrics
    
    def select_sample_cases(self, df: pd.DataFrame, n_samples: int = 3) -> List[Dict]:
        """Select random sample cases for demonstration."""
        # Randomly select n samples
        sample_indices = np.random.choice(len(df), size=min(n_samples, len(df)), replace=False)
        samples = []
        
        for idx in sample_indices:
            row = df.iloc[idx]
            
            sample = {
                'index': int(row['index']),
                'actual': {
                    'steer': float(row['steer_actual']),
                    'throttle': float(row['throttle_actual']),
                    'brake': float(row['brake_actual'])
                },
                'predicted': {
                    'steer': float(row['steer_pred']),
                    'throttle': float(row['throttle_pred']),
                    'brake': float(row['brake_pred'])
                },
                'errors': {
                    'steer': float(abs(row['steer_actual'] - row['steer_pred'])),
                    'throttle': float(abs(row['throttle_actual'] - row['throttle_pred'])),
                    'brake': float(abs(row['brake_actual'] - row['brake_pred']))
                }
            }
            
            # Human-readable interpretations
            sample['actual_interpretation'] = {
                'steer': self.interpret_steering(sample['actual']['steer']),
                'throttle': self.interpret_throttle(sample['actual']['throttle']),
                'brake': self.interpret_brake(sample['actual']['brake']),
                'decision': self.get_driving_decision(
                    sample['actual']['steer'],
                    sample['actual']['throttle'],
                    sample['actual']['brake']
                )
            }
            
            sample['predicted_interpretation'] = {
                'steer': self.interpret_steering(sample['predicted']['steer']),
                'throttle': self.interpret_throttle(sample['predicted']['throttle']),
                'brake': self.interpret_brake(sample['predicted']['brake']),
                'decision': self.get_driving_decision(
                    sample['predicted']['steer'],
                    sample['predicted']['throttle'],
                    sample['predicted']['brake']
                )
            }
            
            samples.append(sample)
        
        return samples
    
    def print_comprehensive_report(self, metrics: Dict, sample_cases: List[Dict]):
        """Print comprehensive evaluation report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*80)
        
        # Overall Performance
        print("\n" + "─"*80)
        print("OVERALL MODEL PERFORMANCE")
        print("─"*80)
        overall = metrics['overall']
        print(f"Total Samples Evaluated: {len(self.dataset)}")
        print(f"Overall MSE:             {overall['mse']:.6f}")
        print(f"Overall MAE:             {overall['mae']:.6f}")
        print(f"Overall R² Score:        {overall['r2']:.6f}")
        print(f"Overall Correlation:     {overall['correlation']:.6f}")
        print(f"Overall Accuracy (±10%): {overall['accuracy_10pct']:.2f}%")
        
        # Individual Output Statistics
        for output_name, display_name in [('steer', 'STEERING'), ('throttle', 'THROTTLE'), ('brake', 'BRAKE')]:
            m = metrics[output_name]
            
            print("\n" + "─"*80)
            print(f"{display_name} STATISTICS")
            print("─"*80)
            
            print(f"\n📊 Error Metrics:")
            print(f"  Mean Squared Error (MSE):     {m['mse']:.6f}")
            print(f"  Root Mean Squared Error:      {m['rmse']:.6f}")
            print(f"  Mean Absolute Error (MAE):    {m['mae']:.6f}")
            print(f"  Median Absolute Error:        {m['median_error']:.6f}")
            print(f"  Maximum Error:                {m['max_error']:.6f}")
            print(f"  Standard Deviation of Error:  {m['std_error']:.6f}")
            
            print(f"\n📈 Performance Metrics:")
            print(f"  R² Score (Coefficient of Determination): {m['r2']:.6f}")
            print(f"  Correlation Coefficient:                 {m['correlation']:.6f}")
            print(f"  Mean Absolute Percentage Error (MAPE):   {m['mape']:.2f}%")
            
            print(f"\n🎯 Accuracy Metrics:")
            print(f"  Predictions within ±5%:   {m['accuracy_5pct']:.2f}%")
            print(f"  Predictions within ±10%:  {m['accuracy_10pct']:.2f}%")
            print(f"  Predictions within ±20%:  {m['accuracy_20pct']:.2f}%")
            
            print(f"\n📉 Distribution Statistics:")
            print(f"  Actual Mean:      {m['mean_target']:.6f}")
            print(f"  Predicted Mean:   {m['mean_pred']:.6f}")
            print(f"  Actual Std Dev:   {m['std_target']:.6f}")
            print(f"  Predicted Std Dev: {m['std_pred']:.6f}")
        
        # Sample Cases
        print("\n" + "="*80)
        print("RANDOM TEST CASES - HUMAN READABLE PREDICTIONS")
        print("="*80)
        
        for i, sample in enumerate(sample_cases, 1):
            print(f"\n{'='*80}")
            print(f"TEST CASE #{i} (Sample Index: {sample['index']})")
            print(f"{'='*80}")
            
            print("\n🎯 ACTUAL (Ground Truth):")
            print(f"  Steering:  {sample['actual']['steer']:+.3f} → {sample['actual_interpretation']['steer']}")
            print(f"  Throttle:  {sample['actual']['throttle']:+.3f} → {sample['actual_interpretation']['throttle']}")
            print(f"  Brake:     {sample['actual']['brake']:+.3f} → {sample['actual_interpretation']['brake']}")
            print(f"  ➜ DECISION: {sample['actual_interpretation']['decision']}")
            
            print("\n🤖 PREDICTED (Model Output):")
            print(f"  Steering:  {sample['predicted']['steer']:+.3f} → {sample['predicted_interpretation']['steer']}")
            print(f"  Throttle:  {sample['predicted']['throttle']:+.3f} → {sample['predicted_interpretation']['throttle']}")
            print(f"  Brake:     {sample['predicted']['brake']:+.3f} → {sample['predicted_interpretation']['brake']}")
            print(f"  ➜ DECISION: {sample['predicted_interpretation']['decision']}")
            
            print("\n📏 ERRORS:")
            print(f"  Steering Error:  {sample['errors']['steer']:.4f}")
            print(f"  Throttle Error:  {sample['errors']['throttle']:.4f}")
            print(f"  Brake Error:     {sample['errors']['brake']:.4f}")
            
            # Judgment
            avg_error = np.mean(list(sample['errors'].values()))
            if avg_error < 0.1:
                judgment = "✅ EXCELLENT prediction"
            elif avg_error < 0.2:
                judgment = "✓ GOOD prediction"
            elif avg_error < 0.3:
                judgment = "~ FAIR prediction"
            else:
                judgment = "✗ POOR prediction"
            print(f"  Average Error: {avg_error:.4f} → {judgment}")
        
        print("\n" + "="*80)
    
    def save_results(self, metrics: Dict, predictions_df: pd.DataFrame, sample_cases: List[Dict]):
        """Save all results to files."""
        # Save metrics
        with open(self.output_dir / 'comprehensive_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Saved metrics to {self.output_dir / 'comprehensive_metrics.json'}")
        
        # Save predictions
        predictions_df.to_csv(self.output_dir / 'predictions.csv', index=False)
        self.logger.info(f"Saved predictions to {self.output_dir / 'predictions.csv'}")
        
        # Save sample cases
        with open(self.output_dir / 'sample_cases.json', 'w') as f:
            json.dump(sample_cases, f, indent=2)
        self.logger.info(f"Saved sample cases to {self.output_dir / 'sample_cases.json'}")
        
        # Save human-readable report
        with open(self.output_dir / 'evaluation_report.txt', 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            self.print_comprehensive_report(metrics, sample_cases)
            sys.stdout = original_stdout
        self.logger.info(f"Saved report to {self.output_dir / 'evaluation_report.txt'}")
    
    def run(self):
        """Run complete evaluation."""
        metrics, predictions_df, sample_cases = self.evaluate()
        self.print_comprehensive_report(metrics, sample_cases)
        self.save_results(metrics, predictions_df, sample_cases)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to processed data directory')
    
    # Data
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate')
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
    
    # Set random seed for reproducible sample selection
    np.random.seed(42)
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator(args)
    evaluator.run()


if __name__ == '__main__':
    main()

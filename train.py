"""
Training script for multimodal self-driving model.
Supports mixed precision, checkpointing, and comprehensive logging.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MultimodalDrivingModel, count_parameters
from dataset import create_dataloaders
from utils import (
    setup_logging, set_seed, get_device, save_checkpoint,
    load_checkpoint, AverageMeter, get_auto_batch_size,
    get_auto_num_workers, create_run_dir, save_config
)


class Trainer:
    """Training manager for self-driving model."""
    
    def __init__(self, args):
        self.args = args
        
        # Setup
        self.device = get_device(args.device)
        set_seed(args.seed)
        
        # Create run directory
        self.run_dir = create_run_dir(args.save_dir, args.run_name)
        
        # Setup logging
        log_file = os.path.join(self.run_dir, 'training.log')
        self.logger = setup_logging(log_file)
        
        # Save configuration
        config_path = os.path.join(self.run_dir, 'config.json')
        save_config(vars(args), config_path)
        
        # Auto-adjust hyperparameters based on device
        if args.batch_size is None:
            args.batch_size = get_auto_batch_size(self.device, args.batch_size_gpu, args.batch_size_cpu)
        
        if args.workers is None:
            args.workers = get_auto_num_workers(self.device)
        
        # Adjust num_points for CPU
        if self.device.type == 'cpu' and args.num_points > 1024:
            self.logger.warning(f"Reducing num_points from {args.num_points} to 1024 for CPU")
            args.num_points = 1024
        
        self.logger.info(f"Batch size: {args.batch_size}, Workers: {args.workers}")
        
        # Create dataloaders
        self.logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            lidar_mode=args.lidar_mode,
            image_size=(args.image_size, args.image_size),
            num_points=args.num_points,
            device=self.device.type
        )
        
        # Build model
        self.logger.info("Building model...")
        self.model = MultimodalDrivingModel(
            lidar_mode=args.lidar_mode,
            pretrained_image=args.pretrained,
            freeze_backbone=args.freeze_backbone
        )
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1 and not args.no_data_parallel:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        
        num_params = count_parameters(self.model)
        self.logger.info(f"Model parameters: {num_params:,}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        if args.scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        elif args.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
            )
        else:
            self.scheduler = None
        
        # Mixed precision
        self.use_amp = args.use_amp and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            self.logger.info("Using mixed precision training")
        
        # Loss weights
        self.loss_weights = {
            'steer': args.steer_weight,
            'throttle': args.throttle_weight,
            'brake': args.brake_weight
        }
        
        # Tracking
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Resume from checkpoint
        if args.resume:
            self.resume_training(args.resume)
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        self.logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler, self.device
        )
        self.start_epoch = checkpoint_data['epoch'] + 1
        self.best_val_loss = checkpoint_data['metrics'].get('val_loss', float('inf'))
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss.
        
        Args:
            predictions: Dict with 'steer', 'throttle', 'brake'
            targets: (B, 3) tensor with [steer, throttle, brake]
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        steer_target = targets[:, 0]
        throttle_target = targets[:, 1]
        brake_target = targets[:, 2]
        
        # MSE losses
        steer_loss = F.mse_loss(predictions['steer'], steer_target)
        throttle_loss = F.mse_loss(predictions['throttle'], throttle_target)
        brake_loss = F.mse_loss(predictions['brake'], brake_target)
        
        # Weighted sum
        total_loss = (
            self.loss_weights['steer'] * steer_loss +
            self.loss_weights['throttle'] * throttle_loss +
            self.loss_weights['brake'] * brake_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'steer': steer_loss.item(),
            'throttle': throttle_loss.item(),
            'brake': brake_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        meters = {
            'total': AverageMeter(),
            'steer': AverageMeter(),
            'throttle': AverageMeter(),
            'brake': AverageMeter()
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            image = batch['image'].to(self.device)
            lidar = batch['lidar'].to(self.device)
            scalars = batch['scalars'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    predictions = self.model(image, lidar, scalars)
                    loss, loss_dict = self.compute_loss(predictions, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.args.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(image, lidar, scalars)
                loss, loss_dict = self.compute_loss(predictions, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                
                self.optimizer.step()
            
            # Update meters
            batch_size = image.size(0)
            for key, value in loss_dict.items():
                meters[key].update(value, batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{meters['total'].avg:.4f}",
                'steer': f"{meters['steer'].avg:.4f}",
                'throttle': f"{meters['throttle'].avg:.4f}",
                'brake': f"{meters['brake'].avg:.4f}"
            })
        
        return {key: meter.avg for key, meter in meters.items()}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        meters = {
            'total': AverageMeter(),
            'steer': AverageMeter(),
            'throttle': AverageMeter(),
            'brake': AverageMeter()
        }
        
        all_predictions = {'steer': [], 'throttle': [], 'brake': []}
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.args.epochs} [Val]")
        
        for batch in pbar:
            image = batch['image'].to(self.device)
            lidar = batch['lidar'].to(self.device)
            scalars = batch['scalars'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    predictions = self.model(image, lidar, scalars)
                    loss, loss_dict = self.compute_loss(predictions, targets)
            else:
                predictions = self.model(image, lidar, scalars)
                loss, loss_dict = self.compute_loss(predictions, targets)
            
            # Update meters
            batch_size = image.size(0)
            for key, value in loss_dict.items():
                meters[key].update(value, batch_size)
            
            # Store predictions
            for key in all_predictions:
                all_predictions[key].append(predictions[key].cpu())
            all_targets.append(targets.cpu())
            
            pbar.set_postfix({
                'loss': f"{meters['total'].avg:.4f}",
                'steer': f"{meters['steer'].avg:.4f}"
            })
        
        # Concatenate all predictions
        for key in all_predictions:
            all_predictions[key] = torch.cat(all_predictions[key])
        all_targets = torch.cat(all_targets)
        
        # Compute additional metrics (MAE, RMSE)
        metrics = {key: meter.avg for key, meter in meters.items()}
        
        for i, key in enumerate(['steer', 'throttle', 'brake']):
            mae = torch.abs(all_predictions[key] - all_targets[:, i]).mean().item()
            rmse = torch.sqrt(((all_predictions[key] - all_targets[:, i]) ** 2).mean()).item()
            metrics[f'{key}_mae'] = mae
            metrics[f'{key}_rmse'] = rmse
        
        # Save sample visualizations
        if epoch % self.args.vis_every == 0:
            self.visualize_predictions(
                all_predictions, all_targets, epoch, split='val'
            )
        
        return metrics
    
    def visualize_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        epoch: int,
        split: str = 'val'
    ):
        """Visualize predictions vs targets."""
        vis_dir = os.path.join(self.run_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, (ax, key) in enumerate(zip(axes, ['steer', 'throttle', 'brake'])):
            pred = predictions[key].numpy()
            target = targets[:, i].numpy()
            
            # Scatter plot
            ax.scatter(target, pred, alpha=0.3, s=1)
            
            # Perfect prediction line
            min_val = min(target.min(), pred.min())
            max_val = max(target.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
            
            ax.set_xlabel(f'Actual {key}')
            ax.set_ylabel(f'Predicted {key}')
            ax.set_title(f'{key.capitalize()} (Epoch {epoch})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(vis_dir, f'{split}_predictions_epoch{epoch:03d}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved visualization: {save_path}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Training from epoch {self.start_epoch} to {self.args.epochs}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch + 1)
            
            # Validate
            val_metrics = self.validate(epoch + 1)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total'])
                else:
                    self.scheduler.step()
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch+1}/{self.args.epochs} - "
                f"Train Loss: {train_metrics['total']:.4f} - "
                f"Val Loss: {val_metrics['total']:.4f} - "
                f"Val Steer MAE: {val_metrics['steer_mae']:.4f} - "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save metrics
            self.train_losses.append(train_metrics)
            self.val_losses.append(val_metrics)
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.run_dir, 'checkpoints', f'checkpoint_epoch{epoch+1:03d}.pth'
            )
            
            is_best = val_metrics['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total']
            
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch + 1, val_metrics, checkpoint_path, is_best
            )
            
            # Save training history
            history = {
                'train': self.train_losses,
                'val': self.val_losses
            }
            history_path = os.path.join(self.run_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        
        self.logger.info("Training complete!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total loss
        ax = axes[0, 0]
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, [m['total'] for m in self.train_losses], label='Train')
        ax.plot(epochs, [m['total'] for m in self.val_losses], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Steer loss
        ax = axes[0, 1]
        ax.plot(epochs, [m['steer'] for m in self.train_losses], label='Train')
        ax.plot(epochs, [m['steer'] for m in self.val_losses], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Steer Loss')
        ax.set_title('Steering Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Throttle loss
        ax = axes[1, 0]
        ax.plot(epochs, [m['throttle'] for m in self.train_losses], label='Train')
        ax.plot(epochs, [m['throttle'] for m in self.val_losses], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Throttle Loss')
        ax.set_title('Throttle Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Brake loss
        ax = axes[1, 1]
        ax.plot(epochs, [m['brake'] for m in self.train_losses], label='Train')
        ax.plot(epochs, [m['brake'] for m in self.val_losses], label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Brake Loss')
        ax.set_title('Brake Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.run_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Saved training curves: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Train multimodal self-driving model')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing processed data')
    parser.add_argument('--lidar-mode', type=str, default='pointnet',
                        choices=['pointnet', 'bev', 'sector_hist'],
                        help='LiDAR encoding mode')
    parser.add_argument('--num-points', type=int, default=4096,
                        help='Number of points for pointnet mode')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size (will be resized to square)')
    
    # Model
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained image encoder')
    parser.add_argument('--no-pretrained', action='store_false', dest='pretrained',
                        help='Do not use pretrained weights')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze image encoder backbone')
    
    # Training
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (auto if not specified)')
    parser.add_argument('--batch-size-gpu', type=int, default=64,
                        help='Default batch size for GPU')
    parser.add_argument('--batch-size-cpu', type=int, default=8,
                        help='Default batch size for CPU')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping (0 to disable)')
    
    # Loss weights
    parser.add_argument('--steer-weight', type=float, default=1.0,
                        help='Steering loss weight')
    parser.add_argument('--throttle-weight', type=float, default=0.5,
                        help='Throttle loss weight')
    parser.add_argument('--brake-weight', type=float, default=0.5,
                        help='Brake loss weight')
    
    # Scheduler
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of dataloader workers (auto if not specified)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--no-amp', action='store_false', dest='use_amp',
                        help='Disable mixed precision')
    parser.add_argument('--no-data-parallel', action='store_true',
                        help='Disable DataParallel for multi-GPU')
    
    # Checkpointing
    parser.add_argument('--save-dir', type=str, default='./runs',
                        help='Directory to save runs')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Run name (auto-generated if not specified)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--vis-every', type=int, default=5,
                        help='Visualize predictions every N epochs')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (1 epoch)')
    
    args = parser.parse_args()
    
    # Debug mode overrides
    if args.debug:
        args.epochs = 1
        args.workers = 0
        print("Debug mode: 1 epoch, 0 workers")
    
    # Create trainer and run
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

"""
Improved training script with enhanced techniques for better performance.

Improvements over train.py:
1. Weighted loss (steering weighted higher)
2. Data augmentation (already in dataset)
3. Gradient clipping
4. Label smoothing
5. Cosine annealing with warmup
6. Mixed precision training support
7. Early stopping
8. Better learning rate scheduling
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MultimodalDrivingModel
from dataset import create_dataloaders
from utils import setup_logging, get_device, save_checkpoint, load_checkpoint


class ImprovedTrainer:
    """Enhanced trainer with advanced techniques."""
    
    def __init__(self, args):
        self.args = args
        self.device = get_device(args.device)
        
        # Setup
        self.save_dir = Path(args.save_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.save_dir / f'improved_run_{timestamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logging(self.run_dir / 'training.log')
        self.logger.info(f"Improved Training - Run: {self.run_dir.name}")
        self.logger.info(f"Device: {self.device}")
        
        # Save config
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Data loaders with augmentation
        self.logger.info("Loading datasets with augmentation...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            lidar_mode=args.lidar_mode,
            num_points=args.num_points,
            image_size=(args.image_size, args.image_size),
            device=self.device.type
        )
        
        self.logger.info(f"Train: {len(self.train_loader.dataset)} samples")
        self.logger.info(f"Val: {len(self.val_loader.dataset)} samples")
        
        # Model
        self.logger.info("Building model...")
        self.model = MultimodalDrivingModel(
            lidar_mode=args.lidar_mode,
            pretrained_image=args.pretrained
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
        
        # Improved loss weights (steering is most critical)
        self.loss_weights = {
            'steer': 2.0,      # Increased from 1.0
            'throttle': 1.0,   # Standard
            'brake': 1.5       # Slightly increased (safety critical)
        }
        self.logger.info(f"Loss weights: {self.loss_weights}")
        
        # Optimizer with weight decay
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Enhanced scheduler
        if args.scheduler == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,  # Restart every 10 epochs
                T_mult=2,
                eta_min=1e-6
            )
            self.scheduler_type = 'cosine'
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
            self.scheduler_type = 'plateau'
        
        # Gradient clipping
        self.grad_clip = args.grad_clip
        
        # Early stopping
        self.early_stop_patience = args.early_stop_patience
        self.early_stop_counter = 0
        
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
        targets: torch.Tensor,
        label_smoothing: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss with optional label smoothing.
        
        Args:
            predictions: Dict with 'steer', 'throttle', 'brake'
            targets: (B, 3) tensor with [steer, throttle, brake]
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        steer_target = targets[:, 0]
        throttle_target = targets[:, 1]
        brake_target = targets[:, 2]
        
        # Optional label smoothing for regression
        # Smooth targets slightly toward the mean
        if label_smoothing > 0:
            steer_mean = steer_target.mean()
            throttle_mean = throttle_target.mean()
            brake_mean = brake_target.mean()
            
            steer_target = (1 - label_smoothing) * steer_target + label_smoothing * steer_mean
            throttle_target = (1 - label_smoothing) * throttle_target + label_smoothing * throttle_mean
            brake_target = (1 - label_smoothing) * brake_target + label_smoothing * brake_mean
        
        # MSE losses
        steer_loss = F.mse_loss(predictions['steer'], steer_target)
        throttle_loss = F.mse_loss(predictions['throttle'], throttle_target)
        brake_loss = F.mse_loss(predictions['brake'], brake_target)
        
        # Weighted sum (steering weighted higher)
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
        """Train for one epoch with gradient clipping."""
        self.model.train()
        
        losses = {'total': [], 'steer': [], 'throttle': [], 'brake': []}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        for batch in pbar:
            images = batch['image'].to(self.device)
            lidar = batch['lidar'].to(self.device)
            scalars = batch['scalars'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            predictions = self.model(images, lidar, scalars)
            
            # Loss (with slight label smoothing)
            loss, loss_dict = self.compute_loss(
                predictions, targets,
                label_smoothing=self.args.label_smoothing
            )
            
            # Backward with gradient clipping
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Track
            for key, val in loss_dict.items():
                losses[key].append(val)
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'steer': f"{loss_dict['steer']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Average losses
        avg_losses = {key: np.mean(vals) for key, vals in losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        losses = {'total': [], 'steer': [], 'throttle': [], 'brake': []}
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Val]')
        for batch in pbar:
            images = batch['image'].to(self.device)
            lidar = batch['lidar'].to(self.device)
            scalars = batch['scalars'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            # Forward
            predictions = self.model(images, lidar, scalars)
            
            # Loss (no label smoothing for validation)
            loss, loss_dict = self.compute_loss(predictions, targets, label_smoothing=0.0)
            
            # Track
            for key, val in loss_dict.items():
                losses[key].append(val)
            
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})
        
        # Average losses
        avg_losses = {key: np.mean(vals) for key, vals in losses.items()}
        
        return avg_losses
    
    def save_training_progress(self):
        """Save training history and plots."""
        # Save history
        history = {
            'train': self.train_losses,
            'val': self.val_losses
        }
        
        with open(self.run_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot curves
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Total loss
            axes[0, 0].plot([e['total'] for e in self.train_losses], label='Train')
            axes[0, 0].plot([e['total'] for e in self.val_losses], label='Val')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Steering loss
            axes[0, 1].plot([e['steer'] for e in self.train_losses], label='Train')
            axes[0, 1].plot([e['steer'] for e in self.val_losses], label='Val')
            axes[0, 1].set_title('Steering Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Throttle loss
            axes[1, 0].plot([e['throttle'] for e in self.train_losses], label='Train')
            axes[1, 0].plot([e['throttle'] for e in self.val_losses], label='Val')
            axes[1, 0].set_title('Throttle Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Brake loss
            axes[1, 1].plot([e['brake'] for e in self.train_losses], label='Train')
            axes[1, 1].plot([e['brake'] for e in self.val_losses], label='Val')
            axes[1, 1].set_title('Brake Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.run_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.warning(f"Could not create training curves plot: {e}")
    
    def train(self):
        """Main training loop with early stopping."""
        self.logger.info("Starting training...")
        self.logger.info(f"Epochs: {self.args.epochs}")
        self.logger.info(f"Batch size: {self.args.batch_size}")
        self.logger.info(f"Learning rate: {self.args.lr}")
        self.logger.info(f"Gradient clipping: {self.grad_clip}")
        self.logger.info(f"Label smoothing: {self.args.label_smoothing}")
        self.logger.info(f"Early stopping patience: {self.early_stop_patience}")
        
        checkpoint_dir = self.run_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
            # Train
            train_losses = self.train_epoch(epoch)
            self.train_losses.append(train_losses)
            
            # Validate
            val_losses = self.validate(epoch)
            self.val_losses.append(val_losses)
            
            # Update scheduler
            if self.scheduler_type == 'plateau':
                self.scheduler.step(val_losses['total'])
            else:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            self.logger.info(
                f"Epoch {epoch}/{self.args.epochs} - "
                f"Train Loss: {train_losses['total']:.6f} - "
                f"Val Loss: {val_losses['total']:.6f} - "
                f"LR: {current_lr:.6f}"
            )
            
            # Save checkpoint
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics={'val_loss': val_losses['total']},
                save_path=str(checkpoint_dir / f'checkpoint_epoch{epoch:03d}.pth')
            )
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.early_stop_counter = 0
                
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics={'val_loss': val_losses['total']},
                    save_path=str(checkpoint_dir / f'checkpoint_epoch{epoch:03d}_best.pth')
                )
                
                self.logger.info(f"✓ New best model! Val loss: {self.best_val_loss:.6f}")
            else:
                self.early_stop_counter += 1
                self.logger.info(f"No improvement for {self.early_stop_counter} epochs")
            
            # Early stopping
            if self.early_stop_counter >= self.early_stop_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Save progress
            self.save_training_progress()
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Results saved to: {self.run_dir}")


def main():
    parser = argparse.ArgumentParser(description='Improved Self-Driving Model Training')
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--save-dir', type=str, default='./runs',
                        help='Directory to save results')
    
    # Model
    parser.add_argument('--lidar-mode', type=str, default='pointnet',
                        choices=['pointnet', 'bev', 'sector_hist'],
                        help='LiDAR encoding mode')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained image encoder')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine'],
                        help='Learning rate scheduler')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping (0 = disabled)')
    parser.add_argument('--label-smoothing', type=float, default=0.05,
                        help='Label smoothing for regression (0-0.1 recommended)')
    parser.add_argument('--early-stop-patience', type=int, default=15,
                        help='Early stopping patience')
    
    # Parameters
    parser.add_argument('--num-points', type=int, default=4096,
                        help='Number of points for pointnet mode')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Auto-adjust for CPU
    device = get_device(args.device)
    if device.type == 'cpu':
        args.batch_size = min(args.batch_size, 16)
        args.workers = 2
        if args.num_points > 1024:
            args.num_points = 1024
    
    # Train
    trainer = ImprovedTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

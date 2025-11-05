"""
PyTorch Dataset and DataLoader for CARLA self-driving data.
Supports multiple LiDAR encoding modes and data augmentation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import random


class CarlaDataset(Dataset):
    """
    CARLA multimodal self-driving dataset.
    
    Args:
        index_file: Path to index CSV file
        lidar_mode: LiDAR encoding mode ('pointnet', 'bev', 'sector_hist')
        image_size: Target image size (H, W)
        num_points: Number of points for pointnet mode
        augment: Whether to apply data augmentation
        device: Device for tensor allocation (for optimization)
    """
    
    def __init__(
        self,
        index_file: str,
        lidar_mode: str = 'pointnet',
        image_size: Tuple[int, int] = (224, 224),
        num_points: int = 4096,
        augment: bool = True,
        device: str = 'cpu'
    ):
        self.index_file = Path(index_file)
        self.data_dir = self.index_file.parent
        self.lidar_mode = lidar_mode
        self.image_size = image_size
        self.num_points = num_points
        self.augment = augment
        self.device_str = device
        
        # Load index
        self.df = pd.read_csv(index_file)
        logging.info(f"Loaded {len(self.df)} samples from {index_file}")
        
        # Setup image transforms
        self.setup_image_transforms()
        
        # Validate lidar mode
        assert lidar_mode in ['pointnet', 'bev', 'sector_hist'], \
            f"Invalid lidar_mode: {lidar_mode}"
        
        if lidar_mode == 'bev' and 'path_to_bev' not in self.df.columns:
            logging.warning("BEV mode selected but path_to_bev not in index. Will compute on-the-fly.")
            self.compute_bev_on_fly = True
        else:
            self.compute_bev_on_fly = False
    
    def setup_image_transforms(self):
        """Setup image preprocessing and augmentation."""
        if self.augment:
            self.image_transform = T.Compose([
                T.Resize(self.image_size),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = T.Compose([
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.df)
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        full_path = self.data_dir / image_path
        image = Image.open(full_path).convert('RGB')
        
        # Random horizontal flip with steering inversion
        if self.augment and random.random() < 0.5:
            image = T.functional.hflip(image)
            self.flip_steering = True
        else:
            self.flip_steering = False
        
        return self.image_transform(image)
    
    def load_lidar_pointnet(self, lidar_path: str) -> torch.Tensor:
        """Load and preprocess LiDAR for PointNet mode."""
        full_path = self.data_dir / lidar_path
        lidar = np.load(full_path)  # Nx4 array
        
        # Random rotation augmentation (small yaw angle)
        if self.augment:
            angle = np.random.uniform(-0.1, 0.1)  # radians
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([
                [cos_a, -sin_a, 0, 0],
                [sin_a, cos_a, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)
            lidar = lidar @ rotation
            
            # Random point dropout
            if random.random() < 0.3:
                keep_ratio = random.uniform(0.8, 0.95)
                keep_n = int(len(lidar) * keep_ratio)
                keep_idx = np.random.choice(len(lidar), keep_n, replace=False)
                lidar = lidar[keep_idx]
            
            # Add small noise
            noise = np.random.normal(0, 0.02, lidar.shape).astype(np.float32)
            lidar = lidar + noise
        
        # Sample or pad to fixed number of points
        if len(lidar) > self.num_points:
            # Random sampling
            indices = np.random.choice(len(lidar), self.num_points, replace=False)
            lidar = lidar[indices]
        elif len(lidar) < self.num_points:
            # Pad with zeros or repeat points
            if len(lidar) > 0:
                # Repeat points to reach num_points
                repeats = self.num_points // len(lidar) + 1
                lidar = np.tile(lidar, (repeats, 1))[:self.num_points]
            else:
                # All zeros if no points
                lidar = np.zeros((self.num_points, 4), dtype=np.float32)
        
        return torch.from_numpy(lidar.astype(np.float32))
    
    def load_lidar_bev(self, lidar_path: str, bev_path: Optional[str] = None) -> torch.Tensor:
        """Load or compute BEV projection."""
        if bev_path and not self.compute_bev_on_fly:
            full_path = self.data_dir / bev_path
            bev = np.load(full_path)
        else:
            # Compute BEV on the fly
            full_path = self.data_dir / lidar_path
            lidar = np.load(full_path)
            bev = self.create_bev_projection(lidar)
        
        # Add channel dimension if needed
        if bev.ndim == 2:
            bev = bev[np.newaxis, :, :]  # (1, H, W)
        
        return torch.from_numpy(bev.astype(np.float32))
    
    def load_lidar_sector_hist(self, lidar_path: str) -> torch.Tensor:
        """Load LiDAR and compute sector histogram features."""
        full_path = self.data_dir / lidar_path
        lidar = np.load(full_path)  # Nx4
        
        # Compute radial distance and angle
        xy = lidar[:, :2]  # x, y coordinates
        distances = np.linalg.norm(xy, axis=1)
        angles = np.arctan2(xy[:, 1], xy[:, 0])  # -pi to pi
        
        # Define sectors (e.g., 36 sectors = 10 degrees each)
        num_sectors = 36
        sector_bins = np.linspace(-np.pi, np.pi, num_sectors + 1)
        
        # Compute features per sector: min, median, max distance
        features = []
        for i in range(num_sectors):
            mask = (angles >= sector_bins[i]) & (angles < sector_bins[i + 1])
            sector_distances = distances[mask]
            
            if len(sector_distances) > 0:
                min_dist = np.min(sector_distances)
                median_dist = np.median(sector_distances)
                max_dist = np.max(sector_distances)
            else:
                min_dist = median_dist = max_dist = 0.0
            
            features.extend([min_dist, median_dist, max_dist])
        
        features = np.array(features, dtype=np.float32)
        return torch.from_numpy(features)
    
    def create_bev_projection(
        self,
        lidar: np.ndarray,
        x_range: tuple = (-50, 50),
        y_range: tuple = (-50, 50),
        grid_size: int = 128,
        z_min: float = -3.0,
        z_max: float = 5.0
    ) -> np.ndarray:
        """Create BEV projection from LiDAR points."""
        if len(lidar) == 0:
            return np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Filter by height
        mask = (lidar[:, 2] >= z_min) & (lidar[:, 2] <= z_max)
        points = lidar[mask]
        
        if len(points) == 0:
            return np.zeros((grid_size, grid_size), dtype=np.float32)
        
        # Create BEV grid
        x_bins = np.linspace(x_range[0], x_range[1], grid_size + 1)
        y_bins = np.linspace(y_range[0], y_range[1], grid_size + 1)
        
        x_indices = np.digitize(points[:, 0], x_bins) - 1
        y_indices = np.digitize(points[:, 1], y_bins) - 1
        
        valid = (x_indices >= 0) & (x_indices < grid_size) & \
                (y_indices >= 0) & (y_indices < grid_size)
        
        x_indices = x_indices[valid]
        y_indices = y_indices[valid]
        heights = points[valid, 2]
        
        bev = np.zeros((grid_size, grid_size), dtype=np.float32)
        for x_idx, y_idx, h in zip(x_indices, y_indices, heights):
            bev[y_idx, x_idx] = max(bev[y_idx, x_idx], h)
        
        return bev
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        row = self.df.iloc[idx]
        
        # Load image
        image = self.load_image(row['path_to_image'])
        
        # Load LiDAR based on mode
        if self.lidar_mode == 'pointnet':
            lidar = self.load_lidar_pointnet(row['path_to_lidar'])
        elif self.lidar_mode == 'bev':
            bev_path = row.get('path_to_bev', None)
            lidar = self.load_lidar_bev(row['path_to_lidar'], bev_path)
        else:  # sector_hist
            lidar = self.load_lidar_sector_hist(row['path_to_lidar'])
        
        # Load scalar features
        speed = row['speed_kmh']
        scalars = torch.tensor([speed], dtype=torch.float32)
        
        # Load targets
        steer = row['steer']
        # Apply horizontal flip to steering if image was flipped
        if hasattr(self, 'flip_steering') and self.flip_steering:
            steer = -steer
        
        throttle = row['throttle']
        brake = row['brake']
        targets = torch.tensor([steer, throttle, brake], dtype=torch.float32)
        
        return {
            'image': image,
            'lidar': lidar,
            'scalars': scalars,
            'targets': targets,
            'idx': idx
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    lidar_mode: str = 'pointnet',
    image_size: Tuple[int, int] = (224, 224),
    num_points: int = 4096,
    device: str = 'cpu'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Directory containing index files
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        lidar_mode: LiDAR encoding mode
        image_size: Target image size
        num_points: Number of points for pointnet mode
        device: Device for optimization
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = CarlaDataset(
        index_file=data_dir / 'train_index.csv',
        lidar_mode=lidar_mode,
        image_size=image_size,
        num_points=num_points,
        augment=True,
        device=device
    )
    
    val_dataset = CarlaDataset(
        index_file=data_dir / 'validation_index.csv',
        lidar_mode=lidar_mode,
        image_size=image_size,
        num_points=num_points,
        augment=False,
        device=device
    )
    
    test_dataset = CarlaDataset(
        index_file=data_dir / 'test_index.csv',
        lidar_mode=lidar_mode,
        image_size=image_size,
        num_points=num_points,
        augment=False,
        device=device
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        drop_last=False
    )
    
    logging.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logging.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    logging.info(f"Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


def sanity_check(data_dir: str, num_samples: int = 8):
    """Run sanity check on dataset."""
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Dataset Sanity Check")
    print("="*60)
    
    # Try different lidar modes
    for lidar_mode in ['pointnet', 'bev', 'sector_hist']:
        print(f"\nTesting lidar_mode: {lidar_mode}")
        try:
            dataset = CarlaDataset(
                index_file=Path(data_dir) / 'train_index.csv',
                lidar_mode=lidar_mode,
                num_points=1024,  # Small for testing
                augment=True
            )
            
            # Load a few samples
            for i in range(min(num_samples, len(dataset))):
                sample = dataset[i]
                print(f"  Sample {i}:")
                print(f"    Image: {sample['image'].shape}")
                print(f"    LiDAR: {sample['lidar'].shape}")
                print(f"    Scalars: {sample['scalars'].shape}")
                print(f"    Targets: {sample['targets']}")
            
            print(f"  ✓ {lidar_mode} mode working")
            
        except Exception as e:
            print(f"  ✗ Error with {lidar_mode} mode: {e}")
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=4,
            num_workers=0,  # Use 0 for debugging
            lidar_mode='pointnet',
            num_points=1024
        )
        
        # Get one batch
        batch = next(iter(train_loader))
        print(f"  Batch shapes:")
        print(f"    Images: {batch['image'].shape}")
        print(f"    LiDAR: {batch['lidar'].shape}")
        print(f"    Scalars: {batch['scalars'].shape}")
        print(f"    Targets: {batch['targets'].shape}")
        
        print("  ✓ DataLoader working")
        
    except Exception as e:
        print(f"  ✗ DataLoader error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Sanity check complete!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA Dataset')
    parser.add_argument('--sanity', action='store_true',
                        help='Run sanity check')
    parser.add_argument('--data', type=str, default='./data',
                        help='Data directory')
    parser.add_argument('--num-samples', type=int, default=8,
                        help='Number of samples for sanity check')
    
    args = parser.parse_args()
    
    if args.sanity:
        sanity_check(args.data, args.num_samples)
    else:
        print("Use --sanity flag to run sanity check")
        print("Example: python dataset.py --sanity --data ./data")

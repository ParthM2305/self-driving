"""
Inference demo script for visualizing model predictions.
Displays RGB images with steering overlays and LiDAR BEV.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from PIL import Image
import cv2

from model import MultimodalDrivingModel
from dataset import CarlaDataset
from utils import setup_logging, load_checkpoint, get_device


class InferenceDemo:
    """Inference demonstration."""
    
    def __init__(self, args):
        self.args = args
        
        # Setup
        self.device = get_device(args.device)
        self.logger = setup_logging()
        
        # Load dataset
        self.logger.info(f"Loading dataset from {args.data_dir}")
        self.dataset = CarlaDataset(
            index_file=Path(args.data_dir) / f'{args.split}_index.csv',
            lidar_mode=args.lidar_mode,
            image_size=(args.image_size, args.image_size),
            num_points=args.num_points,
            augment=False
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
        
        # Output directory
        if args.save_outputs:
            self.output_dir = Path(args.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def denormalize_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Denormalize image tensor to numpy array."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        return img
    
    def render_bev_lidar(
        self,
        lidar_points: np.ndarray,
        size: int = 512,
        x_range: tuple = (-50, 50),
        y_range: tuple = (-50, 50)
    ) -> np.ndarray:
        """
        Render LiDAR points as bird's eye view image.
        
        Args:
            lidar_points: Nx4 array [x, y, z, intensity]
            size: Output image size
            x_range: X-axis range in meters
            y_range: Y-axis range in meters
        
        Returns:
            RGB image (H, W, 3)
        """
        # Create blank canvas
        canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        if len(lidar_points) == 0:
            return canvas
        
        # Extract coordinates
        x = lidar_points[:, 0]
        y = lidar_points[:, 1]
        z = lidar_points[:, 2]
        
        # Normalize to image coordinates
        x_norm = (x - x_range[0]) / (x_range[1] - x_range[0])
        y_norm = (y - y_range[0]) / (y_range[1] - y_range[0])
        
        # Convert to pixel coordinates
        px = (x_norm * size).astype(int)
        py = ((1 - y_norm) * size).astype(int)  # Flip y-axis
        
        # Filter valid points
        valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        px = px[valid]
        py = py[valid]
        z_valid = z[valid]
        
        # Color by height
        z_norm = (z_valid - z_valid.min()) / (z_valid.max() - z_valid.min() + 1e-6)
        colors = plt.cm.jet(z_norm)[:, :3] * 255
        
        # Draw points
        for i in range(len(px)):
            cv2.circle(canvas, (px[i], py[i]), 2, colors[i].tolist(), -1)
        
        # Draw vehicle position (center)
        center = size // 2
        cv2.circle(canvas, (center, center), 8, (255, 0, 0), -1)
        cv2.circle(canvas, (center, center), 10, (0, 0, 0), 2)
        
        # Draw orientation arrow
        arrow_len = 20
        cv2.arrowedLine(canvas, (center, center), (center, center - arrow_len),
                       (0, 0, 0), 3, tipLength=0.3)
        
        return canvas
    
    def draw_steering_overlay(
        self,
        image: np.ndarray,
        steering: float,
        color: tuple = (0, 255, 0),
        label: str = "Pred"
    ) -> np.ndarray:
        """
        Draw steering angle overlay on image.
        
        Args:
            image: RGB image (H, W, 3)
            steering: Steering angle (-1 to 1)
            color: RGB color tuple
            label: Label text
        
        Returns:
            Image with overlay
        """
        img = (image * 255).astype(np.uint8).copy()
        h, w = img.shape[:2]
        
        # Center point (bottom center)
        center_x = w // 2
        center_y = int(h * 0.85)
        
        # Arrow parameters
        arrow_length = min(h, w) // 4
        max_angle = 45  # degrees
        
        # Convert steering to angle
        angle_deg = steering * max_angle
        angle_rad = np.deg2rad(angle_deg)
        
        # Calculate arrow endpoint
        end_x = int(center_x + arrow_length * np.sin(angle_rad))
        end_y = int(center_y - arrow_length * np.cos(angle_rad))
        
        # Draw arrow
        cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y),
                       color, 4, tipLength=0.3)
        
        # Draw label
        text = f"{label}: {steering:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + 30
        
        # Background rectangle for text
        cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        cv2.putText(img, text, (text_x, text_y), font, 0.7, color, 2)
        
        return img
    
    @torch.no_grad()
    def visualize_sample(self, idx: int, save: bool = False):
        """Visualize a single sample prediction."""
        # Get sample
        sample = self.dataset[idx]
        
        # Prepare batch
        image = sample['image'].unsqueeze(0).to(self.device)
        lidar = sample['lidar'].unsqueeze(0).to(self.device)
        scalars = sample['scalars'].unsqueeze(0).to(self.device)
        targets = sample['targets']
        
        # Predict
        predictions = self.model(image, lidar, scalars)
        
        # Extract values
        pred_steer = predictions['steer'].item()
        pred_throttle = predictions['throttle'].item()
        pred_brake = predictions['brake'].item()
        
        actual_steer = targets[0].item()
        actual_throttle = targets[1].item()
        actual_brake = targets[2].item()
        
        # Create visualization
        fig = plt.figure(figsize=(16, 6))
        
        # 1. Image with steering overlay
        ax1 = plt.subplot(1, 3, 1)
        img = self.denormalize_image(sample['image'])
        
        # Draw actual steering (red)
        img_actual = self.draw_steering_overlay(img, actual_steer, (255, 0, 0), "Actual")
        # Draw predicted steering (green)
        img_pred = self.draw_steering_overlay(img_actual, pred_steer, (0, 255, 0), "Pred")
        
        ax1.imshow(img_pred)
        ax1.set_title(f'Front Camera (Sample {idx})', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. LiDAR BEV
        if self.args.render_bev:
            ax2 = plt.subplot(1, 3, 2)
            
            # Get raw lidar points
            row = self.dataset.df.iloc[idx]
            lidar_path = Path(self.args.data_dir) / row['path_to_lidar']
            lidar_points = np.load(lidar_path)
            
            bev_img = self.render_bev_lidar(lidar_points)
            ax2.imshow(bev_img)
            ax2.set_title('LiDAR Bird\'s Eye View', fontsize=14, fontweight='bold')
            ax2.axis('off')
        
        # 3. Control values
        ax3_idx = 3 if self.args.render_bev else 2
        ax3 = plt.subplot(1, 3, ax3_idx)
        ax3.axis('off')
        
        # Create table
        data = [
            ['Control', 'Actual', 'Predicted', 'Error'],
            ['Steering', f'{actual_steer:.4f}', f'{pred_steer:.4f}', f'{pred_steer - actual_steer:.4f}'],
            ['Throttle', f'{actual_throttle:.4f}', f'{pred_throttle:.4f}', f'{pred_throttle - actual_throttle:.4f}'],
            ['Brake', f'{actual_brake:.4f}', f'{pred_brake:.4f}', f'{pred_brake - actual_brake:.4f}'],
        ]
        
        table = ax3.table(cellText=data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color-code error
        for i in range(1, 4):
            error_val = float(data[i][3])
            if abs(error_val) < 0.05:
                color = '#C8E6C9'  # Light green
            elif abs(error_val) < 0.1:
                color = '#FFF9C4'  # Light yellow
            else:
                color = '#FFCDD2'  # Light red
            table[(i, 3)].set_facecolor(color)
        
        ax3.set_title('Control Values', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save and self.args.save_outputs:
            save_path = self.output_dir / f'sample_{idx:04d}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved: {save_path}")
        
        if self.args.show:
            plt.show()
        else:
            plt.close()
        
        # Print to console
        print(f"\nSample {idx}:")
        print(f"  Steering:  Actual={actual_steer:.4f}, Pred={pred_steer:.4f}, Error={pred_steer - actual_steer:.4f}")
        print(f"  Throttle:  Actual={actual_throttle:.4f}, Pred={pred_throttle:.4f}, Error={pred_throttle - actual_throttle:.4f}")
        print(f"  Brake:     Actual={actual_brake:.4f}, Pred={pred_brake:.4f}, Error={pred_brake - actual_brake:.4f}")
    
    def run(self):
        """Run inference demo."""
        self.logger.info(f"Running inference on {self.args.n_samples} samples")
        
        # Select samples
        if self.args.sample_indices:
            indices = self.args.sample_indices
        else:
            # Random samples
            indices = np.random.choice(
                len(self.dataset),
                min(self.args.n_samples, len(self.dataset)),
                replace=False
            )
        
        # Visualize each sample
        for idx in indices:
            self.visualize_sample(idx, save=self.args.save_outputs)
        
        self.logger.info("Inference demo complete!")


def main():
    parser = argparse.ArgumentParser(description='Inference demo for self-driving model')
    
    # Model & data
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing processed data')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='Data split to use')
    parser.add_argument('--lidar-mode', type=str, default='pointnet',
                        choices=['pointnet', 'bev', 'sector_hist'],
                        help='LiDAR encoding mode')
    
    # Sampling
    parser.add_argument('--n-samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--sample-indices', type=int, nargs='+', default=None,
                        help='Specific sample indices to visualize')
    
    # Visualization
    parser.add_argument('--render-bev', action='store_true', default=True,
                        help='Render LiDAR bird\'s eye view')
    parser.add_argument('--no-render-bev', action='store_false', dest='render_bev',
                        help='Skip BEV rendering')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    
    # Output
    parser.add_argument('--save-outputs', action='store_true', default=True,
                        help='Save output visualizations')
    parser.add_argument('--no-save', action='store_false', dest='save_outputs',
                        help='Do not save outputs')
    parser.add_argument('--output-dir', type=str, default='./demo_outputs',
                        help='Output directory for visualizations')
    
    # Parameters
    parser.add_argument('--num-points', type=int, default=4096,
                        help='Number of points for pointnet mode')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size')
    
    # System
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')
    
    args = parser.parse_args()
    
    # Auto-adjust for CPU
    device = get_device(args.device)
    if device.type == 'cpu' and args.num_points > 1024:
        args.num_points = 1024
    
    # Run demo
    demo = InferenceDemo(args)
    demo.run()


if __name__ == '__main__':
    main()

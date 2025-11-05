"""
Model architectures for multimodal self-driving.
Includes ImageEncoder, LiDAREncoder, and FusionNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple, Dict
import logging


class ImageEncoder(nn.Module):
    """
    Image encoder using ResNet18 backbone.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: Freeze early layers
        output_dim: Output feature dimension
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 256
    ):
        super().__init__()
        
        # Load ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logging.info("Image backbone frozen")
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor
        
        Returns:
            (B, output_dim) feature tensor
        """
        # Extract features
        features = self.backbone(x)  # (B, 512, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 512)
        
        # Project
        output = self.projection(features)  # (B, output_dim)
        
        return output


class PointNetEncoder(nn.Module):
    """
    PointNet-style encoder for LiDAR point clouds.
    
    Args:
        input_dim: Input point dimension (typically 4: x, y, z, intensity)
        output_dim: Output feature dimension
    """
    
    def __init__(self, input_dim: int = 4, output_dim: int = 256):
        super().__init__()
        
        # Point-wise MLPs
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        self.dropout = nn.Dropout(0.3)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, input_dim) point cloud tensor
        
        Returns:
            (B, output_dim) feature tensor
        """
        B, N, D = x.shape
        
        # Reshape for BatchNorm: (B, N, D) -> (B*N, D)
        x = x.reshape(B * N, D)
        
        # First MLP
        x = self.mlp1(x)  # (B*N, 128)
        
        # Second MLP
        x = self.mlp2(x)  # (B*N, output_dim)
        
        # Reshape back: (B*N, output_dim) -> (B, N, output_dim)
        x = x.reshape(B, N, self.output_dim)
        
        # Global max pooling across points
        x = torch.max(x, dim=1)[0]  # (B, output_dim)
        
        # Dropout
        x = self.dropout(x)
        
        return x


class BEVCNNEncoder(nn.Module):
    """
    CNN encoder for Bird's Eye View LiDAR representation.
    
    Args:
        input_channels: Number of input channels
        output_dim: Output feature dimension
    """
    
    def __init__(self, input_channels: int = 1, output_dim: int = 256):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) BEV tensor
        
        Returns:
            (B, output_dim) feature tensor
        """
        # Conv layers
        x = self.conv_layers(x)  # (B, 256, H', W')
        
        # Global pooling
        x = self.global_pool(x)  # (B, 256, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, 256)
        
        # FC
        x = self.fc(x)  # (B, output_dim)
        
        return x


class SectorHistEncoder(nn.Module):
    """
    Simple MLP encoder for sector histogram features.
    
    Args:
        input_dim: Input dimension (num_sectors * 3)
        output_dim: Output feature dimension
    """
    
    def __init__(self, input_dim: int = 108, output_dim: int = 256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) feature tensor
        
        Returns:
            (B, output_dim) feature tensor
        """
        return self.mlp(x)


class LiDAREncoder(nn.Module):
    """
    Wrapper for different LiDAR encoding methods.
    
    Args:
        lidar_mode: Encoding mode ('pointnet', 'bev', 'sector_hist')
        output_dim: Output feature dimension
        num_points: Number of points (for pointnet mode)
        bev_channels: BEV input channels (for bev mode)
    """
    
    def __init__(
        self,
        lidar_mode: str = 'pointnet',
        output_dim: int = 256,
        num_points: int = 4096,
        bev_channels: int = 1
    ):
        super().__init__()
        
        self.lidar_mode = lidar_mode
        
        if lidar_mode == 'pointnet':
            self.encoder = PointNetEncoder(input_dim=4, output_dim=output_dim)
        elif lidar_mode == 'bev':
            self.encoder = BEVCNNEncoder(input_channels=bev_channels, output_dim=output_dim)
        elif lidar_mode == 'sector_hist':
            self.encoder = SectorHistEncoder(input_dim=108, output_dim=output_dim)
        else:
            raise ValueError(f"Unknown lidar_mode: {lidar_mode}")
        
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.encoder(x)


class FusionNet(nn.Module):
    """
    Fusion network that combines image, LiDAR, and scalar features
    and predicts steering, throttle, and brake.
    
    Args:
        image_dim: Image feature dimension
        lidar_dim: LiDAR feature dimension
        scalar_dim: Scalar feature dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        image_dim: int = 256,
        lidar_dim: int = 256,
        scalar_dim: int = 1,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Fusion MLP
        fusion_input_dim = image_dim + lidar_dim + scalar_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction heads
        self.steer_head = nn.Linear(64, 1)
        self.throttle_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()  # Throttle in [0, 1]
        )
        self.brake_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()  # Brake in [0, 1]
        )
    
    def forward(
        self,
        image_feat: torch.Tensor,
        lidar_feat: torch.Tensor,
        scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image_feat: (B, image_dim)
            lidar_feat: (B, lidar_dim)
            scalars: (B, scalar_dim)
        
        Returns:
            Tuple of (steer, throttle, brake) predictions
        """
        # Concatenate all features
        fused = torch.cat([image_feat, lidar_feat, scalars], dim=1)
        
        # Fusion network
        x = self.fusion(fused)  # (B, 64)
        
        # Predictions
        steer = self.steer_head(x)  # (B, 1)
        throttle = self.throttle_head(x)  # (B, 1)
        brake = self.brake_head(x)  # (B, 1)
        
        return steer, throttle, brake


class MultimodalDrivingModel(nn.Module):
    """
    Complete multimodal self-driving model.
    
    Args:
        lidar_mode: LiDAR encoding mode
        pretrained_image: Use pretrained image encoder
        freeze_backbone: Freeze image backbone
        image_output_dim: Image encoder output dimension
        lidar_output_dim: LiDAR encoder output dimension
        scalar_dim: Scalar feature dimension
    """
    
    def __init__(
        self,
        lidar_mode: str = 'pointnet',
        pretrained_image: bool = True,
        freeze_backbone: bool = False,
        image_output_dim: int = 256,
        lidar_output_dim: int = 256,
        scalar_dim: int = 1
    ):
        super().__init__()
        
        self.image_encoder = ImageEncoder(
            pretrained=pretrained_image,
            freeze_backbone=freeze_backbone,
            output_dim=image_output_dim
        )
        
        self.lidar_encoder = LiDAREncoder(
            lidar_mode=lidar_mode,
            output_dim=lidar_output_dim
        )
        
        self.fusion_net = FusionNet(
            image_dim=image_output_dim,
            lidar_dim=lidar_output_dim,
            scalar_dim=scalar_dim
        )
        
        self.lidar_mode = lidar_mode
    
    def forward(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
        scalars: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            image: (B, 3, H, W) image tensor
            lidar: LiDAR tensor (shape depends on mode)
            scalars: (B, scalar_dim) scalar features
        
        Returns:
            Dictionary with 'steer', 'throttle', 'brake' predictions
        """
        # Encode modalities
        image_feat = self.image_encoder(image)
        lidar_feat = self.lidar_encoder(lidar)
        
        # Fusion and prediction
        steer, throttle, brake = self.fusion_net(image_feat, lidar_feat, scalars)
        
        return {
            'steer': steer.squeeze(-1),  # (B,)
            'throttle': throttle.squeeze(-1),  # (B,)
            'brake': brake.squeeze(-1)  # (B,)
        }
    
    def get_features(
        self,
        image: torch.Tensor,
        lidar: torch.Tensor,
        scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get intermediate features (for visualization/analysis)."""
        image_feat = self.image_encoder(image)
        lidar_feat = self.lidar_encoder(lidar)
        return image_feat, lidar_feat


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test model architectures."""
    print("\n" + "="*60)
    print("Model Architecture Test")
    print("="*60)
    
    batch_size = 4
    
    # Test ImageEncoder
    print("\n1. ImageEncoder")
    image_encoder = ImageEncoder(pretrained=False, output_dim=256)
    dummy_image = torch.randn(batch_size, 3, 224, 224)
    image_feat = image_encoder(dummy_image)
    print(f"   Input: {dummy_image.shape}")
    print(f"   Output: {image_feat.shape}")
    print(f"   Parameters: {count_parameters(image_encoder):,}")
    
    # Test PointNetEncoder
    print("\n2. PointNetEncoder")
    pointnet = PointNetEncoder(input_dim=4, output_dim=256)
    dummy_points = torch.randn(batch_size, 4096, 4)
    point_feat = pointnet(dummy_points)
    print(f"   Input: {dummy_points.shape}")
    print(f"   Output: {point_feat.shape}")
    print(f"   Parameters: {count_parameters(pointnet):,}")
    
    # Test BEVCNNEncoder
    print("\n3. BEVCNNEncoder")
    bev_encoder = BEVCNNEncoder(input_channels=1, output_dim=256)
    dummy_bev = torch.randn(batch_size, 1, 128, 128)
    bev_feat = bev_encoder(dummy_bev)
    print(f"   Input: {dummy_bev.shape}")
    print(f"   Output: {bev_feat.shape}")
    print(f"   Parameters: {count_parameters(bev_encoder):,}")
    
    # Test FusionNet
    print("\n4. FusionNet")
    fusion = FusionNet(image_dim=256, lidar_dim=256, scalar_dim=1)
    dummy_scalars = torch.randn(batch_size, 1)
    steer, throttle, brake = fusion(image_feat, point_feat, dummy_scalars)
    print(f"   Inputs: image_feat{image_feat.shape}, lidar_feat{point_feat.shape}, scalars{dummy_scalars.shape}")
    print(f"   Outputs: steer{steer.shape}, throttle{throttle.shape}, brake{brake.shape}")
    print(f"   Parameters: {count_parameters(fusion):,}")
    
    # Test complete model
    print("\n5. Complete MultimodalDrivingModel")
    for lidar_mode in ['pointnet', 'bev', 'sector_hist']:
        print(f"\n   Mode: {lidar_mode}")
        model = MultimodalDrivingModel(lidar_mode=lidar_mode, pretrained_image=False)
        
        if lidar_mode == 'pointnet':
            dummy_lidar = torch.randn(batch_size, 4096, 4)
        elif lidar_mode == 'bev':
            dummy_lidar = torch.randn(batch_size, 1, 128, 128)
        else:  # sector_hist
            dummy_lidar = torch.randn(batch_size, 108)
        
        outputs = model(dummy_image, dummy_lidar, dummy_scalars)
        print(f"      LiDAR input: {dummy_lidar.shape}")
        print(f"      Outputs: steer{outputs['steer'].shape}, throttle{outputs['throttle'].shape}, brake{outputs['brake'].shape}")
        print(f"      Total parameters: {count_parameters(model):,}")
    
    print("\n" + "="*60)
    print("All model tests passed!")
    print("="*60)


if __name__ == '__main__':
    test_model()

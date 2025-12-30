"""
Approach 1: Heatmap-based Landmark Detection with ResNet Backbone

Hypothesis:
- Heatmap regression provides spatial probability distributions
- More robust than direct coordinate regression
- Captures uncertainty in landmark localization
- Allows for implicit spatial reasoning

Architecture:
- Encoder: ResNet50 (pretrained on ImageNet)
- Decoder: Upsampling layers to generate heatmaps
- Output: 4 heatmaps (one per landmark) at reduced resolution

Advantages:
1. Implicit handling of landmark ambiguity
2. Spatial context preserved throughout network
3. Visual interpretability through heatmap visualization
4. Proven effective in pose estimation literature

Disadvantages:
1. Higher computational cost (generate full heatmaps)
2. Resolution-dependent (limited by heatmap size)
3. Requires post-processing (argmax to extract coordinates)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Dict


class HeatmapLandmarkDetector(nn.Module):
    """
    Heatmap-based landmark detection model.
    
    Based on the approach from:
    "Regressing Heatmaps for Multiple Landmark Localization using CNNs"
    """
    
    def __init__(
        self,
        num_landmarks: int = 4,
        heatmap_size: Tuple[int, int] = (128, 128),
        backbone: str = 'resnet50',
        pretrained: bool = True,
    ):
        """
        Args:
            num_landmarks: Number of landmarks to detect (4 for BPD/OFD)
            heatmap_size: Size of output heatmaps (H, W)
            backbone: Backbone architecture
            pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        
        # Encoder: ResNet backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract feature layers (remove FC layer)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Get feature dimensions based on backbone
        if 'resnet50' in backbone or 'resnet101' in backbone:
            self.feature_dim = 2048
        else:
            self.feature_dim = 512
        
        # Decoder: Upsampling to generate heatmaps
        self.decoder = nn.Sequential(
            # 2048 -> 512
            nn.Conv2d(self.feature_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 512 -> 256
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 256 -> 128
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 128 -> 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        
        # Final heatmap generation layer
        self.heatmap_head = nn.Conv2d(64, num_landmarks, kernel_size=1)
        
        # Optional: Coordinate regression head (auxiliary task)
        self.coord_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_landmarks * 2),  # (x, y) for each landmark
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Dictionary containing:
                - 'heatmaps': Predicted heatmaps [B, 4, H_hm, W_hm]
                - 'coordinates': Extracted coordinates [B, 4, 2]
                - 'coord_regression': Direct coordinate regression [B, 4, 2]
        """
        # Encoder
        features = self.encoder(x)  # [B, 2048, H/32, W/32]
        
        # Decoder
        decoded = self.decoder(features)  # [B, 64, H/2, W/2]
        
        # Generate heatmaps
        heatmaps = self.heatmap_head(decoded)  # [B, 4, H/2, W/2]
        
        # Resize heatmaps to target size if needed
        if heatmaps.shape[2:] != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps,
                size=self.heatmap_size,
                mode='bilinear',
                align_corners=True
            )
        
        # Extract coordinates from heatmaps
        coordinates = self.heatmaps_to_coordinates(heatmaps)
        
        # Direct coordinate regression (auxiliary)
        coord_regression = self.coord_head(features)
        coord_regression = coord_regression.view(-1, self.num_landmarks, 2)
        
        return {
            'heatmaps': heatmaps,
            'coordinates': coordinates,
            'coord_regression': coord_regression,
        }
    
    def heatmaps_to_coordinates(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Extract landmark coordinates from heatmaps.
        
        Methods:
        1. Argmax: Find peak location (simple but discrete)
        2. Soft-argmax: Weighted average (differentiable, sub-pixel accuracy)
        
        We use soft-argmax for better gradient flow.
        
        Args:
            heatmaps: [B, num_landmarks, H, W]
            
        Returns:
            coordinates: [B, num_landmarks, 2] (x, y)
        """
        batch_size, num_landmarks, h, w = heatmaps.shape
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(batch_size, num_landmarks, -1)
        
        # Apply softmax to get probability distribution
        heatmaps_softmax = F.softmax(heatmaps_flat, dim=2)
        heatmaps_softmax = heatmaps_softmax.view(batch_size, num_landmarks, h, w)
        
        # Create coordinate grids
        x_coords = torch.arange(w, device=heatmaps.device).float()
        y_coords = torch.arange(h, device=heatmaps.device).float()
        
        # Compute expected x and y coordinates
        x_expected = torch.sum(heatmaps_softmax * x_coords.view(1, 1, 1, w), dim=(2, 3))
        y_expected = torch.sum(heatmaps_softmax * y_coords.view(1, 1, h, 1), dim=(2, 3))
        
        # Stack coordinates
        coordinates = torch.stack([x_expected, y_expected], dim=2)
        
        return coordinates


class ImprovedHeatmapDetector(nn.Module):
    """
    Improved heatmap detector with:
    1. Skip connections (U-Net style)
    2. Attention mechanisms
    3. Multi-scale feature fusion
    
    Hypothesis: Multi-scale features capture both fine details and global context
    """
    
    def __init__(
        self,
        num_landmarks: int = 4,
        heatmap_size: Tuple[int, int] = (128, 128),
        backbone: str = 'resnet50',
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        
        # Encoder with intermediate features
        resnet = models.resnet50(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(*list(resnet.children())[:4])  # -> 64 channels
        self.layer1 = resnet.layer1  # -> 256 channels
        self.layer2 = resnet.layer2  # -> 512 channels
        self.layer3 = resnet.layer3  # -> 1024 channels
        self.layer4 = resnet.layer4  # -> 2048 channels
        
        # Decoder with skip connections
        self.up1 = self._make_decoder_block(2048, 1024)
        self.up2 = self._make_decoder_block(2048, 512)  # 1024 + 1024 from skip
        self.up3 = self._make_decoder_block(1024, 256)  # 512 + 512 from skip
        self.up4 = self._make_decoder_block(512, 128)   # 256 + 256 from skip
        
        # Attention modules
        self.attention3 = SpatialAttention(1024)
        self.attention2 = SpatialAttention(512)
        self.attention1 = SpatialAttention(256)
        
        # Final heatmap generation
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks, kernel_size=1),
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a decoder block with upsampling."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with skip connections."""
        # Encoder
        x1 = self.conv1(x)      # 1/4 size, 64 channels
        x2 = self.layer1(x1)    # 1/4 size, 256 channels
        x3 = self.layer2(x2)    # 1/8 size, 512 channels
        x4 = self.layer3(x3)    # 1/16 size, 1024 channels
        x5 = self.layer4(x4)    # 1/32 size, 2048 channels
        
        # Decoder with skip connections
        d4 = self.up1(x5)                                    # 1/16 size, 1024 channels
        d4 = torch.cat([d4, self.attention3(x4)], dim=1)     # Skip connection
        
        d3 = self.up2(d4)                                    # 1/8 size, 512 channels
        d3 = torch.cat([d3, self.attention2(x3)], dim=1)     # Skip connection
        
        d2 = self.up3(d3)                                    # 1/4 size, 256 channels
        d2 = torch.cat([d2, self.attention1(x2)], dim=1)     # Skip connection
        
        d1 = self.up4(d2)                                    # 1/2 size, 128 channels
        
        # Generate heatmaps
        heatmaps = self.final_conv(d1)
        
        # Resize to target size
        if heatmaps.shape[2:] != self.heatmap_size:
            heatmaps = F.interpolate(
                heatmaps,
                size=self.heatmap_size,
                mode='bilinear',
                align_corners=True
            )
        
        # Extract coordinates
        coordinates = HeatmapLandmarkDetector.heatmaps_to_coordinates(None, heatmaps)
        
        return {
            'heatmaps': heatmaps,
            'coordinates': coordinates,
        }


class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on relevant regions.
    
    Rationale: Helps the model focus on cranium region and suppress background
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention."""
        attention = self.conv(x)
        return x * attention


# Loss functions for heatmap training
class HeatmapLoss(nn.Module):
    """
    Combined loss for heatmap-based landmark detection.
    
    Components:
    1. MSE on heatmaps: Primary training signal
    2. MSE on extracted coordinates: Ensures accurate localization
    3. Geometric constraint: Enforces perpendicularity of BPD/OFD
    """
    
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        coord_weight: float = 0.5,
        geometric_weight: float = 0.1,
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.geometric_weight = geometric_weight
    
    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        gt_heatmaps: torch.Tensor,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred_heatmaps: Predicted heatmaps [B, 4, H, W]
            gt_heatmaps: Ground truth heatmaps [B, 4, H, W]
            pred_coords: Predicted coordinates [B, 4, 2]
            gt_coords: Ground truth coordinates [B, 4, 2]
            
        Returns:
            Dictionary with total loss and components
        """
        # Heatmap MSE loss
        heatmap_loss = F.mse_loss(pred_heatmaps, gt_heatmaps)
        
        # Coordinate MSE loss
        coord_loss = F.mse_loss(pred_coords, gt_coords)
        
        # Geometric constraint: BPD and OFD should be perpendicular
        geometric_loss = self.geometric_constraint_loss(pred_coords)
        
        # Total loss
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.coord_weight * coord_loss +
            self.geometric_weight * geometric_loss
        )
        
        return {
            'total': total_loss,
            'heatmap': heatmap_loss,
            'coordinate': coord_loss,
            'geometric': geometric_loss,
        }
    
    def geometric_constraint_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Enforce perpendicularity of BPD and OFD lines.
        
        Medical constraint: BPD and OFD are orthogonal diameters of the ellipse.
        
        Args:
            coords: Landmark coordinates [B, 4, 2]
                    [OFD1, OFD2, BPD1, BPD2]
        """
        # Extract landmarks
        ofd1, ofd2 = coords[:, 0], coords[:, 1]
        bpd1, bpd2 = coords[:, 2], coords[:, 3]
        
        # Compute direction vectors
        ofd_vec = ofd2 - ofd1
        bpd_vec = bpd2 - bpd1
        
        # Dot product (should be close to 0 for perpendicular vectors)
        dot_product = torch.sum(ofd_vec * bpd_vec, dim=1)
        
        # Normalize by vector magnitudes
        ofd_mag = torch.norm(ofd_vec, dim=1) + 1e-6
        bpd_mag = torch.norm(bpd_vec, dim=1) + 1e-6
        
        # Cosine of angle (should be close to 0)
        cos_angle = dot_product / (ofd_mag * bpd_mag)
        
        # Loss: penalize deviation from 0
        loss = torch.mean(cos_angle ** 2)
        
        return loss

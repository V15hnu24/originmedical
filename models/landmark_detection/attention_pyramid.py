"""
Approach 3: Attention-Based Feature Pyramid Network

Hypothesis:
- Multi-scale feature pyramid captures both fine details and global context
- Attention mechanisms focus on relevant anatomical regions
- Inspired by recent work on cephalometric landmark detection

Architecture:
- Backbone: ResNet34 with Feature Pyramid Network (FPN)
- Attention: Spatial and channel attention at each pyramid level
- Output: Coordinates from multi-scale features

Advantages:
1. Handles varying fetal head sizes (scale invariance)
2. Attention suppresses background and artifacts
3. Multi-scale features improve robustness
4. Interpretable attention maps

Disadvantages:
1. More complex architecture
2. Slower training than simple regression
3. More hyperparameters to tune

Based on:
"Cephalometric Landmark Detection by Attentive Feature Pyramid Fusion"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).
    
    Rationale: Different feature channels encode different information.
    Channel attention weights channels by their relevance for landmark detection.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and sigmoid
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).
    
    Rationale: Highlights spatial regions relevant for landmarks (cranium area).
    Suppresses background and ultrasound artifacts.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average and max across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Combines channel and spatial attention sequentially.
    Paper: "CBAM: Convolutional Block Attention Module"
    """
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction.
    
    Rationale:
    - Top-down pathway creates semantically strong features at all scales
    - Lateral connections preserve spatial details
    - Essential for detecting landmarks on varying head sizes
    """
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        
        self.out_channels = out_channels
        
        # Lateral convolutions (reduce channel dimension)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        # Output convolutions (smooth features)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for _ in in_channels_list
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from different scales (low to high res)
            
        Returns:
            List of FPN feature maps
        """
        # Process from high-level (coarse) to low-level (fine)
        laterals = [conv(feat) for feat, conv in zip(features, self.lateral_convs)]
        
        # Top-down pathway with lateral connections
        fpn_features = []
        for i in range(len(laterals) - 1, -1, -1):
            if i == len(laterals) - 1:
                # Highest level (most semantic)
                fpn_feat = laterals[i]
            else:
                # Add upsampled higher-level features
                upsampled = F.interpolate(
                    fpn_feat,
                    size=laterals[i].shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
                fpn_feat = laterals[i] + upsampled
            
            # Apply output convolution
            fpn_feat = self.output_convs[i](fpn_feat)
            fpn_features.insert(0, fpn_feat)
        
        return fpn_features


class AttentionFeaturePyramid(nn.Module):
    """
    Attention-based Feature Pyramid Network for landmark detection.
    
    Combines:
    1. ResNet backbone for feature extraction
    2. Feature Pyramid Network for multi-scale features
    3. CBAM attention at each scale
    4. Coordinate regression from fused features
    """
    
    def __init__(
        self,
        num_landmarks: int = 4,
        backbone: str = 'resnet34',
        pretrained: bool = True,
        fpn_channels: int = 256,
        attention_type: str = 'cbam',  # 'cbam', 'channel', or 'spatial'
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        self.fpn_channels = fpn_channels
        
        # Backbone: ResNet with intermediate features
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.feature_channels = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.feature_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract layers
        self.conv1 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(self.feature_channels, fpn_channels)
        
        # Attention modules for each FPN level
        if attention_type == 'cbam':
            self.attention_modules = nn.ModuleList([
                CBAM(fpn_channels) for _ in self.feature_channels
            ])
        elif attention_type == 'channel':
            self.attention_modules = nn.ModuleList([
                ChannelAttention(fpn_channels) for _ in self.feature_channels
            ])
        elif attention_type == 'spatial':
            self.attention_modules = nn.ModuleList([
                SpatialAttention() for _ in self.feature_channels
            ])
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * len(self.feature_channels), 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(256, num_landmarks * 2),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Dictionary with coordinates and attention maps
        """
        batch_size = x.size(0)
        
        # Extract multi-scale features from backbone
        x1 = self.conv1(x)
        x2 = self.layer1(x1)  # 1/4 resolution
        x3 = self.layer2(x2)  # 1/8 resolution
        x4 = self.layer3(x3)  # 1/16 resolution
        x5 = self.layer4(x4)  # 1/32 resolution
        
        features = [x2, x3, x4, x5]
        
        # Feature Pyramid Network
        fpn_features = self.fpn(features)
        
        # Apply attention to each FPN level
        attended_features = []
        attention_maps = []
        for feat, attn_module in zip(fpn_features, self.attention_modules):
            attended = attn_module(feat)
            attended_features.append(attended)
            
            # Store attention map for visualization
            if isinstance(attn_module, (SpatialAttention, CBAM)):
                # Extract spatial attention map
                avg_out = torch.mean(attended, dim=1, keepdim=True)
                attention_maps.append(avg_out)
        
        # Resize all features to same size for fusion
        target_size = attended_features[0].shape[2:]
        resized_features = []
        for feat in attended_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=True
                )
            resized_features.append(feat)
        
        # Fuse multi-scale features
        fused = torch.cat(resized_features, dim=1)
        fused = self.fusion(fused)
        
        # Global pooling
        pooled = self.global_pool(fused).flatten(1)
        
        # Regression
        coords = self.regression_head(pooled)
        coords = coords.view(batch_size, self.num_landmarks, 2)
        
        return {
            'coordinates': coords,
            'features': pooled,
            'attention_maps': attention_maps,
            'fpn_features': fpn_features,
        }


class HierarchicalAttentionRegression(nn.Module):
    """
    Hierarchical attention-based regression.
    
    Hypothesis: Use coarse-to-fine landmark detection
    1. First detect rough cranium location
    2. Then refine individual landmark positions
    
    Similar to cascaded regression but end-to-end trainable.
    """
    
    def __init__(
        self,
        num_landmarks: int = 4,
        backbone: str = 'resnet34',
        pretrained: bool = True,
        fpn_channels: int = 256,
    ):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        
        # Base FPN model
        self.base_model = AttentionFeaturePyramid(
            num_landmarks=num_landmarks,
            backbone=backbone,
            pretrained=pretrained,
            fpn_channels=fpn_channels,
        )
        
        # Refinement network (takes initial predictions + features)
        self.refinement = nn.Sequential(
            nn.Linear(512 + num_landmarks * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_landmarks * 2),  # Residual offset
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward with coarse-to-fine refinement."""
        # Initial prediction
        outputs = self.base_model(x)
        coords_coarse = outputs['coordinates']
        features = outputs['features']
        
        # Concatenate features and coarse predictions
        coords_flat = coords_coarse.view(coords_coarse.size(0), -1)
        combined = torch.cat([features, coords_flat], dim=1)
        
        # Predict refinement offset
        offset = self.refinement(combined)
        offset = offset.view(-1, self.num_landmarks, 2)
        
        # Refined coordinates
        coords_refined = coords_coarse + offset
        
        return {
            'coordinates': coords_refined,
            'coordinates_coarse': coords_coarse,
            'offset': offset,
            'features': features,
            'attention_maps': outputs['attention_maps'],
        }
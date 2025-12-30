"""
Approach 2: Direct Coordinate Regression with EfficientNet

Hypothesis:
- Direct regression is simpler and faster for well-defined landmarks
- EfficientNet provides excellent accuracy-efficiency tradeoff
- End-to-end training without heatmap intermediate representation

Architecture:
- Encoder: EfficientNet-B3 (pretrained on ImageNet)
- Head: Global pooling + fully connected layers
- Output: 8 values (4 landmarks × 2 coordinates)

Advantages:
1. Fast inference (no heatmap generation)
2. Memory efficient
3. Direct optimization of coordinate accuracy
4. Simple architecture

Disadvantages:
1. Less robust to occlusions/ambiguity
2. No spatial probability distribution
3. Harder to visualize predictions
4. May struggle with outliers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional


class CoordinateRegressionModel(nn.Module):
    """
    Direct coordinate regression for landmark detection.
    
    Uses a CNN backbone + regression head to directly predict
    (x, y) coordinates for each landmark.
    """
    
    def __init__(
        self,
        num_landmarks: int = 4,
        backbone: str = 'efficientnet_b3',
        pretrained: bool = True,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            num_landmarks: Number of landmarks to detect
            backbone: Backbone architecture from timm library
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate in regression head
            use_batch_norm: Use batch normalization in head
        """
        super().__init__()
        
        self.num_landmarks = num_landmarks
        
        # Load backbone from timm (PyTorch Image Models)
        # timm provides many efficient architectures with pretrained weights
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='',  # We'll add custom pooling
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Regression head
        layers = []
        
        # First hidden layer
        layers.extend([
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ])
        
        # Second hidden layer
        layers.extend([
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ])
        
        # Output layer
        layers.append(nn.Linear(256, num_landmarks * 2))
        
        self.regression_head = nn.Sequential(*layers)
        
        # Initialize output layer with small weights
        # Rationale: Start with predictions near image center
        nn.init.normal_(self.regression_head[-1].weight, std=0.001)
        nn.init.constant_(self.regression_head[-1].bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Dictionary containing:
                - 'coordinates': Predicted coordinates [B, num_landmarks, 2]
                - 'features': Intermediate features [B, feature_dim]
        """
        # Extract features
        features = self.backbone(x)  # [B, C, H', W']
        
        # Global pooling
        pooled = self.global_pool(features)  # [B, C, 1, 1]
        pooled = pooled.flatten(1)  # [B, C]
        
        # Regression
        coords = self.regression_head(pooled)  # [B, num_landmarks * 2]
        
        # Reshape to [B, num_landmarks, 2]
        coords = coords.view(-1, self.num_landmarks, 2)
        
        return {
            'coordinates': coords,
            'features': pooled,
        }


class MultiScaleCoordinateRegression(nn.Module):
    """
    Enhanced coordinate regression with multi-scale features.
    
    Hypothesis: Combining features from multiple scales improves localization
    - Low-level features: Fine details (edges, textures)
    - High-level features: Semantic context (cranium location)
    
    Similar to Feature Pyramid Networks but for regression.
    """
    
    def __init__(
        self,
        num_landmarks: int = 4,
        backbone: str = 'efficientnet_b3',
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        
        # Backbone with intermediate features
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,  # Return intermediate features
            out_indices=(1, 2, 3, 4),  # Multiple scales
        )
        
        # Get feature dimensions for each scale
        feature_info = self.backbone.feature_info
        self.feature_dims = [f['num_chs'] for f in feature_info]
        
        # Adaptive pooling for each scale
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in self.feature_dims
        ])
        
        # Total feature dimension
        total_dim = sum(self.feature_dims)
        
        # Regression head with multi-scale features
        self.regression_head = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, num_landmarks * 2),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-scale features."""
        # Extract multi-scale features
        features = self.backbone(x)  # List of [B, C_i, H_i, W_i]
        
        # Pool each scale
        pooled_features = []
        for feat, pool in zip(features, self.pools):
            pooled = pool(feat).flatten(1)
            pooled_features.append(pooled)
        
        # Concatenate all scales
        combined = torch.cat(pooled_features, dim=1)
        
        # Regression
        coords = self.regression_head(combined)
        coords = coords.view(-1, self.num_landmarks, 2)
        
        return {
            'coordinates': coords,
            'features': combined,
        }


class CoordinateRegressionWithConfidence(nn.Module):
    """
    Coordinate regression with uncertainty estimation.
    
    Hypothesis: Predicting confidence helps identify difficult cases
    - Model outputs both coordinates and confidence scores
    - Low confidence indicates potential failures
    - Can be used for active learning or alerting clinicians
    
    Architecture adds a confidence head alongside coordinate head.
    """
    
    def __init__(
        self,
        num_landmarks: int = 4,
        backbone: str = 'efficientnet_b3',
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_landmarks = num_landmarks
        
        # Backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        
        self.feature_dim = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Shared features
        self.shared_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Coordinate head
        self.coord_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_landmarks * 2),
        )
        
        # Confidence head (one confidence per landmark)
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_landmarks),
            nn.Sigmoid(),  # Confidence in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with confidence prediction."""
        # Extract features
        features = self.backbone(x)
        pooled = self.global_pool(features).flatten(1)
        
        # Shared processing
        shared = self.shared_head(pooled)
        
        # Coordinate prediction
        coords = self.coord_head(shared)
        coords = coords.view(-1, self.num_landmarks, 2)
        
        # Confidence prediction
        confidence = self.confidence_head(shared)  # [B, num_landmarks]
        
        return {
            'coordinates': coords,
            'confidence': confidence,
            'features': pooled,
        }


# Loss functions for coordinate regression
class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for robust landmark localization.
    
    Paper: "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    
    Rationale:
    - More robust to outliers than MSE
    - Focuses on difficult samples (large errors)
    - Adaptive behavior: linear for small errors, logarithmic for large errors
    - Better gradient properties than L1 loss
    
    Formula:
    AWing(x) = {
        omega * ln(1 + |x|/epsilon)           if |x| < theta
        alpha * |x| - C                        otherwise
    }
    
    where C is chosen to make the function continuous.
    """
    
    def __init__(
        self,
        omega: float = 14,
        theta: float = 0.5,
        epsilon: float = 1,
        alpha: float = 2.1,
    ):
        """
        Args:
            omega: Scale factor for small errors
            theta: Threshold between regimes
            epsilon: Smoothness parameter
            alpha: Slope for large errors
        """
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        
        # Compute C for continuity
        self.C = self.omega * (1 - np.log(1 + self.omega / self.epsilon * self.theta))
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Adaptive Wing Loss.
        
        Args:
            pred: Predicted coordinates [B, N, 2]
            target: Ground truth coordinates [B, N, 2]
            weights: Optional per-landmark weights [B, N]
            
        Returns:
            Loss value
        """
        # Compute absolute difference
        diff = torch.abs(pred - target)
        
        # Compute loss
        loss = torch.where(
            diff < self.theta,
            self.omega * torch.log(1 + diff / self.epsilon),
            self.alpha * diff - self.C
        )
        
        # Average over coordinates
        loss = loss.mean(dim=-1)  # [B, N]
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        # Average over batch and landmarks
        return loss.mean()


import numpy as np


class CoordinateRegressionLoss(nn.Module):
    """
    Combined loss for coordinate regression.
    
    Components:
    1. Adaptive Wing Loss: Main localization loss
    2. Geometric Constraint: Enforce perpendicularity
    3. Confidence Loss: If using confidence prediction
    """
    
    def __init__(
        self,
        loss_type: str = 'adaptive_wing',
        geometric_weight: float = 0.1,
        use_confidence: bool = False,
    ):
        """
        Args:
            loss_type: 'mse', 'smooth_l1', 'wing', or 'adaptive_wing'
            geometric_weight: Weight for geometric constraint
            use_confidence: Whether using confidence prediction
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.geometric_weight = geometric_weight
        self.use_confidence = use_confidence
        
        # Primary loss function
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        elif loss_type == 'adaptive_wing':
            self.criterion = AdaptiveWingLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted coordinates [B, 4, 2]
            target: Ground truth coordinates [B, 4, 2]
            confidence: Optional confidence scores [B, 4]
            
        Returns:
            Dictionary with total loss and components
        """
        # Main coordinate loss
        if self.use_confidence and confidence is not None:
            # Weight loss by confidence (higher confidence = higher weight)
            coord_loss = self.criterion(pred, target, weights=confidence)
            
            # Confidence regularization (prevent always predicting low confidence)
            confidence_loss = -torch.log(confidence + 1e-6).mean()
        else:
            coord_loss = self.criterion(pred, target)
            confidence_loss = torch.tensor(0.0, device=pred.device)
        
        # Geometric constraint
        geometric_loss = self.geometric_constraint_loss(pred)
        
        # Total loss
        total_loss = coord_loss + self.geometric_weight * geometric_loss
        if self.use_confidence:
            total_loss += 0.1 * confidence_loss
        
        return {
            'total': total_loss,
            'coordinate': coord_loss,
            'geometric': geometric_loss,
            'confidence': confidence_loss,
        }
    
    def geometric_constraint_loss(self, coords: torch.Tensor) -> torch.Tensor:
        """Enforce perpendicularity of BPD and OFD."""
        ofd1, ofd2 = coords[:, 0], coords[:, 1]
        bpd1, bpd2 = coords[:, 2], coords[:, 3]
        
        ofd_vec = ofd2 - ofd1
        bpd_vec = bpd2 - bpd1
        
        dot_product = torch.sum(ofd_vec * bpd_vec, dim=1)
        ofd_mag = torch.norm(ofd_vec, dim=1) + 1e-6
        bpd_mag = torch.norm(bpd_vec, dim=1) + 1e-6
        
        cos_angle = dot_product / (ofd_mag * bpd_mag)
        
        return torch.mean(cos_angle ** 2)

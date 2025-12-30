"""
Segmentation Model 1: U-Net for Cranium Segmentation

Hypothesis:
- U-Net is the gold standard for medical image segmentation
- Skip connections preserve spatial details essential for accurate boundaries
- Proven effective on small datasets with augmentation

Architecture:
- Encoder-decoder with skip connections
- 5 levels of downsampling/upsampling
- Encoder: ResNet34 (pretrained)
- Decoder: Transposed convolutions + skip connections

Advantages:
1. Well-established and reliable
2. Good with limited data
3. Excellent boundary localization
4. Fast training and inference

Disadvantages:
1. Fixed receptive field
2. No multi-scale context aggregation
3. May struggle with very small or large objects
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Dict, Optional


class UNet(nn.Module):
    """
    U-Net for fetal cranium segmentation.
    
    Uses segmentation_models_pytorch for easy implementation
    with various encoder backbones.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        """
        Args:
            encoder_name: Encoder backbone
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels
            classes: Number of output classes (1 for binary)
            activation: Output activation ('sigmoid', 'softmax', or None)
        """
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            Segmentation mask [B, 1, H, W]
        """
        return self.model(x)


class AttentionUNet(nn.Module):
    """
    U-Net with Attention Gates.
    
    Hypothesis:
    - Attention gates highlight relevant features and suppress irrelevant ones
    - Particularly useful for medical imaging with background clutter
    - Improves boundary localization by focusing on edges
    
    Paper: "Attention U-Net: Learning Where to Look for the Pancreas"
    
    Advantages:
    1. Better feature selection
    2. Improved boundary accuracy
    3. Fewer false positives in background
    4. Interpretable attention maps
    
    Disadvantages:
    1. More parameters than standard U-Net
    2. Slightly slower training
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        attention_type: str = 'scse',  # 'scse' or None
        activation: Optional[str] = None,
    ):
        """
        Args:
            encoder_name: Encoder backbone
            encoder_weights: Pretrained weights
            in_channels: Number of input channels
            classes: Number of output classes
            attention_type: Type of attention ('scse' for Spatial and Channel Squeeze & Excitation)
            activation: Output activation
        """
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            decoder_attention_type=attention_type,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class UNetPlusPlus(nn.Module):
    """
    U-Net++ (Nested U-Net) for improved feature propagation.
    
    Hypothesis:
    - Dense skip connections capture features at multiple semantic levels
    - Reduces semantic gap between encoder and decoder
    - Better for complex boundary shapes (like cranium ellipse)
    
    Paper: "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
    
    Advantages:
    1. More flexible feature aggregation
    2. Better gradient flow
    3. Can use deep supervision
    4. Superior performance on complex shapes
    
    Disadvantages:
    1. More parameters and memory
    2. Slower training
    3. Risk of overfitting on small datasets
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        deep_supervision: bool = False,
        activation: Optional[str] = None,
    ):
        """
        Args:
            encoder_name: Encoder backbone
            encoder_weights: Pretrained weights
            in_channels: Number of input channels
            classes: Number of output classes
            deep_supervision: Use deep supervision (multiple outputs)
            activation: Output activation
        """
        super().__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
        
        self.deep_supervision = deep_supervision
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


# Segmentation Loss Functions
class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Rationale:
    - Handles class imbalance (cranium vs background)
    - Directly optimizes Dice coefficient (primary metric)
    - Works well for medical image segmentation
    
    Formula:
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: Predicted mask [B, 1, H, W] (logits or probabilities)
            target: Ground truth mask [B, 1, H, W] (binary)
            
        Returns:
            Dice loss value
        """
        # Apply sigmoid if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for segmentation.
    
    Rationale:
    - Focuses on hard-to-classify pixels
    - Reduces impact of easy background pixels
    - Improves boundary accuracy
    
    Paper: "Focal Loss for Dense Object Detection"
    
    Formula:
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.
        
        Args:
            pred: Predicted mask [B, 1, H, W] (logits)
            target: Ground truth mask [B, 1, H, W] (binary)
            
        Returns:
            Focal loss value
        """
        # Binary cross entropy with logits
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Probability
        p = torch.sigmoid(pred)
        p_t = p * target + (1 - p) * (1 - target)
        
        # Focal term
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        # Focal loss
        loss = alpha_t * focal_term * bce
        
        return loss.mean()


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for segmentation: Dice + Focal + Boundary
    
    Rationale:
    - Dice: Handles class imbalance, optimizes overlap
    - Focal: Focuses on difficult pixels (edges)
    - Boundary: Explicitly penalizes boundary errors
    
    Hypothesis: Combining multiple loss functions improves:
    1. Overall segmentation accuracy (Dice)
    2. Boundary localization (Focal + Boundary)
    3. Robustness to class imbalance (Dice + Focal)
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        boundary_weight: float = 0.2,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted mask [B, 1, H, W] (logits)
            target: Ground truth mask [B, 1, H, W] (binary)
            
        Returns:
            Dictionary with total loss and components
        """
        # Dice loss
        dice = self.dice_loss(pred, target)
        
        # Focal loss
        focal = self.focal_loss(pred, target)
        
        # Boundary loss (compute distance transform)
        boundary = self.boundary_loss(pred, target)
        
        # Total loss
        total = (
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.boundary_weight * boundary
        )
        
        return {
            'total': total,
            'dice': dice,
            'focal': focal,
            'boundary': boundary,
        }
    
    def boundary_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Boundary-aware loss.
        
        Emphasizes errors near the boundary of the cranium.
        Uses Sobel filter to detect edges.
        """
        # Apply sigmoid to predictions
        pred_sig = torch.sigmoid(pred)
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).to(pred.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).to(pred.device)
        
        # Compute gradients
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-6)
        
        # Normalize edge map
        target_edge = (target_edge > 0.1).float()
        
        # Weight BCE by edge proximity
        bce = F.binary_cross_entropy(pred_sig, target, reduction='none')
        weighted_bce = bce * (1 + 5 * target_edge)  # 5x weight on boundaries
        
        return weighted_bce.mean()

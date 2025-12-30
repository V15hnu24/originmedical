"""
Segmentation Model 2: DeepLabV3+ for Cranium Segmentation

Hypothesis:
- Atrous (dilated) convolutions capture multi-scale context
- Atrous Spatial Pyramid Pooling (ASPP) handles varying cranium sizes
- Encoder-decoder architecture preserves spatial details

Architecture:
- Encoder: ResNet50/101 with atrous convolutions
- ASPP: Multiple parallel atrous convolutions with different rates
- Decoder: Lightweight decoder with skip connections

Advantages:
1. Excellent multi-scale context
2. Large receptive field without losing resolution
3. State-of-the-art semantic segmentation
4. Good for varying object sizes

Disadvantages:
1. More computationally expensive
2. Higher memory requirements
3. May be overkill for simple ellipse shapes
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for fetal cranium segmentation.
    
    Paper: "Encoder-Decoder with Atrous Separable Convolution 
            for Semantic Image Segmentation"
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        """
        Args:
            encoder_name: Encoder backbone
            encoder_weights: Pretrained weights
            in_channels: Number of input channels
            classes: Number of output classes
            activation: Output activation
        """
        super().__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class DeepLabV3(nn.Module):
    """
    DeepLabV3 (without decoder) for cranium segmentation.
    
    Simpler than V3+ but still very effective.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet101',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        """
        Args:
            encoder_name: Encoder backbone
            encoder_weights: Pretrained weights
            in_channels: Number of input channels
            classes: Number of output classes
            activation: Output activation
        """
        super().__init__()
        
        self.model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class FPN(nn.Module):
    """
    Feature Pyramid Network for segmentation.
    
    Hypothesis:
    - Multi-scale feature pyramid helps with varying cranium sizes
    - Top-down pathway creates semantically strong features
    - Lighter than DeepLabV3+ but with similar multi-scale benefits
    
    Advantages:
    1. Good balance of accuracy and speed
    2. Multi-scale feature fusion
    3. Works well for objects of varying sizes
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()
        
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class PSPNet(nn.Module):
    """
    Pyramid Scene Parsing Network for segmentation.
    
    Hypothesis:
    - Pyramid pooling module captures global context
    - Multiple pooling scales help understand scene layout
    - Good for understanding cranium within ultrasound image context
    
    Paper: "Pyramid Scene Parsing Network"
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()
        
        self.model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

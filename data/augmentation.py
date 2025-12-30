"""
Data augmentation strategies for fetal ultrasound images.

Key Considerations:
1. Preserve anatomical validity (avoid unrealistic transformations)
2. Account for ultrasound-specific artifacts
3. Augment both images and landmark coordinates consistently
4. Balance dataset diversity with clinical realism
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Tuple


def get_augmentation_pipeline(
    image_size: Tuple[int, int] = (512, 512),
    augment: bool = True,
    mode: str = 'landmark',
    augmentation_strength: str = 'medium',  # 'light', 'medium', 'heavy'
) -> A.Compose:
    """
    Create augmentation pipeline using Albumentations.
    
    Rationale for each augmentation:
    
    1. **Geometric Transformations**:
       - Rotation: Fetal head can be at various angles
       - Affine: Accounts for probe positioning variations
       - Elastic: Simulates natural anatomical variations (subtle)
    
    2. **Intensity Transformations**:
       - Brightness/Contrast: Simulates different ultrasound machine settings
       - Gamma: Adjusts exposure non-linearly (realistic for US)
       - Gaussian Noise: Simulates electronic noise
    
    3. **Ultrasound-Specific**:
       - Blur: Simulates out-of-focus regions
       - Motion Blur: Simulates fetal/maternal movement
    
    Args:
        image_size: Target image size (H, W)
        augment: Whether to apply augmentations (False for validation)
        mode: 'landmark' or 'segmentation'
        augmentation_strength: Intensity of augmentations
        
    Returns:
        Albumentations Compose pipeline
    """
    
    # Set parameters based on strength
    if augmentation_strength == 'light':
        rotation_limit = 10
        scale_limit = 0.1
        shift_limit = 0.05
        brightness_limit = 0.1
        contrast_limit = 0.1
        aug_prob = 0.3
    elif augmentation_strength == 'medium':
        rotation_limit = 20
        scale_limit = 0.15
        shift_limit = 0.1
        brightness_limit = 0.2
        contrast_limit = 0.2
        aug_prob = 0.5
    else:  # heavy
        rotation_limit = 30
        scale_limit = 0.2
        shift_limit = 0.15
        brightness_limit = 0.3
        contrast_limit = 0.3
        aug_prob = 0.7
    
    if not augment:
        # Validation pipeline: only resize and normalize
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    # Training pipeline with augmentations
    transforms = [
        # Resize first to standardize
        A.Resize(image_size[0], image_size[1]),
        
        # ===== GEOMETRIC AUGMENTATIONS =====
        # Rationale: Account for different probe angles and fetal positions
        
        # Rotation: Fetal head can be oriented at various angles
        A.Rotate(
            limit=rotation_limit,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=aug_prob
        ),
        
        # Affine transformations: Scale, shear, translate
        # Simulates probe positioning variations and different fetal sizes
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotation_limit,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=aug_prob
        ),
        
        # Horizontal flip: Valid for ultrasound (probe can be from either side)
        A.HorizontalFlip(p=0.5),
        
        # Elastic deformations: Simulates natural anatomical variations
        # CONSERVATIVE: Ultrasound anatomy should remain realistic
        A.ElasticTransform(
            alpha=30,
            sigma=5,
            alpha_affine=5,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2  # Lower probability to avoid unrealistic anatomy
        ),
        
        # Grid distortion: Simulates ultrasound beam artifacts
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.1,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.2
        ),
        
        # ===== INTENSITY AUGMENTATIONS =====
        # Rationale: Simulate different ultrasound machine settings and gains
        
        # Random brightness and contrast
        # Simulates gain and TGC (Time Gain Compensation) settings
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=aug_prob
        ),
        
        # Gamma correction: Non-linear intensity transformation
        # Common in ultrasound post-processing
        A.RandomGamma(
            gamma_limit=(90, 110),  # More conservative range
            p=aug_prob
        ),
        
        # Hue-Saturation-Value adjustments
        # Less relevant for grayscale US, but helps with color maps
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        
        # ===== NOISE AND BLUR =====
        # Rationale: Simulate image quality variations
        
        # One of: Gaussian blur, motion blur, median blur
        # Simulates out-of-focus, movement, or smoothing artifacts
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
        ], p=0.3),
        
        # Gaussian noise: Simulates electronic noise
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            mean=0,
            p=0.3
        ),
        
        # ===== ADVANCED AUGMENTATIONS =====
        
        # Random shadow: Simulates acoustic shadowing
        # Common artifact in ultrasound
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.2
        ),
        
        # Coarse dropout: Simulates missing data or artifacts
        # Similar to cutout but with larger holes
        A.CoarseDropout(
            max_holes=8,
            max_height=int(image_size[0] * 0.1),
            max_width=int(image_size[1] * 0.1),
            min_holes=1,
            fill_value=0,
            p=0.2
        ),
    ]
    
    # Add normalization and tensor conversion at the end
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Compose with keypoint parameters
    # remove_invisible=False keeps landmarks even if they go off-screen
    # (important for training stability)
    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False,
            label_fields=None
        )
    )


def get_mixup_augmentation(alpha: float = 0.2):
    """
    Mixup augmentation for improved generalization.
    
    Rationale:
    - Encourages smooth decision boundaries
    - Reduces overfitting on small datasets
    - Particularly effective for landmark detection
    
    Formula: x_mix = lambda * x_i + (1 - lambda) * x_j
             y_mix = lambda * y_i + (1 - lambda) * y_j
    
    Args:
        alpha: Beta distribution parameter (higher = more mixing)
        
    Note: Should be applied in the training loop, not in the dataset
    """
    def mixup(images, landmarks, alpha=alpha):
        """
        Apply mixup augmentation to a batch.
        
        Args:
            images: Batch of images [B, C, H, W]
            landmarks: Batch of landmarks [B, 4, 2]
            alpha: Mixup parameter
            
        Returns:
            Mixed images and landmarks
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = np.random.permutation(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_landmarks = lam * landmarks + (1 - lam) * landmarks[index]
        
        return mixed_images, mixed_landmarks, lam, index
    
    return mixup


def get_cutmix_augmentation(alpha: float = 1.0):
    """
    CutMix augmentation: Cut and paste regions between images.
    
    Rationale:
    - More realistic than mixup for medical images
    - Forces model to localize landmarks
    - Improves robustness to occlusions
    
    Args:
        alpha: Beta distribution parameter
    """
    def cutmix(images, landmarks, alpha=alpha):
        """
        Apply cutmix augmentation to a batch.
        
        Args:
            images: Batch of images [B, C, H, W]
            landmarks: Batch of landmarks [B, 4, 2]
            alpha: Cutmix parameter
            
        Returns:
            Mixed images, original landmarks (landmarks not mixed in cutmix)
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = np.random.permutation(batch_size)
        
        _, _, h, w = images.size()
        
        # Generate random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Cut and paste
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        
        return images, landmarks, lam, index
    
    return cutmix


class UltrasoundSpecificAugmentation:
    """
    Custom augmentations specific to ultrasound imaging.
    
    These simulate real ultrasound artifacts and variations.
    """
    
    @staticmethod
    def add_speckle_noise(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """
        Add speckle noise characteristic of ultrasound imaging.
        
        Rationale: Speckle is multiplicative noise, different from additive Gaussian
        Formula: I_noisy = I + I * noise
        """
        noise = np.random.randn(*image.shape) * intensity
        noisy = image + image * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def simulate_acoustic_shadow(
        image: np.ndarray,
        num_shadows: int = 1,
        intensity: float = 0.5
    ) -> np.ndarray:
        """
        Simulate acoustic shadowing artifacts.
        
        Rationale: Common in ultrasound when sound is blocked by bones/calcifications
        """
        h, w = image.shape[:2]
        result = image.copy()
        
        for _ in range(num_shadows):
            # Random shadow position and size
            x = np.random.randint(0, w - 50)
            y = np.random.randint(0, h // 2)
            width = np.random.randint(20, 80)
            
            # Create shadow region (darkening from top to bottom)
            shadow_mask = np.linspace(1, intensity, h - y)
            shadow_mask = shadow_mask[:, np.newaxis]
            
            if len(result.shape) == 3:
                shadow_mask = shadow_mask[:, :, np.newaxis]
            
            result[y:, x:min(x + width, w)] *= shadow_mask[:h - y]
        
        return result.astype(np.uint8)
    
    @staticmethod
    def simulate_attenuation(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Simulate depth-dependent attenuation.
        
        Rationale: Ultrasound signal weakens with depth
        """
        h = image.shape[0]
        attenuation_mask = np.linspace(1, 1 - strength, h)
        attenuation_mask = attenuation_mask[:, np.newaxis]
        
        if len(image.shape) == 3:
            attenuation_mask = attenuation_mask[:, :, np.newaxis]
        
        return (image * attenuation_mask).astype(np.uint8)


# Augmentation strategy summary for documentation
AUGMENTATION_RATIONALE = """
Augmentation Strategy for Fetal Ultrasound Biometry:

1. **Geometric Augmentations** (p=0.5):
   - Purpose: Account for probe positioning and fetal orientation variations
   - Transforms: Rotation (±20°), Scale (±15%), Translation (±10%)
   - Constraint: Preserve anatomical structure (no extreme distortions)

2. **Intensity Augmentations** (p=0.5):
   - Purpose: Simulate different ultrasound machine settings and operators
   - Transforms: Brightness, contrast, gamma correction
   - Constraint: Maintain tissue differentiation

3. **Noise and Blur** (p=0.3):
   - Purpose: Improve robustness to image quality variations
   - Transforms: Gaussian noise, motion blur, Gaussian blur
   - Constraint: Keep landmarks visible

4. **Ultrasound-Specific** (p=0.2):
   - Purpose: Simulate realistic ultrasound artifacts
   - Transforms: Acoustic shadowing, speckle noise, attenuation
   - Constraint: Clinical validity

5. **Regularization** (p=0.2):
   - Purpose: Prevent overfitting on specific image regions
   - Transforms: Coarse dropout, random erasing
   - Constraint: Don't occlude all landmarks

Expected Dataset Expansion:
- Original: 624 images
- With augmentation: ~3,000-5,000 effective samples per epoch
- Validation: No augmentation (consistent evaluation)

Testing Strategy:
- Test-Time Augmentation (TTA): Average predictions over 4-8 augmented versions
- Expected improvement: 1-3% in landmark localization accuracy
"""

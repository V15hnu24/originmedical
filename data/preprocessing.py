"""
Preprocessing utilities for fetal ultrasound images.

Rationale:
- Ultrasound images have unique characteristics (speckle noise, variable contrast)
- CLAHE improves local contrast without losing global structure
- Denoising reduces speckle while preserving anatomical boundaries
- Normalization ensures consistent input to neural networks
"""
import cv2
import numpy as np
from typing import Tuple, Optional
from skimage import exposure
from scipy import ndimage


class UltrasoundPreprocessor:
    """
    Preprocessing pipeline specifically designed for ultrasound images.
    
    Key considerations:
    1. Speckle noise reduction (characteristic of ultrasound)
    2. Contrast enhancement (variable machine settings)
    3. Edge preservation (critical for landmark detection)
    4. Intensity normalization (standardize across devices)
    """
    
    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        denoise: bool = True,
        denoise_strength: float = 10.0,
        normalize: bool = True,
        normalize_method: str = 'zscore',  # 'zscore', 'minmax', or 'imagenet'
    ):
        """
        Args:
            target_size: Target size (H, W) for resizing
            use_clahe: Apply Contrast Limited Adaptive Histogram Equalization
            clahe_clip_limit: CLAHE clipping limit (higher = more contrast)
            clahe_tile_size: CLAHE tile grid size
            denoise: Apply denoising
            denoise_strength: Strength of denoising filter
            normalize: Apply normalization
            normalize_method: Normalization method
        """
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.denoise = denoise
        self.denoise_strength = denoise_strength
        self.normalize = normalize
        self.normalize_method = normalize_method
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input image [H, W, C] or [H, W]
            
        Returns:
            Preprocessed image [H, W, C]
        """
        # Convert to grayscale if needed for processing
        is_color = len(image.shape) == 3
        if not is_color:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Ensure image is uint8 for CLAHE
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Apply CLAHE for contrast enhancement
        # Rationale: Improves local contrast in ultrasound images
        if self.use_clahe:
            image = self._apply_clahe(image)
        
        # Apply denoising
        # Rationale: Reduces speckle noise while preserving edges
        if self.denoise:
            image = self._apply_denoising(image)
        
        # Resize if needed
        if self.target_size is not None:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # Ensure valid range [0, 255] after preprocessing
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Normalize
        # Rationale: Standardizes intensity distribution for neural networks
        if self.normalize:
            image = self._normalize(image)
        
        return image
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        
        CLAHE vs Global Histogram Equalization:
        - CLAHE: Preserves local details, better for medical images
        - Global: Can lose subtle features, overly bright regions
        
        Applied per channel to preserve color information.
        """
        # Create CLAHE object on-demand to avoid pickling issues
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
        
        if len(image.shape) == 3:
            # Apply CLAHE to each channel
            channels = cv2.split(image)
            clahe_channels = [clahe.apply(ch) for ch in channels]
            return cv2.merge(clahe_channels)
        else:
            return clahe.apply(image)
    
    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Non-local Means denoising.
        
        Rationale for NLM:
        - Better preserves edges than Gaussian blur
        - Effective for speckle noise in ultrasound
        - Non-local approach considers image structure
        
        Alternative: Bilateral filter (faster but less effective)
        """
        if len(image.shape) == 3:
            # Use color denoising
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=self.denoise_strength,
                hColor=self.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image,
                None,
                h=self.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
        return denoised
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensities.
        
        Methods:
        - zscore: Zero mean, unit variance (good for varied distributions)
        - minmax: Scale to [0, 1] (preserves original distribution shape)
        - imagenet: Use ImageNet statistics (for transfer learning)
        """
        image = image.astype(np.float32)
        
        if self.normalize_method == 'zscore':
            # Z-score normalization
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            std = np.std(image, axis=(0, 1), keepdims=True) + 1e-6
            image = (image - mean) / std
            
        elif self.normalize_method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = np.min(image, axis=(0, 1), keepdims=True)
            max_val = np.max(image, axis=(0, 1), keepdims=True)
            image = (image - min_val) / (max_val - min_val + 1e-6)
            
        elif self.normalize_method == 'imagenet':
            # ImageNet normalization (for transfer learning)
            # First scale to [0, 1]
            image = image / 255.0
            # Then apply ImageNet statistics
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
        
        return image


class AdaptivePreprocessor(UltrasoundPreprocessor):
    """
    Adaptive preprocessing that adjusts parameters based on image quality.
    
    Hypothesis: Different images require different preprocessing strength
    - Low contrast images: Stronger CLAHE
    - Noisy images: Stronger denoising
    - High quality images: Minimal processing
    """
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptively preprocess based on image quality metrics.
        """
        # Calculate image quality metrics
        contrast = self._estimate_contrast(image)
        noise_level = self._estimate_noise(image)
        
        # Adapt CLAHE based on contrast
        if contrast < 30:  # Low contrast
            self.clahe_clip_limit = 3.0
        elif contrast > 70:  # High contrast
            self.clahe_clip_limit = 1.5
        else:
            self.clahe_clip_limit = 2.0
        
        # Reinitialize CLAHE with new parameters
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size
            )
        
        # Adapt denoising based on noise level
        if noise_level > 20:  # High noise
            self.denoise_strength = 15.0
        elif noise_level < 5:  # Low noise
            self.denoise_strength = 5.0
        else:
            self.denoise_strength = 10.0
        
        # Apply standard preprocessing with adapted parameters
        return super().preprocess(image)
    
    def _estimate_contrast(self, image: np.ndarray) -> float:
        """
        Estimate image contrast using standard deviation.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        return np.std(gray)
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """
        Estimate noise level using Laplacian variance.
        Lower variance indicates higher noise (in ultrasound context).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Use Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Inverse relationship: low variance = high noise in ultrasound
        noise_estimate = 100 / (1 + variance / 100)
        return noise_estimate


def apply_test_time_augmentation(
    image: np.ndarray,
    preprocessor: UltrasoundPreprocessor,
    num_augmentations: int = 4
) -> list:
    """
    Apply multiple preprocessing variants for test-time augmentation.
    
    Rationale:
    - Ensemble predictions improve robustness
    - Different preprocessing can reveal different features
    - Averaging reduces prediction variance
    
    Returns:
        List of preprocessed image variants
    """
    variants = []
    
    # Original preprocessing
    variants.append(preprocessor.preprocess(image))
    
    # Variant 1: Stronger CLAHE
    if num_augmentations >= 2:
        strong_clahe = UltrasoundPreprocessor(
            target_size=preprocessor.target_size,
            use_clahe=True,
            clahe_clip_limit=3.0,
            denoise=preprocessor.denoise,
            normalize=preprocessor.normalize,
        )
        variants.append(strong_clahe.preprocess(image))
    
    # Variant 2: No CLAHE
    if num_augmentations >= 3:
        no_clahe = UltrasoundPreprocessor(
            target_size=preprocessor.target_size,
            use_clahe=False,
            denoise=preprocessor.denoise,
            normalize=preprocessor.normalize,
        )
        variants.append(no_clahe.preprocess(image))
    
    # Variant 3: Different normalization
    if num_augmentations >= 4:
        alt_norm = UltrasoundPreprocessor(
            target_size=preprocessor.target_size,
            use_clahe=preprocessor.use_clahe,
            denoise=preprocessor.denoise,
            normalize=True,
            normalize_method='minmax',
        )
        variants.append(alt_norm.preprocess(image))
    
    return variants

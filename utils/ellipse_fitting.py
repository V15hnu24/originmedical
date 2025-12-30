"""
Ellipse Fitting and Landmark Extraction from Segmentation Masks

This module converts segmentation masks to landmark coordinates using:
1. Ellipse fitting (RANSAC or direct)
2. Geometric analysis to extract BPD and OFD points

Rationale:
- Segmentation provides robust cranium boundary
- Ellipse is clinically accurate model for fetal head
- Geometric extraction ensures perpendicular BPD/OFD
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from scipy import optimize
from sklearn.linear_model import RANSACRegressor


class EllipseFitter:
    """
    Fit ellipse to cranium segmentation mask and extract landmarks.
    
    Methods:
    1. OpenCV ellipse fitting (fast, direct fit)
    2. RANSAC ellipse fitting (robust to outliers)
    3. Least squares ellipse fitting (accurate for clean masks)
    """
    
    def __init__(
        self,
        method: str = 'opencv',  # 'opencv', 'ransac', or 'least_squares'
        min_area: int = 5000,
        max_area: int = 150000,
        ransac_iterations: int = 1000,
        ransac_threshold: float = 2.0,
    ):
        """
        Args:
            method: Ellipse fitting method
            min_area: Minimum valid cranium area (pixels)
            max_area: Maximum valid cranium area (pixels)
            ransac_iterations: Number of RANSAC iterations
            ransac_threshold: RANSAC inlier threshold
        """
        self.method = method
        self.min_area = min_area
        self.max_area = max_area
        self.ransac_iterations = ransac_iterations
        self.ransac_threshold = ransac_threshold
    
    def mask_to_landmarks(
        self,
        mask: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Convert segmentation mask to landmark coordinates.
        
        Args:
            mask: Binary or probability mask [H, W]
            confidence_threshold: Threshold for binarization
            
        Returns:
            Dictionary with:
                - 'landmarks': [4, 2] array of (x, y) coordinates
                - 'ellipse_params': Ellipse parameters (center, axes, angle)
                - 'confidence': Fitting confidence score
        """
        # Binarize mask
        binary_mask = (mask > confidence_threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # Get largest contour (assume it's the cranium)
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        # Validate area
        if area < self.min_area or area > self.max_area:
            return None
        
        # Fit ellipse based on method
        if self.method == 'opencv':
            ellipse_params = self._fit_ellipse_opencv(contour)
        elif self.method == 'ransac':
            ellipse_params = self._fit_ellipse_ransac(contour)
        elif self.method == 'least_squares':
            ellipse_params = self._fit_ellipse_least_squares(contour)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if ellipse_params is None:
            return None
        
        # Extract landmarks from ellipse
        landmarks = self._ellipse_to_landmarks(ellipse_params)
        
        # Compute confidence score
        confidence = self._compute_fitting_confidence(contour, ellipse_params)
        
        return {
            'landmarks': landmarks,
            'ellipse_params': ellipse_params,
            'confidence': confidence,
            'area': area,
        }
    
    def _fit_ellipse_opencv(self, contour: np.ndarray) -> Optional[Tuple]:
        """
        Fit ellipse using OpenCV's fitEllipse.
        
        Fast and direct, works well for clean contours.
        """
        if len(contour) < 5:  # Need at least 5 points for ellipse
            return None
        
        try:
            ellipse = cv2.fitEllipse(contour)
            # ellipse = ((cx, cy), (width, height), angle)
            return ellipse
        except:
            return None
    
    def _fit_ellipse_ransac(self, contour: np.ndarray) -> Optional[Tuple]:
        """
        Fit ellipse using RANSAC for robustness to outliers.
        
        More robust but slower. Good for noisy segmentations.
        """
        points = contour.reshape(-1, 2).astype(np.float32)
        
        if len(points) < 5:
            return None
        
        # Use direct method as base estimator
        base_params = self._fit_ellipse_opencv(contour)
        if base_params is None:
            return None
        
        # RANSAC refinement
        # Convert points to ellipse canonical form for fitting
        # This is a simplified RANSAC approach
        
        best_inliers = 0
        best_params = base_params
        
        for _ in range(self.ransac_iterations // 10):  # Reduced iterations
            # Sample random subset
            if len(points) < 10:
                sample_points = points
            else:
                indices = np.random.choice(len(points), size=min(len(points), 50), replace=False)
                sample_points = points[indices]
            
            # Fit ellipse to sample
            if len(sample_points) < 5:
                continue
            
            try:
                sample_contour = sample_points.reshape(-1, 1, 2).astype(np.int32)
                sample_params = cv2.fitEllipse(sample_contour)
                
                # Count inliers
                distances = self._point_to_ellipse_distance(points, sample_params)
                inliers = np.sum(distances < self.ransac_threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_params = sample_params
            except:
                continue
        
        return best_params
    
    def _fit_ellipse_least_squares(self, contour: np.ndarray) -> Optional[Tuple]:
        """
        Fit ellipse using least squares optimization.
        
        Most accurate for clean data but sensitive to outliers.
        """
        points = contour.reshape(-1, 2).astype(np.float32)
        
        if len(points) < 5:
            return None
        
        # Use OpenCV fit as initial guess
        initial_params = self._fit_ellipse_opencv(contour)
        if initial_params is None:
            return None
        
        # Convert to parameter vector for optimization
        (cx, cy), (width, height), angle = initial_params
        x0 = np.array([cx, cy, width/2, height/2, np.radians(angle)])
        
        # Optimize
        try:
            result = optimize.minimize(
                lambda params: self._ellipse_residual(params, points),
                x0,
                method='L-BFGS-B',
            )
            
            # Convert back to OpenCV format
            cx, cy, a, b, angle_rad = result.x
            return ((cx, cy), (a*2, b*2), np.degrees(angle_rad))
        except:
            return initial_params
    
    def _ellipse_to_landmarks(self, ellipse_params: Tuple) -> np.ndarray:
        """
        Extract BPD and OFD landmark points from ellipse parameters.
        
        Landmarks:
        - OFD1, OFD2: Points along major axis
        - BPD1, BPD2: Points along minor axis
        
        Medical constraint: BPD ⊥ OFD
        
        Args:
            ellipse_params: ((cx, cy), (width, height), angle)
            
        Returns:
            landmarks: [4, 2] array of (x, y) coordinates
        """
        (cx, cy), (width, height), angle = ellipse_params
        
        # Semi-axes
        a = width / 2   # Semi-major (typically OFD)
        b = height / 2  # Semi-minor (typically BPD)
        
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Determine which is longer (OFD should be longer)
        if width >= height:
            # Major axis is OFD
            ofd_angle = angle_rad
            bpd_angle = angle_rad + np.pi / 2
            ofd_length = a
            bpd_length = b
        else:
            # Major axis is BPD (swap)
            bpd_angle = angle_rad
            ofd_angle = angle_rad + np.pi / 2
            bpd_length = a
            ofd_length = b
        
        # Calculate landmark positions
        ofd1 = np.array([
            cx + ofd_length * np.cos(ofd_angle),
            cy + ofd_length * np.sin(ofd_angle)
        ])
        
        ofd2 = np.array([
            cx - ofd_length * np.cos(ofd_angle),
            cy - ofd_length * np.sin(ofd_angle)
        ])
        
        bpd1 = np.array([
            cx + bpd_length * np.cos(bpd_angle),
            cy + bpd_length * np.sin(bpd_angle)
        ])
        
        bpd2 = np.array([
            cx - bpd_length * np.cos(bpd_angle),
            cy - bpd_length * np.sin(bpd_angle)
        ])
        
        # Stack landmarks: [OFD1, OFD2, BPD1, BPD2]
        landmarks = np.stack([ofd1, ofd2, bpd1, bpd2], axis=0)
        
        return landmarks
    
    def _point_to_ellipse_distance(
        self,
        points: np.ndarray,
        ellipse_params: Tuple
    ) -> np.ndarray:
        """
        Compute distance from points to ellipse.
        
        Used for RANSAC inlier counting and confidence estimation.
        """
        (cx, cy), (width, height), angle = ellipse_params
        
        # Transform points to ellipse coordinate system
        angle_rad = np.radians(angle)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        
        # Translate
        px = points[:, 0] - cx
        py = points[:, 1] - cy
        
        # Rotate
        px_rot = px * cos_a - py * sin_a
        py_rot = px * sin_a + py * cos_a
        
        # Compute distance to ellipse
        a = width / 2
        b = height / 2
        
        # Normalized coordinates
        x_norm = px_rot / a
        y_norm = py_rot / b
        
        # Distance (approximate)
        distances = np.abs(np.sqrt(x_norm**2 + y_norm**2) - 1) * min(a, b)
        
        return distances
    
    def _ellipse_residual(self, params: np.ndarray, points: np.ndarray) -> float:
        """Residual function for least squares ellipse fitting."""
        cx, cy, a, b, angle_rad = params
        
        # Transform points
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        
        px = points[:, 0] - cx
        py = points[:, 1] - cy
        
        px_rot = px * cos_a - py * sin_a
        py_rot = px * sin_a + py * cos_a
        
        # Ellipse equation residual
        residual = (px_rot / a)**2 + (py_rot / b)**2 - 1
        
        return np.sum(residual**2)
    
    def _compute_fitting_confidence(
        self,
        contour: np.ndarray,
        ellipse_params: Tuple
    ) -> float:
        """
        Compute confidence score for ellipse fitting.
        
        Higher confidence = better fit
        
        Factors:
        - Mean distance of contour points to ellipse (lower = better)
        - Percentage of inliers (higher = better)
        - Ellipse aspect ratio (should be reasonable for cranium)
        """
        points = contour.reshape(-1, 2).astype(np.float32)
        
        # Distance to ellipse
        distances = self._point_to_ellipse_distance(points, ellipse_params)
        mean_distance = np.mean(distances)
        inlier_ratio = np.sum(distances < 3.0) / len(distances)
        
        # Aspect ratio check (cranium typically 1.1 to 1.4)
        (cx, cy), (width, height), angle = ellipse_params
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        aspect_score = 1.0 - min(abs(aspect_ratio - 1.2) / 0.5, 1.0)
        
        # Combined confidence
        confidence = (
            0.4 * (1.0 / (1.0 + mean_distance / 10)) +  # Distance term
            0.4 * inlier_ratio +                         # Inlier term
            0.2 * aspect_score                           # Aspect ratio term
        )
        
        return float(np.clip(confidence, 0, 1))


def post_process_mask(mask: np.ndarray) -> np.ndarray:
    """
    Post-process segmentation mask to improve ellipse fitting.
    
    Operations:
    1. Remove small connected components (noise)
    2. Fill holes
    3. Morphological smoothing
    
    Args:
        mask: Binary mask [H, W]
        
    Returns:
        Cleaned mask [H, W]
    """
    # Convert to uint8
    mask = (mask * 255).astype(np.uint8)
    
    # Morphological opening (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Fill largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [largest_contour], -1, 255, -1)
        mask = mask_filled
    
    # Morphological closing (smooth boundary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return (mask / 255.0).astype(np.float32)

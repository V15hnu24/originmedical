"""
Evaluation Metrics for Landmark Detection and Segmentation

Key Metrics:
1. Mean Radial Error (MRE): Average distance between predicted and ground truth landmarks
2. Successful Detection Rate (SDR): Percentage of landmarks within threshold
3. Dice Coefficient: Overlap metric for segmentation
4. Clinical Metrics: BPD/OFD measurement errors
"""
import numpy as np
import torch
from typing import Dict, List, Tuple
from scipy.spatial.distance import euclidean


class LandmarkMetrics:
    """
    Metrics for landmark detection evaluation.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (512, 512)):
        """
        Args:
            image_size: Reference image size for normalization
        """
        self.image_size = image_size
    
    def mean_radial_error(
        self,
        pred_landmarks: np.ndarray,
        gt_landmarks: np.ndarray,
        pixel_spacing: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute Mean Radial Error (MRE).
        
        MRE is the average Euclidean distance between predicted and ground truth landmarks.
        
        Args:
            pred_landmarks: Predicted landmarks [N, 4, 2] or [4, 2]
            gt_landmarks: Ground truth landmarks [N, 4, 2] or [4, 2]
            pixel_spacing: mm per pixel for real-world measurements
            
        Returns:
            Dictionary with per-landmark and overall MRE
        """
        if pred_landmarks.ndim == 2:
            pred_landmarks = pred_landmarks[np.newaxis, :]
            gt_landmarks = gt_landmarks[np.newaxis, :]
        
        # Compute Euclidean distances
        distances = np.linalg.norm(pred_landmarks - gt_landmarks, axis=2)  # [N, 4]
        
        # Mean per landmark
        mre_per_landmark = np.mean(distances, axis=0)  # [4]
        
        # Overall MRE
        mre_overall = np.mean(distances)
        
        # Convert to mm if pixel spacing provided
        results = {
            'mre_overall_px': float(mre_overall),
            'mre_overall_mm': float(mre_overall * pixel_spacing),
            'mre_ofd1_px': float(mre_per_landmark[0]),
            'mre_ofd2_px': float(mre_per_landmark[1]),
            'mre_bpd1_px': float(mre_per_landmark[2]),
            'mre_bpd2_px': float(mre_per_landmark[3]),
        }
        
        return results
    
    def successful_detection_rate(
        self,
        pred_landmarks: np.ndarray,
        gt_landmarks: np.ndarray,
        thresholds: List[float] = [2.0, 2.5, 3.0, 4.0],
        pixel_spacing: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute Successful Detection Rate (SDR).
        
        SDR_t = percentage of landmarks with error < threshold t
        
        Args:
            pred_landmarks: Predicted landmarks [N, 4, 2] or [4, 2]
            gt_landmarks: Ground truth landmarks [N, 4, 2] or [4, 2]
            thresholds: List of distance thresholds (mm)
            pixel_spacing: mm per pixel
            
        Returns:
            Dictionary with SDR at each threshold
        """
        if pred_landmarks.ndim == 2:
            pred_landmarks = pred_landmarks[np.newaxis, :]
            gt_landmarks = gt_landmarks[np.newaxis, :]
        
        # Compute distances in mm
        distances = np.linalg.norm(pred_landmarks - gt_landmarks, axis=2)  # [N, 4]
        distances_mm = distances * pixel_spacing
        
        results = {}
        for threshold in thresholds:
            sdr = np.mean(distances_mm < threshold) * 100
            results[f'sdr_{threshold}mm'] = float(sdr)
        
        return results
    
    def clinical_measurements(
        self,
        landmarks: np.ndarray,
        pixel_spacing: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute clinical measurements from landmarks.
        
        Measurements:
        - BPD: Biparietal Diameter (distance between BPD1 and BPD2)
        - OFD: Occipitofrontal Diameter (distance between OFD1 and OFD2)
        - HC: Head Circumference (from ellipse fit)
        
        Args:
            landmarks: Landmarks [4, 2] or [N, 4, 2]
            pixel_spacing: mm per pixel
            
        Returns:
            Dictionary with measurements in mm
        """
        if landmarks.ndim == 2:
            landmarks = landmarks[np.newaxis, :]
        
        # Extract landmarks
        ofd1 = landmarks[:, 0]
        ofd2 = landmarks[:, 1]
        bpd1 = landmarks[:, 2]
        bpd2 = landmarks[:, 3]
        
        # Compute diameters
        ofd = np.linalg.norm(ofd2 - ofd1, axis=1)  # [N]
        bpd = np.linalg.norm(bpd2 - bpd1, axis=1)  # [N]
        
        # Head circumference (approximate from ellipse)
        # HC ≈ π * √(2(a² + b²)) where a=OFD/2, b=BPD/2
        # Ramanujan approximation: π(a+b)[1 + 3h/(10 + √(4-3h))] where h = ((a-b)/(a+b))²
        a = ofd / 2
        b = bpd / 2
        h = ((a - b) / (a + b + 1e-6)) ** 2
        hc = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
        
        # Convert to mm
        results = {
            'bpd_mm': float(np.mean(bpd) * pixel_spacing),
            'ofd_mm': float(np.mean(ofd) * pixel_spacing),
            'hc_mm': float(np.mean(hc) * pixel_spacing),
        }
        
        return results
    
    def measurement_error(
        self,
        pred_landmarks: np.ndarray,
        gt_landmarks: np.ndarray,
        pixel_spacing: float = 1.0,
    ) -> Dict[str, float]:
        """
        Compute errors in clinical measurements.
        
        More clinically relevant than point-wise errors.
        
        Args:
            pred_landmarks: Predicted landmarks [N, 4, 2] or [4, 2]
            gt_landmarks: Ground truth landmarks [N, 4, 2] or [4, 2]
            pixel_spacing: mm per pixel
            
        Returns:
            Dictionary with measurement errors
        """
        pred_measurements = self.clinical_measurements(pred_landmarks, pixel_spacing)
        gt_measurements = self.clinical_measurements(gt_landmarks, pixel_spacing)
        
        errors = {
            'bpd_error_mm': abs(pred_measurements['bpd_mm'] - gt_measurements['bpd_mm']),
            'ofd_error_mm': abs(pred_measurements['ofd_mm'] - gt_measurements['ofd_mm']),
            'hc_error_mm': abs(pred_measurements['hc_mm'] - gt_measurements['hc_mm']),
            'bpd_error_pct': abs(pred_measurements['bpd_mm'] - gt_measurements['bpd_mm']) / gt_measurements['bpd_mm'] * 100,
            'ofd_error_pct': abs(pred_measurements['ofd_mm'] - gt_measurements['ofd_mm']) / gt_measurements['ofd_mm'] * 100,
        }
        
        return errors


class SegmentationMetrics:
    """
    Metrics for segmentation evaluation.
    """
    
    @staticmethod
    def dice_coefficient(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        smooth: float = 1.0,
    ) -> float:
        """
        Compute Dice coefficient (F1 score for segmentation).
        
        Dice = 2 * |X ∩ Y| / (|X| + |Y|)
        
        Args:
            pred_mask: Predicted mask [H, W]
            gt_mask: Ground truth mask [H, W]
            smooth: Smoothing factor
            
        Returns:
            Dice coefficient [0, 1]
        """
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        intersection = np.sum(pred_flat * gt_flat)
        union = np.sum(pred_flat) + np.sum(gt_flat)
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        
        return float(dice)
    
    @staticmethod
    def iou(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        smooth: float = 1.0,
    ) -> float:
        """
        Compute Intersection over Union (IoU / Jaccard index).
        
        IoU = |X ∩ Y| / |X ∪ Y|
        
        Args:
            pred_mask: Predicted mask [H, W]
            gt_mask: Ground truth mask [H, W]
            smooth: Smoothing factor
            
        Returns:
            IoU [0, 1]
        """
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()
        
        intersection = np.sum(pred_flat * gt_flat)
        union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        return float(iou)
    
    @staticmethod
    def hausdorff_distance(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        percentile: int = 95,
    ) -> float:
        """
        Compute Hausdorff distance between boundaries.
        
        Measures maximum boundary error. Uses 95th percentile to reduce
        sensitivity to outliers.
        
        Args:
            pred_mask: Predicted mask [H, W]
            gt_mask: Ground truth mask [H, W]
            percentile: Percentile for robust Hausdorff distance
            
        Returns:
            Hausdorff distance in pixels
        """
        from scipy.ndimage import distance_transform_edt
        
        # Get boundaries
        pred_boundary = pred_mask - cv2.erode(pred_mask.astype(np.uint8), np.ones((3, 3)))
        gt_boundary = gt_mask - cv2.erode(gt_mask.astype(np.uint8), np.ones((3, 3)))
        
        # Distance transforms
        pred_dt = distance_transform_edt(1 - pred_boundary)
        gt_dt = distance_transform_edt(1 - gt_boundary)
        
        # Distances from pred to gt and vice versa
        pred_to_gt = pred_dt[gt_boundary > 0]
        gt_to_pred = gt_dt[pred_boundary > 0]
        
        if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
            return float('inf')
        
        # Hausdorff distance (use percentile for robustness)
        hd = max(
            np.percentile(pred_to_gt, percentile),
            np.percentile(gt_to_pred, percentile)
        )
        
        return float(hd)
    
    @staticmethod
    def average_surface_distance(
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> float:
        """
        Compute average symmetric surface distance.
        
        Measures average boundary error.
        
        Args:
            pred_mask: Predicted mask [H, W]
            gt_mask: Ground truth mask [H, W]
            
        Returns:
            Average surface distance in pixels
        """
        from scipy.ndimage import distance_transform_edt
        
        # Get boundaries
        pred_boundary = pred_mask - cv2.erode(pred_mask.astype(np.uint8), np.ones((3, 3)))
        gt_boundary = gt_mask - cv2.erode(gt_mask.astype(np.uint8), np.ones((3, 3)))
        
        # Distance transforms
        pred_dt = distance_transform_edt(1 - pred_boundary)
        gt_dt = distance_transform_edt(1 - gt_boundary)
        
        # Distances
        pred_to_gt = pred_dt[gt_boundary > 0]
        gt_to_pred = gt_dt[pred_boundary > 0]
        
        if len(pred_to_gt) == 0 or len(gt_to_pred) == 0:
            return float('inf')
        
        # Average
        asd = (np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2
        
        return float(asd)


import cv2


class CombinedMetrics:
    """
    Combined metrics for evaluating the full pipeline:
    Segmentation → Ellipse Fitting → Landmark Extraction
    """
    
    def __init__(self):
        self.landmark_metrics = LandmarkMetrics()
        self.segmentation_metrics = SegmentationMetrics()
    
    def evaluate_pipeline(
        self,
        pred_mask: np.ndarray,
        pred_landmarks: np.ndarray,
        gt_mask: np.ndarray,
        gt_landmarks: np.ndarray,
        pixel_spacing: float = 1.0,
    ) -> Dict[str, float]:
        """
        Evaluate complete pipeline.
        
        Args:
            pred_mask: Predicted segmentation [H, W]
            pred_landmarks: Predicted landmarks [4, 2]
            gt_mask: Ground truth segmentation [H, W]
            gt_landmarks: Ground truth landmarks [4, 2]
            pixel_spacing: mm per pixel
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Segmentation metrics
        results['dice'] = self.segmentation_metrics.dice_coefficient(pred_mask, gt_mask)
        results['iou'] = self.segmentation_metrics.iou(pred_mask, gt_mask)
        
        # Landmark metrics
        mre = self.landmark_metrics.mean_radial_error(pred_landmarks, gt_landmarks, pixel_spacing)
        results.update(mre)
        
        sdr = self.landmark_metrics.successful_detection_rate(pred_landmarks, gt_landmarks, pixel_spacing=pixel_spacing)
        results.update(sdr)
        
        # Clinical measurement errors
        measurement_errors = self.landmark_metrics.measurement_error(pred_landmarks, gt_landmarks, pixel_spacing)
        results.update(measurement_errors)
        
        return results


def compute_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    metric_type: str = 'landmark',
    pixel_spacing: float = 1.0,
) -> Dict[str, float]:
    """
    Compute metrics for a batch of predictions.
    
    Args:
        pred: Predictions [B, ...] (landmarks or masks)
        target: Ground truth [B, ...]
        metric_type: 'landmark' or 'segmentation'
        pixel_spacing: mm per pixel (for real-world measurements)
        
    Returns:
        Dictionary with averaged metrics
    """
    batch_size = pred.size(0)
    
    if metric_type == 'landmark':
        metrics_calculator = LandmarkMetrics()
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        all_mre = []
        all_sdr = []
        for i in range(batch_size):
            # Mean Radial Error
            mre = metrics_calculator.mean_radial_error(pred_np[i], target_np[i], pixel_spacing)
            all_mre.append(mre)
            
            # Successful Detection Rate (accuracy at different thresholds)
            sdr = metrics_calculator.successful_detection_rate(
                pred_np[i], target_np[i],
                thresholds=[2.0, 2.5, 3.0, 4.0, 5.0],  # mm thresholds
                pixel_spacing=pixel_spacing
            )
            all_sdr.append(sdr)
        
        # Average metrics
        avg_metrics = {}
        for key in all_mre[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_mre])
        for key in all_sdr[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_sdr])
        
        return avg_metrics
    
    elif metric_type == 'segmentation':
        metrics_calculator = SegmentationMetrics()
        pred_np = (pred.cpu().numpy() > 0.5).astype(np.float32)
        target_np = target.cpu().numpy()
        
        dice_scores = []
        iou_scores = []
        
        for i in range(batch_size):
            dice = metrics_calculator.dice_coefficient(pred_np[i, 0], target_np[i, 0])
            iou = metrics_calculator.iou(pred_np[i, 0], target_np[i, 0])
            dice_scores.append(dice)
            iou_scores.append(iou)
        
        return {
            'dice': np.mean(dice_scores),
            'iou': np.mean(iou_scores),
        }
    
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

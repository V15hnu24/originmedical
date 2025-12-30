"""
Visualization utilities for fetal ultrasound landmark detection and segmentation.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as MPLEllipse
import seaborn as sns
from typing import Optional, Tuple, Dict, List


class LandmarkVisualizer:
    """
    Visualize landmark detection results.
    """
    
    def __init__(
        self,
        landmark_colors: Dict[str, Tuple[int, int, int]] = None,
        line_thickness: int = 2,
        point_radius: int = 5,
    ):
        """
        Args:
            landmark_colors: Colors for different landmarks (BGR format)
            line_thickness: Thickness for lines
            point_radius: Radius for landmark points
        """
        self.landmark_colors = landmark_colors or {
            'ofd': (0, 0, 255),    # Red for OFD
            'bpd': (255, 0, 0),    # Blue for BPD
            'pred': (0, 255, 0),   # Green for predictions
            'gt': (255, 255, 0),   # Cyan for ground truth
        }
        self.line_thickness = line_thickness
        self.point_radius = point_radius
    
    def draw_landmarks(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        color_mode: str = 'separate',  # 'separate' or 'unified'
        label: str = 'pred',
    ) -> np.ndarray:
        """
        Draw landmarks on image.
        
        Args:
            image: Input image [H, W, 3] (RGB)
            landmarks: Landmark coordinates [4, 2] (x, y)
            color_mode: 'separate' for different colors per biometry, 'unified' for same color
            label: 'pred' or 'gt'
            
        Returns:
            Image with landmarks drawn
        """
        img_draw = image.copy()
        
        # Convert to uint8 if needed
        if img_draw.dtype != np.uint8:
            img_draw = (img_draw * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        ofd1, ofd2, bpd1, bpd2 = landmarks
        
        # Choose colors
        if color_mode == 'separate':
            ofd_color = self.landmark_colors['ofd']
            bpd_color = self.landmark_colors['bpd']
        else:
            ofd_color = bpd_color = self.landmark_colors[label]
        
        # Draw OFD line
        cv2.line(
            img_draw,
            tuple(ofd1.astype(int)),
            tuple(ofd2.astype(int)),
            ofd_color,
            self.line_thickness
        )
        
        # Draw BPD line
        cv2.line(
            img_draw,
            tuple(bpd1.astype(int)),
            tuple(bpd2.astype(int)),
            bpd_color,
            self.line_thickness
        )
        
        # Draw landmark points
        for point, color in [(ofd1, ofd_color), (ofd2, ofd_color), 
                              (bpd1, bpd_color), (bpd2, bpd_color)]:
            cv2.circle(
                img_draw,
                tuple(point.astype(int)),
                self.point_radius,
                color,
                -1
            )
            # White border for visibility
            cv2.circle(
                img_draw,
                tuple(point.astype(int)),
                self.point_radius + 1,
                (255, 255, 255),
                1
            )
        
        # Convert back to RGB
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        
        return img_draw
    
    def compare_predictions(
        self,
        image: np.ndarray,
        pred_landmarks: np.ndarray,
        gt_landmarks: np.ndarray,
        show_error: bool = True,
    ) -> np.ndarray:
        """
        Compare predicted and ground truth landmarks.
        
        Args:
            image: Input image [H, W, 3]
            pred_landmarks: Predicted landmarks [4, 2]
            gt_landmarks: Ground truth landmarks [4, 2]
            show_error: Draw error vectors
            
        Returns:
            Comparison visualization
        """
        img_draw = image.copy()
        if img_draw.dtype != np.uint8:
            img_draw = (img_draw * 255).astype(np.uint8)
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
        
        # Draw ground truth (cyan)
        for i in range(4):
            cv2.circle(
                img_draw,
                tuple(gt_landmarks[i].astype(int)),
                self.point_radius,
                self.landmark_colors['gt'],
                -1
            )
        
        # Draw lines for GT
        cv2.line(img_draw, tuple(gt_landmarks[0].astype(int)), 
                 tuple(gt_landmarks[1].astype(int)), self.landmark_colors['gt'], 2)
        cv2.line(img_draw, tuple(gt_landmarks[2].astype(int)), 
                 tuple(gt_landmarks[3].astype(int)), self.landmark_colors['gt'], 2)
        
        # Draw predictions (green)
        for i in range(4):
            cv2.circle(
                img_draw,
                tuple(pred_landmarks[i].astype(int)),
                self.point_radius - 1,
                self.landmark_colors['pred'],
                -1
            )
        
        # Draw lines for predictions
        cv2.line(img_draw, tuple(pred_landmarks[0].astype(int)), 
                 tuple(pred_landmarks[1].astype(int)), self.landmark_colors['pred'], 2)
        cv2.line(img_draw, tuple(pred_landmarks[2].astype(int)), 
                 tuple(pred_landmarks[3].astype(int)), self.landmark_colors['pred'], 2)
        
        # Draw error vectors
        if show_error:
            for i in range(4):
                cv2.arrowedLine(
                    img_draw,
                    tuple(gt_landmarks[i].astype(int)),
                    tuple(pred_landmarks[i].astype(int)),
                    (255, 255, 255),
                    1,
                    tipLength=0.3
                )
        
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        return img_draw
    
    def visualize_heatmaps(
        self,
        image: np.ndarray,
        heatmaps: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay heatmaps on image.
        
        Args:
            image: Input image [H, W, 3]
            heatmaps: Predicted heatmaps [4, H, W]
            alpha: Transparency for overlay
            
        Returns:
            Image with heatmap overlay
        """
        img_draw = image.copy()
        if img_draw.dtype != np.uint8:
            img_draw = (img_draw * 255).astype(np.uint8)
        
        # Resize heatmaps to image size
        h, w = img_draw.shape[:2]
        heatmaps_resized = []
        for i in range(heatmaps.shape[0]):
            hm = cv2.resize(heatmaps[i], (w, h))
            heatmaps_resized.append(hm)
        
        # Combine heatmaps (max across landmarks)
        combined_heatmap = np.max(heatmaps_resized, axis=0)
        
        # Normalize to [0, 255]
        combined_heatmap = (combined_heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(combined_heatmap, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        blended = cv2.addWeighted(img_draw, 1 - alpha, heatmap_colored, alpha, 0)
        
        return blended


class SegmentationVisualizer:
    """
    Visualize segmentation results.
    """
    
    def __init__(
        self,
        mask_color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5,
    ):
        """
        Args:
            mask_color: Color for mask overlay (RGB)
            alpha: Transparency for overlay
        """
        self.mask_color = mask_color
        self.alpha = alpha
    
    def overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """
        Overlay segmentation mask on image.
        
        Args:
            image: Input image [H, W, 3]
            mask: Binary mask [H, W]
            color: Optional custom color
            
        Returns:
            Image with mask overlay
        """
        img_draw = image.copy()
        if img_draw.dtype != np.uint8:
            img_draw = (img_draw * 255).astype(np.uint8)
        
        # Create colored mask
        color = color or self.mask_color
        mask_colored = np.zeros_like(img_draw)
        mask_colored[mask > 0.5] = color
        
        # Blend
        blended = cv2.addWeighted(img_draw, 1 - self.alpha, mask_colored, self.alpha, 0)
        
        return blended
    
    def draw_contour(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw mask contour on image.
        
        Args:
            image: Input image [H, W, 3]
            mask: Binary mask [H, W]
            color: Contour color (RGB)
            thickness: Line thickness
            
        Returns:
            Image with contour
        """
        img_draw = image.copy()
        if img_draw.dtype != np.uint8:
            img_draw = (img_draw * 255).astype(np.uint8)
        
        # Convert RGB to BGR
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
        
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        color_bgr = (color[2], color[1], color[0])  # RGB to BGR
        cv2.drawContours(img_draw, contours, -1, color_bgr, thickness)
        
        # Convert back to RGB
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        
        return img_draw
    
    def visualize_ellipse(
        self,
        image: np.ndarray,
        ellipse_params: Tuple,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw fitted ellipse on image.
        
        Args:
            image: Input image [H, W, 3]
            ellipse_params: ((cx, cy), (width, height), angle)
            color: Ellipse color (RGB)
            thickness: Line thickness
            
        Returns:
            Image with ellipse
        """
        img_draw = image.copy()
        if img_draw.dtype != np.uint8:
            img_draw = (img_draw * 255).astype(np.uint8)
        
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
        
        # Draw ellipse
        color_bgr = (color[2], color[1], color[0])
        cv2.ellipse(img_draw, ellipse_params, color_bgr, thickness)
        
        img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        
        return img_draw


def create_comparison_grid(
    images: List[np.ndarray],
    titles: List[str],
    rows: int = 2,
    cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """
    Create a grid of comparison images.
    
    Args:
        images: List of images to display
        titles: List of titles for each image
        rows: Number of rows
        cols: Number of columns
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(title, fontsize=12)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[List[float]] = None,
    val_metrics: Optional[List[float]] = None,
    metric_name: str = 'MRE',
    save_path: Optional[str] = None,
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_metrics: Optional training metrics per epoch
        val_metrics: Optional validation metrics per epoch
        metric_name: Name of the metric
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2 if train_metrics else 1, figsize=(15, 5))
    
    if train_metrics is None:
        axes = [axes]
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metric plot (if provided)
    if train_metrics:
        axes[1].plot(train_metrics, label=f'Train {metric_name}', linewidth=2)
        axes[1].plot(val_metrics, label=f'Val {metric_name}', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel(metric_name, fontsize=12)
        axes[1].set_title(f'Training and Validation {metric_name}', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_distribution(
    errors: np.ndarray,
    metric_name: str = 'Radial Error (mm)',
    bins: int = 30,
    save_path: Optional[str] = None,
):
    """
    Plot distribution of errors.
    
    Args:
        errors: Array of errors
        metric_name: Name of the error metric
        bins: Number of histogram bins
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(errors, bins=bins, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(errors), color='red', linestyle='--', 
                     linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    axes[0].axvline(np.median(errors), color='green', linestyle='--', 
                     linewidth=2, label=f'Median: {np.median(errors):.2f}')
    axes[0].set_xlabel(metric_name, fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{metric_name} Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(errors, vert=True)
    axes[1].set_ylabel(metric_name, fontsize=12)
    axes[1].set_title(f'{metric_name} Box Plot', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

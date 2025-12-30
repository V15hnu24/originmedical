"""
Visualization script for Part B: Segmentation-Based Approach

This script visualizes:
1. Original ultrasound images
2. Ground truth segmentation masks (cranium ellipse)
3. Overlay of mask on image
4. Extracted landmarks from ellipse fitting

Usage:
    python utils/visualise_segmentation.py --num_images 10
    python utils/visualise_segmentation.py --num_images 10 --save_dir output/segmentation_vis
"""
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.ellipse_fitting import EllipseFitter


def extract_landmarks_from_mask(mask: np.ndarray, method='ransac'):
    """
    Extract BPD and OFD landmarks from segmentation mask using ellipse fitting.
    
    Args:
        mask: Binary mask [H, W] with 255 for cranium, 0 for background
        method: 'opencv', 'ransac', or 'least_squares'
    
    Returns:
        landmarks: Dictionary with ofd_1, ofd_2, bpd_1, bpd_2 (x, y) tuples
        ellipse_params: (center, axes, angle) for visualization
        confidence: Fitting confidence score
    """
    # Initialize ellipse fitter
    fitter = EllipseFitter(method=method)
    
    # Fit ellipse to mask
    result = fitter.fit_mask_to_landmarks(mask)
    
    if result is None:
        return None, None, 0.0
    
    landmarks, ellipse_params, confidence = result
    
    # Convert to dictionary format
    landmark_dict = {
        'ofd_1_x': landmarks[0, 0],
        'ofd_1_y': landmarks[0, 1],
        'ofd_2_x': landmarks[1, 0],
        'ofd_2_y': landmarks[1, 1],
        'bpd_1_x': landmarks[2, 0],
        'bpd_1_y': landmarks[2, 1],
        'bpd_2_x': landmarks[3, 0],
        'bpd_2_y': landmarks[3, 1],
    }
    
    return landmark_dict, ellipse_params, confidence


def draw_segmentation_overlay(image, mask, landmarks_gt=None, landmarks_pred=None):
    """
    Draw segmentation mask overlay with ground truth and predicted landmarks.
    
    Args:
        image: RGB image [H, W, 3]
        mask: Binary mask [H, W]
        landmarks_gt: Ground truth landmarks (dict)
        landmarks_pred: Predicted landmarks from ellipse fitting (dict)
    
    Returns:
        Visualization image
    """
    # Create overlay
    img_draw = image.copy()
    
    # Resize mask to match image if needed
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
    
    # Create colored overlay (cyan for mask)
    overlay = img_draw.copy()
    overlay[mask > 127] = [0, 255, 255]  # Cyan for cranium
    
    # Blend with original image
    img_draw = cv2.addWeighted(img_draw, 0.7, overlay, 0.3, 0)
    
    # Draw mask contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_draw, contours, -1, (0, 255, 255), 2)
    
    # Draw ground truth landmarks (if provided)
    if landmarks_gt is not None:
        ofd_1_gt = (int(landmarks_gt['ofd_1_x']), int(landmarks_gt['ofd_1_y']))
        ofd_2_gt = (int(landmarks_gt['ofd_2_x']), int(landmarks_gt['ofd_2_y']))
        bpd_1_gt = (int(landmarks_gt['bpd_1_x']), int(landmarks_gt['bpd_1_y']))
        bpd_2_gt = (int(landmarks_gt['bpd_2_x']), int(landmarks_gt['bpd_2_y']))
        
        # Draw lines
        cv2.line(img_draw, ofd_1_gt, ofd_2_gt, (255, 0, 0), 2)  # Red for GT OFD
        cv2.line(img_draw, bpd_1_gt, bpd_2_gt, (0, 0, 255), 2)  # Blue for GT BPD
        
        # Draw points
        for pt in [ofd_1_gt, ofd_2_gt]:
            cv2.circle(img_draw, pt, 5, (255, 0, 0), -1)
            cv2.circle(img_draw, pt, 7, (255, 255, 255), 2)
        for pt in [bpd_1_gt, bpd_2_gt]:
            cv2.circle(img_draw, pt, 5, (0, 0, 255), -1)
            cv2.circle(img_draw, pt, 7, (255, 255, 255), 2)
        
        # Labels
        cv2.putText(img_draw, 'GT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
    
    # Draw predicted landmarks (if provided)
    if landmarks_pred is not None:
        ofd_1_pred = (int(landmarks_pred['ofd_1_x']), int(landmarks_pred['ofd_1_y']))
        ofd_2_pred = (int(landmarks_pred['ofd_2_x']), int(landmarks_pred['ofd_2_y']))
        bpd_1_pred = (int(landmarks_pred['bpd_1_x']), int(landmarks_pred['bpd_1_y']))
        bpd_2_pred = (int(landmarks_pred['bpd_2_x']), int(landmarks_pred['bpd_2_y']))
        
        # Draw lines (dashed style by drawing segments)
        def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10):
            dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            dashes = int(dist / dash_length)
            for i in range(0, dashes, 2):
                start = (
                    int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes),
                    int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)
                )
                end = (
                    int(pt1[0] + (pt2[0] - pt1[0]) * (i + 1) / dashes),
                    int(pt1[1] + (pt2[1] - pt1[1]) * (i + 1) / dashes)
                )
                cv2.line(img, start, end, color, thickness)
        
        draw_dashed_line(img_draw, ofd_1_pred, ofd_2_pred, (255, 100, 100))  # Light red
        draw_dashed_line(img_draw, bpd_1_pred, bpd_2_pred, (100, 100, 255))  # Light blue
        
        # Draw points
        for pt in [ofd_1_pred, ofd_2_pred]:
            cv2.circle(img_draw, pt, 4, (255, 100, 100), -1)
        for pt in [bpd_1_pred, bpd_2_pred]:
            cv2.circle(img_draw, pt, 4, (100, 100, 255), -1)
        
        # Calculate error if GT is available
        if landmarks_gt is not None:
            errors = []
            for key in ['ofd_1_x', 'ofd_1_y', 'ofd_2_x', 'ofd_2_y', 
                       'bpd_1_x', 'bpd_1_y', 'bpd_2_x', 'bpd_2_y']:
                errors.append(abs(landmarks_gt[key] - landmarks_pred[key]))
            mean_error = np.mean(errors)
            
            cv2.putText(img_draw, f'Pred (Error: {mean_error:.1f}px)', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    return img_draw


def visualize_segmentation_dataset(
    csv_path: str,
    images_dir: str,
    masks_dir: str,
    num_images: int = 10,
    save_dir: str = None,
    ellipse_method: str = 'ransac',
):
    """
    Visualize segmentation dataset with masks and extracted landmarks.
    
    Args:
        csv_path: Path to CSV with ground truth landmarks
        images_dir: Directory with ultrasound images
        masks_dir: Directory with segmentation masks
        num_images: Number of images to visualize
        save_dir: Optional directory to save visualizations
        ellipse_method: Method for ellipse fitting ('opencv', 'ransac', 'least_squares')
    """
    # Load annotations
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} annotations from {csv_path}")
    
    # Find images that have both image and mask
    available_images = []
    for idx, row in df.iterrows():
        image_name = row['image_name']
        mask_name = image_name.replace('.png', '_Annotation.png')
        
        image_path = Path(images_dir) / image_name
        mask_path = Path(masks_dir) / mask_name
        
        if image_path.exists() and mask_path.exists():
            available_images.append((idx, image_name, mask_name))
    
    print(f"Found {len(available_images)} images with masks")
    
    # Select random images
    if len(available_images) > num_images:
        np.random.seed(42)
        indices = np.random.choice(len(available_images), size=num_images, replace=False)
        selected = [available_images[i] for i in sorted(indices)]
    else:
        selected = available_images[:num_images]
    
    print(f"\nVisualizing {len(selected)} images with segmentation masks...")
    
    # Create save directory
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics
    all_errors = []
    all_confidences = []
    
    # Process each image
    for vis_idx, (csv_idx, image_name, mask_name) in enumerate(selected):
        row = df.iloc[csv_idx]
        
        # Load image
        image_path = Path(images_dir) / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Failed to load {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = Path(masks_dir) / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Failed to load {mask_path}")
            continue
        
        # Ground truth landmarks
        landmarks_gt = {
            'ofd_1_x': row['ofd_1_x'],
            'ofd_1_y': row['ofd_1_y'],
            'ofd_2_x': row['ofd_2_x'],
            'ofd_2_y': row['ofd_2_y'],
            'bpd_1_x': row['bpd_1_x'],
            'bpd_1_y': row['bpd_1_y'],
            'bpd_2_x': row['bpd_2_x'],
            'bpd_2_y': row['bpd_2_y'],
        }
        
        # Extract landmarks from mask using ellipse fitting
        landmarks_pred, ellipse_params, confidence = extract_landmarks_from_mask(
            mask, method=ellipse_method
        )
        
        if landmarks_pred is None:
            print(f"Warning: Failed to fit ellipse for {image_name}")
            continue
        
        # Calculate error
        errors = []
        for key in landmarks_gt.keys():
            errors.append(abs(landmarks_gt[key] - landmarks_pred[key]))
        mean_error = np.mean(errors)
        all_errors.append(mean_error)
        all_confidences.append(confidence)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original image with GT landmarks
        img_with_gt = image.copy()
        img_with_gt = cv2.cvtColor(img_with_gt, cv2.COLOR_RGB2BGR)
        
        ofd_1_gt = (int(landmarks_gt['ofd_1_x']), int(landmarks_gt['ofd_1_y']))
        ofd_2_gt = (int(landmarks_gt['ofd_2_x']), int(landmarks_gt['ofd_2_y']))
        bpd_1_gt = (int(landmarks_gt['bpd_1_x']), int(landmarks_gt['bpd_1_y']))
        bpd_2_gt = (int(landmarks_gt['bpd_2_x']), int(landmarks_gt['bpd_2_y']))
        
        cv2.line(img_with_gt, ofd_1_gt, ofd_2_gt, (0, 0, 255), 2)
        cv2.line(img_with_gt, bpd_1_gt, bpd_2_gt, (255, 0, 0), 2)
        for pt in [ofd_1_gt, ofd_2_gt, bpd_1_gt, bpd_2_gt]:
            cv2.circle(img_with_gt, pt, 5, (255, 255, 255), -1)
        
        img_with_gt = cv2.cvtColor(img_with_gt, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_with_gt)
        axes[0].set_title('Original + GT Landmarks', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # 3. Overlay with GT and predicted landmarks
        img_overlay = draw_segmentation_overlay(image, mask, landmarks_gt, landmarks_pred)
        axes[2].imshow(img_overlay)
        axes[2].set_title(f'Mask + Landmarks\nError: {mean_error:.2f}px | Conf: {confidence:.3f}', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # Main title
        fig.suptitle(f'{image_name} (Ellipse Method: {ellipse_method})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        if save_dir:
            numeric_prefix = ''.join(c for c in Path(image_name).stem if c.isdigit())
            if numeric_prefix:
                image_number = int(numeric_prefix)
                output_file = save_path / f'{image_number}_segmentation.jpg'
            else:
                output_file = save_path / f'{Path(image_name).stem}_segmentation.jpg'
            
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [{vis_idx+1}/{len(selected)}] {image_name}: Error={mean_error:.2f}px, Conf={confidence:.3f} -> Saved to {output_file}")
        else:
            print(f"  [{vis_idx+1}/{len(selected)}] {image_name}: Error={mean_error:.2f}px, Conf={confidence:.3f}")
            plt.show()
    
    # Print statistics
    if all_errors:
        print("\n" + "="*60)
        print("ELLIPSE FITTING STATISTICS (Mask -> Landmarks)")
        print("="*60)
        print(f"\nMethod: {ellipse_method}")
        print(f"Images processed: {len(all_errors)}")
        print(f"\nLandmark Error (pixels):")
        print(f"  Mean: {np.mean(all_errors):.2f}")
        print(f"  Median: {np.median(all_errors):.2f}")
        print(f"  Std: {np.std(all_errors):.2f}")
        print(f"  Min: {np.min(all_errors):.2f}")
        print(f"  Max: {np.max(all_errors):.2f}")
        print(f"\nFitting Confidence:")
        print(f"  Mean: {np.mean(all_confidences):.3f}")
        print(f"  Median: {np.median(all_confidences):.3f}")
        print(f"  Range: [{np.min(all_confidences):.3f}, {np.max(all_confidences):.3f}]")
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Part B: Segmentation-based approach with masks'
    )
    
    parser.add_argument(
        '--csv_path',
        type=str,
        default='role_challenge_dataset_ground_truth.csv',
        help='Path to CSV with ground truth landmarks'
    )
    
    parser.add_argument(
        '--images_dir',
        type=str,
        default='images',
        help='Directory containing ultrasound images'
    )
    
    parser.add_argument(
        '--masks_dir',
        type=str,
        default='details/masks',
        help='Directory containing segmentation masks'
    )
    
    parser.add_argument(
        '--num_images',
        type=int,
        default=10,
        help='Number of images to visualize'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save visualizations (optional)'
    )
    
    parser.add_argument(
        '--ellipse_method',
        type=str,
        default='ransac',
        choices=['opencv', 'ransac', 'least_squares'],
        help='Method for ellipse fitting'
    )
    
    args = parser.parse_args()
    
    # Check paths
    csv_path = Path(args.csv_path)
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    if not masks_dir.exists():
        print(f"Error: Masks directory not found: {masks_dir}")
        return
    
    print("="*60)
    print("PART B: SEGMENTATION-BASED VISUALIZATION")
    print("="*60)
    print(f"CSV: {csv_path}")
    print(f"Images: {images_dir}")
    print(f"Masks: {masks_dir}")
    print(f"Ellipse method: {args.ellipse_method}")
    print("="*60 + "\n")
    
    visualize_segmentation_dataset(
        str(csv_path),
        str(images_dir),
        str(masks_dir),
        num_images=args.num_images,
        save_dir=args.save_dir,
        ellipse_method=args.ellipse_method,
    )


if __name__ == '__main__':
    main()

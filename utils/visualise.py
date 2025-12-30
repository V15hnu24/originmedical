"""
Visualization script to display ground truth landmarks on ultrasound images.

Usage:
    python utils/visualise.py --num_images 10
    python utils/visualise.py --num_images 20 --save_dir results/gt_visualizations
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


def draw_landmarks_on_image(image, landmarks, image_name=""):
    """
    Draw ground truth landmarks on ultrasound image.
    
    Args:
        image: Input image [H, W, 3] (RGB)
        landmarks: Dictionary with landmark coordinates
        image_name: Name of the image for title
    
    Returns:
        Image with landmarks drawn
    """
    # Make a copy
    img_draw = image.copy()
    
    # Convert to BGR for OpenCV
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    
    # Extract landmarks
    ofd_1 = (int(landmarks['ofd_1_x']), int(landmarks['ofd_1_y']))
    ofd_2 = (int(landmarks['ofd_2_x']), int(landmarks['ofd_2_y']))
    bpd_1 = (int(landmarks['bpd_1_x']), int(landmarks['bpd_1_y']))
    bpd_2 = (int(landmarks['bpd_2_x']), int(landmarks['bpd_2_y']))
    
    # Colors (BGR format)
    ofd_color = (0, 0, 255)    # Red for OFD
    bpd_color = (255, 0, 0)    # Blue for BPD
    
    # Line thickness and point radius
    line_thickness = 3
    point_radius = 7
    
    # Draw OFD line
    cv2.line(img_draw, ofd_1, ofd_2, ofd_color, line_thickness)
    
    # Draw BPD line
    cv2.line(img_draw, bpd_1, bpd_2, bpd_color, line_thickness)
    
    # Draw OFD points
    cv2.circle(img_draw, ofd_1, point_radius, ofd_color, -1)
    cv2.circle(img_draw, ofd_1, point_radius + 2, (255, 255, 255), 2)  # White border
    cv2.circle(img_draw, ofd_2, point_radius, ofd_color, -1)
    cv2.circle(img_draw, ofd_2, point_radius + 2, (255, 255, 255), 2)
    
    # Draw BPD points
    cv2.circle(img_draw, bpd_1, point_radius, bpd_color, -1)
    cv2.circle(img_draw, bpd_1, point_radius + 2, (255, 255, 255), 2)
    cv2.circle(img_draw, bpd_2, point_radius, bpd_color, -1)
    cv2.circle(img_draw, bpd_2, point_radius + 2, (255, 255, 255), 2)
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    # Label OFD
    mid_ofd = ((ofd_1[0] + ofd_2[0]) // 2, (ofd_1[1] + ofd_2[1]) // 2)
    cv2.putText(img_draw, 'OFD', (mid_ofd[0] + 10, mid_ofd[1] - 10), 
                font, font_scale, ofd_color, font_thickness)
    
    # Label BPD
    mid_bpd = ((bpd_1[0] + bpd_2[0]) // 2, (bpd_1[1] + bpd_2[1]) // 2)
    cv2.putText(img_draw, 'BPD', (mid_bpd[0] + 10, mid_bpd[1] - 10), 
                font, font_scale, bpd_color, font_thickness)
    
    # Calculate measurements
    ofd_distance = np.sqrt((ofd_2[0] - ofd_1[0])**2 + (ofd_2[1] - ofd_1[1])**2)
    bpd_distance = np.sqrt((bpd_2[0] - bpd_1[0])**2 + (bpd_2[1] - bpd_1[1])**2)
    
    # Add measurement text
    info_y = 30
    cv2.putText(img_draw, f'OFD: {ofd_distance:.1f} px', (10, info_y), 
                font, 0.7, ofd_color, 2)
    cv2.putText(img_draw, f'BPD: {bpd_distance:.1f} px', (10, info_y + 30), 
                font, 0.7, bpd_color, 2)
    
    # Convert back to RGB
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
    
    return img_draw


def visualize_ground_truth(
    csv_path: str,
    images_dir: str,
    num_images: int = 10,
    save_dir: str = None,
    start_idx: int = 0,
    save_individual: bool = False,
):
    """
    Visualize ground truth landmarks on ultrasound images.
    
    Args:
        csv_path: Path to CSV file with annotations
        images_dir: Directory containing images
        num_images: Number of images to visualize
        save_dir: Optional directory to save visualizations
        start_idx: Starting index in the dataset
        save_individual: If True, save each image separately instead of grid
    """
    # Load annotations
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} annotations from {csv_path}")
    
    # Select random images if we have more than requested
    if len(df) > num_images:
        # Use fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.choice(len(df), size=num_images, replace=False)
        indices = sorted(indices)  # Sort for better visualization
    else:
        indices = range(min(num_images, len(df)))
    
    print(f"\nVisualizing {len(indices)} images...")
    
    # Create save directory if needed
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    images_with_landmarks = []
    
    for idx, img_idx in enumerate(indices):
        row = df.iloc[img_idx]
        image_name = row['image_name']
        image_path = Path(images_dir) / image_name
        
        # Load image
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Failed to load image: {image_path}")
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract landmarks
        landmarks = {
            'ofd_1_x': row['ofd_1_x'],
            'ofd_1_y': row['ofd_1_y'],
            'ofd_2_x': row['ofd_2_x'],
            'ofd_2_y': row['ofd_2_y'],
            'bpd_1_x': row['bpd_1_x'],
            'bpd_1_y': row['bpd_1_y'],
            'bpd_2_x': row['bpd_2_x'],
            'bpd_2_y': row['bpd_2_y'],
        }
        
        # Draw landmarks
        img_with_landmarks = draw_landmarks_on_image(image, landmarks, image_name)
        
        # Calculate measurements
        ofd_distance = np.sqrt((landmarks['ofd_2_x'] - landmarks['ofd_1_x'])**2 + 
                              (landmarks['ofd_2_y'] - landmarks['ofd_1_y'])**2)
        bpd_distance = np.sqrt((landmarks['bpd_2_x'] - landmarks['bpd_1_x'])**2 + 
                              (landmarks['bpd_2_y'] - landmarks['bpd_1_y'])**2)
        
        print(f"  [{idx+1}/{len(indices)}] {image_name}: "
              f"OFD={ofd_distance:.1f}px, BPD={bpd_distance:.1f}px")
        
        # Save individual image if requested
        if save_individual and save_dir:
            fig_single = plt.figure(figsize=(12, 8))
            plt.imshow(img_with_landmarks)
            plt.title(
                f'{image_name}\nOFD: {ofd_distance:.1f}px | BPD: {bpd_distance:.1f}px',
                fontsize=14,
                fontweight='bold'
            )
            plt.axis('off')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='OFD'),
                Patch(facecolor='blue', label='BPD')
            ]
            plt.legend(handles=legend_elements, loc='upper right', fontsize=11)
            
            # Extract numeric prefix from image name (e.g., "001_HC.png" -> "1")
            base_name = Path(image_name).stem
            numeric_prefix = ''.join(c for c in base_name if c.isdigit())
            if numeric_prefix:
                # Remove leading zeros and save as N.jpg
                image_number = int(numeric_prefix)
                output_file = save_path / f'{image_number}.jpg'
            else:
                # Fallback to original name if no numeric prefix found
                output_file = save_path / f'{base_name}.jpg'
            
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig_single)
            print(f"    Saved: {output_file}")
        
        images_with_landmarks.append((img_with_landmarks, image_name, ofd_distance, bpd_distance))
    
    # Create grid visualization if not saving individual
    if not save_individual:
        rows = (len(images_with_landmarks) + 2) // 3  # 3 columns
        cols = min(3, len(images_with_landmarks))
        fig, axes = plt.subplots(rows, cols, figsize=(20, 7 * rows))
        
        # Flatten axes for easier indexing
        if len(images_with_landmarks) > 1:
            axes = axes.flatten() if rows * cols > 1 else [axes]
        else:
            axes = [axes]
        
        for idx, (img, name, ofd, bpd) in enumerate(images_with_landmarks):
            axes[idx].imshow(img)
            axes[idx].set_title(
                f'{name}\nOFD: {ofd:.1f}px | BPD: {bpd:.1f}px',
                fontsize=12,
                fontweight='bold'
            )
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(images_with_landmarks), len(axes)):
            axes[idx].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='OFD (Occipitofrontal Diameter)'),
            Patch(facecolor='blue', label='BPD (Biparietal Diameter)')
        ]
        fig.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save grid if requested
        if save_dir:
            output_file = save_path / f'ground_truth_visualization_{num_images}_images.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nSaved grid visualization to: {output_file}")
        
        # Only show if not saving individual images
        if not save_individual:
            plt.show()
        else:
            plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Calculate all measurements
    all_ofd = []
    all_bpd = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        ofd = np.sqrt((row['ofd_2_x'] - row['ofd_1_x'])**2 + 
                     (row['ofd_2_y'] - row['ofd_1_y'])**2)
        bpd = np.sqrt((row['bpd_2_x'] - row['bpd_1_x'])**2 + 
                     (row['bpd_2_y'] - row['bpd_1_y'])**2)
        all_ofd.append(ofd)
        all_bpd.append(bpd)
    
    all_ofd = np.array(all_ofd)
    all_bpd = np.array(all_bpd)
    
    print(f"\nTotal images: {len(df)}")
    print(f"\nOFD Statistics (pixels):")
    print(f"  Mean: {np.mean(all_ofd):.2f}")
    print(f"  Std:  {np.std(all_ofd):.2f}")
    print(f"  Min:  {np.min(all_ofd):.2f}")
    print(f"  Max:  {np.max(all_ofd):.2f}")
    
    print(f"\nBPD Statistics (pixels):")
    print(f"  Mean: {np.mean(all_bpd):.2f}")
    print(f"  Std:  {np.std(all_bpd):.2f}")
    print(f"  Min:  {np.min(all_bpd):.2f}")
    print(f"  Max:  {np.max(all_bpd):.2f}")
    
    print(f"\nOFD/BPD Ratio:")
    ratio = all_ofd / all_bpd
    print(f"  Mean: {np.mean(ratio):.3f}")
    print(f"  Std:  {np.std(ratio):.3f}")
    print(f"  Range: [{np.min(ratio):.3f}, {np.max(ratio):.3f}]")
    
    print("\n" + "="*60)


def visualize_single_image(
    csv_path: str,
    images_dir: str,
    image_name: str,
):
    """
    Visualize a single image with its ground truth landmarks.
    
    Args:
        csv_path: Path to CSV file
        images_dir: Directory containing images
        image_name: Name of the image to visualize
    """
    # Load annotations
    df = pd.read_csv(csv_path)
    
    # Find the image
    row = df[df['image_name'] == image_name]
    
    if len(row) == 0:
        print(f"Error: Image {image_name} not found in CSV")
        return
    
    row = row.iloc[0]
    
    # Load image
    image_path = Path(images_dir) / image_name
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extract landmarks
    landmarks = {
        'ofd_1_x': row['ofd_1_x'],
        'ofd_1_y': row['ofd_1_y'],
        'ofd_2_x': row['ofd_2_x'],
        'ofd_2_y': row['ofd_2_y'],
        'bpd_1_x': row['bpd_1_x'],
        'bpd_1_y': row['bpd_1_y'],
        'bpd_2_x': row['bpd_2_x'],
        'bpd_2_y': row['bpd_2_y'],
    }
    
    # Draw landmarks
    img_with_landmarks = draw_landmarks_on_image(image, landmarks, image_name)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_landmarks)
    plt.title(f'Ground Truth Landmarks: {image_name}', fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='OFD (Occipitofrontal Diameter)'),
        Patch(facecolor='blue', label='BPD (Biparietal Diameter)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize ground truth landmarks on fetal ultrasound images'
    )
    
    parser.add_argument(
        '--csv_path',
        type=str,
        default='role_challenge_dataset_ground_truth.csv',
        help='Path to CSV file with annotations'
    )
    
    parser.add_argument(
        '--images_dir',
        type=str,
        default='images',
        help='Directory containing images'
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
        '--image_name',
        type=str,
        default=None,
        help='Visualize a specific image by name (optional)'
    )
        
    
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Starting index in the dataset'
    )
    
    parser.add_argument(
        '--save_individual',
        action='store_true',
        help='Save each image individually in addition to grid'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    csv_path = Path(args.csv_path)
    images_dir = Path(args.images_dir)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        print(f"Current directory: {Path.cwd()}")
        return
    
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        print(f"Current directory: {Path.cwd()}")
        return
    
    print("="*60)
    print("FETAL ULTRASOUND GROUND TRUTH VISUALIZATION")
    print("="*60)
    print(f"CSV file: {csv_path}")
    print(f"Images directory: {images_dir}")
    print("="*60 + "\n")
    
    # Visualize specific image or multiple images
    if args.image_name:
        visualize_single_image(str(csv_path), str(images_dir), args.image_name)
    else:
        visualize_ground_truth(
            str(csv_path),
            str(images_dir),
            num_images=args.num_images,
            save_dir=args.save_dir,
            start_idx=args.start_idx,
            save_individual=args.save_individual,
        )


if __name__ == '__main__':
    main()

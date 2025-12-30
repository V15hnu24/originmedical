"""
Test and visualize different augmentation strategies.

Usage:
    python utils/test_augmentations.py --image_path images/001_HC.png --num_samples 9
    python utils/test_augmentations.py --strength light --num_samples 12
"""
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import pandas as pd

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent))

from data.augmentation import get_augmentation_pipeline


def visualize_augmentations(
    image_path: str,
    csv_path: str,
    num_samples: int = 8,
    strength: str = 'medium',
    save_path: str = None
):
    """
    Visualize original + augmentations in a single plot.
    
    Args:
        image_path: Path to input image
        csv_path: Path to CSV with landmarks
        num_samples: Number of augmented samples (original + this many augmented)
        strength: 'light', 'medium', or 'heavy'
        save_path: Optional path to save visualization
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load landmarks
    df = pd.read_csv(csv_path)
    image_name = Path(image_path).name
    row = df[df['image_name'] == image_name].iloc[0]
    
    landmarks = [
        [row['ofd_1_x'], row['ofd_1_y']],
        [row['ofd_2_x'], row['ofd_2_y']],
        [row['bpd_1_x'], row['bpd_1_y']],
        [row['bpd_2_x'], row['bpd_2_y']],
    ]
    
    # Get augmentation pipeline (for augmented samples)
    aug_pipeline = get_augmentation_pipeline(
        image_size=(512, 512),
        augment=True,
        mode='landmark',
        augmentation_strength=strength
    )
    
    # No-aug pipeline (for original)
    no_aug_pipeline = get_augmentation_pipeline(
        image_size=(512, 512),
        augment=False,
        mode='landmark'
    )
    
    # Calculate grid size (original + num_samples augmented)
    total_samples = num_samples + 1
    cols = 3
    rows = (total_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    print(f"Generating visualization: 1 original + {num_samples} augmented samples (strength='{strength}')...")
    
    # First, show original image
    original = no_aug_pipeline(image=image, keypoints=landmarks)
    orig_image = original['image']
    orig_landmarks = original['keypoints']
    
    # Convert tensor back to numpy
    if hasattr(orig_image, 'numpy'):
        orig_image = orig_image.numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        orig_image = orig_image.transpose(1, 2, 0)
        orig_image = std * orig_image + mean
        orig_image = np.clip(orig_image * 255, 0, 255).astype(np.uint8)
    
    # Draw landmarks on original
    orig_image_draw = orig_image.copy()
    if len(orig_landmarks) == 4:
        # Draw OFD (red)
        pt1 = tuple(map(int, orig_landmarks[0]))
        pt2 = tuple(map(int, orig_landmarks[1]))
        cv2.line(orig_image_draw, pt1, pt2, (255, 0, 0), 3)
        cv2.circle(orig_image_draw, pt1, 6, (255, 0, 0), -1)
        cv2.circle(orig_image_draw, pt2, 6, (255, 0, 0), -1)
        
        # Draw BPD (blue)
        pt3 = tuple(map(int, orig_landmarks[2]))
        pt4 = tuple(map(int, orig_landmarks[3]))
        cv2.line(orig_image_draw, pt3, pt4, (0, 0, 255), 3)
        cv2.circle(orig_image_draw, pt3, 6, (0, 0, 255), -1)
        cv2.circle(orig_image_draw, pt4, 6, (0, 0, 255), -1)
        
        # Add labels
        cv2.putText(orig_image_draw, 'OFD', (pt1[0]+10, pt1[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(orig_image_draw, 'BPD', (pt3[0]+10, pt3[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    axes[0].imshow(orig_image_draw)
    axes[0].set_title('ORIGINAL', fontsize=14, fontweight='bold', color='green')
    axes[0].axis('off')
    
    # Now generate augmented samples
    for idx in range(num_samples):
        # Apply augmentation
        augmented = aug_pipeline(image=image, keypoints=landmarks)
        aug_image = augmented['image']
        aug_landmarks = augmented['keypoints']
        
        # Convert tensor back to numpy for visualization
        if hasattr(aug_image, 'numpy'):
            aug_image = aug_image.numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            aug_image = aug_image.transpose(1, 2, 0)
            aug_image = std * aug_image + mean
            aug_image = np.clip(aug_image * 255, 0, 255).astype(np.uint8)
        
        # Draw landmarks
        aug_image_draw = aug_image.copy()
        
        if len(aug_landmarks) == 4:
            # Draw OFD (red)
            pt1 = tuple(map(int, aug_landmarks[0]))
            pt2 = tuple(map(int, aug_landmarks[1]))
            cv2.line(aug_image_draw, pt1, pt2, (255, 0, 0), 2)
            cv2.circle(aug_image_draw, pt1, 5, (255, 0, 0), -1)
            cv2.circle(aug_image_draw, pt2, 5, (255, 0, 0), -1)
            
            # Draw BPD (blue)
            pt3 = tuple(map(int, aug_landmarks[2]))
            pt4 = tuple(map(int, aug_landmarks[3]))
            cv2.line(aug_image_draw, pt3, pt4, (0, 0, 255), 2)
            cv2.circle(aug_image_draw, pt3, 5, (0, 0, 255), -1)
            cv2.circle(aug_image_draw, pt4, 5, (0, 0, 255), -1)
        
        # Display in position idx+1 (after original)
        axes[idx+1].imshow(aug_image_draw)
        axes[idx+1].set_title(f'Augmented {idx+1}', fontsize=12)
        axes[idx+1].axis('off')
    
    # Hide unused subplots
    for idx in range(total_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Original vs Augmented Samples | Strength: {strength.upper()} | Image: {image_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def compare_strengths(
    image_path: str,
    csv_path: str,
    save_dir: str = None
):
    """
    Compare light, medium, and heavy augmentation strengths.
    """
    strengths = ['light', 'medium', 'heavy']
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load landmarks
    df = pd.read_csv(csv_path)
    image_name = Path(image_path).name
    row = df[df['image_name'] == image_name].iloc[0]
    
    landmarks = [
        [row['ofd_1_x'], row['ofd_1_y']],
        [row['ofd_2_x'], row['ofd_2_y']],
        [row['bpd_1_x'], row['bpd_1_y']],
        [row['bpd_2_x'], row['bpd_2_y']],
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row_idx, strength in enumerate(strengths):
        aug_pipeline = get_augmentation_pipeline(
            image_size=(512, 512),
            augment=True,
            mode='landmark',
            augmentation_strength=strength
        )
        
        for col_idx in range(4):
            # Apply augmentation
            augmented = aug_pipeline(image=image, keypoints=landmarks)
            aug_image = augmented['image']
            aug_landmarks = augmented['keypoints']
            
            # Convert tensor back to numpy
            if hasattr(aug_image, 'numpy'):
                aug_image = aug_image.numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                aug_image = aug_image.transpose(1, 2, 0)
                aug_image = std * aug_image + mean
                aug_image = np.clip(aug_image * 255, 0, 255).astype(np.uint8)
            
            # Draw landmarks
            aug_image_draw = aug_image.copy()
            
            if len(aug_landmarks) == 4:
                for i in range(0, 4, 2):
                    color = (255, 0, 0) if i == 0 else (0, 0, 255)
                    pt1 = tuple(map(int, aug_landmarks[i]))
                    pt2 = tuple(map(int, aug_landmarks[i+1]))
                    cv2.line(aug_image_draw, pt1, pt2, color, 2)
                    cv2.circle(aug_image_draw, pt1, 4, color, -1)
                    cv2.circle(aug_image_draw, pt2, 4, color, -1)
            
            # Display
            axes[row_idx, col_idx].imshow(aug_image_draw)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(strength.upper(), fontsize=12, fontweight='bold')
            axes[row_idx, col_idx].axis('off')
    
    plt.suptitle(f'Augmentation Strength Comparison\n{image_name}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(save_dir) / 'augmentation_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Test and visualize augmentations'
    )
    
    parser.add_argument(
        '--image_path',
        type=str,
        default='images/001_HC.png',
        help='Path to input image'
    )
    
    parser.add_argument(
        '--csv_path',
        type=str,
        default='role_challenge_dataset_ground_truth.csv',
        help='Path to CSV with landmarks'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=9,
        help='Number of augmented samples to generate'
    )
    
    parser.add_argument(
        '--strength',
        type=str,
        default='medium',
        choices=['light', 'medium', 'heavy'],
        help='Augmentation strength'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all augmentation strengths'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output/augmentations',
        help='Directory to save visualizations'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("AUGMENTATION VISUALIZATION")
    print("="*60)
    print(f"Image: {args.image_path}")
    print(f"Landmarks CSV: {args.csv_path}")
    print("="*60 + "\n")
    
    if args.compare:
        compare_strengths(
            args.image_path,
            args.csv_path,
            save_dir=args.save_dir
        )
    else:
        save_path = None
        if args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(args.save_dir) / f'augmentation_{args.strength}_{args.num_samples}samples.png'
        
        visualize_augmentations(
            args.image_path,
            args.csv_path,
            num_samples=args.num_samples,
            strength=args.strength,
            save_path=save_path
        )


if __name__ == '__main__':
    main()

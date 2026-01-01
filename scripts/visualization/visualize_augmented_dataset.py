"""
Visualize augmented dataset.

Usage:
    python visualize_augmented_dataset.py
    python visualize_augmented_dataset.py --num_samples 16 --compare_originals
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random


def visualize_random_samples(csv_path: str, images_dir: str, num_samples: int = 12, save_path: str = None):
    """Visualize random samples from augmented dataset."""
    df = pd.read_csv(csv_path)
    
    # Sample random images
    samples = df.sample(n=min(num_samples, len(df)))
    
    # Create grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= num_samples:
            break
            
        image_path = Path(images_dir) / row['image_name']
        
        if not image_path.exists():
            axes[idx].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            axes[idx].axis('off')
            continue
        
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw landmarks
        ofd_1 = (int(row['ofd_1_x']), int(row['ofd_1_y']))
        ofd_2 = (int(row['ofd_2_x']), int(row['ofd_2_y']))
        bpd_1 = (int(row['bpd_1_x']), int(row['bpd_1_y']))
        bpd_2 = (int(row['bpd_2_x']), int(row['bpd_2_y']))
        
        # Draw lines
        cv2.line(image, ofd_1, ofd_2, (255, 0, 0), 2)  # Red for OFD
        cv2.line(image, bpd_1, bpd_2, (0, 0, 255), 2)  # Blue for BPD
        
        # Draw points
        for pt in [ofd_1, ofd_2]:
            cv2.circle(image, pt, 5, (255, 0, 0), -1)
        for pt in [bpd_1, bpd_2]:
            cv2.circle(image, pt, 5, (0, 0, 255), -1)
        
        axes[idx].imshow(image)
        
        # Determine if original or augmented
        img_name = row['image_name']
        if 'orig_' in img_name:
            title = f"Original\n{img_name}"
            axes[idx].set_title(title, fontsize=9, color='green', fontweight='bold')
        elif '_aug' in img_name:
            title = f"Augmented\n{img_name}"
            axes[idx].set_title(title, fontsize=9, color='orange')
        else:
            axes[idx].set_title(img_name, fontsize=9)
        
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Random Samples from Augmented Dataset ({len(df)} total images)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def compare_original_with_augmented(csv_path: str, images_dir: str, num_originals: int = 3, save_path: str = None):
    """Compare original images with their augmented versions."""
    df = pd.read_csv(csv_path)
    
    # Get original images
    orig_df = df[df['image_name'].str.contains('orig_')]
    
    if len(orig_df) == 0:
        print("No original images found in dataset!")
        return
    
    # Sample random originals
    sampled_originals = orig_df.sample(n=min(num_originals, len(orig_df)))
    
    for orig_idx, (_, orig_row) in enumerate(sampled_originals.iterrows()):
        orig_name = orig_row['image_name']
        
        # Find augmented versions (remove 'orig_' prefix and find matching augmented)
        base_name = orig_name.replace('orig_', '')
        base_stem = Path(base_name).stem
        
        # Find all augmented versions
        aug_versions = df[df['image_name'].str.contains(f"{base_stem}_aug")]
        
        if len(aug_versions) == 0:
            print(f"No augmented versions found for {orig_name}")
            continue
        
        # Create figure with original + augmented versions
        num_augs = len(aug_versions)
        cols = min(4, num_augs + 1)
        rows = (num_augs + 1 + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if (num_augs + 1) > 1 else [axes]
        
        # Plot original
        orig_path = Path(images_dir) / orig_name
        if orig_path.exists():
            orig_image = cv2.imread(str(orig_path))
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            
            # Draw landmarks on original
            ofd_1 = (int(orig_row['ofd_1_x']), int(orig_row['ofd_1_y']))
            ofd_2 = (int(orig_row['ofd_2_x']), int(orig_row['ofd_2_y']))
            bpd_1 = (int(orig_row['bpd_1_x']), int(orig_row['bpd_1_y']))
            bpd_2 = (int(orig_row['bpd_2_x']), int(orig_row['bpd_2_y']))
            
            cv2.line(orig_image, ofd_1, ofd_2, (255, 0, 0), 2)
            cv2.line(orig_image, bpd_1, bpd_2, (0, 0, 255), 2)
            for pt in [ofd_1, ofd_2]:
                cv2.circle(orig_image, pt, 5, (255, 0, 0), -1)
            for pt in [bpd_1, bpd_2]:
                cv2.circle(orig_image, pt, 5, (0, 0, 255), -1)
            
            axes[0].imshow(orig_image)
            axes[0].set_title('ORIGINAL', fontsize=12, color='green', fontweight='bold')
            axes[0].axis('off')
        
        # Plot augmented versions
        for aug_idx, (_, aug_row) in enumerate(aug_versions.iterrows(), start=1):
            if aug_idx >= len(axes):
                break
                
            aug_path = Path(images_dir) / aug_row['image_name']
            
            if not aug_path.exists():
                axes[aug_idx].text(0.5, 0.5, 'Not found', ha='center', va='center')
                axes[aug_idx].axis('off')
                continue
            
            aug_image = cv2.imread(str(aug_path))
            aug_image = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
            
            # Draw landmarks
            ofd_1 = (int(aug_row['ofd_1_x']), int(aug_row['ofd_1_y']))
            ofd_2 = (int(aug_row['ofd_2_x']), int(aug_row['ofd_2_y']))
            bpd_1 = (int(aug_row['bpd_1_x']), int(aug_row['bpd_1_y']))
            bpd_2 = (int(aug_row['bpd_2_x']), int(aug_row['bpd_2_y']))
            
            cv2.line(aug_image, ofd_1, ofd_2, (255, 0, 0), 2)
            cv2.line(aug_image, bpd_1, bpd_2, (0, 0, 255), 2)
            for pt in [ofd_1, ofd_2]:
                cv2.circle(aug_image, pt, 5, (255, 0, 0), -1)
            for pt in [bpd_1, bpd_2]:
                cv2.circle(aug_image, pt, 5, (0, 0, 255), -1)
            
            axes[aug_idx].imshow(aug_image)
            axes[aug_idx].set_title(f'Augmented {aug_idx}', fontsize=10, color='orange')
            axes[aug_idx].axis('off')
        
        # Hide unused subplots
        for idx in range(num_augs + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Original vs Augmented Versions: {base_name}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            output_path = Path(save_path).parent / f"{Path(save_path).stem}_comparison_{orig_idx}{Path(save_path).suffix}"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison {orig_idx+1} to: {output_path}")
        
        plt.show()


def print_dataset_stats(csv_path: str):
    """Print statistics about the augmented dataset."""
    df = pd.read_csv(csv_path)
    
    num_originals = len(df[df['image_name'].str.contains('orig_')])
    num_augmented = len(df[df['image_name'].str.contains('_aug')])
    
    print("\n" + "="*60)
    print("AUGMENTED DATASET STATISTICS")
    print("="*60)
    print(f"Total images: {len(df)}")
    print(f"Original images: {num_originals}")
    print(f"Augmented images: {num_augmented}")
    print(f"Expansion factor: {len(df) / num_originals:.2f}x" if num_originals > 0 else "N/A")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize augmented dataset')
    
    parser.add_argument(
        '--csv_path',
        type=str,
        default='data/augmented/augmented_ground_truth.csv',
        help='Path to augmented CSV'
    )
    
    parser.add_argument(
        '--images_dir',
        type=str,
        default='data/augmented/images',
        help='Directory with augmented images'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=12,
        help='Number of random samples to visualize'
    )
    
    parser.add_argument(
        '--compare_originals',
        action='store_true',
        help='Compare original images with their augmented versions'
    )
    
    parser.add_argument(
        '--num_comparisons',
        type=int,
        default=3,
        help='Number of original images to compare with augmented versions'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output/augmented_visualizations',
        help='Directory to save visualizations'
    )
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Print statistics
    print_dataset_stats(args.csv_path)
    
    # Visualize random samples
    print("Generating random samples visualization...")
    visualize_random_samples(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        num_samples=args.num_samples,
        save_path=str(save_dir / 'random_samples.png')
    )
    
    # Compare originals with augmented if requested
    if args.compare_originals:
        print("\nGenerating original vs augmented comparisons...")
        compare_original_with_augmented(
            csv_path=args.csv_path,
            images_dir=args.images_dir,
            num_originals=args.num_comparisons,
            save_path=str(save_dir / 'comparison.png')
        )


if __name__ == '__main__':
    main()

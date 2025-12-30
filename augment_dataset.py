"""
Generate augmented dataset by creating augmented copies of images and updating ground truth.

This creates an expanded dataset with original + augmented samples.

Usage:
    python augment_dataset.py --augmentations_per_image 3 --strength medium
    python augment_dataset.py --augmentations_per_image 5 --strength heavy --output_dir data/augmented
"""
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from data.augmentation import get_augmentation_pipeline


def augment_dataset(
    csv_path: str,
    images_dir: str,
    output_images_dir: str,
    output_csv_path: str,
    augmentations_per_image: int = 3,
    strength: str = 'medium',
    seed: int = 42
):
    """
    Generate augmented dataset.
    
    Args:
        csv_path: Path to original CSV with ground truth
        images_dir: Directory with original images
        output_images_dir: Directory to save augmented images
        output_csv_path: Path to save augmented CSV
        augmentations_per_image: Number of augmented versions per image
        strength: Augmentation strength ('light', 'medium', 'heavy')
        seed: Random seed
    """
    # Set seed
    np.random.seed(seed)
    
    # Create output directory
    output_images_path = Path(output_images_dir)
    output_images_path.mkdir(parents=True, exist_ok=True)
    
    # Load original CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} original samples from {csv_path}")
    
    # Get augmentation pipeline
    aug_pipeline = get_augmentation_pipeline(
        image_size=(512, 512),
        augment=True,
        mode='landmark',
        augmentation_strength=strength
    )
    
    # No-aug pipeline for validation
    no_aug_pipeline = get_augmentation_pipeline(
        image_size=(512, 512),
        augment=False,
        mode='landmark'
    )
    
    # Store augmented data
    augmented_rows = []
    
    print(f"\nGenerating {augmentations_per_image} augmented versions per image...")
    print(f"Augmentation strength: {strength}")
    print(f"Output directory: {output_images_dir}")
    print(f"Output CSV: {output_csv_path}\n")
    
    # Process each image
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting images"):
        image_name = row['image_name']
        image_path = Path(images_dir) / image_name
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Failed to load: {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Original landmarks
        landmarks = [
            [row['ofd_1_x'], row['ofd_1_y']],
            [row['ofd_2_x'], row['ofd_2_y']],
            [row['bpd_1_x'], row['bpd_1_y']],
            [row['bpd_2_x'], row['bpd_2_y']],
        ]
        
        # Save original image (resized to 512x512)
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
        
        # Save original (resized)
        orig_output_name = f"orig_{image_name}"
        orig_output_path = output_images_path / orig_output_name
        cv2.imwrite(str(orig_output_path), cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR))
        
        # Add original to CSV
        if len(orig_landmarks) == 4:
            augmented_rows.append({
                'image_name': orig_output_name,
                'ofd_1_x': orig_landmarks[0][0],
                'ofd_1_y': orig_landmarks[0][1],
                'ofd_2_x': orig_landmarks[1][0],
                'ofd_2_y': orig_landmarks[1][1],
                'bpd_1_x': orig_landmarks[2][0],
                'bpd_1_y': orig_landmarks[2][1],
                'bpd_2_x': orig_landmarks[3][0],
                'bpd_2_y': orig_landmarks[3][1],
            })
        
        # Generate augmented versions
        for aug_idx in range(augmentations_per_image):
            try:
                # Apply augmentation
                augmented = aug_pipeline(image=image, keypoints=landmarks)
                aug_image = augmented['image']
                aug_landmarks = augmented['keypoints']
                
                # Skip if landmarks were lost
                if len(aug_landmarks) != 4:
                    continue
                
                # Convert tensor back to numpy
                if hasattr(aug_image, 'numpy'):
                    aug_image = aug_image.numpy()
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    aug_image = aug_image.transpose(1, 2, 0)
                    aug_image = std * aug_image + mean
                    aug_image = np.clip(aug_image * 255, 0, 255).astype(np.uint8)
                
                # Save augmented image
                base_name = Path(image_name).stem
                ext = Path(image_name).suffix
                aug_output_name = f"{base_name}_aug{aug_idx+1}{ext}"
                aug_output_path = output_images_path / aug_output_name
                
                cv2.imwrite(str(aug_output_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                
                # Add to CSV
                augmented_rows.append({
                    'image_name': aug_output_name,
                    'ofd_1_x': aug_landmarks[0][0],
                    'ofd_1_y': aug_landmarks[0][1],
                    'ofd_2_x': aug_landmarks[1][0],
                    'ofd_2_y': aug_landmarks[1][1],
                    'bpd_1_x': aug_landmarks[2][0],
                    'bpd_1_y': aug_landmarks[2][1],
                    'bpd_2_x': aug_landmarks[3][0],
                    'bpd_2_y': aug_landmarks[3][1],
                })
                
            except Exception as e:
                print(f"Warning: Failed to augment {image_name} (aug {aug_idx+1}): {e}")
                continue
    
    # Create augmented CSV
    augmented_df = pd.DataFrame(augmented_rows)
    augmented_df.to_csv(output_csv_path, index=False)
    
    print(f"\n{'='*60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Original images: {len(df)}")
    print(f"Augmented images generated: {len(augmented_df) - len(df)}")
    print(f"Total images: {len(augmented_df)}")
    print(f"Expansion factor: {len(augmented_df) / len(df):.2f}x")
    print(f"\nAugmented images saved to: {output_images_dir}")
    print(f"Augmented CSV saved to: {output_csv_path}")
    print(f"{'='*60}\n")
    
    return augmented_df


def verify_augmented_data(csv_path: str, images_dir: str, num_samples: int = 5):
    """
    Verify augmented data by checking a few samples.
    """
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(csv_path)
    print(f"Verifying augmented dataset: {len(df)} samples")
    
    # Sample some augmented images
    aug_samples = df[df['image_name'].str.contains('_aug')].sample(min(num_samples, len(df)))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    
    for idx, (_, row) in enumerate(aug_samples.iterrows()):
        image_path = Path(images_dir) / row['image_name']
        
        if not image_path.exists():
            print(f"Warning: {image_path} not found")
            continue
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw landmarks
        ofd_1 = (int(row['ofd_1_x']), int(row['ofd_1_y']))
        ofd_2 = (int(row['ofd_2_x']), int(row['ofd_2_y']))
        bpd_1 = (int(row['bpd_1_x']), int(row['bpd_1_y']))
        bpd_2 = (int(row['bpd_2_x']), int(row['bpd_2_y']))
        
        cv2.line(image, ofd_1, ofd_2, (255, 0, 0), 2)
        cv2.line(image, bpd_1, bpd_2, (0, 0, 255), 2)
        for pt in [ofd_1, ofd_2, bpd_1, bpd_2]:
            cv2.circle(image, pt, 4, (255, 255, 255), -1)
        
        axes[idx].imshow(image)
        axes[idx].set_title(row['image_name'], fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(images_dir).parent / 'augmentation_verification.png', dpi=150, bbox_inches='tight')
    print(f"Verification plot saved to: {Path(images_dir).parent / 'augmentation_verification.png'}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Generate augmented dataset offline'
    )
    
    parser.add_argument(
        '--csv_path',
        type=str,
        default='role_challenge_dataset_ground_truth.csv',
        help='Path to original CSV with ground truth'
    )
    
    parser.add_argument(
        '--images_dir',
        type=str,
        default='images',
        help='Directory with original images'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/augmented',
        help='Output directory for augmented data'
    )
    
    parser.add_argument(
        '--augmentations_per_image',
        type=int,
        default=3,
        help='Number of augmented versions per image'
    )
    
    parser.add_argument(
        '--strength',
        type=str,
        default='medium',
        choices=['light', 'medium', 'heavy'],
        help='Augmentation strength'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify augmented data after generation'
    )
    
    args = parser.parse_args()
    
    # Set up output paths
    output_images_dir = Path(args.output_dir) / 'images'
    output_csv_path = Path(args.output_dir) / 'augmented_ground_truth.csv'
    
    print("="*60)
    print("OFFLINE DATA AUGMENTATION")
    print("="*60)
    print(f"Original CSV: {args.csv_path}")
    print(f"Original images: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Augmentations per image: {args.augmentations_per_image}")
    print(f"Strength: {args.strength}")
    print("="*60 + "\n")
    
    # Generate augmented dataset
    augmented_df = augment_dataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
        output_images_dir=str(output_images_dir),
        output_csv_path=str(output_csv_path),
        augmentations_per_image=args.augmentations_per_image,
        strength=args.strength,
        seed=args.seed
    )
    
    # Verify if requested
    if args.verify:
        print("\nVerifying augmented data...")
        verify_augmented_data(str(output_csv_path), str(output_images_dir), num_samples=5)


if __name__ == '__main__':
    main()

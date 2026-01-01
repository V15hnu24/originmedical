"""
Create GT vs Prediction visualizations with green GT and blue predictions

Usage:
    python visualize_gt_pred_overlay.py --checkpoint checkpoints/coordinate/best/checkpoint.ckpt --num_samples 30
"""
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

import config
from data.dataset import create_dataloaders
from train_landmark import LandmarkDetectionModule


def visualize_gt_pred_overlay(checkpoint_path, num_samples=30, output_dir=None):
    """
    Create visualizations showing GT (green) and predictions (blue) overlaid on images.
    """
    print(f"\n{'='*60}")
    print(f"Creating GT (Green) vs Prediction (Blue) Visualizations")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = LandmarkDetectionModule.load_from_checkpoint(checkpoint_path)
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully on {device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Create validation dataloader
    print("\nLoading validation data...")
    _, val_loader = create_dataloaders(
        csv_path=str(config.CSV_PATH),
        image_dir=str(config.DATA_DIR),
        batch_size=1,
        num_workers=config.NUM_WORKERS,
        train_val_split=config.TRAIN_VAL_SPLIT,
        random_seed=config.RANDOM_SEED,
        image_size=config.IMAGE_SIZE,
        mode='landmark',
    )
    print(f"✓ Loaded {len(val_loader.dataset)} validation samples\n")
    
    # Create output directory
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "unnet_e48" / "visualise"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}\n")
    
    # Process samples
    print(f"Creating visualizations for {num_samples} samples...")
    samples_processed = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Processing", total=num_samples)):
            if samples_processed >= num_samples:
                break
            
            # Move to device
            images = batch['image'].to(device)
            gt_landmarks = batch['landmarks'].to(device)
            image_names = batch['image_name']
            
            # Forward pass
            outputs = model(images)
            pred_landmarks = outputs['coordinates']
            
            # Predictions are in resized image space (512x512)
            pred_np = pred_landmarks[0].cpu().numpy()
            
            # Load original image (before preprocessing)
            image_path = config.DATA_DIR / image_names[0]
            original_image = cv2.imread(str(image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = original_image.shape[:2]
            
            # Load GT landmarks from CSV (these are in ORIGINAL image space)
            import pandas as pd
            val_csv_path = Path(config.CSV_PATH).parent / 'val_split.csv'
            df = pd.read_csv(val_csv_path)
            row = df[df['image_name'] == image_names[0]].iloc[0]
            
            # Get original GT landmarks (already in original image space)
            gt_orig = np.array([
                [row['ofd_1_x'], row['ofd_1_y']],
                [row['ofd_2_x'], row['ofd_2_y']],
                [row['bpd_1_x'], row['bpd_1_y']],
                [row['bpd_2_x'], row['bpd_2_y']],
            ], dtype=np.float32)
            
            # Scale predictions from 512x512 to original image size
            # Calculate scaling factors
            scale_x = orig_w / config.IMAGE_SIZE[1]  # original_width / 512
            scale_y = orig_h / config.IMAGE_SIZE[0]  # original_height / 512
            
            pred_orig = pred_np.copy()
            pred_orig[:, 0] = pred_np[:, 0] * scale_x
            pred_orig[:, 1] = pred_np[:, 1] * scale_y
            
            # Compute errors in original image space
            errors_orig = np.linalg.norm(pred_orig - gt_orig, axis=1)
            mre_orig = errors_orig.mean()
            
            # Create visualization
            create_overlay_visualization(
                original_image,
                gt_orig,
                pred_orig,
                errors_orig,
                mre_orig,
                image_names[0],
                output_dir
            )
            
            samples_processed += 1
    
    print(f"\n✓ Created {samples_processed} visualizations")
    print(f"✓ Saved to: {output_dir}\n")


def create_overlay_visualization(image, gt_landmarks, pred_landmarks, errors, mre, image_name, output_dir):
    """Create a single overlay visualization with GT in green and predictions in blue."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Display image
    ax.imshow(image)
    
    # Define colors
    gt_color = 'lime'  # Green for GT
    pred_color = 'blue'  # Blue for predictions
    landmark_size = 150
    line_width = 3
    
    # Landmark labels
    landmark_labels = ['OFD1', 'OFD2', 'BPD1', 'BPD2']
    
    # Draw GT landmarks and lines
    # Draw OFD line (landmarks 0 and 1)
    ax.plot([gt_landmarks[0, 0], gt_landmarks[1, 0]], 
            [gt_landmarks[0, 1], gt_landmarks[1, 1]], 
            color=gt_color, linewidth=line_width, label='GT OFD', linestyle='-')
    
    # Draw BPD line (landmarks 2 and 3)
    ax.plot([gt_landmarks[2, 0], gt_landmarks[3, 0]], 
            [gt_landmarks[2, 1], gt_landmarks[3, 1]], 
            color=gt_color, linewidth=line_width, label='GT BPD', linestyle='--')
    
    # Draw GT landmarks
    for i, (lm, label) in enumerate(zip(gt_landmarks, landmark_labels)):
        ax.scatter(lm[0], lm[1], c=gt_color, s=landmark_size, marker='o', 
                  edgecolors='white', linewidths=2, 
                  label='GT Landmarks' if i == 0 else '', zorder=5)
        # Add text label
        ax.text(lm[0], lm[1]-20, label, color='white', fontsize=10, 
               fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=gt_color, 
                        edgecolor='white', alpha=0.9, linewidth=1.5), zorder=6)
    
    # Draw Prediction landmarks and lines
    # Draw OFD line (landmarks 0 and 1)
    ax.plot([pred_landmarks[0, 0], pred_landmarks[1, 0]], 
            [pred_landmarks[0, 1], pred_landmarks[1, 1]], 
            color=pred_color, linewidth=line_width, label='Pred OFD', linestyle='-', alpha=0.8)
    
    # Draw BPD line (landmarks 2 and 3)
    ax.plot([pred_landmarks[2, 0], pred_landmarks[3, 0]], 
            [pred_landmarks[2, 1], pred_landmarks[3, 1]], 
            color=pred_color, linewidth=line_width, label='Pred BPD', linestyle='--', alpha=0.8)
    
    # Draw predicted landmarks
    for i, (lm, label, err) in enumerate(zip(pred_landmarks, landmark_labels, errors)):
        ax.scatter(lm[0], lm[1], c=pred_color, s=landmark_size, marker='x', 
                  linewidths=3, 
                  label='Pred Landmarks' if i == 0 else '', zorder=5)
        # Add text label with error
        ax.text(lm[0], lm[1]+25, f'{label}\n({err:.1f}px)', color='white', fontsize=9, 
               fontweight='bold', ha='center',
               bbox=dict(boxstyle='round,pad=0.4', facecolor=pred_color, 
                        edgecolor='white', alpha=0.9, linewidth=1.5), zorder=6)
    
    # Add title with MRE
    clean_name = Path(image_name).stem
    ax.set_title(f'{clean_name} - Mean Radial Error: {mre:.2f} pixels', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{clean_name}.png", dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize GT (green) vs predictions (blue) on images')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--num_samples', type=int, default=30,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save outputs (default: results/unnet_e48/visualise)')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = config.CHECKPOINT_DIR / "coordinate"
        if checkpoint_dir.exists():
            for ckpt in checkpoint_dir.rglob("*.ckpt"):
                print(f"  - {ckpt}")
        return
    
    # Run visualization
    visualize_gt_pred_overlay(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

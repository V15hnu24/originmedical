"""
Create detailed GT vs Prediction visualizations on original images

Usage:
    python visualize_predictions_on_images.py --checkpoint checkpoints/coordinate/best/checkpoint.ckpt --num_samples 30
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


def visualize_predictions_on_images(checkpoint_path, num_samples=30, output_dir=None):
    """
    Create visualizations showing GT vs predictions on original images.
    """
    print(f"\n{'='*60}")
    print(f"Creating GT vs Prediction Visualizations")
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
        output_dir = config.RESULTS_DIR / "gt_vs_predictions"
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
            
            # Compute error
            pred_np = pred_landmarks[0].cpu().numpy()
            gt_np = gt_landmarks[0].cpu().numpy()
            errors = np.linalg.norm(pred_np - gt_np, axis=1)
            mre = errors.mean()
            
            # Load original image (before preprocessing)
            image_path = config.DATA_DIR / image_names[0]
            original_image = cv2.imread(str(image_path))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = original_image.shape[:2]
            
            # Scale landmarks back to original image size
            scale_x = orig_w / config.IMAGE_SIZE[1]
            scale_y = orig_h / config.IMAGE_SIZE[0]
            
            gt_orig = gt_np.copy()
            gt_orig[:, 0] *= scale_x
            gt_orig[:, 1] *= scale_y
            
            pred_orig = pred_np.copy()
            pred_orig[:, 0] *= scale_x
            pred_orig[:, 1] *= scale_y
            
            # Create visualization
            create_gt_pred_comparison(
                original_image,
                gt_orig,
                pred_orig,
                errors,
                mre,
                image_names[0],
                output_dir
            )
            
            samples_processed += 1
    
    print(f"\n✓ Created {samples_processed} visualizations")
    print(f"✓ Saved to: {output_dir}\n")
    
    # Create a grid visualization of multiple samples
    create_grid_visualization(output_dir, num_samples=min(16, samples_processed))


def create_gt_pred_comparison(image, gt_landmarks, pred_landmarks, errors, mre, image_name, output_dir):
    """Create a detailed GT vs prediction comparison."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define colors and labels
    colors = {
        'gt': 'lime',
        'pred': 'red',
        'error': 'yellow'
    }
    
    landmark_labels = ['OFD1', 'OFD2', 'BPD1', 'BPD2']
    landmark_colors = ['cyan', 'cyan', 'magenta', 'magenta']
    
    # 1. Ground Truth Only
    ax1 = axes[0]
    ax1.imshow(image)
    
    # Draw GT landmarks
    for i, (lm, label, color) in enumerate(zip(gt_landmarks, landmark_labels, landmark_colors)):
        ax1.scatter(lm[0], lm[1], c=colors['gt'], s=200, marker='o', 
                   edgecolors='black', linewidths=2, label='GT' if i == 0 else '')
        ax1.text(lm[0], lm[1]-15, label, color='white', fontsize=10, 
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    # Draw BPD and OFD lines
    ax1.plot([gt_landmarks[0, 0], gt_landmarks[1, 0]], 
            [gt_landmarks[0, 1], gt_landmarks[1, 1]], 
            'lime', linewidth=3, label='OFD')
    ax1.plot([gt_landmarks[2, 0], gt_landmarks[3, 0]], 
            [gt_landmarks[2, 1], gt_landmarks[3, 1]], 
            'lime', linewidth=3, linestyle='--', label='BPD')
    
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.axis('off')
    
    # 2. Prediction Only
    ax2 = axes[1]
    ax2.imshow(image)
    
    # Draw predicted landmarks
    for i, (lm, label, color, err) in enumerate(zip(pred_landmarks, landmark_labels, landmark_colors, errors)):
        ax2.scatter(lm[0], lm[1], c=colors['pred'], s=200, marker='x', 
                   linewidths=3, label='Pred' if i == 0 else '')
        ax2.text(lm[0], lm[1]-15, f'{label}\n({err:.1f}px)', color='white', fontsize=9, 
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
    
    # Draw BPD and OFD lines
    ax2.plot([pred_landmarks[0, 0], pred_landmarks[1, 0]], 
            [pred_landmarks[0, 1], pred_landmarks[1, 1]], 
            'red', linewidth=3, label='OFD')
    ax2.plot([pred_landmarks[2, 0], pred_landmarks[3, 0]], 
            [pred_landmarks[2, 1], pred_landmarks[3, 1]], 
            'red', linewidth=3, linestyle='--', label='BPD')
    
    ax2.set_title(f'Prediction (MRE: {mre:.1f}px)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axis('off')
    
    # 3. Overlay with Error Visualization
    ax3 = axes[2]
    ax3.imshow(image)
    
    # Draw error lines first
    for i, (gt_lm, pred_lm) in enumerate(zip(gt_landmarks, pred_landmarks)):
        ax3.plot([gt_lm[0], pred_lm[0]], [gt_lm[1], pred_lm[1]], 
                colors['error'], linewidth=2, alpha=0.8)
    
    # Draw GT landmarks
    for i, lm in enumerate(gt_landmarks):
        ax3.scatter(lm[0], lm[1], c=colors['gt'], s=150, marker='o', 
                   edgecolors='black', linewidths=2, label='GT' if i == 0 else '')
    
    # Draw predicted landmarks
    for i, lm in enumerate(pred_landmarks):
        ax3.scatter(lm[0], lm[1], c=colors['pred'], s=150, marker='x', 
                   linewidths=3, label='Pred' if i == 0 else '')
    
    # Draw BPD and OFD lines for GT
    ax3.plot([gt_landmarks[0, 0], gt_landmarks[1, 0]], 
            [gt_landmarks[0, 1], gt_landmarks[1, 1]], 
            'lime', linewidth=2, alpha=0.6)
    ax3.plot([gt_landmarks[2, 0], gt_landmarks[3, 0]], 
            [gt_landmarks[2, 1], gt_landmarks[3, 1]], 
            'lime', linewidth=2, linestyle='--', alpha=0.6)
    
    # Draw BPD and OFD lines for prediction
    ax3.plot([pred_landmarks[0, 0], pred_landmarks[1, 0]], 
            [pred_landmarks[0, 1], pred_landmarks[1, 1]], 
            'red', linewidth=2, alpha=0.6)
    ax3.plot([pred_landmarks[2, 0], pred_landmarks[3, 0]], 
            [pred_landmarks[2, 1], pred_landmarks[3, 1]], 
            'red', linewidth=2, linestyle='--', alpha=0.6)
    
    ax3.set_title('GT vs Prediction Overlay', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.axis('off')
    
    # Overall title
    clean_name = Path(image_name).stem
    fig.suptitle(f'{clean_name} - Mean Error: {mre:.2f} pixels', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{clean_name}_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()


def create_grid_visualization(output_dir, num_samples=16):
    """Create a grid showing multiple samples."""
    
    print("\nCreating grid visualization...")
    
    # Get all comparison images
    image_files = sorted(output_dir.glob("*_comparison.png"))[:num_samples]
    
    if len(image_files) == 0:
        print("No comparison images found!")
        return
    
    # Calculate grid size
    n_cols = 4
    n_rows = (len(image_files) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_file in enumerate(image_files):
        row = idx // n_cols
        col = idx % n_cols
        
        img = plt.imread(str(img_file))
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(img_file.stem.replace('_comparison', ''), fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(image_files), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('GT vs Prediction Comparison Grid', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    grid_path = output_dir / "comparison_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Grid visualization saved to: {grid_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize GT vs predictions on images')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--num_samples', type=int, default=30,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save outputs (default: results/gt_vs_predictions)')
    
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
    visualize_predictions_on_images(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

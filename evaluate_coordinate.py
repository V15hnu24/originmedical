"""
Evaluation script for coordinate regression landmark detection model

Usage:
    python evaluate_coordinate.py --checkpoint checkpoints/coordinate/best/coordinate_efficientnet_b3_best.ckpt
"""
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).parent))

import config
from data.dataset import create_dataloaders
from train_landmark import LandmarkDetectionModule
from utils.metrics import compute_batch_metrics


def evaluate_model(checkpoint_path, save_visualizations=False, output_dir=None):
    """
    Evaluate a trained coordinate regression model on validation data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        save_visualizations: Whether to save prediction visualizations
        output_dir: Directory to save outputs
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Coordinate Regression Model")
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
        batch_size=1,  # Use batch size 1 for evaluation
        num_workers=config.NUM_WORKERS,
        train_val_split=config.TRAIN_VAL_SPLIT,
        random_seed=config.RANDOM_SEED,
        image_size=config.IMAGE_SIZE,
        mode='landmark',  # landmark mode for coordinate regression
    )
    print(f"✓ Loaded {len(val_loader.dataset)} validation samples\n")
    
    # Initialize metrics storage
    all_errors = []  # Per-landmark errors
    all_mre = []  # Mean radial error per image
    all_bpd_errors = []  # BPD landmark errors
    all_ofd_errors = []  # OFD landmark errors
    
    # For detailed results
    detailed_results = []
    
    # Create output directory for visualizations
    if save_visualizations:
        if output_dir is None:
            output_dir = config.RESULTS_DIR / "coordinate_evaluation"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {output_dir}\n")
    
    # Evaluate
    print("Evaluating on validation set...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Processing")):
            # Move to device
            images = batch['image'].to(device)
            gt_landmarks = batch['landmarks'].to(device)
            image_names = batch['image_name']
            
            # Forward pass
            outputs = model(images)
            pred_landmarks = outputs['coordinates']
            
            # Compute metrics
            metrics = compute_batch_metrics(
                pred_landmarks,
                gt_landmarks,
                metric_type='landmark'
            )
            
            # Store per-image MRE
            mre = metrics['mre_overall_px']
            all_mre.append(mre)
            
            # Per-landmark errors (4 landmarks)
            pred_np = pred_landmarks[0].cpu().numpy()
            gt_np = gt_landmarks[0].cpu().numpy()
            
            landmark_errors = []
            for lm_idx in range(4):
                error = np.linalg.norm(pred_np[lm_idx] - gt_np[lm_idx])
                landmark_errors.append(error)
                all_errors.append(error)
                
                # Separate BPD (landmarks 0,1) and OFD (landmarks 2,3)
                if lm_idx < 2:
                    all_bpd_errors.append(error)
                else:
                    all_ofd_errors.append(error)
            
            # Store detailed results
            detailed_results.append({
                'image_name': image_names[0],
                'mre': mre,
                'landmark_0_error': landmark_errors[0],
                'landmark_1_error': landmark_errors[1],
                'landmark_2_error': landmark_errors[2],
                'landmark_3_error': landmark_errors[3],
                'bpd_mre': np.mean(landmark_errors[:2]),
                'ofd_mre': np.mean(landmark_errors[2:]),
            })
            
            # Save visualization
            if save_visualizations and idx < 20:  # Save first 20 samples
                save_prediction_visualization(
                    images[0].cpu(),
                    gt_landmarks[0].cpu(),
                    pred_landmarks[0].cpu(),
                    image_names[0],
                    output_dir,
                    mre
                )
    
    # Compute aggregate statistics
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Overall Metrics:")
    print(f"  Mean Radial Error (MRE):    {np.mean(all_mre):.2f} ± {np.std(all_mre):.2f} pixels")
    print(f"  Median Radial Error:        {np.median(all_mre):.2f} pixels")
    print(f"  Min Error:                  {np.min(all_mre):.2f} pixels")
    print(f"  Max Error:                  {np.max(all_mre):.2f} pixels")
    
    print(f"\nPer-Landmark Statistics:")
    print(f"  All Landmarks:              {np.mean(all_errors):.2f} ± {np.std(all_errors):.2f} pixels")
    print(f"  BPD Landmarks (0,1):        {np.mean(all_bpd_errors):.2f} ± {np.std(all_bpd_errors):.2f} pixels")
    print(f"  OFD Landmarks (2,3):        {np.mean(all_ofd_errors):.2f} ± {np.std(all_ofd_errors):.2f} pixels")
    
    # Accuracy at different thresholds
    print(f"\nAccuracy at Different Thresholds:")
    for threshold in [2.5, 5.0, 10.0, 15.0]:
        accuracy = (np.array(all_mre) < threshold).mean() * 100
        print(f"  < {threshold:5.1f} pixels:            {accuracy:.1f}%")
    
    print(f"\n{'='*60}\n")
    
    # Save detailed results to CSV
    if output_dir:
        results_df = pd.DataFrame(detailed_results)
        csv_path = output_dir / "evaluation_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Detailed results saved to: {csv_path}")
        
        # Save summary statistics
        summary = {
            'metric': [
                'mre_mean', 'mre_std', 'mre_median',
                'bpd_mean', 'bpd_std',
                'ofd_mean', 'ofd_std',
                'accuracy_2.5px', 'accuracy_5px', 'accuracy_10px', 'accuracy_15px'
            ],
            'value': [
                np.mean(all_mre), np.std(all_mre), np.median(all_mre),
                np.mean(all_bpd_errors), np.std(all_bpd_errors),
                np.mean(all_ofd_errors), np.std(all_ofd_errors),
                (np.array(all_mre) < 2.5).mean() * 100,
                (np.array(all_mre) < 5.0).mean() * 100,
                (np.array(all_mre) < 10.0).mean() * 100,
                (np.array(all_mre) < 15.0).mean() * 100,
            ]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_path = output_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Summary statistics saved to: {summary_path}\n")
    
    return {
        'mre_mean': np.mean(all_mre),
        'mre_std': np.std(all_mre),
        'bpd_mean': np.mean(all_bpd_errors),
        'ofd_mean': np.mean(all_ofd_errors),
    }


def save_prediction_visualization(image, gt_landmarks, pred_landmarks, image_name, output_dir, mre):
    """Save a visualization of the prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Denormalize image for visualization
    image_np = image.permute(1, 2, 0).numpy()
    mean = np.array(config.NORMALIZE_MEAN)
    std = np.array(config.NORMALIZE_STD)
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0, 1)
    
    gt_lm_np = gt_landmarks.numpy()
    pred_lm_np = pred_landmarks.numpy()
    
    # Plot ground truth
    axes[0].imshow(image_np)
    axes[0].scatter(gt_lm_np[:, 0], gt_lm_np[:, 1], c='green', s=100, marker='o', label='Ground Truth')
    # Draw lines for BPD and OFD
    axes[0].plot([gt_lm_np[0, 0], gt_lm_np[1, 0]], [gt_lm_np[0, 1], gt_lm_np[1, 1]], 'g-', linewidth=2, label='BPD')
    axes[0].plot([gt_lm_np[2, 0], gt_lm_np[3, 0]], [gt_lm_np[2, 1], gt_lm_np[3, 1]], 'g--', linewidth=2, label='OFD')
    axes[0].set_title('Ground Truth')
    axes[0].legend()
    axes[0].axis('off')
    
    # Plot prediction
    axes[1].imshow(image_np)
    axes[1].scatter(pred_lm_np[:, 0], pred_lm_np[:, 1], c='red', s=100, marker='x', label='Prediction')
    axes[1].scatter(gt_lm_np[:, 0], gt_lm_np[:, 1], c='green', s=50, marker='o', alpha=0.5, label='Ground Truth')
    # Draw lines
    axes[1].plot([pred_lm_np[0, 0], pred_lm_np[1, 0]], [pred_lm_np[0, 1], pred_lm_np[1, 1]], 'r-', linewidth=2, label='BPD (pred)')
    axes[1].plot([pred_lm_np[2, 0], pred_lm_np[3, 0]], [pred_lm_np[2, 1], pred_lm_np[3, 1]], 'r--', linewidth=2, label='OFD (pred)')
    # Draw error lines
    for i in range(4):
        axes[1].plot([gt_lm_np[i, 0], pred_lm_np[i, 0]], 
                     [gt_lm_np[i, 1], pred_lm_np[i, 1]], 
                     'yellow', linewidth=1, alpha=0.7)
    axes[1].set_title(f'Prediction (MRE: {mre:.2f} px)')
    axes[1].legend()
    axes[1].axis('off')
    
    plt.tight_layout()
    # Clean image name for filename (remove extension and special chars)
    clean_name = Path(image_name).stem
    plt.savefig(output_dir / f"{clean_name}_prediction.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate coordinate regression model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save prediction visualizations')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save outputs (default: results/coordinate_evaluation)')
    
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
    
    # Run evaluation
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        save_visualizations=args.save_visualizations,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

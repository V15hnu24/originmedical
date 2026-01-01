"""
Evaluation script for segmentation models (Part B)

Usage:
    python evaluate_segmentation.py --checkpoint checkpoints/unet/unet_resnet34_best.ckpt
    python evaluate_segmentation.py --checkpoint path/to/model.ckpt --save_visualizations
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
from train_segmentation import SegmentationModule
from utils.metrics import compute_batch_metrics
from utils.ellipse_fitting import EllipseFitter


def evaluate_model(checkpoint_path, save_visualizations=False, output_dir=None):
    """
    Evaluate a trained segmentation model on validation data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        save_visualizations: Whether to save prediction visualizations
        output_dir: Directory to save outputs
    """
    print(f"\n{'='*60}")
    print(f"Evaluating Segmentation Model")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = SegmentationModule.load_from_checkpoint(checkpoint_path)
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
        mode='segmentation',
    )
    print(f"✓ Loaded {len(val_loader.dataset)} validation samples\n")
    
    # Initialize metrics
    all_dice_scores = []
    all_iou_scores = []
    all_precision_scores = []
    all_recall_scores = []
    all_landmark_errors = []
    
    # Ellipse fitter for landmark extraction
    ellipse_fitter = EllipseFitter(method='ransac')
    
    # Create output directory for visualizations
    if save_visualizations:
        if output_dir is None:
            output_dir = config.RESULTS_DIR / "segmentation_evaluation"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {output_dir}\n")
    
    # Evaluate
    print("Evaluating on validation set...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Processing")):
            # Move to device
            images = batch['image'].to(device)
            gt_masks = batch['mask'].to(device)
            image_ids = batch['image_id']
            
            # Forward pass
            pred_masks = model(images)
            pred_masks_sigmoid = torch.sigmoid(pred_masks)
            
            # Compute metrics
            metrics = compute_batch_metrics(
                pred_masks_sigmoid,
                gt_masks,
                metric_type='segmentation'
            )
            
            all_dice_scores.append(metrics['dice'].item())
            all_iou_scores.append(metrics['iou'].item())
            all_precision_scores.append(metrics['precision'].item())
            all_recall_scores.append(metrics['recall'].item())
            
            # Extract landmarks if available
            if 'landmarks' in batch:
                gt_landmarks = batch['landmarks'][0].numpy()
                pred_mask_np = pred_masks_sigmoid[0, 0].cpu().numpy()
                
                result = ellipse_fitter.mask_to_landmarks(pred_mask_np)
                if result is not None:
                    pred_landmarks = result['landmarks']
                    # Compute mean radial error
                    errors = np.linalg.norm(pred_landmarks - gt_landmarks, axis=1)
                    mre = errors.mean()
                    all_landmark_errors.append(mre)
            
            # Save visualization
            if save_visualizations and idx < 20:  # Save first 20 samples
                save_prediction_visualization(
                    images[0].cpu(),
                    gt_masks[0].cpu(),
                    pred_masks_sigmoid[0].cpu(),
                    image_ids[0],
                    output_dir,
                    metrics
                )
    
    # Compute aggregate statistics
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Segmentation Metrics:")
    print(f"  Dice Score:  {np.mean(all_dice_scores):.4f} ± {np.std(all_dice_scores):.4f}")
    print(f"  IoU:         {np.mean(all_iou_scores):.4f} ± {np.std(all_iou_scores):.4f}")
    print(f"  Precision:   {np.mean(all_precision_scores):.4f} ± {np.std(all_precision_scores):.4f}")
    print(f"  Recall:      {np.mean(all_recall_scores):.4f} ± {np.std(all_recall_scores):.4f}")
    
    if all_landmark_errors:
        print(f"\nLandmark Detection (from segmentation):")
        print(f"  Mean Radial Error: {np.mean(all_landmark_errors):.2f} ± {np.std(all_landmark_errors):.2f} pixels")
    
    print(f"\n{'='*60}\n")
    
    # Save detailed results to CSV
    if output_dir:
        results_df = pd.DataFrame({
            'dice': all_dice_scores,
            'iou': all_iou_scores,
            'precision': all_precision_scores,
            'recall': all_recall_scores,
        })
        
        if all_landmark_errors:
            # Pad with NaN if some samples failed landmark extraction
            landmark_errors_padded = all_landmark_errors + [np.nan] * (len(all_dice_scores) - len(all_landmark_errors))
            results_df['landmark_mre'] = landmark_errors_padded
        
        csv_path = output_dir / "evaluation_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Detailed results saved to: {csv_path}")
        
        # Save summary statistics
        summary = {
            'metric': ['dice', 'iou', 'precision', 'recall'],
            'mean': [
                np.mean(all_dice_scores),
                np.mean(all_iou_scores),
                np.mean(all_precision_scores),
                np.mean(all_recall_scores),
            ],
            'std': [
                np.std(all_dice_scores),
                np.std(all_iou_scores),
                np.std(all_precision_scores),
                np.std(all_recall_scores),
            ]
        }
        
        if all_landmark_errors:
            summary['metric'].append('landmark_mre')
            summary['mean'].append(np.mean(all_landmark_errors))
            summary['std'].append(np.std(all_landmark_errors))
        
        summary_df = pd.DataFrame(summary)
        summary_path = output_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Summary statistics saved to: {summary_path}\n")
    
    return {
        'dice': np.mean(all_dice_scores),
        'iou': np.mean(all_iou_scores),
        'precision': np.mean(all_precision_scores),
        'recall': np.mean(all_recall_scores),
        'landmark_mre': np.mean(all_landmark_errors) if all_landmark_errors else None
    }


def save_prediction_visualization(image, gt_mask, pred_mask, image_id, output_dir, metrics):
    """Save a visualization of the prediction."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Denormalize image for visualization
    image_np = image.permute(1, 2, 0).numpy()
    mean = np.array(config.NORMALIZE_MEAN)
    std = np.array(config.NORMALIZE_STD)
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0, 1)
    
    gt_mask_np = gt_mask[0].numpy()
    pred_mask_np = pred_mask[0].numpy()
    pred_mask_binary = (pred_mask_np > 0.5).astype(np.float32)
    
    # Plot original image
    axes[0].imshow(image_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Plot ground truth
    axes[1].imshow(image_np)
    axes[1].imshow(gt_mask_np, alpha=0.5, cmap='jet')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot prediction
    axes[2].imshow(image_np)
    axes[2].imshow(pred_mask_np, alpha=0.5, cmap='jet')
    axes[2].set_title(f'Prediction (Dice: {metrics["dice"]:.3f})')
    axes[2].axis('off')
    
    # Plot overlay
    axes[3].imshow(image_np)
    axes[3].contour(gt_mask_np, colors='g', linewidths=2, levels=[0.5])
    axes[3].contour(pred_mask_binary, colors='r', linewidths=2, levels=[0.5])
    axes[3].set_title('Overlay (GT=green, Pred=red)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{image_id}_prediction.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--save_visualizations', action='store_true',
                       help='Save prediction visualizations')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save outputs (default: results/segmentation_evaluation)')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = config.CHECKPOINT_DIR
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

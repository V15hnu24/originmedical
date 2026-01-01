"""
Debug script to properly handle inference and visualization pipeline

Steps:
1. Load original image
2. Preprocess and resize to 512x512
3. Run inference -> get predictions in 512x512 space
4. Scale predictions back to original image dimensions
5. Load GT from CSV (in original space)
6. Visualize on original image
"""
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(str(Path(__file__).parent))

import config
from train_landmark import LandmarkDetectionModule
from data.preprocessing import UltrasoundPreprocessor


def inference_and_visualize(checkpoint_path, num_samples=30, output_dir=None):
    """
    Complete inference and visualization pipeline.
    """
    print(f"\n{'='*70}")
    print(f"DEBUG: Complete Inference and Visualization Pipeline")
    print(f"{'='*70}\n")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = LandmarkDetectionModule.load_from_checkpoint(checkpoint_path)
        model.to(device)
        model.eval()
        print(f"✓ Model loaded on {device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Load validation split CSV
    val_csv_path = Path(config.CSV_PATH).parent / 'val_split.csv'
    val_df = pd.read_csv(val_csv_path)
    print(f"\n✓ Loaded validation split: {len(val_df)} samples")
    
    # Create preprocessing pipeline (same as during training)
    preprocessor = UltrasoundPreprocessor(
        use_clahe=True,
        denoise=False,
        normalize=False,
    )
    
    # Create transform (inference mode - no augmentation)
    transform = A.Compose([
        A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ToTensorV2(),
    ])
    
    # Create output directory
    if output_dir is None:
        output_dir = config.RESULTS_DIR / "unnet_e48" / "visualise"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {output_dir}\n")
    
    print(f"Processing {num_samples} samples...\n")
    
    # Process samples
    for idx in tqdm(range(min(num_samples, len(val_df))), desc="Inference & Visualization"):
        row = val_df.iloc[idx]
        image_name = row['image_name']
        
        # Step 1: Load ORIGINAL image
        image_path = config.DATA_DIR / image_name
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            print(f"Warning: Could not load {image_path}")
            continue
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = original_image.shape[:2]
        
        print(f"\n[{idx+1}] {image_name}")
        print(f"  Original image size: {orig_w}x{orig_h}")
        
        # Step 2: Get GT landmarks from CSV (in ORIGINAL image space)
        gt_landmarks_orig = np.array([
            [row['ofd_1_x'], row['ofd_1_y']],
            [row['ofd_2_x'], row['ofd_2_y']],
            [row['bpd_1_x'], row['bpd_1_y']],
            [row['bpd_2_x'], row['bpd_2_y']],
        ], dtype=np.float32)
        
        print(f"  GT landmarks (original space):")
        print(f"    OFD1: ({gt_landmarks_orig[0,0]:.1f}, {gt_landmarks_orig[0,1]:.1f})")
        print(f"    OFD2: ({gt_landmarks_orig[1,0]:.1f}, {gt_landmarks_orig[1,1]:.1f})")
        print(f"    BPD1: ({gt_landmarks_orig[2,0]:.1f}, {gt_landmarks_orig[2,1]:.1f})")
        print(f"    BPD2: ({gt_landmarks_orig[3,0]:.1f}, {gt_landmarks_orig[3,1]:.1f})")
        
        # Step 3: Preprocess and resize for model input
        preprocessed_image = preprocessor.preprocess(original_image)
        
        # Apply resize and normalization
        transformed = transform(image=preprocessed_image)
        resized_image = transformed['image']
        
        print(f"  Resized for model: {config.IMAGE_SIZE[1]}x{config.IMAGE_SIZE[0]}")
        
        # Step 4: Run inference
        with torch.no_grad():
            input_tensor = resized_image.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            pred_landmarks_512 = outputs['coordinates'][0].cpu().numpy()
        
        print(f"  Predictions (512x512 space):")
        print(f"    OFD1: ({pred_landmarks_512[0,0]:.1f}, {pred_landmarks_512[0,1]:.1f})")
        print(f"    OFD2: ({pred_landmarks_512[1,0]:.1f}, {pred_landmarks_512[1,1]:.1f})")
        print(f"    BPD1: ({pred_landmarks_512[2,0]:.1f}, {pred_landmarks_512[2,1]:.1f})")
        print(f"    BPD2: ({pred_landmarks_512[3,0]:.1f}, {pred_landmarks_512[3,1]:.1f})")
        
        # Step 5: Scale predictions from 512x512 to original image size
        scale_x = orig_w / config.IMAGE_SIZE[1]
        scale_y = orig_h / config.IMAGE_SIZE[0]
        
        print(f"  Scaling factors: x={scale_x:.4f}, y={scale_y:.4f}")
        
        pred_landmarks_orig = pred_landmarks_512.copy()
        pred_landmarks_orig[:, 0] = pred_landmarks_512[:, 0] * scale_x
        pred_landmarks_orig[:, 1] = pred_landmarks_512[:, 1] * scale_y
        
        print(f"  Predictions (original space after scaling):")
        print(f"    OFD1: ({pred_landmarks_orig[0,0]:.1f}, {pred_landmarks_orig[0,1]:.1f})")
        print(f"    OFD2: ({pred_landmarks_orig[1,0]:.1f}, {pred_landmarks_orig[1,1]:.1f})")
        print(f"    BPD1: ({pred_landmarks_orig[2,0]:.1f}, {pred_landmarks_orig[2,1]:.1f})")
        print(f"    BPD2: ({pred_landmarks_orig[3,0]:.1f}, {pred_landmarks_orig[3,1]:.1f})")
        
        # Step 6: Compute errors
        errors = np.linalg.norm(pred_landmarks_orig - gt_landmarks_orig, axis=1)
        mre = errors.mean()
        
        print(f"  Errors per landmark: [{errors[0]:.1f}, {errors[1]:.1f}, {errors[2]:.1f}, {errors[3]:.1f}] pixels")
        print(f"  Mean Radial Error: {mre:.2f} pixels")
        
        # Step 7: Visualize
        visualize_result(
            original_image,
            gt_landmarks_orig,
            pred_landmarks_orig,
            errors,
            mre,
            image_name,
            output_dir
        )
    
    print(f"\n\n✓ All visualizations saved to: {output_dir}")


def visualize_result(image, gt_landmarks, pred_landmarks, errors, mre, image_name, output_dir):
    """Create visualization with GT (green) and predictions (blue)."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Display original image
    ax.imshow(image)
    
    # Colors
    gt_color = 'lime'
    pred_color = 'blue'
    
    # Draw GT lines
    # OFD line
    ax.plot([gt_landmarks[0, 0], gt_landmarks[1, 0]], 
            [gt_landmarks[0, 1], gt_landmarks[1, 1]], 
            color=gt_color, linewidth=3, label='GT OFD', linestyle='-')
    # BPD line
    ax.plot([gt_landmarks[2, 0], gt_landmarks[3, 0]], 
            [gt_landmarks[2, 1], gt_landmarks[3, 1]], 
            color=gt_color, linewidth=3, label='GT BPD', linestyle='--')
    
    # Draw GT landmarks
    for i, lm in enumerate(gt_landmarks):
        ax.scatter(lm[0], lm[1], c=gt_color, s=150, marker='o', 
                  edgecolors='white', linewidths=2, 
                  label='GT Landmarks' if i == 0 else '', zorder=5)
    
    # Draw Prediction lines
    # OFD line
    ax.plot([pred_landmarks[0, 0], pred_landmarks[1, 0]], 
            [pred_landmarks[0, 1], pred_landmarks[1, 1]], 
            color=pred_color, linewidth=3, label='Pred OFD', linestyle='-', alpha=0.8)
    # BPD line
    ax.plot([pred_landmarks[2, 0], pred_landmarks[3, 0]], 
            [pred_landmarks[2, 1], pred_landmarks[3, 1]], 
            color=pred_color, linewidth=3, label='Pred BPD', linestyle='--', alpha=0.8)
    
    # Draw Prediction landmarks
    for i, (lm, err) in enumerate(zip(pred_landmarks, errors)):
        ax.scatter(lm[0], lm[1], c=pred_color, s=150, marker='x', 
                  linewidths=3, 
                  label='Pred Landmarks' if i == 0 else '', zorder=5)
    
    # Add title
    clean_name = Path(image_name).stem
    ax.set_title(f'{clean_name} - MRE: {mre:.2f} pixels\n' +
                 f'Errors: [{errors[0]:.1f}, {errors[1]:.1f}, {errors[2]:.1f}, {errors[3]:.1f}] px', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{clean_name}.png", dpi=200, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Debug inference and visualization pipeline')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=30,
                       help='Number of samples to process')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    inference_and_visualize(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

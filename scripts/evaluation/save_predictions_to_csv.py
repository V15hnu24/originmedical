"""
Analyze GT format and save predictions vs GT to CSV

This script will:
1. Load validation split
2. Check original image dimensions
3. Understand GT coordinate space
4. Run inference and scale predictions
5. Save detailed CSV with GT and predictions
"""
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(str(Path(__file__).parent))

import config
from train_landmark import LandmarkDetectionModule
from data.preprocessing import UltrasoundPreprocessor


def analyze_and_save_predictions(checkpoint_path, output_csv=None):
    """
    Analyze GT format and save predictions to CSV.
    """
    print(f"\n{'='*70}")
    print(f"Analyzing GT Format and Saving Predictions")
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
    
    # Analyze first few samples to understand GT format
    print(f"\n{'='*70}")
    print("ANALYZING GT FORMAT")
    print(f"{'='*70}\n")
    
    for idx in range(min(5, len(val_df))):
        row = val_df.iloc[idx]
        image_name = row['image_name']
        
        # Load original image
        image_path = config.DATA_DIR / image_name
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            continue
        orig_h, orig_w = original_image.shape[:2]
        
        # Get GT landmarks from CSV
        gt_ofd1 = (row['ofd_1_x'], row['ofd_1_y'])
        gt_ofd2 = (row['ofd_2_x'], row['ofd_2_y'])
        gt_bpd1 = (row['bpd_1_x'], row['bpd_1_y'])
        gt_bpd2 = (row['bpd_2_x'], row['bpd_2_y'])
        
        print(f"[{idx+1}] {image_name}")
        print(f"  Original image dimensions: {orig_w} x {orig_h}")
        print(f"  GT coordinates from CSV:")
        print(f"    OFD1: ({gt_ofd1[0]:.1f}, {gt_ofd1[1]:.1f})")
        print(f"    OFD2: ({gt_ofd2[0]:.1f}, {gt_ofd2[1]:.1f})")
        print(f"    BPD1: ({gt_bpd1[0]:.1f}, {gt_bpd1[1]:.1f})")
        print(f"    BPD2: ({gt_bpd2[0]:.1f}, {gt_bpd2[1]:.1f})")
        
        # Check if GT is in original space or normalized
        max_x = max(gt_ofd1[0], gt_ofd2[0], gt_bpd1[0], gt_bpd2[0])
        max_y = max(gt_ofd1[1], gt_ofd2[1], gt_bpd1[1], gt_bpd2[1])
        
        if max_x > orig_w or max_y > orig_h:
            print(f"  ⚠️  WARNING: GT coordinates exceed image dimensions!")
        elif max_x > 10 and max_y > 10:
            print(f"  ✓ GT appears to be in ORIGINAL image space (pixel coordinates)")
        else:
            print(f"  ⚠️  GT might be normalized (0-1 range)")
        print()
    
    # Create preprocessing pipeline
    preprocessor = UltrasoundPreprocessor(
        use_clahe=True,
        denoise=False,
        normalize=False,
    )
    
    transform = A.Compose([
        A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        A.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
        ToTensorV2(),
    ])
    
    print(f"\n{'='*70}")
    print("RUNNING INFERENCE ON ALL VALIDATION SAMPLES")
    print(f"{'='*70}\n")
    
    # Prepare results list
    results = []
    
    # Process all validation samples
    for idx in tqdm(range(len(val_df)), desc="Processing"):
        row = val_df.iloc[idx]
        image_name = row['image_name']
        
        # Load original image
        image_path = config.DATA_DIR / image_name
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            print(f"Warning: Could not load {image_path}")
            continue
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = original_image.shape[:2]
        
        # Get GT landmarks from CSV (in original image space)
        gt_landmarks_orig = np.array([
            [row['ofd_1_x'], row['ofd_1_y']],
            [row['ofd_2_x'], row['ofd_2_y']],
            [row['bpd_1_x'], row['bpd_1_y']],
            [row['bpd_2_x'], row['bpd_2_y']],
        ], dtype=np.float32)
        
        # Preprocess and resize for model
        preprocessed_image = preprocessor.preprocess(original_image)
        transformed = transform(image=preprocessed_image)
        resized_image = transformed['image']
        
        # Run inference
        with torch.no_grad():
            input_tensor = resized_image.unsqueeze(0).to(device)
            outputs = model(input_tensor)
            pred_landmarks_512 = outputs['coordinates'][0].cpu().numpy()
        
        # Scale predictions from 512x512 to original image size
        scale_x = orig_w / config.IMAGE_SIZE[1]
        scale_y = orig_h / config.IMAGE_SIZE[0]
        
        pred_landmarks_orig = pred_landmarks_512.copy()
        pred_landmarks_orig[:, 0] = pred_landmarks_512[:, 0] * scale_x
        pred_landmarks_orig[:, 1] = pred_landmarks_512[:, 1] * scale_y
        
        # Compute errors
        errors = np.linalg.norm(pred_landmarks_orig - gt_landmarks_orig, axis=1)
        mre = errors.mean()
        
        # Store results
        result = {
            'image_name': image_name,
            'orig_width': orig_w,
            'orig_height': orig_h,
            # GT in original space
            'gt_ofd1_x': gt_landmarks_orig[0, 0],
            'gt_ofd1_y': gt_landmarks_orig[0, 1],
            'gt_ofd2_x': gt_landmarks_orig[1, 0],
            'gt_ofd2_y': gt_landmarks_orig[1, 1],
            'gt_bpd1_x': gt_landmarks_orig[2, 0],
            'gt_bpd1_y': gt_landmarks_orig[2, 1],
            'gt_bpd2_x': gt_landmarks_orig[3, 0],
            'gt_bpd2_y': gt_landmarks_orig[3, 1],
            # Predictions in 512x512 space
            'pred_512_ofd1_x': pred_landmarks_512[0, 0],
            'pred_512_ofd1_y': pred_landmarks_512[0, 1],
            'pred_512_ofd2_x': pred_landmarks_512[1, 0],
            'pred_512_ofd2_y': pred_landmarks_512[1, 1],
            'pred_512_bpd1_x': pred_landmarks_512[2, 0],
            'pred_512_bpd1_y': pred_landmarks_512[2, 1],
            'pred_512_bpd2_x': pred_landmarks_512[3, 0],
            'pred_512_bpd2_y': pred_landmarks_512[3, 1],
            # Predictions scaled to original space
            'pred_orig_ofd1_x': pred_landmarks_orig[0, 0],
            'pred_orig_ofd1_y': pred_landmarks_orig[0, 1],
            'pred_orig_ofd2_x': pred_landmarks_orig[1, 0],
            'pred_orig_ofd2_y': pred_landmarks_orig[1, 1],
            'pred_orig_bpd1_x': pred_landmarks_orig[2, 0],
            'pred_orig_bpd1_y': pred_landmarks_orig[2, 1],
            'pred_orig_bpd2_x': pred_landmarks_orig[3, 0],
            'pred_orig_bpd2_y': pred_landmarks_orig[3, 1],
            # Errors
            'error_ofd1': errors[0],
            'error_ofd2': errors[1],
            'error_bpd1': errors[2],
            'error_bpd2': errors[3],
            'mre': mre,
            # Scaling factors
            'scale_x': scale_x,
            'scale_y': scale_y,
        }
        
        results.append(result)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    if output_csv is None:
        output_csv = config.RESULTS_DIR / "predictions_vs_gt.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    print(f"Total samples processed: {len(results_df)}")
    print(f"✓ Results saved to: {output_csv}")
    print(f"\nOverall Statistics:")
    print(f"  Mean Radial Error: {results_df['mre'].mean():.2f} ± {results_df['mre'].std():.2f} pixels")
    print(f"  Median Error: {results_df['mre'].median():.2f} pixels")
    print(f"  Min Error: {results_df['mre'].min():.2f} pixels")
    print(f"  Max Error: {results_df['mre'].max():.2f} pixels")
    
    print(f"\nImage Dimensions (unique sizes):")
    unique_dims = results_df[['orig_width', 'orig_height']].drop_duplicates()
    for _, dim_row in unique_dims.iterrows():
        count = len(results_df[(results_df['orig_width'] == dim_row['orig_width']) & 
                               (results_df['orig_height'] == dim_row['orig_height'])])
        print(f"  {int(dim_row['orig_width'])} x {int(dim_row['orig_height'])}: {count} images")
    
    print(f"\nPrediction Statistics (in 512x512 space):")
    pred_512_cols = [col for col in results_df.columns if col.startswith('pred_512_')]
    print(f"  X coordinates range: [{results_df[[c for c in pred_512_cols if '_x' in c]].min().min():.1f}, "
          f"{results_df[[c for c in pred_512_cols if '_x' in c]].max().max():.1f}]")
    print(f"  Y coordinates range: [{results_df[[c for c in pred_512_cols if '_y' in c]].min().min():.1f}, "
          f"{results_df[[c for c in pred_512_cols if '_y' in c]].max().max():.1f}]")
    print(f"  Expected range for 512x512: [0, 512]")
    
    print(f"\nGT Statistics (in original space):")
    gt_cols = [col for col in results_df.columns if col.startswith('gt_')]
    print(f"  X coordinates range: [{results_df[[c for c in gt_cols if '_x' in c]].min().min():.1f}, "
          f"{results_df[[c for c in gt_cols if '_x' in c]].max().max():.1f}]")
    print(f"  Y coordinates range: [{results_df[[c for c in gt_cols if '_y' in c]].min().min():.1f}, "
          f"{results_df[[c for c in gt_cols if '_y' in c]].max().max():.1f}]")
    
    print(f"\n{'='*70}\n")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Analyze GT and save predictions to CSV')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path (default: results/predictions_vs_gt.csv)')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    analyze_and_save_predictions(
        checkpoint_path=args.checkpoint,
        output_csv=args.output
    )


if __name__ == '__main__':
    main()

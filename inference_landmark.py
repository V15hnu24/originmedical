"""
Inference script for landmark detection models.

Loads a trained checkpoint and visualizes predictions on validation/test images.

Usage:
    python inference_landmark.py --checkpoint checkpoints/heatmap/best/model.ckpt --num_samples 10
    python inference_landmark.py --checkpoint checkpoints/heatmap/best/model.ckpt --image_path images/001_HC.png
"""
import argparse
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from train_landmark import LandmarkDetectionModule
from data.dataset import FetalUltrasoundDataset
from data.preprocessing import UltrasoundPreprocessor
from data.augmentation import get_augmentation_pipeline
import config


def load_model(checkpoint_path: str):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    model = LandmarkDetectionModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    
    # Move to appropriate device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    return model, device


def inference_single_image(
    model,
    device,
    image_path: str,
    gt_landmarks: np.ndarray = None,
):
    """
    Run inference on a single image.
    
    Args:
        model: Trained model
        device: Device to run on
        image_path: Path to image
        gt_landmarks: Ground truth landmarks [4, 2] (optional)
        
    Returns:
        pred_landmarks: Predicted landmarks [4, 2]
        image: Original image
    """
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get preprocessing pipeline
    aug_pipeline = get_augmentation_pipeline(
        image_size=config.IMAGE_SIZE,
        augment=False,
        mode='landmark'
    )
    
    # Preprocess
    keypoints = gt_landmarks.tolist() if gt_landmarks is not None else [[0, 0]] * 4
    transformed = aug_pipeline(image=image, keypoints=keypoints)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
        pred_coords = outputs['coordinates'].cpu().numpy()[0]  # [4, 2]
    
    return pred_coords, image


def visualize_prediction(
    image: np.ndarray,
    pred_landmarks: np.ndarray,
    gt_landmarks: np.ndarray = None,
    save_path: str = None,
    title: str = None,
):
    """
    Visualize predictions with ground truth.
    
    Args:
        image: Original image
        pred_landmarks: Predicted landmarks [4, 2]
        gt_landmarks: Ground truth landmarks [4, 2] (optional)
        save_path: Path to save visualization
        title: Plot title
    """
    # Resize image to match prediction coordinates (512x512)
    image_resized = cv2.resize(image, config.IMAGE_SIZE)
    image_vis = image_resized.copy()
    
    # Scale landmarks to image size
    h, w = image_resized.shape[:2]
    scale_x = w / config.IMAGE_SIZE[0]
    scale_y = h / config.IMAGE_SIZE[1]
    
    # Draw ground truth (green)
    if gt_landmarks is not None:
        gt_ofd_1 = (int(gt_landmarks[0, 0] * scale_x), int(gt_landmarks[0, 1] * scale_y))
        gt_ofd_2 = (int(gt_landmarks[1, 0] * scale_x), int(gt_landmarks[1, 1] * scale_y))
        gt_bpd_1 = (int(gt_landmarks[2, 0] * scale_x), int(gt_landmarks[2, 1] * scale_y))
        gt_bpd_2 = (int(gt_landmarks[3, 0] * scale_x), int(gt_landmarks[3, 1] * scale_y))
        
        cv2.line(image_vis, gt_ofd_1, gt_ofd_2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.line(image_vis, gt_bpd_1, gt_bpd_2, (0, 255, 0), 2, cv2.LINE_AA)
        for pt in [gt_ofd_1, gt_ofd_2, gt_bpd_1, gt_bpd_2]:
            cv2.circle(image_vis, pt, 6, (0, 255, 0), -1)
            cv2.circle(image_vis, pt, 7, (255, 255, 255), 2)
    
    # Draw predictions (red)
    pred_ofd_1 = (int(pred_landmarks[0, 0] * scale_x), int(pred_landmarks[0, 1] * scale_y))
    pred_ofd_2 = (int(pred_landmarks[1, 0] * scale_x), int(pred_landmarks[1, 1] * scale_y))
    pred_bpd_1 = (int(pred_landmarks[2, 0] * scale_x), int(pred_landmarks[2, 1] * scale_y))
    pred_bpd_2 = (int(pred_landmarks[3, 0] * scale_x), int(pred_landmarks[3, 1] * scale_y))
    
    cv2.line(image_vis, pred_ofd_1, pred_ofd_2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(image_vis, pred_bpd_1, pred_bpd_2, (255, 0, 0), 2, cv2.LINE_AA)
    for pt in [pred_ofd_1, pred_ofd_2, pred_bpd_1, pred_bpd_2]:
        cv2.circle(image_vis, pt, 5, (255, 0, 0), -1)
    
    # Calculate error if GT available
    error_text = ""
    if gt_landmarks is not None:
        mre = np.mean(np.linalg.norm(pred_landmarks - gt_landmarks, axis=1))
        error_text = f"MRE: {mre:.2f} px ({mre*0.15:.2f} mm)"
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_vis)
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Predicted'),
    ]
    if gt_landmarks is not None:
        legend_elements.insert(0, Patch(facecolor='green', label='Ground Truth'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add title
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    if error_text:
        plt.suptitle(error_text, fontsize=12, color='blue')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def inference_dataset(
    model,
    device,
    csv_path: str,
    images_dir: str,
    num_samples: int = 10,
    save_dir: str = 'output/inference',
    split: str = 'val',
):
    """
    Run inference on multiple images from dataset.
    
    Args:
        model: Trained model
        device: Device
        csv_path: Path to CSV
        images_dir: Images directory
        num_samples: Number of samples to visualize
        save_dir: Directory to save results
        split: 'train' or 'val'
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Sample random images
    samples = df.sample(n=min(num_samples, len(df)))
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning inference on {len(samples)} samples...")
    
    all_errors = []
    
    for idx, (_, row) in enumerate(tqdm(samples.iterrows(), total=len(samples))):
        image_name = row['image_name']
        image_path = Path(images_dir) / image_name
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get ground truth
        gt_landmarks = np.array([
            [row['ofd_1_x'], row['ofd_1_y']],
            [row['ofd_2_x'], row['ofd_2_y']],
            [row['bpd_1_x'], row['bpd_1_y']],
            [row['bpd_2_x'], row['bpd_2_y']],
        ])
        
        # Run inference
        pred_landmarks, image = inference_single_image(
            model, device, image_path, gt_landmarks
        )
        
        # Calculate error
        mre = np.mean(np.linalg.norm(pred_landmarks - gt_landmarks, axis=1))
        all_errors.append(mre)
        
        # Visualize
        output_path = save_path / f"inference_{idx:03d}_{image_name}"
        visualize_prediction(
            image,
            pred_landmarks,
            gt_landmarks,
            save_path=str(output_path),
            title=f"{image_name}"
        )
        
        plt.close()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("INFERENCE RESULTS")
    print(f"{'='*60}")
    print(f"Samples evaluated: {len(all_errors)}")
    print(f"Mean Radial Error: {np.mean(all_errors):.2f} ± {np.std(all_errors):.2f} px")
    print(f"                   {np.mean(all_errors)*0.15:.2f} ± {np.std(all_errors)*0.15:.2f} mm")
    print(f"Min error: {np.min(all_errors):.2f} px ({np.min(all_errors)*0.15:.2f} mm)")
    print(f"Max error: {np.max(all_errors):.2f} px ({np.max(all_errors)*0.15:.2f} mm)")
    print(f"\nVisualizations saved to: {save_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Run inference with trained landmark detection model')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--csv_path',
        type=str,
        default='data/augmented/augmented_ground_truth.csv',
        help='Path to CSV with ground truth'
    )
    
    parser.add_argument(
        '--images_dir',
        type=str,
        default='data/augmented/images',
        help='Directory with images'
    )
    
    parser.add_argument(
        '--image_path',
        type=str,
        default=None,
        help='Single image path (overrides dataset mode)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10,
        help='Number of samples to visualize (dataset mode)'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output/inference',
        help='Directory to save visualizations'
    )
    
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.checkpoint)
    
    # Single image mode
    if args.image_path:
        print(f"\nRunning inference on: {args.image_path}")
        pred_landmarks, image = inference_single_image(
            model, device, args.image_path
        )
        
        save_path = Path(args.save_dir) / f"inference_{Path(args.image_path).name}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        visualize_prediction(
            image,
            pred_landmarks,
            save_path=str(save_path),
            title=Path(args.image_path).name
        )
    
    # Dataset mode
    else:
        inference_dataset(
            model,
            device,
            args.csv_path,
            args.images_dir,
            args.num_samples,
            args.save_dir
        )


if __name__ == '__main__':
    main()

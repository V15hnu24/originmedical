"""
Dataset class for fetal ultrasound biometry landmark detection
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.preprocessing import UltrasoundPreprocessor
from data.augmentation import get_augmentation_pipeline


class FetalUltrasoundDataset(Dataset):
    """
    Dataset for fetal ultrasound images with landmark annotations.
    
    Handles both landmark detection and segmentation tasks.
    """
    
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        mode: str = 'landmark',  # 'landmark' or 'segmentation'
        augment: bool = True,
        preprocessor: Optional[UltrasoundPreprocessor] = None,
        heatmap_size: Optional[Tuple[int, int]] = None,
        heatmap_sigma: float = 2.0,
    ):
        """
        Args:
            csv_path: Path to CSV file with annotations
            image_dir: Directory containing images
            image_size: Target image size (H, W)
            mode: 'landmark' for landmark detection, 'segmentation' for segmentation
            augment: Whether to apply augmentations
            preprocessor: Optional preprocessor instance
            heatmap_size: Size for heatmap generation (for heatmap-based models)
            heatmap_sigma: Sigma for Gaussian heatmap generation
        """
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.mode = mode
        self.augment = augment
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        
        # Load annotations
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        
        # Initialize preprocessor (disable normalization - Albumentations will handle it)
        self.preprocessor = preprocessor or UltrasoundPreprocessor(
            use_clahe=True,
            denoise=False,  # Albumentations handles denoising
            normalize=False,  # Albumentations handles normalization
        )
        
        # Get augmentation pipeline
        self.transform = get_augmentation_pipeline(
            image_size=image_size,
            augment=augment,
            mode=mode
        )
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary containing:
                - 'image': Preprocessed image tensor [C, H, W]
                - 'landmarks': Landmark coordinates [4, 2] (x, y)
                - 'heatmaps': Gaussian heatmaps [4, H, W] (if heatmap mode)
                - 'mask': Segmentation mask [1, H, W] (if segmentation mode)
                - 'image_name': Original image filename
        """
        row = self.df.iloc[idx]
        
        # Load image
        image_path = self.image_dir / row['image_name']
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original image size for coordinate normalization
        orig_h, orig_w = image.shape[:2]
        
        # Extract landmarks (OFD1, OFD2, BPD1, BPD2)
        landmarks = np.array([
            [row['ofd_1_x'], row['ofd_1_y']],
            [row['ofd_2_x'], row['ofd_2_y']],
            [row['bpd_1_x'], row['bpd_1_y']],
            [row['bpd_2_x'], row['bpd_2_y']],
        ], dtype=np.float32)
        
        # Apply preprocessing
        image = self.preprocessor.preprocess(image)
        
        # Prepare keypoints for albumentations
        keypoints = landmarks.tolist()
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(
                image=image,
                keypoints=keypoints
            )
            image = transformed['image']
            landmarks = np.array(transformed['keypoints'], dtype=np.float32)
            
            # Ensure we still have 4 landmarks after augmentation
            if len(landmarks) != 4:
                # If augmentation removed some keypoints, use original landmarks
                # and resize them to match the transformed image size
                landmarks = np.array(keypoints, dtype=np.float32)
                # Scale landmarks to match resized image
                scale_x = self.image_size[1] / orig_w
                scale_y = self.image_size[0] / orig_h
                landmarks[:, 0] *= scale_x
                landmarks[:, 1] *= scale_y
        
        # Ensure landmarks shape is correct [4, 2]
        assert landmarks.shape == (4, 2), f"Invalid landmarks shape: {landmarks.shape}"
        
        # Prepare output dictionary
        output = {
            'image': image,
            'landmarks': torch.from_numpy(landmarks),
            'image_name': row['image_name'],
            'orig_size': torch.tensor([orig_h, orig_w], dtype=torch.float32),
        }
        
        # Generate heatmaps if needed
        if self.heatmap_size is not None:
            heatmaps = self._generate_heatmaps(landmarks)
            output['heatmaps'] = torch.from_numpy(heatmaps)
        
        # Generate segmentation mask if needed (approximate from landmarks)
        if self.mode == 'segmentation':
            mask = self._generate_mask_from_landmarks(landmarks)
            output['mask'] = torch.from_numpy(mask)
        
        return output
    
    def _generate_heatmaps(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian heatmaps for each landmark.
        
        Args:
            landmarks: Landmark coordinates [4, 2] in image space
            
        Returns:
            Heatmaps [4, H, W] with Gaussian blobs at landmark locations
        """
        num_landmarks = len(landmarks)
        h, w = self.heatmap_size
        heatmaps = np.zeros((num_landmarks, h, w), dtype=np.float32)
        
        # Scale landmarks to heatmap size
        scale_x = w / self.image_size[1]
        scale_y = h / self.image_size[0]
        
        for i, (x, y) in enumerate(landmarks):
            # Scale coordinates
            hm_x = int(x * scale_x)
            hm_y = int(y * scale_y)
            
            # Skip if out of bounds
            if hm_x < 0 or hm_x >= w or hm_y < 0 or hm_y >= h:
                continue
            
            # Generate Gaussian heatmap
            sigma = self.heatmap_sigma
            size = 6 * sigma + 1
            x_grid = np.arange(0, size)
            y_grid = np.arange(0, size)
            xx, yy = np.meshgrid(x_grid, y_grid)
            
            # Gaussian kernel
            gaussian = np.exp(-((xx - size // 2) ** 2 + (yy - size // 2) ** 2) / (2 * sigma ** 2))
            
            # Paste Gaussian onto heatmap (convert all to integers)
            x1 = int(max(0, hm_x - size // 2))
            y1 = int(max(0, hm_y - size // 2))
            x2 = int(min(w, hm_x + size // 2 + 1))
            y2 = int(min(h, hm_y + size // 2 + 1))
            
            g_x1 = int(size // 2 - (hm_x - x1))
            g_y1 = int(size // 2 - (hm_y - y1))
            g_x2 = int(g_x1 + (x2 - x1))
            g_y2 = int(g_y1 + (y2 - y1))
            
            heatmaps[i, y1:y2, x1:x2] = np.maximum(
                heatmaps[i, y1:y2, x1:x2],
                gaussian[g_y1:g_y2, g_x1:g_x2]
            )
        
        return heatmaps
    
    def _generate_mask_from_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Generate ellipse mask from landmark points (approximate cranium segmentation).
        
        Args:
            landmarks: Landmark coordinates [4, 2]
            
        Returns:
            Binary mask [1, H, W]
        """
        h, w = self.image_size
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Calculate ellipse parameters from landmarks
        ofd_1, ofd_2 = landmarks[0], landmarks[1]
        bpd_1, bpd_2 = landmarks[2], landmarks[3]
        
        # Center of ellipse (intersection of diameters)
        center_x = (ofd_1[0] + ofd_2[0] + bpd_1[0] + bpd_2[0]) / 4
        center_y = (ofd_1[1] + ofd_2[1] + bpd_1[1] + bpd_2[1]) / 4
        center = (int(center_x), int(center_y))
        
        # Semi-axes lengths
        ofd_length = np.linalg.norm(ofd_2 - ofd_1) / 2
        bpd_length = np.linalg.norm(bpd_2 - bpd_1) / 2
        
        # Angle of rotation
        angle = np.degrees(np.arctan2(bpd_2[1] - bpd_1[1], bpd_2[0] - bpd_1[0]))
        
        # Draw ellipse
        axes = (int(ofd_length), int(bpd_length))
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
        
        return mask[np.newaxis, :, :]  # Add channel dimension
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching.
        Filter out any samples with incorrect shapes.
        """
        # Filter out samples with incorrect landmark shapes
        valid_batch = []
        for item in batch:
            if item['landmarks'].shape == (4, 2):
                valid_batch.append(item)
        
        if len(valid_batch) == 0:
            raise ValueError("No valid samples in batch")
        
        images = torch.stack([item['image'] for item in valid_batch])
        landmarks = torch.stack([item['landmarks'] for item in valid_batch])
        image_names = [item['image_name'] for item in valid_batch]
        orig_sizes = torch.stack([item['orig_size'] for item in valid_batch])
        
        output = {
            'image': images,
            'landmarks': landmarks,
            'image_name': image_names,
            'orig_size': orig_sizes,
        }
        
        if 'heatmaps' in valid_batch[0]:
            heatmaps = torch.stack([item['heatmaps'] for item in valid_batch])
            output['heatmaps'] = heatmaps
        
        if 'mask' in valid_batch[0]:
            masks = torch.stack([item['mask'] for item in valid_batch])
            output['mask'] = masks
        
        return output


def create_dataloaders(
    csv_path: str,
    image_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    train_val_split: float = 0.8,
    random_seed: int = 42,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders with stratified split.
    
    Args:
        csv_path: Path to annotations CSV
        image_dir: Directory containing images
        batch_size: Batch size
        num_workers: Number of workers for data loading
        train_val_split: Fraction of data for training
        random_seed: Random seed for reproducibility
        **dataset_kwargs: Additional arguments for FetalUltrasoundDataset
        
    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    df = pd.read_csv(csv_path)
    
    # Stratified split (shuffle)
    np.random.seed(random_seed)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * train_val_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Create train/val CSV subsets
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    # Save temporary CSV files
    train_csv = Path(csv_path).parent / 'train_split.csv'
    val_csv = Path(csv_path).parent / 'val_split.csv'
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # Create datasets
    train_dataset = FetalUltrasoundDataset(
        csv_path=str(train_csv),
        image_dir=image_dir,
        augment=True,
        **dataset_kwargs
    )
    
    val_dataset = FetalUltrasoundDataset(
        csv_path=str(val_csv),
        image_dir=image_dir,
        augment=False,
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=FetalUltrasoundDataset.collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=FetalUltrasoundDataset.collate_fn,
    )
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    return train_loader, val_loader

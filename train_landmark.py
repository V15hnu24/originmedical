"""
Training script for landmark detection models (Part A)

Usage:
    python train_landmark.py --model heatmap --epochs 100 --batch_size 16
    python train_landmark.py --model coordinate --backbone efficientnet_b3
    python train_landmark.py --model attention_pyramid --gpus 1
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config
from data.dataset import FetalUltrasoundDataset, create_dataloaders
from data.preprocessing import UltrasoundPreprocessor
from models.landmark_detection.heatmap_model import HeatmapLandmarkDetector, HeatmapLoss
from models.landmark_detection.coordinate_regression import (
    CoordinateRegressionModel,
    CoordinateRegressionLoss,
)
from models.landmark_detection.attention_pyramid import AttentionFeaturePyramid
from utils.metrics import LandmarkMetrics, compute_batch_metrics


class LandmarkDetectionModule(pl.LightningModule):
    """
    PyTorch Lightning module for landmark detection.
    """
    
    def __init__(
        self,
        model_name: str = 'heatmap',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        if model_name == 'heatmap':
            self.model = HeatmapLandmarkDetector(**model_kwargs)
            self.criterion = HeatmapLoss()
            self.use_heatmaps = True
        elif model_name == 'coordinate':
            self.model = CoordinateRegressionModel(**model_kwargs)
            self.criterion = CoordinateRegressionLoss()
            self.use_heatmaps = False
        elif model_name == 'attention_pyramid':
            self.model = AttentionFeaturePyramid(**model_kwargs)
            self.criterion = CoordinateRegressionLoss()
            self.use_heatmaps = False
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.metrics = LandmarkMetrics()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        gt_landmarks = batch['landmarks']
        
        # Forward pass
        outputs = self(images)
        
        # Compute loss
        if self.use_heatmaps and 'heatmaps' in outputs:
            gt_heatmaps = batch['heatmaps']
            loss_dict = self.criterion(
                outputs['heatmaps'],
                gt_heatmaps,
                outputs['coordinates'],
                gt_landmarks
            )
        else:
            pred_coords = outputs['coordinates']
            loss_dict = self.criterion(pred_coords, gt_landmarks)
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train_{key}_loss', value, prog_bar=(key=='total'), on_step=True, on_epoch=True)
        
        # Check for NaN
        if torch.isnan(loss_dict['total']):
            print(f"\nWARNING: NaN loss detected at batch {batch_idx}")
            print(f"Image stats: min={images.min():.3f}, max={images.max():.3f}, mean={images.mean():.3f}")
            print(f"Landmark stats: min={gt_landmarks.min():.3f}, max={gt_landmarks.max():.3f}")
            return torch.tensor(0.0, requires_grad=True, device=images.device)
        
        return loss_dict['total']
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        gt_landmarks = batch['landmarks']
        
        # Forward pass
        outputs = self(images)
        
        # Compute loss
        if self.use_heatmaps and 'heatmaps' in outputs:
            gt_heatmaps = batch['heatmaps']
            loss_dict = self.criterion(
                outputs['heatmaps'],
                gt_heatmaps,
                outputs['coordinates'],
                gt_landmarks
            )
        else:
            pred_coords = outputs['coordinates']
            loss_dict = self.criterion(pred_coords, gt_landmarks)
        
        # Compute metrics (pixel_spacing: assume ~0.15 mm/pixel for 512x512 ultrasound)
        pred_coords = outputs['coordinates']
        metrics = compute_batch_metrics(pred_coords, gt_landmarks, metric_type='landmark', pixel_spacing=0.15)
        
        # Log validation losses and metrics
        for key, value in loss_dict.items():
            self.log(f'val_{key}_loss', value, prog_bar=(key=='total'), on_step=False, on_epoch=True)
        
        # Log MRE metrics
        mre_keys = [k for k in metrics.keys() if 'mre' in k]
        for key in mre_keys:
            self.log(f'val_{key}', metrics[key], prog_bar=('overall' in key), on_step=False, on_epoch=True)
        
        # Log accuracy (SDR) metrics
        sdr_keys = [k for k in metrics.keys() if 'sdr' in k]
        for key in sdr_keys:
            self.log(f'val_{key}', metrics[key], prog_bar=(key=='sdr_2.5mm'), on_step=False, on_epoch=True)
        
        return loss_dict['total']
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-7,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Train landmark detection model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='heatmap',
                       choices=['heatmap', 'coordinate', 'attention_pyramid'],
                       help='Model architecture')
    parser.add_argument('--backbone', type=str, default='resnet50',
                       help='Backbone architecture')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, 
                       default='data/augmented/augmented_ground_truth.csv',
                       help='Path to CSV with ground truth')
    parser.add_argument('--images_dir', type=str,
                       default='data/augmented/images',
                       help='Directory with images')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--augment_strength', type=str, default='medium',
                       choices=['light', 'medium', 'heavy'],
                       help='Augmentation strength')
    
    # System arguments
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Create dataloaders
    preprocessor = UltrasoundPreprocessor(
        target_size=config.IMAGE_SIZE,
        use_clahe=config.USE_CLAHE,
        normalize_method='imagenet',
    )
    
    print(f"\nUsing dataset:")
    print(f"  CSV: {args.csv_path}")
    print(f"  Images: {args.images_dir}\n")
    
    train_loader, val_loader = create_dataloaders(
        csv_path=args.csv_path,
        image_dir=args.images_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_val_split=config.TRAIN_VAL_SPLIT,
        random_seed=args.seed,
        image_size=config.IMAGE_SIZE,
        mode='landmark',
        heatmap_size=config.HEATMAP_SIZE if args.model == 'heatmap' else None,
        heatmap_sigma=config.HEATMAP_SIGMA if args.model == 'heatmap' else None,
    )
    
    print(f"\nTraining {args.model} model with {args.backbone} backbone")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")
    
    # Create model
    model_config = config.LANDMARK_MODELS.get(args.model, {})
    model_config['backbone'] = args.backbone
    
    module = LandmarkDetectionModule(
        model_name=args.model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        **model_config
    )
    
    # Callbacks
    # Save best models based on validation MRE
    checkpoint_callback_best = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR / args.model / 'best',
        filename=f'{args.model}_{args.backbone}_best_{{epoch:02d}}_{{val_mre_overall_px:.2f}}',
        monitor='val_mre_overall_px',
        mode='min',
        save_top_k=3,
        verbose=True,
    )
    
    # Save checkpoint every 10 epochs
    checkpoint_callback_periodic = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR / args.model / 'periodic',
        filename=f'{args.model}_{args.backbone}_epoch{{epoch:02d}}',
        every_n_epochs=10,
        save_top_k=-1,  # Keep all periodic checkpoints
        verbose=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_mre_overall_px',
        patience=20,
        mode='min',
        verbose=True,
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.LOG_DIR,
        name=f'{args.model}_{args.backbone}',
    )
    
    # Trainer
    # Auto-detect best accelerator: CUDA > MPS > CPU
    if args.gpus > 0:
        if torch.cuda.is_available():
            accelerator = 'gpu'
            devices = args.gpus
        elif torch.backends.mps.is_available():
            accelerator = 'mps'
            devices = 1
        else:
            accelerator = 'cpu'
            devices = 1
    else:
        accelerator = 'cpu'
        devices = 1
    
    print(f"Using accelerator: {accelerator}")
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback_best, checkpoint_callback_periodic, early_stop_callback],
        logger=logger,
        precision=32,  # MPS doesn't support mixed precision yet
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        detect_anomaly=True,  # Detect NaN/Inf
        enable_progress_bar=True,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(module, train_loader, val_loader)
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best validation MRE: {checkpoint_callback.best_model_score:.4f} pixels")


if __name__ == '__main__':
    main()

"""
Training script for segmentation models (Part B)

Usage:
    python train_segmentation.py --model unet --epochs 150 --batch_size 8
    python train_segmentation.py --model attention_unet --backbone resnet50
    python train_segmentation.py --model deeplabv3plus --gpus 1
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

sys.path.append(str(Path(__file__).parent))

import config
from data.dataset import create_dataloaders
from data.preprocessing import UltrasoundPreprocessor
from models.segmentation.unet import UNet, AttentionUNet, CombinedSegmentationLoss
from models.segmentation.deeplabv3 import DeepLabV3Plus, FPN
from utils.metrics import SegmentationMetrics, compute_batch_metrics
from utils.ellipse_fitting import EllipseFitter


class SegmentationModule(pl.LightningModule):
    """
    PyTorch Lightning module for segmentation.
    """
    
    def __init__(
        self,
        model_name: str = 'unet',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        extract_landmarks: bool = True,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        if model_name == 'unet':
            self.model = UNet(**model_kwargs)
        elif model_name == 'attention_unet':
            self.model = AttentionUNet(**model_kwargs)
        elif model_name == 'deeplabv3plus':
            self.model = DeepLabV3Plus(**model_kwargs)
        elif model_name == 'fpn':
            self.model = FPN(**model_kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.criterion = CombinedSegmentationLoss()
        self.metrics = SegmentationMetrics()
        
        # Ellipse fitter for landmark extraction
        if extract_landmarks:
            self.ellipse_fitter = EllipseFitter(method='ransac')
        else:
            self.ellipse_fitter = None
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        gt_masks = batch['mask']
        
        # Forward pass
        pred_masks = self(images)
        
        # Compute loss
        loss_dict = self.criterion(pred_masks, gt_masks)
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train_{key}_loss', value, prog_bar=(key=='total'))
        
        return loss_dict['total']
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        gt_masks = batch['mask']
        gt_landmarks = batch.get('landmarks', None)
        
        # Forward pass
        pred_masks = self(images)
        
        # Compute loss
        loss_dict = self.criterion(pred_masks, gt_masks)
        
        # Compute segmentation metrics
        pred_masks_sigmoid = torch.sigmoid(pred_masks)
        seg_metrics = compute_batch_metrics(
            pred_masks_sigmoid,
            gt_masks,
            metric_type='segmentation'
        )
        
        # Log
        for key, value in loss_dict.items():
            self.log(f'val_{key}_loss', value, prog_bar=(key=='total'))
        
        for key, value in seg_metrics.items():
            self.log(f'val_{key}', value, prog_bar=True)
        
        # Extract landmarks if ground truth available
        if self.ellipse_fitter and gt_landmarks is not None:
            # Extract landmarks from predicted masks
            pred_masks_np = pred_masks_sigmoid.cpu().numpy()
            batch_size = pred_masks_np.shape[0]
            
            landmark_errors = []
            for i in range(batch_size):
                result = self.ellipse_fitter.mask_to_landmarks(pred_masks_np[i, 0])
                if result is not None:
                    pred_lm = result['landmarks']
                    gt_lm = gt_landmarks[i].cpu().numpy()
                    error = torch.tensor(pred_lm - gt_lm).norm(dim=1).mean()
                    landmark_errors.append(error.item())
            
            if landmark_errors:
                self.log('val_landmark_mre', sum(landmark_errors) / len(landmark_errors))
        
        return loss_dict['total']
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_total_loss',
                'interval': 'epoch',
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'attention_unet', 'deeplabv3plus', 'fpn'],
                       help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34',
                       help='Encoder backbone')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Data arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # System arguments
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        csv_path=str(config.CSV_PATH),
        image_dir=str(config.DATA_DIR),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_val_split=config.TRAIN_VAL_SPLIT,
        random_seed=args.seed,
        image_size=config.IMAGE_SIZE,
        mode='segmentation',
    )
    
    print(f"\nTraining {args.model} model with {args.encoder} encoder")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}\n")
    
    # Create model
    model_config = config.SEGMENTATION_MODELS.get(args.model, {})
    model_config['encoder_name'] = args.encoder
    
    module = SegmentationModule(
        model_name=args.model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        extract_landmarks=True,
        **model_config
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR / args.model,
        filename=f'{args.model}_{args.encoder}_{{epoch:02d}}_{{val_dice:.4f}}',
        monitor='val_dice',
        mode='max',
        save_top_k=3,
        verbose=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_dice',
        patience=25,
        mode='max',
        verbose=True,
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.LOG_DIR,
        name=f'{args.model}_{args.encoder}',
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 and torch.cuda.is_available() else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        precision='16-mixed' if config.MIXED_PRECISION else 32,
        log_every_n_steps=config.LOG_INTERVAL,
        gradient_clip_val=1.0,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(module, train_loader, val_loader)
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best validation Dice: {checkpoint_callback.best_model_score:.4f}")


if __name__ == '__main__':
    main()

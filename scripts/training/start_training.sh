#!/bin/bash

# Training script for landmark detection with augmented dataset
# This will train on the augmented dataset (2451 images instead of 622)

echo "=========================================="
echo "LANDMARK DETECTION TRAINING"
echo "=========================================="
echo ""
echo "Using augmented dataset:"
echo "  - Total images: 2451 (3.94x expansion)"
echo "  - Original: 622 images"
echo "  - Augmented: 1829 images"
echo ""
echo "Training will show:"
echo "  - Validation Loss (val_total_loss)"
echo "  - Mean Radial Error in pixels (val_mre_overall_px)"
echo "  - Mean Radial Error in mm (val_mre_overall_mm)"
echo "  - Accuracy at 2.5mm threshold (val_sdr_2.5mm) [%]"
echo "  - Accuracy at 2mm, 3mm, 4mm, 5mm thresholds"
echo ""
echo "=========================================="
echo ""

# Activate virtual environment
source originMedical/bin/activate

# Start training
python train_landmark.py \
  --model heatmap \
  --backbone resnet50 \
  --csv_path data/augmented/augmented_ground_truth.csv \
  --images_dir data/augmented/images \
  --epochs 100 \
  --batch_size 8 \
  --num_workers 0 \
  --gpus 1

echo ""
echo "Training complete! Check logs/ directory for TensorBoard logs."
echo "View training curves with: tensorboard --logdir logs/"

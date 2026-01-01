#!/bin/bash

# Script to augment the dataset

echo "=========================================="
echo "DATA AUGMENTATION"
echo "=========================================="
echo ""
echo "This will create augmented versions of the original dataset"
echo ""

# Activate virtual environment
source originMedical/bin/activate

# Augment dataset with 3 augmentations per image, medium strength
python augment_dataset.py \
  --augmentations_per_image 3 \
  --strength medium \
  --verify

echo ""
echo "=========================================="
echo "Augmentation complete!"
echo "Output: data/augmented/"
echo "  - Images: data/augmented/images/"
echo "  - CSV: data/augmented/augmented_ground_truth.csv"
echo "=========================================="

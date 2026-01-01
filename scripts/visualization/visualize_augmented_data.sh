#!/bin/bash

# Script to visualize augmented dataset

echo "=========================================="
echo "VISUALIZE AUGMENTED DATASET"
echo "=========================================="
echo ""

# Activate virtual environment
source originMedical/bin/activate

# Visualize augmented dataset
# - 16 random samples (mix of original and augmented)
# - Compare 3 original images with their augmented versions
python visualize_augmented_dataset.py \
  --num_samples 16 \
  --compare_originals \
  --num_comparisons 3

echo ""
echo "=========================================="
echo "Visualization complete!"
echo "Output saved to: output/augmented_visualizations/"
echo "  - random_samples.png: 16 random samples"
echo "  - comparison_*.png: Original vs augmented comparisons"
echo "=========================================="

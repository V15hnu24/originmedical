#!/bin/bash

# Script to visualize original dataset

echo "=========================================="
echo "VISUALIZE ORIGINAL DATASET"
echo "=========================================="
echo ""

# Activate virtual environment
source originMedical/bin/activate

# Visualize 10 random samples from original dataset
python utils/visualise.py \
  --csv_path role_challenge_dataset_ground_truth.csv \
  --images_dir images \
  --num_samples 10

echo ""
echo "=========================================="
echo "Visualization complete!"
echo "Check output/ directory for saved images"
echo "=========================================="

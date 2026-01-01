#!/bin/bash

# Repository Organization Script
# This script organizes the Origin Medical project into a clean structure

echo "========================================"
echo "Origin Medical - Repository Organization"
echo "========================================"
echo ""

# Create organized directory structure
echo "Creating organized directory structure..."

# 1. Create main directories
mkdir -p docs
mkdir -p scripts/{training,evaluation,visualization,data}
mkdir -p notebooks
mkdir -p archived
mkdir -p configs

# 2. Move documentation files
echo "Organizing documentation..."
mv APPROACH_AND_METHODOLOGY.md docs/ 2>/dev/null
mv PART_B_SEGMENTATION_GUIDE.md docs/ 2>/dev/null
mv QUICKSTART.md docs/ 2>/dev/null
mv UNET_EVALUATION_GUIDE.md docs/ 2>/dev/null
mv PROJECT_REPORT.md docs/ 2>/dev/null

# Keep README.md in root
# mv README.md docs/ would move it, but we keep it in root

# 3. Move shell scripts
echo "Organizing scripts..."
mv start_training.sh scripts/training/ 2>/dev/null
mv start_train.sh scripts/training/ 2>/dev/null
mv evaluate_coordinate.sh scripts/evaluation/ 2>/dev/null
mv evaluate_unet.sh scripts/evaluation/ 2>/dev/null
mv run_inference.sh scripts/evaluation/ 2>/dev/null

mv visualize_augmented_data.sh scripts/visualization/ 2>/dev/null
mv visualize_original_data.sh scripts/visualization/ 2>/dev/null
mv augment_data.sh scripts/data/ 2>/dev/null

# 4. Move Python utility scripts
echo "Organizing utility scripts..."
mv visualize_evaluation.py scripts/visualization/ 2>/dev/null
mv visualize_predictions_on_images.py scripts/visualization/ 2>/dev/null
mv visualize_gt_pred_overlay.py scripts/visualization/ 2>/dev/null
mv debug_inference_pipeline.py scripts/evaluation/ 2>/dev/null
mv save_predictions_to_csv.py scripts/evaluation/ 2>/dev/null

mv augment_dataset.py scripts/data/ 2>/dev/null
mv visualize_augmented_dataset.py scripts/visualization/ 2>/dev/null

# 5. Move temp/misc files
echo "Moving temporary files..."
mv temp.txt archived/ 2>/dev/null

# 6. Create symlinks for commonly used scripts in root for convenience
echo "Creating convenience symlinks..."
ln -sf scripts/training/start_training.sh ./train.sh 2>/dev/null
ln -sf scripts/evaluation/evaluate_coordinate.sh ./evaluate.sh 2>/dev/null
ln -sf scripts/evaluation/run_inference.sh ./inference.sh 2>/dev/null

echo ""
echo "✓ Repository organized!"
echo ""
echo "New structure:"
echo "  docs/              - All documentation"
echo "  scripts/           - All scripts organized by function"
echo "    ├── training/    - Training scripts"
echo "    ├── evaluation/  - Evaluation scripts"  
echo "    ├── visualization/ - Visualization scripts"
echo "    └── data/        - Data processing scripts"
echo "  models/            - Model architectures (unchanged)"
echo "  data/              - Data loading and preprocessing (unchanged)"
echo "  utils/             - Utility functions (unchanged)"
echo "  checkpoints/       - Model checkpoints (unchanged)"
echo "  results/           - Output results (unchanged)"
echo "  archived/          - Old/temp files"
echo ""
echo "Convenience symlinks in root:"
echo "  train.sh -> scripts/training/start_training.sh"
echo "  evaluate.sh -> scripts/evaluation/evaluate_coordinate.sh"
echo "  inference.sh -> scripts/evaluation/run_inference.sh"
echo ""

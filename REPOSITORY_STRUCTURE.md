# Repository Structure and Organization Guide

## Clean Repository Structure

```
Origin-Medical/
│
├── README.md                          # Project overview and quick start
├── PROJECT_REPORT.md                  # Comprehensive project documentation
├── requirements.txt                   # Python dependencies
├── config.py                          # Central configuration file
│
├── train.sh                          # Symlink to training script
├── evaluate.sh                       # Symlink to evaluation script
├── inference.sh                      # Symlink to inference script
│
├── docs/                             # 📚 All Documentation
│   ├── QUICKSTART.md
│   ├── APPROACH_AND_METHODOLOGY.md
│   ├── PART_B_SEGMENTATION_GUIDE.md
│   └── UNET_EVALUATION_GUIDE.md
│
├── models/                           # 🧠 Model Architectures
│   ├── landmark_detection/
│   │   ├── coordinate_regression.py  # Direct coordinate prediction
│   │   ├── heatmap_model.py         # Heatmap-based detection
│   │   └── attention_pyramid.py      # Attention-based architecture
│   └── segmentation/
│       ├── unet.py                   # U-Net for segmentation
│       └── deeplabv3.py              # DeepLabV3+ architecture
│
├── data/                             # 📊 Data Processing
│   ├── dataset.py                    # PyTorch Dataset classes
│   ├── preprocessing.py              # Image preprocessing
│   ├── augmentation.py               # Data augmentation pipeline
│   └── augmented/                    # Augmented dataset storage
│       ├── augmented_ground_truth.csv
│       ├── train_split.csv
│       ├── val_split.csv
│       └── images/
│
├── utils/                            # 🛠️ Utility Functions
│   ├── metrics.py                    # Evaluation metrics
│   ├── visualization.py              # Visualization utilities
│   ├── ellipse_fitting.py            # Ellipse fitting for landmarks
│   └── visualise_segmentation.py     # Segmentation visualization
│
├── scripts/                          # 📜 Organized Scripts
│   ├── training/
│   │   ├── start_training.sh
│   │   └── start_train.sh
│   ├── evaluation/
│   │   ├── evaluate_coordinate.sh
│   │   ├── evaluate_unet.sh
│   │   ├── run_inference.sh
│   │   ├── debug_inference_pipeline.py
│   │   └── save_predictions_to_csv.py
│   ├── visualization/
│   │   ├── visualize_augmented_data.sh
│   │   ├── visualize_original_data.sh
│   │   ├── visualize_evaluation.py
│   │   ├── visualize_predictions_on_images.py
│   │   ├── visualize_gt_pred_overlay.py
│   │   └── visualize_augmented_dataset.py
│   └── data/
│       ├── augment_data.sh
│       └── augment_dataset.py
│
├── train_landmark.py                 # Main training script (Part A)
├── train_segmentation.py             # Segmentation training (Part B)
├── evaluate_coordinate.py            # Coordinate model evaluation
├── inference_landmark.py             # Inference script
│
├── checkpoints/                      # 💾 Model Checkpoints
│   ├── coordinate/
│   │   ├── best/
│   │   │   └── coordinate_efficientnet_b3_best_epoch=48_val_mre_overall_px=164.88.ckpt
│   │   └── periodic/
│   └── heatmap/
│       ├── best/
│       └── periodic/
│
├── results/                          # 📈 Outputs and Results
│   ├── coordinate_evaluation/
│   │   ├── evaluation_results.csv
│   │   ├── evaluation_summary.csv
│   │   └── performance_dashboard.png
│   ├── unnet_e48/
│   │   └── visualise/              # GT vs Prediction visualizations
│   └── predictions_vs_gt.csv       # Complete predictions dataset
│
├── logs/                             # 📝 Training Logs
│   ├── coordinate_efficientnet_b3/
│   └── heatmap_resnet50/
│
├── images/                           # 🖼️ Original Dataset Images
│   └── [ultrasound images]
│
├── originMedical/                    # 🐍 Python Virtual Environment
│   ├── bin/
│   ├── lib/
│   └── pyvenv.cfg
│
├── archived/                         # 📦 Old/Temporary Files
│   └── temp.txt
│
├── train_split.csv                   # Training set split
├── val_split.csv                     # Validation set split
└── role_challenge_dataset_ground_truth.csv  # Original annotations
```

## File Categories

### Core Training Files
- `train_landmark.py` - Main training entry point for landmark detection
- `train_segmentation.py` - Segmentation model training
- `config.py` - Centralized configuration

### Evaluation Files
- `evaluate_coordinate.py` - Comprehensive model evaluation
- `inference_landmark.py` - Run inference on new images
- `scripts/evaluation/` - Evaluation utilities

### Data Files
- `data/dataset.py` - Dataset loaders
- `data/preprocessing.py` - Preprocessing pipeline
- `data/augmentation.py` - Augmentation strategies

### Model Definitions
- `models/landmark_detection/` - Landmark detection architectures
- `models/segmentation/` - Segmentation architectures

### Utilities
- `utils/metrics.py` - Evaluation metrics (MRE, Dice, IoU)
- `utils/visualization.py` - Plotting and visualization
- `utils/ellipse_fitting.py` - Geometric fitting utilities

### Documentation
- `README.md` - Main project README
- `PROJECT_REPORT.md` - Comprehensive project report
- `docs/` - Additional documentation

## Organization Script

Run the organization script to clean up the repository:

```bash
chmod +x organize_repo.sh
./organize_repo.sh
```

This will:
1. Create organized directory structure
2. Move files to appropriate locations
3. Create convenience symlinks
4. Archive temporary files

## Convenience Commands

After organization, use these shortcuts:

```bash
# Training
./train.sh

# Evaluation
./evaluate.sh checkpoints/coordinate/best/checkpoint.ckpt

# Inference
./inference.sh
```

## Git Ignore Recommendations

Add to `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*.so
*.egg-info/

# Virtual Environment
originMedical/
venv/
env/

# Checkpoints (too large)
checkpoints/*/periodic/
*.ckpt

# Results (generated)
results/
output/
logs/

# Temporary
temp.txt
*.tmp
.DS_Store

# Data (too large, host separately)
images/
data/augmented/images/
```

## Best Practices

### For Development:
1. Keep root directory clean
2. Use symlinks for frequently used scripts
3. Organize by function (training, evaluation, visualization)
4. Document everything in `docs/`

### For Collaboration:
1. Use version control (git)
2. Document changes in commit messages
3. Keep `requirements.txt` updated
4. Use configuration files, not hardcoded paths

### For Production:
1. Separate config for different environments
2. Use environment variables for sensitive data
3. Containerize (Docker) for deployment
4. Version your models (MLflow, DVC)

## Quick Reference

### Training a Model
```bash
python train_landmark.py --model coordinate --epochs 150 --batch_size 16
```

### Evaluating a Model
```bash
python scripts/evaluation/evaluate_coordinate.py \
    --checkpoint checkpoints/coordinate/best/checkpoint.ckpt \
    --save_visualizations
```

### Creating Visualizations
```bash
python scripts/visualization/visualize_gt_pred_overlay.py \
    --checkpoint checkpoints/coordinate/best/checkpoint.ckpt \
    --num_samples 30
```

### Analyzing Results
```bash
python scripts/evaluation/save_predictions_to_csv.py \
    --checkpoint checkpoints/coordinate/best/checkpoint.ckpt
```

## Maintenance

### Regular Tasks:
- [ ] Clean up `results/` periodically
- [ ] Archive old checkpoints
- [ ] Update documentation with changes
- [ ] Run tests before pushing code
- [ ] Review and update `requirements.txt`

### Before Sharing:
- [ ] Remove temporary files
- [ ] Check all paths are relative
- [ ] Update README with any new dependencies
- [ ] Ensure scripts are executable
- [ ] Test on fresh environment

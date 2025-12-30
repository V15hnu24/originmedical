# Quick Start Guide - Fetal Ultrasound Biometry

## Installation

### 1. Create Virtual Environment
```bash
cd "/Users/vishnuvardhan/Desktop/Origin Medical"
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
# venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

## Training Models

### Part A: Landmark Detection

#### Train Heatmap Model (Recommended for beginners)
```bash
python train_landmark.py \
    --model heatmap \
    --backbone resnet50 \
    --epochs 150 \
    --batch_size 16 \
    --lr 1e-4
```

#### Train Direct Regression Model (Fastest)
```bash
python train_landmark.py \
    --model coordinate \
    --backbone efficientnet_b3 \
    --epochs 150 \
    --batch_size 16
```

#### Train Attention Pyramid Model (Best accuracy)
```bash
python train_landmark.py \
    --model attention_pyramid \
    --backbone resnet34 \
    --epochs 150 \
    --batch_size 12
```

### Part B: Segmentation

#### Train U-Net (Recommended baseline)
```bash
python train_segmentation.py \
    --model unet \
    --encoder resnet34 \
    --epochs 150 \
    --batch_size 8
```

#### Train Attention U-Net (Better boundaries)
```bash
python train_segmentation.py \
    --model attention_unet \
    --encoder resnet50 \
    --epochs 150 \
    --batch_size 8
```

#### Train DeepLabV3+ (Best multi-scale)
```bash
python train_segmentation.py \
    --model deeplabv3plus \
    --encoder resnet50 \
    --epochs 150 \
    --batch_size 6
```

## Monitor Training

### TensorBoard
```bash
tensorboard --logdir logs/
# Open browser to http://localhost:6006
```

### Check Outputs
```bash
# Training logs
ls -lh logs/

# Model checkpoints
ls -lh checkpoints/

# Results
ls -lh results/
```

## Evaluate Models

### Evaluate Landmark Detection
```python
import torch
from models.landmark_detection.heatmap_model import HeatmapLandmarkDetector
from utils.metrics import LandmarkMetrics

# Load model
model = HeatmapLandmarkDetector()
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# Evaluate
metrics = LandmarkMetrics()
# ... add evaluation code
```

### Evaluate Segmentation
```python
from models.segmentation.unet import UNet
from utils.metrics import SegmentationMetrics
from utils.ellipse_fitting import EllipseFitter

# Load model
model = UNet()
model.load_state_dict(torch.load('checkpoints/best_seg_model.pth'))
model.eval()

# Fit ellipse and extract landmarks
fitter = EllipseFitter(method='ransac')
# ... add evaluation code
```

## Project Structure Overview

```
Origin Medical/
├── README.md                          # Main documentation
├── APPROACH_AND_METHODOLOGY.md        # Detailed approach document
├── QUICKSTART.md                      # This file
├── requirements.txt                   # Dependencies
├── config.py                          # Configuration
│
├── data/                              # Data processing
│   ├── dataset.py                     # Dataset class
│   ├── preprocessing.py               # Preprocessing utilities
│   └── augmentation.py                # Augmentation strategies
│
├── models/                            # Model architectures
│   ├── landmark_detection/
│   │   ├── heatmap_model.py          # Heatmap-based model
│   │   ├── coordinate_regression.py   # Direct regression
│   │   └── attention_pyramid.py       # FPN + Attention
│   └── segmentation/
│       ├── unet.py                    # U-Net variants
│       └── deeplabv3.py              # DeepLab variants
│
├── utils/                             # Utilities
│   ├── metrics.py                     # Evaluation metrics
│   ├── visualization.py               # Visualization tools
│   └── ellipse_fitting.py            # Ellipse fitting
│
├── train_landmark.py                  # Training script (Part A)
├── train_segmentation.py              # Training script (Part B)
│
├── images/                            # Dataset images (yours)
├── role_challenge_dataset_ground_truth.csv  # Annotations (yours)
│
├── checkpoints/                       # Saved models
├── logs/                              # Training logs
└── results/                           # Evaluation results
```

## Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution:** Reduce batch size
```bash
python train_landmark.py --batch_size 8  # Instead of 16
```

### Issue: Slow Training
**Solution:** 
1. Use mixed precision (already enabled in config)
2. Reduce image size in config.py
3. Use fewer workers: `--num_workers 2`

### Issue: Import Errors
**Solution:** Make sure you're in the right directory
```bash
cd "/Users/vishnuvardhan/Desktop/Origin Medical"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Can't Find Images
**Solution:** Check paths in config.py
```python
# config.py
DATA_DIR = BASE_DIR / "images"  # Should point to your images folder
```

## Next Steps

1. **Start Simple:** Train U-Net and Heatmap model first
2. **Monitor Training:** Use TensorBoard to track progress
3. **Compare Models:** Train multiple variants
4. **Evaluate:** Use provided metrics to compare
5. **Ensemble:** Combine best models for final predictions

## Key Papers Referenced

1. **Heatmap Regression:** Regressing Heatmaps for Multiple Landmark Localization using CNNs
2. **Attention Pyramid:** Cephalometric Landmark Detection by Attentive Feature Pyramid Fusion
3. **U-Net:** Convolutional Networks for Biomedical Image Segmentation
4. **Attention U-Net:** Learning Where to Look for the Pancreas
5. **DeepLabV3+:** Encoder-Decoder with Atrous Separable Convolution

## Contact & Support

For questions about the implementation:
- Review: [APPROACH_AND_METHODOLOGY.md](APPROACH_AND_METHODOLOGY.md)
- Check: [README.md](README.md)

## License & Usage

This code is provided for the Origin Medical role challenge.

---

**Last Updated:** December 29, 2025

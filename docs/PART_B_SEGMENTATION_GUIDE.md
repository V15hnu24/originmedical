# Part B: Segmentation-Based Approach - Implementation Guide

## Overview

For **Part B**, you have ground truth segmentation masks in `details/masks/` that show the cranium ellipse. The workflow is:

1. **Train a segmentation model** to predict cranium masks
2. **Apply ellipse fitting** to the predicted mask
3. **Extract BPD and OFD landmarks** from the fitted ellipse

---

## Dataset Structure

### Ground Truth Masks Location
```
details/masks/
├── 000_HC_Annotation.png
├── 001_HC_Annotation.png
├── 002_HC_Annotation.png
└── ... (622 total masks)
```

### Mask Format
- **Type**: Binary masks (grayscale PNG)
- **Values**: 0 (background), 255 (cranium)
- **Size**: 540x800 pixels
- **Format**: Ellipse-shaped region representing fetal cranium

### Naming Convention
- Image: `001_HC.png` → Mask: `001_HC_Annotation.png`
- Image: `138_2HC.png` → Mask: `138_2HC_Annotation.png`

---

## Processing Pipeline

### Step 1: Dataset Preparation

The existing `data/dataset.py` already handles masks! Check the `FetalUltrasoundDataset` class:

```python
from data.dataset import FetalUltrasoundDataset

# For segmentation training
dataset = FetalUltrasoundDataset(
    csv_path='role_challenge_dataset_ground_truth.csv',
    images_dir='images',
    masks_dir='details/masks',  # Point to your masks
    mode='segmentation',  # Important!
    augment=True
)

# Dataset returns:
# - image: [3, 512, 512] tensor
# - mask: [1, 512, 512] tensor (0/1 binary)
# - landmarks: [4, 2] tensor (for validation)
# - image_name: str
```

**Key points:**
- Masks are automatically loaded by matching image names
- Masks are resized to 512x512 to match image size
- Augmentations are applied consistently to both image and mask
- Values normalized to [0, 1] for segmentation training

### Step 2: Train Segmentation Model

Use the existing `train_segmentation.py` script:

```bash
# Update config.py to point to masks
python train_segmentation.py --model unet --encoder resnet34 --epochs 100

# Try different architectures
python train_segmentation.py --model attention_unet --encoder resnet50
python train_segmentation.py --model deeplabv3plus --encoder efficientnet-b3
python train_segmentation.py --model fpn --encoder resnet50
```

**Models available (already implemented):**
1. **U-Net** - Standard encoder-decoder with skip connections
2. **Attention U-Net** - U-Net + SCSE attention gates
3. **U-Net++** - Nested U-Net with dense connections
4. **DeepLabV3+** - Atrous convolutions for multi-scale context
5. **FPN** - Feature Pyramid Network
6. **PSPNet** - Pyramid Pooling Module

**Loss functions:**
- Dice Loss (overlap measure)
- Focal Loss (handles class imbalance)
- Boundary Loss (emphasizes edges)
- Combined: 0.5×Dice + 0.3×Focal + 0.2×Boundary

### Step 3: Ellipse Fitting (Mask → Landmarks)

The `utils/ellipse_fitting.py` module extracts landmarks from predicted masks:

```python
from utils.ellipse_fitting import EllipseFitter

# Initialize fitter
fitter = EllipseFitter(method='ransac')  # or 'opencv', 'least_squares'

# Fit ellipse to mask
result = fitter.fit_mask_to_landmarks(predicted_mask)

if result is not None:
    landmarks, ellipse_params, confidence = result
    
    # landmarks shape: [4, 2]
    # landmarks[0] = OFD point 1 (x, y)
    # landmarks[1] = OFD point 2 (x, y)
    # landmarks[2] = BPD point 1 (x, y)
    # landmarks[3] = BPD point 2 (x, y)
    
    # confidence: 0.0-1.0 (quality of fit)
```

**Three ellipse fitting methods:**

1. **OpenCV** (fastest)
   - Direct least-squares fitting
   - Fast but sensitive to outliers
   - Good for clean masks

2. **RANSAC** (most robust) ⭐ **Recommended**
   - Iterative fitting with outlier rejection
   - Robust to noise and artifacts
   - Best for real predictions

3. **Least Squares** (most accurate)
   - scipy.optimize for precision
   - Slower but more accurate
   - Good for high-quality masks

**How landmarks are extracted:**
```python
# From fitted ellipse parameters: (center, axes, angle)
# 1. OFD = major axis endpoints (longest diameter)
# 2. BPD = minor axis endpoints (shortest diameter)
# 3. Ensure BPD ⊥ OFD (perpendicular by ellipse geometry)
```

---

## Configuration Updates

Update `config.py` to use masks:

```python
# In config.py

# Part B: Segmentation paths
MASKS_DIR = 'details/masks'

# Dataset mode
DATASET_MODE = 'segmentation'  # or 'landmark' for Part A

# Segmentation settings
MASK_SIZE = (512, 512)  # Resize masks to this size
MASK_THRESHOLD = 0.5    # Binarization threshold for predictions

# Ellipse fitting
ELLIPSE_METHOD = 'ransac'  # 'opencv', 'ransac', or 'least_squares'
RANSAC_ITERATIONS = 1000
RANSAC_THRESHOLD = 5.0
```

---

## Visualization

### Visualize Ground Truth Masks

```bash
# Visualize masks and ellipse fitting accuracy
python utils/visualise_segmentation.py --num_images 10 --save_dir output/segmentation_vis

# Try different ellipse methods
python utils/visualise_segmentation.py --num_images 20 --ellipse_method opencv
python utils/visualise_segmentation.py --num_images 20 --ellipse_method ransac
python utils/visualise_segmentation.py --num_images 20 --ellipse_method least_squares
```

**What you'll see:**
- **Left**: Original image + GT landmarks (from CSV)
- **Middle**: Ground truth segmentation mask
- **Right**: Overlay showing:
  - Cyan: Mask region
  - Solid lines: GT landmarks (from CSV)
  - Dashed lines: Predicted landmarks (from ellipse fit)
  - Error statistics

This shows you how well ellipse fitting can extract landmarks from perfect masks!

---

## Training Workflow for Part B

### 1. **Train Segmentation Model**

```bash
python train_segmentation.py \
    --model unet \
    --encoder resnet34 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4
```

**During training:**
- Model learns to predict cranium masks
- Loss: Dice + Focal + Boundary
- Metrics: Dice, IoU, Hausdorff distance
- Checkpoints saved to `checkpoints/segmentation/`
- Logs saved to `logs/segmentation/`

### 2. **Validation with Ellipse Fitting**

The training script automatically:
1. Generates predicted masks on validation set
2. Fits ellipse to each predicted mask
3. Extracts BPD/OFD landmarks
4. Compares to ground truth landmarks (from CSV)
5. Reports **both** segmentation metrics (Dice) **and** landmark metrics (MRE)

```
Validation Results:
  Dice: 0.942
  IoU: 0.891
  MRE Overall: 3.24 px
  MRE OFD: 3.18 px
  MRE BPD: 3.30 px
  SDR@2.5mm: 78.4%
```

### 3. **Inference on Test Set**

```python
# Example inference script
import torch
from models.segmentation.unet import UNet
from utils.ellipse_fitting import EllipseFitter

# Load trained model
model = UNet(encoder_name='resnet34', num_classes=1)
checkpoint = torch.load('checkpoints/segmentation/best.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Initialize ellipse fitter
fitter = EllipseFitter(method='ransac')

# Predict on new image
with torch.no_grad():
    mask_pred = model(image_tensor)
    mask_pred = torch.sigmoid(mask_pred)
    mask_pred = (mask_pred > 0.5).float()

# Extract landmarks
mask_np = mask_pred[0, 0].cpu().numpy()
landmarks, ellipse, conf = fitter.fit_mask_to_landmarks(mask_np)

print(f"OFD points: {landmarks[0]}, {landmarks[1]}")
print(f"BPD points: {landmarks[2]}, {landmarks[3]}")
print(f"Confidence: {conf:.3f}")
```

---

## Model Variations to Try (Part B Requirement)

You need **at least 3 variations** for Part B. Here are suggestions:

### Variation 1: U-Net with Different Encoders
```bash
python train_segmentation.py --model unet --encoder resnet34
python train_segmentation.py --model unet --encoder resnet50
python train_segmentation.py --model unet --encoder efficientnet-b3
```
**Hypothesis**: Stronger encoders (ResNet50, EfficientNet) capture better features

### Variation 2: Different Architectures
```bash
python train_segmentation.py --model unet --encoder resnet34
python train_segmentation.py --model attention_unet --encoder resnet34
python train_segmentation.py --model deeplabv3plus --encoder resnet34
```
**Hypothesis**: Attention mechanisms and atrous convolutions improve boundary accuracy

### Variation 3: Different Loss Functions
Modify `train_segmentation.py`:
```python
# Experiment 1: Dice only
loss = DiceLoss()

# Experiment 2: Dice + Focal
loss = CombinedSegmentationLoss(dice_weight=0.7, focal_weight=0.3, boundary_weight=0.0)

# Experiment 3: Full combined (default)
loss = CombinedSegmentationLoss(dice_weight=0.5, focal_weight=0.3, boundary_weight=0.2)
```
**Hypothesis**: Boundary loss improves edge precision → better ellipse fitting

### Variation 4: Different Ellipse Fitting Methods
```python
# In train_segmentation.py, change:
fitter = EllipseFitter(method='opencv')    # Fast, direct
fitter = EllipseFitter(method='ransac')    # Robust
fitter = EllipseFitter(method='least_squares')  # Accurate
```
**Hypothesis**: RANSAC is more robust to imperfect segmentation predictions

---

## Evaluation Metrics

### Segmentation Metrics
- **Dice Coefficient**: Overlap between predicted and GT mask (0-1, higher better)
- **IoU (Jaccard)**: Intersection over Union (0-1, higher better)
- **Hausdorff Distance**: Maximum distance between boundaries (pixels, lower better)
- **Average Surface Distance**: Mean boundary mismatch (pixels, lower better)

### Landmark Metrics (from ellipse fitting)
- **MRE (Mean Radial Error)**: Average distance between predicted and GT landmarks (pixels)
- **SDR (Successful Detection Rate)**: % landmarks within threshold (2.0, 2.5, 3.0, 4.0 mm)
- **Clinical Measurements**: BPD, OFD, HC errors (mm)

### Combined Evaluation
The segmentation approach is evaluated on:
1. **Mask Quality**: Dice, IoU (how well does model segment cranium?)
2. **Landmark Accuracy**: MRE, SDR (how accurate are ellipse-fitted landmarks?)

**Expected Performance:**
- Good segmentation (Dice > 0.90) → Good landmarks (MRE < 4px)
- Poor segmentation (Dice < 0.80) → Poor landmarks (MRE > 8px)
- Ellipse fitting quality matters: RANSAC typically reduces MRE by 10-20%

---

## Debugging Tips

### Issue: Masks not loading
```python
# Check mask naming
import os
from pathlib import Path

masks_dir = Path('details/masks')
for img_name in ['000_HC.png', '001_HC.png']:
    mask_name = img_name.replace('.png', '_Annotation.png')
    mask_path = masks_dir / mask_name
    print(f"{img_name} → {mask_name}: {mask_path.exists()}")
```

### Issue: Poor ellipse fitting
```python
# Visualize mask before fitting
import cv2
import matplotlib.pyplot as plt

mask = cv2.imread('details/masks/000_HC_Annotation.png', 0)
plt.imshow(mask, cmap='gray')
plt.title(f'Mask stats: min={mask.min()}, max={mask.max()}, shape={mask.shape}')
plt.show()

# Check contour extraction
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of contours: {len(contours)}")
print(f"Main contour points: {len(contours[0])}")
```

### Issue: Low confidence scores
```python
# Post-process masks before fitting
from utils.ellipse_fitting import EllipseFitter

fitter = EllipseFitter(method='ransac')

# Apply morphological operations
mask_clean = fitter.post_process_mask(mask)

# Then fit
result = fitter.fit_mask_to_landmarks(mask_clean)
```

---

## References

The two papers mentioned for Part B:

1. **U-Net: Convolutional Networks for Biomedical Image Segmentation**  
   https://arxiv.org/pdf/1505.04597  
   - Standard U-Net architecture
   - Skip connections for fine details
   - Implemented in `models/segmentation/unet.py`

2. **Attention U-Net: Learning Where to Look for the Pancreas**  
   https://arxiv.org/abs/1804.03999  
   - Attention gates for focusing on relevant features
   - Implemented in `models/segmentation/unet.py` (AttentionUNet class)

---

## Summary

**Part B Workflow:**
1. ✅ Ground truth masks available in `details/masks/`
2. ✅ Dataset loader handles masks automatically (`mode='segmentation'`)
3. ✅ Train segmentation models (4+ architectures available)
4. ✅ Ellipse fitting extracts landmarks from masks (3 methods)
5. ✅ Evaluate both segmentation quality and landmark accuracy
6. ✅ Visualization tools to verify entire pipeline

**Next Steps:**
1. Run `python utils/visualise_segmentation.py --num_images 10` to see masks and ellipse fitting
2. Start training: `python train_segmentation.py --model unet --encoder resnet34`
3. Try 3+ variations (encoders, architectures, losses, ellipse methods)
4. Compare Part A (direct landmark detection) vs Part B (segmentation → ellipse fitting)

Both approaches are fully implemented and ready to train! 🚀

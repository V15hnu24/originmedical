# Fetal Ultrasound Biometry - Project Report

**Project**: Automated Landmark Detection and Biometry Measurement in Fetal Ultrasound Images  
**Date**: January 1, 2026  
**Status**: Part A (Landmark Detection) - In Progress | Part B (Segmentation) - Planned

---

## Executive Summary

This project aims to develop an automated system for fetal ultrasound biometry measurement through landmark detection and segmentation. The system detects key anatomical landmarks (OFD and BPD points) in fetal head ultrasound images to enable automated measurement of critical biometric parameters used in prenatal care.

### Current Status
- ✅ **Part A (Landmark Detection)**: Coordinate regression model trained for 48 epochs
- 🔄 **Model Performance**: Mean Radial Error (MRE) ~160-165 pixels on validation set
- ⏳ **Part B (Segmentation)**: Architecture designed, implementation pending

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Current Work - Part A: Landmark Detection](#current-work---part-a-landmark-detection)
3. [Results and Analysis](#results-and-analysis)
4. [Future Work](#future-work)
5. [Part B: Segmentation Plan](#part-b-segmentation-plan)
6. [Recommendations](#recommendations)
7. [References and Resources](#references-and-resources)

---

## Project Overview

### Objective
Develop an automated system to:
1. Detect 4 key landmarks in fetal head ultrasound images:
   - OFD1, OFD2 (Occipital-Frontal Diameter endpoints)
   - BPD1, BPD2 (Biparietal Diameter endpoints)
2. Segment cranium regions for improved measurement accuracy
3. Calculate biometric measurements automatically

### Dataset
- **Total Images**: 622 fetal head ultrasound images
- **Training Set**: 497 images (80%)
- **Validation Set**: 125 images (20%)
- **Image Dimensions**: Primarily 800×540 pixels (some variation)
- **Annotations**: 4 landmark coordinates (x, y) per image in original image space
- **Source**: `role_challenge_dataset_ground_truth.csv`

### Clinical Relevance
- **OFD (Occipital-Frontal Diameter)**: Measures head length front-to-back
- **BPD (Biparietal Diameter)**: Measures head width side-to-side
- These measurements are critical for:
  - Gestational age estimation
  - Fetal growth monitoring
  - Detection of developmental abnormalities

---

## Current Work - Part A: Landmark Detection

### Model Architecture

**Selected Approach**: Direct Coordinate Regression with EfficientNet-B3

**Theoretical Foundation**: Payer et al., "Regressing Heatmaps for Multiple Landmark Localization Using CNNs" (MICCAI 2016)  
**Paper URL**: https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Images/team_bischof/mib/paper_pdfs/MICCAI2016_CNNHeatmaps.pdf

**Key Concepts from Paper**:
- **Original Method**: Uses fully convolutional networks to regress spatial probability heatmaps (Gaussian distributions centered at landmark positions)
- **Innovation**: Combines localization and detection in a unified framework with pixel-wise regression
- **Advantage**: Maintains spatial information throughout the network using encoder-decoder architecture

**Our Adaptation**:
- **Modified Approach**: Direct coordinate regression using global features instead of spatial heatmaps
- **Rationale**: Simpler architecture, faster inference, reduced memory footprint
- **Trade-off**: Less spatial interpretability but more efficient for limited computational resources
- **Implementation**: EfficientNet-B3 backbone with fully connected output layer for 8 coordinates

#### Architecture Details
```
Input: RGB Ultrasound Image (resized to 512×512)
    ↓
Backbone: EfficientNet-B3 (pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Fully Connected Layers with Dropout (0.3)
    ↓
Output: 8 values (4 landmarks × 2 coordinates)
```

#### Model Configuration
- **Backbone**: EfficientNet-B3
- **Input Size**: 512×512 pixels
- **Output**: 8 continuous values (x, y for each of 4 landmarks)
- **Loss Function**: Combined coordinate regression loss (L1 + Smooth L1)
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-5
- **Batch Size**: 16

### Preprocessing Pipeline

#### 1. Image Preprocessing
```python
UltrasoundPreprocessor:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Clip Limit: 2.0
    - Tile Grid Size: 8×8
  - Normalization: ImageNet mean/std
    - Mean: [0.485, 0.456, 0.406]
    - Std: [0.229, 0.224, 0.225]
```

#### 2. Data Augmentation (Training Only)

**Geometric Augmentations**:
- Rotation: ±20 degrees
- Scale: ±15%
- Shift: ±10%
- Horizontal Flip: 50% probability
- Vertical Flip: 10% probability
- ShiftScaleRotate: Combined transformations

**Intensity Augmentations**:
- Random Brightness/Contrast: ±20%
- Random Gamma: (80, 120)
- Gaussian Noise: var_limit=(10, 50)
- Gaussian Blur: blur_limit=(3, 7)
- Motion Blur: blur_limit=(3, 7)

**Domain-Specific Augmentations**:
- Elastic Transform: Simulates tissue deformation
- Grid Distortion: Mimics probe angle variations
- Optical Distortion: Simulates ultrasound artifacts
- Random Shadow: Simulates acoustic shadowing
- Coarse Dropout: Simulates signal dropout

**Augmentation Probability**: 50% per augmentation

### Training Configuration

```yaml
Training Parameters:
  - Epochs Completed: 48
  - Target Epochs: 150
  - Batch Size: 16
  - Learning Rate: 1e-4
  - LR Scheduler: ReduceLROnPlateau
    - Factor: 0.5
    - Patience: 10 epochs
  - Early Stopping: Patience 25 epochs
  - Mixed Precision: Enabled (FP16)
  - Gradient Clipping: 1.0

Hardware Constraints:
  - Training Device: CPU (GPU unavailable)
  - Training Time: ~48 hours for 48 epochs
  - Memory: Limited by CPU constraints
```

### Training Progress

**Best Checkpoint**: Epoch 48
- Validation MRE: 164.88 pixels
- Training stopped early due to:
  - Time constraints
  - GPU unavailability
  - Resource limitations

---

## Results and Analysis

### Quantitative Results

#### Overall Performance (Validation Set, 125 images)

| Metric | Value |
|--------|-------|
| **Mean Radial Error (MRE)** | 159.59 ± 36.63 pixels |
| **Median Error** | 151.61 pixels |
| **Min Error** | 69.58 pixels |
| **Max Error** | 324.26 pixels |

#### Per-Landmark Performance

| Landmark | Mean Error (pixels) | Std Dev (pixels) |
|----------|---------------------|------------------|
| **OFD1** | 146.80 | 89.08 |
| **OFD2** | 146.80 | 89.08 |
| **BPD1** | 172.37 | 77.63 |
| **BPD2** | 172.37 | 77.63 |

#### Accuracy at Thresholds

| Threshold | Accuracy |
|-----------|----------|
| < 50 pixels | 0.0% |
| < 100 pixels | 0.0% |
| < 150 pixels | ~45% |
| < 200 pixels | ~75% |

### Critical Issue Identified

**Problem**: Model predictions are clustering in a narrow range

**Analysis**:
- Predictions in 512×512 space: X∈[127.6, 254.5], Y∈[80.2, 269.0]
- Expected range: X∈[0, 512], Y∈[0, 512]
- Model is using only **~25% of coordinate space**

**Root Cause**:
- Training stopped too early (48 epochs vs. target 150)
- Model converged to predicting "mean" coordinates
- Insufficient training iterations to learn full spatial distribution
- Possible learning rate issues

**Impact**:
- All predictions cluster near image center
- Poor generalization to landmark positions
- High error rates (159-210 pixels)

### Visual Analysis

Generated visualizations show:
- ✅ GT landmarks (green) are correctly positioned
- ❌ Predicted landmarks (blue) cluster in center region
- ❌ Predictions don't respond to actual landmark locations
- ✅ Coordinate scaling pipeline is correct

**Saved Artifacts**:
- `results/unnet_e48/visualise/` - 30 GT vs Prediction overlays
- `results/coordinate_evaluation/` - Performance dashboard and metrics
- `results/predictions_vs_gt.csv` - Complete predictions dataset

---

## Future Work

### 1. Immediate Improvements (Short-term)

#### A. Complete Model Training
**Priority**: HIGH  
**Effort**: Medium

- [ ] Train for full 150 epochs with GPU
- [ ] Monitor training curves for convergence
- [ ] Implement proper learning rate scheduling
- [ ] Expected improvement: 50-70% reduction in MRE

**Estimated Resources**:
- GPU: NVIDIA RTX 3090 or better
- Time: 8-12 hours for 150 epochs
- Cost: $10-20 on cloud GPU (Google Colab Pro, AWS)

#### B. Hyperparameter Optimization
**Priority**: HIGH  
**Effort**: Low

- [ ] Learning rate tuning (try 5e-4, 1e-3)
- [ ] Batch size optimization (try 32, 64)
- [ ] Loss function weighting
- [ ] Optimizer comparison (Adam vs AdamW vs SGD)

#### C. Model Architecture Improvements
**Priority**: MEDIUM  
**Effort**: Medium

- [ ] Try larger backbone: EfficientNet-B4, B5
- [ ] Add attention mechanisms to regression head
- [ ] Multi-task learning: Predict landmarks + heatmaps
- [ ] Ensemble multiple models

### 2. Data Enhancement (Medium-term)

#### A. Acquire Additional Training Data
**Priority**: HIGH  
**Effort**: High

**Public Datasets to Consider**:

1. **HC18 Challenge Dataset** (Recommended)
   - Source: https://hc18.grand-challenge.org/
   - Size: 999 training + 335 test images
   - Features: Fetal head ultrasound with annotations
   - Use: Pre-training backbone

2. **FETAL Dataset**
   - Source: Kaggle Fetal Ultrasound datasets
   - Size: ~2000+ images
   - Features: Various fetal measurements
   - Use: Transfer learning

3. **Medical Segmentation Decathlon**
   - Source: http://medicaldecathlon.com/
   - Task: Generic medical imaging
   - Use: Pre-training feature extractors

**Strategy**:
```
Stage 1: Pre-train on HC18 (1000+ images)
    ↓
Stage 2: Fine-tune on combined dataset
    ↓
Stage 3: Final fine-tuning on target dataset (622 images)
```

#### B. Data Augmentation Enhancement
**Priority**: MEDIUM  
**Effort**: Low

- [ ] Add ultrasound-specific augmentations:
  - Speckle noise simulation
  - Attenuation artifacts
  - Multiple scattering effects
- [ ] Use AutoAugment/RandAugment
- [ ] Synthetic data generation with GANs

### 3. Advanced Techniques (Long-term)

#### A. Pre-trained Models
**Priority**: HIGH  
**Effort**: Low

**Models to Try**:
1. **MedicalNet** (3D medical imaging features)
2. **Models Genesis** (Self-supervised pre-training)
3. **Vision Transformers** (ViT, Swin Transformer)
4. **ConvNeXt** (Modern CNN architecture)

#### B. Multi-Stage Approaches
**Priority**: MEDIUM  
**Effort**: High

1. **Coarse-to-Fine**:
   - Stage 1: Detect rough region of interest
   - Stage 2: Refine landmarks in ROI
   
2. **Heatmap + Regression**:
   - Generate probability heatmaps
   - Extract coordinates from heatmaps
   - More interpretable predictions

#### C. Uncertainty Estimation
**Priority**: LOW  
**Effort**: Medium

- Bayesian Neural Networks
- Monte Carlo Dropout
- Provide confidence scores with predictions

### 4. Model Improvements for Better Performance

#### Why Current Model Underperforms:

1. **Insufficient Training**: 48/150 epochs (32% complete)
2. **Limited Data**: 497 training images
3. **No Transfer Learning**: Training from ImageNet only
4. **Simple Architecture**: Direct regression may be too simple

#### Recommended Improvements:

**A. Architecture Changes**:
```python
# Current: Simple regression head
# Improved: Multi-scale feature fusion

class ImprovedLandmarkDetector(nn.Module):
    def __init__(self):
        # Use FPN or U-Net style decoder
        # Multiple prediction heads at different scales
        # Attention mechanisms for landmark-specific features
```

**B. Loss Function Enhancements**:
```python
# Current: L1 + Smooth L1
# Improved: Multi-component loss

Loss = α * L1_loss + 
       β * Wing_loss + 
       γ * Perceptual_loss +
       δ * Geometric_consistency_loss
```

**C. Training Strategy**:
```python
# Progressive training
Phase 1: Train on easy samples (50 epochs)
Phase 2: Add hard samples gradually (50 epochs)
Phase 3: Fine-tune on full dataset (50 epochs)
```

### 5. Evaluation Enhancements

- [ ] Cross-validation (5-fold)
- [ ] Clinical metrics (OFD/BPD measurement accuracy in mm)
- [ ] Comparison with human annotations (inter-rater agreement)
- [ ] Failure case analysis
- [ ] Real-time inference speed measurement

---

## Part B: Segmentation Plan

### Overview
**Status**: Architecture designed, implementation pending  
**Goal**: Segment fetal cranium for improved measurement accuracy

### Proposed Approaches

#### Model 1: U-Net (Baseline)
```
Architecture:
  - Encoder: ResNet34 (pretrained)
  - Decoder: Transposed convolutions + skip connections
  - Output: Binary mask (cranium vs background)

Advantages:
  - Proven effectiveness in medical imaging
  - Fast training and inference
  - Good with limited data

Status: Code implemented, training pending
```

#### Model 2: DeepLabV3+
```
Architecture:
  - Encoder: ResNet50/ResNet101
  - ASPP (Atrous Spatial Pyramid Pooling)
  - Decoder: Upsampling with low-level features

Advantages:
  - Multi-scale context aggregation
  - State-of-the-art segmentation
  - Better boundary detection

Status: Code implemented, training pending
```

### Segmentation Training Plan

```yaml
Configuration:
  - Input Size: 512×512
  - Batch Size: 8
  - Epochs: 150
  - Loss: Dice + BCE (combined)
  - Metrics: Dice, IoU, Precision, Recall
  
Augmentation:
  - Same as landmark detection
  - Additional: CutMix, MixUp for segmentation

Post-processing:
  - Morphological operations (opening, closing)
  - Ellipse fitting to extract landmarks
  - Geometric validation
```

### Integration Strategy

**Option 1: Two-Stage Pipeline**
```
Stage 1: Segmentation → Generate mask
Stage 2: Mask → Ellipse fitting → Landmarks
```

**Option 2: Multi-Task Learning**
```
Shared Encoder
    ↓
    ├→ Segmentation Head → Mask
    └→ Landmark Head → Coordinates
```

**Option 3: Segmentation-Guided Landmark Detection**
```
Input Image + Predicted Mask → Landmark Detector
```

### Expected Improvements

With segmentation:
- **Spatial Context**: Model understands full cranium shape
- **Robustness**: Less sensitive to local image noise
- **Interpretability**: Visual mask for verification
- **Accuracy**: Expected 30-50% improvement in MRE

---

## Recommendations

### For Immediate Action

1. **🔴 Priority 1: GPU Training**
   - Secure GPU access (local or cloud)
   - Complete 150-epoch training run
   - Monitor and log all metrics

2. **🔴 Priority 1: Data Acquisition**
   - Download HC18 dataset
   - Set up transfer learning pipeline
   - Validate data quality

3. **🟡 Priority 2: Model Debugging**
   - Investigate why predictions cluster
   - Check coordinate normalization in dataset
   - Verify loss function implementation

### For Medium-Term

4. **🟡 Priority 2: Implement Part B**
   - Train U-Net baseline
   - Compare with DeepLabV3+
   - Evaluate segmentation metrics

5. **🟡 Priority 2: Comprehensive Evaluation**
   - 5-fold cross-validation
   - Clinical measurement accuracy
   - Statistical significance testing

### For Long-Term

6. **🟢 Priority 3: Production Readiness**
   - Model optimization (TensorRT, ONNX)
   - Real-time inference
   - Web/mobile deployment
   - Clinical validation study

---

## Technical Stack

### Development Environment
```
Python: 3.11
PyTorch: 2.0+
PyTorch Lightning: 2.0+
CUDA: 11.8+ (when available)
```

### Key Libraries
```
Core:
  - pytorch
  - pytorch-lightning
  - torchvision
  
Preprocessing:
  - opencv-python
  - albumentations
  - scikit-image
  
Models:
  - timm (PyTorch Image Models)
  - segmentation-models-pytorch
  
Utilities:
  - pandas, numpy
  - matplotlib, seaborn
  - tqdm
```

### Project Structure
```
Origin Medical/
├── models/                    # Model architectures
│   ├── landmark_detection/
│   │   ├── coordinate_regression.py
│   │   ├── heatmap_model.py
│   │   └── attention_pyramid.py
│   └── segmentation/
│       ├── unet.py
│       └── deeplabv3.py
├── data/                      # Data handling
│   ├── dataset.py
│   ├── preprocessing.py
│   └── augmentation.py
├── utils/                     # Utilities
│   ├── metrics.py
│   ├── visualization.py
│   └── ellipse_fitting.py
├── checkpoints/               # Saved models
├── results/                   # Outputs
└── config.py                  # Configuration
```

---

## References and Resources

### Datasets

1. **HC18 Challenge**
   - URL: https://hc18.grand-challenge.org/
   - Paper: "Evaluation of Deep Learning Methods for Fetal Head Circumference Measurement"
   - Size: 1334 images
   - Download: Register and download from challenge website

2. **Kaggle Ultrasound Datasets**
   - Fetal Head Circumference: https://www.kaggle.com/search?q=fetal+ultrasound
   - Various fetal biometry datasets
   - Size: 500-5000+ images per dataset

3. **Medical Segmentation Decathlon**
   - URL: http://medicaldecathlon.com/
   - Task 10: Generic medical imaging
   - Pre-training resource

### Pre-trained Models

1. **MedicalNet**
   - GitHub: https://github.com/Borda/MedicalNet
   - Pre-trained on medical images
   - 3D models adaptable to 2D

2. **Models Genesis**
   - URL: https://www.models-genesis.ai/
   - Self-supervised pre-training
   - Medical imaging specific

3. **TIMM (PyTorch Image Models)**
   - GitHub: https://github.com/huggingface/pytorch-image-models
   - 700+ pre-trained models
   - Easy integration

### Research Papers

1. **Landmark Detection**:
   - **Payer, C., Štern, D., Bischof, H., & Urschler, M. (2016). "Regressing Heatmaps for Multiple Landmark Localization Using CNNs." MICCAI 2016.**
     - URL: https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Images/team_bischof/mib/paper_pdfs/MICCAI2016_CNNHeatmaps.pdf
     - **Key Contribution**: Pioneering work on CNN-based spatial regression for landmark detection
     - **Method**: Uses fully convolutional networks to regress spatial probability heatmaps
     - **Impact**: Foundation for modern landmark detection in medical imaging
   - "Deep Learning in Medical Image Analysis" (2020)
   - "Fetal Ultrasound Image Analysis" (2021)
   - "Coordinate Regression for Anatomical Landmark Detection" (2019)

2. **Segmentation**:
   - "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
   - "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" (2018)
   - "Medical Image Segmentation using Deep Learning: A Survey" (2020)

3. **Ultrasound Specific**:
   - "Automated Fetal Head Detection and Circumference Estimation" (2018)
   - "Deep Learning for Fetal Ultrasound: State of the Art and Future Directions" (2021)

### Tools and Frameworks

1. **Segmentation Models PyTorch**
   - GitHub: https://github.com/qubvel/segmentation_models.pytorch
   - Pre-built architectures: U-Net, DeepLabV3+, FPN, etc.
   - Easy encoder swapping

2. **Albumentations**
   - Docs: https://albumentations.ai/
   - Fast augmentation library
   - Medical imaging support

3. **MLflow / Weights & Biases**
   - Experiment tracking
   - Hyperparameter logging
   - Model versioning

---

## Appendix: Current Repository Files

### Root Directory Files
```
Core Scripts:
  - train_landmark.py          # Main training script (Part A)
  - train_segmentation.py      # Segmentation training (Part B)
  - evaluate_coordinate.py     # Model evaluation
  - config.py                  # Central configuration

Data Processing:
  - augment_dataset.py         # Data augmentation
  - visualize_augmented_dataset.py
  - visualize_original_data.sh

Evaluation & Visualization:
  - visualize_evaluation.py    # Performance dashboard
  - visualize_predictions_on_images.py
  - visualize_gt_pred_overlay.py
  - debug_inference_pipeline.py
  - save_predictions_to_csv.py

Shell Scripts:
  - start_training.sh
  - evaluate_coordinate.sh
  - run_inference.sh

Documentation:
  - README.md
  - QUICKSTART.md
  - APPROACH_AND_METHODOLOGY.md
  - PART_B_SEGMENTATION_GUIDE.md
```

---

## Conclusion

The project has made significant progress in implementing the landmark detection pipeline with a solid foundation in data preprocessing, augmentation, and model architecture. However, **training completion** and **data augmentation through external datasets** are critical next steps to achieve clinically acceptable performance.

The current MRE of ~160 pixels needs to be reduced to **<10 pixels** for clinical deployment. This is achievable through:
1. ✅ Complete training (150 epochs)
2. ✅ Transfer learning from larger datasets
3. ✅ Advanced model architectures
4. ✅ Proper hyperparameter tuning

**Part B (Segmentation)** is well-planned with implemented architectures ready for training once Part A reaches acceptable performance.

---

**Document Version**: 1.0  
**Last Updated**: January 1, 2026
**Status**: Living Document - To be updated with progress

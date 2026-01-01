# UNet Segmentation Model Evaluation Guide

## Overview
I've created an evaluation script that will test your UNet model and calculate accuracy metrics on the validation data.

## What You Need

### 1. A Trained UNet Model
First, you need to train your UNet model if you haven't already:

```bash
# Train UNet with default settings (ResNet34 backbone)
python train_segmentation.py --model unet --epochs 150 --batch_size 8

# Or with different backbone
python train_segmentation.py --model unet --encoder resnet50 --epochs 150 --batch_size 8
```

The trained model checkpoint will be saved in `checkpoints/unet/`

### 2. Run Evaluation

Once you have a trained checkpoint, evaluate it:

```bash
# Method 1: Using the shell script (easiest)
./evaluate_unet.sh checkpoints/unet/unet_resnet34_XX_X.XXXX.ckpt

# Method 2: Direct Python command
python evaluate_segmentation.py --checkpoint checkpoints/unet/unet_resnet34_XX_X.XXXX.ckpt --save_visualizations
```

## Evaluation Metrics

The evaluation script will calculate:

### Segmentation Metrics:
- **Dice Score**: Measures overlap between predicted and ground truth masks (0-1, higher is better)
- **IoU (Intersection over Union)**: Similar to Dice, measures segmentation quality (0-1, higher is better)
- **Precision**: How many predicted pixels are correct (0-1, higher is better)
- **Recall**: How many ground truth pixels were detected (0-1, higher is better)

### Landmark Metrics (if available):
- **Mean Radial Error (MRE)**: Average distance error in pixels for landmarks extracted from segmentation

## Output Files

The evaluation creates:

1. **Terminal Output**: Summary statistics with mean ± std for all metrics
2. **evaluation_results.csv**: Detailed per-image metrics
3. **evaluation_summary.csv**: Aggregate statistics
4. **Visualization Images**: Side-by-side comparison of predictions vs ground truth (first 20 samples)

All outputs are saved in: `results/segmentation_evaluation/`

## Example Output

```
============================================
EVALUATION RESULTS
============================================

Segmentation Metrics:
  Dice Score:  0.9234 ± 0.0156
  IoU:         0.8591 ± 0.0245
  Precision:   0.9401 ± 0.0189
  Recall:      0.9076 ± 0.0234

Landmark Detection (from segmentation):
  Mean Radial Error: 3.45 ± 1.23 pixels

============================================
```

## Current Status

**Note**: I checked your checkpoints directory and you don't have any UNet segmentation models trained yet. You only have:
- Coordinate regression models (EfficientNet-B3)
- Heatmap models (ResNet50)

## Next Steps

1. **Train the UNet model first**:
   ```bash
   python train_segmentation.py --model unet --epochs 150 --batch_size 8
   ```

2. **Wait for training to complete** (it will save checkpoints in `checkpoints/unet/`)

3. **Run evaluation** on the best checkpoint:
   ```bash
   ./evaluate_unet.sh checkpoints/unet/unet_resnet34_best.ckpt
   ```

## Quick Training Command

If you want to start training now:

```bash
# Make sure you're in the virtual environment
source originMedical/bin/activate

# Start training
python train_segmentation.py --model unet --encoder resnet34 --epochs 150 --batch_size 8 --lr 0.0001
```

## Troubleshooting

**If you get an error about missing dependencies:**
```bash
pip install segmentation-models-pytorch
```

**If you want to test with a smaller number of epochs first:**
```bash
python train_segmentation.py --model unet --epochs 10 --batch_size 4
```

**To monitor training progress:**
```bash
tensorboard --logdir logs/
```

## Questions?

- **Q: How long does training take?**  
  A: Depends on your hardware. With GPU: ~2-3 hours for 150 epochs. With CPU: ~10-20 hours.

- **Q: What's a good Dice score?**  
  A: For medical image segmentation, 0.85+ is good, 0.90+ is excellent.

- **Q: Can I evaluate during training?**  
  A: Yes! Training automatically evaluates on validation data each epoch and logs the metrics.

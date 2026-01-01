#!/bin/bash

# Script to run inference and visualize predictions

echo "=========================================="
echo "LANDMARK DETECTION INFERENCE"
echo "=========================================="
echo ""

# Activate virtual environment
source originMedical/bin/activate

# Find the best checkpoint
CHECKPOINT=$(ls -t checkpoints/heatmap/best/*.ckpt 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No checkpoint found in checkpoints/heatmap/best/"
    echo "Please train a model first using: ./start_training.sh"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
echo ""
echo "Running inference on 10 validation samples..."
echo ""

# Run inference on validation set
python inference_landmark.py \
  --checkpoint "$CHECKPOINT" \
  --csv_path data/augmented/augmented_ground_truth.csv \
  --images_dir data/augmented/images \
  --num_samples 10 \
  --save_dir output/inference

echo ""
echo "=========================================="
echo "Inference complete!"
echo "Results saved to: output/inference/"
echo "=========================================="
echo ""
echo "To run on a single image, use:"
echo "  python inference_landmark.py \\"
echo "    --checkpoint $CHECKPOINT \\"
echo "    --image_path path/to/image.png"

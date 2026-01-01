#!/bin/bash

# Script to evaluate coordinate regression model on validation data

echo "============================================"
echo "Coordinate Regression Model Evaluation"
echo "============================================"
echo ""

# Check if checkpoint path is provided
if [ -z "$1" ]; then
    echo "Usage: ./evaluate_coordinate.sh <path_to_checkpoint>"
    echo ""
    echo "Example:"
    echo "  ./evaluate_coordinate.sh checkpoints/coordinate/best/coordinate_efficientnet_b3_best.ckpt"
    echo ""
    echo "Available checkpoints:"
    find checkpoints/coordinate -name "*.ckpt" 2>/dev/null
    exit 1
fi

CHECKPOINT_PATH=$1

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    find checkpoints/coordinate -name "*.ckpt" 2>/dev/null
    exit 1
fi

echo "Checkpoint: $CHECKPOINT_PATH"
echo ""

# Run evaluation with visualizations
python evaluate_coordinate.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --save_visualizations

echo ""
echo "Evaluation complete!"
echo "Results saved in: results/coordinate_evaluation/"

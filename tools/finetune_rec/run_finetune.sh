#!/bin/bash
# Example script to finetune PP-OCRv5 mobile recognition model on license plate data
#
# Usage:
#   ./run_finetune.sh
#
# Or customize with environment variables:
#   IMAGES_DIR=/path/to/images CSV_FILE=/path/to/labels.csv OUTPUT_DIR=./output ./run_finetune.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PADDLEOCR_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths (can be overridden with environment variables)
IMAGES_DIR="${IMAGES_DIR:-../smartcity-cvfm-computer-vision/license_plate/datasets/pictures/originals/dataset06}"
CSV_FILE="${CSV_FILE:-../smartcity-cvfm-computer-vision/license_plate/datasets/ground_truth/dataset06_ground_truth.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/license_plate_rec}"

# Training parameters
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
GPUS="${GPUS:-0}"

echo "============================================================"
echo "PP-OCRv5 Mobile Recognition Finetuning"
echo "============================================================"
echo "Images dir:    $IMAGES_DIR"
echo "CSV file:      $CSV_FILE"
echo "Output dir:    $OUTPUT_DIR"
echo "Epochs:        $EPOCHS"
echo "Batch size:    $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "GPUs:          $GPUS"
echo "============================================================"

cd "$PADDLEOCR_ROOT"

python tools/finetune_rec/finetune.py \
    --images_dir "$IMAGES_DIR" \
    --ground_truth_csv "$CSV_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --epoch_num "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --gpus "$GPUS"

#!/bin/bash
# Script for generating AI essays using trained CLM model
# Usage: ./run_generate.sh [model_path] [num_essays] [output_file]

cd /qwarium/home/d.a.lanovenko/llm-detect-ai

# Default values
MODEL_PATH=${1:-/tmp/models/r_clm_v2/last}
NUM_ESSAYS=${2:-10}
OUTPUT_FILE=${3:-generated_essays.csv}

echo "=============================================="
echo "AI Essay Generation Script"
echo "=============================================="
echo "Model path: $MODEL_PATH"
echo "Number of essays: $NUM_ESSAYS"
echo "Output file: $OUTPUT_FILE"
echo "=============================================="

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    echo "Please wait for training to complete or specify a different path."
    exit 1
fi

# Run generation
python3 ./code/r_clm/generate_text.py \
    --model_path "$MODEL_PATH" \
    --num_essays "$NUM_ESSAYS" \
    --output "$OUTPUT_FILE" \
    --use_8bit

echo "=============================================="
echo "Generation complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "=============================================="

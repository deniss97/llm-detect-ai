#!/bin/bash
# Script for generating essay variations from existing texts
# Usage: ./run_generate_from_existing.sh [input_file] [mode] [model_path] [output_file]

cd /qwarium/home/d.a.lanovenko/llm-detect-ai

# Default values
INPUT_FILE=${1:-datasets/persuade_2.0_human_scores_demo_id_github.csv}
MODE=${2:-modify}  # modify, rewrite, continue
MODEL_PATH=${3:-/tmp/models/r_clm_v2/last}
OUTPUT_FILE=${4:-modified_essays.csv}
NUM_ESSAYS=${5:-10}

echo "=============================================="
echo "AI Essay Variation Generation Script"
echo "=============================================="
echo "Input file: $INPUT_FILE"
echo "Mode: $MODE"
echo "Model path: $MODEL_PATH"
echo "Output file: $OUTPUT_FILE"
echo "Number of essays to process: $NUM_ESSAYS"
echo "=============================================="

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    echo "Please wait for training to complete or specify a different path."
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Run generation
python3 ./code/r_clm/generate_from_existing.py \
    --model_path "$MODEL_PATH" \
    --input "$INPUT_FILE" \
    --mode "$MODE" \
    --output "$OUTPUT_FILE" \
    --num_essays "$NUM_ESSAYS" \
    --use_8bit

echo "=============================================="
echo "Generation complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "=============================================="

#!/bin/bash
# Script for translating Russian essays to English and generating AI variations
# Usage: ./run_translate_and_generate.sh [input_file] [model_path] [max_essays]

cd /qwarium/home/d.a.lanovenko/llm-detect-ai

# Default values
INPUT_FILE=${1:-datasets/Датасет.csv}
MODEL_PATH=${2:-/tmp/models/r_clm_v2/last}
MAX_ESSAYS=${3:-100}  # Process 100 essays by default for testing

TRANSLATED_OUTPUT="datasets/translated_essays.csv"
GENERATED_OUTPUT="datasets/generated_variations.csv"

echo "=============================================="
echo "Russian Essays Translation & AI Generation"
echo "=============================================="
echo "Input file: $INPUT_FILE"
echo "Model path: $MODEL_PATH"
echo "Max essays to process: $MAX_ESSAYS"
echo "Translated output: $TRANSLATED_OUTPUT"
echo "Generated output: $GENERATED_OUTPUT"
echo "=============================================="

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    echo "Please ensure r_clm model is trained and available."
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    exit 1
fi

# Check if deep-translator is installed
python3 -c "from deep_translator import GoogleTranslator" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: deep-translator not installed. Installing..."
    pip3 install deep-translator --quiet
fi

# Run translation and generation
python3 ./code/r_clm/translate_and_generate.py \
    --input "$INPUT_FILE" \
    --model_path "$MODEL_PATH" \
    --translated_output "$TRANSLATED_OUTPUT" \
    --generated_output "$GENERATED_OUTPUT" \
    --max_essays "$MAX_ESSAYS" \
    --num_variations 1 \
    --use_8bit

echo "=============================================="
echo "Processing complete!"
echo "Translated essays: $TRANSLATED_OUTPUT"
echo "Generated variations: $GENERATED_OUTPUT"
echo "=============================================="

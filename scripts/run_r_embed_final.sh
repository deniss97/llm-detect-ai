#!/bin/bash
# Script to train r_embed model on final_dataset.csv (v2 - balanced dataset)

set -e

echo "=========================================="
echo "Training r_embed on final_dataset.csv (v2)"
echo "=========================================="

cd /qwarium/home/d.a.lanovenko/llm-detect-ai

# Run training
python3 code/train_r_embed.py \
    --config-name conf_r_embed_final

echo "=========================================="
echo "Training completed!"
echo "Model saved to: /qwarium/home/d.a.lanovenko/models/r_embed_final_dataset"
echo "=========================================="

#!/bin/bash
# Training script for r_detect_competition model
# DEPRECATED: Use scripts/run_r_detect.sh instead
# This script is kept for backward compatibility

cd /qwarium/home/d.a.lanovenko/llm-detect-ai

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HOME=/tmp/huggingface_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache

# Create model output directory (persistent storage)
MODEL_DIR="/qwarium/home/d.a.lanovenko/models/r_detect_competition"
mkdir -p "$MODEL_DIR"

# Run training
python3 ./code/train_r_detect.py \
    --config-name conf_r_detect_competition \
    use_wandb=false \
    outputs.model_dir="$MODEL_DIR" \
    hydra.run.dir=/tmp/hydra_competition/%Y-%m-%d/%H-%M-%S \
    2>&1 | tee /qwarium/home/d.a.lanovenko/llm-detect-ai/logs/train_competition.log

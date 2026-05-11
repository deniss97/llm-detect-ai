#!/bin/bash
# Training script for r_detect on final_dataset (Russian essays)
# Usage: /ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect_final.sh
# Example: ./run_r_detect_final.sh

cd /qwarium/home/d.a.lanovenko/llm-detect-ai

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HOME=/tmp/huggingface_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache

# Config for final_dataset
CONFIG_NAME="conf_r_detect_final"

# Create model output directory (persistent storage)
MODEL_DIR="/qwarium/home/d.a.lanovenko/models/r_detect_final_dataset"
mkdir -p "$MODEL_DIR"

# Create logs directory
mkdir -p /qwarium/home/d.a.lanovenko/llm-detect-ai/logs

# Run training
python3 ./code/train_r_detect.py \
    --config-name "$CONFIG_NAME" \
    use_wandb=false \
    outputs.model_dir="$MODEL_DIR" \
    hydra.run.dir=/tmp/hydra_r_detect_final/%Y-%m-%d/%H-%M-%S \
    2>&1 | tee /qwarium/home/d.a.lanovenko/llm-detect-ai/logs/train_r_detect_final.log

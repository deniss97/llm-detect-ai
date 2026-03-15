#!/bin/bash
# Training script for r_dpo (DPO - Direct Preference Optimization)
# This script can be run with qwarium-agent proc spawn for persistent execution
# Usage: /ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_dpo.sh [config_name]
# Example: ./run_r_dpo.sh conf_r_dpo

cd /qwarium/home/d.a.lanovenko/llm-detect-ai

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HOME=/tmp/huggingface_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache

# Default config
CONFIG_NAME=${1:-conf_r_dpo}

# Create model output directory (persistent storage)
MODEL_DIR="/qwarium/home/d.a.lanovenko/models/r_dpo_${CONFIG_NAME}"
mkdir -p "$MODEL_DIR"

# Run training
python3 ./code/train_r_dpo.py \
    --config-name "$CONFIG_NAME" \
    use_wandb=false \
    outputs.model_dir="$MODEL_DIR" \
    hydra.run.dir=/tmp/hydra_r_dpo/%Y-%m-%d/%H-%M-%S \
    2>&1 | tee /qwarium/home/d.a.lanovenko/llm-detect-ai/logs/train_r_dpo_${CONFIG_NAME}.log

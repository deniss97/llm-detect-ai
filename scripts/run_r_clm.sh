#!/bin/bash
# Training script for r_clm (CLM model training)
# This script can be run with qwarium-agent proc spawn for persistent execution
# Usage: /ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_clm.sh [config_name]
# Example: ./run_r_clm.sh conf_r_clm

cd /qwarium/home/d.a.lanovenko/llm-detect-ai

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HOME=/tmp/huggingface_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache

# Default config
CONFIG_NAME=${1:-conf_r_clm}

# Run training
python3 ./code/train_r_clm.py \
    --config-name "$CONFIG_NAME" \
    use_wandb=false \
    hydra.run.dir=/tmp/hydra_r_clm/%Y-%m-%d/%H-%M-%S \
    2>&1 | tee /qwarium/home/d.a.lanovenko/llm-detect-ai/logs/train_r_clm_${CONFIG_NAME}.log

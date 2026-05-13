#!/bin/bash
# Запуск обучения T-lite модели с использованием /tmp для кэша

export HF_HOME=/tmp/huggingface_cache
export HF_HUB_CACHE=/tmp/huggingface_cache/hub
export TRANSFORMERS_CACHE=/tmp/transformers_cache

mkdir -p /tmp/huggingface_cache
mkdir -p /tmp/transformers_cache

cd /qwarium/home/d.a.lanovenko/llm-detect-ai

python3 code/train_r_detect_t_lite.py 2>&1 | tee logs/train_r_detect_t_lite_v2.log

#!/bin/bash
# Обучение Qwen3-8B Detection модели на final_dataset.csv
# Архитектура V2: Qwen3-8B + LoRA + DoRA

set -e

echo "=============================================================================="
echo "ОБУЧЕНИЕ Qwen3-8B Detection модели (V2)"
echo "=============================================================================="

# === Окружение ===
export HF_HOME="/tmp/hf_cache"
export CUDA_VISIBLE_DEVICES=0

LOG_DIR="/qwarium/home/d.a.lanovenko/llm-detect-ai/logs"
mkdir -p "$LOG_DIR"

echo ""
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo ""

# === Запуск обучения ===
cd /qwarium/home/d.a.lanovenko/llm-detect-ai

python3 code/train_r_detect_qwen3.py \
    --config-name conf_r_detect_qwen3 \
    2>&1 | tee "$LOG_DIR/train_r_detect_qwen3.log"

echo ""
echo "=============================================================================="
echo "ОБУЧЕНИЕ ЗАВЕРШЕНО"
echo "=============================================================================="
echo "Лог: $LOG_DIR/train_r_detect_qwen3.log"
echo "Модель: /tmp/llm_cache/models/r_detect_qwen3"

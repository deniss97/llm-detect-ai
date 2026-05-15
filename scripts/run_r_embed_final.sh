#!/bin/bash
# Скрипт для обучения Embedding модели (USER-bge-m3) на триплетах
# Использует правильные сплиты с final_prepared_embed_ranking/

set -e

echo "============================================================"
echo "Обучение Embedding модели (USER-bge-m3)"
echo "============================================================"

# Установка переменных окружения
export HF_HOME="/tmp/hf_home"
export HF_HUB_CACHE="/tmp/hf_home/huggingface/hub"
export TRANSFORMERS_CACHE="/tmp/hf_home/transformers"
export MODEL_OUTPUT_DIR="/tmp/llm_cache/models/r_embed_final"

# Создание директорий
mkdir -p /tmp/llm_cache/models
mkdir -p /qwarium/home/d.a.lanovenko/llm-detect-ai/logs

echo "HF_HOME: $HF_HOME"
echo "MODEL_OUTPUT_DIR: $MODEL_OUTPUT_DIR"
echo ""

# Запуск обучения
cd /qwarium/home/d.a.lanovenko/llm-detect-ai
python3 code/train_r_embed_final.py

echo ""
echo "============================================================"
echo "✅ Обучение завершено!"
echo "Модель сохранена в: $MODEL_OUTPUT_DIR"
echo "============================================================"

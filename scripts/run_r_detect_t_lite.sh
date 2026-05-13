#!/bin/bash
# Скрипт обучения Detection модели на базе T-lite-it-1.0
# Использует /tmp для кэша моделей

set -e

echo "=== Обучение Detection модели (T-lite-it-1.0) ==="
echo "Дата: $(date)"

# Настройка окружения
export HF_HOME=/tmp/huggingface_cache
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "HF_HOME: $HF_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Проверка GPU
echo "=== Проверка GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Проверка наличия модели
if [ ! -d "$HF_HOME/t-tech--T-lite-it-1.0" ]; then
    echo "Загрузка модели T-lite-it-1.0..."
    huggingface-cli download t-tech/T-lite-it-1.0 --local-dir "$HF_HOME/t-tech--T-lite-it-1.0"
else
    echo "✅ Модель T-lite-it-1.0 уже загружена"
fi

# Создание директории для логов
mkdir -p /qwarium/home/d.a.lanovenko/llm-detect-ai/logs

# Запуск обучения
cd /qwarium/home/d.a.lanovenko/llm-detect-ai

echo ""
echo "=== Запуск обучения ==="
python3 code/train_r_detect_t_lite.py 2>&1 | tee logs/train_r_detect_t_lite.log

echo ""
echo "=== Обучение завершено ==="
echo "Логи: logs/train_r_detect_t_lite.log"

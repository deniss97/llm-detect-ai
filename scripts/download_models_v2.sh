#!/bin/bash
# Скрипт загрузки моделей для архитектуры V2
# Все модели загружаются в /tmp (2.3TB доступно)

set -e

echo "=== Настройка окружения для V2 архитектуры ==="

# Создаем директорию для кэша в /tmp
export HF_HOME=/tmp/huggingface_cache
mkdir -p $HF_HOME
echo "HF_HOME установлен в: $HF_HOME"

# Модели для загрузки
MODELS=(
    "t-tech/T-lite-it-1.0"
    "Qwen/Qwen3-8B"
    "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"
    "deepvk/USER-bge-m3"
    "BAAI/bge-reranker-v2-m3"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-7B-Instruct"
)

echo ""
echo "=== Загрузка моделей ==="
for model in "${MODELS[@]}"; do
    echo "Загрузка: $model"
    huggingface-cli download "$model" --local-dir "$HF_HOME/$model" || echo "Ошибка загрузки $model, продолжаем..."
done

echo ""
echo "=== Проверка загруженных моделей ==="
du -sh $HF_HOME/*

echo ""
echo "=== Готово! Модели загружены в $HF_HOME ==="
echo "Для использования установите: export HF_HOME=$HF_HOME"

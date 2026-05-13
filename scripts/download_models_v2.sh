#!/bin/bash
# Скрипт для загрузки моделей ARCHITECTURE_V2.md
# Использует HF mirror для загрузки из РФ

set -e

echo "=========================================="
echo "📥 Загрузка моделей для ARCHITECTURE_V2.md"
echo "=========================================="

# Используем HF mirror для загрузки из РФ
export HF_ENDPOINT=https://hf-mirror.com

# Проверяем наличие huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli не найден!"
    echo "Установите: pip install huggingface_hub"
    exit 1
fi

# Создаем директорию для моделей
MODEL_DIR="${HF_HOME:-/qwarium/home/d.a.lanovenko/.cache/huggingface/hub}"
echo "📂 Модели будут загружены в: $MODEL_DIR"

# Проверяем место
echo ""
echo "📊 Доступное место:"
df -h "$MODEL_DIR" | tail -1

echo ""
echo "⚠️  Требуется ~70-80 GB свободного места"
echo ""

# Список моделей для загрузки
MODELS=(
    "t-tech/T-lite-it-1.0"
    "Qwen/Qwen3-8B"
    "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"
    "deepvk/USER-bge-m3"
    "BAAI/bge-reranker-v2-m3"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-7B-Instruct"
)

echo "📋 Список моделей для загрузки:"
for i in "${!MODELS[@]}"; do
    echo "   $((i+1)). ${MODELS[$i]}"
done

echo ""
read -p "Начать загрузку? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Загрузка отменена"
    exit 0
fi

# Загрузка каждой модели
for model in "${MODELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "📥 Загрузка: $model"
    echo "=========================================="
    
    # Проверяем, загружена ли уже модель
    if [ -d "$MODEL_DIR/$(echo $model | tr '/' '_')" ]; then
        echo "✅ Модель уже загружена (пропускаем)"
        continue
    fi
    
    # Загружаем модель
    huggingface-cli download "$model" --local-dir "$MODEL_DIR/$(echo $model | tr '/' '_')"
    
    echo "✅ Модель загружена: $model"
done

echo ""
echo "=========================================="
echo "✅ Все модели загружены!"
echo "=========================================="
echo ""
echo "📊 Итоговое использование диска:"
df -h "$MODEL_DIR" | tail -1

echo ""
echo "💡 Следующий шаг: создание конфигов для обучения"
echo "   См. TRAINING_PLAN.md (Этапы 2.5.1 - 2.5.7)"
echo ""

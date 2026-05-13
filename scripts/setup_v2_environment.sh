#!/bin/bash
# Скрипт для настройки окружения и загрузки моделей V2.0 с использованием /tmp для кэша

set -e

echo "=========================================="
echo "🔧 Настройка окружения для ARCHITECTURE_V2.md"
echo "=========================================="

# === 1. Настройка HF_HOME в /tmp ===
export HF_HOME=/tmp/huggingface_cache
export HF_ENDPOINT=https://hf-mirror.com

echo "📂 HF_HOME установлен в: $HF_HOME"
mkdir -p $HF_HOME

# === 2. Проверка места в /tmp ===
echo ""
echo "📊 Доступное место в /tmp:"
df -h /tmp | tail -1

# === 3. Проверка установленных пакетов ===
echo ""
echo "📦 Проверка установленных пакетов..."

REQUIRED_PACKAGES=(
    "torch"
    "transformers"
    "peft"
    "accelerate"
    "sentence-transformers"
    "bitsandbytes"
    "faiss-gpu"
    "catboost"
    "pymorphy3"
)

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo "   ✅ $pkg установлен"
    else
        echo "   ⚠️  $pkg НЕ установлен"
    fi
done

# === 4. Проверка FlashAttention ===
echo ""
echo "🔍 Проверка FlashAttention..."
if python3 -c "import flash_attn" 2>/dev/null; then
    echo "   ✅ FlashAttention установлен"
else
    echo "   ⚠️  FlashAttention НЕ установлен"
    echo "   Для установки: pip install flash-attn --no-build-isolation"
fi

# === 5. Создание директорий для моделей ===
echo ""
echo "📂 Создание директорий для моделей..."

MODEL_BASE="/qwarium/home/d.a.lanovenko/models"
mkdir -p $MODEL_BASE

V2_MODELS=(
    "r_detect_t_lite"
    "r_detect_qwen3"
    "r_detect_vikhr"
    "r_embed_bge_m3"
    "r_ranking_bge_v2"
)

for model_dir in "${V2_MODELS[@]}"; do
    mkdir -p "$MODEL_BASE/$model_dir"
    echo "   ✅ Создано: $MODEL_BASE/$model_dir"
done

# === 6. Список моделей для загрузки ===
echo ""
echo "📋 Модели для загрузки (через HF mirror):"
echo ""

MODELS_INFO=(
    "t-tech/T-lite-it-1.0:~14GB:SOTA на русском"
    "Qwen/Qwen3-8B:~16GB:Мультиязычный"
    "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24:~24GB:Российский проект"
    "deepvk/USER-bge-m3:~1GB:Embedding"
    "BAAI/bge-reranker-v2-m3:~1GB:Ranking"
    "Qwen/Qwen2.5-7B:~14GB:Для Binoculars"
    "Qwen/Qwen2.5-7B-Instruct:~14GB:Для Binoculars"
)

for info in "${MODELS_INFO[@]}"; do
    IFS=':' read -r model size desc <<< "$info"
    printf "   %-50s %8s  %s\n" "$model" "$size" "$desc"
done

echo ""
echo "⚠️  Общий размер: ~84 GB"
echo "💡 Кэш будет сохранён в /tmp/huggingface_cache"
echo ""

# === 7. Предложение начать загрузку ===
read -p "Начать загрузку моделей? (y/n): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Загрузка отменена"
    exit 0
fi

# === 8. Загрузка моделей ===
echo ""
echo "=========================================="
echo "📥 Начало загрузки моделей..."
echo "=========================================="

MODELS=(
    "t-tech/T-lite-it-1.0"
    "Qwen/Qwen3-8B"
    "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"
    "deepvk/USER-bge-m3"
    "BAAI/bge-reranker-v2-m3"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-7B-Instruct"
)

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    echo ""
    echo "[$((i+1))/${#MODELS[@]}] Загрузка: $model"
    echo "=========================================="
    
    # Проверяем, загружена ли уже модель
    model_dir_name=$(echo $model | tr '/' '_')
    if [ -d "$HF_HOME/$model_dir_name" ]; then
        echo "✅ Модель уже загружена (пропускаем)"
        continue
    fi
    
    # Загружаем модель
    huggingface-cli download "$model" --local-dir "$HF_HOME/$model_dir_name"
    
    echo "✅ Модель загружена: $model"
done

echo ""
echo "=========================================="
echo "✅ Все модели загружены!"
echo "=========================================="
echo ""
echo "📊 Использование /tmp:"
df -h /tmp | tail -1
echo ""
echo "💡 Следующий шаг: создание конфигов для обучения"
echo "   См. TRAINING_PLAN.md (Этапы 2.5.1 - 2.5.7)"
echo ""

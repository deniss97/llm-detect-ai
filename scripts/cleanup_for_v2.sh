#!/bin/bash
# Скрипт для очистки места перед загрузкой моделей ARCHITECTURE_V2.md
# Освобождает ~70-80 GB для новых моделей

set -e

echo "=========================================="
echo "🧹 Очистка места для моделей V2.0"
echo "=========================================="

# Проверяем текущее место
echo ""
echo "📊 Текущее использование диска:"
df -h /qwarium/home/d.a.lanovenko

echo ""
echo "🔍 Анализ占用 места..."

# Проверяем размер моделей
if [ -d "/qwarium/home/d.a.lanovenko/models" ]; then
    echo ""
    echo "📦 Модели:"
    du -sh /qwarium/home/d.a.lanovenko/models/* 2>/dev/null | sort -rh || echo "  (пусто)"
fi

# Проверяем размер кэша transformers
if [ -d "/qwarium/home/d.a.lanovenko/.cache/huggingface" ]; then
    echo ""
    echo "💾 Кэш HuggingFace:"
    du -sh /qwarium/home/d.a.lanovenko/.cache/huggingface/* 2>/dev/null | sort -rh || echo "  (пусто)"
fi

# Проверяем логи
if [ -d "/qwarium/home/d.a.lanovenko/llm-detect-ai/logs" ]; then
    echo ""
    echo "📝 Логи:"
    du -sh /qwarium/home/d.a.lanovenko/llm-detect-ai/logs/* 2>/dev/null | sort -rh || echo "  (пусто)"
fi

echo ""
echo "=========================================="
echo "⚠️  ВНИМАНИЕ: Диск заполнен на 100%!"
echo "=========================================="
echo ""
echo "Рекомендуемые действия:"
echo ""
echo "1. Удалить старые модели (если не нужны):"
echo "   rm -rf /qwarium/home/d.a.lanovenko/models/r_embed_final_dataset"
echo "   rm -rf /qwarium/home/d.a.lanovenko/models/r_ranking_final_dataset"
echo "   → Освободит ~2.7 GB"
echo ""
echo "2. Очистить кэш HuggingFace:"
echo "   rm -rf /qwarium/home/d.a.lanovenko/.cache/huggingface/hub/*"
echo "   → Освободит ~10-50 GB (зависит от кэша)"
echo ""
echo "3. Удалить старые логи:"
echo "   rm -f /qwarium/home/d.a.lanovenko/llm-detect-ai/logs/*.log"
echo "   → Освободит ~100-500 MB"
echo ""
echo "4. Использовать /tmp для временных моделей:"
echo "   export HF_HOME=/tmp/huggingface"
echo "   → Модели будут скачиваться в /tmp (больше места)"
echo ""
echo "=========================================="
echo "📋 Для ARCHITECTURE_V2.md требуется:"
echo "   ~70-80 GB для 7 новых моделей:"
echo "   - t-tech/T-lite-it-1.0 (14 GB)"
echo "   - Qwen/Qwen3-8B (16 GB)"
echo "   - Vikhrmodels/Vikhr-Nemo-12B (24 GB)"
echo "   - deepvk/USER-bge-m3 (1 GB)"
echo "   - BAAI/bge-reranker-v2-m3 (1 GB)"
echo "   - Qwen/Qwen2.5-7B (14 GB)"
echo "   - Qwen/Qwen2.5-7B-Instruct (14 GB)"
echo "=========================================="

# Автоматическая очистка (раскомментировать при необходимости)
# echo ""
# echo "🗑️  Автоматическая очистка..."
# rm -rf /qwarium/home/d.a.lanovenko/models/r_embed_final_dataset
# rm -rf /qwarium/home/d.a.lanovenko/models/r_ranking_final_dataset
# echo "✅ Старые модели удалены"

echo ""
echo "💡 Запустите этот скрипт с раскомментированными командами"
echo "   для автоматической очистки, или выполните команды вручную."
echo ""

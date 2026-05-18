#!/bin/bash
# Установка окружения для проекта LLM Detect AI
# Версия Python: 3.10+

set -e

echo "🚀 Установка окружения для LLM Detect AI..."

# Создание виртуального окружения (если не существует)
if [ ! -d "venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3.10 -m venv venv
fi

# Активация виртуального окружения
echo "🔌 Активация виртуального окружения..."
source venv/bin/activate

# Обновление pip
echo "📦 Обновление pip..."
pip install --upgrade pip

# Установка зависимостей
echo "📦 Установка зависимостей из requirements.txt..."
pip install -r requirements.txt

echo "✅ Окружение успешно установлено!"
echo ""
echo "Для активации окружения выполните:"
echo "  source venv/bin/activate"
echo ""
echo "Для запуска обучения моделей см. README.md"

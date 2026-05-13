# V2 Architecture - Инструкция по запуску

## 📋 Обзор

Архитектура V2 использует современные LLM модели для детекции AI-генерации текстов:

### Модели V2:
1. **Detection**: T-lite-it-1.0 (Qwen2-based, SOTA на русском)
2. **Embedding**: USER-bge-m3 (универсальные эмбеддинги)
3. **Ranking**: BGE-reranker-v2-m3 (ранжирование)

## 🚀 Быстрый старт

### 1. Настройка окружения

```bash
# Установить HF_HOME в /tmp (для экономии места в домашней директории)
export HF_HOME=/tmp/huggingface_cache
export HF_ENDPOINT=https://hf-mirror.com  # Опционально: mirror для ускорения

# Запустить скрипт настройки
bash scripts/setup_v2_environment.sh
```

### 2. Загрузка моделей

Скрипт `setup_v2_environment.sh` автоматически загрузит все необходимые модели в `/tmp/huggingface_cache`.

**Важно**: Модели занимают ~84 GB, поэтому используется `/tmp` с 2.3TB свободного места.

### 3. Обучение Detection модели (T-lite)

```bash
# Запустить обучение
bash scripts/run_r_detect_t_lite.sh
```

Или вручную:
```bash
export HF_HOME=/tmp/huggingface_cache
cd /qwarium/home/d.a.lanovenko/llm-detect-ai
python3 code/train_r_detect_v2.py
```

## 📁 Структура файлов V2

```
llm-detect-ai/
├── conf/r_detect/
│   └── conf_r_detect_t_lite.yaml    # Конфиг для T-lite
├── code/
│   ├── train_r_detect_v2.py         # Training script V2
│   └── r_detect/
│       ├── ai_dataset.py            # Dataset (обновлён для V2)
│       └── ai_loader.py             # Data collator (обновлён для V2)
├── scripts/
│   ├── setup_v2_environment.sh      # Настройка и загрузка моделей
│   └── run_r_detect_t_lite.sh       # Запуск обучения
└── logs/
    └── train_r_detect_t_lite.log    # Логи обучения
```

## ⚙️ Конфигурация

### Основные параметры (conf_r_detect_t_lite.yaml):

```yaml
# Модель
model_name: t-tech/T-lite-it-1.0
num_labels: 1  # Binary classification

# LoRA/DoRA
lora_config:
  r: 32
  lora_alpha: 64
  use_dora: true
  use_rslora: true

# Training
training_args:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # Эффективный batch = 16
  learning_rate: 1.0e-4
  num_train_epochs: 3
  bf16: true
  gradient_checkpointing: true
```

## 🔧 Технические детали

### Использование /tmp для кэша

Из-за ограничения в 8GB в домашней директории, все модели кэшируются в `/tmp`:

```bash
export HF_HOME=/tmp/huggingface_cache
```

Это даёт доступ к 2.3TB свободного места.

### Оптимизация памяти

- **Batch size = 1** (минимальный для избежания OOM)
- **Gradient accumulation = 16** (эффективный batch = 16)
- **Gradient checkpointing** (экономия памяти)
- **BF16 precision** (использование тензорных ядер H100)
- **max_length = 1024** (сокращение длины последовательности)

### LoRA + DoRA

Используется **DoRA** (Decomposed Low-Rank Adaptation) вместо обычной LoRA:
- Лучшее качество
- Меньше параметров
- Быстрее обучение

Параметры:
- `r=32` (rank)
- `alpha=64` (scaling)
- `use_dora=true`
- `use_rslora=true` (Rank-Stabilized)

## 📊 Ожидаемые результаты

- **AUC-ROC**: ~0.95+ (на валидации)
- **Время обучения**: ~4-6 часов (1 H100)
- **Обучаемые параметры**: 82M (1.15% от общей модели)

## 🐛 Решение проблем

### OOM (Out Of Memory)
- Уменьшите `per_device_train_batch_size` до 1
- Увеличьте `gradient_accumulation_steps`
- Уменьшите `max_length` в конфиге

### Ошибки загрузки моделей
- Проверьте `HF_HOME` (должен указывать на `/tmp/huggingface_cache`)
- Проверьте интернет-соединение
- Попробуйте `HF_ENDPOINT=https://hf-mirror.com`

### Ошибки импорта
```bash
# Обновить пакеты
pip install --upgrade peft transformers accelerate
```

## 📈 Мониторинг обучения

Логи сохраняются в:
- `logs/train_r_detect_t_lite.log`
- `models/r_detect_t_lite/oof_df_last.csv` (предсказания)
- `models/r_detect_t_lite/result_df_last.csv` (метрики)

## 🎯 Следующие шаги

После обучения Detection модели:

1. **Embedding модель** (USER-bge-m3)
2. **Ranking модель** (BGE-reranker-v2-m3)
3. **Ансамблирование** всех трёх моделей

См. `TRAINING_PLAN.md` для полного плана обучения.

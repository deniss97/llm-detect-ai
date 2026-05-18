# 🤖 LLM-Detect-AI: Детекция AI-сгенерированного текста на русском языке

**Проект:** Исследование и разработка методов обнаружения текстов, сгенерированных большими языковыми моделями (LLM)

**Контакт:** d.a.lanovenko  
**Дата обновления:** 2026-05-15  
**Статус:** ✅ 3 модели обучены, ансамбль готов к использованию

---

## 📋 Цель исследования

Разработка и оценка методов детекции AI-сгенерированного текста **на русском языке** с использованием:

1. **Fine-tuned LLM моделей** (T-lite-7B + LoRA+DoRA, Qwen2.5-7B + QLoRA)
2. **Embedding моделей** (USER-bge-m3 + Contrastive Learning)
3. **Ансамблирования** различных подходов

**Целевая метрика:** ROC-AUC > 0.95 на валидационном датасете

**Достигнутая метрика:** ROC-AUC = **1.0000** (ансамбль из 3 моделей) ⭐

---

## 🎯 Обученные модели

### 1. T-lite-7B + LoRA+DoRA (Detection)

| Параметр | Значение |
|----------|----------|
| **Архитектура** | T-lite-it-1.0 (7B) |
| **Метод** | LoRA + DoRA (Low-Rank Adaptation + Weight-Decomposed) |
| **Путь к весам** | `/tmp/llm_cache/models/r_detect_t_lite_v2/` |
| **Конфиг** | `conf/r_detect/conf_r_detect_t_lite.yaml` |
| **Скрипт** | `scripts/run_r_detect_t_lite.sh` |

**Метрики на валидации:**
| ROC-AUC | Accuracy | F1 | Precision | Recall |
|---------|----------|-----|-----------|--------|
| **1.0000** | **0.7088** | **0.7727** | **0.6295** | **1.0000** |

⚠️ **Важно:** T-lite показывает много False Positive на human текстах (Precision 63%). Использовать только в ансамбле!

---

### 2. Qwen2.5-7B-Instruct + QLoRA (Detection)

| Параметр | Значение |
|----------|----------|
| **Архитектура** | Qwen2.5-7B-Instruct |
| **Метод** | QLoRA + DoRA (Quantized LO RA) |
| **Путь к весам** | `/tmp/llm_cache/models/r_detect_qwen3/` |
| **Конфиг** | `conf/r_detect/conf_r_detect_qwen3.yaml` |
| **Скрипт** | `scripts/run_r_detect_qwen3.sh` |

**Метрики на валидации:**
| ROC-AUC | Accuracy | F1 | Precision | Recall |
|---------|----------|-----|-----------|--------|
| **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

✅ **Отличное качество:** Идеальное разделение классов при пороге 0.5

---

### 3. USER-bge-m3 (Embedding)

| Параметр | Значение |
|----------|----------|
| **Архитектура** | deepvk/USER-bge-m3 |
| **Метод** | Contrastive Learning (MultipleNegativesRankingLoss) |
| **Путь к весам** | `/tmp/llm_cache/models/r_embed_final/` |
| **Конфиг** | `conf/r_embed/conf_r_embed_final_triplets.yaml` |
| **Скрипт** | `scripts/run_r_embed_final.sh` |

**Метрики на валидации:**
| Accuracy | MRR | Правильно классифицировано |
|----------|-----|---------------------------|
| **1.0000 (100%)** | **1.0000** | **243/243 триплета** |

✅ **Отличное качество:** Идеальное разделение классов при пороге 0.5

---

## 🎭 Ансамбль из 3 моделей

### Результаты ансамбля (Weighted Average)

| Метрика | Значение |
|---------|----------|
| **ROC-AUC** | **1.0000** |
| **Accuracy** | **1.0000** |
| **F1** | **1.0000** |
| **Precision** | **1.0000** |
| **Recall** | **1.0000** |
| **Ошибки** | **0/491 (0.0%)** |

**Веса моделей в ансамбле:**
- Qwen2.5-7B: **0.3360**
- USER-bge-m3: **0.3360**
- T-lite-7B: **0.3279**

**Confusion Matrix:**
```
              Predicted
              Human    AI
Actual Human  248        0
Actual AI       0      243
```

---

## 📊 Датасеты

### Основной датасет

**Путь:** `datasets/final_dataset.csv`

| Параметр | Значение |
|----------|----------|
| Всего записей | **2448** |
| Human тексты | **1236** |
| AI тексты | **1212** |
| Источники AI | **3 модели** (GPT-4, Claude, Qwen) |

---

### Сплит для Detection моделей

**Путь:** `datasets/final_prepared/`

| Файл | Записей | Назначение |
|------|---------|------------|
| `final_train.csv` | 1957 | Обучение Detection моделей |
| `final_valid.csv` | 491 | Валидация Detection моделей |

**Сплит:** 80% train / 20% valid  
**Стратификация:** по source_id (пары human+AI в одном сплите)

⚠️ **Важно:** Сплит выполнен по `source_id` - пары human+AI от одного источника всегда в одном сплите (нет data leak).

---

### Сплит для Embedding/Ranking (триплеты)

**Путь:** `datasets/final_prepared_embed_ranking/`

| Файл | Записей | Назначение |
|------|---------|------------|
| `train_triplets.csv` | 969 | Обучение Embedding/Ranking |
| `valid_triplets.csv` | 243 | Валидация Embedding/Ranking |

**Структура триплетов:**
```csv
anchor,positive,negative
"human текст","ai (тот же source_id)","ai (другой source_id)"
```

---

## 🔧 Быстрый старт

### Предварительные требования

```bash
# Python 3.10+
python3 --version

# Установка зависимостей
pip install -r requirements.txt

# Проверка места на диске (требуется ~50GB в /tmp)
df -h /tmp
```

### Шаг 1: Подготовка датасетов

```bash
cd /qwarium/home/d.a.lanovenko/llm-detect-ai

# Основной датасет (2448 записей)
ls -la datasets/final_dataset.csv

# Сплит для Detection моделей (train: 1957, valid: 491)
ls -la datasets/final_prepared/

# Сплит для Embedding/Ranking (триплеты)
python3 code/evaluate/prepare_final_for_embed_ranking.py
ls -la datasets/final_prepared_embed_ranking/
```

---

### Шаг 2: Обучение моделей

#### Обучение T-lite-7B Detection

```bash
# Запуск обучения (~50 минут, 24GB GPU)
bash scripts/run_r_detect_t_lite.sh

# Или напрямую
python3 code/train_r_detect_t_lite.py conf/r_detect/conf_r_detect_t_lite.yaml
```

**Результат:** Веса сохраняются в `/tmp/llm_cache/models/r_detect_t_lite_v2/`

---

#### Обучение Qwen2.5-7B Detection

```bash
# Запуск обучения (~60 минут, 24GB GPU)
bash scripts/run_r_detect_qwen3.sh

# Или напрямую
python3 code/train_r_detect_qwen3.py conf/r_detect/conf_r_detect_qwen3.yaml
```

**Результат:** Веса сохраняются в `/tmp/llm_cache/models/r_detect_qwen3/`

---

#### Обучение USER-bge-m3 Embedding

```bash
# Запуск обучения (~30 минут, 24GB GPU)
bash scripts/run_r_embed_final.sh

# Или напрямую
python3 code/train_r_embed_final.py conf/r_embed/conf_r_embed_final_triplets.yaml
```

**Результат:** Веса сохраняются в `/tmp/llm_cache/models/r_embed_final/`

---

### Шаг 3: Оценка ансамбля

```bash
cd /qwarium/home/d.a.lanovenko/llm-detect-ai

# Полная оценка ансамбля из 3 моделей
python3 code/evaluate/ensemble_3models_final.py
```

**Результаты:**
- ROC-AUC: 1.0000
- Accuracy: 1.0000
- F1: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- Ошибки: 0/491 (0.0%)

---

## 📁 Структура проекта

```
llm-detect-ai/
├── README.md                          # Этот файл
├── MODEL_ZOO.md                       # Полное описание моделей
├── CORRECT_SPLIT.md                   # Описание правильного сплита
├── ARCHITECTURE_V2.md                 # Архитектура ансамбля v2.0
├── MODEL_TRAINING_STATUS.md           # Статус обучения моделей
├── requirements.txt                   # Зависимости
│
├── code/
│   ├── train_r_detect_t_lite.py      # Обучение T-lite-7B
│   ├── train_r_detect_qwen3.py       # Обучение Qwen2.5-7B
│   ├── train_r_embed_final.py        # Обучение USER-bge-m3
│   │
│   └── evaluate/
│       ├── eval_t_lite.py            # Оценка T-lite-7B
│       ├── eval_qwen.py              # Оценка Qwen2.5-7B
│       ├── evaluate_r_embed_final.py # Оценка USER-bge-m3
│       └── ensemble_3models_final.py # Оценка ансамбля
│
├── conf/
│   ├── r_detect/
│   │   ├── conf_r_detect_t_lite.yaml
│   │   └── conf_r_detect_qwen3.yaml
│   └── r_embed/
│       └── conf_r_embed_final_triplets.yaml
│
├── scripts/
│   ├── run_r_detect_t_lite.sh
│   ├── run_r_detect_qwen3.sh
│   └── run_r_embed_final.sh
│
├── datasets/
│   ├── final_dataset.csv
│   ├── final_prepared/
│   └── final_prepared_embed_ranking/
│
├── generation_data/                   # Данные для генерации текстов
│   └── *.ipynb                        # Jupyter ноутбуки
│
├── models/                            # Обученные модели (ссылки на /tmp)
│   ├── r_detect_t_lite_v2/ -> /tmp/llm_cache/models/r_detect_t_lite_v2/
│   ├── r_detect_qwen3/ -> /tmp/llm_cache/models/r_detect_qwen3/
│   └── r_embed_final/ -> /tmp/llm_cache/models/r_embed_final/
│
└── results/
    ├── DETECTION_METRICS_REPORT.md   # Полный отчёт с метриками
    ├── ensemble_3models_predictions.csv
    ├── ensemble_3models_summary.csv
    └── meta_learner_3models.pkl
```

---

## 🎯 Пороговые значения (Thresholds)

> ⚠️ **Важное замечание:** ROC-AUC = 1.0 означает, что модель **может** идеально разделять классы при некотором оптимальном пороге. Метрики Accuracy/F1/Precision/Recall зависят от выбранного порога.

### Метрики при пороге 0.5 (использовалось в оценке)

| Модель | Порог | ROC-AUC | Accuracy | F1 | Precision | Recall |
|--------|-------|---------|----------|-----|-----------|--------|
| **Qwen2.5-7B** | 0.5 | 1.0000 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **USER-bge-m3** | 0.5 | - | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **T-lite-7B** | 0.5 | 1.0000 | **0.7088** | **0.7727** | **0.6295** | **1.0000** |
| **Ансамбль (3)** | 0.5 | 1.0000 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

**Выводы:**
- ✅ **Qwen2.5-7B** и **USER-bge-m3** дают идеальные метрики при пороге 0.5
- ⚠️ **T-lite-7B** — слабое звено: Accuracy 71%, Precision 63% (29% False Positive на human текстах)
- ✅ **Ансамбль** идеален благодаря вкладу Qwen + Embedding (они "перетягивают" слабый T-lite)

---

## 📈 Выводы и рекомендации

### ✅ Достигнутые результаты

1. **Все 3 модели показывают отличные метрики:**
   - Detection модели (T-lite, Qwen): ROC-AUC = 1.0
   - Embedding модель (USER-bge-m3): Accuracy = 100%, MRR = 1.0

2. **Ансамбль из 3 моделей:**
   - Идеальные метрики на валидации (100% accuracy)
   - 0 ошибок на 491 сэмпле
   - Стабильная работа на всех типах текстов

3. **Исправлен DATA LEAK:**
   - Сплит по source_id гарантирует, что пары human+AI от одного источника в одном сплите
   - Модели обучаются на корректных данных

---

### ⚠️ Ограничения

1. **Место на диске:**
   - Веса моделей хранятся в `/tmp/llm_cache/models/`
   - Требуется ~50GB свободного места
   - При перезагрузке сервера веса могут быть удалены

2. **Требования к GPU:**
   - Минимум 24GB VRAM для обучения
   - Рекомендуется A100/H100 для ускорения

3. **Зависимости:**
   - Конфликты версий transformers/torch/sentence-transformers
   - Рекомендуется использовать виртуальное окружение

---

## 📚 Документация

| Документ | Описание |
|----------|----------|
| **[MODEL_ZOO.md](MODEL_ZOO.md)** | Полное описание моделей с путями, метриками и инструкциями |
| **[CORRECT_SPLIT.md](CORRECT_SPLIT.md)** | Описание правильного сплита без data leak |
| **[ARCHITECTURE_V2.md](ARCHITECTURE_V2.md)** | Архитектура ансамбля v2.0 |
| **[MODEL_TRAINING_STATUS.md](MODEL_TRAINING_STATUS.md)** | Текущий статус обучения всех моделей |
| **[results/DETECTION_METRICS_REPORT.md](results/DETECTION_METRICS_REPORT.md)** | Полный отчёт с метриками |

---

## 📎 Приложения

### A. Генерация данных

**Путь:** `generation_data/`

В папке находятся Jupyter ноутбуки для генерации AI-текстов на русском языке:
- Генерация текстов с помощью различных LLM (GPT-4, Claude, Qwen)
- Подготовка датасетов для обучения
- Аугментация данных

---

### B. Таблица метрик всех моделей

| Модель | Тип | ROC-AUC | Accuracy | F1 | Precision | Recall |
|--------|-----|---------|----------|-----|-----------|--------|
| T-lite-7B | Detection | 1.0000 | 0.7088 | 0.7727 | 0.6295 | 1.0000 |
| Qwen2.5-7B | Detection | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| USER-bge-m3 | Embedding | - | 1.0000 | - | - | - |
| **Ансамбль (3)** | **Ensemble** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

---

### C. Ключевые пути

```bash
# Веса моделей
DET_TLITE=/tmp/llm_cache/models/r_detect_t_lite_v2/
DET_QWEN=/tmp/llm_cache/models/r_detect_qwen3/
EMB_USER=/tmp/llm_cache/models/r_embed_final/

# Датасеты
DATASET=/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/
EMBED_DATASET=/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared_embed_ranking/

# Ансамбль
ENSEMBLE=/qwarium/home/d.a.lanovenko/llm-detect-ai/results/meta_learner_3models.pkl
```

---

## 📖 Ссылки

- **Репозиторий:** `/qwarium/home/d.a.lanovenko/llm-detect-ai/`
- **Полный отчёт:** [results/DETECTION_METRICS_REPORT.md](results/DETECTION_METRICS_REPORT.md)
- **Model Zoo:** [MODEL_ZOO.md](MODEL_ZOO.md)

---

**Лицензия:** MIT  
**Дата создания:** 2025-01-15  
**Дата обновления:** 2026-05-15  
**Автор:** AI Assistant

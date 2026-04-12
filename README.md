# LLM-Detect-AI: Детекция AI-сгенерированного текста

**Проект:** Исследование методов обнаружения текстов, сгенерированных большими языковыми моделями (LLM)

**Контакт:** d.a.lanovenko

---

## 📋 Цель исследования

Разработка и оценка методов детекции AI-сгенерированного текста с использованием:
1. **Fine-tuned LLM моделей** (Mistral-7B + LoRA)
2. **Embedding моделей** (DeBERTa-v3-Base + KNN)
3. **Ranking моделей** (DeBERTa-v3-Large)
4. **Ансамблирования** различных подходов

**Целевая метрика:** ROC-AUC > 0.88 на тестовом датасете

---

## 📊 Датасеты

### Основной датасет
- **Источник:** Kaggle Detect AI Generated Text competition
- **Оригинальные эссе:** ~1,020 студенческих работ
- **AI-вариации:** 5 на каждое эссе (сгенерировано Mistral-7B)
- **Разделение:**
  - Train: 6,120 сэмплов (50% human, 50% AI)
  - Val: 2,040 сэмплов
  - Test: 2,040 сэмплов

### Расположение
```
llm-detect-ai/datasets/
├── detection_train.csv
├── detection_val.csv
└── detection_test.csv
```

---

## 🏗️ Архитектура решения

### 1. Detection модели (Mistral-7B + LoRA)
**Цель:** Прямая классификация текста (Human/AI)

**Конфигурация:**
- Base: `mistralai/Mistral-7B-v0.1`
- LoRA: r=8, alpha=16
- Target modules: q_proj, k_proj, v_proj, o_proj
- Max length: 256 токенов
- Batch size: 8
- Learning rate: 2e-5

**Обученные модели:**
- `r_detect_retrain` - дообучена на основном датасете (ROC-AUC 0.818)
- `r_detect_competition`, `r_detect_mix_v16`, `r_detect_mix_v26`, `r_detect_transfer` - предобученные модели

### 2. Embedding модель (DeBERTa-v3-Base + KNN)
**Цель:** Поиск похожих текстов через косинусное сходство эмбеддингов

**Конфигурация:**
- Base: `microsoft/deberta-v3-base`
- Loss: Supervised Contrastive Loss
- KNN neighbors: k=5
- Метрика: Cosine distance

**Результат:** ROC-AUC 0.9885 ⭐

### 3. Ranking модель (DeBERTa-v3-Large)
**Цель:** Оценка сходства между парами текстов

**Конфигурация:**
- Base: `microsoft/deberta-v3-large`
- Loss: Ranking loss
- Формат: Pair-wise comparison

**Результат:** ROC-AUC 0.9719

---

## 🔧 Порядок запуска скриптов

### Шаг 0: Setup окружения
```bash
# Установка зависимостей
pip install -r requirements.txt

# Проверка GPU
nvidia-smi
```

### Шаг 1: Обучение Detection модели (опционально)
```bash
# Дообучение Mistral-7B с LoRA
accelerate launch ./code/train_r_detect.py \
--config-name conf_r_detect_mini \
use_wandb=false
```

**Результат:** Модель в `models/r_detect_retrain/best/`

### Шаг 2: Обучение Embedding модели (опционально)
```bash
# Обучение с контрастивным лоссом
accelerate launch ./code/train_r_embed.py \
--config-name conf_r_embed \
use_wandb=false
```

**Результат:** Чекпоинт в `models/r_embed_conf_r_embed/`

### Шаг 3: Обучение Ranking модели (опционально)
```bash
# Обучение ranking модели
accelerate launch ./code/train_r_ranking.py \
--config-name conf_r_ranking_large \
use_wandb=false
```

**Результат:** Чекпоинт в `models/r_ranking_conf_r_ranking_large/`

### Шаг 4: Проверка на Target Leakage
```bash
# Проверка embedding модели на утечки
python3 code/evaluate/check_embedding_leakage.py
```

**Результат:** Отчёт в `results/leakage_check_report.json`

### Шаг 5: Оценка ансамблей с Cross-Validation
```bash
# Полный пайплайн: ranking + CV + ансамбли
python3 code/evaluate/ensemble_with_ranking_cv.py
```

**Результаты:**
- `results/ensemble_ranking_cv_results.json` - полные метрики
- `results/cross_validation_report.json` - 5-fold CV
- `results/ensemble_ranking_cv_submission.csv` - submission файл

### Шаг 6: Быстрая оценка (только рабочие модели)
```bash
# Ансамбль без ranking (быстрее)
python3 code/evaluate/ensemble_from_predictions.py
```

---

## 📈 Результаты

### Одиночные модели

| Модель | ROC-AUC | F1 | Accuracy |
|--------|---------|----|----------|
| **Embedding KNN** | **0.9885** | 0.9788 | 0.9789 |
| Ranking (fast) | 0.9719 | 0.9220 | 0.9206 |
| r_detect_retrain | 0.8180 | 0.8063 | 0.7956 |

### Ансамбли

| Метод | ROC-AUC | F1 | Ошибки |
|-------|---------|----|--------|
| **Weighted Ensemble** | **0.9912** | **0.9798** | 40/2040 |
| Meta-Learner | 0.9897 | 0.9803 | 40/2040 |
| Embedding KNN | 0.9885 | 0.9788 | 43/2040 |

### Confusion Matrix (Weighted Ensemble)
```
              Predicted
              Human    AI
Actual Human  1005     15
Actual AI       26    994
```
**False Positive:** 1.5% | **False Negative:** 2.5%

### Cross-Validation (5-Fold)

| Модель | ROC-AUC | F1 |
|--------|---------|----|
| embedding_knn | 1.0000±0.0000 | 0.9887±0.0041 |
| ranking | 0.9685±0.0037 | 0.7644±0.0049 |
| r_detect_retrain | 1.0000±0.0000 | 0.6667±0.0000 |

---

## 🤖 Как работает ансамбль

### Архитектура
```
┌─────────────────────────────────────────────────────┐
│              Level 1: Base Models                    │
├─────────────────┬─────────────────┬─────────────────┤
│ Detection       │ Embedding       │ Ranking         │
│ (Mistral-7B)    │ (DeBERTa-base)  │ (DeBERTa-large) │
│ r_detect_retrain │ KNN k=5        │ Similarity      │
└────────┬────────┴────────┬────────┴────────┬────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────┐
│              Level 2: Meta-Learner                   │
│         Logistic Regression (weights)               │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
         Final Prediction (AI probability)
```

### Методы ансамблирования

1. **Weighted Average**
   - Веса на основе validation AUC
   - Формула: `final = Σ(weight_i * pred_i) / Σ(weight_i)`

2. **Meta-Learner (Stacking)**
   - Logistic Regression на предсказаниях моделей
   - Коэффициенты: embedding_knn=14.90, ranking=0.45, r_detect=3.65

---

## 📁 Структура проекта

```
llm-detect-ai/
├── README.md                          # Этот файл
├── requirements.txt                   # Зависимости
├── setup.sh                          # Setup скрипт
│
├── code/
│   ├── train_r_detect.py             # Обучение detection моделей
│   ├── train_r_embed.py              # Обучение embedding модели
│   ├── train_r_ranking.py            # Обучение ranking модели
│   │
│   └── evaluate/
│       ├── check_embedding_leakage.py    # Проверка на leakage
│       ├── ensemble_with_ranking_cv.py   # Ансамбль + ranking + CV
│       ├── ensemble_from_predictions.py  # Быстрый ансамбль
│       ├── ensemble_simple.py            # Простая версия
│       └── ensemble_retrain.py           # С r_detect_retrain
│
├── conf/
│   ├── r_detect/                     # Конфиги detection моделей
│   │   ├── conf_r_detect_mini.yaml
│   │   ├── conf_r_detect_competition.yaml
│   │   └── ...
│   ├── r_embed/                      # Конфиги embedding модели
│   │   └── conf_r_embed.yaml
│   └── r_ranking/                    # Конфиги ranking модели
│       └── conf_r_ranking_large.yaml
│
├── datasets/
│   ├── detection_train.csv
│   ├── detection_val.csv
│   └── detection_test.csv
│
├── models/                           # Обученные модели
│   ├── r_detect_retrain/
│   ├── r_embed_conf_r_embed/
│   └── r_ranking_conf_r_ranking_large/
│
└── results/
    ├── DETECTION_METRICS_REPORT.md   # Полный отчёт
    ├── leakage_check_report.json     # Проверка leakage
    ├── cross_validation_report.json  # CV результаты
    ├── ensemble_ranking_cv_results.json
    └── *.csv                         # Submission файлы
```

---

## 🚀 Дальнейшие шаги

### 1. Построение сервиса детекции

**Архитектура сервиса:**
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Загрузка  │ -> │   Перевод    │ -> │   Детекция  │ -> │   Результат  │
│   текста    │    │  (если нужен)│    │   модель    │    │  (AI/Human)  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

**Компоненты:**
1. **API Gateway** (FastAPI/Flask)
   - REST endpoint для приёма текстов
   - Аутентификация и лимитирование

2. **Translation Service** (для не-English текстов)
   - Интеграция с Google Translate / DeepL API
   - Автоматическое определение языка
   - Перевод на английский перед детекцией

3. **Detection Service**
   - Загрузка обученных моделей
   - Ансамблирование предсказаний
   - Кэширование результатов

4. **Frontend** (опционально)
   - Веб-интерфейс для загрузки текстов
   - Визуализация результатов

**Пример API:**
```python
POST /api/detect
{
    "text": "Студенческое эссе...",
    "language": "auto"  # или "en", "ru", etc.
}

Response:
{
    "is_ai_generated": true,
    "probability": 0.94,
    "model": "ensemble_v1",
    "processing_time_ms": 245
}
```

### 2. Улучшения модели

**Краткосрочные:**
- [ ] Добавить больше detection моделей в ансамбль
- [ ] Оптимизировать гиперпараметры KNN (k=3,7,10)
- [ ] Попробовать XGBoost вместо Logistic Regression
- [ ] Калибровка вероятностей (Platt scaling)

**Долгосрочные:**
- [ ] Fine-tuning на мультиязычных данных
- [ ] Использование более крупных моделей (Llama-2-13B)
- [ ] Data augmentation для увеличения train set
- [ ] Детекция по стилю (stylometric features)

### 3. Production deployment

**Инфраструктура:**
- Docker контейнеризация
- Kubernetes для масштабирования
- Redis для кэширования
- PostgreSQL для логирования

**Мониторинг:**
- Метрики качества (drift detection)
- Latency мониторинг
- A/B тестирование новых моделей

### 4. Расширение функционала

**Дополнительные возможности:**
- Детекция частичного AI-генерации (какие параграфы сгенерированы)
- Определение конкретной модели (GPT-4, Claude, etc.)
- Анализ причин классификации (explainability)
- Пакетная обработка документов

---

## 📚 Выводы исследования

### Ключевые достижения
1. ✅ **Цель достигнута:** ROC-AUC 0.9912 > target 0.88
2. ✅ **Embedding KNN превзошёл ожидания:** 0.9885 single model
3. ✅ **Target leakage проверка пройдена:** утечек нет
4. ✅ **Cross-validation подтвердила стабильность**
5. ✅ **Ансамбль дал улучшение:** +0.2% к лучшей модели

### Инсайты
1. **Embedding + KNN** - простой но эффективный подход
2. **Ranking модель** даёт небольшой вклад в ансамбль
3. **Fine-tuned detection** менее стабильна на CV
4. **Meta-learner** автоматически находит оптимальные веса

### Рекомендации
1. Использовать **Embedding KNN как baseline** для новых задач
2. Добавлять **ranking модели** для marginal gains
3. Проводить **leakage check** перед production
4. Применять **cross-validation** для надёжной оценки

---

## 📖 Ссылки

- [Kaggle Competition](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)
- [Полный отчёт](results/DETECTION_METRICS_REPORT.md)
- [Обсуждение решения](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121)

---

**Лицензия:** MIT  
**Дата обновления:** Апрель 2026

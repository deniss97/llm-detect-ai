# Обучение и Оценка на Final Dataset (Русские Сочинения)

## Описание Датасета

`datasets/final_dataset.csv` содержит:
- **1236 оригинальных сочинений** на русском языке (label=human, model=human)
- **1212 сгенерированных сочинений** от трёх моделей:
  - openai/gpt-4o-mini (416 текстов)
  - google/gemini-3.1-flash-lite (416 текстов)
  - qwen/qwen-2.5-72b-instruct (380 текстов)

**Структура CSV:**
```
source_id,text,label,model,prompt_type
0,В своем произведении Горький ставит вопрос...,human,human,none
...
```

## Подготовка Датасета

Датасет автоматически подготавливается при первом запуске обучения.

**Вручную:**
```bash
cd /qwarium/home/d.a.lanovenko/llm-detect-ai
python3 code/evaluate/prepare_final_dataset.py
```

**Результат:**
- `datasets/final_prepared/final_train.csv` - 1958 сэмплов (80%)
- `datasets/final_prepared/final_valid.csv` - 490 сэмплов (20%)
- `datasets/final_prepared/final_full.csv` - полный датасет (2448 сэмплов)

**Распределение классов:**
- Train: 989 human, 969 AI
- Valid: 247 human, 243 AI

---

## Обучение Модели

### Конфигурация

Файл: `conf/r_detect/conf_r_detect_final.yaml`

**Параметры:**
- **Модель:** mistralai/Mistral-7B-v0.1 + LoRA (r=8, alpha=16)
- **Max Length:** 512 токенов (адаптировано для русских текстов)
- **Эпохи:** 3
- **Batch Size:** 1 (gradient accumulation=16, эффективный batch=16)
- **Learning Rate:** 2e-5
- **Target Modules:** q_proj, k_proj, v_proj, o_proj

### Запуск Обучения

**Вариант 1: Через скрипт**
```bash
/qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect_final.sh
```

**Вариант 2: Через qwarium-agent (рекомендуется для долгого обучения)**
```bash
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect_final.sh
```

**Вариант 3: Прямой запуск**
```bash
cd /qwarium/home/d.a.lanovenko/llm-detect-ai

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HOME=/tmp/huggingface_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache

python3 code/train_r_detect.py \
    --config-name conf_r_detect_final \
    use_wandb=false \
    outputs.model_dir=/qwarium/home/d.a.lanovenko/models/r_detect_final_dataset
```

### Логи и Сохранение

**Логи:**
- `/qwarium/home/d.a.lanovenko/llm-detect-ai/logs/train_r_detect_final.log`

**Модели:**
- `/qwarium/home/d.a.lanovenko/models/r_detect_final_dataset/last/` - последняя версия
- `/qwarium/home/d.a.lanovenko/models/r_detect_final_dataset/best/` - лучшая версия (AUC > 0.85)

**Метрики во время обучения:**
- `oof_df_best.csv` - предсказания на валидации для лучшей модели
- `result_df_best.csv` - полные результаты (predictions + truths)

---

## Оценка Модели

### Запуск Оценки

**Вариант 1: Через скрипт**
```bash
python3 /qwarium/home/d.a.lanovenko/llm-detect-ai/code/evaluate/eval_r_detect_final.py
```

**Вариант 2: С указанием путей**
```bash
python3 /qwarium/home/d.a.lanovenko/llm-detect-ai/code/evaluate/eval_r_detect_final.py \
    /qwarium/home/d.a.lanovenko/models/r_detect_final_dataset/best \
    /qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/final_valid.csv
```

### Выходные Данные

**Метрики:**
- AUC-ROC (основная метрика)
- Accuracy (при пороге 0.5)
- F1 Score (при пороге 0.5)
- Classification Report (precision, recall, F1 для каждого класса)

**Файлы результатов:**
- `final_test_results.csv` - предсказания для каждого сэмпла
- `final_metrics_summary.csv` - сводные метрики

---

## Ожидаемые Результаты

**Целевые метрики (на валидации):**
- **AUC-ROC:** > 0.85 (save_trigger в конфиге)
- **F1 Score:** > 0.80
- **Accuracy:** > 0.80

**Время обучения:**
- ~2-4 часа на GPU (Mistral-7B + LoRA, 3 эпохи, 1958 сэмплов)

---

## Структура Файлов

```
llm-detect-ai/
├── datasets/
│   ├── final_dataset.csv              # Исходный датасет
│   └── final_prepared/                # Подготовленные данные
│       ├── final_train.csv            # Train split (1958)
│       ├── final_valid.csv            # Valid split (490)
│       └── final_full.csv             # Полный датасет (2448)
│
├── code/
│   ├── train_r_detect.py              # Скрипт обучения (обновлён)
│   └── evaluate/
│       ├── prepare_final_dataset.py   # Подготовка датасета
│       └── eval_r_detect_final.py     # Оценка модели
│
├── conf/r_detect/
│   └── conf_r_detect_final.yaml       # Конфигурация обучения
│
├── scripts/
│   └── run_r_detect_final.sh          # Скрипт запуска обучения
│
└── models/
    └── r_detect_final_dataset/        # Сохранённые модели
        ├── best/                      # Лучшая модель
        │   ├── adapter_config.json
        │   ├── adapter_model.safetensors
        │   ├── tokenizer files...
        │   ├── final_test_results.csv
        │   └── final_metrics_summary.csv
        └── last/                      # Последняя модель
```

---

## Troubleshooting

### Ошибка: "CUDA out of memory"
- Уменьшите `max_length` в конфиге (например, до 384)
- Уменьшите `gradient_accumulation_steps`
- Включите 4-bit quantization в конфиге:
  ```yaml
  model:
    use_4bit: true
  ```

### Ошибка: "Model not found"
- Проверьте, что модель Mistral-7B-v0.1 закэширована в `/tmp/huggingface_cache`
- При необходимости скачайте заново:
  ```bash
  huggingface-cli download mistralai/Mistral-7B-v0.1
  ```

### Низкие метрики (AUC < 0.75)
- Увеличьте количество эпох (до 5-10)
- Увеличьте `max_length` для захвата большего контекста
- Попробуйте другие target modules в LoRA
- Добавьте data augmentation

---

## Дальнейшие Улучшения

1. **Cross-Validation:** Запустить 5-fold CV для более надёжной оценки
2. **Ensemble:** Объединить с другими моделями (Embedding, Ranking)
3. **Hyperparameter Tuning:** Подобрать оптимальные гиперпараметры
4. **Data Augmentation:** Добавить больше вариаций текстов
5. **Multi-Model:** Обучить отдельные модели для каждой AI-модели

---

## Контакты

По вопросам: d.a.lanovenko

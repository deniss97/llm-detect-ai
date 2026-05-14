# Правильный сплит для final_dataset.csv

## Проблема предыдущих сплитов

**Старые сплиты** (`final_prepared/`, `final_prepared_v2/` в `/tmp/`):
- ❌ **Data Leak**: Пары human+AI были разделены между train и valid
- ❌ `source_id_check` создавался из `id` (essay_0, essay_1, ...), а не из оригинального `source_id`
- ❌ Для source_id=0: human был в train, а AI имел `source_id_check=1256` и мог быть в valid
- ❌ Метрики были **некорректны** (AUC=1.0 из-за data leak или AUC=0.65 из-за разделения пар)

## Правильный сплит (текущий)

**Путь:** `/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/`

**Скрипт подготовки:** `code/evaluate/prepare_final_dataset_v2.py`

### Характеристики:
- ✅ **Split by source_id**: Все записи с одинаковым `source_id` (human + AI версии) находятся в одном сплите
- ✅ **Нет data leak**: 0 source_id пересекаются между train и valid
- ✅ **Пары сохранены**: 969 source_id с парами human+AI в train, 243 в valid

### Статистика:
```
Dataset: final_dataset.csv
Total: 2448 записей (1236 human + 1212 AI)
Unique source_id: 1236

Train: 1957 записей (988 human + 969 AI)
  - 969 source_id с парами human+AI
  - 19 source_id только human (без AI пар)

Valid: 491 записей (248 human + 243 AI)
  - 243 source_id с парами human+AI
  - 5 source_id только human (без AI пар)
```

### Пример ID:
```
source_id=0 в train:
  - essay_0_1 (human)
  - essay_0_2 (AI: openai/gpt-4o-mini)

source_id=1 в valid:
  - essay_1_1 (human)
  - essay_1_2 (AI: google/gemini-3.1-flash-lite)
```

## Использование в обучении

Все модели должны использовать этот сплит:

### T-lite-7B:
```yaml
# conf/r_detect/conf_r_detect_t_lite.yaml
train_dataset: datasets/final_prepared/final_train.csv
valid_dataset: datasets/final_prepared/final_valid.csv
label_column: generated
```

### Qwen3-8B:
```yaml
# conf/r_detect/conf_r_detect_qwen3.yaml
data:
  train_path: datasets/final_prepared/final_train.csv
  valid_path: datasets/final_prepared/final_valid.csv
  target_column: generated
```

## Проверка корректности

```python
import pandas as pd

train_df = pd.read_csv('datasets/final_prepared/final_train.csv')
valid_df = pd.read_csv('datasets/final_prepared/final_valid.csv')

# Проверка: нет пересечений source_id
train_sids = set(train_df['source_id'].unique())
valid_sids = set(valid_df['source_id'].unique())
assert len(train_sids & valid_sids) == 0, "Data leak!"

# Проверка: пары в одном сплите
for split_name, split_df in [("Train", train_df), ("Valid", valid_df)]:
    grouped = split_df.groupby('source_id')['generated'].apply(lambda x: tuple(sorted(x.unique())))
    pairs = sum(1 for _, labels in grouped.items() if len(labels) == 2)
    print(f"{split_name}: {pairs} source_id с парами human+AI")
```

## Модели, обученные на правильном сплите

1. **T-lite-7B** (переобучить)
   - Путь: `/tmp/llm_cache/models/r_detect_t_lite_v2/`
   - Статус: Требует переобучения на правильном сплите

2. **Qwen3-8B** (обучить)
   - Путь: `/tmp/llm_cache/models/r_detect_qwen3/`
   - Статус: Готов к обучению

3. **Embedding KNN** (zero-shot)
   - Модель: `deepvk/USER-bge-m3`
   - Статус: Готов (обучен на external ai_mix_v26)

4. **Ranking** (zero-shot)
   - Модель: `BAAI/bge-reranker-v2-m3`
   - Статус: Готов (обучен на external ai_mix_for_ranking)

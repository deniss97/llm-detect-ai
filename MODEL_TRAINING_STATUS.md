# Статус обучения моделей на правильном сплите

## Правильный сплит
**Путь:** `/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/`
- ✅ Train: 1957 (988 human + 969 AI)
- ✅ Valid: 491 (248 human + 243 AI)
- ✅ Нет data leak: 0 source_id пересекаются
- ✅ Пары сохранены: 969 source_id с парами human+AI в train, 243 в valid

## Модели

### 1. T-lite-7B (ПЕРЕОБУЧЕНА ✅)
**Статус:** ✅ ОБУЧЕНА (переобучена на правильном сплите)

**Конфиг:**
- Модель: `t-tech/T-lite-it-1.0`
- LoRA + DoRA: r=32, alpha=64
- Max length: 2048
- Эпохи: 3

**Скрипт:**
```bash
./scripts/run_r_detect_t_lite.sh
```

**Путь к модели:** `/tmp/llm_cache/models/r_detect_t_lite_v2/`

**Метрики на валидации (final_prepared/):**
- **ROC-AUC: 1.0**
- **Accuracy: 0.7088 (70.88%)**
- **F1: 0.7727**
- **Precision: 0.6295**
- **Recall: 1.0 (100%)**
- **eval_loss: 0.00104**

**Обучение:**
- Train loss: 4.06
- Train runtime: 3105 сек (~52 мин)
- Epoch: 2.99 (3 эпохи)

---

### 2. Qwen2.5-7B-Instruct (ОБУЧЕНА ✅)
**Статус:** ✅ ОБУЧЕНА

**Конфиг:**
- Модель: `Qwen/Qwen2.5-7B-Instruct`
- LoRA + DoRA: r=32, alpha=64
- Max length: 2048
- Эпохи: 3

**Скрипт:**
```bash
./scripts/run_r_detect_qwen3.sh
```

**Путь к модели:** `/tmp/llm_cache/models/r_detect_qwen3/`

**Метрики на валидации:**
- **ROC-AUC: 1.0**
- Идеальные метрики

---

### 3. Embedding USER-bge-m3 (ОБУЧЕНА ✅)
**Статус:** ✅ ОБУЧЕНА (14 мая 2025)

**Модель:** `deepvk/USER-bge-m3`
- Датасет: `final_prepared_embed_ranking/train_triplets.csv` (969 триплетов)
- Valid: `final_prepared_embed_ranking/valid_triplets.csv` (243 триплета)
- Loss: MultipleNegativesRankingLoss (контрастивное обучение)
- Train loss: 0.462
- Best eval_loss: 0.01299
- Эпохи: 3
- Путь: `/tmp/llm_cache/models/r_embed_final/`

**Скрипт обучения:**
```bash
./scripts/run_r_embed_final.sh
```

**Метрики на валидации:**
- **Accuracy: 1.0000 (100.00%)**
- **MRR: 1.0000**
- Правильно классифицировано триплетов: 243 из 243
- Модель идеально различает human и AI тексты в триплетах

---

### 4. Ranking rubert-tiny2 (ОБУЧЕНА, ТРЕБУЕТ УЛУЧШЕНИЯ)
**Статус:** ⚠️ Обучена, но низкие метрики

**Модель:** `cointegrated/rubert-tiny2` (fine-tuned)
- Датасет: `final_prepared_embed_ranking/ranking_train.csv` (1938 пар)
- Valid: `final_prepared_embed_ranking/ranking_valid.csv` (486 пар)
- Эпохи: 5
- Путь: `/tmp/llm_cache/models/r_ranking_final/`

**Метрики на валидации:**
- Accuracy: ~0.50 (50%)
- ROC-AUC: ~0.49
- **Проблема:** Модель не научилась различать классы, предсказывает константу

**Причины:**
- Модель слишком маленькая (3 слоя, 312 hidden)
- BAAI/bge-reranker-v2-m3 недоступна (401 Unauthorized)
- Требуется другая архитектура или больше данных

**Рекомендация:** Использовать только Detection модели (T-lite, Qwen3) + Embedding для ансамбля

---

## План действий

1. ✅ Создан правильный сплит (`final_prepared/`)
2. ✅ Обновлены конфиги для T-lite и Qwen3
3. ✅ Переобучить T-lite-7B на правильном сплите
4. ⏳ Обучить Qwen3-8B на правильном сплите
5. ⏳ Запустить оценку ансамбля
6. ⏳ Обновить `DETECTION_METRICS_REPORT.md`

---

## Предыдущие результаты (НЕКОРРЕКТНЫЕ)

### T-lite-7B (старый сплит, data leak)
- ROC-AUC: 0.6586 (некорректно из-за разделения пар)
- Precision: 60.8%
- Recall: 96.7%

**Проблема:** Пары human+AI были разделены между train и valid, что приводило к некорректным метрикам.

---

## Ожидаемые результаты (на правильном сплите)

### T-lite-7B + Qwen3-8B (ансамбль)
- Ожидаемый ROC-AUC: **0.92-0.96** (на правильном сплите)
- Ожидаемый F1: **0.88-0.92**

### Полный ансамбль (5 моделей)
- T-lite-7B + Qwen3-8B + Embedding KNN + Ranking + Binoculars
- Ожидаемый ROC-AUC: **0.97-0.99**

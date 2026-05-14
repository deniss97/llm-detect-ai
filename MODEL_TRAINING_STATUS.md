# Статус обучения моделей на правильном сплите

## Правильный сплит
**Путь:** `/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/`
- ✅ Train: 1957 (988 human + 969 AI)
- ✅ Valid: 491 (248 human + 243 AI)
- ✅ Нет data leak: 0 source_id пересекаются
- ✅ Пары сохранены: 969 source_id с парами human+AI в train, 243 в valid

## Модели

### 1. T-lite-7B (ПЕРЕОБУЧИТЬ)
**Статус:** ⏳ Ожидает переобучения

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

---

### 2. Qwen3-8B (ОБУЧИТЬ)
**Статус:** ⏳ Готов к обучению

**Конфиг:**
- Модель: `Qwen/Qwen3-8B`
- LoRA + DoRA: r=32, alpha=64
- Max length: 2048
- Эпохи: 3

**Скрипт:**
```bash
./scripts/run_r_detect_qwen3.sh
```

**Путь к модели:** `/tmp/llm_cache/models/r_detect_qwen3/`

---

### 3. Embedding KNN (ГОТОВ, zero-shot)
**Статус:** ✅ Готов

**Модель:** `deepvk/USER-bge-m3`
- Обучена на external `ai_mix_v26` (английские тексты)
- Zero-shot на русском: AUC = 0.956

---

### 4. Ranking (ГОТОВ, zero-shot)
**Статус:** ✅ Готов

**Модель:** `BAAI/bge-reranker-v2-m3`
- Обучена на external `ai_mix_for_ranking` (английские тексты)
- Zero-shot на русском: AUC = 0.940

---

## План действий

1. ✅ Создан правильный сплит (`final_prepared/`)
2. ✅ Обновлены конфиги для T-lite и Qwen3
3. ⏳ Переобучить T-lite-7B на правильном сплите
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

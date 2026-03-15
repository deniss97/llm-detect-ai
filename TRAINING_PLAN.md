# План Обучения Моделей (Progressive Complexity)

## Статус Датасетов

### ✅ Доступные датасеты
| Файл | Размер | Назначение |
|------|--------|------------|
| `datasets/train_essays.csv` | 4.4 MB | Основные данные для обучения detect |
| `datasets/test_essays.csv` | 90 B | Тестовые данные |
| `datasets/train_prompts.csv` | 27 KB | Промпты для генерации |

### ✅ Внешние датасеты (скачаны в /tmp с симлинками)
| Датасет | Размер | Формат | Для чего | Статус |
|---------|--------|--------|----------|--------|
| `ai_mix_for_ranking` | 174 MB | CSV | Ranking, Embed | ✅ Скачан |
| `ai_mix_v16` | 306 MB | CSV | Detect mix_v16 | ✅ Скачан |
| `ai_mix_v26` | 195 MB | Parquet | Detect mix_v26, Embed | ✅ Скачан |

---

## Этапы Обучения (по возрастанию сложности)

### 🔹 Этап 1: Detect Mini (✅ ЗАВЕРШЕН)
**Конфиг:** `conf_r_detect_mini.yaml`
- **Модель:** Mistral-7B-v0.1 + LoRA (r=8)
- **Датасет:** `train_essays.csv`
- **Эпохи:** 1
- **Сложность:** Минимальная (тестовая конфигурация)
- **Статус:** ✅ Обучено

---

### 🔹 Этап 2: Detect Competition (✅ ЗАВЕРШЕН)
**Конфиг:** `conf_r_detect_competition.yaml`
- **Модель:** Mistral-7B-v0.1 + LoRA (r=4)
- **Датасет:** `train_essays.csv`
- **Эпохи:** 2
- **Сложность:** Базовая
- **Статус:** ✅ Обучено

---

### 🔹 Этап 3: Detect Mix v16 (🔄 СЛЕДУЮЩИЙ)
**Конфиг:** `conf_r_detect_mix_v16.yaml`
- **Модель:** Mistral-7B-v0.1 + LoRA (r=8)
- **Датасет:** Требуется `ai_mix_v16` (больший датасет)
- **Эпохи:** 16
- **Max Length:** 1296 токенов
- **Сложность:** Средняя
- **Статус:** ⏳ Требуется подготовка датасета

**Команда для запуска:**
```bash
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_mix_v16
```

---

### 🔹 Этап 4: Detect Mix v26
**Конфиг:** `conf_r_detect_mix_v26.yaml`
- **Модель:** Mistral-7B-v0.1 + LoRA (r=16) ↑
- **Датасет:** Требуется `ai_mix_v26`
- **Эпохи:** 16
- **Max Length:** 1296 токенов
- **Сложность:** Средняя+ (больше параметров LoRA)
- **Статус:** ⏳ Требуется подготовка датасета

**Команда для запуска:**
```bash
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_mix_v26
```

---

### 🔹 Этап 5: Detect Transfer Learning
**Конфиг:** `conf_r_detect_transfer.yaml`
- **Модель:** Transfer с `r_detect_mix_v16/best`
- **Датасет:** `train_essays.csv` (fine-tuning)
- **Эпохи:** 2
- **Сложность:** Высокая (требует предобученную модель)
- **Статус:** ⏳ Зависит от Этапа 3

**Команда для запуска:**
```bash
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_transfer
```

---

### 🔹 Этап 6: Ranking Model
**Конфиг:** `conf_r_ranking_large.yaml`
- **Модель:** DeBERTa-v3-large
- **Датасет:** `datasets/external/ai_mix_for_ranking` ❌
- **Эпохи:** 2 (рекомендуется 16)
- **Сложность:** Высокая (требует внешний датасет)
- **Статус:** ❌ Требуется скачать датасет

**Команда для запуска:**
```bash
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_ranking.sh conf_r_ranking_large
```

---

### 🔹 Этап 7: Embedding Model
**Конфиг:** `conf_r_embed.yaml`
- **Модель:** Embedding модель
- **Датасет:** Требуется проверить
- **Сложность:** Высокая
- **Статус:** ⏳ Требуется проверка конфига

**Команда для запуска:**
```bash
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_embed.sh
```

---

### 🔹 Этап 8: CLM Models (разные варианты)
**Конфиги:** `conf/r_clm/*.yaml`
- **Модели:** GPT-2, OPT, Pythia, Llama variants, Mistral
- **Сложность:** Очень высокая (полное обучение или fine-tuning)
- **Статус:** ⏳ Требуется выбор конфигурации

**Доступные конфиги:**
- `conf_r_clm_gpt2.yaml` - GPT-2 (базовая)
- `conf_r_clm_tiny_llama.yaml` - Tiny Llama (легкая)
- `conf_r_clm_lite_llama.yaml` - Lite Llama (средняя)
- `conf_r_clm_llama13b.yaml` - Llama 13B (очень тяжелая)
- `conf_r_clm_mistral_persuade.yaml` - Mistral Persuade

**Команда для запуска:**
```bash
# Lite вариант
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_clm.sh conf_r_clm_tiny_llama

# Полное обучение с нуля
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_clm_from_scratch.sh conf_r_clm_tiny_llama
```

---

### 🔹 Этап 9: DPO (Direct Preference Optimization)
**Конфиг:** `conf_r_dpo.yaml`
- **Модель:** DPO fine-tuning
- **Датасет:** Требуется preference dataset
- **Сложность:** Максимальная
- **Статус:** ❌ Требуется подготовка датасета

**Команда для запуска:**
```bash
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_dpo.sh
```

---

## 📋 План Действий

### ✅ Выполненные действия:
1. ✅ Все конфиги обновлены (пути к моделям и датасетам исправлены)
2. ✅ Скрипты для запуска созданы (`scripts/run_*.sh`)
3. ✅ Скрипт для скачивания датасетов создан (`scripts/download_datasets.sh`)
4. ✅ Код обучения обновлен для поддержки внешних датасетов (`external_data_dir`)
5. ✅ Датасет `train_essays.csv` доступен для всех этапов

### ⏳ Следующие шаги:

**ШАГ 1: Скачать внешние датасеты в /tmp (с симлинками)**
```bash
/qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/download_datasets.sh
```
Это скачает:
- `ai-bin7-mix-v1` → `ai_mix_for_ranking` (для ranking и embed)
- `ai-mix-v16` → `ai_mix_v16` (для detect mix_v16)
- `ai-mix-v26` → `ai_mix_v26` (для detect mix_v26 и embed)

**ШАГ 2: Запустить обучение по этапам**

```bash
# Этап 3: Detect Mix v16 (Mistral-7B + LoRA r=8, 16 эпох)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_mix_v16

# Этап 4: Detect Mix v26 (Mistral-7B + LoRA r=16, 16 эпох)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_mix_v26

# Этап 5: Transfer Learning (использует модель из Этапа 3)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_transfer

# Этап 6: Ranking Model (DeBERTa-large)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_ranking.sh

# Этап 7: Embedding Model (DeBERTa-base)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_embed.sh
```

---

## 📊 Матрица Сложности

| Этап | Модель | Датасет | Эпохи | LoRA Rank | GPU Hours (est.) |
|------|--------|---------|-------|-----------|------------------|
| 1 | Mistral-7B | essays | 1 | 8 | ~0.5 |
| 2 | Mistral-7B | essays | 2 | 4 | ~0.5 |
| 3 | Mistral-7B | mix_v16 | 16 | 8 | ~4-6 |
| 4 | Mistral-7B | mix_v26 | 16 | 16 | ~6-8 |
| 5 | Mistral-7B | transfer | 2 | 16 | ~1 |
| 6 | DeBERTa-large | ranking | 2-16 | N/A | ~2-4 |
| 7 | Embedding | ? | ? | N/A | ? |
| 8 | CLM variants | ? | ? | Full/LoRA | ~10-50 |
| 9 | DPO | preference | ? | Full | ~5-10 |

---

## 🔧 Быстрый Старт

```bash
# Следующий этап (Detect Mix v16)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_mix_v16

# Проверка статуса процесса
ps aux | grep train_r_detect

# Просмотр логов
tail -f /qwarium/home/d.a.lanovenko/llm-detect-ai/logs/train_r_detect_conf_r_detect_mix_v16.log

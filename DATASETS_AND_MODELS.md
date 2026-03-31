# Датасеты и Модели - Полная Карта Взаимосвязей

## 📊 Обзор Датасетов

### Базовые Датасеты

| Название | Путь | Размер | Формат | Источник |
|----------|------|--------|--------|----------|
| **train_essays** | `datasets/train_essays.csv` | 4.4 MB | CSV | Kaggle Competition |
| **test_essays** | `datasets/test_essays.csv` | 90 B | CSV | Kaggle Competition |
| **train_prompts** | `datasets/train_prompts.csv` | 27 KB | CSV | Kaggle Competition |

### Внешние Датасеты (скачаны в /tmp)

| Название | Путь | Размер | Формат | Kaggle Dataset |
|----------|------|--------|--------|----------------|
| **ai_mix_v16** | `/tmp/datasets/external/ai_mix_v16/` | 306 MB | CSV | `conjuring92/ai-mix-v16` |
| **ai_mix_v26** | `/tmp/datasets/external/ai_mix_v26/` | 195 MB | Parquet | `conjuring92/ai-mix-v26` |
| **ai_mix_for_ranking** | `/tmp/datasets/external/ai_mix_for_ranking/` | 174 MB | CSV | `conjuring92/ai-bin7-mix-v1` |
| **persuade_2.0** | `datasets/persuade_2.0_human_scores_demo_id_github.csv` | 72 MB | CSV | `nbroad/persaude-corpus-2` |

---

## 🗺️ Карта Использования Датасетов

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ВСЕ ДАТАСЕТЫ                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. train_essays.csv (4.4 MB) - БАЗОВЫЙ                                     │
│  2. ai_mix_v16/train_essays.csv (306 MB) - РАСШИРЕННЫЙ                      │
│  3. ai_mix_v26/*.parquet (195 MB) - РАСШИРЕННЫЙ + VALID                     │
│  4. ai_mix_for_ranking/train_essays.csv (174 MB) - ДЛЯ RANKING/EMBED        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ЭТАПЫ ОБУЧЕНИЯ                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Детальная Таблица: Модель → Датасет

| Этап | Модель | Конфиг | Основной Датасет | Внешний Датасет | Итоговый Размер |
|------|--------|--------|------------------|-----------------|-----------------|
| **1** | Detect Mini | `conf_r_detect_mini.yaml` | train_essays.csv | ❌ Нет | ~4.4 MB |
| **2** | Detect Competition | `conf_r_detect_competition.yaml` | train_essays.csv | ❌ Нет | ~4.4 MB |
| **3** | Detect Mix v16 | `conf_r_detect_mix_v16.yaml` | train_essays.csv | ✅ ai_mix_v16 | ~310 MB |
| **4** | Detect Mix v26 | `conf_r_detect_mix_v26.yaml` | train_essays.csv | ✅ ai_mix_v26 | ~200 MB |
| **5** | Detect Transfer | `conf_r_detect_transfer.yaml` | train_essays.csv | ❌ (использует модель Этапа 3) | ~4.4 MB |
| **6** | Ranking | `conf_r_ranking_large.yaml` | train_essays.csv | ✅ ai_mix_for_ranking | ~178 MB |
| **7** | Embedding | `conf_r_embed.yaml` | train_essays.csv | ✅ ai_mix_v26 | ~200 MB |
| **8** | CLM | `conf/r_clm/*.yaml` | persuade_2.0 | ✅ PERSUADE 2.0 | ~72 MB |
| **9** | DPO | `conf_r_dpo.yaml` | Требуется preference | ❌ Пока нет | - |

---

## 🔗 Взаимосвязи Между Моделями

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ЗАВИСИМОСТИ МОДЕЛЕЙ                                   │
└──────────────────────────────────────────────────────────────────────────────┘

Этап 1: Detect Mini (Mistral-7B + LoRA r=8)
         │
         │ ✅ Независимый
         │
         ▼
Этап 2: Detect Competition (Mistral-7B + LoRA r=4)
         │
         │ ✅ Независимый
         │
         ▼
Этап 3: Detect Mix v16 (Mistral-7B + LoRA r=8, 16 эпох)
         │
         │ ⚠️ Требует ai_mix_v16
         │
         ▼
Этап 4: Detect Mix v26 (Mistral-7B + LoRA r=16, 16 эпох)
         │
         │ ⚠️ Требует ai_mix_v26
         │
         ▼
Этап 5: Detect Transfer (Mistral-7B + LoRA r=16, 2 эпохи)
         │
         │ 🔗 ЗАВИСИТ от Этапа 3!
         │    (использует weights из r_detect_mix_v16/best)
         │
         ▼
Этап 6: Ranking (DeBERTa-v3-large)
         │
         │ ⚠️ Требует ai_mix_for_ranking
         │    ✅ НЕ зависит от предыдущих этапов
         │
         ▼
Этап 7: Embedding (DeBERTa-v3-base)
         │
         │ ⚠️ Требует ai_mix_v26
         │    ✅ НЕ зависит от предыдущих этапов
         │
         ▼
Этап 8: CLM (разные варианты)
         │
         │ ✅ Независимый (генеративные модели)
         │
         ▼
Этап 9: DPO (Direct Preference Optimization)
         │
         │ 🔗 ЗАВИСИТ от CLM моделей (Этап 8)
         │    Требуется preference dataset
```

---

## 📁 Структура Датасетов

### 1. train_essays.csv (Базовый)
```
Колонки:
- id: уникальный идентификатор эссе
- prompt_id: идентификатор промпта
- text: текст эссе
- generated: 0 (human) или 1 (AI)
```

### 2. ai_mix_v16/train_essays.csv
```
Колонки:
- id: уникальный идентификатор
- prompt_id: идентификатор промпта  
- text: текст эссе
- generated: 0 или 1

Особенности:
- Больше данных для обучения
- Разнообразные AI-генерированные тексты
```

### 3. ai_mix_v26/*.parquet
```
Файлы:
- train_essays.parquet (обучение)
- valid_essays.parquet (валидация)

Колонки:
- id, prompt_id, text, generated

Особенности:
- Готовое разделение train/valid
- Формат Parquet (быстрее чтение)
```

### 4. ai_mix_for_ranking/train_essays.csv
```
Колонки:
- id, prompt_id, text, generated

Особенности:
- Специально подготовлен для ranking задачи
- Сбалансированные классы
```

---

## 🚀 Порядок Запуска Обучения

### Вариант A: Последовательный (Рекомендуется)
```bash
# 1. Detect Mix v16 (базовая модель для transfer)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_mix_v16

# 2. Detect Mix v26 (независимо)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_mix_v26

# 3. Detect Transfer (после завершения Этапа 3!)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_detect.sh conf_r_detect_transfer

# 4. Ranking (параллельно)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_ranking.sh

# 5. Embedding (параллельно)
/ml_core_binaries/qwarium-agent proc spawn -- /qwarium/home/d.a.lanovenko/llm-detect-ai/scripts/run_r_embed.sh
```

### Вариант B: Параллельный (для экономии времени)
```bash
# Можно запускать параллельно (независимые):
# - Detect Mix v16
# - Detect Mix v26  
# - Ranking
# - Embedding

# Transfer запускается ТОЛЬКО после завершения Mix v16
```

---

## ⚠️ Критические Зависимости

| Зависимость | Тип | Последствия |
|-------------|-----|-------------|
| **Transfer → Mix v16** | 🔗 Критичная | Transfer не запустится без весов Mix v16 |
| **Mix v16 → ai_mix_v16** | ⚠️ Датасет | Ошибка при отсутствии датасета |
| **Mix v26 → ai_mix_v26** | ⚠️ Датасет | Ошибка при отсутствии датасета |
| **Ranking → ai_mix_for_ranking** | ⚠️ Датасет | Ошибка при отсутствии датасета |
| **Embed → ai_mix_v26** | ⚠️ Датасет | Ошибка при отсутствии датасета |

---

## 💾 Расположение Файлов

```
/qwarium/home/d.a.lanovenko/
├── llm-detect-ai/
│   ├── datasets/
│   │   ├── train_essays.csv          # Базовый датасет
│   │   └── external/                 # Симлинки на /tmp
│   │       ├── ai_mix_v16 -> /tmp/datasets/external/ai_mix_v16
│   │       ├── ai_mix_v26 -> /tmp/datasets/external/ai_mix_v26
│   │       └── ai_mix_for_ranking -> /tmp/datasets/external/ai_mix_for_ranking
│   ├── scripts/
│   │   ├── download_datasets.sh      # Скрипт скачивания
│   │   └── run_*.sh                  # Скрипты запуска
│   └── conf/
│       └── r_*/conf_*.yaml           # Конфигурации
│
├── models/                           # Сохранение моделей
│   ├── r_detect_mix_v16/
│   ├── r_detect_mix_v26/
│   ├── r_detect_transfer/
│   ├── r_ranking/
│   └── r_embed/
│
└── /tmp/datasets/external/           # Временное хранение датасетов
    ├── ai_mix_v16/
    ├── ai_mix_v26/
    └── ai_mix_for_ranking/
```

---

## 📝 Примечания

1. **Датасеты в /tmp** могут быть удалены при перезагрузке сервера. Для постоянного хранения скопируйте в `/qwarium/home/d.a.lanovenko/datasets/external/`

2. **Transfer Learning** требует успешного завершения обучения Mix v16 и сохранения весов в `models/r_detect_mix_v16/best/`

3. **CLM** этап использует датасет PERSUADE 2.0 (скачан в `datasets/persuade_2.0_human_scores_demo_id_github.csv`)

4. **DPO** этап требует preference dataset (пока не доступен)

4. **Валидация**: Для Detect Mix v26 уже есть готовый valid набор, для остальных используется случайное разделение 99/1

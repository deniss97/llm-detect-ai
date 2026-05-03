# 📍 Расположение весов моделей и компонентов ансамбля

**Дата:** 3 мая 2026  
**Автор:** d.a.lanovenko

---

## 🗂️ Общая структура

Все модели хранятся в директории:
```
/qwarium/home/d.a.lanovenko/models/
```

---

## 🎯 Компоненты ансамбля

### 1️⃣ Detection модели (Mistral-7B + LoRA)

#### 1.1 r_detect_retrain ⭐ (основная detection модель)
**Путь:** `/qwarium/home/d.a.lanovenko/models/r_detect_retrain/`

```
r_detect_retrain/
├── best/
│   ├── adapter_config.json          # Конфигурация LoRA
│   └── adapter_model.safetensors    # Веса LoRA адаптера (~30MB)
├── last/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── oof_df_best.csv                  # OOF предсказания (train)
├── oof_df_last.csv
├── result_df_best.csv               # Предсказания на test/val
└── result_df_last.csv
```

**Base модель:** `mistralai/Mistral-7B-v0.1` (загружается из HuggingFace)  
**LoRA веса:** `best/adapter_model.safetensors`  
**ROC-AUC:** 0.818

---

#### 1.2 r_detect_mix_v16
**Путь:** `/qwarium/home/d.a.lanovenko/models/r_detect_conf_r_detect_mix_v16/`

```
r_detect_conf_r_detect_mix_v16/
├── best/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── last/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── oof_df_best.csv
├── oof_df_last.csv
├── result_df_best.csv
└── result_df_last.csv
```

**Base модель:** `mistralai/Mistral-7B-v0.1`  
**Конфигурация:** `conf/r_detect/conf_r_detect_mix_v16.yaml`

---

#### 1.3 r_detect_mix_v26
**Путь:** `/qwarium/home/d.a.lanovenko/models/r_detect_conf_r_detect_mix_v26/`

```
r_detect_conf_r_detect_mix_v26/
├── last/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── oof_df_last.csv
└── result_df_last.csv
```

**Base модель:** `mistralai/Mistral-7B-v0.1`  
**Конфигурация:** `conf/r_detect/conf_r_detect_mix_v26.yaml`

---

#### 1.4 r_detect_transfer
**Путь:** `/qwarium/home/d.a.lanovenko/models/r_detect_conf_r_detect_transfer/`

```
r_detect_conf_r_detect_transfer/
├── last/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── oof_df_last.csv
└── result_df_last.csv
```

**Base модель:** `mistralai/Mistral-7B-v0.1`  
**Конфигурация:** `conf/r_detect/conf_r_detect_transfer.yaml`

---

#### 1.5 r_detect_competition
**Путь:** `/qwarium/home/d.a.lanovenko/models/r_detect_competition/`

```
r_detect_competition/
├── last/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── oof_df_last.csv
└── result_df_last.csv
```

**Base модель:** `mistralai/Mistral-7B-v0.1`  
**Конфигурация:** `conf/r_detect/conf_r_detect_competition.yaml`

---

### 2️⃣ Embedding модель (DeBERTa-v3-Base + KNN) ⭐⭐⭐

**Путь:** `/qwarium/home/d.a.lanovenko/models/r_embed_conf_r_embed/`

```
r_embed_conf_r_embed/
├── detect_ai_model_best.pth.tar     # Полные веса модели (~700MB)
└── detect_ai_model_last.pth.tar
```

**Base модель:** `microsoft/deberta-v3-base`  
**Веса:** `detect_ai_model_best.pth.tar` (738 MB)  
**ROC-AUC:** 0.9885 (ЛУЧШАЯ одиночная модель)

**Структура checkpoint:**
```python
checkpoint = {
    'state_dict': {...},      # Веса модели
    'epoch': ...,             # Эпоха
    'optimizer': {...},       # Состояние оптимизатора
    'config': {...}           # Конфигурация
}
```

---

### 3️⃣ Ranking модель (DeBERTa-v3-Large)

**Путь:** `/qwarium/home/d.a.lanovenko/models/r_ranking_conf_r_ranking_large/`

```
r_ranking_conf_r_ranking_large/
├── oof_df_best.csv           # OOF предсказания
├── oof_df_last.csv
├── result_df_best.csv        # Предсказания на test/val
└── result_df_last.csv
```

⚠️ **ВНИМАНИЕ:** Веса модели НЕ сохранены в этой директории!

**Base модель:** `microsoft/deberta-v3-large`  
**ROC-AUC:** 0.9719

**Где веса:**
- Вариант 1: Веса загружаются динамически из HuggingFace при инференсе
- Вариант 2: Веса могут быть в `/tmp/models/r_ranking_*` (symlink)
- Вариант 3: Нужно дообучить модель заново

**Рекомендация:** Сохранить веса после обучения:
```python
torch.save({
    'state_dict': model.state_dict(),
    'config': config
}, '/qwarium/home/d.a.lanovenko/models/r_ranking_conf_r_ranking_large/ranking_model_best.pth.tar')
```

---

## 📊 Сводная таблица компонентов

| Компонент | Путь | Тип | Размер | ROC-AUC | Статус |
|-----------|------|-----|--------|---------|--------|
| **r_detect_retrain** | `models/r_detect_retrain/best/` | LoRA Adapter | ~30MB | 0.818 | ✅ Готов |
| **r_detect_mix_v16** | `models/r_detect_conf_r_detect_mix_v16/best/` | LoRA Adapter | ~30MB | - | ✅ Готов |
| **r_detect_mix_v26** | `models/r_detect_conf_r_detect_mix_v26/last/` | LoRA Adapter | ~30MB | - | ✅ Готов |
| **r_detect_transfer** | `models/r_detect_conf_r_detect_transfer/last/` | LoRA Adapter | ~30MB | - | ✅ Готов |
| **r_detect_competition** | `models/r_detect_competition/last/` | LoRA Adapter | ~30MB | - | ✅ Готов |
| **r_embed_conf_r_embed** | `models/r_embed_conf_r_embed/` | Full Model | 738MB | **0.9885** | ✅ Готов |
| **r_ranking_conf_r_ranking_large** | `models/r_ranking_conf_r_ranking_large/` | ❌ НЕТ ВЕСОВ | - | 0.9719 | ⚠️ Требуется |

---

## 🔧 Как использовать каждый компонент

### Detection модели (LoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base модель
base_model_path = "mistralai/Mistral-7B-v0.1"
adapter_path = "/qwarium/home/d.a.lanovenko/models/r_detect_retrain/best/"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
```

### Embedding модель

```python
import torch
from transformers import AutoModel, AutoTokenizer

checkpoint_path = "/qwarium/home/d.a.lanovenko/models/r_embed_conf_r_embed/detect_ai_model_best.pth.tar"
base_model_path = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModel.from_pretrained(base_model_path)

# Загрузка весов
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)
```

### Ranking модель (требует сохранения)

```python
# После обучения сохранить:
torch.save({
    'state_dict': model.state_dict(),
    'config': config
}, '/qwarium/home/d.a.lanovenko/models/r_ranking_conf_r_ranking_large/ranking_model_best.pth.tar')

# Загрузка:
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
```

---

## 📁 Symlinks (символические ссылки)

```
models/r_detect_mini -> /tmp/models/r_detect_mini
models/r_detect_competition_tmp -> /tmp/models/r_detect_competition
```

⚠️ **Внимание:** Эти ссылки могут указывать на временные файлы в `/tmp/`, которые могут быть удалены.

---

## 🎯 Минимальный набор для воспроизведения ансамбля

Для воспроизведения лучшего ансамбля (ROC-AUC 0.9912) необходимы:

### Обязательные компоненты:
1. ✅ `models/r_embed_conf_r_embed/detect_ai_model_best.pth.tar` (738MB)
2. ✅ `models/r_detect_retrain/best/adapter_model.safetensors` (~30MB)
3. ⚠️ `models/r_ranking_conf_r_ranking_large/` — **нужно сохранить веса!**

### Base модели (загружаются из HuggingFace):
- `mistralai/Mistral-7B-v0.1`
- `microsoft/deberta-v3-base`
- `microsoft/deberta-v3-large`

---

## 📋 Чеклист для полного бэкапа

```bash
# 1. Проверка наличия всех весов
ls -lh /qwarium/home/d.a.lanovenko/models/r_embed_conf_r_embed/*.pth.tar
ls -lh /qwarium/home/d.a.lanovenko/models/r_detect_retrain/best/adapter_model.safetensors
ls -lh /qwarium/home/d.a.lanovenko/models/r_ranking_conf_r_ranking_large/*.pth.tar  # ⚠️ Может отсутствовать

# 2. Копирование на внешний носитель
rsync -av /qwarium/home/d.a.lanovenko/models/ /path/to/backup/models/

# 3. Сохранение ranking модели (если отсутствует)
# Запустить обучение и сохранить веса
```

---

## 🚨 Критические замечания

1. **Ranking модель не имеет сохранённых весов** — нужно сохранить после обучения
2. **Symlinks в /tmp/** — могут быть удалены при перезагрузке
3. **Base модели** — загружаются из HuggingFace, нужен интернет для первого запуска
4. **Размер embedding модели** — 738MB, учитывать при бэкапе

---

## 📞 Контакты

По вопросам доступа к весам моделей обращаться: d.a.lanovenko

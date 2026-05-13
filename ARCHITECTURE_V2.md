# Современная Архитектура Детекции AI-Текста v2.0 🚀

## Обновлённая архитектура для русского языка (2024-2025)

### 📐 Архитектура ансамбля

```
┌──────────────────────────────────────────────────────────────┐
│                  Сочинение на русском                         │
└─────────────────────────┬────────────────────────────────────┘
                          ▼
        ┌─────────────────────────────────┐
        │     3 ПАРАЛЛЕЛЬНЫХ КОМПОНЕНТА   │
        └─────────────────────────────────┘
                          │
    ┌────────────┬────────┼────────┬─────────────┐
    ▼            ▼        ▼        ▼             ▼
┌───────┐   ┌───────┐ ┌──────┐ ┌──────┐  ┌─────────────┐
│Detect │   │Detect │ │Embed │ │Rank  │  │  Zero-shot  │
│T-lite │   │Qwen3  │ │bge-m3│ │bge-v2│  │ Binoculars  │
│ +DoRA │   │ +DoRA │ │+contr│ │      │  │ +DetectGPT  │
└───┬───┘   └───┬───┘ └──┬───┘ └──┬───┘  └──────┬──────┘
    │           │        │        │              │
    └───────────┴────────┴────────┴──────────────┘
                         ▼
            ┌────────────────────────┐
            │   Stacking Meta-model  │
            │   (CatBoost / LogReg)  │
            └────────────┬───────────┘
                         ▼
                    P(AI) ∈ [0,1]
```

---

## 🎯 1. Detection Модели — Главный Апгрейд

### ❌ Старое: `mistralai/Mistral-7B-v0.1`
**Проблема:** обучена в основном на английском, для русского — слабые representations.

### ✅ Новое: ансамбль из 3 моделей

#### 🥇 Detector #1: **T-lite-it-1.0** (T-Bank)
```python
model_name = "t-tech/T-lite-it-1.0"  # 7B параметров
```
**Преимущества:**
- SOTA на русском в классе 7B (Q1 2025)
- Специально дообучена T-Bank на русском корпусе поверх Qwen2.5
- Понимает культурный контекст, школьную лексику

#### 🥈 Detector #2: **Qwen3-8B** (Alibaba)
```python
model_name = "Qwen/Qwen3-8B"
```
**Преимущества:**
- Один из лучших мультиязычных open-source LLM
- Контекст 128K, поддержка thinking-mode
- Архитектурно отличается от T-lite → разнообразие в ансамбле

#### 🥉 Detector #3: **Vikhr-Nemo-12B-Instruct-R**
```python
model_name = "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"
```
**Преимущества:**
- Российский проект, заточен под русский с нуля
- 12B → больше capacity
- Третье архитектурное семейство (Mistral) → ортогональность ансамбля

---

## 🔍 2. Embedding Модель — Апгрейд для Русского

### ❌ Старое: `microsoft/deberta-v3-base`
**Проблема:** английская модель, для русского нужна `mdeberta` (она слабее).

### ✅ Новое: **`deepvk/USER-bge-m3`**
```python
model_name = "deepvk/USER-bge-m3"  # 0.5B, SOTA для русского
```

**Преимущества:**
- Дотюнен из BGE-M3 на русском (deepvk — известная команда)
- Топ MTEB-ru в категории до 1B
- Контекст 8192 токена → влезают целые сочинения
- Лёгкий → дотюнинг на H100 за 30 мин

### 🔧 Современный Contrastive Learning
```python
# Matryoshka + MultipleNegativesRankingLoss
train_loss = losses.MatryoshkaLoss(
    model=model,
    loss=losses.MultipleNegativesRankingLoss(model),
    matryoshka_dims=[1024, 768, 512, 256, 128],
)
```

### 🚀 FAISS HNSW вместо KNN
```python
# HNSW для миллионных корпусов (быстрее обычного KNN)
index = faiss.IndexHNSWFlat(dim, 32)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 64
```

---

## 🏆 3. Ranking Модель — Апгрейд

### ❌ Старое: `microsoft/deberta-v3-large`
**Проблема:** английская модель, теряет качество на русском.

### ✅ Новое: **`BAAI/bge-reranker-v2-m3`**
```python
model_name = "BAAI/bge-reranker-v2-m3"  # 0.5B, мультиязычный SOTA
```

**Преимущества:**
- Лучший мультиязычный reranker на MTEB
- Изначально pair-wise архитектура
- Контекст 8192

---

## 🆕 4. Новые Компоненты Ансамбля

### A. Binoculars (zero-shot) 🔭
```python
# Отношение perplexity к cross-perplexity
# для пары моделей (base, instruct)
score = ppl / x_ppl  # < 0.9 → вероятно AI
```

### B. Fast-DetectGPT (zero-shot) 🔍
```python
# AI-тексты лежат в локальных максимумах
# log-probability surface
```

### C. Лингвистические фичи + CatBoost 📊
```python
# Морфология, пунктуация, burstiness
# через pymorphy3 + CatBoost на GPU
```

---

## 📊 5. Ожидаемое Улучшение

| Компонент | Было | Стало | Δ ROC-AUC |
|-----------|------|-------|-----------|
| Detection #1 | Mistral-7B (0.818) | T-lite-7B + DoRA | **+0.10-0.14** → ~0.93-0.96 |
| Detection #2 | — | Qwen3-8B + DoRA | новый |
| Detection #3 | — | Vikhr-Nemo-12B + QLoRA | новый |
| Embedding | DeBERTa-v3 (0.9885) | USER-bge-m3 | **+0.005-0.01** |
| Ranking | DeBERTa-large (0.9719) | bge-reranker-v2-m3 | **+0.013-0.018** |
| + Binoculars | — | Qwen2.5-7B base+instruct | +0.005-0.015 |
| + Fast-DetectGPT | — | Qwen2.5-7B | +0.003-0.01 |
| + Лингв. фичи | — | pymorphy3 + CatBoost | +0.003-0.008 |
| **Финальный ансамбль** | — | Stacking CatBoost | **🎯 0.995-0.998** |

---

## 🖥️ 6. Использование H100-80GB

### Оценка времени и VRAM

| Этап | Модель | VRAM (peak) | Время |
|------|--------|-------------|-------|
| Дотюнинг T-lite-7B + DoRA | T-lite | ~28 GB | ~1.5 ч |
| Дотюнинг Qwen3-8B + DoRA | Qwen3 | ~32 GB | ~2 ч |
| Дотюнинг Vikhr-Nemo-12B + QLoRA | Vikhr | ~24 GB (4-bit) | ~2.5 ч |
| Дотюнинг USER-bge-m3 | bge-m3 | ~16 GB | ~30 мин |
| Дотюнинг bge-reranker-v2-m3 | bge-reranker | ~14 GB | ~30 мин |
| Binoculars (inference) | 2× Qwen2.5-7B | ~32 GB | ~30 мин |
| Fast-DetectGPT (inference) | Qwen2.5-7B | ~16 GB | ~20 мин |
| Inference всех детекторов | — | ~30 GB | ~1.5 ч |
| **ИТОГО на H100-1x** | | | **~10-12 часов** |

**Бюджет:** ~$25-50 на полный пайплайн при цене H100 ~$2-4/час.

---

## 📋 7. Чеклист для Запуска

```bash
# 1. Окружение
conda create -n detect python=3.11
conda activate detect
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.0 peft==0.13.0 accelerate==1.0.0
pip install sentence-transformers==3.3.0
pip install flash-attn==2.7.0 --no-build-isolation
pip install bitsandbytes catboost faiss-gpu pymorphy3 wandb

# 2. Скачивание моделей (с HF mirror, если из РФ)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download t-tech/T-lite-it-1.0
huggingface-cli download Qwen/Qwen3-8B
huggingface-cli download Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24
huggingface-cli download deepvk/USER-bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3
huggingface-cli download Qwen/Qwen2.5-7B
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# Итого: ~70-80 GB на диске
```

---

## 🚀 8. План Действий (1-2 недели)

| Неделя | День | Задача |
|--------|------|--------|
| **1** | 1-2 | Скачивание моделей, подготовка окружения |
| | 3-4 | Дотюнинг T-lite-7B + DoRA |
| | 5-6 | Дотюнинг Qwen3-8B и Vikhr-Nemo-12B |
| | 7 | Дотюнинг USER-bge-m3 + FAISS HNSW |
| **2** | 8 | Дотюнинг bge-reranker-v2-m3 |
| | 9 | Реализация Binoculars + Fast-DetectGPT |
| | 10 | CatBoost на лингв. фичах |
| | 11-12 | Stacking, калибровка, тестирование |
| | 13-14 | Анализ ошибок, SHAP, написание диплома |

---

## 🎯 TL;DR

**Меняем:**
- Mistral-7B → **T-lite-7B + Qwen3-8B + Vikhr-Nemo-12B** (3 модели в ансамбле)
- DeBERTa-v3-base → **deepvk/USER-bge-m3** (русский SOTA эмбеддер)
- DeBERTa-v3-large → **BAAI/bge-reranker-v2-m3** (мультиязычный SOTA reranker)

**Добавляем:**
- DoRA + RSLoRA (вместо обычной LoRA)
- FlashAttention-2/3 (для H100)
- Длинный контекст 2048 (вместо 256)
- Binoculars + Fast-DetectGPT (zero-shot)
- CatBoost на pymorphy3 фичах
- Stacking-meta модель

**Получаем:**
- ROC-AUC: **0.818 → 0.995-0.998**
- Время на H100-1x: **~10-12 часов**
- Бюджет: **~$25-50** на GPU + **$30-80** на API-аугментацию

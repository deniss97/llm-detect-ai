# 📖 Инструкция по использованию Qwen2.5-7B Detection модели

## 📍 Путь к модели

```
/tmp/llm_cache/models/r_detect_qwen3/
```

### Структура файлов:
```
r_detect_qwen3/
├── adapter_config.json          # Конфигурация LoRA+DoRA
├── adapter_model.safetensors    # Веса адаптера (~330 MB)
├── tokenizer_config.json        # Конфигурация токенизатора
├── training_args.bin            # Параметры обучения
└── special_tokens_map.json      # Специальные токены
```

**Базовая модель:** `Qwen/Qwen2.5-7B-Instruct` (загружается автоматически из HuggingFace)

---

## 🚀 Быстрый старт (Inference)

### Вариант 1: Использование готового скрипта

```bash
python3 /qwarium/home/d.a.lanovenko/llm-detect-ai/code/evaluate/eval_r_detect_qwen3.py \
    --model_path /tmp/llm_cache/models/r_detect_qwen3 \
    --test_data /qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/final_valid.csv \
    --output_dir /qwarium/home/d.a.lanovenko/llm-detect-ai/results/
```

### Вариант 2: Программное использование в коде

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import pandas as pd

# === 1. Загрузка модели ===
MODEL_PATH = "/tmp/llm_cache/models/r_detect_qwen3"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

print("Загрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Загрузка базовой модели...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # Требуется flash-attn
)

print("Загрузка LoRA адаптера...")
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

# === 2. Функция для предсказания ===
def predict_ai_probability(text: str, threshold: float = 0.10) -> dict:
    """
    Предсказывает вероятность того, что текст сгенерирован AI.
    
    Args:
        text: Текст сочинения
        threshold: Порог классификации (по умолчанию 0.10)
    
    Returns:
        dict с результатами:
            - ai_probability: вероятность AI (0-1)
            - is_ai: булево значение (True если AI)
            - confidence: уверенность модели
    """
    # Токенизация
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True
    ).to(model.device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        ai_prob = probabilities[0, 1].item()  # Вероятность класса AI
    
    # Классификация
    is_ai = ai_prob > threshold
    
    return {
        "ai_probability": ai_prob,
        "is_ai": is_ai,
        "confidence": abs(ai_prob - threshold),
        "threshold": threshold
    }

# === 3. Пример использования ===
text_human = "В своем произведении Горький ставит вопрос о смысле жизни..."
text_ai = "В современном мире вопрос о смысле жизни является актуальным..."

result_human = predict_ai_probability(text_human)
result_ai = predict_ai_probability(text_ai)

print(f"Human текст: AI probability = {result_human['ai_probability']:.4f}, "
      f"is_ai = {result_human['is_ai']}")
print(f"AI текст: AI probability = {result_ai['ai_probability']:.4f}, "
      f"is_ai = {result_ai['is_ai']}")

# === 4. Пакетная обработка ===
def predict_batch(texts: list, threshold: float = 0.10, batch_size: int = 8) -> list:
    """
    Пакетная обработка текстов.
    
    Args:
        texts: Список текстов
        threshold: Порог классификации
        batch_size: Размер батча
    
    Returns:
        Список словарей с результатами
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Токенизация батча
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        ).to(model.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            ai_probs = probabilities[:, 1].cpu().numpy()
        
        # Обработка результатов
        for j, ai_prob in enumerate(ai_probs):
            results.append({
                "text": batch_texts[j][:100] + "...",  # Первые 100 символов
                "ai_probability": float(ai_prob),
                "is_ai": bool(ai_prob > threshold),
                "confidence": float(abs(ai_prob - threshold))
            })
    
    return results

# Пример пакетной обработки
texts = [
    "Текст 1...",
    "Текст 2...",
    "Текст 3...",
]

batch_results = predict_batch(texts)
for result in batch_results:
    print(f"Text: {result['text'][:50]}... | AI prob: {result['ai_probability']:.4f} | AI: {result['is_ai']}")
```

---

## 📊 Параметры модели

| Параметр | Значение |
|----------|----------|
| **Базовая модель** | Qwen/Qwen2.5-7B-Instruct |
| **Адаптер** | LoRA + DoRA |
| **Rank (r)** | 32 |
| **Alpha** | 64 |
| **Dropout** | 0.05 |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Max length** | 2048 токенов |
| **Precision** | BFloat16 |
| **Attention** | FlashAttention-2 |
| **Threshold** | 0.10 (оптимальный) |

---

## 🎯 Метрики на тесте (final_dataset)

| Метрика | Значение |
|---------|----------|
| **ROC-AUC** | **1.0000** |
| **Accuracy** | **1.0000** (100%) |
| **F1 Score** | **1.0000** |
| **Precision** | **1.0000** |
| **Recall** | **1.0000** |
| **Samples** | 491 (248 human + 243 AI) |

### Метрики по моделям:
| Модель | Samples | Accuracy |
|--------|---------|----------|
| human | 248 | 1.0 |
| google/gemini-3.1-flash-lite | 90 | 1.0 |
| qwen/qwen-2.5-72b-instruct | 73 | 1.0 |
| openai/gpt-4o-mini | 80 | 1.0 |

---

## ⚙️ Требования

```bash
# Основные зависимости
torch==2.4.0
transformers==4.46.0
peft==0.19.1
accelerate==1.13.0

# Для FlashAttention-2
flash-attn==2.7.0

# Дополнительные
pandas
numpy
scikit-learn
```

### Установка:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.46.0 peft==0.19.1 accelerate==1.13.0
pip install flash-attn==2.7.0 --no-build-isolation
pip install pandas numpy scikit-learn
```

---

## 🔧 Настройка порога классификации

По умолчанию используется **threshold = 0.10**, что обеспечивает идеальные метрики.

Если нужно изменить баланс между Precision и Recall:

```python
# Более строгий порог (меньше False Positive, но больше False Negative)
result = predict_ai_probability(text, threshold=0.50)

# Более мягкий порог (больше False Positive, но меньше False Negative)
result = predict_ai_probability(text, threshold=0.05)
```

### Рекомендации:
- **threshold = 0.10** - оптимальный баланс (использовался в обучении)
- **threshold = 0.05** - максимальный Recall (ловить все AI тексты)
- **threshold = 0.50** - максимальный Precision (минимизировать ложные срабатывания)

---

## 📁 Примеры использования

### Пример 1: Проверка одного текста

```python
from code.evaluate.eval_r_detect_qwen3 import predict_ai_probability

text = "В своем произведении автор поднимает важные вопросы..."
result = predict_ai_probability(text)

print(f"AI вероятность: {result['ai_probability']:.2%}")
print(f"Это AI текст: {'Да' if result['is_ai'] else 'Нет'}")
print(f"Уверенность: {result['confidence']:.2%}")
```

### Пример 2: Обработка CSV файла

```python
import pandas as pd
from code.evaluate.eval_r_detect_qwen3 import predict_batch

# Загрузка данных
df = pd.read_csv("essays.csv")
texts = df["text"].tolist()

# Предсказания
results = predict_batch(texts, threshold=0.10, batch_size=8)

# Сохранение результатов
df["ai_probability"] = [r["ai_probability"] for r in results]
df["is_ai"] = [r["is_ai"] for r in results]
df.to_csv("essays_with_predictions.csv", index=False)

print(f"Обработано {len(df)} текстов")
print(f"AI текстов: {df['is_ai'].sum()}")
print(f"Human текстов: {len(df) - df['is_ai'].sum()}")
```

### Пример 3: Интеграция в API

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    threshold: float = 0.10

@app.post("/predict")
async def predict(request: TextRequest):
    result = predict_ai_probability(request.text, request.threshold)
    return result

# Запуск: uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## 🚨 Возможные проблемы и решения

### Проблема 1: "FlashAttention2 has been toggled on, but it cannot be used"

**Решение:** Установите flash-attn или отключите его в коде:

```bash
# Вариант A: Установить flash-attn
pip install flash-attn==2.7.0 --no-build-isolation

# Вариант B: Отключить в коде (медленнее)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    attn_implementation="eager"  # вместо "flash_attention_2"
)
```

### Проблема 2: "CUDA out of memory"

**Решение:** Уменьшите max_length или используйте gradient checkpointing:

```python
inputs = tokenizer(
    text,
    max_length=1024,  # вместо 2048
    truncation=True
)
```

### Проблема 3: Модель загружается долго

**Решение:** Используйте кэширование:

```bash
export HF_HOME=/tmp/huggingface_cache
```

Или загрузите модель заранее:

```python
from transformers import AutoModelForSequenceClassification
AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

---

## 📞 Контакты и поддержка

При возникновении проблем:
1. Проверьте логи обучения: `/qwarium/home/d.a.lanovenko/llm-detect-ai/logs/train_r_detect_qwen3.log`
2. Изучите метрики: `/qwarium/home/d.a.lanovenko/llm-detect-ai/results/qwen3_final_dataset_metrics.json`
3. Откройте issue в репозитории

---

## 📚 Дополнительные ресурсы

- [HuggingFace Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [PEFT документация](https://huggingface.co/docs/peft)
- [FlashAttention документация](https://github.com/Dao-AILab/flash-attention)
- [Отчет с метриками](DETECTION_METRICS_REPORT.md)

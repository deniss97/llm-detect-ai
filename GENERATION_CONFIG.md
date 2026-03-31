# 📚 Конфигурация генерации AI-текстов для детекции

## 1. Модель для генерации

### Mistral-7B-Instruct-v0.2

| Параметр | Значение |
|----------|----------|
| **Название** | `mistralai/Mistral-7B-Instruct-v0.2` |
| **Архитектура** | Transformer Decoder-only |
| **Количество параметров** | 7 миллиардов |
| **Языки** | Multilingual (включая русский и английский) |
| **Размер модели** | ~14 GB (float16) |
| **Контекстное окно** | 8192 токена |
| **Лицензия** | Apache 2.0 |

### Почему Mistral-7B?

| Преимущество | Описание |
|--------------|----------|
| ✅ **Multilingual** | Понимает русский контекст, что критично для наших данных |
| ✅ **Качество генерации** | Лучше сохраняет культурный контекст, чем TinyLlama |
| ✅ **Размер** | Достаточно компактна для локального запуска |
| ✅ **Инструктивная версия** | Optimized для следования инструкциям |

---

## 2. Параметры генерации

### 2.1 Критически важные параметры

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| **`prefix_ratio`** | `0.6` (60%) | **Ключевой параметр!** Сохраняет 60% оригинального текста, чтобы AI продолжал в том же культурном контексте. Предотвращает замену русских авторов на западные. |
| **`temperature`** | `0.7-0.9` | Баланс между креативностью и когерентностью. Добавляем случайность (0.6 + random * 0.3) для разнообразия вариаций. |
| **`max_length`** | `1024` токена | Достаточно для полного эссе с запасом. |

### 2.2 Дополнительные параметры

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| **`top_p`** | `0.9` | Nucleus sampling — отбираем 90% вероятностной массы. Уменьшает бессмысленные генерации. |
| **`top_k`** | `50` | Ограничиваем выборку 50 наиболее вероятных токенов. |
| **`repetition_penalty`** | `1.1` | Штраф за повторения. Предотвращает зацикливание текста. |
| **`num_variations`** | `5` | 5 вариаций на каждое эссе для аугментации данных. |

---

## 3. Формат промпта

### Структура промпта для Mistral

```
<s>[INST] {instruction} [/INST] {prefix}
```

Где:

**`instruction`** — метаданные эссе:
```
Prompt: {название_темы}
Task: Essay Writing
Score: {оценка 4-6}
Student Grade Level: {класс 9-11}
English Language Learner: Non-ELL
Disability Status: No Disability
```

**`prefix`** — первые 60% оригинального эссе на английском

### Пример полного промпта

```
<s>[INST] 
Prompt: haraktery-i-sudby-pesy-na-dne_var0
Task: Essay Writing
Score: 5
Student Grade Level: 10
English Language Learner: Non-ELL
Disability Status: No Disability
[/INST] In his work, Gorky poses the question: "What is better, truth or compassion? What is more necessary? In fact, this question applies to absolutely every character in the play, because it talks about the tragic fates of people who find themselves at the bottom of social life. All the characters are different, each has their own destiny, their own path, which led them to the place where the play takes place - the shelter. Take, for example, the Actor. This is a drunkard trying in vain to get back to work...
```

---

## 4. Пайплайн генерации

### Шаг 1: Загрузка данных
```
datasets/Датасет.csv (русские эссе)
       ↓
```

### Шаг 2: Перевод на английский
```
Google Translate → datasets/translated_essays.csv
       ↓
```

### Шаг 3: Генерация вариаций
```
Mistral-7B + prefix_ratio=0.6 → datasets/generated_variations_mistral.csv
       ↓
```

### Шаг 4: Финальный датасет
```
Для каждого оригинала:
├── original_en (перевод 100%)
├── generated_en (AI: 60% оригинал + 40% генерация)
└── label: 0 = human, 1 = AI
```

---

## 5. Команда для запуска

### Полная генерация (все эссе, 5 вариаций)

```bash
cd /qwarium/home/d.a.lanovenko/llm-detect-ai

python3 code/r_clm/test_mistral_generate.py \
    --model_path /tmp/models/mistral_7b/last \
    --input datasets/translated_essays.csv \
    --output datasets/generated_variations_mistral_full.csv \
    --prefix_ratio 0.6 \
    --num_variations 5 \
    --max_essays -1 \
    --max_length 1024 \
    --seed 42
```

### Параметры командной строки

| Аргумент | Значение по умолчанию | Описание |
|----------|----------------------|----------|
| `--model_path` | `/tmp/models/mistral_7b/last` | Путь к сохранённой модели |
| `--input` | `datasets/translated_essays.csv` | Входной CSV с переведёнными эссе |
| `--output` | `datasets/generated_variations_mistral.csv` | Выходной CSV с генерациями |
| `--prefix_ratio` | `0.3` | Доля оригинала для continuation |
| `--num_variations` | `1` | Количество вариаций на эссе |
| `--max_essays` | `-1` | Максимум эссе (-1 = все) |
| `--max_length` | `1024` | Максимальная длина генерации |
| `--seed` | `42` | Random seed для воспроизводимости |

---

## 6. Ожидаемые результаты

### Статистика генерации

| Метрика | Значение |
|---------|----------|
| **Всего эссе** | ~1236 |
| **Вариаций на эссе** | 5 |
| **Итого AI-текстов** | ~6180 |
| **Время на эссе** | ~12 секунд |
| **Общее время** | ~21 час (последовательно) |

### Качество генерации (ожидаемое)

| Метрика | Значение |
|---------|----------|
| **Word overlap с оригиналом** | 65-75% |
| **Сохранение культурного контекста** | ✅ Да |
| **Связность текста** | ✅ Хорошая |
| **Галлюцинации** | ⚠️ Минимальные |

---

## 7. Почему это важно для детекции

### Проблема cultural bias (решена!)

**Без prefix_ratio (ПЛОХО):**
```
Оригинал: "Раскольников убил старушку..." (Достоевский)
AI: "Harry Potter defeated the villain..." (Западная литература)
```
→ Модель детектирует **культурные различия**, а не AI-паттерны!

**С prefix_ratio=0.6 (ХОРОШО):**
```
Оригинал: "Раскольников убил старушку..." (Достоевский)
AI: "Раскольников понял, что его теория неверна..." (Достоевский)
```
→ Модель детектирует **AI-стиль**, культурный контекст одинаковый!

---

## 8. Аппаратные требования

| Требование | Значение |
|------------|----------|
| **GPU память** | 14 GB минимум |
| **RAM** | 16 GB рекомендуется |
| **Диск** | 20 GB свободно (модель + кэш) |
| **Время** | ~12 сек/эссе |

---

## 9. Файлы

| Файл | Описание |
|------|----------|
| `code/r_clm/test_mistral_generate.py` | Скрипт генерации |
| `datasets/translated_essays.csv` | Переведённые оригиналы |
| `datasets/generated_variations_mistral.csv` | Результат генерации |
| `/tmp/models/mistral_7b/last/` | Сохранённая модель |

---

## 10. Контакты и поддержка

При возникновении проблем:
1. Проверьте наличие модели: `ls -la /tmp/models/mistral_7b/last/`
2. Проверьте память GPU: `nvidia-smi`
3. Убедитесь, что `translated_essays.csv` существует

# Скрипты для обучения моделей

Эти скрипты предназначены для запуска обучения моделей через `qwarium-agent proc spawn` для устойчивого выполнения, которое сохраняется после отключения SSH сессии.

## Доступные скрипты

| Скрипт | Назначение |
|--------|---------|
| `run_r_detect_t_lite.sh` | Обучение T-lite-7B Detection модели |
| `run_r_detect_qwen3.sh` | Обучение Qwen2.5-7B Detection модели |
| `run_r_embed_final.sh` | Обучение USER-bge-m3 Embedding модели |

## Использование

### Базовое использование

```bash
# Сделать скрипты исполняемыми (только первый раз)
chmod +x scripts/*.sh

# Запустить обучение T-lite-7B
/ml_core_binaries/qwarium-agent proc spawn -- scripts/run_r_detect_t_lite.sh

# Запустить обучение Qwen2.5-7B
/ml_core_binaries/qwarium-agent proc spawn -- scripts/run_r_detect_qwen3.sh

# Запустить обучение USER-bge-m3
/ml_core_binaries/qwarium-agent proc spawn -- scripts/run_r_embed_final.sh
```

## Переменные окружения

Следующие переменные окружения устанавливаются автоматически:
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` - Оптимизация памяти
- `HF_HOME=/tmp/huggingface_cache` - Расположение кэша HuggingFace
- `HF_DATASETS_CACHE=/tmp/hf_datasets_cache` - Расположение кэша датасетов

## Расположение результатов

### Модели (сохраняются в /tmp/llm_cache/models/)
- T-lite-7B: `/tmp/llm_cache/models/r_detect_t_lite_v2/`
- Qwen2.5-7B: `/tmp/llm_cache/models/r_detect_qwen3/`
- USER-bge-m3: `/tmp/llm_cache/models/r_embed_final/`

### Логи
Все логи сохраняются в стандартный вывод процесса.

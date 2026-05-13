#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка модели T-lite-it-1.0 на final_dataset.csv
Упрощенная версия с прямой загрузкой весов
"""

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ОЦЕНКА T-LITE DETECTION МОДЕЛИ НА FINAL_DATASET")
print("=" * 80)

# === Загрузка данных ===
print("\n[1/5] Загрузка данных...")
df = pd.read_csv('/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_dataset.csv')

# Разделение на train/valid/test (те же сплиты, что и при обучении)
np.random.seed(42)

# Создаем сплит по source_id (чтобы пары human+AI были вместе)
unique_source_ids = df['source_id'].unique()
np.random.shuffle(unique_source_ids)

n_total = len(unique_source_ids)
n_train = int(0.8 * n_total)
n_valid = int(0.1 * n_total)

train_ids = unique_source_ids[:n_train]
valid_ids = unique_source_ids[n_train:n_train + n_valid]
test_ids = unique_source_ids[n_train + n_valid:]

train_df = df[df['source_id'].isin(train_ids)].reset_index(drop=True)
valid_df = df[df['source_id'].isin(valid_ids)].reset_index(drop=True)
test_df = df[df['source_id'].isin(test_ids)].reset_index(drop=True)

print(f"Train: {len(train_df)} сэмплов")
print(f"Valid: {len(valid_df)} сэмплов")
print(f"Test: {len(test_df)} сэмплов")

# === Загрузка модели ===
print("\n[2/5] Загрузка модели...")
os.environ['HF_HOME'] = '/tmp/huggingface_cache_eval'

model_path = '/tmp/llm_cache/models/r_detect_t_lite_v2'
base_model_name = 't-tech/T-lite-it-1.0'

print(f"Используем HF_HOME={os.environ['HF_HOME']}")

# Загружаем конфиг адаптера
with open(f'{model_path}/adapter_config.json', 'r') as f:
    adapter_config = json.load(f)

print(f"Конфигурация адаптера:")
print(f"  - r: {adapter_config['r']}")
print(f"  - lora_alpha: {adapter_config['lora_alpha']}")
print(f"  - use_dora: {adapter_config['use_dora']}")
print(f"  - use_rslora: {adapter_config['use_rslora']}")

# Загружаем токенизатор
print(f"\nЗагрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir='/tmp/huggingface_cache_eval')

# Загружаем базовую модель
print(f"Загрузка базовой модели...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    cache_dir='/tmp/huggingface_cache_eval'
)

# Загружаем веса адаптера вручную
print(f"Загрузка LoRA весов...")
import glob
adapter_files = glob.glob(f'{model_path}/adapter_*.safetensors')
print(f"Найдены файлы: {adapter_files}")

from peft import LoraConfig, get_peft_model, TaskType

# Создаем конфигурацию LoRA
lora_config = LoraConfig(
    r=adapter_config['r'],
    lora_alpha=adapter_config['lora_alpha'],
    lora_dropout=adapter_config['lora_dropout'],
    target_modules=adapter_config['target_modules'],
    use_rslora=adapter_config['use_rslora'],
    use_dora=adapter_config['use_dora'],
    task_type=TaskType.SEQ_CLS,
    modules_to_save=['score'],
)

# Применяем LoRA к базовой модели
model = get_peft_model(base_model, lora_config)

# Загружаем веса адаптера
from safetensors.torch import load_file
adapter_weights = {}
for adapter_file in adapter_files:
    weights = load_file(adapter_file)
    for key, value in weights.items():
        adapter_weights[key] = value

# Загружаем веса в модель
model.load_state_dict(adapter_weights, strict=False)
model.eval()

print(f"Модель загружена: {base_model_name}")
print(f"Адаптер загружен: {model_path}")

# === Инференс ===
print("\n[3/5] Инференс на тесте...")

def predict_batch(texts, model, tokenizer, batch_size=16):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=2048,
            padding=True,
            return_tensors='pt'
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Для num_labels=1 используем sigmoid
            probs = torch.sigmoid(logits).flatten().float()  # конвертируем в float32
            predictions.extend(probs.cpu().numpy())
    
    return np.array(predictions)

test_texts = test_df['text'].tolist()
test_labels = (test_df['model'] != 'human').astype(int).values  # 1 = AI, 0 = human

print(f"Предсказания для {len(test_texts)} сэмплов...")
test_probs = predict_batch(test_texts, model, tokenizer, batch_size=8)

# === Метрики ===
print("\n[4/5] Вычисление метрик...")

# ROC-AUC
roc_auc = roc_auc_score(test_labels, test_probs)

# Оптимальный threshold по Youden's J statistic
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Оптимальный threshold: {optimal_threshold:.4f}")

# Метрики с оптимальным threshold
test_preds = (test_probs >= optimal_threshold).astype(int)

accuracy = accuracy_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)

print(f"\n{'='*60}")
print(f"ОБЩИЕ МЕТРИКИ (threshold = {optimal_threshold:.4f})")
print(f"{'='*60}")
print(f"ROC-AUC:  {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:   {recall:.4f}")

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)
print(f"\nConfusion Matrix:")
print(f"[[{cm[0,0]:4d}  {cm[0,1]:4d}]  [True Negative  False Positive]")
print(f" [{cm[1,0]:4d}  {cm[1,1]:4d}]] [False Negative True Positive]")

# Метрики по моделям
print(f"\n{'='*60}")
print("МЕТРИКИ ПО МОДЕЛЯМ")
print(f"{'='*60}")

results = []
for model_name in test_df['model'].unique():
    model_mask = test_df['model'] == model_name
    model_labels = (test_df.loc[model_mask, 'model'] != 'human').astype(int).values
    model_probs = test_probs[model_mask]
    model_preds = (model_probs >= optimal_threshold).astype(int)
    
    if len(model_labels) > 0:
        model_acc = accuracy_score(model_labels, model_preds)
        if len(np.unique(model_labels)) > 1:
            model_f1 = f1_score(model_labels, model_preds)
        else:
            model_f1 = 'N/A'
        
        print(f"{model_name:40s} | Samples: {len(model_labels):4d} | Accuracy: {model_acc:.4f} | F1: {model_f1}")
        
        results.append({
            'model': model_name,
            'samples': len(model_labels),
            'accuracy': model_acc,
            'f1': model_f1 if model_f1 != 'N/A' else None
        })

# Сохранение результатов
print("\n[5/5] Сохранение результатов...")

test_results_df = test_df.copy()
test_results_df['prediction_prob'] = test_probs
test_results_df['prediction'] = test_preds
test_results_df['true_label'] = test_labels

test_results_df.to_csv('/qwarium/home/d.a.lanovenko/llm-detect-ai/results/t_lite_test_results.csv', index=False)
print(f"Результаты сохранены: results/t_lite_test_results.csv")

# Метрики по моделям
metrics_df = pd.DataFrame(results)
metrics_df.to_csv('/qwarium/home/d.a.lanovenko/llm-detect-ai/results/t_lite_metrics_by_model.csv', index=False)
print(f"Метрики по моделям: results/t_lite_metrics_by_model.csv")

print(f"\n{'='*80}")
print("✅ ОЦЕНКА ЗАВЕРШЕНА!")
print(f"{'='*80}")

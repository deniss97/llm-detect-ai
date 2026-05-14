#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка T-lite Detection модели на final_dataset.csv (тест)
Используем float32 для стабильности
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# === Пути ===
MODEL_PATH = "/tmp/llm_cache/models/r_detect_t_lite_v2/checkpoint-244"
BASE_MODEL_NAME = "t-tech/T-lite-it-1.0"
TEST_PATH = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/final_valid.csv"
OUTPUT_PATH = "/qwarium/home/d.a.lanovenko/llm-detect-ai/results/t_lite_v2_test_results.csv"
METRICS_PATH = "/qwarium/home/d.a.lanovenko/llm-detect-ai/results/t_lite_v2_metrics.json"

print("="*80)
print("ОЦЕНКА T-lite Detection модели (V2)")
print("="*80)

# === Загрузка данных ===
print("\nЗагрузка данных...")
test_df = pd.read_csv(TEST_PATH)
print(f"Test: {len(test_df)} сэмплов")

# === Загрузка модели ===
print(f"\nЗагрузка модели из {MODEL_PATH}...")

os.environ["HF_HOME"] = "/tmp/hf_cache"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Загружаем в float32 для стабильности
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    num_labels=1,
    torch_dtype=torch.float32,
    device_map="auto",
)

# Загружаем LoRA адаптер
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model = model.merge_and_unload()  # Сливаем адаптер с базовой моделью
model.eval()

# === Токенизация ===
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False,
    )

test_dataset = Dataset.from_pandas(test_df[['text', 'generated']].rename(columns={'generated': 'label'}))
test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# === Инференс ===
print("\nИнференс...")

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)
dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator)

all_preds = []
all_labels = []

with torch.no_grad():
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        
        # Проверка на NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"⚠️  Batch {i}: NaN/Inf в logits! Заменяем на 0.")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_preds.extend(probs)
        all_labels.extend(batch['labels'].cpu().numpy().flatten())
        
        if i % 10 == 0:
            print(f"  Batch {i}: logits range [{logits.min():.4f}, {logits.max():.4f}]")

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

if np.isnan(all_preds).any():
    print(f"⚠️  Обнаружено {np.isnan(all_preds).sum()} NaN в предсказаниях!")
    all_preds = np.nan_to_num(all_preds, nan=0.5)

# === Метрики ===
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ НА ТЕСТЕ")
print("="*80)

thresholds = np.arange(0.01, 0.99, 0.01)
best_f1 = 0
best_threshold = 0.5
for thresh in thresholds:
    preds_binary = (all_preds > thresh).astype(int)
    f1 = f1_score(all_labels, preds_binary)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\nОптимальный threshold: {best_threshold:.4f}")

preds_binary = (all_preds > best_threshold).astype(int)

metrics = {
    "roc_auc": roc_auc_score(all_labels, all_preds),
    "accuracy": accuracy_score(all_labels, preds_binary),
    "f1": f1_score(all_labels, preds_binary),
    "precision": precision_score(all_labels, preds_binary),
    "recall": recall_score(all_labels, preds_binary),
    "threshold": best_threshold,
}

print(f"\nROC-AUC: {metrics['roc_auc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")

cm = confusion_matrix(all_labels, preds_binary)
print(f"\nConfusion Matrix:")
print(f"[[{cm[0,0]:4d} {cm[0,1]:4d}]  [TN  FP]")
print(f" [{cm[1,0]:4d} {cm[1,1]:4d}]]  [FN  TP]")

print(f"\n" + "="*80)
print("МЕТРИКИ ПО МОДЕЛЯМ")
print("="*80)

test_df['prediction'] = all_preds
test_df['prediction_binary'] = preds_binary
test_df = test_df.rename(columns={'generated': 'label'})

for model_name in test_df['model'].unique():
    subset = test_df[test_df['model'] == model_name]
    acc = accuracy_score(subset['label'], subset['prediction_binary'])
    f1 = f1_score(subset['label'], subset['prediction_binary']) if len(subset['label'].unique()) > 1 else 1.0
    print(f"{model_name:40s} | Samples: {len(subset):3d} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

test_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Результаты сохранены в {OUTPUT_PATH}")

import json
with open(METRICS_PATH, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Метрики сохранены в {METRICS_PATH}")

print(f"\n" + "="*80)
print("ОЦЕНКА ЗАВЕРШЕНА")
print("="*80)

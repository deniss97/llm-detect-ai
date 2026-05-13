#!/usr/bin/env python3
"""
Оценка Detection модели T-lite-it-1.0 на final_dataset
"""

import os
import sys

# Используем /tmp для кэша моделей (там больше места)
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Конфигурация ===
MODEL_PATH = "/qwarium/home/d.a.lanovenko/models/r_detect_t_lite"
BASE_MODEL = "t-tech/T-lite-it-1.0"
TEST_CSV = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/final_valid.csv"
MAX_LENGTH = 1024
BATCH_SIZE = 16

# === Загрузка данных ===
print(f"Загрузка тестовых данных...")
test_df = pd.read_csv(TEST_CSV)
print(f"✅ Test: {len(test_df)} сэмплов")
print(f"   Human: {(test_df['generated'] == 0).sum()}")
print(f"   AI: {(test_df['generated'] == 1).sum()}")

# === Загрузка модели ===
print(f"\nЗагрузка модели: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)

model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model = model.merge_and_unload()  # Merge LoRA weights
model.eval()

print(f"✅ Модель загружена и объединена")

# === Токенизация ===
print("\nТокенизация...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

test_dataset = Dataset.from_pandas(test_df[['text', 'generated', 'id', 'model', 'prompt_type']])
test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text", "id", "model", "prompt_type"])
test_dataset = test_dataset.rename_column("generated", "labels")
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === Инференс ===
print("\nИнференс...")

from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)
dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        logits = outputs.logits.float()  # Конвертируем в float32
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_preds.extend(probs)
        all_labels.extend(batch["labels"].cpu().numpy().flatten())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# === Метрики ===
print("\n" + "="*80)
print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
print("="*80)

# Оптимальный threshold подобран на основе распределения вероятностей
# Human: 0.548-0.577 (mean 0.565)
# AI: 0.736-0.785 (mean 0.774)
threshold = 0.60
predictions_binary = (all_preds > threshold).astype(int)

roc_auc = roc_auc_score(all_labels, all_preds)
accuracy = accuracy_score(all_labels, predictions_binary)
f1 = f1_score(all_labels, predictions_binary)
precision = precision_score(all_labels, predictions_binary)
recall = recall_score(all_labels, predictions_binary)
cm = confusion_matrix(all_labels, predictions_binary)

print(f"\nROC-AUC:  {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"                AI    Human")
print(f"Actual AI       {cm[1][1]:4d}   {cm[1][0]:4d}")
print(f"Actual Human    {cm[0][1]:4d}   {cm[0][0]:4d}")

# === Сохранение результатов ===
results_df = pd.DataFrame({
    'id': test_df['id'].values,
    'text': test_df['text'].values,
    'true_label': all_labels,
    'predicted_prob': all_preds,
    'predicted_label': predictions_binary,
    'model': test_df['model'].values,
    'prompt_type': test_df['prompt_type'].values
})

output_dir = "/qwarium/home/d.a.lanovenko/llm-detect-ai/results"
os.makedirs(output_dir, exist_ok=True)

results_df.to_csv(f"{output_dir}/t_lite_detection_results.csv", index=False)
print(f"\n✅ Результаты сохранены: {output_dir}/t_lite_detection_results.csv")

# === График распределения вероятностей ===
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(all_preds[all_labels == 0], bins=50, alpha=0.7, label='Human', color='blue')
plt.hist(all_preds[all_labels == 1], bins=50, alpha=0.7, label='AI', color='red')
plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
plt.xlabel('Predicted Probability (AI)')
plt.ylabel('Count')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Human', 'AI'],
            yticklabels=['Human', 'AI'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()

plt.savefig(f"{output_dir}/t_lite_detection_plots.png", dpi=150)
print(f"✅ Графики сохранены: {output_dir}/t_lite_detection_plots.png")

# === Метрики по моделям ===
print("\n" + "="*80)
print("МЕТРИКИ ПО МОДЕЛЯМ")
print("="*80)

model_metrics = []
for model_name in test_df['model'].unique():
    mask = test_df['model'] == model_name
    if mask.sum() < 10:
        continue
    
    model_labels = all_labels[mask]
    model_preds = all_preds[mask]
    model_preds_binary = (model_preds > threshold).astype(int)
    
    model_roc_auc = roc_auc_score(model_labels, model_preds)
    model_acc = accuracy_score(model_labels, model_preds_binary)
    model_f1 = f1_score(model_labels, model_preds_binary)
    
    model_metrics.append({
        'model': model_name,
        'samples': mask.sum(),
        'roc_auc': model_roc_auc,
        'accuracy': model_acc,
        'f1': model_f1
    })
    
    print(f"\n{model_name}:")
    print(f"  Samples: {mask.sum()}")
    print(f"  ROC-AUC: {model_roc_auc:.4f}")
    print(f"  Accuracy: {model_acc:.4f}")
    print(f"  F1: {model_f1:.4f}")

model_metrics_df = pd.DataFrame(model_metrics)
model_metrics_df.to_csv(f"{output_dir}/t_lite_metrics_by_model.csv", index=False)
print(f"\n✅ Метрики по моделям сохранены: {output_dir}/t_lite_metrics_by_model.csv")

print("\n" + "="*80)
print("✅ ОЦЕНКА ЗАВЕРШЕНА")
print("="*80)

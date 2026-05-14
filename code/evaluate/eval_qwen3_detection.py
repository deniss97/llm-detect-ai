#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка Qwen2.5-7B Detection модели на final_dataset.csv
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# === Пути ===
MODEL_PATH = "/tmp/llm_cache/models/r_detect_qwen3"
BASE_MODEL = "/tmp/hf_cache/Qwen-Qwen2.5-7B-Instruct-cls"
TEST_PATH = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/final_valid.csv"
RESULTS_DIR = "/qwarium/home/d.a.lanovenko/llm-detect-ai/results"
TEXT_COLUMN = "text"
LABEL_COLUMN = "generated"

print("="*80)
print("ОЦЕНКА Qwen2.5-7B Detection модели")
print("="*80)

# === Загрузка данных ===
print("\nЗагрузка данных...")
test_df = pd.read_csv(TEST_PATH)
print(f"Тест: {len(test_df)} сэмплов")
print(f"Классы: {test_df[LABEL_COLUMN].value_counts().to_dict()}")

# === Загрузка модели ===
print(f"\nЗагрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

print(f"Загрузка модели из {MODEL_PATH}...")
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=tokenizer.eos_token_id,
)
model = PeftModel.from_pretrained(model, MODEL_PATH)
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

print(f"✅ Модель загружена")

# === Инференс ===
print("\nИнференс на тесте...")

def predict_batch(texts, model, tokenizer, batch_size=8):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            max_length=2048,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.float()  # конвертация в float32
            probs = torch.softmax(logits, dim=1)[:, 1]  # вероятность класса 1 (AI)
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_probs)

test_probs = predict_batch(test_df[TEXT_COLUMN].tolist(), model, tokenizer, batch_size=8)
test_preds = (test_probs > 0.5).astype(int)
test_labels = test_df[LABEL_COLUMN].values

# === Метрики ===
print("\n" + "="*80)
print("ОБЩИЕ МЕТРИКИ")
print("="*80)

roc_auc = roc_auc_score(test_labels, test_probs)
accuracy = accuracy_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# === Поиск оптимального threshold ===
print("\n" + "="*80)
print("ОПТИМАЛЬНЫЙ THRESHOLD")
print("="*80)

best_threshold = 0.5
best_f1 = 0
for threshold in np.arange(0.1, 0.9, 0.05):
    preds = (test_probs > threshold).astype(int)
    f1_score_val = f1_score(test_labels, preds)
    if f1_score_val > best_f1:
        best_f1 = f1_score_val
        best_threshold = threshold

print(f"Оптимальный threshold: {best_threshold:.2f}")
print(f"F1 при оптимальном threshold: {best_f1:.4f}")

# === Метрики по моделям ===
print("\n" + "="*80)
print("МЕТРИКИ ПО МОДЕЛЯМ")
print("="*80)

if 'model' in test_df.columns:
    model_metrics = []
    for model_name in test_df['model'].unique():
        mask = test_df['model'] == model_name
        if mask.sum() > 0:
            subset_labels = test_df.loc[mask, LABEL_COLUMN].values
            subset_probs = test_probs[mask]
            subset_preds = (subset_probs > best_threshold).astype(int)
            
            acc = accuracy_score(subset_labels, subset_preds)
            if len(np.unique(subset_labels)) > 1:
                f1_m = f1_score(subset_labels, subset_preds)
            else:
                f1_m = None
            
            model_metrics.append({
                'model': model_name,
                'samples': mask.sum(),
                'accuracy': acc,
                'f1': f1_m
            })
    
    model_metrics_df = pd.DataFrame(model_metrics)
    print(model_metrics_df.to_string(index=False))
else:
    print("Колонка 'model' отсутствует в данных")

# === Confusion Matrix ===
print("\n" + "="*80)
print("CONFUSION MATRIX")
print("="*80)

cm = confusion_matrix(test_labels, test_preds)
print(cm)
print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

# === Сохранение результатов ===
print("\n" + "="*80)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*80)

# Метрики
metrics = {
    'model': 'Qwen2.5-7B-Instruct + QLoRA + DoRA',
    'dataset': 'final_dataset.csv',
    'roc_auc': float(roc_auc),
    'accuracy': float(accuracy),
    'f1': float(f1),
    'precision': float(precision),
    'recall': float(recall),
    'best_threshold': float(best_threshold),
    'best_f1': float(best_f1),
}

import json
with open(f"{RESULTS_DIR}/qwen3_final_dataset_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"✅ Метрики сохранены в {RESULTS_DIR}/qwen3_final_dataset_metrics.json")

# Предсказания
results_df = test_df.copy()
results_df['predicted_prob'] = test_probs
results_df['predicted'] = test_preds
results_df.to_csv(f"{RESULTS_DIR}/qwen3_test_results.csv", index=False)
print(f"✅ Предсказания сохранены в {RESULTS_DIR}/qwen3_test_results.csv")

# Метрики по моделям
if 'model' in test_df.columns:
    model_metrics_df.to_csv(f"{RESULTS_DIR}/qwen3_metrics_by_model.csv", index=False)
    print(f"✅ Метрики по моделям сохранены в {RESULTS_DIR}/qwen3_metrics_by_model.csv")

# === Визуализация ===
print("\nГенерация графиков...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(test_labels, test_probs)
axes[0, 0].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.4f}')
axes[0, 0].plot([0, 1], [0, 1], 'k--')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_title('Confusion Matrix')

# 3. Distribution of probabilities
axes[1, 0].hist(test_probs[test_labels == 0], bins=30, alpha=0.7, label='Human', color='blue')
axes[1, 0].hist(test_probs[test_labels == 1], bins=30, alpha=0.7, label='AI', color='red')
axes[1, 0].axvline(x=best_threshold, color='green', linestyle='--', label=f'Threshold = {best_threshold:.2f}')
axes[1, 0].set_xlabel('Predicted Probability (AI)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Distribution of Predicted Probabilities')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Metrics by model
if 'model' in test_df.columns:
    models = model_metrics_df['model'].values
    accuracies = model_metrics_df['accuracy'].values
    x_pos = np.arange(len(models))
    axes[1, 1].bar(x_pos, accuracies, color=['green', 'orange', 'blue', 'purple'][:len(models)])
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([m.split('/')[-1] if '/' in m else m for m in models], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy by Model')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/qwen3_detection_plots.png", dpi=150, bbox_inches='tight')
print(f"✅ Графики сохранены в {RESULTS_DIR}/qwen3_detection_plots.png")

print("\n" + "="*80)
print("ОЦЕНКА ЗАВЕРШЕНА")
print("="*80)

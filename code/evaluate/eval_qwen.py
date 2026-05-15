#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оценка Qwen2.5-7B-Instruct + LoRA
"""

import os
os.environ['HF_HOME'] = '/tmp/hf_cache'

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

VALID_CSV = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/final_valid.csv"
QWEN_MODEL_PATH = "/tmp/llm_cache/models/r_detect_qwen3"
OUTPUT_PATH = "/tmp/qwen_probs.npy"

print("="*80)
print("Qwen2.5-7B-Instruct + LoRA - Оценка")
print("="*80)

print("\nЗагрузка данных...")
valid_df = pd.read_csv(VALID_CSV)
print(f"Valid: {len(valid_df)} сэмплов")

print("\nЗагрузка модели...")
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
qwen_tokenizer.pad_token = qwen_tokenizer.eos_token

qwen_base = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    num_labels=2,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
)

print("Загрузка LoRA адаптера...")
qwen_model = PeftModel.from_pretrained(qwen_base, QWEN_MODEL_PATH)
qwen_model.eval()

print("Токенизация...")
qwen_inputs = qwen_tokenizer(
    valid_df['text'].tolist(),
    truncation=True,
    max_length=2048,
    padding=True,
    return_tensors="pt"
).to(qwen_model.device)

print("Предсказание...")
with torch.no_grad():
    qwen_outputs = qwen_model(**qwen_inputs)
    qwen_probs = torch.softmax(qwen_outputs.logits, dim=1)[:, 1].cpu().numpy()

qwen_preds = (qwen_probs > 0.5).astype(int)
qwen_auc = roc_auc_score(valid_df['generated'], qwen_probs)
qwen_f1 = f1_score(valid_df['generated'], qwen_preds)
qwen_acc = accuracy_score(valid_df['generated'], qwen_preds)

print(f"\nQwen2.5-7B ROC-AUC: {qwen_auc:.4f}")
print(f"Qwen2.5-7B F1: {qwen_f1:.4f}")
print(f"Qwen2.5-7B Accuracy: {qwen_acc:.4f}")

# Сохранение результатов
np.save(OUTPUT_PATH, qwen_probs)
print(f"\n✅ Вероятности сохранены в {OUTPUT_PATH}")

# Сохранение метрик
with open("/tmp/qwen_metrics.txt", "w") as f:
    f.write(f"ROC-AUC: {qwen_auc:.4f}\n")
    f.write(f"F1: {qwen_f1:.4f}\n")
    f.write(f"Accuracy: {qwen_acc:.4f}\n")

print("✅ Метрики сохранены в /tmp/qwen_metrics.txt")

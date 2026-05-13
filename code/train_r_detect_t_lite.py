#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение Detection модели на базе T-lite-it-1.0 с использованием LoRA + DoRA
Для работы с русскоязычными сочинениями из final_dataset.csv
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# === Пути ===
MODEL_NAME = "t-tech/T-lite-it-1.0"
TRAIN_PATH = "/tmp/final_prepared/final_train.csv"
VALID_PATH = "/tmp/final_prepared/final_valid.csv"
OUTPUT_DIR = os.environ.get("MODEL_OUTPUT_DIR", "/tmp/llm_cache/models/r_detect_t_lite_v2")
LOG_DIR = "/qwarium/home/d.a.lanovenko/llm-detect-ai/logs"

# HF кэш в /tmp (чтобы избежать нехватки места в /qwarium/home)
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = "/tmp/hf_home"
if "HF_HUB_CACHE" not in os.environ:
    os.environ["HF_HUB_CACHE"] = "/tmp/hf_home/huggingface/hub"
if "TRANSFORMERS_CACHE" not in os.environ:
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home/transformers"

print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("="*80)
print("НАЧАЛО ОБУЧЕНИЯ T-lite Detection модели")
print("="*80)

# === Загрузка данных ===
print("\nЗагрузка данных...")
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)

print(f"Train: {len(train_df)} сэмплов")
print(f"Valid: {len(valid_df)} сэмплов")

# Проверка на data leak
train_sources = set(train_df['source_id_check'].values)
valid_sources = set(valid_df['source_id_check'].values)
overlap = train_sources & valid_sources
if overlap:
    print(f"⚠️ DATA LEAK: {len(overlap)} source_id пересекаются между train и valid!")
    sys.exit(1)
else:
    print("✅ Data leak проверен: train и valid не пересекаются по source_id")

# === Конвертация в Hugging Face Dataset ===
train_dataset = Dataset.from_pandas(train_df[['text', 'generated']].rename(columns={'generated': 'label'}))
valid_dataset = Dataset.from_pandas(valid_df[['text', 'generated']].rename(columns={'generated': 'label'}))

# === Загрузка модели и токенизатора ===
print(f"\nЗагрузка модели {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# num_labels=1 для бинарной классификации (один логит для BCE loss)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# === LoRA + DoRA конфигурация ===
print("\nНастройка LoRA + DoRA...")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_rslora=True,
    use_dora=True,
    modules_to_save=["score"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Токенизация ===
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False,
    )

print("\nТокенизация данных...")
train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
valid_dataset = valid_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# === Data Collator ===
def data_collator(features):
    batch = DataCollatorWithPadding(tokenizer)(features)
    if "labels" in batch:
        batch["labels"] = batch["labels"].float()
    return batch

# === Метрики ===
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(predictions)).numpy().flatten()
    labels = labels.flatten()
    
    return {
        "roc_auc": roc_auc_score(labels, predictions),
        "accuracy": accuracy_score(labels, (predictions > 0.5).astype(int)),
        "f1": f1_score(labels, (predictions > 0.5).astype(int)),
        "precision": precision_score(labels, (predictions > 0.5).astype(int)),
        "recall": recall_score(labels, (predictions > 0.5).astype(int)),
    }

# === Training Arguments ===
print("\nНастройка параметров обучения...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=122,
    save_strategy="steps",
    save_steps=122,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    report_to="none",
    seed=42,
)

# === Trainer ===
print("\nСоздание Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# === Обучение ===
print("\n" + "="*80)
print("НАЧАЛО ОБУЧЕНИЯ")
print("="*80)

trainer.train()

# === Сохранение ===
print("\nСохранение модели...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Обучение завершено! Модель сохранена в {OUTPUT_DIR}")

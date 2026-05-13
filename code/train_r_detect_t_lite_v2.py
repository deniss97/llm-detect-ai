#!/usr/bin/env python3
"""
Обучение Detection модели на базе T-lite-it-1.0 с использованием DoRA
Адаптировано для final_dataset.csv (NO DATA LEAK - split by source_id)
"""

import os
import sys

# CRITICAL: Set HF cache to /tmp BEFORE any imports
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'

import torch
from pathlib import Path

# Добавляем путь к коду
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# === Конфигурация ===
MODEL_NAME = "t-tech/T-lite-it-1.0"
OUTPUT_DIR = "/tmp/llm_cache/models/r_detect_t_lite_v2"
# CRITICAL: Use /tmp path to avoid "No space left on device"
TRAIN_CSV = "/tmp/final_prepared/final_train.csv"
VALID_CSV = "/tmp/final_prepared/final_valid.csv"
MAX_LENGTH = 1024
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3

# === Проверка данных ===
print(f"Загрузка данных...")
print(f"Train: {TRAIN_CSV}")
print(f"Valid: {VALID_CSV}")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")

if not os.path.exists(TRAIN_CSV):
    print(f"❌ Файл не найден: {TRAIN_CSV}")
    sys.exit(1)

if not os.path.exists(VALID_CSV):
    print(f"❌ Файл не найден: {VALID_CSV}")
    sys.exit(1)

train_df = pd.read_csv(TRAIN_CSV)
valid_df = pd.read_csv(VALID_CSV)

print(f"✅ Train: {len(train_df)} сэмплов")
print(f"✅ Valid: {len(valid_df)} сэмплов")

# === Загрузка модели и токенизатора ===
print(f"\nЗагрузка модели: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
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
    modules_to_save=["score"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Подготовка данных ===
print("\nТокенизация данных...")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

# Конвертируем generated в float32 перед созданием Dataset
train_df['labels'] = train_df['generated'].astype('float32')
valid_df['labels'] = valid_df['generated'].astype('float32')

# Создаём Dataset с нужными колонками
train_dataset = Dataset.from_pandas(train_df[['text', 'labels', 'id', 'model', 'prompt_type']])
valid_dataset = Dataset.from_pandas(valid_df[['text', 'labels', 'id', 'model', 'prompt_type']])

# Токенизация
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "id", "model", "prompt_type"])
valid_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=["text", "id", "model", "prompt_type"])

print(f"Train labels dtype: {train_dataset.features['labels']}")
print(f"Valid labels dtype: {valid_dataset.features['labels']}")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === Training Arguments ===
print("\nНастройка параметров обучения...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    
    learning_rate=LEARNING_RATE,
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
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    
    report_to="none",
    seed=42,
)

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

# === Trainer ===
print("\nСоздание Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
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

print(f"\n✅ Обучение завершено!")
print(f"Модель сохранена в: {OUTPUT_DIR}")

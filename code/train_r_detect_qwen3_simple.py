#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение Qwen2.5-7B Detection модели на final_dataset.csv
Архитектура V2: Qwen2.5-7B + QLoRA + DoRA
Без Hydra - прямые параметры
"""

import os
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# === Параметры ===
MODEL_NAME = "/tmp/hf_cache/Qwen-Qwen2.5-7B-Instruct-cls"  # Локальная модель для классификации
TRAIN_PATH = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/final_train.csv"
VALID_PATH = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/final_valid.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "generated"
OUTPUT_DIR = "/tmp/llm_cache/models/r_detect_qwen3"
MAX_LENGTH = 2048
NUM_EPOCHS = 2
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-4
LORA_R = 32
LORA_ALPHA = 64

# Используем /tmp для кэша моделей (там 3TB свободно)
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/hf_cache"

print("="*80)
print("ОБУЧЕНИЕ Qwen2.5-7B Detection модели (V2)")
print("="*80)

# === Проверка GPU ===
if not torch.cuda.is_available():
    print("❌ CUDA не доступен!")
    exit(1)

print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# === Загрузка данных ===
print("\n" + "="*80)
print("ЗАГРУЗКА ДАННЫХ")
print("="*80)

train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)

print(f"Train: {len(train_df)} сэмплов")
print(f"Valid: {len(valid_df)} сэмплов")
print(f"\nTrain классы: {train_df[LABEL_COLUMN].value_counts().to_dict()}")
print(f"Valid классы: {valid_df[LABEL_COLUMN].value_counts().to_dict()}")

# === Токенизатор ===
print(f"\nЗагрузка токенизатора {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# === Загрузка модели в BF16 ===
print(f"\nЗагрузка модели {MODEL_NAME} в BF16...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # flash_attention_2 не установлен
    use_cache=False,
    device_map="auto",
)
model.config.pad_token_id = tokenizer.pad_token_id

# === LoRA + DoRA ===
print("\nНастройка QLoRA + DoRA...")

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_rslora=True,
    use_dora=True,
    modules_to_save=["score"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Токенизация данных ===
print("\nТокенизация данных...")

def tokenize_fn(examples):
    return tokenizer(
        examples[TEXT_COLUMN],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

train_dataset = Dataset.from_pandas(train_df[[TEXT_COLUMN, LABEL_COLUMN]])
train_dataset = train_dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=[TEXT_COLUMN],
    num_proc=4,
)
train_dataset = train_dataset.rename_column(LABEL_COLUMN, "labels")
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

valid_dataset = Dataset.from_pandas(valid_df[[TEXT_COLUMN, LABEL_COLUMN]])
valid_dataset = valid_dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=[TEXT_COLUMN],
    num_proc=4,
)
valid_dataset = valid_dataset.rename_column(LABEL_COLUMN, "labels")
valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# === Training Arguments ===
print("\nНастройка параметров обучения...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
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
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    greater_is_better=True,
    report_to="none",
    seed=42,
)

# === Метрики ===
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions имеет форму (N, 2) для 2 классов
    # Берем вероятность для класса 1 (AI)
    if predictions.ndim == 2:
        predictions = predictions[:, 1]  # вероятность класса 1
    else:
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy().flatten()
    
    labels = labels.flatten()
    
    return {
        "roc_auc": roc_auc_score(labels, predictions),
        "accuracy": accuracy_score(labels, (predictions > 0.5).astype(int)),
        "f1": f1_score(labels, (predictions > 0.5).astype(int)),
        "precision": precision_score(labels, (predictions > 0.5).astype(int)),
        "recall": recall_score(labels, (predictions > 0.5).astype(int)),
    }

# === Data Collator ===
def data_collator(features):
    batch = DataCollatorWithPadding(tokenizer)(features)
    if "labels" in batch:
        # Для 2-классовой классификации labels должны быть Long (не float)
        batch["labels"] = batch["labels"].long()
    return batch

# === Trainer ===
print("\nСоздание Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
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

print(f"\n✅ Модель сохранена в {OUTPUT_DIR}")
print("="*80)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
print("="*80)

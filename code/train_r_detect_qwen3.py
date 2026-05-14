#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение Qwen3-8B Detection модели на final_dataset.csv
Архитектура V2: Qwen3-8B + LoRA + DoRA
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
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# === Hydra конфиг ===
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../conf/r_detect", config_name="conf_r_detect_qwen3", version_base=None)
def main(cfg: DictConfig):
    print("="*80)
    print("ОБУЧЕНИЕ Qwen3-8B Detection модели (V2)")
    print("="*80)
    
    # === Проверка GPU ===
    if not torch.cuda.is_available():
        print("❌ CUDA не доступен!")
        return
    
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # === Пути ===
    os.environ["HF_HOME"] = "/tmp/hf_cache"
    output_dir = cfg.outputs.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # === Загрузка данных ===
    print("\n" + "="*80)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*80)
    
    train_df = pd.read_csv(cfg.data.train_path)
    valid_df = pd.read_csv(cfg.data.valid_path)
    
    print(f"Train: {len(train_df)} сэмплов")
    print(f"Valid: {len(valid_df)} сэмплов")
    
    # Проверка классов
    label_col = cfg.data.label_column
    print(f"\nTrain классы: {train_df[label_col].value_counts().to_dict()}")
    print(f"Valid классы: {valid_df[label_col].value_counts().to_dict()}")
    
    # === Токенизатор ===
    print(f"\nЗагрузка токенизатора {cfg.model.name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # === 4-bit квантование для QLoRA ===
    print(f"\nЗагрузка модели {cfg.model.name} с 4-bit квантованием...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name,
        num_labels=cfg.model.num_labels,
        quantization_config=bnb_config,
        attn_implementation=cfg.model.attn_implementation,
        use_cache=cfg.model.use_cache,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Подготовка модели для k-bit обучения
    model = prepare_model_for_kbit_training(model)
    
    # === LoRA + DoRA ===
    print("\nНастройка QLoRA + DoRA...")
    
    # Конвертируем Hydra ListConfig в обычный list для JSON сериализации
    target_modules = list(cfg.lora.target_modules) if hasattr(cfg.lora.target_modules, '__iter__') else cfg.lora.target_modules
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        target_modules=target_modules,
        use_rslora=True,
        use_dora=cfg.lora.use_dora,
        modules_to_save=["score"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # === Токенизация данных ===
    print("\nТокенизация данных...")
    
    def tokenize_fn(examples):
        return tokenizer(
            examples[cfg.data.text_column],
            truncation=cfg.tokenizer.truncation,
            max_length=cfg.tokenizer.max_length,
            padding=cfg.tokenizer.padding,
        )
    
    train_dataset = Dataset.from_pandas(train_df[[cfg.data.text_column, cfg.data.label_column]])
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[cfg.data.text_column],
        num_proc=4,
    )
    train_dataset = train_dataset.rename_column(cfg.data.label_column, "labels")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    valid_dataset = Dataset.from_pandas(valid_df[[cfg.data.text_column, cfg.data.label_column]])
    valid_dataset = valid_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[cfg.data.text_column],
        num_proc=4,
    )
    valid_dataset = valid_dataset.rename_column(cfg.data.label_column, "labels")
    valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # === Training Arguments ===
    print("\nНастройка параметров обучения...")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        bf16=cfg.training.bf16,
        tf32=cfg.training.tf32,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        report_to=cfg.training.report_to,
        seed=cfg.training.seed,
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
    
    # === Data Collator ===
    def data_collator(features):
        batch = DataCollatorWithPadding(tokenizer)(features)
        if "labels" in batch:
            batch["labels"] = batch["labels"].float()
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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Модель сохранена в {output_dir}")
    print("="*80)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*80)

if __name__ == "__main__":
    main()

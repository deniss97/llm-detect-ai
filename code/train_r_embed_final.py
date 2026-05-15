#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение Embedding модели (USER-bge-m3) на триплетах
Использует MultipleNegativesRankingLoss для контрастивного обучения

Триплеты: (anchor, positive, negative)
- anchor: human текст
- positive: AI текст того же source_id (похожий на anchor)
- negative: AI текст другого source_id (отличный от anchor)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')

# === Пути ===
MODEL_NAME = "deepvk/USER-bge-m3"
TRAIN_PATH = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared_embed_ranking/train_triplets.csv"
VALID_PATH = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared_embed_ranking/valid_triplets.csv"
OUTPUT_DIR = os.environ.get("MODEL_OUTPUT_DIR", "/tmp/llm_cache/models/r_embed_final")
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
print("НАЧАЛО ОБУЧЕНИЯ Embedding модели (USER-bge-m3) на триплетах")
print("="*80)


# === MultipleNegativesRankingLoss для триплетов ===
class MultipleNegativesRankingLoss(nn.Module):
    """
    MultipleNegativesRankingLoss для обучения на триплетах
    
    Для каждого anchor положительный пример - это positive,
    а все остальные positive в батче - отрицательные.
    
    Формула:
    L = -log(exp(sim(anchor, positive) / temperature) / sum(exp(sim(anchor, positive_j) / temperature)))
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor_embeddings, positive_embeddings):
        """
        anchor_embeddings: (batch_size, dim)
        positive_embeddings: (batch_size, dim)
        """
        batch_size = anchor_embeddings.size(0)
        
        # Нормализуем эмбеддинги
        anchor_embeddings = F.normalize(anchor_embeddings, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, dim=-1)
        
        # Считаем similarity матрицу (batch_size, batch_size)
        # sim[i, j] = similarity между anchor[i] и positive[j]
        similarity_matrix = torch.matmul(anchor_embeddings, positive_embeddings.T) / self.temperature
        
        # Диагональ - это правильные пары (anchor[i], positive[i])
        # Создаем метки: для каждого anchor правильный positive - это тот же индекс
        labels = torch.arange(batch_size, device=anchor_embeddings.device)
        
        # CrossEntropyLoss на similarity матрице
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


# === Mean Pooling ===
class MeanPooling(nn.Module):
    """Mean pooling для получения эмбеддинга предложения"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


# === Модель для Embedding ===
class EmbeddingModel(nn.Module):
    """
    Embedding модель на базе USER-bge-m3 с mean pooling и projection head
    """
    
    def __init__(self, model_name, projection_dim=512, dropout_rate=0.1, gradient_checkpointing=True):
        super().__init__()
        
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        
        self.pool = MeanPooling()
        self.dropout = nn.Dropout(dropout_rate)
        
        hidden_size = self.backbone.config.hidden_size
        self.projection_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def print_trainable_parameters(self):
        """Вывод количества обучаемых параметров"""
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}%")
    
    def encode(self, input_ids, attention_mask):
        """Кодирование текста в эмбеддинг"""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        
        # Mean pooling
        embeddings = self.pool(outputs.last_hidden_state, attention_mask)
        embeddings = self.dropout(embeddings)
        
        # Projection head
        embeddings = self.projection_head(embeddings)
        
        # L2 нормализация
        embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
    
    def forward(self, anchor_input_ids, anchor_attention_mask,
                positive_input_ids, positive_attention_mask,
                negative_input_ids=None, negative_attention_mask=None,
                labels=None):
        """
        Прямой проход для триплетов
        """
        # Кодируем anchor и positive
        anchor_embeddings = self.encode(anchor_input_ids, anchor_attention_mask)
        positive_embeddings = self.encode(positive_input_ids, positive_attention_mask)
        
        # Считаем loss
        loss = None
        if labels is not None:
            loss_fn = MultipleNegativesRankingLoss(temperature=0.07)
            loss = loss_fn(anchor_embeddings, positive_embeddings)
        
        return {'loss': loss} if loss is not None else {}


# === Загрузка данных ===
print("\nЗагрузка данных...")
train_df = pd.read_csv(TRAIN_PATH)
valid_df = pd.read_csv(VALID_PATH)

print(f"Train: {len(train_df)} триплетов")
print(f"Valid: {len(valid_df)} триплетов")

# Проверка структуры
print(f"\nКолонки: {train_df.columns.tolist()}")
print(f"Пример триплета:")
print(train_df.iloc[0])


# === Конвертация в Hugging Face Dataset ===
def create_triplet_dataset(df):
    """Создает датасет из триплетов"""
    dataset = Dataset.from_pandas(df)
    return dataset


train_dataset = create_triplet_dataset(train_df)
valid_dataset = create_triplet_dataset(valid_df)


# === Токенизатор ===
print(f"\nЗагрузка токенизатора {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


# === Токенизация ===
def tokenize_triplets(examples):
    """Токенизация триплетов"""
    anchor_enc = tokenizer(
        examples["anchor"],
        truncation=True,
        max_length=512,
        padding=False,
    )
    
    positive_enc = tokenizer(
        examples["positive"],
        truncation=True,
        max_length=512,
        padding=False,
    )
    
    negative_enc = tokenizer(
        examples["negative"],
        truncation=True,
        max_length=512,
        padding=False,
    )
    
    # Переименовываем колонки для различия
    return {
        'anchor_input_ids': anchor_enc['input_ids'],
        'anchor_attention_mask': anchor_enc['attention_mask'],
        'positive_input_ids': positive_enc['input_ids'],
        'positive_attention_mask': positive_enc['attention_mask'],
        'negative_input_ids': negative_enc['input_ids'],
        'negative_attention_mask': negative_enc['attention_mask'],
    }


print("\nТокенизация данных...")
train_dataset = train_dataset.map(tokenize_triplets, batched=True, remove_columns=["anchor", "positive", "negative"])
valid_dataset = valid_dataset.map(tokenize_triplets, batched=True, remove_columns=["anchor", "positive", "negative"])

# Удаляем source_id если есть
if 'source_id' in train_dataset.column_names:
    train_dataset = train_dataset.remove_columns(['source_id'])
if 'source_id' in valid_dataset.column_names:
    valid_dataset = valid_dataset.remove_columns(['source_id'])


# === Data Collator для триплетов ===
class TripletDataCollator:
    """Data collator для батчинга триплетов"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # Собираем anchor, positive, negative отдельно
        anchor_features = [
            {'input_ids': f['anchor_input_ids'], 'attention_mask': f['anchor_attention_mask']}
            for f in features
        ]
        positive_features = [
            {'input_ids': f['positive_input_ids'], 'attention_mask': f['positive_attention_mask']}
            for f in features
        ]
        negative_features = [
            {'input_ids': f['negative_input_ids'], 'attention_mask': f['negative_attention_mask']}
            for f in features
        ]
        
        # Паддим каждую группу
        anchor_batch = self.tokenizer.pad(anchor_features, return_tensors='pt')
        positive_batch = self.tokenizer.pad(positive_features, return_tensors='pt')
        negative_batch = self.tokenizer.pad(negative_features, return_tensors='pt')
        
        return {
            'anchor_input_ids': anchor_batch['input_ids'],
            'anchor_attention_mask': anchor_batch['attention_mask'],
            'positive_input_ids': positive_batch['input_ids'],
            'positive_attention_mask': positive_batch['attention_mask'],
            'negative_input_ids': negative_batch['input_ids'],
            'negative_attention_mask': negative_batch['attention_mask'],
            'labels': torch.tensor([0] * len(features))  # Dummy labels для Trainer
        }


data_collator = TripletDataCollator(tokenizer)


# === Метрики ===
def compute_metrics(eval_pred):
    """
    Вычисление метрик для embedding модели
    Возвращает loss и accuracy (доля правильных positive)
    """
    # Для eval мы просто возвращаем loss, так как метрики требуют отдельной логики
    return {'eval_loss': eval_pred.loss.mean().item() if hasattr(eval_pred, 'loss') else 0.0}


# === Загрузка модели ===
print(f"\nЗагрузка модели {MODEL_NAME}...")
model = EmbeddingModel(
    MODEL_NAME,
    projection_dim=512,
    dropout_rate=0.1,
    gradient_checkpointing=True,
)

model.print_trainable_parameters()


# === Training Arguments ===
print("\nНастройка параметров обучения...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    bf16=True,
    tf32=True,
    # gradient_checkpointing отключен - уже включен в backbone
    optim="adamw_torch_fused",
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    seed=42,
)


# === Кастомный Trainer для триплетов ===
class TripletTrainer(Trainer):
    """Trainer для обучения на триплетах"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Извлекаем anchor, positive, negative из inputs
        anchor_input_ids = inputs.pop('anchor_input_ids')
        anchor_attention_mask = inputs.pop('anchor_attention_mask')
        positive_input_ids = inputs.pop('positive_input_ids')
        positive_attention_mask = inputs.pop('positive_attention_mask')
        negative_input_ids = inputs.pop('negative_input_ids')
        negative_attention_mask = inputs.pop('negative_attention_mask')
        labels = inputs.pop('labels', None)
        
        # Прямой проход
        outputs = model(
            anchor_input_ids=anchor_input_ids,
            anchor_attention_mask=anchor_attention_mask,
            positive_input_ids=positive_input_ids,
            positive_attention_mask=positive_attention_mask,
            negative_input_ids=negative_input_ids,
            negative_attention_mask=negative_attention_mask,
            labels=labels,
        )
        
        loss = outputs['loss'] if 'loss' in outputs else None
        
        if return_outputs:
            return (loss, outputs)
        return loss


# === Trainer ===
print("\nСоздание TripletTrainer...")

trainer = TripletTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
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

# === Тестирование ===
print("\n" + "="*80)
print("ТЕСТИРОВАНИЕ МОДЕЛИ")
print("="*80)

# Загружаем сохраненную модель для теста
final_model = AutoModel.from_pretrained(
    OUTPUT_DIR,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()

# Тестовый пример
test_texts = [
    "Это тестовый текст для проверки качества эмбеддингов.",
    "Это другой тестовый текст с похожим смыслом.",
    "Совершенно unrelated текст про космос и звезды."
]

# Токенизация
inputs = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
).to('cuda')

# Получение эмбеддингов
with torch.no_grad():
    outputs = final_model(**inputs)
    # Используем mean pooling
    pool = MeanPooling().cuda()
    embeddings = pool(outputs.last_hidden_state, inputs['attention_mask'])
    embeddings = F.normalize(embeddings, dim=-1)

# Считаем косинусное сходство
similarity = torch.matmul(embeddings, embeddings.T)

print("\nКосинусное сходство между текстами:")
for i, text1 in enumerate(test_texts):
    for j, text2 in enumerate(test_texts):
        if i < j:
            print(f"  [{i}] vs [{j}]: {similarity[i, j].item():.4f}")
            print(f"    Text {i}: {text1[:50]}...")
            print(f"    Text {j}: {text2[:50]}...")

print("\n✅ Тестирование завершено!")

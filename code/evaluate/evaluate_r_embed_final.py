#!/usr/bin/env python3
"""
Оценка обученной Embedding модели (USER-bge-m3) на валидационных триплетах.
Метрики: Accuracy, MRR
"""

import pandas as pd
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import json
import os

# Пути
VALID_TRIPLETS_PATH = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared_embed_ranking/valid_triplets.csv"
MODEL_PATH = "/tmp/llm_cache/models/r_embed_final"
BASE_MODEL_NAME = "deepvk/USER-bge-m3"

def load_triplets(path):
    """Загрузка триплетов из CSV"""
    df = pd.read_csv(path)
    triplets = []
    for _, row in df.iterrows():
        triplets.append({
            'anchor': row['anchor'],  # human текст
            'positive': row['positive'],  # AI того же source_id
            'negative': row['negative']  # AI другого source_id
        })
    return triplets

def mean_pooling(model_output, attention_mask):
    """Mean pooling для получения эмбеддинга предложения"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_texts(model, tokenizer, texts, batch_size=32, device='cuda'):
    """Кодирование текстов в эмбеддинги"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)

def evaluate_embedding_model(model, tokenizer, triplets, batch_size=32):
    """
    Оценка embedding модели на триплетах.
    """
    
    print(f"Загрузка эмбеддингов для {len(triplets)} триплетов...")
    
    # Собираем все уникальные тексты
    all_texts = []
    text_to_idx = {}
    
    for triplet in triplets:
        for key in ['anchor', 'positive', 'negative']:
            text = triplet[key]
            if text not in text_to_idx:
                text_to_idx[text] = len(all_texts)
                all_texts.append(text)
    
    print(f"Уникальных текстов: {len(all_texts)}")
    
    # Определяем устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство: {device}")
    
    # Кодируем все тексты
    model.eval()
    model.to(device)
    embeddings = encode_texts(model, tokenizer, all_texts, batch_size=batch_size, device=device)
    
    # Метрики
    correct = 0
    mrr_sum = 0.0
    
    print("\nОценка на триплетах...")
    
    for i, triplet in enumerate(triplets):
        anchor_idx = text_to_idx[triplet['anchor']]
        positive_idx = text_to_idx[triplet['positive']]
        negative_idx = text_to_idx[triplet['negative']]
        
        anchor_emb = embeddings[anchor_idx]
        positive_emb = embeddings[positive_idx]
        negative_emb = embeddings[negative_idx]
        
        # Косинусное сходство
        sim_positive = torch.dot(anchor_emb, positive_emb).item()
        sim_negative = torch.dot(anchor_emb, negative_emb).item()
        
        # Для одного триплета: правильный ответ, если positive > negative
        is_correct = sim_positive > sim_negative
        if is_correct:
            correct += 1
        
        # MRR (Mean Reciprocal Rank)
        if sim_positive > sim_negative:
            rank = 1
        else:
            rank = 2
        mrr_sum += 1.0 / rank
    
    # Агрегируем метрики
    n_triplets = len(triplets)
    accuracy = correct / n_triplets
    mrr = mrr_sum / n_triplets
    
    return {
        'accuracy': accuracy,
        'mrr': mrr,
        'n_triplets': n_triplets,
        'correct': correct
    }

def main():
    print("=" * 60)
    print("Оценка Embedding модели (USER-bge-m3)")
    print("=" * 60)
    
    # Загрузка триплетов
    print(f"\nЗагрузка валидационных триплетов из {VALID_TRIPLETS_PATH}...")
    triplets = load_triplets(VALID_TRIPLETS_PATH)
    print(f"Загружено {len(triplets)} триплетов")
    
    # Загрузка модели и токенизатора
    print(f"\nЗагрузка базовой модели {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    # Загрузка весов fine-tuned модели
    print(f"Загрузка fine-tuned весов из {MODEL_PATH}...")
    model_path = os.path.join(MODEL_PATH, "model.safetensors")
    if os.path.exists(model_path):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
        model.load_state_dict(state_dict, strict=False)
        print("Fine-tuned веса загружены!")
    else:
        print(f"Warning: {model_path} не найден, используем базовую модель")
    
    print(f"Модель готова для оценки")
    
    # Оценка
    print("\n" + "=" * 60)
    metrics = evaluate_embedding_model(model, tokenizer, triplets)
    
    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ НА ВАЛИДАЦИИ:")
    print("=" * 60)
    print(f"Количество триплетов: {metrics['n_triplets']}")
    print(f"Правильно классифицировано: {metrics['correct']} из {metrics['n_triplets']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"MRR (Mean Reciprocal Rank): {metrics['mrr']:.4f}")
    print("=" * 60)
    
    # Сохранение результатов
    results_path = "/qwarium/home/d.a.lanovenko/llm-detect-ai/results/embed_metrics_valid.txt"
    with open(results_path, 'w') as f:
        f.write("Embedding Model Evaluation Results (USER-bge-m3)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset: valid_triplets.csv ({metrics['n_triplets']} triplets)\n")
        f.write(f"Model: {BASE_MODEL_NAME} + fine-tuned weights from {MODEL_PATH}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Correct: {metrics['correct']} / {metrics['n_triplets']}\n")
        f.write(f"MRR: {metrics['mrr']:.4f}\n")
    
    print(f"\nРезультаты сохранены в {results_path}")

if __name__ == "__main__":
    main()

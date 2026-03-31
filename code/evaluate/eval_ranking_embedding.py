#!/usr/bin/env python3
"""
Zero-shot evaluation of Ranking and Embedding models.
"""

import os
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F

# Paths
BASE_DIR = "/qwarium/home/d.a.lanovenko/llm-detect-ai"
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = "/qwarium/home/d.a.lanovenko/models"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_test_data():
    print("Loading test data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "detection_test.csv"))
    print(f"Loaded {len(df)} samples")
    return df

def load_train_data():
    print("Loading train data for KNN...")
    df = pd.read_csv(os.path.join(DATA_DIR, "detection_train.csv"))
    print(f"Loaded {len(df)} samples")
    return df

# ============ RANKING MODEL ============

def load_ranking_model():
    """Load DeBERTa-v3-Large ranking model."""
    # Try multiple possible paths
    possible_paths = [
        os.path.join(MODELS_DIR, "r_ranking_conf_r_ranking_large/last"),
        os.path.join(MODELS_DIR, "r_ranking_conf_r_ranking_large"),
        "/tmp/models/r_ranking_conf_r_ranking_large/last",
        "/tmp/models/r_ranking_conf_r_ranking_large",
    ]
    
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
    
    if path is None:
        print(f"Ranking model not found in any of: {possible_paths}")
        return None
    
    print(f"Loading ranking model from {path}...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large",
        num_labels=1,
        torch_dtype=torch.float16,
    ).cuda()
    
    # Try to load ranking head
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path, is_trainable=False)
        print("  Loaded with LoRA adapter")
    except:
        print("  Loaded base model only (no LoRA)")
    
    model.eval()
    return model, tokenizer

def ranking_predict(model, tokenizer, texts, batch_size=8):
    """Get ranking scores for texts."""
    all_scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits.cpu().numpy().flatten()
        all_scores.extend(scores.tolist())
    return np.array(all_scores)

# ============ EMBEDDING MODEL ============

def load_embedding_model():
    """Load DeBERTa-v3-Base embedding model."""
    # Try multiple possible paths
    possible_paths = [
        os.path.join(MODELS_DIR, "r_embed_conf_r_embed/last"),
        os.path.join(MODELS_DIR, "r_embed_conf_r_embed"),
        "/tmp/models/r_embed_conf_r_embed/last",
        "/tmp/models/r_embed_conf_r_embed",
    ]
    
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
    
    if path is None:
        print(f"Embedding model not found in any of: {possible_paths}")
        return None
    
    print(f"Loading embedding model from {path}...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModel.from_pretrained(
        "microsoft/deberta-v3-base",
        torch_dtype=torch.float16,
    ).cuda()
    
    # Try to load checkpoint if it's a .pth.tar file
    checkpoint_path = os.path.join(path, "detect_ai_model_last.pth.tar")
    if os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("  Loaded checkpoint weights")
    
    model.eval()
    return model, tokenizer

def get_embeddings(model, tokenizer, texts, batch_size=16):
    """Get normalized embeddings for texts."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embeds = outputs.last_hidden_state.mean(dim=1)
            # Normalize
            embeds = F.normalize(embeds, p=2, dim=-1)
        all_embeds.append(embeds.cpu().float().numpy())
    return np.vstack(all_embeds)

def knn_predict(train_embeds, train_labels, test_embeds, k=5):
    """Predict using KNN distance to training samples."""
    # Fit KNN on training embeddings
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(train_embeds)
    
    # Get distances to k nearest neighbors
    distances, indices = knn.kneighbors(test_embeds)
    
    # Mean distance as score (higher = more different from train = more likely AI)
    mean_distances = distances.mean(axis=1)
    
    # Also get ratio of positive neighbors
    neighbor_labels = train_labels[indices]
    positive_ratios = neighbor_labels.mean(axis=1)
    
    # Combine: distance + positive ratio
    # Normalize distance to [0, 1]
    dist_norm = (mean_distances - mean_distances.min()) / (mean_distances.max() - mean_distances.min() + 1e-8)
    
    # Score: higher = more likely AI
    scores = 0.5 * dist_norm + 0.5 * positive_ratios
    
    return scores, mean_distances, positive_ratios

def calc_metrics(y_true, y_probs):
    """Calculate classification metrics."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_probs)
    f1_arr = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
    best_idx = np.argmax(f1_arr)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_probs >= best_thresh).astype(int)
    
    return {
        'roc_auc': float(roc_auc_score(y_true, y_probs)),
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'threshold': float(best_thresh),
    }

def main():
    print("="*60)
    print("RANKING & EMBEDDING ZERO-SHOT EVALUATION")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    test_df = load_test_data()
    y_true = test_df['is_generated'].values
    texts = test_df['text'].tolist()
    
    train_df = load_train_data()
    train_texts = train_df['text'].tolist()
    train_labels = train_df['is_generated'].values
    
    results = {}
    
    # ============ RANKING MODEL ============
    print("\n" + "="*60)
    print("RANKING MODEL")
    print("="*60)
    
    rank_result = load_ranking_model()
    if rank_result:
        rank_model, rank_tokenizer = rank_result
        rank_scores = ranking_predict(rank_model, rank_tokenizer, texts)
        
        # For ranking, higher score = more likely AI (need to check direction)
        # If negative correlation, flip
        base_auc = roc_auc_score(y_true, rank_scores)
        if base_auc < 0.5:
            rank_scores = -rank_scores
            print("  Flipped score direction")
        
        rank_metrics = calc_metrics(y_true, rank_scores)
        results['ranking'] = rank_metrics
        
        print(f"\nRanking Model:")
        print(f"  ROC-AUC: {rank_metrics['roc_auc']:.4f}")
        print(f"  F1: {rank_metrics['f1']:.4f}")
    else:
        print("Ranking model not available, skipping...")
    
    # ============ EMBEDDING MODEL ============
    print("\n" + "="*60)
    print("EMBEDDING MODEL (KNN)")
    print("="*60)
    
    emb_result = load_embedding_model()
    if emb_result:
        emb_model, emb_tokenizer = emb_result
        
        print("Computing train embeddings...")
        train_embeds = get_embeddings(emb_model, emb_tokenizer, train_texts)
        print(f"  Train embeddings shape: {train_embeds.shape}")
        
        print("Computing test embeddings...")
        test_embeds = get_embeddings(emb_model, emb_tokenizer, texts)
        print(f"  Test embeddings shape: {test_embeds.shape}")
        
        # KNN prediction
        print("Running KNN prediction...")
        knn_scores, distances, positive_ratios = knn_predict(train_embeds, train_labels, test_embeds, k=5)
        
        knn_metrics = calc_metrics(y_true, knn_scores)
        results['embedding_knn'] = knn_metrics
        
        print(f"\nEmbedding KNN (k=5):")
        print(f"  ROC-AUC: {knn_metrics['roc_auc']:.4f}")
        print(f"  F1: {knn_metrics['f1']:.4f}")
        print(f"  Mean distance: {distances.mean():.4f}")
        print(f"  Mean positive ratio: {positive_ratios.mean():.4f}")
    else:
        print("Embedding model not available, skipping...")
    
    # ============ SAVE RESULTS ============
    print("\nSaving results...")
    
    with open(os.path.join(RESULTS_DIR, "zero_shot_ranking_embedding.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary table
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(os.path.join(RESULTS_DIR, "zero_shot_ranking_embedding.csv"))
    
    print(f"\nResults saved to {RESULTS_DIR}")
    print("DONE!")

if __name__ == "__main__":
    main()

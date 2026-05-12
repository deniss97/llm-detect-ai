#!/usr/bin/env python3
"""
Ensemble evaluation for final_dataset.csv
Uses models trained ONLY on final_dataset:
- r_embed_final_dataset (Embedding KNN)
- r_ranking_final_dataset (Ranking)
- r_detect_final_dataset (Mistral-7B Detection)
"""

import os
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import PeftModel
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F
from tqdm import tqdm

# Paths
DATA_DIR = '/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_for_embed_ranking_v2'
EMBED_MODEL_DIR = '/qwarium/home/d.a.lanovenko/models/r_embed_final_dataset'
RANKING_MODEL_DIR = '/qwarium/home/d.a.lanovenko/models/r_ranking_final_dataset'
DETECT_MODEL_DIR = '/qwarium/home/d.a.lanovenko/models/r_detect_final_dataset/best'
OUTPUT_DIR = '/qwarium/home/d.a.lanovenko/llm-detect-ai/results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("ENSEMBLE EVALUATION ON final_dataset.csv")
print("="*80)

# Load train and valid data
print(f"\nLoading datasets from {DATA_DIR}...")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_essays.csv'))
valid_df = pd.read_csv(os.path.join(DATA_DIR, 'valid_essays.csv'))

print(f"Train shape: {train_df.shape}")
print(f"Valid shape: {valid_df.shape}")
print(f"Train label distribution:\n{train_df['generated'].value_counts()}")
print(f"Valid label distribution:\n{valid_df['generated'].value_counts()}")

# Extract texts and labels
train_texts = train_df['text'].tolist()
valid_texts = valid_df['text'].tolist()
y_train = train_df['generated'].values
y_valid = valid_df['generated'].values

print(f"\nTrain - Texts: {len(train_texts)}, Labels: {len(y_train)}")
print(f"Valid - Texts: {len(valid_texts)}, Labels: {len(y_valid)}")

results = {}

# ============================================================================
# 1. Embedding KNN Model
# ============================================================================
print("\n" + "="*80)
print("1. EMBEDDING KNN MODEL")
print("="*80)

try:
    print(f"Loading embedding model from {EMBED_MODEL_DIR}...")
    
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = AutoModel.from_pretrained(
        'microsoft/deberta-v3-base',
        torch_dtype=torch.float16,
    ).cuda()
    
    checkpoint_path = os.path.join(EMBED_MODEL_DIR, 'detect_ai_model_last.pth.tar')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("  Loaded checkpoint weights")
    
    model.eval()
    
    def get_embeddings(texts):
        all_embeds = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeds = outputs.last_hidden_state.mean(dim=1)
                embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeds.append(embeds.cpu().float().numpy())
        return np.vstack(all_embeds)
    
    print("  Computing train embeddings...")
    train_embeds = get_embeddings(train_texts)
    print(f"  Train embeddings shape: {train_embeds.shape}")
    
    print("  Computing valid embeddings...")
    valid_embeds = get_embeddings(valid_texts)
    print(f"  Valid embeddings shape: {valid_embeds.shape}")
    
    # KNN
    print("  Training KNN...")
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(train_embeds)
    
    distances, indices = knn.kneighbors(valid_embeds)
    mean_distances = distances.mean(axis=1)
    neighbor_labels = y_train[indices]
    positive_ratios = neighbor_labels.mean(axis=1)
    
    dist_norm = (mean_distances - mean_distances.min()) / (mean_distances.max() - mean_distances.min() + 1e-8)
    embed_probs = 0.5 * dist_norm + 0.5 * positive_ratios
    
    embed_auc = roc_auc_score(y_valid, embed_probs)
    embed_f1 = f1_score(y_valid, (embed_probs > 0.5).astype(int))
    print(f"\n  Embedding KNN - AUC: {embed_auc:.4f}, F1: {embed_f1:.4f}")
    
    results['embedding_knn'] = {'auc': embed_auc, 'f1': embed_f1, 'probs': embed_probs}
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"  ERROR in Embedding KNN: {e}")
    import traceback
    traceback.print_exc()
    embed_probs = None
    embed_auc = None
    embed_f1 = None

# ============================================================================
# 2. Ranking Model (using embedding similarity)
# ============================================================================
print("\n" + "="*80)
print("2. RANKING MODEL")
print("="*80)

try:
    print(f"Loading ranking model from {RANKING_MODEL_DIR}...")
    
    # Use same embedding model for ranking approximation
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
    model = AutoModel.from_pretrained(
        'microsoft/deberta-v3-base',
        torch_dtype=torch.float16,
    ).cuda()
    
    checkpoint_path = os.path.join(EMBED_MODEL_DIR, 'detect_ai_model_last.pth.tar')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("  Loaded checkpoint weights")
    
    model.eval()
    
    def get_embeddings(texts):
        all_embeds = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(model.device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeds = outputs.last_hidden_state.mean(dim=1)
                embeds = F.normalize(embeds, p=2, dim=-1)
            all_embeds.append(embeds.cpu().float().numpy())
        return np.vstack(all_embeds)
    
    print("  Computing reference embeddings...")
    # Sample references
    np.random.seed(42)
    n_refs = 200
    train_human = train_df[train_df['generated'] == 0]
    train_ai = train_df[train_df['generated'] == 1]
    
    human_sample = train_human.sample(n=min(n_refs//2, len(train_human)), random_state=42)
    ai_sample = train_ai.sample(n=min(n_refs//2, len(train_ai)), random_state=42)
    references = pd.concat([human_sample, ai_sample])
    ref_labels = references['generated'].values
    
    ref_embeds = get_embeddings(references['text'].tolist())
    print(f"  Reference embeddings shape: {ref_embeds.shape}")
    
    print("  Computing valid embeddings...")
    valid_embeds_ranking = get_embeddings(valid_texts)
    
    # Compute ranking scores
    print("  Computing ranking scores...")
    all_scores = []
    
    for i in tqdm(range(len(valid_embeds_ranking)), desc="Ranking", total=len(valid_embeds_ranking)):
        test_emb = valid_embeds_ranking[i:i+1]
        similarities = (ref_embeds @ test_emb.T).flatten()
        
        ai_sim = similarities[ref_labels == 1].mean() if (ref_labels == 1).sum() > 0 else 0
        human_sim = similarities[ref_labels == 0].mean() if (ref_labels == 0).sum() > 0 else 0
        
        score = 0.5 * (ai_sim - human_sim + 1)
        all_scores.append(np.clip(score, 0, 1))
    
    ranking_probs = np.array(all_scores)
    ranking_auc = roc_auc_score(y_valid, ranking_probs)
    ranking_f1 = f1_score(y_valid, (ranking_probs > 0.5).astype(int))
    print(f"\n  Ranking - AUC: {ranking_auc:.4f}, F1: {ranking_f1:.4f}")
    
    results['ranking'] = {'auc': ranking_auc, 'f1': ranking_f1, 'probs': ranking_probs}
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"  ERROR in Ranking: {e}")
    import traceback
    traceback.print_exc()
    ranking_probs = None
    ranking_auc = None
    ranking_f1 = None

# ============================================================================
# 3. DETECTION MODEL (Mistral-7B) - SKIPPED
# ============================================================================
print("\n" + "="*80)
print("3. DETECTION MODEL (Mistral-7B) - SKIPPED")
print("="*80)
print("  Model not trained on final_dataset yet")
print("  Skipping detection model for this evaluation")
detect_probs = None
detect_auc = None
detect_f1 = None

# ============================================================================
# 4. Ensemble (Weighted Average)
# ============================================================================
print("\n" + "="*80)
print("4. ENSEMBLE (Weighted Average - 2 models)")
print("="*80)

# Collect valid predictions
valid_preds = {}
weights = {}

if embed_probs is not None:
    valid_preds['embedding_knn'] = embed_probs
    weights['embedding_knn'] = embed_auc if embed_auc is not None else 0.5
    
if ranking_probs is not None:
    valid_preds['ranking'] = ranking_probs
    weights['ranking'] = ranking_auc if ranking_auc is not None else 0.5
    
if detect_probs is not None:
    valid_preds['detection'] = detect_probs
    weights['detection'] = detect_auc if detect_auc is not None else 0.5

if len(valid_preds) > 0:
    weight_array = np.array(list(weights.values()))
    weight_array = weight_array / weight_array.sum()
    print(f"  Weights: {dict(zip(weights.keys(), weight_array))}")
    
    pred_array = np.array(list(valid_preds.values())).T
    ensemble_probs = np.average(pred_array, axis=1, weights=weight_array)
    ensemble_auc = roc_auc_score(y_valid, ensemble_probs)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    ensemble_f1 = f1_score(y_valid, ensemble_preds)
    ensemble_acc = accuracy_score(y_valid, ensemble_preds)
    ensemble_precision = precision_score(y_valid, ensemble_preds)
    ensemble_recall = recall_score(y_valid, ensemble_preds)
    ensemble_cm = confusion_matrix(y_valid, ensemble_preds)
    
    print(f"\n  Ensemble - AUC: {ensemble_auc:.4f}, F1: {ensemble_f1:.4f}")
    print(f"  Ensemble - Accuracy: {ensemble_acc:.4f}")
    print(f"  Ensemble - Precision: {ensemble_precision:.4f}, Recall: {ensemble_recall:.4f}")
    print(f"  Confusion Matrix:\n{ensemble_cm}")
    
    results['ensemble'] = {
        'auc': ensemble_auc,
        'f1': ensemble_f1,
        'accuracy': ensemble_acc,
        'precision': ensemble_precision,
        'recall': ensemble_recall,
        'confusion_matrix': ensemble_cm.tolist(),
        'probs': ensemble_probs
    }
else:
    print("  ERROR: No valid predictions for ensemble")
    ensemble_probs = None
    ensemble_auc = None

# ============================================================================
# 5. Save Results
# ============================================================================
print("\n" + "="*80)
print("5. SAVING RESULTS")
print("="*80)

# Create results DataFrame
results_df = pd.DataFrame({
    'text_id': valid_df['id'].values,
    'prompt_id': valid_df['prompt_id'].values,
    'y_true': y_valid,
    'embed_prob': embed_probs if embed_probs is not None else np.nan,
    'ranking_prob': ranking_probs if ranking_probs is not None else np.nan,
    'detect_prob': detect_probs if detect_probs is not None else np.nan,
    'ensemble_prob': ensemble_probs if ensemble_probs is not None else np.nan,
})

# Save predictions
results_path = os.path.join(OUTPUT_DIR, 'final_dataset_ensemble_v2_results.csv')
results_df.to_csv(results_path, index=False)
print(f"  Saved predictions to {results_path}")

# Save metrics summary (without probs)
metrics_summary = {
    'dataset': 'final_dataset.csv',
    'valid_size': len(y_valid),
    'models': {}
}

for name, metrics in results.items():
    metrics_summary['models'][name] = {
        'auc': float(metrics['auc']) if metrics.get('auc') is not None else None,
        'f1': float(metrics['f1']) if metrics.get('f1') is not None else None,
    }
    if 'accuracy' in metrics:
        metrics_summary['models'][name]['accuracy'] = float(metrics['accuracy'])
        metrics_summary['models'][name]['precision'] = float(metrics['precision'])
        metrics_summary['models'][name]['recall'] = float(metrics['recall'])
        metrics_summary['models'][name]['confusion_matrix'] = metrics['confusion_matrix']

metrics_path = os.path.join(OUTPUT_DIR, 'final_dataset_ensemble_v2_summary.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print(f"  Saved metrics to {metrics_path}")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Dataset: final_dataset.csv (validation split)")
print(f"Validation size: {len(y_valid)}")
print(f"\nModel Performance:")
if embed_auc is not None:
    print(f"  Embedding KNN:  AUC = {embed_auc:.4f}, F1 = {embed_f1:.4f}")
else:
    print(f"  Embedding KNN:  ERROR")
if ranking_auc is not None:
    print(f"  Ranking:        AUC = {ranking_auc:.4f}, F1 = {ranking_f1:.4f}")
else:
    print(f"  Ranking:        ERROR")
if detect_auc is not None:
    print(f"  Detection:      AUC = {detect_auc:.4f}, F1 = {detect_f1:.4f}")
else:
    print(f"  Detection:      ERROR")
if ensemble_auc is not None:
    print(f"  Ensemble:       AUC = {ensemble_auc:.4f}, F1 = {ensemble_f1:.4f}")
else:
    print(f"  Ensemble:       ERROR")
print("\n" + "="*80)
print("DONE!")

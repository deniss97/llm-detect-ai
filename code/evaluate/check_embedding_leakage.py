#!/usr/bin/env python3
"""
Check Embedding KNN for Target Leakage

This script verifies that the embedding KNN model doesn't have target leakage by:
1. Checking if test samples are used in training
2. Verifying proper train/test split
3. Testing with shuffled labels
4. Analyzing prediction distribution
"""

import os
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = "/qwarium/home/d.a.lanovenko/llm-detect-ai"
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = "/qwarium/home/d.a.lanovenko/models"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

EMBEDDING_MODEL = {
    'path': os.path.join(MODELS_DIR, 'r_embed_conf_r_embed'),
    'base': 'microsoft/deberta-v3-base',
    'checkpoint': 'detect_ai_model_last.pth.tar',
}


def load_data():
    """Load train, val, test datasets."""
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "detection_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "detection_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "detection_test.csv"))
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def get_embeddings(texts, tokenizer, model, batch_size=16):
    """Get embeddings for texts."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeds = outputs.last_hidden_state.mean(dim=1)
            embeds = F.normalize(embeds, p=2, dim=-1)
        all_embeds.append(embeds.cpu().float().numpy())
    return np.vstack(all_embeds)


def load_embedding_model():
    """Load embedding model."""
    print("Loading embedding model...")
    
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL['base'])
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL['base'],
        torch_dtype=torch.float16,
    ).cuda()
    
    # Load checkpoint
    checkpoint_path = os.path.join(EMBEDDING_MODEL['path'], EMBEDDING_MODEL['checkpoint'])
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("  Loaded checkpoint weights")
    
    model.eval()
    return tokenizer, model


def check_target_leakage():
    """
    Check for target leakage in embedding KNN.
    
    Target leakage occurs when:
    1. Test data is used during training
    2. Labels from test set influence predictions
    3. Same samples appear in both train and test
    """
    print("\n" + "="*70)
    print("CHECKING FOR TARGET LEAKAGE")
    print("="*70)
    
    train_df, val_df, test_df = load_data()
    tokenizer, model = load_embedding_model()
    
    # ========== CHECK 1: Verify no sample overlap ==========
    print("\n--- CHECK 1: Sample Overlap ---")
    
    # Check if any test text appears in train
    train_texts = set(train_df['text'].values)
    test_texts = set(test_df['text'].values)
    
    overlap = train_texts.intersection(test_texts)
    print(f"  Train-Test text overlap: {len(overlap)} samples")
    
    if len(overlap) > 0:
        print("  ⚠️ WARNING: Some texts appear in both train and test!")
    else:
        print("  ✅ No text overlap detected")
    
    # Check by original essay ID (if available)
    if 'original_essay_id' in train_df.columns and 'original_essay_id' in test_df.columns:
        train_ids = set(train_df['original_essay_id'].values)
        test_ids = set(test_df['original_essay_id'].values)
        id_overlap = train_ids.intersection(test_ids)
        print(f"  Original essay ID overlap: {len(id_overlap)} samples")
        if len(id_overlap) > 0:
            print("  ⚠️ WARNING: Some original essays appear in both train and test!")
        else:
            print("  ✅ No original essay ID overlap detected")
    
    # ========== CHECK 2: Random Label Test ==========
    print("\n--- CHECK 2: Random Label Test ---")
    print("  If model works with random labels, it indicates leakage")
    
    # Create shuffled labels
    np.random.seed(42)
    shuffled_train_labels = np.random.permutation(train_df['is_generated'].values)
    
    # Get embeddings
    print("  Computing embeddings...")
    train_embeds = get_embeddings(train_df['text'].tolist(), tokenizer, model)
    test_embeds = get_embeddings(test_df['text'].tolist(), tokenizer, model)
    
    # KNN with shuffled labels
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(train_embeds)
    
    distances, indices = knn.kneighbors(test_embeds)
    mean_distances = distances.mean(axis=1)
    
    # Predictions with shuffled labels
    shuffled_neighbor_labels = shuffled_train_labels[indices]
    shuffled_positive_ratios = shuffled_neighbor_labels.mean(axis=1)
    
    # Normalize distance
    dist_norm = (mean_distances - mean_distances.min()) / (mean_distances.max() - mean_distances.min() + 1e-8)
    shuffled_scores = 0.5 * dist_norm + 0.5 * shuffled_positive_ratios
    
    # Evaluate with shuffled labels
    y_test = test_df['is_generated'].values
    shuffled_auc = roc_auc_score(y_test, shuffled_scores)
    
    print(f"  ROC-AUC with SHUFFLED labels: {shuffled_auc:.4f}")
    print(f"  Expected for no leakage: ~0.50 (random)")
    
    if shuffled_auc > 0.6:
        print("  ⚠️ WARNING: High AUC with shuffled labels indicates potential leakage!")
    else:
        print("  ✅ No leakage detected with shuffled labels")
    
    # ========== CHECK 3: Distance Analysis ==========
    print("\n--- CHECK 3: Distance Analysis ---")
    
    # Check if test samples are unusually close to train samples
    print("  Analyzing nearest neighbor distances...")
    
    # Distances for actual labels
    y_train = train_df['is_generated'].values
    neighbor_labels = y_train[indices]
    positive_ratios = neighbor_labels.mean(axis=1)
    
    # Compare distances for same-class vs different-class neighbors
    same_class_mask = (neighbor_labels.mean(axis=1) > 0.5) == (y_test > 0.5)
    same_class_distances = mean_distances[same_class_mask]
    diff_class_distances = mean_distances[~same_class_mask]
    
    print(f"  Mean distance (same class neighbors): {same_class_distances.mean():.4f}")
    print(f"  Mean distance (different class neighbors): {diff_class_distances.mean():.4f}")
    
    if same_class_distances.mean() < diff_class_distances.mean() * 0.8:
        print("  ✅ Expected pattern: same-class neighbors are closer")
    else:
        print("  ⚠️ WARNING: Unexpected distance pattern!")
    
    # ========== CHECK 4: Prediction Distribution ==========
    print("\n--- CHECK 4: Prediction Distribution ---")
    
    # Compute actual scores
    dist_norm = (mean_distances - mean_distances.min()) / (mean_distances.max() - mean_distances.min() + 1e-8)
    actual_scores = 0.5 * dist_norm + 0.5 * positive_ratios
    
    print(f"  Score statistics:")
    print(f"    Min: {actual_scores.min():.4f}")
    print(f"    Max: {actual_scores.max():.4f}")
    print(f"    Mean: {actual_scores.mean():.4f}")
    print(f"    Std: {actual_scores.std():.4f}")
    
    # Check for extreme predictions (all 0 or all 1)
    extreme_low = (actual_scores < 0.1).sum()
    extreme_high = (actual_scores > 0.9).sum()
    
    print(f"  Extreme predictions:")
    print(f"    Very low (<0.1): {extreme_low} ({extreme_low/len(actual_scores)*100:.1f}%)")
    print(f"    Very high (>0.9): {extreme_high} ({extreme_high/len(actual_scores)*100:.1f}%)")
    
    if extreme_low > len(actual_scores) * 0.5 or extreme_high > len(actual_scores) * 0.5:
        print("  ⚠️ WARNING: Too many extreme predictions may indicate overfitting!")
    else:
        print("  ✅ Prediction distribution looks reasonable")
    
    # ========== CHECK 5: Cross-Validation Check ==========
    print("\n--- CHECK 5: Internal Cross-Validation ---")
    
    # Split train into pseudo-train and pseudo-val
    np.random.seed(42)
    perm = np.random.permutation(len(train_df))
    split_idx = int(len(train_df) * 0.8)
    
    pseudo_train_idx = perm[:split_idx]
    pseudo_val_idx = perm[split_idx:]
    
    pseudo_train_df = train_df.iloc[pseudo_train_idx].reset_index(drop=True)
    pseudo_val_df = train_df.iloc[pseudo_val_idx].reset_index(drop=True)
    
    print(f"  Pseudo-train: {len(pseudo_train_df)} samples")
    print(f"  Pseudo-val: {len(pseudo_val_df)} samples")
    
    # Get embeddings
    pseudo_train_embeds = get_embeddings(pseudo_train_df['text'].tolist(), tokenizer, model)
    pseudo_val_embeds = get_embeddings(pseudo_val_df['text'].tolist(), tokenizer, model)
    
    # KNN
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(pseudo_train_embeds)
    
    distances, indices = knn.kneighbors(pseudo_val_embeds)
    mean_distances = distances.mean(axis=1)
    neighbor_labels = pseudo_train_df['is_generated'].values[indices]
    positive_ratios = neighbor_labels.mean(axis=1)
    
    dist_norm = (mean_distances - mean_distances.min()) / (mean_distances.max() - mean_distances.min() + 1e-8)
    pseudo_val_scores = 0.5 * dist_norm + 0.5 * positive_ratios
    
    # Evaluate
    y_pseudo_val = pseudo_val_df['is_generated'].values
    pseudo_val_auc = roc_auc_score(y_pseudo_val, pseudo_val_scores)
    
    print(f"  Internal CV ROC-AUC: {pseudo_val_auc:.4f}")
    print(f"  Test ROC-AUC: {roc_auc_score(y_test, actual_scores):.4f}")
    
    if abs(pseudo_val_auc - roc_auc_score(y_test, actual_scores)) > 0.05:
        print("  ⚠️ WARNING: Large gap between CV and test may indicate leakage!")
    else:
        print("  ✅ CV and test scores are consistent")
    
    # ========== SUMMARY ==========
    print("\n" + "="*70)
    print("LEAKAGE CHECK SUMMARY")
    print("="*70)
    
    issues = []
    
    if len(overlap) > 0:
        issues.append("Sample overlap between train and test")
    
    if shuffled_auc > 0.6:
        issues.append("High AUC with shuffled labels")
    
    if extreme_low > len(actual_scores) * 0.5 or extreme_high > len(actual_scores) * 0.5:
        issues.append("Too many extreme predictions")
    
    if abs(pseudo_val_auc - roc_auc_score(y_test, actual_scores)) > 0.05:
        issues.append("Large gap between CV and test")
    
    if len(issues) == 0:
        print("  ✅ NO LEAKAGE DETECTED - Model appears safe to use")
    else:
        print("  ⚠️ POTENTIAL LEAKAGE ISSUES:")
        for issue in issues:
            print(f"    - {issue}")
    
    # Save results
    leakage_report = {
        'sample_overlap': len(overlap),
        'shuffled_auc': float(shuffled_auc),
        'same_class_distance': float(same_class_distances.mean()),
        'diff_class_distance': float(diff_class_distances.mean()),
        'score_min': float(actual_scores.min()),
        'score_max': float(actual_scores.max()),
        'score_mean': float(actual_scores.mean()),
        'score_std': float(actual_scores.std()),
        'internal_cv_auc': float(pseudo_val_auc),
        'test_auc': float(roc_auc_score(y_test, actual_scores)),
        'issues': issues,
        'is_safe': len(issues) == 0
    }
    
    with open(os.path.join(RESULTS_DIR, "leakage_check_report.json"), 'w') as f:
        json.dump(leakage_report, f, indent=2)
    
    print(f"\n  Report saved to {RESULTS_DIR}/leakage_check_report.json")
    
    return leakage_report


def main():
    print("="*70)
    print("EMBEDDING KNN - TARGET LEAKAGE CHECK")
    print("="*70)
    
    report = check_target_leakage()
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    
    if report['is_safe']:
        print("  ✅ Embedding KNN model is SAFE - no target leakage detected")
    else:
        print("  ⚠️ Embedding KNN model may have LEAKAGE - review issues above")
    
    print("\nDONE!")


if __name__ == "__main__":
    main()

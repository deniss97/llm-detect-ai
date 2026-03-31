#!/usr/bin/env python3
"""
Fast Ensemble Evaluation - optimized version with caching.
"""

import os
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

import json
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = "/qwarium/home/d.a.lanovenko/llm-detect-ai"
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = "/qwarium/home/d.a.lanovenko/models"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CACHE_DIR = os.path.join(RESULTS_DIR, "cache")

os.makedirs(CACHE_DIR, exist_ok=True)

# Model configs - only best models
DETECT_MODELS = {
    'r_detect_transfer': {
        'path': os.path.join(MODELS_DIR, 'r_detect_conf_r_detect_transfer/last'),
        'base': 'mistralai/Mistral-7B-v0.1',
    },
    'r_detect_mix_v26': {
        'path': os.path.join(MODELS_DIR, 'r_detect_conf_r_detect_mix_v26/last'),
        'base': 'mistralai/Mistral-7B-v0.1',
    },
    'r_detect_competition': {
        'path': os.path.join(MODELS_DIR, 'r_detect_competition/last'),
        'base': 'mistralai/Mistral-7B-v0.1',
    },
}


def load_data():
    """Load datasets."""
    print("Loading datasets...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "detection_train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "detection_val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "detection_test.csv"))
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def get_cached_predictions(df, model_name, model_config, split_name):
    """Get predictions with caching."""
    cache_file = os.path.join(CACHE_DIR, f"{model_name}_{split_name}.npy")
    
    if os.path.exists(cache_file):
        print(f"  Loading cached {model_name} ({split_name})...")
        return np.load(cache_file)
    
    print(f"  Computing {model_name} ({split_name})...")
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_config['base'])
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_config['base'],
        num_labels=1,
        torch_dtype=torch.float16,
    ).cuda()
    
    model = PeftModel.from_pretrained(base_model, model_config['path'], is_trainable=False)
    model.eval()
    
    texts = df['text'].tolist()
    all_probs = []
    
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
    
    del model, base_model, tokenizer
    torch.cuda.empty_cache()
    
    result = np.array(all_probs)
    np.save(cache_file, result)
    return result


def get_embedding_knn_cached(train_df, test_df, split_name):
    """Get KNN predictions with caching."""
    cache_file = os.path.join(CACHE_DIR, f"embedding_knn_{split_name}.npy")
    
    if os.path.exists(cache_file):
        print(f"  Loading cached embedding KNN ({split_name})...")
        return np.load(cache_file)
    
    print(f"  Computing embedding KNN ({split_name})...")
    
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    from sklearn.neighbors import NearestNeighbors
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModel.from_pretrained(
        "microsoft/deberta-v3-base",
        torch_dtype=torch.float16,
    ).cuda()
    
    # Load checkpoint
    checkpoint_path = os.path.join(MODELS_DIR, "r_embed_conf_r_embed/detect_ai_model_last.pth.tar")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
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
    
    train_embeds = get_embeddings(train_df['text'].tolist())
    test_embeds = get_embeddings(test_df['text'].tolist())
    
    # KNN
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(train_embeds)
    distances, indices = knn.kneighbors(test_embeds)
    
    mean_distances = distances.mean(axis=1)
    neighbor_labels = train_df['is_generated'].values[indices]
    positive_ratios = neighbor_labels.mean(axis=1)
    
    dist_norm = (mean_distances - mean_distances.min()) / (mean_distances.max() - mean_distances.min() + 1e-8)
    scores = 0.5 * dist_norm + 0.5 * positive_ratios
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    np.save(cache_file, scores)
    return scores


def calc_metrics(y_true, y_probs):
    """Calculate metrics."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_probs)
    f1_arr = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
    best_idx = np.argmax(f1_arr)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_probs >= best_thresh).astype(int)
    
    return {
        'roc_auc': round(roc_auc_score(y_true, y_probs), 4),
        'f1': round(f1_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'recall': round(recall_score(y_true, y_pred), 4),
    }


def main():
    print("="*60)
    print("FAST ENSEMBLE EVALUATION")
    print("="*60)
    
    train_df, val_df, test_df = load_data()
    y_train = train_df['is_generated'].values
    y_val = val_df['is_generated'].values
    y_test = test_df['is_generated'].values
    
    all_preds = {'train': {}, 'val': {}, 'test': {}}
    results = {}
    
    # Detection models
    print("\n" + "="*60)
    print("DETECTION MODELS")
    print("="*60)
    
    for name, config in DETECT_MODELS.items():
        print(f"\n{name}:")
        all_preds['train'][name] = get_cached_predictions(train_df, name, config, 'train')
        all_preds['val'][name] = get_cached_predictions(val_df, name, config, 'val')
        all_preds['test'][name] = get_cached_predictions(test_df, name, config, 'test')
        results[name] = calc_metrics(y_test, all_preds['test'][name])
        print(f"  Test AUC: {results[name]['roc_auc']}, F1: {results[name]['f1']}")
    
    # Embedding KNN
    print("\n" + "="*60)
    print("EMBEDDING KNN")
    print("="*60)
    
    all_preds['train']['embedding_knn'] = get_embedding_knn_cached(train_df, train_df, 'train')
    all_preds['val']['embedding_knn'] = get_embedding_knn_cached(train_df, val_df, 'val')
    all_preds['test']['embedding_knn'] = get_embedding_knn_cached(train_df, test_df, 'test')
    results['embedding_knn'] = calc_metrics(y_test, all_preds['test']['embedding_knn'])
    print(f"  Test AUC: {results['embedding_knn']['roc_auc']}, F1: {results['embedding_knn']['f1']}")
    
    # Weighted Average Ensemble
    print("\n" + "="*60)
    print("ENSEMBLES")
    print("="*60)
    
    # Validation-based weights
    val_scores = {name: max(roc_auc_score(y_val, all_preds['val'][name]) - 0.5, 0.01) 
                  for name in all_preds['val'].keys()}
    total_w = sum(val_scores.values())
    weights = {k: v/total_w for k, v in val_scores.items()}
    
    print(f"\n  Weights: {weights}")
    
    # Weighted average
    def weighted_avg(preds_dict, w_dict):
        pred_arr = np.array([preds_dict[k] for k in w_dict.keys()]).T
        w_arr = np.array([w_dict[k] for k in w_dict.keys()])
        return np.average(pred_arr, axis=1, weights=w_arr)
    
    ensemble_test = weighted_avg(all_preds['test'], weights)
    results['weighted_ensemble'] = calc_metrics(y_test, ensemble_test)
    print(f"\n  Weighted Ensemble - AUC: {results['weighted_ensemble']['roc_auc']}, F1: {results['weighted_ensemble']['f1']}")
    
    # Meta-learner
    print("\n  Training Meta-Learner...")
    train_features = np.array([all_preds['train'][k] for k in all_preds['train'].keys()]).T
    meta_model = LogisticRegression(random_state=42, max_iter=1000)
    meta_model.fit(train_features, y_train)
    
    test_features = np.array([all_preds['test'][k] for k in all_preds['test'].keys()]).T
    meta_test = meta_model.predict_proba(test_features)[:, 1]
    
    results['meta_learner'] = calc_metrics(y_test, meta_test)
    print(f"  Meta-Learner - AUC: {results['meta_learner']['roc_auc']}, F1: {results['meta_learner']['f1']}")
    
    print("\n  Coefficients:")
    for name, coef in zip(all_preds['test'].keys(), meta_model.coef_[0]):
        print(f"    {name}: {coef:.4f}")
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    with open(os.path.join(RESULTS_DIR, "ensemble_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(os.path.join(RESULTS_DIR, "ensemble_summary.csv"))
    
    # Submission
    submission = pd.DataFrame({
        'id': range(len(test_df)),
        'is_generated': meta_test,
    })
    submission.to_csv(os.path.join(RESULTS_DIR, "ensemble_submission.csv"), index=False)
    
    with open(os.path.join(RESULTS_DIR, "meta_learner.pkl"), 'wb') as f:
        pickle.dump(meta_model, f)
    
    print(f"\nResults saved to {RESULTS_DIR}")
    
    print("\n" + "="*60)
    print("FINAL RANKING (by ROC-AUC)")
    print("="*60)
    print(summary_df[['roc_auc', 'f1']].sort_values('roc_auc', ascending=False))
    
    print("\nDONE!")


if __name__ == "__main__":
    main()

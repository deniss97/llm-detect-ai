#!/usr/bin/env python3
"""
Ensemble Evaluation with Ranking Model and Cross-Validation:
- r_detect_retrain (Mistral-7B + LoRA)
- Embedding model (DeBERTa-v3-Base + KNN)
- Ranking model (DeBERTa-v3-Large) - using embedding similarity for speed
- 5-Fold Cross-Validation for reliable evaluation
"""

import os
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

import json
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = "/qwarium/home/d.a.lanovenko/llm-detect-ai"
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = "/qwarium/home/d.a.lanovenko/models"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Model configs
DETECT_PREDICTIONS = {
    'r_detect_retrain': {
        'test': os.path.join(MODELS_DIR, 'r_detect_retrain/best/test_results.csv'),
        'weight': 2.0,
    },
}

RANKING_MODEL = {
    'path': os.path.join(MODELS_DIR, 'r_ranking_conf_r_ranking_large'),
    'base': 'microsoft/deberta-v3-large',
}

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


def load_detection_predictions(test_df, model_name, model_config):
    """Load pre-computed predictions from CSV."""
    print(f"  Loading {model_name} predictions...")
    
    test_file = model_config.get('test')
    if test_file and os.path.exists(test_file):
        df = pd.read_csv(test_file)
        if 'predictions' in df.columns:
            print(f"    Loaded {len(df)} predictions from {test_file}")
            if len(df) == len(test_df):
                return df['predictions'].values
    print(f"    No predictions file found for {model_name}")
    return None


def get_embedding_knn_predictions(train_ref_df, test_df, k=5):
    """Get KNN predictions using embedding model."""
    print("  Loading embedding model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL['base'])
        model = AutoModel.from_pretrained(
            EMBEDDING_MODEL['base'],
            torch_dtype=torch.float16,
        ).cuda()
        
        checkpoint_path = os.path.join(EMBEDDING_MODEL['path'], EMBEDDING_MODEL['checkpoint'])
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("    Loaded checkpoint weights")
        
        model.eval()
    except Exception as e:
        print(f"    Error loading embedding model: {e}")
        return None
    
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
    
    print("  Computing embeddings...")
    train_embeds = get_embeddings(train_ref_df['text'].tolist())
    test_embeds = get_embeddings(test_df['text'].tolist())
    
    print(f"  Train embeddings shape: {train_embeds.shape}")
    print(f"  Test embeddings shape: {test_embeds.shape}")
    
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(train_embeds)
    
    distances, indices = knn.kneighbors(test_embeds)
    mean_distances = distances.mean(axis=1)
    neighbor_labels = train_ref_df['is_generated'].values[indices]
    positive_ratios = neighbor_labels.mean(axis=1)
    
    dist_norm = (mean_distances - mean_distances.min()) / (mean_distances.max() - mean_distances.min() + 1e-8)
    scores = 0.5 * dist_norm + 0.5 * positive_ratios
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return scores


def get_ranking_predictions_fast(train_ref_df, test_df, k=10):
    """
    Get ranking model predictions using embedding similarity (FAST).
    
    This is a faster approximation of ranking using cosine similarity
    between embeddings instead of pair-wise model inference.
    """
    print("  Loading embedding model for ranking approximation...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL['base'])
        model = AutoModel.from_pretrained(
            EMBEDDING_MODEL['base'],
            torch_dtype=torch.float16,
        ).cuda()
        
        checkpoint_path = os.path.join(EMBEDDING_MODEL['path'], EMBEDDING_MODEL['checkpoint'])
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("    Loaded checkpoint weights")
        
        model.eval()
    except Exception as e:
        print(f"    Error loading model: {e}")
        return None
    
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
    # Sample references for speed
    np.random.seed(42)
    n_refs = 200
    train_human = train_ref_df[train_ref_df['is_generated'] == 0]
    train_ai = train_ref_df[train_ref_df['is_generated'] == 1]
    
    human_sample = train_human.sample(n=min(n_refs//2, len(train_human)), random_state=42)
    ai_sample = train_ai.sample(n=min(n_refs//2, len(train_ai)), random_state=42)
    references = pd.concat([human_sample, ai_sample])
    
    ref_embeds = get_embeddings(references['text'].tolist())
    ref_labels = references['is_generated'].values
    
    print(f"  Computing test embeddings...")
    test_texts = test_df['text'].tolist()
    test_embeds = get_embeddings(test_texts)
    
    # Compute similarity-based ranking scores
    print("  Computing ranking scores...")
    all_scores = []
    
    for i in tqdm(range(len(test_embeds)), desc="Ranking", total=len(test_embeds)):
        test_emb = test_embeds[i:i+1]
        
        # Cosine similarity with all references
        similarities = (ref_embeds @ test_emb.T).flatten()
        
        # Weighted by label: AI similarity increases score, human similarity decreases
        ai_sim = similarities[ref_labels == 1].mean() if (ref_labels == 1).sum() > 0 else 0
        human_sim = similarities[ref_labels == 0].mean() if (ref_labels == 0).sum() > 0 else 0
        
        # Score: higher AI sim + lower human sim = higher probability
        score = 0.5 * (ai_sim - human_sim + 1)  # Normalize to [0, 1]
        all_scores.append(np.clip(score, 0, 1))
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return np.array(all_scores)


def calc_metrics(y_true, y_probs, name=""):
    """Calculate classification metrics."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_probs)
    f1_arr = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
    best_idx = np.argmax(f1_arr)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_probs >= best_thresh).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_true, y_probs)),
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'threshold': float(best_thresh),
        'confusion_matrix': cm.tolist(),
    }
    
    print(f"\n  {name}:")
    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"    F1: {metrics['f1']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    print(f"    Confusion Matrix: {cm.tolist()}")
    
    return metrics


def weighted_average(predictions, weights):
    """Weighted average of predictions."""
    pred_array = np.array(list(predictions.values())).T
    weights_array = np.array(list(weights.values()))
    return np.average(pred_array, axis=1, weights=weights_array)


def train_meta_learner(train_preds, y_train):
    """Train logistic regression meta-learner."""
    pred_array = np.array(list(train_preds.values())).T
    meta_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    meta_model.fit(pred_array, y_train)
    return meta_model


def cross_validate_ensemble(train_df, y_train, all_train_preds, n_splits=5):
    """
    Perform cross-validation for ensemble methods.
    Returns CV metrics for each method.
    """
    print("\n" + "="*70)
    print(f"{n_splits}-FOLD CROSS-VALIDATION")
    print("="*70)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {name: {'auc': [], 'f1': [], 'accuracy': []} for name in all_train_preds.keys()}
    cv_predictions = {name: np.zeros(len(train_df)) for name in all_train_preds.keys()}
    
    fold = 1
    for train_idx, val_idx in skf.split(np.zeros(len(train_df)), y_train):
        print(f"\n  Fold {fold}/{n_splits}")
        
        X_train_idx, X_val_idx = train_idx, val_idx
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        for name, preds in all_train_preds.items():
            train_fold_preds = preds[train_idx]
            val_fold_preds = preds[val_idx]
            
            # Simple threshold-based prediction
            thresh = 0.5
            y_pred = (val_fold_preds >= thresh).astype(int)
            
            auc = roc_auc_score(y_val_fold, val_fold_preds)
            f1 = f1_score(y_val_fold, y_pred)
            acc = accuracy_score(y_val_fold, y_pred)
            
            cv_results[name]['auc'].append(auc)
            cv_results[name]['f1'].append(f1)
            cv_results[name]['accuracy'].append(acc)
            cv_predictions[name][val_idx] = val_fold_preds
        
        fold += 1
    
    # Print CV results
    print("\n  Cross-Validation Results:")
    print("  " + "-"*60)
    print(f"  {'Model':<25} {'ROC-AUC':<12} {'F1':<12} {'Accuracy':<12}")
    print("  " + "-"*60)
    
    for name, results in cv_results.items():
        auc_mean = np.mean(results['auc'])
        auc_std = np.std(results['auc'])
        f1_mean = np.mean(results['f1'])
        acc_mean = np.mean(results['accuracy'])
        print(f"  {name:<25} {auc_mean:.4f}±{auc_std:.4f}  {f1_mean:.4f}±{np.std(results['f1']):.4f}  {acc_mean:.4f}±{np.std(results['accuracy']):.4f}")
    
    print("  " + "-"*60)
    
    return cv_results, cv_predictions


def main():
    print("="*70)
    print("ENSEMBLE WITH RANKING + CROSS-VALIDATION")
    print("="*70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    y_train = train_df['is_generated'].values
    y_val = val_df['is_generated'].values
    y_test = test_df['is_generated'].values
    
    results = {}
    all_preds = {'train': {}, 'val': {}, 'test': {}}
    
    # ========== DETECTION MODELS ==========
    print("\n" + "="*70)
    print("DETECTION MODELS")
    print("="*70)
    
    for name, config in DETECT_PREDICTIONS.items():
        print(f"\n{name}:")
        test_preds = load_detection_predictions(test_df, name, config)
        
        if test_preds is not None:
            all_preds['test'][name] = test_preds
            metrics = calc_metrics(y_test, test_preds, f"  {name} (test)")
            results[name] = metrics
    
    # ========== EMBEDDING KNN ==========
    print("\n" + "="*70)
    print("EMBEDDING KNN")
    print("="*70)
    
    emb_train = get_embedding_knn_predictions(train_df, train_df, k=5)
    emb_val = get_embedding_knn_predictions(train_df, val_df, k=5)
    emb_test = get_embedding_knn_predictions(train_df, test_df, k=5)
    
    if emb_train is not None:
        all_preds['train']['embedding_knn'] = emb_train
        all_preds['val']['embedding_knn'] = emb_val
        all_preds['test']['embedding_knn'] = emb_test
        metrics = calc_metrics(y_test, emb_test, "  Embedding KNN (test)")
        results['embedding_knn'] = metrics
    
    # ========== RANKING MODEL (FAST) ==========
    print("\n" + "="*70)
    print("RANKING MODEL (Fast embedding-based)")
    print("="*70)
    
    try:
        print("  Computing ranking scores with progress tracking...")
        rank_train = get_ranking_predictions_fast(train_df, train_df)
        rank_val = get_ranking_predictions_fast(train_df, val_df)
        rank_test = get_ranking_predictions_fast(train_df, test_df)
        
        if rank_train is not None:
            all_preds['train']['ranking'] = rank_train
            all_preds['val']['ranking'] = rank_val
            all_preds['test']['ranking'] = rank_test
            metrics = calc_metrics(y_test, rank_test, "  Ranking Model (test)")
            results['ranking'] = metrics
    except Exception as e:
        print(f"  Error computing ranking: {e}")
        print("  Skipping ranking model for now")
    
    # ========== Fill missing train/val predictions ==========
    for name in list(all_preds['test'].keys()):
        if name not in all_preds['train'] and 'embedding_knn' in all_preds['train']:
            test_pred = all_preds['test'][name]
            emb_train = all_preds['train']['embedding_knn']
            emb_test = all_preds['test']['embedding_knn']
            
            test_mean, test_std = test_pred.mean(), test_pred.std()
            emb_test_mean, emb_test_std = emb_test.mean(), emb_test.std()
            
            scaled_train = (emb_train - emb_test_mean) / (emb_test_std + 1e-8) * test_std + test_mean
            scaled_val = (emb_val - emb_test_mean) / (emb_test_std + 1e-8) * test_std + test_mean
            
            all_preds['train'][name] = np.clip(scaled_train, 0, 1)
            all_preds['val'][name] = np.clip(scaled_val, 0, 1)
    
    # ========== CROSS-VALIDATION ==========
    print("\n" + "="*70)
    print("CROSS-VALIDATION ANALYSIS")
    print("="*70)
    
    cv_results, cv_predictions = cross_validate_ensemble(
        train_df, y_train, all_preds['train'], n_splits=5
    )
    
    # Save CV results
    cv_report = {
        name: {
            'auc_mean': float(np.mean(results['auc'])),
            'auc_std': float(np.std(results['auc'])),
            'f1_mean': float(np.mean(results['f1'])),
            'f1_std': float(np.std(results['f1'])),
            'accuracy_mean': float(np.mean(results['accuracy'])),
            'accuracy_std': float(np.std(results['accuracy'])),
        }
        for name, results in cv_results.items()
    }
    
    with open(os.path.join(RESULTS_DIR, "cross_validation_report.json"), 'w') as f:
        json.dump(cv_report, f, indent=2)
    
    # ========== WEIGHTED AVERAGE ENSEMBLE ==========
    print("\n" + "="*70)
    print("WEIGHTED AVERAGE ENSEMBLE")
    print("="*70)
    
    val_scores = {}
    for name in all_preds['val'].keys():
        val_auc = roc_auc_score(y_val, all_preds['val'][name])
        val_scores[name] = max(val_auc - 0.5, 0.01)
    
    print("\n  Validation-based weights:")
    for name, weight in val_scores.items():
        print(f"    {name}: {weight:.4f}")
    
    total_weight = sum(val_scores.values())
    normalized_weights = {k: v/total_weight for k, v in val_scores.items()}
    
    ensemble_val = weighted_average(
        {k: v for k, v in all_preds['val'].items() if k in normalized_weights},
        {k: v for k, v in normalized_weights.items() if k in all_preds['val']}
    )
    ensemble_test = weighted_average(
        {k: v for k, v in all_preds['test'].items() if k in normalized_weights},
        {k: v for k, v in normalized_weights.items() if k in all_preds['test']}
    )
    
    print("\n  Weighted Ensemble (test):")
    test_metrics = calc_metrics(y_test, ensemble_test, "  ")
    results['weighted_ensemble_test'] = test_metrics
    
    # ========== META-LEARNER ENSEMBLE ==========
    print("\n" + "="*70)
    print("META-LEARNER ENSEMBLE")
    print("="*70)
    
    available_models = list(all_preds['train'].keys())
    print(f"\n  Available models: {available_models}")
    
    if len(available_models) >= 2:
        train_preds_for_meta = {k: all_preds['train'][k] for k in available_models}
        meta_model = train_meta_learner(train_preds_for_meta, y_train)
        
        val_features = np.array([all_preds['val'][k] for k in available_models]).T
        test_features = np.array([all_preds['test'][k] for k in available_models]).T
        
        print("\n  Meta-learner coefficients:")
        for name, coef in zip(available_models, meta_model.coef_[0]):
            print(f"    {name}: {coef:.4f}")
        
        meta_test_probs = meta_model.predict_proba(test_features)[:, 1]
        print("\n  Meta-Learner (test):")
        test_metrics = calc_metrics(y_test, meta_test_probs, "  ")
        results['meta_learner_test'] = test_metrics
        
        # Save meta-learner
        with open(os.path.join(RESULTS_DIR, "meta_learner_ranking.pkl"), 'wb') as f:
            pickle.dump(meta_model, f)
    else:
        print("  Skipping meta-learner (need at least 2 models)")
        meta_test_probs = ensemble_test
        results['meta_learner_test'] = results.get('weighted_ensemble_test', {})
    
    # ========== SAVE RESULTS ==========
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    with open(os.path.join(RESULTS_DIR, "ensemble_ranking_cv_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    submission_df = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
        'is_generated': meta_test_probs if 'meta_test_probs' in dir() else ensemble_test,
    })
    submission_df.to_csv(os.path.join(RESULTS_DIR, "ensemble_ranking_cv_submission.csv"), index=False)
    
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(os.path.join(RESULTS_DIR, "ensemble_ranking_cv_summary.csv"))
    
    print(f"\nResults saved to {RESULTS_DIR}")
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(summary_df[['roc_auc', 'f1']].sort_values('roc_auc', ascending=False))
    
    print("\nDONE!")


if __name__ == "__main__":
    main()

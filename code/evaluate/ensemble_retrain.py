#!/usr/bin/env python3
"""
Ensemble Evaluation with Fine-tuned Models:
- r_detect_retrain (Mistral-7B + LoRA) - best performing detection model
- Embedding model (DeBERTa-v3-Base + KNN)
- Weighted averaging and meta-learning
"""

import os
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

import json
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = "/qwarium/home/d.a.lanovenko/llm-detect-ai"
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = "/qwarium/home/d.a.lanovenko/models"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Model configs - only working models
DETECT_MODELS = {
    'r_detect_retrain': {
        'path': os.path.join(MODELS_DIR, 'r_detect_retrain/best'),
        'base': 'mistralai/Mistral-7B-v0.1',
        'weight': 2.0,  # Higher weight due to best performance
    },
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


def get_detection_predictions(df, model_name, model_config):
    """Get predictions from a detection model."""
    print(f"  Loading {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_config['base'])
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_config['base'],
            num_labels=1,
            torch_dtype=torch.float16,
        ).cuda()
        
        model = PeftModel.from_pretrained(base_model, model_config['path'], is_trainable=False)
        model.eval()
    except Exception as e:
        print(f"    Error loading {model_name}: {e}")
        return None
    
    texts = df['text'].tolist()
    all_probs = []
    
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
    
    del model, base_model, tokenizer
    torch.cuda.empty_cache()
    
    return np.array(all_probs)


def get_embedding_knn_predictions(train_ref_df, test_df, k=5):
    """Get KNN predictions using embedding model."""
    print("  Loading embedding model...")
    
    try:
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
    
    # KNN
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(train_embeds)
    
    distances, indices = knn.kneighbors(test_embeds)
    mean_distances = distances.mean(axis=1)
    neighbor_labels = train_ref_df['is_generated'].values[indices]
    positive_ratios = neighbor_labels.mean(axis=1)
    
    # Normalize distance
    dist_norm = (mean_distances - mean_distances.min()) / (mean_distances.max() - mean_distances.min() + 1e-8)
    
    # Combined score
    scores = 0.5 * dist_norm + 0.5 * positive_ratios
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return scores


def calc_metrics(y_true, y_probs, name=""):
    """Calculate classification metrics."""
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_probs)
    f1_arr = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
    best_idx = np.argmax(f1_arr)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_probs >= best_thresh).astype(int)
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_true, y_probs)),
        'f1': float(f1_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'threshold': float(best_thresh),
    }
    
    print(f"\n  {name}:")
    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"    F1: {metrics['f1']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    Accuracy: {metrics['accuracy']:.4f}")
    
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


def main():
    print("="*70)
    print("ENSEMBLE EVALUATION (Fine-tuned + Embedding)")
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
    print("DETECTION MODELS (Fine-tuned)")
    print("="*70)
    
    for name, config in DETECT_MODELS.items():
        print(f"\n{name}:")
        
        train_preds = get_detection_predictions(train_df, name, config)
        val_preds = get_detection_predictions(val_df, name, config)
        test_preds = get_detection_predictions(test_df, name, config)
        
        if train_preds is not None:
            all_preds['train'][name] = train_preds
            all_preds['val'][name] = val_preds
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
    
    # ========== WEIGHTED AVERAGE ENSEMBLE ==========
    print("\n" + "="*70)
    print("WEIGHTED AVERAGE ENSEMBLE")
    print("="*70)
    
    # Use validation-based weights
    val_scores = {}
    for name in all_preds['val'].keys():
        val_auc = roc_auc_score(y_val, all_preds['val'][name])
        val_scores[name] = max(val_auc - 0.5, 0.01)  # Weight based on val performance
    
    print("\n  Validation AUC scores:")
    for name in val_scores.keys():
        auc = roc_auc_score(y_val, all_preds['val'][name])
        print(f"    {name}: {auc:.4f}")
    
    print("\n  Validation-based weights:")
    for name, weight in val_scores.items():
        print(f"    {name}: {weight:.4f}")
    
    # Normalize weights
    total_weight = sum(val_scores.values())
    normalized_weights = {k: v/total_weight for k, v in val_scores.items()}
    
    # Weighted average
    ensemble_val = weighted_average(
        {k: v for k, v in all_preds['val'].items() if k in normalized_weights},
        {k: v for k, v in normalized_weights.items() if k in all_preds['val']}
    )
    ensemble_test = weighted_average(
        {k: v for k, v in all_preds['test'].items() if k in normalized_weights},
        {k: v for k, v in normalized_weights.items() if k in all_preds['test']}
    )
    
    print("\n  Weighted Ensemble (val):")
    val_metrics = calc_metrics(y_val, ensemble_val, "  ")
    results['weighted_ensemble_val'] = val_metrics
    
    print("\n  Weighted Ensemble (test):")
    test_metrics = calc_metrics(y_test, ensemble_test, "  ")
    results['weighted_ensemble_test'] = test_metrics
    
    # ========== META-LEARNER ENSEMBLE ==========
    print("\n" + "="*70)
    print("META-LEARNER ENSEMBLE (Logistic Regression)")
    print("="*70)
    
    # Get available models (only those that loaded successfully)
    available_models = list(all_preds['train'].keys())
    print(f"\n  Available models for meta-learner: {available_models}")
    
    meta_model = None
    
    if len(available_models) < 2:
        print("  Skipping meta-learner (need at least 2 models)")
        # Use single model predictions
        meta_val_probs = all_preds['val'][available_models[0]]
        meta_test_probs = all_preds['test'][available_models[0]]
        results['meta_learner_val'] = results.get('weighted_ensemble_val', {})
        results['meta_learner_test'] = results.get('weighted_ensemble_test', {})
    else:
        # Train on train set
        train_preds_for_meta = {k: all_preds['train'][k] for k in available_models}
        meta_model = train_meta_learner(train_preds_for_meta, y_train)
        
        # Prepare features for val and test
        val_features = np.array([all_preds['val'][k] for k in available_models]).T
        test_features = np.array([all_preds['test'][k] for k in available_models]).T
        
        print("\n  Meta-learner coefficients:")
        for name, coef in zip(available_models, meta_model.coef_[0]):
            print(f"    {name}: {coef:.4f}")
        
        meta_val_probs = meta_model.predict_proba(val_features)[:, 1]
        meta_test_probs = meta_model.predict_proba(test_features)[:, 1]
        
        print("\n  Meta-Learner (val):")
        val_metrics = calc_metrics(y_val, meta_val_probs, "  ")
        results['meta_learner_val'] = val_metrics
        
        print("\n  Meta-Learner (test):")
        test_metrics = calc_metrics(y_test, meta_test_probs, "  ")
        results['meta_learner_test'] = test_metrics
    
    # ========== SAVE RESULTS ==========
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Save full results
    with open(os.path.join(RESULTS_DIR, "ensemble_retrain_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions for submission
    submission_df = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
        'is_generated': meta_test_probs if meta_model is not None else ensemble_test,
    })
    submission_df.to_csv(os.path.join(RESULTS_DIR, "ensemble_retrain_submission.csv"), index=False)
    
    # Save meta-learner model
    if meta_model is not None:
        with open(os.path.join(RESULTS_DIR, "meta_learner_retrain.pkl"), 'wb') as f:
            pickle.dump(meta_model, f)
        print("  Saved meta-learner model")
    else:
        print("  Meta-learner model not saved (not trained)")
    
    # Summary table
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(os.path.join(RESULTS_DIR, "ensemble_retrain_summary.csv"))
    
    print(f"\nResults saved to {RESULTS_DIR}")
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(summary_df[['roc_auc', 'f1']].sort_values('roc_auc', ascending=False))
    
    print("\nDONE!")


if __name__ == "__main__":
    main()

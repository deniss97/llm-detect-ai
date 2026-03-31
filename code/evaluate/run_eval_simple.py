#!/usr/bin/env python3
"""
Simple zero-shot evaluation script.
"""

import os
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

import sys
import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Paths
BASE_DIR = "/qwarium/home/d.a.lanovenko/llm-detect-ai"
DATA_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = "/qwarium/home/d.a.lanovenko/models"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Model configs
MODELS = {
    "r_detect_competition": {
        "path": os.path.join(MODELS_DIR, "r_detect_competition/last"),
        "backbone": "mistralai/Mistral-7B-v0.1",
    },
    "r_detect_mix_v16": {
        "path": os.path.join(MODELS_DIR, "r_detect_conf_r_detect_mix_v16/last"),
        "backbone": "mistralai/Mistral-7B-v0.1",
    },
    "r_detect_mix_v26": {
        "path": os.path.join(MODELS_DIR, "r_detect_conf_r_detect_mix_v26/last"),
        "backbone": "mistralai/Mistral-7B-v0.1",
    },
    "r_detect_transfer": {
        "path": os.path.join(MODELS_DIR, "r_detect_conf_r_detect_transfer/last"),
        "backbone": "mistralai/Mistral-7B-v0.1",
    },
}

def load_model(name, config):
    print(f"Loading {name}...")
    if not os.path.exists(config['path']):
        print(f"  Path not found: {config['path']}")
        return None
    
    tokenizer = AutoTokenizer.from_pretrained(config['backbone'], use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config['backbone'],
        num_labels=1,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    model = PeftModel.from_pretrained(base_model, config['path'], is_trainable=False)
    model.eval()
    print(f"  OK")
    return model, tokenizer

def predict(model, tokenizer, texts, max_length=256, batch_size=4):
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
        all_probs.extend(probs.tolist())
    return np.array(all_probs)

def calc_metrics(y_true, y_probs):
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
    print("ZERO-SHOT EVALUATION")
    print("="*60)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load test data
    print("\nLoading test data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "detection_test.csv"))
    y_true = df['is_generated'].values
    texts = df['text'].tolist()
    var_indices = df['variation_idx'].values
    print(f"Loaded {len(df)} samples")
    
    # Load models and predict
    predictions = {}
    metrics = {}
    
    for name, config in MODELS.items():
        result = load_model(name, config)
        if result is None:
            print(f"Skipping {name}")
            continue
        
        model, tokenizer = result
        probs = predict(model, tokenizer, texts)
        predictions[name] = probs
        metrics[name] = calc_metrics(y_true, probs)
        
        print(f"\n{name}:")
        print(f"  ROC-AUC: {metrics[name]['roc_auc']:.4f}")
        print(f"  F1: {metrics[name]['f1']:.4f}")
    
    # Ensembles
    print("\n" + "="*60)
    print("ENSEMBLES")
    print("="*60)
    
    model_names = list(predictions.keys())
    ensemble_metrics = {}
    
    # Pairwise ensembles
    for i, n1 in enumerate(model_names):
        for n2 in model_names[i+1:]:
            name = f"{n1}+{n2}"
            probs = (predictions[n1] + predictions[n2]) / 2
            ensemble_metrics[name] = calc_metrics(y_true, probs)
            print(f"{name}: AUC={ensemble_metrics[name]['roc_auc']:.4f}")
    
    # Save results
    print("\nSaving results...")
    
    single_df = pd.DataFrame(metrics).T
    single_df.to_csv(os.path.join(RESULTS_DIR, "zero_shot_single_metrics.csv"))
    
    ensemble_df = pd.DataFrame(ensemble_metrics).T
    ensemble_df.to_csv(os.path.join(RESULTS_DIR, "zero_shot_ensemble_metrics.csv"))
    
    with open(os.path.join(RESULTS_DIR, "zero_shot_summary.json"), 'w') as f:
        json.dump({
            'single': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics.items()},
            'ensembles': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in ensemble_metrics.items()},
        }, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}")
    print("DONE!")

if __name__ == "__main__":
    main()

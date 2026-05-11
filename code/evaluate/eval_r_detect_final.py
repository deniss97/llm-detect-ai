#!/usr/bin/env python3
"""
Evaluate r_detect model trained on final_dataset
"""

import os
import sys
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from peft import PeftModel

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


def evaluate_model(model_path, test_csv_path, device='cuda'):
    """
    Evaluate model on test dataset
    
    Args:
        model_path: Path to trained model directory
        test_csv_path: Path to test CSV file
        device: Device to use for inference
    
    Returns:
        auc, acc, f1: Metrics
    """
    print(f"Loading model from: {model_path}")
    print(f"Loading test data from: {test_csv_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model from cache
    base_model_path = "/tmp/huggingface_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da"
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=True
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Load test data
    df = pd.read_csv(test_csv_path)
    print(f"Test data shape: {df.shape}")
    
    # Handle different column names
    if 'generated' in df.columns:
        label_col = 'generated'
    elif 'is_generated' in df.columns:
        label_col = 'is_generated'
    elif 'label' in df.columns:
        # Convert label (human/ai) to generated (0/1)
        df['generated'] = (df['label'] == 'ai').astype(int)
        label_col = 'generated'
    else:
        raise ValueError("No label column found in test data")
    
    print(f"Class distribution: {df[label_col].value_counts().to_dict()}")
    
    # Create dataset
    dataset = TestDataset(
        df['text'].values,
        df[label_col].values,
        tokenizer,
        max_length=512
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Predict
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy().tolist())
    
    # Calculate metrics
    all_predictions = torch.sigmoid(torch.tensor(all_predictions)).numpy()
    
    auc = roc_auc_score(all_labels, all_predictions)
    acc = accuracy_score(all_labels, (all_predictions > 0.5).astype(int))
    f1 = f1_score(all_labels, (all_predictions > 0.5).astype(int))
    
    print(f"\n=== Test Set Performance ===")
    print(f"Total samples: {len(all_labels)}")
    print(f"Class distribution: {{0: {all_labels.count(0)}, 1: {all_labels.count(1)}}}")
    print(f"\nAUC-ROC: {auc:.6f}")
    print(f"Accuracy (threshold=0.5): {acc:.4f}")
    print(f"F1 Score (threshold=0.5): {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, (all_predictions > 0.5).astype(int), digits=4))
    
    # Save predictions
    result_df = pd.DataFrame({
        'id': range(len(all_predictions)),
        'predictions': all_predictions,
        'truths': all_labels
    })
    
    # Save to model directory
    output_dir = os.path.dirname(model_path) if os.path.basename(model_path) in ['best', 'last'] else model_path
    os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(os.path.join(output_dir, 'final_test_results.csv'), index=False)
    print(f"\nPredictions saved to: {os.path.join(output_dir, 'final_test_results.csv')}")
    
    # Save metrics summary
    metrics = {
        'auc': auc,
        'accuracy': acc,
        'f1': f1,
        'num_samples': len(all_labels),
        'num_positive': all_labels.count(1),
        'num_negative': all_labels.count(0)
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'final_metrics_summary.csv'), index=False)
    print(f"Metrics saved to: {os.path.join(output_dir, 'final_metrics_summary.csv')}")
    
    return auc, acc, f1


if __name__ == "__main__":
    # Default paths
    model_path = "/qwarium/home/d.a.lanovenko/models/r_detect_final_dataset/best"
    test_csv_path = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared/final_valid.csv"
    
    # Allow command line override
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        test_csv_path = sys.argv[2]
    
    evaluate_model(model_path, test_csv_path)

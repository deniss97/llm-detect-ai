#!/usr/bin/env python3
"""
Prepare final_dataset.csv for training
Converts the dataset to the format expected by train_r_detect.py:
- source_id -> id
- label (human/ai) -> generated (0/1)
- Keeps text column as is
"""

import pandas as pd
import os

def prepare_final_dataset(input_path, output_dir, test_split=0.2, seed=42):
    """
    Prepare final_dataset.csv for training
    
    Args:
        input_path: Path to final_dataset.csv
        output_dir: Directory to save prepared datasets
        test_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
    """
    # Load dataset
    df = pd.read_csv(input_path)
    
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert label to generated (0 for human, 1 for ai)
    df['generated'] = (df['label'] == 'ai').astype(int)
    
    # Create id column from source_id
    df['id'] = [f"essay_{i}" for i in range(len(df))]
    
    # Reorder columns to match expected format
    df = df[['id', 'text', 'generated', 'model', 'prompt_type']]
    
    print(f"\nLabel distribution:")
    print(f"  Human (generated=0): {(df['generated'] == 0).sum()}")
    print(f"  AI (generated=1): {(df['generated'] == 1).sum()}")
    
    print(f"\nModel distribution:")
    print(df['model'].value_counts())
    
    # Split into train and validation
    from sklearn.model_selection import train_test_split
    
    # Stratified split to maintain class balance
    train_df, valid_df = train_test_split(
        df, 
        test_size=test_split, 
        random_state=seed,
        stratify=df['generated']
    )
    
    print(f"\nTrain/Valid split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Valid: {len(valid_df)} samples")
    
    print(f"\nTrain class distribution:")
    print(train_df['generated'].value_counts().to_dict())
    
    print(f"\nValid class distribution:")
    print(valid_df['generated'].value_counts().to_dict())
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    train_path = os.path.join(output_dir, "final_train.csv")
    valid_path = os.path.join(output_dir, "final_valid.csv")
    full_path = os.path.join(output_dir, "final_full.csv")
    
    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    df.to_csv(full_path, index=False)
    
    print(f"\nSaved datasets:")
    print(f"  Train: {train_path}")
    print(f"  Valid: {valid_path}")
    print(f"  Full: {full_path}")
    
    return train_df, valid_df


if __name__ == "__main__":
    input_path = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_dataset.csv"
    output_dir = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared"
    
    train_df, valid_df = prepare_final_dataset(input_path, output_dir)
    
    print("\n✅ Dataset preparation complete!")

#!/usr/bin/env python3
"""
Prepare final_dataset.csv for training
CRITICAL FIX: Split by source_id to keep human+AI pairs together!

Converts the dataset to the format expected by train_r_detect.py:
- source_id -> id
- label (human/ai) -> generated (0/1)
- Keeps text column as is

IMPORTANT: Split is done by source_id to prevent data leak!
All rows with the same source_id (human + AI versions) go to the same split.
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
    df = df[['id', 'text', 'generated', 'model', 'prompt_type', 'source_id']]
    
    print(f"\nLabel distribution:")
    print(f"  Human (generated=0): {(df['generated'] == 0).sum()}")
    print(f"  AI (generated=1): {(df['generated'] == 1).sum()}")
    
    print(f"\nModel distribution:")
    print(df['model'].value_counts())
    
    # CRITICAL FIX: Split by source_id to keep pairs together!
    from sklearn.model_selection import train_test_split
    
    # Get unique source_ids
    unique_source_ids = df['source_id'].unique()
    print(f"\nUnique source_ids: {len(unique_source_ids)}")
    
    # Stratified split by source_id (not by individual rows!)
    # This ensures all rows with the same source_id go to the same split
    train_source_ids, valid_source_ids = train_test_split(
        unique_source_ids,
        test_size=test_split,
        random_state=seed,
        stratify=[1 if sid < 1212 else 0 for sid in unique_source_ids]  # Stratify: 1212 pairs + 24 human-only
    )
    
    print(f"\nTrain source_ids: {len(train_source_ids)}")
    print(f"Valid source_ids: {len(valid_source_ids)}")
    
    # Filter dataframe by source_ids
    train_df = df[df['source_id'].isin(train_source_ids)].copy()
    valid_df = df[df['source_id'].isin(valid_source_ids)].copy()
    
    # Drop source_id column (not needed for training)
    train_df = train_df.drop(columns=['source_id'])
    valid_df = valid_df.drop(columns=['source_id'])
    
    print(f"\nTrain/Valid split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Valid: {len(valid_df)} samples")
    
    print(f"\nTrain class distribution:")
    print(train_df['generated'].value_counts().to_dict())
    
    print(f"\nValid class distribution:")
    print(valid_df['generated'].value_counts().to_dict())
    
    # VERIFY: Check that pairs are kept together
    print("\n=== VERIFYING NO DATA LEAK ===")
    
    # For train: check if any source_id has both human and AI
    train_grouped = train_df.groupby('id')
    train_mixed = sum(1 for _, g in train_grouped if g['generated'].nunique() == 2)
    print(f"Train: {train_mixed} essays with both human+AI (expected: 0, because id is unique per row)")
    
    # Better check: for each original source_id, verify all rows are in same split
    # We need to reconstruct source_id from id
    train_df['source_id_check'] = train_df['id'].apply(lambda x: int(x.split('_')[1]))
    valid_df['source_id_check'] = valid_df['id'].apply(lambda x: int(x.split('_')[1]))
    
    # Check pairs
    pairs_split = 0
    pairs_together = 0
    
    for sid in range(1212):  # source_id 0-1211 have pairs
        in_train = (train_df['source_id_check'] == sid)
        in_valid = (valid_df['source_id_check'] == sid)
        
        if in_train.any() and in_valid.any():
            pairs_split += 1
        elif in_train.any() or in_valid.any():
            pairs_together += 1
    
    print(f"\nPairs verification:")
    print(f"  Pairs kept together: {pairs_together}")
    print(f"  Pairs SPLIT (DATA LEAK): {pairs_split}")
    
    if pairs_split > 0:
        print("❌ ERROR: Data leak detected! Pairs are split between train and valid!")
        raise ValueError("Data leak: pairs are split between train and valid")
    else:
        print("✅ VERIFIED: No data leak - all pairs are kept together!")
    
    # Drop helper column
    train_df = train_df.drop(columns=['source_id_check'])
    valid_df = valid_df.drop(columns=['source_id_check'])
    
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

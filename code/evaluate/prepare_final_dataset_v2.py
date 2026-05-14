#!/usr/bin/env python3
"""
Prepare final_dataset.csv for training - FIXED VERSION
CRITICAL: Create proper ID that groups human+AI pairs together!

The issue: original script created unique ID for each row (essay_0, essay_1, ...)
But we need ID that groups human+AI pairs (same source_id should have same base ID)

Solution: Use source_id as the base for ID, so human and AI versions share the same base ID
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def prepare_final_dataset_fixed(input_path, output_dir, test_split=0.2, seed=42):
    """
    Prepare final_dataset.csv with CORRECT ID scheme
    
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
    
    # CRITICAL FIX: Create ID based on source_id, not row index
    # This way human and AI versions of the same essay have related IDs
    # Format: essay_{source_id}_{model_short}
    # But for grouping, we'll use source_id directly
    
    print(f"\nOriginal source_id distribution:")
    print(df.groupby('source_id').size().value_counts())
    
    # Stratified split by source_id to keep pairs together!
    unique_source_ids = df['source_id'].unique()
    print(f"\nUnique source_ids: {len(unique_source_ids)}")
    
    # Create stratification: 1 if source_id has AI pair, 0 if human-only
    # source_id 0-1211 have pairs, 1212-1235 are human-only
    stratify_labels = [1 if sid < 1212 else 0 for sid in unique_source_ids]
    
    # Split by source_id
    train_source_ids, valid_source_ids = train_test_split(
        unique_source_ids,
        test_size=test_split,
        random_state=seed,
        stratify=stratify_labels
    )
    
    print(f"\nTrain source_ids: {len(train_source_ids)}")
    print(f"Valid source_ids: {len(valid_source_ids)}")
    
    # Filter dataframe by source_ids
    train_df = df[df['source_id'].isin(train_source_ids)].copy()
    valid_df = df[df['source_id'].isin(valid_source_ids)].copy()
    
    # Create proper ID: essay_{source_id}_{counter}
    # This ensures human and AI have different but related IDs
    def create_id(row, counter_dict):
        sid = row['source_id']
        if sid not in counter_dict:
            counter_dict[sid] = 0
        counter_dict[sid] += 1
        return f"essay_{sid}_{counter_dict[sid]}"
    
    train_counter = {}
    valid_counter = {}
    
    train_df['id'] = train_df.apply(lambda row: create_id(row, train_counter), axis=1)
    valid_df['id'] = valid_df.apply(lambda row: create_id(row, valid_counter), axis=1)
    
    # Reorder columns
    train_df = train_df[['id', 'text', 'generated', 'model', 'prompt_type', 'source_id']]
    valid_df = valid_df[['id', 'text', 'generated', 'model', 'prompt_type', 'source_id']]
    
    print(f"\nTrain/Valid split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Valid: {len(valid_df)} samples")
    
    print(f"\nTrain class distribution:")
    print(train_df['generated'].value_counts().to_dict())
    
    print(f"\nValid class distribution:")
    print(valid_df['generated'].value_counts().to_dict())
    
    # VERIFY: Check that pairs are kept together
    print("\n=== VERIFY NO DATA LEAK ===")
    
    # Check that no source_id appears in both splits
    train_sids = set(train_df['source_id'].unique())
    valid_sids = set(valid_df['source_id'].unique())
    
    overlap = train_sids & valid_sids
    print(f"Source_id overlap between train/valid: {len(overlap)} (should be 0)")
    
    if len(overlap) > 0:
        print("❌ ERROR: Data leak detected!")
        raise ValueError("Data leak: source_ids overlap between train and valid")
    
    # Check that pairs are together within each split
    for split_name, split_df in [("Train", train_df), ("Valid", valid_df)]:
        grouped = split_df.groupby('source_id')['generated'].apply(lambda x: tuple(sorted(x.unique())))
        pairs_mixed = sum(1 for _, labels in grouped.items() if len(labels) == 2)
        print(f"{split_name}: {pairs_mixed} source_ids with both human+AI")
    
    print("✅ VERIFIED: No data leak - all pairs are kept together!")
    
    # Drop source_id column (not needed for training, but keep for verification)
    # Actually, let's keep it for now
    
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
    
    # Show example
    print("\n=== EXAMPLE IDs ===")
    print("Train sample:")
    print(train_df[train_df['source_id'] == 0][['id', 'generated', 'model', 'source_id']].to_string())
    
    return train_df, valid_df


if __name__ == "__main__":
    input_path = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_dataset.csv"
    output_dir = "/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_prepared_v2"
    
    train_df, valid_df = prepare_final_dataset_fixed(input_path, output_dir)
    
    print("\n✅ Dataset preparation complete!")

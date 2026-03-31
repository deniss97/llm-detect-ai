#!/usr/bin/env python3
"""
Script to prepare detection dataset from generated essay variations.

This script:
1. Loads generated essay variations (generated_variations_mistral_full.csv)
2. Creates a dataset with 'is_generated' label (0=original, 1=generated)
3. Splits into train/val/test with stratification by original essay
   (all 5 variations of one essay go to the same split)
4. Saves the datasets for evaluation

Usage:
    python code/evaluate/prepare_detection_dataset.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Paths
BASE_DIR = "/qwarium/home/d.a.lanovenko/llm-detect-ai"
INPUT_FILE = os.path.join(BASE_DIR, "datasets/generated_variations_mistral_full.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets")

# Split ratios
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def load_generated_essays(filepath: str) -> pd.DataFrame:
    """Load the generated essays dataset."""
    print(f"Loading generated essays from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows")
    return df


def extract_essay_index(essay_id: str) -> int:
    """Extract original essay index from generated essay id.
    
    Args:
        essay_id: Format like 'essay_0_var0', 'essay_123_var4'
    
    Returns:
        Original essay index (int)
    """
    # Format: essay_IDX_varN
    parts = essay_id.replace('essay_', '').split('_var')
    return int(parts[0])


def create_detection_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create detection dataset with is_generated label.
    
    For each row in generated_variations:
    - original_en -> is_generated = 0 (original human text)
    - generated_en -> is_generated = 1 (AI generated text)
    
    Returns:
        DataFrame with columns: text, is_generated, essay_idx, variation_idx, source, year
    """
    print("\nCreating detection dataset...")
    
    detection_rows = []
    
    for idx, row in df.iterrows():
        essay_idx = extract_essay_index(row['id'])
        variation_idx = row.get('variation_idx', 0)
        source = row.get('source', 'unknown')
        year = row.get('year', 2020)
        prompt_name = row.get('prompt_name', 'unknown')
        
        # Original essay (is_generated = 0)
        original_en = row.get('original_en', '')
        if original_en and str(original_en).strip() and not pd.isna(original_en):
            detection_rows.append({
                'text': str(original_en).strip(),
                'is_generated': 0,
                'essay_idx': essay_idx,
                'variation_idx': -1,  # -1 for original
                'source': source,
                'year': year,
                'prompt_name': prompt_name,
                'original_essay_idx': essay_idx,
            })
        
        # Generated variation (is_generated = 1)
        generated_en = row.get('generated_en', '')
        if generated_en and str(generated_en).strip() and not pd.isna(generated_en):
            detection_rows.append({
                'text': str(generated_en).strip(),
                'is_generated': 1,
                'essay_idx': essay_idx,
                'variation_idx': variation_idx,
                'source': source,
                'year': year,
                'prompt_name': prompt_name,
                'original_essay_idx': essay_idx,
            })
    
    detection_df = pd.DataFrame(detection_rows)
    
    print(f"Created detection dataset with {len(detection_df)} samples")
    print(f"  - Original (is_generated=0): {len(detection_df[detection_df['is_generated']==0])}")
    print(f"  - Generated (is_generated=1): {len(detection_df[detection_df['is_generated']==1])}")
    
    return detection_df


def stratified_split_by_essay(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42
) -> tuple:
    """
    Split dataset with stratification by original essay index.
    
    All variations of the same original essay go to the same split.
    
    Args:
        df: Detection dataset with essay_idx column
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
    
    Returns:
        Tuple of (train_df, val_df, test_df, split_info)
    """
    print("\nPerforming stratified split by original essay...")
    
    # Get unique essay indices
    unique_essays = df['essay_idx'].unique()
    print(f"Total unique original essays: {len(unique_essays)}")
    
    # First split: train vs (val + test)
    train_essays, temp_essays = train_test_split(
        unique_essays,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True
    )
    
    # Second split: val vs test (from remaining)
    remaining_ratio = val_ratio / (val_ratio + test_ratio)
    val_essays, test_essays = train_test_split(
        temp_essays,
        train_size=remaining_ratio,
        random_state=seed,
        shuffle=True
    )
    
    print(f"Split essay counts: train={len(train_essays)}, val={len(val_essays)}, test={len(test_essays)}")
    
    # Filter dataframe by essay indices
    train_df = df[df['essay_idx'].isin(train_essays)].reset_index(drop=True)
    val_df = df[df['essay_idx'].isin(val_essays)].reset_index(drop=True)
    test_df = df[df['essay_idx'].isin(test_essays)].reset_index(drop=True)
    
    # Create split info
    split_info = {
        'seed': seed,
        'ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'essay_counts': {
            'train': len(train_essays),
            'val': len(val_essays),
            'test': len(test_essays),
            'total': len(unique_essays)
        },
        'sample_counts': {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df),
            'total': len(df)
        },
        'train_essay_indices': sorted([int(x) for x in train_essays]),
        'val_essay_indices': sorted([int(x) for x in val_essays]),
        'test_essay_indices': sorted([int(x) for x in test_essays])
    }
    
    return train_df, val_df, test_df, split_info


def print_dataset_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_info: dict
):
    """Print detailed statistics about the splits."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name} split:")
        print(f"  Total samples: {len(df)}")
        print(f"  Unique essays: {df['essay_idx'].nunique()}")
        
        # Class distribution
        original_count = len(df[df['is_generated'] == 0])
        generated_count = len(df[df['is_generated'] == 1])
        print(f"  Original (is_generated=0): {original_count} ({100*original_count/len(df):.1f}%)")
        print(f"  Generated (is_generated=1): {generated_count} ({100*generated_count/len(df):.1f}%)")
        
        # Variation distribution (for generated only)
        if generated_count > 0:
            gen_df = df[df['is_generated'] == 1]
            var_counts = gen_df['variation_idx'].value_counts().sort_index()
            print(f"  Variations distribution:")
            for var_idx, count in var_counts.items():
                print(f"    var{var_idx}: {count} samples")
    
    print("\n" + "="*70)


def save_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_info: dict,
    output_dir: str
):
    """Save datasets and split info to files."""
    print(f"\nSaving datasets to: {output_dir}")
    
    # Save CSV files
    train_path = os.path.join(output_dir, "detection_train.csv")
    val_path = os.path.join(output_dir, "detection_val.csv")
    test_path = os.path.join(output_dir, "detection_test.csv")
    full_path = os.path.join(output_dir, "detection_full.csv")
    
    # Combine all for full dataset
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Save to CSV
    full_df.to_csv(full_path, index=False, encoding='utf-8')
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    val_df.to_csv(val_path, index=False, encoding='utf-8')
    test_df.to_csv(test_path, index=False, encoding='utf-8')
    
    print(f"  Saved: {full_path} ({len(full_df)} samples)")
    print(f"  Saved: {train_path} ({len(train_df)} samples)")
    print(f"  Saved: {val_path} ({len(val_df)} samples)")
    print(f"  Saved: {test_path} ({len(test_df)} samples)")
    
    # Save split indices as JSON
    split_info_path = os.path.join(output_dir, "detection_split_indices.json")
    with open(split_info_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {split_info_path}")


def main():
    """Main function to prepare detection dataset."""
    print("="*70)
    print("PREPARING DETECTION DATASET")
    print("="*70)
    
    # Step 1: Load generated essays
    df = load_generated_essays(INPUT_FILE)
    
    # Step 2: Create detection dataset with labels
    detection_df = create_detection_dataset(df)
    
    # Step 3: Stratified split by original essay
    train_df, val_df, test_df, split_info = stratified_split_by_essay(
        detection_df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED
    )
    
    # Step 4: Print statistics
    print_dataset_statistics(train_df, val_df, test_df, split_info)
    
    # Step 5: Save datasets
    save_datasets(train_df, val_df, test_df, split_info, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE!")
    print("="*70)
    
    return train_df, val_df, test_df, split_info


if __name__ == "__main__":
    main()

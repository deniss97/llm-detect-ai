import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Load final_dataset.csv
input_path = '/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_dataset.csv'
output_dir = '/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_for_embed_ranking_v3'

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)

print(f"Original dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Transform to expected format
transformed_df = pd.DataFrame()
transformed_df['id'] = range(len(df))
# Use source_id as prompt_id (each original essay + its AI versions form a group)
transformed_df['prompt_id'] = df['source_id']
transformed_df['text'] = df['text']
transformed_df['generated'] = (df['label'] == 'ai').astype(int)

# Filter to keep only prompt_ids that have BOTH human and AI texts
print("\nFiltering prompt_ids to keep only those with both human and AI texts...")

# Group by prompt_id and check if both classes exist
prompt_stats = transformed_df.groupby('prompt_id')['generated'].agg(['sum', 'count'])
prompt_stats['has_human'] = prompt_stats['count'] - prompt_stats['sum'] > 0
prompt_stats['has_ai'] = prompt_stats['sum'] > 0
prompt_stats['has_both'] = prompt_stats['has_human'] & prompt_stats['has_ai']

valid_prompts = prompt_stats[prompt_stats['has_both']].index.tolist()
print(f"Total unique prompt_ids: {transformed_df['prompt_id'].nunique()}")
print(f"Prompt_ids with both classes: {len(valid_prompts)}")

# Filter dataframe
filtered_df = transformed_df[transformed_df['prompt_id'].isin(valid_prompts)].copy()
filtered_df = filtered_df.reset_index(drop=True)

# Remap prompt_id to 0-N range (for compatibility with training script that expects prompt_id <= 8)
# We'll use only first 9 prompts (0-8) for training, as the script expects
unique_prompts = sorted(filtered_df['prompt_id'].unique())[:9]
print(f"\nUsing first 9 prompt_ids: {unique_prompts}")

# Filter to only these 9 prompts
filtered_df = filtered_df[filtered_df['prompt_id'].isin(unique_prompts)].copy()

# Remap prompt_id to 0-8 range
prompt_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_prompts)}
filtered_df['prompt_id'] = filtered_df['prompt_id'].map(prompt_mapping)

print(f"\nFinal dataset shape: {filtered_df.shape}")
print(f"Generated distribution:\n{filtered_df['generated'].value_counts()}")
print(f"Prompt_id distribution:\n{filtered_df['prompt_id'].value_counts().sort_index()}")

# Split into train/valid (90/10)
train_df, valid_df = train_test_split(
    filtered_df, 
    test_size=0.1, 
    random_state=42,
    stratify=filtered_df['generated']
)

print(f"\nTrain shape: {train_df.shape}")
print(f"Valid shape: {valid_df.shape}")
print(f"Train generated distribution:\n{train_df['generated'].value_counts()}")
print(f"Valid generated distribution:\n{valid_df['generated'].value_counts()}")

# Save
train_df.to_csv(os.path.join(output_dir, 'train_essays.csv'), index=False)
valid_df.to_csv(os.path.join(output_dir, 'valid_essays.csv'), index=False)
filtered_df.to_csv(os.path.join(output_dir, 'all_essays.csv'), index=False)

print(f"\nSaved to {output_dir}")
print(f"Files: train_essays.csv, valid_essays.csv, all_essays.csv")

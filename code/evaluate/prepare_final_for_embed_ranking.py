import pandas as pd
import os

# Load final_dataset.csv
input_path = '/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_dataset.csv'
output_dir = '/qwarium/home/d.a.lanovenko/llm-detect-ai/datasets/final_for_embed_ranking'

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)

print(f"Original dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Label distribution:\n{df['label'].value_counts()}")

# Transform to expected format:
# - id: unique identifier
# - prompt_id: we can use source_id as prompt_id (groups texts by original essay)
# - text: the essay text
# - generated: 0 for human, 1 for AI

transformed_df = pd.DataFrame()
transformed_df['id'] = range(len(df))
transformed_df['prompt_id'] = df['source_id']
transformed_df['text'] = df['text']
transformed_df['generated'] = (df['label'] == 'ai').astype(int)

print(f"\nTransformed dataset shape: {transformed_df.shape}")
print(f"Generated distribution:\n{transformed_df['generated'].value_counts()}")

# Split into train/valid (90/10)
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    transformed_df, 
    test_size=0.1, 
    random_state=42,
    stratify=transformed_df['generated']
)

print(f"\nTrain shape: {train_df.shape}")
print(f"Valid shape: {valid_df.shape}")
print(f"Train generated distribution:\n{train_df['generated'].value_counts()}")
print(f"Valid generated distribution:\n{valid_df['generated'].value_counts()}")

# Save
train_df.to_csv(os.path.join(output_dir, 'train_essays.csv'), index=False)
valid_df.to_csv(os.path.join(output_dir, 'valid_essays.csv'), index=False)
transformed_df.to_csv(os.path.join(output_dir, 'all_essays.csv'), index=False)

print(f"\nSaved to {output_dir}")
print(f"Files: train_essays.csv, valid_essays.csv, all_essays.csv")

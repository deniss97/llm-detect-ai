#!/usr/bin/env python3
"""
Test script to generate essay variations using Mistral-7B model.
"""

import argparse
import os
import random
import time
from typing import List, Optional, Dict, Set

import pandas as pd
import torch
import warnings
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

# Suppress transformers warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id`")
warnings.filterwarnings("ignore", message=".*deprecated.*")


def load_existing_results(output_path: str, num_variations: int) -> Set[str]:
    """
    Load existing results and return set of essay indices that already have all variations.
    
    Args:
        output_path: Path to output CSV file
        num_variations: Expected number of variations per essay
    
    Returns:
        Set of essay indices (as strings) that are complete
    """
    if not os.path.exists(output_path):
        return set()
    
    try:
        existing_df = pd.read_csv(output_path)
        if len(existing_df) == 0:
            return set()
        
        # Count variations per essay
        essay_counts = existing_df.groupby('id').size()
        
        # Extract essay index from id (format: essay_IDX_varN)
        complete_essays = set()
        for essay_id, count in essay_counts.items():
            if count >= num_variations:
                # Extract index from id like "essay_0_var0"
                essay_idx = essay_id.split('_var')[0].replace('essay_', '')
                complete_essays.add(essay_idx)
        
        return complete_essays
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return set()


def save_progress(df: pd.DataFrame, output_path: str):
    """Save current progress to CSV."""
    df.to_csv(output_path, index=False, encoding='utf-8')


def get_instruction(
    prompt_name: str = "Unknown",
    task: str = "Essay Writing",
    score: int = 5,
    grade_level: int = 10,
    ell_status: str = "Non-ELL",
    disability_status: str = "No Disability",
) -> str:
    """Create instruction string from metadata."""
    return f"""
Prompt: {prompt_name}
Task: {task}
Score: {score}
Student Grade Level: {grade_level}
English Language Learner: {ell_status}
Disability Status: {disability_status}
    """.strip()


def format_prompt_mistral(instruction: str, prefix: str = "") -> str:
    """Format the full prompt for Mistral model using chat template."""
    return f"<s>[INST] {instruction} [/INST] {prefix}"


def load_model(model_path: str):
    """Load the trained model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.eval()
    print(f"Model loaded successfully!")
    return model, tokenizer


def generate_variation(
    model,
    tokenizer,
    original_text: str,
    instruction: str,
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    prefix_ratio: float = 0.3,
) -> str:
    """
    Generate a variation of the original text using Mistral.
    """
    # Take a prefix of the original text
    words = original_text.split()
    prefix_len = max(10, int(len(words) * prefix_ratio))
    prefix = " ".join(words[:prefix_len])
    
    # Create the full prompt using Mistral format
    full_prompt = format_prompt_mistral(instruction, prefix)
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length - input_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # Combine prefix with generated continuation
    full_text = f"{prefix} {generated_text}".strip()
    
    return full_text


def main():
    parser = argparse.ArgumentParser(
        description="Test Mistral-7B for essay variation generation with resume support"
    )
    
    parser.add_argument("--model_path", type=str, default="/tmp/models/mistral_7b/last",
                        help="Path to Mistral model")
    parser.add_argument("--input", type=str, default="llm-detect-ai/datasets/translated_essays.csv",
                        help="Input CSV file with translated essays")
    parser.add_argument("--output", type=str, default="llm-detect-ai/datasets/generated_variations_mistral.csv",
                        help="Output CSV for generated variations")
    parser.add_argument("--num_variations", type=int, default=5,
                        help="Number of variations per essay")
    parser.add_argument("--max_essays", type=int, default=-1,
                        help="Maximum essays to process (-1 = all)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum length of generated text")
    parser.add_argument("--prefix_ratio", type=float, default=0.3,
                        help="Ratio of original text to use as prefix")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save progress every N essays")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load input data
    print(f"Loading essays from: {args.input}")
    df = pd.read_csv(args.input)
    
    if args.max_essays > 0:
        df = df.head(args.max_essays)
    
    print(f"Processing {len(df)} essays...")
    
    # Load model
    print("\n" + "="*60)
    print("Loading Mistral-7B model")
    print("="*60)
    
    model, tokenizer = load_model(args.model_path)
    
    # Generate variations
    print("\n" + "="*60)
    print("Generating variations with Mistral-7B")
    print("="*60)
    
    # Filter out essays without valid English translation
    df = df[df['Текст_en'].notna() & (df['Текст_en'] != '')].reset_index(drop=True)
    print(f"Essays with valid English translation: {len(df)}")
    
    # Load existing results for resume capability
    print(f"\nChecking for existing results in: {args.output}")
    complete_essays = load_existing_results(args.output, args.num_variations)
    print(f"Already complete essays (with {args.num_variations} variations): {len(complete_essays)}")
    
    # Load existing results into DataFrame
    if os.path.exists(args.output):
        try:
            results_df = pd.read_csv(args.output)
            results = results_df.to_dict('records')
            print(f"Loaded {len(results)} existing variations")
        except Exception as e:
            print(f"Warning: Could not load existing results, starting fresh: {e}")
            results = []
    else:
        results = []
    
    skipped = 0
    saved_count = len(results)
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        essay_idx_str = str(idx)
        
        # Skip if this essay is already complete
        if essay_idx_str in complete_essays:
            skipped += 1
            continue
        
        # Use actual column names from translated_essays.csv
        original_ru = row.get('Текст', '')
        original_en = row.get('Текст_en', '')
        
        if not original_en or pd.isna(original_en) or str(original_en).strip() == '':
            skipped += 1
            continue
        
        source = row.get('Источник', 'Unknown')
        year = row.get('Год', 2020)
        prompt_name = source.split('/')[-1].replace('.html', '').replace('#respond', '') if pd.notna(source) and source else 'Essay'
        
        for var_idx in range(args.num_variations):
            # Check if this specific variation already exists
            var_id = f'essay_{idx}_var{var_idx}'
            if any(r['id'] == var_id for r in results):
                continue
            
            instruction = get_instruction(
                prompt_name=f"{prompt_name}_var{var_idx}",
                task="Essay Writing",
                score=random.randint(4, 6),
                grade_level=random.randint(9, 11),
                ell_status="Non-ELL",
                disability_status="No Disability",
            )
            
            generated = generate_variation(
                model, tokenizer, original_en, instruction,
                max_length=args.max_length,
                prefix_ratio=args.prefix_ratio,
                temperature=0.7 + random.random() * 0.2,
            )
            
            result = {
                'id': var_id,
                'original_ru': original_ru,
                'original_en': original_en,
                'generated_en': generated,
                'source': source,
                'year': year,
                'prompt_name': prompt_name,
                'variation_idx': var_idx,
                'model': 'mistral_7b',
            }
            results.append(result)
        
        # Mark this essay as complete
        complete_essays.add(essay_idx_str)
        
        # Save progress periodically
        if len(results) - saved_count >= args.save_interval * args.num_variations:
            result_df = pd.DataFrame(results)
            save_progress(result_df, args.output)
            print(f"\n💾 Progress saved: {len(results)} variations ({len(complete_essays)} essays complete)")
            saved_count = len(results)
    
    # Final save
    result_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    result_df.to_csv(args.output, index=False, encoding='utf-8')
    print(f"\nResults saved to: {args.output}")
    
    # Print samples
    if len(result_df) > 0:
        print("\n" + "="*80)
        print("GENERATED VARIATION SAMPLES (Mistral-7B)")
        print("="*80)
        
        for idx, row in result_df.head(3).iterrows():
            print(f"\n{'='*80}")
            print(f"ESSAY: {row['id']}")
            print(f"{'='*80}")
            
            print(f"\n📝 ORIGINAL ENGLISH (first 400 chars):")
            print(row['original_en'][:400])
            
            print(f"\n🤖 GENERATED BY MISTRAL (first 400 chars):")
            print(row['generated_en'][:400])
            
            # Calculate overlap
            orig_words = set(row['original_en'].lower().split())
            gen_words = set(row['generated_en'].lower().split())
            overlap = len(orig_words & gen_words) / max(len(orig_words), 1) * 100
            
            print(f"\n📊 STATISTICS:")
            print(f"  - Original length: {len(row['original_en'])} chars")
            print(f"  - Generated length: {len(row['generated_en'])} chars")
            print(f"  - Word overlap: {overlap:.1f}%")
            print("-"*80)
    
    print(f"\n✅ Total variations generated: {len(result_df)}")
    print(f"📁 Output file: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script for generating AI variations from already translated essays.
This script skips translation and only generates variations using the r_clm model.

Usage:
    python generate_variations_only.py --input datasets/translated_essays_full.csv --output datasets/generated_variations.csv --model_path /tmp/models/r_clm_v2/last
"""

import argparse
import os
import random
import time
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


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
            max_length=max_length,
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


def process_and_generate(
    df: pd.DataFrame,
    model,
    tokenizer,
    text_column_ru: str = 'Текст',
    text_column_en: str = 'Текст_en',
    output_file: Optional[str] = None,
    num_variations: int = 1,
    **generation_kwargs,
) -> pd.DataFrame:
    """
    Process translated essays and generate AI variations.
    Skips rows without valid English translation.
    
    Args:
        df: DataFrame with translated texts
        model: The trained model
        tokenizer: Tokenizer
        text_column_ru: Column with Russian texts
        text_column_en: Column with English translations
        output_file: Optional path to save results
        num_variations: Number of variations per essay
        **generation_kwargs: Arguments for generation
    
    Returns:
        DataFrame with original and generated texts
    """
    results = []
    skipped_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating variations"):
        original_ru = row.get(text_column_ru, '')
        original_en = row.get(text_column_en, '')
        
        # Skip rows without valid English translation
        if not original_en or pd.isna(original_en) or str(original_en).strip() == '':
            skipped_count += 1
            continue
        
        # Create instruction from metadata
        source = row.get('Источник', 'Unknown')
        year = row.get('Год', 2020)
        
        # Extract prompt name from source
        prompt_name = source.split('/')[-1].replace('.html', '').replace('#respond', '') if source else 'Essay'
        
        for var_idx in range(num_variations):
            instruction = get_instruction(
                prompt_name=f"{prompt_name}_var{var_idx}",
                task="Essay Writing",
                score=random.randint(4, 6),
                grade_level=random.randint(9, 11),
                ell_status="Non-ELL",
                disability_status="No Disability",
            )
            
            # Add some randomness to generation
            var_kwargs = generation_kwargs.copy()
            var_kwargs['temperature'] = 0.6 + random.random() * 0.3
            var_kwargs['prefix_ratio'] = 0.2 + random.random() * 0.2
            
            generated = generate_variation(
                model, tokenizer, original_en, instruction, **var_kwargs
            )
            
            result = {
                'id': f'essay_{idx}_var{var_idx}',
                'original_ru': original_ru,
                'original_en': original_en,
                'generated_en': generated,
                'source': source,
                'year': year,
                'prompt_name': prompt_name,
                'variation_idx': var_idx,
            }
            results.append(result)
    
    result_df = pd.DataFrame(results)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Results saved to: {output_file}")
    
    print(f"\nSkipped rows (no translation): {skipped_count}")
    print(f"Processed rows: {len(df) - skipped_count}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate AI variations from already translated essays (no translation step)"
    )
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="/tmp/models/mistral_7b/last",
                        help="Path to trained model (Mistral-7B-Instruct-v0.2)")
    parser.add_argument("--use_8bit", action="store_true", default=True,
                        help="Use 8-bit quantization")
    
    # Input/Output arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file with already translated essays")
    parser.add_argument("--output", type=str, default="generated_variations.csv",
                        help="Output CSV for generated variations")
    parser.add_argument("--text_column_ru", type=str, default="Текст",
                        help="Column name containing Russian text")
    parser.add_argument("--text_column_en", type=str, default="Текст_en",
                        help="Column name containing English translation")
    
    # Generation arguments
    parser.add_argument("--num_variations", type=int, default=1,
                        help="Number of variations per essay")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum length of generated text")
    parser.add_argument("--prefix_ratio", type=float, default=0.6,
                        help="Ratio of original text to use as prefix")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top_p")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    
    # Processing
    parser.add_argument("--max_essays", type=int, default=-1,
                        help="Maximum essays to process (-1 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load input data
    print(f"Loading translated essays from: {args.input}")
    # Try to detect separator
    try:
        df = pd.read_csv(args.input, sep=';')
    except pd.errors.ParserError:
        df = pd.read_csv(args.input, sep=',')
    
    if args.max_essays > 0:
        df = df.head(args.max_essays)
    
    # Count valid translations
    valid_count = sum(1 for _, row in df.iterrows() 
                      if pd.notna(row.get(args.text_column_en, '')) 
                      and str(row.get(args.text_column_en, '')).strip() != '')
    
    print(f"Total essays: {len(df)}")
    print(f"Essays with valid English translation: {valid_count}")
    print(f"Processing {valid_count} essays for generation...")
    
    # Generate AI variations (skip translation step)
    print("\n" + "="*60)
    print("GENERATING AI VARIATIONS using r_clm model")
    print("="*60)
    
    model, tokenizer = load_model(args.model_path)
    
    generation_kwargs = {
        "max_length": args.max_length,
        "prefix_ratio": args.prefix_ratio,
        "repetition_penalty": args.repetition_penalty,
    }
    
    result_df = process_and_generate(
        df,
        model,
        tokenizer,
        text_column_ru=args.text_column_ru,
        text_column_en=args.text_column_en,
        output_file=args.output,
        num_variations=args.num_variations,
        **generation_kwargs,
    )
    
    # Print samples
    print("\n" + "="*80)
    print("GENERATED VARIATION SAMPLES")
    print("="*80)
    
    for idx, row in result_df.head(2).iterrows():
        print(f"\n--- {row['id']} ---")
        print(f"Original EN (first 200 chars):\n{row['original_en'][:200]}...")
        print(f"\nGenerated (first 300 chars):\n{row['generated_en'][:300]}...")
        print("-"*80)
    
    print(f"\nTotal variations generated: {len(result_df)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script for translating Russian essays to English and generating AI variations.
This script:
1. Reads Russian essays from CSV
2. Translates them to English using Google Translate
3. Generates AI variations using the trained r_clm model

Usage:
    python translate_and_generate.py --input datasets/Датасет.csv --output translated_essays.csv --model_path /tmp/models/r_clm_v2/last
"""

import argparse
import os
import random
import time
from typing import List, Optional, Dict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

# Try to import translation library
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False
    print("Warning: deep_translator not installed. Install with: pip install deep-translator")


def translate_text(text: str, source: str = 'ru', target: str = 'en', max_chars: int = 5000) -> str:
    """
    Translate text from source to target language.
    
    Args:
        text: Text to translate
        source: Source language code (e.g., 'ru')
        target: Target language code (e.g., 'en')
        max_chars: Maximum characters per translation chunk
    
    Returns:
        Translated text
    """
    if not HAS_TRANSLATOR:
        raise ImportError("deep_translator library not installed")
    
    translator = GoogleTranslator(source=source, target=target)
    
    # Split long texts into chunks
    if len(text) <= max_chars:
        return translator.translate(text)
    
    # Split by sentences for long texts
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    translated_parts = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if current_length + len(sentence) + 2 <= max_chars:
            current_chunk.append(sentence)
            current_length += len(sentence) + 2
        else:
            if current_chunk:
                chunk_text = '. '.join(current_chunk) + '.'
                translated = translator.translate(chunk_text)
                translated_parts.append(translated)
            current_chunk = [sentence]
            current_length = len(sentence)
    
    # Translate remaining chunk
    if current_chunk:
        chunk_text = '. '.join(current_chunk) + '.'
        translated = translator.translate(chunk_text)
        translated_parts.append(translated)
    
    return ' '.join(translated_parts)


def is_valid_translation(val) -> bool:
    """Check if a translation value is valid (not empty, not an error message)."""
    if pd.isna(val) or val == '':
        return False
    if 'Translation error' in str(val) or 'Request exception' in str(val):
        return False
    return True


def translate_dataframe(
    df: pd.DataFrame,
    text_column: str = 'Текст',
    source: str = 'ru',
    target: str = 'en',
    output_path: Optional[str] = None,
    batch_size: int = 10,
    max_retries: int = 5,
    retry_delay: float = 10.0,
    request_delay: float = 3.0,
) -> pd.DataFrame:
    """
    Translate all texts in a DataFrame column.
    
    Args:
        df: Input DataFrame
        text_column: Column containing texts to translate
        source: Source language
        target: Target language
        output_path: Optional path to save intermediate results
        batch_size: Save progress every N translations
        max_retries: Maximum retry attempts for failed translations
        retry_delay: Delay in seconds between retries
        request_delay: Delay between EVERY request to avoid rate limiting
    
    Returns:
        DataFrame with translated texts
    """
    if not HAS_TRANSLATOR:
        raise ImportError("deep_translator library not installed")
    
    # Create output column if it doesn't exist
    output_col = f'{text_column}_en'
    if output_col not in df.columns:
        df[output_col] = ''
    
    # Count already translated rows
    already_translated = sum(1 for idx, row in df.iterrows() if is_valid_translation(row.get(output_col, '')))
    print(f"Already translated: {already_translated}/{len(df)}")
    print(f"Using request_delay={request_delay}s between each translation to avoid rate limiting")
    
    # Track progress
    total = len(df)
    progress_bar = tqdm(total=total, desc="Translating", initial=already_translated)
    
    successful_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        # Skip if already has valid translation
        if is_valid_translation(row.get(output_col, '')):
            progress_bar.update(1)
            continue
        
        # Try to translate with retries
        text = row[text_column]
        if pd.isna(text) or not str(text).strip():
            df.at[idx, output_col] = ''
            progress_bar.update(1)
            continue
        
        # Always delay before each request to avoid rate limiting
        if successful_count > 0 or error_count > 0:
            time.sleep(request_delay)
        
        # Retry loop
        success = False
        for attempt in range(max_retries):
            try:
                translated = translate_text(str(text), source, target)
                df.at[idx, output_col] = translated
                success = True
                successful_count += 1
                break
            except Exception as e:
                print(f"Error translating row {idx} (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
        
        if not success:
            df.at[idx, output_col] = f"[Translation error: max retries exceeded]"
            error_count += 1
            print(f"  Row {idx} marked as error after {max_retries} attempts")
        
        progress_bar.update(1)
        
        # Save intermediate results
        if idx % batch_size == 0 and output_path:
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"\nProgress saved: {successful_count} successful, {error_count} errors")
    
    progress_bar.close()
    
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Translation saved to: {output_path}")
    
    # Print summary
    valid_count = sum(1 for idx, row in df.iterrows() if is_valid_translation(row.get(output_col, '')))
    error_count = sum(1 for idx, row in df.iterrows() if not is_valid_translation(row.get(output_col, '')) and pd.notna(row.get(text_column, '')) and str(row.get(text_column, '')).strip())
    print(f"Translation complete: {valid_count}/{len(df)} successful, {error_count} errors")
    
    return df


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


def load_model(model_path: str, use_8bit: bool = False):
    """Load the trained model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_8bit:
        # Use 8-bit quantization for lower memory usage
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=False,
        )
    else:
        # Use float32 for better stability (avoid NaN issues)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=False,
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
    Generate a variation of the original text using Mistral format.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        original_text: Original English text
        instruction: Instruction string with metadata
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repetition
        prefix_ratio: Ratio of original text to use as prefix
    
    Returns:
        Generated variation
    """
    # Take a prefix of the original text
    words = original_text.split()
    prefix_len = max(10, int(len(words) * prefix_ratio))
    prefix = " ".join(words[:prefix_len])
    
    # Create the full prompt using Mistral chat format
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
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating variations"):
        original_ru = row.get(text_column_ru, '')
        original_en = row.get(text_column_en, '')
        
        if not original_en or pd.isna(original_en):
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
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Translate Russian essays to English and generate AI variations"
    )
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="/tmp/models/r_clm_v2/last",
                        help="Path to trained r_clm model")
    parser.add_argument("--use_8bit", action="store_true", default=True,
                        help="Use 8-bit quantization")
    
    # Input/Output arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file with Russian essays")
    parser.add_argument("--translated_output", type=str, default="translated_essays.csv",
                        help="Output CSV for translated essays")
    parser.add_argument("--generated_output", type=str, default="generated_variations.csv",
                        help="Output CSV for generated variations")
    parser.add_argument("--text_column", type=str, default="Текст",
                        help="Column name containing Russian text")
    
    # Translation arguments
    parser.add_argument("--source_lang", type=str, default="ru",
                        help="Source language code")
    parser.add_argument("--target_lang", type=str, default="en",
                        help="Target language code")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Save translation progress every N essays")
    parser.add_argument("--skip_translation", action="store_true", default=False,
                        help="Skip translation step and use already translated file")
    
    # Generation arguments
    parser.add_argument("--num_variations", type=int, default=1,
                        help="Number of variations per essay")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum length of generated text")
    parser.add_argument("--prefix_ratio", type=float, default=0.3,
                        help="Ratio of original text to use as prefix")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty")
    
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
    print(f"Loading essays from: {args.input}")
    # Try to detect separator
    try:
        df = pd.read_csv(args.input, sep=';')
    except pd.errors.ParserError:
        df = pd.read_csv(args.input, sep=',')
    
    if args.max_essays > 0:
        df = df.head(args.max_essays)
    
    print(f"Processing {len(df)} essays...")
    
    # Step 1: Translate to English (or skip if already translated)
    print("\n" + "="*60)
    if args.skip_translation:
        print("STEP 1: SKIPPING TRANSLATION (using pre-translated file)")
        print("="*60)
        print(f"Using already translated file: {args.input}")
        translated_df = df
    else:
        print("STEP 1: Translating Russian essays to English")
        print("="*60)
        
        if not HAS_TRANSLATOR:
            print("ERROR: deep_translator not installed!")
            print("Please install: pip install deep-translator")
            return
        
        translated_df = translate_dataframe(
            df,
            text_column=args.text_column,
            source=args.source_lang,
            target=args.target_lang,
            output_path=args.translated_output,
            batch_size=args.batch_size,
        )
        
        print(f"\nTranslation complete! Saved to: {args.translated_output}")
        print(f"Sample translation:")
        print(f"  Original (first 100 chars): {translated_df.iloc[0][args.text_column][:100]}...")
        print(f"  Translated (first 100 chars): {translated_df.iloc[0][f'{args.text_column}_en'][:100]}...")
    
    # Step 2: Generate AI variations
    print("\n" + "="*60)
    print("STEP 2: Generating AI variations using r_clm model")
    print("="*60)
    
    model, tokenizer = load_model(args.model_path, use_8bit=args.use_8bit)
    
    generation_kwargs = {
        "max_length": args.max_length,
        "prefix_ratio": args.prefix_ratio,
        "repetition_penalty": args.repetition_penalty,
    }
    
    result_df = process_and_generate(
        translated_df,
        model,
        tokenizer,
        text_column_ru=args.text_column,
        text_column_en=f'{args.text_column}_en',
        output_file=args.generated_output,
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
    print(f"Saved to: {args.generated_output}")


if __name__ == "__main__":
    main()

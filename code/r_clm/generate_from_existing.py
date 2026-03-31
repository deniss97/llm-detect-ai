#!/usr/bin/env python3
"""
Script for generating AI essays by modifying existing texts.
This script takes existing essays and generates variations/rewrites using the trained CLM model.

Usage:
    python generate_from_existing.py --model_path /tmp/models/r_clm_v2/last --input existing_essays.csv --output modified_essays.csv
"""

import argparse
import os
import random
from typing import List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def get_instruction(
    prompt_name: str = "Unknown",
    task: str = "Writing",
    score: int = -1,
    grade_level: int = -1,
    ell_status: str = "Unknown",
    disability_status: str = "Unknown",
) -> str:
    """Create instruction string from metadata."""
    ret = f"""
Prompt: {prompt_name}
Task: {task}
Score: {score}
Student Grade Level: {grade_level}
English Language Learner: {ell_status}
Disability Status: {disability_status}
    """.strip()
    return ret


def format_prompt(instruction: str) -> str:
    """Format the full prompt for the model."""
    return f"### Instruction:\n{instruction}\n\n### Response: "


def load_model(model_path: str, use_8bit: bool = True):
    """Load the trained model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model.eval()
    print(f"Model loaded successfully!")
    return model, tokenizer


def modify_essay(
    model,
    tokenizer,
    existing_text: str,
    instruction: str,
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    prefix_ratio: float = 0.3,
) -> str:
    """
    Generate a modified version of an existing essay.
    
    This works by:
    1. Taking the instruction + the beginning of the existing essay as prompt
    2. Letting the model continue from there
    3. This creates a variation that follows the same style/theme
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        existing_text: The original essay text
        instruction: Instruction string with metadata
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repetition
        do_sample: Whether to use sampling
        prefix_ratio: Ratio of original text to use as prefix (0.1-0.5 recommended)
    
    Returns:
        Modified/continued essay text
    """
    # Take a prefix of the existing text
    words = existing_text.split()
    prefix_len = max(10, int(len(words) * prefix_ratio))
    prefix = " ".join(words[:prefix_len])
    
    # Create the full prompt
    base_prompt = format_prompt(instruction)
    full_prompt = f"{base_prompt}{prefix}"
    
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
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (after the prefix)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # Combine prefix with generated continuation
    full_text = f"{prefix} {generated_text}".strip()
    
    return full_text


def rewrite_essay(
    model,
    tokenizer,
    existing_text: str,
    instruction: str,
    max_length: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    do_sample: bool = True,
) -> str:
    """
    Rewrite an existing essay completely.
    
    This works by:
    1. Providing only the instruction (metadata)
    2. Letting the model generate a completely new essay
    3. The model has learned the style from training, so it will produce similar content
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        existing_text: The original essay (used for metadata extraction only)
        instruction: Instruction string with metadata
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher for more variation)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repetition
        do_sample: Whether to use sampling
    
    Returns:
        Completely rewritten essay
    """
    prompt = format_prompt(instruction)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return generated_text.strip()


def continue_essay(
    model,
    tokenizer,
    existing_text: str,
    instruction: str,
    continuation_ratio: float = 0.5,
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """
    Continue an existing essay from a certain point.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        existing_text: The original essay text
        instruction: Instruction string with metadata
        continuation_ratio: Ratio of text to keep before continuing (0.5 = continue from middle)
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repetition
        do_sample: Whether to use sampling
    
    Returns:
        Completed essay with continuation
    """
    # Split the text at the continuation point
    words = existing_text.split()
    split_point = int(len(words) * continuation_ratio)
    prefix = " ".join(words[:split_point])
    
    # Create prompt with instruction and prefix
    base_prompt = format_prompt(instruction)
    full_prompt = f"{base_prompt}{prefix}"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    # Calculate how many new tokens to generate
    max_new_tokens = max_length - input_length
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    full_text = f"{prefix} {generated_text}".strip()
    
    return full_text


def process_essays(
    model,
    tokenizer,
    df: pd.DataFrame,
    mode: str = "modify",
    output_file: Optional[str] = None,
    **generation_kwargs,
) -> pd.DataFrame:
    """
    Process a DataFrame of essays.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        df: DataFrame with essay data (must have 'text' column)
        mode: One of "modify", "rewrite", "continue"
        output_file: Optional path to save results
        **generation_kwargs: Arguments for generation functions
    
    Returns:
        DataFrame with original and generated texts
    """
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing essays ({mode})"):
        # Extract metadata
        text = row.get("text", row.get("full_text", ""))
        prompt_name = row.get("prompt_name", row.get("competition_set", "Unknown"))
        task = row.get("task", row.get("discourse_type", "Writing"))
        score = row.get("holistic_essay_score", row.get("score", -1))
        grade_level = row.get("grade_level", row.get("grade", -1))
        ell_status = row.get("ell_status", "Unknown")
        disability_status = row.get("student_disability_status", "Unknown")
        
        instruction = get_instruction(
            prompt_name=prompt_name,
            task=task,
            score=score,
            grade_level=grade_level,
            ell_status=ell_status,
            disability_status=disability_status,
        )
        
        if mode == "modify":
            generated_text = modify_essay(
                model, tokenizer, text, instruction, **generation_kwargs
            )
        elif mode == "rewrite":
            generated_text = rewrite_essay(
                model, tokenizer, text, instruction, **generation_kwargs
            )
        elif mode == "continue":
            generated_text = continue_essay(
                model, tokenizer, text, instruction, **generation_kwargs
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        result = {
            "original_text": text,
            "generated_text": generated_text,
            "mode": mode,
            "prompt_name": prompt_name,
            "task": task,
            "score": score,
            "grade_level": grade_level,
            "ell_status": ell_status,
            "disability_status": disability_status,
        }
        results.append(result)
    
    result_df = pd.DataFrame(results)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        result_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate essay variations from existing texts using trained CLM model"
    )
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="/tmp/models/r_clm_v2/last",
                        help="Path to trained model")
    parser.add_argument("--use_8bit", action="store_true", default=True,
                        help="Use 8-bit quantization")
    
    # Input/Output arguments
    parser.add_argument("--input", type=str, required=True,
                        help="Input CSV file with existing essays")
    parser.add_argument("--output", type=str, default="modified_essays.csv",
                        help="Output CSV file path")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing essay text")
    parser.add_argument("--num_essays", type=int, default=-1,
                        help="Number of essays to process (-1 = all)")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, default="modify",
                        choices=["modify", "rewrite", "continue"],
                        help="Generation mode: modify (partial rewrite), rewrite (full), continue")
    parser.add_argument("--prefix_ratio", type=float, default=0.3,
                        help="For modify mode: ratio of original text to keep as prefix")
    parser.add_argument("--continuation_ratio", type=float, default=0.5,
                        help="For continue mode: ratio of text before continuation point")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top_p")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load input data
    print(f"Loading essays from: {args.input}")
    df = pd.read_csv(args.input)
    
    if args.text_column not in df.columns:
        available = list(df.columns)
        raise ValueError(f"Column '{args.text_column}' not found. Available: {available}")
    
    if args.num_essays > 0:
        df = df.head(args.num_essays)
    
    print(f"Processing {len(df)} essays in '{args.mode}' mode...")
    
    # Load model
    model, tokenizer = load_model(args.model_path, use_8bit=args.use_8bit)
    
    # Process essays
    generation_kwargs = {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "prefix_ratio": args.prefix_ratio,
        "continuation_ratio": args.continuation_ratio,
    }
    
    result_df = process_essays(
        model, tokenizer, df,
        mode=args.mode,
        output_file=args.output,
        **generation_kwargs,
    )
    
    # Print samples
    print("\n" + "="*80)
    print("GENERATED ESSAY SAMPLES")
    print("="*80)
    
    for idx, row in result_df.head(2).iterrows():
        print(f"\n--- Essay {idx+1} ---")
        print(f"Mode: {args.mode}")
        print(f"Original (first 200 chars):\n{row['original_text'][:200]}...")
        print(f"\nGenerated (first 300 chars):\n{row['generated_text'][:300]}...")
        print("-"*80)
    
    print(f"\nTotal essays processed: {len(result_df)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script for generating AI essays using trained CLM model.
This script generates texts from scratch based on provided metadata.

Usage:
    python generate_text.py --model_path /tmp/models/r_clm_v2/last --output generated_essays.csv
"""

import argparse
import os
import random
from typing import List, Optional

import pandas as pd
import torch
from peft import PeftModel
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


def generate_essay(
    model,
    tokenizer,
    instruction: str,
    max_length: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """
    Generate an essay based on the instruction.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        instruction: Instruction string with metadata
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repetition
        do_sample: Whether to use sampling or greedy decoding
    
    Returns:
        Generated essay text
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


def generate_single_essay(
    model,
    tokenizer,
    prompt_name: str = "Argumentative Essay",
    task: str = "Argumentative Writing",
    score: int = 5,
    grade_level: int = 10,
    ell_status: str = "Non-ELL",
    disability_status: str = "No Disability",
    **generation_kwargs,
) -> dict:
    """Generate a single essay with specified metadata."""
    instruction = get_instruction(
        prompt_name=prompt_name,
        task=task,
        score=score,
        grade_level=grade_level,
        ell_status=ell_status,
        disability_status=disability_status,
    )
    
    essay = generate_essay(model, tokenizer, instruction, **generation_kwargs)
    
    return {
        "prompt_name": prompt_name,
        "task": task,
        "score": score,
        "grade_level": grade_level,
        "ell_status": ell_status,
        "disability_status": disability_status,
        "instruction": instruction,
        "generated_text": essay,
    }


def generate_batch_essays(
    model,
    tokenizer,
    metadata_list: List[dict],
    output_file: Optional[str] = None,
    **generation_kwargs,
) -> pd.DataFrame:
    """
    Generate multiple essays from a list of metadata dictionaries.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer
        metadata_list: List of dictionaries with metadata for each essay
        output_file: Optional path to save results as CSV
        **generation_kwargs: Arguments passed to generate_essay()
    
    Returns:
        DataFrame with generated essays
    """
    results = []
    
    for metadata in tqdm(metadata_list, desc="Generating essays"):
        result = generate_single_essay(
            model, tokenizer,
            prompt_name=metadata.get("prompt_name", "Unknown"),
            task=metadata.get("task", "Writing"),
            score=metadata.get("score", -1),
            grade_level=metadata.get("grade_level", -1),
            ell_status=metadata.get("ell_status", "Unknown"),
            disability_status=metadata.get("disability_status", "Unknown"),
            **generation_kwargs,
        )
        results.append(result)
    
    df = pd.DataFrame(results)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate AI essays using trained CLM model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="/tmp/models/r_clm_v2/last",
                        help="Path to trained model")
    parser.add_argument("--use_8bit", action="store_true", default=True,
                        help="Use 8-bit quantization")
    
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
    parser.add_argument("--num_essays", type=int, default=5,
                        help="Number of essays to generate")
    
    # Output arguments
    parser.add_argument("--output", type=str, default="generated_essays.csv",
                        help="Output CSV file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model
    model, tokenizer = load_model(args.model_path, use_8bit=args.use_8bit)
    
    # Generate sample prompts
    prompt_names = [
        "Argumentative Essay on Climate Change",
        "Persuasive Essay about Technology",
        "Narrative Essay about Personal Experience",
        "Expository Essay on Education",
        "Descriptive Essay about Nature",
    ]
    
    metadata_list = []
    for i in range(args.num_essays):
        metadata = {
            "prompt_name": prompt_names[i % len(prompt_names)],
            "task": random.choice(["Argumentative Writing", "Persuasive Writing", 
                                   "Narrative Writing", "Expository Writing"]),
            "score": random.randint(3, 6),
            "grade_level": random.randint(6, 12),
            "ell_status": random.choice(["Non-ELL", "ELL", "Unknown"]),
            "disability_status": random.choice(["No Disability", "Has Disability", "Unknown"]),
        }
        metadata_list.append(metadata)
    
    # Generate essays
    print(f"\nGenerating {args.num_essays} essays...")
    df = generate_batch_essays(
        model, tokenizer, metadata_list,
        output_file=args.output,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    
    # Print samples
    print("\n" + "="*80)
    print("GENERATED ESSAY SAMPLES")
    print("="*80)
    
    for idx, row in df.head(3).iterrows():
        print(f"\n--- Essay {idx+1} ---")
        print(f"Prompt: {row['prompt_name']}")
        print(f"Score: {row['score']}, Grade: {row['grade_level']}")
        print(f"Text:\n{row['generated_text'][:500]}...")
        print("-"*80)
    
    print(f"\nTotal essays generated: {len(df)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

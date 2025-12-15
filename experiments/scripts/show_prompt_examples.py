"""
Show examples of prompts and model outputs for analysis
"""

import sys
from pathlib import Path
import torch

# Add project root and src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dts_implementation.models.pytorch_hf_wrapper import load_recommended_model
from time_series_datasets.m4.M4QADataset import M4QADataset
from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

def print_separator(char='=', length=80):
    print(char * length)

def show_prompt_and_output(model, prompt_text, max_tokens=100):
    """Show prompt and model's generated output"""
    
    # Encode
    tokens = model.encode_text(prompt_text)
    
    print(f"\n📝 PROMPT ({len(prompt_text)} chars, {len(tokens)} tokens):")
    print("─" * 80)
    # Show first 500 chars
    preview = prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text
    print(preview)
    print("─" * 80)
    
    # Generate
    print(f"\n🤖 GENERATING (max {max_tokens} new tokens)...")
    tokens_tensor = torch.tensor([tokens]).to(model.device)
    
    with torch.no_grad():
        output_ids = model.model.generate(
            tokens_tensor,
            max_new_tokens=max_tokens,
            temperature=0.8,
            do_sample=True,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.eos_token_id
        )
    
    # Decode full output
    full_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove prompt)
    generated_only = full_text[len(prompt_text):] if full_text.startswith(prompt_text[:50]) else full_text
    
    print(f"\n💬 MODEL OUTPUT ({len(generated_only)} chars):")
    print("─" * 80)
    print(generated_only)
    print("─" * 80)
    
    return generated_only


def main():
    print("\n" + "="*80)
    print("  📊 PROMPT & OUTPUT EXAMPLES")
    print("="*80 + "\n")
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Device: {device}\n")
    
    # Load model (1B for quick test)
    print("Loading model...")
    model = load_recommended_model("1b-instruct", device=device)
    print()
    
    # Test 1: Simple prompt
    print_separator()
    print("  TEST 1: SIMPLE PROMPT")
    print_separator()
    
    simple_prompt = "The time series shows an upward trend. Based on this pattern, the forecast for the next period is"
    show_prompt_and_output(model, simple_prompt, max_tokens=50)
    
    # Test 2: M4 Dataset Sample
    print("\n\n" + "="*80)
    print("  TEST 2: REAL M4 DATASET SAMPLE")
    print("="*80)
    
    print("\nLoading M4 dataset...")
    m4_dataset = M4QADataset(
        split='test',
        EOS_TOKEN=model.tokenizer.eos_token,
        format_sample_str=False
    )
    
    sample = m4_dataset[0]
    # Check what keys are available
    print(f"Sample keys: {sample.keys()}")
    
    # Try different possible key names
    if 'input' in sample:
        prompt = sample['input']
        ground_truth = sample['output']
    elif 'text' in sample:
        prompt = sample['text']
        ground_truth = sample.get('answer', '')
    else:
        # Just use the whole sample as a string
        prompt = str(sample)
        ground_truth = ""
    
    print(f"✅ Loaded sample 0 from M4 dataset")
    
    output = show_prompt_and_output(model, prompt, max_tokens=150)
    
    print(f"\n💡 GROUND TRUTH ({len(ground_truth)} chars):")
    print("─" * 80)
    print(ground_truth)
    print("─" * 80)
    
    # Test 3: HAR Dataset Sample
    print("\n\n" + "="*80)
    print("  TEST 3: REAL HAR DATASET SAMPLE")
    print("="*80)
    
    print("\nLoading HAR dataset...")
    har_dataset = HARCoTQADataset(
        split='test',
        EOS_TOKEN=model.tokenizer.eos_token,
        format_sample_str=False
    )
    
    sample = har_dataset[0]
    prompt = sample['input']
    ground_truth = sample['output']
    
    print(f"✅ Loaded sample 0 from HAR dataset")
    
    output = show_prompt_and_output(model, prompt, max_tokens=200)
    
    print(f"\n💡 GROUND TRUTH ({len(ground_truth)} chars):")
    print("─" * 80)
    print(ground_truth[:500] + "..." if len(ground_truth) > 500 else ground_truth)
    print("─" * 80)
    
    # Analysis
    print("\n\n" + "="*80)
    print("  🔍 ANALYSIS")
    print("="*80 + "\n")
    
    print("Key observations:")
    print("  1. Check if outputs are coherent (not gibberish)")
    print("  2. Check if model understands time series context")
    print("  3. Compare generated vs ground truth quality")
    print("  4. Assess if instruction-tuning is working")
    print()
    
    print("Expected with REAL weights:")
    print("  ✅ Coherent sentences")
    print("  ✅ Relevant to the prompt")
    print("  ✅ Attempts to describe/analyze data")
    print("  ✅ May not be perfect, but understandable")
    print()
    
    print("If still seeing gibberish:")
    print("  ❌ Model weights not properly loaded")
    print("  ❌ Tokenization issues")
    print("  ❌ Need to debug further")
    print()
    
    print("="*80)
    print("  ✅ EXAMPLES COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()


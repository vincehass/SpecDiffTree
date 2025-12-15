"""
Test actual model outputs on real prompts
Compare model output vs. expected ground truth
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper
from time_series_datasets.m4.M4QADataset import M4QADataset
from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def test_model_on_prompt(model, dataset_name, prompt, ground_truth, max_tokens=150):
    """Test model on a single prompt and show output"""
    
    print(f"📊 Dataset: {dataset_name}")
    print(f"📝 Prompt Length: {len(prompt)} chars, ~{len(prompt.split())} words")
    print()
    
    # Show first 500 chars of prompt
    print("🔵 INPUT PROMPT (first 500 chars):")
    print("─" * 80)
    print(prompt[:500] + ("..." if len(prompt) > 500 else ""))
    print("─" * 80)
    print()
    
    # Encode prompt
    prompt_tokens = model.encode_text(prompt)
    if hasattr(prompt_tokens, 'shape'):
        num_tokens = prompt_tokens.shape[1] if len(prompt_tokens.shape) > 1 else prompt_tokens.shape[0]
    else:
        num_tokens = len(prompt_tokens)
    print(f"🔢 Tokenized to {num_tokens} tokens")
    print()
    
    # Generate with model
    print(f"🤖 GENERATING (max {max_tokens} new tokens)...")
    result = model.generate_sequence(prompt_tokens, max_tokens=max_tokens, temperature=0.8)
    model_output = result['text']
    print()
    
    # Extract just the generated part (remove prompt)
    if model_output.startswith(prompt):
        generated_only = model_output[len(prompt):]
    else:
        generated_only = model_output
    
    print("🟢 MODEL OUTPUT (generated text only):")
    print("─" * 80)
    print(generated_only)
    print("─" * 80)
    print(f"Generated: {len(generated_only)} chars, ~{len(generated_only.split())} words")
    print()
    
    print("🎯 EXPECTED OUTPUT (ground truth, first 500 chars):")
    print("─" * 80)
    print(ground_truth[:500] + ("..." if len(ground_truth) > 500 else ""))
    print("─" * 80)
    print()
    
    # Simple comparison
    print("📈 QUICK ANALYSIS:")
    if len(generated_only.strip()) < 10:
        print("  ⚠️  Output is very short!")
    else:
        print(f"  ✓ Generated {len(generated_only)} characters")
    
    # Check if it's coherent (not just repeated tokens)
    words = generated_only.split()
    unique_words = set(words)
    if len(words) > 0:
        diversity = len(unique_words) / len(words)
        print(f"  ✓ Word diversity: {diversity:.2%} ({len(unique_words)} unique / {len(words)} total)")
    
    print()


def main():
    print_section("🔬 TESTING MODEL OUTPUTS ON REAL PROMPTS")
    
    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🎮 Using CUDA GPU")
    else:
        device = "cpu"
        print("💻 Using CPU")
    
    print()
    
    # Load model
    print("Loading Llama 3.2 1B Instruct (PyTorch)...")
    model = PyTorchHFWrapper(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        device=device
    )
    print()
    
    # Test 1: M4 Dataset
    print_section("TEST 1: M4 TIME SERIES CAPTIONING")
    
    print("Loading M4QADataset...")
    m4_dataset = M4QADataset(
        split='test',
        EOS_TOKEN=model.tokenizer.eos_token,
        format_sample_str=False
    )
    print(f"✅ Loaded {len(m4_dataset)} samples\n")
    
    # Get first sample
    sample = m4_dataset[0]
    
    # Construct full prompt
    pre = sample.get('pre_prompt', '')
    ts_text = sample.get('time_series_text', '')
    post = sample.get('post_prompt', '')
    
    if isinstance(ts_text, list):
        ts_text = ' '.join(str(x) for x in ts_text)
    
    full_prompt = str(pre) + str(ts_text) + str(post)
    ground_truth = sample.get('answer', 'N/A')
    
    test_model_on_prompt(model, "M4 Competition", full_prompt, ground_truth, max_tokens=200)
    
    # Test 2: HAR Dataset  
    print_section("TEST 2: HAR ACTIVITY RECOGNITION (Chain-of-Thought)")
    
    print("Loading HARCoTQADataset...")
    har_dataset = HARCoTQADataset(
        split='test',
        EOS_TOKEN=model.tokenizer.eos_token,
        format_sample_str=False
    )
    print(f"✅ Loaded {len(har_dataset)} samples\n")
    
    # Get first sample
    sample = har_dataset[0]
    
    # Construct full prompt
    pre = sample.get('pre_prompt', '')
    ts_text = sample.get('time_series_text', '')
    post = sample.get('post_prompt', '')
    
    if isinstance(ts_text, list):
        ts_text = ' '.join(str(x) for x in ts_text)
    
    full_prompt = str(pre) + str(ts_text) + str(post)
    ground_truth = sample.get('answer', 'N/A')
    
    test_model_on_prompt(model, "HAR with CoT", full_prompt, ground_truth, max_tokens=200)
    
    print_section("✅ MODEL OUTPUT TESTING COMPLETE")
    
    print("Key Observations:")
    print("  1. Model receives REAL prompts with actual data")
    print("  2. Model generates actual completions (not random)")
    print("  3. Compare model output quality to ground truth")
    print("  4. This shows baseline model performance (no tree search)")
    print()
    print("Next: Run MaxEnt-TS tree search to improve outputs!")
    print()


if __name__ == "__main__":
    main()


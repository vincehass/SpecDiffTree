"""
Stages 2-3 Evaluation with PyTorch + HuggingFace Models

Uses:
1. PyTorch with proper HuggingFace model loading (REAL weights!)
2. Real M4 and HAR datasets
3. Mistral-7B-Instruct or Llama models
4. MaxEnt-TS tree search algorithm
"""

import sys
import json
from pathlib import Path
import time
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dts_implementation.search.maxent_ts import MaxEntTS
from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper, load_recommended_model
from dts_implementation.rewards.spectral_reward import SpectralReward
from src.time_series_datasets.m4.M4QADataset import M4QADataset
from src.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

print("\n" + "="*80)
print("  🚀 STAGES 2-3 WITH PYTORCH + HUGGINGFACE")
print("="*80 + "\n")

print("Advantages of this approach:")
print("  ✅ PyTorch with REAL model weights (not random!)")
print("  ✅ HuggingFace pre-trained models (Mistral-7B, Llama, etc.)")
print("  ✅ Real M4 and HAR datasets")
print("  ✅ MPS support for Apple Silicon")
print("  ✅ Proven to work - no MLX loading issues")
print()

# Configuration
NUM_SAMPLES_STAGE2 = 3  # M4 time series captioning
NUM_SAMPLES_STAGE3 = 3  # HAR activity recognition
NUM_ROLLOUTS = 5        # Tree search rollouts (5 for speed, 10-20 for quality)
EXPANSION_K = 3         # Top-k expansion (3 for speed, 4 for quality)

# Detect device
if torch.cuda.is_available():
    device = "cuda"
    print("🎮 Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = "mps"
    print("🍎 Using Apple Silicon (MPS)")
else:
    device = "cpu"
    print("💻 Using CPU")

print()

# Model selection
MODEL_OPTIONS = {
    "1": ("1b-instruct", "Llama 3.2 1B Instruct (fast, ~2GB RAM)"),
    "2": ("3b-instruct", "Llama 3.2 3B Instruct (balanced, ~6GB RAM)"),
    "3": ("7b-mistral", "Mistral 7B Instruct (best quality, ~14GB RAM)"),
}

print("="*80)
print("  📥 MODEL SELECTION")
print("="*80 + "\n")

print("Available models:")
for key, (model_key, description) in MODEL_OPTIONS.items():
    print(f"  {key}. {description}")

# Auto-select based on device and memory
if device == "mps":
    selected_model = "7b-mistral"  # Mac with MPS can handle 7B
    print(f"\n✅ Auto-selected: Mistral 7B (recommended for Apple Silicon)")
elif device == "cuda":
    selected_model = "7b-mistral"
    print(f"\n✅ Auto-selected: Mistral 7B (recommended for CUDA)")
else:
    selected_model = "3b-instruct"  # CPU: use smaller model
    print(f"\n✅ Auto-selected: Llama 3B (recommended for CPU)")

print()
sys.stdout.flush()

print("="*80)
print("  📥 LOADING MODEL")
print("="*80 + "\n")
sys.stdout.flush()

# Load model
model = load_recommended_model(selected_model, device=device)
print()
sys.stdout.flush()

# Initialize spectral reward
reward_fn = SpectralReward(
    freq_weight=0.5,
    temporal_weight=0.5,
    normalize=True
)

# Initialize searcher
searcher = MaxEntTS(
    model=model,
    reward_function=reward_fn,
    num_rollouts=NUM_ROLLOUTS,
    expansion_k=EXPANSION_K,
    max_depth=5,
    temperature=0.8,
    verbose=True
)

print("="*80)
print("  📊 STAGE 2: M4 TIME SERIES CAPTIONING")
print("="*80 + "\n")
sys.stdout.flush()

# Load Stage 2 dataset (REAL M4)
print("Loading M4QADataset (real M4 competition data)...")
stage2_dataset = M4QADataset(
    split='test',
    EOS_TOKEN=model.tokenizer.eos_token,
    format_sample_str=False
)

stage2_results = {
    'dataset': 'M4 Time Series Captioning (Real M4 Competition)',
    'model': f"{selected_model} (PyTorch)",
    'device': device,
    'num_rollouts': NUM_ROLLOUTS,
    'expansion_k': EXPANSION_K,
    'results': []
}

print(f"✅ Dataset loaded: {len(stage2_dataset)} samples")
print(f"Testing on {NUM_SAMPLES_STAGE2} samples...")
print()
sys.stdout.flush()

for i in range(NUM_SAMPLES_STAGE2):
    print(f"\n{'─'*80}")
    print(f"  Sample {i+1}/{NUM_SAMPLES_STAGE2}")
    print(f"{'─'*80}\n")
    sys.stdout.flush()
    
    # Get sample from real M4 dataset
    sample = stage2_dataset[i]
    prompt = sample['input']
    ground_truth = sample['output']
    
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Ground truth length: {len(ground_truth)} chars")
    sys.stdout.flush()
    
    # Tokenize
    prompt_tokens = model.encode_text(prompt)
    ground_truth_tokens = model.encode_text(ground_truth)
    
    print(f"Running search with {NUM_ROLLOUTS} rollouts...")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Run search
    result = searcher.search(
        prompt_tokens=prompt_tokens,
        ground_truth_tokens=ground_truth_tokens,
        max_new_tokens=200
    )
    
    elapsed = time.time() - start_time
    
    # Extract results
    generated_text = result['best_text']
    best_reward = result['best_reward']
    tree_stats = result['tree_stats']
    nodes = tree_stats['total_nodes']
    
    print(f"\n✅ Complete!")
    print(f"  • Nodes: {nodes}")
    print(f"  • Time: {elapsed:.2f}s")
    print(f"  • Reward: {best_reward:.4f}")
    print(f"  • Output length: {len(generated_text)} chars")
    
    # Show preview
    preview = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
    print(f"\n  Output preview:")
    print(f"  '{preview}'")
    sys.stdout.flush()
    
    # Save result
    stage2_results['results'].append({
        'sample_id': i,
        'prompt': prompt,
        'ground_truth': ground_truth,
        'generated_output': generated_text,
        'best_reward': float(best_reward),
        'nodes': int(nodes),
        'time': float(elapsed),
        'tree_stats': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                      for k, v in tree_stats.items()}
    })

# Calculate Stage 2 averages
stage2_results['avg_nodes'] = np.mean([r['nodes'] for r in stage2_results['results']])
stage2_results['avg_time'] = np.mean([r['time'] for r in stage2_results['results']])
stage2_results['avg_reward'] = np.mean([r['best_reward'] for r in stage2_results['results']])

print("\n" + "="*80)
print(f"  STAGE 2 COMPLETE - Avg Nodes: {stage2_results['avg_nodes']:.1f}, "
      f"Avg Time: {stage2_results['avg_time']:.2f}s")
print("="*80 + "\n")
sys.stdout.flush()

print("="*80)
print("  📊 STAGE 3: HUMAN ACTIVITY RECOGNITION CoT")
print("="*80 + "\n")
sys.stdout.flush()

# Load Stage 3 dataset (REAL HAR)
print("Loading HARCoTQADataset (real HAR accelerometer data with CoT)...")
stage3_dataset = HARCoTQADataset(
    split='test',
    EOS_TOKEN=model.tokenizer.eos_token,
    format_sample_str=False
)

stage3_results = {
    'dataset': 'HAR with Chain-of-Thought (Real Accelerometer Data)',
    'model': f"{selected_model} (PyTorch)",
    'device': device,
    'num_rollouts': NUM_ROLLOUTS,
    'expansion_k': EXPANSION_K,
    'results': []
}

print(f"✅ Dataset loaded: {len(stage3_dataset)} samples")
print(f"Testing on {NUM_SAMPLES_STAGE3} samples...")
print()
sys.stdout.flush()

for i in range(NUM_SAMPLES_STAGE3):
    print(f"\n{'─'*80}")
    print(f"  Sample {i+1}/{NUM_SAMPLES_STAGE3}")
    print(f"{'─'*80}\n")
    sys.stdout.flush()
    
    # Get sample from real HAR dataset
    sample = stage3_dataset[i]
    prompt = sample['input']
    ground_truth = sample['output']
    
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Ground truth length: {len(ground_truth)} chars")
    sys.stdout.flush()
    
    # Tokenize
    prompt_tokens = model.encode_text(prompt)
    ground_truth_tokens = model.encode_text(ground_truth)
    
    print(f"Running search with {NUM_ROLLOUTS} rollouts...")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Run search
    result = searcher.search(
        prompt_tokens=prompt_tokens,
        ground_truth_tokens=ground_truth_tokens,
        max_new_tokens=300
    )
    
    elapsed = time.time() - start_time
    
    # Extract results
    generated_text = result['best_text']
    best_reward = result['best_reward']
    tree_stats = result['tree_stats']
    nodes = tree_stats['total_nodes']
    
    print(f"\n✅ Complete!")
    print(f"  • Nodes: {nodes}")
    print(f"  • Time: {elapsed:.2f}s")
    print(f"  • Reward: {best_reward:.4f}")
    print(f"  • Output length: {len(generated_text)} chars")
    
    # Show preview
    preview = generated_text[:200] + "..." if len(generated_text) > 200 else generated_text
    print(f"\n  Output preview:")
    print(f"  '{preview}'")
    sys.stdout.flush()
    
    # Save result
    stage3_results['results'].append({
        'sample_id': i,
        'prompt': prompt,
        'ground_truth': ground_truth,
        'generated_output': generated_text,
        'best_reward': float(best_reward),
        'nodes': int(nodes),
        'time': float(elapsed),
        'tree_stats': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                      for k, v in tree_stats.items()}
    })

# Calculate Stage 3 averages
stage3_results['avg_nodes'] = np.mean([r['nodes'] for r in stage3_results['results']])
stage3_results['avg_time'] = np.mean([r['time'] for r in stage3_results['results']])
stage3_results['avg_reward'] = np.mean([r['best_reward'] for r in stage3_results['results']])

print("\n" + "="*80)
print(f"  STAGE 3 COMPLETE - Avg Nodes: {stage3_results['avg_nodes']:.1f}, "
      f"Avg Time: {stage3_results['avg_time']:.2f}s")
print("="*80 + "\n")
sys.stdout.flush()

# Save all results
output_dir = Path("evaluation/results")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "stages_2_3_PYTORCH.json"

results = {
    'stage2': stage2_results,
    'stage3': stage3_results,
    'system_info': {
        'device': device,
        'model': selected_model,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available()
    }
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print("  ✅ EVALUATION COMPLETE WITH PYTORCH!")
print("="*80 + "\n")

print(f"Results saved to: {output_file}")
print()
print("SUMMARY:")
print(f"  Stage 2 (M4): {stage2_results['avg_nodes']:.1f} nodes, {stage2_results['avg_time']:.2f}s avg")
print(f"  Stage 3 (HAR): {stage3_results['avg_nodes']:.1f} nodes, {stage3_results['avg_time']:.2f}s avg")
print()
print("Key improvements over previous attempts:")
print("  ✅ REAL model weights loaded (not random!)")
print("  ✅ Coherent text generation expected")
print("  ✅ Proper time series understanding")
print("  ✅ Works on CPU, CUDA, and MPS")
print()
print("View detailed results:")
print(f"  python view_detailed_results.py")
print()
print("="*80 + "\n")

"""
Stages 2-3 Evaluation with BETTER MODEL (Mistral-7B, non-4bit)

Key improvements:
1. Uses REAL M4 competition dataset for Stage 2 (not simulation)
2. Uses REAL HAR accelerometer dataset for Stage 3 (not simulation)
3. Uses Mistral-7B-Instruct (7B params, non-4bit) for quality outputs
4. Instruction-tuned model capable of proper Q&A reasoning
"""

import sys
import json
from pathlib import Path
import time
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dts_implementation.search.maxent_ts import MaxEntTS
from dts_implementation.models.mlx_direct_loader import SimplifiedMLXWrapper
from dts_implementation.rewards.spectral_reward import SpectralReward
from src.time_series_datasets.m4.M4QADataset import M4QADataset
from src.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

print("\n" + "="*80)
print("  🚀 STAGES 2-3 WITH BETTER MODEL (Mistral-7B)")
print("="*80 + "\n")

print("Key improvements over previous run:")
print("  ✅ Mistral-7B-Instruct (vs Llama 1B)")
print("  ✅ 7× more parameters (7B vs 1B)")
print("  ✅ NOT 4-bit quantized (likely fp16 full precision)")
print("  ✅ Instruction-tuned for Q&A tasks")
print("  ✅ Should generate coherent, meaningful text")
print()

# Configuration
NUM_SAMPLES_STAGE2 = 3  # M4 time series captioning
NUM_SAMPLES_STAGE3 = 3  # HAR activity recognition
NUM_ROLLOUTS = 5        # Tree search rollouts
EXPANSION_K = 3         # Top-k expansion

print("="*80)
print("  📥 LOADING BETTER MODEL (This will take a moment...)")
print("="*80 + "\n")
sys.stdout.flush()

# Load better model
model_id = "mlx-community/Mistral-7B-Instruct-v0.2"
print(f"Model: {model_id}")
print(f"Size: ~7B parameters (~14GB RAM)")
print(f"Quality: Much better than 4-bit 1B model")
print()
sys.stdout.flush()

model = SimplifiedMLXWrapper(model_id=model_id)
print()
print("✅ Model loaded successfully!")
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
print("  📊 STAGE 2: M4 TIME SERIES CAPTIONING (Real Competition Data)")
print("="*80 + "\n")
sys.stdout.flush()

# Load Stage 2 dataset (REAL M4 DATASET)
print("Loading M4QADataset (real M4 competition data)...")
stage2_dataset = M4QADataset(
    split='test',
    EOS_TOKEN=model.tokenizer.eos_token,
    format_sample_str=False
)

stage2_results = {
    'dataset': 'M4 Time Series Captioning (Real M4 Competition)',
    'model': model_id,
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
    prompt = sample['input']  # M4QADataset uses 'input' and 'output'
    ground_truth = sample['output']
    
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Ground truth length: {len(ground_truth)} chars")
    sys.stdout.flush()
    
    # Tokenize prompt
    prompt_tokens = model.encode_text(prompt)
    ground_truth_tokens = model.encode_text(ground_truth)
    
    print(f"Running search with {NUM_ROLLOUTS} rollouts...")
    sys.stdout.flush()
    
    start_time = time.time()
    
    # Run search
    result = searcher.search(
        prompt_tokens=prompt_tokens,
        ground_truth_tokens=ground_truth_tokens,
        max_new_tokens=200  # Allow longer outputs for captions
    )
    
    elapsed = time.time() - start_time
    
    # Extract results correctly
    generated_text = result['best_text']
    best_reward = result['best_reward']
    tree_stats = result['tree_stats']
    nodes = tree_stats['total_nodes']
    
    print(f"\n✅ Complete!")
    print(f"  • Nodes: {nodes}")
    print(f"  • Time: {elapsed:.2f}s")
    print(f"  • Reward: {best_reward:.4f}")
    print(f"  • Output length: {len(generated_text)} chars")
    
    # Show first 200 chars of output
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
print("  📊 STAGE 3: HUMAN ACTIVITY RECOGNITION CoT (Real Accelerometer Data)")
print("="*80 + "\n")
sys.stdout.flush()

# Load Stage 3 dataset (REAL HAR DATASET with Chain-of-Thought)
print("Loading HARCoTQADataset (real HAR accelerometer data with CoT)...")
stage3_dataset = HARCoTQADataset(
    split='test',
    EOS_TOKEN=model.tokenizer.eos_token,
    format_sample_str=False
)

stage3_results = {
    'dataset': 'HAR with Chain-of-Thought (Real Accelerometer Data)',
    'model': model_id,
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
    prompt = sample['input']  # HARCoTQADataset uses 'input' and 'output'
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
        max_new_tokens=300  # Allow longer outputs for reasoning
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
    
    # Show first 200 chars of output
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
output_file = output_dir / "stages_2_3_BETTER_MODEL.json"

results = {
    'stage2': stage2_results,
    'stage3': stage3_results,
    'model_info': {
        'model_id': model_id,
        'size': '7B parameters',
        'quantization': 'fp16 (likely)',
        'improvement_over_previous': '7× parameters, full precision'
    }
}

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print("  ✅ EVALUATION COMPLETE WITH BETTER MODEL!")
print("="*80 + "\n")

print(f"Results saved to: {output_file}")
print()
print("SUMMARY:")
print(f"  Stage 2 (M4): {stage2_results['avg_nodes']:.1f} nodes, {stage2_results['avg_time']:.2f}s avg")
print(f"  Stage 3 (HAR): {stage3_results['avg_nodes']:.1f} nodes, {stage3_results['avg_time']:.2f}s avg")
print()
print("Expected improvements:")
print("  ✅ Coherent, meaningful text (not gibberish)")
print("  ✅ Better reasoning for HAR classification")
print("  ✅ Proper time series descriptions for M4")
print("  ✅ Actual Q&A capabilities")
print()
print("View detailed results:")
print(f"  python view_detailed_results.py")
print()
print("="*80 + "\n")


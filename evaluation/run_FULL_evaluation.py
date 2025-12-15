"""
FULL EVALUATION: Stages 2 & 3 with MaxEnt-TS

This script runs a comprehensive evaluation with:
- More samples (10 per stage)
- More rollouts (30 per sample)
- Real M4 and HAR datasets
- Detailed model output vs expected output comparison
- Performance statistics and visualizations
"""

import sys
import json
from pathlib import Path
import time
import numpy as np
import torch
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig
from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper
from dts_implementation.rewards.spectral_reward import SpectralReward
from time_series_datasets.m4.M4QADataset import M4QADataset
from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

# ========================================
# CONFIGURATION
# ========================================

NUM_SAMPLES_PER_STAGE = 10  # More samples for better statistics
NUM_ROLLOUTS = 30           # More rollouts for better search
EXPANSION_K = 4             # Top-k expansion
MAX_NEW_TOKENS = 250        # Max tokens to generate

print("\n" + "="*80)
print("  🚀 FULL EVALUATION: MaxEnt-TS on Stages 2 & 3")
print("="*80 + "\n")

print("Configuration:")
print(f"  • Samples per stage: {NUM_SAMPLES_PER_STAGE}")
print(f"  • Rollouts per sample: {NUM_ROLLOUTS}")
print(f"  • Expansion K: {EXPANSION_K}")
print(f"  • Max new tokens: {MAX_NEW_TOKENS}")
print()

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
sys.stdout.flush()

# ========================================
# LOAD MODEL
# ========================================

print("="*80)
print("  📥 LOADING MODEL")
print("="*80 + "\n")

# Load Llama 3.2 1B (fast and effective) using PyTorch wrapper
model = PyTorchHFWrapper(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    device=device
)
print("✅ Model loaded successfully")
print()
sys.stdout.flush()

# ========================================
# INITIALIZE SEARCHER
# ========================================

print("="*80)
print("  🔬 INITIALIZING MaxEnt-TS")
print("="*80 + "\n")

# Create dummy reference time series for reward
reference_ts = np.sin(np.linspace(0, 10 * np.pi, 256)) + 0.1 * np.random.randn(256)
reward = SpectralReward(gamma=1.0)
reward.set_context(reference_ts)

# Configure MaxEnt-TS
config = MaxEntTSConfig(
    num_rollouts=NUM_ROLLOUTS,
    temperature=1.0,
    expansion_k=EXPANSION_K,
    max_seq_length=512,
    verbose=False  # Turn off per-rollout logging for cleaner output
)

searcher = None  # Will create fresh for each search

print("✅ MaxEnt-TS configured")
print()
sys.stdout.flush()

# ========================================
# STAGE 2: M4 TIME SERIES CAPTIONING
# ========================================

print("="*80)
print("  📊 STAGE 2: M4 TIME SERIES CAPTIONING")
print("="*80 + "\n")

# Load M4 dataset
print("Loading M4QADataset...")
m4_dataset = M4QADataset(
    split='test',
    EOS_TOKEN=model.tokenizer.eos_token,
    format_sample_str=False
)
print(f"✅ Loaded {len(m4_dataset)} M4 test samples")
print(f"   Testing on first {NUM_SAMPLES_PER_STAGE} samples...")
print()
sys.stdout.flush()

stage2_results = {
    'stage': 'Stage 2: M4 Time Series Captioning',
    'dataset': 'M4 Competition',
    'model': f"Llama 3.2 1B ({device})",
    'num_samples': NUM_SAMPLES_PER_STAGE,
    'num_rollouts': NUM_ROLLOUTS,
    'expansion_k': EXPANSION_K,
    'results': []
}

total_nodes_s2 = 0
total_time_s2 = 0

for i in range(NUM_SAMPLES_PER_STAGE):
    print(f"{'─'*80}")
    print(f"  Sample {i+1}/{NUM_SAMPLES_PER_STAGE}")
    print(f"{'─'*80}\n")
    sys.stdout.flush()
    
    try:
        # Get sample
        sample = m4_dataset[i]
        prompt_text = sample.get('input', sample.get('prompt', ''))
        expected_text = sample.get('output', sample.get('answer', ''))
        
        # Extract time series data if available
        ts_mean = sample.get('pre_prompt', '').split('mean ')[-1].split(' and')[0] if 'mean' in sample.get('pre_prompt', '') else 'N/A'
        ts_std = sample.get('pre_prompt', '').split('std ')[-1].split(':')[0] if 'std' in sample.get('pre_prompt', '') else 'N/A'
        
        print(f"Time series stats: mean={ts_mean}, std={ts_std}")
        print(f"Prompt length: {len(prompt_text)} chars")
        print(f"Expected output length: {len(expected_text)} chars")
        sys.stdout.flush()
        
        # Tokenize
        prompt_tokens = model.tokenizer.encode(prompt_text, add_special_tokens=True)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        print(f"Tokenized: {len(prompt_tokens)} tokens")
        print(f"Running MaxEnt-TS search with {NUM_ROLLOUTS} rollouts...")
        sys.stdout.flush()
        
        # Run search
        start_time = time.time()
        searcher = MaxEntTS(model, reward, config)
        best_node = searcher.search(prompt_tensor)
        tree_stats = searcher._get_tree_stats()
        elapsed = time.time() - start_time
        
        # Extract generated text
        if hasattr(best_node, 'token_ids'):
            token_ids = best_node.token_ids
            if isinstance(token_ids, torch.Tensor):
                if token_ids.dim() == 2:
                    token_ids = token_ids[0]
                token_ids = token_ids.cpu().tolist()
            generated_text = model.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            generated_text = "<error: no output>"
        
        nodes = tree_stats.get('total_nodes', tree_stats.get('nodes_explored', 0))
        total_nodes_s2 += nodes
        total_time_s2 += elapsed
        
        print(f"\n✅ Search complete!")
        print(f"  • Nodes explored: {nodes}")
        print(f"  • Time: {elapsed:.1f}s")
        print(f"  • Output length: {len(generated_text)} chars")
        print(f"\n  Model output preview: {generated_text[:150]}...")
        print(f"\n  Expected output preview: {expected_text[:150]}...")
        sys.stdout.flush()
        
        # Save result
        stage2_results['results'].append({
            'sample_id': i,
            'prompt': prompt_text,
            'expected_output': expected_text,
            'model_output': generated_text,
            'time_series_mean': ts_mean,
            'time_series_std': ts_std,
            'nodes_explored': int(nodes),
            'search_time': float(elapsed),
            'output_length': len(generated_text),
            'expected_length': len(expected_text),
            'tree_stats': {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
                          for k, v in tree_stats.items()}
        })
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        stage2_results['results'].append({
            'sample_id': i,
            'error': str(e),
            'nodes_explored': 0,
            'search_time': 0.0
        })
    
    print()

# Stage 2 summary
avg_nodes_s2 = total_nodes_s2 / NUM_SAMPLES_PER_STAGE
avg_time_s2 = total_time_s2 / NUM_SAMPLES_PER_STAGE

stage2_results['avg_nodes'] = float(avg_nodes_s2)
stage2_results['avg_time'] = float(avg_time_s2)
stage2_results['total_time'] = float(total_time_s2)

print("="*80)
print(f"  STAGE 2 COMPLETE")
print("="*80)
print(f"  • Avg nodes: {avg_nodes_s2:.1f}")
print(f"  • Avg time: {avg_time_s2:.1f}s")
print(f"  • Total time: {total_time_s2:.1f}s ({total_time_s2/60:.1f} min)")
print("="*80 + "\n")
sys.stdout.flush()

# ========================================
# STAGE 3: HAR ACTIVITY RECOGNITION
# ========================================

print("="*80)
print("  🏃 STAGE 3: HAR ACTIVITY RECOGNITION (Chain-of-Thought)")
print("="*80 + "\n")

# Load HAR dataset
print("Loading HARCoTQADataset...")
har_dataset = HARCoTQADataset(
    split='test',
    EOS_TOKEN=model.tokenizer.eos_token,
    format_sample_str=False
)
print(f"✅ Loaded {len(har_dataset)} HAR test samples")
print(f"   Testing on first {NUM_SAMPLES_PER_STAGE} samples...")
print()
sys.stdout.flush()

stage3_results = {
    'stage': 'Stage 3: HAR Activity Recognition with CoT',
    'dataset': 'HAR with Chain-of-Thought',
    'model': f"Llama 3.2 1B ({device})",
    'num_samples': NUM_SAMPLES_PER_STAGE,
    'num_rollouts': NUM_ROLLOUTS,
    'expansion_k': EXPANSION_K,
    'results': []
}

total_nodes_s3 = 0
total_time_s3 = 0

for i in range(NUM_SAMPLES_PER_STAGE):
    print(f"{'─'*80}")
    print(f"  Sample {i+1}/{NUM_SAMPLES_PER_STAGE}")
    print(f"{'─'*80}\n")
    sys.stdout.flush()
    
    try:
        # Get sample
        sample = har_dataset[i]
        prompt_text = sample.get('input', sample.get('prompt', ''))
        expected_text = sample.get('output', sample.get('answer', ''))
        
        print(f"Prompt length: {len(prompt_text)} chars")
        print(f"Expected output length: {len(expected_text)} chars")
        sys.stdout.flush()
        
        # Tokenize
        prompt_tokens = model.tokenizer.encode(prompt_text, add_special_tokens=True)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        print(f"Tokenized: {len(prompt_tokens)} tokens")
        print(f"Running MaxEnt-TS search with {NUM_ROLLOUTS} rollouts...")
        sys.stdout.flush()
        
        # Run search
        start_time = time.time()
        searcher = MaxEntTS(model, reward, config)
        best_node = searcher.search(prompt_tensor)
        tree_stats = searcher._get_tree_stats()
        elapsed = time.time() - start_time
        
        # Extract generated text
        if hasattr(best_node, 'token_ids'):
            token_ids = best_node.token_ids
            if isinstance(token_ids, torch.Tensor):
                if token_ids.dim() == 2:
                    token_ids = token_ids[0]
                token_ids = token_ids.cpu().tolist()
            generated_text = model.tokenizer.decode(token_ids, skip_special_tokens=True)
        else:
            generated_text = "<error: no output>"
        
        nodes = tree_stats.get('total_nodes', tree_stats.get('nodes_explored', 0))
        total_nodes_s3 += nodes
        total_time_s3 += elapsed
        
        # Extract activity classification from expected and model output
        expected_activity = 'N/A'
        model_activity = 'N/A'
        
        if 'Answer:' in expected_text:
            expected_activity = expected_text.split('Answer:')[-1].strip().split()[0]
        if 'Answer:' in generated_text:
            model_activity = generated_text.split('Answer:')[-1].strip().split()[0]
        
        correct = (expected_activity != 'N/A' and model_activity == expected_activity)
        
        print(f"\n✅ Search complete!")
        print(f"  • Nodes explored: {nodes}")
        print(f"  • Time: {elapsed:.1f}s")
        print(f"  • Expected activity: {expected_activity}")
        print(f"  • Model predicted: {model_activity}")
        print(f"  • Correct: {'✓' if correct else '✗'}")
        print(f"\n  Model output preview: {generated_text[:150]}...")
        print(f"\n  Expected output preview: {expected_text[:150]}...")
        sys.stdout.flush()
        
        # Save result
        stage3_results['results'].append({
            'sample_id': i,
            'prompt': prompt_text,
            'expected_output': expected_text,
            'model_output': generated_text,
            'expected_activity': expected_activity,
            'model_activity': model_activity,
            'correct': correct,
            'nodes_explored': int(nodes),
            'search_time': float(elapsed),
            'output_length': len(generated_text),
            'expected_length': len(expected_text),
            'tree_stats': {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
                          for k, v in tree_stats.items()}
        })
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        stage3_results['results'].append({
            'sample_id': i,
            'error': str(e),
            'nodes_explored': 0,
            'search_time': 0.0
        })
    
    print()

# Stage 3 summary
avg_nodes_s3 = total_nodes_s3 / NUM_SAMPLES_PER_STAGE
avg_time_s3 = total_time_s3 / NUM_SAMPLES_PER_STAGE
correct_count = sum(1 for r in stage3_results['results'] if r.get('correct', False))
accuracy = correct_count / NUM_SAMPLES_PER_STAGE

stage3_results['avg_nodes'] = float(avg_nodes_s3)
stage3_results['avg_time'] = float(avg_time_s3)
stage3_results['total_time'] = float(total_time_s3)
stage3_results['accuracy'] = float(accuracy)
stage3_results['correct_count'] = correct_count

print("="*80)
print(f"  STAGE 3 COMPLETE")
print("="*80)
print(f"  • Avg nodes: {avg_nodes_s3:.1f}")
print(f"  • Avg time: {avg_time_s3:.1f}s")
print(f"  • Total time: {total_time_s3:.1f}s ({total_time_s3/60:.1f} min)")
print(f"  • Accuracy: {accuracy:.1%} ({correct_count}/{NUM_SAMPLES_PER_STAGE})")
print("="*80 + "\n")
sys.stdout.flush()

# ========================================
# SAVE RESULTS
# ========================================

print("="*80)
print("  💾 SAVING RESULTS")
print("="*80 + "\n")

output_dir = Path("evaluation/results")
output_dir.mkdir(parents=True, exist_ok=True)

# Save individual stage results
s2_file = output_dir / "FULL_stage2_results.json"
s3_file = output_dir / "FULL_stage3_results.json"

with open(s2_file, 'w') as f:
    json.dump(stage2_results, f, indent=2)
print(f"✅ Stage 2 results: {s2_file}")

with open(s3_file, 'w') as f:
    json.dump(stage3_results, f, indent=2)
print(f"✅ Stage 3 results: {s3_file}")

# Save combined summary
combined_results = {
    'evaluation_type': 'Full MaxEnt-TS Evaluation',
    'timestamp': datetime.now().isoformat(),
    'configuration': {
        'model': f"Llama 3.2 1B ({device})",
        'num_samples_per_stage': NUM_SAMPLES_PER_STAGE,
        'num_rollouts': NUM_ROLLOUTS,
        'expansion_k': EXPANSION_K,
        'max_new_tokens': MAX_NEW_TOKENS
    },
    'stage2': stage2_results,
    'stage3': stage3_results,
    'overall': {
        'total_samples': NUM_SAMPLES_PER_STAGE * 2,
        'total_nodes': total_nodes_s2 + total_nodes_s3,
        'total_time': total_time_s2 + total_time_s3,
        'avg_nodes_per_sample': (total_nodes_s2 + total_nodes_s3) / (NUM_SAMPLES_PER_STAGE * 2),
        'avg_time_per_sample': (total_time_s2 + total_time_s3) / (NUM_SAMPLES_PER_STAGE * 2)
    }
}

combined_file = output_dir / "FULL_evaluation_results.json"
with open(combined_file, 'w') as f:
    json.dump(combined_results, f, indent=2)
print(f"✅ Combined results: {combined_file}")

print()
print("="*80)
print("  ✅ FULL EVALUATION COMPLETE")
print("="*80 + "\n")

# Final summary
total_time_all = total_time_s2 + total_time_s3
print("FINAL SUMMARY:")
print(f"  • Total samples evaluated: {NUM_SAMPLES_PER_STAGE * 2}")
print(f"  • Total rollouts: {NUM_ROLLOUTS * NUM_SAMPLES_PER_STAGE * 2}")
print(f"  • Total nodes explored: {total_nodes_s2 + total_nodes_s3:.0f}")
print(f"  • Total time: {total_time_all:.1f}s ({total_time_all/60:.1f} minutes)")
print()
print(f"Stage 2 (M4 Captioning):")
print(f"  • {NUM_SAMPLES_PER_STAGE} samples")
print(f"  • Avg: {avg_nodes_s2:.1f} nodes, {avg_time_s2:.1f}s per sample")
print()
print(f"Stage 3 (HAR with CoT):")
print(f"  • {NUM_SAMPLES_PER_STAGE} samples")
print(f"  • Avg: {avg_nodes_s3:.1f} nodes, {avg_time_s3:.1f}s per sample")
print(f"  • Accuracy: {accuracy:.1%} ({correct_count}/{NUM_SAMPLES_PER_STAGE})")
print()
print("="*80 + "\n")

print("📊 Next steps:")
print("  1. Run: python analyze_full_evaluation.py")
print("  2. View detailed results and comparisons")
print("  3. See model outputs vs. expected outputs")
print()


"""
Comprehensive Comparison of All Methods

Compares:
1. Greedy Baseline (no search)
2. MCTS (Standard Monte Carlo Tree Search)
3. DTS (Diffusion Tree Sampling with Soft Bellman)
4. DTS* (DTS with UCT selection - greedy)
5. MaxEnt-TS (Our implementation)

This will help identify where our implementation differs from the paper.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "baselines"))

import torch
import time
import numpy as np
from typing import Dict, List

# Import all methods
from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper
from dts_implementation.rewards.spectral_reward import SpectralReward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig

from mcts_baseline import MCTSTextGenerator, MCTSConfig
from dts_baseline import DTSTextGenerator, DTSStarTextGenerator, DTSConfig

# Import datasets
from time_series_datasets.m4.M4QADataset import M4QADataset


print("\n" + "="*80)
print("  🔬 COMPREHENSIVE METHOD COMPARISON")
print("="*80 + "\n")

# Configuration
NUM_SAMPLES = 3  # Test on 3 samples
NUM_ROLLOUTS = 20  # Fair comparison - same compute budget

print("Configuration:")
print(f"  • Samples: {NUM_SAMPLES}")
print(f"  • Rollouts/Simulations: {NUM_ROLLOUTS}")
print()

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
print("📥 Loading model...")
model = PyTorchHFWrapper(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    device=device
)
print("✅ Model loaded")
print()

# Load dataset
print("📊 Loading M4 dataset...")
dataset = M4QADataset(
    split='test',
    EOS_TOKEN=model.tokenizer.eos_token,
    format_sample_str=False
)
print(f"✅ Dataset loaded: {len(dataset)} samples")
print()

# Create reward function
reference_ts = np.sin(np.linspace(0, 10 * np.pi, 256))
reward_fn_obj = SpectralReward(gamma=1.0)
reward_fn_obj.set_context(reference_ts)

def reward_fn(tokens: torch.Tensor) -> float:
    """Wrapper for reward function"""
    try:
        # Decode tokens
        if tokens.dim() == 2:
            tokens = tokens[0]
        text = model.tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
        
        # Simple reward: length + coherence heuristic
        reward = len(text) / 100.0  # Reward for generating text
        
        # Bonus for completing generation
        if model.eos_token_id in tokens.tolist():
            reward += 1.0
        
        return reward
    except:
        return 0.0


# Initialize all methods
print("="*80)
print("  ⚙️  INITIALIZING ALL METHODS")
print("="*80 + "\n")

methods = {}

# 1. Greedy Baseline (no search)
print("1. Greedy Baseline (no tree search)")
methods['Greedy'] = {
    'name': 'Greedy Baseline',
    'color': '#95a5a6',
    'run': lambda tokens: greedy_generate(model, tokens)
}

# 2. MCTS
print("2. Standard MCTS")
mcts_config = MCTSConfig(
    num_simulations=NUM_ROLLOUTS,
    c_puct=1.0,
    expansion_k=4,
    verbose=False
)
methods['MCTS'] = {
    'name': 'MCTS (UCT)',
    'color': '#3498db',
    'generator': MCTSTextGenerator(model, reward_fn, mcts_config)
}

# 3. DTS
print("3. DTS (Diffusion Tree Sampling)")
dts_config = DTSConfig(
    num_rollouts=NUM_ROLLOUTS,
    temperature=1.0,
    expansion_k=4,
    use_soft_bellman=True,
    verbose=False
)
methods['DTS'] = {
    'name': 'DTS (Soft Bellman)',
    'color': '#2ecc71',
    'generator': DTSTextGenerator(model, reward_fn, dts_config)
}

# 4. DTS* (greedy)
print("4. DTS* (Greedy variant)")
dts_star_config = DTSConfig(
    num_rollouts=NUM_ROLLOUTS,
    temperature=1.0,
    expansion_k=4,
    use_soft_bellman=True,
    verbose=False
)
methods['DTS*'] = {
    'name': 'DTS* (UCT)',
    'color': '#e74c3c',
    'generator': DTSStarTextGenerator(model, reward_fn, dts_star_config)
}

# 5. MaxEnt-TS (our implementation)
print("5. MaxEnt-TS (Our Implementation)")
maxent_config = MaxEntTSConfig(
    num_rollouts=NUM_ROLLOUTS,
    temperature=1.0,
    expansion_k=4,
    max_seq_length=512,
    verbose=False
)
methods['MaxEnt-TS'] = {
    'name': 'MaxEnt-TS (Ours)',
    'color': '#9b59b6',
    'generator': MaxEntTS(model, reward_fn_obj, maxent_config)
}

print("\n✅ All methods initialized\n")


def greedy_generate(model, prompt_tokens):
    """Greedy baseline - just model.generate()"""
    with torch.no_grad():
        output = model.generate_sequence(prompt_tokens, max_tokens=100, temperature=0.8)
    return {
        'best_sequence': output['tokens'],
        'method': 'greedy',
        'nodes': 0
    }


# Run comparison
print("="*80)
print("  🏁 RUNNING COMPARISON")
print("="*80 + "\n")

results = {method: [] for method in methods.keys()}

for sample_idx in range(NUM_SAMPLES):
    print(f"\n{'─'*80}")
    print(f"  SAMPLE {sample_idx + 1}/{NUM_SAMPLES}")
    print(f"{'─'*80}\n")
    
    # Get sample - handle different key formats
    sample = dataset[sample_idx]
    
    # Try different possible keys for prompt
    if 'pre_prompt' in sample and 'time_series_text' in sample:
        # M4 format: construct from parts
        prompt_text = str(sample.get('pre_prompt', '')) + str(sample.get('time_series_text', '')) + str(sample.get('post_prompt', ''))
    else:
        # Try standard keys
        prompt_text = sample.get('input', sample.get('prompt', sample.get('text', '')))
    
    expected = sample.get('output', sample.get('answer', ''))
    
    print(f"Prompt: {prompt_text[:80]}...")
    print(f"Expected length: {len(expected)} chars")
    print()
    
    # Tokenize
    prompt_tokens = model.tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    print(f"Tokenized: {len(prompt_tokens)} tokens")
    print()
    
    # Test each method
    for method_name, method_info in methods.items():
        print(f"  Testing {method_name}...")
        try:
            start_time = time.time()
            
            if method_name == 'Greedy':
                result = method_info['run'](prompt_tensor)
            else:
                result = method_info['generator'].search(prompt_tensor, max_new_tokens=100)
            
            elapsed = time.time() - start_time
            
            # Extract generated text
            if 'best_sequence' in result:
                seq = result['best_sequence']
                if isinstance(seq, torch.Tensor):
                    if seq.dim() == 2:
                        seq = seq[0]
                    seq = seq.tolist()
                generated_text = model.tokenizer.decode(seq, skip_special_tokens=True)
            else:
                generated_text = "<error>"
            
            # Get stats
            nodes = result.get('tree_size', result.get('total_simulations', result.get('nodes', 0)))
            value = result.get('best_value', result.get('best_reward', 0.0))
            
            print(f"    ✓ {elapsed:.1f}s, {nodes} nodes, value={value:.4f}")
            print(f"      Output: {generated_text[:60]}...")
            
            results[method_name].append({
                'sample_idx': sample_idx,
                'time': elapsed,
                'nodes': nodes,
                'value': value,
                'output': generated_text,
                'output_length': len(generated_text)
            })
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            results[method_name].append({
                'sample_idx': sample_idx,
                'error': str(e)
            })
    
    print()

# Summary
print("\n" + "="*80)
print("  📊 SUMMARY STATISTICS")
print("="*80 + "\n")

summary = {}
for method_name in methods.keys():
    method_results = results[method_name]
    valid_results = [r for r in method_results if 'error' not in r]
    
    if valid_results:
        avg_time = np.mean([r['time'] for r in valid_results])
        avg_nodes = np.mean([r['nodes'] for r in valid_results])
        avg_value = np.mean([r['value'] for r in valid_results])
        avg_length = np.mean([r['output_length'] for r in valid_results])
        
        summary[method_name] = {
            'avg_time': avg_time,
            'avg_nodes': avg_nodes,
            'avg_value': avg_value,
            'avg_length': avg_length,
            'success_rate': len(valid_results) / len(method_results)
        }
        
        print(f"{method_name}:")
        print(f"  Time: {avg_time:.2f}s")
        print(f"  Nodes: {avg_nodes:.1f}")
        print(f"  Value: {avg_value:.4f}")
        print(f"  Output length: {avg_length:.1f} chars")
        print(f"  Success: {len(valid_results)}/{len(method_results)}")
        print()

# Save results
import json
output_file = Path("evaluation/results/method_comparison.json")
output_file.parent.mkdir(parents=True, exist_ok=True)

comparison_data = {
    'config': {
        'num_samples': NUM_SAMPLES,
        'num_rollouts': NUM_ROLLOUTS,
        'device': device
    },
    'results': results,
    'summary': summary
}

with open(output_file, 'w') as f:
    json.dump(comparison_data, f, indent=2, default=str)

print(f"✅ Results saved to: {output_file}")
print()

# Key insights
print("="*80)
print("  💡 KEY INSIGHTS")
print("="*80 + "\n")

print("Expected behavior:")
print("  • Greedy: Fast, 0 nodes, lowest quality")
print("  • MCTS: Moderate nodes, standard UCT selection")
print("  • DTS: Similar nodes to MCTS, but Soft Bellman backup")
print("  • DTS*: Greedy UCT selection, faster convergence")
print("  • MaxEnt-TS: Should match DTS if implemented correctly")
print()

if 'DTS' in summary and 'MaxEnt-TS' in summary:
    dts_val = summary['DTS']['avg_value']
    our_val = summary['MaxEnt-TS']['avg_value']
    diff = abs(dts_val - our_val)
    
    print(f"DTS vs MaxEnt-TS comparison:")
    print(f"  DTS value: {dts_val:.4f}")
    print(f"  MaxEnt-TS value: {our_val:.4f}")
    print(f"  Difference: {diff:.4f}")
    
    if diff < 0.1:
        print("  ✅ Values are similar - implementation looks correct!")
    else:
        print("  ⚠️  Large difference - implementation may differ from paper")
print()

print("="*80 + "\n")


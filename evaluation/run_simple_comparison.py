"""
Simple Comparison Script - Get REAL Performance Numbers
==========================================================

Compares Greedy, MCTS, DTS, DTS*, and MaxEnt-TS on ACTUAL data.
"""

import os
import sys
import torch
import time
import json
from dataclasses import dataclass
from typing import Dict, List

# Add paths
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('baselines'))

# Import methods
from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig
from dts_implementation.rewards.spectral_reward import SpectralReward

from baselines.mcts_baseline import MCTSTextGenerator, MCTSConfig
from baselines.dts_baseline import DTSTextGenerator, DTSConfig


@dataclass
class ComparisonResult:
    method: str
    sample_idx: int
    prompt: str
    expected: str
    generated: str
    nodes_explored: int
    time_seconds: float
    success: bool
    error: str = ""


def greedy_baseline(model: PyTorchHFWrapper, prompt_tokens: torch.Tensor, max_tokens: int = 50) -> Dict:
    """Simple greedy baseline"""
    start_time = time.time()
    
    with torch.no_grad():
        output = model.model.generate(
            input_ids=prompt_tokens,
            max_new_tokens=max_tokens,
            do_sample=False,  # Greedy
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.eos_token_id
        )
    
    elapsed = time.time() - start_time
    generated_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    
    return {
        'text': generated_text,
        'nodes': len(output[0]),  # Approximate
        'time': elapsed
    }


def run_comparison(num_samples: int = 3, num_rollouts: int = 10):
    """
    Run comprehensive comparison
    """
    print("=" * 80)
    print("  🔬 SIMPLE COMPARISON - REAL PERFORMANCE NUMBERS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Samples: {num_samples}")
    print(f"  • Rollouts/Simulations: {num_rollouts}")
    print()
    
    # Check device
    if torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("🚀 Using CUDA")
    else:
        device = "cpu"
        print("💻 Using CPU")
    print()
    
    # Load model
    print("📥 Loading model...")
    model = PyTorchHFWrapper("meta-llama/Llama-3.2-1B-Instruct", device=device)
    print("✅ Model loaded\n")
    
    # Define test prompts (simple, no dataset dependency)
    test_cases = [
        {
            'prompt': "The capital of France is",
            'expected': "Paris"
        },
        {
            'prompt': "2 + 2 equals",
            'expected': "4"
        },
        {
            'prompt': "The largest planet in our solar system is",
            'expected': "Jupiter"
        }
    ]
    
    # Limit to requested samples
    test_cases = test_cases[:num_samples]
    
    # Define methods to test
    methods = ['Greedy', 'MCTS', 'DTS', 'DTS*', 'MaxEnt-TS']
    
    # Store all results
    all_results = []
    
    # Run each sample through each method
    for sample_idx, test_case in enumerate(test_cases):
        prompt_text = test_case['prompt']
        expected = test_case['expected']
        
        print(f"\n{'='*80}")
        print(f"📝 Sample {sample_idx + 1}/{len(test_cases)}: '{prompt_text}'")
        print(f"   Expected: '{expected}'")
        print(f"{'='*80}\n")
        
        # Encode prompt
        prompt_tokens = model.tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        
        # Test each method
        for method_name in methods:
            print(f"⚙️  Testing {method_name}...")
            
            result = ComparisonResult(
                method=method_name,
                sample_idx=sample_idx,
                prompt=prompt_text,
                expected=expected,
                generated="",
                nodes_explored=0,
                time_seconds=0.0,
                success=False
            )
            
            try:
                if method_name == 'Greedy':
                    output = greedy_baseline(model, prompt_tokens, max_tokens=20)
                    result.generated = output['text']
                    result.nodes_explored = output['nodes']
                    result.time_seconds = output['time']
                    result.success = True
                
                elif method_name == 'MCTS':
                    config = MCTSConfig(num_simulations=num_rollouts, expansion_k=3, verbose=False)
                    def dummy_reward(x): return 0.5
                    mcts = MCTSTextGenerator(model, dummy_reward, config)
                    
                    search_result = mcts.search(prompt_tokens[0], max_new_tokens=20)
                    result.generated = search_result['best_text']
                    result.nodes_explored = search_result['nodes_explored']
                    result.time_seconds = search_result['time']
                    result.success = True
                
                elif method_name == 'DTS':
                    config = DTSConfig(num_rollouts=num_rollouts, expansion_k=3, verbose=False)
                    def dummy_reward(x): return 0.5
                    dts = DTSTextGenerator(model, dummy_reward, config)
                    
                    search_result = dts.search(prompt_tokens[0], max_new_tokens=20)
                    result.generated = search_result['best_text']
                    result.nodes_explored = search_result['nodes_explored']
                    result.time_seconds = search_result['time']
                    result.success = True
                
                elif method_name == 'DTS*':
                    config = DTSConfig(num_rollouts=num_rollouts, expansion_k=3, verbose=False, use_soft_bellman=False)
                    def dummy_reward(x): return 0.5
                    dts_greedy = DTSTextGenerator(model, dummy_reward, config)
                    
                    search_result = dts_greedy.search(prompt_tokens[0], max_new_tokens=20)
                    result.generated = search_result['best_text']
                    result.nodes_explored = search_result['nodes_explored']
                    result.time_seconds = search_result['time']
                    result.success = True
                
                elif method_name == 'MaxEnt-TS':
                    config = MaxEntTSConfig(
                        num_rollouts=num_rollouts,
                        expansion_k=3,
                        temperature=1.0,
                        gamma=0.95,
                        verbose=False
                    )
                    def dummy_reward(x): return 0.5
                    maxent = MaxEntTS(model, dummy_reward, config)
                    
                    search_result = maxent.search(prompt_tokens[0], max_new_tokens=20)
                    result.generated = search_result['best_text']
                    result.nodes_explored = search_result['tree_stats']['total_nodes']
                    result.time_seconds = search_result['tree_stats']['time']
                    result.success = True
                
                print(f"   ✅ Generated: '{result.generated[:80]}...'")
                print(f"   📊 Nodes: {result.nodes_explored}, Time: {result.time_seconds:.2f}s\n")
                
            except Exception as e:
                result.error = str(e)
                print(f"   ❌ Error: {str(e)}\n")
            
            all_results.append(result)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    
    for method_name in methods:
        method_results = [r for r in all_results if r.method == method_name]
        successful = [r for r in method_results if r.success]
        
        if successful:
            avg_nodes = sum(r.nodes_explored for r in successful) / len(successful)
            avg_time = sum(r.time_seconds for r in successful) / len(successful)
            success_rate = len(successful) / len(method_results) * 100
            
            print(f"\n{method_name}:")
            print(f"  Success Rate: {success_rate:.0f}%")
            print(f"  Avg Nodes: {avg_nodes:.1f}")
            print(f"  Avg Time: {avg_time:.3f}s")
        else:
            print(f"\n{method_name}: All attempts failed")
    
    # Save detailed results
    results_file = "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump([{
            'method': r.method,
            'sample_idx': r.sample_idx,
            'prompt': r.prompt,
            'expected': r.expected,
            'generated': r.generated,
            'nodes_explored': r.nodes_explored,
            'time_seconds': r.time_seconds,
            'success': r.success,
            'error': r.error
        } for r in all_results], f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    run_comparison(num_samples=3, num_rollouts=10)


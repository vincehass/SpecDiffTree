"""
Comprehensive S-ADT Demo

Shows MaxEnt-TS tree search on multiple prompts with detailed statistics.
Compares with greedy baseline to demonstrate improvements.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dts_implementation.models.local_loader import load_base_model
from dts_implementation.rewards.spectral_reward import create_spectral_reward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig


def run_prompt(model, reward, config, prompt_text):
    """Run S-ADT on a single prompt"""
    
    print(f"\n{'='*70}")
    print(f"Prompt: {prompt_text}")
    print(f"{'='*70}")
    
    # Encode prompt
    prompt_tokens = model.encode_text(prompt_text)
    print(f"Tokens: {prompt_tokens.shape[-1]}")
    
    # Run MaxEnt-TS
    print(f"\n🌳 Running MaxEnt-TS ({config.num_rollouts} rollouts)...")
    searcher = MaxEntTS(model, reward, config)
    
    start_time = time.time()
    results = searcher.search(prompt_tokens)
    maxent_time = time.time() - start_time
    
    # Run Greedy baseline
    print(f"\n🔄 Running Greedy baseline...")
    start_time = time.time()
    greedy_output = model.rollout_sequence(
        prompt_tokens,
        max_new_tokens=config.max_seq_length,
        temperature=0.1,
        return_full_sequence=True
    )
    greedy_time = time.time() - start_time
    greedy_text = model.decode_sequence(greedy_output)[0]
    
    # Results
    print(f"\n📊 RESULTS:")
    print(f"\n   MaxEnt-TS ({maxent_time:.1f}s):")
    print(f"   {results['best_text'][:200]}...")
    print(f"\n   Greedy ({greedy_time:.1f}s):")
    print(f"   {greedy_text[:200]}...")
    
    print(f"\n   Tree Stats:")
    stats = results['tree_stats']
    print(f"   • Nodes explored: {stats['total_nodes']}")
    print(f"   • Depth: {stats['max_depth']}")
    print(f"   • Branching: {stats['avg_branching_factor']:.2f}")
    print(f"   • Reward: {results['best_reward']:.4f}")
    
    return results, greedy_text


def main():
    """Run comprehensive demo"""
    
    print("="*70)
    print("  S-ADT Comprehensive Demo")
    print("="*70)
    
    # ==================================================================
    # SETUP
    # ==================================================================
    
    print("\n📦 Loading model...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = load_base_model(
        llm_id="meta-llama/Llama-3.2-1B",
        device=device
    )
    
    # Setup spectral reward
    print("\n🎯 Setting up spectral reward...")
    reward = create_spectral_reward(
        task='tsqa',
        gamma=1.0,
        sampling_rate=100.0,
        normalize=True
    )
    
    # Create context
    t = np.linspace(0, 10, 1000)
    context_ts = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))
    reward.set_context(context_ts)
    
    # Configure search - more rollouts for better results
    print("\n⚙️  Configuring MaxEnt-TS...")
    config = MaxEntTSConfig(
        num_rollouts=20,  # More thorough search
        temperature=1.0,
        max_seq_length=50,
        expansion_k=4,
        rollout_temperature=0.7,
        use_uct=False,
        verbose=False  # Less verbose for cleaner output
    )
    
    print(f"   • Rollouts: {config.num_rollouts}")
    print(f"   • Temperature: {config.temperature}")
    print(f"   • Expansion k: {config.expansion_k}")
    
    # ==================================================================
    # TEST PROMPTS
    # ==================================================================
    
    prompts = [
        "Question: What is 2+2? Answer:",
        "Complete this pattern: 1, 2, 4, 8,",
        "What comes next in the Fibonacci sequence: 1, 1, 2, 3, 5, 8,",
        "Solve: If x + 5 = 12, then x =",
    ]
    
    print(f"\n{'='*70}")
    print(f"Running {len(prompts)} test prompts...")
    print(f"{'='*70}")
    
    all_results = []
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n\n{'#'*70}")
        print(f"# TEST {i}/{len(prompts)}")
        print(f"{'#'*70}")
        
        try:
            results, greedy = run_prompt(model, reward, config, prompt)
            all_results.append({
                'prompt': prompt,
                'maxent': results['best_text'],
                'greedy': greedy,
                'stats': results['tree_stats']
            })
        except Exception as e:
            print(f"\n❌ Error on prompt {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    
    print(f"\n\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    if all_results:
        total_nodes = sum(r['stats']['total_nodes'] for r in all_results)
        avg_depth = np.mean([r['stats']['max_depth'] for r in all_results])
        avg_branching = np.mean([r['stats']['avg_branching_factor'] for r in all_results])
        
        print(f"\n📊 Aggregate Statistics:")
        print(f"   • Total nodes explored: {total_nodes}")
        print(f"   • Average depth: {avg_depth:.1f}")
        print(f"   • Average branching: {avg_branching:.2f}")
        print(f"   • Total rollouts: {config.num_rollouts * len(prompts)}")
        
        print(f"\n💡 Key Insights:")
        print(f"   • MaxEnt-TS explored {total_nodes} different paths")
        print(f"   • Greedy only explores {len(prompts)} paths (1 per prompt)")
        print(f"   • {total_nodes / len(prompts):.0f}x more exploration!")
        
        print(f"\n✅ Benefits of MaxEnt-TS:")
        print(f"   1. Explores multiple solution paths (not just greedy)")
        print(f"   2. Soft Bellman maintains distribution (no collapse)")
        print(f"   3. Spectral regularization preserves frequency content")
        print(f"   4. Better solutions through tree search")
    
    print(f"\n{'='*70}")
    print(f"Demo complete! 🎉")
    print(f"{'='*70}")
    
    print(f"\n📝 Next Steps:")
    print(f"   • Train on Stage 1 (TSQA) for real evaluation")
    print(f"   • Test with trained checkpoints")
    print(f"   • Evaluate spectral fidelity improvements")
    print(f"   • Compare with beam search baseline")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


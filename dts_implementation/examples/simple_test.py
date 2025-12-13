"""
Simple S-ADT Test - Minimal Example

Tests MaxEnt-TS with a base OpenTSLM model (no training data needed).
This verifies the entire pipeline works end-to-end.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dts_implementation.models.local_loader import load_base_model
from dts_implementation.rewards.spectral_reward import create_spectral_reward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig

def main():
    """Run simple S-ADT test"""
    
    print("="*70)
    print("  Simple S-ADT Test - Minimal Example")
    print("="*70)
    
    # ==================================================================
    # 1. SETUP
    # ==================================================================
    
    print("\n📦 Step 1: Loading model...")
    
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"   Using device: {device}")
    
    # Load base OpenTSLM model
    print(f"\n   Loading OpenTSLMSP (base Llama 3.2 1B)...")
    model = load_base_model(
        llm_id="meta-llama/Llama-3.2-1B",
        device=device
    )
    
    # ==================================================================
    # 2. CREATE SPECTRAL REWARD
    # ==================================================================
    
    print("\n🎯 Step 2: Setting up spectral reward...")
    
    # Create spectral reward computer
    spectral_reward = create_spectral_reward(
        task='tsqa',
        gamma=1.0,
        sampling_rate=100.0,
        spectral_metric='l1',
        normalize=True
    )
    
    # Create dummy context time series
    t = np.linspace(0, 10, 1000)
    context_ts = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))
    spectral_reward.set_context(context_ts)
    
    print(f"   ✅ Spectral reward configured (γ={spectral_reward.gamma})")
    
    # ==================================================================
    # 3. PREPARE PROMPT
    # ==================================================================
    
    print("\n✍️  Step 3: Creating prompt...")
    
    prompt_text = "Question: What is 2+2? Answer:"
    
    print(f"\n   Prompt: {prompt_text}")
    
    # Encode prompt
    prompt_tokens = model.encode_text(prompt_text)
    print(f"   Encoded to {prompt_tokens.shape[-1]} tokens")
    
    # ==================================================================
    # 4. CONFIGURE SEARCH
    # ==================================================================
    
    print("\n⚙️  Step 4: Configuring MaxEnt-TS...")
    
    config = MaxEntTSConfig(
        num_rollouts=5,  # Very small for quick test
        temperature=1.0,
        max_seq_length=30,
        expansion_k=3,
        expansion_temperature=1.0,
        rollout_temperature=0.7,
        rollout_top_k=50,
        use_uct=False,
        gamma=1.0,
        verbose=True
    )
    
    print(f"   Configuration:")
    print(f"      Rollouts: {config.num_rollouts} (small for quick test)")
    print(f"      Temperature: {config.temperature}")
    print(f"      Expansion top-k: {config.expansion_k}")
    
    # ==================================================================
    # 5. RUN SEARCH
    # ==================================================================
    
    print("\n🔍 Step 5: Running MaxEnt-TS search...")
    print(f"{'='*70}\n")
    
    # Initialize searcher
    searcher = MaxEntTS(
        model=model,
        reward=spectral_reward,
        config=config
    )
    
    # Run search
    try:
        results = searcher.search(
            prompt_tokens=prompt_tokens,
            ground_truth=None
        )
        
        print(f"\n{'='*70}")
        
        # ==================================================================
        # 6. ANALYZE RESULTS
        # ==================================================================
        
        print("\n📊 Step 6: Results Analysis")
        print("="*70)
        
        print(f"\n🏆 Best Sequence:")
        print(f"   Text: {results['best_text']}")
        print(f"   Reward: {results['best_reward']:.4f}")
        
        print(f"\n🌳 Tree Statistics:")
        stats = results['tree_stats']
        print(f"   Total nodes explored: {stats['total_nodes']}")
        print(f"   Maximum depth: {stats['max_depth']}")
        print(f"   Avg branching factor: {stats['avg_branching_factor']:.2f}")
        print(f"   Total rollouts: {stats['rollouts']}")
        
        # ==================================================================
        # 7. COMPARISON WITH GREEDY
        # ==================================================================
        
        print("\n🔄 Step 7: Comparison with greedy baseline...")
        
        # Generate with standard greedy decoding
        greedy_output = model.rollout_sequence(
            prompt_tokens,
            max_new_tokens=config.max_seq_length,
            temperature=0.1,
            return_full_sequence=True
        )
        greedy_text = model.decode_sequence(greedy_output)[0]
        
        print(f"\n   MaxEnt-TS: {results['best_text']}")
        print(f"   Greedy:    {greedy_text}")
        
        print("\n" + "="*70)
        print("✅ Test complete!")
        print("="*70)
        print("\n🎉 S-ADT implementation is working! The full pipeline is functional.")
        print("   Next: Test with real OpenTSLM trained models and TSQA dataset.")
        
    except Exception as e:
        print(f"\n❌ Error during search: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run test
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


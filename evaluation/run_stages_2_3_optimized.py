"""
Run S-ADT evaluation on Stages 2 and 3 using MLX (optimized - load model once).
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
import numpy as np
import time
import json
from datetime import datetime

from dts_implementation.models.mlx_direct_loader import SimplifiedMLXWrapper
from dts_implementation.rewards.spectral_reward import SpectralReward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig, TokenNode


def main():
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║      🚀 RUNNING STAGES 2 & 3 WITH MLX (OPTIMIZED)               ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    # Load model ONCE for both stages
    print("📥 Loading MLX model (Llama 3.2 1B - 4bit)...")
    sys.stdout.flush()
    model = SimplifiedMLXWrapper()
    print("✅ Model loaded and ready for both stages!\n")
    sys.stdout.flush()
    
    print("DEBUG: About to define test prompts...")
    sys.stdout.flush()
    # Test prompts for both stages
    test_prompts = {
        2: [  # M4 Captioning
            "Describe this time series pattern:",
            "Generate a caption for the following data:",
            "Explain the trend observed in this series:",
        ],
        3: [  # HAR CoT
            "Analyze the sensor data and identify the activity being performed:",
            "What activity is shown in this data? Provide step-by-step reasoning:",
            "Classify the activity from the sensor readings:",
        ]
    }
    print("DEBUG: Test prompts defined")
    
    # Setup reward (same for both stages)
    print("DEBUG: Creating reference time series...")
    reference_ts = np.sin(np.linspace(0, 10 * np.pi, 256)) + 0.1 * np.random.randn(256)
    print("DEBUG: Initializing SpectralReward...")
    reward = SpectralReward(gamma=1.0)
    print("DEBUG: Setting reward context...")
    reward.set_context(reference_ts)
    print("DEBUG: Reward setup complete")
    
    # Configure S-ADT
    print("DEBUG: Configuring S-ADT...")
    config = MaxEntTSConfig(
        num_rollouts=20,
        temperature=1.0,
        expansion_k=4
    )
    print("DEBUG: Config created")
    
    all_results = {}
    
    print("DEBUG: About to enter main loop for stages 2 and 3...")
    # Run both stages
    for stage_num in [2, 3]:
        print(f"DEBUG: Entering loop for stage {stage_num}")
        print(f"\n{'='*70}")
        print(f"  STAGE {stage_num} EVALUATION")
        print(f"{'='*70}\n")
        
        prompts = test_prompts[stage_num]
        results = []
        total_nodes = 0
        total_time = 0
        
        print(f"🔬 Running {len(prompts)} test prompts...\n")
        
        for i, prompt in enumerate(prompts):
            print(f"Prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
            
            try:
                # Tokenize
                prompt_tokens = model.tokenizer.encode(prompt)
                prompt_array = mx.array([prompt_tokens])
                
                # Run S-ADT
                start_time = time.time()
                searcher = MaxEntTS(model, reward, config)
                result = searcher.search(prompt_array)
                elapsed = time.time() - start_time
                
                # Extract results
                tree_stats = result['tree_stats']
                generated_text = result['best_text']
                best_reward = result['best_reward']
                nodes = tree_stats['total_nodes']
                
                total_nodes += nodes
                total_time += elapsed
                
                print(f"  ✅ {nodes} nodes, {elapsed:.1f}s")
                print(f"     Output: {generated_text[:80]}...\n")
                
                results.append({
                    'prompt': prompt,
                    'nodes': nodes,
                    'time': elapsed,
                    'output': generated_text,
                    'best_reward': best_reward,
                    'tree_stats': tree_stats
                })
                
            except Exception as e:
                print(f"  ❌ Failed: {e}\n")
                import traceback
                traceback.print_exc()
                results.append({
                    'prompt': prompt,
                    'nodes': 0,
                    'time': 0,
                    'error': str(e)
                })
        
        # Summary
        avg_nodes = total_nodes / len(prompts) if prompts else 0
        avg_time = total_time / len(prompts) if prompts else 0
        
        print(f"\n{'='*70}")
        print(f"  STAGE {stage_num} SUMMARY")
        print(f"{'='*70}\n")
        print(f"Prompts: {len(prompts)}")
        print(f"Rollouts: 20")
        print(f"Avg Nodes: {avg_nodes:.1f}")
        print(f"Avg Time: {avg_time:.1f}s")
        print(f"Total Time: {total_time:.1f}s")
        print(f"{'='*70}\n")
        
        # Save results
        output = {
            'stage': stage_num,
            'framework': 'MLX',
            'num_prompts': len(prompts),
            'num_rollouts': 20,
            'avg_nodes': avg_nodes,
            'avg_time': avg_time,
            'total_time': total_time,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        all_results[f'stage{stage_num}'] = output
        
        output_file = f'evaluation/results/stage{stage_num}_mlx_final.json'
        Path('evaluation/results').mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ Results saved to {output_file}\n")
    
    # Combined summary
    print(f"\n{'='*70}")
    print("  📊 COMBINED SUMMARY")
    print(f"{'='*70}\n")
    for stage_key, data in all_results.items():
        stage_num = data['stage']
        print(f"Stage {stage_num}:")
        print(f"  • Avg Nodes: {data['avg_nodes']:.1f}")
        print(f"  • Avg Time: {data['avg_time']:.1f}s")
    print(f"\nTotal Time: {sum(d['total_time'] for d in all_results.values()):.1f}s")
    print(f"{'='*70}\n")
    
    # Save aggregate
    aggregate_file = 'evaluation/results/stages_2_3_aggregate.json'
    with open(aggregate_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("✅ All results saved!")
    print(f"📊 Aggregate: {aggregate_file}\n")


if __name__ == "__main__":
    main()


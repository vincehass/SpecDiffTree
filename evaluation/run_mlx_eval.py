"""
MLX-based evaluation for all 5 OpenTSLM stages with S-ADT.
Uses MLX for Apple Silicon (M1/M3 Max).
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import json
from datetime import datetime
import time
import mlx.core as mx

# Import MLX model wrapper
from dts_implementation.models.mlx_loader import MLXModelWrapper
from dts_implementation.rewards.spectral_reward import SpectralReward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig, TokenNode


def create_spectral_reward(reference_ts, gamma=1.0):
    """Create spectral reward"""
    reward = SpectralReward(gamma=gamma)
    reward.set_context(reference_ts)
    return reward


def run_stage_evaluation(stage: int, num_prompts: int = 5, num_rollouts: int = 20):
    """
    Run evaluation for a single stage using MLX.
    """
    print(f"\n{'='*70}")
    print(f"  STAGE {stage} EVALUATION (MLX)")
    print(f"{'='*70}\n")
    
    # Stage-specific test prompts
    test_prompts = {
        1: [  # TSQA (MCQ)
            "Question: What is 2+2? Answer:",
            "Question: Is this trend increasing? A) Yes B) No C) Stable D) Unknown Answer:",
            "Question: Does this show seasonality? A) Yes B) No C) Cannot determine Answer:",
            "Question: What is the main pattern? A) Linear B) Exponential C) Cyclical D) Random Answer:",
            "Question: Is there an anomaly? A) Yes, beginning B) Yes, middle C) Yes, end D) No Answer:",
        ],
        2: [  # M4 Captioning
            "Describe the following time series pattern:",
            "Generate a caption for this time series:",
            "Explain the trend in this data:",
        ],
        3: [  # HAR CoT
            "Analyze the sensor data and identify the activity:",
            "What activity is being performed? Explain:",
        ],
        4: [  # Sleep CoT
            "Classify the sleep stage from EEG signal:",
            "What sleep stage is present? Provide reasoning:",
        ],
        5: [  # ECG QA
            "Analyze the ECG signal and identify arrhythmias:",
            "What cardiac condition is present?:",
        ]
    }
    
    prompts = test_prompts.get(stage, test_prompts[1])[:num_prompts]
    
    # Load MLX model
    print(f"📥 Loading MLX model (Llama 3.2 1B)...")
    model = MLXModelWrapper("mlx-community/Llama-3.2-1B-Instruct-4bit")
    
    # Setup reward
    reference_ts = np.sin(np.linspace(0, 10 * np.pi, 256)) + 0.1 * np.random.randn(256)
    reward = create_spectral_reward(reference_ts, gamma=1.0)
    
    # Configure search
    config = MaxEntTSConfig(num_rollouts=num_rollouts, temperature=1.0, expansion_k=4)
    
    # Run evaluation
    results = []
    total_sadt_nodes = 0
    total_greedy_nodes = 0
    total_sadt_time = 0
    total_greedy_time = 0
    
    print(f"\n🔬 Running {len(prompts)} test prompts...\n")
    
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
        
        # Tokenize
        prompt_tokens = model.tokenizer.encode(prompt)
        prompt_array = mx.array([prompt_tokens])
        
        # Run S-ADT
        start_time = time.time()
        try:
            searcher = MaxEntTS(model, reward, config)
            best_node = searcher.search(prompt_array)
            tree_stats = searcher._get_tree_stats()
            
            sadt_time = time.time() - start_time
            sadt_nodes = tree_stats['nodes_explored']
            
            # Get generated text
            if isinstance(best_node, TokenNode):
                if isinstance(best_node.token_ids, mx.array):
                    token_ids = best_node.token_ids.tolist()
                else:
                    token_ids = best_node.token_ids
                generated_text = model.decode_sequence(token_ids)
            else:
                generated_text = ""
            
            print(f"  ✅ S-ADT: {sadt_nodes} nodes, {sadt_time:.1f}s")
            
            total_sadt_nodes += sadt_nodes
            total_sadt_time += sadt_time
            
        except Exception as e:
            print(f"  ❌ S-ADT failed: {e}")
            sadt_nodes = 0
            sadt_time = 0
            tree_stats = {}
            generated_text = ""
        
        # Run Greedy baseline
        start_time = time.time()
        try:
            greedy_result = model.generate_sequence(prompt_array, max_tokens=50, temperature=0.0)
            greedy_text = greedy_result['text']
            greedy_time = time.time() - start_time
            greedy_nodes = 1  # Greedy explores 1 path
            
            print(f"  ✅ Greedy: {greedy_nodes} nodes, {greedy_time:.1f}s")
            
            total_greedy_nodes += greedy_nodes
            total_greedy_time += greedy_time
            
        except Exception as e:
            print(f"  ❌ Greedy failed: {e}")
            greedy_nodes = 1
            greedy_time = 0
            greedy_text = ""
        
        results.append({
            'prompt': prompt,
            'sadt': {
                'text': generated_text,
                'nodes': sadt_nodes,
                'time': sadt_time,
                'tree_stats': tree_stats
            },
            'greedy': {
                'text': greedy_text,
                'nodes': greedy_nodes,
                'time': greedy_time
            }
        })
    
    # Compute aggregate stats
    avg_sadt_nodes = total_sadt_nodes / len(prompts) if prompts else 0
    avg_greedy_nodes = total_greedy_nodes / len(prompts) if prompts else 0
    exploration_ratio = avg_sadt_nodes / avg_greedy_nodes if avg_greedy_nodes > 0 else 0
    
    summary = {
        'stage': stage,
        'framework': 'MLX',
        'num_prompts': len(prompts),
        'num_rollouts': num_rollouts,
        'avg_sadt_nodes': avg_sadt_nodes,
        'avg_greedy_nodes': avg_greedy_nodes,
        'exploration_ratio': exploration_ratio,
        'avg_sadt_time': total_sadt_time / len(prompts) if prompts else 0,
        'avg_greedy_time': total_greedy_time / len(prompts) if prompts else 0,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  STAGE {stage} SUMMARY (MLX)")
    print(f"{'='*70}\n")
    print(f"Prompts: {len(prompts)}")
    print(f"Rollouts: {num_rollouts}")
    print(f"Framework: MLX (Apple Silicon)")
    print(f"\nExploration:")
    print(f"  S-ADT:  {avg_sadt_nodes:.1f} nodes avg ({total_sadt_time:.1f}s total)")
    print(f"  Greedy: {avg_greedy_nodes:.1f} nodes avg ({total_greedy_time:.1f}s total)")
    print(f"  Ratio:  {exploration_ratio:.1f}x more exploration!")
    print(f"\n{'='*70}\n")
    
    return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=int, nargs='+', default=[1],
                        help='Stages to evaluate')
    parser.add_argument('--num-prompts', type=int, default=3,
                        help='Number of prompts per stage')
    parser.add_argument('--num-rollouts', type=int, default=20,
                        help='Number of rollouts for S-ADT')
    parser.add_argument('--output-dir', type=str, default='evaluation/results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n╔══════════════════════════════════════════════════════════════════╗")
    print(f"║     🚀 MLX EVALUATION - STAGES {args.stages}                      ║")
    print(f"║           (Optimized for Apple Silicon M1/M3 Max)               ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝\n")
    
    all_results = {}
    
    for stage in args.stages:
        summary = run_stage_evaluation(stage, args.num_prompts, args.num_rollouts)
        all_results[f'stage{stage}'] = summary
        
        # Save individual results
        output_file = output_dir / f'stage{stage}_mlx_eval.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved to {output_file}\n")
    
    # Save aggregate
    aggregate_file = output_dir / 'mlx_eval_all_stages.json'
    with open(aggregate_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"{'='*70}")
    print(f"  ✅ MLX EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📊 Aggregate: {aggregate_file}\n")
    
    # Print summary table
    print(f"\n{'='*70}")
    print(f"  📊 SUMMARY TABLE")
    print(f"{'='*70}\n")
    print(f"{'Stage':<10} {'Nodes (S-ADT)':<15} {'Nodes (Greedy)':<15} {'Ratio':<10}")
    print(f"{'-'*70}")
    for stage_key, data in all_results.items():
        stage_num = data['stage']
        sadt = data['avg_sadt_nodes']
        greedy = data['avg_greedy_nodes']
        ratio = data['exploration_ratio']
        print(f"Stage {stage_num:<4} {sadt:<15.1f} {greedy:<15.1f} {ratio:<10.1f}x")
    print(f"{'-'*70}\n")


if __name__ == "__main__":
    main()


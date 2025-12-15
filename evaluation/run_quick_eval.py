"""
Quick evaluation using the working comprehensive demo approach.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import json
from datetime import datetime
import time

# Use the working demo approach
from dts_implementation.models.local_loader import load_base_model
from dts_implementation.rewards.spectral_reward import SpectralReward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig, TokenNode
import mlx.core as mx


def create_spectral_reward(reference_ts, gamma=1.0):
    """Create spectral reward"""
    reward = SpectralReward(gamma=gamma)
    reward.set_context(reference_ts)
    return reward


def run_stage_evaluation(stage: int, num_prompts: int = 5, num_rollouts: int = 20):
    """
    Run evaluation for a single stage using working demo approach.
    """
    print(f"\n{'='*70}")
    print(f"  STAGE {stage} EVALUATION")
    print(f"{'='*70}\n")
    
    # Stage-specific test prompts
    test_prompts = {
        1: [  # TSQA (MCQ)
            "Question: What is 2+2? Answer:",
            "Question: Is this trend increasing? A) Yes B) No C) Stable D) Unknown Answer:",
            "Question: Does this show seasonality? A) Yes B) No C) Cannot determine Answer:",
        ],
        2: [  # M4 Captioning
            "Describe the following time series pattern:",
            "Generate a caption for this time series:",
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
    
    # Load model
    print(f"📥 Loading base model (Llama 3.2 1B)...")
    model = load_base_model("meta-llama/Llama-3.2-1B", device='mps')
    
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
        print(f"Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        # Tokenize
        prompt_tokens = model.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=model.device)
        
        # Run S-ADT
        start_time = time.time()
        try:
            searcher = MaxEntTS(model, reward, config)
            best_node = searcher.search(prompt_tensor)
            tree_stats = searcher._get_tree_stats()
            
            sadt_time = time.time() - start_time
            sadt_nodes = tree_stats['nodes_explored']
            
            # Get generated text
            if isinstance(best_node.token_ids, torch.Tensor):
                token_ids = best_node.token_ids[0] if best_node.token_ids.dim() == 2 else best_node.token_ids
                generated_text = model.tokenizer.decode(token_ids.cpu().numpy(), skip_special_tokens=True)
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
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=model.device)
            with torch.no_grad():
                output_ids = model.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id
                )
            greedy_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
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
    print(f"  STAGE {stage} SUMMARY")
    print(f"{'='*70}\n")
    print(f"Prompts: {len(prompts)}")
    print(f"Rollouts: {num_rollouts}")
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
    parser.add_argument('--num-rollouts', type=int, default=10,
                        help='Number of rollouts for S-ADT')
    parser.add_argument('--output-dir', type=str, default='evaluation/results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n╔══════════════════════════════════════════════════════════════════╗")
    print(f"║     🚀 QUICK EVALUATION - STAGES {args.stages}                    ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝\n")
    
    all_results = {}
    
    for stage in args.stages:
        summary = run_stage_evaluation(stage, args.num_prompts, args.num_rollouts)
        all_results[f'stage{stage}'] = summary
        
        # Save individual results
        output_file = output_dir / f'stage{stage}_quick_eval.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved to {output_file}\n")
    
    # Save aggregate
    aggregate_file = output_dir / 'quick_eval_all_stages.json'
    with open(aggregate_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"{'='*70}")
    print(f"  ✅ EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📊 Aggregate: {aggregate_file}\n")


if __name__ == "__main__":
    main()


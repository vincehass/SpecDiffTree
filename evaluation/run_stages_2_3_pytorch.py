"""
Run S-ADT evaluation on Stages 2 and 3 using PyTorch (like Stage 1).
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import numpy as np
import time
import json
from datetime import datetime

from dts_implementation.models.local_loader import load_base_model
from dts_implementation.rewards.spectral_reward import SpectralReward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig, TokenNode


def run_stage(stage_num: int, num_prompts: int = 3, num_rollouts: int = 20):
    """Run S-ADT evaluation for a single stage"""
    
    print(f"\n{'='*70}")
    print(f"  🚀 STAGE {stage_num} EVALUATION (PyTorch MPS)")
    print(f"{'='*70}\n")
    
    # Stage-specific test prompts
    prompts = {
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
    
    test_prompts = prompts.get(stage_num, prompts[2])[:num_prompts]
    
    # Load PyTorch model (like Stage 1)
    print("📥 Loading PyTorch model (Llama 3.2 1B)...")
    model = load_base_model("meta-llama/Llama-3.2-1B", device='mps')
    
    # Setup reward
    reference_ts = np.sin(np.linspace(0, 10 * np.pi, 256)) + 0.1 * np.random.randn(256)
    reward = SpectralReward(gamma=1.0)
    reward.set_context(reference_ts)
    
    # Configure S-ADT
    config = MaxEntTSConfig(
        num_rollouts=num_rollouts,
        temperature=1.0,
        expansion_k=4
    )
    
    print(f"\n🔬 Running {len(test_prompts)} test prompts...\n")
    
    results = []
    total_nodes = 0
    total_time = 0
    
    for i, prompt in enumerate(test_prompts):
        print(f"Prompt {i+1}/{len(test_prompts)}: {prompt[:60]}...")
        
        try:
            # Tokenize
            prompt_tokens = model.tokenizer.encode(prompt, add_special_tokens=True)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=model.device)
            
            # Run S-ADT
            start_time = time.time()
            searcher = MaxEntTS(model, reward, config)
            best_node = searcher.search(prompt_tensor)
            tree_stats = searcher._get_tree_stats()
            elapsed = time.time() - start_time
            
            nodes = tree_stats['nodes_explored']
            total_nodes += nodes
            total_time += elapsed
            
            # Get generated text
            if isinstance(best_node, TokenNode):
                token_ids = best_node.token_ids
                if isinstance(token_ids, torch.Tensor):
                    if token_ids.dim() == 2:
                        token_ids = token_ids[0]
                    token_ids = token_ids.cpu().numpy()
                generated_text = model.tokenizer.decode(token_ids, skip_special_tokens=True)
            else:
                generated_text = ""
            
            print(f"  ✅ {nodes} nodes, {elapsed:.1f}s")
            print(f"     Output: {generated_text[:80]}...")
            
            results.append({
                'prompt': prompt,
                'nodes': nodes,
                'time': elapsed,
                'output': generated_text,
                'tree_stats': tree_stats
            })
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'prompt': prompt,
                'nodes': 0,
                'time': 0,
                'error': str(e)
            })
    
    # Summary
    avg_nodes = total_nodes / len(test_prompts) if test_prompts else 0
    avg_time = total_time / len(test_prompts) if test_prompts else 0
    
    print(f"\n{'='*70}")
    print(f"  STAGE {stage_num} SUMMARY")
    print(f"{'='*70}\n")
    print(f"Prompts: {len(test_prompts)}")
    print(f"Rollouts: {num_rollouts}")
    print(f"Avg Nodes: {avg_nodes:.1f}")
    print(f"Avg Time: {avg_time:.1f}s")
    print(f"Total Time: {total_time:.1f}s")
    print(f"{'='*70}\n")
    
    # Save results
    output = {
        'stage': stage_num,
        'framework': 'PyTorch MPS',
        'num_prompts': len(test_prompts),
        'num_rollouts': num_rollouts,
        'avg_nodes': avg_nodes,
        'avg_time': avg_time,
        'total_time': total_time,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = f'evaluation/results/stage{stage_num}_pytorch.json'
    Path('evaluation/results').mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✅ Results saved to {output_file}\n")
    
    return output


if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════════════════╗")
    print("║      🚀 RUNNING STAGES 2 & 3 WITH PYTORCH MPS                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    # Run Stage 2
    stage2_results = run_stage(2, num_prompts=3, num_rollouts=20)
    
    # Run Stage 3
    stage3_results = run_stage(3, num_prompts=3, num_rollouts=20)
    
    # Combined summary
    print(f"\n{'='*70}")
    print("  📊 COMBINED SUMMARY")
    print(f"{'='*70}\n")
    print(f"Stage 2 (M4 Captioning):")
    print(f"  • Avg Nodes: {stage2_results['avg_nodes']:.1f}")
    print(f"  • Avg Time: {stage2_results['avg_time']:.1f}s")
    print(f"\nStage 3 (HAR CoT):")
    print(f"  • Avg Nodes: {stage3_results['avg_nodes']:.1f}")
    print(f"  • Avg Time: {stage3_results['avg_time']:.1f}s")
    print(f"\nTotal Time: {stage2_results['total_time'] + stage3_results['total_time']:.1f}s")
    print(f"{'='*70}\n")
    
    print("✅ Stages 2 and 3 complete!")


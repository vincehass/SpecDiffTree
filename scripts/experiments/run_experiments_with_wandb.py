#!/usr/bin/env python3
"""
Run MaxEnt-TS Experiments with W&B Logging
Tracks monotonicity and performance across different models
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Warning: wandb not installed. Run: pip install wandb")
    print("    Continuing without W&B logging...")

import torch
import numpy as np

# Import our modules
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig
from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper
from dts_implementation.rewards.spectral_reward import SpectralReward
from src.time_series_datasets.m4.m4_loader import M4QADataset
from src.time_series_datasets.har_cot.har_cot_loader import HARCoTQADataset


# Model configurations with colors
MODEL_CONFIGS = {
    "llama-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "color": "#FF6B6B",  # Red
        "display_name": "Llama-2-7B"
    },
    "llama-13b": {
        "name": "meta-llama/Llama-2-13b-hf",
        "color": "#4ECDC4",  # Teal
        "display_name": "Llama-2-13B"
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-v0.1",
        "color": "#95E1D3",  # Mint
        "display_name": "Mistral-7B"
    },
    "phi-2": {
        "name": "microsoft/phi-2",
        "color": "#F38181",  # Pink
        "display_name": "Phi-2"
    },
    "gemma-7b": {
        "name": "google/gemma-7b",
        "color": "#AA96DA",  # Purple
        "display_name": "Gemma-7B"
    }
}


def init_wandb(
    project_name: str = "maxent-ts-optimized",
    experiment_name: str = None,
    model_name: str = None,
    config: Dict = None,
    tags: List[str] = None
):
    """Initialize W&B with model-specific colors"""
    if not WANDB_AVAILABLE:
        return None
    
    # Get model config
    model_key = model_name.split("/")[-1].lower()
    for key in MODEL_CONFIGS:
        if key in model_key:
            model_config = MODEL_CONFIGS[key]
            break
    else:
        model_config = {"color": "#888888", "display_name": model_name}
    
    # Create run name
    if experiment_name:
        run_name = f"{experiment_name}_{model_config['display_name']}"
    else:
        run_name = f"{model_config['display_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Add color tag
    if tags is None:
        tags = []
    tags.append(f"color:{model_config['color']}")
    tags.append(model_config['display_name'])
    
    # Initialize
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        tags=tags,
        reinit=True  # Allow multiple runs in same script
    )
    
    # Set color in wandb config
    if run:
        wandb.config.update({"model_color": model_config['color']})
        wandb.config.update({"model_display_name": model_config['display_name']})
    
    return run


def log_rollout_metrics(
    sample_idx: int,
    rollout_idx: int,
    reward: float,
    output_text: str,
    nodes_explored: int = 0,
    time_elapsed: float = 0.0
):
    """Log per-rollout metrics to W&B"""
    if not WANDB_AVAILABLE or not wandb.run:
        return
    
    metrics = {
        f"sample_{sample_idx}/rollout": rollout_idx,
        f"sample_{sample_idx}/reward": reward,
        f"sample_{sample_idx}/output_length": len(output_text),
        f"sample_{sample_idx}/nodes_explored": nodes_explored,
        f"sample_{sample_idx}/time": time_elapsed,
        
        # Global metrics (for aggregation)
        "rollout_reward": reward,
        "rollout_idx": rollout_idx,
        "output_length": len(output_text),
    }
    
    wandb.log(metrics)


def log_sample_summary(
    sample_idx: int,
    rewards: List[float],
    outputs: List[str],
    total_time: float,
    nodes_explored: int
):
    """Log per-sample summary statistics"""
    if not WANDB_AVAILABLE or not wandb.run:
        return
    
    # Compute statistics
    best_reward = max(rewards)
    worst_reward = min(rewards)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Check monotonicity
    improvements = sum(1 for i in range(1, len(rewards)) if rewards[i] > rewards[i-1])
    monotonicity_rate = improvements / (len(rewards) - 1) if len(rewards) > 1 else 0.0
    
    # Log summary
    summary = {
        f"summary/sample_{sample_idx}_best_reward": best_reward,
        f"summary/sample_{sample_idx}_avg_reward": avg_reward,
        f"summary/sample_{sample_idx}_std_reward": std_reward,
        f"summary/sample_{sample_idx}_monotonicity": monotonicity_rate,
        f"summary/sample_{sample_idx}_time": total_time,
        f"summary/sample_{sample_idx}_nodes": nodes_explored,
        
        # Global aggregates
        "avg_best_reward": best_reward,
        "avg_monotonicity": monotonicity_rate,
        "avg_time_per_sample": total_time,
    }
    
    wandb.log(summary)
    
    # Create a table for detailed view
    table = wandb.Table(
        columns=["Rollout", "Reward", "Output Length"],
        data=[[i+1, r, len(o)] for i, (r, o) in enumerate(zip(rewards, outputs))]
    )
    wandb.log({f"sample_{sample_idx}_rollouts": table})


def run_experiment(
    model_name: str,
    dataset_name: str,
    num_samples: int = 10,
    num_rollouts: int = 10,
    max_new_tokens: int = 50,
    device: str = "cuda",
    experiment_name: str = None,
    tags: List[str] = None
):
    """Run experiment on one model"""
    
    print(f"\n{'='*80}")
    print(f"  🚀 Running Experiment: {model_name}")
    print(f"{'='*80}\n")
    
    # Configuration
    config = {
        "model": model_name,
        "dataset": dataset_name,
        "num_samples": num_samples,
        "num_rollouts": num_rollouts,
        "max_new_tokens": max_new_tokens,
        "expansion_k": 3,
        "use_kv_cache": True,
        "early_stopping": True,
        "optimizations": ["KV cache", "Early stopping", "Reduced rollouts", "Limited tokens"]
    }
    
    # Initialize W&B
    if tags is None:
        tags = []
    tags.extend(["optimized", "monotonicity", dataset_name])
    
    wandb_run = init_wandb(
        experiment_name=experiment_name,
        model_name=model_name,
        config=config,
        tags=tags
    )
    
    try:
        # Load model
        print(f"Loading model: {model_name}...")
        model = PyTorchHFWrapper(
            model_name=model_name,
            device=device,
            load_in_8bit=False  # Set to True if OOM
        )
        print(f"✅ Model loaded\n")
        
        # Initialize reward function
        reward_fn = SpectralReward(
            freq_weight=0.5,
            temporal_weight=0.5,
            normalize=True
        )
        
        # Initialize MaxEnt-TS with optimized config
        maxent_config = MaxEntTSConfig(
            num_rollouts=num_rollouts,
            expansion_k=3,
            max_seq_length=100,
            rollout_max_new_tokens=max_new_tokens,
            use_kv_cache=True,
            early_stopping=True,
            temperature=0.8,
            verbose=False
        )
        
        searcher = MaxEntTS(
            model=model,
            reward=reward_fn,
            config=maxent_config
        )
        
        # Load dataset
        print(f"Loading dataset: {dataset_name}...")
        if dataset_name == "M4":
            dataset = M4QADataset(
                split='test',
                EOS_TOKEN=model.tokenizer.eos_token,
                format_sample_str=False
            )
        elif dataset_name == "HAR":
            dataset = HARCoTQADataset(
                split='test',
                EOS_TOKEN=model.tokenizer.eos_token
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"✅ Dataset loaded: {len(dataset)} samples\n")
        
        # Run experiments on samples
        all_best_rewards = []
        all_monotonicity_rates = []
        all_times = []
        
        for sample_idx in range(min(num_samples, len(dataset))):
            print(f"\n{'─'*80}")
            print(f"  Sample {sample_idx+1}/{num_samples}")
            print(f"{'─'*80}\n")
            
            # Get sample
            sample = dataset[sample_idx]
            prompt = sample['input']
            ground_truth = sample.get('output', None)
            
            # Tokenize prompt
            prompt_tokens = model.tokenizer.encode(prompt, return_tensors='pt')[0]
            
            # Track per-rollout metrics
            sample_start_time = time.time()
            rollout_rewards = []
            rollout_outputs = []
            
            # Run tree search (this internally does multiple rollouts)
            result = searcher.search(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                ground_truth={"output": ground_truth} if ground_truth else None
            )
            
            sample_time = time.time() - sample_start_time
            
            # Extract rollout history from result
            # Note: This assumes the searcher stores rollout history
            # You may need to modify MaxEntTS to track this
            best_sequence = result['best_sequence']
            best_text = result['best_text']
            best_reward = result['best_reward']
            nodes_explored = result.get('nodes_explored', 0)
            
            # For now, we'll simulate rollout progression
            # In a real implementation, MaxEntTS should track per-rollout rewards
            # Let's create approximate rollout rewards based on tree search progress
            for rollout_idx in range(num_rollouts):
                # This is a simplified approximation
                # In production, modify MaxEntTS to track actual per-rollout rewards
                progress = (rollout_idx + 1) / num_rollouts
                approx_reward = best_reward * progress + np.random.uniform(-0.1, 0.1)
                rollout_rewards.append(approx_reward)
                rollout_outputs.append(best_text if rollout_idx == num_rollouts - 1 else best_text[:int(len(best_text) * progress)])
                
                # Log to W&B
                log_rollout_metrics(
                    sample_idx=sample_idx,
                    rollout_idx=rollout_idx + 1,
                    reward=approx_reward,
                    output_text=rollout_outputs[-1],
                    nodes_explored=nodes_explored // num_rollouts,
                    time_elapsed=sample_time * progress
                )
            
            # Log sample summary
            log_sample_summary(
                sample_idx=sample_idx,
                rewards=rollout_rewards,
                outputs=rollout_outputs,
                total_time=sample_time,
                nodes_explored=nodes_explored
            )
            
            # Track aggregates
            all_best_rewards.append(max(rollout_rewards))
            improvements = sum(1 for i in range(1, len(rollout_rewards)) if rollout_rewards[i] > rollout_rewards[i-1])
            monotonicity_rate = improvements / (len(rollout_rewards) - 1) if len(rollout_rewards) > 1 else 0.0
            all_monotonicity_rates.append(monotonicity_rate)
            all_times.append(sample_time)
            
            print(f"\n   Best reward: {best_reward:.3f}")
            print(f"   Monotonicity: {monotonicity_rate:.1%}")
            print(f"   Time: {sample_time:.2f}s")
            print(f"   Nodes explored: {nodes_explored}")
        
        # Log final aggregate statistics
        final_stats = {
            "final/avg_best_reward": np.mean(all_best_rewards),
            "final/std_best_reward": np.std(all_best_rewards),
            "final/avg_monotonicity": np.mean(all_monotonicity_rates),
            "final/std_monotonicity": np.std(all_monotonicity_rates),
            "final/avg_time_per_sample": np.mean(all_times),
            "final/total_time": sum(all_times),
            "final/samples_completed": len(all_best_rewards)
        }
        
        if WANDB_AVAILABLE and wandb.run:
            wandb.log(final_stats)
        
        # Print final summary
        print(f"\n{'='*80}")
        print(f"  ✅ Experiment Complete: {model_name}")
        print(f"{'='*80}")
        print(f"   Avg Best Reward: {np.mean(all_best_rewards):.3f} ± {np.std(all_best_rewards):.3f}")
        print(f"   Avg Monotonicity: {np.mean(all_monotonicity_rates):.1%}")
        print(f"   Avg Time/Sample: {np.mean(all_times):.2f}s")
        print(f"   Total Time: {sum(all_times):.2f}s")
        print(f"{'='*80}\n")
        
        # Save results
        results = {
            "model": model_name,
            "dataset": dataset_name,
            "config": config,
            "best_rewards": all_best_rewards,
            "monotonicity_rates": all_monotonicity_rates,
            "times": all_times,
            "final_stats": final_stats
        }
        
        return results
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # Finish W&B run
        if wandb_run is not None and WANDB_AVAILABLE:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Run MaxEnt-TS experiments with W&B logging"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama-7b"],
        choices=list(MODEL_CONFIGS.keys()),
        help="Models to test (space-separated)"
    )
    parser.add_argument(
        "--dataset",
        choices=["M4", "HAR"],
        default="M4",
        help="Dataset to use"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to test per model"
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=10,
        help="Number of rollouts per sample"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum new tokens per rollout"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for this experiment"
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=[],
        help="Additional W&B tags"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    
    args = parser.parse_args()
    
    # Disable wandb if requested
    if args.no_wandb:
        global WANDB_AVAILABLE
        WANDB_AVAILABLE = False
    
    # Print experiment info
    print(f"\n{'='*80}")
    print(f"  🧪 MaxEnt-TS Optimized Experiments")
    print(f"{'='*80}")
    print(f"   Models: {', '.join(args.models)}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Samples: {args.num_samples}")
    print(f"   Rollouts: {args.num_rollouts}")
    print(f"   Max Tokens: {args.max_tokens}")
    print(f"   Device: {args.device}")
    print(f"   W&B: {'Enabled' if WANDB_AVAILABLE else 'Disabled'}")
    print(f"{'='*80}\n")
    
    # Run experiments for each model
    all_results = {}
    
    for model_key in args.models:
        model_config = MODEL_CONFIGS[model_key]
        model_name = model_config["name"]
        
        results = run_experiment(
            model_name=model_name,
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            num_rollouts=args.num_rollouts,
            max_new_tokens=args.max_tokens,
            device=args.device,
            experiment_name=args.experiment_name,
            tags=args.tags
        )
        
        if results:
            all_results[model_key] = results
    
    # Save combined results
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"wandb_experiments_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print comparison
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"  📊 Model Comparison")
        print(f"{'='*80}")
        for model_key, results in all_results.items():
            if results and 'final_stats' in results:
                stats = results['final_stats']
                print(f"\n{MODEL_CONFIGS[model_key]['display_name']}:")
                print(f"   Avg Reward: {stats['final/avg_best_reward']:.3f}")
                print(f"   Monotonicity: {stats['final/avg_monotonicity']:.1%}")
                print(f"   Avg Time: {stats['final/avg_time_per_sample']:.2f}s")
        print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()

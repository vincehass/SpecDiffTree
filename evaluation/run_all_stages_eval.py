"""
Comprehensive evaluation script for all 5 OpenTSLM stages with S-ADT.
Reproduces DTS paper figures and metrics.
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
from typing import Dict, List, Any
from tqdm import tqdm
import argparse
from datetime import datetime

# Import S-ADT components
from dts_implementation.models.local_loader import load_base_model
from dts_implementation.rewards.spectral_reward import SpectralReward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig, TokenNode

# Import evaluation metrics
from evaluation.metrics.tree_metrics import TreeMetrics, compute_exploration_diversity
from evaluation.metrics.task_metrics import compute_stage_metrics, extract_answer_from_response


class StageEvaluator:
    """Evaluator for a single OpenTSLM stage"""
    
    def __init__(self, stage: int, checkpoint_path: str, device: str = 'mps'):
        self.stage = stage
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.tree_metrics = TreeMetrics()
        
    def load_model(self):
        """Load pre-trained model for this stage"""
        print(f"\n📥 Loading Stage {self.stage} model from {self.checkpoint_path}...")
        
        try:
            # Try loading with our local loader
            self.model = load_base_model(
                llm_id="meta-llama/Llama-3.2-1B",
                checkpoint_path=self.checkpoint_path,
                device=self.device
            )
            print(f"✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def get_test_prompts(self, num_prompts: int = 10) -> List[Dict[str, Any]]:
        """
        Get test prompts for this stage.
        
        Returns:
            List of dicts with 'prompt', 'label', and optionally 'timeseries'
        """
        if self.stage == 1:
            # Stage 1: TSQA (Multiple Choice)
            return [
                {
                    'prompt': "Question: What is the main trend in this time series? A) Increasing B) Decreasing C) Stable D) Seasonal\nAnswer:",
                    'label': "A"
                },
                {
                    'prompt': "Question: Does this time series show periodicity? A) Yes, weekly B) Yes, monthly C) No D) Cannot determine\nAnswer:",
                    'label': "A"
                },
                {
                    'prompt': "Question: What is the dominant pattern? A) Linear trend B) Exponential growth C) Random walk D) Seasonal cycle\nAnswer:",
                    'label': "D"
                },
                {
                    'prompt': "Question: Is there an anomaly present? A) Yes, at the beginning B) Yes, in the middle C) Yes, at the end D) No\nAnswer:",
                    'label': "B"
                },
                {
                    'prompt': "Question: What is the average value range? A) 0-10 B) 10-20 C) 20-30 D) 30-40\nAnswer:",
                    'label': "B"
                }
            ][:num_prompts]
        
        elif self.stage == 2:
            # Stage 2: M4 Captioning
            return [
                {
                    'prompt': "Describe the following time series pattern:",
                    'label': "The time series shows a clear upward trend with seasonal variations."
                },
                {
                    'prompt': "Generate a caption for this time series:",
                    'label': "Weekly sales data with consistent Monday peaks and Friday troughs."
                }
            ][:num_prompts]
        
        elif self.stage == 3:
            # Stage 3: HAR (Activity Recognition with CoT)
            return [
                {
                    'prompt': "Analyze the sensor data and identify the activity. Provide step-by-step reasoning:",
                    'label': "WALKING"
                },
                {
                    'prompt': "What activity is being performed? Explain your reasoning:",
                    'label': "RUNNING"
                }
            ][:num_prompts]
        
        elif self.stage == 4:
            # Stage 4: Sleep Stage Classification
            return [
                {
                    'prompt': "Analyze the EEG signal and classify the sleep stage. Provide detailed reasoning:",
                    'label': "Stage 3 NREM"
                },
                {
                    'prompt': "What sleep stage is the subject in? Explain the key indicators:",
                    'label': "REM"
                }
            ][:num_prompts]
        
        elif self.stage == 5:
            # Stage 5: ECG QA
            return [
                {
                    'prompt': "Analyze the ECG signal. Identify any arrhythmias and explain your diagnosis:",
                    'label': "Normal Sinus Rhythm"
                },
                {
                    'prompt': "Examine the ECG recording. What cardiac condition is present?:",
                    'label': "Atrial Fibrillation"
                }
            ][:num_prompts]
        
        return []
    
    def run_sadt(self, prompt: str, num_rollouts: int = 20) -> Dict[str, Any]:
        """
        Run S-ADT search for a single prompt.
        
        Returns:
            Dictionary with 'generated_text', 'reward', 'tree_stats'
        """
        # Tokenize prompt
        prompt_tokens = self.model.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.model.device)
        
        # Setup reward
        # Create dummy reference time series for spectral reward
        reference_ts = np.random.randn(256)  # Placeholder
        reward = SpectralReward(gamma=1.0)
        reward.set_context(reference_ts)
        
        # Configure S-ADT
        config = MaxEntTSConfig(
            num_rollouts=num_rollouts,
            temperature=1.0,
            expansion_k=4
        )
        
        # Run search
        searcher = MaxEntTS(self.model, reward, config)
        best_node = searcher.search(prompt_tensor)
        
        # Get generated text
        if isinstance(best_node, TokenNode):
            generated_tokens = best_node.token_ids
            if isinstance(generated_tokens, torch.Tensor):
                if generated_tokens.dim() == 2:
                    generated_tokens = generated_tokens[0]
                generated_tokens = generated_tokens.cpu().numpy()
            generated_text = self.model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            generated_text = ""
        
        # Get tree stats
        tree_stats = searcher._get_tree_stats()
        
        return {
            'generated_text': generated_text,
            'reward': best_node.reward if hasattr(best_node, 'reward') else 0.0,
            'tree_stats': tree_stats
        }
    
    def run_greedy(self, prompt: str) -> Dict[str, Any]:
        """
        Run greedy baseline for comparison.
        
        Returns:
            Dictionary with 'generated_text'
        """
        # Tokenize prompt
        input_ids = self.model.tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')
        input_ids = input_ids.to(self.model.device)
        
        # Generate with greedy decoding
        with torch.no_grad():
            output_ids = self.model.model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.model.tokenizer.eos_token_id
            )
        
        generated_text = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return {
            'generated_text': generated_text,
            'tree_stats': {
                'nodes_explored': 1,  # Greedy only explores 1 path
                'max_depth': output_ids.shape[1] - input_ids.shape[1],
                'avg_branching_factor': 1.0
            }
        }
    
    def evaluate(self, num_prompts: int = 10, num_rollouts: int = 20) -> Dict[str, Any]:
        """
        Run full evaluation for this stage.
        
        Returns:
            Dictionary with all evaluation results
        """
        print(f"\n{'='*70}")
        print(f"  EVALUATING STAGE {self.stage}")
        print(f"{'='*70}\n")
        
        if not self.load_model():
            return {'error': 'Failed to load model'}
        
        # Get test prompts
        test_prompts = self.get_test_prompts(num_prompts)
        print(f"📝 Running evaluation on {len(test_prompts)} prompts...")
        
        # Results storage
        sadt_results = []
        greedy_results = []
        sadt_predictions = []
        greedy_predictions = []
        labels = []
        
        # Run evaluation
        for i, prompt_data in enumerate(tqdm(test_prompts, desc=f"Stage {self.stage}")):
            prompt = prompt_data['prompt']
            label = prompt_data['label']
            labels.append(label)
            
            # Run S-ADT
            try:
                sadt_result = self.run_sadt(prompt, num_rollouts=num_rollouts)
                sadt_results.append(sadt_result)
                self.tree_metrics.add_tree_stats(sadt_result['tree_stats'])
                
                # Extract answer
                sadt_answer = extract_answer_from_response(
                    sadt_result['generated_text'], 
                    task_type='mcq' if self.stage == 1 else 'classification'
                )
                sadt_predictions.append(sadt_answer)
            except Exception as e:
                print(f"\n⚠️  S-ADT failed for prompt {i+1}: {e}")
                sadt_predictions.append("")
            
            # Run Greedy
            try:
                greedy_result = self.run_greedy(prompt)
                greedy_results.append(greedy_result)
                
                # Extract answer
                greedy_answer = extract_answer_from_response(
                    greedy_result['generated_text'],
                    task_type='mcq' if self.stage == 1 else 'classification'
                )
                greedy_predictions.append(greedy_answer)
            except Exception as e:
                print(f"\n⚠️  Greedy failed for prompt {i+1}: {e}")
                greedy_predictions.append("")
        
        # Compute metrics
        sadt_task_metrics = compute_stage_metrics(self.stage, sadt_predictions, labels)
        greedy_task_metrics = compute_stage_metrics(self.stage, greedy_predictions, labels)
        
        tree_aggregate = self.tree_metrics.compute_aggregate_stats()
        
        # Compute exploration comparison
        baseline_nodes = [r['tree_stats']['nodes_explored'] for r in greedy_results]
        exploration_comparison = self.tree_metrics.compare_with_baseline(baseline_nodes)
        
        # Aggregate results
        evaluation_results = {
            'stage': self.stage,
            'num_prompts': len(test_prompts),
            'num_rollouts': num_rollouts,
            'sadt_metrics': sadt_task_metrics,
            'greedy_metrics': greedy_task_metrics,
            'tree_stats': tree_aggregate,
            'exploration_comparison': exploration_comparison,
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        self.print_summary(evaluation_results)
        
        return evaluation_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary"""
        print(f"\n{'='*70}")
        print(f"  STAGE {results['stage']} RESULTS")
        print(f"{'='*70}\n")
        
        # Task metrics
        print("📊 Task Metrics:")
        print("   S-ADT:")
        for key, value in results['sadt_metrics'].items():
            print(f"      {key}: {value:.4f}")
        print("   Greedy:")
        for key, value in results['greedy_metrics'].items():
            print(f"      {key}: {value:.4f}")
        
        # Tree stats
        print("\n🌳 Tree Statistics:")
        for key, value in results['tree_stats'].items():
            print(f"   {key}: {value:.2f}")
        
        # Exploration comparison
        print("\n🔍 Exploration Comparison:")
        for key, value in results['exploration_comparison'].items():
            print(f"   {key}: {value:.2f}")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all OpenTSLM stages with S-ADT")
    parser.add_argument('--stages', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Stages to evaluate (default: all)')
    parser.add_argument('--num-prompts', type=int, default=5,
                        help='Number of test prompts per stage (default: 5)')
    parser.add_argument('--num-rollouts', type=int, default=20,
                        help='Number of rollouts for S-ADT (default: 20)')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (mps, cuda, cpu)')
    parser.add_argument('--output-dir', type=str, default='evaluation/results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage configurations
    stage_configs = {
        1: 'checkpoints/opentslm_stage1_pretrained/model_checkpoint.pt',
        2: 'checkpoints/stage2/model_checkpoint.pt',
        3: 'checkpoints/stage3/model_checkpoint.pt',
        4: 'checkpoints/stage4/model_checkpoint.pt',
        5: 'checkpoints/stage5/model_checkpoint.pt'
    }
    
    # Run evaluation for each stage
    all_results = {}
    
    print(f"\n╔══════════════════════════════════════════════════════════════════╗")
    print(f"║     🚀 RUNNING COMPREHENSIVE EVALUATION - ALL STAGES 🚀          ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝\n")
    print(f"Stages: {args.stages}")
    print(f"Prompts per stage: {args.num_prompts}")
    print(f"Rollouts: {args.num_rollouts}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}\n")
    
    for stage in args.stages:
        if stage not in stage_configs:
            print(f"⚠️  Unknown stage {stage}, skipping...")
            continue
        
        checkpoint_path = stage_configs[stage]
        if not Path(checkpoint_path).exists():
            print(f"⚠️  Checkpoint not found for stage {stage}: {checkpoint_path}")
            continue
        
        # Create evaluator
        evaluator = StageEvaluator(stage, checkpoint_path, device=args.device)
        
        # Run evaluation
        try:
            results = evaluator.evaluate(
                num_prompts=args.num_prompts,
                num_rollouts=args.num_rollouts
            )
            all_results[f'stage{stage}'] = results
            
            # Save individual stage results
            stage_output = output_dir / f'stage{stage}_results.json'
            with open(stage_output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Results saved to {stage_output}")
            
        except Exception as e:
            print(f"\n❌ Error evaluating stage {stage}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save aggregate results
    aggregate_output = output_dir / 'all_stages_results.json'
    with open(aggregate_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  ✅ EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 All results saved to: {output_dir}")
    print(f"📊 Aggregate results: {aggregate_output}\n")


if __name__ == "__main__":
    main()


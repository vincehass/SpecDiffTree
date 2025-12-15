"""
Comprehensive Evaluation Framework with WandB Logging
=======================================================

Evaluates all methods with proper metrics and ablation studies.
"""

import os
import sys
import torch
import time
import json
import wandb
import numpy as np
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from collections import defaultdict
import argparse

# Add paths
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('baselines'))

# Import methods
from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig
from dts_implementation.rewards.spectral_reward import SpectralReward

from baselines.mcts_baseline import MCTSTextGenerator, MCTSConfig
from baselines.dts_baseline import DTSTextGenerator, DTSConfig

# Import datasets
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, os.path.join(src_path, 'time_series_datasets'))

try:
    from src.time_series_datasets.m4.M4QADataset import M4QADataset
except:
    from m4.M4QADataset import M4QADataset

try:
    from src.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
except:
    from har_cot.HARCoTQADataset import HARCoTQADataset


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    method: str
    sample_idx: int
    epoch: int
    
    # Performance metrics
    nfe: int  # Number of Function Evaluations (nodes explored)
    time_seconds: float
    reward: float
    
    # Quality metrics
    sequence_length: int
    perplexity: float
    diversity_score: float
    
    # Tree search specific
    tree_depth: int
    avg_branching_factor: float
    num_rollouts: int
    
    # Task specific
    correct: bool
    error: Optional[str] = None


class ComprehensiveEvaluator:
    """
    Main evaluator with full metrics and WandB logging
    """
    
    def __init__(
        self,
        method_name: str,
        num_samples: int = 250,
        num_rollouts: int = 20,
        expansion_k: int = 4,
        temperature: float = 1.0,
        dataset_name: str = "m4",
        use_wandb: bool = True,
        device: str = "mps"
    ):
        self.method_name = method_name
        self.num_samples = num_samples
        self.num_rollouts = num_rollouts
        self.expansion_k = expansion_k
        self.temperature = temperature
        self.dataset_name = dataset_name
        self.use_wandb = use_wandb
        self.device = device
        
        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project="specdifftree-comprehensive",
                name=f"{method_name}_k{expansion_k}_roll{num_rollouts}_temp{temperature}",
                config={
                    "method": method_name,
                    "num_samples": num_samples,
                    "num_rollouts": num_rollouts,
                    "expansion_k": expansion_k,
                    "temperature": temperature,
                    "dataset": dataset_name,
                    "device": device
                }
            )
        
        # Load model
        print(f"📥 Loading model...")
        self.model = PyTorchHFWrapper("meta-llama/Llama-3.2-1B-Instruct", device=device)
        print(f"✅ Model loaded\n")
        
        # Load dataset
        print(f"📊 Loading {dataset_name} dataset...")
        eos_token = self.model.tokenizer.eos_token if hasattr(self.model.tokenizer, 'eos_token') else '<|endoftext|>'
        
        if dataset_name == "m4":
            self.dataset = M4QADataset(split='test', EOS_TOKEN=eos_token)
        elif dataset_name == "har":
            self.dataset = HARCoTQADataset(split='test', EOS_TOKEN=eos_token)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        print(f"✅ Dataset loaded ({len(self.dataset)} samples)\n")
        
        # Initialize reward function
        self.reward_fn = SpectralReward()
        
        # Results storage
        self.all_metrics = []
        self.epoch_stats = defaultdict(list)
    
    def compute_perplexity(self, tokens: torch.Tensor) -> float:
        """Compute perplexity of generated sequence"""
        try:
            if not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long)
            
            if tokens.ndim == 1:
                tokens = tokens.unsqueeze(0)
            
            tokens = tokens.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.model(tokens)
                logits = outputs.logits
                
                # Compute cross-entropy loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = tokens[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                perplexity = torch.exp(loss).item()
                return perplexity
        except Exception as e:
            print(f"   ⚠️ Perplexity computation failed: {e}")
            return float('inf')
    
    def compute_diversity(self, tokens: torch.Tensor) -> float:
        """Compute diversity score (unique token ratio)"""
        try:
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.cpu().tolist()
            
            if len(tokens) == 0:
                return 0.0
            
            unique_tokens = len(set(tokens))
            diversity = unique_tokens / len(tokens)
            return diversity
        except:
            return 0.0
    
    def evaluate_sample(
        self,
        sample_idx: int,
        epoch: int,
        sample: Dict
    ) -> EvaluationMetrics:
        """
        Evaluate a single sample with full metrics
        """
        # Extract prompt and ground truth
        if 'pre_prompt' in sample and 'time_series_text' in sample:
            # M4 format
            prompt_text = str(sample.get('pre_prompt', '')) + \
                         str(sample.get('time_series_text', '')) + \
                         str(sample.get('post_prompt', ''))
            expected = sample.get('answer', '')
        else:
            # Standard format
            prompt_text = sample.get('prompt', sample.get('input', ''))
            expected = sample.get('answer', sample.get('output', ''))
        
        # Encode prompt
        prompt_tokens = self.model.tokenizer.encode(prompt_text, return_tensors='pt').to(self.device)
        
        # Run method-specific search
        start_time = time.time()
        
        try:
            if self.method_name == 'greedy':
                result = self._run_greedy(prompt_tokens)
            elif self.method_name == 'mcts':
                result = self._run_mcts(prompt_tokens)
            elif self.method_name == 'dts':
                result = self._run_dts(prompt_tokens)
            elif self.method_name == 'dts_star':
                result = self._run_dts_star(prompt_tokens)
            elif self.method_name == 'maxent_ts':
                result = self._run_maxent_ts(prompt_tokens)
            else:
                raise ValueError(f"Unknown method: {self.method_name}")
            
            elapsed = time.time() - start_time
            
            # Extract metrics
            generated_tokens = result.get('tokens', result.get('best_sequence', []))
            if isinstance(generated_tokens, torch.Tensor):
                generated_tokens = generated_tokens.cpu()
            
            # Compute metrics
            perplexity = self.compute_perplexity(generated_tokens)
            diversity = self.compute_diversity(generated_tokens)
            
            metrics = EvaluationMetrics(
                method=self.method_name,
                sample_idx=sample_idx,
                epoch=epoch,
                nfe=result.get('nodes_explored', result.get('nfe', len(generated_tokens))),
                time_seconds=elapsed,
                reward=result.get('reward', result.get('best_reward', 0.0)),
                sequence_length=len(generated_tokens) if isinstance(generated_tokens, list) else generated_tokens.shape[-1],
                perplexity=perplexity,
                diversity_score=diversity,
                tree_depth=result.get('tree_depth', 0),
                avg_branching_factor=result.get('avg_branching', 0.0),
                num_rollouts=self.num_rollouts,
                correct=self._check_correctness(result.get('text', ''), expected)
            )
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return EvaluationMetrics(
                method=self.method_name,
                sample_idx=sample_idx,
                epoch=epoch,
                nfe=0,
                time_seconds=0.0,
                reward=0.0,
                sequence_length=0,
                perplexity=float('inf'),
                diversity_score=0.0,
                tree_depth=0,
                avg_branching_factor=0.0,
                num_rollouts=self.num_rollouts,
                correct=False,
                error=str(e)
            )
    
    def _run_greedy(self, prompt_tokens: torch.Tensor) -> Dict:
        """Run greedy baseline"""
        with torch.no_grad():
            output = self.model.model.generate(
                input_ids=prompt_tokens,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.model.tokenizer.pad_token_id,
                eos_token_id=self.model.eos_token_id
            )
        
        text = self.model.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return {
            'tokens': output[0].cpu().tolist(),
            'text': text,
            'nodes_explored': len(output[0]),
            'reward': 0.0,
            'nfe': len(output[0])
        }
    
    def _run_mcts(self, prompt_tokens: torch.Tensor) -> Dict:
        """Run MCTS"""
        config = MCTSConfig(
            num_simulations=self.num_rollouts,
            expansion_k=self.expansion_k,
            temperature=self.temperature,
            verbose=False
        )
        mcts = MCTSTextGenerator(self.model, self.reward_fn, config)
        return mcts.search(prompt_tokens[0], max_new_tokens=50)
    
    def _run_dts(self, prompt_tokens: torch.Tensor) -> Dict:
        """Run DTS"""
        config = DTSConfig(
            num_rollouts=self.num_rollouts,
            expansion_k=self.expansion_k,
            temperature=self.temperature,
            verbose=False
        )
        dts = DTSTextGenerator(self.model, self.reward_fn, config)
        return dts.search(prompt_tokens[0], max_new_tokens=50)
    
    def _run_dts_star(self, prompt_tokens: torch.Tensor) -> Dict:
        """Run DTS*"""
        config = DTSConfig(
            num_rollouts=self.num_rollouts,
            expansion_k=self.expansion_k,
            temperature=self.temperature,
            use_soft_bellman=False,  # DTS* uses standard backup
            verbose=False
        )
        dts_star = DTSTextGenerator(self.model, self.reward_fn, config)
        return dts_star.search(prompt_tokens[0], max_new_tokens=50)
    
    def _run_maxent_ts(self, prompt_tokens: torch.Tensor) -> Dict:
        """Run MaxEnt-TS"""
        config = MaxEntTSConfig(
            num_rollouts=self.num_rollouts,
            expansion_k=self.expansion_k,
            temperature=self.temperature,
            verbose=False
        )
        maxent = MaxEntTS(self.model, self.reward_fn, config)
        result = maxent.search(prompt_tokens[0], max_new_tokens=50)
        
        # Add NFE
        result['nfe'] = result['tree_stats']['total_nodes']
        result['nodes_explored'] = result['tree_stats']['total_nodes']
        result['tree_depth'] = result['tree_stats'].get('max_depth', 0)
        result['avg_branching'] = result['tree_stats'].get('avg_branching_factor', 0.0)
        result['reward'] = result.get('best_reward', 0.0)
        result['tokens'] = result['best_sequence']
        result['text'] = result['best_text']
        
        return result
    
    def _check_correctness(self, generated: str, expected: str) -> bool:
        """
        Check if generated text contains the expected answer
        
        Args:
            generated: Generated text
            expected: Expected answer
        
        Returns:
            bool: True if answer is correct
        """
        if not expected or not generated:
            return False
        
        # Clean strings
        generated_clean = str(generated).lower().strip()
        expected_clean = str(expected).lower().strip()
        
        # Exact match
        if expected_clean == generated_clean:
            return True
        
        # Substring match
        if expected_clean in generated_clean:
            return True
        
        # Check if generated contains the key parts (split by whitespace)
        expected_words = expected_clean.split()
        if len(expected_words) > 0:
            # For numeric answers, check if number appears
            try:
                expected_num = float(expected_clean)
                # Try to find the number in generated text
                numbers_in_generated = re.findall(r'[-+]?\d*\.?\d+', generated_clean)
                for num_str in numbers_in_generated:
                    try:
                        gen_num = float(num_str)
                        # Allow 10% tolerance for numeric answers
                        if abs(gen_num - expected_num) / max(abs(expected_num), 1e-6) < 0.1:
                            return True
                    except:
                        continue
            except:
                # Not a number - check word overlap
                generated_words = set(generated_clean.split())
                expected_words_set = set(expected_words)
                # If more than 70% of expected words are in generated, consider correct
                if len(expected_words_set) > 0:
                    overlap = len(expected_words_set & generated_words) / len(expected_words_set)
                    if overlap >= 0.7:
                        return True
        
        return False
    
    def run_epoch(self, epoch: int):
        """Run one epoch of evaluation"""
        print(f"\n{'='*80}")
        print(f"📊 EPOCH {epoch + 1} - {self.method_name.upper()}")
        print(f"{'='*80}\n")
        
        epoch_metrics = []
        
        # Limit samples
        num_eval = min(self.num_samples, len(self.dataset))
        
        for i in range(num_eval):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{num_eval} samples...")
            
            sample = self.dataset[i]
            metrics = self.evaluate_sample(i, epoch, sample)
            epoch_metrics.append(metrics)
            self.all_metrics.append(metrics)
            
            # Log to WandB
            if self.use_wandb and metrics.error is None:
                wandb.log({
                    "epoch": epoch,
                    "sample_idx": i,
                    "nfe": metrics.nfe,
                    "time": metrics.time_seconds,
                    "reward": metrics.reward,
                    "sequence_length": metrics.sequence_length,
                    "perplexity": metrics.perplexity if metrics.perplexity != float('inf') else 1000,
                    "diversity": metrics.diversity_score,
                    "tree_depth": metrics.tree_depth,
                    "branching_factor": metrics.avg_branching_factor,
                    "correct": 1.0 if metrics.correct else 0.0
                })
        
        # Compute epoch statistics
        self._compute_epoch_stats(epoch, epoch_metrics)
    
    def _compute_epoch_stats(self, epoch: int, metrics: List[EvaluationMetrics]):
        """Compute and log epoch-level statistics"""
        valid_metrics = [m for m in metrics if m.error is None]
        
        if not valid_metrics:
            print("⚠️ No valid metrics in this epoch")
            return
        
        stats = {
            'avg_nfe': np.mean([m.nfe for m in valid_metrics]),
            'avg_time': np.mean([m.time_seconds for m in valid_metrics]),
            'avg_reward': np.mean([m.reward for m in valid_metrics]),
            'avg_seq_length': np.mean([m.sequence_length for m in valid_metrics]),
            'avg_perplexity': np.mean([m.perplexity for m in valid_metrics if m.perplexity != float('inf')]),
            'avg_diversity': np.mean([m.diversity_score for m in valid_metrics]),
            'accuracy': np.mean([1.0 if m.correct else 0.0 for m in valid_metrics]),
            'success_rate': len(valid_metrics) / len(metrics)
        }
        
        self.epoch_stats[epoch] = stats
        
        # Log to WandB
        if self.use_wandb:
            wandb.log({
                f"epoch_{k}": v for k, v in stats.items()
            })
        
        # Print summary
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   NFE: {stats['avg_nfe']:.1f}")
        print(f"   Time: {stats['avg_time']:.3f}s")
        print(f"   Reward: {stats['avg_reward']:.4f}")
        print(f"   Accuracy: {stats['accuracy']*100:.1f}%")
        print(f"   Diversity: {stats['avg_diversity']:.4f}")
    
    def save_results(self, output_dir: str = "results"):
        """Save all results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(
            output_dir,
            f"{self.method_name}_k{self.expansion_k}_roll{self.num_rollouts}.json"
        )
        
        with open(metrics_file, 'w') as f:
            json.dump([asdict(m) for m in self.all_metrics], f, indent=2)
        
        print(f"\n💾 Results saved to: {metrics_file}")
    
    def finish(self):
        """Cleanup"""
        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation')
    parser.add_argument('--method', type=str, required=True,
                       choices=['greedy', 'mcts', 'dts', 'dts_star', 'maxent_ts'])
    parser.add_argument('--num_samples', type=int, default=250)
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--expansion_k', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='m4', choices=['m4', 'har'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--device', type=str, default='mps')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        method_name=args.method,
        num_samples=args.num_samples,
        num_rollouts=args.num_rollouts,
        expansion_k=args.expansion_k,
        temperature=args.temperature,
        dataset_name=args.dataset,
        use_wandb=not args.no_wandb,
        device=args.device
    )
    
    # Run epochs
    for epoch in range(args.epochs):
        evaluator.run_epoch(epoch)
    
    # Save results
    evaluator.save_results()
    evaluator.finish()


if __name__ == "__main__":
    main()


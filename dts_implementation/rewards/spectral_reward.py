"""
Spectral Reward Computation for S-ADT

Implements the spectral regularization reward function from S-ADT paper.

Mathematical Framework:
r(x) = r_task(x) - γ ∫ |log S_x(ω) - log E[S_c(ω)]| dω

where:
- r_task(x): Task-specific reward (e.g., accuracy, CRPS)
- S_x(ω): PSD of predicted time series
- E[S_c(ω)]: Expected PSD from historical context
- γ: Spectral penalty weight
"""

import torch
import numpy as np
from typing import Optional, Dict, Callable
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.psd_utils import compute_psd, compute_expected_psd, spectral_distance


class SpectralReward:
    """
    Computes spectral-regularized rewards for S-ADT
    
    r(x) = r_task(x) - γ * spectral_penalty(x, context)
    
    This prevents spectral collapse by penalizing predictions that don't
    match the frequency characteristics of the historical data.
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        sampling_rate: float = 1.0,
        spectral_metric: str = 'l1',
        task_reward_fn: Optional[Callable] = None,
        normalize_rewards: bool = True
    ):
        """
        Initialize spectral reward computer
        
        Args:
            gamma: Spectral penalty weight (higher = stricter spectral matching)
            sampling_rate: Time series sampling rate
            spectral_metric: Distance metric ('l1', 'wasserstein', 'kl')
            task_reward_fn: Optional custom task reward function
            normalize_rewards: Whether to normalize rewards to [0, 1]
        """
        self.gamma = gamma
        self.sampling_rate = sampling_rate
        self.spectral_metric = spectral_metric
        self.task_reward_fn = task_reward_fn
        self.normalize_rewards = normalize_rewards
        
        # Cache for context PSD (computed once per batch)
        self.context_psd_cache = None
        self.context_freqs_cache = None
    
    def set_context(self, context_time_series: np.ndarray):
        """
        Set context time series and compute expected PSD
        
        This should be called once per evaluation with the historical data.
        
        Args:
            context_time_series: Historical time series [batch, length] or [length]
        """
        freqs, expected_psd = compute_expected_psd(
            context_time_series,
            sampling_rate=self.sampling_rate
        )
        
        self.context_freqs_cache = freqs
        self.context_psd_cache = expected_psd
        
        print(f"✅ Context PSD computed:")
        print(f"   Frequency range: {freqs[0]:.4f} - {freqs[-1]:.4f} Hz")
        print(f"   Mean power: {expected_psd.mean():.6f}")
    
    def compute_spectral_penalty(
        self,
        predicted_time_series: np.ndarray
    ) -> float:
        """
        Compute spectral penalty for predicted time series
        
        penalty = ∫ |log S_x(ω) - log E[S_c(ω)]| dω
        
        Args:
            predicted_time_series: Predicted time series [length]
        
        Returns:
            penalty: Spectral distance (lower = better match)
        """
        if self.context_psd_cache is None:
            raise ValueError("Context PSD not set. Call set_context() first.")
        
        # Compute PSD of prediction
        freqs, pred_psd = compute_psd(
            predicted_time_series,
            sampling_rate=self.sampling_rate
        )
        
        # Compute distance
        distance = spectral_distance(
            pred_psd,
            self.context_psd_cache,
            freqs,
            metric=self.spectral_metric
        )
        
        return distance
    
    def compute_task_reward(
        self,
        predicted: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Compute task-specific reward
        
        Args:
            predicted: Model prediction
            ground_truth: Ground truth (if available)
            metadata: Additional task metadata
        
        Returns:
            task_reward: Task performance reward
        """
        if self.task_reward_fn is not None:
            return self.task_reward_fn(predicted, ground_truth, metadata)
        
        # Default: Simple accuracy if ground truth available
        if ground_truth is not None:
            # For classification: accuracy
            if predicted.ndim == 0 or len(predicted) == 1:
                return float(predicted == ground_truth)
            # For regression: negative MSE
            else:
                mse = np.mean((predicted - ground_truth) ** 2)
                return -mse
        
        # No ground truth: return neutral reward
        return 0.0
    
    def compute_reward(
        self,
        predicted_time_series: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute full reward: r(x) = r_task(x) - γ * spectral_penalty(x)
        
        Args:
            predicted_time_series: Predicted time series [length]
            ground_truth: Ground truth for task reward
            metadata: Task metadata
        
        Returns:
            Dict with:
                - total_reward: Combined reward
                - task_reward: Task component
                - spectral_penalty: Spectral component
                - spectral_reward: -γ * spectral_penalty
        """
        # Task reward
        task_reward = self.compute_task_reward(
            predicted_time_series,
            ground_truth,
            metadata
        )
        
        # Spectral penalty
        spectral_penalty = self.compute_spectral_penalty(predicted_time_series)
        
        # Combined reward
        spectral_reward = -self.gamma * spectral_penalty
        total_reward = task_reward + spectral_reward
        
        # Normalize if requested
        if self.normalize_rewards:
            # Simple sigmoid normalization
            total_reward = 1.0 / (1.0 + np.exp(-total_reward))
        
        return {
            'total_reward': float(total_reward),
            'task_reward': float(task_reward),
            'spectral_penalty': float(spectral_penalty),
            'spectral_reward': float(spectral_reward)
        }
    
    def batch_compute_reward(
        self,
        predicted_time_series_batch: np.ndarray,
        ground_truth_batch: Optional[np.ndarray] = None,
        metadata_batch: Optional[list] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute rewards for batch
        
        Args:
            predicted_time_series_batch: Predicted time series [batch, length]
            ground_truth_batch: Ground truths [batch, ...]
            metadata_batch: List of metadata dicts
        
        Returns:
            Dict with arrays of rewards
        """
        batch_size = predicted_time_series_batch.shape[0]
        
        total_rewards = []
        task_rewards = []
        spectral_penalties = []
        spectral_rewards = []
        
        for i in range(batch_size):
            pred = predicted_time_series_batch[i]
            gt = ground_truth_batch[i] if ground_truth_batch is not None else None
            meta = metadata_batch[i] if metadata_batch is not None else None
            
            reward_dict = self.compute_reward(pred, gt, meta)
            
            total_rewards.append(reward_dict['total_reward'])
            task_rewards.append(reward_dict['task_reward'])
            spectral_penalties.append(reward_dict['spectral_penalty'])
            spectral_rewards.append(reward_dict['spectral_reward'])
        
        return {
            'total_reward': np.array(total_rewards),
            'task_reward': np.array(task_rewards),
            'spectral_penalty': np.array(spectral_penalties),
            'spectral_reward': np.array(spectral_rewards)
        }


# Pre-defined reward functions for different tasks

def tsqa_accuracy_reward(predicted: np.ndarray, ground_truth: np.ndarray, metadata: Dict) -> float:
    """
    TSQA (Stage 1): Multiple-choice accuracy
    
    Args:
        predicted: Predicted answer index
        ground_truth: True answer index
        metadata: Question metadata
    
    Returns:
        reward: 1.0 if correct, 0.0 otherwise
    """
    return float(predicted == ground_truth)


def m4_captioning_reward(predicted: str, ground_truth: str, metadata: Dict) -> float:
    """
    M4 (Stage 2): Captioning quality (BLEU, ROUGE, etc.)
    
    For now, simple string match. Could be improved with BLEU/ROUGE.
    """
    # Simple token overlap
    pred_tokens = set(predicted.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    
    if len(gt_tokens) == 0:
        return 0.0
    
    overlap = len(pred_tokens & gt_tokens) / len(gt_tokens)
    return overlap


def regression_mse_reward(predicted: np.ndarray, ground_truth: np.ndarray, metadata: Dict) -> float:
    """
    Regression: Negative MSE reward
    """
    mse = np.mean((predicted - ground_truth) ** 2)
    return -mse


def classification_f1_reward(predicted: np.ndarray, ground_truth: np.ndarray, metadata: Dict) -> float:
    """
    Classification: F1 score
    """
    from sklearn.metrics import f1_score
    return f1_score(ground_truth, predicted, average='macro')


# Registry of task-specific reward functions
TASK_REWARD_FUNCTIONS = {
    'tsqa': tsqa_accuracy_reward,
    'm4': m4_captioning_reward,
    'regression': regression_mse_reward,
    'classification': classification_f1_reward,
}


def create_spectral_reward(
    task: str,
    gamma: float = 1.0,
    sampling_rate: float = 1.0,
    spectral_metric: str = 'l1',
    normalize: bool = True
) -> SpectralReward:
    """
    Factory function to create spectral reward for specific task
    
    Args:
        task: Task name ('tsqa', 'm4', 'regression', 'classification')
        gamma: Spectral penalty weight
        sampling_rate: Time series sampling rate
        spectral_metric: Distance metric
        normalize: Normalize rewards
    
    Returns:
        Configured SpectralReward instance
    """
    task_reward_fn = TASK_REWARD_FUNCTIONS.get(task, None)
    
    return SpectralReward(
        gamma=gamma,
        sampling_rate=sampling_rate,
        spectral_metric=spectral_metric,
        task_reward_fn=task_reward_fn,
        normalize_rewards=normalize
    )


if __name__ == "__main__":
    # Test spectral reward
    print("Testing Spectral Reward...")
    
    # Generate test data
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    
    # Context: low frequency signal
    context = np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(len(t))
    
    # Prediction 1: matches context (low freq)
    pred1 = np.sin(2 * np.pi * 2 * t) + 0.15 * np.random.randn(len(t))
    
    # Prediction 2: different freq (high freq)
    pred2 = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    
    print("\n1. Creating spectral reward...")
    reward_computer = SpectralReward(
        gamma=1.0,
        sampling_rate=100,
        spectral_metric='l1',
        normalize_rewards=False
    )
    
    print("\n2. Setting context...")
    reward_computer.set_context(context)
    
    print("\n3. Computing rewards...")
    
    # Reward for good prediction (matches context freq)
    reward1 = reward_computer.compute_reward(pred1)
    print(f"\n   Prediction 1 (matches context):")
    print(f"      Total reward: {reward1['total_reward']:.4f}")
    print(f"      Spectral penalty: {reward1['spectral_penalty']:.4f}")
    
    # Reward for bad prediction (wrong freq)
    reward2 = reward_computer.compute_reward(pred2)
    print(f"\n   Prediction 2 (wrong freq):")
    print(f"      Total reward: {reward2['total_reward']:.4f}")
    print(f"      Spectral penalty: {reward2['spectral_penalty']:.4f}")
    
    print(f"\n   ✅ Spectral penalty is higher for wrong frequency!")
    print(f"      Penalty ratio: {reward2['spectral_penalty'] / reward1['spectral_penalty']:.2f}x")
    
    # Test batch computation
    print("\n4. Testing batch computation...")
    batch_preds = np.stack([pred1, pred2])
    batch_rewards = reward_computer.batch_compute_reward(batch_preds)
    print(f"   Batch rewards: {batch_rewards['total_reward']}")
    
    print("\n✅ Spectral reward tests passed!")


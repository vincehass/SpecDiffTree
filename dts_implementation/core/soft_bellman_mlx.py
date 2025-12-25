"""
Pure MLX Soft Bellman Backup for DTS
Key innovation that prevents spectral collapse by using LogSumExp instead of max

Based on S-ADT paper: "Unlike greedy methods that converge to E[x], 
Soft Bellman maintains the full probability distribution"

Pure MLX implementation - no PyTorch dependencies!
"""

import mlx.core as mx
import numpy as np
from typing import List, Optional
from .dts_node_mlx import MCTSNode


def soft_bellman_backup(
    terminal_node: MCTSNode,
    reward: float,
    temperature: float = 0.1
):
    """
    Pure MLX Soft Bellman Backup with LogSumExp aggregation
    
    Propagates terminal reward r(x_0) back through the tree using:
    V_t(x_t) = (1/λ) log E[exp(λ V_{t-1}(x_{t-1}))]
    
    This maintains probability mass across multimodal solutions,
    preventing the spectral collapse that occurs with greedy/max backups.
    
    Args:
        terminal_node: Terminal node with reward
        reward: Terminal reward r(x_0)
        temperature: Inverse temperature λ (higher = more stochastic)
    
    Key Insight (from S-ADT paper):
        Greedy: V_t = max(V_{t-1}) → collapses to mode (E[x])
        Soft:   V_t = (1/λ) log E[exp(λ V_{t-1})] → maintains distribution
        
        This preserves high-frequency content that greedy destroys!
    """
    current = terminal_node
    value = reward
    
    while current is not None:
        # Update visit count
        current.increment_visit()
        
        # Soft value aggregation (not max!)
        if current.children:
            child_values = [child.value_est for child in current.children]
            
            # LogSumExp for numerical stability (Pure MLX!)
            # V = (1/λ) * log(sum(exp(λ * v_i)))
            values_array = mx.array(child_values)
            value = (1.0 / temperature) * float(
                mx.logsumexp(temperature * values_array, axis=0)
            )
        
        # Update node value
        current.update_value(value)
        
        # Move to parent
        current = current.parent


def soft_bellman_backup_batch(
    terminal_nodes: List[MCTSNode],
    rewards: List[float],
    temperature: float = 0.1
):
    """
    Batch version of Soft Bellman backup for efficiency
    
    Args:
        terminal_nodes: List of terminal nodes
        rewards: Corresponding rewards
        temperature: Inverse temperature λ
    """
    for node, reward in zip(terminal_nodes, rewards):
        soft_bellman_backup(node, reward, temperature)


def greedy_backup(terminal_node: MCTSNode, reward: float):
    """
    Greedy (max) backup - for comparison/baseline
    
    This is what standard MCTS does, but it causes spectral collapse!
    V_t(x_t) = max_{children} V_{t-1}(x_{t-1})
    
    Args:
        terminal_node: Terminal node with reward
        reward: Terminal reward r(x_0)
    """
    current = terminal_node
    value = reward
    
    while current is not None:
        current.increment_visit()
        
        # Greedy max (causes spectral collapse!)
        if current.children:
            child_values = [child.value_est for child in current.children]
            value = max(child_values)
        
        current.update_value(value)
        current = current.parent


def compute_soft_value(
    child_values: List[float],
    temperature: float
) -> float:
    """
    Compute soft value from children using LogSumExp (Pure MLX)
    
    V = (1/λ) * log(sum(exp(λ * v_i)))
    
    Args:
        child_values: List of child value estimates
        temperature: Inverse temperature λ
    
    Returns:
        Soft aggregated value
    """
    if not child_values:
        return 0.0
    
    values_array = mx.array(child_values)
    return (1.0 / temperature) * float(
        mx.logsumexp(temperature * values_array, axis=0)
    )


def compute_boltzmann_policy(
    child_values: List[float],
    temperature: float,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Compute Boltzmann (softmax) policy over children (Pure MLX)
    
    π(child_i) ∝ exp(λ * V(child_i))
    
    This is used for SELECTION phase of DTS.
    
    Args:
        child_values: List of child value estimates
        temperature: Inverse temperature λ
        epsilon: Small constant for numerical stability
    
    Returns:
        Probability distribution over children (guaranteed to sum to 1.0)
    """
    if not child_values:
        return np.array([])
    
    values = mx.array(child_values)
    
    # Softmax with temperature
    # Subtract max for numerical stability
    exp_values = mx.exp(temperature * (values - mx.max(values)))
    probs = exp_values / mx.sum(exp_values)
    
    # Convert to numpy
    probs_np = np.array(probs.tolist())
    
    # Add epsilon and renormalize to guarantee sum=1.0
    probs_np = probs_np + epsilon
    probs_np = probs_np / np.sum(probs_np)
    
    # Final validation
    assert np.isclose(np.sum(probs_np), 1.0), f"Probabilities sum to {np.sum(probs_np)}, not 1.0"
    
    return probs_np


def sample_child_boltzmann(
    node: MCTSNode,
    temperature: float,
    exploration_prob: float = 0.0
) -> Optional[MCTSNode]:
    """
    Sample a child using Boltzmann policy (Pure MLX)
    
    With probability exploration_prob, sample uniformly (exploration).
    Otherwise, sample proportional to exp(λ * V).
    
    Args:
        node: Parent node
        temperature: Inverse temperature λ
        exploration_prob: Probability of uniform exploration
    
    Returns:
        Selected child node (or None if no children)
    """
    if not node.children:
        return None
    
    if len(node.children) == 1:
        return node.children[0]
    
    # Epsilon-greedy exploration
    if exploration_prob > 0 and np.random.random() < exploration_prob:
        return np.random.choice(node.children)
    
    # Boltzmann sampling with error handling
    try:
        child_values = [child.value_est for child in node.children]
        probs = compute_boltzmann_policy(child_values, temperature)
        return np.random.choice(node.children, p=probs)
    except (ValueError, AssertionError) as e:
        # Fallback to uniform sampling if probability computation fails
        print(f"Warning: Boltzmann sampling failed ({e}), using uniform sampling")
        return np.random.choice(node.children)


def select_best_child(node: MCTSNode) -> Optional[MCTSNode]:
    """
    Select child with highest value (for final extraction)
    
    Args:
        node: Parent node
    
    Returns:
        Child with highest value
    """
    if not node.children:
        return None
    
    return max(node.children, key=lambda c: c.value_est)


class ValueTracker:
    """
    Track value evolution during tree search for diagnostics
    """
    
    def __init__(self):
        self.root_values = []
        self.terminal_rewards = []
        self.rollout_idx = []
    
    def log_rollout(self, root_value: float, terminal_reward: float, rollout: int):
        """Log values after a rollout"""
        self.root_values.append(root_value)
        self.terminal_rewards.append(terminal_reward)
        self.rollout_idx.append(rollout)
    
    def get_stats(self) -> dict:
        """Get summary statistics"""
        return {
            'mean_root_value': np.mean(self.root_values) if self.root_values else 0,
            'mean_terminal_reward': np.mean(self.terminal_rewards) if self.terminal_rewards else 0,
            'num_rollouts': len(self.rollout_idx),
            'value_improvement': (
                self.root_values[-1] - self.root_values[0]
                if len(self.root_values) > 1 else 0
            )
        }

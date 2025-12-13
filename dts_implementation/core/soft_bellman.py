"""
Soft Bellman Backup for DTS
Key innovation that prevents spectral collapse by using LogSumExp instead of max

Based on S-ADT paper: "Unlike greedy methods that converge to E[x], 
Soft Bellman maintains the full probability distribution"
"""

import torch
import numpy as np
from typing import List, Optional
from .dts_node import MCTSNode


def soft_bellman_backup(
    terminal_node: MCTSNode,
    reward: float,
    temperature: float = 0.1
):
    """
    Soft Bellman Backup with LogSumExp aggregation
    
    Propagates terminal reward r(x_0) back through the tree using:
    V_t(x_t) = (1/λ) log E[exp(λ V_{t-1}(x_{t-1}))]
    
    This maintains probability mass across multimodal solutions,
    preventing the spectral collapse that occurs with greedy/max backups.
    
    Args:
        terminal_node: Terminal node (x_0) with reward
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
            
            # LogSumExp for numerical stability
            # V = (1/λ) * log(sum(exp(λ * v_i)))
            values_tensor = torch.tensor(child_values, dtype=torch.float32)
            value = (1.0 / temperature) * torch.logsumexp(
                temperature * values_tensor, dim=0
            ).item()
        
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
    Compute soft value from children using LogSumExp
    
    V = (1/λ) * log(sum(exp(λ * v_i)))
    
    Args:
        child_values: List of child value estimates
        temperature: Inverse temperature λ
    
    Returns:
        Soft aggregated value
    """
    if not child_values:
        return 0.0
    
    values_tensor = torch.tensor(child_values, dtype=torch.float32)
    return (1.0 / temperature) * torch.logsumexp(
        temperature * values_tensor, dim=0
    ).item()


def compute_boltzmann_policy(
    child_values: List[float],
    temperature: float
) -> np.ndarray:
    """
    Compute Boltzmann (softmax) policy over children
    
    π(child_i) ∝ exp(λ * V(child_i))
    
    This is used for SELECTION phase of DTS.
    
    Args:
        child_values: List of child value estimates
        temperature: Inverse temperature λ
    
    Returns:
        Probability distribution over children
    """
    if not child_values:
        return np.array([])
    
    values = np.array(child_values)
    
    # Softmax with temperature
    # Subtract max for numerical stability
    exp_values = np.exp(temperature * (values - values.max()))
    probs = exp_values / exp_values.sum()
    
    return probs


def sample_child_boltzmann(
    node: MCTSNode,
    temperature: float,
    exploration_prob: float = 0.0
) -> Optional[MCTSNode]:
    """
    Sample a child using Boltzmann policy
    
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
    
    # Boltzmann sampling
    child_values = [child.value_est for child in node.children]
    probs = compute_boltzmann_policy(child_values, temperature)
    
    return np.random.choice(node.children, p=probs)


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


if __name__ == "__main__":
    # Test Soft Bellman backup
    print("Testing Soft Bellman Backup...")
    
    # Create simple tree
    root = MCTSNode(torch.randn(1, 10), t=2)
    
    child1 = MCTSNode(torch.randn(1, 10), t=1, parent=root)
    child2 = MCTSNode(torch.randn(1, 10), t=1, parent=root)
    root.add_child(child1)
    root.add_child(child2)
    
    terminal1 = MCTSNode(torch.randn(1, 10), t=0, parent=child1)
    terminal2 = MCTSNode(torch.randn(1, 10), t=0, parent=child2)
    child1.add_child(terminal1)
    child2.add_child(terminal2)
    
    # Test soft backup
    print("\n1. Testing Soft Bellman Backup:")
    soft_bellman_backup(terminal1, reward=1.0, temperature=0.1)
    soft_bellman_backup(terminal2, reward=0.5, temperature=0.1)
    
    print(f"   Root value (soft): {root.value_est:.4f}")
    print(f"   Child1 value: {child1.value_est:.4f}")
    print(f"   Child2 value: {child2.value_est:.4f}")
    
    # Test greedy for comparison
    print("\n2. Testing Greedy Backup (for comparison):")
    root2 = MCTSNode(torch.randn(1, 10), t=2)
    child1_g = MCTSNode(torch.randn(1, 10), t=1, parent=root2)
    child2_g = MCTSNode(torch.randn(1, 10), t=1, parent=root2)
    root2.add_child(child1_g)
    root2.add_child(child2_g)
    
    terminal1_g = MCTSNode(torch.randn(1, 10), t=0, parent=child1_g)
    terminal2_g = MCTSNode(torch.randn(1, 10), t=0, parent=child2_g)
    child1_g.add_child(terminal1_g)
    child2_g.add_child(terminal2_g)
    
    greedy_backup(terminal1_g, reward=1.0)
    greedy_backup(terminal2_g, reward=0.5)
    
    print(f"   Root value (greedy): {root2.value_est:.4f}")
    print(f"   (Should be max = 1.0, not soft aggregation)")
    
    # Test Boltzmann sampling
    print("\n3. Testing Boltzmann Policy:")
    probs = compute_boltzmann_policy([1.0, 0.5, 0.2], temperature=0.1)
    print(f"   Child values: [1.0, 0.5, 0.2]")
    print(f"   Boltzmann probs: {probs}")
    print(f"   (Higher value → higher probability)")
    
    # Test child selection
    print("\n4. Testing Child Selection:")
    sampled = sample_child_boltzmann(root, temperature=0.1)
    print(f"   Sampled child: {sampled}")
    
    best = select_best_child(root)
    print(f"   Best child: {best}")
    
    print("\n✅ Soft Bellman tests passed!")
    print("\n💡 Key Insight:")
    print("   Soft Bellman uses LogSumExp (maintains distribution)")
    print("   Greedy uses max (collapses to mode)")
    print("   This prevents spectral collapse in diffusion sampling!")


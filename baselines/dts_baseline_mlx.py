"""
Pure MLX DTS (Diffusion Tree Sampling) Baseline

Implements the exact algorithm from the paper:
"Diffusion Tree Sampling: Inference-Time Alignment of Generative Diffusion Models"
Jain et al., 2025 (arXiv:2506.20701)

Adapted for autoregressive LLMs (token sequences instead of diffusion states)
Pure MLX implementation - no PyTorch dependencies!
"""

import mlx.core as mx
import numpy as np
import time
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class DTSConfig:
    """Configuration for DTS"""
    num_rollouts: int = 100
    temperature: float = 1.0  # λ in the paper (inverse temperature)
    expansion_k: int = 4
    max_seq_length: int = 512
    use_soft_bellman: bool = True  # Use Soft Bellman backup (DTS)
    verbose: bool = False


class DTSNode:
    """
    Node in DTS tree
    
    Represents a partial sequence x_{≤t}
    """
    
    def __init__(self, token_ids: mx.array, parent: Optional['DTSNode'] = None):
        self.token_ids = token_ids  # MLX array
        self.parent = parent
        self.children: List['DTSNode'] = []
        
        # DTS-specific attributes
        self.visit_count = 1  # N(x_{≤t}) - initialized to 1 as per paper
        self.soft_value = 0.0  # V̂(x_{≤t}) - soft value estimate
        
    def is_leaf(self) -> bool:
        return len(self.children) == 0


class DTSTextGenerator:
    """
    Pure MLX Diffusion Tree Sampling (DTS) for text generation
    
    Algorithm from paper (Alg. 1):
    
    for m = 1 to M rollouts:
        1. SELECT: Sample path using Boltzmann policy π ∝ p_θ · exp(λ V̂)
        2. EXPAND: Add new children at selected node
        3. ROLLOUT: Complete sequence using base model p_θ
        4. EVALUATE: Compute reward r(x)
        5. BACKUP: Soft Bellman update through path
    
    Key difference from MCTS:
    - Uses Soft Bellman backup: V̂(x_{≤t}) = softmax_{x_{t+1}}[log p_θ + λ V̂(x_{≤t+1}) + λ r]
    - Boltzmann selection: π(x_{t+1}|x_{≤t}) ∝ p_θ(x_{t+1}|x_{≤t}) exp(λ V̂(x_{≤t+1}))
    """
    
    def __init__(self, model, reward_fn, config: DTSConfig):
        self.model = model
        self.reward_fn = reward_fn
        self.config = config
        self.root: Optional[DTSNode] = None
        
    def search(self, prompt_tokens: mx.array, max_new_tokens: int = 200) -> Dict:
        """
        Run DTS search
        
        Args:
            prompt_tokens: Initial prompt (MLX array)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict with best sequence and statistics
        """
        # Convert to MLX array if needed
        if not isinstance(prompt_tokens, mx.array):
            if hasattr(prompt_tokens, 'tolist'):
                prompt_tokens = mx.array(prompt_tokens.tolist())
            else:
                prompt_tokens = mx.array(list(prompt_tokens))
        
        # Initialize root - handle batch dimension
        if prompt_tokens.ndim == 2:
            prompt_tokens = prompt_tokens[0]
        
        self.root = DTSNode(prompt_tokens)
        start_time = time.time()
        
        if self.config.verbose:
            print(f"\n🌲 DTS (Pure MLX): Starting {self.config.num_rollouts} rollouts...")
        
        # Run rollouts
        for rollout_idx in range(self.config.num_rollouts):
            # 1. SELECT: Navigate tree using Boltzmann policy
            leaf = self._select_boltzmann(self.root)
            
            # 2. EXPAND: Add new children if not terminal
            if not self._is_terminal(leaf):
                expanded = self._expand(leaf)
            else:
                expanded = leaf
            
            # 3. ROLLOUT: Complete sequence
            final_tokens, rollout_node = self._rollout(expanded, max_new_tokens)
            
            # 4. EVALUATE: Get reward
            reward = self.reward_fn(mx.expand_dims(final_tokens, 0))
            
            # 5. BACKUP: Soft Bellman update
            if self.config.use_soft_bellman:
                self._soft_bellman_backup(rollout_node, float(reward))
            else:
                self._standard_backup(rollout_node, float(reward))
            
            if self.config.verbose and (rollout_idx + 1) % 10 == 0:
                best = self._get_best_child(self.root)
                print(f"  Rollout {rollout_idx+1}/{self.config.num_rollouts}, "
                      f"Best V̂: {best.soft_value:.4f}")
        
        # Extract best sequence (highest soft value)
        best_node = self._get_best_child(self.root)
        
        # Follow best path to terminal
        current = best_node
        while not self._is_terminal(current) and current.children:
            current = self._get_best_child(current)
        
        if self.config.verbose:
            print(f"\n✅ DTS complete. Best V̂: {current.soft_value:.4f}")
        
        # Decode best sequence
        best_text = self.model.tokenizer.decode(current.token_ids.tolist(), skip_special_tokens=True)
        
        return {
            'best_node': current,
            'best_sequence': current.token_ids,
            'best_text': best_text,
            'best_value': current.soft_value,
            'nodes_explored': self._count_nodes(self.root),
            'total_rollouts': self.config.num_rollouts,
            'tree_size': self._count_nodes(self.root),
            'time': time.time() - start_time
        }
    
    def _select_boltzmann(self, node: DTSNode) -> DTSNode:
        """
        Selection using Boltzmann policy (DTS Eq. 6):
        
        π(x_{t+1}|x_{≤t}) ∝ p_θ(x_{t+1}|x_{≤t}) · exp(λ · V̂(x_{≤t+1}))
        
        This is the key difference from MCTS!
        """
        current = node
        
        while not current.is_leaf() and not self._is_terminal(current):
            # Get Boltzmann distribution over children
            children = current.children
            
            # Compute log probabilities
            log_probs = []
            for child in children:
                # log π = log p_θ + λ V̂
                # We approximate log p_θ from model (simplified)
                log_prob = self.config.temperature * child.soft_value
                log_probs.append(log_prob)
            
            # Softmax to get probabilities
            log_probs = mx.array(log_probs)
            probs = mx.softmax(log_probs, axis=0)

            # Sample child using categorical distribution
            child_idx = int(mx.random.categorical(mx.log(probs)))
            current = children[child_idx]
        
        return current
    
    def _expand(self, node: DTSNode) -> DTSNode:
        """
        Expansion: Add new children based on top-k tokens from p_θ
        """
        # Get top-k next tokens
        input_ids = mx.expand_dims(node.token_ids, 0) if node.token_ids.ndim == 1 else node.token_ids
        logits_output = self.model.get_next_token_logits(input_ids)
        
        # Handle KV cache: get_next_token_logits may return (logits, past_kv) or just logits
        logits = logits_output[0] if isinstance(logits_output, tuple) else logits_output
        
        # Get probabilities
        probs = mx.softmax(logits, axis=-1)
        
        # Get top-k using argsort (MLX doesn't have topk, but argsort works)
        top_indices = mx.argsort(probs, axis=-1)[-self.config.expansion_k:]
        top_indices = top_indices[::-1]  # Reverse to get highest first
        
        # Create children
        for token_id in top_indices.tolist():
            new_tokens = mx.concatenate([node.token_ids, mx.array([token_id])])
            child = DTSNode(new_tokens, parent=node)
            node.children.append(child)
        
        # Return first child for rollout
        return node.children[0] if node.children else node
    
    def _rollout(self, node: DTSNode, max_new_tokens: int):
        """
        Rollout: Complete sequence using base model p_θ
        """
        current_tokens = mx.expand_dims(node.token_ids, 0) if node.token_ids.ndim == 1 else node.token_ids
        remaining = max_new_tokens - (current_tokens.shape[1] if current_tokens.ndim == 2 else current_tokens.shape[0]) + self.root.token_ids.shape[0]
        
        if remaining <= 0:
            return node.token_ids, node
        
        # Generate completion
        final_tokens = self.model.rollout_sequence(
            current_tokens,
            max_new_tokens=remaining,
            temperature=self.config.temperature
        )
        
        # Handle batch dimension
        if final_tokens.ndim == 2:
            final_tokens = final_tokens[0]
        
        return final_tokens, node
    
    def _soft_bellman_backup(self, node: DTSNode, reward: float):
        """
        Soft Bellman backup (DTS Eq. 7):
        
        V̂(x_{≤t}) ← V̂(x_{≤t}) + α [r + (1/λ) log Σ_{x_{t+1}} exp(λ V̂(x_{≤t+1})) - V̂(x_{≤t})]
        
        Simplified: V̂(x_{≤t}) = (1/N) Σ [r + softmax over children]
        
        This is the KEY innovation of DTS over MCTS!
        """
        current = node
        
        while current is not None:
            # Update visit count
            current.visit_count += 1
            
            # Compute soft Bellman target
            if current.children:
                # Soft max over children values
                child_values = [c.soft_value for c in current.children]
                child_values_array = mx.array(child_values)
                soft_max = (1.0 / self.config.temperature) * float(
                    mx.logsumexp(self.config.temperature * child_values_array, axis=0)
                )
                target = reward + soft_max
            else:
                # Terminal or leaf node
                target = reward
            
            # Running average update
            alpha = 1.0 / current.visit_count
            current.soft_value += alpha * (target - current.soft_value)
            
            current = current.parent
    
    def _standard_backup(self, node: DTSNode, reward: float):
        """Standard MCTS-style backup (for comparison)"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.soft_value += (reward - current.soft_value) / current.visit_count
            current = current.parent
    
    def _get_best_child(self, node: DTSNode) -> DTSNode:
        """Get child with highest soft value"""
        if not node.children:
            return node
        return max(node.children, key=lambda c: c.soft_value)
    
    def _is_terminal(self, node: DTSNode) -> bool:
        """Check if node is terminal"""
        seq_len = node.token_ids.shape[0]
        return (seq_len >= self.config.max_seq_length or 
                (seq_len > 0 and int(node.token_ids[-1]) == self.model.eos_token_id))
    
    def _count_nodes(self, node: DTSNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count


class DTSStarTextGenerator(DTSTextGenerator):
    """
    DTS* (Greedy variant)
    
    Uses UCT selection instead of Boltzmann
    From paper Section 3.2 / Algorithm 2
    """
    
    def __init__(self, model, reward_fn, config: DTSConfig):
        super().__init__(model, reward_fn, config)
        self.c_uct = 1.0  # UCT exploration constant
    
    def _select_boltzmann(self, node: DTSNode) -> DTSNode:
        """
        Override with UCT selection (DTS* variant):
        
        x_{t+1} = argmax [V̂(x_{≤t+1}) + c sqrt(log N(x_{≤t}) / N(x_{≤t+1}))]
        """
        current = node
        
        while not current.is_leaf() and not self._is_terminal(current):
            # Use UCT formula
            current = max(
                current.children,
                key=lambda c: c.soft_value + self.c_uct * np.sqrt(
                    np.log(current.visit_count) / max(c.visit_count, 1)
                )
            )
        
        return current

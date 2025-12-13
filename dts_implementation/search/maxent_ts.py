"""
Maximum Entropy Tree Search for Autoregressive Models (MaxEnt-TS)

Implements token-level tree search adapted from Diffusion Tree Sampling (DTS).

Mathematical Framework from MaximumEntropyTreeSearchforAutoregressive.md:
- State: Partial token sequence x_{≤t}
- Transition: p_θ(x_{t+1}|x_{≤t}) from OpenTSLM
- Soft Bellman: V_t(x_{≤t}) = (1/λ) log E[exp(λ V_{t+1}(x_{≤t+1}))]
- Optimal Policy: π*(x_{≤t+1}|x_{≤t}) ∝ p_θ(x_{t+1}|x_{≤t}) exp(λ V_{t+1}(x_{≤t+1}))

Algorithm Steps:
1. SELECTION: Navigate tree using Boltzmann/UCT policy
2. EXPANSION: Generate new children (top-k next tokens)
3. ROLLOUT: Complete sequence using base model
4. BACKUP: Propagate rewards with Soft Bellman
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dts_node import MCTSNode, MetaRootNode
from core.soft_bellman import soft_bellman_backup, sample_child_boltzmann
from models.local_loader import LocalOpenTSLMWrapper
from rewards.spectral_reward import SpectralReward


@dataclass
class MaxEntTSConfig:
    """Configuration for MaxEnt-TS"""
    
    # Tree search parameters
    num_rollouts: int = 100  # M in DTS paper
    temperature: float = 1.0  # λ (lambda) in equations
    max_seq_length: int = 200  # Maximum sequence length
    
    # Expansion parameters
    expansion_k: int = 5  # Top-k tokens to expand
    expansion_temperature: float = 1.0
    
    # Rollout parameters
    rollout_temperature: float = 0.8
    rollout_top_k: Optional[int] = 50
    rollout_top_p: Optional[float] = 0.9
    
    # UCT parameters (for DTS* variant)
    use_uct: bool = False  # If True, use UCT instead of Boltzmann
    c_uct: float = 1.0  # UCT exploration constant
    
    # Spectral reward parameters
    gamma: float = 1.0  # Spectral penalty weight
    spectral_metric: str = 'l1'
    
    # Efficiency
    use_kv_cache: bool = True
    verbose: bool = True


class TokenNode(MCTSNode):
    """
    Node for token-level tree search
    
    Extends MCTSNode with token-specific data:
    - token_ids: Current token sequence
    - kv_cache: Cached key-values for efficiency
    """
    
    def __init__(
        self,
        token_ids: torch.Tensor,
        t: int,  # Sequence position (0 = start, T = end)
        parent: Optional['TokenNode'] = None,
        kv_cache: Optional[tuple] = None
    ):
        # Use token_ids as x_t for MCTSNode
        super().__init__(x_t=token_ids, t=t, parent=parent)
        self.token_ids = token_ids
        self.kv_cache = kv_cache
        
        # Token-specific attributes
        self.token_logprobs: Dict[int, float] = {}  # p_θ(token|prefix)
        self.token_children: Dict[int, 'TokenNode'] = {}  # Map token_id -> child node
    
    def add_token_child(self, token_id: int, child_node: 'TokenNode'):
        """Add a child node for a specific token"""
        self.token_children[token_id] = child_node
        self.add_child(child_node)
    
    def get_token_child(self, token_id: int) -> Optional['TokenNode']:
        """Get child node for a specific token"""
        return self.token_children.get(token_id, None)
    
    def is_terminal(self, max_length: int, eos_token_id: int) -> bool:
        """Check if this is a terminal node"""
        # Terminal if: reached max length OR generated EOS token
        
        # Handle both 1D (MLX) and 2D (PyTorch) arrays
        if hasattr(self.token_ids, 'ndim'):
            if self.token_ids.ndim == 1:
                seq_len = self.token_ids.shape[0]
                last_token = int(self.token_ids[-1]) if seq_len > 0 else None
            else:
                seq_len = self.token_ids.shape[-1]
                last_token = self.token_ids[0, -1].item() if seq_len > 0 else None
        else:
            # Fallback for other types
            seq_len = len(self.token_ids) if hasattr(self.token_ids, '__len__') else 0
            last_token = int(self.token_ids[-1]) if seq_len > 0 else None
        
        return (seq_len >= max_length) or (last_token == eos_token_id)


class MaxEntTS:
    """
    Maximum Entropy Tree Search for Autoregressive Models
    
    Adapts Diffusion Tree Sampling to token generation with OpenTSLM.
    """
    
    def __init__(
        self,
        model: LocalOpenTSLMWrapper,
        reward: SpectralReward,
        config: MaxEntTSConfig
    ):
        """
        Initialize MaxEnt-TS
        
        Args:
            model: Wrapped OpenTSLM model (provides p_θ)
            reward: Spectral reward computer
            config: Search configuration
        """
        self.model = model
        self.reward = reward
        self.config = config
        
        self.root: Optional[TokenNode] = None
        self.rollout_count = 0
        
        if config.verbose:
            print(f"🌳 MaxEnt-TS initialized:")
            print(f"   Rollouts: {config.num_rollouts}")
            print(f"   Temperature (λ): {config.temperature}")
            print(f"   Expansion top-k: {config.expansion_k}")
            print(f"   Max sequence length: {config.max_seq_length}")
            print(f"   Spectral γ: {config.gamma}")
    
    def initialize_root(self, prompt_tokens: torch.Tensor):
        """
        Initialize root node with prompt
        
        Args:
            prompt_tokens: Initial token sequence [1, seq_len]
        """
        # Handle both 1D (MLX) and 2D (PyTorch) arrays
        if hasattr(prompt_tokens, 'ndim') and prompt_tokens.ndim == 1:
            prompt_len = prompt_tokens.shape[0]
        else:
            prompt_len = prompt_tokens.shape[-1]
        
        self.root = TokenNode(
            token_ids=prompt_tokens,
            t=prompt_len,  # Start at prompt length
            parent=None
        )
        self.rollout_count = 0
        
        if self.config.verbose:
            decoded = self.model.decode_sequence(prompt_tokens)[0]
            print(f"\n🌱 Root initialized:")
            print(f"   Prompt: '{decoded}'")
            print(f"   Length: {prompt_tokens.shape[-1]} tokens")
    
    def select(self, node: TokenNode) -> TokenNode:
        """
        SELECTION phase: Navigate tree to leaf
        
        Uses Boltzmann policy (DTS) or UCT (DTS*).
        
        Args:
            node: Current node
        
        Returns:
            Selected leaf node for expansion
        """
        current = node
        
        while not current.is_leaf() and not current.is_terminal(
            self.config.max_seq_length,
            self.model.eos_token_id
        ):
            if self.config.use_uct:
                # UCT selection (DTS*)
                current = self._select_uct(current)
            else:
                # Boltzmann selection (DTS)
                current = sample_child_boltzmann(
                    current,
                    temperature=self.config.temperature
                )
        
        return current
    
    def _select_uct(self, node: TokenNode) -> TokenNode:
        """
        UCT selection for DTS*
        
        x_{t+1} = argmax [V(x_{≤t+1}) + c * sqrt(log N(x_{≤t}) / N(x_{≤t+1}))]
        """
        best_score = float('-inf')
        best_child = None
        
        log_parent_visits = np.log(node.visit_count + 1)
        
        for child in node.children:
            # Value term
            value = child.value_est
            
            # Exploration bonus
            exploration = self.config.c_uct * np.sqrt(
                log_parent_visits / (child.visit_count + 1)
            )
            
            score = value + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, node: TokenNode) -> TokenNode:
        """
        EXPANSION phase: Generate new child nodes
        
        Sample top-k next tokens from p_θ(x_{t+1}|x_{≤t})
        
        Args:
            node: Node to expand
        
        Returns:
            Newly created child node
        """
        # Get top-k next tokens
        top_tokens, top_probs = self.model.get_top_k_tokens(
            node.token_ids,
            k=self.config.expansion_k,
            temperature=self.config.expansion_temperature
        )
        
        # Create children for top-k tokens
        children_created = []
        
        for i in range(top_tokens.shape[1]):
            token_id = top_tokens[0, i].item()
            token_prob = top_probs[0, i].item()
            
            # Skip if already expanded
            if node.get_token_child(token_id) is not None:
                continue
            
            # Create new token sequence
            # Handle both MLX (1D) and PyTorch (2D) arrays
            if hasattr(node.token_ids, 'ndim') and node.token_ids.ndim == 1:
                # MLX 1D array - convert to list, append, convert back
                import mlx.core as mx
                token_list = node.token_ids.tolist() if hasattr(node.token_ids, 'tolist') else list(node.token_ids)
                token_list.append(token_id)
                new_tokens = mx.array(token_list)
            else:
                # PyTorch 2D tensor
                new_tokens = torch.cat([
                    node.token_ids,
                    torch.tensor([[token_id]], device=node.token_ids.device)
                ], dim=1)
            
            # Create child node
            child = TokenNode(
                token_ids=new_tokens,
                t=node.t + 1,
                parent=node,
                kv_cache=None  # TODO: Implement KV cache
            )
            
            # Store token probability
            node.token_logprobs[token_id] = np.log(token_prob)
            
            # Add to parent
            node.add_token_child(token_id, child)
            children_created.append(child)
        
        # Return first created child (or random if multiple)
        if children_created:
            return np.random.choice(children_created)
        else:
            return node  # All tokens already expanded
    
    def rollout(self, node: TokenNode) -> Tuple[torch.Tensor, str]:
        """
        ROLLOUT phase: Complete sequence from current node
        
        Uses base OpenTSLM model p_θ to generate remaining tokens.
        
        Args:
            node: Starting node
        
        Returns:
            complete_sequence: Full token sequence
            decoded_text: Decoded text
        """
        # Check if already terminal
        if node.is_terminal(self.config.max_seq_length, self.model.eos_token_id):
            return node.token_ids, self.model.decode_sequence(node.token_ids)[0]
        
        # Generate remaining tokens
        # Handle both 1D (MLX) and 2D (PyTorch) arrays
        if hasattr(node.token_ids, 'ndim') and node.token_ids.ndim == 1:
            current_len = node.token_ids.shape[0]
        else:
            current_len = node.token_ids.shape[-1]
        remaining_tokens = self.config.max_seq_length - current_len
        
        complete_sequence = self.model.rollout_sequence(
            node.token_ids,
            max_new_tokens=remaining_tokens,
            temperature=self.config.rollout_temperature,
            top_k=self.config.rollout_top_k,
            top_p=self.config.rollout_top_p,
            return_full_sequence=True
        )
        
        decoded = self.model.decode_sequence(complete_sequence)[0]
        
        return complete_sequence, decoded
    
    def evaluate_reward(
        self,
        decoded_text: str,
        ground_truth: Optional[Dict] = None
    ) -> float:
        """
        Evaluate terminal reward r(x)
        
        This should:
        1. Parse decoded text to extract time series prediction
        2. Compute spectral reward
        
        Args:
            decoded_text: Model-generated text
            ground_truth: Optional ground truth data
        
        Returns:
            reward: Total reward (task + spectral)
        """
        # TODO: Implement proper parsing of OpenTSLM output
        # For now, return a placeholder
        
        # In real implementation, we need to:
        # 1. Extract the predicted time series from decoded_text
        # 2. Convert to numpy array
        # 3. Compute spectral reward
        
        # Placeholder: random reward for testing
        reward = np.random.randn()
        
        return reward
    
    def backup(self, node: TokenNode, reward: float):
        """
        BACKUP phase: Propagate reward up the tree
        
        Uses Soft Bellman backup from core.soft_bellman
        
        Args:
            node: Terminal node
            reward: Reward to propagate
        """
        soft_bellman_backup(
            node,
            reward,
            temperature=self.config.temperature
        )
    
    def search(
        self,
        prompt_tokens: torch.Tensor,
        ground_truth: Optional[Dict] = None
    ) -> Dict:
        """
        Run full MaxEnt-TS search
        
        Algorithm:
        1. Initialize root with prompt
        2. For M rollouts:
            a. SELECT: Navigate to leaf
            b. EXPAND: Create new children
            c. ROLLOUT: Complete sequence
            d. EVALUATE: Compute reward
            e. BACKUP: Propagate with Soft Bellman
        3. Return best sequence
        
        Args:
            prompt_tokens: Initial prompt [1, seq_len]
            ground_truth: Optional ground truth for evaluation
        
        Returns:
            Dict with:
                - best_sequence: Best token sequence
                - best_text: Decoded text
                - best_reward: Reward value
                - tree_stats: Tree statistics
        """
        # Initialize
        self.initialize_root(prompt_tokens)
        
        if self.config.verbose:
            print(f"\n🔍 Starting search with {self.config.num_rollouts} rollouts...")
        
        # Main search loop
        for rollout_idx in range(self.config.num_rollouts):
            # 1. SELECT
            leaf = self.select(self.root)
            
            # 2. EXPAND (if not terminal)
            if not leaf.is_terminal(self.config.max_seq_length, self.model.eos_token_id):
                expanded_node = self.expand(leaf)
            else:
                expanded_node = leaf
            
            # 3. ROLLOUT
            complete_seq, decoded_text = self.rollout(expanded_node)
            
            # 4. EVALUATE
            reward = self.evaluate_reward(decoded_text, ground_truth)
            
            # 5. BACKUP
            self.backup(expanded_node, reward)
            
            self.rollout_count += 1
            
            if self.config.verbose and (rollout_idx + 1) % 10 == 0:
                print(f"   Rollout {rollout_idx+1}/{self.config.num_rollouts} | "
                      f"Reward: {reward:.4f} | "
                      f"Tree size: {self._count_nodes()}")
        
        # Extract best path
        best_sequence, best_text, best_reward = self._extract_best_path()
        
        if self.config.verbose:
            print(f"\n✅ Search complete!")
            print(f"   Total nodes: {self._count_nodes()}")
            print(f"   Best reward: {best_reward:.4f}")
            print(f"   Best output: '{best_text[:100]}...'")
        
        return {
            'best_sequence': best_sequence,
            'best_text': best_text,
            'best_reward': best_reward,
            'tree_stats': self._get_tree_stats()
        }
    
    def _extract_best_path(self) -> Tuple[torch.Tensor, str, float]:
        """
        Extract highest-reward path from tree
        
        Returns:
            best_sequence: Token sequence
            best_text: Decoded text
            best_reward: Reward value
        """
        # Traverse tree, always picking best child
        current = self.root
        
        while not current.is_leaf():
            # Pick child with highest value
            best_child = max(current.children, key=lambda c: c.value_est)
            current = best_child
        
        # If leaf is not terminal, complete with rollout
        if not current.is_terminal(self.config.max_seq_length, self.model.eos_token_id):
            complete_seq, decoded = self.rollout(current)
        else:
            complete_seq = current.token_ids
            decoded = self.model.decode_sequence(complete_seq)[0]
        
        reward = current.value_est
        
        return complete_seq, decoded, reward
    
    def _count_nodes(self) -> int:
        """Count total nodes in tree"""
        if self.root is None:
            return 0
        
        count = 0
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            count += 1
            queue.extend(node.children)
        
        return count
    
    def _get_tree_stats(self) -> Dict:
        """Get tree statistics"""
        if self.root is None:
            return {}
        
        total_nodes = self._count_nodes()
        max_depth = self._get_max_depth()
        avg_branching = self._get_avg_branching()
        
        return {
            'total_nodes': total_nodes,
            'max_depth': max_depth,
            'avg_branching_factor': avg_branching,
            'rollouts': self.rollout_count
        }
    
    def _get_max_depth(self) -> int:
        """Get maximum tree depth"""
        if self.root is None:
            return 0
        
        max_depth = 0
        queue = [(self.root, 0)]
        
        while queue:
            node, depth = queue.pop(0)
            max_depth = max(max_depth, depth)
            for child in node.children:
                queue.append((child, depth + 1))
        
        return max_depth
    
    def _get_avg_branching(self) -> float:
        """Get average branching factor"""
        if self.root is None:
            return 0.0
        
        total_children = 0
        total_nodes = 0
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            if not node.is_leaf():
                total_children += len(node.children)
                total_nodes += 1
            queue.extend(node.children)
        
        return total_children / total_nodes if total_nodes > 0 else 0.0


if __name__ == "__main__":
    print("Testing MaxEnt-TS...")
    
    # This requires a trained OpenTSLM model
    # For now, just verify imports work
    print("✅ Imports successful!")
    print("   - TokenNode")
    print("   - MaxEntTS")
    print("   - MaxEntTSConfig")
    
    print("\nTo run full test, load an OpenTSLM model and run search()")


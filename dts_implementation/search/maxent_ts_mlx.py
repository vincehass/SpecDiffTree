"""
Pure MLX Maximum Entropy Tree Search for Autoregressive Models (MaxEnt-TS)

Implements token-level tree search adapted from Diffusion Tree Sampling (DTS).
Pure MLX implementation - no PyTorch dependencies!

Algorithm Steps:
1. SELECTION: Navigate tree using Boltzmann policy
2. EXPANSION: Generate new children (top-k next tokens)
3. ROLLOUT: Complete sequence using base model
4. BACKUP: Propagate rewards with Soft Bellman
"""

import mlx.core as mx
import numpy as np
import time
from typing import Optional, Dict
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dts_node_mlx import MCTSNode
from core.soft_bellman_mlx import soft_bellman_backup, sample_child_boltzmann


@dataclass
class MaxEntTSConfig:
    """Configuration for MaxEnt-TS - Pure MLX"""
    
    # Tree search parameters
    num_rollouts: int = 10
    temperature: float = 1.0  # λ (lambda) in equations
    max_seq_length: int = 100
    
    # Expansion parameters
    expansion_k: int = 3
    expansion_temperature: float = 1.0
    
    # Rollout parameters
    rollout_temperature: float = 0.8
    rollout_max_new_tokens: int = 50
    
    # UCT parameters (for DTS* variant)
    use_uct: bool = False
    c_uct: float = 1.0
    
    # Efficiency
    early_stopping: bool = True
    verbose: bool = True


class TokenNode(MCTSNode):
    """
    Node for token-level tree search (Pure MLX)
    
    Extends MCTSNode with token-specific data.
    """
    
    def __init__(
        self,
        token_ids: mx.array,
        parent: Optional['TokenNode'] = None
    ):
        super().__init__(token_ids=token_ids, parent=parent)
        self.token_ids = token_ids
        self.token_children: Dict[int, 'TokenNode'] = {}
    
    def add_token_child(self, token_id: int, child_node: 'TokenNode'):
        """Add a child node for a specific token"""
        self.token_children[token_id] = child_node
        self.add_child(child_node)
    
    def get_token_child(self, token_id: int) -> Optional['TokenNode']:
        """Get child node for a specific token"""
        return self.token_children.get(token_id, None)


class MaxEntTS:
    """
    Pure MLX Maximum Entropy Tree Search for Autoregressive Models
    
    Adapts Diffusion Tree Sampling to token generation.
    """
    
    def __init__(self, model, reward_fn, config: MaxEntTSConfig):
        """
        Initialize MaxEnt-TS
        
        Args:
            model: MLX model wrapper
            reward_fn: Reward function
            config: Search configuration
        """
        self.model = model
        self.reward_fn = reward_fn
        self.config = config
        self.root: Optional[TokenNode] = None
        self.rollout_count = 0
        
        if config.verbose:
            print(f"🌳 MaxEnt-TS (Pure MLX) initialized:")
            print(f"   Rollouts: {config.num_rollouts}")
            print(f"   Temperature (λ): {config.temperature}")
            print(f"   Expansion top-k: {config.expansion_k}")
    
    def search(self, prompt_tokens: mx.array, max_new_tokens: int = 50) -> Dict:
        """
        Run MaxEnt-TS search
        
        Args:
            prompt_tokens: Initial prompt (MLX array)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict with best sequence and statistics
        """
        # Initialize root
        if not isinstance(prompt_tokens, mx.array):
            if hasattr(prompt_tokens, 'tolist'):
                prompt_tokens = mx.array(prompt_tokens.tolist())
            else:
                prompt_tokens = mx.array(prompt_tokens)
        
        if prompt_tokens.ndim == 2:
            prompt_tokens = prompt_tokens[0]
        
        self.root = TokenNode(token_ids=prompt_tokens, parent=None)
        self.rollout_count = 0
        start_time = time.time()
        
        if self.config.verbose:
            print(f"\n🌱 Starting MaxEnt-TS search...")
        
        # Run rollouts
        for rollout_idx in range(self.config.num_rollouts):
            # 1. SELECTION
            leaf = self.select(self.root)
            
            # 2. EXPANSION
            if not self._is_terminal(leaf):
                expanded = self.expand(leaf)
            else:
                expanded = leaf
            
            # 3. ROLLOUT
            final_tokens = self.rollout(expanded, max_new_tokens)
            
            # 4. EVALUATE
            reward = self.reward_fn(mx.expand_dims(final_tokens, 0))
            
            # 5. BACKUP
            self._backup(expanded, float(reward))
            
            self.rollout_count += 1
            
            if self.config.verbose and (rollout_idx + 1) % 10 == 0:
                print(f"  Rollout {rollout_idx+1}/{self.config.num_rollouts}")
        
        # Extract best sequence
        best_node = self._get_best_leaf()
        best_sequence = best_node.token_ids
        
        if self.config.verbose:
            print(f"✅ MaxEnt-TS complete. Best value: {best_node.value_est:.4f}")
        
        # Decode best sequence
        best_text = self.model.tokenizer.decode(best_sequence.tolist(), skip_special_tokens=True)
        
        return {
            'best_node': best_node,
            'best_sequence': best_sequence,
            'best_text': best_text,
            'best_value': best_node.value_est,
            'nodes_explored': self._count_nodes(self.root),
            'total_rollouts': self.config.num_rollouts,
            'time': time.time() - start_time
        }
    
    def select(self, node: TokenNode) -> TokenNode:
        """
        SELECTION phase: Navigate tree to leaf using Boltzmann policy
        
        Args:
            node: Current node
        
        Returns:
            Selected leaf node
        """
        current = node
        
        while not current.is_leaf() and not self._is_terminal(current):
            if self.config.use_uct:
                current = self._select_uct(current)
            else:
                current = sample_child_boltzmann(
                    current,
                    temperature=self.config.temperature
                )
                if current is None:
                    break
        
        return current if current is not None else node
    
    def _select_uct(self, node: TokenNode) -> TokenNode:
        """UCT selection for DTS*"""
        best_score = float('-inf')
        best_child = None
        
        log_parent_visits = np.log(node.visit_count + 1)
        
        for child in node.children:
            value = child.value_est
            exploration = self.config.c_uct * np.sqrt(
                log_parent_visits / (child.visit_count + 1)
            )
            score = value + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child if best_child is not None else node.children[0]
    
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
            k=self.config.expansion_k
        )
        
        # Create children for top-k tokens
        children_created = []
        
        # Convert to lists if needed
        if isinstance(top_tokens, mx.array):
            tokens_list = top_tokens.tolist()
            probs_list = top_probs.tolist()
        else:
            tokens_list = list(top_tokens)
            probs_list = list(top_probs)
        
        for token_id, token_prob in zip(tokens_list, probs_list):
            # Skip if already expanded
            if node.get_token_child(int(token_id)) is not None:
                continue
            
            # Create new token sequence
            new_tokens = mx.concatenate([node.token_ids, mx.array([int(token_id)])])
            
            # Create child node
            child = TokenNode(token_ids=new_tokens, parent=node)
            node.add_token_child(int(token_id), child)
            children_created.append(child)
        
        # Return first created child (or self if none created)
        return children_created[0] if children_created else node
    
    def rollout(self, node: TokenNode, max_new_tokens: int) -> mx.array:
        """
        ROLLOUT phase: Complete sequence using base model
        
        Args:
            node: Starting node
            max_new_tokens: Maximum new tokens
        
        Returns:
            Final token sequence (MLX array)
        """
        current_tokens = node.token_ids
        remaining = max_new_tokens - (current_tokens.shape[0] - self.root.token_ids.shape[0])
        
        if remaining <= 0:
            return current_tokens
        
        # Complete sequence
        final_tokens = self.model.rollout_sequence(
            mx.expand_dims(current_tokens, 0),
            max_new_tokens=remaining,
            temperature=self.config.rollout_temperature
        )
        
        # Handle batch dimension
        if final_tokens.ndim == 2:
            final_tokens = final_tokens[0]
        
        return final_tokens
    
    def _backup(self, node: TokenNode, reward: float):
        """
        BACKUP phase: Propagate reward using Soft Bellman
        
        Args:
            node: Node to start backup from
            reward: Terminal reward
        """
        soft_bellman_backup(node, reward, self.config.temperature)
    
    def _is_terminal(self, node: TokenNode) -> bool:
        """Check if node is terminal"""
        seq_len = node.token_ids.shape[0]
        
        # Check max length
        if seq_len >= self.config.max_seq_length:
            return True
        
        # Check EOS token
        if seq_len > 0:
            last_token = int(node.token_ids[-1])
            if last_token == self.model.eos_token_id:
                return True
        
        return False
    
    def _get_best_leaf(self) -> TokenNode:
        """Get leaf node with highest value"""
        all_nodes = self._get_all_nodes(self.root)
        leaves = [n for n in all_nodes if n.is_leaf() or self._is_terminal(n)]
        
        if not leaves:
            return self.root
        
        return max(leaves, key=lambda n: n.value_est)
    
    def _get_all_nodes(self, node: TokenNode) -> list:
        """Get all nodes in tree via BFS"""
        nodes = []
        queue = [node]
        
        while queue:
            current = queue.pop(0)
            nodes.append(current)
            queue.extend(current.children)
        
        return nodes
    
    def _count_nodes(self, node: TokenNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

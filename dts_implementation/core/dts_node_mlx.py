"""
Pure MLX DTS Node Implementation
Adapted from dts_node.py for pure MLX (no PyTorch dependencies)

Each node represents a token sequence state in autoregressive generation.
"""

import mlx.core as mx
import numpy as np
from typing import Optional, List


class MCTSNode:
    """
    Pure MLX MCTS Tree Node for Token-level Tree Search
    
    Represents a partial token sequence.
    Children are possible next token extensions.
    
    Attributes:
        token_ids (mx.array): Current token sequence
        parent (MCTSNode): Parent node in the tree
        children (List[MCTSNode]): List of child nodes
        visit_count (int): Number of times this node has been visited
        value_est (float): Estimated soft value V(token_ids)
        kv_cache: Cached key-values for efficiency (optional)
    """
    
    def __init__(
        self,
        token_ids: mx.array,
        parent: Optional['MCTSNode'] = None,
        kv_cache=None
    ):
        """
        Initialize MCTS Node
        
        Args:
            token_ids: Token sequence (MLX array)
            parent: Parent node (None for root)
            kv_cache: Optional KV cache for efficiency
        """
        self.token_ids = token_ids
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visit_count = 1  # Initialize to 1 (as per DTS paper)
        self.value_est = 0.0  # Soft value estimate
        self.kv_cache = kv_cache  # For O(n) instead of O(n²)
        
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)"""
        return len(self.children) == 0
    
    def is_terminal(self, max_length: int, eos_token_id: Optional[int] = None) -> bool:
        """
        Check if node represents a terminal state
        
        Args:
            max_length: Maximum sequence length
            eos_token_id: EOS token ID (optional)
            
        Returns:
            True if terminal (reached max length or EOS)
        """
        seq_len = self.token_ids.shape[0] if self.token_ids.ndim == 1 else self.token_ids.shape[1]
        
        # Check max length
        if seq_len >= max_length:
            return True
        
        # Check EOS token
        if eos_token_id is not None and seq_len > 0:
            last_token = int(self.token_ids[-1]) if self.token_ids.ndim == 1 else int(self.token_ids[0, -1])
            if last_token == eos_token_id:
                return True
        
        return False
    
    def update_value(self, value: float):
        """
        Update node's value estimate
        
        Args:
            value: New value estimate
        """
        self.value_est = value
    
    def add_child(self, child: 'MCTSNode'):
        """
        Add a child node
        
        Args:
            child: Child node to add
        """
        self.children.append(child)
    
    def increment_visit(self):
        """Increment visit count"""
        self.visit_count += 1
    
    def get_path_to_root(self) -> List['MCTSNode']:
        """
        Get path from this node to root
        
        Returns:
            List of nodes from current to root
        """
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path
    
    def get_depth(self) -> int:
        """
        Get depth of node in tree (distance from root)
        
        Returns:
            Depth (0 for root)
        """
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth
    
    def get_sequence_length(self) -> int:
        """Get length of token sequence"""
        return self.token_ids.shape[0] if self.token_ids.ndim == 1 else self.token_ids.shape[1]
    
    def __repr__(self) -> str:
        return (f"MCTSNode(seq_len={self.get_sequence_length()}, "
                f"visits={self.visit_count}, value={self.value_est:.4f}, "
                f"children={len(self.children)})")


class MetaRootNode(MCTSNode):
    """
    Special meta-root node for initializing tree search.
    Its children are created from the initial prompt.
    """
    
    def __init__(self, prompt_tokens: mx.array):
        """
        Initialize meta-root node
        
        Args:
            prompt_tokens: Initial prompt tokens (MLX array)
        """
        super().__init__(
            token_ids=prompt_tokens,
            parent=None,
            kv_cache=None
        )
    
    def is_leaf(self) -> bool:
        """Meta-root is leaf only if it has no children"""
        return len(self.children) == 0
    
    def is_terminal(self, max_length: int, eos_token_id: Optional[int] = None) -> bool:
        """Meta-root is never terminal"""
        return False


class DTSTree:
    """
    Container for DTS search tree with utility methods
    
    Tracks the entire tree structure and provides methods for:
    - Tree traversal
    - Statistics collection
    - Trajectory extraction
    """
    
    def __init__(self, root: MCTSNode):
        """
        Initialize tree
        
        Args:
            root: Root node of the tree
        """
        self.root = root
        self.num_rollouts = 0
        self.total_nfes = 0
    
    def get_all_nodes(self) -> List[MCTSNode]:
        """
        Get all nodes in tree via BFS
        
        Returns:
            List of all nodes
        """
        nodes = []
        queue = [self.root]
        
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue.extend(node.children)
        
        return nodes
    
    def get_terminal_nodes(self, max_length: int, eos_token_id: Optional[int] = None) -> List[MCTSNode]:
        """
        Get all terminal nodes
        
        Args:
            max_length: Maximum sequence length
            eos_token_id: EOS token ID (optional)
            
        Returns:
            List of terminal nodes
        """
        all_nodes = self.get_all_nodes()
        return [node for node in all_nodes if node.is_terminal(max_length, eos_token_id)]
    
    def get_best_terminal_node(self, max_length: int, eos_token_id: Optional[int] = None) -> MCTSNode:
        """
        Get terminal node with highest value
        
        Args:
            max_length: Maximum sequence length
            eos_token_id: EOS token ID (optional)
            
        Returns:
            Best terminal node
        """
        terminal_nodes = self.get_terminal_nodes(max_length, eos_token_id)
        if not terminal_nodes:
            # If no terminal nodes, return best leaf
            all_nodes = self.get_all_nodes()
            leaves = [n for n in all_nodes if n.is_leaf()]
            if not leaves:
                return self.root
            return max(leaves, key=lambda n: n.value_est)
        
        return max(terminal_nodes, key=lambda n: n.value_est)
    
    def get_tree_stats(self) -> dict:
        """
        Get tree statistics
        
        Returns:
            Dictionary of statistics
        """
        nodes = self.get_all_nodes()
        
        if not nodes:
            return {
                'total_nodes': 0,
                'max_depth': 0,
                'avg_visits': 0,
                'avg_value': 0,
                'num_rollouts': self.num_rollouts,
                'total_nfes': self.total_nfes
            }
        
        return {
            'total_nodes': len(nodes),
            'max_depth': max(node.get_depth() for node in nodes),
            'avg_visits': np.mean([node.visit_count for node in nodes]),
            'avg_value': np.mean([node.value_est for node in nodes]),
            'num_rollouts': self.num_rollouts,
            'total_nfes': self.total_nfes
        }
    
    def extract_best_trajectory(self, max_length: int, eos_token_id: Optional[int] = None) -> List[mx.array]:
        """
        Extract best trajectory from root to best terminal node
        
        Args:
            max_length: Maximum sequence length
            eos_token_id: EOS token ID (optional)
            
        Returns:
            List of token sequences [initial, ..., final]
        """
        best_terminal = self.get_best_terminal_node(max_length, eos_token_id)
        path = best_terminal.get_path_to_root()
        path.reverse()  # Root to terminal
        
        return [node.token_ids for node in path]
    
    def __repr__(self) -> str:
        stats = self.get_tree_stats()
        return (f"DTSTree(nodes={stats['total_nodes']}, "
                f"depth={stats['max_depth']}, "
                f"rollouts={stats['num_rollouts']})")

"""
DTS Node Implementation for OpenTSLM
Adapted from: https://github.com/vineetjain96/Diffusion-Tree-Sampling

Each node represents a state x_t at diffusion timestep t in the reverse process.
"""

import torch
import numpy as np
from typing import Optional, List


class MCTSNode:
    """
    MCTS Tree Node for Diffusion Tree Sampling
    
    Represents a state x_t at diffusion timestep t.
    Children are possible next states x_{t-1} sampled from p_θ(x_{t-1} | x_t).
    
    Attributes:
        x_t (torch.Tensor): The noisy sample at timestep t. Shape: [batch_size, ...]
        t (int): Current diffusion timestep index (T → 0)
        parent (MCTSNode): Parent node in the tree
        children (List[MCTSNode]): List of child nodes
        visit_count (int): Number of times this node has been visited
        value_est (float): Estimated soft value V_t(x_t)
        noise_pred (torch.Tensor): Cached noise prediction for efficiency
    """
    
    def __init__(
        self,
        x_t: torch.Tensor,
        t: int,
        parent: Optional['MCTSNode'] = None,
        timestep_value: Optional[float] = None
    ):
        """
        Initialize MCTS Node
        
        Args:
            x_t: Noisy sample at timestep t
            t: Timestep index (T → 0)
            parent: Parent node (None for root)
            timestep_value: Actual scheduler timestep value (optional)
        """
        self.x_t = x_t
        self.t = t
        self.timestep_value = timestep_value
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visit_count = 1  # Initialize to 1 (as per DTS paper)
        self.value_est = 0.0  # Soft value estimate V_t(x_t)
        self.noise_pred: Optional[torch.Tensor] = None  # Cached for efficiency
        
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children or at t=0)"""
        return (self.t <= 0) or (len(self.children) == 0)
    
    def is_terminal(self) -> bool:
        """Check if node represents a terminal state (x_0)"""
        return self.t == 0
    
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
    
    def __repr__(self) -> str:
        return (f"MCTSNode(t={self.t}, visits={self.visit_count}, "
                f"value={self.value_est:.4f}, children={len(self.children)})")


class MetaRootNode(MCTSNode):
    """
    Special meta-root node that doesn't correspond to any actual diffusion timestep.
    Its children are created by sampling from random noise x_T ~ N(0, I).
    
    This allows the tree search to start from multiple initial noise samples.
    """
    
    def __init__(self, data_shape: tuple, device: str = 'cpu'):
        """
        Initialize meta-root node
        
        Args:
            data_shape: Shape of data (e.g., (1, seq_len, dim))
            device: Device to create tensors on
        """
        # Virtual timestep one more than max
        virtual_timestep = 1000  # Placeholder
        
        # Create dummy tensor with right shape
        dummy_x = torch.zeros(data_shape, device=device)
        
        # Initialize with virtual timestep
        super().__init__(
            x_t=dummy_x,
            t=virtual_timestep,
            parent=None,
            timestep_value=None
        )
    
    def is_leaf(self) -> bool:
        """Meta-root is never a leaf"""
        return len(self.children) == 0
    
    def is_terminal(self) -> bool:
        """Meta-root is never terminal"""
        return False


class DTSTree:
    """
    Container for DTS search tree with utility methods
    
    Tracks the entire tree structure and provides methods for:
    - Tree traversal
    - Statistics collection
    - Visualization
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
    
    def get_terminal_nodes(self) -> List[MCTSNode]:
        """
        Get all terminal nodes (x_0)
        
        Returns:
            List of terminal nodes
        """
        all_nodes = self.get_all_nodes()
        return [node for node in all_nodes if node.is_terminal()]
    
    def get_best_terminal_node(self) -> MCTSNode:
        """
        Get terminal node with highest value
        
        Returns:
            Best terminal node
        """
        terminal_nodes = self.get_terminal_nodes()
        if not terminal_nodes:
            raise ValueError("No terminal nodes in tree")
        
        return max(terminal_nodes, key=lambda n: n.value_est)
    
    def get_tree_stats(self) -> dict:
        """
        Get tree statistics
        
        Returns:
            Dictionary of statistics
        """
        nodes = self.get_all_nodes()
        terminal_nodes = self.get_terminal_nodes()
        
        return {
            'total_nodes': len(nodes),
            'terminal_nodes': len(terminal_nodes),
            'max_depth': max(node.get_depth() for node in nodes) if nodes else 0,
            'avg_visits': np.mean([node.visit_count for node in nodes]) if nodes else 0,
            'avg_value': np.mean([node.value_est for node in nodes]) if nodes else 0,
            'num_rollouts': self.num_rollouts,
            'total_nfes': self.total_nfes
        }
    
    def extract_best_trajectory(self) -> List[torch.Tensor]:
        """
        Extract best trajectory from root to best terminal node
        
        Returns:
            List of states [x_T, x_{T-1}, ..., x_0]
        """
        best_terminal = self.get_best_terminal_node()
        path = best_terminal.get_path_to_root()
        path.reverse()  # Root to terminal
        
        return [node.x_t for node in path]
    
    def __repr__(self) -> str:
        stats = self.get_tree_stats()
        return (f"DTSTree(nodes={stats['total_nodes']}, "
                f"terminals={stats['terminal_nodes']}, "
                f"depth={stats['max_depth']}, "
                f"rollouts={stats['num_rollouts']})")


if __name__ == "__main__":
    # Test node creation
    print("Testing MCTSNode...")
    
    # Create root node
    x_T = torch.randn(1, 10, 128)  # Example: batch=1, seq_len=10, dim=128
    root = MCTSNode(x_T, t=50)
    
    # Create children
    x_49 = torch.randn(1, 10, 128)
    child1 = MCTSNode(x_49, t=49, parent=root)
    root.add_child(child1)
    
    x_48 = torch.randn(1, 10, 128)
    child2 = MCTSNode(x_48, t=48, parent=child1)
    child1.add_child(child2)
    
    # Test methods
    print(f"Root: {root}")
    print(f"Root is leaf: {root.is_leaf()}")
    print(f"Child1: {child1}")
    print(f"Child2 depth: {child2.get_depth()}")
    print(f"Child2 path length: {len(child2.get_path_to_root())}")
    
    # Test tree
    tree = DTSTree(root)
    print(f"\nTree: {tree}")
    print(f"Stats: {tree.get_tree_stats()}")
    
    print("\n✅ MCTSNode tests passed!")


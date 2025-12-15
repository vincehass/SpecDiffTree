"""
Standard MCTS Baseline for Text Generation

Classic Monte Carlo Tree Search with UCT (Upper Confidence Bound for Trees)
Reference: Browne et al., "A Survey of Monte Carlo Tree Search Methods" (2012)
"""

import torch
import numpy as np
import time
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class MCTSConfig:
    """Configuration for MCTS"""
    num_simulations: int = 100
    c_puct: float = 1.0  # Exploration constant
    expansion_k: int = 4  # Number of children to expand
    max_seq_length: int = 512
    temperature: float = 1.0
    verbose: bool = False


class MCTSNode:
    """Node in MCTS tree for text generation"""
    
    def __init__(self, token_ids: torch.Tensor, parent: Optional['MCTSNode'] = None):
        self.token_ids = token_ids  # Sequence up to this node
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visit_count = 0
        self.total_reward = 0.0
        self.untried_actions: List[int] = []  # Tokens not yet expanded
        
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0
    
    def get_q_value(self) -> float:
        """Get average reward (exploitation)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count
    
    def get_uct_value(self, c_puct: float, parent_visits: int) -> float:
        """
        Get UCT value: Q(s,a) + c * sqrt(log(N(s)) / N(s,a))
        
        This is the classic UCB1 formula used in MCTS.
        """
        if self.visit_count == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        exploitation = self.get_q_value()
        exploration = c_puct * np.sqrt(np.log(parent_visits) / self.visit_count)
        return exploitation + exploration


class MCTSTextGenerator:
    """
    Standard MCTS for text generation
    
    Algorithm:
    1. Selection: Use UCT to select path to leaf
    2. Expansion: Add new children
    3. Simulation: Rollout to completion
    4. Backpropagation: Update values
    """
    
    def __init__(self, model, reward_fn, config: MCTSConfig):
        self.model = model
        self.reward_fn = reward_fn
        self.config = config
        self.root: Optional[MCTSNode] = None
        
    def search(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 200) -> Dict:
        """
        Run MCTS search
        
        Args:
            prompt_tokens: Initial prompt [batch_size, seq_len] or [seq_len]
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dict with best sequence and statistics
        """
        # Convert to tensor if needed
        if isinstance(prompt_tokens, list):
            prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)
        elif not isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = torch.tensor(list(prompt_tokens), dtype=torch.long)
        
        # Ensure correct dtype
        if prompt_tokens.dtype != torch.long:
            prompt_tokens = prompt_tokens.long()
        
        # Initialize root - handle batch dimension
        if prompt_tokens.dim() == 2:
            prompt_tokens = prompt_tokens[0]
        
        self.root = MCTSNode(prompt_tokens)
        start_time = time.time()
        
        if self.config.verbose:
            print(f"\n🌳 MCTS: Starting {self.config.num_simulations} simulations...")
        
        # Run simulations
        for sim in range(self.config.num_simulations):
            # 1. SELECTION
            node = self._select(self.root)
            
            # 2. EXPANSION
            if not self._is_terminal(node):
                node = self._expand(node)
            
            # 3. SIMULATION (ROLLOUT)
            reward = self._simulate(node, max_new_tokens)
            
            # 4. BACKPROPAGATION
            self._backpropagate(node, reward)
            
            if self.config.verbose and (sim + 1) % 10 == 0:
                print(f"  Simulation {sim+1}/{self.config.num_simulations}, "
                      f"Best Q: {self._get_best_child(self.root).get_q_value():.4f}")
        
        # Extract best path
        best_node = self._get_best_child(self.root)
        best_sequence = best_node.token_ids
        
        if self.config.verbose:
            print(f"\n✅ MCTS complete. Best Q-value: {best_node.get_q_value():.4f}")
        
        # Decode best sequence
        best_text = self.model.tokenizer.decode(best_sequence, skip_special_tokens=True)
        
        return {
            'best_node': best_node,
            'best_sequence': best_sequence,
            'best_text': best_text,
            'best_reward': best_node.get_q_value(),
            'nodes_explored': self._count_nodes(self.root),
            'total_simulations': self.config.num_simulations,
            'tree_size': self._count_nodes(self.root),
            'time': time.time() - start_time
        }
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Selection phase: Use UCT to navigate to a leaf
        """
        current = node
        
        while not current.is_leaf() and not self._is_terminal(current):
            if not current.is_fully_expanded():
                return current  # Return for expansion
            
            # Select child with highest UCT value
            current = max(
                current.children,
                key=lambda c: c.get_uct_value(self.config.c_puct, current.visit_count)
            )
        
        return current
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Add new child node
        """
        # Ensure node tokens are in correct format
        if isinstance(node.token_ids, list):
            node.token_ids = torch.tensor(node.token_ids, dtype=torch.long)
        elif node.token_ids.dtype != torch.long:
            node.token_ids = node.token_ids.long()
        
        # Get top-k tokens from model
        if len(node.untried_actions) == 0:
            # First expansion - get top-k tokens
            with torch.no_grad():
                input_ids = node.token_ids.unsqueeze(0)
                logits = self.model.get_next_token_logits(input_ids)
                top_tokens, top_probs = torch.topk(
                    torch.softmax(logits, dim=-1),
                    k=self.config.expansion_k
                )
                node.untried_actions = top_tokens.squeeze(0).tolist()
        
        # Pick an untried action
        action = node.untried_actions.pop(0)
        
        # Create child node with correct dtype
        new_tokens = torch.cat([
            node.token_ids, 
            torch.tensor([action], dtype=torch.long, device=node.token_ids.device)
        ])
        child = MCTSNode(new_tokens, parent=node)
        node.children.append(child)
        
        return child
    
    def _simulate(self, node: MCTSNode, max_new_tokens: int) -> float:
        """
        Simulation phase: Rollout to terminal state and evaluate
        """
        # Complete sequence using model's generate
        current_tokens = node.token_ids.unsqueeze(0)
        remaining = max_new_tokens - (current_tokens.shape[1] - self.root.token_ids.shape[0])
        
        if remaining <= 0:
            # Already at max length
            final_tokens = current_tokens
        else:
            # Rollout
            with torch.no_grad():
                final_tokens = self.model.rollout_sequence(
                    current_tokens,
                    max_new_tokens=remaining,
                    temperature=self.config.temperature
                )
        
        # Evaluate reward
        reward = self.reward_fn(final_tokens)
        
        return float(reward)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: Update values up the tree
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current = current.parent
    
    def _get_best_child(self, node: MCTSNode) -> MCTSNode:
        """Get child with highest Q-value (most visits)"""
        if not node.children:
            return node
        return max(node.children, key=lambda c: c.visit_count)
    
    def _is_terminal(self, node: MCTSNode) -> bool:
        """Check if node is terminal"""
        seq_len = node.token_ids.shape[0]
        return (seq_len >= self.config.max_seq_length or 
                (seq_len > 0 and int(node.token_ids[-1]) == self.model.eos_token_id))
    
    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count


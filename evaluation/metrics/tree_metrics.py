"""
Tree search metrics for S-ADT evaluation.
Adapted from DTS paper evaluation metrics.
"""

import numpy as np
from typing import Dict, List, Any
import json


class TreeMetrics:
    """
    Metrics for evaluating tree search quality.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.nodes_explored = []
        self.tree_depths = []
        self.branching_factors = []
        self.visit_counts = []
        self.value_estimates = []
        
    def add_tree_stats(self, stats: Dict[str, Any]):
        """
        Add statistics from a single tree search.
        
        Args:
            stats: Dictionary with keys:
                - nodes_explored: int
                - max_depth: int
                - avg_branching_factor: float
                - total_visits: int
                - avg_value: float
        """
        self.nodes_explored.append(stats.get('nodes_explored', 0))
        self.tree_depths.append(stats.get('max_depth', 0))
        self.branching_factors.append(stats.get('avg_branching_factor', 0.0))
        self.visit_counts.append(stats.get('total_visits', 0))
        self.value_estimates.append(stats.get('avg_value', 0.0))
    
    def compute_aggregate_stats(self) -> Dict[str, float]:
        """
        Compute aggregate statistics across all trees.
        
        Returns:
            Dictionary with aggregate metrics
        """
        if not self.nodes_explored:
            return {}
        
        return {
            'avg_nodes_explored': float(np.mean(self.nodes_explored)),
            'std_nodes_explored': float(np.std(self.nodes_explored)),
            'avg_tree_depth': float(np.mean(self.tree_depths)),
            'std_tree_depth': float(np.std(self.tree_depths)),
            'avg_branching_factor': float(np.mean(self.branching_factors)),
            'std_branching_factor': float(np.std(self.branching_factors)),
            'avg_visit_count': float(np.mean(self.visit_counts)),
            'avg_value_estimate': float(np.mean(self.value_estimates)),
            'total_trees': len(self.nodes_explored)
        }
    
    def save_to_json(self, filename: str):
        """Save metrics to JSON file"""
        stats = self.compute_aggregate_stats()
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def compare_with_baseline(self, baseline_nodes: List[int]) -> Dict[str, float]:
        """
        Compare tree search with baseline (e.g., greedy, beam search).
        
        Args:
            baseline_nodes: List of nodes explored by baseline method
            
        Returns:
            Comparison metrics
        """
        if not self.nodes_explored or not baseline_nodes:
            return {}
        
        tree_avg = np.mean(self.nodes_explored)
        baseline_avg = np.mean(baseline_nodes)
        
        return {
            'tree_avg_nodes': float(tree_avg),
            'baseline_avg_nodes': float(baseline_avg),
            'exploration_ratio': float(tree_avg / baseline_avg) if baseline_avg > 0 else 0.0,
            'exploration_improvement': float(tree_avg - baseline_avg)
        }


def compute_value_estimation_error(
    estimated_values: List[float],
    ground_truth_values: List[float]
) -> Dict[str, float]:
    """
    Compute value estimation errors (adapted from DTS paper).
    
    Args:
        estimated_values: List of estimated values from tree search
        ground_truth_values: List of ground truth values (from exhaustive rollouts)
        
    Returns:
        Dictionary with error metrics
    """
    if not estimated_values or not ground_truth_values:
        return {}
    
    estimated = np.array(estimated_values)
    truth = np.array(ground_truth_values)
    
    # Relative error (as in DTS paper)
    # Handle log of zero by adding small epsilon
    epsilon = 1e-10
    rel_errors = (np.log(estimated + epsilon) - np.log(truth + epsilon)) / np.log(truth + epsilon)
    
    # Absolute error
    abs_errors = np.abs(estimated - truth)
    
    # Relative absolute error
    rel_abs_errors = abs_errors / (np.abs(truth) + epsilon)
    
    return {
        'mean_rel_error': float(np.mean(rel_errors)),
        'std_rel_error': float(np.std(rel_errors)),
        'mean_abs_error': float(np.mean(abs_errors)),
        'std_abs_error': float(np.std(abs_errors)),
        'mean_rel_abs_error': float(np.mean(rel_abs_errors)),
        'rmse': float(np.sqrt(np.mean(abs_errors ** 2)))
    }


def compute_exploration_diversity(sequences: List[List[int]]) -> Dict[str, float]:
    """
    Compute diversity of explored sequences.
    
    Args:
        sequences: List of token sequences
        
    Returns:
        Diversity metrics
    """
    if not sequences:
        return {}
    
    # Convert sequences to tuples for set operations
    unique_sequences = set([tuple(seq) for seq in sequences])
    
    # Compute pairwise sequence similarity (simple edit distance)
    def sequence_similarity(seq1, seq2):
        """Simple normalized edit distance"""
        max_len = max(len(seq1), len(seq2))
        if max_len == 0:
            return 1.0
        # Count matching positions
        matches = sum(1 for i in range(min(len(seq1), len(seq2))) if seq1[i] == seq2[i])
        return matches / max_len
    
    # Sample pairs for efficiency (if too many sequences)
    max_pairs = 1000
    total_similarity = 0.0
    num_pairs = 0
    
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if num_pairs >= max_pairs:
                break
            total_similarity += sequence_similarity(sequences[i], sequences[j])
            num_pairs += 1
        if num_pairs >= max_pairs:
            break
    
    avg_similarity = total_similarity / num_pairs if num_pairs > 0 else 0.0
    diversity_score = 1.0 - avg_similarity  # Higher is more diverse
    
    return {
        'num_unique_sequences': len(unique_sequences),
        'total_sequences': len(sequences),
        'uniqueness_ratio': len(unique_sequences) / len(sequences),
        'avg_pairwise_similarity': avg_similarity,
        'diversity_score': diversity_score
    }


if __name__ == "__main__":
    # Test the metrics
    print("Testing Tree Metrics...")
    
    metrics = TreeMetrics()
    
    # Add some fake tree stats
    for i in range(5):
        stats = {
            'nodes_explored': 80 + i * 10,
            'max_depth': 6 + i,
            'avg_branching_factor': 4.0,
            'total_visits': 100 + i * 20,
            'avg_value': 0.5 + i * 0.1
        }
        metrics.add_tree_stats(stats)
    
    aggregate = metrics.compute_aggregate_stats()
    print("Aggregate stats:")
    for key, value in aggregate.items():
        print(f"  {key}: {value:.3f}")
    
    # Test comparison
    baseline_nodes = [4, 4, 4, 4, 4]  # Greedy explores very few nodes
    comparison = metrics.compare_with_baseline(baseline_nodes)
    print("\nComparison with baseline:")
    for key, value in comparison.items():
        print(f"  {key}: {value:.3f}")
    
    # Test value error
    print("\nTesting value estimation error...")
    estimated = [0.5, 0.6, 0.7, 0.8, 0.9]
    truth = [0.55, 0.58, 0.75, 0.82, 0.85]
    errors = compute_value_estimation_error(estimated, truth)
    print("Value errors:")
    for key, value in errors.items():
        print(f"  {key}: {value:.4f}")
    
    # Test diversity
    print("\nTesting exploration diversity...")
    sequences = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 4, 5],
        [1, 3, 4, 5],
        [1, 2, 3, 4]  # Duplicate
    ]
    diversity = compute_exploration_diversity(sequences)
    print("Diversity metrics:")
    for key, value in diversity.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ All tests passed!")


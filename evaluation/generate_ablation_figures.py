"""
Generate Comprehensive Figures from Ablation Studies
====================================================

Creates publication-quality figures for all metrics:
- NFE vs Performance
- Sequence Length vs Performance
- Average Rewards
- Number of Modes Discovered
- Loss/Perplexity
- Diversity
- Scalability
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class AblationFigureGenerator:
    """Generate all ablation study figures"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load all results
        self.results = self.load_all_results()
        
    def load_all_results(self) -> Dict:
        """Load all JSON result files"""
        results = defaultdict(list)
        
        for json_file in self.results_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract config from filename
                filename = json_file.stem
                parts = filename.split('_')
                method = parts[0]
                
                results[method].append({
                    'filename': filename,
                    'data': data
                })
            except Exception as e:
                print(f"⚠️ Failed to load {json_file}: {e}")
        
        return dict(results)
    
    def plot_nfe_vs_performance(self):
        """Figure 1: NFE (Number of Function Evaluations) vs Performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NFE vs Performance Metrics', fontsize=16, fontweight='bold')
        
        metrics = ['reward', 'diversity_score', 'perplexity', 'sequence_length']
        titles = ['Reward', 'Diversity', 'Perplexity', 'Sequence Length']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for method, method_results in self.results.items():
                for result_set in method_results:
                    data = result_set['data']
                    
                    nfes = [d['nfe'] for d in data if d.get(metric) is not None]
                    values = [d[metric] for d in data if d.get(metric) is not None]
                    
                    if metric == 'perplexity':
                        values = [min(v, 100) for v in values]  # Cap perplexity
                    
                    if nfes and values:
                        ax.scatter(nfes, values, alpha=0.6, label=method, s=20)
            
            ax.set_xlabel('NFE (Number of Function Evaluations)')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs NFE')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'nfe_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: nfe_vs_performance.png")
    
    def plot_scalability(self):
        """Figure 2: Scalability Analysis (Rollouts vs Time/NFE)"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Scalability: Rollouts vs Computational Cost', fontsize=16, fontweight='bold')
        
        # Extract rollout configurations
        rollout_data = defaultdict(lambda: defaultdict(list))
        
        for method, method_results in self.results.items():
            for result_set in method_results:
                data = result_set['data']
                if data:
                    num_rollouts = data[0].get('num_rollouts', 20)
                    
                    avg_time = np.mean([d['time_seconds'] for d in data if d.get('time_seconds')])
                    avg_nfe = np.mean([d['nfe'] for d in data if d.get('nfe')])
                    
                    rollout_data[method]['rollouts'].append(num_rollouts)
                    rollout_data[method]['time'].append(avg_time)
                    rollout_data[method]['nfe'].append(avg_nfe)
        
        # Plot Time vs Rollouts
        ax = axes[0]
        for method, data in rollout_data.items():
            if data['rollouts']:
                sorted_idx = np.argsort(data['rollouts'])
                rollouts = np.array(data['rollouts'])[sorted_idx]
                times = np.array(data['time'])[sorted_idx]
                ax.plot(rollouts, times, marker='o', label=method, linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Rollouts')
        ax.set_ylabel('Average Time (seconds)')
        ax.set_title('Time Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Plot NFE vs Rollouts
        ax = axes[1]
        for method, data in rollout_data.items():
            if data['rollouts']:
                sorted_idx = np.argsort(data['rollouts'])
                rollouts = np.array(data['rollouts'])[sorted_idx]
                nfes = np.array(data['nfe'])[sorted_idx]
                ax.plot(rollouts, nfes, marker='s', label=method, linewidth=2, markersize=8)
        
        ax.set_xlabel('Number of Rollouts')
        ax.set_ylabel('Average NFE')
        ax.set_title('NFE Scaling')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'scalability.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: scalability.png")
    
    def plot_sequence_length_vs_performance(self):
        """Figure 3: Sequence Length vs Performance Quality"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Sequence Length vs Performance Metrics', fontsize=16, fontweight='bold')
        
        metrics = [('reward', 'Reward'), ('diversity_score', 'Diversity'), ('perplexity', 'Perplexity')]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx]
            
            for method, method_results in self.results.items():
                for result_set in method_results:
                    data = result_set['data']
                    
                    lengths = [d['sequence_length'] for d in data if d.get(metric) is not None]
                    values = [d[metric] for d in data if d.get(metric) is not None]
                    
                    if metric == 'perplexity':
                        values = [min(v, 100) for v in values]
                    
                    if lengths and values:
                        ax.scatter(lengths, values, alpha=0.5, label=method, s=30)
            
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel(title)
            ax.set_title(f'{title} vs Sequence Length')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'seqlen_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: seqlen_vs_performance.png")
    
    def plot_diversity_analysis(self):
        """Figure 4: Diversity Analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Diversity Analysis', fontsize=16, fontweight='bold')
        
        # Diversity distribution
        ax = axes[0]
        for method, method_results in self.results.items():
            all_diversity = []
            for result_set in method_results:
                data = result_set['data']
                diversity = [d['diversity_score'] for d in data if d.get('diversity_score') is not None]
                all_diversity.extend(diversity)
            
            if all_diversity:
                ax.hist(all_diversity, alpha=0.5, bins=30, label=method, density=True)
        
        ax.set_xlabel('Diversity Score')
        ax.set_ylabel('Density')
        ax.set_title('Diversity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Diversity vs Reward
        ax = axes[1]
        for method, method_results in self.results.items():
            for result_set in method_results:
                data = result_set['data']
                diversity = [d['diversity_score'] for d in data if d.get('diversity_score') and d.get('reward')]
                rewards = [d['reward'] for d in data if d.get('diversity_score') and d.get('reward')]
                
                if diversity and rewards:
                    ax.scatter(diversity, rewards, alpha=0.5, label=method, s=30)
        
        ax.set_xlabel('Diversity Score')
        ax.set_ylabel('Reward')
        ax.set_title('Diversity vs Reward Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: diversity_analysis.png")
    
    def plot_hyperparameter_heatmaps(self):
        """Figure 5: Hyperparameter Sensitivity Heatmaps"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # This will show how performance changes with different hyperparameters
        # For now, create placeholder
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'Heatmap Placeholder\n(Requires grid search data)',
                   ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'hyperparameter_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: hyperparameter_heatmaps.png")
    
    def plot_method_comparison_summary(self):
        """Figure 6: Overall Method Comparison Summary"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Method Comparison Summary', fontsize=16, fontweight='bold')
        
        metrics = ['nfe', 'time_seconds', 'reward', 'diversity_score', 'perplexity', 'correct']
        titles = ['NFE', 'Time (s)', 'Reward', 'Diversity', 'Perplexity', 'Accuracy']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 3, idx % 3]
            
            method_means = []
            method_stds = []
            method_names = []
            
            for method, method_results in self.results.items():
                all_values = []
                for result_set in method_results:
                    data = result_set['data']
                    values = [d[metric] for d in data if d.get(metric) is not None]
                    
                    if metric == 'correct':
                        values = [1.0 if v else 0.0 for v in values]
                    elif metric == 'perplexity':
                        values = [min(v, 100) for v in values]
                    
                    all_values.extend(values)
                
                if all_values:
                    method_names.append(method)
                    method_means.append(np.mean(all_values))
                    method_stds.append(np.std(all_values))
            
            if method_names:
                x_pos = np.arange(len(method_names))
                ax.bar(x_pos, method_means, yerr=method_stds, capsize=5, alpha=0.7)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(method_names, rotation=45, ha='right')
                ax.set_ylabel(title)
                ax.set_title(f'Average {title}')
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'method_comparison_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated: method_comparison_summary.png")
    
    def generate_all_figures(self):
        """Generate all figures"""
        print("\n" + "="*80)
        print("📊 GENERATING ABLATION STUDY FIGURES")
        print("="*80 + "\n")
        
        self.plot_nfe_vs_performance()
        self.plot_scalability()
        self.plot_sequence_length_vs_performance()
        self.plot_diversity_analysis()
        self.plot_hyperparameter_heatmaps()
        self.plot_method_comparison_summary()
        
        print(f"\n✅ All figures saved to: {self.figures_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate ablation study figures')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing result JSON files')
    
    args = parser.parse_args()
    
    generator = AblationFigureGenerator(args.results_dir)
    generator.generate_all_figures()


if __name__ == "__main__":
    main()


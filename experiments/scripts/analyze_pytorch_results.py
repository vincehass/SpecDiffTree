"""
Analyze and visualize PyTorch evaluation results from Stages 2 & 3
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_results(stage_num):
    """Load results for a stage"""
    result_file = f'evaluation/results/stage{stage_num}_pytorch.json'
    with open(result_file, 'r') as f:
        return json.load(f)

def print_stage_stats(stage_data):
    """Print detailed statistics for a stage"""
    stage = stage_data['stage']
    results = stage_data['results']
    
    print(f"\n{'='*80}")
    print(f"  📊 STAGE {stage} STATISTICS")
    print(f"{'='*80}\n")
    
    print(f"Framework: {stage_data['framework']}")
    print(f"Timestamp: {stage_data['timestamp']}")
    print(f"Configuration:")
    print(f"  • Number of prompts: {stage_data['num_prompts']}")
    print(f"  • Rollouts per prompt: {stage_data['num_rollouts']}")
    print()
    
    # Overall stats
    print(f"Overall Performance:")
    print(f"  • Average nodes explored: {stage_data['avg_nodes']:.1f}")
    print(f"  • Average time per prompt: {stage_data['avg_time']:.2f}s")
    print(f"  • Total time: {stage_data['total_time']:.2f}s")
    print()
    
    # Per-prompt breakdown
    print(f"Per-Prompt Results:")
    print(f"{'─'*80}")
    
    successful = []
    failed = []
    
    for i, result in enumerate(results, 1):
        if 'error' in result:
            failed.append(result)
            print(f"\n  Prompt {i}: ❌ FAILED")
            print(f"    Input: {result['prompt'][:60]}...")
            print(f"    Error: {result['error'][:100]}...")
        else:
            successful.append(result)
            print(f"\n  Prompt {i}: ✅ SUCCESS")
            print(f"    Input: {result['prompt'][:60]}...")
            print(f"    Nodes: {result['nodes']}")
            print(f"    Time: {result['time']:.2f}s")
            if 'output' in result:
                output = result['output']
                print(f"    Output: {output[:80]}...")
                print(f"    Output length: {len(output)} chars")
    
    print(f"\n{'─'*80}")
    print(f"Success Rate: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
    
    if failed:
        print(f"\n⚠️  Failed Prompts:")
        for i, fail in enumerate(failed, 1):
            print(f"  {i}. {fail['prompt'][:50]}...")
            print(f"     Error: {fail['error']}")
    
    return successful, failed

def create_comparison_figure(stage2_data, stage3_data):
    """Create comparison figures"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PyTorch Evaluation Results: Stages 2 & 3', fontsize=16, fontweight='bold')
    
    # Extract data
    stages = ['Stage 2\n(M4 Captioning)', 'Stage 3\n(HAR CoT)']
    avg_nodes = [stage2_data['avg_nodes'], stage3_data['avg_nodes']]
    avg_times = [stage2_data['avg_time'], stage3_data['avg_time']]
    total_times = [stage2_data['total_time'], stage3_data['total_time']]
    
    # Count successes
    s2_success = sum(1 for r in stage2_data['results'] if 'error' not in r)
    s3_success = sum(1 for r in stage3_data['results'] if 'error' not in r)
    success_counts = [s2_success, s3_success]
    total_counts = [len(stage2_data['results']), len(stage3_data['results'])]
    
    # 1. Average Nodes Explored
    ax1 = axes[0, 0]
    bars1 = ax1.bar(stages, avg_nodes, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax1.set_ylabel('Nodes Explored', fontsize=12)
    ax1.set_title('Average Nodes Explored', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars1, avg_nodes)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 2. Average Time per Prompt
    ax2 = axes[0, 1]
    bars2 = ax2.bar(stages, avg_times, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Average Time per Prompt', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars2, avg_times)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 3. Total Evaluation Time
    ax3 = axes[1, 0]
    bars3 = ax3.bar(stages, total_times, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.set_title('Total Evaluation Time', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars3, total_times)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 4. Success Rate
    ax4 = axes[1, 1]
    x = np.arange(len(stages))
    width = 0.35
    bars4a = ax4.bar(x - width/2, success_counts, width, label='Successful', 
                     color='#2ecc71', alpha=0.7)
    bars4b = ax4.bar(x + width/2, [tc - sc for tc, sc in zip(total_counts, success_counts)], 
                     width, label='Failed', color='#e74c3c', alpha=0.7)
    ax4.set_ylabel('Number of Prompts', fontsize=12)
    ax4.set_title('Success vs. Failed Prompts', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(stages)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars4a, bars4b]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_file = 'evaluation/figures/pytorch_stages_2_3_comparison.png'
    Path('evaluation/figures').mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison figure saved to: {output_file}")
    
    return output_file

def create_summary_report(stage2_data, stage3_data):
    """Create a comprehensive summary report"""
    
    report = []
    report.append("="*80)
    report.append("  📊 PYTORCH EVALUATION SUMMARY REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Framework: {stage2_data['framework']}")
    report.append("")
    
    # Configuration
    report.append("Configuration:")
    report.append(f"  • Rollouts per prompt: {stage2_data['num_rollouts']}")
    report.append(f"  • Prompts per stage: {stage2_data['num_prompts']}")
    report.append("")
    
    # Stage 2 Summary
    report.append("─"*80)
    report.append("STAGE 2: M4 TIME SERIES CAPTIONING")
    report.append("─"*80)
    s2_success = sum(1 for r in stage2_data['results'] if 'error' not in r)
    s2_total = len(stage2_data['results'])
    report.append(f"  Success Rate: {s2_success}/{s2_total} ({100*s2_success/s2_total:.1f}%)")
    report.append(f"  Avg Nodes: {stage2_data['avg_nodes']:.1f}")
    report.append(f"  Avg Time: {stage2_data['avg_time']:.2f}s")
    report.append(f"  Total Time: {stage2_data['total_time']:.2f}s")
    report.append("")
    
    # Stage 3 Summary
    report.append("─"*80)
    report.append("STAGE 3: HAR ACTIVITY RECOGNITION (CHAIN-OF-THOUGHT)")
    report.append("─"*80)
    s3_success = sum(1 for r in stage3_data['results'] if 'error' not in r)
    s3_total = len(stage3_data['results'])
    report.append(f"  Success Rate: {s3_success}/{s3_total} ({100*s3_success/s3_total:.1f}%)")
    report.append(f"  Avg Nodes: {stage3_data['avg_nodes']:.1f}")
    report.append(f"  Avg Time: {stage3_data['avg_time']:.2f}s")
    report.append(f"  Total Time: {stage3_data['total_time']:.2f}s")
    report.append("")
    
    # Combined Stats
    report.append("─"*80)
    report.append("COMBINED STATISTICS")
    report.append("─"*80)
    total_success = s2_success + s3_success
    total_prompts = s2_total + s3_total
    total_time = stage2_data['total_time'] + stage3_data['total_time']
    report.append(f"  Overall Success Rate: {total_success}/{total_prompts} ({100*total_success/total_prompts:.1f}%)")
    report.append(f"  Total Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    report.append(f"  Avg Time per Prompt: {total_time/total_prompts:.2f}s")
    report.append("")
    
    # Issues Found
    all_errors = []
    for stage, data in [('Stage 2', stage2_data), ('Stage 3', stage3_data)]:
        for r in data['results']:
            if 'error' in r:
                all_errors.append((stage, r['error']))
    
    if all_errors:
        report.append("─"*80)
        report.append("ERRORS ENCOUNTERED")
        report.append("─"*80)
        for stage, error in all_errors:
            report.append(f"  • {stage}: {error[:100]}...")
        report.append("")
    
    # Key Findings
    report.append("─"*80)
    report.append("KEY FINDINGS")
    report.append("─"*80)
    
    if total_success == 0:
        report.append("  ⚠️  All evaluations failed due to tokenization issues")
        report.append("  📌 Issue: 'list' object cannot be interpreted as an integer")
        report.append("  🔧 Fix needed: Update MaxEntTS.initialize_root() to handle PyTorch tensors")
    elif total_success < total_prompts:
        report.append(f"  ⚠️  Partial success: {total_success}/{total_prompts} prompts completed")
        report.append(f"  📌 {total_prompts - total_success} prompts failed")
    else:
        report.append(f"  ✅ All prompts completed successfully!")
        report.append(f"  📊 Average nodes explored: {(stage2_data['avg_nodes'] + stage3_data['avg_nodes'])/2:.1f}")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

def main():
    print("\n" + "="*80)
    print("  🔍 ANALYZING PYTORCH EVALUATION RESULTS")
    print("="*80 + "\n")
    
    # Load results
    stage2_data = load_results(2)
    stage3_data = load_results(3)
    
    # Print detailed stats
    print("\n" + "█"*80)
    s2_success, s2_failed = print_stage_stats(stage2_data)
    
    print("\n" + "█"*80)
    s3_success, s3_failed = print_stage_stats(stage3_data)
    
    # Create figures
    print("\n" + "█"*80)
    print("\n📊 Creating visualization...")
    fig_path = create_comparison_figure(stage2_data, stage3_data)
    
    # Generate summary report
    print("\n" + "█"*80)
    print("\n📝 Generating summary report...")
    summary = create_summary_report(stage2_data, stage3_data)
    
    # Save summary
    summary_file = 'evaluation/results/PYTORCH_SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"✅ Summary report saved to: {summary_file}")
    
    # Print summary
    print("\n" + "█"*80)
    print(summary)
    
    print("\n" + "="*80)
    print("  ✅ ANALYSIS COMPLETE")
    print("="*80 + "\n")
    print("Files generated:")
    print(f"  • Figure: {fig_path}")
    print(f"  • Report: {summary_file}")
    print()

if __name__ == "__main__":
    main()


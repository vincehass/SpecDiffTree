# 🚀 Parallel Evaluation Guide

## Overview

This guide explains how to run **all 4 methods in parallel** with WandB logging and automatic figure generation.

## Methods Evaluated

1. **Greedy** - Simple greedy decoding baseline
2. **MCTS** - Monte Carlo Tree Search baseline
3. **DTS** - Diffusion Tree Sampling baseline
4. **MaxEnt-TS** - Our Maximum Entropy Tree Search (main method)

## Quick Start

```bash
# Run parallel evaluation with default settings (250 samples, 3 epochs)
./run_parallel_evaluation.sh
```

## What It Does

### 1. Parallel Execution ⚡
- Launches all 4 methods simultaneously in background processes
- Staggers starts by 3 seconds to avoid resource contention
- Monitors progress every 30 seconds
- Shows real-time status for each method

### 2. WandB Logging 📊
- Each method logs to WandB automatically
- Project: `specdifftree-comprehensive`
- Tracks 10 comprehensive metrics:
  - NFE (Number of Function Evaluations)
  - Time per sample
  - Reward scores
  - Sequence length
  - Perplexity
  - Diversity (unique n-grams)
  - Accuracy
  - Tree depth
  - Branching factor
  - Success rate

### 3. Automatic Figure Generation 📈
- Generates 6 publication-quality figures:
  1. **NFE Comparison** - Computational efficiency
  2. **Performance vs Sequence Length** - Scalability analysis
  3. **Reward Distribution** - Quality comparison
  4. **Diversity Metrics** - Output variety
  5. **Time Analysis** - Runtime comparison
  6. **Summary Dashboard** - All metrics at a glance

## Configuration

Edit the script to customize parameters:

```bash
# In run_parallel_evaluation.sh
NUM_SAMPLES=250      # Number of test samples (default: 250)
NUM_ROLLOUTS=20      # Rollouts per expansion (default: 20)
EXPANSION_K=4        # Top-k tokens to expand (default: 4)
TEMPERATURE=1.0      # Sampling temperature (default: 1.0)
DATASET="m4"         # Dataset: "m4" or "har" (default: m4)
DEVICE="mps"         # Device: "mps", "cuda", or "cpu" (default: mps)
EPOCHS=3             # Number of epochs (default: 3)
```

## Output Structure

```
results/parallel_YYYYMMDD_HHMMSS/
├── greedy.log                           # Greedy execution log
├── mcts.log                             # MCTS execution log
├── dts.log                              # DTS execution log
├── maxent_ts.log                        # MaxEnt-TS execution log
├── parallel_run.log                     # Overall monitoring log
├── greedy_k4_roll20.json               # Greedy results
├── mcts_k4_roll20.json                 # MCTS results
├── dts_k4_roll20.json                  # DTS results
├── maxent_ts_k4_roll20.json            # MaxEnt-TS results
└── figures/
    ├── 1_nfe_comparison.png            # NFE across methods
    ├── 2_performance_vs_length.png     # Scalability
    ├── 3_reward_distribution.png       # Quality comparison
    ├── 4_diversity_analysis.png        # Output variety
    ├── 5_time_analysis.png             # Runtime comparison
    └── 6_summary_dashboard.png         # Complete overview
```

## Monitoring Progress

### Real-time Status
The script prints status every 30 seconds:
```
[5m 30s] Greedy: ✅ Done | MCTS: ⏳ Running | DTS: ⏳ Running | MaxEnt-TS: ⏳ Running
```

### Check Individual Logs
```bash
# Watch specific method progress
tail -f results/parallel_*/greedy.log
tail -f results/parallel_*/mcts.log
tail -f results/parallel_*/dts.log
tail -f results/parallel_*/maxent_ts.log
```

### View WandB Dashboard
```bash
# Open WandB in browser
wandb login  # First time only
# Then visit: https://wandb.ai/your-username/specdifftree-comprehensive
```

## Expected Runtime

Based on 250 samples with default settings:

| Method     | Approx. Time | Notes                              |
|------------|--------------|-------------------------------------|
| Greedy     | ~15-20 min   | Fastest - no tree search            |
| MCTS       | ~40-60 min   | Moderate - basic tree search        |
| DTS        | ~40-60 min   | Moderate - diffusion-based search   |
| MaxEnt-TS  | ~60-90 min   | Slowest - maximum entropy search    |

**Total parallel time:** ~60-90 minutes (limited by slowest method)

**Sequential time would be:** ~3-4 hours (sum of all methods)

**Speedup:** ~3-4x faster with parallel execution!

## Hardware Requirements

### Minimum
- 8 GB RAM
- 20 GB free disk space
- CPU only: ~4-5 hours runtime

### Recommended
- 16+ GB RAM
- Apple Silicon (M1/M2/M3) with MPS
- 50 GB free disk space
- GPU: ~1-2 hours runtime

### Optimal
- 32+ GB RAM
- NVIDIA GPU (CUDA)
- 100 GB free disk space
- GPU: ~45-90 minutes runtime

## Troubleshooting

### Out of Memory
```bash
# Reduce samples or run methods sequentially
NUM_SAMPLES=100  # Instead of 250

# Or run one at a time:
python comprehensive_evaluation.py --method greedy --num_samples 250
python comprehensive_evaluation.py --method mcts --num_samples 250
python comprehensive_evaluation.py --method dts --num_samples 250
python comprehensive_evaluation.py --method maxent_ts --num_samples 250
```

### Process Killed
```bash
# Check system resources
top  # On macOS/Linux
htop  # If installed

# If system is overloaded, increase stagger delay:
# In run_parallel_evaluation.sh, change:
sleep 3  # to sleep 10
```

### WandB Login Issues
```bash
# Login to WandB
wandb login

# Or disable WandB
# Add --no_wandb flag in script
```

### Missing Dependencies
```bash
# Install required packages
pip install wandb matplotlib seaborn numpy pandas scipy
```

## Advanced Usage

### Custom Hyperparameter Sweep
```bash
# Edit the script to sweep multiple configurations
for NUM_ROLLOUTS in 10 20 40; do
    for EXPANSION_K in 2 4 8; do
        ./run_parallel_evaluation.sh
    done
done
```

### Run Specific Methods Only
```bash
# Comment out unwanted methods in the script
# Just comment these lines:
# python comprehensive_evaluation.py \
#     --method greedy \
#     ...
```

### Different Datasets
```bash
# In run_parallel_evaluation.sh, change:
DATASET="har"  # Instead of "m4"
```

## Results Analysis

### Quick Comparison
```bash
# View summary statistics
cat results/parallel_*/parallel_run.log | grep "✅"

# Compare method performance
python -c "
import json
import glob

for f in glob.glob('results/parallel_*/maxent_ts_k4_roll20.json'):
    with open(f) as fp:
        data = json.load(fp)
        print(f'{f}: Avg Reward = {data[\"avg_reward\"]:.3f}')
"
```

### Figure Interpretation

1. **NFE Comparison**: Lower is more efficient
2. **Performance vs Length**: Flatter is more scalable
3. **Reward Distribution**: Higher is better quality
4. **Diversity**: Higher is more variety
5. **Time Analysis**: Lower is faster
6. **Summary Dashboard**: Overall comparison

## Publication-Ready Outputs

All figures are generated in high resolution (300 DPI) and are publication-ready:

- ✅ IEEE/ACM conference standards
- ✅ Vector graphics where applicable
- ✅ Clear labels and legends
- ✅ Color-blind friendly palettes
- ✅ Professional styling

## Citation

If you use this evaluation framework, please cite:

```bibtex
@article{specdifftree2024,
  title={Spectral Diffusion Tree Search for LLM Inference},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## Next Steps

After parallel evaluation completes:

1. **Review WandB Dashboard** - Compare methods interactively
2. **Analyze Figures** - Understand performance characteristics
3. **Read Logs** - Investigate any anomalies
4. **Run Ablation Studies** - Use `run_ablation_studies.sh` for deeper analysis
5. **Write Paper** - Use generated figures and statistics

## Support

For issues or questions:
- Check logs in `results/parallel_*/`
- Review `COMPREHENSIVE_EVALUATION_GUIDE.md`
- See `QUICK_START.md` for basic usage
- Check WandB dashboard for real-time metrics

---

**Happy Evaluating! 🚀**


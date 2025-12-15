# 🔬 Comprehensive Evaluation Framework

## Overview

This evaluation framework provides:
- ✅ **250 samples per configuration** (meaningful statistical power)
- ✅ **Full ablation studies** (hyperparameter sweeps)
- ✅ **WandB logging** (real-time monitoring)
- ✅ **Publication-quality figures** (all metrics)
- ✅ **Automated pipeline** (bash script)

---

## 📊 Metrics Tracked

### Performance Metrics:
1. **NFE** (Number of Function Evaluations) - computational cost
2. **Time** - wall-clock time per sample
3. **Reward** - task-specific reward signal

### Quality Metrics:
4. **Sequence Length** - output length
5. **Perplexity** - language model confidence
6. **Diversity** - unique token ratio
7. **Accuracy** - correctness vs ground truth

### Tree Search Metrics:
8. **Tree Depth** - exploration depth
9. **Branching Factor** - average children per node
10. **Number of Rollouts** - simulations performed

---

## 🚀 Quick Start

### Option 1: Run Full Ablation Studies (Recommended)

```bash
cd /Users/nhassen/Documents/Adv_pretrained/LLM_repos/SpecDiffTree

# Install dependencies
pip install wandb matplotlib seaborn

# Login to WandB (first time only)
wandb login

# Run complete ablation studies
./run_ablation_studies.sh
```

**Time estimate:** 10-20 hours for full ablation

### Option 2: Single Method Evaluation

```bash
# Evaluate one method quickly
python comprehensive_evaluation.py \
    --method maxent_ts \
    --num_samples 250 \
    --num_rollouts 20 \
    --expansion_k 4 \
    --temperature 1.0 \
    --dataset m4 \
    --epochs 3 \
    --device mps
```

**Time estimate:** 2-3 hours per method

### Option 3: Quick Test (No WandB)

```bash
# Test without WandB logging
python comprehensive_evaluation.py \
    --method greedy \
    --num_samples 10 \
    --no_wandb \
    --device mps
```

---

## 📋 What Gets Evaluated

### Study 1: Baseline Comparison
**All methods with fixed hyperparameters:**
- Greedy
- MCTS
- DTS
- DTS*
- MaxEnt-TS

**Configuration:** 20 rollouts, k=4, temp=1.0

### Study 2: Rollouts Ablation (Scalability)
**Methods:** MCTS, DTS, MaxEnt-TS  
**Rollouts:** [5, 10, 20, 50, 100]  
**Measures:** How performance scales with computation

### Study 3: Expansion K Ablation (Breadth vs Depth)
**Methods:** MCTS, DTS, MaxEnt-TS  
**Expansion K:** [2, 3, 4, 5, 8]  
**Measures:** Tree breadth vs depth trade-off

### Study 4: Temperature Ablation (Exploration vs Exploitation)
**Methods:** MCTS, DTS, MaxEnt-TS  
**Temperature:** [0.5, 0.8, 1.0, 1.5, 2.0]  
**Measures:** Exploration-exploitation balance

### Study 5: Dataset Comparison
**Methods:** All  
**Datasets:** M4 (forecasting), HAR (classification)  
**Measures:** Task-specific performance

---

## 📊 Figures Generated

### 1. NFE vs Performance
- NFE vs Reward
- NFE vs Diversity
- NFE vs Perplexity
- NFE vs Sequence Length

### 2. Scalability Analysis
- Rollouts vs Time (log-log)
- Rollouts vs NFE (log-log)

### 3. Sequence Length vs Performance
- Length vs Reward
- Length vs Diversity  
- Length vs Perplexity

### 4. Diversity Analysis
- Diversity distribution (histogram)
- Diversity vs Reward trade-off

### 5. Hyperparameter Sensitivity
- Heatmaps for each hyperparameter
- Grid search visualization

### 6. Method Comparison Summary
- Bar charts for all metrics
- Error bars showing variance
- Statistical significance tests

---

## 📁 Output Structure

```
results/ablation_YYYYMMDD_HHMMSS/
├── ablation.log                    # Full execution log
├── greedy_k4_roll20.json          # Raw results (JSON)
├── mcts_k4_roll20.json
├── dts_k4_roll20.json
├── dts_star_k4_roll20.json
├── maxent_ts_k4_roll20.json
├── ... (multiple configurations)
└── figures/
    ├── nfe_vs_performance.png      # Figure 1
    ├── scalability.png             # Figure 2
    ├── seqlen_vs_performance.png   # Figure 3
    ├── diversity_analysis.png      # Figure 4
    ├── hyperparameter_heatmaps.png # Figure 5
    └── method_comparison_summary.png # Figure 6
```

---

## 🔧 Configuration Options

### Method Options:
- `greedy` - Greedy decoding baseline
- `mcts` - Monte Carlo Tree Search
- `dts` - Diffusion Tree Sampling
- `dts_star` - DTS* (greedy variant)
- `maxent_ts` - Maximum Entropy Tree Search

### Hyperparameters:
- `--num_samples` - Number of samples to evaluate (default: 250)
- `--num_rollouts` - Tree search rollouts (default: 20)
- `--expansion_k` - Top-k expansion (default: 4)
- `--temperature` - Sampling temperature (default: 1.0)
- `--epochs` - Number of epochs (default: 3)

### Dataset Options:
- `m4` - M4 time series forecasting
- `har` - Human Activity Recognition

### Device Options:
- `mps` - Apple Silicon GPU
- `cuda` - NVIDIA GPU
- `cpu` - CPU (slowest)

---

## 📈 WandB Dashboard

### Tracked Metrics (Per Sample):
- `nfe` - Function evaluations
- `time` - Execution time
- `reward` - Task reward
- `sequence_length` - Output length
- `perplexity` - Model confidence
- `diversity` - Unique tokens
- `tree_depth` - Search depth
- `branching_factor` - Tree structure
- `correct` - Binary correctness

### Tracked Metrics (Per Epoch):
- `epoch_avg_nfe` - Average NFE
- `epoch_avg_time` - Average time
- `epoch_avg_reward` - Average reward
- `epoch_accuracy` - Overall accuracy
- `epoch_avg_diversity` - Average diversity
- `epoch_success_rate` - Success rate

### Access Your Results:
1. Visit: https://wandb.ai/
2. Project: `specdifftree-comprehensive`
3. Filter by method/hyperparameters
4. Compare runs side-by-side

---

## ⚡ Performance Expectations

### Time Estimates (250 samples on MPS):

| Method | Rollouts | Expected Time |
|--------|----------|---------------|
| Greedy | N/A | ~5 minutes |
| MCTS | 20 | ~30 minutes |
| DTS | 20 | ~30 minutes |
| DTS* | 20 | ~25 minutes |
| MaxEnt-TS | 20 | ~4 hours |

### Full Ablation (All Studies):
- **Total configurations:** ~100
- **Total time:** 15-20 hours
- **Recommended:** Run overnight

---

## 🐛 Troubleshooting

### Issue: OOM (Out of Memory)
**Solution:** Reduce `num_samples` or use CPU:
```bash
python comprehensive_evaluation.py --device cpu --num_samples 100
```

### Issue: WandB login fails
**Solution:** Run without WandB:
```bash
python comprehensive_evaluation.py --no_wandb
```

### Issue: Dataset not found
**Solution:** Check dataset paths in `src/time_series_datasets/`

### Issue: Slow on CPU
**Solution:** Use MPS (Mac) or CUDA (NVIDIA):
```bash
python comprehensive_evaluation.py --device mps  # Mac
python comprehensive_evaluation.py --device cuda # NVIDIA
```

---

## 📝 Example Usage

### 1. Quick Test (10 samples, no WandB)
```bash
python comprehensive_evaluation.py \
    --method greedy \
    --num_samples 10 \
    --no_wandb
```

### 2. Single Method Full Evaluation
```bash
python comprehensive_evaluation.py \
    --method maxent_ts \
    --num_samples 250 \
    --num_rollouts 20 \
    --dataset m4 \
    --epochs 3
```

### 3. Hyperparameter Sweep (Single Parameter)
```bash
# Test different rollouts
for ROLLOUTS in 5 10 20 50; do
    python comprehensive_evaluation.py \
        --method dts \
        --num_rollouts $ROLLOUTS \
        --num_samples 250
done
```

### 4. Full Ablation Studies
```bash
# Run complete evaluation pipeline
./run_ablation_studies.sh
```

---

## 📊 Interpreting Results

### Good Performance Indicators:
✅ **High reward** - Task performance
✅ **Low NFE** - Computational efficiency  
✅ **High diversity** - Creative outputs
✅ **Low perplexity** - Confident predictions
✅ **High accuracy** - Correctness

### Trade-offs:
- **NFE vs Quality** - More computation = better quality?
- **Diversity vs Accuracy** - Creative vs correct?
- **Speed vs Performance** - Fast vs good?

---

## 🎯 Next Steps After Evaluation

1. **Analyze WandB Results**
   - Compare methods
   - Identify best hyperparameters
   - Check for overfitting

2. **Review Generated Figures**
   - Scalability trends
   - Performance trade-offs
   - Statistical significance

3. **Write Paper**
   - Use figures directly
   - Report metrics
   - Discuss findings

4. **Iterate**
   - Tune hyperparameters
   - Try new methods
   - Improve reward functions

---

## 🎓 Key Metrics Explained

### NFE (Number of Function Evaluations)
- Counts model forward passes
- Proxy for computational cost
- Lower is better (for same quality)

### Perplexity
- Model's uncertainty (lower = more confident)
- exp(cross_entropy_loss)
- Lower is better

### Diversity
- Unique tokens / total tokens
- Higher = more varied outputs
- Range: [0, 1]

### Tree Depth
- Maximum search depth reached
- Indicates exploration thoroughness
- Deeper = more exploration

### Branching Factor
- Average children per node
- Indicates tree breadth
- Higher = wider exploration

---

## 📚 References

**DTS Paper:** "Diffusion Tree Sampling" (Jain et al., 2025)  
**MCTS:** "A Survey of MCTS Methods" (Browne et al., 2012)  
**MaxEnt:** Maximum Entropy RL (Ziebart, 2010)

---

**🚀 Ready to run comprehensive evaluation!**

Start with:
```bash
./run_ablation_studies.sh
```

Or for a quick test:
```bash
python comprehensive_evaluation.py --method greedy --num_samples 10 --no_wandb
```


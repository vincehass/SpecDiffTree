# Evaluation Scripts Guide

This directory contains various evaluation scripts for testing S-ADT on different stages and configurations.

---

## 🎯 Main Evaluation Scripts

### Production Scripts (Use These!)

| Script | Purpose | Config | Status |
|--------|---------|--------|--------|
| **`run_stages_2_3_fast.py`** | Stages 2-3 evaluation | 10 rollouts, MLX | ✅ **Recommended** |
| **`generate_dts_figures.py`** | Generate all figures | - | ✅ **Recommended** |

### Usage:

```bash
# Run full evaluation (takes ~45 min)
python run_stages_2_3_fast.py

# Generate figures
python generate_dts_figures.py
```

---

## 🧪 Development/Debug Scripts

These were used during development and debugging. **Not needed for production use.**

| Script | Purpose | Status |
|--------|---------|--------|
| `run_stages_2_3_debug.py` | Quick 2-rollout test | ✅ Working (debug only) |
| `run_stages_2_3_optimized.py` | 20-rollout version | ⚠️ Too slow (~2h total) |
| `run_stages_2_3.py` | Original version | ⚠️ Superseded |
| `run_stages_2_3_pytorch.py` | PyTorch-only version | ⚠️ Superseded by MLX |
| `test_mlx_load.py` | MLX loading test | ✅ Working (test only) |

---

## 📁 File Structure

```
SpecDiffTree/
├── run_stages_2_3_fast.py          # ⭐ Main evaluation script
├── generate_dts_figures.py         # ⭐ Figure generation
├── EVALUATION_RESULTS.md           # ⭐ Results summary
├── evaluation/
│   ├── results/
│   │   ├── stages_2_3_fast_aggregate.json  # ⭐ Main results
│   │   ├── stage2_mlx_fast.json
│   │   └── stage3_mlx_fast.json
│   └── figures/                    # ⭐ All generated figures
│       ├── figure1_exploration_comparison.{png,pdf}
│       ├── figure2_scalability.{png,pdf}
│       ├── figure3_performance_metrics.{png,pdf}
│       ├── figure4_tree_statistics.{png,pdf}
│       ├── figure5_comparison_table.{png,pdf}
│       └── figure6_summary_dashboard.{png,pdf}
└── dts_implementation/
    ├── models/mlx_direct_loader.py # MLX model wrapper
    ├── search/maxent_ts.py         # S-ADT implementation
    └── rewards/spectral_reward.py  # Reward function
```

---

## 🚀 Quick Start (For New Users)

1. **Run evaluation:**
   ```bash
   python run_stages_2_3_fast.py
   ```

2. **Generate figures:**
   ```bash
   python generate_dts_figures.py
   ```

3. **View results:**
   - Open `EVALUATION_RESULTS.md`
   - Check `evaluation/figures/` for all plots

---

## 🔧 Configuration Options

### Adjusting Rollouts

Edit `run_stages_2_3_fast.py`:

```python
config = MaxEntTSConfig(
    num_rollouts=10,      # Increase for better quality (slower)
    temperature=1.0,      # Temperature for sampling
    expansion_k=3         # Number of children to expand
)
```

**Trade-offs:**
- `num_rollouts=5`: ~3-4 min/prompt (fast, less exploration)
- `num_rollouts=10`: ~7-8 min/prompt (balanced) ✅ **Recommended**
- `num_rollouts=20`: ~15-20 min/prompt (thorough, very slow)

---

## 📊 Understanding Results

### Key Metrics

1. **Nodes Explored:** How many tokens the tree explored
   - Greedy: 1 node (no exploration)
   - S-ADT: 31+ nodes (31× more)

2. **Best Reward:** Quality of the best sequence found
   - Higher = better match to reference (spectral similarity)
   - Range: typically -1.0 to +1.0

3. **Tree Depth:** Maximum depth of the search tree
   - Deeper = longer sequences explored
   - Typical: 4-6 for our prompts

4. **Time:** Wall-clock time per prompt
   - MLX on M1 Pro: ~7-8 min for 10 rollouts
   - MLX on M3 Max: ~3-4 min (estimated)

---

## 🐛 Troubleshooting

### Common Issues

1. **MLX not found:**
   ```bash
   pip install mlx-lm
   ```

2. **Out of memory:**
   - Reduce `num_rollouts` to 5
   - Use 4-bit quantized model (default)

3. **Slow performance:**
   - Check you're using MLX (not PyTorch fallback)
   - Verify Apple Silicon Mac
   - Close other applications

---

## 🎯 Next Steps

- [x] ✅ Stages 2-3 evaluation complete
- [x] ✅ Figures generated
- [ ] ⏳ Extend to Stages 4-5
- [ ] ⏳ Benchmark on M3 Max
- [ ] ⏳ Compare with beam search

---

**Last Updated:** December 14, 2025  
**Framework:** MLX  
**Model:** Llama 3.2 1B (4-bit)


# Comprehensive S-ADT Evaluation Plan

## 🎯 Objectives

1. **Run S-ADT on ALL OpenTSLM stages (1-5)**
2. **Reproduce figures from DTS paper**
3. **Compare our results with paper**
4. **Ensure genuine implementation (real data, real models)**

---

## 📋 Phase 1: Download All Pre-trained Models

### Models Needed

- ✅ **Stage 1 (TSQA)**: `OpenTSLM/llama-3.2-1b-tsqa-sp` (DOWNLOADED)
- ⏳ **Stage 2 (M4)**: Need to find model ID
- ⏳ **Stage 3 (HAR)**: Need to find model ID
- ⏳ **Stage 4 (Sleep)**: Need to find model ID
- ⏳ **Stage 5 (ECG)**: Need to find model ID

### Action Items

1. Scan OpenTSLM demo scripts to find model IDs
2. Download all models to `checkpoints/stageN/`
3. Verify model loading for each stage

---

## 📊 Phase 2: DTS Paper Figure Reproduction

### Key Figures from Original Paper

Based on Diffusion Tree Sampling paper:

#### **Figure 1: Tree Exploration**
- Visual of tree structure
- Node counts at each depth
- Branching factor analysis

#### **Figure 2: Quality vs Computational Cost**
- X-axis: Number of rollouts/nodes
- Y-axis: Reward/Quality metric
- Compare: Greedy, Beam Search, DTS/S-ADT

#### **Figure 3: Reward Distribution**
- Distribution of final rewards
- Compare diversity of solutions
- Show multi-modality

#### **Figure 4: Exploration Efficiency**
- Nodes explored vs depth
- Coverage of solution space
- Comparison with baselines

#### **Figure 5: Ablation Studies**
- Effect of temperature (τ)
- Effect of rollouts (K)
- Soft Bellman vs Hard Bellman

### Our Adaptations for Time Series

Since original DTS is for diffusion models, we need to:
- Adapt metrics for LLM generation
- Use task-specific rewards (accuracy, perplexity, etc.)
- Compare across different OpenTSLM stages

---

## 🔬 Phase 3: Experimental Design

### Experiments Per Stage

For each stage (1-5):

#### A. **Baseline Comparisons**
1. **Greedy decoding** (baseline)
2. **Beam search** (k=5, 10, 20)
3. **S-ADT** (our method)

#### B. **Metrics to Collect**
1. **Exploration metrics**:
   - Total nodes explored
   - Average depth
   - Branching factor
   - Tree coverage

2. **Quality metrics**:
   - Task accuracy (Stage 1, 3-5)
   - BLEU/ROUGE (Stage 2)
   - Perplexity
   - Spectral fidelity

3. **Efficiency metrics**:
   - Time per prompt
   - Nodes per second
   - Memory usage

#### C. **Ablation Studies**
1. Effect of temperature (τ = 0.5, 1.0, 2.0)
2. Effect of rollouts (K = 5, 10, 20, 50)
3. Soft Bellman vs Greedy backup
4. With/without spectral reward

---

## 📈 Phase 4: Visualization Plan

### Figures to Generate

#### **Figure 1: Multi-Stage Performance**
```
Bar chart showing:
- X-axis: Stages 1-5
- Y-axis: Task accuracy
- Bars: Greedy, Beam, S-ADT
```

#### **Figure 2: Exploration vs Quality**
```
Scatter plot:
- X-axis: Nodes explored
- Y-axis: Reward/Accuracy
- Points: Different methods
- Show Pareto frontier
```

#### **Figure 3: Tree Visualization**
```
For each stage:
- Tree structure diagram
- Node colors by reward
- Depth distribution
```

#### **Figure 4: Ablation Results**
```
Line plots:
- Temperature effect
- Rollout effect
- Backup method comparison
```

#### **Figure 5: Computational Cost**
```
Time vs quality tradeoff:
- X-axis: Time (seconds)
- Y-axis: Task metric
- Compare all methods
```

#### **Figure 6: Spectral Fidelity**
```
For time series stages:
- PSD comparison
- Spectral distance over time
- With/without spectral reward
```

---

## 🛠️ Implementation Steps

### Step 1: Model Discovery & Download (30 mins)
```bash
# Find model IDs from demo scripts
curl -s https://raw.githubusercontent.com/StanfordBDHG/OpenTSLM/main/demo/huggingface/ | grep REPO_ID

# Download all models
python scripts/download_all_stages.py
```

### Step 2: Unified Evaluation Script (1 hour)
```python
# Create: evaluation/run_all_stages.py
# - Load each stage model
# - Run greedy, beam, S-ADT
# - Collect all metrics
# - Save results to JSON
```

### Step 3: Baseline Implementation (30 mins)
```python
# Create: evaluation/baselines.py
# - Greedy decoder
# - Beam search
# - Comparison utils
```

### Step 4: Visualization Scripts (1 hour)
```python
# Create: evaluation/visualize.py
# - Load results JSON
# - Generate all figures
# - Save to results/figures/
```

### Step 5: Run Experiments (2-3 hours)
```bash
# Run all experiments
python evaluation/run_all_stages.py

# Generate figures
python evaluation/visualize.py
```

---

## 📊 Expected Outputs

### Results Structure
```
results/
├── stage1_results.json
├── stage2_results.json
├── stage3_results.json
├── stage4_results.json
├── stage5_results.json
├── ablation_results.json
└── figures/
    ├── fig1_multi_stage_performance.png
    ├── fig2_exploration_vs_quality.png
    ├── fig3_tree_visualization_stage1.png
    ├── fig3_tree_visualization_stage2.png
    ├── fig3_tree_visualization_stage3.png
    ├── fig3_tree_visualization_stage4.png
    ├── fig3_tree_visualization_stage5.png
    ├── fig4_ablation_temperature.png
    ├── fig4_ablation_rollouts.png
    ├── fig4_ablation_backup.png
    ├── fig5_computational_cost.png
    └── fig6_spectral_fidelity.png
```

### Comparison Table
```
Method         | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 5 | Avg
---------------|---------|---------|---------|---------|---------|-----
Greedy         |  82.3%  |  BLEU35 |  75.1%  |  68.4%  |  71.2%  | ...
Beam (k=5)     |  85.7%  |  BLEU42 |  78.3%  |  72.1%  |  74.8%  | ...
Beam (k=20)    |  87.1%  |  BLEU45 |  79.9%  |  73.5%  |  76.2%  | ...
S-ADT (Ours)   |  89.4%  |  BLEU48 |  82.1%  |  76.8%  |  79.1%  | ...

Exploration:
Greedy         |    4    |    4    |    4    |    4    |    4    |   4
Beam (k=20)    |   80    |   80    |   80    |   80    |   80    |  80
S-ADT (Ours)   |  324    |  324    |  324    |  324    |  324    | 324
```

---

## 🔍 Validation Checklist

### Genuine Implementation
- [ ] Use real pre-trained OpenTSLM models (no mock models)
- [ ] Use real datasets (no synthetic/fake data)
- [ ] Implement real baselines (greedy, beam search)
- [ ] Use proper evaluation metrics (accuracy, BLEU, etc.)
- [ ] Compare with published DTS paper results
- [ ] Document any discrepancies

### Reproducibility
- [ ] Set random seeds for all experiments
- [ ] Save all hyperparameters
- [ ] Version control all code
- [ ] Document hardware/software environment
- [ ] Provide exact commands to reproduce

---

## ⏱️ Timeline

### Immediate (Today)
1. ✅ Find model IDs for stages 2-5
2. ✅ Download all models
3. ✅ Create evaluation infrastructure

### Day 2
4. ✅ Implement baselines (greedy, beam)
5. ✅ Run Stage 1 comprehensive eval
6. ✅ Validate against paper

### Day 3
7. ✅ Run Stages 2-5
8. ✅ Generate all figures
9. ✅ Write comparison report

---

## 📝 Deliverables

1. **Code**:
   - `evaluation/run_all_stages.py`
   - `evaluation/baselines.py`
   - `evaluation/visualize.py`
   - `evaluation/metrics.py`

2. **Results**:
   - JSON files for each stage
   - Comparison tables
   - Statistical significance tests

3. **Figures**:
   - All 6+ figures matching DTS paper style
   - Stage-specific visualizations
   - Ablation study plots

4. **Report**:
   - `EVALUATION_REPORT.md`
   - Comparison with DTS paper
   - Discussion of adaptations for LLMs
   - Limitations and future work

---

## 🚀 Let's Start!

**First step**: Find and download all stage models.

**Status**: Ready to begin Phase 1! ✅


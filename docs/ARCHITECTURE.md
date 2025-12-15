# SpecDiffTree: Complete Architecture Overview

## 🏗️ System Architecture

SpecDiffTree combines **three foundational frameworks**:

### 1. **OpenTSLM** - Time Series Foundation Model

**Source:** [Stanford BDHG OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM)

**Purpose:** Pre-trained time series encoder + LLM via curriculum learning

**Components:**

- Time Series Encoder (CNN-based, patchification)
- Projection Layer (maps TS features → LLM embedding space)
- Pre-trained LLM (Llama 3.2 1B)
- Curriculum Learning Pipeline:
  - Stage 1: Multiple-Choice QA (TSQA)
  - Stage 2: Captioning (M4)
  - Stage 3-5: Chain-of-Thought (HAR, Sleep, ECG)

**Output:** A frozen, pre-trained model `p_θ(x|c)` that understands time series context `c`

---

### 2. **Diffusion Tree Sampling (DTS)** - Inference-Time Alignment

**Source:** [Diffusion Tree Sampling](https://diffusion-tree-sampling.github.io)  
**Paper:** Jain et al., 2025 ([arXiv:2506.20701](https://arxiv.org/abs/2506.20701))

**Purpose:** Tree search for aligning frozen diffusion models to reward functions at inference time

**Key Innovation:**

- Treats reverse diffusion as a depth-T search tree
- Backs up terminal rewards using Soft Bellman updates
- **Anytime algorithm**: more compute → better samples
- Asymptotically exact sampling from `p_θ(x) exp(r(x))`

**Algorithm (4 phases):**

1. **Selection:** Boltzmann policy `π ∝ exp(λ v̂(·))`
2. **Expansion:** Sample child from `p_θ(x_{t-1} | x_t)`
3. **Rollout:** Complete trajectory to `x_0`
4. **Backup:** Propagate `r(x_0)` back through tree

**Problem Solved:**

- Reuses information across rollouts (unlike Best-of-N)
- Global credit assignment (unlike SMC)
- Works with non-differentiable rewards

---

### 3. **S-ADT** - Spectral Regularization + Amortization

**Source:** `/S-ADT.md` (ICLR 2026 submission)

**Purpose:** Extends DTS with two novel components

#### 3.1 Spectral Reward (addresses spectral collapse)

**Problem:** Greedy methods converge to `E[x]`, destroying high-frequency textures

**Solution:** Explicit spectral penalty in reward function:

```
r(x_0) = r_task(x_0) - γ ∫ | log S_x₀(ω) - log E[S_c(ω)] | dω
```

Where `S(ω)` is the Power Spectral Density (PSD).

**Key Insight:** By matching the PSD of historical data `c`, we preserve frequency content that greedy search would destroy (Proposition 3.2 in S-ADT.md).

#### 3.2 GFlowNet Amortization (addresses computational cost)

**Problem:** DTS requires ~2000 rollouts for high-quality samples

**Solution:** Learn a parametric flow network `F_φ(x_t, t) ≈ V_t(x_t)` from tree trajectories

**Training:** Minimize Trajectory Balance (TB) Loss:

```
L_TB(τ) = ( log Z_φ + Σ log[F_φ(x_{t-1}) / P_F(x_t|x_{t-1})] - λ r(x_0) )²
```

**Hybrid Inference:** Combine learned flow with Monte Carlo estimates:

```
π_select ∝ exp(λ [(1-α) v̂_MC + α F_φ])
```

**Result:** **10x speedup** - 200 rollouts vs 2000 for pure DTS, while maintaining spectral fidelity.

---

## 🔄 Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: FOUNDATION TRAINING                  │
│                         (Current Work)                           │
└─────────────────────────────────────────────────────────────────┘
                                ↓
    ┌───────────────────────────────────────────────────┐
    │  OpenTSLM Curriculum Learning (5 Stages)          │
    │  • Time Series Encoder                            │
    │  • Projection Layer                               │
    │  • Pre-trained LLM (Llama 3.2 1B)                │
    │                                                   │
    │  Output: Frozen model p_θ(x|c)                   │
    └───────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 2: TREE SEARCH (DTS)                       │
│                      (To Be Implemented)                         │
└─────────────────────────────────────────────────────────────────┘
                                ↓
    ┌───────────────────────────────────────────────────┐
    │  Diffusion Tree Sampling                          │
    │  • Build search tree from noise x_T               │
    │  • Use p_θ as transition model                    │
    │  • Back up rewards with Soft Bellman              │
    │                                                   │
    │  Output: High-quality trajectories τ              │
    └───────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3: SPECTRAL REGULARIZATION                    │
│                      (To Be Implemented)                         │
└─────────────────────────────────────────────────────────────────┘
                                ↓
    ┌───────────────────────────────────────────────────┐
    │  S-ADT Extensions to DTS                          │
    │  1. Spectral Reward:                              │
    │     r(x) = r_task(x) - γ * spectral_penalty       │
    │                                                   │
    │  2. GFlowNet Amortization:                       │
    │     Train F_φ on tree trajectories                │
    │     Minimize TB Loss                              │
    │                                                   │
    │  Output: Fast inference with F_φ guidance         │
    └───────────────────────────────────────────────────┘
                                ↓
    ┌───────────────────────────────────────────────────┐
    │  Final SpecDiffTree System                        │
    │  • Frozen OpenTSLM as prior                       │
    │  • DTS tree search with spectral rewards          │
    │  • F_φ for fast inference (200 vs 2000 rollouts) │
    │                                                   │
    │  Output: High-fidelity time series with           │
    │          preserved spectral characteristics       │
    └───────────────────────────────────────────────────┘
```

---

## 📂 Implementation Status

### ✅ Completed

- [x] OpenTSLM encoder architecture (`src/model/encoder/`)
- [x] Projection layers (`src/model/projector/`)
- [x] Curriculum learning pipeline (`curriculum_learning.py`)
- [x] MLX conversion for Apple Silicon (`mlx_training/`)

### 🚧 In Progress

- [ ] Fix Conv1d bug in `mlx_model_pretrained.py`
- [ ] Complete Stage 1 training (TSQA)

### ⏳ To Be Implemented

- [ ] DTS tree search mechanism
  - [ ] Tree data structure
  - [ ] Selection/Expansion/Rollout/Backup phases
  - [ ] Soft Bellman value updates
- [ ] S-ADT spectral rewards
  - [ ] PSD computation
  - [ ] Spectral penalty term
- [ ] GFlowNet amortization
  - [ ] Flow network F_φ architecture
  - [ ] TB loss implementation
  - [ ] Hybrid inference (MC + learned flow)

---

## 🎯 Current Focus: Phase 1 - Foundation Training

**Goal:** Train OpenTSLM Stage 1 (TSQA) on MLX

**Why MLX?** Optimized for Apple Silicon (M3 Max), efficient inference and fine-tuning

**Architecture:**

```
Time Series [batch, 1, 256]
         ↓
   CNN Encoder (trainable ~500K params)
         ↓
   Projection Layer (trainable ~2.5M params)
         ↓
   Pre-trained Llama 3.2 1B (frozen ~1B params)
         ↓
   LM Head (trainable ~260M params)
         ↓
   Predictions [batch, seq_len, vocab_size]
```

**Current Bug:** Conv1d dimension mismatch (5 min fix)

**Expected Training Time:** 1-2 hours for 2 epochs

---

## 📚 References

1. **OpenTSLM:** Langer et al., 2025 - [GitHub](https://github.com/StanfordBDHG/OpenTSLM)
2. **Diffusion Tree Sampling:** Jain et al., 2025 - [Website](https://diffusion-tree-sampling.github.io) | [arXiv:2506.20701](https://arxiv.org/abs/2506.20701)
3. **S-ADT:** Anonymous, ICLR 2026 (under review) - See `S-ADT.md`
4. **GFlowNets:** Bengio et al., 2021 - [arXiv:2111.09266](https://arxiv.org/abs/2111.09266)

---

**Last Updated:** Dec 13, 2025  
**Current Phase:** 1 (Foundation Training)  
**Next Milestone:** Complete Stage 1 training, then implement DTS + S-ADT

# SpecDiffTree: Spectral-Regularized Amortized Diffusion Trees

[![Conference](https://img.shields.io/badge/ICLR-2026_Submission-blue)](https://openreview.net)
[![Task](https://img.shields.io/badge/Task-Time_Series_Alignment-green)](#)
[![Method](https://img.shields.io/badge/Method-Diffusion_Trees_+_GFlowNets-orange)]()
[![Base](https://img.shields.io/badge/Built_on-OpenTSLM-purple)](https://github.com/StanfordBDHG/OpenTSLM)

**SpecDiffTree** extends [OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM) with **Spectral-Regularized Amortized Diffusion Trees (S-ADT)**, a framework for aligning pre-trained time series diffusion models to complex, non-differentiable objectives at inference time.

Built on OpenTSLM's curriculum learning framework, S-ADT addresses two critical failures in existing alignment methods:
- **Spectral Collapse**: Greedy guidance destroys high-frequency textures
- **Computational Inefficiency**: MCTS requires thousands of function evaluations

By combining Soft Bellman backups with GFlowNet amortization, S-ADT achieves high-fidelity spectral textures with **10x fewer function evaluations** than standard tree search.

---

## 🎯 Key Idea

**OpenTSLM** provides the foundation: a pre-trained time series language model trained through curriculum learning (MCQ → Captioning → Chain-of-Thought reasoning).

**S-ADT** adds inference-time alignment: Given OpenTSLM's frozen diffusion prior, we align it to complex rewards (spectral fidelity, constraints, task objectives) without retraining.

```
OpenTSLM (Pre-trained)  →  S-ADT (Inference Alignment)  →  Task-Specific Outputs
```

---

## 📖 Table of Contents
- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [Mathematical Framework](#-mathematical-framework)
- [Integration with OpenTSLM](#-integration-with-opentslm)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Citation](#-citation)

---

## 🔥 The Problem

### OpenTSLM Success
OpenTSLM's curriculum learning produces excellent time series models:
- **Stage 1**: Multiple-choice QA (TSQA dataset)
- **Stage 2**: Captioning (M4 dataset)
- **Stages 3-5**: Chain-of-thought reasoning (HAR, Sleep, ECG)

### The Alignment Challenge
At inference, we often need to satisfy additional constraints:
- Preserve spectral characteristics from historical data
- Satisfy hard constraints (non-negativity, bounds)
- Optimize for non-differentiable metrics (CRPS, DTW)

**Standard approaches fail:**
1. **Gradient guidance** (DPS): Assumes differentiable rewards
2. **Classifier-free guidance**: Requires training multiple models
3. **Greedy search**: Converges to mean, destroying texture

---

## 💡 Our Solution

### S-ADT: Three Key Components

#### 1. Spectral-Regularized Tree Search
Build a Monte Carlo search tree that preserves frequency content:

$$
r(\mathbf{x}_0) = r_{\text{task}}(\mathbf{x}_0) - \gamma \int \left| \log S_{\mathbf{x}_0}(\omega) - \log \mathbb{E}[S_{\mathbf{c}}(\omega)] \right| d\omega
$$

- Uses OpenTSLM's reverse diffusion as the transition model
- Backs up values with **Soft Bellman** (LogSumExp), not max
- Preserves multimodal spectral structure

#### 2. GFlowNet Amortization
Learn to predict search tree values with a parametric network $F_\phi$:

$$
\mathcal{L}_{TB}(\tau) = \left( \log Z_\phi + \sum_{t=T}^1 \log \frac{F_\phi(\mathbf{x}_{t-1})}{P_F(\mathbf{x}_t|\mathbf{x}_{t-1})} - \lambda r(\mathbf{x}_0) \right)^2
$$

- Trains on trajectories harvested from tree search
- **10x speedup**: 200 rollouts vs 2000 for pure search

#### 3. Hybrid Inference
Combine learned flow with Monte Carlo estimates:

$$
\pi_{\text{select}} \propto \exp(\lambda [ (1-\alpha)\hat{v}_{\text{MC}} + \alpha F_\phi ])
$$

---

## 🧮 Mathematical Framework

### The Alignment Problem
Given OpenTSLM's frozen prior $p_\theta(\mathbf{x}|\mathbf{c})$, sample from:

$$
\pi^*(\mathbf{x}) \propto p_\theta(\mathbf{x}|\mathbf{c}) \exp(\lambda r(\mathbf{x}))
$$

### Spectral Collapse Theorem
**Why greedy fails**: Greedy search approximates $\mathbb{E}[\mathbf{x}]$, which acts as a low-pass filter.

**Proposition**: The Power Spectral Density of the greedy estimator is bounded:

$$
S_{\hat{\mathbf{x}}}(\omega) = \| \mathcal{F}(\mathbb{E}[\mathbf{x}])(\omega) \|^2 \leq \mathbb{E} [ \| \mathcal{F}(\mathbf{x})(\omega) \|^2 ]
$$

*Proof*: Jensen's inequality on $\|\cdot\|^2$ (convex). Averaging destroys phase information.

### Soft Bellman Backup
Preserve probability mass across modes:

$$
V_t(\mathbf{x}_t) = \frac{1}{\lambda} \log \mathbb{E}_{p_\theta(\cdot|\mathbf{x}_t)} \left[ \exp(\lambda V_{t-1}(\mathbf{x}_{t-1})) \right]
$$

Uses LogSumExp instead of max, preventing collapse to single mode.

---

## 🔗 Integration with OpenTSLM

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      OpenTSLM Foundation                      │
│  (Pre-trained via Curriculum Learning: Stages 1-5)           │
├─────────────────────────────────────────────────────────────┤
│  • Time Series Encoder (TSEncoder)                           │
│  • Projector (to LLM space)                                  │
│  • LLM Backbone (Llama 3.2 1B)                              │
│  • Diffusion Model (for generation)                          │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    S-ADT Extension                           │
│              (Inference-Time Alignment)                      │
├─────────────────────────────────────────────────────────────┤
│  1. Tree Search Module                                       │
│     • Uses OpenTSLM's diffusion as transition model         │
│     • Spectral reward function                               │
│     • Soft Bellman backup                                    │
│                                                              │
│  2. GFlowNet Amortization                                   │
│     • Flow network F_φ                                       │
│     • Trajectory Balance loss                                │
│     • Learns from tree search buffer                         │
│                                                              │
│  3. Hybrid Inference                                        │
│     • Combines MC estimates + learned flow                   │
│     • 10x speedup over pure search                          │
└─────────────────────────────────────────────────────────────┘
```

### Code Structure

```python
# OpenTSLM components (pre-trained)
from src.model.llm import OpenTSLMSP, OpenTSLMFlamingo
from curriculum_learning import CurriculumTrainer

# S-ADT extensions (this work)
from src.alignment.tree_search import SpectralTreeSearch
from src.alignment.gflownet import FlowNetwork, TrajectoryBalanceLoss
from src.alignment.inference import HybridInference
```

### Using Pre-trained OpenTSLM Models

```python
# Load pre-trained OpenTSLM checkpoint
model = OpenTSLMSP.from_pretrained("checkpoints/stage5_final.pt")

# Initialize S-ADT on top
sadt = SpectralTreeSearch(
    diffusion_model=model,
    spectral_gamma=1.0,
    temperature=0.5
)

# Run alignment
aligned_forecast = sadt.search(history_context, reward_fn, n_rollouts=200)
```

---

## 💻 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/vincehass/SpecDiffTree.git
cd SpecDiffTree

# Create environment
conda create -n specdifftree python=3.10
conda activate specdifftree

# Install dependencies
pip install -r requirements.txt

# Initialize OpenTSLM submodule (if needed)
git submodule update --init src/open_flamingo
pip install -e src/open_flamingo --no-deps
```

---

## 🚀 Usage

### 1. Use Pre-trained OpenTSLM

If you have a pre-trained OpenTSLM model:

```python
from src.model.llm import OpenTSLMSP
from src.alignment import SpectralTreeSearch

# Load OpenTSLM
model = OpenTSLMSP.from_pretrained("path/to/checkpoint.pt")

# Add S-ADT alignment
sadt = SpectralTreeSearch(model, spectral_gamma=1.0)
forecast = sadt.align(history, reward_fn)
```

### 2. Train OpenTSLM from Scratch

Follow OpenTSLM's curriculum:

```bash
# Stage 1: Multiple Choice QA
python curriculum_learning.py --stage stage1_mcq --epochs 30

# Stage 2: Captioning
python curriculum_learning.py --stage stage2_captioning --epochs 20

# Stages 3-5: Chain of Thought
python curriculum_learning.py --stage stage3_cot --epochs 60
```

### 3. Run S-ADT Inference

#### Pure Tree Search (Baseline)
```bash
python scripts/run_inference.py \
  --model checkpoints/opentslm_stage5.pt \
  --method spectral_tree \
  --dataset ETTh1 \
  --n_rollouts 2000
```

#### Train GFlowNet Amortization
```bash
python scripts/train_gflownet.py \
  --buffer results/tree_buffer.pkl \
  --lr 1e-4 \
  --epochs 100
```

#### Fast Inference with S-ADT
```bash
python scripts/run_inference.py \
  --model checkpoints/opentslm_stage5.pt \
  --method sadt \
  --flow_checkpoint checkpoints/flow_net.pt \
  --n_rollouts 200
```

---

## 📊 Results

Performance on **ETTh1** benchmark (96-step horizon), using OpenTSLM Stage 5 as base model.

| Method | CRPS ↓ | Spec-W1 ↓ | Reward ↑ | NFE |
|--------|--------|-----------|----------|-----|
| OpenTSLM (Base) | 0.385 | 0.45 | -12.4 | 1 |
| + DPS Guidance | 0.410 | 0.42 | -8.5 | 50 |
| + SMC Steering | 0.390 | 0.35 | -6.2 | 250 |
| + Tree Search | 0.375 | **0.15** | **-2.1** | 2000 |
| **+ S-ADT (Ours)** | **0.371** | **0.16** | -2.3 | **200** |

**Key Findings**:
- S-ADT matches expensive tree search quality
- **10x computational speedup** (200 vs 2000 NFE)
- Preserves spectral fidelity (Spec-W1 ≈ 0.16)

---

## 🏗️ Repository Structure

```
SpecDiffTree/
├── src/
│   ├── model/                # OpenTSLM models
│   │   ├── llm/             # OpenTSLMSP, OpenTSLMFlamingo
│   │   └── encoders/        # Time series encoders
│   ├── datasets/            # TSQA, M4, HAR, Sleep, ECG loaders
│   ├── alignment/           # S-ADT implementation (NEW)
│   │   ├── tree_search.py   # Spectral tree search
│   │   ├── gflownet.py      # Flow network & TB loss
│   │   └── inference.py     # Hybrid inference
│   └── open_flamingo/       # Flamingo architecture (submodule)
├── scripts/
│   ├── run_inference.py     # Inference with S-ADT
│   └── train_gflownet.py    # Train amortization
├── configs/                 # Experiment configs
├── curriculum_learning.py   # OpenTSLM training
└── README.md               # This file
```

---

## 🔬 Key Contributions

1. **Spectral Collapse Theorem**: Formal proof of why greedy alignment fails
2. **Soft Bellman + Spectral Rewards**: Solution preserving high-frequency content
3. **GFlowNet Amortization**: 10x speedup via learned search
4. **OpenTSLM Integration**: Extends curriculum-learned models with alignment

---

## 📜 Citation

```bibtex
@inproceedings{specdifftree2026,
  title={Spectral-Regularized Amortized Diffusion Trees: Scalable Inference-Time Alignment for Time Series},
  author={Anonymous Authors},
  booktitle={Under Review at ICLR 2026},
  year={2026}
}
```

---

## 🙏 Acknowledgements

This work builds upon:
- **OpenTSLM** - Stanford BDHG (Foundation model via curriculum learning)
- **Diffusion Tree Sampling** - Jain et al., 2025
- **GFlowNets** - Bengio et al., 2021

---

## 📝 License

MIT License - see [LICENSE.md](LICENSE.md)

---

## 📧 Contact

For questions, open an issue on GitHub.

---

**Status**: Under Review at ICLR 2026 | Built on OpenTSLM 🚀

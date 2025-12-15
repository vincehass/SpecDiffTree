# S-ADT: Spectral-Regularized Amortized Diffusion Trees

[![Conference](https://img.shields.io/badge/ICLR-2026_Submission-blue)](https://openreview.net)
[![Task](https://img.shields.io/badge/Task-Time_Series_Alignment-green)](https://github.com/StanfordBDHG/OpenTSLM)
[![Method](https://img.shields.io/badge/Method-Diffusion_Trees_+_GFlowNets-orange)]()

This repository contains the official implementation of **Spectral-Regularized Amortized Diffusion Trees (S-ADT)**, a framework for aligning pre-trained time series diffusion models to complex, non-differentiable objectives at inference time.

S-ADT addresses two critical failures in existing alignment methods: **Spectral Collapse** (oversmoothing due to greedy guidance) and **Computational Inefficiency** (high inference cost of MCTS). By combining Soft Bellman backups with Generative Flow Network (GFlowNet) amortization, S-ADT achieves high-fidelity spectral textures with **10x fewer function evaluations** than standard search.

---

## 📖 Table of Contents
- [Abstract](#-abstract)
- [Mathematical Breakdown](#-mathematical-breakdown)
  - [The Alignment Problem](#1-the-alignment-problem)
  - [The Spectral Collapse Theorem](#2-the-spectral-collapse-theorem)
  - [Solution: Soft Bellman & Spectral Rewards](#3-solution-soft-bellman--spectral-rewards)
  - [Amortization: The GFlowNet Connection](#4-amortization-the-gflownet-connection)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Citation](#-citation)

---

## 🧩 Abstract

Aligning pre-trained diffusion models to complex objectives at inference time is a fundamental challenge in time series forecasting. While methods like Diffusion Tree Sampling (DTS) offer asymptotically exact posterior sampling, they suffer from:
1.  **Spectral Collapse:** Greedy search converges to the conditional expectation, obliterating high-frequency textures.
2.  **Computational Inefficiency:** Non-parametric trees forget values between queries.

S-ADT formulates alignment as a stochastic optimal control problem. We introduce a **Spectral Soft Bellman Backup** to propagate frequency-domain constraints and a **GFlowNet Amortization** module that learns a parametric flow function $F_\phi$ from the search tree, allowing the model to "learn to search" efficiently.

---

## 🧮 Mathematical Breakdown

### 1. The Alignment Problem
Given a frozen, pre-trained diffusion prior $p_\theta(\mathbf{x}|\mathbf{c})$ conditioned on history $\mathbf{c}$, and a reward function $r(\mathbf{x})$, we aim to sample from the twisted posterior:

$$
\pi^*(\mathbf{x}) \propto p_\theta(\mathbf{x}|\mathbf{c}) \exp(\lambda r(\mathbf{x}))
$$

where $\lambda$ is the inverse temperature.

### 2. The Spectral Collapse Theorem
Why do standard methods (Guidance, DPS, Greedy Search) fail for time series?
Greedy methods approximate the mode, which converges to the conditional expectation $\hat{\mathbf{x}} \approx \mathbb{E}_{\pi^*}[\mathbf{x}]$.

**Proposition 3.2 (Greedy Spectral Collapse):**
Let $\mathcal{F}$ be the Fourier transform. The Power Spectral Density (PSD) of the greedy estimator is strictly lower than the expected PSD of the true distribution.

$$
S_{\hat{\mathbf{x}}}(\omega) = \| \mathcal{F}(\mathbb{E}[\mathbf{x}])(\omega) \|^2 \leq \mathbb{E} [ \| \mathcal{F}(\mathbf{x})(\omega) \|^2 ] = \mathbb{E}[S_{\mathbf{x}}(\omega)]
$$

*Proof Intuition:* By Jensen's inequality on the convex norm function $\|\cdot\|^2$, the operation of averaging (expectation) commutes destructively with spectral magnitude when phase is uncertain. This acts as a low-pass filter, removing texture.

### 3. Solution: Soft Bellman & Spectral Rewards
To solve collapse, we must sample proportional to value, not maximize it. We define the **Soft Value Function** $V_t(\mathbf{x}_t)$:

$$
V_t(\mathbf{x}_t) = \frac{1}{\lambda} \log \mathbb{E}_{p_\theta(\cdot|\mathbf{x}_t)} \left[ \exp(\lambda V_{t-1}(\mathbf{x}_{t-1})) \right]
$$

We explicitly enforce texture matching using a **Spectral Reward** $r_S$:

$$
r(\mathbf{x}_0) = r_{\text{task}}(\mathbf{x}_0) - \gamma \int \left| \log S_{\mathbf{x}_0}(\omega) - \log \mathbb{E}[S_{\mathbf{c}_{\text{ext}}}(\omega)] \right| d\omega
$$

### 4. Amortization: The GFlowNet Connection
Calculating $V_t$ via Monte Carlo tree search is expensive ($O(10^3)$ steps). We amortize this by learning a parametric flow network $F_\phi(\mathbf{x}_t, t) \approx V_t(\mathbf{x}_t)$.

We minimize the **Trajectory Balance (TB) Loss** over trajectories $\tau$ harvested from the tree:

$$
\mathcal{L}_{TB}(\tau) = \left( \log Z_\phi + \sum_{t=T}^1 \left( \lambda F_\phi(\mathbf{x}_{t-1}) - \log \mathbb{E}_{z}[e^{\lambda F_\phi(z)}] \right) - \lambda r(\mathbf{x}_0) \right)^2
$$

Minimizing this ensures $F_\phi$ converges to the true Soft Value $V_t$, allowing us to use $F_\phi$ as a heuristic to guide future searches with minimal compute.

---

## ⚙️ Methodology

S-ADT operates in two phases:

1.  **Spectral-Regularized Tree Search:**
    *   Constructs a search tree rooted at noise $\mathbf{x}_T$.
    *   Expands nodes using the pre-trained diffusion model.
    *   Backs up rewards using the **LogSumExp** operator (Eq. 7 in paper) rather than max, preserving probability mass across multimodal spectral phases.

2.  **Amortized Learning:**
    *   Treats the diffusion process as a GFlowNet.
    *   Trains $F_\phi$ on the search buffer.
    *   **Hybrid Selection:** During inference, selects children using a mixture of the Monte Carlo estimate $\hat{v}$ (high variance, unbiased) and the learned flow $F_\phi$ (low variance, biased).

    $$
    \pi_{\text{select}} \propto \exp(\lambda [ (1-\alpha)\hat{v} + \alpha F_\phi ])
    $$

---

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/S-ADT.git
cd S-ADT

# Create environment
conda create -n sadt python=3.10
conda activate sadt

# Install dependencies (requires PyTorch and OpenTSLM)
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run Pure Tree Search (DTS with Spectral Reward)
This runs the non-amortized baseline to gather high-fidelity trajectories.

```bash
python run_inference.py \
  --backbone diff-k \
  --dataset ETTh1 \
  --method dts \
  --spectral_gamma 1.0 \
  --n_rollouts 2000
```

### 2. Train Amortization (S-ADT)
Train the flow network $F_\phi$ using trajectories from the previous step.

```bash
python train_amortization.py \
  --buffer_path ./results/dts_buffer.pkl \
  --lr 1e-4 \
  --epochs 100
```

### 3. Fast Inference (S-ADT)
Run inference using the trained flow network for guidance (10x speedup).

```bash
python run_inference.py \
  --backbone diff-k \
  --dataset ETTh1 \
  --method sadt \
  --checkpoint ./checkpoints/flow_net.pt \
  --n_rollouts 200
```

---

## 📊 Results

Performance on **ETTh1** benchmark (Horizon 96). S-ADT matches the spectral fidelity of expensive search (DTS) with the speed of SMC.

| Method | CRPS ($\downarrow$) | Spec-W1 ($\downarrow$) | Reward ($\uparrow$) | NFE (Cost) |
| :--- | :---: | :---: | :---: | :---: |
| OpenTSLM (Base) | 0.385 | 0.45 | -12.4 | 1 |
| DPS (Gradient Guidance) | 0.410 | 0.42 | -8.5 | 50 |
| SMC-Steering | 0.390 | 0.35 | -6.2 | 250 |
| DTS (Pure Search) | 0.375 | **0.15** | **-2.1** | 2000 |
| **S-ADT (Ours)** | **0.371** | **0.16** | -2.3 | **200** |

* **Spec-W1:** Wasserstein-1 distance between forecast PSD and history PSD (Lower is better).
* **NFE:** Number of Function Evaluations.

---

## 📜 Citation

If you find this work useful, please cite the ICLR 2026 submission:

```bibtex
@inproceedings{sadt2026,
  title={Spectral-Regularized Amortized Diffusion Trees: Scalable Inference-Time Alignment for Time Series},
  author={Anonymous Authors},
  booktitle={Under Review at ICLR 2026},
  year={2026}
}
```

Acknowledgements to **DTS** (Jain et al., 2025) and **OpenTSLM** (Langer et al., 2025) for the foundational frameworks used in this work.
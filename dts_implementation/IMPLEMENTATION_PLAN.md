# DTS + S-ADT Implementation Plan for OpenTSLM

**Date:** Dec 13, 2025  
**Goal:** Implement Diffusion Tree Sampling with Spectral Regularization for OpenTSLM Stages 1-5

---

## 🎯 Overview

We are implementing inference-time alignment for **pre-trained OpenTSLM models** using:

1. **DTS (Diffusion Tree Sampling)** - Base tree search algorithm  
   Source: https://github.com/vineetjain96/Diffusion-Tree-Sampling

2. **S-ADT (Spectral-Regularized Amortized Diffusion Trees)** - Our novel contribution  
   Source: `S-ADT.md`

3. **OpenTSLM Stages 1-5** - Pre-trained time series models  
   Source: Existing OpenTSLM checkpoints

---

## 📊 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Pre-trained OpenTSLM Models (FROZEN - No Training)          │
│  ├── Stage 1: TSQA (Multiple-Choice QA)                      │
│  ├── Stage 2: M4 (Captioning)                                │
│  ├── Stage 3: HAR CoT (Human Activity Recognition)           │
│  ├── Stage 4: Sleep CoT (Sleep Stage Classification)         │
│  └── Stage 5: ECG QA CoT (Electrocardiogram QA)             │
└──────────────────────────────────────────────────────────────┘
                          ↓ (inference only)
┌──────────────────────────────────────────────────────────────┐
│  DTS Tree Search (Inference Algorithm - No Training)         │
│  ├── MCTSNode: Track states x_t at timestep t               │
│  ├── Selection: Boltzmann policy π ∝ exp(λ v̂)               │
│  ├── Expansion: Sample from p_θ(x_{t-1} | x_t)               │
│  ├── Rollout: Complete trajectory to x_0                     │
│  └── Backup: Soft Bellman value updates                      │
└──────────────────────────────────────────────────────────────┘
                          ↓ (evaluation)
┌──────────────────────────────────────────────────────────────┐
│  S-ADT Spectral Rewards (Our Contribution - No Training)     │
│  ├── Task Reward: r_task(x_0) - stage-specific              │
│  ├── Spectral Penalty: γ ∫ |log S_x - log E[S_c]| dω        │
│  └── Total: r(x_0) = r_task(x_0) - spectral_penalty         │
└──────────────────────────────────────────────────────────────┘
                          ↓ (optional speedup)
┌──────────────────────────────────────────────────────────────┐
│  GFlowNet Amortization (Small Training Component)            │
│  ├── Flow Network F_φ: Learns to predict search values       │
│  ├── Training: TB loss on harvested trajectories            │
│  └── Result: 10x speedup (200 vs 2000 rollouts)             │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚫 What We're NOT Doing

❌ Training OpenTSLM from scratch  
❌ Training the LLM backbone  
❌ Training the time series encoder  
✅ Only inference-time tree search + optional small F_φ training

---

## 📁 Implementation Structure

```
dts_implementation/
├── core/
│   ├── dts_node.py          # MCTS tree node
│   ├── dts_sampler.py       # Main DTS algorithm
│   └── soft_bellman.py      # Value backup logic
├── rewards/
│   ├── spectral_reward.py   # S-ADT spectral penalty
│   ├── stage1_reward.py     # TSQA task reward
│   ├── stage2_reward.py     # M4 task reward
│   ├── stage3_reward.py     # HAR task reward
│   ├── stage4_reward.py     # Sleep task reward
│   └── stage5_reward.py     # ECG task reward
├── models/
│   ├── opentslm_wrapper.py  # Load pre-trained OpenTSLM
│   └── gflownet.py          # Optional F_φ network
├── utils/
│   ├── psd_utils.py         # Power Spectral Density computation
│   └── tree_utils.py        # Tree visualization/debugging
├── run_dts_stage1.py        # Run DTS on Stage 1
├── run_dts_stage2.py        # Run DTS on Stage 2
├── run_dts_stage3.py        # Run DTS on Stage 3
├── run_dts_stage4.py        # Run DTS on Stage 4
├── run_dts_stage5.py        # Run DTS on Stage 5
└── IMPLEMENTATION_PLAN.md   # This file
```

---

## 🔧 Implementation Steps

### Phase 1: Core DTS (Priority)

- [ ] **Step 1:** Implement `MCTSNode` class

  - State x_t representation
  - Value estimation v̂(x_t)
  - Visit count tracking
  - Children management

- [ ] **Step 2:** Implement `DTSSampler` class

  - Selection (Boltzmann policy)
  - Expansion (sample from p_θ)
  - Rollout (complete trajectory)
  - Backup (Soft Bellman updates)

- [ ] **Step 3:** Implement `SoftBellmanBackup`
  - LogSumExp value aggregation
  - Temperature parameter λ
  - Prevent spectral collapse

### Phase 2: OpenTSLM Integration

- [ ] **Step 4:** Create `OpenTSLMWrapper`

  - Load pre-trained checkpoints
  - Wrap forward pass for DTS
  - Handle stage-specific inputs
  - Diffusion process interface

- [ ] **Step 5:** Implement stage-specific rewards
  - Stage 1: Accuracy for TSQA
  - Stage 2: BLEU/ROUGE for M4
  - Stage 3: F1-score for HAR
  - Stage 4: Cohen's Kappa for Sleep
  - Stage 5: Accuracy for ECG QA

### Phase 3: S-ADT Spectral Rewards

- [ ] **Step 6:** Implement PSD computation

  - FFT for time series
  - Power Spectral Density
  - Handle variable-length sequences

- [ ] **Step 7:** Implement spectral reward
  - Compute PSD for x_0
  - Compute expected PSD for context c
  - Wasserstein distance in frequency domain
  - Tunable penalty weight γ

### Phase 4: GFlowNet (Optional Speedup)

- [ ] **Step 8:** Implement Flow Network F_φ

  - Small MLP architecture
  - Input: (x_t, t)
  - Output: value estimate

- [ ] **Step 9:** Train F_φ on harvested trajectories
  - Trajectory Balance (TB) loss
  - Buffer management
  - Training loop

### Phase 5: Testing & Integration

- [ ] **Step 10:** Test on Stage 1 (TSQA)
- [ ] **Step 11:** Extend to Stages 2-5
- [ ] **Step 12:** Compare with baselines
- [ ] **Step 13:** Document results

---

## 🎓 Key Algorithms

### DTS Algorithm (from paper)

```python
def dts_rollout(x_T, model, reward_fn, num_rollouts):
    """
    DTS: Diffusion Tree Sampling

    Args:
        x_T: Initial noise
        model: Pre-trained diffusion model p_θ
        reward_fn: Black-box reward r(x_0)
        num_rollouts: Number of MCTS iterations

    Returns:
        x_0: Final sample
        tree: Search tree with values
    """
    root = MCTSNode(x_T, t=T)

    for _ in range(num_rollouts):
        # 1. SELECTION: Choose path using Boltzmann policy
        node = root
        while not node.is_leaf():
            node = select_child_boltzmann(node, temperature=λ)

        # 2. EXPANSION: Sample next state
        if node.t > 0:
            x_prev = model.p_sample(node.x_t, node.t)
            child = MCTSNode(x_prev, t=node.t-1, parent=node)
            node.children.append(child)
            node = child

        # 3. ROLLOUT: Complete trajectory
        x_0 = rollout_to_x0(node, model)

        # 4. BACKUP: Update values with Soft Bellman
        r = reward_fn(x_0)
        backup_soft_bellman(node, r, temperature=λ)

    # Return best sample
    return select_best_x0(root)
```

### S-ADT Spectral Reward (from S-ADT.md)

```python
def spectral_reward(x_0, context_c, r_task, gamma=1.0):
    """
    S-ADT Spectral-Regularized Reward

    Args:
        x_0: Predicted time series
        context_c: Historical context
        r_task: Task-specific reward
        gamma: Spectral penalty weight

    Returns:
        r: Total reward with spectral regularization
    """
    # Compute PSDs
    S_x0 = compute_psd(x_0)
    S_c_expected = compute_expected_psd(context_c)

    # Spectral penalty (Wasserstein in log-freq space)
    spectral_penalty = np.trapz(
        np.abs(np.log(S_x0) - np.log(S_c_expected)),
        dx=freq_resolution
    )

    # Total reward
    r = r_task(x_0) - gamma * spectral_penalty

    return r
```

### Soft Bellman Backup (from paper)

```python
def soft_bellman_backup(node, reward, temperature):
    """
    Soft Bellman Backup (LogSumExp)
    Prevents spectral collapse by maintaining diversity

    V_t(x_t) = (1/λ) log E[exp(λ V_{t-1}(x_{t-1}))]
    """
    current = node
    value = reward

    while current is not None:
        # Update visit count
        current.visit_count += 1

        # Soft update (not max!)
        if current.children:
            child_values = [child.value_est for child in current.children]
            # LogSumExp aggregation
            value = (1/temperature) * torch.logsumexp(
                temperature * torch.tensor(child_values), dim=0
            ).item()

        current.value_est = value
        current = current.parent
```

---

## 📊 Expected Results

Based on S-ADT paper (Table in S-ADT.md):

| Method            | CRPS ↓    | Spec-W1 ↓ | Reward ↑ | NFE     |
| ----------------- | --------- | --------- | -------- | ------- |
| OpenTSLM (Base)   | 0.385     | 0.45      | -12.4    | 1       |
| DTS (Tree Search) | 0.375     | 0.15      | -2.1     | 2000    |
| **S-ADT (Ours)**  | **0.371** | **0.16**  | **-2.3** | **200** |

**Key Improvements:**

- ✅ Better CRPS than base OpenTSLM
- ✅ Much better spectral fidelity (Spec-W1)
- ✅ 10x fewer rollouts than pure DTS

---

## 🧪 Testing Strategy

### Unit Tests

1. Test MCTSNode creation and updates
2. Test DTS selection/expansion/backup
3. Test PSD computation
4. Test spectral reward calculation

### Integration Tests

1. Test DTS with dummy OpenTSLM
2. Test full pipeline on Stage 1
3. Compare DTS vs baseline

### Performance Tests

1. Measure NFE (Number of Function Evaluations)
2. Measure wall-clock time
3. Compare 200 vs 2000 rollouts

---

## 🚀 Quick Start (After Implementation)

```bash
cd /Users/nhassen/Documents/Adv_pretrained/LLM_repos/SpecDiffTree

# Run DTS on Stage 1 (TSQA)
python dts_implementation/run_dts_stage1.py \
    --checkpoint results/opentslm_stage1_checkpoint.pt \
    --num_rollouts 200 \
    --spectral_gamma 1.0 \
    --temperature 0.1

# Run full DTS (2000 rollouts)
python dts_implementation/run_dts_stage1.py \
    --checkpoint results/opentslm_stage1_checkpoint.pt \
    --num_rollouts 2000 \
    --spectral_gamma 1.0

# Train GFlowNet F_φ
python dts_implementation/train_gflownet.py \
    --trajectory_buffer dts_trajectories.pkl \
    --epochs 100
```

---

## 📚 References

1. **DTS Paper:** Jain et al., 2025 - "Diffusion Tree Sampling: Scalable inference-time alignment of diffusion models"

   - arXiv: [2506.20701](https://arxiv.org/abs/2506.20701)
   - Code: https://github.com/vineetjain96/Diffusion-Tree-Sampling

2. **S-ADT:** See `S-ADT.md` in this repository

3. **OpenTSLM:** Stanford BDHG - https://github.com/StanfordBDHG/OpenTSLM

4. **GFlowNets:** Bengio et al., 2021 - [arXiv:2111.09266](https://arxiv.org/abs/2111.09266)

---

## ✅ Current Status

- [x] Stopped incorrect training approach
- [x] Cloned DTS reference implementation
- [x] Understood DTS algorithm structure
- [x] Created implementation plan
- [ ] **Next:** Implement core DTS components

---

**Last Updated:** Dec 13, 2025, 4:15 PM  
**Implementation Phase:** Planning → Core DTS  
**Estimated Time:** 4-6 hours for full implementation

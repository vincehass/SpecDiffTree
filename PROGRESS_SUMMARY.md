# SpecDiffTree Progress Summary - Dec 13, 2025

## 🏗️ Project Architecture (3-Part System)

**See `ARCHITECTURE.md` for complete technical details**

SpecDiffTree combines three frameworks:

### 1. **OpenTSLM** - Foundation Model (Phase 1 - Current Work)

- **Source:** [Stanford BDHG](https://github.com/StanfordBDHG/OpenTSLM)
- **Purpose:** Time series encoder + LLM via curriculum learning
- **Status:** Training Stage 1 (TSQA) on MLX

### 2. **Diffusion Tree Sampling (DTS)** - Tree Search (Phase 2 - Next)

- **Source:** [Jain et al., 2025](https://diffusion-tree-sampling.github.io)
- **Purpose:** Inference-time alignment via tree search
- **Status:** To be implemented after Phase 1 complete

### 3. **S-ADT** - Spectral Regularization + Amortization (Phase 3 - Final)

- **Source:** `S-ADT.md` (ICLR 2026 submission)
- **Purpose:** Spectral rewards + GFlowNet for 10x speedup
- **Status:** To be implemented after DTS

---

## 🎯 Current Work: Phase 1 - OpenTSLM Stage 1 Training

### Architecture

```
Time Series [batch, 1, 256]
         ↓
   CNN Encoder (~500K params, trainable)
         ↓
   Projection Layer (~2.5M params, trainable)
         ↓
   Pre-trained Llama 3.2 1B (~1B params, FROZEN)
         ↓
   LM Head (~260M params, trainable)
         ↓
   Predictions [batch, seq_len, vocab_size]
```

**Total Trainable:** ~263M parameters  
**Total Model Size:** ~1.3B parameters

---

## 📋 What's Done

### ✅ Yesterday's Progress

1. **Discovered MLX Limitations:**
   - ❌ Training from scratch: too slow (43+ hours)
   - ❌ Zero gradients with large models
2. **Found the Right Approach:**

   - ✅ Use pre-trained MLX Llama (already downloaded!)
   - ✅ Only train encoder+projector+LM head
   - ✅ Should complete in 1-2 hours

3. **Created Files:**
   - ✅ `mlx_model_pretrained.py` (99% done)
   - ✅ Updated `mlx_trainer.py`
   - ✅ Architecture documentation

### ✅ All Previous Fixes Applied

- Built-in cross-entropy loss
- Logit clipping
- Conservative learning rate
- Sequence length handling
- Better diagnostics

---

## 🐛 Current Bug to Fix Today

**Conv1d Dimension Mismatch:**

```python
ValueError: [conv] Expect the input channels in the input and weight array to match
but got shapes - input: (2,1,256) and weight: (128,7,1)
```

**Location:** `mlx_model_pretrained.py`, line ~37 in `TimeSeriesEncoder.__init__`

**Fix Required:**
The Conv1d layer in MLX might have different parameter ordering than PyTorch.

**Options to try:**

1. Check MLX Conv1d documentation for parameter order
2. Verify `in_channels` vs `out_channels` ordering
3. Test with explicit shape debugging

---

## 🚀 Today's Plan

### Step 1: Fix Conv1d Bug (5-10 minutes)

```bash
cd /Users/nhassen/Documents/Adv_pretrained/LLM_repos/SpecDiffTree
source opentslm_env/bin/activate
export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH

# Test model creation
python -c "from mlx_training.mlx_model_pretrained import create_model_pretrained; \
           model = create_model_pretrained({'patch_size': 4}, 128256)"
```

### Step 2: Verify Forward Pass (5 minutes)

```python
import mlx.core as mx
from mlx_training.mlx_model_pretrained import create_model_pretrained

config = {'llm_id': 'mlx-community/Llama-3.2-1B-Instruct-4bit', 'patch_size': 4}
model = create_model_pretrained(config, 128256)

# Test
ts = mx.random.normal((2, 1, 256))
input_ids = mx.random.randint(0, 128256, (2, 50))
labels = mx.random.randint(0, 128256, (2, 50))

logits, loss = model(ts, input_ids=input_ids, labels=labels)
print(f"✅ Logits: {logits.shape}, Loss: {loss.item():.4f}")
```

### Step 3: Run Training! (1-2 hours)

```bash
python -B mlx_training/mlx_trainer.py --config configs/mlx/quick_test.yaml
```

**Expected Results:**

- ✅ Fast training: ~10-20 it/s
- ✅ Stable loss (no NaN)
- ✅ Real gradients flowing
- ⏱️ Complete in 1-2 hours

---

## 📁 Key Files

**Main Implementation:**

- `mlx_training/mlx_model_pretrained.py` (needs Conv1d fix)
- `mlx_training/mlx_trainer.py` (ready)
- `mlx_training/mlx_data.py` (ready)

**Config:**

- `configs/mlx/quick_test.yaml`

**Documentation:**

- `ARCHITECTURE.md` (complete system overview)
- `S-ADT.md` (methodology)
- `README.md` (project overview)

**Environment:**

- Virtual env: `opentslm_env`
- Python: 3.12
- MLX: 0.30.0
- mlx-lm: 0.28.4

---

## 🔧 Quick Reference Commands

```bash
# Activate environment
cd /Users/nhassen/Documents/Adv_pretrained/LLM_repos/SpecDiffTree
source opentslm_env/bin/activate
export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH

# Test model
python -c "from mlx_training.mlx_model_pretrained import create_model_pretrained; \
           model = create_model_pretrained({'patch_size': 4}, 128256)"

# Run training
python -B mlx_training/mlx_trainer.py --config configs/mlx/quick_test.yaml

# Monitor W&B
# https://wandb.ai/nadhirvincenthassen/specdifftree
```

---

## 💡 Why This Approach Will Work

1. **Pre-trained LLM**: Already knows language patterns
2. **Focused Training**: Only ~263M trainable params (encoder+projector+head)
3. **MLX Optimized**: This is exactly what MLX was designed for
4. **Fast Convergence**: Frozen LLM = stable, fast training
5. **Low Memory**: 4-bit quantized model fits in M3 Max memory

---

## 🎯 Success Criteria - Phase 1

- [x] Model loads pre-trained Llama ✅
- [x] Trainer updated to use pretrained model ✅
- [ ] Conv1d dimensions fixed
- [ ] Forward pass works
- [ ] Training loop runs
- [ ] Loss decreases (no NaN)
- [ ] Complete 2 epochs in < 2 hours

---

## 🔮 Next Steps After Phase 1

Once Stage 1 training completes:

1. **Complete OpenTSLM Curriculum** (Stages 2-5)

   - M4 captioning
   - HAR CoT
   - Sleep CoT
   - ECG QA CoT

2. **Implement DTS Tree Search**

   - Tree data structure
   - Selection/Expansion/Rollout/Backup
   - Soft Bellman updates

3. **Add S-ADT Extensions**
   - Spectral reward computation
   - GFlowNet flow network F_φ
   - Trajectory Balance loss
   - Hybrid inference

---

## 📚 References

1. **OpenTSLM:** [GitHub](https://github.com/StanfordBDHG/OpenTSLM)
2. **DTS:** [Website](https://diffusion-tree-sampling.github.io) | [arXiv:2506.20701](https://arxiv.org/abs/2506.20701)
3. **S-ADT:** See `S-ADT.md` (ICLR 2026 submission)
4. **MLX:** [Documentation](https://ml-explore.github.io/mlx)

---

**Last Updated:** Dec 13, 2025, 12:15 PM  
**Current Phase:** 1 (Foundation Training - Stage 1)  
**Estimated Time to Working Training:** 10-20 minutes of fixes + 1-2 hours training  
**W&B Dashboard:** https://wandb.ai/nadhirvincenthassen/specdifftree

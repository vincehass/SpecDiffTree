# Sequential Implementation Plan: B → C → A

**Status:** Checkpoint directories exist but are empty  
**Approach:** Implement with dummy model, plug in real checkpoints later

---

## 🎯 Step B: Prepare Pre-trained Checkpoints

### Current Situation:
```
✅ Checkpoint structure exists:
   results/Llama_3_2_1B/OpenTSLMSP/
   ├── stage1_mcq/checkpoints/        (empty)
   ├── stage2_captioning/checkpoints/ (empty)
   ├── stage3_cot/checkpoints/        (empty)
   ├── stage4_sleep_cot/checkpoints/  (empty)
   └── stage5_ecg_cot/checkpoints/    (empty)

❌ No actual checkpoint files found
```

### Options:

**Option 1: Use Dummy Model (RECOMMENDED for now)**
- ✅ Implement DTS algorithm with mock OpenTSLM
- ✅ Test all components work correctly
- ✅ Plug in real checkpoints later when available
- ⏱️ Can start immediately

**Option 2: Train Stage 1**
- ❌ Would take ~50 hours on current setup
- ❌ Not needed for DTS implementation
- ⚠️ Can do later if needed

**Option 3: Download Pre-trained (if available)**
- ❓ Need to check if OpenTSLM provides pre-trained weights
- ❓ Check HuggingFace or Stanford BDHG repo

---

## 📋 Step C: Focus on Stage 1 (TSQA)

### Why Stage 1 First?
- ✅ Simplest task (Multiple-Choice QA)
- ✅ Clear reward function (accuracy)
- ✅ Easiest to validate
- ✅ Can extend to other stages once working

### Stage 1 Specifics:
```python
# TSQA Dataset
Task: Multiple-choice question answering on time series
Input: Time series + Question + 4 choices
Output: Predicted answer (A/B/C/D)
Reward: Accuracy (0 or 1)

# Model: OpenTSLMSP (Simple Projection)
Architecture:
  Time Series Encoder
  → Projection Layer
  → LLM (Llama 3.2 1B)
  → Answer prediction
```

---

## 🔧 Step A: Implement Full DTS for Stage 1

### Phase 1: OpenTSLM Wrapper (with dummy fallback)
```python
class OpenTSLMWrapper:
    def __init__(self, checkpoint_path=None, use_dummy=False):
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model = load_checkpoint(checkpoint_path)
        elif use_dummy:
            self.model = DummyOpenTSLM()  # For testing
        else:
            raise ValueError("No checkpoint or dummy model")
    
    def forward(self, x_t, t, context):
        """Diffusion step: x_t → x_{t-1}"""
        return self.model.denoise(x_t, t, context)
    
    def predict_answer(self, x_0, question, choices):
        """Final prediction on denoised x_0"""
        return self.model.qa_forward(x_0, question, choices)
```

### Phase 2: Stage 1 Reward Function
```python
def stage1_reward(x_0, question, choices, true_answer, context, gamma=1.0):
    """
    S-ADT reward for Stage 1 (TSQA)
    
    Args:
        x_0: Denoised time series
        question: QA question text
        choices: List of 4 answer choices
        true_answer: Ground truth answer
        context: Historical time series for spectral matching
        gamma: Spectral penalty weight
    
    Returns:
        r: Total reward (task + spectral)
    """
    # Task reward: Accuracy
    predicted = model.predict_answer(x_0, question, choices)
    r_task = 1.0 if predicted == true_answer else 0.0
    
    # Spectral penalty
    S_x0 = compute_psd(x_0)
    S_context = compute_psd(context)
    spectral_penalty = wasserstein_distance(S_x0, S_context)
    
    # Total reward
    r = r_task - gamma * spectral_penalty
    
    return r
```

### Phase 3: Main DTS Algorithm
```python
def run_dts_stage1(
    question,
    choices,
    true_answer,
    context,
    model,
    num_rollouts=200,
    temperature=0.1,
    gamma=1.0
):
    """
    Run DTS on Stage 1 (TSQA)
    
    Returns:
        best_x0: Best denoised time series
        best_answer: Predicted answer
        tree: Search tree with all trajectories
    """
    # Initialize from noise
    x_T = torch.randn_like(context)
    root = MCTSNode(x_T, t=model.num_timesteps)
    
    for rollout in range(num_rollouts):
        # 1. SELECTION
        node = root
        while not node.is_leaf():
            node = sample_child_boltzmann(node, temperature)
        
        # 2. EXPANSION
        if node.t > 0:
            x_prev = model.forward(node.x_t, node.t, context)
            child = MCTSNode(x_prev, t=node.t-1, parent=node)
            node.add_child(child)
            node = child
        
        # 3. ROLLOUT
        x_0 = rollout_to_x0(node, model, context)
        
        # 4. BACKUP
        r = stage1_reward(x_0, question, choices, true_answer, context, gamma)
        soft_bellman_backup(node, r, temperature)
    
    # Extract best
    tree = DTSTree(root)
    best_x0 = tree.extract_best_trajectory()[-1]
    best_answer = model.predict_answer(best_x0, question, choices)
    
    return best_x0, best_answer, tree
```

---

## 📊 Implementation Checklist

### ✅ Phase 1: Foundation (DONE)
- [x] MCTSNode class
- [x] Soft Bellman backup
- [x] Tree utilities

### 🚧 Phase 2: Model Wrapper (NEXT)
- [ ] Create `OpenTSLMWrapper` class
- [ ] Implement `DummyOpenTSLM` for testing
- [ ] Handle checkpoint loading (when available)
- [ ] Diffusion process interface

### 📝 Phase 3: Spectral Rewards
- [ ] Implement PSD computation
- [ ] Implement Wasserstein distance
- [ ] Create `stage1_reward` function
- [ ] Test spectral penalty calculation

### 🎯 Phase 4: Main DTS Algorithm
- [ ] Implement `DTSSampler` class
- [ ] Selection phase
- [ ] Expansion phase
- [ ] Rollout phase
- [ ] Backup phase (already have soft_bellman)

### 🧪 Phase 5: Testing
- [ ] Unit tests with dummy model
- [ ] Integration test on sample TSQA data
- [ ] Compare DTS vs baseline
- [ ] Measure NFE and performance

### 🔄 Phase 6: Real Checkpoints
- [ ] Obtain pre-trained Stage 1 checkpoint
- [ ] Load real model
- [ ] Re-run experiments
- [ ] Validate results match paper

---

## 🚀 Next Immediate Steps

1. **Create DummyOpenTSLM** (for testing)
   ```python
   class DummyOpenTSLM(nn.Module):
       """Dummy model that mimics OpenTSLM API for testing"""
       def denoise(self, x_t, t, context):
           # Simple Gaussian denoising
           return x_t + torch.randn_like(x_t) * 0.1
       
       def qa_forward(self, x_0, question, choices):
           # Random answer for testing
           return random.choice(['A', 'B', 'C', 'D'])
   ```

2. **Implement OpenTSLMWrapper**
   - Support both real checkpoints and dummy model
   - Unified API for DTS

3. **Implement Spectral Reward**
   - PSD computation via FFT
   - Spectral penalty calculation

4. **Implement DTSSampler**
   - Full 4-phase algorithm
   - Integrate all components

5. **Test End-to-End**
   - Run on sample TSQA data
   - Verify tree search works
   - Measure performance

---

## 💡 Advantages of This Approach

✅ **Can start immediately** (don't need to wait for checkpoints)  
✅ **Test DTS implementation** independently  
✅ **Validate algorithm** with dummy model  
✅ **Easy to plug in** real checkpoints later  
✅ **Modular design** makes debugging easier

---

## 🎯 Success Criteria

### For Dummy Model:
- [x] DTS algorithm runs without errors
- [ ] Tree builds correctly
- [ ] Values propagate via Soft Bellman
- [ ] Can extract best trajectory

### For Real Model (later):
- [ ] Accuracy improves over baseline
- [ ] Spectral fidelity preserved
- [ ] Matches S-ADT paper results
- [ ] 10x speedup with GFlowNet

---

**Current Status:** Ready to implement Phase 2 (Model Wrapper)  
**Next Action:** Create DummyOpenTSLM + OpenTSLMWrapper  
**Estimated Time:** 1-2 hours for full DTS implementation


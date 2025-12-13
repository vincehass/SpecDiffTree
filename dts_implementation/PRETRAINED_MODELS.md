# Pre-trained OpenTSLM Models on HuggingFace

**Source:** [StanfordBDHG/OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM/tree/main/demo/huggingface)

---

## 📦 Available Models (All 5 Stages)

| Stage | Task | HuggingFace Model ID | Architecture |
|-------|------|---------------------|--------------|
| **Stage 1** | TSQA (Multiple-Choice QA) | `OpenTSLM/llama-3.2-1b-tsqa-sp` | Simple Projection |
| **Stage 2** | M4 (Captioning) | `OpenTSLM/llama-3.2-1b-m4-sp` | Simple Projection |
| **Stage 3** | HAR CoT (Activity Recognition) | `OpenTSLM/llama-3.2-1b-har-sp` | Simple Projection |
| **Stage 4** | Sleep CoT (Sleep Stage) | `OpenTSLM/llama-3.2-1b-sleep-sp` | Simple Projection |
| **Stage 5** | ECG QA CoT (ECG Analysis) | `OpenTSLM/llama-3.2-1b-ecg-sp` | Simple Projection |

---

## 💡 Loading Models

```python
from model.llm.OpenTSLM import OpenTSLM
import torch

# Stage 1 (TSQA)
model = OpenTSLM.load_pretrained(
    "OpenTSLM/llama-3.2-1b-tsqa-sp",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Model has .generate() method
predictions = model.generate(batch, max_new_tokens=200)
```

---

## ⚠️ **IMPORTANT REALIZATION**

### OpenTSLM is NOT a Diffusion Model!

**OpenTSLM Architecture:**
```
Time Series → Encoder → Projection → LLM → Direct Prediction
```

It's a **direct prediction model** (like GPT), not a diffusion model!

**DTS (Diffusion Tree Sampling) expects:**
```
Noise x_T → Denoise → x_{T-1} → ... → x_0 (reverse diffusion)
```

---

## 🤔 **The Problem**

DTS is designed for **diffusion models** that have:
1. Forward process: x_0 → ... → x_T (add noise)
2. Reverse process: x_T → ... → x_0 (denoise)
3. Stochastic sampling at each timestep

**But OpenTSLM:**
- Does **direct prediction** (not iterative denoising)
- No noise schedule or timesteps
- Deterministic forward pass

---

## 🔧 **Two Possible Solutions**

### Option A: Adapt OpenTSLM to Diffusion Framework

**Idea:** Treat OpenTSLM's generation process as "denoising"

```python
# Pseudo-code
def opentslm_as_diffusion(x_t, t, context):
    """
    Treat LLM token generation as denoising steps
    - t=T: Start with random tokens
    - t=0: Fully generated answer
    """
    # Each timestep = one token generation
    if t == 0:
        return model.generate(context, max_new_tokens=1)
    else:
        # Partial generation
        return model.generate(context, max_new_tokens=T-t)
```

**Pros:**
- Can use DTS algorithm
- Tree search over generation paths
- Spectral rewards still apply

**Cons:**
- Not a true diffusion model
- May not match DTS paper assumptions

### Option B: Use Tree Search Directly (No Diffusion)

**Idea:** Tree search over LLM decoding, not diffusion

```python
# Beam search / MCTS over token generation
def tree_search_generation(model, context, reward_fn):
    """
    Build tree over possible token sequences
    - Each node = partial generation
    - Children = next token choices
    - Reward = final answer quality + spectral match
    """
    # Similar to DTS but for autoregressive generation
```

**Pros:**
- More natural for LLMs
- Similar to constrained decoding
- Still gets spectral regularization

**Cons:**
- Different from DTS paper
- May need new algorithm design

---

## 📊 **Which Approach for S-ADT + OpenTSLM?**

Looking at the S-ADT paper, they mention:

> "Given a frozen, pre-trained diffusion prior p_θ(x|c)"

But OpenTSLM is **not** a diffusion model. The paper assumes:
- Time series diffusion models
- Reverse denoising process
- Stochastic sampling

**Possible interpretations:**

1. **S-ADT is designed for actual diffusion models**, not OpenTSLM
   - Need to find/train a time series diffusion model
   - Then apply DTS + S-ADT

2. **Adapt the concept to OpenTSLM**
   - Use tree search over LLM generation
   - Apply spectral regularization
   - Different algorithm but same spirit

---

## 🎯 **Recommended Path Forward**

**I recommend asking the user to clarify:**

### Question 1: Are we using OpenTSLM or a diffusion model?

- **If OpenTSLM:** Need to adapt DTS to work with autoregressive LLMs
- **If Diffusion:** Need to find/train a time series diffusion model (not OpenTSLM)

### Question 2: What does the S-ADT paper actually use?

Looking at S-ADT.md:
```
"Given OpenTSLM's frozen prior p_θ(x|c), sample from:
π*(x) ∝ p_θ(x|c) exp(λ r(x))"
```

This suggests S-ADT **extends OpenTSLM**, but OpenTSLM is not a diffusion model in the traditional sense.

**Possible resolution:**
- OpenTSLM might have a diffusion-based component we haven't seen
- OR S-ADT adapts the diffusion concept to LLMs
- OR there's a misunderstanding in the methodology

---

## 🚨 **Action Needed**

Before implementing, we need to clarify:

1. **Is OpenTSLM a diffusion model?**
   - If yes: Where's the forward/reverse process?
   - If no: How does DTS apply?

2. **What does S-ADT actually do?**
   - Tree search over diffusion steps?
   - OR tree search over LLM generation?

3. **What should we implement?**
   - Pure DTS (needs diffusion model)
   - Adapted DTS for LLMs
   - Something else

---

**Let me check the OpenTSLM code to see if there's a diffusion component...**


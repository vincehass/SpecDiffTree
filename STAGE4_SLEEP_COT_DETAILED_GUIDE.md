<!--
This source file is part of the OpenTSLM open-source project

SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

# Stage 4: Sleep Stage Classification Chain-of-Thought - Detailed Mathematical Guide

**A Comprehensive Mathematical and Algorithmic Explanation of Stage 4 Training**

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset: SleepEDF CoT](#dataset-sleepedf-cot)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Evaluation](#evaluation)
7. [Implementation Algorithms](#implementation-algorithms)
8. [Comparison with Previous Stages](#comparison-with-previous-stages)

---

## Overview

### Stage 4 Purpose

Stage 4 (Sleep Stage Classification Chain-of-Thought) is the **sleep neuroscience reasoning stage** of the OpenTSLM curriculum learning pipeline. Building on Stages 1-3, it teaches the model:

1. **Neurophysiological reasoning**: How to analyze EEG signals for sleep staging
2. **Chain-of-thought explanation**: How to articulate step-by-step sleep stage classification reasoning
3. **Clinical sleep interpretation**: How to connect EEG patterns to sleep stages
4. **Structured reasoning**: How to provide rationale before final sleep stage classification

**Why Stage 4 After Stages 1-3?**
- Stage 1 taught basic time series understanding (classification/MCQ)
- Stage 2 extended to detailed text generation (captioning)
- Stage 3 introduced medical chain-of-thought reasoning (ECG analysis)
- Stage 4 applies CoT reasoning to neurophysiological signals (sleep EEG)
- Extends medical reasoning to sleep medicine domain

**Key Statistics:**
- **Dataset**: SleepEDF Chain-of-Thought (custom-created with GPT-4o)
- **Samples**: Variable across train/val/test splits
- **Task Type**: Chain-of-Thought Sleep Stage Classification
- **Time Series**: Single-channel EEG signals (30-second epochs)
- **EEG Duration**: 30 seconds (standard sleep staging window)
- **Sampling Rate**: 100 Hz
- **Samples per Epoch**: ~3,000 (30 seconds × 100 Hz)
- **Training Epochs**: 60 (with early stopping)
- **Sleep Stages**: 6 classes (Wake, N1, N2, N3, REM, Movement)
- **Metric**: Test Loss and Perplexity (reasoning quality)

---

## Dataset: SleepEDF CoT

### Dataset Description

The SleepEDF Chain-of-Thought dataset combines polysomnographic EEG recordings with AI-generated chain-of-thought reasoning for sleep stage classification. Each sample consists of:

**Input:**
- A single-channel EEG recording: $\mathbf{x} \in \mathbb{R}^L$ where $L \approx 3000$
- Sleep staging context: $C$ (30-second epoch analysis)

**Output:**
- Chain-of-thought reasoning: $R$ (detailed step-by-step sleep stage analysis)
- Final sleep stage classification: $A \in \{\text{Wake}, \text{N1}, \text{N2}, \text{N3}, \text{REM}, \text{Movement}\}$

### Data Format

Each sample in the SleepEDF CoT dataset has the following structure:

```json
{
  "time_series": [[x1, x2, ..., x3000]],    // Single-channel EEG (nested list)
  "label": "Non-REM stage 2",                // Ground truth sleep stage
  "rationale": "Looking at the EEG signal..."  // CoT reasoning + answer
}
```

### Mathematical Representation

Let $\mathcal{D}_{\text{Sleep-CoT}} = \{(\mathbf{x}_i, R_i, A_i)\}_{i=1}^N$ be the dataset with $N$ samples, where:

- $\mathbf{x}_i \in \mathbb{R}^{L}$: Single-channel EEG recording ($L \approx 3000$ samples)
- $R_i \in \mathcal{V}^*$: Chain-of-thought reasoning (detailed analysis)
- $A_i \in \mathcal{A}$: Final sleep stage classification
- $\mathcal{A} = \{\text{Wake}, \text{Non-REM stage 1}, \text{Non-REM stage 2}, \text{Non-REM stage 3}, \text{REM sleep}, \text{Movement}\}$

### Data Splits

The dataset is split into three subsets using stratified sampling:

$$
\mathcal{D}_{\text{Sleep-CoT}} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}
$$

**Split proportions:**
- Training: 80% of data (stratified by sleep stage)
- Validation: 10% of data (stratified by sleep stage)
- Test: 10% of data (stratified by sleep stage)

**Splitting Algorithm:**
```
1. Load SleepEDF CoT CSV data (sleep_cot.csv)
2. Parse each row:
   - time_series: Parse nested list [[...]] to extract EEG data
   - label: Sleep stage classification
   - rationale: Chain-of-thought reasoning + answer
3. Perform stratified split by label:
   - Ensures all sleep stages represented in each split
   - Preserves class distribution
4. Create HuggingFace Dataset objects
5. Return D_train, D_val, D_test
```

### EEG Signal Characteristics

**EEG Recording Properties:**

| Property | Value | Description |
|----------|-------|-------------|
| **Channels** | 1 (single-channel EEG) | Frontal EEG electrode |
| **Sampling Rate** | 100 Hz | Standard PSG sampling |
| **Duration** | 30 seconds | Standard sleep staging epoch |
| **Samples per Epoch** | 3,000 | 30s × 100 Hz |
| **Signal Type** | Raw EEG voltage | Microvolts (μV) |

**Sleep Stage Characteristics:**

Each sleep stage has distinct EEG patterns:

| Sleep Stage | EEG Characteristics | Frequency Bands |
|-------------|---------------------|-----------------|
| **Wake** | Low amplitude, high frequency | Alpha (8-13 Hz), Beta (13-30 Hz) |
| **N1** | Theta activity, vertex sharp waves | Theta (4-8 Hz) |
| **N2** | Sleep spindles, K-complexes | Sigma (12-16 Hz) |
| **N3** | High amplitude slow waves | Delta (0.5-4 Hz) |
| **REM** | Low amplitude mixed frequency | Similar to wake, theta prominent |
| **Movement** | Artifacts, high amplitude transients | Broadband noise |

### Chain-of-Thought Reasoning Structure

**Typical CoT Reasoning Contains:**

1. **Signal Overview**: General observation of EEG characteristics
2. **Amplitude Analysis**: Assessment of signal amplitude (high/low voltage)
3. **Frequency Content**: Identification of dominant frequency bands
4. **Pattern Recognition**: Detection of sleep-specific waveforms
   - Sleep spindles (N2)
   - K-complexes (N2)
   - Slow waves (N3)
   - Alpha waves (Wake)
   - Theta activity (N1, REM)
5. **Diagnostic Reasoning**: Step-by-step logic toward sleep stage
6. **Final Answer**: Clear sleep stage classification

**Example CoT Reasoning:**

```
The EEG signal shows moderate to high amplitude activity with a mixed frequency 
pattern. Looking at the overall morphology, I observe several characteristic 
features. First, the amplitude is relatively low compared to deep sleep stages, 
suggesting lighter sleep or wakefulness. The frequency content appears to contain 
significant theta activity (4-8 Hz), which is visible as rhythmic oscillations. 
Additionally, there are brief bursts of higher frequency activity that could 
represent sleep spindles - these are 12-16 Hz oscillations lasting 0.5-1 seconds, 
which are hallmark features of Stage 2 sleep. I can also identify what appear to 
be K-complexes, which are sudden sharp negative deflections followed by positive 
components. These two features together - sleep spindles and K-complexes - are 
the defining characteristics of Non-REM Stage 2 sleep. The absence of dominant 
slow-wave activity rules out Stage 3, while the presence of these specific 
patterns rules out Stage 1, REM, or wakefulness. Answer: Non-REM stage 2
```

### Data Preprocessing

**1. EEG Signal Loading:**

Each EEG epoch is loaded from the CSV file:

```python
import ast
import numpy as np

# Parse nested list structure
time_series_str = row['time_series']  # String: "[[x1, x2, ..., x3000]]"
parsed = ast.literal_eval(time_series_str)  # Parse to Python list

# Extract inner list (the actual time series)
if isinstance(parsed, list) and len(parsed) == 1:
    time_series = np.array(parsed[0], dtype=np.float32)  # Shape: [3000]
```

**2. Per-Epoch Normalization:**

For each EEG epoch $\mathbf{x} = [x_1, \ldots, x_L]$, apply z-score normalization:

$$
\mu = \frac{1}{L} \sum_{t=1}^{L} x_t
$$

$$
\sigma = \sqrt{\frac{1}{L} \sum_{t=1}^{L} (x_t - \mu)^2}
$$

$$
\tilde{x}_t = \frac{x_t - \mu}{\max(\sigma, \epsilon)}, \quad \epsilon = 10^{-6}
$$

The normalized epoch is: $\tilde{\mathbf{x}} = [\tilde{x}_1, \ldots, \tilde{x}_L]$

**Rationale**: Z-score normalization removes baseline shifts and amplitude variations while preserving frequency content and morphology.

**3. Padding for Batch Processing:**

EEG epochs within a batch may have slightly different lengths. Pad to maximum length divisible by patch size $P = 4$:

$$
L_{\text{max}} = \left\lceil \frac{\max(L_1, \ldots, L_B)}{P} \right\rceil \times P
$$

For epoch $i$ with $L_i < L_{\text{max}}$:

$$
\mathbf{x}_i^{\text{padded}} = [\tilde{x}_{i,1}, \ldots, \tilde{x}_{i,L_i}, \underbrace{0, \ldots, 0}_{L_{\text{max}} - L_i}]
$$

**4. Prompt Construction:**

Each sample is formatted as a sleep staging analysis prompt:

$$
\text{Prompt}_i = P_{\text{role}} \oplus P_{\text{instructions}}
$$

where:
- $P_{\text{role}} = \text{"You are given a 30-second EEG time series segment..."}$
- $P_{\text{instructions}} = \text{"Instructions: Analyze objectively, reason methodically..."}$
- $\oplus$ denotes string concatenation

**Pre-prompt (Role and Context):**
```
You are given a 30-second EEG time series segment. Your task is to classify 
the sleep stage based on analysis of the data.

Instructions:
- Analyze the data objectively without presuming a particular label.
- Reason carefully and methodically about what the signal patterns suggest 
  regarding sleep stage.
- Write your reasoning as a single, coherent paragraph. Do not use bullet 
  points, lists, or section headers.
- Only reveal the correct class at the very end.
- Never state that you are uncertain or unable to classify the data. You 
  must always provide a rationale and a final answer.
```

**Post-prompt (Output Format Instructions):**
```
Possible sleep stages are:
Wake, Non-REM stage 1, Non-REM stage 2, Non-REM stage 3, REM sleep, Movement

- Please now write your rationale. Make sure that your last word is the 
  answer. You MUST end your response with "Answer: "
```

**5. Time Series Text Description:**

Before presenting the EEG data, include descriptive statistics:

```python
mean = float(np.mean(series))
std = float(np.std(series))
text = f"The following is the EEG time series, it has mean {mean:.4f} and std {std:.4f}:"
```

This provides the model with explicit statistical context about the signal.

**6. Chain-of-Thought Target Construction:**

The target output is the complete rationale (which includes the answer at the end):

$$
\text{Target}_i = R_i
$$

where $R_i$ already contains the reasoning and concludes with "Answer: [sleep_stage]".

---

## Mathematical Formulation

### Problem Formulation

Given:
- Single-channel EEG recording: $\mathbf{x} \in \mathbb{R}^L$ where $L \approx 3000$
- Sleep staging context: $C$ (30-second epoch analysis task)

Goal:
- Generate chain-of-thought reasoning: $\hat{R} = [\hat{r}_1, \hat{r}_2, \ldots, \hat{r}_K]$
- Generate final sleep stage: $\hat{A} \in \mathcal{A}$

This is a **conditional text generation with structured output** problem for sleep stage classification.

### Model Function

The model learns a conditional probability distribution over reasoning + answer:

$$
P_\theta(R | \mathbf{x}, C) = \prod_{k=1}^{K} P_\theta(r_k | \mathbf{x}, C, r_{<k})
$$

where $R = [r_1, \ldots, r_{K_R}, a_1, \ldots, a_{K_A}]$ is the complete output sequence (reasoning + answer).

The model function is:

$$
f_\theta: (\mathbb{R}^{L}, \mathcal{V}^*) \rightarrow \Delta^{|\mathcal{V}|}
$$

where $\Delta^{|\mathcal{V}|}$ is the probability simplex over the vocabulary.

### Loss Function

For chain-of-thought sleep staging, we use **causal language modeling loss** (cross-entropy):

$$
\mathcal{L}(\theta; \mathbf{x}, C, R) = -\frac{1}{K} \sum_{k=1}^{K} \log P_\theta(r_k | \mathbf{x}, C, r_{<k})
$$

Expanded form:

$$
\mathcal{L}(\theta) = -\frac{1}{K} \left[\sum_{k=1}^{K_R} \log P_\theta(r_k | \mathbf{x}, C, r_{<k}) + \sum_{m=1}^{K_A} \log P_\theta(a_m | \mathbf{x}, C, R, a_{<m})\right]
$$

This ensures the model learns to:
1. Generate coherent sleep staging reasoning given EEG and context
2. Produce correct sleep stage conditioned on the reasoning

### Total Training Objective

Over the entire training dataset $\mathcal{D}_{\text{train}}$:

$$
\mathcal{L}_{\text{total}}(\theta) = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(\mathbf{x}_i, C_i, R_i) \in \mathcal{D}_{\text{train}}} \mathcal{L}(\theta; \mathbf{x}_i, C_i, R_i)
$$

### Optimization Objective

$$
\theta^* = \arg\min_{\theta} \mathcal{L}_{\text{total}}(\theta) + \lambda \|\theta\|_2^2
$$

where $\lambda = 10^{-2}$ is the weight decay (L2 regularization) coefficient.

### Perplexity Metric

Perplexity measures the quality of chain-of-thought reasoning:

$$
\text{Perplexity}(\mathcal{D}) = \exp\left(\mathcal{L}_{\text{total}}(\theta)\right)
$$

Lower perplexity indicates better reasoning quality:
- Lower perplexity → More confident and coherent sleep staging reasoning
- Higher perplexity → Uncertain or incoherent reasoning

---

## Model Architecture

The architecture for Stage 4 is **identical** to Stages 1-3, but with sleep-specific EEG data and sleep staging chain-of-thought reasoning. The model consists of three main components:

### 1. Time Series Encoder (Single-Channel EEG)

**Architecture**: Transformer-CNN Encoder (same as Stages 1-3)

**Input**: $\mathbf{x} \in \mathbb{R}^{B \times L}$ (batch of single-channel EEG epochs)

**Processing**: Each EEG epoch is encoded independently

**Output**: $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times N \times d_{\text{enc}}}$ (encoded features)

where:
- $B$: Batch size
- $L$: EEG samples per epoch (padded, typically ~3000)
- $N = L / P$: Number of patches
- $P = 4$: Patch size
- $d_{\text{enc}} = 128$: Encoder embedding dimension

#### Single-Channel Encoding Process

**Per-Epoch Encoding:**

For each EEG epoch $\mathbf{x} \in \mathbb{R}^{L}$:

1. **Patch Embedding**: 
   $$
   \mathbf{P} = \text{Conv1D}(\mathbf{x}) \in \mathbb{R}^{N \times d_{\text{enc}}}
   $$

2. **Positional Encoding**: 
   $$
   \mathbf{P}' = \mathbf{P} + \mathbf{E}_{\text{pos}}[:N, :] \in \mathbb{R}^{N \times d_{\text{enc}}}
   $$

3. **Transformer Encoding**: 
   $$
   \mathbf{H}_{\text{enc}} = \text{TransformerEncoder}_{L=6}(\mathbf{P}')
   $$

**Total Sequence Length:**

$$
N = \frac{L}{4} \approx \frac{3000}{4} = 750 \text{ tokens}
$$

This is the time series token representation length.

### 2. Projector

**Architecture**: MLP with LayerNorm and GELU (same as Stages 1-3)

**Input**: $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times N \times d_{\text{enc}}}$

**Output**: $\mathbf{H}_{\text{proj}} \in \mathbb{R}^{B \times N \times d_{\text{llm}}}$

**Mathematical Formulation:**

$$
\mathbf{H}_{\text{proj}} = \text{Dropout}(\text{GELU}(\text{Linear}(\text{LayerNorm}(\mathbf{H}_{\text{enc}}))))
$$

### 3. Large Language Model (LLM)

**Architecture**: Pre-trained causal LM (Llama/Gemma, same as Stages 1-3)

**Input**: Combined sequence of projected EEG embeddings and text tokens

**Output**: Probability distribution over vocabulary for next token prediction

#### Input Sequence Construction for Sleep CoT

For a sample $(\mathbf{x}, C, R)$:

**1. Text Tokenization:**

$$
C_{\text{tokens}} = \text{Tokenize}(P_{\text{role}} + P_{\text{instructions}} + \text{EEG description}) = [c_1, \ldots, c_M]
$$

$$
R_{\text{tokens}} = \text{Tokenize}(R) = [r_1, r_2, \ldots, r_K]
$$

where $R$ includes both reasoning and the final answer.

**2. Text Embedding:**

$$
\mathbf{H}_C = \text{Embed}_{\text{LLM}}(C_{\text{tokens}}) \in \mathbb{R}^{M \times d_{\text{llm}}}
$$

$$
\mathbf{H}_R = \text{Embed}_{\text{LLM}}(R_{\text{tokens}}) \in \mathbb{R}^{K \times d_{\text{llm}}}
$$

**3. Sequence Concatenation:**

$$
\mathbf{H}_{\text{input}} = [\mathbf{H}_C; \mathbf{H}_{\text{proj}}; \mathbf{H}_R] \in \mathbb{R}^{(M+N+K) \times d_{\text{llm}}}
$$

**Total sequence length**: $T = M + N + K$ (typically 1,000-1,500 tokens)

Note: Sequence length components:
- Context/prompt ($M$): ~150-250 tokens
- EEG encoding ($N$): ~750 tokens
- Reasoning + answer ($K$): ~100-400 tokens

#### Autoregressive Chain-of-Thought Generation

During training, the model learns:

$$
P_\theta(r_k | \mathbf{x}, C, r_{<k}) = \text{Softmax}(\text{LLM}([\mathbf{H}_C; \mathbf{H}_{\text{proj}}; \mathbf{H}_{r_{<k}}]))_k
$$

### Model Initialization for Stage 4

**Key Difference from Stages 1-3:**

Stage 4 **initializes** from the best Stage 3 checkpoint:

$$
\theta_{\text{Stage4}}^{(0)} = \theta_{\text{Stage3}}^*
$$

where $\theta_{\text{Stage3}}^*$ is the best model from Stage 3 training (ECG CoT).

This provides:
1. **Pre-trained encoder**: Already understands time series patterns from Stages 1-3
2. **Pre-trained projector**: Already maps time series to LLM space effectively
3. **Pre-trained LLM**: Already capable of medical chain-of-thought reasoning
4. **Warm start**: Fast convergence for sleep-specific reasoning task

**Curriculum Learning Progression:**

$$
\text{Stage 1 (MCQ)} \rightarrow \text{Stage 2 (Captioning)} \rightarrow \text{Stage 3 (ECG CoT)} \rightarrow \text{Stage 4 (Sleep CoT)}
$$

Each stage builds on the previous:
- Stage 1: Basic understanding
- Stage 2: Detailed generation
- Stage 3: Medical reasoning (cardiac)
- Stage 4: Medical reasoning (neurophysiological)

### Parameter Count

**Trainable Parameters (Stage 4):**

| Component | Parameters | Trainable | Notes |
|-----------|-----------|-----------|-------|
| **Encoder** | ~5M | ✅ Yes | Fine-tuned from Stage 3 |
| **Projector** | ~260K | ✅ Yes | Fine-tuned from Stage 3 |
| **LLM (Llama-3.2-1B)** | ~1.2B | ❌ No (frozen) | Or ✅ with LoRA |
| **LoRA (if enabled)** | ~2-4M | ✅ Yes | Optional LLM fine-tuning |
| **Total Trainable** | ~5.3M (or ~9M with LoRA) | | |

---

## Training Process

### Training Configuration

**Hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 60 | Maximum training epochs |
| Batch Size | 4 | Samples per batch |
| Learning Rate (Encoder) | $2 \times 10^{-4}$ | LR for encoder |
| Learning Rate (Projector) | $1 \times 10^{-4}$ | LR for projector |
| Learning Rate (LoRA) | $2 \times 10^{-4}$ | LR for LoRA adapters (if enabled) |
| Weight Decay | $1 \times 10^{-2}$ | L2 regularization |
| Gradient Clipping | 1.0 | Max gradient norm |
| Warmup Fraction | 0.03 | Fraction of steps for LR warmup |
| Early Stopping Patience | 5 | Epochs without improvement to stop |
| Patch Size | 4 | Time series patch size |

**Note**: Stage 4 requires 60 epochs (same as Stage 3) because chain-of-thought reasoning is a complex task requiring extended training.

### Curriculum Learning Connection

Stage 4 builds on Stage 3:

$$
\theta_{\text{Stage4}}^{(0)} \leftarrow \text{BestCheckpoint}(\text{Stage3})
$$

This enables:
1. **Knowledge transfer**: Medical reasoning capabilities transfer
2. **Faster convergence**: Model starts with strong foundation
3. **Better performance**: Curriculum approach critical for complex reasoning
4. **Domain adaptation**: Fine-tune existing medical reasoning for sleep staging

### Learning Rate Schedule

**Warmup + Linear Decay:**

Let $T_{\text{total}}$ be the total number of training steps, and $T_{\text{warmup}} = 0.03 \times T_{\text{total}}$.

For step $t$:

$$
\eta(t) = \begin{cases}
\eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \text{ (warmup)} \\
\eta_{\max} \cdot \frac{T_{\text{total}} - t}{T_{\text{total}} - T_{\text{warmup}}} & \text{if } t \geq T_{\text{warmup}} \text{ (decay)}
\end{cases}
$$

### Optimization Algorithm

**AdamW Optimizer** (same as Stages 1-3):

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t
$$

$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2
$$

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta_t \left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \theta_{t-1}\right)
$$

where:
- $\mathbf{g}_t = \nabla_\theta \mathcal{L}(\theta_{t-1})$: Gradient
- $\beta_1 = 0.9$, $\beta_2 = 0.999$: Momentum parameters
- $\epsilon = 10^{-8}$: Numerical stability
- $\lambda = 0.01$: Weight decay

### Training Loop

**Algorithm: Stage 4 Training**

```
Input: Training data D_train, validation data D_val, test data D_test
       Stage 3 best checkpoint θ_stage3
Output: Trained model parameters θ*

1. Load Stage 3 checkpoint:
   encoder_params ← θ_stage3.encoder
   projector_params ← θ_stage3.projector
   lora_params ← θ_stage3.lora (if enabled)
   
2. Initialize optimizer groups:
   optimizer_groups = [
       {params: encoder_params, lr: 2e-4, weight_decay: 1e-2},
       {params: projector_params, lr: 1e-4, weight_decay: 1e-2},
       {params: lora_params, lr: 2e-4, weight_decay: 1e-2}  # if LoRA enabled
     ]
   optimizer ← AdamW(optimizer_groups)

3. Initialize scheduler:
   total_steps ← num_epochs × len(D_train) / batch_size
   warmup_steps ← 0.03 × total_steps
   scheduler ← LinearWarmupScheduler(optimizer, warmup_steps, total_steps)

4. Training loop:
   best_val_loss ← ∞
   epochs_no_improve ← 0
   
   for epoch = 1 to 60:
       // Training phase
       model.train()
       train_loss ← 0
       
       for batch in DataLoader(D_train, batch_size=4, shuffle=True):
           // Forward pass
           X_batch, C_batch, R_batch ← batch
           loss ← compute_cot_loss(X_batch, C_batch, R_batch)
           
           // Backward pass
           optimizer.zero_grad()
           loss.backward()
           
           // Gradient clipping
           clip_grad_norm_(model.parameters(), max_norm=1.0)
           
           // Optimizer step
           optimizer.step()
           scheduler.step()
           
           train_loss ← train_loss + loss.item()
       
       avg_train_loss ← train_loss / len(D_train)
       
       // Validation phase
       model.eval()
       val_loss ← 0
       
       with torch.no_grad():
           for batch in DataLoader(D_val, batch_size=4):
               X_batch, C_batch, R_batch ← batch
               loss ← compute_cot_loss(X_batch, C_batch, R_batch)
               val_loss ← val_loss + loss.item()
       
       avg_val_loss ← val_loss / len(D_val)
       perplexity ← exp(avg_val_loss)
       
       // Early stopping check
       if avg_val_loss < best_val_loss - 1e-4:
           best_val_loss ← avg_val_loss
           epochs_no_improve ← 0
           save_checkpoint(model, optimizer, epoch)
           print("✓ New best model saved")
       else:
           epochs_no_improve ← epochs_no_improve + 1
           if epochs_no_improve ≥ 5:
               print("Early stopping triggered")
               break
       
       print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, perplexity={perplexity:.2f}")
   
5. Load best checkpoint:
   load_checkpoint(model)

6. Evaluate on test set:
   test_metrics ← evaluate_cot_reasoning(model, D_test)
   
7. Return model, test_metrics
```

### Loss Computation for Sleep CoT

For a single sample $(\mathbf{x}, C, R)$:

**Step 1: Load and normalize EEG**
```
# Load EEG signal
x ← parse_time_series(sample['time_series'])  # Shape: [3000]

# Normalize
mean ← np.mean(x)
std ← np.std(x)
x_norm ← (x - mean) / max(std, 1e-6)          # Shape: [3000]
```

**Step 2: Encode EEG**
```
H_enc ← Encoder(x_norm)                        # Shape: [N, d_enc]
                                               # where N ≈ 750
```

**Step 3: Project to LLM space**
```
H_proj ← Projector(H_enc)                      # Shape: [N, d_llm]
```

**Step 4: Prepare text**
```
# Create prompt with EEG description
prompt ← create_sleep_prompt(mean, std)        # Includes role + instructions
C_tokens ← Tokenize(prompt)                    # Shape: [M]

# Chain-of-thought reasoning (includes answer)
R_tokens ← Tokenize(rationale)                 # Shape: [K]
```

**Step 5: Embed and concatenate**
```
H_C ← Embed(C_tokens)                          # Shape: [M, d_llm]
H_R ← Embed(R_tokens)                          # Shape: [K, d_llm]

H_input ← concat([H_C, H_proj, H_R], dim=0)   # Shape: [M+N+K, d_llm]
```

**Step 6: LLM forward pass**
```
logits ← LLM(H_input)                          # Shape: [M+N+K, |V|]
```

**Step 7: Extract CoT logits and compute loss**
```
# CoT tokens start after context and EEG embeddings
cot_start_idx ← M + N
cot_logits ← logits[cot_start_idx : cot_start_idx+K-1, :]  # Shape: [K-1, |V|]

# Target tokens (shifted by 1 for next-token prediction)
target_tokens ← R_tokens[1:]                   # Shape: [K-1]

# Compute cross-entropy loss
loss ← CrossEntropyLoss(cot_logits, target_tokens)
```

Mathematically:
$$
\mathcal{L} = -\frac{1}{K-1} \sum_{k=1}^{K-1} \log P_\theta(r_{k+1} | \mathbf{x}, C, r_{\leq k})
$$

where $R = [r_1, \ldots, r_K]$ is the complete CoT sequence (reasoning + answer).

---

## Evaluation

### Evaluation Metrics

**Primary Metrics:**

1. **Test Loss** (Cross-Entropy):
   $$
   \mathcal{L}_{\text{test}} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{(\mathbf{x}, C, R) \in \mathcal{D}_{\text{test}}} \mathcal{L}(\theta; \mathbf{x}, C, R)
   $$

2. **Perplexity**:
   $$
   \text{PPL} = \exp(\mathcal{L}_{\text{test}})
   $$

**Interpretation:**
- **Lower test loss** = Better reasoning generation quality
- **Lower perplexity** = More confident and coherent CoT reasoning
- Typical good perplexity: 8-25 for sleep staging reasoning tasks

**Secondary Metrics (Post-Processing):**

3. **Sleep Stage Accuracy**: Extract final sleep stage and compare with ground truth

   For each prediction:
   ```python
   predicted_stage = extract_sleep_stage(generated_text)  # After "Answer: "
   ground_truth_stage = sample['label']
   accuracy = predicted_stage == ground_truth_stage
   ```
   
   $$
   \text{Accuracy} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{i=1}^{|\mathcal{D}_{\text{test}}|} \mathbb{1}[\hat{A}_i = A_i]
   $$

4. **Per-Class F1 Score**: Classification metrics for each sleep stage

   For each sleep stage $s \in \mathcal{A}$:
   $$
   F1_s = \frac{2 \cdot P_s \cdot R_s}{P_s + R_s}
   $$
   
   where $P_s$ is precision and $R_s$ is recall for sleep stage $s$.

5. **Macro-F1 Score**: Average F1 across all sleep stages

   $$
   \text{Macro-F1} = \frac{1}{|\mathcal{A}|} \sum_{s \in \mathcal{A}} F1_s
   $$

6. **Cohen's Kappa**: Agreement metric accounting for chance

   $$
   \kappa = \frac{p_o - p_e}{1 - p_e}
   $$
   
   where $p_o$ is observed agreement and $p_e$ is expected agreement by chance.

### Inference Process

**Chain-of-Thought Sleep Staging Algorithm:**

For a test sample $(\mathbf{x}_{\text{test}}, C_{\text{test}})$:

```
1. Preprocess EEG:
   x ← parse_time_series(sample['time_series'])
   mean ← np.mean(x)
   std ← np.std(x)
   x_norm ← (x - mean) / max(std, 1e-6)
   
2. Encode EEG:
   H_enc ← Encoder(x_norm)
   H_proj ← Projector(H_enc)
   
3. Prepare prompt:
   prompt ← create_sleep_prompt(mean, std)
   H_C ← Embed(Tokenize(prompt))
   H_input ← concat([H_C, H_proj], dim=0)
   
4. Generate CoT reasoning autoregressively:
   generated_tokens ← []
   current_input ← H_input
   
   for step = 1 to max_new_tokens (e.g., 400):
       // Forward pass
       logits ← LLM(current_input)
       
       // Get next token probability
       next_token_logits ← logits[-1, :]
       next_token_prob ← Softmax(next_token_logits / temperature)
       
       // Greedy decode (for evaluation)
       next_token ← argmax(next_token_prob)
       
       // Check for end of sequence
       if next_token == EOS_token:
           break
       
       // Append to generated sequence
       generated_tokens.append(next_token)
       
       // Update input for next iteration
       next_token_emb ← Embed(next_token)
       current_input ← concat([current_input, next_token_emb], dim=0)
   
5. Decode tokens to text:
   cot_text ← Detokenize(generated_tokens)
   
6. Extract reasoning and sleep stage:
   if "Answer:" in cot_text:
       reasoning ← cot_text.split("Answer:")[0].strip()
       sleep_stage ← cot_text.split("Answer:")[-1].strip()
   else:
       reasoning ← cot_text
       sleep_stage ← "Wake"  # Default if no answer found
   
7. Normalize sleep stage:
   sleep_stage ← normalize_sleep_stage(sleep_stage)
   
8. Return reasoning, sleep_stage
```

**Temperature and Sampling:**

For CoT reasoning, temperature can be used to control creativity:

$$
P'(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

where:
- $T = 1.0$: Standard sampling (default for evaluation)
- $T < 1.0$: More conservative, deterministic reasoning
- $T > 1.0$: More diverse, creative reasoning

### Evaluation Algorithm

```
Function: evaluate_stage4(model, D_test)

Input: Trained model, test dataset D_test
Output: Test loss, perplexity, accuracy, confusion matrix, and predictions

1. Initialize:
   total_loss ← 0
   total_tokens ← 0
   predictions ← []
   ground_truths ← []
   correct_answers ← 0
   confusion_matrix ← zeros(6, 6)  # 6 sleep stages
   
2. Set model to evaluation mode:
   model.eval()
   
3. Disable gradient computation:
   with torch.no_grad():
       
       for (x, C, R, label) in D_test:
           // Compute loss
           loss ← compute_cot_loss(x, C, R)
           K ← len(Tokenize(R))
           
           total_loss ← total_loss + loss.item() × K
           total_tokens ← total_tokens + K
           
           // Generate prediction
           reasoning_pred, stage_pred ← model.generate_cot(x, C, max_new_tokens=400)
           
           // Extract ground truth stage
           stage_true ← label
           
           // Store predictions
           predictions.append({
               "generated_stage": stage_pred,
               "target_stage": stage_true,
               "reasoning": reasoning_pred,
               "target_reasoning": R
           })
           ground_truths.append(stage_true)
           
           // Check accuracy (after normalization)
           if normalize_sleep_stage(stage_pred) == normalize_sleep_stage(stage_true):
               correct_answers ← correct_answers + 1
           
           // Update confusion matrix
           i ← sleep_stage_to_index(stage_true)
           j ← sleep_stage_to_index(stage_pred)
           confusion_matrix[i, j] ← confusion_matrix[i, j] + 1
   
4. Compute metrics:
   avg_test_loss ← total_loss / total_tokens
   perplexity ← exp(avg_test_loss)
   accuracy ← correct_answers / len(D_test)
   
5. Compute per-class metrics:
   per_class_f1 ← compute_per_class_f1(confusion_matrix)
   macro_f1 ← mean(per_class_f1)
   cohens_kappa ← compute_cohens_kappa(confusion_matrix)
   
6. Save results:
   results ← {
       "test_loss": avg_test_loss,
       "perplexity": perplexity,
       "accuracy": accuracy,
       "macro_f1": macro_f1,
       "cohens_kappa": cohens_kappa,
       "total_samples": len(D_test),
       "total_tokens": total_tokens,
       "per_class_f1": per_class_f1,
       "confusion_matrix": confusion_matrix
   }
   
   save_json(results, "stage4_sleep_cot/results/metrics.json")
   save_jsonl(predictions, "stage4_sleep_cot/results/test_predictions.jsonl")
   
7. Print summary:
   print(f"Test Loss: {avg_test_loss:.4f}")
   print(f"Perplexity: {perplexity:.2f}")
   print(f"Accuracy: {accuracy:.4f}")
   print(f"Macro-F1: {macro_f1:.4f}")
   print(f"Cohen's Kappa: {cohens_kappa:.4f}")
   print(f"Avg tokens per sample: {total_tokens/len(D_test):.1f}")
   
   // Print per-class F1 scores
   for stage, f1 in zip(sleep_stages, per_class_f1):
       print(f"{stage} F1: {f1:.4f}")
   
8. Return results
```

### Answer Extraction and Normalization

**Extraction Function:**

```python
def extract_sleep_stage(text: str) -> str:
    """Extract the final sleep stage from CoT reasoning."""
    if "Answer:" in text:
        stage = text.split("Answer:")[-1].strip()
    else:
        return text.strip()
    
    # Remove end-of-text tokens
    stage = re.sub(r'<\|.*?\|>|<eos>$', '', stage).strip()
    
    # Remove trailing periods
    stage = re.sub(r'\.$', '', stage).strip()
    
    return stage
```

**Normalization Function:**

```python
def normalize_sleep_stage(stage: str) -> str:
    """Normalize sleep stage for comparison."""
    # Convert to lowercase
    stage = stage.lower()
    
    # Remove punctuation
    stage = stage.rstrip('.,!?;:')
    
    # Strip whitespace
    stage = stage.strip()
    
    # Map common variations to standard names
    stage_mappings = {
        "wake": "wake",
        "wakefulness": "wake",
        "awake": "wake",
        "n1": "non-rem stage 1",
        "stage 1": "non-rem stage 1",
        "non-rem 1": "non-rem stage 1",
        "n2": "non-rem stage 2",
        "stage 2": "non-rem stage 2",
        "non-rem 2": "non-rem stage 2",
        "n3": "non-rem stage 3",
        "stage 3": "non-rem stage 3",
        "non-rem 3": "non-rem stage 3",
        "deep sleep": "non-rem stage 3",
        "slow wave sleep": "non-rem stage 3",
        "rem": "rem sleep",
        "rapid eye movement": "rem sleep",
        "movement": "movement",
        "artifact": "movement"
    }
    
    return stage_mappings.get(stage, stage)
```

### Expected Performance

**Typical Stage 4 Results:**

| Model | Test Loss | Perplexity | Accuracy | Macro-F1 | Cohen's κ | Training Time | GPU Memory |
|-------|-----------|------------|----------|----------|-----------|---------------|------------|
| **OpenTSLMSP** (Llama-3.2-1B) | 1.8-2.3 | 6-10 | 0.75-0.85 | 0.70-0.80 | 0.65-0.75 | ~8-12 hours | ~12-16GB |
| **OpenTSLMSP** (Gemma-3-270m) | 2.1-2.6 | 8-14 | 0.70-0.80 | 0.65-0.75 | 0.60-0.70 | ~6-10 hours | ~10-14GB |
| **OpenTSLMFlamingo** (Llama-3.2-1B) | 1.6-2.1 | 5-8 | 0.80-0.88 | 0.75-0.83 | 0.70-0.78 | ~10-14 hours | ~14-18GB |

*Note: Results vary based on random seed, hardware, exact configuration, and whether LoRA is enabled*

**Performance Interpretation:**
- **PPL < 8**: Excellent - coherent sleep staging reasoning
- **PPL 8-15**: Good - generally sound reasoning with minor issues
- **PPL 15-30**: Acceptable - reasoning captures main points but may lack detail
- **PPL > 30**: Poor - reasoning may be incoherent or generic

**Accuracy vs. Perplexity:**
- Lower perplexity typically correlates with higher accuracy
- Sleep staging is challenging even for experts (inter-rater agreement ~80-85%)
- Both metrics important for evaluating model quality

**Per-Class Performance:**

Typical F1 scores by sleep stage (ordered by difficulty):

| Sleep Stage | Typical F1 | Challenge Level |
|-------------|-----------|-----------------|
| **Wake** | 0.85-0.95 | Easiest - distinct high-frequency pattern |
| **N3** | 0.80-0.90 | Easy - prominent slow waves |
| **REM** | 0.75-0.85 | Moderate - similar to wake, needs context |
| **N2** | 0.70-0.80 | Moderate - requires spindle/K-complex detection |
| **N1** | 0.60-0.75 | Hard - transitional, short duration |
| **Movement** | 0.55-0.70 | Hardest - artifacts, rare class |

---

## Implementation Algorithms

### Algorithm 1: SleepEDF CoT Data Loading

```
Function: load_and_preprocess_SleepEDF_CoT()

Output: D_train, D_val, D_test (preprocessed datasets)

1. Download required datasets (if not present):
   if not exists(SLEEP_DATA_DIR):
       download_sleepedf_cot()
   
2. Load CoT data from CSV file:
   df ← read_csv("sleep_cot.csv")
   
3. Parse time series data:
   for i, row in enumerate(df):
       // Parse nested list structure [[...]]
       time_series_str ← row['time_series']
       parsed ← ast.literal_eval(time_series_str)
       
       // Extract inner list (the actual EEG data)
       if isinstance(parsed, list) and len(parsed) == 1:
           time_series ← parsed[0]
           assert len(time_series) == 3000  # Validate length
       else:
           raise ValueError("Invalid time series format")
       
       df.at[i, 'time_series'] ← time_series
   
4. Perform stratified split by label:
   // First split: train+val vs test
   train_val_df, test_df ← train_test_split(
       df,
       test_size=0.1,
       random_state=42,
       stratify=df['label']
   )
   
   // Second split: train vs val
   train_df, val_df ← train_test_split(
       train_val_df,
       test_size=0.1/(1-0.1),  # Adjust fraction
       random_state=43,
       stratify=train_val_df['label']
   )
   
5. Create HuggingFace datasets:
   D_train ← Dataset.from_pandas(train_df)
   D_val ← Dataset.from_pandas(val_df)
   D_test ← Dataset.from_pandas(test_df)
   
6. Print statistics:
   print(f"Train: {len(D_train)} samples")
   print(f"Val: {len(D_val)} samples")
   print(f"Test: {len(D_test)} samples")
   
   // Print label distribution
   for split_name, split_data in [("Train", D_train), ("Val", D_val), ("Test", D_test)]:
       label_counts ← Counter(split_data['label'])
       print(f"\n{split_name} label distribution:")
       for label, count in label_counts.items():
           print(f"  {label}: {count} ({count/len(split_data)*100:.1f}%)")
   
7. Return D_train, D_val, D_test
```

### Algorithm 2: Single-Channel EEG Loading and Preprocessing

```
Function: load_and_preprocess_eeg(time_series_data)

Input: Raw EEG time series (list or array)
Output: Normalized EEG tensor

1. Convert to numpy array:
   x ← np.array(time_series_data, dtype=np.float32)  # Shape: [3000]
   
2. Validate length:
   assert len(x) == 3000, f"Expected 3000 samples, got {len(x)}"
   
3. Compute statistics:
   μ ← mean(x)
   σ ← std(x)
   
4. Z-score normalization:
   min_std ← 1e-6  # Minimum std to avoid division by zero
   σ_safe ← max(σ, min_std)
   x_norm ← (x - μ) / σ_safe
   
5. Convert to tensor:
   x_tensor ← torch.tensor(x_norm, dtype=torch.float32)
   
6. Return x_tensor, μ, σ  # Also return stats for prompt
```

### Algorithm 3: Sleep CoT Loss Computation

```
Function: compute_sleep_cot_loss(model, x_eeg, context, rationale)

Input: Model, single-channel EEG, context text, rationale text
Output: Loss value

1. Normalize EEG:
   mean ← np.mean(x_eeg)
   std ← np.std(x_eeg)
   x_norm ← (x_eeg - mean) / max(std, 1e-6)
   
2. Encode EEG:
   H_enc ← model.encoder(x_norm)                 # Shape: [N, d_enc]
   
3. Project to LLM space:
   H_proj ← model.projector(H_enc)               # Shape: [N, d_llm]
   
4. Prepare context prompt:
   prompt ← create_sleep_prompt(mean, std, context)
   C_tokens ← tokenize(prompt)                   # Shape: [M]
   
5. Prepare CoT target:
   R_tokens ← tokenize(rationale)                # Shape: [K]
   
6. Embed text:
   H_C ← model.llm.embed(C_tokens)               # Shape: [M, d_llm]
   H_R ← model.llm.embed(R_tokens)               # Shape: [K, d_llm]
   
7. Create input sequence:
   H_input ← concat([H_C, H_proj, H_R], dim=0)  # Shape: [M+N+K, d_llm]
   
8. Create labels:
   // Labels are -100 for non-CoT tokens (ignored in loss)
   labels ← [-100] × (M + N) + R_tokens[1:]     # Shape: [M+N+K-1]
   
9. LLM forward pass:
   logits ← model.llm(H_input)                   # Shape: [M+N+K, |V|]
   
10. Compute cross-entropy loss:
   // Only compute loss on CoT tokens
   cot_start ← M + N
   cot_logits ← logits[cot_start:cot_start+K-1, :] # Shape: [K-1, |V|]
   cot_targets ← R_tokens[1:]                    # Shape: [K-1]
   
   loss ← CrossEntropyLoss(cot_logits, cot_targets)
   
11. Return loss
```

### Algorithm 4: Sleep CoT Generation with Sleep Stage Extraction

```
Function: generate_sleep_cot(model, x_eeg, context, max_tokens=400, temperature=1.0)

Input: Model, single-channel EEG, context, generation parameters
Output: Generated reasoning text and extracted sleep stage

1. Normalize EEG:
   mean ← np.mean(x_eeg)
   std ← np.std(x_eeg)
   x_norm ← (x_eeg - mean) / max(std, 1e-6)
   
2. Encode EEG:
   H_enc ← model.encoder(x_norm)
   H_proj ← model.projector(H_enc)
   
3. Prepare context prompt:
   prompt ← create_sleep_prompt(mean, std, context)
   C_tokens ← tokenize(prompt)
   H_C ← model.llm.embed(C_tokens)
   
4. Initialize generation:
   H_input ← concat([H_C, H_proj], dim=0)        # Shape: [M+N, d_llm]
   generated_tokens ← []
   
5. Autoregressive generation:
   for step = 1 to max_tokens:
       // Forward pass
       logits ← model.llm(H_input)[-1, :]        # Shape: [|V|]
       
       // Apply temperature
       logits ← logits / temperature
       
       // Compute probabilities
       probs ← Softmax(logits)
       
       // Greedy decode (for evaluation)
       next_token ← argmax(probs)
       
       // Check for EOS
       if next_token == EOS_token:
           break
       
       // Append token
       generated_tokens.append(next_token)
       
       // Update input
       next_token_emb ← model.llm.embed(next_token)
       H_input ← concat([H_input, next_token_emb], dim=0)
   
6. Decode to text:
   cot_text ← detokenize(generated_tokens)
   
7. Extract reasoning and sleep stage:
   if "Answer:" in cot_text:
       reasoning ← cot_text.split("Answer:")[0].strip()
       sleep_stage ← cot_text.split("Answer:")[-1].strip()
   else:
       reasoning ← cot_text
       sleep_stage ← "Wake"  # Default
   
8. Clean and normalize sleep stage:
   sleep_stage ← sleep_stage.rstrip('.,!?;:').strip()
   sleep_stage ← normalize_sleep_stage(sleep_stage)
   
9. Return reasoning, sleep_stage
```

### Algorithm 5: Per-Class F1 and Cohen's Kappa Computation

```
Function: compute_sleep_metrics(predictions, ground_truths)

Input: List of predicted sleep stages, list of ground truth sleep stages
Output: Per-class F1 scores, macro-F1, Cohen's kappa, confusion matrix

1. Define sleep stages:
   stages ← ["Wake", "Non-REM stage 1", "Non-REM stage 2", 
             "Non-REM stage 3", "REM sleep", "Movement"]
   num_classes ← len(stages)
   
2. Initialize confusion matrix:
   confusion_matrix ← zeros(num_classes, num_classes)
   
3. Build confusion matrix:
   for i = 0 to len(predictions) - 1:
       pred ← normalize_sleep_stage(predictions[i])
       true ← normalize_sleep_stage(ground_truths[i])
       
       pred_idx ← stages.index(pred) if pred in stages else 0
       true_idx ← stages.index(true) if true in stages else 0
       
       confusion_matrix[true_idx, pred_idx] += 1
   
4. Compute per-class metrics:
   per_class_f1 ← []
   
   for i = 0 to num_classes - 1:
       TP ← confusion_matrix[i, i]
       FP ← sum(confusion_matrix[:, i]) - TP
       FN ← sum(confusion_matrix[i, :]) - TP
       
       precision ← TP / (TP + FP) if (TP + FP) > 0 else 0
       recall ← TP / (TP + FN) if (TP + FN) > 0 else 0
       
       if precision + recall > 0:
           f1 ← 2 × precision × recall / (precision + recall)
       else:
           f1 ← 0
       
       per_class_f1.append(f1)
   
5. Compute macro-F1:
   macro_f1 ← mean(per_class_f1)
   
6. Compute Cohen's kappa:
   N ← sum(confusion_matrix)  # Total samples
   
   // Observed agreement
   p_o ← trace(confusion_matrix) / N
   
   // Expected agreement
   p_e ← 0
   for i = 0 to num_classes - 1:
       row_sum ← sum(confusion_matrix[i, :])
       col_sum ← sum(confusion_matrix[:, i])
       p_e += (row_sum × col_sum) / (N × N)
   
   // Cohen's kappa
   if 1 - p_e > 0:
       kappa ← (p_o - p_e) / (1 - p_e)
   else:
       kappa ← 0
   
7. Return per_class_f1, macro_f1, kappa, confusion_matrix
```

---

## Comparison with Previous Stages

### Key Differences

| Aspect | Stage 3 (ECG CoT) | Stage 4 (Sleep CoT) |
|--------|-------------------|---------------------|
| **Task** | ECG Question Answering | Sleep Stage Classification |
| **Domain** | Cardiac electrophysiology | Neurophysiology / Sleep medicine |
| **Time Series Type** | Multi-lead (6-lead ECG) | Single-channel (EEG) |
| **Signal Duration** | 10 seconds | 30 seconds |
| **Sampling Rate** | ~100 Hz | 100 Hz |
| **Total Samples** | ~1,000 per lead | ~3,000 |
| **Time Series Tokens** | ~1,500 (6 leads × 250) | ~750 (1 channel × 750) |
| **Total Sequence** | ~1,800-2,500 tokens | ~1,000-1,500 tokens |
| **Output Classes** | Variable (question-dependent) | 6 fixed sleep stages |
| **Question Type** | Diagnostic verification | Classification |
| **Clinical Context** | Patient demographics, question | Sleep epoch context |
| **Typical Accuracy** | 0.70-0.85 | 0.75-0.88 |
| **Inter-Expert Agreement** | ~0.80-0.85 | ~0.80-0.85 |
| **Epochs** | 60 | 60 |

### Similarities

1. **Architecture**: Identical encoder, projector, and LLM across all stages
2. **Loss Function**: Both use cross-entropy (causal language modeling)
3. **Optimization**: Same AdamW with warmup + linear decay
4. **Learning Rates**: Identical (encoder: 2e-4, projector: 1e-4)
5. **Curriculum**: Both initialize from previous stage checkpoint
6. **CoT Reasoning**: Both use chain-of-thought reasoning structure
7. **Hyperparameters**: Same batch size, gradient clipping, weight decay
8. **Training Duration**: Both use 60 epochs with early stopping

### Curriculum Learning Progression

**Full Pipeline:**

$$
\text{Stage 1} \rightarrow \text{Stage 2} \rightarrow \text{Stage 3} \rightarrow \text{Stage 4}
$$

**Skills Acquired:**

| Stage | Primary Skill | Domain | Reasoning Type |
|-------|--------------|--------|----------------|
| **Stage 1** | Pattern recognition | General time series | Implicit |
| **Stage 2** | Detailed description | Economic/demographic | Descriptive |
| **Stage 3** | Medical reasoning | Cardiac | Explicit (ECG) |
| **Stage 4** | Medical reasoning | Neurophysiological | Explicit (Sleep) |

**Mathematical Progression:**

1. **Stage 1**: Learn $P_\theta(A | \mathbf{x}, Q)$ - Answer given time series
2. **Stage 2**: Learn $P_\theta(C | \mathbf{x})$ - Detailed caption generation
3. **Stage 3**: Learn $P_\theta(R, A | \mathbf{X}_{\text{ECG}}, Q)$ - ECG reasoning
4. **Stage 4**: Learn $P_\theta(R, A | \mathbf{x}_{\text{EEG}}, C)$ - Sleep staging reasoning

**Why This Order?**

The curriculum is carefully designed:

$$
\text{Classification} \rightarrow \text{Generation} \rightarrow \text{Medical Reasoning (Cardiac)} \rightarrow \text{Medical Reasoning (Neural)}
$$

1. **Stage 1**: Establishes time series understanding
   - Model learns: "What is this pattern?"
   - Output: Simple labels

2. **Stage 2**: Extends to natural language generation
   - Model learns: "How do I describe this in detail?"
   - Output: Fluent, coherent text

3. **Stage 3**: Introduces medical chain-of-thought reasoning
   - Model learns: "How do I explain cardiac diagnosis?"
   - Output: Structured ECG reasoning → conclusion

4. **Stage 4**: Extends medical reasoning to neurophysiology
   - Model learns: "How do I explain sleep staging?"
   - Output: Structured EEG reasoning → sleep stage
   - Leverages: Medical reasoning framework from Stage 3

**Benefits of Stage 3 → Stage 4 Transfer:**

- ✅ Medical reasoning framework already established
- ✅ Chain-of-thought structure already learned
- ✅ Time series understanding already strong
- ✅ Faster convergence (domain adaptation vs. from scratch)
- ✅ Better performance (warm start from medical reasoning)

---

## Summary

### Stage 4 Key Takeaways

1. **Purpose**: Sleep stage classification with chain-of-thought neurophysiological reasoning
2. **Dataset**: SleepEDF CoT with single-channel EEG signals and GPT-4o generated reasoning
3. **Architecture**: Same as Stages 1-3, initialized from Stage 3 checkpoint
4. **Training**: 60 epochs, early stopping, focus on test loss/perplexity
5. **Metrics**: Test loss, perplexity, accuracy, macro-F1, and Cohen's kappa
6. **Challenge**: 6-class sleep staging with clinical-grade chain-of-thought reasoning
7. **Output**: Trained model capable of sleep stage classification with expert-level reasoning

### Mathematical Components Summary

| Component | Mathematical Operation | Dimensionality |
|-----------|------------------------|----------------|
| **Input** | $\mathbf{x} \in \mathbb{R}^L$ | $[B, L]$ → single-channel EEG (L≈3000) |
| **Normalization** | $\tilde{x}_t = \frac{x_t - \mu}{\max(\sigma, \epsilon)}$ | Per-epoch z-score |
| **Encoder** | $\mathbf{H}_{\text{enc}} = \text{TransformerEncoder}(\text{PatchEmbed}(\tilde{\mathbf{x}}))$ | $[B, L] \rightarrow [B, N, d_{\text{enc}}]$ (N≈750) |
| **Projector** | $\mathbf{H}_{\text{proj}} = \text{MLP}(\mathbf{H}_{\text{enc}})$ | $[B, N, d_{\text{enc}}] \rightarrow [B, N, d_{\text{llm}}]$ |
| **LLM** | $\mathbf{L} = \text{LLM}([\mathbf{H}_C; \mathbf{H}_{\text{proj}}; \mathbf{H}_R])$ | $[B, M+N+K, d_{\text{llm}}] \rightarrow [B, M+N+K, |\mathcal{V}|]$ |
| **Loss** | $\mathcal{L} = -\frac{1}{K} \sum_{k=1}^{K} \log P_\theta(r_k \mid \mathbf{x}, C, r_{<k})$ | Scalar |
| **Perplexity** | $\text{PPL} = \exp(\mathcal{L})$ | Scalar |

### Next Steps

After completing Stage 4:
1. Model checkpoint saved to `results/{llm_id}/OpenTSLM*/stage4_sleep_cot/checkpoints/best_model.pt`
2. Evaluation metrics saved to `results/{llm_id}/OpenTSLM*/stage4_sleep_cot/results/metrics.json`
3. Predictions saved to `results/{llm_id}/OpenTSLM*/stage4_sleep_cot/results/test_predictions.jsonl`
4. Model now capable of sleep stage classification with chain-of-thought reasoning
5. Can be used as initialization for Stage 5 (ECG CoT) if curriculum continues
6. Evaluation parser available: `evaluation/opentslm/sleep/parse_sleep_cot_data.py`

### Complete Curriculum Learning Pipeline

**Full Training Sequence:**

```bash
# Train all stages in sequence (including Stage 4)
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage1_mcq stage2_captioning stage3_cot stage4_sleep_cot \
    --device cuda \
    --gradient_checkpointing
```

**Or train only Stage 4 (requires Stages 1-3 completed):**

```bash
# Train only Stage 4 (sleep staging CoT)
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage4_sleep_cot \
    --device cuda
```

**Evaluate only (skip training):**

```bash
# Evaluate existing Stage 4 checkpoint
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage4_sleep_cot \
    --eval_only \
    --device cuda
```

**Stage Dependencies:**

```
Stage 1 (TSQA MCQ)
    ↓
    [Checkpoint saved]
    ↓
Stage 2 (M4 Captioning)
    ↓
    [Checkpoint saved]
    ↓
Stage 3 (ECG-QA CoT)
    ↓
    [Checkpoint saved]
    ↓
Stage 4 (Sleep CoT)  ← We are here
    ↓
    [Checkpoint saved - can be used for Stage 5]
    ↓
Stage 5 (ECG CoT - if continuing curriculum)
```

### Performance Optimization Tips

**For Better Sleep Staging Quality:**

1. **Use larger batch sizes** (if memory allows): Better gradient estimates for imbalanced classes
2. **Enable gradient checkpointing**: Allows longer sequences
3. **Use LoRA**: Fine-tune LLM layers for better reasoning
4. **Increase epochs**: Sleep staging reasoning benefits from extended training
5. **Monitor per-class F1**: Identifies weak sleep stages (typically N1 and Movement)

**Memory Optimization:**

- Single-channel EEG encoding creates ~750 tokens (manageable)
- Total sequence length ~1,000-1,500 tokens (moderate)
- Enable gradient checkpointing to reduce memory
- Use batch size of 4 (or 2 if memory constrained)

**Addressing Class Imbalance:**

- Sleep stages are naturally imbalanced (N2 most common, Movement rare)
- Consider using BalancedBatchSampler for single-GPU training
- Monitor per-class F1 scores to identify underperforming stages
- May need longer training for rare classes (N1, Movement)

---

## References

### SleepEDF Dataset

The SleepEDF dataset is derived from the Sleep-EDF Database:

```bibtex
@article{kemp2000sleep,
  title={Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG},
  author={Kemp, Bob and Zwinderman, Aeilko H and Tuk, Bert and Kamphuisen, Hein AC and Oberye, Joost JL},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={47},
  number={9},
  pages={1185--1194},
  year={2000},
  publisher={IEEE}
}
```

**Database**: [PhysioNet Sleep-EDF Database](https://physionet.org/content/sleep-edfx/1.0.0/)

### Sleep Staging Standards

Sleep staging follows the American Academy of Sleep Medicine (AASM) manual:

```bibtex
@book{iber2007aasm,
  title={The AASM manual for the scoring of sleep and associated events: rules, terminology and technical specifications},
  author={Iber, Conrad},
  year={2007},
  publisher={American Academy of Sleep Medicine}
}
```

### OpenTSLM Paper

**Paper**: [OpenTSLM: An Open-Source Time Series Language Model](https://doi.org/10.13140/RG.2.2.14827.60963)

**Implementation**: See `curriculum_learning.py`, `src/time_series_datasets/sleep/`, and `evaluation/opentslm/sleep/`

### Chain-of-Thought Reasoning

```bibtex
@article{wei2022chain,
  title={Chain-of-thought prompting elicits reasoning in large language models},
  author={Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Xia, Fei and Chi, Ed and Le, Quoc V and Zhou, Denny},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={24824--24837},
  year={2022}
}
```

---

**End of Stage 4 Detailed Guide**


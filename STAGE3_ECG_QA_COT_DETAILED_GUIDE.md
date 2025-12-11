<!--
This source file is part of the OpenTSLM open-source project

SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

# Stage 3: ECG-QA Chain-of-Thought - Detailed Mathematical Guide

**A Comprehensive Mathematical and Algorithmic Explanation of Stage 3 Training**

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset: ECG-QA CoT](#dataset-ecg-qa-cot)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Evaluation](#evaluation)
7. [Implementation Algorithms](#implementation-algorithms)
8. [Comparison with Previous Stages](#comparison-with-previous-stages)

---

## Overview

### Stage 3 Purpose

Stage 3 (ECG-QA Chain-of-Thought) is the **advanced reasoning stage** of the OpenTSLM curriculum learning pipeline. Building on Stages 1 and 2, it teaches the model:

1. **Medical reasoning**: How to analyze complex medical time series (ECG data)
2. **Chain-of-thought explanation**: How to articulate step-by-step diagnostic reasoning
3. **Clinical interpretation**: How to connect ECG patterns to cardiac conditions
4. **Structured reasoning**: How to provide rationale before final answers

**Why Stage 3 After Stages 1 & 2?**
- Stage 1 taught basic time series understanding (classification/MCQ)
- Stage 2 extended to detailed text generation (captioning)
- Stage 3 combines both: **structured reasoning** → **final answer**
- Transitions from description to diagnostic reasoning

**Key Statistics:**
- **Dataset**: ECG-QA Chain-of-Thought (custom-created with GPT-4o)
- **Samples**: Variable across train/val/test splits
- **Task Type**: Chain-of-Thought Question Answering
- **Time Series**: 12-lead ECG signals (~1,000 samples per lead @ 100Hz)
- **ECG Duration**: 10 seconds (typical clinical recording)
- **Leads Used**: 6 leads (I, II, III, aVR, aVL, aVF)
- **Training Epochs**: 60 (with early stopping)
- **Metric**: Test Loss and Perplexity (reasoning quality)

---

## Dataset: ECG-QA CoT

### Dataset Description

The ECG-QA Chain-of-Thought dataset combines clinical ECG recordings from PTB-XL with AI-generated chain-of-thought reasoning. Each sample consists of:

**Input:**
- A 12-lead ECG recording: $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_{12}]$ where each $\mathbf{x}_i \in \mathbb{R}^L$
- A clinical question: $Q$ (natural language text about cardiac diagnosis)
- Clinical context: $C$ (patient demographics, recording quality)
- Question type: $\tau \in \{\text{single-verify}, \text{single-choice}, \text{comparison}\}$

**Output:**
- Chain-of-thought reasoning: $R$ (detailed step-by-step analysis)
- Final answer: $A$ (diagnostic conclusion)

### Data Format

Each sample in the ECG-QA CoT dataset has the following structure:

```json
{
  "ecg_id": [12345],                          // PTB-XL record ID(s)
  "ecg_paths": ["path/to/ecg.dat"],           // Physical signal file paths
  "clinical_contexts": ["76-year-old male..."], // Patient metadata
  "question": "Does this ECG show symptoms of...?", // Clinical question
  "question_type": "single-verify",            // Question category
  "template_id": 42,                           // Question template ID
  "answer": "yes",                             // Ground truth answer
  "rationale": "Looking at the ECG signal..."  // CoT reasoning
}
```

### Mathematical Representation

Let $\mathcal{D}_{\text{ECG-CoT}} = \{(\mathbf{X}_i, Q_i, C_i, R_i, A_i, \tau_i)\}_{i=1}^N$ be the dataset with $N$ samples, where:

- $\mathbf{X}_i = [\mathbf{x}_{i,1}, \ldots, \mathbf{x}_{i,L_{\text{leads}}}]$: Multi-lead ECG recording
- $\mathbf{x}_{i,j} \in \mathbb{R}^{L_{\text{sample}}}$: Single ECG lead ($L_{\text{sample}} \approx 1000$ samples)
- $Q_i \in \mathcal{V}^*$: Clinical question (sequence of tokens)
- $C_i \in \mathcal{V}^*$: Clinical context (patient info, recording quality)
- $R_i \in \mathcal{V}^*$: Chain-of-thought reasoning (detailed analysis)
- $A_i \in \mathcal{V}^*$: Final answer (diagnostic conclusion)
- $\tau_i \in \mathcal{T}$: Question type from set $\mathcal{T}$

### Data Splits

The dataset is split into three subsets:

$$
\mathcal{D}_{\text{ECG-CoT}} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}
$$

**Approximate proportions** (exact numbers vary):
- $|\mathcal{D}_{\text{train}}|$: Training samples
- $|\mathcal{D}_{\text{val}}|$: Validation samples
- $|\mathcal{D}_{\text{test}}|$: Test samples

**Splitting Algorithm:**
```
1. Load ECG-QA CoT data from CSV files:
   - ecg_qa_cot_train.csv
   - ecg_qa_cot_val.csv
   - ecg_qa_cot_test.csv
2. Parse each row for required fields
3. Resolve PTB-XL file paths for each ecg_id
4. Validate signal files exist (.dat and .hea)
5. Construct HuggingFace Dataset objects
6. Return D_train, D_val, D_test
```

### ECG Signal Characteristics

**ECG Recording Properties:**

| Property | Value | Description |
|----------|-------|-------------|
| **Leads** | 6 selected (I, II, III, aVR, aVL, aVF) | Limb leads for cardiac analysis |
| **Sampling Rate** | ~100 Hz | PTB-XL standard sampling |
| **Duration** | 10 seconds | Standard clinical recording |
| **Samples per Lead** | ~1,000 | Total time series length |
| **Total Dimensions** | $L_{\text{leads}} \times L_{\text{sample}}$ | Multi-variate time series |

**ECG Lead Selection:**

For OpenTSLM, we use **6 limb leads**:

$$
\mathbf{X} = [\mathbf{x}_{\text{I}}, \mathbf{x}_{\text{II}}, \mathbf{x}_{\text{III}}, \mathbf{x}_{\text{aVR}}, \mathbf{x}_{\text{aVL}}, \mathbf{x}_{\text{aVF}}]
$$

where each lead captures electrical activity from different cardiac perspectives.

### Chain-of-Thought Reasoning Structure

**Typical CoT Reasoning Contains:**

1. **Initial Observation**: General patterns in ECG morphology
2. **Detailed Analysis**: Examination of intervals, waves, segments
3. **Pattern Recognition**: Identification of abnormalities or normal features
4. **Clinical Interpretation**: Connection to cardiac physiology
5. **Diagnostic Reasoning**: Step-by-step logic toward conclusion
6. **Final Answer**: Clear diagnostic statement

**Example CoT Reasoning:**

```
Looking at the ECG signal, I observe the overall rhythm and morphology of the 
cardiac cycle. The P waves appear regular with consistent PR intervals around 
160ms, suggesting normal atrioventricular conduction. The QRS complexes show 
a normal duration of approximately 100ms, indicating proper ventricular 
depolarization. However, examining the T waves more closely, there is notable 
flattening in leads I and aVL, with some inversion in lead III. These T wave 
abnormalities are non-specific and could indicate various conditions including 
electrolyte imbalances, myocardial ischemia, or ventricular strain. The pattern 
does not clearly point to a specific diagnostic category but represents 
measurable deviation from normal T wave morphology. Given the presence of 
these observable T wave changes without clear pathological significance, this 
meets the criteria for non-diagnostic T abnormalities. Answer: yes
```

### Question Types and Templates

**Template-Based Questions:**

The dataset uses **template_id** to categorize questions:

| Template Type | Description | Example |
|---------------|-------------|---------|
| **Diagnostic Verification** | Yes/no questions about conditions | "Does this ECG show...?" |
| **Morphology Assessment** | Questions about waveform features | "Are the QRS complexes normal?" |
| **Rhythm Analysis** | Questions about cardiac rhythm | "Is the heart rhythm regular?" |
| **Interval Measurement** | Questions about timing intervals | "Is the PR interval prolonged?" |

**Possible Answers per Template:**

Each template has a defined set of valid answers:
- Binary: `["yes", "no", "not sure"]`
- Specific choices: `["normal", "abnormal", "borderline"]`
- Measurements: `["normal", "prolonged", "shortened"]`

### Data Preprocessing

**1. ECG Signal Loading:**

Each ECG recording is loaded from PTB-XL format:

```python
import wfdb  # WaveForm DataBase library

# Load ECG signal from .dat and .hea files
ecg_record = wfdb.rdrecord(ecg_base_path)
ecg_signals = ecg_record.p_signal  # Shape: [samples, 12_leads]
```

**2. Lead Selection:**

Extract the 6 limb leads:

$$
\mathbf{X}_{\text{selected}} = \mathbf{X}[:, [0, 1, 2, 3, 4, 5]]
$$

where indices correspond to leads I, II, III, aVR, aVL, aVF.

**3. Per-Lead Normalization:**

For each lead $\mathbf{x}_j = [x_{j,1}, \ldots, x_{j,L}]$, apply z-score normalization:

$$
\mu_j = \frac{1}{L} \sum_{t=1}^{L} x_{j,t}
$$

$$
\sigma_j = \sqrt{\frac{1}{L} \sum_{t=1}^{L} (x_{j,t} - \mu_j)^2}
$$

$$
\tilde{x}_{j,t} = \frac{x_{j,t} - \mu_j}{\sigma_j + \epsilon}, \quad \epsilon = 10^{-8}
$$

The normalized lead is: $\tilde{\mathbf{x}}_j = [\tilde{x}_{j,1}, \ldots, \tilde{x}_{j,L}]$

**4. Padding for Batch Processing:**

ECG signals within a batch may have slightly different lengths. Pad to maximum length divisible by patch size $P = 4$:

$$
L_{\text{max}} = \left\lceil \frac{\max(L_1, \ldots, L_B)}{P} \right\rceil \times P
$$

For lead $j$ with $L_j < L_{\text{max}}$:

$$
\mathbf{x}_j^{\text{padded}} = [\tilde{x}_{j,1}, \ldots, \tilde{x}_{j,L_j}, \underbrace{0, \ldots, 0}_{L_{\text{max}} - L_j}]
$$

**5. Prompt Construction:**

Each sample is formatted as a detailed clinical prompt:

$$
\text{Prompt}_i = P_{\text{role}} \oplus C_i \oplus Q_i \oplus P_{\text{instructions}}
$$

where:
- $P_{\text{role}} = \text{"You are an expert cardiologist analyzing an ECG..."}$
- $C_i = \text{Clinical context with patient info}$
- $Q_i = \text{"Question: "} \oplus Q_i$
- $P_{\text{instructions}} = \text{"Instructions for CoT reasoning..."}$
- $\oplus$ denotes string concatenation

**6. Chain-of-Thought Target Construction:**

The target output combines reasoning and answer:

$$
\text{Target}_i = R_i \oplus \text{" Answer: "} \oplus A_i
$$

where:
- $R_i$: Chain-of-thought reasoning (paragraph form)
- $A_i$: Final diagnostic answer

---

## Mathematical Formulation

### Problem Formulation

Given:
- Multi-lead ECG recording: $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_L]$ where $L$ is number of leads
- Clinical question with context: $Q$ (sequence of tokens)

Goal:
- Generate chain-of-thought reasoning: $\hat{R} = [\hat{r}_1, \hat{r}_2, \ldots, \hat{r}_K]$
- Generate final answer: $\hat{A} = [\hat{a}_1, \hat{a}_2, \ldots, \hat{a}_M]$

This is a **conditional text generation with structured output** problem.

### Model Function

The model learns a conditional probability distribution over reasoning + answer:

$$
P_\theta(R, A | \mathbf{X}, Q) = \prod_{k=1}^{K+M} P_\theta(t_k | \mathbf{X}, Q, t_{<k})
$$

where $t = [r_1, \ldots, r_K, a_1, \ldots, a_M]$ is the complete output sequence.

The model function is:

$$
f_\theta: (\mathbb{R}^{L \times L_{\text{sample}}}, \mathcal{V}^*) \rightarrow \Delta^{|\mathcal{V}|}
$$

where $\Delta^{|\mathcal{V}|}$ is the probability simplex over the vocabulary.

### Loss Function

For chain-of-thought generation, we use **causal language modeling loss** (cross-entropy):

$$
\mathcal{L}(\theta; \mathbf{X}, Q, R, A) = -\frac{1}{K+M} \sum_{k=1}^{K+M} \log P_\theta(t_k | \mathbf{X}, Q, t_{<k})
$$

Expanded form:

$$
\mathcal{L}(\theta) = -\frac{1}{K+M} \left[\sum_{k=1}^{K} \log P_\theta(r_k | \mathbf{X}, Q, r_{<k}) + \sum_{m=1}^{M} \log P_\theta(a_m | \mathbf{X}, Q, R, a_{<m})\right]
$$

This ensures the model learns to:
1. Generate coherent reasoning given ECG and question
2. Produce correct answer conditioned on the reasoning

### Total Training Objective

Over the entire training dataset $\mathcal{D}_{\text{train}}$:

$$
\mathcal{L}_{\text{total}}(\theta) = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(\mathbf{X}_i, Q_i, R_i, A_i) \in \mathcal{D}_{\text{train}}} \mathcal{L}(\theta; \mathbf{X}_i, Q_i, R_i, A_i)
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
- Lower perplexity → More confident and coherent reasoning
- Higher perplexity → Uncertain or incoherent reasoning

---

## Model Architecture

The architecture for Stage 3 is **identical** to Stages 1 and 2, but with different data modality (multi-lead ECG) and longer output sequences (CoT reasoning). The model consists of three main components:

### 1. Time Series Encoder (Multi-Lead)

**Architecture**: Transformer-CNN Encoder (same as Stages 1 & 2)

**Input**: $\mathbf{X} \in \mathbb{R}^{B \times L_{\text{leads}} \times L_{\text{sample}}}$ (batch of multi-lead ECG)

**Processing**: Each lead is encoded independently, then concatenated

**Output**: $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times (L_{\text{leads}} \times N) \times d_{\text{enc}}}$ (encoded features)

where:
- $B$: Batch size
- $L_{\text{leads}} = 6$: Number of ECG leads
- $L_{\text{sample}}$: ECG samples per lead (padded, typically ~1000)
- $N = L_{\text{sample}} / P$: Number of patches per lead
- $P = 4$: Patch size
- $d_{\text{enc}} = 128$: Encoder embedding dimension

#### Multi-Lead Encoding Process

**Per-Lead Encoding:**

For each lead $\mathbf{x}_j \in \mathbb{R}^{L_{\text{sample}}}$:

1. **Patch Embedding**: 
   $$
   \mathbf{P}_j = \text{Conv1D}(\mathbf{x}_j) \in \mathbb{R}^{N \times d_{\text{enc}}}
   $$

2. **Positional Encoding**: 
   $$
   \mathbf{P}_j' = \mathbf{P}_j + \mathbf{E}_{\text{pos}}[:N, :] \in \mathbb{R}^{N \times d_{\text{enc}}}
   $$

3. **Transformer Encoding**: 
   $$
   \mathbf{H}_{\text{enc}, j} = \text{TransformerEncoder}_{L=6}(\mathbf{P}_j')
   $$

**Lead Concatenation:**

$$
\mathbf{H}_{\text{enc}} = [\mathbf{H}_{\text{enc}, 1}; \mathbf{H}_{\text{enc}, 2}; \ldots; \mathbf{H}_{\text{enc}, L_{\text{leads}}}] \in \mathbb{R}^{(L_{\text{leads}} \times N) \times d_{\text{enc}}}
$$

This creates a unified representation of all ECG leads.

**Total Sequence Length:**

$$
N_{\text{total}} = L_{\text{leads}} \times N = 6 \times \frac{L_{\text{sample}}}{4} \approx 6 \times 250 = 1500 \text{ tokens}
$$

### 2. Projector

**Architecture**: MLP with LayerNorm and GELU (same as Stages 1 & 2)

**Input**: $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times N_{\text{total}} \times d_{\text{enc}}}$

**Output**: $\mathbf{H}_{\text{proj}} \in \mathbb{R}^{B \times N_{\text{total}} \times d_{\text{llm}}}$

**Mathematical Formulation:**

$$
\mathbf{H}_{\text{proj}} = \text{Dropout}(\text{GELU}(\text{Linear}(\text{LayerNorm}(\mathbf{H}_{\text{enc}}))))
$$

### 3. Large Language Model (LLM)

**Architecture**: Pre-trained causal LM (Llama/Gemma, same as Stages 1 & 2)

**Input**: Combined sequence of projected ECG embeddings and text tokens

**Output**: Probability distribution over vocabulary for next token prediction

#### Input Sequence Construction for ECG CoT

For a sample $(\mathbf{X}, Q, R, A)$:

**1. Text Tokenization:**

$$
Q_{\text{tokens}} = \text{Tokenize}(P_{\text{role}} + C + Q + P_{\text{instructions}}) = [q_1, \ldots, q_M]
$$

$$
R_{\text{tokens}} = \text{Tokenize}(R) = [r_1, r_2, \ldots, r_K]
$$

$$
A_{\text{tokens}} = \text{Tokenize}("Answer: " + A) = [a_1, a_2, \ldots, a_N]
$$

**2. Text Embedding:**

$$
\mathbf{H}_Q = \text{Embed}_{\text{LLM}}(Q_{\text{tokens}}) \in \mathbb{R}^{M \times d_{\text{llm}}}
$$

$$
\mathbf{H}_R = \text{Embed}_{\text{LLM}}(R_{\text{tokens}}) \in \mathbb{R}^{K \times d_{\text{llm}}}
$$

$$
\mathbf{H}_A = \text{Embed}_{\text{LLM}}(A_{\text{tokens}}) \in \mathbb{R}^{N \times d_{\text{llm}}}
$$

**3. Sequence Concatenation:**

$$
\mathbf{H}_{\text{input}} = [\mathbf{H}_Q; \mathbf{H}_{\text{proj}}; \mathbf{H}_R; \mathbf{H}_A] \in \mathbb{R}^{(M+N_{\text{total}}+K+N) \times d_{\text{llm}}}
$$

**Total sequence length**: $T = M + N_{\text{total}} + K + N$ (typically 1,800-2,500 tokens)

Note: This is significantly longer than Stages 1 and 2 due to:
- Multi-lead ECG (6 leads × ~250 patches = 1,500 tokens)
- Longer prompt with clinical context
- Extended chain-of-thought reasoning

#### Autoregressive Chain-of-Thought Generation

During training, the model learns:

$$
P_\theta(r_k | \mathbf{X}, Q, r_{<k}) = \text{Softmax}(\text{LLM}([\mathbf{H}_Q; \mathbf{H}_{\text{proj}}; \mathbf{H}_{r_{<k}}]))_k
$$

$$
P_\theta(a_m | \mathbf{X}, Q, R, a_{<m}) = \text{Softmax}(\text{LLM}([\mathbf{H}_Q; \mathbf{H}_{\text{proj}}; \mathbf{H}_R; \mathbf{H}_{a_{<m}}]))_m
$$

### Model Initialization for Stage 3

**Key Difference from Stages 1 & 2:**

Stage 3 **initializes** from the best Stage 2 checkpoint:

$$
\theta_{\text{Stage3}}^{(0)} = \theta_{\text{Stage2}}^*
$$

where $\theta_{\text{Stage2}}^*$ is the best model from Stage 2 training.

This provides:
1. **Pre-trained encoder**: Already understands time series patterns from Stages 1 & 2
2. **Pre-trained projector**: Already maps time series to LLM space effectively
3. **Pre-trained LLM**: Already capable of detailed language generation (from Stage 2)
4. **Warm start**: Much faster convergence for complex reasoning task

**Curriculum Learning Progression:**

$$
\text{Stage 1 (MCQ)} \rightarrow \text{Stage 2 (Captioning)} \rightarrow \text{Stage 3 (CoT)}
$$

Each stage builds on the previous:
- Stage 1: Basic understanding
- Stage 2: Detailed generation
- Stage 3: Structured reasoning

### Parameter Count

**Trainable Parameters (Stage 3):**

| Component | Parameters | Trainable | Notes |
|-----------|-----------|-----------|-------|
| **Encoder** | ~5M | ✅ Yes | Fine-tuned from Stage 2 |
| **Projector** | ~260K | ✅ Yes | Fine-tuned from Stage 2 |
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

**Note**: Stage 3 requires more epochs (60 vs 30/20) because chain-of-thought reasoning is a more complex task requiring extended training.

### Curriculum Learning Connection

Stage 3 builds on Stage 2:

$$
\theta_{\text{Stage3}}^{(0)} \leftarrow \text{BestCheckpoint}(\text{Stage2})
$$

This enables:
1. **Knowledge transfer**: Time series understanding and generation capabilities transfer
2. **Faster convergence**: Model starts with strong foundation
3. **Better performance**: Curriculum approach critical for complex reasoning
4. **Task-specific adaptation**: Fine-tune existing capabilities for medical reasoning

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

**AdamW Optimizer** (same as Stages 1 & 2):

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

**Algorithm: Stage 3 Training**

```
Input: Training data D_train, validation data D_val, test data D_test
       Stage 2 best checkpoint θ_stage2
Output: Trained model parameters θ*

1. Load Stage 2 checkpoint:
   encoder_params ← θ_stage2.encoder
   projector_params ← θ_stage2.projector
   lora_params ← θ_stage2.lora (if enabled)
   
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
           X_batch, Q_batch, R_batch, A_batch ← batch
           loss ← compute_cot_loss(X_batch, Q_batch, R_batch, A_batch)
           
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
               X_batch, Q_batch, R_batch, A_batch ← batch
               loss ← compute_cot_loss(X_batch, Q_batch, R_batch, A_batch)
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

### Loss Computation for Chain-of-Thought

For a single sample $(\mathbf{X}, Q, R, A)$:

**Step 1: Load and encode multi-lead ECG**
```
# Load ECG signals for each lead
X_leads ← [load_ecg_lead(path, lead_idx) for lead_idx in [0,1,2,3,4,5]]

# Normalize each lead independently
X_norm ← [normalize(x_lead) for x_lead in X_leads]

# Encode each lead
H_enc_leads ← [Encoder(x_lead) for x_lead in X_norm]  # Each: [N, d_enc]

# Concatenate all lead encodings
H_enc ← concat(H_enc_leads, dim=0)                     # Shape: [6N, d_enc]
```

**Step 2: Project to LLM space**
```
H_proj ← Projector(H_enc)                              # Shape: [6N, d_llm]
```

**Step 3: Prepare text**
```
# Create full prompt with clinical context
prompt ← create_clinical_prompt(Q, clinical_context)
Q_tokens ← Tokenize(prompt)                            # Shape: [M]

# Chain-of-thought reasoning + answer
CoT_full ← R + " Answer: " + A
CoT_tokens ← Tokenize(CoT_full)                        # Shape: [K]
```

**Step 4: Embed and concatenate**
```
H_Q ← Embed(Q_tokens)                                  # Shape: [M, d_llm]
H_CoT ← Embed(CoT_tokens)                              # Shape: [K, d_llm]

H_input ← concat([H_Q, H_proj, H_CoT], dim=0)         # Shape: [M+6N+K, d_llm]
```

**Step 5: LLM forward pass**
```
logits ← LLM(H_input)                                  # Shape: [M+6N+K, |V|]
```

**Step 6: Extract CoT logits and compute loss**
```
# CoT tokens start after question and ECG embeddings
cot_start_idx ← M + 6N
cot_logits ← logits[cot_start_idx : cot_start_idx+K-1, :]  # Shape: [K-1, |V|]

# Target tokens (shifted by 1 for next-token prediction)
target_tokens ← CoT_tokens[1:]                         # Shape: [K-1]

# Compute cross-entropy loss
loss ← CrossEntropyLoss(cot_logits, target_tokens)
```

Mathematically:
$$
\mathcal{L} = -\frac{1}{K-1} \sum_{k=1}^{K-1} \log P_\theta(t_{k+1} | \mathbf{X}, Q, t_{\leq k})
$$

where $t = [r_1, \ldots, r_{K_R}, a_1, \ldots, a_{K_A}]$ is the complete CoT sequence.

---

## Evaluation

### Evaluation Metrics

**Primary Metrics:**

1. **Test Loss** (Cross-Entropy):
   $$
   \mathcal{L}_{\text{test}} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{(\mathbf{X}, Q, R, A) \in \mathcal{D}_{\text{test}}} \mathcal{L}(\theta; \mathbf{X}, Q, R, A)
   $$

2. **Perplexity**:
   $$
   \text{PPL} = \exp(\mathcal{L}_{\text{test}})
   $$

**Interpretation:**
- **Lower test loss** = Better reasoning generation quality
- **Lower perplexity** = More confident and coherent CoT reasoning
- Typical good perplexity: 10-30 for medical reasoning tasks

**Secondary Metrics (Post-Processing):**

3. **Answer Accuracy**: Extract final answer and compare with ground truth

   For each prediction:
   ```python
   predicted_answer = extract_answer(generated_text)  # After "Answer: "
   ground_truth_answer = extract_answer(target_text)
   accuracy = predicted_answer == ground_truth_answer
   ```
   
   $$
   \text{Accuracy} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{i=1}^{|\mathcal{D}_{\text{test}}|} \mathbb{1}[\hat{A}_i = A_i]
   $$

4. **Template-Specific F1 Score**: Per-template classification metrics

   For each template $\tau$:
   $$
   F1_\tau = \frac{2 \cdot P_\tau \cdot R_\tau}{P_\tau + R_\tau}
   $$
   
   where $P_\tau$ is precision and $R_\tau$ is recall for template $\tau$.

5. **Macro-F1 Score**: Average F1 across all templates

   $$
   \text{Macro-F1} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} F1_\tau
   $$

### Inference Process

**Chain-of-Thought Generation Algorithm:**

For a test sample $(\mathbf{X}_{\text{test}}, Q_{\text{test}})$:

```
1. Preprocess ECG:
   X_leads ← load_ecg_leads(ecg_path)
   X_norm ← [normalize(x_lead) for x_lead in X_leads]
   
2. Encode ECG:
   H_enc_leads ← [Encoder(x_lead) for x_lead in X_norm]
   H_enc ← concat(H_enc_leads, dim=0)
   H_proj ← Projector(H_enc)
   
3. Prepare prompt:
   prompt ← create_clinical_prompt(Q_test, clinical_context)
   H_Q ← Embed(Tokenize(prompt))
   H_input ← concat([H_Q, H_proj], dim=0)
   
4. Generate CoT reasoning autoregressively:
   generated_tokens ← []
   current_input ← H_input
   
   for step = 1 to max_new_tokens (e.g., 400):
       // Forward pass
       logits ← LLM(current_input)
       
       // Get next token probability
       next_token_logits ← logits[-1, :]
       next_token_prob ← Softmax(next_token_logits / temperature)
       
       // Greedy or sampling decode
       if use_sampling:
           next_token ← nucleus_sample(next_token_prob, p=0.9)
       else:
           next_token ← argmax(next_token_prob)  # Greedy
       
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
   
6. Extract reasoning and answer:
   if "Answer:" in cot_text:
       reasoning ← cot_text.split("Answer:")[0].strip()
       answer ← cot_text.split("Answer:")[-1].strip()
   else:
       reasoning ← cot_text
       answer ← "not sure"  # Default if no answer found
   
7. Return reasoning, answer
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
Function: evaluate_stage3(model, D_test)

Input: Trained model, test dataset D_test
Output: Test loss, perplexity, accuracy, and predictions

1. Initialize:
   total_loss ← 0
   total_tokens ← 0
   predictions ← []
   ground_truths ← []
   correct_answers ← 0
   
2. Set model to evaluation mode:
   model.eval()
   
3. Disable gradient computation:
   with torch.no_grad():
       
       for (X, Q, R, A) in D_test:
           // Compute loss
           loss ← compute_cot_loss(X, Q, R, A)
           K ← len(Tokenize(R + " Answer: " + A))
           
           total_loss ← total_loss + loss.item() × K
           total_tokens ← total_tokens + K
           
           // Generate prediction
           reasoning_pred, answer_pred ← model.generate_cot(X, Q, max_new_tokens=400)
           
           // Extract ground truth answer
           answer_true ← A
           
           // Store predictions
           predictions.append({
               "generated_answer": answer_pred,
               "target_answer": answer_true,
               "reasoning": reasoning_pred,
               "target_reasoning": R,
               "template_id": get_template_id(sample),
               "question_type": get_question_type(sample)
           })
           ground_truths.append(answer_true)
           
           // Check accuracy (after normalization)
           if normalize_answer(answer_pred) == normalize_answer(answer_true):
               correct_answers ← correct_answers + 1
   
4. Compute metrics:
   avg_test_loss ← total_loss / total_tokens
   perplexity ← exp(avg_test_loss)
   accuracy ← correct_answers / len(D_test)
   
5. Compute per-template metrics:
   template_f1_scores ← compute_template_f1(predictions)
   macro_f1 ← compute_macro_f1(template_f1_scores)
   
6. Save results:
   results ← {
       "test_loss": avg_test_loss,
       "perplexity": perplexity,
       "accuracy": accuracy,
       "macro_f1": macro_f1,
       "total_samples": len(D_test),
       "total_tokens": total_tokens,
       "template_metrics": template_f1_scores
   }
   
   save_json(results, "stage3_ecg_cot/results/metrics.json")
   save_jsonl(predictions, "stage3_ecg_cot/results/test_predictions.jsonl")
   
7. Print summary:
   print(f"Test Loss: {avg_test_loss:.4f}")
   print(f"Perplexity: {perplexity:.2f}")
   print(f"Accuracy: {accuracy:.4f}")
   print(f"Macro-F1: {macro_f1:.4f}")
   print(f"Avg tokens per sample: {total_tokens/len(D_test):.1f}")
   
8. Return results
```

### Answer Extraction and Normalization

**Extraction Function:**

```python
def extract_answer(text: str) -> str:
    """Extract the final answer from CoT reasoning."""
    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()
    else:
        return text.strip()
    
    # Remove end-of-text tokens
    answer = re.sub(r'<\|.*?\|>|<eos>$', '', answer).strip()
    
    # Remove trailing periods
    answer = re.sub(r'\.$', '', answer).strip()
    
    return answer
```

**Normalization Function:**

```python
def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove punctuation
    answer = answer.rstrip('.,!?;:')
    
    # Strip whitespace
    answer = answer.strip()
    
    return answer
```

### Expected Performance

**Typical Stage 3 Results:**

| Model | Test Loss | Perplexity | Accuracy | Macro-F1 | Training Time | GPU Memory |
|-------|-----------|------------|----------|----------|---------------|------------|
| **OpenTSLMSP** (Llama-3.2-1B) | 2.0-2.5 | 7-12 | 0.70-0.80 | 0.65-0.75 | ~8-12 hours | ~12-16GB |
| **OpenTSLMSP** (Gemma-3-270m) | 2.3-2.8 | 10-16 | 0.65-0.75 | 0.60-0.70 | ~6-10 hours | ~10-14GB |
| **OpenTSLMFlamingo** (Llama-3.2-1B) | 1.8-2.3 | 6-10 | 0.75-0.85 | 0.70-0.80 | ~10-14 hours | ~14-18GB |

*Note: Results vary based on random seed, hardware, exact configuration, and whether LoRA is enabled*

**Performance Interpretation:**
- **PPL < 10**: Excellent - coherent clinical reasoning
- **PPL 10-20**: Good - generally sound reasoning with minor issues
- **PPL 20-40**: Acceptable - reasoning captures main points but may lack detail
- **PPL > 40**: Poor - reasoning may be incoherent or generic

**Accuracy vs. Perplexity:**
- Lower perplexity typically correlates with higher accuracy
- However, perfect reasoning doesn't guarantee correct answers (medical complexity)
- Both metrics important for evaluating model quality

---

## Implementation Algorithms

### Algorithm 1: ECG-QA CoT Data Loading

```
Function: load_and_preprocess_ECG_QA_CoT()

Output: D_train, D_val, D_test (preprocessed datasets)

1. Download required datasets (if not present):
   if not exists(ECG_QA_COT_DIR):
       download_ecg_qa_cot()
   if not exists(PTBXL_DIR):
       download_ptbxl()
   
2. Load CoT data from CSV files:
   train_df ← read_csv("ecg_qa_cot_train.csv")
   val_df ← read_csv("ecg_qa_cot_val.csv")
   test_df ← read_csv("ecg_qa_cot_test.csv")
   
3. For each split (train, val, test):
   samples ← []
   
   for row in split_df:
       // Extract fields
       ecg_id ← parse_ecg_id(row["ecg_id"])
       question ← row["question"]
       answer ← row["answer"]
       template_id ← row["template_id"]
       question_type ← row["question_type"]
       clinical_context ← row["clinical_context"]
       rationale ← row["rationale"]
       
       // Resolve PTB-XL file paths
       ecg_base_path ← get_ptbxl_ecg_path(ecg_id)
       dat_path ← ecg_base_path + ".dat"
       hea_path ← ecg_base_path + ".hea"
       
       // Validate files exist
       if not exists(dat_path) or not exists(hea_path):
           raise FileNotFoundError(f"ECG files not found for {ecg_id}")
       
       // Create sample dictionary
       sample ← {
           "ecg_id": [ecg_id],
           "ecg_paths": [dat_path],
           "clinical_contexts": [clinical_context],
           "question": question,
           "answer": answer,
           "template_id": template_id,
           "question_type": question_type,
           "rationale": rationale
       }
       
       samples.append(sample)
   
4. Convert to HuggingFace datasets:
   D_train ← Dataset.from_list(train_samples)
   D_val ← Dataset.from_list(val_samples)
   D_test ← Dataset.from_list(test_samples)
   
5. Print statistics:
   print(f"Train: {len(D_train)} samples")
   print(f"Val: {len(D_val)} samples")
   print(f"Test: {len(D_test)} samples")
   
6. Return D_train, D_val, D_test
```

### Algorithm 2: Multi-Lead ECG Loading and Preprocessing

```
Function: load_and_preprocess_ecg(ecg_path, lead_indices=[0,1,2,3,4,5])

Input: Path to ECG file, lead indices to extract
Output: Normalized multi-lead ECG tensor

1. Load ECG signal using wfdb:
   import wfdb
   ecg_record ← wfdb.rdrecord(ecg_path)
   ecg_signals ← ecg_record.p_signal              # Shape: [samples, 12]
   
2. Select limb leads:
   selected_leads ← ecg_signals[:, lead_indices]  # Shape: [samples, 6]
   
3. Normalize each lead independently:
   normalized_leads ← []
   
   for lead_idx in range(6):
       x ← selected_leads[:, lead_idx]             # Shape: [samples]
       
       // Compute statistics
       μ ← mean(x)
       σ ← std(x)
       
       // Z-score normalization
       x_norm ← (x - μ) / (σ + 1e-8)
       
       normalized_leads.append(x_norm)
   
4. Stack normalized leads:
   X ← stack(normalized_leads, axis=1)            # Shape: [samples, 6]
   
5. Convert to tensor:
   X_tensor ← torch.tensor(X, dtype=torch.float32)
   
6. Return X_tensor
```

### Algorithm 3: Chain-of-Thought Loss Computation

```
Function: compute_cot_loss(model, X_ecg, question, reasoning, answer)

Input: Model, multi-lead ECG, question text, reasoning text, answer text
Output: Loss value

1. Encode multi-lead ECG:
   H_enc_list ← []
   
   for lead_idx in range(6):
       // Extract single lead
       x_lead ← X_ecg[:, lead_idx]                # Shape: [samples]
       
       // Normalize
       x_norm ← (x_lead - mean(x_lead)) / (std(x_lead) + 1e-8)
       
       // Encode
       H_enc_lead ← model.encoder(x_norm)         # Shape: [N, d_enc]
       H_enc_list.append(H_enc_lead)
   
   // Concatenate all leads
   H_enc ← concat(H_enc_list, dim=0)             # Shape: [6N, d_enc]
   
2. Project to LLM space:
   H_proj ← model.projector(H_enc)               # Shape: [6N, d_llm]
   
3. Prepare prompt:
   prompt ← create_clinical_prompt(question, clinical_context)
   Q_tokens ← tokenize(prompt)                   # Shape: [M]
   
4. Prepare CoT target:
   cot_full ← reasoning + " Answer: " + answer
   CoT_tokens ← tokenize(cot_full)               # Shape: [K]
   
5. Embed text:
   H_Q ← model.llm.embed(Q_tokens)               # Shape: [M, d_llm]
   H_CoT ← model.llm.embed(CoT_tokens)           # Shape: [K, d_llm]
   
6. Create input sequence:
   H_input ← concat([H_Q, H_proj, H_CoT], dim=0) # Shape: [M+6N+K, d_llm]
   
7. Create labels:
   // Labels are -100 for non-CoT tokens (ignored in loss)
   labels ← [-100] × (M + 6N) + CoT_tokens[1:]   # Shape: [M+6N+K-1]
   
8. LLM forward pass:
   logits ← model.llm(H_input)                   # Shape: [M+6N+K, |V|]
   
9. Compute cross-entropy loss:
   // Only compute loss on CoT tokens
   cot_start ← M + 6N
   cot_logits ← logits[cot_start:cot_start+K-1, :] # Shape: [K-1, |V|]
   cot_targets ← CoT_tokens[1:]                  # Shape: [K-1]
   
   loss ← CrossEntropyLoss(cot_logits, cot_targets)
   
10. Return loss
```

### Algorithm 4: Chain-of-Thought Generation with Answer Extraction

```
Function: generate_cot_with_answer(model, X_ecg, question, max_tokens=400, temperature=1.0)

Input: Model, multi-lead ECG, question, generation parameters
Output: Generated reasoning text and extracted answer

1. Encode multi-lead ECG:
   H_enc_list ← []
   for lead_idx in range(6):
       x_lead ← X_ecg[:, lead_idx]
       x_norm ← (x_lead - mean(x_lead)) / (std(x_lead) + 1e-8)
       H_enc_lead ← model.encoder(x_norm)
       H_enc_list.append(H_enc_lead)
   H_enc ← concat(H_enc_list, dim=0)
   
2. Project to LLM space:
   H_proj ← model.projector(H_enc)
   
3. Prepare prompt:
   prompt ← create_clinical_prompt(question, clinical_context)
   Q_tokens ← tokenize(prompt)
   H_Q ← model.llm.embed(Q_tokens)
   
4. Initialize generation:
   H_input ← concat([H_Q, H_proj], dim=0)        # Shape: [M+6N, d_llm]
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
   
7. Extract reasoning and answer:
   if "Answer:" in cot_text:
       reasoning ← cot_text.split("Answer:")[0].strip()
       answer ← cot_text.split("Answer:")[-1].strip()
   else:
       reasoning ← cot_text
       answer ← "not sure"
   
8. Clean answer:
   answer ← answer.rstrip('.,!?;:').strip()
   
9. Return reasoning, answer
```

### Algorithm 5: Per-Template F1 Score Computation

```
Function: compute_template_f1_scores(predictions)

Input: List of predictions with template_id, generated_answer, target_answer
Output: Dictionary of F1 scores per template and overall macro-F1

1. Group predictions by template_id:
   template_groups ← defaultdict(list)
   for pred in predictions:
       template_id ← pred["template_id"]
       template_groups[template_id].append(pred)
   
2. For each template:
   template_f1_scores ← {}
   
   for template_id, preds in template_groups:
       // Get possible answers for this template
       possible_answers ← get_possible_answers_for_template(template_id)
       
       // Initialize counts for each answer class
       class_counts ← {}
       for ans in possible_answers:
           class_counts[ans] ← {"tp": 0, "fp": 0, "fn": 0}
       
       // Count TP, FP, FN
       for pred in preds:
           gt ← normalize_answer(pred["target_answer"])
           pred_ans ← normalize_answer(pred["generated_answer"])
           
           if gt in class_counts:
               if pred_ans == gt:
                   class_counts[gt]["tp"] += 1
               else:
                   class_counts[gt]["fn"] += 1
                   if pred_ans in class_counts:
                       class_counts[pred_ans]["fp"] += 1
       
       // Compute F1 for each class
       class_f1_list ← []
       for ans, counts in class_counts:
           tp ← counts["tp"]
           fp ← counts["fp"]
           fn ← counts["fn"]
           
           precision ← tp / (tp + fp) if (tp + fp) > 0 else 0
           recall ← tp / (tp + fn) if (tp + fn) > 0 else 0
           f1 ← 2 × precision × recall / (precision + recall) if (precision + recall) > 0 else 0
           
           class_f1_list.append(f1)
       
       // Macro-F1 for this template
       template_macro_f1 ← mean(class_f1_list)
       template_f1_scores[template_id] ← template_macro_f1
   
3. Compute overall macro-F1:
   overall_macro_f1 ← mean(list(template_f1_scores.values()))
   
4. Return template_f1_scores, overall_macro_f1
```

---

## Comparison with Previous Stages

### Key Differences

| Aspect | Stage 1 (TSQA) | Stage 2 (M4 Captioning) | Stage 3 (ECG-QA CoT) |
|--------|----------------|-------------------------|----------------------|
| **Task** | Multiple Choice QA | Free-form Captioning | Chain-of-Thought Reasoning |
| **Output** | Short answer (1-3 words) | Long caption (50-300 tokens) | Reasoning + answer (100-400 tokens) |
| **Output Structure** | Simple answer | Descriptive paragraph | Structured CoT + final answer |
| **Metric** | Accuracy | Test Loss / Perplexity | Test Loss / Perplexity / Accuracy |
| **Evaluation** | Exact match | Language modeling quality | Reasoning + answer extraction |
| **Time Series Type** | Univariate | Univariate | Multi-variate (6-lead ECG) |
| **Sequence Length** | ~500-800 tokens | ~600-1000 tokens | ~1800-2500 tokens |
| **Domain** | General time series | Economic/demographic | Medical (cardiac) |
| **Epochs** | 30 | 20 | 60 |
| **Generation** | Greedy decode | Nucleus sampling | Greedy/sampling |
| **Reasoning Type** | Implicit | Descriptive | Explicit step-by-step |

### Similarities

1. **Architecture**: Identical encoder, projector, and LLM across all stages
2. **Loss Function**: All use cross-entropy (causal language modeling)
3. **Optimization**: Same AdamW with warmup + linear decay
4. **Learning Rates**: Consistent across stages (encoder: 2e-4, projector: 1e-4)
5. **Curriculum**: Each stage initializes from previous best checkpoint
6. **Hyperparameters**: Similar batch size, gradient clipping, weight decay

### Curriculum Learning Progression

**Complexity Hierarchy:**

$$
\text{Stage 1 (MCQ)} \rightarrow \text{Stage 2 (Captioning)} \rightarrow \text{Stage 3 (CoT)}
$$

**Skills Acquired:**

| Stage | Primary Skill | Output Type | Reasoning Depth |
|-------|--------------|-------------|-----------------|
| **Stage 1** | Pattern recognition | Classification | Implicit |
| **Stage 2** | Detailed description | Generation | Descriptive |
| **Stage 3** | Step-by-step reasoning | Structured explanation | Explicit |

**Mathematical Progression:**

1. **Stage 1**: Learn $P_\theta(A | \mathbf{x}, Q)$ - Answer given time series
2. **Stage 2**: Learn $P_\theta(C | \mathbf{x})$ - Detailed caption generation
3. **Stage 3**: Learn $P_\theta(R, A | \mathbf{X}, Q)$ - Reasoning + answer generation

**Why This Order?**

The curriculum is carefully designed:

$$
\text{Simple Classification} \rightarrow \text{Open-ended Generation} \rightarrow \text{Structured Reasoning}
$$

1. **Stage 1**: Establishes time series understanding
   - Model learns: "What is this pattern?"
   - Output: Simple labels

2. **Stage 2**: Extends to natural language generation
   - Model learns: "How do I describe this in detail?"
   - Output: Fluent, coherent text

3. **Stage 3**: Combines understanding + generation for reasoning
   - Model learns: "How do I explain my diagnostic process?"
   - Output: Structured reasoning → conclusion

Without this progression:
- ❌ Training CoT from scratch often fails (reasoning too complex)
- ❌ Jumping directly to medical reasoning lacks foundation
- ✅ Curriculum learning enables successful complex reasoning

---

## Summary

### Stage 3 Key Takeaways

1. **Purpose**: Advanced reasoning stage teaching medical chain-of-thought analysis
2. **Dataset**: ECG-QA CoT with multi-lead ECG signals and GPT-4o generated reasoning
3. **Architecture**: Same as Stages 1 & 2, initialized from Stage 2 checkpoint
4. **Training**: 60 epochs, early stopping, focus on test loss/perplexity
5. **Metric**: Test loss, perplexity, accuracy, and macro-F1 score
6. **Challenge**: Longest sequences (~2000 tokens), most complex reasoning
7. **Output**: Trained model capable of clinical ECG reasoning

### Mathematical Components Summary

| Component | Mathematical Operation | Dimensionality |
|-----------|------------------------|----------------|
| **Input** | $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_6]$ | $[B, 6, L]$ → multi-lead ECG |
| **Normalization** | $\tilde{x}_{j,t} = \frac{x_{j,t} - \mu_j}{\sigma_j + \epsilon}$ | Per-lead z-score |
| **Encoder (per lead)** | $\mathbf{H}_{\text{enc}, j} = \text{TransformerEncoder}(\text{PatchEmbed}(\tilde{\mathbf{x}}_j))$ | $[B, L] \rightarrow [B, N, d_{\text{enc}}]$ |
| **Lead Concatenation** | $\mathbf{H}_{\text{enc}} = [\mathbf{H}_{\text{enc}, 1}; \ldots; \mathbf{H}_{\text{enc}, 6}]$ | $[B, 6N, d_{\text{enc}}]$ |
| **Projector** | $\mathbf{H}_{\text{proj}} = \text{MLP}(\mathbf{H}_{\text{enc}})$ | $[B, 6N, d_{\text{enc}}] \rightarrow [B, 6N, d_{\text{llm}}]$ |
| **LLM** | $\mathbf{L} = \text{LLM}([\mathbf{H}_Q; \mathbf{H}_{\text{proj}}; \mathbf{H}_{\text{CoT}}])$ | $[B, M+6N+K, d_{\text{llm}}] \rightarrow [B, M+6N+K, |\mathcal{V}|]$ |
| **Loss** | $\mathcal{L} = -\frac{1}{K} \sum_{k=1}^{K} \log P_\theta(t_k \mid \mathbf{X}, Q, t_{<k})$ | Scalar |
| **Perplexity** | $\text{PPL} = \exp(\mathcal{L})$ | Scalar |

### Next Steps

After completing Stage 3:
1. Model checkpoint saved to `results/{llm_id}/OpenTSLM*/stage5_ecg_cot/checkpoints/best_model.pt`
2. Evaluation metrics saved to `results/{llm_id}/OpenTSLM*/stage5_ecg_cot/results/metrics.json`
3. Predictions saved to `results/{llm_id}/OpenTSLM*/stage5_ecg_cot/results/test_predictions.jsonl`
4. Model now capable of clinical chain-of-thought reasoning
5. Can be further fine-tuned on additional medical reasoning tasks
6. Evaluation parser available: `evaluation/opentslm/ecg_qa_cot/parse_ecg_qa_cot_data.py`

### Complete Curriculum Learning Pipeline

**Full Training Sequence:**

```bash
# Train all stages in sequence
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage1_mcq stage2_captioning stage5_ecg_cot \
    --device cuda \
    --gradient_checkpointing
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
Stage 3 (ECG-QA CoT)  ← We are here
    ↓
    [Final model: clinical reasoning capable]
```

### Performance Optimization Tips

**For Better Reasoning Quality:**

1. **Use larger batch sizes** (if memory allows): Better gradient estimates
2. **Enable gradient checkpointing**: Allows longer sequences
3. **Use LoRA**: Fine-tune LLM layers for better reasoning
4. **Increase epochs**: CoT reasoning benefits from extended training
5. **Monitor per-template F1**: Identifies weak question types

**Memory Optimization:**

- Multi-lead ECG encoding creates long sequences (~1500 tokens from ECG alone)
- Enable gradient checkpointing to reduce memory
- Consider reducing leads if memory constrained (use 3 leads instead of 6)
- Use smaller batch size (batch_size=2 or 1)

---

## References

### ECG-QA Paper

```bibtex
@article{oh2023ecg,
  title={ECG-QA: A Comprehensive Question Answering Dataset Combined With Electrocardiogram},
  author={Oh, Jungwoo and Lee, Gyubok and Bae, Seongsu and Kwon, Joon-myoung and Choi, Edward},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={66277--66288},
  year={2023}
}
```

**Paper**: [ECG-QA: A Comprehensive Question Answering Dataset Combined With Electrocardiogram](https://arxiv.org/abs/2306.15681)

**Original Repository**: https://github.com/Jwoo5/ecg-qa

### PTB-XL Paper

```bibtex
@article{wagner2020ptb,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Kreiseler, Dieter and Lunze, Franziska I and Samek, Wojciech and Schaeffter, Tobias},
  journal={Nature Scientific Data},
  volume={7},
  number={1},
  pages={1--15},
  year={2020}
}
```

**Paper**: [PTB-XL, a large publicly available electrocardiography dataset](https://www.nature.com/articles/s41597-020-0495-6)

### OpenTSLM Paper

**Paper**: [OpenTSLM: An Open-Source Time Series Language Model](https://doi.org/10.13140/RG.2.2.14827.60963)

**Implementation**: See `curriculum_learning.py`, `src/time_series_datasets/ecg_qa/`, and `evaluation/opentslm/ecg_qa_cot/`

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

**End of Stage 3 Detailed Guide**


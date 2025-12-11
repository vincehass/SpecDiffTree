<!--
This source file is part of the OpenTSLM open-source project

SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

# Stage 1: TSQA (Time Series Question Answering) - Detailed Mathematical Guide

**A Comprehensive Mathematical and Algorithmic Explanation of Stage 1 Training**

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset: TSQA](#dataset-tsqa)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Evaluation](#evaluation)
7. [Implementation Algorithms](#implementation-algorithms)

---

## Overview

### Stage 1 Purpose

Stage 1 (TSQA - Time Series Question Answering) is the **foundation stage** of the OpenTSLM curriculum learning pipeline. It teaches the model:

1. **Basic time series understanding**: How to process and interpret time series patterns
2. **Question answering**: How to map time series data to natural language answers
3. **Multiple choice reasoning**: How to select correct answers from given options

**Why Start Here?**
- Simple task structure (MCQ format)
- Clear evaluation metric (accuracy)
- Builds fundamental time series-to-text mapping
- Foundation for more complex reasoning in later stages

**Key Statistics:**
- **Dataset**: TSQA (ChengsenWang/TSQA on Hugging Face)
- **Samples**: ~7,000 total (6,300 train / 630 val / 700 test)
- **Task Type**: Multiple Choice Questions (MCQ)
- **Time Series**: Variable length (10 to 10,000 points)
- **Training Epochs**: 30 (with early stopping)
- **Metric**: Accuracy

---

## Dataset: TSQA

### Dataset Description

The TSQA dataset contains time series with associated multiple-choice questions. Each sample consists of:

**Input:**
- A univariate time series: $\mathbf{x} = [x_1, x_2, \ldots, x_L] \in \mathbb{R}^L$
- A question: $Q$ (natural language text)
- A task type: $\tau$ (e.g., "trend", "shape", "value")

**Output:**
- An answer: $A$ (natural language text, typically one word or short phrase)

### Data Format

Each sample in the TSQA dataset has the following structure:

```json
{
  "Series": "[1.2, 3.4, 2.1, ...]",     // Time series as JSON array
  "Question": "What is the trend?",      // Natural language question
  "Task": "trend",                       // Task category
  "Answer": "increasing"                 // Ground truth answer
}
```

### Mathematical Representation

Let $\mathcal{D} = \{(\mathbf{x}_i, Q_i, \tau_i, A_i)\}_{i=1}^N$ be the dataset with $N$ samples, where:

- $\mathbf{x}_i \in \mathbb{R}^{L_i}$: Time series of length $L_i$
- $Q_i \in \mathcal{V}^*$: Question (sequence of tokens from vocabulary $\mathcal{V}$)
- $\tau_i \in \mathcal{T}$: Task type from set $\mathcal{T}$ (e.g., {trend, shape, value, ...})
- $A_i \in \mathcal{V}^*$: Answer (sequence of tokens)

### Data Splits

The dataset is split into three subsets:

$$
\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}
$$

With proportions:
- $|\mathcal{D}_{\text{train}}| = 0.81 \times N \approx 6,300$ samples
- $|\mathcal{D}_{\text{val}}| = 0.09 \times N \approx 630$ samples
- $|\mathcal{D}_{\text{test}}| = 0.10 \times N \approx 700$ samples

**Splitting Algorithm:**
```
1. Load full dataset D with N samples
2. Shuffle D with seed=42 for reproducibility
3. Split: D_temp, D_test = split(D, test_size=0.10)
4. Split: D_train, D_val = split(D_temp, test_size=0.09/0.90)
5. Return D_train, D_val, D_test
```

### Data Preprocessing

**1. Time Series Normalization:**

For each time series $\mathbf{x} = [x_1, \ldots, x_L]$, apply z-score normalization:

$$
\mu = \frac{1}{L} \sum_{i=1}^{L} x_i
$$

$$
\sigma = \sqrt{\frac{1}{L} \sum_{i=1}^{L} (x_i - \mu)^2}
$$

$$
\tilde{x}_i = \frac{x_i - \mu}{\sigma + \epsilon}, \quad \epsilon = 10^{-8}
$$

The normalized series is: $\tilde{\mathbf{x}} = [\tilde{x}_1, \ldots, \tilde{x}_L]$

**Why normalize?**
- Removes scale dependency
- Stabilizes training
- Allows model to focus on patterns rather than absolute values

**2. Padding for Batch Processing:**

Time series have variable lengths $L_1, L_2, \ldots, L_B$ in a batch. We pad to the maximum length that is divisible by patch size $P$:

$$
L_{\text{max}} = \left\lceil \frac{\max(L_1, \ldots, L_B)}{P} \right\rceil \times P
$$

For series $i$ with $L_i < L_{\text{max}}$:

$$
\mathbf{x}_i^{\text{padded}} = [\tilde{x}_1, \ldots, \tilde{x}_{L_i}, \underbrace{0, \ldots, 0}_{L_{\text{max}} - L_i}]
$$

**3. Prompt Construction:**

Each sample is formatted as a prompt:

$$
\text{Prompt}_i = Q_i \oplus \text{" This is the time series, it has mean "} \oplus \mu_i \oplus \text{" and std "} \oplus \sigma_i \oplus \text{". Predict the "} \oplus \tau_i \oplus \text{" Answer:"}
$$

where $\oplus$ denotes string concatenation.

---

## Mathematical Formulation

### Problem Formulation

Given:
- Time series: $\mathbf{x} \in \mathbb{R}^L$
- Question: $Q = [q_1, q_2, \ldots, q_M]$ (sequence of $M$ tokens)

Goal:
- Predict answer: $\hat{A} = [\hat{a}_1, \hat{a}_2, \ldots, \hat{a}_K]$ (sequence of $K$ tokens)

### Model Function

The model learns a mapping:

$$
f_\theta: (\mathbb{R}^L, \mathcal{V}^M) \rightarrow \mathcal{V}^K
$$

where $\theta$ represents all trainable parameters.

### Loss Function

For multiple-choice QA, we use **causal language modeling loss** (cross-entropy):

$$
\mathcal{L}(\theta) = -\frac{1}{K} \sum_{k=1}^{K} \log P(a_k | \mathbf{x}, Q, a_{<k}; \theta)
$$

where:
- $a_k$: The $k$-th token of the ground truth answer
- $a_{<k} = [a_1, \ldots, a_{k-1}]$: All previous tokens
- $P(a_k | \cdot)$: Probability distribution over vocabulary $\mathcal{V}$

### Total Training Objective

Over the entire training dataset $\mathcal{D}_{\text{train}}$:

$$
\mathcal{L}_{\text{total}}(\theta) = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(\mathbf{x}_i, Q_i, A_i) \in \mathcal{D}_{\text{train}}} \mathcal{L}_i(\theta)
$$

### Optimization Objective

$$
\theta^* = \arg\min_{\theta} \mathcal{L}_{\text{total}}(\theta) + \lambda \|\theta\|_2^2
$$

where $\lambda = 10^{-2}$ is the weight decay (L2 regularization) coefficient.

---

## Model Architecture

The OpenTSLM model for Stage 1 consists of three main components:

### 1. Time Series Encoder

**Purpose**: Transform raw time series into sequence of embeddings

**Architecture**: Transformer-CNN Encoder

**Input**: $\mathbf{x} \in \mathbb{R}^{B \times L}$ (batch of time series)

**Output**: $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times N \times d_{\text{enc}}}$ (encoded features)

where:
- $B$: Batch size
- $L$: Time series length (padded)
- $N = L / P$: Number of patches
- $P = 4$: Patch size
- $d_{\text{enc}} = 128$: Encoder embedding dimension

#### Encoder Forward Pass

**Step 1: Patch Embedding**

Split time series into non-overlapping patches:

$$
\mathbf{x} = [\underbrace{x_1, \ldots, x_P}_{\text{patch 1}}, \underbrace{x_{P+1}, \ldots, x_{2P}}_{\text{patch 2}}, \ldots, \underbrace{x_{(N-1)P+1}, \ldots, x_{NP}}_{\text{patch N}}]
$$

Apply 1D convolution with kernel size $P$ and stride $P$:

$$
\mathbf{p}_i = \text{Conv1D}([x_{(i-1)P+1}, \ldots, x_{iP}]) \in \mathbb{R}^{d_{\text{enc}}}, \quad i = 1, \ldots, N
$$

Result: $\mathbf{P} = [\mathbf{p}_1, \mathbf{p}_2, \ldots, \mathbf{p}_N] \in \mathbb{R}^{N \times d_{\text{enc}}}$

**Step 2: Positional Encoding**

Add learnable positional embeddings:

$$
\mathbf{P}' = \mathbf{P} + \mathbf{E}_{\text{pos}}[:N, :] \in \mathbb{R}^{N \times d_{\text{enc}}}
$$

where $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{N_{\max} \times d_{\text{enc}}}$ is a learnable matrix ($N_{\max} = 2600$).

**Step 3: Layer Normalization and Dropout**

$$
\mathbf{P}'' = \text{Dropout}(\text{LayerNorm}(\mathbf{P}'))
$$

**Step 4: Transformer Encoder**

Apply $L_{\text{enc}} = 6$ Transformer encoder layers:

$$
\mathbf{H}_{\text{enc}} = \text{TransformerEncoder}_{L_{\text{enc}}}(\mathbf{P}'') \in \mathbb{R}^{N \times d_{\text{enc}}}
$$

Each layer consists of:
- Multi-head self-attention (8 heads)
- Feed-forward network (FFN with hidden dim 1024)
- Residual connections and layer normalization

**Transformer Layer $\ell$:**

$$
\mathbf{Z}^{(\ell)} = \text{LayerNorm}(\mathbf{H}^{(\ell-1)} + \text{MultiHeadAttention}(\mathbf{H}^{(\ell-1)}))
$$

$$
\mathbf{H}^{(\ell)} = \text{LayerNorm}(\mathbf{Z}^{(\ell)} + \text{FFN}(\mathbf{Z}^{(\ell)}))
$$

where:

$$
\text{FFN}(\mathbf{z}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2
$$

### 2. Projector

**Purpose**: Map encoder output to LLM embedding space

**Architecture**: MLP with LayerNorm and GELU activation

**Input**: $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times N \times d_{\text{enc}}}$

**Output**: $\mathbf{H}_{\text{proj}} \in \mathbb{R}^{B \times N \times d_{\text{llm}}}$

where $d_{\text{llm}}$ is the LLM hidden dimension (e.g., 2048 for Llama-3.2-1B).

**Forward Pass:**

$$
\mathbf{H}_{\text{proj}} = \text{Dropout}(\text{GELU}(\text{Linear}(\text{LayerNorm}(\mathbf{H}_{\text{enc}}))))
$$

Explicitly:

$$
\mathbf{H}_{\text{norm}} = \text{LayerNorm}(\mathbf{H}_{\text{enc}})
$$

$$
\mathbf{H}_{\text{linear}} = \mathbf{W}_{\text{proj}} \mathbf{H}_{\text{norm}} + \mathbf{b}_{\text{proj}}
$$

$$
\mathbf{H}_{\text{proj}} = \text{Dropout}(\text{GELU}(\mathbf{H}_{\text{linear}}))
$$

where $\mathbf{W}_{\text{proj}} \in \mathbb{R}^{d_{\text{llm}} \times d_{\text{enc}}}$ and $\mathbf{b}_{\text{proj}} \in \mathbb{R}^{d_{\text{llm}}}$.

### 3. Large Language Model (LLM)

**Purpose**: Generate natural language answers conditioned on time series and question

**Architecture**: Pre-trained causal LM (e.g., Llama-3.2-1B, Gemma-3-270m)

**Input**: Combined sequence of projected time series embeddings and text tokens

**Output**: Probability distribution over vocabulary for next token prediction

#### Input Sequence Construction

For a sample $(\mathbf{x}, Q, A)$:

**1. Text Tokenization:**

$$
Q_{\text{tokens}} = \text{Tokenize}(Q) = [q_1, q_2, \ldots, q_M]
$$

$$
A_{\text{tokens}} = \text{Tokenize}(A) = [a_1, a_2, \ldots, a_K]
$$

**2. Text Embedding:**

$$
\mathbf{H}_Q = \text{Embed}_{\text{LLM}}(Q_{\text{tokens}}) \in \mathbb{R}^{M \times d_{\text{llm}}}
$$

$$
\mathbf{H}_A = \text{Embed}_{\text{LLM}}(A_{\text{tokens}}) \in \mathbb{R}^{K \times d_{\text{llm}}}
$$

**3. Sequence Concatenation:**

$$
\mathbf{H}_{\text{input}} = [\mathbf{H}_Q; \mathbf{H}_{\text{proj}}; \mathbf{H}_A] \in \mathbb{R}^{(M+N+K) \times d_{\text{llm}}}
$$

where $;$ denotes concatenation along the sequence dimension.

**Total sequence length**: $T = M + N + K$

#### LLM Forward Pass

**1. Causal Attention Mask:**

Create causal mask $\mathbf{M} \in \{0, -\infty\}^{T \times T}$ where:

$$
M_{ij} = \begin{cases}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}
$$

This ensures that position $i$ can only attend to positions $\leq i$ (autoregressive property).

**2. LLM Layers:**

Apply $L_{\text{llm}}$ Transformer decoder layers:

$$
\mathbf{H}_{\text{llm}}^{(0)} = \mathbf{H}_{\text{input}}
$$

For $\ell = 1, \ldots, L_{\text{llm}}$:

$$
\mathbf{H}_{\text{llm}}^{(\ell)} = \text{TransformerDecoderLayer}_\ell(\mathbf{H}_{\text{llm}}^{(\ell-1)}, \mathbf{M})
$$

**3. Output Projection:**

$$
\mathbf{L} = \mathbf{W}_{\text{out}} \mathbf{H}_{\text{llm}}^{(L_{\text{llm}})} + \mathbf{b}_{\text{out}} \in \mathbb{R}^{T \times |\mathcal{V}|}
$$

where $|\mathcal{V}|$ is the vocabulary size.

**4. Softmax for Token Probabilities:**

$$
P(w | \mathbf{x}, Q, a_{<k}) = \text{Softmax}(\mathbf{L}[M+N+k-1, :]) \in \mathbb{R}^{|\mathcal{V}|}
$$

### Complete Forward Pass

Given input $(\mathbf{x}, Q, A)$:

1. **Normalize time series**: $\tilde{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sigma + \epsilon}$

2. **Encode time series**: $\mathbf{H}_{\text{enc}} = \text{Encoder}(\tilde{\mathbf{x}})$

3. **Project to LLM space**: $\mathbf{H}_{\text{proj}} = \text{Projector}(\mathbf{H}_{\text{enc}})$

4. **Embed text**: $\mathbf{H}_Q = \text{Embed}(Q)$, $\mathbf{H}_A = \text{Embed}(A)$

5. **Concatenate**: $\mathbf{H}_{\text{input}} = [\mathbf{H}_Q; \mathbf{H}_{\text{proj}}; \mathbf{H}_A]$

6. **LLM forward**: $\mathbf{L} = \text{LLM}(\mathbf{H}_{\text{input}})$

7. **Compute loss**: 
   $$
   \mathcal{L} = -\frac{1}{K} \sum_{k=1}^{K} \log \text{Softmax}(\mathbf{L}[M+N+k-1, :])[a_k]
   $$

### Parameter Count

**Trainable Parameters (Stage 1):**

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| **Encoder** | ~5M | ✅ Yes |
| **Projector** | ~260K | ✅ Yes |
| **LLM (Llama-3.2-1B)** | ~1.2B | ❌ No (frozen) |
| **LoRA (if enabled)** | ~2-4M | ✅ Yes |
| **Total Trainable** | ~5.3M (or ~9M with LoRA) | |

---

## Training Process

### Training Configuration

**Hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 30 | Maximum training epochs |
| Batch Size | 4 | Samples per batch |
| Learning Rate (Encoder) | $2 \times 10^{-4}$ | LR for encoder |
| Learning Rate (Projector) | $1 \times 10^{-4}$ | LR for projector |
| Learning Rate (LoRA) | $2 \times 10^{-4}$ | LR for LoRA adapters (if enabled) |
| Weight Decay | $1 \times 10^{-2}$ | L2 regularization |
| Gradient Clipping | 1.0 | Max gradient norm |
| Warmup Fraction | 0.03 | Fraction of steps for LR warmup |
| Early Stopping Patience | 5 | Epochs without improvement to stop |
| Patch Size | 4 | Time series patch size |

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

where $\eta_{\max}$ is the maximum learning rate (different for each component).

### Optimization Algorithm

**AdamW Optimizer:**

Parameters $\theta$ are updated using AdamW:

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

**Algorithm: Stage 1 Training**

```
Input: Training data D_train, validation data D_val, test data D_test
Output: Trained model parameters θ*

1. Initialize model components:
   - encoder_params ← TransformerCNNEncoder parameters
   - projector_params ← MLPProjector parameters
   - lora_params ← LoRA adapter parameters (if enabled)

2. Initialize optimizer groups:
   - optimizer_groups = [
       {params: encoder_params, lr: 2e-4, weight_decay: 1e-2},
       {params: projector_params, lr: 1e-4, weight_decay: 1e-2},
       {params: lora_params, lr: 2e-4, weight_decay: 1e-2}  # if LoRA enabled
     ]
   - optimizer ← AdamW(optimizer_groups)

3. Initialize scheduler:
   - total_steps ← num_epochs × len(D_train) / batch_size
   - warmup_steps ← 0.03 × total_steps
   - scheduler ← LinearWarmupScheduler(optimizer, warmup_steps, total_steps)

4. Training loop:
   best_val_loss ← ∞
   epochs_no_improve ← 0
   
   for epoch = 1 to max_epochs:
       // Training phase
       model.train()
       train_loss ← 0
       
       for batch in DataLoader(D_train, batch_size=4, shuffle=True):
           // Forward pass
           x_batch, Q_batch, A_batch ← batch
           loss ← compute_loss(x_batch, Q_batch, A_batch)
           
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
               x_batch, Q_batch, A_batch ← batch
               loss ← compute_loss(x_batch, Q_batch, A_batch)
               val_loss ← val_loss + loss.item()
       
       avg_val_loss ← val_loss / len(D_val)
       
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
       
       print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
   
5. Load best checkpoint:
   load_checkpoint(model)

6. Evaluate on test set:
   metrics ← evaluate(model, D_test)
   
7. Return model, metrics
```

### Loss Computation Details

For a single sample $(\mathbf{x}, Q, A)$:

**Step 1: Forward pass through model**
```
H_enc ← Encoder(normalize(x))                    # Shape: [N, d_enc]
H_proj ← Projector(H_enc)                        # Shape: [N, d_llm]
H_Q ← Embed(Tokenize(Q))                         # Shape: [M, d_llm]
H_A ← Embed(Tokenize(A))                         # Shape: [K, d_llm]
H_input ← concat([H_Q, H_proj, H_A], dim=0)     # Shape: [M+N+K, d_llm]
```

**Step 2: LLM forward pass**
```
logits ← LLM(H_input)                            # Shape: [M+N+K, |V|]
```

**Step 3: Extract answer token logits**
```
answer_start_idx ← M + N
answer_logits ← logits[answer_start_idx : answer_start_idx+K-1, :]
```

**Step 4: Compute cross-entropy loss**
```
A_tokens ← Tokenize(A)                           # [a_1, a_2, ..., a_K]
target_tokens ← A_tokens[1:]                     # Shift by 1 for next-token prediction

loss ← CrossEntropyLoss(answer_logits, target_tokens)
```

Mathematically:
$$
\mathcal{L} = -\frac{1}{K-1} \sum_{k=1}^{K-1} \log P(a_{k+1} | \mathbf{x}, Q, a_{\leq k}; \theta)
$$

---

## Evaluation

### Evaluation Metrics

**Primary Metric: Accuracy**

$$
\text{Accuracy} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{i=1}^{|\mathcal{D}_{\text{test}}|} \mathbb{1}[\hat{A}_i = A_i]
$$

where:
- $\hat{A}_i$: Predicted answer for sample $i$
- $A_i$: Ground truth answer
- $\mathbb{1}[\cdot]$: Indicator function (1 if true, 0 if false)

### Inference Process

**Generation Algorithm:**

For a test sample $(\mathbf{x}_{\text{test}}, Q_{\text{test}})$:

```
1. Preprocess:
   x_norm ← normalize(x_test)
   
2. Encode:
   H_enc ← Encoder(x_norm)
   H_proj ← Projector(H_enc)
   
3. Prepare prompt:
   H_Q ← Embed(Tokenize(Q_test))
   H_input ← concat([H_Q, H_proj], dim=0)
   
4. Generate answer tokens autoregressively:
   generated_tokens ← []
   current_input ← H_input
   
   for step = 1 to max_new_tokens:
       // Forward pass
       logits ← LLM(current_input)
       
       // Get next token probability
       next_token_logits ← logits[-1, :]
       next_token_prob ← Softmax(next_token_logits)
       
       // Sample or greedy decode
       if sampling:
           next_token ← sample(next_token_prob)
       else:
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
   A_pred ← Detokenize(generated_tokens)
   
6. Return A_pred
```

### Evaluation Algorithm

```
Function: evaluate_stage1(model, D_test)

Input: Trained model, test dataset D_test
Output: Accuracy score and predictions

1. Initialize:
   predictions ← []
   ground_truths ← []
   correct ← 0
   total ← 0
   
2. Set model to evaluation mode:
   model.eval()
   
3. Disable gradient computation:
   with torch.no_grad():
       
       for (x, Q, A) in D_test:
           // Generate prediction
           A_pred ← model.generate(x, Q, max_new_tokens=50)
           
           // Clean predictions
           A_pred_clean ← clean_text(A_pred)
           A_true_clean ← clean_text(A)
           
           // Store predictions
           predictions.append(A_pred_clean)
           ground_truths.append(A_true_clean)
           
           // Check if correct
           if A_pred_clean == A_true_clean:
               correct ← correct + 1
           
           total ← total + 1
   
4. Compute accuracy:
   accuracy ← correct / total
   
5. Save results:
   results ← {
       "predictions": predictions,
       "ground_truths": ground_truths,
       "accuracy": accuracy,
       "total_samples": total,
       "correct_predictions": correct
   }
   
   save_json(results, "stage1_mcq/results/metrics.json")
   save_jsonl(predictions, "stage1_mcq/results/test_predictions.jsonl")
   
6. Return accuracy, results
```

### Text Cleaning for Comparison

To ensure fair comparison, both predictions and ground truths are cleaned:

```
Function: clean_text(text)

1. Convert to lowercase: text ← text.lower()
2. Remove punctuation: text ← remove_punctuation(text)
3. Remove extra whitespace: text ← text.strip()
4. Remove special tokens: text ← remove_special_tokens(text)
5. Return text
```

### Expected Performance

**Typical Stage 1 Results:**

| Model | Accuracy | Training Time | GPU Memory |
|-------|----------|---------------|------------|
| **OpenTSLMSP** (Llama-3.2-1B) | 75-85% | ~2-3 hours | ~8GB |
| **OpenTSLMSP** (Gemma-3-270m) | 70-80% | ~1-2 hours | ~6GB |
| **OpenTSLMFlamingo** (Llama-3.2-1B) | 78-88% | ~3-4 hours | ~10GB |

*Note: Results vary based on random seed, hardware, and exact configuration*

---

## Implementation Algorithms

### Algorithm 1: Data Loading and Preprocessing

```
Function: load_and_preprocess_TSQA()

Output: D_train, D_val, D_test (preprocessed datasets)

1. Load raw dataset:
   D_raw ← load_dataset("ChengsenWang/TSQA", split="train")
   
2. Split dataset:
   D_temp, D_test ← train_test_split(D_raw, test_size=0.10, seed=42)
   D_train, D_val ← train_test_split(D_temp, test_size=0.09/0.90, seed=42)
   
3. For each split D in {D_train, D_val, D_test}:
   
   D_processed ← []
   
   for sample in D:
       // Extract fields
       x ← parse_json(sample["Series"])              // Convert JSON to array
       Q ← sample["Question"]
       A ← sample["Answer"]
       task ← sample["Task"]
       
       // Normalize time series
       μ ← mean(x)
       σ ← std(x)
       x_norm ← (x - μ) / (σ + 1e-8)
       
       // Create prompt
       pre_prompt ← Q
       time_series_text ← f"This is the time series, it has mean {μ:.4f} and std {σ:.4f}."
       post_prompt ← f"Predict the {task} Answer:"
       
       // Create processed sample
       processed_sample ← {
           "time_series": x_norm,
           "time_series_text": time_series_text,
           "pre_prompt": pre_prompt,
           "post_prompt": post_prompt,
           "answer": A
       }
       
       D_processed.append(processed_sample)
   
4. Return D_train, D_val, D_test
```

### Algorithm 2: Batch Collation with Padding

```
Function: collate_batch(samples, patch_size=4)

Input: List of samples [(x_1, Q_1, A_1), ..., (x_B, Q_B, A_B)]
Output: Batched and padded data

1. Find maximum length in batch:
   L_max ← max(len(x_i) for i in samples)
   
2. Round up to nearest multiple of patch_size:
   L_padded ← ceil(L_max / patch_size) × patch_size
   
3. Pad all time series:
   X_batch ← []
   Q_batch ← []
   A_batch ← []
   
   for (x, Q, A) in samples:
       // Pad time series
       if len(x) < L_padded:
           padding ← [0] × (L_padded - len(x))
           x_padded ← concat([x, padding])
       else:
           x_padded ← x[:L_padded]
       
       X_batch.append(x_padded)
       Q_batch.append(Q)
       A_batch.append(A)
   
4. Convert to tensors:
   X_batch ← torch.stack(X_batch)                   # Shape: [B, L_padded]
   
5. Return X_batch, Q_batch, A_batch
```

### Algorithm 3: Complete Training Step

```
Function: training_step(model, optimizer, scheduler, batch)

Input: Model, optimizer, scheduler, batch data
Output: Loss value

1. Unpack batch:
   X, Q_list, A_list ← batch                        # X: [B, L]
   
2. Zero gradients:
   optimizer.zero_grad()
   
3. Forward pass through encoder:
   H_enc ← model.encoder(X)                         # [B, N, d_enc]
   
4. Forward pass through projector:
   H_proj ← model.projector(H_enc)                  # [B, N, d_llm]
   
5. Process each sample in batch:
   total_loss ← 0
   
   for b = 1 to B:
       // Tokenize text
       Q_tokens ← tokenize(Q_list[b])
       A_tokens ← tokenize(A_list[b])
       
       // Embed text
       H_Q ← model.llm.embed(Q_tokens)              # [M_b, d_llm]
       H_A ← model.llm.embed(A_tokens)              # [K_b, d_llm]
       
       // Concatenate sequence
       H_input ← concat([H_Q, H_proj[b], H_A], dim=0)  # [M_b+N+K_b, d_llm]
       
       // Create labels (shifted by 1)
       labels ← create_labels(Q_tokens, A_tokens, M_b, N)
       
       // LLM forward pass
       logits ← model.llm(H_input)                  # [M_b+N+K_b, |V|]
       
       // Compute loss (only on answer tokens)
       answer_logits ← logits[M_b+N : M_b+N+K_b-1, :]
       answer_labels ← A_tokens[1:]
       
       loss_b ← CrossEntropyLoss(answer_logits, answer_labels)
       total_loss ← total_loss + loss_b
   
6. Average loss over batch:
   loss ← total_loss / B
   
7. Backward pass:
   loss.backward()
   
8. Gradient clipping:
   clip_grad_norm_(model.parameters(), max_norm=1.0)
   
9. Optimizer step:
   optimizer.step()
   scheduler.step()
   
10. Return loss.item()
```

### Algorithm 4: Model Checkpoint Saving

```
Function: save_checkpoint(model, optimizer, epoch, val_loss, save_path)

Input: Model, optimizer, epoch number, validation loss, save path

1. Create checkpoint dictionary:
   checkpoint ← {
       "epoch": epoch,
       "val_loss": val_loss,
       "encoder_state": model.encoder.state_dict(),
       "projector_state": model.projector.state_dict(),
   }
   
2. If LoRA is enabled:
   if model.lora_enabled:
       checkpoint["lora_state"] ← model.get_lora_state()
       checkpoint["lora_enabled"] ← True
   
3. If optimizer state should be saved:
   if save_optimizer:
       checkpoint["optimizer_state"] ← optimizer.state_dict()
   
4. Save to disk:
   torch.save(checkpoint, save_path)
   
5. Save loss history:
   loss_history_path ← save_path.replace("best_model.pt", "loss_history.txt")
   append_to_file(loss_history_path, f"Epoch {epoch}: {val_loss:.4f}\n")
   
6. Print confirmation:
   print(f"✓ Checkpoint saved: {save_path}")
```

---

## Summary

### Stage 1 Key Takeaways

1. **Purpose**: Foundation stage teaching basic time series understanding and QA
2. **Dataset**: TSQA with ~7,000 MCQ samples
3. **Architecture**: Encoder → Projector → LLM (frozen or LoRA)
4. **Training**: 30 epochs, early stopping, AdamW optimizer with warmup
5. **Metric**: Accuracy (75-85% typical)
6. **Output**: Trained model checkpoint for Stage 2 initialization

### Mathematical Components Summary

| Component | Mathematical Operation | Dimensionality |
|-----------|------------------------|----------------|
| **Input** | $\mathbf{x} \in \mathbb{R}^L$ | $[B, L]$ → time series |
| **Normalization** | $\tilde{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sigma + \epsilon}$ | $[B, L]$ → normalized |
| **Patch Embedding** | $\mathbf{P} = \text{Conv1D}(\tilde{\mathbf{x}})$ | $[B, L] \rightarrow [B, N, d_{\text{enc}}]$ |
| **Transformer Encoder** | $\mathbf{H}_{\text{enc}} = \text{TransformerEncoder}(\mathbf{P})$ | $[B, N, d_{\text{enc}}] \rightarrow [B, N, d_{\text{enc}}]$ |
| **Projector** | $\mathbf{H}_{\text{proj}} = \text{MLP}(\mathbf{H}_{\text{enc}})$ | $[B, N, d_{\text{enc}}] \rightarrow [B, N, d_{\text{llm}}]$ |
| **LLM** | $\mathbf{L} = \text{LLM}([\mathbf{H}_Q; \mathbf{H}_{\text{proj}}; \mathbf{H}_A])$ | $[B, M+N+K, d_{\text{llm}}] \rightarrow [B, M+N+K, |\mathcal{V}|]$ |
| **Loss** | $\mathcal{L} = -\frac{1}{K} \sum_{k=1}^{K} \log P(a_k \mid \cdot)$ | Scalar |

### Next Steps

After completing Stage 1:
1. Model checkpoint saved to `results/{llm_id}/OpenTSLMSP/stage1_mcq/checkpoints/best_model.pt`
2. Evaluation metrics saved to `results/{llm_id}/OpenTSLMSP/stage1_mcq/results/metrics.json`
3. Model ready for Stage 2 (Captioning) initialization
4. Encoder and projector have learned basic time series → embedding mapping
5. Foundation established for more complex reasoning tasks

---

## References

- **TSQA Dataset**: ChengsenWang/TSQA on Hugging Face
- **OpenTSLM Paper**: [Link to paper](https://doi.org/10.13140/RG.2.2.14827.60963)
- **Implementation**: See `curriculum_learning.py` and `src/time_series_datasets/TSQADataset.py`

---

**End of Stage 1 Detailed Guide**


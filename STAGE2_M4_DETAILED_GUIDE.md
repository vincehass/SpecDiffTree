<!--
This source file is part of the OpenTSLM open-source project

SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

# Stage 2: M4 Captioning - Detailed Mathematical Guide

**A Comprehensive Mathematical and Algorithmic Explanation of Stage 2 Training**

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset: M4 Time Series Captioning](#dataset-m4-time-series-captioning)
3. [Mathematical Formulation](#mathematical-formulation)
4. [Model Architecture](#model-architecture)
5. [Training Process](#training-process)
6. [Evaluation](#evaluation)
7. [Implementation Algorithms](#implementation-algorithms)
8. [Comparison with Stage 1](#comparison-with-stage-1)

---

## Overview

### Stage 2 Purpose

Stage 2 (M4 Captioning) is the **generative language stage** of the OpenTSLM curriculum learning pipeline. Building on Stage 1's foundation, it teaches the model:

1. **Detailed language generation**: How to produce fluent, detailed descriptions
2. **Pattern description**: How to articulate time series patterns in natural language
3. **Long-form text generation**: How to generate multi-sentence explanations
4. **Domain adaptation**: How to describe various time series domains (economic, demographic, etc.)

**Why Stage 2 After Stage 1?**
- Stage 1 taught basic time series understanding (classification/MCQ)
- Stage 2 extends to **generation** - producing full descriptions
- Transitions from discriminative (selecting answers) to generative (creating text)
- Prepares for chain-of-thought reasoning in later stages

**Key Statistics:**
- **Dataset**: M4 Time Series Caption Dataset (custom-created)
- **Samples**: ~23,000 total across all frequencies (18,400 train / 2,300 val / 2,300 test)
- **Task Type**: Caption Generation (free-form text)
- **Time Series**: Variable length (12 to 2,794 points depending on frequency)
- **Captions**: ChatGPT-generated detailed descriptions
- **Training Epochs**: 20 (with early stopping)
- **Metric**: Perplexity and Test Loss

---

## Dataset: M4 Time Series Captioning

### Dataset Description

The M4 Time Series Caption Dataset combines time series from the M4 forecasting competition with AI-generated captions. Each sample consists of:

**Input:**
- A univariate time series: $\mathbf{x} = [x_1, x_2, \ldots, x_L] \in \mathbb{R}^L$
- A task description: "Please generate a detailed caption..."
- Frequency metadata: $f \in \{\text{Yearly, Quarterly, Monthly, Weekly, Daily, Hourly}\}$

**Output:**
- A detailed caption: $C$ (natural language description, 50-300 words)

### Data Format

Each sample in the M4 dataset has the following structure:

```json
{
  "id": "M1234",                           // Unique M4 series ID
  "frequency": "Monthly",                  // Temporal frequency
  "series": [45.2, 46.8, 47.1, ...],      // Time series values
  "caption": "This monthly time series shows a clear upward trend..."  // Caption
}
```

### Mathematical Representation

Let $\mathcal{D}_{\text{M4}} = \{(\mathbf{x}_i, f_i, C_i)\}_{i=1}^N$ be the dataset with $N$ samples, where:

- $\mathbf{x}_i \in \mathbb{R}^{L_i}$: Time series of length $L_i$ (varies by frequency)
- $f_i \in \mathcal{F}$: Frequency from set $\mathcal{F} = \{\text{Yearly}, \text{Quarterly}, \text{Monthly}, \text{Weekly}, \text{Daily}, \text{Hourly}\}$
- $C_i \in \mathcal{V}^*$: Caption (sequence of tokens from vocabulary $\mathcal{V}$)

### Data Splits

The dataset is split into three subsets:

$$
\mathcal{D}_{\text{M4}} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}
$$

With proportions:
- $|\mathcal{D}_{\text{train}}| = 0.81 \times N \approx 18,400$ samples
- $|\mathcal{D}_{\text{val}}| = 0.09 \times N \approx 2,300$ samples
- $|\mathcal{D}_{\text{test}}| = 0.10 \times N \approx 2,300$ samples

**Splitting Algorithm:**
```
1. Load all M4 data across frequencies: D_full
2. Concatenate all frequency datasets
3. Shuffle D_full with seed=42 for reproducibility
4. Split: D_temp, D_test = split(D_full, test_size=0.10)
5. Split: D_train, D_val = split(D_temp, test_size=0.09/0.90)
6. Return D_train, D_val, D_test
```

### Frequency Distribution

The dataset contains time series from multiple temporal frequencies:

| Frequency | Typical Length | Number of Samples | Percentage |
|-----------|----------------|-------------------|------------|
| **Yearly** | 12-50 | ~1,000 | ~4% |
| **Quarterly** | 20-100 | ~3,000 | ~13% |
| **Monthly** | 50-200 | ~8,000 | ~35% |
| **Weekly** | 100-400 | ~6,000 | ~26% |
| **Daily** | 200-2,000 | ~4,000 | ~17% |
| **Hourly** | 500-2,794 | ~1,000 | ~4% |

**Length Statistics:**
- Minimum length: 12 (Yearly)
- Maximum length: 2,794 (Hourly)
- Median length: 150-200
- Mean length: ~300

### Caption Characteristics

**Caption Length Distribution:**

Let $\ell(C_i)$ be the length of caption $C_i$ in tokens. Statistics:

$$
\mathbb{E}[\ell(C)] \approx 100 \text{ tokens} \quad (\text{mean})
$$

$$
\sigma[\ell(C)] \approx 40 \text{ tokens} \quad (\text{std})
$$

$$
\min(\ell(C)) \approx 30 \text{ tokens}, \quad \max(\ell(C)) \approx 400 \text{ tokens}
$$

**Caption Structure:**

Typical captions contain:
1. **Overview**: General description of the time series (1-2 sentences)
2. **Trend Analysis**: Long-term patterns (increasing/decreasing/stable)
3. **Seasonality**: Periodic patterns if present
4. **Volatility**: Variability and fluctuations
5. **Notable Features**: Anomalies, outliers, regime changes
6. **Context**: Domain-specific interpretation when applicable

**Example Caption:**
```
"This monthly time series exhibits a clear upward trend over the observed period, 
starting from a baseline around 45 and reaching values near 95 by the end. The series 
shows moderate volatility with occasional sharp increases followed by brief corrections. 
A notable feature is the accelerated growth in the latter half of the series. The 
overall pattern suggests consistent expansion with cyclical fluctuations superimposed 
on the underlying trend. No major anomalies or structural breaks are evident."
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

**2. Padding for Batch Processing:**

Time series have variable lengths $L_1, L_2, \ldots, L_B$ in a batch. We pad to the maximum length that is divisible by patch size $P = 4$:

$$
L_{\text{max}} = \left\lceil \frac{\max(L_1, \ldots, L_B)}{P} \right\rceil \times P
$$

For series $i$ with $L_i < L_{\text{max}}$:

$$
\mathbf{x}_i^{\text{padded}} = [\tilde{x}_1, \ldots, \tilde{x}_{L_i}, \underbrace{0, \ldots, 0}_{L_{\text{max}} - L_i}]
$$

**3. Prompt Construction:**

Each sample is formatted as a captioning prompt:

$$
\text{Prompt}_i = P_{\text{pre}} \oplus T_i \oplus P_{\text{post}}
$$

where:
- $P_{\text{pre}} = \text{"You are an expert in time series analysis."}$
- $T_i = \text{"This is the time series, it has mean } \mu_i \text{ and std } \sigma_i\text{:"}$
- $P_{\text{post}} = \text{"Please generate a detailed caption for this time-series, describing it as accurately as possible."}$
- $\oplus$ denotes string concatenation

**4. Caption Tokenization:**

Captions are tokenized using the LLM's tokenizer:

$$
C = \text{"This monthly time series..."} \rightarrow [c_1, c_2, \ldots, c_K]
$$

where $K$ varies per caption (typically 50-300 tokens).

---

## Mathematical Formulation

### Problem Formulation

Given:
- Time series: $\mathbf{x} \in \mathbb{R}^L$
- Task prompt: $P = [p_1, p_2, \ldots, p_M]$ (sequence of $M$ tokens)

Goal:
- Generate detailed caption: $\hat{C} = [\hat{c}_1, \hat{c}_2, \ldots, \hat{c}_K]$ (sequence of $K$ tokens)

This is a **conditional text generation** problem.

### Model Function

The model learns a conditional probability distribution:

$$
P_\theta(C | \mathbf{x}) = \prod_{k=1}^{K} P_\theta(c_k | \mathbf{x}, c_{<k})
$$

where:
- $\theta$: All trainable parameters
- $c_{<k} = [c_1, \ldots, c_{k-1}]$: All previous tokens

The model function is:

$$
f_\theta: (\mathbb{R}^L, \mathcal{V}^M) \rightarrow \Delta^{|\mathcal{V}|}
$$

where $\Delta^{|\mathcal{V}|}$ is the probability simplex over the vocabulary.

### Loss Function

For caption generation, we use **causal language modeling loss** (cross-entropy):

$$
\mathcal{L}(\theta; \mathbf{x}, C) = -\frac{1}{K} \sum_{k=1}^{K} \log P_\theta(c_k | \mathbf{x}, c_{<k})
$$

Expanded form:

$$
\mathcal{L}(\theta; \mathbf{x}, C) = -\frac{1}{K} \sum_{k=1}^{K} \log \frac{\exp(z_{c_k})}{\sum_{v \in \mathcal{V}} \exp(z_v)}
$$

where $z_v$ is the logit for vocabulary token $v$.

### Total Training Objective

Over the entire training dataset $\mathcal{D}_{\text{train}}$:

$$
\mathcal{L}_{\text{total}}(\theta) = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(\mathbf{x}_i, C_i) \in \mathcal{D}_{\text{train}}} \mathcal{L}(\theta; \mathbf{x}_i, C_i)
$$

### Optimization Objective

$$
\theta^* = \arg\min_{\theta} \mathcal{L}_{\text{total}}(\theta) + \lambda \|\theta\|_2^2
$$

where $\lambda = 10^{-2}$ is the weight decay (L2 regularization) coefficient.

### Perplexity Metric

Perplexity is used to evaluate the quality of caption generation:

$$
\text{Perplexity}(\mathcal{D}) = \exp\left(\mathcal{L}_{\text{total}}(\theta)\right)
$$

Lower perplexity indicates better caption quality. Interpretation:
- Perplexity of 10: Model effectively chooses among ~10 likely tokens per position
- Perplexity of 100: Model uncertain among ~100 tokens per position

---

## Model Architecture

The architecture for Stage 2 is **identical** to Stage 1, but with different training objectives and data. The model consists of three main components:

### 1. Time Series Encoder

**Architecture**: Transformer-CNN Encoder (same as Stage 1)

**Input**: $\mathbf{x} \in \mathbb{R}^{B \times L}$ (batch of time series)

**Output**: $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times N \times d_{\text{enc}}}$ (encoded features)

where:
- $B$: Batch size
- $L$: Time series length (padded)
- $N = L / P$: Number of patches
- $P = 4$: Patch size
- $d_{\text{enc}} = 128$: Encoder embedding dimension

#### Encoder Operations

**Mathematical Formulation:**

$$
\mathbf{H}_{\text{enc}} = \text{TransformerEncoder}(\text{PosEnc}(\text{PatchEmbed}(\mathbf{x})))
$$

Detailed steps:

1. **Patch Embedding**: $\mathbf{P} = \text{Conv1D}(\mathbf{x}) \in \mathbb{R}^{B \times N \times d_{\text{enc}}}$

2. **Positional Encoding**: $\mathbf{P}' = \mathbf{P} + \mathbf{E}_{\text{pos}}[:N, :] \in \mathbb{R}^{B \times N \times d_{\text{enc}}}$

3. **Normalization**: $\mathbf{P}'' = \text{Dropout}(\text{LayerNorm}(\mathbf{P}'))$

4. **Transformer Layers**: $\mathbf{H}_{\text{enc}} = \text{TransformerEncoder}_{L=6}(\mathbf{P}'')$

### 2. Projector

**Architecture**: MLP with LayerNorm and GELU (same as Stage 1)

**Input**: $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times N \times d_{\text{enc}}}$

**Output**: $\mathbf{H}_{\text{proj}} \in \mathbb{R}^{B \times N \times d_{\text{llm}}}$

**Mathematical Formulation:**

$$
\mathbf{H}_{\text{proj}} = \text{Dropout}(\text{GELU}(\text{Linear}(\text{LayerNorm}(\mathbf{H}_{\text{enc}}))))
$$

### 3. Large Language Model (LLM)

**Architecture**: Pre-trained causal LM (Llama/Gemma, same as Stage 1)

**Input**: Combined sequence of projected time series embeddings and text tokens

**Output**: Probability distribution over vocabulary for next token prediction

#### Input Sequence Construction for Captioning

For a sample $(\mathbf{x}, C)$:

**1. Text Tokenization:**

$$
P_{\text{tokens}} = \text{Tokenize}(P_{\text{pre}} + T + P_{\text{post}}) = [p_1, \ldots, p_M]
$$

$$
C_{\text{tokens}} = \text{Tokenize}(C) = [c_1, c_2, \ldots, c_K]
$$

**2. Text Embedding:**

$$
\mathbf{H}_P = \text{Embed}_{\text{LLM}}(P_{\text{tokens}}) \in \mathbb{R}^{M \times d_{\text{llm}}}
$$

$$
\mathbf{H}_C = \text{Embed}_{\text{LLM}}(C_{\text{tokens}}) \in \mathbb{R}^{K \times d_{\text{llm}}}
$$

**3. Sequence Concatenation:**

$$
\mathbf{H}_{\text{input}} = [\mathbf{H}_P; \mathbf{H}_{\text{proj}}; \mathbf{H}_C] \in \mathbb{R}^{(M+N+K) \times d_{\text{llm}}}
$$

**Total sequence length**: $T = M + N + K$ (typically 200-500 tokens for M4)

#### Autoregressive Generation

During training, the model learns:

$$
P_\theta(c_k | \mathbf{x}, c_{<k}) = \text{Softmax}(\text{LLM}([\mathbf{H}_P; \mathbf{H}_{\text{proj}}; \mathbf{H}_{c_{<k}}]))_k
$$

where $\mathbf{H}_{c_{<k}}$ are embeddings of tokens $c_1, \ldots, c_{k-1}$.

### Model Initialization for Stage 2

**Key Difference from Stage 1:**

Stage 2 **initializes** from the best Stage 1 checkpoint:

$$
\theta_{\text{Stage2}}^{(0)} = \theta_{\text{Stage1}}^*
$$

where $\theta_{\text{Stage1}}^*$ is the best model from Stage 1 training.

This provides:
1. **Pre-trained encoder**: Already understands basic time series patterns
2. **Pre-trained projector**: Already maps time series to LLM space
3. **Warm start**: Faster convergence and better performance

### Parameter Count

**Trainable Parameters (Stage 2):**

| Component | Parameters | Trainable | Notes |
|-----------|-----------|-----------|-------|
| **Encoder** | ~5M | ✅ Yes | Fine-tuned from Stage 1 |
| **Projector** | ~260K | ✅ Yes | Fine-tuned from Stage 1 |
| **LLM (Llama-3.2-1B)** | ~1.2B | ❌ No (frozen) | Or ✅ with LoRA |
| **LoRA (if enabled)** | ~2-4M | ✅ Yes | Optional LLM fine-tuning |
| **Total Trainable** | ~5.3M (or ~9M with LoRA) | | |

---

## Training Process

### Training Configuration

**Hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 20 | Maximum training epochs |
| Batch Size | 4 | Samples per batch |
| Learning Rate (Encoder) | $2 \times 10^{-4}$ | LR for encoder |
| Learning Rate (Projector) | $1 \times 10^{-4}$ | LR for projector |
| Learning Rate (LoRA) | $2 \times 10^{-4}$ | LR for LoRA adapters (if enabled) |
| Weight Decay | $1 \times 10^{-2}$ | L2 regularization |
| Gradient Clipping | 1.0 | Max gradient norm |
| Warmup Fraction | 0.03 | Fraction of steps for LR warmup |
| Early Stopping Patience | 5 | Epochs without improvement to stop |
| Patch Size | 4 | Time series patch size |

**Note**: Learning rates are kept the same as Stage 1 because we're continuing curriculum learning (not pure fine-tuning).

### Curriculum Learning Connection

Stage 2 builds on Stage 1:

$$
\theta_{\text{Stage2}}^{(0)} \leftarrow \text{BestCheckpoint}(\text{Stage1})
$$

This enables:
1. **Knowledge transfer**: Basic time series understanding transfers
2. **Faster convergence**: Model starts from a good initialization
3. **Better performance**: Curriculum approach outperforms training from scratch

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

**AdamW Optimizer** (same as Stage 1):

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

**Algorithm: Stage 2 Training**

```
Input: Training data D_train, validation data D_val, test data D_test
       Stage 1 best checkpoint θ_stage1
Output: Trained model parameters θ*

1. Load Stage 1 checkpoint:
   encoder_params ← θ_stage1.encoder
   projector_params ← θ_stage1.projector
   lora_params ← θ_stage1.lora (if enabled)
   
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
   
   for epoch = 1 to max_epochs:
       // Training phase
       model.train()
       train_loss ← 0
       
       for batch in DataLoader(D_train, batch_size=4, shuffle=True):
           // Forward pass
           x_batch, C_batch ← batch
           loss ← compute_caption_loss(x_batch, C_batch)
           
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
               x_batch, C_batch ← batch
               loss ← compute_caption_loss(x_batch, C_batch)
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
   test_metrics ← evaluate_captions(model, D_test)
   
7. Return model, test_metrics
```

### Loss Computation for Captioning

For a single sample $(\mathbf{x}, C)$:

**Step 1: Forward pass through model**
```
H_enc ← Encoder(normalize(x))                      # Shape: [N, d_enc]
H_proj ← Projector(H_enc)                          # Shape: [N, d_llm]
H_P ← Embed(Tokenize(P))                           # Shape: [M, d_llm]
H_C ← Embed(Tokenize(C))                           # Shape: [K, d_llm]
H_input ← concat([H_P, H_proj, H_C], dim=0)       # Shape: [M+N+K, d_llm]
```

**Step 2: LLM forward pass**
```
logits ← LLM(H_input)                              # Shape: [M+N+K, |V|]
```

**Step 3: Extract caption token logits**
```
caption_start_idx ← M + N
caption_logits ← logits[caption_start_idx : caption_start_idx+K-1, :]  # Shape: [K-1, |V|]
```

**Step 4: Compute cross-entropy loss**
```
C_tokens ← Tokenize(C)                             # [c_1, c_2, ..., c_K]
target_tokens ← C_tokens[1:]                       # Shift by 1 for next-token prediction

loss ← CrossEntropyLoss(caption_logits, target_tokens)
```

Mathematically:
$$
\mathcal{L} = -\frac{1}{K-1} \sum_{k=1}^{K-1} \log P_\theta(c_{k+1} | \mathbf{x}, c_{\leq k})
$$

---

## Evaluation

### Evaluation Metrics

**Primary Metrics:**

1. **Test Loss** (Cross-Entropy):
   $$
   \mathcal{L}_{\text{test}} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{(\mathbf{x}, C) \in \mathcal{D}_{\text{test}}} \mathcal{L}(\theta; \mathbf{x}, C)
   $$

2. **Perplexity**:
   $$
   \text{PPL} = \exp(\mathcal{L}_{\text{test}})
   $$

**Interpretation:**
- **Lower test loss** = Better caption generation quality
- **Lower perplexity** = More confident and coherent predictions
- Typical good perplexity: 10-30 for this task

**Secondary Metrics (Optional):**

For qualitative evaluation, compute:

3. **BLEU Score** (n-gram overlap with reference):
   $$
   \text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)
   $$
   where $p_n$ is the n-gram precision.

4. **ROUGE Score** (recall-oriented):
   $$
   \text{ROUGE-L} = \frac{2 \cdot P \cdot R}{P + R}
   $$
   where $P$ and $R$ are precision and recall of longest common subsequence.

5. **Caption Length Accuracy**:
   $$
   \text{Length Ratio} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{i=1}^{|\mathcal{D}_{\text{test}}|} \frac{|\hat{C}_i|}{|C_i|}
   $$

### Inference Process

**Caption Generation Algorithm:**

For a test sample $(\mathbf{x}_{\text{test}})$:

```
1. Preprocess:
   x_norm ← normalize(x_test)
   
2. Encode:
   H_enc ← Encoder(x_norm)
   H_proj ← Projector(H_enc)
   
3. Prepare prompt:
   P ← "You are an expert... [full prompt]"
   H_P ← Embed(Tokenize(P))
   H_input ← concat([H_P, H_proj], dim=0)
   
4. Generate caption tokens autoregressively:
   generated_tokens ← []
   current_input ← H_input
   
   for step = 1 to max_new_tokens (e.g., 300):
       // Forward pass
       logits ← LLM(current_input)
       
       // Get next token probability
       next_token_logits ← logits[-1, :]
       next_token_prob ← Softmax(next_token_logits / temperature)
       
       // Sample with nucleus sampling (top-p)
       if use_nucleus_sampling:
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
   C_pred ← Detokenize(generated_tokens)
   
6. Return C_pred
```

**Nucleus Sampling (Top-p):**

Instead of always picking the highest probability token (greedy), use nucleus sampling for more diverse captions:

$$
\mathcal{V}_p = \min\left\{V' \subseteq \mathcal{V} : \sum_{v \in V'} P(v) \geq p\right\}
$$

Sample $v \sim P(\cdot | v \in \mathcal{V}_p)$ with typical $p = 0.9$.

### Evaluation Algorithm

```
Function: evaluate_stage2(model, D_test)

Input: Trained model, test dataset D_test
Output: Test loss, perplexity, and sample predictions

1. Initialize:
   total_loss ← 0
   total_tokens ← 0
   predictions ← []
   ground_truths ← []
   
2. Set model to evaluation mode:
   model.eval()
   
3. Disable gradient computation:
   with torch.no_grad():
       
       for (x, C) in D_test:
           // Compute loss
           loss ← compute_caption_loss(x, C)
           K ← len(Tokenize(C))
           
           total_loss ← total_loss + loss.item() × K
           total_tokens ← total_tokens + K
           
           // Generate prediction (for qualitative analysis)
           if sample_predictions:
               C_pred ← model.generate(x, max_new_tokens=300)
               predictions.append(C_pred)
               ground_truths.append(C)
   
4. Compute metrics:
   avg_test_loss ← total_loss / total_tokens
   perplexity ← exp(avg_test_loss)
   
5. Save results:
   results ← {
       "test_loss": avg_test_loss,
       "perplexity": perplexity,
       "total_samples": len(D_test),
       "total_tokens": total_tokens
   }
   
   if sample_predictions:
       results["sample_predictions"] ← predictions[:100]
       results["sample_ground_truths"] ← ground_truths[:100]
   
   save_json(results, "stage2_captioning/results/metrics.json")
   save_jsonl(predictions, "stage2_captioning/results/test_predictions.jsonl")
   
6. Print summary:
   print(f"Test Loss: {avg_test_loss:.4f}")
   print(f"Perplexity: {perplexity:.2f}")
   print(f"Avg tokens per caption: {total_tokens/len(D_test):.1f}")
   
7. Return results
```

### Expected Performance

**Typical Stage 2 Results:**

| Model | Test Loss | Perplexity | Training Time | GPU Memory |
|-------|-----------|------------|---------------|------------|
| **OpenTSLMSP** (Llama-3.2-1B) | 2.5-3.0 | 12-20 | ~4-6 hours | ~8-10GB |
| **OpenTSLMSP** (Gemma-3-270m) | 2.8-3.3 | 16-27 | ~2-4 hours | ~6-8GB |
| **OpenTSLMFlamingo** (Llama-3.2-1B) | 2.3-2.8 | 10-16 | ~6-8 hours | ~10-12GB |

*Note: Results vary based on random seed, hardware, and exact configuration*

**Perplexity Interpretation:**
- **PPL < 15**: Excellent - captions are coherent and detailed
- **PPL 15-25**: Good - captions are generally fluent with minor issues
- **PPL 25-40**: Acceptable - captions capture main points but may lack detail
- **PPL > 40**: Poor - captions may be incoherent or generic

---

## Implementation Algorithms

### Algorithm 1: M4 Data Loading and Preprocessing

```
Function: load_and_preprocess_M4()

Output: D_train, D_val, D_test (preprocessed datasets)

1. Download and extract M4 dataset (if not present):
   if not exists(M4_DATA_DIR):
       download_from_url(M4_RELEASE_URL)
       extract_zip(M4_ZIP_FILE, M4_DATA_DIR)
   
2. Load data for all frequencies:
   all_samples ← []
   
   for frequency in [Yearly, Quarterly, Monthly, Weekly, Daily, Hourly]:
       // Load series and captions
       series_df ← read_csv(f"m4_series_{frequency}.csv")
       captions_df ← read_csv(f"m4_captions_{frequency}.csv")
       
       // Merge on ID
       merged_df ← merge(series_df, captions_df, on="id")
       
       // Convert to samples
       for row in merged_df:
           series ← parse_json(row["series"])
           caption ← row["caption"]
           
           // Create sample
           sample ← {
               "id": row["id"],
               "frequency": frequency,
               "series": series,
               "caption": caption
           }
           all_samples.append(sample)
   
3. Create dataset:
   D_full ← Dataset.from_list(all_samples)
   
4. Split dataset:
   D_temp, D_test ← train_test_split(D_full, test_size=0.10, seed=42)
   D_train, D_val ← train_test_split(D_temp, test_size=0.09/0.90, seed=43)
   
5. Preprocess each split:
   for D in [D_train, D_val, D_test]:
       D_processed ← []
       
       for sample in D:
           // Extract fields
           x ← sample["series"]
           C ← sample["caption"]
           freq ← sample["frequency"]
           
           // Normalize time series
           μ ← mean(x)
           σ ← std(x)
           x_norm ← (x - μ) / (σ + 1e-8)
           
           // Create prompts
           pre_prompt ← "You are an expert in time series analysis."
           time_series_text ← f"This is the time series, it has mean {μ:.4f} and std {σ:.4f}:"
           post_prompt ← "Please generate a detailed caption for this time-series, describing it as accurately as possible."
           
           // Create processed sample
           processed_sample ← {
               "time_series": x_norm,
               "time_series_text": time_series_text,
               "pre_prompt": pre_prompt,
               "post_prompt": post_prompt,
               "answer": C,
               "frequency": freq,
               "id": sample["id"]
           }
           
           D_processed.append(processed_sample)
   
6. Print statistics:
   print(f"Train: {len(D_train)} samples")
   print(f"Val: {len(D_val)} samples")
   print(f"Test: {len(D_test)} samples")
   print(f"Total: {len(D_train) + len(D_val) + len(D_test)} samples")
   
7. Return D_train, D_val, D_test
```

### Algorithm 2: Caption Loss Computation

```
Function: compute_caption_loss(model, x, caption)

Input: Model, time series x, caption text
Output: Loss value

1. Normalize time series:
   x_norm ← (x - mean(x)) / (std(x) + 1e-8)
   
2. Encode time series:
   H_enc ← model.encoder(x_norm)                   # [N, d_enc]
   H_proj ← model.projector(H_enc)                 # [N, d_llm]
   
3. Prepare prompt text:
   P ← create_prompt(x)  # Includes pre-prompt, TS info, post-prompt
   
4. Tokenize:
   P_tokens ← tokenize(P)                          # [p_1, ..., p_M]
   C_tokens ← tokenize(caption)                    # [c_1, ..., c_K]
   
5. Embed text:
   H_P ← model.llm.embed(P_tokens)                 # [M, d_llm]
   H_C ← model.llm.embed(C_tokens)                 # [K, d_llm]
   
6. Create input sequence:
   H_input ← concat([H_P, H_proj, H_C], dim=0)    # [M+N+K, d_llm]
   
7. Create labels:
   // Labels are -100 for non-caption tokens (ignored in loss)
   labels ← [-100] × (M + N) + C_tokens[1:]       # [M+N+K-1]
   
8. LLM forward pass:
   logits ← model.llm(H_input)                     # [M+N+K, |V|]
   
9. Compute cross-entropy loss:
   // Only compute loss on caption tokens
   caption_logits ← logits[M+N:M+N+K-1, :]        # [K-1, |V|]
   caption_targets ← C_tokens[1:]                  # [K-1]
   
   loss ← CrossEntropyLoss(caption_logits, caption_targets)
   
10. Return loss
```

### Algorithm 3: Caption Generation with Nucleus Sampling

```
Function: generate_caption(model, x, max_tokens=300, temperature=1.0, top_p=0.9)

Input: Model, time series x, generation parameters
Output: Generated caption text

1. Normalize and encode:
   x_norm ← (x - mean(x)) / (std(x) + 1e-8)
   H_enc ← model.encoder(x_norm)
   H_proj ← model.projector(H_enc)
   
2. Prepare prompt:
   P ← create_prompt(x)
   P_tokens ← tokenize(P)
   H_P ← model.llm.embed(P_tokens)
   
3. Initialize:
   H_input ← concat([H_P, H_proj], dim=0)         # [M+N, d_llm]
   generated_tokens ← []
   
4. Autoregressive generation:
   for step = 1 to max_tokens:
       // Forward pass
       logits ← model.llm(H_input)[-1, :]         # [|V|]
       
       // Apply temperature
       logits ← logits / temperature
       
       // Compute probabilities
       probs ← Softmax(logits)
       
       // Nucleus sampling (top-p)
       sorted_probs, sorted_indices ← sort_descending(probs)
       cumsum_probs ← cumulative_sum(sorted_probs)
       
       // Find nucleus
       nucleus_mask ← cumsum_probs <= top_p
       if not any(nucleus_mask):
           nucleus_mask[0] ← True  # Always keep at least top token
       
       nucleus_indices ← sorted_indices[nucleus_mask]
       nucleus_probs ← sorted_probs[nucleus_mask]
       
       // Renormalize and sample
       nucleus_probs ← nucleus_probs / sum(nucleus_probs)
       next_token ← sample(nucleus_indices, p=nucleus_probs)
       
       // Check for EOS
       if next_token == EOS_token:
           break
       
       // Append token
       generated_tokens.append(next_token)
       
       // Update input
       next_token_emb ← model.llm.embed(next_token)
       H_input ← concat([H_input, next_token_emb], dim=0)
   
5. Decode to text:
   caption ← detokenize(generated_tokens)
   
6. Post-process:
   caption ← caption.strip()
   
7. Return caption
```

### Algorithm 4: Batch Processing with Variable Lengths

```
Function: collate_m4_batch(samples, patch_size=4)

Input: List of samples [(x_1, C_1), ..., (x_B, C_B)]
Output: Batched and padded data

1. Find maximum time series length:
   L_max_ts ← max(len(x_i) for (x_i, C_i) in samples)
   L_padded_ts ← ceil(L_max_ts / patch_size) × patch_size
   
2. Find maximum caption length:
   L_max_cap ← max(len(tokenize(C_i)) for (x_i, C_i) in samples)
   
3. Pad time series:
   X_batch ← []
   for (x, C) in samples:
       if len(x) < L_padded_ts:
           padding ← [0] × (L_padded_ts - len(x))
           x_padded ← concat([x, padding])
       else:
           x_padded ← x[:L_padded_ts]
       X_batch.append(x_padded)
   
4. Collect captions:
   C_batch ← [C for (x, C) in samples]
   
5. Convert to tensors:
   X_batch ← torch.stack(X_batch)                  # [B, L_padded_ts]
   
6. Create metadata:
   metadata ← {
       "max_ts_length": L_padded_ts,
       "max_caption_length": L_max_cap,
       "frequencies": [sample["frequency"] for sample in samples],
       "ids": [sample["id"] for sample in samples]
   }
   
7. Return X_batch, C_batch, metadata
```

---

## Comparison with Stage 1

### Key Differences

| Aspect | Stage 1 (TSQA) | Stage 2 (M4 Captioning) |
|--------|----------------|-------------------------|
| **Task** | Multiple Choice QA | Free-form Caption Generation |
| **Output** | Short answer (1-3 words) | Long caption (50-300 tokens) |
| **Metric** | Accuracy | Test Loss / Perplexity |
| **Evaluation** | Exact match | Language modeling quality |
| **Generation** | Greedy decode sufficient | Nucleus sampling recommended |
| **Caption Length** | ~2-5 tokens | ~100-150 tokens |
| **Dataset Size** | ~7,000 samples | ~23,000 samples |
| **Frequencies** | N/A | 6 frequencies (Yearly to Hourly) |
| **Learning Focus** | Classification | Generation |

### Similarities

1. **Architecture**: Identical encoder, projector, and LLM
2. **Loss Function**: Both use cross-entropy (but different scope)
3. **Optimization**: Same AdamW with warmup schedule
4. **Initialization**: Stage 2 loads Stage 1's best checkpoint
5. **Hyperparameters**: Similar learning rates and training config

### Why This Progression?

**Curriculum Learning Justification:**

$$
\text{Stage 1 (MCQ)} \rightarrow \text{Stage 2 (Captioning)} \rightarrow \text{Stage 3+ (CoT)}
$$

1. **Stage 1**: Teaches basic time series understanding
   - Model learns: "What pattern does this show?"
   - Output: Simple classification

2. **Stage 2**: Teaches detailed language generation
   - Model learns: "How do I describe this pattern in detail?"
   - Output: Fluent, multi-sentence descriptions

3. **Stage 3+**: Teaches reasoning and explanation
   - Model learns: "Why is this the answer? Show your work."
   - Output: Chain-of-thought rationales

This gradual increase in complexity is key to curriculum learning success.

---

## Summary

### Stage 2 Key Takeaways

1. **Purpose**: Generation stage teaching detailed caption creation
2. **Dataset**: M4 with ~23,000 captioned time series across 6 frequencies
3. **Architecture**: Same as Stage 1, initialized from Stage 1 checkpoint
4. **Training**: 20 epochs, early stopping, focus on test loss/perplexity
5. **Metric**: Perplexity (10-20 typical for good performance)
6. **Output**: Trained model checkpoint for Stage 3 initialization

### Mathematical Components Summary

| Component | Mathematical Operation | Dimensionality |
|-----------|------------------------|----------------|
| **Input** | $\mathbf{x} \in \mathbb{R}^L$ | $[B, L]$ → time series |
| **Normalization** | $\tilde{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sigma + \epsilon}$ | $[B, L]$ → normalized |
| **Encoder** | $\mathbf{H}_{\text{enc}} = \text{TransformerEncoder}(\text{PatchEmbed}(\tilde{\mathbf{x}}))$ | $[B, L] \rightarrow [B, N, d_{\text{enc}}]$ |
| **Projector** | $\mathbf{H}_{\text{proj}} = \text{MLP}(\mathbf{H}_{\text{enc}})$ | $[B, N, d_{\text{enc}}] \rightarrow [B, N, d_{\text{llm}}]$ |
| **LLM** | $\mathbf{L} = \text{LLM}([\mathbf{H}_P; \mathbf{H}_{\text{proj}}; \mathbf{H}_C])$ | $[B, M+N+K, d_{\text{llm}}] \rightarrow [B, M+N+K, |\mathcal{V}|]$ |
| **Loss** | $\mathcal{L} = -\frac{1}{K} \sum_{k=1}^{K} \log P_\theta(c_k \mid \mathbf{x}, c_{<k})$ | Scalar |
| **Perplexity** | $\text{PPL} = \exp(\mathcal{L})$ | Scalar |

### Next Steps

After completing Stage 2:
1. Model checkpoint saved to `results/{llm_id}/OpenTSLMSP/stage2_captioning/checkpoints/best_model.pt`
2. Evaluation metrics saved to `results/{llm_id}/OpenTSLMSP/stage2_captioning/results/metrics.json`
3. Model ready for Stage 3 (CoT Reasoning) initialization
4. Model has learned detailed descriptive language generation
5. Foundation established for chain-of-thought reasoning

---

## References

- **M4 Dataset**: M4 Time Series Caption Dataset (custom-created)
- **M4 Competition**: https://github.com/Mcompetitions/M4-methods
- **OpenTSLM Paper**: [Link to paper](https://doi.org/10.13140/RG.2.2.14827.60963)
- **Implementation**: See `curriculum_learning.py` and `src/time_series_datasets/m4/`

---

**End of Stage 2 Detailed Guide**


<!--
This source file is part of the OpenTSLM open-source project

SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

# Stage 5: ECG-QA Chain-of-Thought - Detailed Mathematical Guide

**A Comprehensive Mathematical and Algorithmic Explanation of Stage 5 Training**

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

### Stage 5 Purpose

Stage 5 (ECG-QA Chain-of-Thought) is the **final advanced medical reasoning stage** of the OpenTSLM curriculum learning pipeline. Building on Stages 1-4, it teaches the model:

1. **Advanced cardiac diagnostic reasoning**: How to analyze complex multi-lead ECG signals
2. **Medical question answering**: How to answer specific clinical questions about ECG findings
3. **Clinical interpretation with context**: How to incorporate patient demographics and clinical context
4. **Multi-modal medical reasoning**: How to combine time series analysis with medical knowledge

**Why Stage 5 After Stages 1-4?**
- Stage 1 taught basic time series understanding (classification/MCQ)
- Stage 2 extended to detailed text generation (captioning)
- Stage 3 introduced CoT reasoning for activity recognition (HAR)
- Stage 4 extended medical reasoning to neurophysiology (sleep staging)
- Stage 5 applies advanced medical reasoning to cardiac diagnosis with clinical questions
- Final stage that combines all learned capabilities

**Key Statistics:**
- **Dataset**: ECG-QA Chain-of-Thought (custom-created with GPT-4o from PTB-XL + ECG-QA)
- **Samples**: Variable across train/val/test splits (thousands of samples)
- **Task Type**: Medical Question Answering with Chain-of-Thought
- **Time Series**: 12-lead ECG signals (10-second recordings at 100Hz)
- **ECG Duration**: 10 seconds (standard clinical recording)
- **Sampling Rate**: 100 Hz (downsampled from 500Hz if needed)
- **Samples per Lead**: ~1,000 (10 seconds × 100 Hz)
- **Leads Used**: All 12 standard ECG leads (I, II, III, aVR, aVL, aVF, V1-V6)
- **Training Epochs**: 60 (with early stopping)
- **Question Types**: Diagnostic verification, morphology assessment, rhythm analysis
- **Metric**: Test Loss, Perplexity, Accuracy, Template-specific F1

---

## Dataset: ECG-QA CoT

### Dataset Description

The ECG-QA Chain-of-Thought dataset combines clinical ECG recordings from PTB-XL with AI-generated chain-of-thought reasoning for medical question answering. Each sample consists of:

**Input:**
- A 12-lead ECG recording: $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_{12}]$ where each $\mathbf{x}_i \in \mathbb{R}^L$
- A clinical question: $Q$ (natural language about cardiac diagnosis)
- Clinical context: $C$ (patient demographics, recording quality, metadata)
- Question type: $\tau \in \mathcal{T}$ (diagnostic verification, morphology, etc.)
- Template ID: $t_{\text{id}}$ (categorizes question structure)

**Output:**
- Chain-of-thought reasoning: $R$ (detailed step-by-step ECG analysis)
- Final answer: $A$ (diagnostic conclusion, typically yes/no/not sure)

### Data Format

Each sample in the ECG-QA CoT dataset has the following structure:

```json
{
  "ecg_id": [12345],                                    // PTB-XL record ID(s)
  "ecg_paths": ["path/to/records100/12000/12345_lr"],  // ECG file paths (without extension)
  "clinical_contexts": ["76-year-old male, sinus rhythm..."], // Patient metadata
  "question": "Does this ECG show symptoms of myocardial infarction?",
  "question_type": "single-verify",                     // Question category
  "template_id": 42,                                    // Question template ID
  "answer": "yes",                                      // Ground truth answer
  "rationale": "Looking at the ECG signal across all leads..." // CoT reasoning + answer
}
```

### Mathematical Representation

Let $\mathcal{D}_{\text{ECG-CoT}} = \{(\mathbf{X}_i, Q_i, C_i, R_i, A_i, \tau_i, t_i)\}_{i=1}^N$ be the dataset with $N$ samples, where:

- $\mathbf{X}_i = [\mathbf{x}_{i,1}, \ldots, \mathbf{x}_{i,12}]$: Multi-lead ECG recording (12 leads)
- $\mathbf{x}_{i,j} \in \mathbb{R}^{L}$: Single ECG lead ($L \approx 1000$ samples at 100Hz)
- $Q_i \in \mathcal{V}^*$: Clinical question (sequence of tokens)
- $C_i \in \mathcal{V}^*$: Clinical context (patient info, recording quality)
- $R_i \in \mathcal{V}^*$: Chain-of-thought reasoning (detailed ECG analysis)
- $A_i \in \mathcal{A}$: Final answer (typically from a template-specific set)
- $\tau_i \in \mathcal{T}$: Question type from set $\mathcal{T}$
- $t_i \in \mathbb{N}$: Template ID

### Data Splits

The dataset is split into three subsets:

$$
\mathcal{D}_{\text{ECG-CoT}} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}
$$

**Splitting is based on ECG-QA CoT CSV files:**
- `ecg_qa_cot_train.csv`: Training samples
- `ecg_qa_cot_val.csv`: Validation samples
- `ecg_qa_cot_test.csv`: Test samples

**Splitting Algorithm:**
```
1. Load ECG-QA CoT data from pre-split CSV files
2. For each sample:
   - Parse question, answer, rationale, template_id, question_type
   - Parse ecg_id (can be single or multiple ECG recordings)
   - Resolve PTB-XL file paths for each ecg_id
   - Parse clinical_contexts
3. Validate ECG files exist (.dat and .hea files)
4. Construct HuggingFace Dataset objects
5. Return D_train, D_val, D_test
```

### ECG Signal Characteristics

**ECG Recording Properties:**

| Property | Value | Description |
|----------|-------|-------------|
| **Leads** | 12 (full standard ECG) | I, II, III, aVR, aVL, aVF, V1-V6 |
| **Sampling Rate** | 100 Hz | Downsampled from 500Hz if needed |
| **Duration** | 10 seconds | Standard clinical recording |
| **Samples per Lead** | ~1,000 | 10s × 100 Hz |
| **Total Dimensions** | 12 × 1,000 = 12,000 | Multi-variate time series |
| **Source** | PTB-XL Database | Large public ECG database |

**12-Lead ECG Configuration:**

ECG leads capture electrical activity from different cardiac perspectives:

$$
\mathbf{X} = [\mathbf{x}_{\text{I}}, \mathbf{x}_{\text{II}}, \mathbf{x}_{\text{III}}, \mathbf{x}_{\text{aVR}}, \mathbf{x}_{\text{aVL}}, \mathbf{x}_{\text{aVF}}, \mathbf{x}_{\text{V1}}, \mathbf{x}_{\text{V2}}, \mathbf{x}_{\text{V3}}, \mathbf{x}_{\text{V4}}, \mathbf{x}_{\text{V5}}, \mathbf{x}_{\text{V6}}]
$$

**Lead Groups:**
- **Limb Leads (6)**: I, II, III, aVR, aVL, aVF - frontal plane view
- **Precordial Leads (6)**: V1-V6 - horizontal plane view

### Chain-of-Thought Reasoning Structure

**Typical CoT Reasoning Contains:**

1. **Overall Assessment**: General observation of ECG morphology and rhythm
2. **Lead-by-Lead Analysis**: Examination of specific leads and their patterns
3. **Interval Measurements**: Analysis of PR, QRS, QT intervals
4. **Waveform Morphology**: P waves, QRS complexes, T waves, ST segments
5. **Pattern Recognition**: Identification of abnormalities or normal features
6. **Clinical Correlation**: Connection to cardiac pathophysiology
7. **Diagnostic Reasoning**: Step-by-step logic toward answer
8. **Final Answer**: Clear response to the clinical question

**Example CoT Reasoning:**

```
Looking at the ECG signal across all leads, I'll systematically analyze the cardiac 
cycle to address the question about myocardial infarction. Starting with the limb 
leads, the rhythm appears regular with consistent RR intervals, suggesting a stable 
sinus rhythm. The P waves are present and upright in leads I, II, and aVF, which is 
normal. The PR interval measures approximately 160ms, within the normal range of 
120-200ms. Examining the QRS complexes, they appear widened at approximately 110ms 
duration, which is at the upper limit of normal. Most significantly, I observe ST 
segment changes in multiple leads. In leads II, III, and aVF (inferior leads), there 
is notable ST segment elevation of 2-3mm above the baseline. Additionally, leads V3 
and V4 show reciprocal ST depression. These ST segment changes are highly suggestive 
of an acute myocardial infarction, specifically involving the inferior wall based on 
the lead distribution. The presence of pathological Q waves in lead III further 
supports this diagnosis, indicating possible myocardial tissue damage. T wave 
morphology shows some inversion in the affected leads, consistent with ischemic 
changes. The combination of ST elevation in anatomically contiguous leads (II, III, 
aVF), reciprocal changes, and Q waves strongly indicates an acute inferior myocardial 
infarction. Answer: yes
```

### Question Types and Templates

**Template-Based Questions:**

The dataset uses **template_id** to categorize questions. Each template has:
- A specific question structure
- A defined set of possible answers
- Specific clinical focus

| Template Type | Description | Example Question | Possible Answers |
|---------------|-------------|------------------|------------------|
| **Diagnostic Verification** | Yes/no questions about conditions | "Does this ECG show symptoms of MI?" | yes, no, not sure |
| **Morphology Assessment** | Questions about waveform features | "Are the QRS complexes normal?" | normal, abnormal, borderline |
| **Rhythm Analysis** | Questions about cardiac rhythm | "Is atrial fibrillation present?" | yes, no, not sure |
| **Interval Measurement** | Questions about timing intervals | "Is the PR interval prolonged?" | yes, no, borderline |
| **Comparison Questions** | Compare between multiple ECGs | "Which ECG shows more severe changes?" | first, second, similar |

**Template-Answer Mapping:**

Each template has a predefined set of valid answers stored in `answers_for_each_template.csv`:

```csv
template_id,classes
1,"['yes', 'no', 'not sure']"
2,"['normal', 'abnormal']"
3,"['present', 'absent', 'indeterminate']"
...
```

### Data Preprocessing

**1. ECG Signal Loading:**

Each ECG recording is loaded from PTB-XL format using the `wfdb` library:

```python
import wfdb

# Load ECG signal from .dat and .hea files
ecg_base_path = "path/to/records100/12000/12345_lr"
ecg_record = wfdb.rdrecord(ecg_base_path)
ecg_signals = ecg_record.p_signal  # Shape: [samples, 12_leads]
```

**PTB-XL has two sampling rates:**
- High-resolution: 500 Hz
- Low-resolution: 100 Hz (suffix `_lr`)

For consistency, we use 100 Hz data (or downsample 500 Hz to 100 Hz).

**2. Downsampling (if needed):**

If ECG is 500 Hz (typically 5000 samples), downsample to 100 Hz:

```python
if len(ecg_signals) > 1000:  # 500Hz data
    # Downsample by factor of 5: take every 5th sample
    downsampled_signals = ecg_signals[::5, :]  # Shape: [1000, 12]
else:  # Already 100Hz
    downsampled_signals = ecg_signals
```

Mathematically:
$$
\mathbf{x}_{\text{100Hz}}[n] = \mathbf{x}_{\text{500Hz}}[5n], \quad n = 0, 1, \ldots, 999
$$

**3. Per-Lead Normalization:**

For each lead $j$, apply z-score normalization:

For lead $\mathbf{x}_j = [x_{j,1}, \ldots, x_{j,L}]$ where $L = 1000$:

$$
\mu_j = \frac{1}{L} \sum_{t=1}^{L} x_{j,t}
$$

$$
\sigma_j = \sqrt{\frac{1}{L} \sum_{t=1}^{L} (x_{j,t} - \mu_j)^2}
$$

$$
\tilde{x}_{j,t} = \frac{x_{j,t} - \mu_j}{\max(\sigma_j, \epsilon)}, \quad \epsilon = 10^{-6}
$$

The normalized lead is: $\tilde{\mathbf{x}}_j = [\tilde{x}_{j,1}, \ldots, \tilde{x}_{j,L}]$

**Rationale**: Per-lead normalization preserves relative amplitude differences between leads while removing baseline variations.

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

Each sample is formatted as a comprehensive clinical prompt:

$$
\text{Prompt}_i = P_{\text{role}} \oplus C_i \oplus Q_i \oplus P_{\text{instructions}} \oplus P_{\text{answers}}
$$

where:
- $P_{\text{role}} = \text{"You are an expert cardiologist analyzing an ECG..."}$
- $C_i = \text{"Clinical Context: "} \oplus C_i$
- $Q_i = \text{"Question: "} \oplus Q_i$
- $P_{\text{instructions}} = \text{"Instructions: Analyze step-by-step..."}$
- $P_{\text{answers}} = \text{"Possible answers: [yes, no, not sure]"}$
- $\oplus$ denotes string concatenation

**Full Pre-Prompt Template:**
```
You are an expert cardiologist analyzing an ECG (electrocardiogram).

Clinical Context: {clinical_context}

Your task is to examine the ECG signal and answer the following medical question:

Question: {question}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding 
  the cardiac condition.
- Write your rationale as a single, natural paragraph — do not use bullet 
  points, numbered steps, or section headings.
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate 
  to the question.
```

**Post-Prompt Template:**
```
Based on your analysis of the ECG data, select your answer from the following 
options:
{possible_answers}

- Make sure that your last word is the answer. You MUST end your response with 
  "Answer: "
```

**6. Lead Description in Prompt:**

For each ECG lead, include descriptive text with statistics:

```python
for lead_idx, lead_name in enumerate(["I", "II", "III", "aVR", "aVL", "aVF", 
                                       "V1", "V2", "V3", "V4", "V5", "V6"]):
    mean_val = np.mean(normalized_signal)
    std_val = np.std(normalized_signal)
    text = f"This is ECG Lead {lead_name}, it has mean {mean_val:.4f} and std {std_val:.4f}:"
```

**7. Chain-of-Thought Target Construction:**

The target output is the complete rationale (which includes the answer at the end):

$$
\text{Target}_i = R_i
$$

where $R_i$ contains both reasoning and concludes with "Answer: [answer]".

---

## Mathematical Formulation

### Problem Formulation

Given:
- Multi-lead ECG recording: $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_{12}]$ where each $\mathbf{x}_i \in \mathbb{R}^L$
- Clinical question with context: $Q, C$ (sequences of tokens)
- Question type and template: $\tau, t_{\text{id}}$

Goal:
- Generate chain-of-thought reasoning: $\hat{R} = [\hat{r}_1, \hat{r}_2, \ldots, \hat{r}_K]$
- Generate final answer: $\hat{A}$ (embedded within $\hat{R}$ after "Answer:")

This is a **conditional text generation with structured medical output** problem.

### Model Function

The model learns a conditional probability distribution over reasoning + answer:

$$
P_\theta(R | \mathbf{X}, Q, C, \tau) = \prod_{k=1}^{K} P_\theta(r_k | \mathbf{X}, Q, C, \tau, r_{<k})
$$

where $R = [r_1, \ldots, r_K]$ is the complete output sequence (reasoning + answer).

The model function is:

$$
f_\theta: (\mathbb{R}^{12 \times L}, \mathcal{V}^*, \mathcal{V}^*, \mathcal{T}) \rightarrow \Delta^{|\mathcal{V}|}
$$

where $\Delta^{|\mathcal{V}|}$ is the probability simplex over the vocabulary.

### Loss Function

For chain-of-thought ECG question answering, we use **causal language modeling loss** (cross-entropy):

$$
\mathcal{L}(\theta; \mathbf{X}, Q, C, R) = -\frac{1}{K} \sum_{k=1}^{K} \log P_\theta(r_k | \mathbf{X}, Q, C, r_{<k})
$$

This ensures the model learns to:
1. Generate coherent medical reasoning given ECG, question, and context
2. Produce correct answer conditioned on the reasoning
3. Incorporate clinical context and question type appropriately

### Total Training Objective

Over the entire training dataset $\mathcal{D}_{\text{train}}$:

$$
\mathcal{L}_{\text{total}}(\theta) = \frac{1}{|\mathcal{D}_{\text{train}}|} \sum_{(\mathbf{X}_i, Q_i, C_i, R_i) \in \mathcal{D}_{\text{train}}} \mathcal{L}(\theta; \mathbf{X}_i, Q_i, C_i, R_i)
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
- Lower perplexity → More confident and coherent ECG diagnostic reasoning
- Higher perplexity → Uncertain or incoherent reasoning

---

## Model Architecture

The architecture for Stage 5 is **identical** to Stages 1-4, but with 12-lead ECG data and medical question answering. The model consists of three main components:

### 1. Time Series Encoder (12-Lead ECG)

**Architecture**: Transformer-CNN Encoder (same as Stages 1-4)

**Input**: $\mathbf{X} \in \mathbb{R}^{B \times 12 \times L}$ (batch of 12-lead ECG)

**Processing**: Each lead is encoded independently, then concatenated

**Output**: $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times (12 \times N) \times d_{\text{enc}}}$ (encoded features)

where:
- $B$: Batch size
- $12$: Number of ECG leads
- $L$: ECG samples per lead (padded, typically ~1000)
- $N = L / P$: Number of patches per lead
- $P = 4$: Patch size
- $d_{\text{enc}} = 128$: Encoder embedding dimension

#### Multi-Lead Encoding Process

**Per-Lead Encoding:**

For each lead $\mathbf{x}_j \in \mathbb{R}^{L}$:

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
\mathbf{H}_{\text{enc}} = [\mathbf{H}_{\text{enc}, 1}; \mathbf{H}_{\text{enc}, 2}; \ldots; \mathbf{H}_{\text{enc}, 12}] \in \mathbb{R}^{(12 \times N) \times d_{\text{enc}}}
$$

This creates a unified representation of all 12 ECG leads.

**Total Sequence Length:**

$$
N_{\text{total}} = 12 \times N = 12 \times \frac{L}{4} \approx 12 \times 250 = 3000 \text{ ECG tokens}
$$

This is the longest time series representation in the curriculum (12 leads vs 6 leads in some variants).

### 2. Projector

**Architecture**: MLP with LayerNorm and GELU (same as Stages 1-4)

**Input**: $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{B \times N_{\text{total}} \times d_{\text{enc}}}$

**Output**: $\mathbf{H}_{\text{proj}} \in \mathbb{R}^{B \times N_{\text{total}} \times d_{\text{llm}}}$

**Mathematical Formulation:**

$$
\mathbf{H}_{\text{proj}} = \text{Dropout}(\text{GELU}(\text{Linear}(\text{LayerNorm}(\mathbf{H}_{\text{enc}}))))
$$

### 3. Large Language Model (LLM)

**Architecture**: Pre-trained causal LM (Llama/Gemma, same as Stages 1-4)

**Input**: Combined sequence of projected ECG embeddings and text tokens

**Output**: Probability distribution over vocabulary for next token prediction

#### Input Sequence Construction for ECG QA CoT

For a sample $(\mathbf{X}, Q, C, R)$:

**1. Text Tokenization:**

$$
\text{Prompt}_{\text{tokens}} = \text{Tokenize}(P_{\text{role}} + C + Q + P_{\text{instructions}} + P_{\text{answers}}) = [p_1, \ldots, p_M]
$$

$$
R_{\text{tokens}} = \text{Tokenize}(R) = [r_1, r_2, \ldots, r_K]
$$

where $R$ includes both reasoning and the final answer.

**2. Text Embedding:**

$$
\mathbf{H}_P = \text{Embed}_{\text{LLM}}(\text{Prompt}_{\text{tokens}}) \in \mathbb{R}^{M \times d_{\text{llm}}}
$$

$$
\mathbf{H}_R = \text{Embed}_{\text{LLM}}(R_{\text{tokens}}) \in \mathbb{R}^{K \times d_{\text{llm}}}
$$

**3. Sequence Concatenation:**

$$
\mathbf{H}_{\text{input}} = [\mathbf{H}_P; \mathbf{H}_{\text{proj}}; \mathbf{H}_R] \in \mathbb{R}^{(M+N_{\text{total}}+K) \times d_{\text{llm}}}
$$

**Total sequence length**: $T = M + N_{\text{total}} + K$ (typically 3,500-4,500 tokens)

Note: This is the longest sequence in the curriculum:
- Prompt ($M$): ~300-500 tokens (role + context + question + instructions)
- ECG encoding ($N_{\text{total}}$): ~3,000 tokens (12 leads × 250 patches)
- Reasoning + answer ($K$): ~200-500 tokens

#### Autoregressive Chain-of-Thought Generation

During training, the model learns:

$$
P_\theta(r_k | \mathbf{X}, Q, C, r_{<k}) = \text{Softmax}(\text{LLM}([\mathbf{H}_P; \mathbf{H}_{\text{proj}}; \mathbf{H}_{r_{<k}}]))_k
$$

### Model Initialization for Stage 5

**Key Difference from Stages 1-4:**

Stage 5 **initializes** from the best Stage 4 checkpoint:

$$
\theta_{\text{Stage5}}^{(0)} = \theta_{\text{Stage4}}^*
$$

where $\theta_{\text{Stage4}}^*$ is the best model from Stage 4 training (Sleep CoT).

This provides:
1. **Pre-trained encoder**: Already understands diverse time series patterns
2. **Pre-trained projector**: Already maps time series to LLM space effectively
3. **Pre-trained LLM**: Already capable of medical chain-of-thought reasoning (from Stages 3-4)
4. **Warm start**: Faster convergence for complex cardiac diagnostic task
5. **Transfer learning**: Leverages sleep staging reasoning for cardiac reasoning

**Complete Curriculum Learning Progression:**

$$
\text{Stage 1} \rightarrow \text{Stage 2} \rightarrow \text{Stage 3} \rightarrow \text{Stage 4} \rightarrow \text{Stage 5}
$$

$$
\text{MCQ} \rightarrow \text{Captioning} \rightarrow \text{HAR CoT} \rightarrow \text{Sleep CoT} \rightarrow \text{ECG QA CoT}
$$

Each stage builds on the previous:
- Stage 1: Basic classification
- Stage 2: Detailed generation
- Stage 3: Activity recognition reasoning
- Stage 4: Neurophysiological reasoning
- Stage 5: Advanced cardiac diagnostic reasoning (final stage)

### Parameter Count

**Trainable Parameters (Stage 5):**

| Component | Parameters | Trainable | Notes |
|-----------|-----------|-----------|-------|
| **Encoder** | ~5M | ✅ Yes | Fine-tuned from Stage 4 |
| **Projector** | ~260K | ✅ Yes | Fine-tuned from Stage 4 |
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
| Batch Size | 4 (or smaller) | Samples per batch (limited by long sequences) |
| Learning Rate (Encoder) | $2 \times 10^{-4}$ | LR for encoder |
| Learning Rate (Projector) | $1 \times 10^{-4}$ | LR for projector |
| Learning Rate (LoRA) | $2 \times 10^{-4}$ | LR for LoRA adapters (if enabled) |
| Weight Decay | $1 \times 10^{-2}$ | L2 regularization |
| Gradient Clipping | 1.0 | Max gradient norm |
| Warmup Fraction | 0.03 | Fraction of steps for LR warmup |
| Early Stopping Patience | 5 | Epochs without improvement to stop |
| Patch Size | 4 | Time series patch size |
| Gradient Checkpointing | Recommended | Reduces memory for long sequences |

**Note**: Stage 5 requires 60 epochs and may benefit from gradient checkpointing due to the longest sequences in the curriculum (~3,000 ECG tokens).

### Curriculum Learning Connection

Stage 5 builds on Stage 4:

$$
\theta_{\text{Stage5}}^{(0)} \leftarrow \text{BestCheckpoint}(\text{Stage4})
$$

This enables:
1. **Knowledge transfer**: All medical reasoning capabilities from Stages 3-4 transfer
2. **Faster convergence**: Model starts with strong medical foundation
3. **Better performance**: Curriculum critical for complex cardiac diagnostics
4. **Final refinement**: Fine-tune for specific ECG question answering task

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

**AdamW Optimizer** (same as Stages 1-4):

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

**Algorithm: Stage 5 Training**

```
Input: Training data D_train, validation data D_val, test data D_test
       Stage 4 best checkpoint θ_stage4
Output: Trained model parameters θ*

1. Load Stage 4 checkpoint:
   encoder_params ← θ_stage4.encoder
   projector_params ← θ_stage4.projector
   lora_params ← θ_stage4.lora (if enabled)
   
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
           X_batch, Q_batch, C_batch, R_batch ← batch
           loss ← compute_ecg_qa_cot_loss(X_batch, Q_batch, C_batch, R_batch)
           
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
               X_batch, Q_batch, C_batch, R_batch ← batch
               loss ← compute_ecg_qa_cot_loss(X_batch, Q_batch, C_batch, R_batch)
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
   test_metrics ← evaluate_ecg_qa_cot(model, D_test)
   
7. Return model, test_metrics
```

### Loss Computation for ECG QA CoT

For a single sample $(\mathbf{X}, Q, C, R)$:

**Step 1: Load and normalize 12-lead ECG**
```
# Load ECG signal from PTB-XL
ecg_record ← wfdb.rdrecord(ecg_path)
ecg_signals ← ecg_record.p_signal                 # Shape: [samples, 12]

# Downsample if needed (500Hz → 100Hz)
if len(ecg_signals) > 1000:
    ecg_signals ← ecg_signals[::5, :]             # Shape: [1000, 12]

# Normalize each lead independently
normalized_leads ← []
for lead_idx = 0 to 11:
    lead_signal ← ecg_signals[:, lead_idx]
    mean_val ← np.mean(lead_signal)
    std_val ← np.std(lead_signal)
    normalized_lead ← (lead_signal - mean_val) / max(std_val, 1e-6)
    normalized_leads.append(normalized_lead)
```

**Step 2: Encode all leads**
```
H_enc_leads ← []
for lead in normalized_leads:
    H_enc_lead ← Encoder(lead)                    # Shape: [N, d_enc]
    H_enc_leads.append(H_enc_lead)

# Concatenate all lead encodings
H_enc ← concat(H_enc_leads, dim=0)                # Shape: [12N, d_enc]
```

**Step 3: Project to LLM space**
```
H_proj ← Projector(H_enc)                         # Shape: [12N, d_llm]
```

**Step 4: Prepare text**
```
# Create full prompt with clinical context and question
prompt ← create_ecg_qa_prompt(Q, C, template_id)
Prompt_tokens ← Tokenize(prompt)                  # Shape: [M]

# Chain-of-thought reasoning (includes answer)
R_tokens ← Tokenize(rationale)                    # Shape: [K]
```

**Step 5: Embed and concatenate**
```
H_P ← Embed(Prompt_tokens)                        # Shape: [M, d_llm]
H_R ← Embed(R_tokens)                             # Shape: [K, d_llm]

H_input ← concat([H_P, H_proj, H_R], dim=0)      # Shape: [M+12N+K, d_llm]
```

**Step 6: LLM forward pass**
```
logits ← LLM(H_input)                             # Shape: [M+12N+K, |V|]
```

**Step 7: Extract CoT logits and compute loss**
```
# CoT tokens start after prompt and ECG embeddings
cot_start_idx ← M + 12N
cot_logits ← logits[cot_start_idx : cot_start_idx+K-1, :]  # Shape: [K-1, |V|]

# Target tokens (shifted by 1 for next-token prediction)
target_tokens ← R_tokens[1:]                      # Shape: [K-1]

# Compute cross-entropy loss
loss ← CrossEntropyLoss(cot_logits, target_tokens)
```

Mathematically:
$$
\mathcal{L} = -\frac{1}{K-1} \sum_{k=1}^{K-1} \log P_\theta(r_{k+1} | \mathbf{X}, Q, C, r_{\leq k})
$$

---

## Evaluation

### Evaluation Metrics

**Primary Metrics:**

1. **Test Loss** (Cross-Entropy):
   $$
   \mathcal{L}_{\text{test}} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{(\mathbf{X}, Q, C, R) \in \mathcal{D}_{\text{test}}} \mathcal{L}(\theta; \mathbf{X}, Q, C, R)
   $$

2. **Perplexity**:
   $$
   \text{PPL} = \exp(\mathcal{L}_{\text{test}})
   $$

**Interpretation:**
- **Lower test loss** = Better reasoning generation quality
- **Lower perplexity** = More confident and coherent CoT reasoning
- Typical good perplexity: 6-20 for ECG diagnostic reasoning tasks

**Secondary Metrics (Post-Processing):**

3. **Answer Accuracy**: Extract final answer and compare with ground truth

   For each prediction:
   ```python
   predicted_answer = extract_answer(generated_text)  # After "Answer: "
   ground_truth_answer = sample['answer']
   accuracy = normalize_answer(predicted_answer) == normalize_answer(ground_truth_answer)
   ```
   
   $$
   \text{Accuracy} = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{i=1}^{|\mathcal{D}_{\text{test}}|} \mathbb{1}[\hat{A}_i = A_i]
   $$

4. **Template-Specific F1 Score**: Per-template classification metrics

   For each template $t \in \mathcal{T}_{\text{templates}}$:
   $$
   F1_t = \frac{2 \cdot P_t \cdot R_t}{P_t + R_t}
   $$
   
   where $P_t$ is precision and $R_t$ is recall for template $t$.

5. **Macro-F1 Score**: Average F1 across all templates

   $$
   \text{Macro-F1} = \frac{1}{|\mathcal{T}_{\text{templates}}|} \sum_{t \in \mathcal{T}_{\text{templates}}} F1_t
   $$

6. **Question-Type Specific Metrics**: Performance breakdown by question type

   For each question type $\tau \in \mathcal{T}$:
   - Accuracy per type
   - F1 per type
   - Sample counts per type

### Inference Process

**Chain-of-Thought ECG QA Generation Algorithm:**

For a test sample $(\mathbf{X}_{\text{test}}, Q_{\text{test}}, C_{\text{test}})$:

```
1. Load and preprocess 12-lead ECG:
   ecg_signals ← load_ecg(ecg_path)
   if len(ecg_signals) > 1000:
       ecg_signals ← ecg_signals[::5, :]          # Downsample
   
   normalized_leads ← []
   for lead in ecg_signals.T:
       normalized_lead ← (lead - mean(lead)) / max(std(lead), 1e-6)
       normalized_leads.append(normalized_lead)
   
2. Encode all leads:
   H_enc_leads ← [Encoder(lead) for lead in normalized_leads]
   H_enc ← concat(H_enc_leads, dim=0)
   H_proj ← Projector(H_enc)
   
3. Prepare prompt:
   prompt ← create_ecg_qa_prompt(Q_test, C_test, template_id)
   H_P ← Embed(Tokenize(prompt))
   H_input ← concat([H_P, H_proj], dim=0)
   
4. Generate CoT reasoning autoregressively:
   generated_tokens ← []
   current_input ← H_input
   
   for step = 1 to max_new_tokens (e.g., 500):
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
   
6. Extract reasoning and answer:
   if "Answer:" in cot_text:
       reasoning ← cot_text.split("Answer:")[0].strip()
       answer ← cot_text.split("Answer:")[-1].strip()
   else:
       reasoning ← cot_text
       answer ← "not sure"  # Default
   
7. Normalize answer:
   answer ← normalize_ecg_answer(answer)
   
8. Return reasoning, answer
```

### Evaluation Algorithm

```
Function: evaluate_stage5(model, D_test)

Input: Trained model, test dataset D_test
Output: Test loss, perplexity, accuracy, template metrics, and predictions

1. Initialize:
   total_loss ← 0
   total_tokens ← 0
   predictions ← []
   ground_truths ← []
   correct_answers ← 0
   template_predictions ← defaultdict(list)  # Group by template
   
2. Set model to evaluation mode:
   model.eval()
   
3. Disable gradient computation:
   with torch.no_grad():
       
       for (X, Q, C, R, answer, template_id) in D_test:
           // Compute loss
           loss ← compute_ecg_qa_cot_loss(X, Q, C, R)
           K ← len(Tokenize(R))
           
           total_loss ← total_loss + loss.item() × K
           total_tokens ← total_tokens + K
           
           // Generate prediction
           reasoning_pred, answer_pred ← model.generate_ecg_qa_cot(
               X, Q, C, max_new_tokens=500
           )
           
           // Extract ground truth answer
           answer_true ← answer
           
           // Store predictions
           pred_record = {
               "generated_answer": answer_pred,
               "target_answer": answer_true,
               "reasoning": reasoning_pred,
               "target_reasoning": R,
               "template_id": template_id,
               "question": Q,
               "ecg_id": get_ecg_id(sample)
           }
           predictions.append(pred_record)
           ground_truths.append(answer_true)
           
           // Group by template
           template_predictions[template_id].append(pred_record)
           
           // Check accuracy (after normalization)
           if normalize_ecg_answer(answer_pred) == normalize_ecg_answer(answer_true):
               correct_answers ← correct_answers + 1
   
4. Compute overall metrics:
   avg_test_loss ← total_loss / total_tokens
   perplexity ← exp(avg_test_loss)
   accuracy ← correct_answers / len(D_test)
   
5. Compute per-template metrics:
   template_f1_scores ← {}
   for template_id, template_preds in template_predictions:
       f1 ← compute_template_f1(template_preds)
       template_f1_scores[template_id] ← f1
   
   macro_f1 ← mean(list(template_f1_scores.values()))
   
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
   
   save_json(results, "stage5_ecg_cot/results/metrics.json")
   save_jsonl(predictions, "stage5_ecg_cot/results/test_predictions.jsonl")
   
7. Print summary:
   print(f"Test Loss: {avg_test_loss:.4f}")
   print(f"Perplexity: {perplexity:.2f}")
   print(f"Accuracy: {accuracy:.4f}")
   print(f"Macro-F1: {macro_f1:.4f}")
   print(f"Avg tokens per sample: {total_tokens/len(D_test):.1f}")
   
   // Print per-template metrics
   print("\nPer-Template Performance:")
   for template_id, f1 in sorted(template_f1_scores.items()):
       print(f"  Template {template_id}: F1={f1:.4f}")
   
8. Return results
```

### Answer Extraction and Normalization

**Extraction Function:**

```python
def extract_ecg_answer(text: str) -> str:
    """Extract the final answer from ECG CoT reasoning."""
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
def normalize_ecg_answer(answer: str) -> str:
    """Normalize ECG answer for comparison."""
    # Convert to lowercase
    answer = answer.lower()
    
    # Remove punctuation
    answer = answer.rstrip('.,!?;:')
    
    # Strip whitespace
    answer = answer.strip()
    
    # Map common variations
    answer_mappings = {
        "yes": "yes",
        "y": "yes",
        "true": "yes",
        "no": "no",
        "n": "no",
        "false": "no",
        "not sure": "not sure",
        "unsure": "not sure",
        "indeterminate": "not sure",
        "normal": "normal",
        "abnormal": "abnormal",
        "borderline": "borderline"
    }
    
    return answer_mappings.get(answer, answer)
```

### Expected Performance

**Typical Stage 5 Results:**

| Model | Test Loss | Perplexity | Accuracy | Macro-F1 | Training Time | GPU Memory |
|-------|-----------|------------|----------|----------|---------------|------------|
| **OpenTSLMSP** (Llama-3.2-1B) | 1.6-2.1 | 5-8 | 0.75-0.85 | 0.70-0.80 | ~10-16 hours | ~16-20GB |
| **OpenTSLMSP** (Gemma-3-270m) | 1.9-2.4 | 7-11 | 0.70-0.80 | 0.65-0.75 | ~8-14 hours | ~14-18GB |
| **OpenTSLMFlamingo** (Llama-3.2-1B) | 1.4-1.9 | 4-7 | 0.80-0.88 | 0.75-0.83 | ~12-18 hours | ~18-24GB |

*Note: Results vary based on random seed, hardware, exact configuration, and whether LoRA is enabled*

**Performance Interpretation:**
- **PPL < 6**: Excellent - coherent and clinically sound ECG reasoning
- **PPL 6-12**: Good - generally sound reasoning with minor issues
- **PPL 12-25**: Acceptable - reasoning captures main points but may lack clinical precision
- **PPL > 25**: Poor - reasoning may be incoherent or clinically unsound

**Accuracy vs. Perplexity:**
- Lower perplexity typically correlates with higher accuracy
- ECG interpretation is challenging even for cardiologists
- Both metrics important for clinical reliability assessment

**Per-Template Performance:**

Different question templates have varying difficulty:

| Template Type | Typical F1 | Challenge Level |
|---------------|-----------|-----------------|
| **Basic Rhythm** | 0.85-0.92 | Easy - clear rhythm patterns |
| **Normal/Abnormal** | 0.80-0.88 | Moderate - requires synthesis |
| **Specific Conditions** | 0.70-0.82 | Hard - requires detailed knowledge |
| **Comparison** | 0.65-0.78 | Hardest - multi-ECG reasoning |

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
   
2. Load CoT data from pre-split CSV files:
   train_df ← read_csv("ecg_qa_cot_train.csv")
   val_df ← read_csv("ecg_qa_cot_val.csv")
   test_df ← read_csv("ecg_qa_cot_test.csv")
   
3. For each split:
   samples ← []
   
   for row in split_df:
       // Parse fields
       ecg_id ← parse_ecg_id(row["ecg_id"])            // Can be single or multiple
       question ← row["question"]
       answer ← row["answer"]
       template_id ← row["template_id"]
       question_type ← row["question_type"]
       clinical_context ← row["clinical_contexts"]
       rationale ← row["rationale"]
       
       // Resolve PTB-XL file paths
       ecg_paths ← []
       for id in ecg_id:
           ecg_base_path ← get_ptbxl_ecg_path(id)    // e.g., "records100/12000/12345_lr"
           ecg_paths.append(ecg_base_path)
       
       // Validate files exist
       for ecg_path in ecg_paths:
           dat_path ← ecg_path + ".dat"
           hea_path ← ecg_path + ".hea"
           if not exists(dat_path) or not exists(hea_path):
               raise FileNotFoundError(f"ECG files not found: {ecg_path}")
       
       // Create sample dictionary
       sample ← {
           "ecg_id": ecg_id,
           "ecg_paths": ecg_paths,
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
   
5. Preload ECG data for performance:
   preload_all_ecg_data([D_train, D_val, D_test])
   
6. Print statistics:
   print(f"Train: {len(D_train)} samples")
   print(f"Val: {len(D_val)} samples")
   print(f"Test: {len(D_test)} samples")
   
   // Print template distribution
   for split_name, split_data in [("Train", D_train), ("Val", D_val), ("Test", D_test)]:
       template_counts ← Counter([s['template_id'] for s in split_data])
       print(f"\n{split_name} template distribution:")
       for template_id, count in template_counts.most_common(10):
           print(f"  Template {template_id}: {count} samples")
   
7. Return D_train, D_val, D_test
```

### Algorithm 2: 12-Lead ECG Loading and Preprocessing

```
Function: load_and_preprocess_12lead_ecg(ecg_path)

Input: Path to ECG file (without extension)
Output: Normalized 12-lead ECG tensor, statistics per lead

1. Load ECG signal using wfdb:
   import wfdb
   ecg_record ← wfdb.rdrecord(ecg_path)
   ecg_signals ← ecg_record.p_signal              # Shape: [samples, 12]
   
2. Validate signal:
   assert ecg_signals.shape[1] == 12, "Expected 12 leads"
   
3. Downsample if needed (500Hz → 100Hz):
   if len(ecg_signals) > 1000:
       ecg_signals ← ecg_signals[::5, :]          # Shape: [1000, 12]
   
4. Normalize each lead independently:
   normalized_leads ← []
   lead_stats ← []
   
   for lead_idx = 0 to 11:
       lead_signal ← ecg_signals[:, lead_idx]    # Shape: [1000]
       
       // Compute statistics
       μ ← mean(lead_signal)
       σ ← std(lead_signal)
       
       // Z-score normalization
       min_std ← 1e-6
       σ_safe ← max(σ, min_std)
       lead_norm ← (lead_signal - μ) / σ_safe
       
       normalized_leads.append(lead_norm)
       lead_stats.append((μ, σ))
   
5. Stack normalized leads:
   X_12lead ← stack(normalized_leads, axis=1)    # Shape: [1000, 12]
   
6. Convert to tensor:
   X_tensor ← torch.tensor(X_12lead, dtype=torch.float32)
   
7. Return X_tensor, lead_stats
```

### Algorithm 3: ECG QA CoT Loss Computation

```
Function: compute_ecg_qa_cot_loss(model, X_ecg, question, context, rationale)

Input: Model, 12-lead ECG, question, context, rationale
Output: Loss value

1. Normalize 12-lead ECG:
   normalized_leads ← []
   for lead_idx = 0 to 11:
       lead ← X_ecg[:, lead_idx]
       lead_norm ← (lead - mean(lead)) / max(std(lead), 1e-6)
       normalized_leads.append(lead_norm)
   
2. Encode all leads:
   H_enc_list ← []
   for lead in normalized_leads:
       H_enc_lead ← model.encoder(lead)          # Shape: [N, d_enc]
       H_enc_list.append(H_enc_lead)
   
   // Concatenate all leads
   H_enc ← concat(H_enc_list, dim=0)             # Shape: [12N, d_enc]
   
3. Project to LLM space:
   H_proj ← model.projector(H_enc)               # Shape: [12N, d_llm]
   
4. Prepare prompt:
   prompt ← create_ecg_qa_prompt(question, context, template_id)
   P_tokens ← tokenize(prompt)                   # Shape: [M]
   
5. Prepare CoT target:
   R_tokens ← tokenize(rationale)                # Shape: [K]
   
6. Embed text:
   H_P ← model.llm.embed(P_tokens)               # Shape: [M, d_llm]
   H_R ← model.llm.embed(R_tokens)               # Shape: [K, d_llm]
   
7. Create input sequence:
   H_input ← concat([H_P, H_proj, H_R], dim=0)  # Shape: [M+12N+K, d_llm]
   
8. Create labels:
   // Labels are -100 for non-CoT tokens (ignored in loss)
   labels ← [-100] × (M + 12N) + R_tokens[1:]   # Shape: [M+12N+K-1]
   
9. LLM forward pass:
   logits ← model.llm(H_input)                   # Shape: [M+12N+K, |V|]
   
10. Compute cross-entropy loss:
   // Only compute loss on CoT tokens
   cot_start ← M + 12N
   cot_logits ← logits[cot_start:cot_start+K-1, :] # Shape: [K-1, |V|]
   cot_targets ← R_tokens[1:]                    # Shape: [K-1]
   
   loss ← CrossEntropyLoss(cot_logits, cot_targets)
   
11. Return loss
```

### Algorithm 4: ECG QA CoT Generation with Answer Extraction

```
Function: generate_ecg_qa_cot(model, X_ecg, question, context, template_id, 
                                max_tokens=500, temperature=1.0)

Input: Model, 12-lead ECG, question, context, template_id, generation parameters
Output: Generated reasoning text and extracted answer

1. Normalize 12-lead ECG:
   normalized_leads ← []
   for lead_idx = 0 to 11:
       lead ← X_ecg[:, lead_idx]
       lead_norm ← (lead - mean(lead)) / max(std(lead), 1e-6)
       normalized_leads.append(lead_norm)
   
2. Encode all leads:
   H_enc_list ← []
   for lead in normalized_leads:
       H_enc_lead ← model.encoder(lead)
       H_enc_list.append(H_enc_lead)
   H_enc ← concat(H_enc_list, dim=0)
   
3. Project to LLM space:
   H_proj ← model.projector(H_enc)
   
4. Prepare prompt:
   prompt ← create_ecg_qa_prompt(question, context, template_id)
   P_tokens ← tokenize(prompt)
   H_P ← model.llm.embed(P_tokens)
   
5. Initialize generation:
   H_input ← concat([H_P, H_proj], dim=0)        # Shape: [M+12N, d_llm]
   generated_tokens ← []
   
6. Autoregressive generation:
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
   
7. Decode to text:
   cot_text ← detokenize(generated_tokens)
   
8. Extract reasoning and answer:
   if "Answer:" in cot_text:
       reasoning ← cot_text.split("Answer:")[0].strip()
       answer ← cot_text.split("Answer:")[-1].strip()
   else:
       reasoning ← cot_text
       answer ← "not sure"  # Default
   
9. Clean and normalize answer:
   answer ← answer.rstrip('.,!?;:').strip()
   answer ← normalize_ecg_answer(answer)
   
10. Return reasoning, answer
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
   
   for template_id, preds in template_groups.items():
       // Get possible answers for this template
       possible_answers ← get_possible_answers_for_template(template_id)
       
       // Initialize counts for each answer class
       class_counts ← {}
       for ans in possible_answers:
           class_counts[ans] ← {"tp": 0, "fp": 0, "fn": 0}
       
       // Count TP, FP, FN
       for pred in preds:
           gt ← normalize_ecg_answer(pred["target_answer"])
           pred_ans ← normalize_ecg_answer(pred["generated_answer"])
           
           if gt in class_counts:
               if pred_ans == gt:
                   class_counts[gt]["tp"] += 1
               else:
                   class_counts[gt]["fn"] += 1
                   if pred_ans in class_counts:
                       class_counts[pred_ans]["fp"] += 1
       
       // Compute F1 for each class
       class_f1_list ← []
       for ans, counts in class_counts.items():
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

| Aspect | Stage 4 (Sleep CoT) | Stage 5 (ECG QA CoT) |
|--------|---------------------|----------------------|
| **Task** | Sleep Stage Classification | Medical Question Answering |
| **Domain** | Neurophysiology | Cardiac electrophysiology |
| **Time Series Type** | Single-channel EEG | 12-lead ECG |
| **Signal Duration** | 30 seconds | 10 seconds |
| **Sampling Rate** | 100 Hz | 100 Hz |
| **Total Samples** | ~3,000 | ~12,000 (12 × 1,000) |
| **Time Series Tokens** | ~750 | ~3,000 (12 leads × 250) |
| **Total Sequence** | ~1,000-1,500 tokens | ~3,500-4,500 tokens |
| **Output Classes** | 6 fixed sleep stages | Variable (template-dependent) |
| **Question Type** | Classification | Question answering |
| **Clinical Context** | Sleep epoch analysis | Patient demographics + ECG metadata |
| **Typical Accuracy** | 0.75-0.88 | 0.75-0.88 |
| **Epochs** | 60 | 60 |
| **Memory Requirements** | ~12-16GB | ~16-24GB (longer sequences) |

### Similarities

1. **Architecture**: Identical encoder, projector, and LLM across all stages
2. **Loss Function**: Both use cross-entropy (causal language modeling)
3. **Optimization**: Same AdamW with warmup + linear decay
4. **Learning Rates**: Identical (encoder: 2e-4, projector: 1e-4)
5. **Curriculum**: Both initialize from previous stage checkpoint
6. **CoT Reasoning**: Both use chain-of-thought reasoning structure
7. **Medical Domain**: Both are medical reasoning tasks
8. **Training Duration**: Both use 60 epochs with early stopping

### Complete Curriculum Learning Progression

**Full Pipeline:**

$$
\text{Stage 1} \rightarrow \text{Stage 2} \rightarrow \text{Stage 3} \rightarrow \text{Stage 4} \rightarrow \text{Stage 5}
$$

$$
\text{MCQ} \rightarrow \text{Captioning} \rightarrow \text{HAR CoT} \rightarrow \text{Sleep CoT} \rightarrow \text{ECG QA CoT}
$$

**Skills Acquired:**

| Stage | Primary Skill | Domain | Signal Type | Reasoning Type |
|-------|--------------|--------|-------------|----------------|
| **Stage 1** | Pattern recognition | General TS | Univariate | Implicit |
| **Stage 2** | Detailed description | Economic | Univariate | Descriptive |
| **Stage 3** | Activity reasoning | Physical activity | Triaxial accel. | Explicit CoT |
| **Stage 4** | Sleep staging | Neurophysiology | Single EEG | Explicit CoT |
| **Stage 5** | Cardiac diagnosis | Cardiology | 12-lead ECG | Explicit CoT + QA |

**Mathematical Progression:**

1. **Stage 1**: Learn $P_\theta(A | \mathbf{x}, Q)$ - Classification
2. **Stage 2**: Learn $P_\theta(C | \mathbf{x})$ - Caption generation
3. **Stage 3**: Learn $P_\theta(R, A | \mathbf{x}_{\text{HAR}}, C)$ - HAR reasoning
4. **Stage 4**: Learn $P_\theta(R, A | \mathbf{x}_{\text{EEG}}, C)$ - Sleep reasoning
5. **Stage 5**: Learn $P_\theta(R, A | \mathbf{X}_{\text{ECG}}, Q, C)$ - ECG QA reasoning

**Why This Order?**

The curriculum is carefully designed for progressive complexity:

$$
\text{Classification} \rightarrow \text{Generation} \rightarrow \text{Activity CoT} \rightarrow \text{Sleep CoT} \rightarrow \text{ECG QA CoT}
$$

1. **Stage 1-2**: Foundation
   - Basic time series understanding
   - Natural language generation capability

2. **Stage 3**: First CoT
   - Introduce chain-of-thought reasoning
   - Simple triaxial signal (3 channels)

3. **Stage 4**: Medical CoT
   - Apply CoT to medical domain
   - Single-channel medical signal (EEG)
   - Classification task

4. **Stage 5**: Advanced Medical CoT + QA
   - Complex multi-lead medical signal (12-lead ECG)
   - Question answering (not just classification)
   - Incorporate clinical context
   - Final integration of all capabilities

**Benefits of Full Curriculum:**

- ✅ Progressive complexity prevents training failure
- ✅ Each stage builds specific capabilities
- ✅ Medical reasoning emerges through stages 3-5
- ✅ Final stage leverages all previous learning
- ✅ State-of-the-art performance on complex cardiac diagnostics

---

## Summary

### Stage 5 Key Takeaways

1. **Purpose**: Final stage - advanced cardiac diagnostic reasoning with medical QA
2. **Dataset**: ECG-QA CoT with 12-lead ECG signals and GPT-4o generated reasoning
3. **Architecture**: Same as Stages 1-4, initialized from Stage 4 checkpoint
4. **Training**: 60 epochs, early stopping, focus on test loss/perplexity/accuracy
5. **Metrics**: Test loss, perplexity, accuracy, template-specific F1, macro-F1
6. **Challenge**: Longest sequences (~3,000 ECG tokens), most complex medical reasoning
7. **Output**: Trained model capable of clinical-grade ECG question answering with reasoning

### Mathematical Components Summary

| Component | Mathematical Operation | Dimensionality |
|-----------|------------------------|----------------|
| **Input** | $\mathbf{X} = [\mathbf{x}_1, \ldots, \mathbf{x}_{12}]$ | $[B, 12, L]$ → 12-lead ECG (L≈1000) |
| **Normalization** | $\tilde{x}_{j,t} = \frac{x_{j,t} - \mu_j}{\max(\sigma_j, \epsilon)}$ | Per-lead z-score |
| **Encoder (per lead)** | $\mathbf{H}_{\text{enc}, j} = \text{TransformerEncoder}(\text{PatchEmbed}(\tilde{\mathbf{x}}_j))$ | $[B, L] \rightarrow [B, N, d_{\text{enc}}]$ |
| **Lead Concatenation** | $\mathbf{H}_{\text{enc}} = [\mathbf{H}_{\text{enc}, 1}; \ldots; \mathbf{H}_{\text{enc}, 12}]$ | $[B, 12N, d_{\text{enc}}]$ |
| **Projector** | $\mathbf{H}_{\text{proj}} = \text{MLP}(\mathbf{H}_{\text{enc}})$ | $[B, 12N, d_{\text{enc}}] \rightarrow [B, 12N, d_{\text{llm}}]$ |
| **LLM** | $\mathbf{L} = \text{LLM}([\mathbf{H}_P; \mathbf{H}_{\text{proj}}; \mathbf{H}_R])$ | $[B, M+12N+K, d_{\text{llm}}] \rightarrow [B, M+12N+K, |\mathcal{V}|]$ |
| **Loss** | $\mathcal{L} = -\frac{1}{K} \sum_{k=1}^{K} \log P_\theta(r_k \mid \mathbf{X}, Q, C, r_{<k})$ | Scalar |
| **Perplexity** | $\text{PPL} = \exp(\mathcal{L})$ | Scalar |

### Next Steps

After completing Stage 5:
1. Model checkpoint saved to `results/{llm_id}/OpenTSLM*/stage5_ecg_cot/checkpoints/best_model.pt`
2. Evaluation metrics saved to `results/{llm_id}/OpenTSLM*/stage5_ecg_cot/results/metrics.json`
3. Predictions saved to `results/{llm_id}/OpenTSLM*/stage5_ecg_cot/results/test_predictions.jsonl`
4. Model now capable of clinical-grade ECG question answering with chain-of-thought reasoning
5. Can be deployed for clinical decision support (with appropriate validation)
6. Can be fine-tuned on specific ECG tasks or other medical time series
7. Evaluation parser available: `evaluation/opentslm/ecg_qa_cot/parse_ecg_qa_cot_data.py`

### Complete Curriculum Learning Pipeline

**Full Training Sequence:**

```bash
# Train all 5 stages in sequence (full curriculum)
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage1_mcq stage2_captioning stage3_cot stage4_sleep_cot stage5_ecg_cot \
    --device cuda \
    --gradient_checkpointing
```

**Or train only Stage 5 (requires Stages 1-4 completed):**

```bash
# Train only Stage 5 (ECG QA CoT)
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage5_ecg_cot \
    --device cuda \
    --gradient_checkpointing  # Recommended for long sequences
```

**Evaluate only (skip training):**

```bash
# Evaluate existing Stage 5 checkpoint
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage5_ecg_cot \
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
Stage 3 (HAR CoT)
    ↓
    [Checkpoint saved]
    ↓
Stage 4 (Sleep CoT)
    ↓
    [Checkpoint saved]
    ↓
Stage 5 (ECG QA CoT)  ← We are here (FINAL STAGE)
    ↓
    [Final trained model: clinical-grade ECG QA with reasoning]
```

### Performance Optimization Tips

**For Better ECG QA Quality:**

1. **Use gradient checkpointing**: Essential for 12-lead ECG (long sequences ~3,000 tokens)
2. **Reduce batch size if needed**: Use batch_size=2 or even 1 if memory constrained
3. **Enable LoRA**: Fine-tune LLM layers for better medical reasoning
4. **Increase epochs**: Complex ECG reasoning benefits from extended training
5. **Monitor per-template F1**: Identifies weak question types
6. **Exclude comparison questions initially**: Focus on simpler question types first

**Memory Optimization:**

- 12-lead ECG encoding creates ~3,000 tokens (very long)
- Total sequence length ~3,500-4,500 tokens (longest in curriculum)
- **MUST use gradient checkpointing** for reasonable memory usage
- Consider using smaller batch size (batch_size=2 or 1)
- Consider excluding comparison questions to reduce sequence length
- Use mixed precision training (automatic with PyTorch)

**Example with Memory Optimizations:**

```bash
# Optimized for limited GPU memory
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id meta-llama/Llama-3.2-1B \
    --stages stage5_ecg_cot \
    --device cuda \
    --gradient_checkpointing \
    --batch_size 2
```

### Clinical Deployment Considerations

For clinical deployment of Stage 5 models:

1. **Validation**: Extensive validation on held-out clinical data required
2. **Regulatory**: Consider regulatory requirements (FDA, CE marking, etc.)
3. **Interpretability**: Chain-of-thought reasoning provides explainability
4. **Uncertainty**: Model outputs "not sure" for uncertain cases
5. **Human oversight**: Always require cardiologist review for clinical decisions
6. **Performance monitoring**: Continuous monitoring in deployment
7. **Ethical considerations**: Fair performance across demographic groups

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

### PTB-XL Dataset

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

**Database**: [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/)

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

### Medical AI and Interpretability

```bibtex
@article{topol2019high,
  title={High-performance medicine: the convergence of human and artificial intelligence},
  author={Topol, Eric J},
  journal={Nature medicine},
  volume={25},
  number={1},
  pages={44--56},
  year={2019}
}
```

---

**End of Stage 5 Detailed Guide - Final Stage of OpenTSLM Curriculum**

**Congratulations!** You have completed the comprehensive guide for the final stage of the OpenTSLM curriculum learning pipeline. The model is now capable of advanced medical reasoning on complex time series data.


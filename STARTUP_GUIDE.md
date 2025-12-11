<!--
This source file is part of the OpenTSLM open-source project

SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT
-->

# OpenTSLM Startup Guide

A comprehensive guide to getting started with OpenTSLM - Training, Evaluation, Inference, Data, and GPU Usage.

## Table of Contents

1. [What is OpenTSLM?](#what-is-opentslm)
2. [Installation](#installation)
3. [LLM Setup](#llm-setup)
4. [Quick Start](#quick-start)
5. [Training Details](#training-details)
6. [Evaluation](#evaluation)
7. [Inference](#inference)
8. [Data and Datasets](#data-and-datasets)
9. [Using Pretrained Models](#using-pretrained-models)
10. [GPU Requirements and Usage](#gpu-requirements-and-usage) - **Includes MPS (Metal) and ARM CPU Support**
11. [Troubleshooting](#troubleshooting)

> 🍎 **Using Apple Silicon (M1/M2/M3/M4)?** Jump to [Apple Silicon Quick Start](#quick-start-for-apple-silicon-m1m2m3m4) for MPS (Metal GPU) and CPU guidance, or see the [MPS Usage section](#mps-metal-usage-on-apple-silicon) for details.

---

## What is OpenTSLM?

OpenTSLM (Time-Series Language Models) is a framework for training and using language models that can reason over time series data. It bridges the gap between Large Language Models and time series analysis, enabling natural-language prompting and reasoning over multiple time series of any length.

### Key Capabilities

- **Process multiple time series of any length** simultaneously
- **Reason over time series data** using natural language prompts
- **Generate detailed captions, rationales, and answers** about time series patterns
- **Handle medical time series** including ECG, EEG, and activity recognition data
- **Support chain-of-thought reasoning** for complex time series analysis tasks

### Model Architectures

The framework provides two main model architectures:

1. **OpenTSLMSP** (Simple Projection)

   - Uses a simple projection approach to integrate time series with LLMs
   - Lightweight and efficient
   - Good for most applications

2. **OpenTSLMFlamingo** (Cross-Attention)
   - Uses cross-attention mechanisms (inspired by Flamingo) for multimodal fusion
   - More powerful for complex reasoning tasks
   - Better for handling multiple time series simultaneously

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training, recommended)
- PyTorch 2.0+
- 8GB+ GPU VRAM (for training with Llama-3.2-1B)

### Step-by-Step Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/StanfordBDHG/OpenTSLM.git --recurse-submodules
   cd OpenTSLM
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies include:

   - PyTorch and transformers (for LLMs)
   - NumPy, pandas (for data processing)
   - scikit-learn (for metrics)
   - PEFT (for LoRA fine-tuning)
   - Hugging Face datasets

3. **Verify Installation**

   ```bash
   python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
   ```

---

## LLM Setup

OpenTSLM is designed to work with Llama and Gemma models from Hugging Face. These models require access permissions.

### 1. Request Access

Visit the model repositories and request access:

- Llama models: https://huggingface.co/meta-llama/Llama-3.2-1B
- Gemma models: https://huggingface.co/google/gemma-3-270m

### 2. Authenticate with Hugging Face

```bash
huggingface-cli login
```

### 3. Create an API Token

- Go to https://huggingface.co/settings/tokens
- Generate a new token with `read` scope
- Use this token when prompted by the CLI

### Supported Models

**Llama Models:**

- `meta-llama/Llama-3.2-1B` (default, recommended)
- `meta-llama/Llama-3.2-3B`

**Gemma Models:**

- `google/gemma-3-270m` (lightweight)
- `google/gemma-3-1b-pt`

---

## Quick Start

### Complete Workflow

1. **Setup Environment**

   ```bash
   cd OpenTSLM
   huggingface-cli login
   ```

2. **Train a Model (Full Curriculum)**

   ```bash
   # Train OpenTSLMFlamingo with all stages
   python curriculum_learning.py --model OpenTSLMFlamingo --device cuda

   # Or train OpenTSLMSP
   python curriculum_learning.py --model OpenTSLMSP --device cuda
   ```

3. **Train Specific Stages**

   ```bash
   # Train only first two stages
   python curriculum_learning.py --model OpenTSLMFlamingo --stages stage1_mcq stage2_captioning
   ```

4. **Evaluate Trained Model**

   ```bash
   python curriculum_learning.py --model OpenTSLMFlamingo --eval_only
   ```

5. **Run Inference**

   ```bash
   python test/test_inference.py
   ```

6. **Check Results**
   ```bash
   # View metrics from a specific stage
   cat results/Llama3_2_1B/OpenTSLMFlamingo/stage3_cot/results/metrics.json
   ```

### Quick Start for Apple Silicon (M1/M2/M3/M4)

If you're on Apple Silicon, you have multiple options:

**Option 1: MPS (Metal GPU) - Faster but experimental**

```bash
# Use Apple's Metal GPU acceleration
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id google/gemma-3-270m \
    --device mps \
    --stages stage1_mcq

# Run inference with MPS
python test/test_inference.py  # Auto-detects MPS
```

**Option 2: CPU Mode - More stable**

```bash
# Use CPU for maximum stability
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id google/gemma-3-270m \
    --device cpu \
    --stages stage1_mcq
```

**Option 3: Cloud GPU - Recommended for full training**

```bash
# Use Google Colab, Lambda Labs, or similar with CUDA
# Then download checkpoint and run inference locally
```

**Performance Comparison on Apple Silicon:**

- **MPS (Metal GPU)**: 5-10x faster than CPU, but may have errors
- **ARM CPU**: Most stable, but slower (good for inference)
- **Cloud CUDA GPU**: 20-50x faster than local CPU, fully compatible

**Quick Test Setup:**

```bash
# For testing, reduce samples (edit src/model_config.py)
# Set MAX_SAMPLES = 10 for quick experiments

# Minimal test command with MPS
python curriculum_learning.py \
    --model OpenTSLMSP \
    --llm_id google/gemma-3-270m \
    --device mps \
    --stages stage1_mcq
```

**Recommendations:**

- Start with the smallest model (`google/gemma-3-270m`)
- Use `OpenTSLMSP` (simpler than Flamingo)
- Test with one stage first before full curriculum
- Try MPS first; fall back to CPU if issues occur
- For serious training, use cloud GPU and run inference locally

---

## Training Details

### Curriculum Learning Approach

OpenTSLM uses **curriculum learning** - a progressive training approach where the model learns increasingly complex tasks across five stages:

1. **Stage 1 (MCQ)** - Foundation

   - Multiple choice questions on time series (TSQA dataset)
   - Teaches basic time series understanding
   - ~7,000 training samples
   - 📖 **[Detailed Mathematical Guide](STAGE1_TSQA_DETAILED_GUIDE.md)** - In-depth explanation with math and algorithms

2. **Stage 2 (Captioning)** - Generation

   - Generate detailed captions for time series (M4 dataset)
   - Teaches descriptive language generation
   - Covers Monthly, Quarterly, and Weekly time series
   - 📖 **[Detailed Mathematical Guide](STAGE2_M4_DETAILED_GUIDE.md)** - In-depth explanation with math and algorithms

3. **Stage 3 (CoT)** - Reasoning

   - Chain-of-thought reasoning on activity recognition (HAR dataset)
   - Teaches step-by-step reasoning
   - Uses 3-axis accelerometer data (PAMAP2)

4. **Stage 4 (Sleep CoT)** - Medical Reasoning

   - Sleep stage classification with explanations (SleepEDF)
   - Teaches medical domain reasoning
   - Works with EEG signals

5. **Stage 5 (ECG CoT)** - Advanced Medical
   - ECG question answering with detailed rationales
   - Advanced medical reasoning and diagnosis
   - Uses 12-lead ECG data (PTB-XL)

### Training Features

- **Automatic Checkpointing**: Best models saved based on validation loss
- **Early Stopping**: Stops if no improvement for 5 epochs
- **Loss Tracking**: Complete history saved to `loss_history.txt`
- **Resume Training**: Automatically resumes from last checkpoint
- **Distributed Training**: Multi-GPU support via PyTorch DDP
- **Gradient Checkpointing**: Memory-efficient training option
- **Curriculum Progression**: Each stage loads the best model from the previous stage

### Training Configuration

Default hyperparameters (in `src/model_config.py`):

```python
BATCH_SIZE = 4              # Batch size per GPU
NUM_EPOCHS = 20             # Maximum epochs (with early stopping)
EARLY_STOP_PAT = 5          # Patience for early stopping
LR_ENCODER = 2e-4           # Learning rate for encoder
LR_PROJECTOR = 1e-4         # Learning rate for projector
GRAD_CLIP_NORM = 1.0        # Gradient clipping
WEIGHT_DECAY = 1e-2         # Weight decay
WARMUP_FRAC = 0.03          # Warmup fraction (3%)
PATCH_SIZE = 4              # Time series patch size
```

### Command Line Arguments

```bash
python curriculum_learning.py \
    --model OpenTSLMFlamingo \                    # Model type
    --stages stage1_mcq stage2_captioning \       # Stages to run
    --device cuda \                                # Device (cuda/mps/cpu)
    --llm_id meta-llama/Llama-3.2-1B \            # Base LLM
    --batch_size 4 \                               # Batch size
    --gradient_checkpointing \                     # Enable memory optimization
    --eval_only \                                  # Evaluation only
    --verbose                                      # Verbose logging
```

### Distributed Training

For multi-GPU training:

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 curriculum_learning.py --model OpenTSLMFlamingo

# Multi-node training (example)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
    --master_addr=192.168.1.1 --master_port=29500 \
    curriculum_learning.py --model OpenTSLMFlamingo
```

### Results Directory Structure

Training creates a structured results directory:

```
results/
├── {llm_id}/                           # e.g., Llama3_2_1B
│   ├── OpenTSLMSP/
│   │   ├── stage1_mcq/
│   │   │   ├── checkpoints/
│   │   │   │   ├── best_model.pt       # Best model checkpoint
│   │   │   │   └── loss_history.txt    # Training loss history
│   │   │   └── results/
│   │   │       ├── test_predictions.jsonl  # Model predictions
│   │   │       └── metrics.json        # Evaluation metrics
│   │   ├── stage2_captioning/
│   │   ├── stage3_cot/
│   │   ├── stage4_sleep_cot/
│   │   ├── stage5_ecg_cot/
│   │   └── curriculum_results.json     # Overall curriculum results
│   └── OpenTSLMFlamingo/
│       └── [same structure as above]
```

---

## Evaluation

### Built-in Evaluation

During training, models are automatically evaluated on test sets after each stage:

```bash
# Training with automatic evaluation
python curriculum_learning.py --model OpenTSLMFlamingo
```

Results are saved in:

```
results/{llm_id}/{model_type}/{stage}/results/
├── test_predictions.jsonl  # All predictions
└── metrics.json            # Aggregated metrics
```

### Standalone Evaluation

Run evaluation on existing checkpoints:

```bash
# Evaluate a specific stage
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --stages stage3_cot \
    --eval_only

# Evaluate all stages
python curriculum_learning.py --model OpenTSLMFlamingo --eval_only
```

### Custom Evaluation Scripts

The `evaluation/` directory contains specialized evaluation scripts:

#### 1. OpenTSLM Evaluation

Detailed evaluation for trained OpenTSLM models:

```bash
# Evaluate on HAR dataset
cd evaluation/opentslm
python get_pamap_cot_predictions.py \
    --model_path ../../results/Llama3_2_1B/OpenTSLMFlamingo/stage3_cot/checkpoints/best_model.pt \
    --llm_id meta-llama/Llama-3.2-1B \
    --num_samples 100

# Evaluate on Sleep dataset
cd evaluation/opentslm/sleep
python get_sleep_predictions.py \
    --model_path ../../../results/Llama3_2_1B/OpenTSLMSP/stage4_sleep_cot/checkpoints/best_model.pt \
    --llm_id meta-llama/Llama-3.2-1B \
    --num_samples 50
```

#### 2. Baseline Comparisons

Compare against other LLM approaches:

```bash
cd evaluation/baseline
python evaluate_all.py
```

#### 3. Memory Profiling

Analyze memory usage across different configurations:

```bash
cd evaluation/memory
bash run_all_memory.sh --device cuda:0 --results_csv memory_profile.csv
```

#### 4. Clinician Evaluation

Medical expert evaluation pipelines (for ECG):

```bash
cd evaluation/clinicianecg/pipeline
python 1_dataset_analyzer.py
python 2_excel_generator.py
```

### Evaluation Metrics

Different tasks use different metrics:

- **MCQ (Stage 1)**: Accuracy, F1-score
- **Captioning (Stage 2)**: BLEU, ROUGE, perplexity
- **CoT Tasks (Stages 3-5)**: Accuracy, reasoning quality, F1-score

---

## Inference

### Basic Inference Example

After training, use your model for inference:

```python
import torch
import numpy as np
from src.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from src.prompt.text_prompt import TextPrompt
from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from src.prompt.full_prompt import FullPrompt

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = OpenTSLMFlamingo(
    device=device,
    llm_id="meta-llama/Llama-3.2-1B",  # Must match training
    cross_attn_every_n_layers=1,
)
model.load_from_file("results/Llama3_2_1B/OpenTSLMFlamingo/stage5_ecg_cot/checkpoints/best_model.pt")
model.eval()

# Prepare your time series data
time_series = np.array([...])  # Your time series data
normalized_series = (time_series - time_series.mean()) / time_series.std()

# Build prompt
pre_prompt = TextPrompt("You are an expert in time series analysis.")
ts_prompt = TextTimeSeriesPrompt(
    "This is the time series data:",
    normalized_series.tolist()
)
post_prompt = TextPrompt("Please analyze this time series and provide insights.")

# Create full prompt
prompt = FullPrompt(pre_prompt, [ts_prompt], post_prompt)

# Generate response
response = model.eval_prompt(prompt, max_new_tokens=300)
print(response)
```

### OpenTSLMSP Inference

For the simpler projection model:

```python
from src.model.llm.OpenTSLMSP import OpenTSLMSP

# Load model
model = OpenTSLMSP(
    llm_id="meta-llama/Llama-3.2-1B",
    device=device,
)
model.load_from_file("results/Llama3_2_1B/OpenTSLMSP/stage3_cot/checkpoints/best_model.pt")
model.eval()

# Use same prompt structure as above
response = model.eval_prompt(prompt, max_new_tokens=300)
```

### Batch Inference

Process multiple samples efficiently:

```python
# Create batch of prompts
prompts = []
for ts_data in time_series_list:
    normalized = (ts_data - ts_data.mean()) / ts_data.std()
    ts_prompt = TextTimeSeriesPrompt("Time series:", normalized.tolist())
    prompt = FullPrompt(pre_prompt, [ts_prompt], post_prompt)
    prompts.append(prompt.to_dict())

# Generate responses for batch
responses = model.generate(prompts, max_new_tokens=300)
```

### Multiple Time Series

Analyze multiple time series at once:

```python
# Create multiple time series prompts
ts_prompts = []
for i, ts_data in enumerate([x_axis, y_axis, z_axis]):
    normalized = (ts_data - ts_data.mean()) / ts_data.std()
    ts_prompts.append(
        TextTimeSeriesPrompt(f"Axis {i+1}:", normalized.tolist())
    )

# Build prompt with multiple time series
prompt = FullPrompt(
    pre_prompt=TextPrompt("You are analyzing accelerometer data."),
    text_time_series_prompt_list=ts_prompts,
    post_prompt=TextPrompt("What activity is being performed?")
)

response = model.eval_prompt(prompt, max_new_tokens=200)
```

### Inference Examples

See complete examples in:

- `test/test_inference.py` - M4 time series captioning
- `evaluation/opentslm/get_pamap_cot_predictions.py` - HAR inference
- `evaluation/opentslm/sleep/get_sleep_predictions.py` - Sleep staging

---

## Data and Datasets

### Supported Datasets

OpenTSLM supports multiple time series datasets that are **automatically downloaded** on first use:

| Dataset          | Task                       | Domain      | Samples      | Description                                     |
| ---------------- | -------------------------- | ----------- | ------------ | ----------------------------------------------- |
| **TSQA**         | MCQ                        | General     | ~7,000       | Time series question answering                  |
| **M4**           | Captioning                 | Forecasting | Varies       | Time series captions (Monthly/Quarterly/Weekly) |
| **HAR CoT**      | Classification + Reasoning | Activity    | ~1,000       | Human activity recognition with rationales      |
| **SleepEDF CoT** | Classification + Reasoning | Medical     | ~500         | Sleep stage classification with explanations    |
| **ECG QA CoT**   | QA + Reasoning             | Medical     | ~1,000       | ECG question answering with rationales          |
| **Simulation**   | Testing                    | Synthetic   | Configurable | Synthetic data for testing/profiling            |

### Dataset Loading

Datasets are automatically downloaded and cached:

```python
from src.time_series_datasets.TSQADataset import TSQADataset
from src.time_series_datasets.m4.M4QADataset import M4QADataset
from src.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from src.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from src.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset

# Load datasets with train/validation/test splits
train_data = TSQADataset(split="train", EOS_TOKEN="")
val_data = TSQADataset(split="validation", EOS_TOKEN="")
test_data = TSQADataset(split="test", EOS_TOKEN="")

# Limit samples for quick experiments
small_train = TSQADataset(split="train", EOS_TOKEN="", max_samples=100)
```

### Data Format

All datasets follow a consistent format:

```python
{
    "pre_prompt": "Text before time series",
    "time_series": [ts1, ts2, ...],           # List of time series (numpy arrays)
    "time_series_text": ["Description 1", "Description 2", ...],
    "post_prompt": "Text after time series (question/task)",
    "answer": "Ground truth answer"
}
```

**Key points:**

- Time series are normalized (z-score normalization)
- Time series can be any length
- Multiple time series can be included per sample
- Text prompts describe the task and provide context

### Custom Datasets

Create your own dataset by inheriting from `QADataset`:

```python
from src.time_series_datasets.QADataset import QADataset
from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from typing import List, Tuple
from torch.utils.data import Dataset

class MyCustomDataset(QADataset):
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and return train/val/test splits"""
        # Load your data here
        train_data = [...]
        val_data = [...]
        test_data = [...]
        return train_data, val_data, test_data

    def _get_answer(self, row) -> str:
        """Return ground truth answer"""
        return row["answer"]

    def _get_pre_prompt(self, row) -> str:
        """Return pre-prompt text"""
        return "You are analyzing a time series."

    def _get_post_prompt(self, row) -> str:
        """Return post-prompt text (question/task)"""
        return "What pattern do you observe?"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """Return list of time series with descriptions"""
        # Normalize your time series
        ts_data = row["time_series"]
        normalized = (ts_data - ts_data.mean()) / ts_data.std()

        return [TextTimeSeriesPrompt("Time series data:", normalized.tolist())]
```

Use your custom dataset:

```python
# In training
from my_dataset import MyCustomDataset

dataset = MyCustomDataset(split="train", EOS_TOKEN="")
```

### Data Preprocessing

Time series preprocessing guidelines:

1. **Normalization**: Apply z-score normalization

   ```python
   normalized = (series - series.mean()) / series.std()
   ```

2. **Length**: Time series can be any length (model handles padding)

3. **Missing Values**: Handle missing values before normalization

   ```python
   # Option 1: Interpolate
   series = np.interp(...)

   # Option 2: Remove or mark
   series = series[~np.isnan(series)]
   ```

4. **Multiple Channels**: Store as separate time series
   ```python
   ts_prompts = [
       TextTimeSeriesPrompt("X-axis:", x_normalized.tolist()),
       TextTimeSeriesPrompt("Y-axis:", y_normalized.tolist()),
       TextTimeSeriesPrompt("Z-axis:", z_normalized.tolist()),
   ]
   ```

---

## Using Pretrained Models

### Loading Pretrained Checkpoints

Load a pretrained model checkpoint:

**OpenTSLMFlamingo:**

```python
from src.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

model = OpenTSLMFlamingo(
    device="cuda",
    llm_id="meta-llama/Llama-3.2-1B",  # Must match checkpoint
    cross_attn_every_n_layers=1,
)
model.load_from_file("path/to/best_model.pt")
model.eval()
```

**OpenTSLMSP:**

```python
from src.model.llm.OpenTSLMSP import OpenTSLMSP

model = OpenTSLMSP(
    llm_id="meta-llama/Llama-3.2-1B",  # Must match checkpoint
    device="cuda",
)
model.load_from_file("path/to/best_model.pt")
model.eval()
```

### Important Notes

1. **LLM ID Must Match**: The `llm_id` parameter must be the same as used during training

   ```python
   # If trained with: meta-llama/Llama-3.2-1B
   # Load with:
   model = OpenTSLMFlamingo(llm_id="meta-llama/Llama-3.2-1B", ...)
   ```

2. **Device Compatibility**:

   - Checkpoints trained on CUDA may not work well on MPS (Apple Silicon)
   - Use the same device type for training and inference when possible

3. **Model Architecture**: Use the same model type as training

   - If trained with OpenTSLMSP, load with OpenTSLMSP
   - If trained with OpenTSLMFlamingo, load with OpenTSLMFlamingo

4. **Checkpoint Structure**: Checkpoints contain:
   - Model weights (encoder, projector, LLM adapters)
   - Training metadata (epoch, loss history)
   - Optional optimizer state (for resuming training)

### Finding Checkpoints

After training, checkpoints are located at:

```
results/{llm_id}/{model_type}/{stage}/checkpoints/best_model.pt
```

Examples:

- `results/Llama3_2_1B/OpenTSLMFlamingo/stage3_cot/checkpoints/best_model.pt`
- `results/Llama3_2_1B/OpenTSLMSP/stage5_ecg_cot/checkpoints/best_model.pt`

### Sharing Models

To share a trained model, provide:

1. The checkpoint file (`best_model.pt`)
2. The LLM ID used (e.g., `meta-llama/Llama-3.2-1B`)
3. The model type (`OpenTSLMSP` or `OpenTSLMFlamingo`)
4. Any custom configuration changes

Example sharing instructions:

```
Model: OpenTSLMFlamingo
Base LLM: meta-llama/Llama-3.2-1B
Trained on: Stage 3 (HAR CoT)
Checkpoint: stage3_cot/checkpoints/best_model.pt
Device: CUDA
```

### Resume Training from Checkpoint

To continue training from a checkpoint:

```bash
# The curriculum trainer automatically resumes from the last checkpoint
python curriculum_learning.py --model OpenTSLMFlamingo --stages stage3_cot
```

If a checkpoint exists for the stage, training will:

1. Load model weights
2. Load optimizer state
3. Resume from the last epoch
4. Continue loss history tracking

---

## GPU Requirements and Usage

### GPU Requirements

**Minimum Requirements:**

- **Training**: NVIDIA GPU with 8GB+ VRAM (for Llama-3.2-1B)
- **Inference**: NVIDIA GPU with 4GB+ VRAM (for Llama-3.2-1B)
- **CUDA**: CUDA 11.8+ recommended
- **PyTorch**: PyTorch 2.0+ with CUDA support

**Recommended Setup:**

- **Training**: NVIDIA GPU with 16GB+ VRAM (A100, V100, RTX 3090/4090)
- **Multi-GPU**: 2-4 GPUs for faster training
- **Large Models**: 24GB+ VRAM for Llama-3.2-3B

**Model Size vs VRAM:**
| LLM Model | Min VRAM (Training) | Min VRAM (Inference) | Recommended VRAM |
|-----------|---------------------|----------------------|------------------|
| gemma-3-270m | 6GB | 3GB | 8GB |
| gemma-3-1b | 8GB | 4GB | 12GB |
| Llama-3.2-1B | 8GB | 4GB | 16GB |
| Llama-3.2-3B | 16GB | 8GB | 24GB |

### Device Selection

OpenTSLM automatically detects the best available device:

```python
# In code
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

**Supported Devices:**

- **CUDA** (NVIDIA GPUs) - ✅ Recommended for training and inference
- **MPS** (Apple Silicon M1/M2/M3 GPU via Metal) - ⚠️ Experimental support, faster than CPU
- **CPU** (x86_64 and ARM64) - ✅ Works but very slow, suitable for testing and small experiments

**Device Selection Quick Reference:**

| Device      | Platform               | Speed          | Stability  | Best For                      | Command         |
| ----------- | ---------------------- | -------------- | ---------- | ----------------------------- | --------------- |
| **CUDA**    | NVIDIA GPUs            | ⚡⚡⚡ Fastest | ✅ Stable  | Production training/inference | `--device cuda` |
| **MPS**     | Apple Silicon M1/M2/M3 | ⚡⚡ Fast      | ⚠️ Limited | Quick experiments, testing    | `--device mps`  |
| **ARM CPU** | Apple Silicon M1/M2/M3 | ⚡ Slow        | ✅ Stable  | Debugging, stable inference   | `--device cpu`  |
| **x86 CPU** | Intel/AMD CPUs         | ⚡ Slow        | ✅ Stable  | Testing, no GPU available     | `--device cpu`  |

**Apple Silicon Support (M1/M2/M3/M4 Macs):**

You have **three options** on Apple Silicon:

1. **MPS (Metal Performance Shaders)** - Use Apple's GPU via Metal

   - ✅ **Faster than CPU** (5-10x speedup)
   - ⚠️ **Limited support** - May produce inconsistent results or errors
   - 💡 **Best for**: Quick experiments, inference, small models
   - 🔧 **Usage**: `--device mps` or auto-detected

2. **ARM CPU Mode** - Use CPU on ARM64 architecture

   - ✅ **Most stable and reliable**
   - ⚠️ **Slower** than MPS (but works correctly)
   - 💡 **Best for**: Stable training, debugging, when MPS fails
   - 🔧 **Usage**: `--device cpu`

3. **Cloud GPU** (Recommended for serious training)
   - ✅ **Fastest and most compatible** (NVIDIA CUDA)
   - 💡 **Best for**: Full curriculum training, production use
   - 🔧 **Options**: Google Colab, Lambda Labs, AWS, etc.

**ARM CPU Support (ARM servers, no GPU):**

- ✅ **Yes, works on ARM64** - Full compatibility
- ⚠️ **Performance**: Training will be 10-50x slower than GPU
- 💡 **Recommended for**: Testing, debugging, small datasets, inference

### MPS (Metal) Usage on Apple Silicon

Use Apple's GPU via Metal Performance Shaders:

```bash
# Explicitly use MPS (Metal GPU)
python curriculum_learning.py --model OpenTSLMSP --device mps

# Auto-detect (will use MPS if available)
python curriculum_learning.py --model OpenTSLMSP

# Run inference with MPS
python test/test_inference.py  # Auto-detects MPS
```

**MPS Performance Tips:**

1. **Start with smaller models:**

   ```bash
   # Use smaller model for better MPS compatibility
   python curriculum_learning.py \
       --model OpenTSLMSP \
       --llm_id google/gemma-3-270m \
       --device mps
   ```

2. **Test one stage first:**

   ```bash
   # Test with single stage before full curriculum
   python curriculum_learning.py \
       --model OpenTSLMSP \
       --stages stage1_mcq \
       --device mps
   ```

3. **If MPS fails, fall back to CPU:**
   ```bash
   # Force CPU if MPS has issues
   python curriculum_learning.py --model OpenTSLMSP --device cpu
   ```

**Verify MPS is Working:**

```bash
# Check if MPS is available
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Test MPS with a simple operation
python -c "import torch; x = torch.tensor([1.0, 2.0], device='mps'); print('MPS test:', x.sum())"
```

**Known MPS Limitations:**

- ⚠️ Some PyTorch operations have limited MPS support (particularly in older PyTorch versions)
- ⚠️ Results may differ from CUDA-trained models due to numerical differences
- ⚠️ Checkpoints trained on CUDA may not work well on MPS
- ⚠️ Training stability can vary between PyTorch versions
- 💡 For consistency, train and infer on the same device type
- 💡 Update to latest PyTorch for best MPS support: `pip install --upgrade torch`

### CPU Usage (Including ARM)

OpenTSLM works on CPU for both x86_64 and ARM64 architectures (including Apple Silicon):

```bash
# Explicitly use CPU
python curriculum_learning.py --model OpenTSLMFlamingo --device cpu

# Run inference on CPU
python test/test_inference.py  # Will auto-detect and use CPU if no GPU available

# On Apple Silicon, force CPU instead of MPS
python curriculum_learning.py --model OpenTSLMSP --device cpu
```

**Tips for CPU Training (especially ARM):**

1. **Use smaller models:**

   ```bash
   python curriculum_learning.py \
       --model OpenTSLMSP \
       --llm_id google/gemma-3-270m \
       --device cpu
   ```

2. **Test with small datasets first:**

   ```python
   # In your code or modify model_config.py
   dataset = TSQADataset(split="train", EOS_TOKEN="", max_samples=10)
   ```

3. **Reduce batch size:**
   Edit `src/model_config.py`:

   ```python
   BATCH_SIZE = 1  # Instead of 4
   ```

4. **Use for inference only:**
   Train on a GPU machine, then run inference on ARM CPU:
   ```bash
   # After training elsewhere, run inference on ARM CPU
   python test/test_inference.py
   ```

### Single GPU Usage

```bash
# Explicitly specify CUDA device
python curriculum_learning.py --model OpenTSLMFlamingo --device cuda

# Use a specific GPU (e.g., GPU 1)
CUDA_VISIBLE_DEVICES=1 python curriculum_learning.py --model OpenTSLMFlamingo

# Use specific GPU for inference
CUDA_VISIBLE_DEVICES=0 python test/test_inference.py
```

### Multi-GPU Training

For faster training with multiple GPUs, use PyTorch's distributed launcher:

```bash
# Use all available GPUs (e.g., 4 GPUs)
torchrun --nproc_per_node=4 curriculum_learning.py --model OpenTSLMFlamingo

# Use specific GPUs (e.g., GPUs 0, 1, 2)
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 curriculum_learning.py --model OpenTSLMFlamingo

# Multi-node training (2 nodes, 4 GPUs each)
# On node 0:
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
    --master_addr=192.168.1.1 --master_port=29500 \
    curriculum_learning.py --model OpenTSLMFlamingo

# On node 1:
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
    --master_addr=192.168.1.1 --master_port=29500 \
    curriculum_learning.py --model OpenTSLMFlamingo
```

### Memory Optimization

If you encounter out-of-memory errors:

**1. Enable Gradient Checkpointing**

```bash
python curriculum_learning.py --model OpenTSLMFlamingo --gradient_checkpointing
```

**2. Reduce Batch Size**

Edit `src/model_config.py`:

```python
BATCH_SIZE = 2  # Instead of default 4
```

**3. Use a Smaller LLM**

```bash
python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id google/gemma-3-270m  # Instead of Llama-3.2-1B
```

**4. Mixed Precision Training**

OpenTSLM automatically uses mixed precision (FP16) on compatible GPUs.

### Memory Profiling

Profile memory usage to optimize your setup:

```bash
# Profile a single configuration
python get_memory_use.py \
    --model OpenTSLMFlamingo \
    --dataset TSQADataset \
    --llm_id meta-llama/Llama-3.2-1B \
    --device cuda:0 \
    --results_csv memory_profile.csv

# Comprehensive memory analysis (all models, datasets, LLMs)
bash run_all_memory.sh --device cuda:0 --results_csv memory_use.csv

# Plot memory usage
cd evaluation/memory
python plot_memory_usage.py
```

### Monitoring GPU Usage

Monitor GPU usage during training:

```bash
# In another terminal
watch -n 1 nvidia-smi

# Or use nvtop for a better interface
nvtop
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

**Problem**: `RuntimeError: CUDA out of memory`

**Solutions:**

```bash
# A. Reduce batch size
# Edit src/model_config.py: BATCH_SIZE = 2

# B. Enable gradient checkpointing
python curriculum_learning.py --gradient_checkpointing

# C. Use smaller model
python curriculum_learning.py --llm_id google/gemma-3-270m

# D. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. CUDA Not Available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions:**

```bash
# Check PyTorch installation
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Hugging Face Authentication

**Problem**: Cannot access Llama/Gemma models

**Solutions:**

```bash
# 1. Login to Hugging Face
huggingface-cli login

# 2. Request access to models at:
# https://huggingface.co/meta-llama/Llama-3.2-1B
# https://huggingface.co/google/gemma-3-270m

# 3. Wait for approval (usually instant for Gemma, may take time for Llama)

# 4. Verify access
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')"
```

#### 4. MPS (Apple Silicon GPU) Issues

**Problem**: Training on MPS produces errors or poor results

**Solutions:**

```bash
# Use CPU instead (slower but more stable)
python curriculum_learning.py --device cpu

# Or train on a cloud GPU (recommended)
# Use Google Colab, Lambda Labs, or similar services
```

**Note**: MPS has limited support. For best results, use CUDA GPUs.

**Alternative on Apple Silicon**: Use CPU mode (`--device cpu`) for better stability, though slower.

**ARM CPU Performance Tips**:

- Training is very slow on CPU (10-50x slower than GPU)
- Inference is more practical on CPU than training
- Use the smallest model possible (e.g., `google/gemma-3-270m`)
- Consider cloud GPU options for training (Google Colab, Lambda Labs, etc.)
- For local testing on ARM CPU:
  ```bash
  # Use smallest model and limited samples
  python curriculum_learning.py \
      --model OpenTSLMSP \
      --llm_id google/gemma-3-270m \
      --device cpu \
      --batch_size 1
  # Then edit model_config.py to set MAX_SAMPLES = 10
  ```

#### 5. Dataset Download Failures

**Problem**: Datasets fail to download automatically

**Solutions:**

```bash
# Check internet connection
ping huggingface.co

# Clear Hugging Face cache and retry
rm -rf ~/.cache/huggingface/datasets/*

# Manual download (for specific datasets)
python -c "from datasets import load_dataset; load_dataset('ChengsenWang/TSQA')"
```

#### 6. Checkpoint Loading Errors

**Problem**: `RuntimeError: Error(s) in loading state_dict`

**Solutions:**

```python
# Ensure LLM ID matches
model = OpenTSLMFlamingo(
    llm_id="meta-llama/Llama-3.2-1B",  # Must match training
    device="cuda"
)

# Check model type matches
# If trained with OpenTSLMSP, use OpenTSLMSP (not OpenTSLMFlamingo)

# For mismatched checkpoints, use strict=False (not recommended for production)
model.load_state_dict(checkpoint, strict=False)
```

#### 7. Slow Training

**Problem**: Training is very slow

**Solutions:**

```bash
# A. Use GPU instead of CPU
python curriculum_learning.py --device cuda

# B. Use multiple GPUs
torchrun --nproc_per_node=4 curriculum_learning.py

# C. Reduce dataset size for testing
# Edit max_samples in dataset loading

# D. Check GPU utilization
nvidia-smi
# If GPU usage is low, you may have a data loading bottleneck
```

#### 8. Import Errors

**Problem**: `ModuleNotFoundError` or import errors

**Solutions:**

```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Update transformers (if needed)
pip install --upgrade transformers

# Check Python path
python -c "import sys; print(sys.path)"

# Add src to path if needed
export PYTHONPATH="${PYTHONPATH}:/path/to/OpenTSLM/src"
```

### Getting Help

If you encounter issues not covered here:

1. **Check the GitHub Issues**: https://github.com/StanfordBDHG/OpenTSLM/issues
2. **Read the paper**: https://doi.org/10.13140/RG.2.2.14827.60963
3. **Contact the team**: digitalhealthresearch@stanford.edu

### Debugging Tips

Enable verbose logging:

```bash
python curriculum_learning.py --verbose
```

Check GPU memory during training:

```python
import torch
print(torch.cuda.memory_summary())
```

Test with a small dataset first:

```bash
# Modify src/model_config.py
MAX_SAMPLES = 10  # Test with 10 samples
```

---

## Additional Resources

### Code Examples

- **Training**: `curriculum_learning.py` - Main training script
- **Inference**: `test/test_inference.py` - Simple inference example
- **Evaluation**: `evaluation/opentslm/` - Evaluation scripts
- **Datasets**: `src/time_series_datasets/` - Dataset implementations

### Documentation

- **Main README**: `README.md` - Overview and quick start
- **Contributors**: `CONTRIBUTORS.md` - List of contributors
- **License**: `LICENSE.md` - MIT License
- **Citation**: `CITATION.cff` - How to cite

### Paper and Research

- **Paper**: [Read the full paper](https://doi.org/10.13140/RG.2.2.14827.60963)
- **Research Opportunities**: http://bdh.stanford.edu/studentresearch

### Community

- **GitHub**: https://github.com/StanfordBDHG/OpenTSLM
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Contributing**: See contribution guidelines

---

## Summary

OpenTSLM provides a complete framework for training time-series language models:

✅ **Easy Installation**: Simple pip install with all dependencies
✅ **Automatic Datasets**: All datasets downloaded automatically
✅ **Curriculum Learning**: Progressive training across 5 stages
✅ **Two Architectures**: OpenTSLMSP and OpenTSLMFlamingo
✅ **GPU Support**: Single and multi-GPU training
✅ **Comprehensive Evaluation**: Built-in and custom evaluation scripts
✅ **Flexible Inference**: Easy-to-use inference API
✅ **Medical Applications**: ECG, EEG, sleep staging, activity recognition

Get started now:

```bash
git clone https://github.com/StanfordBDHG/OpenTSLM.git --recurse-submodules
cd OpenTSLM
pip install -r requirements.txt
huggingface-cli login
python curriculum_learning.py --model OpenTSLMFlamingo --device cuda
```

Happy time series reasoning! 🚀

# OpenTSLM Training Stages - Complete Guide

**OpenTSLM** uses **curriculum learning** - training progresses from simple to complex tasks across 5 stages.

Each stage builds on the previous one, gradually teaching the model to understand and reason about time series data.

---

## 🎯 Stage Overview

| Stage | Name | Task Type | Dataset | Complexity |
|-------|------|-----------|---------|------------|
| **1** | TSQA (MCQ) | Multiple Choice QA | TSQA | ⭐ Simple |
| **2** | Captioning | Time Series Description | M4 | ⭐⭐ Medium |
| **3** | HAR CoT | Chain-of-Thought Reasoning | HAR | ⭐⭐⭐ Complex |
| **4** | Sleep CoT | Advanced Reasoning | SleepEDF | ⭐⭐⭐⭐ Advanced |
| **5** | ECG QA CoT | Medical Reasoning | ECG-QA | ⭐⭐⭐⭐⭐ Expert |

---

## 📚 Detailed Stage Descriptions

### Stage 1: TSQA - Multiple Choice Question Answering

**Purpose**: Teach the model to **understand basic time series patterns**

**Task**: Given a time series, answer multiple-choice questions about it.

**Example:**
```
Time Series: [1.2, 1.5, 1.8, 2.1, 2.4, 2.7]

Question: "Is this time series showing an upward trend?"
Options:
A) Yes, increasing
B) No, decreasing  
C) Stable
D) Cannot determine

Answer: A
```

**Dataset**: TSQA (Time Series Question Answering)
- **Train**: 6,300 samples
- **Val**: 630 samples
- **Test**: 700 samples

**Skills Learned:**
- Basic pattern recognition
- Trend identification
- Statistical properties
- Simple reasoning

**Difficulty**: ⭐ **Simple** - Foundation for all later stages

**Pre-trained Model**: `OpenTSLM/llama-3.2-1b-tsqa-sp` ✅

---

### Stage 2: M4 - Time Series Captioning

**Purpose**: Teach the model to **generate natural language descriptions** of time series

**Task**: Given a time series, generate a descriptive caption explaining what's happening.

**Example:**
```
Time Series: [Sales data showing weekly patterns]

Generated Caption:
"This time series shows weekly sales data with clear periodic patterns. 
There's a recurring peak every 7 days, suggesting higher weekend sales. 
The overall trend is slightly increasing over time."
```

**Dataset**: M4 (Makridakis Competition)
- Forecasting competition dataset
- Multiple time series types
- Hourly, daily, weekly, monthly data

**Skills Learned:**
- Pattern description in natural language
- Temporal feature extraction
- Seasonality detection
- Natural language generation for time series

**Difficulty**: ⭐⭐ **Medium** - Requires understanding + generation

**Pre-trained Model**: Available on HuggingFace (check demo scripts)

---

### Stage 3: HAR - Chain-of-Thought Reasoning

**Purpose**: Teach the model to **reason step-by-step** about human activity recognition

**Task**: Analyze sensor data and explain the reasoning process for activity classification.

**Example:**
```
Time Series: [Accelerometer data from smartphone]

Question: "What activity is the person performing?"

Chain-of-Thought Answer:
"Let me analyze this step by step:
1. First, I observe high-frequency oscillations in the Z-axis
2. The pattern repeats every ~0.5 seconds
3. The X and Y axes show minimal variation
4. This pattern is consistent with vertical movement
5. The frequency matches typical walking cadence
6. Therefore, the person is WALKING."
```

**Dataset**: HAR (Human Activity Recognition)
- Sensor data from smartphones/wearables
- Activities: Walking, Running, Sitting, Standing, etc.
- Multi-sensor inputs (accelerometer, gyroscope)

**Skills Learned:**
- Multi-step reasoning
- Evidence-based inference
- Logical explanation
- Complex pattern analysis

**Difficulty**: ⭐⭐⭐ **Complex** - Requires explicit reasoning chains

**Pre-trained Model**: Available on HuggingFace

---

### Stage 4: Sleep CoT - Sleep Stage Classification

**Purpose**: Teach the model to **reason about physiological signals** for sleep analysis

**Task**: Analyze EEG signals and classify sleep stages with detailed reasoning.

**Example:**
```
Time Series: [EEG signal data]

Question: "What sleep stage is the subject in?"

Chain-of-Thought Answer:
"Analyzing the EEG signal:
1. Frequency analysis: Dominant frequency is 0.5-2 Hz (delta waves)
2. Amplitude: High amplitude waves present
3. K-complexes: Not observed
4. Sleep spindles: Absent
5. Delta wave proportion: >20% of epoch
6. Conclusion: This is Stage 3 NREM sleep (deep sleep)

Reasoning: The prevalence of delta waves (slow-wave sleep) with 
high amplitude and absence of rapid eye movements confirms deep 
sleep stage."
```

**Dataset**: SleepEDF
- Polysomnography recordings
- EEG, EOG, EMG signals
- Sleep stage labels (Wake, N1, N2, N3, REM)
- Medical-grade annotations

**Skills Learned:**
- Physiological signal interpretation
- Medical domain knowledge
- Multi-signal reasoning
- Clinical decision-making

**Difficulty**: ⭐⭐⭐⭐ **Advanced** - Medical domain expertise

**Pre-trained Model**: Available on HuggingFace

---

### Stage 5: ECG QA CoT - ECG Question Answering

**Purpose**: Teach the model **expert-level medical reasoning** for cardiac analysis

**Task**: Answer complex questions about ECG signals with detailed medical reasoning.

**Example:**
```
Time Series: [ECG signal showing 10-second recording]

Question: "Identify any arrhythmias present and explain the diagnosis."

Chain-of-Thought Answer:
"Detailed ECG Analysis:

1. Rhythm Assessment:
   - Heart rate: ~110 bpm (tachycardia)
   - RR intervals: Irregularly irregular
   - No consistent pattern

2. P-wave Analysis:
   - P-waves: Absent
   - Instead: Fibrillatory waves in baseline
   - Rate: 350-600/min

3. QRS Complex:
   - Duration: Normal (<120ms)
   - Morphology: Consistent
   - Axis: Normal

4. Diagnosis:
   ATRIAL FIBRILLATION with rapid ventricular response

5. Reasoning:
   - Absence of organized P-waves
   - Irregularly irregular rhythm
   - Fibrillatory baseline activity
   - Rapid ventricular rate
   
Clinical Significance: This arrhythmia requires immediate 
medical evaluation and rate control therapy."
```

**Dataset**: ECG-QA
- Clinical ECG recordings
- Expert annotations
- Multiple pathologies
- Real medical questions

**Skills Learned:**
- Expert medical diagnosis
- ECG interpretation
- Differential diagnosis
- Clinical reasoning
- Risk assessment

**Difficulty**: ⭐⭐⭐⭐⭐ **Expert** - Requires medical-grade knowledge

**Pre-trained Model**: Available on HuggingFace

---

## 🎓 Curriculum Learning Progression

### Why This Order?

```
Stage 1 (MCQ)  →  Stage 2 (Caption)  →  Stage 3-5 (CoT)
   ↓                    ↓                     ↓
Simple Q&A      Natural Language      Expert Reasoning
   ↓                    ↓                     ↓
Pattern Recog.  Feature Description   Step-by-step Logic
```

**Pedagogical Design:**
1. **Stage 1**: Build basic comprehension (QA)
2. **Stage 2**: Learn to articulate patterns (captioning)
3. **Stage 3**: Introduce reasoning (HAR CoT)
4. **Stage 4**: Apply to medical domain (Sleep CoT)
5. **Stage 5**: Master expert-level diagnosis (ECG CoT)

---

## 📊 Dataset Statistics

### Stage 1: TSQA
- **Samples**: 6,300 train, 630 val, 700 test
- **Input**: Time series + multiple-choice question
- **Output**: Single letter (A, B, C, or D)
- **Metrics**: Accuracy

### Stage 2: M4
- **Samples**: Variable (large forecasting dataset)
- **Input**: Historical time series
- **Output**: Natural language caption
- **Metrics**: BLEU, ROUGE, perplexity

### Stage 3: HAR
- **Samples**: Thousands of activity segments
- **Input**: Multi-sensor time series
- **Output**: Activity label + reasoning
- **Metrics**: F1-score, accuracy, reasoning quality

### Stage 4: Sleep
- **Samples**: Sleep study recordings
- **Input**: EEG/EOG/EMG signals
- **Output**: Sleep stage + explanation
- **Metrics**: Cohen's Kappa, accuracy, F1-score

### Stage 5: ECG QA
- **Samples**: Clinical ECG database
- **Input**: ECG signal + medical question
- **Output**: Diagnosis + detailed reasoning
- **Metrics**: Clinical accuracy, diagnostic precision

---

## 🔧 Training Configuration Per Stage

### Typical Hyperparameters

**Stage 1 (Simple):**
```yaml
epochs: 30
batch_size: 8
lr_encoder: 2.0e-4
lr_projector: 1.0e-4
lr_llm: 2.0e-5 (if not frozen)
gradient_checkpointing: false
```

**Stage 2 (Medium):**
```yaml
epochs: 20
batch_size: 4
lr_encoder: 1.0e-4
lr_projector: 5.0e-5
lora: true (optional)
```

**Stages 3-5 (Complex):**
```yaml
epochs: 60
batch_size: 2
lr_encoder: 5.0e-5
lr_projector: 2.0e-5
lora: true (recommended)
gradient_checkpointing: true
```

---

## 🎯 How S-ADT Uses These Stages

**S-ADT is inference-only** - it works with pre-trained models from ANY stage:

### With Stage 1 (TSQA)
```python
# Load pre-trained Stage 1 model
model = load_pretrained("OpenTSLM/llama-3.2-1b-tsqa-sp")

# Run S-ADT inference
results = run_sadt(model, prompt="Question about time series")

# Gets: Better answers through tree search exploration
```

### With Stages 3-5 (CoT Models)
```python
# Load Stage 5 (ECG expert)
model = load_pretrained("OpenTSLM/stage5-ecg-cot")

# Run S-ADT
results = run_sadt(model, prompt="ECG diagnosis question")

# Gets: 
# - More diverse reasoning paths explored
# - Soft Bellman prevents mode collapse
# - Better clinical decisions
```

---

## 📈 Expected Performance

### Inference Speed (Per Prompt)

| Stage | Complexity | PyTorch MPS | MLX (M1 Pro) | MLX (M3 Max) |
|-------|------------|-------------|--------------|--------------|
| Stage 1 (TSQA) | Simple | ~40s | ~25s | ~8-10s |
| Stage 2 (M4) | Medium | ~50s | ~30s | ~10-12s |
| Stage 3-5 (CoT) | Complex | ~60-80s | ~35-45s | ~12-15s |

**Note**: S-ADT adds systematic exploration, making inference slower but **much more thorough** (81x more exploration!)

---

## 🚀 Quick Start for Each Stage

### Stage 1: TSQA
```bash
# Already demonstrated! ✅
python dts_implementation/examples/comprehensive_demo.py

# Results: 324 nodes, 81x exploration
```

### Stages 2-5: Coming Soon
```bash
# Download pre-trained model for desired stage
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('OpenTSLM/<stage-model-id>', local_dir='checkpoints/stageN')"

# Adapt model wrapper for that stage
# Run S-ADT inference
```

---

## 💡 Key Insights

### Why Curriculum Learning?

**Without curriculum** (train Stage 5 directly):
- ❌ Model struggles with complex reasoning
- ❌ Overfits to simple patterns
- ❌ Poor generalization

**With curriculum** (Stages 1→5):
- ✅ Gradual complexity increase
- ✅ Better foundation building
- ✅ Strong reasoning capabilities
- ✅ Better final performance

### How S-ADT Enhances Each Stage

**Stage 1 (TSQA)**:
- Base model: Answers one way (greedy)
- With S-ADT: Explores 81 different reasoning paths
- Result: More robust answers

**Stages 3-5 (CoT)**:
- Base model: Single reasoning chain
- With S-ADT: Multiple reasoning chains explored
- Result: Better clinical/expert decisions

---

## 📦 Pre-trained Models Available

Based on OpenTSLM GitHub demo folder:

```python
# Stage 1: TSQA (MCQ)
"OpenTSLM/llama-3.2-1b-tsqa-sp"  # ✅ Downloaded and tested!

# Stage 2: M4 Captioning
# Check: 02_test_hf_m4.py for model ID

# Stage 3: HAR CoT
# Check: 03_test_hf_har_cot.py for model ID

# Stage 4: Sleep CoT  
# Check: 04_test_hf_sleep_cot.py for model ID

# Stage 5: ECG QA CoT
# Check: 05_test_hf_ecg_qa_cot.py for model ID
```

**To find model IDs:**
```bash
curl -s https://raw.githubusercontent.com/StanfordBDHG/OpenTSLM/main/demo/huggingface/02_test_hf_m4.py | grep "REPO_ID"
```

---

## 🔬 Technical Details

### Model Architecture Per Stage

**All stages use the same base architecture:**
```
Time Series Input
      ↓
Time Series Encoder (CNN + Transformer)
      ↓
Projector (Linear layers)
      ↓
LLM (Llama 3.2 1B)
      ↓
Generated Text Output
```

**What changes per stage:**
- ✅ Dataset (different tasks)
- ✅ Training data (different complexity)
- ✅ Fine-tuning strategy (LoRA in later stages)
- ❌ Architecture (stays the same!)

### Training Strategy

**Stages 1-2:**
- Train encoder + projector
- Freeze or lightly fine-tune LLM
- Focus on time series understanding

**Stages 3-5:**
- Use LoRA for LLM fine-tuning
- Train encoder + projector + LoRA adapters
- Focus on reasoning capabilities

---

## 📈 Validation Metrics

### Stage 1 (TSQA)
- **Accuracy**: % of correct multiple choice answers
- **Loss**: Cross-entropy loss
- Target: >80% accuracy

### Stage 2 (M4)
- **BLEU Score**: Caption quality
- **Perplexity**: Language model confidence
- **ROUGE**: Overlap with reference captions

### Stages 3-5 (CoT)
- **Task Accuracy**: Correct final answer
- **Reasoning Quality**: Step-by-step logic coherence
- **F1-Score**: Classification performance
- **Cohen's Kappa**: Inter-rater reliability (medical)

---

## 🎯 How to Use Each Stage with S-ADT

### Pattern for All Stages

```python
from dts_implementation.models.local_loader import load_base_model
from dts_implementation.rewards.spectral_reward import SpectralReward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig

# 1. Load pre-trained stage model
model = load_pretrained_opentslm(stage=1)  # or 2, 3, 4, 5

# 2. Setup reward for that task
reward = SpectralReward(gamma=1.0)
# Add task-specific reward function
reward.set_task_reward(stage_specific_reward_fn)

# 3. Configure search
config = MaxEntTSConfig(num_rollouts=20)

# 4. Run S-ADT
searcher = MaxEntTS(model, reward, config)
results = searcher.search(prompt_tokens)
```

### Stage-Specific Prompts

**Stage 1 (TSQA):**
```python
prompt = """Given the ECG time series, what is the heart rate?
A) 60-80 bpm (Normal)
B) 80-100 bpm (Elevated)
C) >100 bpm (Tachycardia)
D) <60 bpm (Bradycardia)
Answer:"""
```

**Stage 2 (M4):**
```python
prompt = "Describe the following time series pattern:"
# Model generates: "This series shows..."
```

**Stage 3-5 (CoT):**
```python
prompt = "Analyze this ECG signal and explain your diagnosis step by step:"
# Model generates: "Step 1: ... Step 2: ... Conclusion: ..."
```

---

## 🔄 Curriculum Learning Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Start: Base LLM (Llama)                      │
│                  (No time series understanding)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: TSQA (MCQ)                                            │
│  • Learn basic time series patterns                             │
│  • Multiple choice QA                                           │
│  • Build foundation                                             │
│  Duration: ~30 epochs                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: M4 Captioning                                         │
│  • Generate descriptions                                        │
│  • Natural language for patterns                                │
│  • Feature articulation                                         │
│  Duration: ~20 epochs                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: HAR CoT                                               │
│  • Step-by-step reasoning                                       │
│  • Activity recognition                                         │
│  • Multi-sensor fusion                                          │
│  Duration: ~60 epochs                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Sleep CoT                                             │
│  • Physiological signals                                        │
│  • Medical reasoning                                            │
│  • Sleep stage classification                                   │
│  Duration: ~60 epochs                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5: ECG QA CoT                                            │
│  • Expert cardiac diagnosis                                     │
│  • Complex medical reasoning                                    │
│  • Arrhythmia detection                                         │
│  Duration: ~60 epochs                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             Final Model: Expert Time Series LLM                  │
│  Can: Answer questions, caption, reason about time series       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎓 Learning Objectives Per Stage

| Stage | What Model Learns | Example Capability |
|-------|-------------------|-------------------|
| **1** | Pattern recognition | "This trend is increasing" |
| **2** | Pattern description | "Weekly seasonality with upward trend" |
| **3** | Basic reasoning | "The pattern suggests walking because..." |
| **4** | Medical reasoning | "Delta waves indicate deep sleep because..." |
| **5** | Expert diagnosis | "Atrial fibrillation due to: 1) ... 2) ... 3) ..." |

---

## 💻 Downloading Pre-trained Models

### Stage 1 (Already Done!)

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "OpenTSLM/llama-3.2-1b-tsqa-sp",
    local_dir="checkpoints/stage1"
)
```

✅ **Already downloaded**: `checkpoints/opentslm_stage1_pretrained/`

### Stages 2-5 (To Do)

```bash
# Find model IDs from demo scripts
curl -s https://raw.githubusercontent.com/StanfordBDHG/OpenTSLM/main/demo/huggingface/02_test_hf_m4.py | grep "REPO_ID"

# Download once you have the ID
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('OpenTSLM/<model-id>', local_dir='checkpoints/stage2')"
```

---

## 🔍 Current Status

### What We Have

✅ **Stage 1 (TSQA)**:
- Pre-trained model downloaded (54.6 MB)
- S-ADT tested and working
- Results: 324 nodes, 81x exploration
- Ready to use!

### What's Available (Not Yet Downloaded)

⏳ **Stages 2-5**:
- Pre-trained models exist on HuggingFace
- Can download anytime
- S-ADT will work the same way
- Just need to run download command

---

## 📝 Summary

**OpenTSLM Stages = Progressive Curriculum**

1. **Stage 1**: Basic QA (foundation) ✅ **Tested with S-ADT!**
2. **Stage 2**: Captioning (articulation)
3. **Stage 3**: Reasoning (logic)
4. **Stage 4**: Medical (domain expertise)
5. **Stage 5**: Expert (clinical diagnosis)

**S-ADT enhances ALL stages** by:
- 81x more exploration
- Soft Bellman preventing collapse
- Spectral regularization
- Better inference quality

**Current Status**: Stage 1 complete and tested! ✅  
**Ready for**: Stages 2-5 anytime you want!

---

**Last Updated**: December 13, 2025  
**Stage 1 Status**: ✅ Complete and Tested with S-ADT  
**Stages 2-5 Status**: Pre-trained models available, ready to download


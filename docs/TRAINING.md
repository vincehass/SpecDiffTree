# Training Guide - SpecDiffTree

This guide covers training SpecDiffTree models using the OpenTSLM foundation with S-ADT extensions.

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv specdifftree_env
source specdifftree_env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install wandb

# Authenticate
huggingface-cli login
wandb login
```

### 2. Run Quick Test

```bash
# Test with Apple Silicon GPU (MPS)
python scripts/training/train_with_wandb.py \
  --config configs/mps/quick_test.yaml

# Or with CUDA
python scripts/training/train_with_wandb.py \
  --config configs/cuda/standard.yaml
```

---

## 📊 Training Configurations

### Apple Silicon (MPS)

Optimized for Apple Silicon GPU acceleration:

| Config | Duration | Purpose | Batch Size |
|--------|----------|---------|------------|
| `mps/quick_test.yaml` | 2-4h | Development/testing | 4 |
| `mps/full_training.yaml` | 12-18h | Production | 8 |
| `mps/conservative.yaml` | 16-24h | If NaN issues | 2 |

**MPS Environment Setup**:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### NVIDIA GPU (CUDA)

| Config | Duration | Batch Size |
|--------|----------|------------|
| `cuda/standard.yaml` | 8-12h | 16 |

### CPU Fallback

| Config | Duration | Batch Size |
|--------|----------|------------|
| `cpu/standard.yaml` | 40-60h | 2 |

---

## 🎯 Training Scripts

### Single Experiment

```bash
./scripts/training/train_single.sh --config configs/mps/full_training.yaml
```

### With Weights & Biases

```bash
python scripts/training/train_with_wandb.py \
  --config configs/mps/full_training.yaml \
  --wandb_entity your_username \
  --wandb_project specdifftree
```

### Ablation Study

```bash
./scripts/training/run_ablation.sh \
  --config configs/ablation.yaml
```

---

## 🔧 Hardware-Specific Tips

### Apple Silicon

**Best Performance**:
- Use MPS configs (batch_size 8)
- Set environment variables for stability
- Close other apps to free GPU memory
- Plug into power (don't run on battery)

**If NaN Loss**:
- Use `conservative.yaml` config
- Reduce batch size to 2
- Lower learning rates

### NVIDIA GPU

**Best Performance**:
- Use CUDA configs (batch_size 16)
- Enable mixed precision training
- Use distributed training for multi-GPU

### CPU

**Best Performance**:
- Use gradient checkpointing
- Reduce batch size to 2
- Consider overnight/weekend runs

---

## 📈 Expected Results

### Stage 1 (TSQA)

| Epochs | Test Accuracy | Training Time (MPS) |
|--------|---------------|---------------------|
| 2 | 30-50% | 2-4 hours |
| 30 | 60-75% | 12-18 hours |

### Performance by Device

| Device | Speed | Batch Size | Time (30 epochs) |
|--------|-------|------------|------------------|
| Apple Silicon | 5-10x CPU | 8 | 12-18h |
| NVIDIA GPU | 8-12x CPU | 16 | 8-12h |
| CPU | 1x (baseline) | 2 | 40-60h |

---

## 🔍 Monitoring

### Real-time Logs

```bash
tail -f training.log
```

### Weights & Biases Dashboard

```bash
# View at: https://wandb.ai/your_username/specdifftree
```

### Check GPU Usage

**Apple Silicon**:
- Activity Monitor → Window → GPU History

**NVIDIA**:
```bash
watch -n 1 nvidia-smi
```

---

## 🐛 Troubleshooting

### NaN Loss

**Cause**: Numerical instability

**Solutions**:
1. Use conservative config
2. Reduce learning rate
3. Increase gradient clipping
4. Use smaller batch size

```yaml
training:
  batch_size: 2
  lr_encoder: 1.0e-4
  lr_projector: 5.0e-5
  grad_clip_norm: 0.5
```

### Out of Memory

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use smaller model
4. Close other applications

```yaml
training:
  batch_size: 2  # or 4
model:
  gradient_checkpointing: true
```

### Slow Training

**Check**:
1. GPU is being used (check Activity Monitor/nvidia-smi)
2. Batch size is appropriate
3. No other processes using GPU
4. MPS fallback enabled (Apple Silicon)

---

## 📦 Curriculum Learning

Follow OpenTSLM's 5-stage curriculum:

```bash
# Stage 1: Multiple Choice QA
python curriculum_learning.py --stage stage1_mcq --epochs 30

# Stage 2: Captioning  
python curriculum_learning.py --stage stage2_captioning --epochs 20

# Stages 3-5: Chain of Thought
python curriculum_learning.py --stage stage3_cot --epochs 60
python curriculum_learning.py --stage stage4_sleep_cot --epochs 60
python curriculum_learning.py --stage stage5_ecg_cot --epochs 60
```

---

## 🎓 Best Practices

1. **Start with quick test** (2 epochs) to verify setup
2. **Monitor early** - check first few iterations for NaN
3. **Use W&B** - essential for experiment tracking
4. **Conservative first** - use conservative config on new hardware
5. **Checkpoint frequently** - save every N epochs
6. **Overnight runs** - ideal for 12-18 hour trainings

---

## 📝 Configuration Format

```yaml
experiment:
  name: "experiment_name"
  tags: ["tag1", "tag2"]
  notes: "Description"

dataset:
  name: "TSQA"
  num_samples_train: 6300

model:
  type: "OpenTSLMSP"
  llm_id: "meta-llama/Llama-3.2-1B"
  gradient_checkpointing: false

training:
  num_epochs: 30
  batch_size: 8
  lr_encoder: 2.0e-4
  lr_projector: 1.0e-4

device:
  type: "mps"  # or "cuda", "cpu"
```

---

## 🚀 Next Steps

After training:
1. Evaluate on test set
2. Run S-ADT alignment for inference
3. Compare with baseline OpenTSLM
4. Tune hyperparameters via ablation

---

For S-ADT inference alignment, see [S-ADT.md](../S-ADT.md).


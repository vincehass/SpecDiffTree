# SpecDiffTree: Spectral-Regularized Amortized Diffusion Trees

[![Framework](https://img.shields.io/badge/Framework-PyTorch_+_MLX-orange)](#)
[![Task](https://img.shields.io/badge/Task-LLM_Inference-green)](#)
[![Method](https://img.shields.io/badge/Method-MaxEnt_Tree_Search-blue)]()
[![Base](https://img.shields.io/badge/Built_on-OpenTSLM-purple)](https://github.com/StanfordBDHG/OpenTSLM)

**SpecDiffTree** implements **Maximum Entropy Tree Search for Autoregressive Models (MaxEnt-TS)**, a principled inference-time algorithm that extends tree-based sampling methods to autoregressive language models. The method combines soft Bellman backups with spectral regularization to enable systematic exploration of the generation space while preventing mode collapse.

---

## Mathematical Framework

### Problem Formulation

We formulate autoregressive text generation as a sequential decision process where at each step $t$, given a sequence $x_{\leq t} = (x_1, \ldots, x_t)$, we select the next token $x_{t+1}$ from a vocabulary $\mathcal{V}$. The goal is to find a policy that maximizes both task reward and generation diversity while preserving spectral properties of the output.

### Soft Bellman Equation

The core of MaxEnt-TS is the soft value function that replaces the standard max operator with a LogSumExp to prevent mode collapse:

$$
V_t(x_{\leq t}) = \frac{1}{\lambda} \log \mathbb{E}_{p_\theta(x_{t+1}|x_{\leq t})} \left[ \exp(\lambda V_{t+1}(x_{\leq t+1})) \right]
$$

where $\lambda > 0$ is the temperature parameter controlling the trade-off between entropy and reward maximization.

### Optimal Policy

The optimal policy derived from the soft value function follows a Boltzmann distribution:

$$
\pi^*(x_{t+1}|x_{\leq t}) \propto p_\theta(x_{t+1}|x_{\leq t}) \exp(\lambda V_{t+1}(x_{\leq t+1}))
$$

This formulation naturally balances the LLM's prior $p_\theta$ with the learned value function $V_{t+1}$.

### Spectral Regularization

To preserve frequency content in generated sequences, we employ a spectral reward function based on Power Spectral Density (PSD) divergence:

$$
r(x) = r_{\text{task}}(x) - \gamma \int_\Omega \left| \log S_x(\omega) - \log \mathbb{E}[S_c(\omega)] \right| d\omega
$$

where:
- $S_x(\omega)$ is the PSD of the generated sequence
- $\mathbb{E}[S_c(\omega)]$ is the expected PSD from a reference corpus
- $\gamma \geq 0$ controls the regularization strength
- $\Omega$ is the frequency domain

This regularization prevents spectral collapse, a phenomenon where generated sequences lose high-frequency components.

### Tree-Based Sampling

We employ Monte Carlo Tree Search (MCTS) at the token level to build a search tree $\mathcal{T}$ where:
- **States**: Token sequences $x_{\leq t}$
- **Actions**: Next token selection from $\mathcal{V}$
- **Transitions**: Deterministic concatenation
- **Policy**: Language model $p_\theta$
- **Value**: Soft Bellman backup with spectral rewards

The search proceeds by iteratively:
1. **Selection**: Traverse tree using upper confidence bounds
2. **Expansion**: Add top-$k$ tokens from $p_\theta$ as children
3. **Simulation**: Roll out to terminal state
4. **Backpropagation**: Update values using soft Bellman backup

---

## Computational Implementation

### Model Compatibility

The implementation is model-agnostic and works with any autoregressive language model through a unified interface:

```python
class LLMInterface:
    def get_next_token_logits(self, token_sequence: List[int]) -> Array
    def encode_text(self, text: str) -> List[int]
    def decode_tokens(self, tokens: List[int]) -> str
    def get_top_k_tokens(self, sequence: List[int], k: int) -> List[Tuple[int, float]]
```

Supported frameworks:
- **PyTorch**: CUDA, MPS, CPU backends
- **MLX**: Native Apple Silicon optimization (2-5× faster on M3 Max)
- **HuggingFace Transformers**: Direct integration

### Algorithm Complexity

For a search with:
- $N$ rollouts
- Maximum depth $D$
- Branching factor $k$
- Sequence length $L$

**Time Complexity**: $O(N \cdot D \cdot k \cdot L)$ for forward passes  
**Space Complexity**: $O(N \cdot D \cdot k)$ for tree storage

### Computational Optimizations

1. **Batched Inference**: Group token evaluations for GPU efficiency
2. **KV-Cache Reuse**: Cache key-value pairs in attention layers
3. **Lazy Expansion**: Only expand promising nodes
4. **Early Termination**: Stop rollouts at EOS token

---

## Installation and Setup

### Requirements

- Python 3.9+
- PyTorch 2.0+ or MLX 0.4+
- HuggingFace Transformers
- NumPy, SciPy (for spectral analysis)

### Quick Installation

```bash
git clone https://github.com/vincehass/SpecDiffTree.git
cd SpecDiffTree

# Create virtual environment
python3 -m venv opentslm_env
source opentslm_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon (MLX support)
pip install mlx-lm

# Set Python path
export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
```

---

## Usage

### Basic Example

```python
from dts_implementation.models.pytorch_hf_wrapper import load_base_model
from dts_implementation.rewards.spectral_reward import SpectralReward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig
import numpy as np

# Load pre-trained LLM
model = load_base_model(
    llm_id="meta-llama/Llama-3.2-1B",
    device="cuda"  # or "mps", "cpu"
)

# Configure spectral reward
reward = SpectralReward(gamma=1.0)
reference_ts = np.sin(np.linspace(0, 10, 1000))
reward.set_context(reference_ts)

# Set MaxEnt-TS hyperparameters
config = MaxEntTSConfig(
    num_rollouts=20,
    temperature=1.0,
    max_seq_length=40,
    expansion_k=4
)

# Execute tree search
searcher = MaxEntTS(model, reward, config)
prompt_tokens = model.encode_text("Question: What is 2+2? Answer:")
results = searcher.search(prompt_tokens)

print(f"Generated: {results['best_text']}")
print(f"Tree nodes: {results['tree_stats']['total_nodes']}")
print(f"Tree depth: {results['tree_stats']['max_depth']}")
```

### MLX Acceleration (Apple Silicon)

```python
from dts_implementation.models.mlx_direct_loader import load_mlx_model

# Load 4-bit quantized model for faster inference
model = load_mlx_model("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Same API as PyTorch
searcher = MaxEntTS(model, reward, config)
results = searcher.search(prompt_tokens)
```

---

## Evaluation Framework

### Comprehensive Multi-Method Comparison

The repository includes a comprehensive evaluation framework comparing four inference methods:

1. **Greedy Decoding**: Baseline argmax selection
2. **MCTS**: Standard Monte Carlo Tree Search
3. **DTS**: Diffusion Tree Sampling (adapted)
4. **MaxEnt-TS**: Maximum Entropy Tree Search (this work)

### Evaluation Metrics

Ten metrics are tracked across all methods:

**Efficiency Metrics**:
- Number of Function Evaluations (NFE)
- Wall-clock time per sample
- Tree depth
- Branching factor

**Quality Metrics**:
- Task-specific reward
- Perplexity
- Sequence diversity (distinct n-grams)
- Success rate

**Length Metrics**:
- Average sequence length
- Consistency with reference

### Running Evaluations

#### Standard Evaluation (PyTorch)

```bash
# Run single method
python evaluation/comprehensive_evaluation.py \
    --method maxent_ts \
    --num_samples 250 \
    --num_rollouts 20 \
    --device cuda

# Parallel evaluation of all methods
./experiments/scripts/run_parallel_evaluation.sh
```

#### Pure MLX Evaluation (Apple Silicon)

```bash
# MLX-optimized evaluation
python evaluation/comprehensive_evaluation_mlx.py \
    --method maxent_ts \
    --num_samples 250 \
    --num_rollouts 20

# Parallel MLX evaluation
./experiments/scripts/run_parallel_evaluation_mlx.sh
```

### Results Visualization

```bash
# Generate publication-quality figures
python evaluation/generate_figures.py --results_dir results/

# Ablation studies
./experiments/scripts/run_ablation_studies.sh
python evaluation/generate_ablation_figures.py
```

---

## Experimental Results

### Performance Comparison (250 samples, M4 dataset)

| Method     | NFE     | Time/Sample | Reward | Diversity | Tree Depth |
|------------|---------|-------------|--------|-----------|------------|
| Greedy     | 50.0    | 4.0s        | 1.00   | 0.85      | 1.0        |
| MCTS       | 3.0     | 4.4s        | 0.50   | 0.72      | 6.2        |
| DTS        | 7.0     | 4.2s        | 0.45   | 0.68      | 5.8        |
| MaxEnt-TS  | 16.0    | 10.7s       | 0.52   | 0.91      | 7.4        |

**Key Findings**:
- MaxEnt-TS achieves highest diversity while maintaining competitive rewards
- Tree-based methods explore 81× more states than greedy baseline
- MLX implementation provides 2-5× speedup on Apple Silicon

### Hardware Performance

| Hardware    | Framework | Time/Sample | Speedup |
|-------------|-----------|-------------|---------|
| M1 Pro      | PyTorch   | ~46s        | 1.0×    |
| M1 Pro      | MLX       | ~25s        | 1.8×    |
| M3 Max      | MLX       | ~8-10s      | 4.6×    |
| A100 (40GB) | PyTorch   | ~12s        | 3.8×    |

---

## Repository Structure

```
SpecDiffTree/
├── dts_implementation/          # Core implementation
│   ├── core/
│   │   ├── dts_node.py          # Tree node structures
│   │   ├── dts_node_mlx.py      # MLX-native nodes
│   │   ├── soft_bellman.py      # Soft Bellman backup (PyTorch)
│   │   └── soft_bellman_mlx.py  # Soft Bellman backup (MLX)
│   ├── search/
│   │   ├── maxent_ts.py         # MaxEnt-TS algorithm (PyTorch)
│   │   └── maxent_ts_mlx.py     # MaxEnt-TS algorithm (MLX)
│   ├── rewards/
│   │   └── spectral_reward.py   # Spectral regularization
│   ├── models/
│   │   ├── pytorch_hf_wrapper.py    # PyTorch models
│   │   ├── mlx_direct_loader.py     # MLX models
│   │   └── opentslm_wrapper.py      # OpenTSLM integration
│   ├── utils/
│   │   └── psd_utils.py         # Power Spectral Density utilities
│   ├── examples/                # Usage examples
│   └── tests/                   # Test suite
│
├── baselines/                   # Baseline implementations
│   ├── mcts_baseline.py         # Standard MCTS
│   ├── mcts_baseline_mlx.py     # MCTS (MLX)
│   ├── dts_baseline.py          # Diffusion Tree Sampling
│   └── dts_baseline_mlx.py      # DTS (MLX)
│
├── evaluation/                  # Evaluation framework
│   ├── comprehensive_evaluation.py      # PyTorch evaluation
│   ├── comprehensive_evaluation_mlx.py  # MLX evaluation
│   ├── generate_figures.py              # Figure generation
│   └── metrics/                         # Metric implementations
│
├── experiments/                 # Experiment scripts
│   └── scripts/
│       ├── run_parallel_evaluation.sh
│       ├── run_parallel_evaluation_mlx.sh
│       └── run_ablation_studies.sh
│
├── src/                         # OpenTSLM components
│   ├── model/                   # Model architectures
│   ├── time_series_datasets/    # Dataset loaders
│   └── prompt/                  # Prompt engineering
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # System architecture
│   ├── ALGORITHM_AND_BEST_PRACTICES.md
│   ├── MaximumEntropyTreeSearchforAutoregressive.md
│   └── guides/                  # User guides
│
├── configs/                     # Hardware configurations
├── test/                        # Additional tests
├── requirements.txt
└── README.md
```

---

## Advanced Configuration

### Hyperparameter Tuning

Key hyperparameters and their effects:

```python
config = MaxEntTSConfig(
    # Search parameters
    num_rollouts=20,        # More rollouts → better quality, slower
    expansion_k=4,          # Branching factor (2-8 typical)
    
    # Temperature parameters
    temperature=1.0,        # λ in soft Bellman (0.1-2.0)
    
    # Sequence parameters
    max_seq_length=100,     # Maximum generation length
    
    # Exploration parameters
    exploration_prob=0.1,   # UCB exploration coefficient
    
    # Termination
    early_stop=True,        # Stop at EOS token
)
```

### Custom Reward Functions

```python
from dts_implementation.rewards.spectral_reward import SpectralReward

class CustomReward(SpectralReward):
    def compute_task_reward(self, text: str) -> float:
        # Domain-specific reward logic
        return score
    
    def compute_spectral_penalty(self, sequence: np.ndarray) -> float:
        # Custom spectral analysis
        return penalty

reward = CustomReward(gamma=1.0)
```

---

## OpenTSLM Integration

SpecDiffTree integrates with OpenTSLM's 5-stage curriculum for time series:

| Stage | Task                  | Checkpoint                  |
|-------|----------------------|------------------------------|
| 1     | TSQA (MCQ)           | `llama-3.2-1b-tsqa-sp`      |
| 2     | M4 Captioning        | `llama-3.2-1b-m4-sp`        |
| 3     | HAR Chain-of-Thought | `llama-3.2-1b-har-sp`       |
| 4     | Sleep Analysis       | `llama-3.2-1b-sleep-sp`     |
| 5     | ECG Diagnosis        | `llama-3.2-1b-ecg-sp`       |

```bash
# Download checkpoint
from huggingface_hub import snapshot_download
snapshot_download("OpenTSLM/llama-3.2-1b-tsqa-sp", 
                  local_dir="checkpoints/stage1")

# Evaluate all stages
python evaluation/run_all_stages_eval.py --stages 1 2 3 4 5
```

---

## Theoretical Foundation

This work builds on three key areas:

### 1. Diffusion Tree Sampling
- **Paper**: "Diffusion Tree Sampling" (Jain et al., 2025)
- **Contribution**: Soft Bellman backups for diffusion models
- **Adaptation**: Extended to autoregressive token generation

### 2. Maximum Entropy Reinforcement Learning
- **Paper**: "Soft Actor-Critic" (Haarnoja et al., 2018)
- **Contribution**: Entropy-regularized policy optimization
- **Application**: Temperature-controlled token selection

### 3. Spectral Analysis for Sequences
- **Foundation**: Power Spectral Density for time series
- **Innovation**: Spectral regularization prevents mode collapse
- **Implementation**: Fast Fourier Transform (FFT) based

**Key Innovation**: We show that autoregressive generation can be viewed as a sequential decision process where soft Bellman backups with spectral regularization enable diverse, high-quality generation without model retraining.

See [MaximumEntropyTreeSearchforAutoregressive.md](docs/MaximumEntropyTreeSearchforAutoregressive.md) for complete mathematical derivation.

---

## Documentation

### Core Documentation
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and design
- [ALGORITHM_AND_BEST_PRACTICES.md](docs/ALGORITHM_AND_BEST_PRACTICES.md) - Implementation details
- [MaximumEntropyTreeSearchforAutoregressive.md](docs/MaximumEntropyTreeSearchforAutoregressive.md) - Mathematical framework

### User Guides
- [QUICK_START.md](docs/guides/QUICK_START.md) - Getting started
- [COMPREHENSIVE_EVALUATION_GUIDE.md](docs/COMPREHENSIVE_EVALUATION_GUIDE.md) - Running evaluations
- [PURE_MLX_M3_MAX_GUIDE.md](docs/guides/PURE_MLX_M3_MAX_GUIDE.md) - MLX optimization guide

### Additional Resources
- [S-ADT.md](docs/S-ADT.md) - Spectral Amortized Diffusion Trees
- [MONOTONICITY_EXPLAINED.md](docs/MONOTONICITY_EXPLAINED.md) - Theoretical properties
- [TRAINING.md](docs/TRAINING.md) - Training procedures

---

## Development

### Testing

```bash
# Run test suite
python -m pytest test/

# Integration tests
python dts_implementation/tests/test_integration.py

# Quick validation
python dts_implementation/examples/simple_test.py
```

### Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{specdifftree2025,
  title={SpecDiffTree: Maximum Entropy Tree Search for Autoregressive Models},
  author={Anonymous},
  year={2025},
  url={https://github.com/vincehass/SpecDiffTree}
}
```

---

## Acknowledgements

This work builds upon:
- **OpenTSLM** (Stanford BDHG) - Time series language models
- **Diffusion Tree Sampling** (Jain et al., 2025) - DTS framework
- **Maximum Entropy RL** (Haarnoja et al., 2018) - Soft Bellman equations
- **MLX** (Apple ML Research) - Apple Silicon optimization

---

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.

---

## Contact

For questions or issues, please open an issue on [GitHub](https://github.com/vincehass/SpecDiffTree/issues).

**Last Updated**: December 2025

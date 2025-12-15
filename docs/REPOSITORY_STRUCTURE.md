# 📁 Repository Organization

## Clean Root Directory

The repository root now contains **only 5 essential files**:

```
SpecDiffTree/
├── README.md           # Main documentation
├── LICENSE.md          # MIT License
├── CITATION.cff        # Citation information
├── requirements.txt    # Python dependencies
└── __init__.py         # Python package marker
```

Everything else is organized into logical directories!

---

## Directory Structure

### Core Implementation
```
dts_implementation/     # MaxEnt-TS Core Algorithm
├── core/              # Tree structures & Soft Bellman
├── search/            # MaxEnt-TS search algorithm
├── rewards/           # Spectral reward functions
├── models/            # Model wrappers (PyTorch, MLX)
├── utils/             # Utilities (PSD, etc.)
├── examples/          # Example scripts
└── tests/             # Test suite
```

### Baseline Methods
```
baselines/             # Comparison Methods
├── mcts_baseline.py   # MCTS implementation
├── dts_baseline.py    # DTS implementation
└── __init__.py
```

### Evaluation Framework
```
evaluation/                      # Comprehensive Evaluation
├── comprehensive_evaluation.py  # Main evaluation script
├── compare_all_methods.py      # Method comparison
├── generate_ablation_figures.py # Figure generation
└── run_stages_*.py             # Stage-specific evaluations
```

### Experiments
```
experiments/           # Experiment Scripts & Logs
├── scripts/          # Bash scripts
│   ├── run_parallel_evaluation.sh  # Parallel evaluation
│   ├── run_ablation_studies.sh     # Ablation studies
│   └── *.py          # Utility scripts
└── logs/             # Execution logs
```

### Data & Datasets
```
src/                      # OpenTSLM Integration
├── model/               # Model architectures
├── time_series_datasets/# Dataset loaders
│   ├── m4/             # M4 forecasting
│   ├── har_cot/        # HAR activity recognition
│   └── simulation/     # Synthetic data
└── prompt/             # Prompt engineering

data/                   # Raw datasets (gitignored)
```

### Documentation
```
docs/                  # All Documentation
├── guides/           # User guides
│   ├── COMPREHENSIVE_EVALUATION_GUIDE.md
│   ├── PARALLEL_EVALUATION_GUIDE.md
│   └── *.md
├── status/           # Status reports & bug fixes
│   └── BUG*.md
├── plans/            # Session plans (gitignored)
│   └── *_PLAN.md
├── ARCHITECTURE.md   # System architecture
├── S-ADT.md          # Algorithm paper
├── CONTRIBUTORS.md   # Contributors
└── *.md             # Other documentation
```

### Configuration & Assets
```
configs/              # Configuration files
└── mlx/             # MLX-specific configs

assets/              # Images, figures, etc.
```

### Results & Outputs
```
results/             # Evaluation results (gitignored)
wandb/              # WandB logs (gitignored)
checkpoints/        # Model checkpoints (gitignored)
checkpoints_mlx/    # MLX checkpoints (gitignored)
```

---

## What's Ignored (.gitignore)

### Heavy Files (Not Committed)
- `opentslm_env/` - Virtual environment
- `checkpoints/` - Model weights
- `results/` - Evaluation outputs
- `wandb/` - WandB logs
- `*.log` - Log files
- `data/` - Raw datasets

### Session-Specific Docs (Not Committed)
- `docs/plans/*_PLAN.md` - Daily plans
- `docs/plans/TOMORROW_*.md` - Next session plans
- `docs/plans/TODAY_*.md` - Current session notes
- `docs/status/*_STATUS.md` - Status updates
- `docs/guides/QUICK_*.md` - Quick guides
- `docs/guides/RUN_*.md` - Run instructions

### Essential Docs (Committed)
- `README.md` - Always committed
- `LICENSE.md` - Always committed
- `CITATION.cff` - Always committed
- `docs/ARCHITECTURE.md` - Core documentation
- `docs/S-ADT.md` - Algorithm description
- `docs/CONTRIBUTORS.md` - Contributors list

---

## Benefits of This Organization

### ✅ Clean Root
- Only 5 files in root directory
- Easy to navigate
- Professional appearance
- Clear entry points

### ✅ Logical Grouping
- Implementation separate from evaluation
- Documentation organized by type
- Experiments isolated from core code
- Clear module boundaries

### ✅ Easy Discovery
- Related files together
- Predictable locations
- Self-documenting structure
- IDE-friendly

### ✅ Scalability
- Easy to add new methods (baselines/)
- Easy to add new evaluations (evaluation/)
- Easy to add new docs (docs/)
- Clean git history

---

## Quick Navigation

### Running Experiments
```bash
# Go to experiments
cd experiments/scripts/

# Run parallel evaluation
./run_parallel_evaluation.sh

# Run ablation studies
./run_ablation_studies.sh
```

### Viewing Results
```bash
# Check latest results
ls results/parallel_*/

# View figures
open results/parallel_*/figures/

# Read evaluation docs
cat docs/guides/COMPREHENSIVE_EVALUATION_GUIDE.md
```

### Development
```bash
# Core algorithm
cd dts_implementation/

# Add new baseline
cd baselines/

# Run tests
cd dts_implementation/tests/
python test_integration.py
```

### Documentation
```bash
# Read guides
ls docs/guides/

# Check status reports
ls docs/status/

# View architecture
cat docs/ARCHITECTURE.md
```

---

## Migration from Old Structure

### Before (Messy)
```
SpecDiffTree/
├── 50+ markdown files in root 😱
├── Test scripts everywhere
├── Logs scattered
├── No clear organization
└── Hard to find anything
```

### After (Clean)
```
SpecDiffTree/
├── 5 essential files only ✨
├── docs/ - All documentation
├── evaluation/ - All evaluation code
├── experiments/ - All scripts & logs
├── Clear, logical structure
└── Easy to navigate
```

---

## Maintenance

### Adding New Files

**Documentation:**
- Guides → `docs/guides/`
- Status → `docs/status/`
- Plans → `docs/plans/` (gitignored)

**Code:**
- Evaluation scripts → `evaluation/`
- Experiment scripts → `experiments/scripts/`
- Tests → appropriate `tests/` subdirectory

**Results:**
- Evaluation results → `results/` (gitignored)
- Logs → `experiments/logs/` (gitignored)
- Figures → `results/*/figures/` (gitignored)

### Cleaning Up

```bash
# Remove old session files
rm docs/plans/*_PLAN.md
rm docs/status/*_STATUS.md

# Clean old results
rm -rf results/parallel_20*

# Clean logs
rm experiments/logs/*.log
```

---

## For Contributors

When adding new code:

1. **Core algorithm changes** → `dts_implementation/`
2. **New baseline methods** → `baselines/`
3. **New evaluation code** → `evaluation/`
4. **New experiments** → `experiments/scripts/`
5. **Documentation** → `docs/`
6. **Tests** → appropriate `tests/` directory

Always keep root directory clean!

---

**Last Updated:** Dec 15, 2025  
**Structure Version:** 2.0 (Clean & Organized)


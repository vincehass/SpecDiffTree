# Schematic Creation Summary

## Overview

Created a publication-quality schematic diagram for the SpecDiffTree/MaxEnt-TS methodology, adapted from the original time-series captioning approach.

## Files Created

### 1. Main Schematic

- **File**: `assets/specdifftree_schematic.png`
- **Size**: 485 KB
- **Resolution**: 300 DPI (publication quality)
- **Dimensions**: 16" × 10"

### 2. Generation Script

- **File**: `assets/generate_schematic.py`
- **Purpose**: Generates the schematic using matplotlib
- **Customizable**: Colors, text, layout can be easily modified

### 3. Assets Documentation

- **File**: `assets/README.md`
- **Purpose**: Documents all visual assets in the repository

## Schematic Components

### Top Section

- **Title**: "MaxEnt Tree Search for Autoregressive Models"
- **Capabilities Box**: Highlights key features
- **Example Prompt**: Medical diagnosis use case

### Left Section: Search Tree Exploration

- Visual tree structure showing branching
- Root node → Level 1 nodes (A, B, C) → Level 2 nodes (1, 2, 3)
- **Highlight**: "81× more exploration vs. greedy decoding"

### Center Flow

1. **Text Encoder** (light peach)
   - Tokenizer + Embedder
2. **Pre-trained LLM** (light purple)
   - Llama 3.2, OpenTSLM, etc.
   - No retraining required!
3. **MaxEnt-TS Search** (light pink/red)
   - Main component with 4 sub-components

### MaxEnt-TS Components (4 Sequential Steps)

1. **Token-Level MCTS** - Systematic exploration
2. **Soft Bellman Backup** - LogSumExp aggregation
3. **Spectral Rewards** - Frequency preservation
4. **Boltzmann Policy** - Temperature-controlled sampling

### Right Section: Key Formulations

- **Soft Bellman Equation**:
  ```
  V_t(x_≤t) = (1/λ) log E[exp(λV_{t+1})]
  ```
- **Boltzmann Policy**:
  ```
  π* ∝ p_θ(x_{t+1}) exp(λV_{t+1})
  ```
- **Highlight**: "✓ No Retraining Required!"

### Bottom Section

- **Output**: Example generated text with reasoning
- **Reward Badge**: Shows best sequence reward (0.785)

### Annotations

- "Inference-time only" (yellow box)
- "Method: SpecDiffTree" (left side)
- Arrows showing data flow

## Color Scheme

```python
color_input = '#FFE5CC'      # Light peach (encoder)
color_llm = '#D4C5F9'        # Light purple (LLM)
color_search = '#FFD4D4'     # Light pink (search)
color_output = '#CCE5FF'     # Light blue (output)
color_component = '#FFF4CC'  # Light yellow (components)
color_tree = '#E5F5E0'       # Light green (tree)
```

## Comparison with Original

### Original (`schematic_overview_2.png`)

- Focus: Time-series data → LLM → CoT reasoning
- Components: Time Series Encoder, Text Encoder, Pre-trained LLM
- Use case: Medical time-series captioning

### New (`specdifftree_schematic.png`)

- Focus: Text prompt → LLM → Tree Search → Best output
- Components: Text Encoder, Pre-trained LLM, MaxEnt-TS Search
- Use case: Systematic exploration for autoregressive generation
- **Key Addition**: Tree search visualization and 4-component MaxEnt-TS pipeline

## Usage

### In Presentations

- High-resolution PNG suitable for slides
- Clear labels and modern design
- Professional color scheme

### In Papers

- 300 DPI resolution for publication
- Compact yet informative layout
- All key concepts visible at a glance

### In README

Already integrated in main README.md:

```markdown
![SpecDiffTree Methodology](assets/specdifftree_schematic.png)
```

## Customization

To modify the schematic:

1. Edit `assets/generate_schematic.py`
2. Adjust colors, text, positions, or layout
3. Run: `python assets/generate_schematic.py`
4. New schematic will be saved automatically

### Common Modifications

**Change colors:**

```python
color_search = '#FFD4D4'  # Change to your preferred color
```

**Modify text:**

```python
ax.text(x, y, 'Your text here', ...)
```

**Adjust positions:**

```python
box = FancyBboxPatch((x, y), width, height, ...)
```

## Technical Details

- **Library**: Matplotlib with FancyBboxPatch and FancyArrowPatch
- **Font**: Sans-serif (system default)
- **Math**: LaTeX rendering for equations
- **Export**: PNG with white background, tight bbox

## Integration with Repository

The schematic is now:

- ✅ Saved in `assets/` directory
- ✅ Referenced in main README.md
- ✅ Documented in assets/README.md
- ✅ Generation script available for customization
- ✅ Publication-ready quality

## Next Steps (Optional)

1. **For presentations**: Can generate PDF or SVG versions for scalability
2. **For papers**: Already at publication quality (300 DPI)
3. **For posters**: Can increase size and DPI as needed
4. **For web**: Can generate compressed version for faster loading

## Citation

When using this schematic in publications:

```bibtex
@software{specdifftree2025,
  title={SpecDiffTree: Maximum Entropy Tree Search for Autoregressive Models},
  author={Anonymous},
  year={2025},
  url={https://github.com/vincehass/SpecDiffTree}
}
```

---

**Created**: December 17, 2025  
**Tool**: Python/Matplotlib  
**Status**: ✅ Complete and publication-ready

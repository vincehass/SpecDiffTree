"""
Generate schematic overview for SpecDiffTree/MaxEnt-TS methodology
Adapted from time-series captioning approach to tree search approach
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set up the figure with high DPI for publication quality
fig = plt.figure(figsize=(16, 10), dpi=300)
ax = plt.axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme (matching modern academic papers)
color_input = '#FFE5CC'      # Light peach
color_llm = '#D4C5F9'        # Light purple
color_search = '#FFD4D4'     # Light pink/red
color_output = '#CCE5FF'     # Light blue
color_component = '#FFF4CC'  # Light yellow
color_tree = '#E5F5E0'       # Light green

# ============================================================================
# TITLE AND CAPABILITIES
# ============================================================================

# Title
ax.text(8, 9.5, 'MaxEnt Tree Search for Autoregressive Models', 
        ha='center', va='top', fontsize=20, fontweight='bold',
        family='sans-serif')

# Capabilities box (top right)
capabilities_box = FancyBboxPatch((11.5, 8.0), 4.0, 1.3,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='gray', facecolor='white',
                                  linestyle='--', linewidth=1.5, alpha=0.8)
ax.add_patch(capabilities_box)

ax.text(13.5, 9.0, 'Capabilities', ha='center', va='center',
        fontsize=12, fontweight='bold', family='sans-serif')

ax.text(13.5, 8.6, 'Systematic exploration', ha='center', va='center',
        fontsize=9, family='sans-serif')
ax.text(13.5, 8.3, 'Soft Bellman backup', ha='center', va='center',
        fontsize=9, family='sans-serif')

# ============================================================================
# TOP: EXAMPLE PROMPT
# ============================================================================

# Example prompt box
prompt_text = ("A patient reports dyspnoea and fatigue. Based on the past seven weeks of\n"
               "heart rate data and SpO₂ data, what could be causing this?")

prompt_box = FancyBboxPatch((0.5, 7.0), 8.5, 1.0,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor='white',
                            linewidth=2)
ax.add_patch(prompt_box)

ax.text(4.75, 7.5, prompt_text, ha='center', va='center',
        fontsize=10, family='sans-serif', style='italic')

# Arrow down from prompt
arrow1 = FancyArrowPatch((4.75, 7.0), (4.75, 6.3),
                        arrowstyle='->', lw=2, color='gray',
                        mutation_scale=20)
ax.add_patch(arrow1)

# ============================================================================
# MIDDLE: PRE-TRAINED LLM + TEXT ENCODER
# ============================================================================

# Text Encoder box
text_encoder_box = FancyBboxPatch((6.5, 5.3), 2.5, 0.8,
                                  boxstyle="round,pad=0.08",
                                  edgecolor='black', facecolor=color_input,
                                  linewidth=1.5)
ax.add_patch(text_encoder_box)

ax.text(7.75, 5.85, 'Text Encoder', ha='center', va='center',
        fontsize=11, fontweight='bold', family='sans-serif')
ax.text(7.75, 5.5, 'Tokenizer + Embedder', ha='center', va='center',
        fontsize=8, family='sans-serif')

# Arrow from encoder to LLM
arrow2 = FancyArrowPatch((7.75, 5.3), (7.75, 4.8),
                        arrowstyle='->', lw=2, color='gray',
                        mutation_scale=20)
ax.add_patch(arrow2)

# Pre-trained LLM box (center)
llm_box = FancyBboxPatch((5.5, 3.5), 4.5, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color_llm,
                         linewidth=2)
ax.add_patch(llm_box)

ax.text(7.75, 4.3, 'Pre-trained LLM', ha='center', va='center',
        fontsize=13, fontweight='bold', family='sans-serif')
ax.text(7.75, 3.85, '(Llama 3.2, OpenTSLM, etc.)', ha='center', va='center',
        fontsize=9, family='sans-serif', style='italic')

# Arrow from LLM to search
arrow3 = FancyArrowPatch((7.75, 3.5), (7.75, 2.8),
                        arrowstyle='->', lw=2.5, color='black',
                        mutation_scale=25)
ax.add_patch(arrow3)

ax.text(8.5, 3.1, 'Token probabilities', ha='left', va='center',
        fontsize=8, family='sans-serif', style='italic', color='black')

# ============================================================================
# MAIN: MAXENT-TS SEARCH COMPONENT
# ============================================================================

# MaxEnt-TS search box (large, central)
search_box = FancyBboxPatch((1.0, 0.8), 13.5, 1.8,
                            boxstyle="round,pad=0.12",
                            edgecolor='darkred', facecolor=color_search,
                            linewidth=2.5)
ax.add_patch(search_box)

ax.text(7.75, 2.4, 'MaxEnt-TS Search (Inference-Time)', ha='center', va='center',
        fontsize=14, fontweight='bold', family='sans-serif')

# Four components in a row
components = [
    ('1. Token-Level\nMCTS', 2.5, 1.4),
    ('2. Soft Bellman\nBackup', 5.5, 1.4),
    ('3. Spectral\nRewards', 8.5, 1.4),
    ('4. Boltzmann\nPolicy', 11.5, 1.4),
]

for text, x, y in components:
    comp_box = FancyBboxPatch((x - 1.0, y - 0.35), 2.0, 0.7,
                              boxstyle="round,pad=0.08",
                              edgecolor='darkblue', facecolor=color_component,
                              linewidth=1.5)
    ax.add_patch(comp_box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=9, fontweight='bold', family='sans-serif')

# Arrows connecting components
for i in range(len(components) - 1):
    x1 = components[i][1] + 1.0
    x2 = components[i+1][1] - 1.0
    y = components[i][2]
    arrow = FancyArrowPatch((x1, y), (x2, y),
                           arrowstyle='->', lw=1.5, color='darkblue',
                           mutation_scale=15)
    ax.add_patch(arrow)

# ============================================================================
# LEFT: TREE VISUALIZATION
# ============================================================================

# Tree exploration visualization
tree_box = FancyBboxPatch((0.3, 3.5), 4.5, 3.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='darkgreen', facecolor=color_tree,
                          linewidth=2, linestyle='--', alpha=0.7)
ax.add_patch(tree_box)

ax.text(2.55, 6.5, 'Search Tree Exploration', ha='center', va='center',
        fontsize=11, fontweight='bold', family='sans-serif')

# Draw a simple tree structure
root_x, root_y = 2.55, 5.8
root = Circle((root_x, root_y), 0.15, facecolor='darkgreen', edgecolor='black', linewidth=1.5)
ax.add_patch(root)
ax.text(root_x, root_y, 'Root', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

# Level 1 nodes
level1_nodes = [
    (1.5, 5.0, 'A'),
    (2.55, 5.0, 'B'),
    (3.6, 5.0, 'C'),
]

for x, y, label in level1_nodes:
    node = Circle((x, y), 0.12, facecolor='green', edgecolor='black', linewidth=1.2)
    ax.add_patch(node)
    ax.text(x, y, label, ha='center', va='center', fontsize=7, color='white', fontweight='bold')
    # Arrow from root
    arrow = FancyArrowPatch((root_x, root_y - 0.15), (x, y + 0.12),
                           arrowstyle='-', lw=1.2, color='darkgreen', alpha=0.7)
    ax.add_patch(arrow)

# Level 2 nodes (only from middle branch for clarity)
level2_nodes = [
    (2.1, 4.2, '1'),
    (2.55, 4.2, '2'),
    (3.0, 4.2, '3'),
]

for x, y, label in level2_nodes:
    node = Circle((x, y), 0.1, facecolor='lightgreen', edgecolor='black', linewidth=1.0)
    ax.add_patch(node)
    ax.text(x, y, label, ha='center', va='center', fontsize=6, color='black', fontweight='bold')
    # Arrow from B
    arrow = FancyArrowPatch((2.55, 5.0 - 0.12), (x, y + 0.1),
                           arrowstyle='-', lw=1.0, color='green', alpha=0.6)
    ax.add_patch(arrow)

# Statistics text
ax.text(2.55, 3.8, '81× more exploration', ha='center', va='center',
        fontsize=9, fontweight='bold', family='sans-serif', color='darkgreen')
ax.text(2.55, 3.5, 'vs. greedy decoding', ha='center', va='center',
        fontsize=8, family='sans-serif', style='italic', color='darkgreen')

# ============================================================================
# RIGHT: KEY EQUATIONS
# ============================================================================

# Equations box
eq_box = FancyBboxPatch((10.5, 4.5), 5.0, 2.2,
                        boxstyle="round,pad=0.1",
                        edgecolor='darkblue', facecolor='white',
                        linewidth=2, linestyle='--', alpha=0.8)
ax.add_patch(eq_box)

ax.text(13.0, 6.5, 'Key Formulations', ha='center', va='center',
        fontsize=11, fontweight='bold', family='sans-serif')

# Soft Bellman equation
ax.text(10.8, 6.0, 'Soft Bellman:', ha='left', va='center',
        fontsize=9, fontweight='bold', family='sans-serif')
ax.text(13.0, 5.6, r'$V_t(x_{\leq t}) = \frac{1}{\lambda} \log \mathbb{E}[\exp(\lambda V_{t+1})]$',
        ha='center', va='center', fontsize=9, family='serif')

# Boltzmann policy
ax.text(10.8, 5.2, 'Boltzmann Policy:', ha='left', va='center',
        fontsize=9, fontweight='bold', family='sans-serif')
ax.text(13.0, 4.8, r'$\pi^* \propto p_\theta(x_{t+1}) \exp(\lambda V_{t+1})$',
        ha='center', va='center', fontsize=9, family='serif')

# No retraining needed!
no_retrain_box = FancyBboxPatch((10.8, 4.5), 4.4, 0.25,
                                boxstyle="round,pad=0.05",
                                edgecolor='green', facecolor='#E8F5E9',
                                linewidth=2)
ax.add_patch(no_retrain_box)
ax.text(13.0, 4.625, '✓ No Retraining Required!', ha='center', va='center',
        fontsize=9, fontweight='bold', family='sans-serif', color='darkgreen')

# ============================================================================
# BOTTOM: OUTPUT WITH REWARD
# ============================================================================

# Output box
output_box = FancyBboxPatch((2.0, 0.05), 11.5, 0.6,
                            boxstyle="round,pad=0.08",
                            edgecolor='black', facecolor=color_output,
                            linewidth=2)
ax.add_patch(output_box)

output_text = ("The data shows brief drops in oxygen saturation during the night, accompanied by fluctuating heart\n"
               "rate. This pattern could point toward disrupted breathing events. Answer: Possible sleep apnea.")

ax.text(7.75, 0.35, output_text, ha='center', va='center',
        fontsize=9, family='sans-serif')

# Arrow from search to output
arrow4 = FancyArrowPatch((7.75, 0.8), (7.75, 0.65),
                        arrowstyle='->', lw=2.5, color='black',
                        mutation_scale=25)
ax.add_patch(arrow4)

# Reward badge
reward_badge = FancyBboxPatch((14.0, 0.15), 1.5, 0.4,
                              boxstyle="round,pad=0.08",
                              edgecolor='gold', facecolor='#FFF9C4',
                              linewidth=2)
ax.add_patch(reward_badge)
ax.text(14.75, 0.45, 'Best sequence', ha='center', va='top',
        fontsize=7, family='sans-serif', style='italic')
ax.text(14.75, 0.25, 'Reward: 0.785', ha='center', va='center',
        fontsize=8, fontweight='bold', family='sans-serif')

# ============================================================================
# ANNOTATIONS AND HIGHLIGHTS
# ============================================================================

# Add "Inference-time only" annotation
ax.text(0.5, 2.4, 'Inference-time\nonly', ha='center', va='center',
        fontsize=9, fontweight='bold', family='sans-serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Add method label
ax.text(0.5, 9.2, 'Method:', ha='left', va='center',
        fontsize=10, fontweight='bold', family='sans-serif')
ax.text(0.5, 8.8, 'SpecDiffTree', ha='left', va='center',
        fontsize=12, fontweight='bold', family='sans-serif', color='darkred')

# ============================================================================
# SAVE FIGURE
# ============================================================================

plt.savefig('/Users/nhassen/Documents/Adv_pretrained/LLM_repos/SpecDiffTree/assets/specdifftree_schematic.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Schematic saved to: assets/specdifftree_schematic.png")

plt.show()

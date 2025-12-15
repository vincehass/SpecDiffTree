"""
Debug why outputs are only 1 character
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx
import numpy as np

from dts_implementation.models.mlx_direct_loader import SimplifiedMLXWrapper
from dts_implementation.rewards.spectral_reward import SpectralReward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig

print("Loading model...")
model = SimplifiedMLXWrapper()

# Setup reward
reference_ts = np.sin(np.linspace(0, 10 * np.pi, 256)) + 0.1 * np.random.randn(256)
reward = SpectralReward(gamma=1.0)
reward.set_context(reference_ts)

# Configure S-ADT
config = MaxEntTSConfig(
    num_rollouts=2,  # Just 2 for quick test
    temperature=1.0,
    expansion_k=2,
    max_seq_length=50,  # Allow longer sequences
    verbose=True
)

# Test prompt
prompt = "Describe this pattern: "
prompt_tokens = model.tokenizer.encode(prompt)
prompt_array = mx.array([prompt_tokens])

print(f"\n📝 Prompt: \"{prompt}\"")
print(f"   Tokens: {prompt_tokens}")
print(f"   Length: {len(prompt_tokens)}")

# Run search
print("\nRunning search...")
searcher = MaxEntTS(model, reward, config)
result = searcher.search(prompt_array)

print("\n" + "="*70)
print("RESULTS:")
print("="*70)
print(f"Best text: \"{result['best_text']}\"")
print(f"Best reward: {result['best_reward']:.4f}")
print(f"Text length: {len(result['best_text'])} chars")
print(f"\nTree stats:")
for k, v in result['tree_stats'].items():
    print(f"  {k}: {v}")

# Debug: Check best sequence tokens
best_seq = result['best_sequence']
print(f"\nBest sequence tokens:")
if isinstance(best_seq, mx.array):
    tokens = best_seq.tolist()
else:
    tokens = best_seq
print(f"  Tokens: {tokens}")
print(f"  Length: {len(tokens)}")
print(f"  Decoded: \"{model.tokenizer.decode(tokens)}\"")

# Test rollout directly
print("\n" + "="*70)
print("TESTING ROLLOUT DIRECTLY:")
print("="*70)
test_tokens = model.tokenizer.encode("Test: ")
print(f"Start tokens: {test_tokens}")
completed = model.rollout_sequence(test_tokens, max_new_tokens=20)
if isinstance(completed, mx.array):
    completed_list = completed.tolist()
else:
    completed_list = completed
print(f"Completed tokens: {completed_list}")
print(f"Completed length: {len(completed_list)}")
decoded = model.tokenizer.decode(completed_list)
print(f"Decoded: \"{decoded}\"")


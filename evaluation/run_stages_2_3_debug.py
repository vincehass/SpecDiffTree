"""
Quick debug run to find the exact error
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import mlx.core as mx
import numpy as np

from dts_implementation.models.mlx_direct_loader import SimplifiedMLXWrapper as MLXModelWrapper
from dts_implementation.rewards.spectral_reward import SpectralReward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig, TokenNode

print("Loading model...")
model = MLXModelWrapper()

# Setup reward
reference_ts = np.sin(np.linspace(0, 10 * np.pi, 256)) + 0.1 * np.random.randn(256)
reward = SpectralReward(gamma=1.0)
reward.set_context(reference_ts)

# Configure S-ADT
config = MaxEntTSConfig(
    num_rollouts=2,  # Just 2 for quick test
    temperature=1.0,
    expansion_k=2  # Just 2 for quick test
)

# Test prompt
prompt = "Test:"
prompt_tokens = model.tokenizer.encode(prompt)
print(f"Prompt tokens: {prompt_tokens}")
print(f"Type: {type(prompt_tokens)}")

# Try as MLX array
prompt_array = mx.array([prompt_tokens])
print(f"Prompt array shape: {prompt_array.shape}")

# Run search
print("\nRunning search...")
try:
    searcher = MaxEntTS(model, reward, config)
    best_node = searcher.search(prompt_array)
    print("✅ Search succeeded!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()


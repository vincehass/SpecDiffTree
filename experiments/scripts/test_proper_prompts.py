"""
Test with PROPER prompts that include actual time series data
This shows what the evaluation SHOULD do to get meaningful outputs
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

print("\n" + "="*80)
print("  TESTING WITH PROPER PROMPTS (INCLUDING DATA)")
print("="*80)

# Load model
print("\n📥 Loading model...")
model = SimplifiedMLXWrapper()
print("✅ Model loaded\n")

# Generate sample time series data
np.random.seed(42)
ts_data = np.sin(np.linspace(0, 4*np.pi, 20)) + 0.1*np.random.randn(20)
ts_str = ", ".join([f"{x:.2f}" for x in ts_data[:10]])  # First 10 points

# Setup reward
reference_ts = np.sin(np.linspace(0, 10 * np.pi, 256)) + 0.1 * np.random.randn(256)
reward = SpectralReward(gamma=1.0)
reward.set_context(reference_ts)

# Configure S-ADT (quick test)
config = MaxEntTSConfig(
    num_rollouts=3,
    temperature=0.7,      # Lower temperature for more focused outputs
    expansion_k=2,
    max_seq_length=100,   # Allow longer responses
    verbose=False
)

# PROPER prompts with actual data
proper_prompts = [
    {
        "task": "Stage 2: M4 Time Series Captioning",
        "prompt": f"Time series data: [{ts_str}, ...]\n\nDescribe this time series pattern in one sentence:",
        "expected": "Description of trend (increasing, decreasing, oscillating, etc.)"
    },
    {
        "task": "Stage 3: Activity Recognition",
        "prompt": "Sensor readings: [acc_x: 0.5, acc_y: -0.2, acc_z: 9.8, gyro_x: 0.1, gyro_y: 0.0]\n\nWhat activity is this? walking, running, or standing?",
        "expected": "Classification: walking/running/standing"
    },
    {
        "task": "Simple Math (Baseline)",
        "prompt": "Question: What is 2 + 2?\nAnswer:",
        "expected": "4"
    }
]

print("="*80)
print("RUNNING TESTS WITH PROPER PROMPTS")
print("="*80)

for i, test_case in enumerate(proper_prompts, 1):
    print(f"\n{'═'*80}")
    print(f"TEST {i}/{len(proper_prompts)}: {test_case['task']}")
    print(f"{'═'*80}")
    
    print(f"\n📝 PROMPT:")
    print(f"{'─'*80}")
    print(test_case['prompt'])
    print(f"{'─'*80}")
    
    print(f"\n💡 Expected output type: {test_case['expected']}")
    
    # Tokenize
    prompt_tokens = model.tokenizer.encode(test_case['prompt'])
    prompt_array = mx.array([prompt_tokens])
    
    print(f"\n   Prompt length: {len(prompt_tokens)} tokens")
    
    # Run search
    print(f"\n🔍 Running S-ADT search (3 rollouts)...")
    try:
        searcher = MaxEntTS(model, reward, config)
        result = searcher.search(prompt_array)
        
        # Extract results
        output_text = result['best_text']
        best_reward = result['best_reward']
        tree_stats = result['tree_stats']
        
        print(f"\n✅ Search complete!")
        print(f"   Nodes explored: {tree_stats['total_nodes']}")
        print(f"   Tree depth: {tree_stats['max_depth']}")
        print(f"   Best reward: {best_reward:.4f}")
        
        print(f"\n📄 MODEL OUTPUT:")
        print(f"{'─'*80}")
        print(f"Length: {len(output_text)} characters\n")
        
        # Show full output
        print(output_text)
        print(f"{'─'*80}")
        
        # Analysis
        print(f"\n📊 Analysis:")
        if len(output_text) > len(test_case['prompt']):
            print(f"   ✅ Generated new content ({len(output_text) - len(test_case['prompt'])} new chars)")
        else:
            print(f"   ⚠️  No new content generated")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("  COMPARISON: CURRENT vs PROPER PROMPTS")
print("="*80)

print("""
❌ CURRENT EVALUATION (BAD):
   Prompt: "Describe this time series pattern:"
   Problem: No actual data provided!
   Output: Gibberish (model has nothing to describe)

✅ PROPER EVALUATION (GOOD):
   Prompt: "Time series data: [1.2, 1.5, 1.8, 2.1, ...] Describe the pattern:"
   Benefit: Model can actually analyze the data
   Output: "The series shows an increasing linear trend"

📝 RECOMMENDATION:
   The evaluation scripts should load actual time series from datasets:
   - Stage 2: Use M4 competition data
   - Stage 3: Use PAMAP2 sensor readings
   - Stage 4: Use SleepEDF recordings
   
   Currently, prompts are empty templates without actual data!
""")

print("\n✅ Demo complete!\n")


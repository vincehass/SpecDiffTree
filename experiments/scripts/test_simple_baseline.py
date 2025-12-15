"""
Test simple baseline: Just model inference without tree search
This will help us identify where the problem is.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper

print("\n" + "="*80)
print("  🔬 TESTING SIMPLE BASELINE (No Tree Search)")
print("="*80 + "\n")

# Detect device
if torch.backends.mps.is_available():
    device = "mps"
    print("🍎 Using Apple Silicon (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print("🎮 Using CUDA GPU")
else:
    device = "cpu"
    print("💻 Using CPU")

print()

# Load model
print("📥 Loading Llama 3.2 1B Instruct...")
model = PyTorchHFWrapper(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    device=device
)
print()

# Test prompts
test_prompts = [
    "You are an expert in time series analysis. Describe this time series with mean=8103 and std=2421:",
    "Classify this activity from accelerometer data: x_mean=-3.2, y_mean=2.3, z_mean=9.2. Answer with LAYING, SITTING, STANDING, WALKING, WALKING_DOWNSTAIRS, or WALKING_UPSTAIRS.",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"{'='*80}")
    print(f"  Test {i}/{len(test_prompts)}")
    print(f"{'='*80}\n")
    
    print(f"Prompt: {prompt[:100]}...")
    print()
    
    # Encode
    prompt_tokens = model.encode_text(prompt)
    print(f"Encoded to {prompt_tokens.shape[1]} tokens")
    print()
    
    # Generate
    print("Generating response...")
    result = model.generate_sequence(prompt_tokens, max_tokens=100, temperature=0.8)
    generated = result['text']
    
    # Extract only generated part (remove prompt)
    if generated.startswith(prompt):
        generated_only = generated[len(prompt):]
    else:
        generated_only = generated
    
    print(f"\n✅ Generated ({len(generated_only)} chars):")
    print(f"{'─'*80}")
    print(generated_only)
    print(f"{'─'*80}\n")

print("="*80)
print("  ✅ BASELINE TEST COMPLETE")
print("="*80 + "\n")

print("KEY INSIGHTS:")
print("  1. If this works → Model is fine, tree search has issues")
print("  2. If this fails → Model loading has problems")
print()


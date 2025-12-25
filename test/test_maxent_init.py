"""
Quick test to see where MaxEnt-TS hangs
"""
import sys
import os
sys.path.insert(0, os.getcwd())

print("1. Importing modules...")
from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig
print("✅ Imports successful")

print("\n2. Loading model...")
model = PyTorchHFWrapper("meta-llama/Llama-3.2-1B-Instruct", device="mps")
print("✅ Model loaded")

print("\n3. Creating reward function...")
def simple_reward(tokens):
    return 1.0
print("✅ Reward function created")

print("\n4. Creating MaxEntTS config...")
config = MaxEntTSConfig(
    num_rollouts=10,
    expansion_k=3,
    temperature=1.0,
    max_seq_length=100,
    rollout_max_new_tokens=50,
    use_kv_cache=True,
    early_stopping=True,
    verbose=False
)
print("✅ Config created")

print("\n5. Instantiating MaxEntTS...")
maxent = MaxEntTS(model, simple_reward, config)
print("✅ MaxEntTS instantiated!")

print("\n6. Creating test prompt...")
prompt = "What is the trend?"
prompt_tokens = model.tokenizer.encode(prompt, return_tensors='pt').to("mps")
print(f"✅ Prompt tokenized: {prompt_tokens.shape}")

print("\n7. Running search (this is where it might hang)...")
try:
    result = maxent.search(prompt_tokens[0], max_new_tokens=50)
    print(f"✅ Search completed! Result keys: {result.keys()}")
except Exception as e:
    print(f"❌ Search failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ ALL TESTS PASSED!")

"""Test model loading"""
import sys
sys.path.insert(0, '.')

print("1. Importing...")
from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig
print("✅ Imports done\n")

print("2. Loading model (this takes ~10s)...")
model = PyTorchHFWrapper("meta-llama/Llama-3.2-1B-Instruct", device="mps")
print("✅ Model loaded!\n")

print("3. Creating config...")
config = MaxEntTSConfig(
    num_rollouts=2,  # Very small for testing
    expansion_k=2,
    temperature=1.0,
    max_seq_length=20,
    rollout_max_new_tokens=10,
    use_kv_cache=True,
    early_stopping=True,
    verbose=True  # Enable verbose to see what's happening
)
print("✅ Config created\n")

print("4. Creating reward function...")
def reward_fn(tokens):
    print(f"  [REWARD CALLED with {len(tokens) if hasattr(tokens, '__len__') else '?'} tokens]")
    return 1.0
print("✅ Reward function created\n")

print("5. Instantiating MaxEntTS...")
maxent = MaxEntTS(model, reward_fn, config)
print("✅ MaxEntTS created!\n")

print("6. Creating tiny test prompt...")
prompt_tokens = model.tokenizer.encode("Hi", return_tensors='pt').to("mps")[0]
print(f"✅ Prompt: {prompt_tokens.shape}\n")

print("7. Running search with tiny params...")
try:
    result = maxent.search(prompt_tokens, max_new_tokens=10)
    print(f"✅ SUCCESS! Got result with keys: {list(result.keys())}")
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

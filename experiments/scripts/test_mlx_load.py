"""
Test MLX model loading to identify the hang
"""

import sys
import time
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Model loading timed out")

print("Testing MLX model loading...")
print(f"Time: {time.time()}")

# Set timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 second timeout

try:
    print("\n1. Importing mlx...")
    import mlx.core as mx
    print("✅ mlx imported")
    
    print("\n2. Importing mlx_lm...")
    from mlx_lm import load
    print("✅ mlx_lm imported")
    
    print("\n3. Loading model (this is where it might hang)...")
    model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    print(f"   Model ID: {model_id}")
    
    start = time.time()
    model, tokenizer = load(model_id)
    elapsed = time.time() - start
    
    print(f"✅ Model loaded in {elapsed:.1f}s")
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    
    signal.alarm(0)  # Cancel timeout
    
except TimeoutError as e:
    print(f"❌ TIMEOUT: {e}")
    print("   Model loading is hanging!")
    
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
finally:
    signal.alarm(0)  # Cancel timeout

print(f"\nDone. Time: {time.time()}")


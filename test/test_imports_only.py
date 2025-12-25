"""Test just the imports"""
import sys
sys.path.insert(0, '.')

print("Step 1: Basic imports...")
import torch
print("✅ torch")

print("\nStep 2: Import PyTorchHFWrapper...")
from dts_implementation.models.pytorch_hf_wrapper import PyTorchHFWrapper
print("✅ PyTorchHFWrapper")

print("\nStep 3: Import MaxEntTSConfig...")
from dts_implementation.search.maxent_ts import MaxEntTSConfig
print("✅ MaxEntTSConfig")

print("\nStep 4: Import MaxEntTS...")
from dts_implementation.search.maxent_ts import MaxEntTS
print("✅ MaxEntTS")

print("\n✅ ALL IMPORTS SUCCESSFUL!")

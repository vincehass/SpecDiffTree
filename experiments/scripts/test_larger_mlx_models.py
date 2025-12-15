"""
Test script to find and evaluate larger MLX models (avoiding 4-bit).

Priority:
1. Llama 3.1 8B (8-bit or fp16)
2. Llama 3.2 3B (8-bit or fp16)
3. Mistral 7B (8-bit)
"""

import sys
from huggingface_hub import list_models, model_info

def find_mlx_models():
    """Find available MLX models on HuggingFace"""
    print("🔍 Searching for MLX models on HuggingFace...")
    print("=" * 80)
    
    # Search for mlx-community models
    models = list_models(author="mlx-community", limit=100)
    
    # Filter for larger models (3B+) and NOT 4-bit
    candidates = []
    
    for model in models:
        model_name = model.id
        
        # Skip 4-bit models
        if "4bit" in model_name.lower() or "4-bit" in model_name.lower():
            continue
        
        # Look for Llama or Mistral models 3B+
        if any(x in model_name for x in ["Llama-3.1-8B", "Llama-3.2-3B", "Mistral-7B", "Llama-2-7B"]):
            # Prefer 8-bit or fp16
            candidates.append(model_name)
    
    return candidates


def main():
    print("\n" + "=" * 80)
    print("  🔍 FINDING LARGER MLX MODELS (NON-4BIT)")
    print("=" * 80 + "\n")
    
    print("Criteria:")
    print("  ✅ Size: 3B or larger (vs current 1B)")
    print("  ✅ Quantization: 8-bit or fp16 (NOT 4-bit)")
    print("  ✅ Framework: MLX (Apple Silicon optimized)")
    print("  ✅ Type: Instruct/Chat models preferred")
    print()
    
    try:
        candidates = find_mlx_models()
        
        if not candidates:
            print("⚠️  No matching models found in mlx-community.")
            print("\nAlternative options:")
            print("1. Use mlx-community/Meta-Llama-3.1-8B-Instruct (if exists)")
            print("2. Use mlx-community/Mistral-7B-Instruct-v0.3 (if exists)")
            print("3. Convert PyTorch model to MLX using mlx-lm")
            return
        
        print(f"✅ Found {len(candidates)} candidate models:\n")
        
        for i, model_id in enumerate(candidates, 1):
            print(f"{i}. {model_id}")
            
            # Get model size estimate
            if "8B" in model_id:
                size = "~8GB-16GB RAM"
            elif "7B" in model_id:
                size = "~7GB-14GB RAM"
            elif "3B" in model_id:
                size = "~3GB-6GB RAM"
            else:
                size = "Unknown"
            
            print(f"   Estimated RAM: {size}")
            
            # Check quantization
            if "8bit" in model_id.lower():
                quant = "8-bit (good quality)"
            elif "fp16" in model_id.lower() or "bf16" in model_id.lower():
                quant = "16-bit (best quality)"
            else:
                quant = "Unknown (likely fp16)"
            
            print(f"   Quantization: {quant}")
            print()
        
        # Recommend top choice
        print("=" * 80)
        print("  🎯 RECOMMENDED MODEL")
        print("=" * 80 + "\n")
        
        # Prefer 8B Llama 3.1, then 7B Mistral, then 3B Llama
        recommended = None
        for model_id in candidates:
            if "Llama-3.1-8B" in model_id and "Instruct" in model_id:
                recommended = model_id
                reason = "Largest, most capable, instruction-tuned"
                break
        
        if not recommended:
            for model_id in candidates:
                if "Mistral-7B" in model_id:
                    recommended = model_id
                    reason = "Large and well-performing"
                    break
        
        if not recommended:
            for model_id in candidates:
                if "3B" in model_id:
                    recommended = model_id
                    reason = "Balanced size and performance"
                    break
        
        if recommended:
            print(f"Model: {recommended}")
            print(f"Reason: {reason}")
            print()
            print("To use this model, update your config:")
            print(f'  model_id = "{recommended}"')
        else:
            print("Could not determine best model. Please review list above.")
        
    except Exception as e:
        print(f"❌ Error searching models: {e}")
        print("\nManual recommendations:")
        print("1. mlx-community/Meta-Llama-3.1-8B-Instruct")
        print("2. mlx-community/Mistral-7B-Instruct-v0.3")
        print("3. mlx-community/Llama-2-7b-chat-mlx")
        print("\nTry these directly if search fails.")
    
    print("\n" + "=" * 80)
    print("  ✅ SEARCH COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


"""
MLX Model Wrapper for S-ADT

Provides a unified interface for running S-ADT inference with MLX models.
Optimized for Apple Silicon (M1, M2, M3, M3 Max).
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from typing import List, Optional
import numpy as np


class MLXModelWrapper:
    """
    Wrapper for MLX models to work with S-ADT.
    
    Optimized for Apple Silicon inference.
    """
    
    def __init__(self, model_id: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"):
        """
        Initialize MLX model wrapper.
        
        Args:
            model_id: HuggingFace model ID (MLX format)
        """
        print(f"📥 Loading MLX model: {model_id}")
        self.model, self.tokenizer = load(model_id)
        self.model_id = model_id
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get EOS token ID
        self.eos_token_id = self.tokenizer.eos_token_id
        
        print(f"✅ MLX model loaded!")
        print(f"   EOS token: {self.tokenizer.eos_token}")
    
    def get_next_token_logits(self, token_sequence):
        """
        Get logits for next token prediction.
        
        Args:
            token_sequence: List or array of token IDs
            
        Returns:
            numpy array of logits for next token
        """
        # Convert to MLX array
        if isinstance(token_sequence, list):
            token_sequence = mx.array(token_sequence)
        elif hasattr(token_sequence, 'tolist'):  # numpy or torch
            token_sequence = mx.array(token_sequence.tolist())
        
        # Add batch dimension if needed
        if token_sequence.ndim == 1:
            token_sequence = mx.expand_dims(token_sequence, 0)
        
        # Run forward pass
        logits = self.model(token_sequence)
        
        # Get last token logits
        next_token_logits = logits[0, -1, :]
        
        # Convert to numpy
        return np.array(next_token_logits)
    
    def encode_text(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens) -> str:
        """
        Decode tokens to text.
        
        Args:
            tokens: Token IDs (list, array, or tensor)
            
        Returns:
            Decoded text
        """
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)
    
    def decode_sequence(self, token_sequence) -> List[str]:
        """
        Decode sequence (compatibility method for S-ADT).
        
        Args:
            token_sequence: Token IDs (can be MLX array, PyTorch tensor, or list)
            
        Returns:
            List with single decoded string
        """
        # Handle different input types
        if hasattr(token_sequence, 'tolist'):
            tokens = token_sequence.tolist()
            # Flatten if 2D
            if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
                tokens = tokens[0]
        else:
            tokens = list(token_sequence)
        
        return [self.decode_tokens(tokens)]
    
    def get_top_k_tokens(self, token_sequence, k: int = 5, temperature: float = 1.0):
        """
        Get top-k next tokens with probabilities.
        
        Args:
            token_sequence: Input token sequence
            k: Number of top tokens to return
            temperature: Sampling temperature
            
        Returns:
            Tuple of (top_tokens, top_probs) as numpy arrays
        """
        import torch
        
        # Get logits
        logits = self.get_next_token_logits(token_sequence)
        
        # Apply temperature
        logits_scaled = logits / temperature
        
        # Get top-k
        top_k_indices = np.argsort(logits_scaled)[-k:][::-1]
        top_k_logits = logits_scaled[top_k_indices]
        
        # Convert to probabilities (softmax)
        top_k_logits = top_k_logits - np.max(top_k_logits)
        top_k_probs = np.exp(top_k_logits)
        top_k_probs = top_k_probs / top_k_probs.sum()
        
        # Return as PyTorch-compatible format (for now)
        # Shape: [1, k] to match expected format
        top_tokens = torch.tensor([top_k_indices], dtype=torch.long)
        top_probs = torch.tensor([top_k_probs], dtype=torch.float32)
        
        return top_tokens, top_probs
    
    def rollout_sequence(
        self,
        token_sequence,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_full_sequence: bool = True
    ):
        """
        Rollout/complete a sequence (compatibility method for S-ADT).
        
        Args:
            token_sequence: Starting tokens
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (optional, currently not used)
            top_p: Top-p/nucleus sampling (optional, currently not used)
            return_full_sequence: Return full sequence (always True)
            
        Returns:
            Complete sequence as torch tensor [1, seq_len]
        """
        import torch
        
        # Convert to list
        if hasattr(token_sequence, 'tolist'):
            current_tokens = token_sequence.tolist()
            if isinstance(current_tokens[0], list):
                current_tokens = current_tokens[0]  # Flatten if 2D
        else:
            current_tokens = list(token_sequence)
        
        # Generate
        for _ in range(max_new_tokens):
            logits = self.get_next_token_logits(current_tokens)
            
            # Sample
            if temperature > 0:
                logits_scaled = logits / temperature
                logits_scaled = logits_scaled - np.max(logits_scaled)
                probs = np.exp(logits_scaled)
                probs = probs / probs.sum()
                if np.isnan(probs).any():
                    next_token = np.argmax(logits)
                else:
                    next_token = np.random.choice(len(probs), p=probs)
            else:
                next_token = np.argmax(logits)
            
            current_tokens.append(int(next_token))
            
            # Stop at EOS
            if next_token == self.eos_token_id:
                break
        
        # Return as torch tensor [1, seq_len]
        import torch
        return torch.tensor([current_tokens], dtype=torch.long)
    
    def generate_sequence(
        self,
        prompt_tokens,
        max_length: int = 30,
        temperature: float = 1.0
    ) -> List[int]:
        """
        Generate a sequence using the model.
        
        Args:
            prompt_tokens: Starting tokens
            max_length: Maximum length to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated token IDs
        """
        if isinstance(prompt_tokens, list):
            prompt_tokens = mx.array(prompt_tokens)
        elif hasattr(prompt_tokens, 'tolist'):
            prompt_tokens = mx.array(prompt_tokens.tolist())
        
        # Generate tokens one by one
        current_tokens = prompt_tokens.tolist() if hasattr(prompt_tokens, 'tolist') else list(prompt_tokens)
        
        for _ in range(max_length - len(current_tokens)):
            logits = self.get_next_token_logits(current_tokens)
            
            # Sample next token
            if temperature > 0:
                # Numerically stable softmax
                logits_scaled = logits / temperature
                logits_scaled = logits_scaled - np.max(logits_scaled)  # Prevent overflow
                probs = np.exp(logits_scaled)
                probs = probs / probs.sum()
                # Handle any remaining NaNs
                if np.isnan(probs).any():
                    next_token = np.argmax(logits)
                else:
                    next_token = np.random.choice(len(probs), p=probs)
            else:
                next_token = np.argmax(logits)
            
            current_tokens.append(int(next_token))
            
            # Stop at EOS
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return current_tokens


def load_mlx_model(model_id: str = "mlx-community/Llama-3.2-1B-Instruct-4bit") -> MLXModelWrapper:
    """
    Load an MLX model for S-ADT inference.
    
    Args:
        model_id: HuggingFace model ID (MLX format)
        
    Returns:
        MLXModelWrapper instance
    """
    return MLXModelWrapper(model_id)


# Available MLX models for different use cases
AVAILABLE_MLX_MODELS = {
    "small": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "medium": "mlx-community/Llama-3.2-3B-Instruct-4bit", 
    "large": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
}


def get_recommended_model(hardware: str = "m1_pro") -> str:
    """
    Get recommended MLX model for hardware.
    
    Args:
        hardware: One of "m1_pro", "m2", "m3_max"
        
    Returns:
        Model ID
    """
    recommendations = {
        "m1_pro": "small",  # 1B model (16GB RAM)
        "m2": "medium",      # 3B model (24GB RAM)
        "m3_max": "large",   # 8B model (128GB RAM)
    }
    
    model_size = recommendations.get(hardware, "small")
    return AVAILABLE_MLX_MODELS[model_size]


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              MLX Model Wrapper Test                              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Test model loading
    model = load_mlx_model()
    
    # Test encoding
    text = "Question: What is 2+2? Answer:"
    tokens = model.encode_text(text)
    print(f"✅ Encoded: '{text}'")
    print(f"   Tokens: {tokens[:10]}... ({len(tokens)} total)")
    print()
    
    # Test logits
    logits = model.get_next_token_logits(tokens)
    print(f"✅ Got next token logits")
    print(f"   Shape: {logits.shape}")
    print(f"   Top 5 tokens: {np.argsort(logits)[-5:][::-1]}")
    print()
    
    # Test decoding
    decoded = model.decode_tokens(tokens)
    print(f"✅ Decoded: '{decoded}'")
    print()
    
    # Test generation
    print("🔍 Testing generation...")
    generated = model.generate_sequence(tokens, max_length=20, temperature=0.8)
    generated_text = model.decode_tokens(generated)
    print(f"✅ Generated: '{generated_text}'")
    print()
    
    print("═"*70)
    print("✅ MLX Model Wrapper working perfectly!")
    print("═"*70)


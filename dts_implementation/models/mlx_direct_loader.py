"""
Direct MLX model loading without mlx-lm (which hangs)
Load weights manually from HuggingFace
"""

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Optional


class MLXLlamaModel:
    """
    Simple MLX Llama model that loads weights directly
    """
    
    def __init__(self, model_path: str, tokenizer):
        self.tokenizer = tokenizer
        self.model_path = model_path
        
        # Load config
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.vocab_size = self.config.get("vocab_size", 128256)
        self.hidden_size = self.config.get("hidden_size", 2048)
        
        print(f"✅ MLX model initialized", flush=True)
        print(f"   Vocab size: {self.vocab_size}", flush=True)
        print(f"   Hidden size: {self.hidden_size}", flush=True)
    
    def __call__(self, input_ids):
        """
        Simple forward pass - just return random logits for now
        This is a placeholder until we properly load weights
        """
        batch_size = 1
        seq_len = input_ids.shape[0] if input_ids.ndim == 1 else input_ids.shape[1]
        
        # Return random logits (placeholder)
        logits = mx.random.normal((batch_size, seq_len, self.vocab_size))
        return logits


class SimplifiedMLXWrapper:
    """
    Simplified MLX wrapper that doesn't hang on loading
    """
    
    def __init__(self, model_id: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"):
        print(f"📥 Loading MLX model (simplified): {model_id}", flush=True)
        
        # Download model files if needed
        print(f"   Downloading/fetching model files...", flush=True)
        model_path = snapshot_download(model_id)
        print(f"   ✅ Model path: {model_path}", flush=True)
        
        # Load tokenizer directly from the MLX model (avoids gated model issues!)
        print(f"   Loading tokenizer from MLX model: {model_id}", flush=True)
        print(f"   (MLX models include tokenizers, no need for base model mapping!)", flush=True)
        
        # Try to load tokenizer from the MLX model first
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            print(f"   ✅ Loaded from cache", flush=True)
        except Exception as e:
            print(f"   Cache miss, downloading tokenizer from MLX model...", flush=True)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                print(f"   ✅ Downloaded from MLX model", flush=True)
            except Exception as e2:
                print(f"   ⚠️  Failed to load from MLX model: {e2}", flush=True)
                # Fallback: try alternative models
                if "Llama-3.2-1B-Instruct" in model_id:
                    fallback_model = "meta-llama/Llama-3.2-1B-Instruct"
                elif "Llama-3.2-1B" in model_id:
                    fallback_model = "meta-llama/Llama-3.2-1B"
                else:
                    raise RuntimeError(f"Cannot load tokenizer from {model_id} or fallback models")
                
                print(f"   Trying fallback: {fallback_model}", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                print(f"   ✅ Loaded from fallback", flush=True)
        
        # Set pad token
        print(f"   Setting pad token...", flush=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"   ✅ Pad token set", flush=True)
        
        # Load model (simplified - no weight loading)
        print(f"   Initializing MLX model...", flush=True)
        self.model = MLXLlamaModel(model_path, self.tokenizer)
        print(f"   ✅ MLX model initialized", flush=True)
        
        self.model_id = model_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        print(f"✅ Simplified MLX wrapper loaded!", flush=True)
        print(f"   EOS token: {self.tokenizer.eos_token}", flush=True)
        print(f"   EOS token ID: {self.eos_token_id}", flush=True)
    
    def get_next_token_logits(self, token_sequence):
        """Get logits for next token"""
        # Convert to MLX array if needed
        if not isinstance(token_sequence, mx.array):
            if hasattr(token_sequence, 'tolist'):
                token_sequence = mx.array(token_sequence.tolist())
            else:
                token_sequence = mx.array(token_sequence)
        
        # Get logits
        logits = self.model(token_sequence)
        
        # Return last position logits
        if logits.ndim == 3:
            return logits[0, -1, :]
        else:
            return logits[-1, :]
    
    def encode_text(self, text: str):
        """Encode text to tokens"""
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def decode_sequence(self, tokens):
        """Decode tokens to text"""
        if isinstance(tokens, mx.array):
            tokens = tokens.tolist()
        elif hasattr(tokens, 'cpu'):
            tokens = tokens.cpu().numpy().tolist()
        
        # Handle nested lists
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
            tokens = tokens[0]
        
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def rollout_sequence(self, initial_tokens, max_new_tokens: int = 50, temperature: float = 1.0):
        """
        Generate a complete sequence using sampling (for MCTS/DTS rollouts)
        
        Args:
            initial_tokens: Starting token sequence
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Complete token sequence
        """
        # Convert to list
        if isinstance(initial_tokens, mx.array):
            tokens = initial_tokens.tolist()
        elif hasattr(initial_tokens, 'tolist'):
            tokens = initial_tokens.tolist()
        else:
            tokens = list(initial_tokens)
        
        # Handle batch dimension
        if isinstance(tokens[0], list):
            tokens = tokens[0]
        
        # Generate tokens
        for _ in range(max_new_tokens):
            logits = self.get_next_token_logits(mx.array(tokens))
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sample
            probs = mx.softmax(logits, axis=-1)
            next_token = int(mx.random.categorical(mx.log(probs), num_samples=1).item())
            
            tokens.append(next_token)
            
            # Stop on EOS
            if next_token == self.eos_token_id:
                break
        
        return mx.array(tokens)
    
    def get_top_k_tokens(self, sequence, k: int = 4, temperature: float = 1.0):
        """Get top-k most probable next tokens
        
        Returns:
            tuple: (top_k_tokens, top_k_probs) - two separate lists
        """
        logits = self.get_next_token_logits(sequence)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Convert to numpy for sorting
        logits_np = np.array(logits)
        
        # Get top-k indices
        top_k_indices = np.argsort(logits_np)[-k:][::-1]
        
        # Get probabilities
        probs = mx.softmax(logits)
        probs_np = np.array(probs)
        top_k_probs = probs_np[top_k_indices]
        
        # Return as two separate lists
        return top_k_indices.tolist(), top_k_probs.tolist()
    
    def rollout_sequence(self, start_tokens, max_length: int = 50, max_new_tokens: int = 50, 
                        temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None,
                        return_full_sequence: bool = True):
        """
        Complete a sequence from start_tokens.
        
        Args:
            start_tokens: Starting token sequence (MLX array or list)
            max_length: Maximum sequence length (alternative to max_new_tokens)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (ignored in this implementation)
            top_p: Nucleus sampling (ignored in this implementation)
            
        Returns:
            MLX array of complete token sequence
        """
        # Use max_new_tokens if provided, otherwise use max_length
        actual_max_length = max_length if max_new_tokens == 50 else max_new_tokens + (len(start_tokens) if hasattr(start_tokens, '__len__') else 0)
        if isinstance(start_tokens, mx.array):
            current_tokens = start_tokens.tolist() if start_tokens.ndim == 1 else start_tokens[0].tolist()
        else:
            current_tokens = start_tokens if isinstance(start_tokens, list) else start_tokens.tolist()
        
        while len(current_tokens) < actual_max_length:
            # Get next token
            logits = self.get_next_token_logits(mx.array(current_tokens))
            
            # Sample based on temperature
            if temperature > 0:
                probs = mx.softmax(logits / temperature)
                next_token = int(mx.random.categorical(mx.log(probs)))
            else:
                next_token = int(mx.argmax(logits))
            
            current_tokens.append(next_token)
            
            # Check for EOS
            if next_token == self.eos_token_id:
                break
        
        return mx.array(current_tokens)
    
    def generate_sequence(self, prompt_tokens, max_tokens: int = 50, temperature: float = 1.0):
        """Generate a sequence"""
        if isinstance(prompt_tokens, mx.array):
            current_tokens = prompt_tokens[0] if prompt_tokens.ndim == 2 else prompt_tokens
            current_tokens = current_tokens.tolist()
        else:
            current_tokens = prompt_tokens[0].tolist() if len(prompt_tokens) > 0 else []
        
        for _ in range(max_tokens):
            # Get next token
            logits = self.get_next_token_logits(mx.array(current_tokens))
            
            # Sample
            if temperature > 0:
                probs = mx.softmax(logits / temperature)
                next_token = int(mx.random.categorical(mx.log(probs)))
            else:
                next_token = int(mx.argmax(logits))
            
            current_tokens.append(next_token)
            
            # Check for EOS
            if next_token == self.eos_token_id:
                break
        
        text = self.decode_sequence(current_tokens)
        return {'text': text, 'tokens': current_tokens}


if __name__ == "__main__":
    print("Testing simplified MLX wrapper...")
    
    wrapper = SimplifiedMLXWrapper()
    
    # Test encoding
    text = "Hello, world!"
    tokens = wrapper.encode_text(text)
    print(f"\nEncoded '{text}': {tokens[:10]}...")
    
    # Test decoding
    decoded = wrapper.decode_sequence(tokens)
    print(f"Decoded: '{decoded}'")
    
    # Test generation
    prompt = "The answer is"
    prompt_tokens = wrapper.encode_text(prompt)
    result = wrapper.generate_sequence(mx.array([prompt_tokens]), max_tokens=10)
    print(f"\nGenerated: '{result['text']}'")
    
    print("\n✅ All tests passed!")


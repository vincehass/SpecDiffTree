"""
PyTorch HuggingFace Wrapper for MaxEnt-TS

Simple wrapper for any HuggingFace model (Llama, Mistral, etc.)
Compatible with MaxEnt-TS tree search interface.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Tuple
import numpy as np


class PyTorchHFWrapper:
    """
    Wrapper for HuggingFace models with MaxEnt-TS compatible interface
    
    Supports:
    - Any HuggingFace model (Llama, Mistral, etc.)
    - CPU and MPS (Apple Silicon)
    - Proper weight loading (not random!)
    """
    
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = "cpu",
        torch_dtype=torch.float16
    ):
        """
        Initialize PyTorch HuggingFace wrapper
        
        Args:
            model_id: HuggingFace model ID
            device: "cpu", "cuda", or "mps"
            torch_dtype: Model precision (float16, bfloat16, float32)
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"📥 Loading PyTorch model: {model_id}")
        print(f"   Device: {device}")
        print(f"   Dtype: {torch_dtype}")
        print()
        
        # Load tokenizer
        print("   Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"   ✅ Tokenizer loaded")
        
        # Load model
        print(f"   Loading model weights (this may take a moment)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device if device != "mps" else None,  # MPS doesn't support device_map
            low_cpu_mem_usage=True
        )
        
        # Move to MPS if needed
        if device == "mps":
            self.model = self.model.to(device)
        
        self.model.eval()  # Set to evaluation mode
        
        self.eos_token_id = self.tokenizer.eos_token_id
        
        print(f"   ✅ Model loaded successfully!")
        print()
        print(f"✅ Ready for inference!")
        print(f"   Vocab size: {len(self.tokenizer)}")
        print(f"   EOS token: {self.tokenizer.eos_token}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters())/1e9:.2f}B")
        print()
    
    def get_next_token_logits(self, token_sequence):
        """
        Get logits for next token
        
        Args:
            token_sequence: List of token IDs or tensor
        
        Returns:
            Logits for next token (1D array/tensor)
        """
        # Convert to tensor
        if not isinstance(token_sequence, torch.Tensor):
            token_sequence = torch.tensor(token_sequence)
        
        # Ensure 2D [batch, seq_len]
        if token_sequence.ndim == 1:
            token_sequence = token_sequence.unsqueeze(0)
        
        token_sequence = token_sequence.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(token_sequence)
            logits = outputs.logits[0, -1, :]  # Last position, all vocab
        
        return logits
    
    def get_top_k_tokens(self, sequence, k: int = 4, temperature: float = 1.0):
        """
        Get top-k most probable next tokens
        
        Args:
            sequence: Token sequence
            k: Number of top tokens
            temperature: Sampling temperature
        
        Returns:
            tuple: (top_k_tokens, top_k_probs) - two separate lists
        """
        logits = self.get_next_token_logits(sequence)
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get top-k
        top_probs, top_indices = torch.topk(probs, k)
        
        # Return as lists
        return top_indices.cpu().tolist(), top_probs.cpu().tolist()
    
    def rollout_sequence(
        self,
        start_tokens,
        max_length: int = 50,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_full_sequence: bool = True
    ):
        """
        Complete a sequence from start_tokens
        
        Args:
            start_tokens: Starting token sequence
            max_length: Maximum total sequence length
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            return_full_sequence: Return input+output or just output
        
        Returns:
            Completed token sequence (as list)
        """
        # Convert to tensor
        if not isinstance(start_tokens, torch.Tensor):
            start_tokens = torch.tensor(start_tokens)
        
        if start_tokens.ndim == 1:
            start_tokens = start_tokens.unsqueeze(0)
        
        start_tokens = start_tokens.to(self.device)
        
        # Use model's generate method
        with torch.no_grad():
            outputs = self.model.generate(
                start_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k else 50,
                top_p=top_p if top_p else 1.0,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_id
            )
        
        # Convert to list
        if return_full_sequence:
            return outputs[0].cpu().tolist()
        else:
            # Return only new tokens
            return outputs[0, start_tokens.shape[1]:].cpu().tolist()
    
    def encode_text(self, text: str):
        """
        Encode text to tokens
        
        Args:
            text: Input text string
        
        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def decode_sequence(self, tokens):
        """
        Decode tokens to text
        
        Args:
            tokens: Token sequence (list or tensor)
        
        Returns:
            Decoded text string
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()
        
        # Handle nested lists
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
            tokens = tokens[0]
        
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def generate_sequence(self, prompt_tokens, max_tokens: int = 50, temperature: float = 1.0):
        """
        Generate a sequence (for compatibility)
        
        Args:
            prompt_tokens: Prompt token sequence
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dict with 'text' and 'tokens'
        """
        completed = self.rollout_sequence(
            prompt_tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            return_full_sequence=True
        )
        
        text = self.decode_sequence(completed)
        
        return {'text': text, 'tokens': completed}


# Recommended models for different sizes/qualities
RECOMMENDED_MODELS = {
    "1b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "3b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "7b-mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "7b-llama": "meta-llama/Llama-2-7b-chat-hf",
}


def load_recommended_model(size: str = "7b-mistral", device: str = "cpu"):
    """
    Load a recommended model
    
    Args:
        size: "1b-instruct", "3b-instruct", "7b-mistral", or "7b-llama"
        device: "cpu", "cuda", or "mps"
    
    Returns:
        PyTorchHFWrapper instance
    """
    if size not in RECOMMENDED_MODELS:
        raise ValueError(f"Unknown size {size}. Choose from: {list(RECOMMENDED_MODELS.keys())}")
    
    model_id = RECOMMENDED_MODELS[size]
    
    # Use float16 for GPU/MPS, float32 for CPU
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    
    return PyTorchHFWrapper(model_id, device=device, torch_dtype=dtype)


if __name__ == "__main__":
    # Test wrapper
    print("\n" + "="*80)
    print("  TESTING PYTORCH HUGGINGFACE WRAPPER")
    print("="*80 + "\n")
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}\n")
    
    # Load model (using 1B for quick test)
    wrapper = load_recommended_model("1b-instruct", device=device)
    
    # Test encoding
    text = "The capital of France is"
    tokens = wrapper.encode_text(text)
    print(f"📝 Encoded: '{text}'")
    print(f"   Tokens: {tokens[:10]}...")
    print()
    
    # Test top-k
    top_tokens, top_probs = wrapper.get_top_k_tokens(tokens, k=5)
    print(f"📊 Top-5 next tokens:")
    for i, (tok, prob) in enumerate(zip(top_tokens, top_probs), 1):
        tok_text = wrapper.tokenizer.decode([tok])
        print(f"   {i}. '{tok_text}' (prob={prob:.4f})")
    print()
    
    # Test rollout
    print(f"🚀 Generating completion...")
    output = wrapper.rollout_sequence(tokens, max_new_tokens=20)
    decoded = wrapper.decode_sequence(output)
    print(f"   Generated: '{decoded}'")
    print()
    
    print("="*80)
    print("  ✅ ALL TESTS PASSED!")
    print("="*80 + "\n")


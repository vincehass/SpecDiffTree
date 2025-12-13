"""
OpenTSLM Wrapper for MaxEnt-TS (Token-Level Tree Search)

Wraps pre-trained OpenTSLM models to provide the base policy p_θ for tree search.

Mathematical Framework from MaximumEntropyTreeSearchforAutoregressive.md:
- p_θ(x_{t+1}|x_{≤t}): Autoregressive next-token distribution
- Used for Expansion and Rollout in tree search
"""

import sys
import os
from pathlib import Path
import torch
from typing import Optional, List, Dict, Tuple

# Add OpenTSLM src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.model.llm.OpenTSLM import OpenTSLM
from transformers import AutoTokenizer


class OpenTSLMWrapper:
    """
    Wrapper for pre-trained OpenTSLM models
    
    Provides unified interface for MaxEnt-TS tree search:
    - Load from HuggingFace
    - Get next-token probabilities p_θ(x_{t+1}|x_{≤t})
    - Complete rollouts
    - Decode final sequences
    
    Supports all 5 OpenTSLM stages
    """
    
    def __init__(
        self,
        repo_id: str,
        device: str = "cpu",
        enable_lora: bool = False
    ):
        """
        Initialize OpenTSLM wrapper
        
        Args:
            repo_id: HuggingFace model ID (e.g., "OpenTSLM/llama-3.2-1b-tsqa-sp")
            device: Device to load model on
            enable_lora: Whether to enable LoRA adapters
        """
        self.repo_id = repo_id
        self.device = device
        self.enable_lora = enable_lora
        
        print(f"📥 Loading OpenTSLM from {repo_id}...")
        
        # Load pre-trained model
        self.model = OpenTSLM.load_pretrained(
            repo_id,
            device=device,
            enable_lora=enable_lora
        )
        self.model.eval()  # Set to evaluation mode
        
        # Get tokenizer
        self.tokenizer = self.model.get_tokenizer()
        self.eos_token = self.model.get_eos_token()
        self.eos_token_id = self.tokenizer.encode(self.eos_token, add_special_tokens=False)[0]
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {device}")
        print(f"   EOS token: {self.eos_token}")
        print(f"   Vocab size: {len(self.tokenizer)}")
    
    def get_next_token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get logits for next token given current sequence
        
        This is p_θ(x_{t+1}|x_{≤t}) in the mathematical framework.
        
        Args:
            input_ids: Token sequence [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
        
        Returns:
            logits: Next-token logits [batch, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Get logits for last position (next token)
            logits = outputs.logits[:, -1, :]
        
        return logits
    
    def get_next_token_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get probabilities for next token
        
        Args:
            input_ids: Token sequence [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            temperature: Sampling temperature
        
        Returns:
            probs: Next-token probabilities [batch, vocab_size]
        """
        logits = self.get_next_token_logits(input_ids, attention_mask)
        probs = torch.softmax(logits / temperature, dim=-1)
        return probs
    
    def sample_next_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample next token from p_θ(x_{t+1}|x_{≤t})
        
        Used for EXPANSION phase of tree search.
        
        Args:
            input_ids: Current token sequence
            attention_mask: Attention mask
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling
        
        Returns:
            next_token: Sampled token ID [batch]
            log_prob: Log probability of sampled token [batch]
        """
        logits = self.get_next_token_logits(input_ids, attention_mask)
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits.scatter_(1, indices_to_remove, float('-inf'))
        
        # Sample token
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # Get log probability
        log_prob = torch.log(probs.gather(1, next_token.unsqueeze(-1))).squeeze(-1)
        
        return next_token, log_prob
    
    def get_top_k_tokens(
        self,
        input_ids: torch.Tensor,
        k: int = 5,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k most likely next tokens
        
        Used for EXPANSION to create multiple children.
        
        Args:
            input_ids: Current token sequence [batch, seq_len]
            k: Number of top tokens to return
            temperature: Sampling temperature
        
        Returns:
            top_tokens: Top-k token IDs [batch, k]
            top_probs: Top-k probabilities [batch, k]
        """
        logits = self.get_next_token_logits(input_ids)
        probs = torch.softmax(logits / temperature, dim=-1)
        
        top_probs, top_tokens = torch.topk(probs, k, dim=-1)
        
        return top_tokens, top_probs
    
    def rollout_sequence(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_full_sequence: bool = True
    ) -> torch.Tensor:
        """
        Complete sequence from current prefix using base model
        
        Used for ROLLOUT phase: Complete trajectory from x_{≤t} to x.
        
        Args:
            input_ids: Current token prefix [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            return_full_sequence: If True, return input+output; if False, only new tokens
        
        Returns:
            sequence: Completed token sequence
        """
        with torch.no_grad():
            # Use model's generate method
            outputs = self.model.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_id
            )
        
        if return_full_sequence:
            return outputs
        else:
            # Return only new tokens
            return outputs[:, input_ids.shape[1]:]
    
    def decode_sequence(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs to text
        
        Args:
            token_ids: Token sequence [batch, seq_len]
        
        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to token IDs
        
        Args:
            text: Input text string
        
        Returns:
            token_ids: Token sequence [1, seq_len]
        """
        encoded = self.tokenizer.encode(text, return_tensors='pt')
        return encoded.to(self.device)
    
    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, tuple]:
        """
        Forward pass with KV cache for efficiency
        
        Args:
            input_ids: Token IDs
            past_key_values: Cached key-value pairs
            use_cache: Whether to return cache
        
        Returns:
            logits: Next-token logits
            past_key_values: Updated cache
        """
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache
            )
        
        return outputs.logits[:, -1, :], outputs.past_key_values if use_cache else None


# Pre-defined model IDs for all 5 stages
STAGE_MODELS = {
    1: "OpenTSLM/llama-3.2-1b-tsqa-sp",     # TSQA (Multiple-Choice QA)
    2: "OpenTSLM/llama-3.2-1b-m4-sp",       # M4 (Captioning)
    3: "OpenTSLM/llama-3.2-1b-har-sp",      # HAR CoT
    4: "OpenTSLM/llama-3.2-1b-sleep-sp",    # Sleep CoT
    5: "OpenTSLM/llama-3.2-1b-ecg-sp",      # ECG QA CoT
}


def load_stage_model(
    stage: int,
    device: str = "cpu",
    enable_lora: bool = False
) -> OpenTSLMWrapper:
    """
    Load pre-trained model for a specific stage
    
    Args:
        stage: Stage number (1-5)
        device: Device to load on
        enable_lora: Enable LoRA adapters
    
    Returns:
        Wrapped OpenTSLM model
    """
    if stage not in STAGE_MODELS:
        raise ValueError(f"Invalid stage {stage}. Must be 1-5.")
    
    repo_id = STAGE_MODELS[stage]
    return OpenTSLMWrapper(repo_id, device=device, enable_lora=enable_lora)


if __name__ == "__main__":
    # Test wrapper
    print("Testing OpenTSLM Wrapper...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Stage 1 model
    wrapper = load_stage_model(stage=1, device=device)
    
    # Test encoding
    text = "What is the time series pattern?"
    tokens = wrapper.encode_text(text)
    print(f"\n📝 Encoded text:")
    print(f"   Input: {text}")
    print(f"   Tokens: {tokens.shape}")
    
    # Test next-token prediction
    print(f"\n🎲 Getting next-token probabilities...")
    probs = wrapper.get_next_token_probs(tokens)
    print(f"   Probs shape: {probs.shape}")
    
    # Test top-k tokens
    top_tokens, top_probs = wrapper.get_top_k_tokens(tokens, k=5)
    print(f"\n📊 Top-5 next tokens:")
    for i in range(5):
        token_id = top_tokens[0, i].item()
        prob = top_probs[0, i].item()
        token_text = wrapper.tokenizer.decode([token_id])
        print(f"   {i+1}. '{token_text}' (prob={prob:.4f})")
    
    # Test rollout
    print(f"\n🚀 Testing rollout...")
    output = wrapper.rollout_sequence(tokens, max_new_tokens=20)
    decoded = wrapper.decode_sequence(output)
    print(f"   Generated: {decoded[0]}")
    
    print("\n✅ Wrapper tests passed!")


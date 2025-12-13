"""
Local OpenTSLM Model Loader for S-ADT

Loads locally trained OpenTSLM models for inference with MaxEnt-TS.
This uses the existing OpenTSLMSP class from the codebase.
"""

import sys
from pathlib import Path
import torch
from typing import Optional, Tuple, List

# Add OpenTSLM src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.model.llm.OpenTSLMSP import OpenTSLMSP


class LocalOpenTSLMWrapper:
    """
    Wrapper for locally trained OpenTSLM models
    
    Uses the existing OpenTSLMSP class for inference with S-ADT.
    """
    
    def __init__(
        self,
        model: OpenTSLMSP,
        device: str = "cpu"
    ):
        """
        Initialize wrapper with OpenTSLMSP model
        
        Args:
            model: OpenTSLMSP instance
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.tokenizer = model.tokenizer
        
        # Get EOS token
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        
        print(f"✅ Model loaded on {device}")
        print(f"   Vocab size: {len(self.tokenizer)}")
        print(f"   EOS token: {self.eos_token}")
    
    @classmethod
    def from_llm_id(
        cls,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cpu",
        checkpoint_path: Optional[str] = None
    ) -> 'LocalOpenTSLMWrapper':
        """
        Create model from base LLM ID
        
        Args:
            llm_id: Base LLM ID (e.g., "meta-llama/Llama-3.2-1B")
            device: Device to load on
            checkpoint_path: Optional path to trained checkpoint
        
        Returns:
            Wrapped model ready for inference
        """
        print(f"📥 Loading OpenTSLMSP model...")
        print(f"   Base LLM: {llm_id}")
        print(f"   Device: {device}")
        
        # Create OpenTSLMSP model
        model = OpenTSLMSP(
            llm_id=llm_id,
            device=device
        )
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            print(f"   Loading checkpoint: {checkpoint_path}")
            model.load_from_file(checkpoint_path)
        
        model.llm.eval()
        
        return cls(model, device)
    
    def get_tokenizer(self):
        """Get tokenizer"""
        return self.tokenizer
    
    def get_eos_token(self):
        """Get EOS token string"""
        return self.eos_token
    
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
        
        Returns:
            Token IDs [1, seq_len]
        """
        encoded = self.tokenizer.encode(text, return_tensors='pt')
        return encoded.to(self.device)
    
    def decode_sequence(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode token IDs to text
        
        Args:
            token_ids: Token sequence [batch, seq_len]
        
        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    
    def get_next_token_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get logits for next token
        
        Args:
            input_ids: Token sequence [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
        
        Returns:
            Logits for next token [batch, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model.llm(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Get logits for last position
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
            input_ids: Token sequence
            attention_mask: Attention mask
            temperature: Sampling temperature
        
        Returns:
            Probabilities [batch, vocab_size]
        """
        logits = self.get_next_token_logits(input_ids, attention_mask)
        probs = torch.softmax(logits / temperature, dim=-1)
        return probs
    
    def get_top_k_tokens(
        self,
        input_ids: torch.Tensor,
        k: int = 5,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k most likely next tokens
        
        Args:
            input_ids: Token sequence
            k: Number of top tokens
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
        Complete sequence using model generation
        
        Args:
            input_ids: Starting sequence
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            return_full_sequence: Return input+output or only new tokens
        
        Returns:
            Generated sequence
        """
        with torch.no_grad():
            outputs = self.model.llm.generate(
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
            return outputs[:, input_ids.shape[1]:]


def load_base_model(
    llm_id: str = "meta-llama/Llama-3.2-1B",
    device: str = "cpu",
    checkpoint_path: Optional[str] = None
) -> LocalOpenTSLMWrapper:
    """
    Load OpenTSLM model from base LLM
    
    Args:
        llm_id: Base LLM ID
        device: Device to load on
        checkpoint_path: Optional trained checkpoint
    
    Returns:
        Loaded model wrapper
    """
    return LocalOpenTSLMWrapper.from_llm_id(
        llm_id=llm_id,
        device=device,
        checkpoint_path=checkpoint_path
    )


if __name__ == "__main__":
    # Test loading
    print("Testing Local OpenTSLM Model Loading...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Try loading base model
    print("\n" + "="*70)
    print("Testing OpenTSLMSP Model Loading")
    print("="*70)
    
    try:
        # Use smaller model for testing
        model = load_base_model(
            llm_id="meta-llama/Llama-3.2-1B",
            device=device
        )
        
        # Test encoding
        test_text = "What is the pattern in this time series?"
        print(f"\n✅ Test 1: Encoding text...")
        tokens = model.encode_text(test_text)
        print(f"   Text: {test_text}")
        print(f"   Tokens shape: {tokens.shape}")
        
        # Test next-token prediction
        print(f"\n✅ Test 2: Getting next-token logits...")
        logits = model.get_next_token_logits(tokens)
        print(f"   Logits shape: {logits.shape}")
        
        # Test top-k
        print(f"\n✅ Test 3: Getting top-5 tokens...")
        top_tokens, top_probs = model.get_top_k_tokens(tokens, k=5)
        print(f"   Top 5 next tokens:")
        for i in range(5):
            tok_id = top_tokens[0, i].item()
            prob = top_probs[0, i].item()
            tok_str = model.tokenizer.decode([tok_id])
            print(f"      {i+1}. '{tok_str}' (prob={prob:.4f})")
        
        # Test generation
        print(f"\n✅ Test 4: Generating sequence...")
        output = model.rollout_sequence(tokens, max_new_tokens=20)
        decoded = model.decode_sequence(output)
        print(f"   Generated: {decoded[0]}")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nModel is ready for S-ADT inference! 🎉")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Note: This test requires:")
        print("   1. Internet connection (to download base LLM)")
        print("   2. HuggingFace credentials (for gated models)")
        print("   3. Sufficient disk space (~2GB)")


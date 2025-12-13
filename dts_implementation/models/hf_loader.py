"""
Simplified HuggingFace Model Loader for S-ADT

Loads pre-trained OpenTSLM models from HuggingFace for inference with MaxEnt-TS.

Available models:
- OpenTSLM/llama-3.2-1b-tsqa-sp (Stage 1: TSQA)
- OpenTSLM/llama-3.2-1b-m4-sp (Stage 2: M4)  
- OpenTSLM/llama-3.2-1b-har-sp (Stage 3: HAR)
- OpenTSLM/llama-3.2-1b-sleep-sp (Stage 4: Sleep)
- OpenTSLM/llama-3.2-1b-ecg-sp (Stage 5: ECG)
"""

import sys
from pathlib import Path
import torch
from typing import Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add OpenTSLM src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class SimplifiedOpenTSLMWrapper:
    """
    Simplified wrapper for pre-trained OpenTSLM models from HuggingFace
    
    This is a lightweight interface specifically for S-ADT inference,
    bypassing the full OpenTSLMSP initialization.
    """
    
    def __init__(
        self,
        llm_model,
        tokenizer,
        device: str = "cpu"
    ):
        """
        Initialize wrapper with loaded model and tokenizer
        
        Args:
            llm_model: Pre-loaded HuggingFace model
            tokenizer: Pre-loaded tokenizer
            device: Device to run on
        """
        self.model = llm_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Get EOS token
        self.eos_token = tokenizer.eos_token
        self.eos_token_id = tokenizer.eos_token_id
        
        print(f"✅ Model loaded on {device}")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   EOS token: {self.eos_token}")
    
    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        device: str = "cpu",
        torch_dtype = torch.float32
    ) -> 'SimplifiedOpenTSLMWrapper':
        """
        Load pre-trained model from HuggingFace
        
        Args:
            repo_id: HuggingFace model ID (e.g., "OpenTSLM/llama-3.2-1b-tsqa-sp")
            device: Device to load on
            torch_dtype: Data type for model weights
        
        Returns:
            Wrapped model ready for inference
        """
        print(f"📥 Loading model from HuggingFace: {repo_id}")
        
        # Load tokenizer
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        print(f"   Loading model (dtype={torch_dtype})...")
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            torch_dtype=torch_dtype,
            device_map={"": device} if device != "cpu" else None,
            low_cpu_mem_usage=True
        )
        
        return cls(model, tokenizer, device)
    
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
            outputs = self.model(
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
            outputs = self.model.generate(
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


# Model registry
STAGE_MODELS = {
    1: "OpenTSLM/llama-3.2-1b-tsqa-sp",
    2: "OpenTSLM/llama-3.2-1b-m4-sp",
    3: "OpenTSLM/llama-3.2-1b-har-sp",
    4: "OpenTSLM/llama-3.2-1b-sleep-sp",
    5: "OpenTSLM/llama-3.2-1b-ecg-sp",
}


def load_stage_model(
    stage: int,
    device: str = "cpu",
    torch_dtype = torch.float32
) -> SimplifiedOpenTSLMWrapper:
    """
    Load pre-trained model for a specific stage
    
    Args:
        stage: Stage number (1-5)
        device: Device to load on
        torch_dtype: Model data type
    
    Returns:
        Loaded model wrapper
    """
    if stage not in STAGE_MODELS:
        raise ValueError(f"Invalid stage {stage}. Must be 1-5.")
    
    repo_id = STAGE_MODELS[stage]
    return SimplifiedOpenTSLMWrapper.from_pretrained(
        repo_id,
        device=device,
        torch_dtype=torch_dtype
    )


if __name__ == "__main__":
    # Test loading
    print("Testing HuggingFace model loading...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Try loading Stage 1 model
    print("\n" + "="*70)
    print("Testing Stage 1 (TSQA) Model Loading")
    print("="*70)
    
    try:
        model = load_stage_model(stage=1, device=device)
        
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
        print("   1. Internet connection (to download from HuggingFace)")
        print("   2. HuggingFace credentials (if model is gated)")
        print("   3. Sufficient disk space (~2GB per model)")


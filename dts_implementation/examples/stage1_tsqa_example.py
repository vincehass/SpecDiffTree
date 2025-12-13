"""
End-to-End Example: MaxEnt-TS on Stage 1 (TSQA)

Demonstrates the complete S-ADT pipeline:
1. Load pre-trained OpenTSLM (Stage 1: TSQA)
2. Load TSQA dataset
3. Set context PSD from historical data
4. Run MaxEnt-TS tree search
5. Evaluate results

This is a minimal working example showing how all components integrate.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dts_implementation.models.opentslm_wrapper import load_stage_model
from dts_implementation.rewards.spectral_reward import create_spectral_reward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig
from src.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from src.dataset.multiple_choice_qa import TSQADataset


def main():
    """Run full MaxEnt-TS pipeline on TSQA"""
    
    print("="*70)
    print("  MaxEnt-TS for OpenTSLM - Stage 1 (TSQA) Example")
    print("="*70)
    
    # ==================================================================
    # 1. SETUP
    # ==================================================================
    
    print("\n📦 Step 1: Loading model and dataset...")
    
    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"   Using device: {device}")
    
    # Load OpenTSLM Stage 1 model
    print(f"\n   Loading OpenTSLM Stage 1 (TSQA)...")
    model = load_stage_model(stage=1, device=device)
    
    # Load TSQA dataset (just a few samples for demo)
    print(f"\n   Loading TSQA dataset...")
    
    # Create prompt template
    prompt_template = TextTimeSeriesPrompt()
    eos_token = model.eos_token
    
    # Load dataset
    dataset = TSQADataset(
        split='test',
        prompt_template=prompt_template,
        EOS_TOKEN=eos_token
    )
    
    print(f"   ✅ Dataset loaded: {len(dataset)} samples")
    
    # ==================================================================
    # 2. SELECT A SAMPLE
    # ==================================================================
    
    print("\n📝 Step 2: Selecting sample question...")
    
    # Get first sample
    sample_idx = 0
    sample = dataset[sample_idx]
    
    # Extract components
    time_series = sample['time_series']  # [channels, length]
    question = sample['question']
    choices = sample['choices']
    answer_idx = sample['answer']
    
    print(f"\n   Question: {question}")
    print(f"   Choices:")
    for i, choice in enumerate(choices):
        marker = "✓" if i == answer_idx else " "
        print(f"      [{marker}] {chr(65+i)}. {choice}")
    
    print(f"\n   Time series shape: {time_series.shape}")
    
    # ==================================================================
    # 3. CREATE SPECTRAL REWARD
    # ==================================================================
    
    print("\n🎯 Step 3: Setting up spectral reward...")
    
    # Create spectral reward computer
    spectral_reward = create_spectral_reward(
        task='tsqa',
        gamma=1.0,  # Spectral penalty weight
        sampling_rate=100.0,  # Adjust based on your data
        spectral_metric='l1',
        normalize=True
    )
    
    # Set context from historical data
    # In real scenario, this would be from historical time series
    # For now, use the question's time series as context
    context_ts = time_series.numpy()
    if context_ts.ndim == 2:
        context_ts = context_ts[0]  # Take first channel
    
    spectral_reward.set_context(context_ts)
    
    print(f"   ✅ Spectral reward configured (γ={spectral_reward.gamma})")
    
    # ==================================================================
    # 4. PREPARE PROMPT
    # ==================================================================
    
    print("\n✍️  Step 4: Creating prompt...")
    
    # Build prompt using OpenTSLM's format
    # Format: "[TS] <time_series_values> [Question] <question> [Choices] A. ... B. ... [Answer]"
    
    # Simplify time series to string (this is how OpenTSLM expects it)
    # In practice, OpenTSLM encodes time series through its encoder
    ts_str = f"<time_series_{sample_idx}>"  # Placeholder
    
    prompt_text = f"{ts_str}\n\nQuestion: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt_text += f"{chr(65+i)}. {choice}\n"
    prompt_text += "\nAnswer:"
    
    print(f"\n   Prompt length: {len(prompt_text)} chars")
    print(f"   Preview: {prompt_text[:200]}...")
    
    # Encode prompt
    prompt_tokens = model.encode_text(prompt_text)
    print(f"   Encoded to {prompt_tokens.shape[-1]} tokens")
    
    # ==================================================================
    # 5. CONFIGURE SEARCH
    # ==================================================================
    
    print("\n⚙️  Step 5: Configuring MaxEnt-TS...")
    
    config = MaxEntTSConfig(
        num_rollouts=20,  # Small for demo (use 100+ for real)
        temperature=1.0,  # λ in equations
        max_seq_length=50,  # Short answers for MCQ
        expansion_k=4,  # Try 4 token choices (A, B, C, D)
        expansion_temperature=1.0,
        rollout_temperature=0.7,  # Slightly greedy for completion
        rollout_top_k=50,
        use_uct=False,  # Use Boltzmann selection
        gamma=1.0,  # Match spectral reward
        verbose=True
    )
    
    print(f"   Configuration:")
    print(f"      Rollouts: {config.num_rollouts}")
    print(f"      Temperature: {config.temperature}")
    print(f"      Expansion top-k: {config.expansion_k}")
    
    # ==================================================================
    # 6. RUN SEARCH
    # ==================================================================
    
    print("\n🔍 Step 6: Running MaxEnt-TS search...")
    print(f"{'='*70}\n")
    
    # Initialize searcher
    searcher = MaxEntTS(
        model=model,
        reward=spectral_reward,
        config=config
    )
    
    # Run search
    results = searcher.search(
        prompt_tokens=prompt_tokens,
        ground_truth={
            'answer': answer_idx,
            'choices': choices
        }
    )
    
    print(f"\n{'='*70}")
    
    # ==================================================================
    # 7. ANALYZE RESULTS
    # ==================================================================
    
    print("\n📊 Step 7: Results Analysis")
    print("="*70)
    
    print(f"\n🏆 Best Sequence:")
    print(f"   Tokens: {results['best_sequence'].shape}")
    print(f"   Text: {results['best_text']}")
    print(f"   Reward: {results['best_reward']:.4f}")
    
    print(f"\n🌳 Tree Statistics:")
    stats = results['tree_stats']
    print(f"   Total nodes explored: {stats['total_nodes']}")
    print(f"   Maximum depth: {stats['max_depth']}")
    print(f"   Avg branching factor: {stats['avg_branching_factor']:.2f}")
    print(f"   Total rollouts: {stats['rollouts']}")
    
    # Parse answer from text
    predicted_answer = extract_answer(results['best_text'])
    correct = (predicted_answer == chr(65 + answer_idx))
    
    print(f"\n✅ Evaluation:")
    print(f"   Ground truth: {chr(65 + answer_idx)}")
    print(f"   Predicted: {predicted_answer}")
    print(f"   Correct: {'✓' if correct else '✗'}")
    
    # ==================================================================
    # 8. COMPARISON WITH GREEDY
    # ==================================================================
    
    print("\n🔄 Step 8: Comparison with greedy baseline...")
    
    # Generate with standard greedy decoding
    greedy_output = model.rollout_sequence(
        prompt_tokens,
        max_new_tokens=config.max_seq_length,
        temperature=0.1,  # Near-greedy
        return_full_sequence=True
    )
    greedy_text = model.decode_sequence(greedy_output)[0]
    greedy_answer = extract_answer(greedy_text)
    
    print(f"\n   Greedy output: {greedy_text}")
    print(f"   Greedy answer: {greedy_answer}")
    
    print(f"\n📈 Comparison:")
    print(f"   MaxEnt-TS: {predicted_answer} {'✓' if correct else '✗'}")
    print(f"   Greedy: {greedy_answer} {'✓' if (greedy_answer == chr(65 + answer_idx)) else '✗'}")
    
    print("\n" + "="*70)
    print("✅ Example complete!")
    print("="*70)


def extract_answer(text: str) -> str:
    """
    Extract answer (A, B, C, or D) from model output
    
    Args:
        text: Generated text
    
    Returns:
        Answer letter (A-D) or '?' if not found
    """
    text = text.upper()
    
    # Look for patterns like "A", "B)", "ANSWER: C", etc.
    for letter in ['A', 'B', 'C', 'D']:
        if f"ANSWER: {letter}" in text or f"ANSWER {letter}" in text:
            return letter
        if text.strip().endswith(letter):
            return letter
        if f"{letter}." in text or f"{letter})" in text:
            return letter
    
    # Default: return first letter found
    for letter in ['A', 'B', 'C', 'D']:
        if letter in text:
            return letter
    
    return '?'


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run example
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: This example requires:")
        print("  1. OpenTSLM model from HuggingFace")
        print("  2. TSQA dataset")
        print("  3. All dependencies installed")


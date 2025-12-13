"""
Stage 1 (TSQA) Evaluation with Pre-trained Checkpoint

Tests S-ADT with real pre-trained OpenTSLM model on TSQA dataset.
Compares MaxEnt-TS vs Greedy baseline on actual time series questions.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dts_implementation.models.local_loader import load_base_model
from dts_implementation.rewards.spectral_reward import create_spectral_reward
from dts_implementation.search.maxent_ts import MaxEntTS, MaxEntTSConfig
from time_series_datasets.TSQADataset import TSQADataset
from prompt.text_time_series_prompt import TextTimeSeriesPrompt


def extract_answer(text):
    """Extract answer (A, B, C, or D) from model output"""
    text = text.upper()
    
    # Look for patterns
    for letter in ['A', 'B', 'C', 'D']:
        if f"ANSWER: {letter}" in text or f"ANSWER {letter}" in text:
            return letter
        if text.strip().endswith(letter):
            return letter
        if f"{letter}." in text or f"{letter})" in text:
            # Check if it's the first occurrence and near the end
            idx = text.find(letter)
            if idx > len(text) * 0.5:  # In second half
                return letter
    
    # Default: return first letter found
    for letter in ['A', 'B', 'C', 'D']:
        if letter in text:
            return letter
    
    return '?'


def test_sample(model, reward, config, sample, sample_idx):
    """Test S-ADT on a single TSQA sample"""
    
    # Extract components
    time_series = sample['time_series']  # [channels, length]
    question = sample['question']
    choices = sample['choices']
    answer_idx = sample['answer']
    answer_letter = chr(65 + answer_idx)  # A, B, C, D
    
    print(f"\n{'='*70}")
    print(f"Sample {sample_idx}")
    print(f"{'='*70}")
    print(f"Question: {question}")
    print(f"Choices:")
    for i, choice in enumerate(choices):
        marker = "✓" if i == answer_idx else " "
        print(f"  [{marker}] {chr(65+i)}. {choice}")
    print(f"Time series shape: {time_series.shape}")
    
    # Build prompt (simplified - real OpenTSLM uses more complex formatting)
    prompt_text = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt_text += f"{chr(65+i)}. {choice}\n"
    prompt_text += "\nAnswer:"
    
    # Encode prompt
    prompt_tokens = model.encode_text(prompt_text)
    print(f"\nPrompt tokens: {prompt_tokens.shape[-1]}")
    
    # Update spectral reward context with this time series
    ts_data = time_series.numpy() if isinstance(time_series, torch.Tensor) else time_series
    if ts_data.ndim == 2:
        ts_data = ts_data[0]  # Take first channel
    reward.set_context(ts_data)
    
    # Run MaxEnt-TS
    print(f"\n🌳 Running MaxEnt-TS...")
    searcher = MaxEntTS(model, reward, config)
    
    start_time = time.time()
    try:
        results = searcher.search(prompt_tokens)
        maxent_time = time.time() - start_time
        maxent_text = results['best_text']
        maxent_answer = extract_answer(maxent_text)
        maxent_correct = (maxent_answer == answer_letter)
    except Exception as e:
        print(f"   ❌ MaxEnt-TS failed: {e}")
        maxent_text = "ERROR"
        maxent_answer = "?"
        maxent_correct = False
        maxent_time = 0
        results = {'tree_stats': {'total_nodes': 0, 'max_depth': 0, 'avg_branching_factor': 0}}
    
    # Run Greedy
    print(f"\n🔄 Running Greedy baseline...")
    start_time = time.time()
    try:
        greedy_output = model.rollout_sequence(
            prompt_tokens,
            max_new_tokens=50,
            temperature=0.1,
            return_full_sequence=True
        )
        greedy_time = time.time() - start_time
        greedy_text = model.decode_sequence(greedy_output)[0]
        greedy_answer = extract_answer(greedy_text)
        greedy_correct = (greedy_answer == answer_letter)
    except Exception as e:
        print(f"   ❌ Greedy failed: {e}")
        greedy_text = "ERROR"
        greedy_answer = "?"
        greedy_correct = False
        greedy_time = 0
    
    # Results
    print(f"\n📊 RESULTS:")
    print(f"\n   Ground Truth: {answer_letter}")
    
    print(f"\n   MaxEnt-TS ({maxent_time:.1f}s):")
    print(f"      Answer: {maxent_answer} {'✅' if maxent_correct else '❌'}")
    print(f"      Text: {maxent_text[:150]}...")
    if results['tree_stats']['total_nodes'] > 0:
        print(f"      Nodes: {results['tree_stats']['total_nodes']}, "
              f"Depth: {results['tree_stats']['max_depth']}, "
              f"Branching: {results['tree_stats']['avg_branching_factor']:.2f}")
    
    print(f"\n   Greedy ({greedy_time:.1f}s):")
    print(f"      Answer: {greedy_answer} {'✅' if greedy_correct else '❌'}")
    print(f"      Text: {greedy_text[:150]}...")
    
    return {
        'question': question,
        'ground_truth': answer_letter,
        'maxent_answer': maxent_answer,
        'maxent_correct': maxent_correct,
        'maxent_time': maxent_time,
        'greedy_answer': greedy_answer,
        'greedy_correct': greedy_correct,
        'greedy_time': greedy_time,
        'tree_stats': results['tree_stats'] if 'tree_stats' in results else {}
    }


def main():
    """Run Stage 1 (TSQA) evaluation"""
    
    print("="*70)
    print("  Stage 1 (TSQA) Evaluation with Pre-trained Model")
    print("="*70)
    
    # ==================================================================
    # SETUP
    # ==================================================================
    
    print("\n📦 Step 1: Loading pre-trained model...")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Device: {device}")
    
    # Find downloaded checkpoint
    checkpoint_path = "./checkpoints_pretrained/models--OpenTSLM--llama-3.2-1b-tsqa-sp/snapshots/1904441f0d87c458d6e9376de04ada5f4c9ab5b7/best_model-llama_3_2_1b-tsqa.pt"
    
    print(f"   Loading checkpoint: {Path(checkpoint_path).name}")
    
    # Load model with checkpoint
    model = load_base_model(
        llm_id="meta-llama/Llama-3.2-1B",
        device=device,
        checkpoint_path=checkpoint_path
    )
    
    print("   ✅ Model loaded with pre-trained weights!")
    
    # ==================================================================
    # LOAD TSQA DATASET
    # ==================================================================
    
    print("\n📊 Step 2: Loading TSQA dataset...")
    
    # Create prompt template
    prompt_template = TextTimeSeriesPrompt()
    eos_token = model.get_eos_token()
    
    # Load test split
    try:
        dataset = TSQADataset(
            split='test',
            prompt_template=prompt_template,
            EOS_TOKEN=eos_token
        )
        print(f"   ✅ Loaded {len(dataset)} test samples")
    except Exception as e:
        print(f"   ⚠️  Could not load TSQA dataset: {e}")
        print("   Using synthetic samples instead...")
        dataset = None
    
    # ==================================================================
    # SETUP REWARD & CONFIG
    # ==================================================================
    
    print("\n🎯 Step 3: Setting up S-ADT...")
    
    # Spectral reward
    reward = create_spectral_reward(
        task='tsqa',
        gamma=1.0,
        sampling_rate=100.0,
        normalize=True
    )
    
    # Configure search
    config = MaxEntTSConfig(
        num_rollouts=10,  # Moderate for speed
        temperature=1.0,
        max_seq_length=50,
        expansion_k=4,
        rollout_temperature=0.7,
        use_uct=False,
        verbose=False
    )
    
    print(f"   Config: {config.num_rollouts} rollouts, k={config.expansion_k}")
    
    # ==================================================================
    # RUN TESTS
    # ==================================================================
    
    print(f"\n🔍 Step 4: Testing on TSQA samples...")
    print(f"{'='*70}\n")
    
    # Test on a few samples
    num_samples = 5 if dataset else 0
    results = []
    
    if dataset:
        for i in range(min(num_samples, len(dataset))):
            try:
                sample = dataset[i]
                result = test_sample(model, reward, config, sample, i+1)
                results.append(result)
            except Exception as e:
                print(f"\n❌ Error on sample {i+1}: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("⚠️  No dataset available - skipping tests")
        print("   To test with real data, ensure TSQA dataset is downloaded")
    
    # ==================================================================
    # SUMMARY
    # ==================================================================
    
    if results:
        print(f"\n\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        maxent_correct = sum(1 for r in results if r['maxent_correct'])
        greedy_correct = sum(1 for r in results if r['greedy_correct'])
        
        print(f"\n📊 Accuracy:")
        print(f"   MaxEnt-TS: {maxent_correct}/{len(results)} = {100*maxent_correct/len(results):.1f}%")
        print(f"   Greedy:    {greedy_correct}/{len(results)} = {100*greedy_correct/len(results):.1f}%")
        
        avg_maxent_time = np.mean([r['maxent_time'] for r in results])
        avg_greedy_time = np.mean([r['greedy_time'] for r in results])
        
        print(f"\n⏱️  Average Time:")
        print(f"   MaxEnt-TS: {avg_maxent_time:.1f}s")
        print(f"   Greedy:    {avg_greedy_time:.1f}s")
        
        total_nodes = sum(r['tree_stats'].get('total_nodes', 0) for r in results)
        print(f"\n🌳 Total Tree Nodes Explored: {total_nodes}")
        
        print(f"\n💡 Key Insights:")
        if maxent_correct > greedy_correct:
            print(f"   ✅ MaxEnt-TS is {maxent_correct - greedy_correct} more accurate!")
        elif maxent_correct == greedy_correct:
            print(f"   ⚖️  Both methods achieved same accuracy")
        else:
            print(f"   ⚠️  Greedy performed better on this small sample")
        
        print(f"   • MaxEnt-TS explored {total_nodes} paths")
        print(f"   • Greedy only explored {len(results)} paths (1 per question)")
        print(f"   • {total_nodes/len(results):.0f}x more exploration with MaxEnt-TS")
    
    print(f"\n{'='*70}")
    print("✅ Stage 1 (TSQA) evaluation complete!")
    print(f"{'='*70}")
    
    print(f"\n📝 Next Steps:")
    print(f"   • Download checkpoints for Stages 2-5")
    print(f"   • Run same evaluation on each stage")
    print(f"   • Compare S-ADT improvements across stages")
    print(f"   • Evaluate spectral fidelity metrics")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


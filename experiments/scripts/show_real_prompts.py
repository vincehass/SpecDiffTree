"""
Show REAL evaluation prompts from M4 and HAR datasets
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from time_series_datasets.m4.M4QADataset import M4QADataset
from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def show_prompt(dataset_name, sample):
    """Display a sample with its full prompt and expected answer"""
    
    print(f"Dataset: {dataset_name}")
    print("-" * 80)
    
    # Get input and output (try different key names)
    if 'input' in sample:
        prompt = sample['input']
        answer = sample['output']
    elif 'text' in sample:
        prompt = sample['text']
        answer = sample.get('answer', 'N/A')
    else:
        print(f"Available keys: {sample.keys()}")
        return
    
    print(f"\n📝 FULL PROMPT TO MODEL ({len(prompt)} characters):")
    print("─" * 80)
    print(prompt)
    print("─" * 80)
    
    print(f"\n💡 EXPECTED OUTPUT / GROUND TRUTH ({len(answer)} characters):")
    print("─" * 80)
    print(answer)
    print("─" * 80)
    
    print(f"\n📊 STATISTICS:")
    print(f"  • Prompt length: {len(prompt)} chars, ~{len(prompt.split())} words")
    print(f"  • Answer length: {len(answer)} chars, ~{len(answer.split())} words")
    
    # Check for time series data in prompt
    if any(x in prompt for x in ['mean', 'std', 'time series', 'data points']):
        print(f"  • Contains: Time series statistics ✅")
    if any(x in prompt for x in ['accelerometer', 'x-axis', 'y-axis', 'z-axis']):
        print(f"  • Contains: Accelerometer data ✅")


def main():
    print_section("🔍 REAL EVALUATION PROMPTS FROM DATASETS")
    
    print("Loading datasets (this may take a moment)...\n")
    
    # Load M4 dataset
    print("Loading M4QADataset...")
    try:
        m4_dataset = M4QADataset(
            split='test',
            EOS_TOKEN='</s>',  # Dummy token, just for loading
            format_sample_str=False
        )
        print(f"✅ Loaded: {len(m4_dataset)} samples\n")
        
        # Show 2 M4 samples
        print_section("SAMPLE 1: M4 TIME SERIES CAPTIONING")
        show_prompt("M4 Competition Dataset", m4_dataset[0])
        
        print_section("SAMPLE 2: M4 TIME SERIES CAPTIONING")
        show_prompt("M4 Competition Dataset", m4_dataset[1])
        
    except Exception as e:
        print(f"❌ Error loading M4: {e}\n")
    
    # Load HAR dataset
    print_section("LOADING HAR DATASET")
    print("Loading HARCoTQADataset...")
    try:
        har_dataset = HARCoTQADataset(
            split='test',
            EOS_TOKEN='</s>',  # Dummy token, just for loading
            format_sample_str=False
        )
        print(f"✅ Loaded: {len(har_dataset)} samples\n")
        
        # Show 2 HAR samples
        print_section("SAMPLE 3: HAR ACTIVITY RECOGNITION (CoT)")
        show_prompt("HAR with Chain-of-Thought", har_dataset[0])
        
        print_section("SAMPLE 4: HAR ACTIVITY RECOGNITION (CoT)")
        show_prompt("HAR with Chain-of-Thought", har_dataset[1])
        
    except Exception as e:
        print(f"❌ Error loading HAR: {e}\n")
    
    print_section("✅ PROMPT EXAMPLES COMPLETE")
    
    print("Key Observations:")
    print("  1. Prompts contain REAL time series data")
    print("  2. Ground truth provides expected reasoning/descriptions")
    print("  3. These are the ACTUAL inputs to MaxEnt-TS tree search")
    print("  4. Model must understand data AND generate coherent analysis")
    print()


if __name__ == "__main__":
    main()


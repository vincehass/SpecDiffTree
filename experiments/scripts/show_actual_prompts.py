"""
Show ACTUAL complete evaluation prompts (what the model really sees)
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

def show_complete_prompt(dataset_name, sample_idx, sample):
    """Display the complete prompt as the model sees it"""
    
    print(f"Dataset: {dataset_name}")
    print(f"Sample Index: {sample_idx}")
    print("-" * 80)
    
    # Construct full prompt from parts
    if 'pre_prompt' in sample and 'time_series_text' in sample:
        pre = sample.get('pre_prompt', '')
        ts_text = sample.get('time_series_text', '')
        post = sample.get('post_prompt', '')
        
        # Convert ts_text to string if it's a list
        if isinstance(ts_text, list):
            ts_text = ' '.join(str(x) for x in ts_text)
        
        full_prompt = str(pre) + str(ts_text) + str(post)
    else:
        full_prompt = str(sample)
    
    answer = sample.get('answer', 'N/A')
    
    print(f"\n📝 FULL INPUT PROMPT TO MODEL:")
    print(f"   ({len(full_prompt)} characters, {len(full_prompt.split())} words)")
    print("─" * 80)
    # Show first 1000 chars
    if len(full_prompt) > 1000:
        print(full_prompt[:1000])
        print(f"\n... [truncated, showing first 1000 of {len(full_prompt)} chars] ...")
    else:
        print(full_prompt)
    print("─" * 80)
    
    print(f"\n💡 EXPECTED ANSWER/OUTPUT:")
    print(f"   ({len(answer)} characters, {len(answer.split())} words)")
    print("─" * 80)
    # Show first 500 chars of answer
    if len(answer) > 500:
        print(answer[:500])
        print(f"\n... [truncated, showing first 500 of {len(answer)} chars] ...")
    else:
        print(answer)
    print("─" * 80)
    
    # Show breakdown
    if 'pre_prompt' in sample:
        print(f"\n📊 PROMPT STRUCTURE:")
        print(f"  • Pre-prompt: {len(sample.get('pre_prompt', ''))} chars")
        print(f"  • Time series data: {len(sample.get('time_series_text', ''))} chars")
        print(f"  • Post-prompt: {len(sample.get('post_prompt', ''))} chars")
        print(f"  • Total: {len(full_prompt)} chars")


def main():
    print_section("🔍 ACTUAL EVALUATION PROMPTS (What Model Sees)")
    
    # Load M4 dataset
    print("Loading M4QADataset...")
    try:
        m4_dataset = M4QADataset(
            split='test',
            EOS_TOKEN='</s>',
            format_sample_str=False
        )
        print(f"✅ Loaded: {len(m4_dataset)} samples\n")
        
        # Show 2 M4 samples
        print_section("EXAMPLE 1: M4 TIME SERIES CAPTIONING")
        show_complete_prompt("M4 Competition Dataset", 0, m4_dataset[0])
        
        print_section("EXAMPLE 2: M4 TIME SERIES CAPTIONING")
        show_complete_prompt("M4 Competition Dataset", 5, m4_dataset[5])
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Load HAR dataset
    print_section("LOADING HAR DATASET")
    print("Loading HARCoTQADataset...")
    try:
        har_dataset = HARCoTQADataset(
            split='test',
            EOS_TOKEN='</s>',
            format_sample_str=False
        )
        print(f"✅ Loaded: {len(har_dataset)} samples\n")
        
        # Show 2 HAR samples
        print_section("EXAMPLE 3: HAR ACTIVITY RECOGNITION (Chain-of-Thought)")
        show_complete_prompt("HAR with CoT", 0, har_dataset[0])
        
        print_section("EXAMPLE 4: HAR ACTIVITY RECOGNITION (Chain-of-Thought)")
        show_complete_prompt("HAR with CoT", 10, har_dataset[10])
        
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
    
    print_section("✅ REAL PROMPT EXAMPLES COMPLETE")
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("  1. These are the EXACT prompts fed to MaxEnt-TS tree search")
    print("  2. Prompts contain REAL numerical time series data")
    print("  3. Expected answers show detailed reasoning/analysis")
    print("  4. Model must:")
    print("     - Understand time series patterns")
    print("     - Generate coherent analytical text")
    print("     - Follow chain-of-thought reasoning")
    print()


if __name__ == "__main__":
    main()


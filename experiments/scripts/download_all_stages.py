#!/usr/bin/env python3
"""
Download all OpenTSLM pre-trained models for Stages 2-5
"""

import os
from huggingface_hub import snapshot_download
from pathlib import Path

# Model IDs for each stage (using recommended 1B + SP models)
MODELS = {
    "stage2_m4": "OpenTSLM/llama-3.2-1b-m4-sp",
    "stage3_har": "OpenTSLM/llama-3.2-1b-har-sp",
    "stage4_sleep": "OpenTSLM/llama-3.2-1b-sleep-sp",
    "stage5_ecg": "OpenTSLM/llama-3.2-1b-ecg-sp",
}

def download_model(stage_name, model_id, base_dir="checkpoints"):
    """Download a model from HuggingFace"""
    print(f"\n{'='*70}")
    print(f"📥 Downloading {stage_name}: {model_id}")
    print(f"{'='*70}\n")
    
    local_dir = Path(base_dir) / stage_name
    local_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\n✅ {stage_name} downloaded successfully to {local_dir}")
        
        # Check downloaded size
        total_size = sum(f.stat().st_size for f in local_dir.rglob('*') if f.is_file())
        print(f"   Size: {total_size / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {stage_name}: {e}")
        return False

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     Downloading OpenTSLM Pre-trained Models (Stages 2-5)        ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")
    
    # Check if Stage 1 is already downloaded
    stage1_path = Path("checkpoints/opentslm_stage1_pretrained")
    if stage1_path.exists():
        print("✅ Stage 1 (TSQA): Already downloaded")
        print(f"   Location: {stage1_path}")
        size1 = sum(f.stat().st_size for f in stage1_path.rglob('*') if f.is_file())
        print(f"   Size: {size1 / (1024*1024):.1f} MB\n")
    else:
        print("⚠️  Stage 1 (TSQA): Not found\n")
    
    # Download Stages 2-5
    results = {}
    for stage_name, model_id in MODELS.items():
        success = download_model(stage_name, model_id)
        results[stage_name] = success
    
    # Summary
    print("\n" + "="*70)
    print("📊 DOWNLOAD SUMMARY")
    print("="*70 + "\n")
    
    print("Status by Stage:")
    print(f"  Stage 1 (TSQA):  ✅ Already downloaded")
    for stage_name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        stage_num = stage_name.split('_')[0].replace('stage', '')
        stage_task = stage_name.split('_')[1].upper()
        print(f"  Stage {stage_num} ({stage_task:5s}): {status}")
    
    successful_downloads = sum(results.values())
    total_stages = len(results)
    
    print(f"\n✅ Downloaded: {successful_downloads}/{total_stages} models")
    
    if successful_downloads == total_stages:
        print("\n🎉 All models downloaded successfully!")
        print("\nNext step: Run S-ADT evaluation on all stages")
    else:
        print("\n⚠️  Some downloads failed. Check errors above.")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()


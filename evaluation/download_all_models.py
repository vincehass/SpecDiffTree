"""
Download all pre-trained OpenTSLM models for Stages 1-5.
"""

import os
from huggingface_hub import snapshot_download
from pathlib import Path

# Model IDs for all stages
MODELS = {
    "stage1": "OpenTSLM/llama-3.2-1b-tsqa-sp",
    "stage2": "OpenTSLM/llama-3.2-1b-m4-sp",
    "stage3": "OpenTSLM/llama-3.2-1b-har-sp",
    "stage4": "OpenTSLM/llama-3.2-1b-sleep-sp",
    "stage5": "OpenTSLM/llama-3.2-1b-ecg-sp",
}

def download_model(stage_name: str, repo_id: str, base_dir: str = "checkpoints"):
    """Download a single model."""
    local_dir = Path(base_dir) / stage_name
    
    print(f"\n{'='*70}")
    print(f"📥 Downloading {stage_name.upper()}")
    print(f"{'='*70}")
    print(f"Repository: {repo_id}")
    print(f"Local path: {local_dir}")
    
    try:
        # Check if already downloaded
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"✅ {stage_name} already exists, skipping...")
            return True
        
        # Download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            resume_download=True,
        )
        print(f"✅ Downloaded {stage_name} successfully!")
        return True
    
    except Exception as e:
        print(f"❌ Error downloading {stage_name}: {e}")
        return False

def main():
    """Download all models."""
    print("\n" + "="*70)
    print("🎯 OpenTSLM Model Downloader")
    print("="*70)
    print("\nDownloading pre-trained models for all 5 stages...")
    
    results = {}
    for stage_name, repo_id in MODELS.items():
        results[stage_name] = download_model(stage_name, repo_id)
    
    # Summary
    print("\n" + "="*70)
    print("📊 DOWNLOAD SUMMARY")
    print("="*70)
    
    for stage_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{stage_name.upper():10} | {status}")
    
    total = len(results)
    successful = sum(results.values())
    print(f"\nTotal: {successful}/{total} models downloaded successfully")
    
    if successful == total:
        print("\n🎉 All models ready for evaluation!")
    else:
        print("\n⚠️  Some models failed to download. Check errors above.")
    
    print("="*70)

if __name__ == "__main__":
    main()


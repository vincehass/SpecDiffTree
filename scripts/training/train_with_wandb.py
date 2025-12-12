#!/usr/bin/env python3
"""
SpecDiffTree Training Script with W&B Integration
Spectral Diffusion Tree - Spectral-Regularized Amortized Diffusion Trees
Supports hyperparameter configuration and experiment tracking
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Import after path setup
import curriculum_learning
from curriculum_learning import CurriculumTrainer

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Warning: wandb not installed. Run: pip install wandb")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def print_experiment_info(config: Dict[str, Any]):
    """Print detailed experiment information."""
    print("\n" + "="*80)
    print("🧪 EXPERIMENT CONFIGURATION")
    print("="*80)
    
    # Experiment info
    if 'experiment' in config:
        print("\n📋 Experiment Details:")
        print(f"   Name: {config['experiment'].get('name', 'N/A')}")
        print(f"   Tags: {', '.join(config['experiment'].get('tags', []))}")
        print(f"   Notes: {config['experiment'].get('notes', 'N/A')}")
    
    # Dataset info
    if 'dataset' in config:
        print("\n📊 Dataset Information:")
        print(f"   Name: {config['dataset'].get('name', 'N/A')}")
        print(f"   Task: {config['dataset'].get('task', 'N/A')}")
        print(f"   Train samples: {config['dataset'].get('num_samples_train', 'N/A'):,}")
        print(f"   Val samples: {config['dataset'].get('num_samples_val', 'N/A'):,}")
        print(f"   Test samples: {config['dataset'].get('num_samples_test', 'N/A'):,}")
        print(f"   Description: {config['dataset'].get('description', 'N/A')}")
    
    # Model info
    if 'model' in config:
        print("\n🤖 Model Configuration:")
        print(f"   Type: {config['model'].get('type', 'N/A')}")
        print(f"   LLM: {config['model'].get('llm_id', 'N/A')}")
        print(f"   Gradient Checkpointing: {config['model'].get('gradient_checkpointing', False)}")
        if 'lora' in config['model']:
            print(f"   LoRA Enabled: {config['model']['lora'].get('enabled', False)}")
    
    # Training hyperparameters
    if 'training' in config:
        print("\n⚙️  Training Hyperparameters:")
        print(f"   Epochs: {config['training'].get('num_epochs', 'N/A')}")
        print(f"   Batch Size: {config['training'].get('batch_size', 'N/A')}")
        print(f"   Learning Rate (Encoder): {config['training'].get('lr_encoder', 'N/A'):.2e}")
        print(f"   Learning Rate (Projector): {config['training'].get('lr_projector', 'N/A'):.2e}")
        print(f"   Weight Decay: {config['training'].get('weight_decay', 'N/A'):.2e}")
        print(f"   Gradient Clipping: {config['training'].get('grad_clip_norm', 'N/A')}")
        print(f"   Warmup Fraction: {config['training'].get('warmup_fraction', 'N/A')}")
        print(f"   Early Stopping Patience: {config['training'].get('early_stopping_patience', 'N/A')}")
        print(f"   Patch Size: {config['training'].get('patch_size', 'N/A')}")
    
    # Device info
    if 'device' in config:
        print("\n💻 Device Configuration:")
        print(f"   Device Type: {config['device'].get('type', 'N/A')}")
    
    print("\n" + "="*80 + "\n")


def init_wandb(config: Dict[str, Any], wandb_entity: str = None, wandb_project: str = "opentslm"):
    """Initialize Weights & Biases logging."""
    if not WANDB_AVAILABLE:
        print("⚠️  Skipping W&B initialization (not installed)")
        return None
    
    # Flatten config for W&B
    wandb_config = {
        "experiment_name": config.get('experiment', {}).get('name', 'unnamed'),
        "model_type": config.get('model', {}).get('type', 'unknown'),
        "llm_id": config.get('model', {}).get('llm_id', 'unknown'),
        "dataset": config.get('dataset', {}).get('name', 'unknown'),
        "num_epochs": config.get('training', {}).get('num_epochs', 0),
        "batch_size": config.get('training', {}).get('batch_size', 0),
        "lr_encoder": config.get('training', {}).get('lr_encoder', 0),
        "lr_projector": config.get('training', {}).get('lr_projector', 0),
        "lr_base": config.get('training', {}).get('lr_base', 0),
        "weight_decay": config.get('training', {}).get('weight_decay', 0),
        "device": config.get('device', {}).get('type', 'unknown'),
        "gradient_checkpointing": config.get('model', {}).get('gradient_checkpointing', False),
    }
    
    # Initialize W&B
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=config.get('experiment', {}).get('name', 'unnamed'),
        tags=config.get('experiment', {}).get('tags', []),
        notes=config.get('experiment', {}).get('notes', ''),
        config=wandb_config,
    )
    
    print(f"✅ W&B initialized: {run.url}")
    return run


def main():
    parser = argparse.ArgumentParser(
        description="Train OpenTSLM with W&B Integration and Config Management"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="nadhirvincenthassen",
        help="W&B entity (username or team)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="opentslm",
        help="W&B project name"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Run evaluation only (no training)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"📂 Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Print experiment information
    print_experiment_info(config)
    
    # Initialize W&B
    wandb_run = None
    if not args.no_wandb and WANDB_AVAILABLE:
        try:
            wandb_run = init_wandb(config, args.wandb_entity, args.wandb_project)
        except Exception as e:
            print(f"⚠️  Failed to initialize W&B: {e}")
            print("   Continuing without W&B logging...")
    
    # Extract parameters from config
    model_type = config.get('model', {}).get('type', 'OpenTSLMSP')
    llm_id = config.get('model', {}).get('llm_id', 'meta-llama/Llama-3.2-1B')
    device = config.get('device', {}).get('type', 'mps')
    gradient_checkpointing = config.get('model', {}).get('gradient_checkpointing', False)
    batch_size = config.get('training', {}).get('batch_size', 4)
    
    # Determine which stage to run
    # For now, we'll assume stage1_mcq based on config name
    # You can make this more sophisticated
    config_name = Path(args.config).stem
    if 'stage1' in config_name or 'mcq' in config_name:
        stages = ['stage1_mcq']
    elif 'stage2' in config_name or 'caption' in config_name:
        stages = ['stage2_captioning']
    elif 'stage3' in config_name or 'cot' in config_name:
        stages = ['stage3_cot']
    elif 'stage4' in config_name or 'sleep' in config_name:
        stages = ['stage4_sleep_cot']
    elif 'stage5' in config_name or 'ecg' in config_name:
        stages = ['stage5_ecg_cot']
    else:
        stages = ['stage1_mcq']  # Default
    
    print(f"🎯 Training Stage: {', '.join(stages)}")
    print(f"🔄 Starting training...\n")
    
    # Initialize trainer
    trainer = CurriculumTrainer(
        model_type=model_type,
        device=device,
        gradient_checkpointing=gradient_checkpointing,
        llm_id=llm_id,
    )
    
    # Run training
    try:
        results = trainer.run_curriculum(
            stages=stages,
            batch_size=batch_size,
            eval_only=args.eval_only
        )
        
        # Log results to W&B
        if wandb_run is not None:
            for stage, metrics in results.items():
                # Log metrics with stage prefix
                wandb_metrics = {f"{stage}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
                wandb.log(wandb_metrics)
            
            print(f"\n✅ Results logged to W&B: {wandb_run.url}")
        
        # Print final results
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETED")
        print("="*80)
        for stage, metrics in results.items():
            print(f"\n{stage.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
        print("\n" + "="*80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        if wandb_run is not None:
            wandb.finish(exit_code=1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if wandb_run is not None:
            wandb.finish(exit_code=1)
        raise
    finally:
        # Finish W&B run
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()


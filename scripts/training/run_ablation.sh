#!/bin/bash

#############################################################################
# SpecDiffTree Ablation Study Script
# Spectral Diffusion Tree - Spectral-Regularized Amortized Diffusion Trees
# 
# This script runs multiple training experiments with different hyperparameters
# for ablation studies. Each experiment is logged to W&B for comparison.
#
# Usage:
#   ./run_ablation.sh [options]
#
# Options:
#   --config FILE    Path to ablation config file (default: configs/ablation_configs.yaml)
#   --experiment N   Run only experiment N (default: run all)
#   --wandb_entity   W&B entity name (default: nadhirvincenthassen)
#   --no_wandb       Disable W&B logging
#   --dry_run        Print commands without executing
#############################################################################

set -e  # Exit on error

# Default values
ABLATION_CONFIG="configs/m3max_ablation.yaml"
WANDB_ENTITY="nadhirvincenthassen"
WANDB_PROJECT="specdifftree"
EXPERIMENT_ID=""
NO_WANDB=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            ABLATION_CONFIG="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        --wandb_entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --no_wandb)
            NO_WANDB=true
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       OpenTSLM Ablation Study Runner${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Ablation Config: $ABLATION_CONFIG"
echo "  W&B Entity: $WANDB_ENTITY"
echo "  W&B Project: $WANDB_PROJECT"
echo "  Dry Run: $DRY_RUN"
echo ""

# Check if config file exists
if [ ! -f "$ABLATION_CONFIG" ]; then
    echo -e "${RED}Error: Ablation config file not found: $ABLATION_CONFIG${NC}"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Parse YAML config and extract experiments
# This is a simplified version - you may need to install yq for complex parsing
echo -e "${YELLOW}Parsing ablation experiments...${NC}"

# Create temporary Python script to parse YAML
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << 'PYTHON_EOF'
import sys
import yaml

config_file = sys.argv[1]
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

experiments = config.get('ablation_experiments', [])
for i, exp in enumerate(experiments):
    print(f"EXP_{i}_NAME={exp.get('name', 'unnamed')}")
    print(f"EXP_{i}_STAGE={exp.get('stage', 'stage1_mcq')}")
    print(f"EXP_{i}_MODEL={exp.get('model_type', 'OpenTSLMSP')}")
    print(f"EXP_{i}_LLM={exp.get('llm_id', 'meta-llama/Llama-3.2-1B')}")
    print(f"EXP_{i}_BATCH={exp.get('batch_size', 4)}")
    print(f"EXP_{i}_LR_ENC={exp.get('lr_encoder', '2.0e-4')}")
    print(f"EXP_{i}_LR_PROJ={exp.get('lr_projector', '1.0e-4')}")
    print(f"EXP_{i}_LR_BASE={exp.get('lr_base', '2.0e-4')}")
    print(f"EXP_{i}_EPOCHS={exp.get('num_epochs', 30)}")
    print(f"EXP_{i}_DEVICE={exp.get('device', 'mps')}")
    print(f"EXP_{i}_GRAD_CKPT={exp.get('gradient_checkpointing', True)}")
    print(f"EXP_{i}_DESC={exp.get('description', 'No description')}")
    print(f"---EXP_{i}_END---")
PYTHON_EOF

# Execute Python script and capture output
EXPERIMENTS_DATA=$(python "$TEMP_SCRIPT" "$ABLATION_CONFIG")
rm "$TEMP_SCRIPT"

# Count experiments
NUM_EXPERIMENTS=$(echo "$EXPERIMENTS_DATA" | grep -c "---EXP_.*_END---")
echo -e "${GREEN}Found $NUM_EXPERIMENTS experiments${NC}"
echo ""

# Function to create temporary config for an experiment
create_temp_config() {
    local exp_id=$1
    local config_file=$2
    
    # Extract experiment data
    eval $(echo "$EXPERIMENTS_DATA" | grep "^EXP_${exp_id}_")
    
    local name_var="EXP_${exp_id}_NAME"
    local stage_var="EXP_${exp_id}_STAGE"
    local model_var="EXP_${exp_id}_MODEL"
    local llm_var="EXP_${exp_id}_LLM"
    local batch_var="EXP_${exp_id}_BATCH"
    local lr_enc_var="EXP_${exp_id}_LR_ENC"
    local lr_proj_var="EXP_${exp_id}_LR_PROJ"
    local lr_base_var="EXP_${exp_id}_LR_BASE"
    local epochs_var="EXP_${exp_id}_EPOCHS"
    local device_var="EXP_${exp_id}_DEVICE"
    local grad_ckpt_var="EXP_${exp_id}_GRAD_CKPT"
    local desc_var="EXP_${exp_id}_DESC"
    
    # Create temporary config file
    cat > "$config_file" << EOF
experiment:
  name: "${!name_var}"
  tags: ["ablation", "${!stage_var}", "experiment_${exp_id}"]
  notes: "${!desc_var}"

dataset:
  name: "TSQA"
  task: "Multiple Choice Question Answering"
  num_samples_train: 6300
  num_samples_val: 630
  num_samples_test: 700

model:
  type: "${!model_var}"
  llm_id: "${!llm_var}"
  gradient_checkpointing: ${!grad_ckpt_var}

training:
  num_epochs: ${!epochs_var}
  batch_size: ${!batch_var}
  lr_encoder: ${!lr_enc_var}
  lr_projector: ${!lr_proj_var}
  lr_base: ${!lr_base_var}
  weight_decay: 1.0e-2
  grad_clip_norm: 1.0
  warmup_fraction: 0.03
  early_stopping_patience: 5
  patch_size: 4

device:
  type: "${!device_var}"
EOF
}

# Function to run a single experiment
run_experiment() {
    local exp_id=$1
    
    # Extract experiment name
    eval $(echo "$EXPERIMENTS_DATA" | grep "^EXP_${exp_id}_NAME=")
    local name_var="EXP_${exp_id}_NAME"
    local desc_var="EXP_${exp_id}_DESC"
    
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}Experiment $((exp_id + 1))/$NUM_EXPERIMENTS: ${!name_var}${NC}"
    echo -e "${BLUE}Description: ${!desc_var}${NC}"
    echo -e "${BLUE}============================================================${NC}"
    
    # Create temporary config
    TEMP_CONFIG=$(mktemp --suffix=.yaml)
    create_temp_config $exp_id "$TEMP_CONFIG"
    
    echo -e "${YELLOW}Generated config:${NC}"
    cat "$TEMP_CONFIG"
    echo ""
    
    # Build command
    CMD="python train_with_wandb.py --config $TEMP_CONFIG --wandb_entity $WANDB_ENTITY --wandb_project $WANDB_PROJECT"
    
    if [ "$NO_WANDB" = true ]; then
        CMD="$CMD --no_wandb"
    fi
    
    echo -e "${GREEN}Command: $CMD${NC}"
    echo ""
    
    # Execute or dry run
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would execute: $CMD${NC}"
    else
        # Execute command
        if eval "$CMD"; then
            echo -e "${GREEN}✅ Experiment completed successfully${NC}"
        else
            echo -e "${RED}❌ Experiment failed${NC}"
            rm "$TEMP_CONFIG"
            return 1
        fi
    fi
    
    # Cleanup
    rm "$TEMP_CONFIG"
    echo ""
}

# Run experiments
if [ -n "$EXPERIMENT_ID" ]; then
    # Run specific experiment
    echo -e "${YELLOW}Running experiment $EXPERIMENT_ID only${NC}"
    run_experiment $((EXPERIMENT_ID - 1))
else
    # Run all experiments
    echo -e "${YELLOW}Running all $NUM_EXPERIMENTS experiments${NC}"
    echo ""
    
    for ((i=0; i<NUM_EXPERIMENTS; i++)); do
        run_experiment $i
        
        # Add delay between experiments
        if [ $i -lt $((NUM_EXPERIMENTS - 1)) ]; then
            echo -e "${YELLOW}Waiting 10 seconds before next experiment...${NC}"
            sleep 10
        fi
    done
fi

echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}🎉 Ablation study completed!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "View results at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"


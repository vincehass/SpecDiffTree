#!/bin/bash

#############################################################################
# SpecDiffTree Single Training Run with W&B Integration
# Spectral Diffusion Tree - Spectral-Regularized Amortized Diffusion Trees
# 
# This script runs a single training experiment with detailed logging
# and W&B integration for experiment tracking.
#
# Usage:
#   ./train_single.sh [options]
#
# Examples:
#   # Train Stage 1 with default config
#   ./train_single.sh --config configs/stage1_mcq.yaml
#
#   # Train with custom W&B settings
#   ./train_single.sh --config configs/stage1_mcq.yaml --wandb_entity myusername
#
#   # Evaluation only (no training)
#   ./train_single.sh --config configs/stage1_mcq.yaml --eval_only
#############################################################################

set -e  # Exit on error

# Default values
CONFIG_FILE=""
WANDB_ENTITY="nadhirvincenthassen"
WANDB_PROJECT="specdifftree"
NO_WANDB=false
EVAL_ONLY=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}     SpecDiffTree Training with W&B Integration${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

# Print usage
print_usage() {
    echo "Usage: $0 --config CONFIG_FILE [options]"
    echo ""
    echo "Required:"
    echo "  --config FILE        Path to configuration YAML file"
    echo ""
    echo "Optional:"
    echo "  --wandb_entity NAME  W&B entity (default: nadhirvincenthassen)"
    echo "  --wandb_project NAME W&B project (default: opentslm)"
    echo "  --no_wandb           Disable W&B logging"
    echo "  --eval_only          Run evaluation only (no training)"
    echo "  --verbose            Enable verbose logging"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --config configs/stage1_mcq.yaml"
    echo "  $0 --config configs/stage1_mcq.yaml --eval_only"
    echo "  $0 --config configs/stage1_mcq.yaml --no_wandb --verbose"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
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
        --eval_only)
            EVAL_ONLY=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            print_header
            echo ""
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo ""
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: --config is required${NC}"
    echo ""
    print_usage
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Print header
print_header
echo ""

# Print configuration
echo -e "${CYAN}📋 Configuration:${NC}"
echo "  Config File: $CONFIG_FILE"
echo "  W&B Entity: $WANDB_ENTITY"
echo "  W&B Project: $WANDB_PROJECT"
echo "  W&B Logging: $([ "$NO_WANDB" = true ] && echo "Disabled" || echo "Enabled")"
echo "  Mode: $([ "$EVAL_ONLY" = true ] && echo "Evaluation Only" || echo "Training + Evaluation")"
echo "  Verbose: $([ "$VERBOSE" = true ] && echo "Enabled" || echo "Disabled")"
echo ""

# Check Python environment
echo -e "${YELLOW}🔍 Checking Python environment...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
echo "  Python: $PYTHON_VERSION"

# Check if in virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "  ${GREEN}✓ Virtual environment: $VIRTUAL_ENV${NC}"
else
    echo -e "  ${YELLOW}⚠ Not in a virtual environment${NC}"
    echo -e "  ${YELLOW}  Consider activating: source opentslm_env/bin/activate${NC}"
fi
echo ""

# Check required packages
echo -e "${YELLOW}🔍 Checking required packages...${NC}"
PACKAGES_OK=true

if ! python -c "import torch" 2>/dev/null; then
    echo -e "  ${RED}✗ PyTorch not found${NC}"
    PACKAGES_OK=false
else
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "  ${GREEN}✓ PyTorch: $TORCH_VERSION${NC}"
fi

if ! python -c "import transformers" 2>/dev/null; then
    echo -e "  ${RED}✗ Transformers not found${NC}"
    PACKAGES_OK=false
else
    echo -e "  ${GREEN}✓ Transformers installed${NC}"
fi

if ! python -c "import yaml" 2>/dev/null; then
    echo -e "  ${RED}✗ PyYAML not found${NC}"
    PACKAGES_OK=false
else
    echo -e "  ${GREEN}✓ PyYAML installed${NC}"
fi

if [ "$NO_WANDB" = false ]; then
    if ! python -c "import wandb" 2>/dev/null; then
        echo -e "  ${YELLOW}⚠ W&B not found (will continue without W&B)${NC}"
        echo -e "  ${YELLOW}  Install with: pip install wandb${NC}"
    else
        echo -e "  ${GREEN}✓ W&B installed${NC}"
    fi
fi

if [ "$PACKAGES_OK" = false ]; then
    echo ""
    echo -e "${RED}Error: Required packages are missing${NC}"
    echo -e "${YELLOW}Install with: pip install -r requirements.txt${NC}"
    exit 1
fi
echo ""

# Check GPU availability
echo -e "${YELLOW}🔍 Checking GPU availability...${NC}"
GPU_INFO=$(python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())" 2>&1)
echo "  $GPU_INFO"
echo ""

# Display config contents
echo -e "${CYAN}📄 Configuration file contents:${NC}"
echo -e "${YELLOW}────────────────────────────────────────────────────────────${NC}"
cat "$CONFIG_FILE"
echo -e "${YELLOW}────────────────────────────────────────────────────────────${NC}"
echo ""

# Build command
CMD="python train_with_wandb.py"
CMD="$CMD --config \"$CONFIG_FILE\""
CMD="$CMD --wandb_entity \"$WANDB_ENTITY\""
CMD="$CMD --wandb_project \"$WANDB_PROJECT\""

if [ "$NO_WANDB" = true ]; then
    CMD="$CMD --no_wandb"
fi

if [ "$EVAL_ONLY" = true ]; then
    CMD="$CMD --eval_only"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

# Confirm execution
echo -e "${GREEN}🚀 Ready to start training${NC}"
echo ""
echo -e "${CYAN}Command to execute:${NC}"
echo "  $CMD"
echo ""
echo -e "${YELLOW}Press Ctrl+C to cancel, or wait 5 seconds to continue...${NC}"
sleep 5
echo ""

# Execute command
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}▶ Starting training...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

START_TIME=$(date +%s)

# Execute with proper error handling
if eval "$CMD"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}Training Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
    echo ""
    
    if [ "$NO_WANDB" = false ]; then
        echo -e "${CYAN}📊 View results at:${NC}"
        echo "  https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
        echo ""
    fi
    
    echo -e "${CYAN}📁 Results saved to:${NC}"
    echo "  results/"
    echo ""
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}❌ Training failed!${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${RED}Duration before failure: ${DURATION}s${NC}"
    echo ""
    exit 1
fi


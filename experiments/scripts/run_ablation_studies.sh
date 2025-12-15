#!/bin/bash

################################################################################
# Comprehensive Ablation Studies & Hyperparameter Sweep
################################################################################
#
# This script runs systematic evaluations across:
# - All methods (Greedy, MCTS, DTS, DTS*, MaxEnt-TS)
# - Multiple hyperparameters (num_rollouts, expansion_k, temperature)
# - 250 samples per configuration
# - Full metrics logged to WandB
#
################################################################################

set -e  # Exit on error

# Configuration
NUM_SAMPLES=250
DATASET="m4"  # or "har"
DEVICE="mps"  # or "cuda" or "cpu"
EPOCHS=3

# Output directory
RESULTS_DIR="results/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/ablation.log"
echo "Starting ablation studies at $(date)" | tee "$LOG_FILE"
echo "Results will be saved to: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

################################################################################
# Study 1: Baseline Comparison (Fixed Hyperparameters)
################################################################################

echo "=============================================================================="  | tee -a "$LOG_FILE"
echo "STUDY 1: Baseline Method Comparison"  | tee -a "$LOG_FILE"
echo "=============================================================================="  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"

ROLLOUTS=20
EXPANSION_K=4
TEMPERATURE=1.0

for METHOD in greedy mcts dts dts_star maxent_ts; do
    echo "Running $METHOD..." | tee -a "$LOG_FILE"
    python comprehensive_evaluation.py \
        --method $METHOD \
        --num_samples $NUM_SAMPLES \
        --num_rollouts $ROLLOUTS \
        --expansion_k $EXPANSION_K \
        --temperature $TEMPERATURE \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --device $DEVICE \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "✅ $METHOD complete" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

################################################################################
# Study 2: Number of Rollouts Ablation (Scalability)
################################################################################

echo "=============================================================================="  | tee -a "$LOG_FILE"
echo "STUDY 2: Number of Rollouts Ablation (Scalability)"  | tee -a "$LOG_FILE"
echo "=============================================================================="  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"

EXPANSION_K=4
TEMPERATURE=1.0

for ROLLOUTS in 5 10 20 50 100; do
    for METHOD in mcts dts maxent_ts; do
        echo "Running $METHOD with $ROLLOUTS rollouts..." | tee -a "$LOG_FILE"
        python comprehensive_evaluation.py \
            --method $METHOD \
            --num_samples $NUM_SAMPLES \
            --num_rollouts $ROLLOUTS \
            --expansion_k $EXPANSION_K \
            --temperature $TEMPERATURE \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --device $DEVICE \
            2>&1 | tee -a "$LOG_FILE"
        
        echo "✅ $METHOD (rollouts=$ROLLOUTS) complete" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

################################################################################
# Study 3: Expansion K Ablation (Breadth vs Depth)
################################################################################

echo "=============================================================================="  | tee -a "$LOG_FILE"
echo "STUDY 3: Expansion K Ablation (Breadth vs Depth)"  | tee -a "$LOG_FILE"
echo "=============================================================================="  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"

ROLLOUTS=20
TEMPERATURE=1.0

for EXPANSION_K in 2 3 4 5 8; do
    for METHOD in mcts dts maxent_ts; do
        echo "Running $METHOD with expansion_k=$EXPANSION_K..." | tee -a "$LOG_FILE"
        python comprehensive_evaluation.py \
            --method $METHOD \
            --num_samples $NUM_SAMPLES \
            --num_rollouts $ROLLOUTS \
            --expansion_k $EXPANSION_K \
            --temperature $TEMPERATURE \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --device $DEVICE \
            2>&1 | tee -a "$LOG_FILE"
        
        echo "✅ $METHOD (expansion_k=$EXPANSION_K) complete" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

################################################################################
# Study 4: Temperature Ablation (Exploration vs Exploitation)
################################################################################

echo "=============================================================================="  | tee -a "$LOG_FILE"
echo "STUDY 4: Temperature Ablation (Exploration vs Exploitation)"  | tee -a "$LOG_FILE"
echo "=============================================================================="  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"

ROLLOUTS=20
EXPANSION_K=4

for TEMPERATURE in 0.5 0.8 1.0 1.5 2.0; do
    for METHOD in mcts dts maxent_ts; do
        echo "Running $METHOD with temperature=$TEMPERATURE..." | tee -a "$LOG_FILE"
        python comprehensive_evaluation.py \
            --method $METHOD \
            --num_samples $NUM_SAMPLES \
            --num_rollouts $ROLLOUTS \
            --expansion_k $EXPANSION_K \
            --temperature $TEMPERATURE \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --device $DEVICE \
            2>&1 | tee -a "$LOG_FILE"
        
        echo "✅ $METHOD (temperature=$TEMPERATURE) complete" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

################################################################################
# Study 5: Dataset Comparison (M4 vs HAR)
################################################################################

echo "=============================================================================="  | tee -a "$LOG_FILE"
echo "STUDY 5: Dataset Comparison (M4 vs HAR)"  | tee -a "$LOG_FILE"
echo "=============================================================================="  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"

ROLLOUTS=20
EXPANSION_K=4
TEMPERATURE=1.0

for DATASET in m4 har; do
    for METHOD in greedy mcts dts maxent_ts; do
        echo "Running $METHOD on $DATASET..." | tee -a "$LOG_FILE"
        python comprehensive_evaluation.py \
            --method $METHOD \
            --num_samples $NUM_SAMPLES \
            --num_rollouts $ROLLOUTS \
            --expansion_k $EXPANSION_K \
            --temperature $TEMPERATURE \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --device $DEVICE \
            2>&1 | tee -a "$LOG_FILE"
        
        echo "✅ $METHOD on $DATASET complete" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

################################################################################
# Generate Figures and Analysis
################################################################################

echo "=============================================================================="  | tee -a "$LOG_FILE"
echo "Generating figures and analysis..."  | tee -a "$LOG_FILE"
echo "=============================================================================="  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"

python generate_ablation_figures.py --results_dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"

################################################################################
# Summary
################################################################################

echo ""  | tee -a "$LOG_FILE"
echo "=============================================================================="  | tee -a "$LOG_FILE"
echo "✅ ALL ABLATION STUDIES COMPLETE!"  | tee -a "$LOG_FILE"
echo "=============================================================================="  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"
echo "Results saved to: $RESULTS_DIR"  | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE"  | tee -a "$LOG_FILE"
echo "WandB project: specdifftree-comprehensive"  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"
echo "Next steps:"  | tee -a "$LOG_FILE"
echo "1. View results on WandB: https://wandb.ai/"  | tee -a "$LOG_FILE"
echo "2. Check generated figures in: $RESULTS_DIR/figures/"  | tee -a "$LOG_FILE"
echo "3. Review summary report: $RESULTS_DIR/summary_report.md"  | tee -a "$LOG_FILE"
echo ""  | tee -a "$LOG_FILE"
echo "Completed at $(date)"  | tee -a "$LOG_FILE"


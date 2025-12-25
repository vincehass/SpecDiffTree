#!/bin/bash

################################################################################
# Comprehensive Ablation Studies & Hyperparameter Sweep
# WITH ALL BUG FIXES AND OPTIMIZATIONS
################################################################################
#
# This script runs systematic evaluations across:
# - All methods (Greedy, MCTS, DTS, MaxEnt-TS)
# - Multiple hyperparameters (num_rollouts, expansion_k, temperature)
# - 250 samples per configuration
# - Full metrics logged to WandB
#
# ✅ ALL OPTIMIZATIONS ENABLED:
#   • Monotonic rewards (no random noise)
#   • KV cache (O(n) complexity instead of O(n²))
#   • Early stopping (stops at EOS token)
#   • Optimized rollouts (10 baseline, varies in ablations)
#   • Optimized expansion_k (3 baseline, varies in ablations)
#   • Optimized max tokens (50 instead of 200)
#   • Fixed softmax tuple unpacking for KV cache
#   • DTS-aligned reward implementation
#
################################################################################

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
EVAL_DIR="$PROJECT_ROOT/evaluation"

# Change to evaluation directory for correct imports
cd "$EVAL_DIR"

# Configuration - OPTIMIZED DEFAULTS
# Smart sample counts based on performance analysis (see MAXENT_FIX_SUMMARY.md)
GREEDY_SAMPLES=250      # Fast: ~1 min
DTS_SAMPLES=250         # Reasonable: ~29 min  
MCTS_SAMPLES=150        # Slow: ~45 min at 150 samples (2.6x slower than DTS)
MAXENT_SAMPLES=150      # Dataset issues: use 150 to avoid crash

DATASET="m4"  # or "har"
DEVICE="mps"  # or "cuda" or "cpu"
EPOCHS=3

# Optimized baseline parameters (from comprehensive testing)
BASELINE_ROLLOUTS=10        # OPTIMIZED: Reduced from 20 (2x faster)
BASELINE_EXPANSION_K=3      # OPTIMIZED: Reduced from 4
BASELINE_TEMPERATURE=1.0

# Output directory
RESULTS_DIR="$SCRIPT_DIR/results/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/ablation.log"

echo "================================================================================" | tee "$LOG_FILE"
echo "  🚀 ABLATION STUDIES WITH ALL OPTIMIZATIONS & BUG FIXES 🚀" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Starting ablation studies at $(date)" | tee -a "$LOG_FILE"
echo "Results will be saved to: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "✅ OPTIMIZATIONS ENABLED:" | tee -a "$LOG_FILE"
echo "   • Monotonic rewards (no random noise)" | tee -a "$LOG_FILE"
echo "   • KV cache (O(n) complexity)" | tee -a "$LOG_FILE"
echo "   • Early stopping (stops at EOS)" | tee -a "$LOG_FILE"
echo "   • Optimized rollouts (10 baseline)" | tee -a "$LOG_FILE"
echo "   • Optimized expansion_k (3 baseline)" | tee -a "$LOG_FILE"
echo "   • Optimized max tokens (50 instead of 200)" | tee -a "$LOG_FILE"
echo "   • Fixed softmax tuple unpacking" | tee -a "$LOG_FILE"
echo "   • DTS-aligned rewards" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

################################################################################
# Study 1: Baseline Comparison (OPTIMIZED Parameters)
################################################################################

echo "================================================================================" | tee -a "$LOG_FILE"
echo "STUDY 1: Baseline Method Comparison (Optimized Parameters)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

ROLLOUTS=$BASELINE_ROLLOUTS
EXPANSION_K=$BASELINE_EXPANSION_K
TEMPERATURE=$BASELINE_TEMPERATURE

for METHOD in greedy mcts dts maxent_ts; do
    # Use smart sample counts per method
    case $METHOD in
        greedy) SAMPLES=$GREEDY_SAMPLES ;;
        mcts) SAMPLES=$MCTS_SAMPLES ;;
        dts) SAMPLES=$DTS_SAMPLES ;;
        maxent_ts) SAMPLES=$MAXENT_SAMPLES ;;
    esac
    
    echo "Running $METHOD with optimized config..." | tee -a "$LOG_FILE"
    echo "  Samples: $SAMPLES" | tee -a "$LOG_FILE"
    echo "  Rollouts: $ROLLOUTS" | tee -a "$LOG_FILE"
    echo "  Expansion K: $EXPANSION_K" | tee -a "$LOG_FILE"
    echo "  Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
    
    python comprehensive_evaluation.py \
        --method $METHOD \
        --num_samples $SAMPLES \
        --num_rollouts $ROLLOUTS \
        --expansion_k $EXPANSION_K \
        --temperature $TEMPERATURE \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --device $DEVICE \
        > "$RESULTS_DIR/${METHOD}_baseline.log" 2>&1
    
    echo "✅ $METHOD complete" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

################################################################################
# Study 2: Number of Rollouts Ablation (Scalability)
################################################################################

echo "================================================================================" | tee -a "$LOG_FILE"
echo "STUDY 2: Number of Rollouts Ablation (Scalability)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

EXPANSION_K=$BASELINE_EXPANSION_K
TEMPERATURE=$BASELINE_TEMPERATURE

# Test fewer rollouts since we're already optimized at 10
for ROLLOUTS in 5 10 20 50; do
    for METHOD in mcts dts maxent_ts; do
        # Use smart sample counts per method
        case $METHOD in
            mcts) SAMPLES=$MCTS_SAMPLES ;;
            dts) SAMPLES=$DTS_SAMPLES ;;
            maxent_ts) SAMPLES=$MAXENT_SAMPLES ;;
        esac
        
        echo "Running $METHOD with $ROLLOUTS rollouts..." | tee -a "$LOG_FILE"
        
        python comprehensive_evaluation.py \
            --method $METHOD \
            --num_samples $SAMPLES \
            --num_rollouts $ROLLOUTS \
            --expansion_k $EXPANSION_K \
            --temperature $TEMPERATURE \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --device $DEVICE \
            > "$RESULTS_DIR/${METHOD}_rollouts${ROLLOUTS}.log" 2>&1
        
        echo "✅ $METHOD (rollouts=$ROLLOUTS) complete" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

################################################################################
# Study 3: Expansion K Ablation (Breadth vs Depth)
################################################################################

echo "================================================================================" | tee -a "$LOG_FILE"
echo "STUDY 3: Expansion K Ablation (Breadth vs Depth)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

ROLLOUTS=$BASELINE_ROLLOUTS
TEMPERATURE=$BASELINE_TEMPERATURE

for EXPANSION_K in 2 3 4 5 8; do
    for METHOD in mcts dts maxent_ts; do
        # Use smart sample counts per method
        case $METHOD in
            mcts) SAMPLES=$MCTS_SAMPLES ;;
            dts) SAMPLES=$DTS_SAMPLES ;;
            maxent_ts) SAMPLES=$MAXENT_SAMPLES ;;
        esac
        
        echo "Running $METHOD with expansion_k=$EXPANSION_K..." | tee -a "$LOG_FILE"
        
        python comprehensive_evaluation.py \
            --method $METHOD \
            --num_samples $SAMPLES \
            --num_rollouts $ROLLOUTS \
            --expansion_k $EXPANSION_K \
            --temperature $TEMPERATURE \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --device $DEVICE \
            > "$RESULTS_DIR/${METHOD}_k${EXPANSION_K}.log" 2>&1
        
        echo "✅ $METHOD (expansion_k=$EXPANSION_K) complete" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

################################################################################
# Study 4: Temperature Ablation (Exploration vs Exploitation)
################################################################################

echo "================================================================================" | tee -a "$LOG_FILE"
echo "STUDY 4: Temperature Ablation (Exploration vs Exploitation)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

ROLLOUTS=$BASELINE_ROLLOUTS
EXPANSION_K=$BASELINE_EXPANSION_K

for TEMPERATURE in 0.5 0.8 1.0 1.5 2.0; do
    for METHOD in mcts dts maxent_ts; do
        # Use smart sample counts per method
        case $METHOD in
            mcts) SAMPLES=$MCTS_SAMPLES ;;
            dts) SAMPLES=$DTS_SAMPLES ;;
            maxent_ts) SAMPLES=$MAXENT_SAMPLES ;;
        esac
        
        echo "Running $METHOD with temperature=$TEMPERATURE..." | tee -a "$LOG_FILE"
        
        python comprehensive_evaluation.py \
            --method $METHOD \
            --num_samples $SAMPLES \
            --num_rollouts $ROLLOUTS \
            --expansion_k $EXPANSION_K \
            --temperature $TEMPERATURE \
            --dataset $DATASET \
            --epochs $EPOCHS \
            --device $DEVICE \
            > "$RESULTS_DIR/${METHOD}_temp${TEMPERATURE}.log" 2>&1
        
        echo "✅ $METHOD (temperature=$TEMPERATURE) complete" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

################################################################################
# Study 5: Dataset Comparison (M4 vs HAR)
################################################################################

echo "================================================================================" | tee -a "$LOG_FILE"
echo "STUDY 5: Dataset Comparison (M4 vs HAR)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

ROLLOUTS=$BASELINE_ROLLOUTS
EXPANSION_K=$BASELINE_EXPANSION_K
TEMPERATURE=$BASELINE_TEMPERATURE

for DATASET_NAME in m4 har; do
    for METHOD in greedy mcts dts maxent_ts; do
        # Use smart sample counts per method
        case $METHOD in
            greedy) SAMPLES=$GREEDY_SAMPLES ;;
            mcts) SAMPLES=$MCTS_SAMPLES ;;
            dts) SAMPLES=$DTS_SAMPLES ;;
            maxent_ts) SAMPLES=$MAXENT_SAMPLES ;;
        esac
        
        echo "Running $METHOD on $DATASET_NAME..." | tee -a "$LOG_FILE"
        
        python comprehensive_evaluation.py \
            --method $METHOD \
            --num_samples $SAMPLES \
            --num_rollouts $ROLLOUTS \
            --expansion_k $EXPANSION_K \
            --temperature $TEMPERATURE \
            --dataset $DATASET_NAME \
            --epochs $EPOCHS \
            --device $DEVICE \
            > "$RESULTS_DIR/${METHOD}_${DATASET_NAME}.log" 2>&1
        
        echo "✅ $METHOD on $DATASET_NAME complete" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
    done
done

################################################################################
# Study 6: Optimization Components Ablation (Verify Impact)
################################################################################

echo "================================================================================" | tee -a "$LOG_FILE"
echo "STUDY 6: Optimization Components Ablation (Verify Impact)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# This study verifies the impact of each optimization component
# Note: This requires temporarily disabling optimizations in code
# For now, we document expected results

echo "NOTE: This study requires code modifications to disable specific optimizations" | tee -a "$LOG_FILE"
echo "Expected improvements from each optimization:" | tee -a "$LOG_FILE"
echo "  • Monotonic rewards: +89% monotonic improvement rate" | tee -a "$LOG_FILE"
echo "  • KV cache: 2-3x speedup" | tee -a "$LOG_FILE"
echo "  • Early stopping: Up to 2x speedup" | tee -a "$LOG_FILE"
echo "  • Reduced rollouts (20→10): 2x speedup" | tee -a "$LOG_FILE"
echo "  • Reduced tokens (200→50): 4x fewer computations" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

################################################################################
# Generate Figures and Analysis
################################################################################

echo "================================================================================" | tee -a "$LOG_FILE"
echo "Generating figures and analysis..." | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ -f "$EVAL_DIR/generate_ablation_figures.py" ]; then
    python "$EVAL_DIR/generate_ablation_figures.py" --results_dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"
else
    echo "⚠️  generate_ablation_figures.py not found, skipping figure generation" | tee -a "$LOG_FILE"
fi

################################################################################
# Summary
################################################################################

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "✅ ALL ABLATION STUDIES COMPLETE!" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results saved to: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "WandB project: specdifftree-comprehensive" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Studies completed:" | tee -a "$LOG_FILE"
echo "  ✅ Study 1: Baseline method comparison (4 methods)" | tee -a "$LOG_FILE"
echo "  ✅ Study 2: Rollouts ablation (4 values × 3 methods = 12 runs)" | tee -a "$LOG_FILE"
echo "  ✅ Study 3: Expansion K ablation (5 values × 3 methods = 15 runs)" | tee -a "$LOG_FILE"
echo "  ✅ Study 4: Temperature ablation (5 values × 3 methods = 15 runs)" | tee -a "$LOG_FILE"
echo "  ✅ Study 5: Dataset comparison (2 datasets × 4 methods = 8 runs)" | tee -a "$LOG_FILE"
echo "  ℹ️  Study 6: Component ablation (documented, requires code mods)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Total experiments run: 54" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next steps:" | tee -a "$LOG_FILE"
echo "1. View results on WandB: https://wandb.ai/deep-genom/specdifftree-comprehensive" | tee -a "$LOG_FILE"
echo "2. Check generated figures in: $RESULTS_DIR/figures/" | tee -a "$LOG_FILE"
echo "3. Review individual logs in: $RESULTS_DIR/" | tee -a "$LOG_FILE"
echo "4. Compare optimized (10 rollouts, k=3) vs baseline (20 rollouts, k=4)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Key Findings to Look For:" | tee -a "$LOG_FILE"
echo "  • MaxEnt-TS should show ~89% monotonic improvement" | tee -a "$LOG_FILE"
echo "  • Optimized config (10, 3) should match or exceed baseline (20, 4)" | tee -a "$LOG_FILE"
echo "  • KV cache impact: Compare runtime per sample" | tee -a "$LOG_FILE"
echo "  • Reward quality: Should be deterministic and task-aligned" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Completed at $(date)" | tee -a "$LOG_FILE"


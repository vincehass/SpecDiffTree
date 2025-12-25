#!/bin/bash

################################################################################
# Parallel Evaluation with WandB - MaxEnt-TS, MCTS, Greedy, DTS
################################################################################

set -e

# Configuration - OPTIMIZED (10 rollouts for 2x speedup)
# Smart sample counts based on performance analysis
GREEDY_SAMPLES=250      # Fast: ~1 min
DTS_SAMPLES=250         # Reasonable: ~29 min  
MCTS_SAMPLES=150        # Slow: ~45 min at 150 samples (2.6x slower than DTS)
MAXENT_SAMPLES=150      # Dataset issues: use 150 to avoid crash

NUM_ROLLOUTS=10          # OPTIMIZED: Reduced from 20 (2x faster)
EXPANSION_K=3            # OPTIMIZED: Reduced from 4 (faster expansion)
TEMPERATURE=1.0
DATASET="m4"
DEVICE="mps"
EPOCHS=3

# Output directory
RESULTS_DIR="results/parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/parallel_run.log"

echo "================================================================================" | tee "$LOG_FILE"
echo "  🚀 PARALLEL EVALUATION - Greedy, MCTS, DTS, MaxEnt-TS with WandB" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  • Smart Sample Counts:" | tee -a "$LOG_FILE"
echo "    - Greedy:    $GREEDY_SAMPLES samples (~1 min)" | tee -a "$LOG_FILE"
echo "    - DTS:       $DTS_SAMPLES samples (~29 min)" | tee -a "$LOG_FILE"
echo "    - MCTS:      $MCTS_SAMPLES samples (~45 min)" | tee -a "$LOG_FILE"
echo "    - MaxEnt-TS: $MAXENT_SAMPLES samples (~45 min)" | tee -a "$LOG_FILE"
echo "  • Rollouts: $NUM_ROLLOUTS" | tee -a "$LOG_FILE"
echo "  • Expansion K: $EXPANSION_K" | tee -a "$LOG_FILE"
echo "  • Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "  • Dataset: $DATASET" | tee -a "$LOG_FILE"
echo "  • Device: $DEVICE" | tee -a "$LOG_FILE"
echo "  • Epochs: $EPOCHS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  ✅ All bug fixes applied (MPS, KV cache, monotonic rewards)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results directory: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "Expected total time: ~2 hours (running in parallel)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

################################################################################
# Run Methods in Parallel
################################################################################

echo "🔬 Starting parallel evaluation..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run Greedy in background
echo "▶️  Starting Greedy ($GREEDY_SAMPLES samples)..." | tee -a "$LOG_FILE"
(cd ../../evaluation && python comprehensive_evaluation.py \
    --method greedy \
    --num_samples $GREEDY_SAMPLES \
    --num_rollouts $NUM_ROLLOUTS \
    --expansion_k $EXPANSION_K \
    --temperature $TEMPERATURE \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --device $DEVICE \
    > "../experiments/scripts/$RESULTS_DIR/greedy.log" 2>&1) &

GREEDY_PID=$!
echo "   Greedy started (PID: $GREEDY_PID)" | tee -a "$LOG_FILE"

# Small delay to avoid resource contention at startup
sleep 3

# Run MCTS in background
echo "▶️  Starting MCTS ($MCTS_SAMPLES samples)..." | tee -a "$LOG_FILE"
(cd ../../evaluation && python comprehensive_evaluation.py \
    --method mcts \
    --num_samples $MCTS_SAMPLES \
    --num_rollouts $NUM_ROLLOUTS \
    --expansion_k $EXPANSION_K \
    --temperature $TEMPERATURE \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --device $DEVICE \
    > "../experiments/scripts/$RESULTS_DIR/mcts.log" 2>&1) &

MCTS_PID=$!
echo "   MCTS started (PID: $MCTS_PID)" | tee -a "$LOG_FILE"

sleep 3

# Run DTS in background
echo "▶️  Starting DTS ($DTS_SAMPLES samples)..." | tee -a "$LOG_FILE"
(cd ../../evaluation && python comprehensive_evaluation.py \
    --method dts \
    --num_samples $DTS_SAMPLES \
    --num_rollouts $NUM_ROLLOUTS \
    --expansion_k $EXPANSION_K \
    --temperature $TEMPERATURE \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --device $DEVICE \
    > "../experiments/scripts/$RESULTS_DIR/dts.log" 2>&1) &

DTS_PID=$!
echo "   DTS started (PID: $DTS_PID)" | tee -a "$LOG_FILE"

sleep 3

# Run MaxEnt-TS in background
echo "▶️  Starting MaxEnt-TS ($MAXENT_SAMPLES samples)..." | tee -a "$LOG_FILE"
(cd ../../evaluation && python comprehensive_evaluation.py \
    --method maxent_ts \
    --num_samples $MAXENT_SAMPLES \
    --num_rollouts $NUM_ROLLOUTS \
    --expansion_k $EXPANSION_K \
    --temperature $TEMPERATURE \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --device $DEVICE \
    > "../experiments/scripts/$RESULTS_DIR/maxent_ts.log" 2>&1) &

MAXENT_PID=$!
echo "   MaxEnt-TS started (PID: $MAXENT_PID)" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✅ All 4 methods running in parallel!" | tee -a "$LOG_FILE"
echo "   Greedy:    PID $GREEDY_PID" | tee -a "$LOG_FILE"
echo "   MCTS:      PID $MCTS_PID" | tee -a "$LOG_FILE"
echo "   DTS:       PID $DTS_PID" | tee -a "$LOG_FILE"
echo "   MaxEnt-TS: PID $MAXENT_PID" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

################################################################################
# Monitor Progress
################################################################################

echo "⏳ Monitoring progress..." | tee -a "$LOG_FILE"
echo "   (Check logs: $RESULTS_DIR/*.log)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Function to check if process is still running
is_running() {
    kill -0 $1 2>/dev/null
}

# Monitor loop
START_TIME=$(date +%s)
while is_running $GREEDY_PID || is_running $MCTS_PID || is_running $DTS_PID || is_running $MAXENT_PID; do
    ELAPSED=$(($(date +%s) - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    # Check status
    GREEDY_STATUS="✅ Done"
    MCTS_STATUS="✅ Done"
    DTS_STATUS="✅ Done"
    MAXENT_STATUS="✅ Done"
    
    if is_running $GREEDY_PID; then
        GREEDY_STATUS="⏳ Running"
    fi
    
    if is_running $MCTS_PID; then
        MCTS_STATUS="⏳ Running"
    fi
    
    if is_running $DTS_PID; then
        DTS_STATUS="⏳ Running"
    fi
    
    if is_running $MAXENT_PID; then
        MAXENT_STATUS="⏳ Running"
    fi
    
    # Print status every 30 seconds
    echo "[${MINUTES}m ${SECONDS}s] Greedy: $GREEDY_STATUS | MCTS: $MCTS_STATUS | DTS: $DTS_STATUS | MaxEnt-TS: $MAXENT_STATUS" | tee -a "$LOG_FILE"
    
    sleep 30
done

TOTAL_TIME=$(($(date +%s) - START_TIME))
TOTAL_MINUTES=$((TOTAL_TIME / 60))
TOTAL_SECONDS=$((TOTAL_TIME % 60))

echo "" | tee -a "$LOG_FILE"
echo "✅ All 4 evaluations complete!" | tee -a "$LOG_FILE"
echo "   Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

################################################################################
# Move Results to Output Directory
################################################################################

echo "📦 Organizing results..." | tee -a "$LOG_FILE"

# Move result JSON files to results directory
if [ -f "results/greedy_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" ]; then
    mv "results/greedy_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" "$RESULTS_DIR/"
    echo "   ✅ Moved Greedy results" | tee -a "$LOG_FILE"
fi

if [ -f "results/mcts_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" ]; then
    mv "results/mcts_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" "$RESULTS_DIR/"
    echo "   ✅ Moved MCTS results" | tee -a "$LOG_FILE"
fi

if [ -f "results/dts_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" ]; then
    mv "results/dts_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" "$RESULTS_DIR/"
    echo "   ✅ Moved DTS results" | tee -a "$LOG_FILE"
fi

if [ -f "results/maxent_ts_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" ]; then
    mv "results/maxent_ts_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" "$RESULTS_DIR/"
    echo "   ✅ Moved MaxEnt-TS results" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"

################################################################################
# Generate Figures
################################################################################

echo "================================================================================" | tee -a "$LOG_FILE"
echo "  📊 GENERATING FIGURES" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

python generate_ablation_figures.py --results_dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG_FILE"

################################################################################
# Summary
################################################################################

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "  ✅ EVALUATION COMPLETE!" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📁 Results location: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📊 Generated files:" | tee -a "$LOG_FILE"
echo "   • greedy_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" | tee -a "$LOG_FILE"
echo "   • mcts_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" | tee -a "$LOG_FILE"
echo "   • dts_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" | tee -a "$LOG_FILE"
echo "   • maxent_ts_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" | tee -a "$LOG_FILE"
echo "   • figures/*.png (6 figure types)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📈 View on WandB:" | tee -a "$LOG_FILE"
echo "   https://wandb.ai/your-username/specdifftree-comprehensive" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📖 Check logs:" | tee -a "$LOG_FILE"
echo "   Greedy:    $RESULTS_DIR/greedy.log" | tee -a "$LOG_FILE"
echo "   MCTS:      $RESULTS_DIR/mcts.log" | tee -a "$LOG_FILE"
echo "   DTS:       $RESULTS_DIR/dts.log" | tee -a "$LOG_FILE"
echo "   MaxEnt-TS: $RESULTS_DIR/maxent_ts.log" | tee -a "$LOG_FILE"
echo "   Summary:   $RESULTS_DIR/parallel_run.log" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "⏱️  Total execution time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Completed at $(date)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"


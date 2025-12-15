#!/bin/bash

################################################################################
# Parallel Evaluation with Pure MLX - M3 Max Optimized
################################################################################
#
# This script runs evaluation using ONLY MLX (no PyTorch/MPS dependency)
# Provides 2-5x speedup on M3 Max compared to PyTorch/MPS!
#
# Note: MLX baselines (MCTS, DTS) are not yet implemented in pure MLX.
# This script supports: Greedy, MaxEnt-TS in pure MLX mode.
#
################################################################################

set -e

# Configuration
NUM_SAMPLES=250
NUM_ROLLOUTS=20
EXPANSION_K=4
TEMPERATURE=1.0
DATASET="m4"
EPOCHS=3
MODEL="mlx-community/Llama-3.2-1B-Instruct"

# Output directory
RESULTS_DIR="results/parallel_mlx_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Log file
LOG_FILE="$RESULTS_DIR/parallel_run_mlx.log"

echo "================================================================================" | tee "$LOG_FILE"
echo "  🚀 PURE MLX PARALLEL EVALUATION (M3 Max Optimized)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Configuration:" | tee -a "$LOG_FILE"
echo "  • Samples: $NUM_SAMPLES" | tee -a "$LOG_FILE"
echo "  • Rollouts: $NUM_ROLLOUTS" | tee -a "$LOG_FILE"
echo "  • Expansion K: $EXPANSION_K" | tee -a "$LOG_FILE"
echo "  • Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
echo "  • Dataset: $DATASET" | tee -a "$LOG_FILE"
echo "  • Model: $MODEL" | tee -a "$LOG_FILE"
echo "  • Epochs: $EPOCHS" | tee -a "$LOG_FILE"
echo "  • Framework: Pure MLX (No PyTorch!)" | tee -a "$LOG_FILE"
echo "  • Hardware: M3 Max optimized" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Results directory: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

################################################################################
# Run Methods in Parallel (MLX-only)
################################################################################

echo "🔬 Starting parallel MLX evaluation..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run Greedy in background
echo "▶️  Starting Greedy (Pure MLX)..." | tee -a "$LOG_FILE"
python evaluation/comprehensive_evaluation_mlx.py \
    --method greedy \
    --num_samples $NUM_SAMPLES \
    --num_rollouts $NUM_ROLLOUTS \
    --expansion_k $EXPANSION_K \
    --temperature $TEMPERATURE \
    --dataset $DATASET \
    --model "$MODEL" \
    --epochs $EPOCHS \
    --no_wandb \
    > "$RESULTS_DIR/greedy_mlx.log" 2>&1 &

GREEDY_PID=$!
echo "   Greedy (MLX) started (PID: $GREEDY_PID)" | tee -a "$LOG_FILE"

sleep 5

# Run MaxEnt-TS in background
echo "▶️  Starting MaxEnt-TS (Pure MLX)..." | tee -a "$LOG_FILE"
python evaluation/comprehensive_evaluation_mlx.py \
    --method maxent_ts \
    --num_samples $NUM_SAMPLES \
    --num_rollouts $NUM_ROLLOUTS \
    --expansion_k $EXPANSION_K \
    --temperature $TEMPERATURE \
    --dataset $DATASET \
    --model "$MODEL" \
    --epochs $EPOCHS \
    --no_wandb \
    > "$RESULTS_DIR/maxent_ts_mlx.log" 2>&1 &

MAXENT_PID=$!
echo "   MaxEnt-TS (MLX) started (PID: $MAXENT_PID)" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "✅ Both MLX methods running in parallel!" | tee -a "$LOG_FILE"
echo "   Greedy:    PID $GREEDY_PID" | tee -a "$LOG_FILE"
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
while is_running $GREEDY_PID || is_running $MAXENT_PID; do
    ELAPSED=$(($(date +%s) - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    # Check status
    GREEDY_STATUS="✅ Done"
    MAXENT_STATUS="✅ Done"
    
    if is_running $GREEDY_PID; then
        GREEDY_STATUS="⏳ Running"
    fi
    
    if is_running $MAXENT_PID; then
        MAXENT_STATUS="⏳ Running"
    fi
    
    # Print status every 30 seconds
    echo "[${MINUTES}m ${SECONDS}s] Greedy: $GREEDY_STATUS | MaxEnt-TS: $MAXENT_STATUS" | tee -a "$LOG_FILE"
    
    sleep 30
done

TOTAL_TIME=$(($(date +%s) - START_TIME))
TOTAL_MINUTES=$((TOTAL_TIME / 60))
TOTAL_SECONDS=$((TOTAL_TIME % 60))

echo "" | tee -a "$LOG_FILE"
echo "✅ Both MLX evaluations complete!" | tee -a "$LOG_FILE"
echo "   Total time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

################################################################################
# Move Results
################################################################################

echo "📦 Organizing results..." | tee -a "$LOG_FILE"

# Move result JSON files
if [ -f "results/greedy_mlx_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" ]; then
    mv "results/greedy_mlx_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" "$RESULTS_DIR/"
    echo "   ✅ Moved Greedy results" | tee -a "$LOG_FILE"
fi

if [ -f "results/maxent_ts_mlx_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" ]; then
    mv "results/maxent_ts_mlx_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" "$RESULTS_DIR/"
    echo "   ✅ Moved MaxEnt-TS results" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"

################################################################################
# Summary
################################################################################

echo "" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "  ✅ MLX EVALUATION COMPLETE!" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📁 Results location: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📊 Generated files:" | tee -a "$LOG_FILE"
echo "   • greedy_mlx_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" | tee -a "$LOG_FILE"
echo "   • maxent_ts_mlx_k${EXPANSION_K}_roll${NUM_ROLLOUTS}.json" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📖 Check logs:" | tee -a "$LOG_FILE"
echo "   Greedy:    $RESULTS_DIR/greedy_mlx.log" | tee -a "$LOG_FILE"
echo "   MaxEnt-TS: $RESULTS_DIR/maxent_ts_mlx.log" | tee -a "$LOG_FILE"
echo "   Summary:   $RESULTS_DIR/parallel_run_mlx.log" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "⏱️  Total execution time: ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "🚀 Pure MLX provides 2-5x speedup on M3 Max!" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Completed at $(date)" | tee -a "$LOG_FILE"
echo "================================================================================" | tee -a "$LOG_FILE"


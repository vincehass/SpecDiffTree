#!/bin/bash

################################################################################
# Sequential Evaluation - Memory-Optimized (Run One at a Time)
################################################################################

set -e

# Configuration (Memory-Optimized)
GREEDY_SAMPLES=200
MCTS_SAMPLES=100
DTS_SAMPLES=150
MAXENT_SAMPLES=100
NUM_ROLLOUTS=10
EXPANSION_K=3
TEMPERATURE=1.0
DATASET="m4"
EPOCHS=3
DEVICE="mps"

# Output directory
RESULTS_DIR="results/sequential_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "================================================================================"
echo "  🔬 SEQUENTIAL EVALUATION (Memory-Optimized)"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  • Greedy:     $GREEDY_SAMPLES samples"
echo "  • MCTS:       $MCTS_SAMPLES samples"
echo "  • DTS:        $DTS_SAMPLES samples"
echo "  • MaxEnt-TS:  $MAXENT_SAMPLES samples"
echo "  • Rollouts:   $NUM_ROLLOUTS"
echo "  • Device:     $DEVICE"
echo "  • Mode:       Sequential (one at a time, low memory)"
echo ""

cd ../../evaluation

################################################################################
# 1. Greedy (Fast Baseline)
################################################################################

echo "▶️  [1/4] Running Greedy..."
python comprehensive_evaluation.py \
    --method greedy \
    --num_samples $GREEDY_SAMPLES \
    --num_rollouts $NUM_ROLLOUTS \
    --expansion_k $EXPANSION_K \
    --temperature $TEMPERATURE \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --device $DEVICE \
    > "../experiments/scripts/$RESULTS_DIR/greedy.log" 2>&1

echo "   ✅ Greedy complete!"
echo ""

################################################################################
# 2. MCTS
################################################################################

echo "▶️  [2/4] Running MCTS..."
python comprehensive_evaluation.py \
    --method mcts \
    --num_samples $MCTS_SAMPLES \
    --num_rollouts $NUM_ROLLOUTS \
    --expansion_k $EXPANSION_K \
    --temperature $TEMPERATURE \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --device $DEVICE \
    > "../experiments/scripts/$RESULTS_DIR/mcts.log" 2>&1

echo "   ✅ MCTS complete!"
echo ""

################################################################################
# 3. DTS
################################################################################

echo "▶️  [3/4] Running DTS..."
python comprehensive_evaluation.py \
    --method dts \
    --num_samples $DTS_SAMPLES \
    --num_rollouts $NUM_ROLLOUTS \
    --expansion_k $EXPANSION_K \
    --temperature $TEMPERATURE \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --device $DEVICE \
    > "../experiments/scripts/$RESULTS_DIR/dts.log" 2>&1

echo "   ✅ DTS complete!"
echo ""

################################################################################
# 4. MaxEnt-TS
################################################################################

echo "▶️  [4/4] Running MaxEnt-TS..."
python comprehensive_evaluation.py \
    --method maxent_ts \
    --num_samples $MAXENT_SAMPLES \
    --num_rollouts $NUM_ROLLOUTS \
    --expansion_k $EXPANSION_K \
    --temperature $TEMPERATURE \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --device $DEVICE \
    > "../experiments/scripts/$RESULTS_DIR/maxent_ts.log" 2>&1

echo "   ✅ MaxEnt-TS complete!"
echo ""

echo "================================================================================"
echo "  ✅ ALL EVALUATIONS COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""


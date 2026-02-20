#!/bin/bash
# =============================================================================
# Full Cross-Validation Experiments for ICDAR 2026 Paper
# =============================================================================
# 
# This script runs 5-fold cross-validation for all three models:
#   1. Enhanced CNN (58M params)
#   2. Transformer (50.8M params)  
#   3. Multi-Channel CNN (70.3M params)
#
# Each model is trained with Korniienko images on the expanded dataset (1,720 samples)
#
# Usage:
#   ./run_full_experiments.sh          # Run all experiments
#   ./run_full_experiments.sh enhanced # Run only enhanced model
#   ./run_full_experiments.sh quick    # Quick test (3 folds, 5 epochs)
#
# Results saved to: evaluation_results/full_experiments_YYYYMMDD_HHMMSS/
# =============================================================================

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="evaluation_results/full_experiments_${TIMESTAMP}"
FOLDS=5
EPOCHS=30
BATCH_SIZE=8
LR=1e-4

# Models to run
MODELS=("enhanced" "transformer" "multichannel")

# Parse arguments
if [ "$1" == "quick" ]; then
    FOLDS=3
    EPOCHS=5
    echo "==> QUICK MODE: 3 folds, 5 epochs"
elif [ "$1" != "" ] && [ "$1" != "all" ]; then
    # Run single model
    MODELS=("$1")
    echo "==> Running single model: $1"
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Log file
LOG_FILE="${OUTPUT_BASE}/experiment_log.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================================================="
echo " Saint Sophia Graffiti Recognition - Full Experiments"
echo " Started: $(date)"
echo "============================================================================="
echo ""
echo "Configuration:"
echo "  Output: $OUTPUT_BASE"
echo "  Folds: $FOLDS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  Models: ${MODELS[*]}"
echo ""

# Check dataset
echo "Checking dataset..."
DATA_CSV="data/complete_dataset.csv"
if [ ! -f "$DATA_CSV" ]; then
    echo "ERROR: Dataset not found: $DATA_CSV"
    exit 1
fi
SAMPLE_COUNT=$(tail -n +2 "$DATA_CSV" | wc -l)
echo "  Dataset: $DATA_CSV"
echo "  Total inscriptions: $SAMPLE_COUNT"
echo ""

# Track results
declare -A MODEL_CER
declare -A MODEL_SEQ_ACC

# Run each model
for MODEL in "${MODELS[@]}"; do
    echo "============================================================================="
    echo " MODEL: $MODEL"
    echo " Started: $(date)"
    echo "============================================================================="
    
    MODEL_OUTPUT="${OUTPUT_BASE}/${MODEL}"
    mkdir -p "$MODEL_OUTPUT"
    
    # Run cross-validation
    echo ""
    echo "Running ${FOLDS}-fold cross-validation..."
    echo ""
    
    python cross_validate.py \
        --model "$MODEL" \
        --folds "$FOLDS" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --use_korniienko \
        --data_csv "$DATA_CSV" \
        --output_dir "$MODEL_OUTPUT" \
        2>&1 | tee "${MODEL_OUTPUT}/training.log"
    
    # Extract results
    SUMMARY_FILE=$(find "$MODEL_OUTPUT" -name "summary.json" | head -1)
    if [ -f "$SUMMARY_FILE" ]; then
        CER=$(python -c "import json; d=json.load(open('$SUMMARY_FILE')); print(f\"{d.get('cer_mean', 0)*100:.2f}\")")
        SEQ_ACC=$(python -c "import json; d=json.load(open('$SUMMARY_FILE')); print(f\"{d.get('sequence_accuracy_mean', 0)*100:.2f}\")")
        MODEL_CER[$MODEL]=$CER
        MODEL_SEQ_ACC[$MODEL]=$SEQ_ACC
        echo ""
        echo "  Results: CER=${CER}%, Seq Accuracy=${SEQ_ACC}%"
    fi
    
    echo ""
    echo " $MODEL completed: $(date)"
    echo ""
done

# Final summary
echo ""
echo "============================================================================="
echo " FINAL SUMMARY"
echo " Completed: $(date)"
echo "============================================================================="
echo ""
echo "Model               | CER (%)     | Seq Accuracy (%)"
echo "--------------------|-------------|------------------"
for MODEL in "${MODELS[@]}"; do
    printf "%-19s | %-11s | %s\n" "$MODEL" "${MODEL_CER[$MODEL]:-N/A}" "${MODEL_SEQ_ACC[$MODEL]:-N/A}"
done
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""

# Create summary JSON
cat > "${OUTPUT_BASE}/final_summary.json" << EOF
{
    "timestamp": "$TIMESTAMP",
    "dataset": "$DATA_CSV",
    "sample_count": $SAMPLE_COUNT,
    "folds": $FOLDS,
    "epochs": $EPOCHS,
    "batch_size": $BATCH_SIZE,
    "learning_rate": "$LR",
    "results": {
EOF

# Add model results
first=true
for MODEL in "${MODELS[@]}"; do
    if [ "$first" = true ]; then
        first=false
    else
        echo "," >> "${OUTPUT_BASE}/final_summary.json"
    fi
    echo -n "        \"$MODEL\": {\"cer\": ${MODEL_CER[$MODEL]:-0}, \"seq_accuracy\": ${MODEL_SEQ_ACC[$MODEL]:-0}}" >> "${OUTPUT_BASE}/final_summary.json"
done

cat >> "${OUTPUT_BASE}/final_summary.json" << EOF

    }
}
EOF

echo "Summary saved to: ${OUTPUT_BASE}/final_summary.json"
echo ""
echo "Done!"

#!/bin/bash
# ===========================================================================
# Master Experiment Runner — Pyramid Mixup Full Matrix (5 groups, 12 configs)
# ===========================================================================
# All experiments run via the unified Python runner.
#
# Usage:
#   bash run_all_experiments.sh              # single GPU, sequential
#
# Results: ./experiment_results/master_sweep_YYYYMMDD_HHMMSS/
#   all_results.json   — JSON with all metrics
#   MASTER_SUMMARY.log — text log of the full run
# ===========================================================================
set -uo pipefail

NGPU=${1:-1}
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULT_DIR="./experiment_results/master_sweep_${TIMESTAMP}"

echo "============================================================"
echo "  Master Experiment Runner"
echo "  Result dir: $RESULT_DIR"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

mkdir -p "$RESULT_DIR"

cd "$(dirname "$0")"

if [ "$NGPU" -gt 1 ]; then
    echo "[WARN] Multi-GPU ($NGPU) not yet supported in unified runner. Using single GPU."
fi

python3 experiments/run_experiments.py \
    --output_dir "$RESULT_DIR" \
    --alpha 5.0 --gamma 1.0 --num_levels 3 \
    --sampler_real_ratio 0.30 --n_epochs 10 \
    2>&1 | tee "${RESULT_DIR}/MASTER_SUMMARY.log"

echo ""
echo "============================================================"
echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results: $RESULT_DIR"
echo "  Summary: ${RESULT_DIR}/MASTER_SUMMARY.log"
echo "============================================================"

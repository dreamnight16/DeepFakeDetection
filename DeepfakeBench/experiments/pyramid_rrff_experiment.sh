#!/bin/bash
# ===========================================================================
# Pyramid RR+FF Mixup Experiment (No RF Cross-Class Pairs)
# ===========================================================================
# Based on Exp2 (Stripped Mixup Loss) from pyramid_loss_ablation.
#
# Key change from Exp2:
#   - RR and FF pairs both use Laplacian pyramid mixup
#     (instead of pixel-space blending as in Exp2)
#   - RF pairs are NOT generated at all
#     (instead of generated-then-stripped as in Exp2)
#
# Achieved via mixup_mode='lap_pyramid_rrff' (new mixup function).
#
# Output:
#   - Accuracy (frame-level, per dataset + average)
#   - Confusion matrix (per dataset, TN/FP/FN/TP)
#   - Score distributions: KDE plots Real vs Fake (train set + test sets)
#
# Usage:
#   bash experiments/pyramid_rrff_experiment.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="./experiment_results/pyramid_rrff_${TIMESTAMP}"

echo "============================================================"
echo "  Pyramid RR+FF Mixup Experiment (No RF)"
echo "  Mode:      lap_pyramid_rrff"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Start:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python3 experiments/pyramid_rrff_experiment.py \
    --train_dataset "FaceForensics++" \
    --val_dataset "Celeb-DF-v2" \
    --test_datasets "WDF" "FFIW" "Celeb-DF-v2" "DeepFakeDetection" "DFDC" "DFDCP" "DeeperForensics-1.0" \
    --output_dir "${OUTPUT_DIR}" \
    --alpha 5.0 \
    --gamma 1.0 \
    --num_levels 3 \
    --sampler_real_ratio 0.30 \
    --n_epochs 10

echo ""
echo "============================================================"
echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results: ${OUTPUT_DIR}"
echo "============================================================"

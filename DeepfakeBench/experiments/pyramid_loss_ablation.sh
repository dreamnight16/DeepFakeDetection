#!/bin/bash
# ===========================================================================
# Pyramid Mixup Loss Ablation Experiment
# ===========================================================================
# Two experiments sharing the same pyramid image mixing:
#
#   Experiment 1: Original Pyramid Mixup
#     Energy-based soft labels (ỹ = 1 − (1 − e_f)^γ) → soft-label CE loss.
#
#   Experiment 2: Stripped Mixup Loss
#     Same pyramid image mixing, but standard hard-label CE loss
#     (soft labels stripped → mixup used purely as data augmentation).
#
# Output per experiment:
#   - Accuracy (frame-level, per dataset + average)
#   - Confusion matrix (per dataset, TN/FP/FN/TP)
#   - Score distributions: KDE plots Real vs Fake (train set + test sets)
#
# Usage:
#   bash experiments/pyramid_loss_ablation.sh                    # default: lap_pyramid mode
#   bash experiments/pyramid_loss_ablation.sh lap_pyramid_label0_full  # label0_full mode
#   bash experiments/pyramid_loss_ablation.sh lap_pyramid       4  # 4-GPU DDP
# ===========================================================================
set -euo pipefail

PYRAMID_MODE="${1:-lap_pyramid}"
NGPU="${2:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="./experiment_results/pyramid_loss_ablation_${PYRAMID_MODE}_${TIMESTAMP}"

echo "============================================================"
echo "  Pyramid Mixup Loss Ablation"
echo "  Mode:      ${PYRAMID_MODE}"
echo "  GPU:       ${NGPU}"
echo "  Output:    ${OUTPUT_DIR}"
echo "  Start:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── Build common args ──────────────────────────────────────────────────────
COMMON_ARGS=(
    --pyramid_mode "${PYRAMID_MODE}"
    --train_dataset "FaceForensics++"
    --val_dataset "Celeb-DF-v2"
    --test_datasets "Celeb-DF-v2" "DFDC" "DFDCP" "DeepFakeDetection"
    --output_dir "${OUTPUT_DIR}"
    --alpha 5.0
    --gamma 1.0
    --num_levels 3
    --sampler v1
    --sampler_real_ratio 0.30
    --n_epochs 10
)

# ── Run experiment ─────────────────────────────────────────────────────────
if [ "$NGPU" -gt 1 ]; then
    echo "[INFO] Multi-GPU training not directly supported by this script."
    echo "       Run experiments/pyramid_loss_ablation.py directly with DDP."
    exit 1
else
    python3 experiments/pyramid_loss_ablation.py "${COMMON_ARGS[@]}"
fi

echo ""
echo "============================================================"
echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results: ${OUTPUT_DIR}"
echo "============================================================"

#!/bin/bash
# ===========================================================================
# Sweep: Video-Level Information Graph Theory — Detector Ablation
# ===========================================================================
# Three independent detectors evaluated with video clips (T frames each):
#
#   Baseline    — temporal mean pool + MLP
#   MI-T        — temporal mutual information between frames
#   GT          — graph topology (Laplacian spectrum + smoothness)
#   GNN         — spatio-temporal GCN
#
# Protocol:
#   Train: FF++ c23 video clips
#   Test:  FF++ c23 + Celeb-DF v2
#   Metric: clip-level AUC
#
# Usage:
#   bash sweep_video_info_graph.sh          # single GPU
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXP_PY="${PROJECT_ROOT}/experiments/validate_video_info_graph.py"
YAML="${PROJECT_ROOT}/DeepfakeBench/training/config/detector/effort.yaml"
SWEEP_LOG="sweep_video_info_graph.log"
TRAIN_DS="FaceForensics++"
TEST_DS="Celeb-DF-v2 FaceForensics++"
CLIP_SIZE=8
MAX_TRAIN=1000
MAX_TEST=500
EPOCHS=20
LR=1e-3
BATCH_SIZE=4
SEED=1024

> "$SWEEP_LOG"

run_one() {
    local TAG=$1
    local DETECTOR=$2

    local SAFE_TAG=$(echo "$TAG" | sed 's/[][ \/()]/_/g')
    local EXP_LOG="sweep_video_${SAFE_TAG}.log"
    echo "===== $TAG ====="

    echo "  [run] $EXP_LOG"
    python3 "$EXP_PY" \
        --detector_path "$YAML" \
        --train_dataset $TRAIN_DS \
        --test_datasets $TEST_DS \
        --clip_size "$CLIP_SIZE" \
        --max_train_clips "$MAX_TRAIN" \
        --max_test_clips "$MAX_TEST" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --seed "$SEED" \
        --detector "$DETECTOR" \
        > "$EXP_LOG" 2>&1 || {
            echo "$TAG | RUN_ERROR" | tee -a "$SWEEP_LOG"
            return
        }

    # Per-dataset results
    for DS in $TEST_DS; do
        echo "    dataset: $DS" | tee -a "$SWEEP_LOG"
        awk -v ds="$DS" '
            BEGIN { in_section=0 }
            /Test: / { if ($2 == ds) in_section=1; else in_section=0 }
            in_section && /^  [A-Z]/ {
                method=$1
                for (i=2; i<=NF && method !~ /[0-9]/; i++)
                    method=method" "$i
                gsub(/ \(\*\)/, "", method)
                auc=$(NF-3); acc=$(NF-2); f1=$(NF-1); ap=$NF
                if (auc ~ /^[0-9]/)
                    printf "      %-30s auc=%-8s acc=%-8s f1=%-8s ap=%s\n",
                           method, auc, acc, f1, ap
            }
        ' "$EXP_LOG" | tee -a "$SWEEP_LOG"
    done

    echo "$TAG | DONE" | tee -a "$SWEEP_LOG"
}

echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Video Info Graph — Detector Ablation"                     | tee -a "$SWEEP_LOG"
echo "  Train: $TRAIN_DS  ($MAX_TRAIN clips)"                     | tee -a "$SWEEP_LOG"
echo "  Test:  $TEST_DS  ($MAX_TEST clips each)"                  | tee -a "$SWEEP_LOG"
echo "  Clip: T=$CLIP_SIZE  epochs=$EPOCHS  lr=$LR"               | tee -a "$SWEEP_LOG"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"                       | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

RUNS=(
    "Baseline            all"
    "MI-T                mi"
    "GT                  gt"
    "GNN                 gnn"
)

TOTAL=${#RUNS[@]}
for i in "${!RUNS[@]}"; do
    idx=$((i + 1))
    read TAG DETECTOR <<< "${RUNS[$i]}"
    run_one "[$idx/$TOTAL] $TAG" "$DETECTOR"
    echo "" | tee -a "$SWEEP_LOG"
done

echo "" | tee -a "$SWEEP_LOG"
echo "=== Done.  Log: $SWEEP_LOG ===" | tee -a "$SWEEP_LOG"

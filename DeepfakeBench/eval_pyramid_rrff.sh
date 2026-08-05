#!/bin/bash
# ===========================================================================
# Evaluate pyramid RR+FF experiment checkpoints with testall.py
# ===========================================================================
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
TEST_DS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"
RESULT_DIR="./experiment_results/pyramid_rrff_20260805_190317"

G1_DIR="${RESULT_DIR}/logs_g1_rf_stripped/effort_2026-08-05-19-03-20"
G2_DIR="${RESULT_DIR}/logs_g2_rf_not_generated/effort_2026-08-05-20-00-39"

CKPT_G1="${G1_DIR}/test/avg/ckpt_best.pth"
CKPT_G2="${G2_DIR}/test/avg/ckpt_best.pth"

EVAL_LOG="eval_pyramid_rrff.log"
> "$EVAL_LOG"

do_eval() {
    local TAG=$1
    local CKPT=$2

    echo "" | tee -a "$EVAL_LOG"
    echo "===== ${TAG} =====" | tee -a "$EVAL_LOG"
    echo "  ckpt: $CKPT" | tee -a "$EVAL_LOG"
    echo "" | tee -a "$EVAL_LOG"

    TMP_YAML=$(mktemp /tmp/effort_eval_XXXXXX.yaml)
    cp "$YAML" "$TMP_YAML"

    python3 testall.py \
        --detector_path "$TMP_YAML" \
        --weights_path "$CKPT" \
        --test_datasets $TEST_DS \
        2>&1 | tee -a "$EVAL_LOG"

    rm -f "$TMP_YAML"

    echo "" | tee -a "$EVAL_LOG"
    echo "--- per-dataset:" | tee -a "$EVAL_LOG"
    grep -E "^(dataset:|video_auc|auc|acc):" "$EVAL_LOG" | tail -28 | tee -a "$EVAL_LOG"
    echo "" | tee -a "$EVAL_LOG"
}

echo "============================================================" | tee -a "$EVAL_LOG"
echo "  Evaluate: Pyramid RR+FF Experiment"                        | tee -a "$EVAL_LOG"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"                     | tee -a "$EVAL_LOG"
echo "============================================================" | tee -a "$EVAL_LOG"

do_eval "G1: RF generated + stripped"  "$CKPT_G1"
do_eval "G2: RF not generated"         "$CKPT_G2"

echo "" | tee -a "$EVAL_LOG"
echo "============================================================" | tee -a "$EVAL_LOG"
echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"                     | tee -a "$EVAL_LOG"
echo "  Log: $EVAL_LOG"                                            | tee -a "$EVAL_LOG"
echo "============================================================" | tee -a "$EVAL_LOG"

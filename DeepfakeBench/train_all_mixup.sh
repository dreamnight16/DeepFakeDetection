#!/bin/bash
# Train all modes (except TTA) with fixed hyperparameters γ=5 α=10 K=3
#   Mixup modes:   original / asymmetric / lap_pyramid / no-mixup
#   Selections:    hardest / random / mean  (asymmetric only)
#   Domains:       rgb / hf / lf / ycbcr_hf / ycbcr_lf
#   Margin loss:   off / add / replace
#   Opt wrapper:   null / sam / pcgrad
#
# Output:
#   train_<NAME>.log          per-run full training log
#   sweep_all.log             tab-separated summary: video_auc / auc / acc
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
LOG_DIR="./zhiyuanyan/logs/benchv2/icml25/release"
TRAIN_DATASET="FaceForensics++"
VAL_DATASET="Celeb-DF-v2"
TEST_DATASETS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"

GAMMA=5.0
ALPHA=10.0
K=3

SWEEP_LOG="sweep_all.log"
> "$SWEEP_LOG"

cp "$YAML" "${YAML}.bak"
trap "mv '${YAML}.bak' '$YAML'" EXIT

run_one() {
    local NAME="$1"  MODE="$2"     SEL="$3"  DOM="$4"
    local MLOSS="$5" OPTWRAP="$6"
    local TRAIN_LOG="train_${NAME}.log"

    echo "===== [$(date '+%H:%M:%S')] ${NAME} ====="

    TMP_YAML=$(mktemp /tmp/effort_XXXXXX.yaml)
    cp "$YAML" "$TMP_YAML"

    if [ "$MODE" = "none" ]; then
        sed -i "s/^use_mixup:.*/use_mixup: false/" "$TMP_YAML"
    else
        sed -i "s/^use_mixup:.*/use_mixup: true/"               "$TMP_YAML"
        sed -i "s/^mixup_mode:.*/mixup_mode: ${MODE}/"           "$TMP_YAML"
        sed -i "s/^mixup_selection:.*/mixup_selection: ${SEL}/"  "$TMP_YAML"
        sed -i "s/^mix_domain:.*/mix_domain: ${DOM}/"            "$TMP_YAML"
    fi
    sed -i "s/^mixup_k:.*/mixup_k: ${K}/"                       "$TMP_YAML"
    sed -i "s/^mixup_gamma:.*/mixup_gamma: ${GAMMA}/"           "$TMP_YAML"
    sed -i "s/^mixup_alpha:.*/mixup_alpha: ${ALPHA}/"           "$TMP_YAML"
    sed -i "s/^margin_loss_mode:.*/margin_loss_mode: ${MLOSS}/"    "$TMP_YAML"
    sed -i "s/^optimizer_wrapper:.*/optimizer_wrapper: ${OPTWRAP}/" "$TMP_YAML"

    echo "  [train] log -> $TRAIN_LOG"
    python3 ./training/train.py \
        --detector_path "$TMP_YAML" \
        --train_dataset "$TRAIN_DATASET" \
        --test_dataset "$VAL_DATASET" \
        > "$TRAIN_LOG" 2>&1 || {
        echo "$NAME | TRAIN_ERROR" | tee -a "$SWEEP_LOG"
        rm -f "$TMP_YAML"
        return
    }

    # Find best checkpoint (try avg first, then direct)
    CKPT=$(ls -td "${LOG_DIR}"/effort_*/test/avg/ckpt_best.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        CKPT=$(ls -td "${LOG_DIR}"/effort_*/test/"${VAL_DATASET}"/ckpt_best.pth 2>/dev/null | head -1)
    fi
    if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
        echo "$NAME | CKPT_NOT_FOUND" | tee -a "$SWEEP_LOG"
        rm -f "$TMP_YAML"
        return
    fi

    echo "  [eval] testing on: $TEST_DATASETS"
    OUT=$(python3 testall.py \
        --detector_path "$TMP_YAML" \
        --weights_path "$CKPT" \
        --test_datasets $TEST_DATASETS 2>/dev/null)

    V_AUC=$(echo "$OUT" | awk '/^video_auc:/{v=$2} END{print v}')
    AUC=$(echo "$OUT"   | awk '/^auc:/{v=$2} END{print v}')
    ACC=$(echo "$OUT"   | awk '/^acc:/{v=$2} END{print v}')

    echo "$NAME | video_auc=${V_AUC:-NA} | auc=${AUC:-NA} | acc=${ACC:-NA}" | tee -a "$SWEEP_LOG"
    rm -f "$TMP_YAML"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration entries: NAME|MODE|SEL|DOMAIN|MARGIN_LOSS|OPT_WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════
ENTRIES=(
    # ── A. Mixup domain sweep (asymmetric + hardest, all 5 domains) ────────
    "asym_hardest_rgb|asymmetric|hardest|rgb|off|null"
    "asym_hardest_hf|asymmetric|hardest|hf|off|null"
    "asym_hardest_lf|asymmetric|hardest|lf|off|null"
    "asym_hardest_ycbcr_hf|asymmetric|hardest|ycbcr_hf|off|null"
    "asym_hardest_ycbcr_lf|asymmetric|hardest|ycbcr_lf|off|null"

    # ── B. Mixup domain sweep (original, all 5 domains) ────────────────────
    "orig_rgb|original|random|rgb|off|null"
    "orig_hf|original|random|hf|off|null"
    "orig_lf|original|random|lf|off|null"
    "orig_ycbcr_hf|original|random|ycbcr_hf|off|null"
    "orig_ycbcr_lf|original|random|ycbcr_lf|off|null"

    # ── C. Selection ablation (asymmetric + rgb, random / mean) ────────────
    "asym_random_rgb|asymmetric|random|rgb|off|null"
    "asym_mean_rgb|asymmetric|mean|rgb|off|null"

    # ── D. Selection × domain (random + mean) for hf only ──────────────────
    "asym_random_hf|asymmetric|random|hf|off|null"
    "asym_mean_hf|asymmetric|mean|hf|off|null"

    # ── E. Lap pyramid ─────────────────────────────────────────────────────
    "lap_pyramid|lap_pyramid|hardest|rgb|off|null"

    # ── F. No-mixup baseline ──────────────────────────────────────────────
    "no_mixup|none|hardest|rgb|off|null"

    # ── G. Margin loss (with default mixup: asym+hardest+rgb) ─────────────
    "margin_add|asymmetric|hardest|rgb|add|null"
    "margin_replace|asymmetric|hardest|rgb|replace|null"

    # ── H. Optimizer wrapper (with default mixup: asym+hardest+rgb) ───────
    "optwr_sam|asymmetric|hardest|rgb|off|sam"
    "optwr_pcgrad|asymmetric|hardest|rgb|off|pcgrad"
)

TOTAL=${#ENTRIES[@]}
echo "Total modes: ${TOTAL}"
echo "Hyperparams: gamma=${GAMMA}  alpha=${ALPHA}  K=${K}"
echo "Train: ${TRAIN_DATASET}  Val: ${VAL_DATASET}"
echo "Test:  ${TEST_DATASETS}"
echo ""

run_idx=0
for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r NAME MODE SEL DOM MLOSS OPTWRAP <<< "$entry"
    run_idx=$((run_idx + 1))
    echo "[${run_idx}/${TOTAL}]"
    run_one "$NAME" "$MODE" "$SEL" "$DOM" "$MLOSS" "$OPTWRAP"
done

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Summary (fixed γ=${GAMMA} α=${ALPHA} K=${K})" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
printf "  %-30s | %-9s | %-9s | %s\n" \
    "name" "video_auc" "auc" "acc" | tee -a "$SWEEP_LOG"
echo "  ------------------------------|-----------|-----------|------" | tee -a "$SWEEP_LOG"

for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r NAME MODE SEL DOM MLOSS OPTWRAP <<< "$entry"
    BEST_LINE=$(grep "^${NAME} " "$SWEEP_LOG" 2>/dev/null | tail -1)
    if [ -n "$BEST_LINE" ]; then
        B_V=$(echo "$BEST_LINE"   | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
        B_AUC=$(echo "$BEST_LINE" | sed 's/.*auc=\([0-9.]*\).*/\1/')
        B_ACC=$(echo "$BEST_LINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
        printf "  %-30s | %-9s | %-9s | %s\n" \
            "$NAME" "${B_V:-NA}" "${B_AUC:-NA}" "${B_ACC:-NA}" | tee -a "$SWEEP_LOG"
    else
        echo "  ${NAME}  | no result" | tee -a "$SWEEP_LOG"
    fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "All ${TOTAL} modes done! Summary → ${SWEEP_LOG}" | tee -a "$SWEEP_LOG"

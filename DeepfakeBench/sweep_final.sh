#!/bin/bash
# Final sweep: all mix modes
#   asymmetric (hardest_k, K=3):  rgb / hf / lf / ycbcr_hf / ycbcr_lf
#   lap_pyramid
#   no-mixup baseline
# Two param combos: γ=1 α=5  and  γ=5 α=10
# Usage:
#   bash sweep_final.sh              # single GPU
#   bash sweep_final.sh 4            # 4 GPUs via DDP (torchrun)
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
LOG_DIR="./zhiyuanyan/logs/benchv2/icml25/release"
SWEEP_LOG="sweep_final.log"
TRAIN_DS="FaceForensics++"
VAL_DS="Celeb-DF-v2"
TEST_DS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"
NGPU=${1:-1}

> "$SWEEP_LOG"

run_one() {
    local TAG=$1 MODE=$2 G=$3 A=$4
    if [ "$MODE" = "no-mixup" ]; then
        local TRAIN_LOG="sweep_final_train_no_mixup.log"
    else
        local TRAIN_LOG="sweep_final_train_${MODE}_g${G}_a${A}.log"
    fi
    echo "===== $TAG ====="

    TMP_YAML=$(mktemp /tmp/effort_final_XXXXXX.yaml)
    cp "$YAML" "$TMP_YAML"

    if [ "$MODE" = "no-mixup" ]; then
        sed -i "s/^use_mixup:.*/use_mixup: false/" "$TMP_YAML"
    elif [ "$MODE" = "lap_pyramid" ]; then
        sed -i "s/^use_mixup:.*/use_mixup: true/"              "$TMP_YAML"
        sed -i "s/^mixup_mode:.*/mixup_mode: lap_pyramid/"     "$TMP_YAML"
        sed -i "s/^mixup_gamma:.*/mixup_gamma: ${G}/"          "$TMP_YAML"
        sed -i "s/^mixup_alpha:.*/mixup_alpha: ${A}/"          "$TMP_YAML"
    else
        sed -i "s/^use_mixup:.*/use_mixup: true/"               "$TMP_YAML"
        sed -i "s/^mixup_mode:.*/mixup_mode: asymmetric/"       "$TMP_YAML"
        sed -i "s/^mixup_selection:.*/mixup_selection: hardest/" "$TMP_YAML"
        sed -i "s/^mix_domain:.*/mix_domain: ${MODE}/"          "$TMP_YAML"
        sed -i "s/^mixup_k:.*/mixup_k: 3/"                      "$TMP_YAML"
        sed -i "s/^mixup_gamma:.*/mixup_gamma: ${G}/"           "$TMP_YAML"
        sed -i "s/^mixup_alpha:.*/mixup_alpha: ${A}/"           "$TMP_YAML"
    fi

    if [ "$NGPU" -gt 1 ]; then
        sed -i "s/^train_batchSize:.*/train_batchSize: $((32 * NGPU))/" "$TMP_YAML"
    fi

    echo "  [train] log -> $TRAIN_LOG"
    if [ "$NGPU" -gt 1 ]; then
        torchrun --nproc_per_node=$NGPU ./training/train.py \
            --ddp \
            --detector_path "$TMP_YAML" \
            --train_dataset "$TRAIN_DS" \
            --test_dataset "$VAL_DS" \
            > "$TRAIN_LOG" 2>&1 || { echo "$TAG | TRAIN_ERROR" | tee -a "$SWEEP_LOG"; rm -f "$TMP_YAML"; return; }
    else
        python3 ./training/train.py \
            --detector_path "$TMP_YAML" \
            --train_dataset "$TRAIN_DS" \
            --test_dataset "$VAL_DS" \
            > "$TRAIN_LOG" 2>&1 || { echo "$TAG | TRAIN_ERROR" | tee -a "$SWEEP_LOG"; rm -f "$TMP_YAML"; return; }
    fi

    CKPT=$(ls -td "${LOG_DIR}"/effort_*/test/avg/ckpt_best.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "$TAG | CKPT_NOT_FOUND" | tee -a "$SWEEP_LOG"
        rm -f "$TMP_YAML"
        return
    fi

    echo "  [eval] testing on: $TEST_DS"
    OUT=$(python3 testall.py \
        --detector_path "$TMP_YAML" \
        --weights_path "$CKPT" \
        --test_datasets $TEST_DS 2>/dev/null)

    V_AUC=$(echo "$OUT" | awk '/^video_auc:/{v=$2} END{print v}')
    AUC=$(echo "$OUT"   | awk '/^auc:/{v=$2} END{print v}')
    ACC=$(echo "$OUT"   | awk '/^acc:/{v=$2} END{print v}')

    echo "$TAG | video_auc=${V_AUC:-NA} | auc=${AUC:-NA} | acc=${ACC:-NA}" | tee -a "$SWEEP_LOG"
    rm -f "$TMP_YAML"
}

# 5 mix_domain modes (asymmetric + hardest, K=3) + lap_pyramid
MODES=("rgb" "hf" "lf" "ycbcr_hf" "ycbcr_lf" "lap_pyramid")
COMBOS=("1.0 5.0" "5.0 10.0")
TOTAL=$(( ${#MODES[@]} * ${#COMBOS[@]} + 1 ))   # +1 for no-mixup baseline

# ── Param sweep ──────────────────────────────────────────────────────────────
run_idx=0
for COMBO in "${COMBOS[@]}"; do
    read G A <<< "$COMBO"
    for MODE in "${MODES[@]}"; do
        run_idx=$((run_idx + 1))
        run_one "[$run_idx/$TOTAL] mode=$MODE gamma=$G alpha=$A" "$MODE" "$G" "$A"
    done
done

# ── no-mixup baseline ────────────────────────────────────────────────────────
run_idx=$((run_idx + 1))
run_one "[$run_idx/$TOTAL] mode=no-mixup" "no-mixup" "0" "0"

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Summary                          ($(date '+%Y-%m-%d %H:%M'))" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
printf "  %-14s | %-5s | %-5s | %-9s | %-9s | %s\n" \
    "mode" "gamma" "alpha" "video_auc" "auc" "acc" | tee -a "$SWEEP_LOG"
echo "  --------------|-------|-------|-----------|-----------|------" | tee -a "$SWEEP_LOG"

for COMBO in "${COMBOS[@]}"; do
    read G A <<< "$COMBO"
    for MODE in "${MODES[@]}"; do
        LINE=$(grep "mode=$MODE gamma=$G alpha=$A " "$SWEEP_LOG" 2>/dev/null | head -1)
        if [ -n "$LINE" ]; then
            B_V=$(echo "$LINE"   | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
            B_AUC=$(echo "$LINE" | sed 's/.*auc=\([0-9.]*\).*/\1/')
            B_ACC=$(echo "$LINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
            printf "  %-14s | %-5s | %-5s | %-9s | %-9s | %s\n" \
                "$MODE" "$G" "$A" "${B_V:-NA}" "${B_AUC:-NA}" "${B_ACC:-NA}" | tee -a "$SWEEP_LOG"
        else
            echo "  $MODE  g=$G a=$A  | no result" | tee -a "$SWEEP_LOG"
        fi
    done
done

# no-mixup row
NOLINE=$(grep "mode=no-mixup " "$SWEEP_LOG" 2>/dev/null | head -1)
if [ -n "$NOLINE" ]; then
    B_V=$(echo "$NOLINE"   | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
    B_AUC=$(echo "$NOLINE" | sed 's/.*auc=\([0-9.]*\).*/\1/')
    B_ACC=$(echo "$NOLINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
    printf "  %-14s | %-5s | %-5s | %-9s | %-9s | %s\n" \
        "no-mixup" "—" "—" "${B_V:-NA}" "${B_AUC:-NA}" "${B_ACC:-NA}" | tee -a "$SWEEP_LOG"
else
    echo "  no-mixup  | no result" | tee -a "$SWEEP_LOG"
fi

echo "" | tee -a "$SWEEP_LOG"
echo "=== All $TOTAL runs done.  Log saved to $SWEEP_LOG ===" | tee -a "$SWEEP_LOG"

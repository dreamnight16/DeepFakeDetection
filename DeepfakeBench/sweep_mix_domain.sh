#!/bin/bash
# Mix Domain sweep: coarse grid (gamma x alpha) for each mix_domain mode
# K=3 fixed, then compare best result per mode
# Usage:
#   bash sweep_mix_domain.sh              # single GPU
#   bash sweep_mix_domain.sh 4            # 4 GPUs via DDP (torchrun)
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
LOG_DIR="./zhiyuanyan/logs/benchv2/icml25/release"
SWEEP_LOG="sweep_mix_domain.log"
TRAIN_DS="FaceForensics++"
VAL_DS="Celeb-DF-v2"
TEST_DS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"
NGPU=${1:-1}

> "$SWEEP_LOG"

MODES=("rgb" "hf" "ycbcr_hf")
GAMMAS=(3.0)
ALPHAS=(10.0)

run_one() {
    local MODE=$1 G=$2 A=$3 TAG=$4
    local TRAIN_LOG="sweep_domain_train_${MODE}_g${G}_a${A}.log"
    echo "===== $TAG ====="

    TMP_YAML=$(mktemp /tmp/effort_domain_XXXXXX.yaml)
    cp "$YAML" "$TMP_YAML"
    sed -i "s/^use_mixup:.*/use_mixup: true/"               "$TMP_YAML"
    sed -i "s/^mixup_mode:.*/mixup_mode: asymmetric/"       "$TMP_YAML"
    sed -i "s/^mixup_selection:.*/mixup_selection: hardest/" "$TMP_YAML"
    sed -i "s/^mix_domain:.*/mix_domain: ${MODE}/"          "$TMP_YAML"
    sed -i "s/^mixup_k:.*/mixup_k: 3/"                      "$TMP_YAML"
    sed -i "s/^mixup_gamma:.*/mixup_gamma: ${G}/"           "$TMP_YAML"
    sed -i "s/^mixup_alpha:.*/mixup_alpha: ${A}/"            "$TMP_YAML"

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

# ====== Coarse sweep per mode (4 gamma x 4 alpha = 16 per mode) ======
PER_MODE=16
TOTAL=$(( ${#MODES[@]} * PER_MODE ))
run_idx=0

for MODE in "${MODES[@]}"; do
    echo "=== Mode: $MODE  (coarse sweep, K=3) ===" | tee -a "$SWEEP_LOG"
    for G in "${GAMMAS[@]}"; do
        for A in "${ALPHAS[@]}"; do
            run_idx=$((run_idx + 1))
            run_one "$MODE" "$G" "$A" "[$run_idx/$TOTAL] mode=$MODE gamma=$G alpha=$A"
        done
    done
    echo "" | tee -a "$SWEEP_LOG"
done

# ====== Final: best per mode by video_auc ======
echo "" | tee -a "$SWEEP_LOG"
echo "=== Final: Best per mode by video_auc ===" | tee -a "$SWEEP_LOG"
echo "  mode       | gamma | alpha | video_auc | auc       | acc" | tee -a "$SWEEP_LOG"
echo "  -----------|-------|-------|-----------|-----------|-----" | tee -a "$SWEEP_LOG"

for MODE in "${MODES[@]}"; do
    BEST_LINE=$(grep "mode=$MODE " "$SWEEP_LOG" | grep 'video_auc=' | while read l; do
        v=$(echo "$l" | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
        printf '%s\t%s\n' "$v" "$l"
    done | sort -nr -k1 | head -1 | cut -f2-)

    if [ -n "$BEST_LINE" ]; then
        B_G=$(echo "$BEST_LINE"   | sed 's/.*gamma=\([0-9.]*\).*/\1/')
        B_A=$(echo "$BEST_LINE"   | sed 's/.*alpha=\([0-9.]*\).*/\1/')
        B_V=$(echo "$BEST_LINE"   | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
        B_AUC=$(echo "$BEST_LINE" | sed 's/.*auc=\([0-9.]*\).*/\1/')
        B_ACC=$(echo "$BEST_LINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
        printf "  %-10s | %-5s | %-5s | %-9s | %-9s | %s\n" \
            "$MODE" "$B_G" "$B_A" "${B_V:-NA}" "${B_AUC:-NA}" "${B_ACC:-NA}" | tee -a "$SWEEP_LOG"
    else
        echo "  $MODE  | no valid result" | tee -a "$SWEEP_LOG"
    fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "=== Log saved to $SWEEP_LOG ===" | tee -a "$SWEEP_LOG"

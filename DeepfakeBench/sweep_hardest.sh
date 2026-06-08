#!/bin/bash
# Hardest Mixup sweep: Phase 1 coarse grid -> Phase 2 fine around top-3
# Usage:
#   bash sweep_hardest.sh              # single GPU
#   bash sweep_hardest.sh 4            # 4 GPUs via DDP (torchrun)
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
LOG_DIR="./zhiyuanyan/logs/benchv2/icml25/release"
SWEEP_LOG="sweep_hardest.log"
TRAIN_DS="FaceForensics++"
VAL_DS="Celeb-DF-v2"
TEST_DS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"
NGPU=${1:-1}   # default 1 GPU, pass e.g. 4 for DDP

> "$SWEEP_LOG"

run_one() {
    local K=$1 G=$2 A=$3 TAG=$4
    echo "===== $TAG ====="

    TMP_YAML=$(mktemp /tmp/effort_sweep_XXXXXX.yaml)
    cp "$YAML" "$TMP_YAML"
    sed -i "s/^mixup_mode:.*/mixup_mode: asymmetric/"       "$TMP_YAML"
    sed -i "s/^mixup_selection:.*/mixup_selection: hardest/" "$TMP_YAML"
    sed -i "s/^use_mixup:.*/use_mixup: true/"               "$TMP_YAML"
    sed -i "s/^mixup_k:.*/mixup_k: ${K}/"                   "$TMP_YAML"
    sed -i "s/^mixup_gamma:.*/mixup_gamma: ${G}/"           "$TMP_YAML"
    sed -i "s/^mixup_alpha:.*/mixup_alpha: ${A}/"           "$TMP_YAML"

    TRAIN_LOG="train_$(echo "$TAG" | tr '/ ' '_-').log"
    if [ "$NGPU" -gt 1 ]; then
        torchrun --nproc_per_node=$NGPU ./training/train.py \
            --ddp \
            --detector_path "$TMP_YAML" \
            --train_dataset "$TRAIN_DS" \
            --test_dataset "$VAL_DS" \
            2>&1 | tee "$TRAIN_LOG" || { echo "$TAG | TRAIN_ERROR" | tee -a "$SWEEP_LOG"; rm -f "$TMP_YAML"; return; }
    else
        python3 ./training/train.py \
            --detector_path "$TMP_YAML" \
            --train_dataset "$TRAIN_DS" \
            --test_dataset "$VAL_DS" \
            2>&1 | tee "$TRAIN_LOG" || { echo "$TAG | TRAIN_ERROR" | tee -a "$SWEEP_LOG"; rm -f "$TMP_YAML"; return; }
    fi

    CKPT=$(ls -td "${LOG_DIR}"/effort_*/test/avg/ckpt_best.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "$TAG | CKPT_NOT_FOUND" | tee -a "$SWEEP_LOG"
        rm -f "$TMP_YAML"
        return
    fi

    OUT=$(python3 testall.py \
        --detector_path "$TMP_YAML" \
        --weights_path "$CKPT" \
        --test_datasets $TEST_DS 2>/dev/null)

    V_AUC=$(echo "$OUT" | awk '/^video_auc:/{v=$2} END{print v}')
    AUC=$(echo "$OUT"   | awk '/^auc:/{v=$2} END{print v}')
    ACC=$(echo "$OUT"   | awk '/^acc:/{v=$2} END{print v}')

    echo "$TAG | video_auc=${V_AUC:-NA} | auc=${AUC:-NA} acc=${ACC:-NA}" | tee -a "$SWEEP_LOG"
    rm -f "$TMP_YAML"
}

# ====== Phase 1: Coarse Grid (4x4x4 = 64) ======
echo "=== Phase 1: Coarse Sweep (${NGPU} GPU(s)) ===" | tee -a "$SWEEP_LOG"
i=0; TOTAL=64
for K in 1 2 3 5; do
for G in 0.5 1.0 3.0 5.0; do
for A in 1.0 3.0 5.0 10.0; do
    i=$((i+1))
    run_one "$K" "$G" "$A" "[$i/$TOTAL] K=$K gamma=$G alpha=$A"
done
done
done

# ====== Phase 2: Fine Sweep around Top-3 ======
echo "" | tee -a "$SWEEP_LOG"
echo "=== Phase 2: Fine Sweep ===" | tee -a "$SWEEP_LOG"

TOP3=$(grep '^\[' "$SWEEP_LOG" | while read l; do
    v=$(echo "$l" | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
    printf '%s\t%s\n' "$v" "$l"
done | sort -nr -k1 | head -3 | cut -f2-)
echo "=== Phase 1 Top-3 ===" | tee -a "$SWEEP_LOG"
echo "$TOP3" | tee -a "$SWEEP_LOG"

declare -A SEEN
i=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    K0=$(echo "$line" | sed 's/.*K=\([0-9]*\).*/\1/')
    G0=$(echo "$line" | sed 's/.*gamma=\([0-9.]*\).*/\1/')
    A0=$(echo "$line" | sed 's/.*alpha=\([0-9.]*\).*/\1/')

    for K in $(seq $((K0>1?K0-1:1)) $((K0+1))); do
        for G in $(awk -v g="$G0" 'BEGIN{printf "%.1f %.1f %.1f", g-0.5, g, g+0.5}'); do
            for A in $(awk -v a="$A0" 'BEGIN{printf "%.1f %.1f %.1f", a-2, a, a+2}'); do
                awk -v g="$G" 'BEGIN{exit(g<=0.1)}' || continue
                awk -v a="$A" 'BEGIN{exit(a<=0.1)}' || continue
                KEY="${K}_${G}_${A}"
                [ "${SEEN[$KEY]}" = "1" ] && continue
                SEEN[$KEY]=1
                i=$((i+1))
                run_one "$K" "$G" "$A" "[fine $i] K=$K gamma=$G alpha=$A"
            done
        done
    done
done <<< "$TOP3"

# ====== Final Summary ======
echo "" | tee -a "$SWEEP_LOG"
echo "=== Final: Top-10 by video_auc ===" | tee -a "$SWEEP_LOG"
grep 'video_auc=' "$SWEEP_LOG" | while read l; do
    v=$(echo "$l" | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
    printf '%s\t%s\n' "$v" "$l"
done | sort -nr -k1 | head -10 | cut -f2- | tee -a "$SWEEP_LOG"
echo "=== All Done ===" | tee -a "$SWEEP_LOG"

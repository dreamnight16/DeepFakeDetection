#!/bin/bash
# ===========================================================================
# Sweep: Laplacian Pyramid Mixup — sampler comparison
#   1) pyramid + v1 sampler (real_ratio=0.5, baseline)
#   2) pyramid + v2 sampler (RF-only pairs)
#   3) pyramid + v1 sampler + high-real-image ratios (0.7, 0.85)
#   4) no-mixup baseline (v1 sampler)
#
# Fixed: gamma=1.0, alpha=5.0, trainer_v2
# ===========================================================================
# Usage:
#   bash sweep_pyramid_sampler.sh              # single GPU
#   bash sweep_pyramid_sampler.sh 4            # 4-GPU DDP (torchrun)
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
LOG_DIR_BASE="./zhiyuanyan/logs/benchv2/icml25/pyramid_sampler_sweep"
SWEEP_LOG="sweep_pyramid_sampler.log"
TRAIN_DS="FaceForensics++"
VAL_DS="Celeb-DF-v2"
TEST_DS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"
NGPU=${1:-1}
GAMMA=1.0
ALPHA=5.0
TRAINER=trainer_v2

# ── Create patched train.py copy (safe for concurrent runs) ──────────────
TRAIN_PY="./training/train_pyramid_sweep.py"
cp ./training/train.py "$TRAIN_PY"
sed -i "s/^from trainer\..* import Trainer/from trainer.${TRAINER} import Trainer/" "$TRAIN_PY"
trap "rm -f $TRAIN_PY" EXIT

> "$SWEEP_LOG"

run_one() {
    local TAG=$1            # display tag
    local SAMPLER=$2        # "v1" | "v2" | "no-mixup"
    local REAL_RATIO=${3:-0.5}

    local SAFE_TAG=$(echo "$TAG" | sed 's/[][ \/]/_/g')
    local TRAIN_LOG="sweep_pyramid_${SAFE_TAG}_train.log"
    echo "===== $TAG ====="

    TMP_YAML=$(mktemp /tmp/effort_pyramid_sweep_XXXXXX.yaml)
    cp "$YAML" "$TMP_YAML"

    # ── Set sweep-specific log_dir ───────────────────────────────────────
    local LOG_DIR="${LOG_DIR_BASE}/${SAFE_TAG}"
    mkdir -p "$LOG_DIR"
    sed -i "s|^log_dir:.*|log_dir: ${LOG_DIR}|" "$TMP_YAML"

    # ── Configure sampler and mixup ──────────────────────────────────────
    if [ "$SAMPLER" = "no-mixup" ]; then
        sed -i "s/^use_mixup:.*/use_mixup: false/"                       "$TMP_YAML"
        sed -i "s/^balance_sampler_v2:.*/balance_sampler_v2: false/"      "$TMP_YAML"
        sed -i "s/^use_balance_batch_sampler:.*/use_balance_batch_sampler: true/" "$TMP_YAML"
        sed -i "s/^sampler_real_ratio:.*/sampler_real_ratio: 0.5/"        "$TMP_YAML"
    elif [ "$SAMPLER" = "v2" ]; then
        sed -i "s/^use_mixup:.*/use_mixup: true/"                         "$TMP_YAML"
        sed -i "s/^mixup_mode:.*/mixup_mode: lap_pyramid/"                "$TMP_YAML"
        sed -i "s/^mixup_gamma:.*/mixup_gamma: ${GAMMA}/"                 "$TMP_YAML"
        sed -i "s/^mixup_alpha:.*/mixup_alpha: ${ALPHA}/"                 "$TMP_YAML"
        sed -i "s/^balance_sampler_v2:.*/balance_sampler_v2: true/"       "$TMP_YAML"
        sed -i "s/^use_balance_batch_sampler:.*/use_balance_batch_sampler: false/" "$TMP_YAML"
    else  # v1
        sed -i "s/^use_mixup:.*/use_mixup: true/"                         "$TMP_YAML"
        sed -i "s/^mixup_mode:.*/mixup_mode: lap_pyramid/"                "$TMP_YAML"
        sed -i "s/^mixup_gamma:.*/mixup_gamma: ${GAMMA}/"                 "$TMP_YAML"
        sed -i "s/^mixup_alpha:.*/mixup_alpha: ${ALPHA}/"                 "$TMP_YAML"
        sed -i "s/^balance_sampler_v2:.*/balance_sampler_v2: false/"      "$TMP_YAML"
        sed -i "s/^use_balance_batch_sampler:.*/use_balance_batch_sampler: true/" "$TMP_YAML"
        sed -i "s/^sampler_real_ratio:.*/sampler_real_ratio: ${REAL_RATIO}/" "$TMP_YAML"
    fi

    if [ "$NGPU" -gt 1 ]; then
        sed -i "s/^train_batchSize:.*/train_batchSize: $((32 * NGPU))/" "$TMP_YAML"
    fi

    # ── Train ────────────────────────────────────────────────────────────
    echo "  [train] log -> $TRAIN_LOG"
    if [ "$NGPU" -gt 1 ]; then
        torchrun --nproc_per_node=$NGPU "$TRAIN_PY" \
            --ddp \
            --detector_path "$TMP_YAML" \
            --train_dataset "$TRAIN_DS" \
            --test_dataset "$VAL_DS" \
            > "$TRAIN_LOG" 2>&1 || { echo "$TAG | TRAIN_ERROR" | tee -a "$SWEEP_LOG"; rm -f "$TMP_YAML"; return; }
    else
        python3 "$TRAIN_PY" \
            --detector_path "$TMP_YAML" \
            --train_dataset "$TRAIN_DS" \
            --test_dataset "$VAL_DS" \
            > "$TRAIN_LOG" 2>&1 || { echo "$TAG | TRAIN_ERROR" | tee -a "$SWEEP_LOG"; rm -f "$TMP_YAML"; return; }
    fi

    # ── Find checkpoint ──────────────────────────────────────────────────
    CKPT=$(ls -td "${LOG_DIR}"/effort_*/test/avg/ckpt_best.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "$TAG | CKPT_NOT_FOUND" | tee -a "$SWEEP_LOG"
        rm -f "$TMP_YAML"
        return
    fi

    # ── Evaluate ─────────────────────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════════════
# Experiment matrix
# ═══════════════════════════════════════════════════════════════════════════
# Format: "TAG" "SAMPLER" "REAL_RATIO"
RUNS=(
    # 1) pyramid + v1 (baseline, equal real/fake)
    "pyramid_v1_rr50     v1   0.5"
    # 2) pyramid + v2 (RF-only pairs)
    "pyramid_v2          v2    —"
    # 3) pyramid + v1 + high-real-image ratios
    "pyramid_v1_rr70     v1   0.7"
    "pyramid_v1_rr85     v1   0.85"
    # 4) no-mixup baseline
    "no_mixup            no-mixup  —"
)

TOTAL=${#RUNS[@]}
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Pyramid Sampler Sweep — $TOTAL configs"                     | tee -a "$SWEEP_LOG"
echo "  Fixed: γ=$GAMMA  α=$ALPHA  trainer=$TRAINER"               | tee -a "$SWEEP_LOG"
echo "  GPU mode: NGPU=$NGPU"                                       | tee -a "$SWEEP_LOG"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"                       | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

for i in "${!RUNS[@]}"; do
    idx=$((i + 1))
    read TAG SAMPLER RR <<< "${RUNS[$i]}"
    # Replace "—" with a safe default for no-mixup
    RATIO_VAL=$( [ "$RR" = "—" ] && echo "n/a" || echo "$RR" )
    run_one "[$idx/$TOTAL] $TAG" "$SAMPLER" "$RATIO_VAL"
    echo "" | tee -a "$SWEEP_LOG"
done

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Summary — Pyramid Sampler Comparison  ($(date '+%Y-%m-%d %H:%M'))" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
printf "  %-20s | %-7s | %-4s | %-9s | %-9s | %s\n" \
    "config" "sampler" "rr" "video_auc" "auc" "acc" | tee -a "$SWEEP_LOG"
echo "  ---------------------|---------|------|-----------|-----------|------" | tee -a "$SWEEP_LOG"

for RUN_LINE in "${RUNS[@]}"; do
    read TAG SAMPLER RR <<< "$RUN_LINE"
    LINE=$(grep "^${TAG} |" "$SWEEP_LOG" 2>/dev/null | head -1)
    RR_DISPLAY=$( [ "$RR" = "—" ] && echo "—" || echo "$RR" )
    if [ -n "$LINE" ]; then
        B_V=$(echo "$LINE"   | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
        B_AUC=$(echo "$LINE" | sed 's/.*auc=\([0-9.]*\).*/\1/')
        B_ACC=$(echo "$LINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
        printf "  %-20s | %-7s | %-4s | %-9s | %-9s | %s\n" \
            "$TAG" "$SAMPLER" "$RR_DISPLAY" "${B_V:-NA}" "${B_AUC:-NA}" "${B_ACC:-NA}" | tee -a "$SWEEP_LOG"
    else
        printf "  %-20s | %-7s | %-4s | %-9s | %-9s | %s\n" \
            "$TAG" "$SAMPLER" "$RR_DISPLAY" "—" "—" "—" | tee -a "$SWEEP_LOG"
    fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "=== All $TOTAL runs done.  Log saved to $SWEEP_LOG ===" | tee -a "$SWEEP_LOG"

#!/bin/bash
# ===========================================================================
# Sweep: Laplacian Pyramid Variants — personal experiments
#   1) lap_pyramid_fake_base   — fake image as coarse-structure base
#   2) lap_pyramid_hardest     — real base + K-candidate hardest selection
#   3) lap_pyramid_fb_hardest  — fake base + K-candidate hardest selection
#
# Fixed: gamma=1.0, alpha=5.0, trainer_v2, v1 sampler (real_ratio=0.5)
# ===========================================================================
# Usage:
#   bash sweep_pyramid_variants.sh              # single GPU
#   bash sweep_pyramid_variants.sh 4            # 4-GPU DDP (torchrun)
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
LOG_DIR_BASE="./zhiyuanyan/logs/benchv2/icml25/pyramid_variants_sweep"
SWEEP_LOG="sweep_pyramid_variants.log"
TRAIN_DS="FaceForensics++"
VAL_DS="Celeb-DF-v2"
TEST_DS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"
NGPU=${1:-1}
GAMMA=1.0
ALPHA=5.0
TRAINER=trainer_v2

# ── Create patched train.py copy (safe for concurrent runs) ──────────────
TRAIN_PY="./training/train_pyramid_variants_sweep.py"
cp ./training/train.py "$TRAIN_PY"
sed -i "s/^from trainer\..* import Trainer/from trainer.${TRAINER} import Trainer/" "$TRAIN_PY"
trap "rm -f $TRAIN_PY" EXIT

> "$SWEEP_LOG"

run_one() {
    local TAG=$1            # display tag
    local MODE=$2           # mixup_mode value
    local K_VAL=${3:-3}     # mixup_k (only used for hardest modes)
    local REAL_RATIO=${4:-0.5}

    local SAFE_TAG=$(echo "$TAG" | sed 's/[][ \/]/_/g')
    local TRAIN_LOG="sweep_pyramid_variants_${SAFE_TAG}_train.log"
    echo "===== $TAG ====="

    TMP_YAML=$(mktemp /tmp/effort_pyramid_variants_XXXXXX.yaml)
    cp "$YAML" "$TMP_YAML"

    # ── Set sweep-specific log_dir ───────────────────────────────────────
    local LOG_DIR="${LOG_DIR_BASE}/${SAFE_TAG}"
    mkdir -p "$LOG_DIR"
    sed -i "s|^log_dir:.*|log_dir: ${LOG_DIR}|" "$TMP_YAML"

    # ── Common: enable mixup + pyramid + v1 sampler ──────────────────────
    sed -i "s/^use_mixup:.*/use_mixup: true/"                         "$TMP_YAML"
    sed -i "s/^mixup_mode:.*/mixup_mode: ${MODE}/"                    "$TMP_YAML"
    sed -i "s/^mixup_gamma:.*/mixup_gamma: ${GAMMA}/"                 "$TMP_YAML"
    sed -i "s/^mixup_alpha:.*/mixup_alpha: ${ALPHA}/"                 "$TMP_YAML"
    sed -i "s/^mixup_k:.*/mixup_k: ${K_VAL}/"                         "$TMP_YAML"
    sed -i "s/^mixup_selection:.*/mixup_selection: hardest/"           "$TMP_YAML"
    sed -i "s/^balance_sampler_v2:.*/balance_sampler_v2: false/"      "$TMP_YAML"
    sed -i "s/^use_balance_batch_sampler:.*/use_balance_batch_sampler: true/" "$TMP_YAML"
    sed -i "s/^sampler_real_ratio:.*/sampler_real_ratio: ${REAL_RATIO}/" "$TMP_YAML"

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

    # ── Evaluate ────────────────────────────────────────────────────────
    echo "  [eval] testing on: $TEST_DS"
    EVAL_LOG="${LOG_DIR}/eval_${SAFE_TAG}.log"
    python3 testall.py \
        --detector_path "$TMP_YAML" \
        --weights_path "$CKPT" \
        --test_datasets $TEST_DS \
        > "$EVAL_LOG" 2>&1

    # Per-dataset breakdown
    echo "  [eval] per-dataset:" | tee -a "$SWEEP_LOG"
    awk '/^dataset: /{ds=$0} /^(auc|video_auc|acc):/{printf "    %s  %s\n", ds, $0}' "$EVAL_LOG" | tee -a "$SWEEP_LOG"

    # Show any warnings
    if grep -q "WARNING" "$EVAL_LOG"; then
        echo "  [eval] WARNINGS:" | tee -a "$SWEEP_LOG"
        grep "WARNING" "$EVAL_LOG" | sed 's/^/    /' | tee -a "$SWEEP_LOG"
    fi

    # Extract avg metrics (last occurrence = average block)
    V_AUC=$(awk '/^video_auc:/{v=$2} END{print v}' "$EVAL_LOG")
    AUC=$(awk   '/^auc:/{v=$2} END{print v}' "$EVAL_LOG")
    ACC=$(awk   '/^acc:/{v=$2} END{print v}' "$EVAL_LOG")

    echo "$TAG | video_auc=${V_AUC:-NA} | auc=${AUC:-NA} | acc=${ACC:-NA}" | tee -a "$SWEEP_LOG"
    rm -f "$TMP_YAML"
}

# ═══════════════════════════════════════════════════════════════════════════
# Experiment matrix
# ═══════════════════════════════════════════════════════════════════════════
# Format: "TAG" "MIXUP_MODE" "K" "REAL_RATIO"
RUNS=(
    # ── Variant 1: fake-base (no K needed, K=1 is a no-op placeholder) ──
    "fake_base         lap_pyramid_fake_base   1   0.5"

    # ── Variant 2: real base + hardest K-selection ───────────────────────
    "pyramid_hardest_k3    lap_pyramid_hardest     3   0.5"

    # ── Variant 3: fake base + hardest K-selection ───────────────────────
    "fb_hardest_k3         lap_pyramid_fb_hardest  3   0.5"
)

TOTAL=${#RUNS[@]}
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Pyramid Variants Sweep — $TOTAL configs"                     | tee -a "$SWEEP_LOG"
echo "  Fixed: γ=$GAMMA  α=$ALPHA  trainer=$TRAINER"               | tee -a "$SWEEP_LOG"
echo "  GPU mode: NGPU=$NGPU"                                       | tee -a "$SWEEP_LOG"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"                       | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

for i in "${!RUNS[@]}"; do
    idx=$((i + 1))
    read TAG MODE KV RR <<< "${RUNS[$i]}"
    run_one "[$idx/$TOTAL] $TAG" "$MODE" "$KV" "$RR"
    echo "" | tee -a "$SWEEP_LOG"
done

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Summary — Pyramid Variants  ($(date '+%Y-%m-%d %H:%M'))" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
printf "  %-22s | %-25s | %-3s | %-9s | %-9s | %s\n" \
    "config" "mode" "K" "video_auc" "auc" "acc" | tee -a "$SWEEP_LOG"
echo "  ------------------------|---------------------------|-----|-----------|-----------|------" | tee -a "$SWEEP_LOG"

for RUN_LINE in "${RUNS[@]}"; do
    read TAG MODE KV RR <<< "$RUN_LINE"
    LINE=$(grep "^${TAG} |" "$SWEEP_LOG" 2>/dev/null | head -1)
    if [ -n "$LINE" ]; then
        B_V=$(echo "$LINE"   | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
        B_AUC=$(echo "$LINE" | sed 's/.*auc=\([0-9.]*\).*/\1/')
        B_ACC=$(echo "$LINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
        printf "  %-22s | %-25s | %-3s | %-9s | %-9s | %s\n" \
            "$TAG" "$MODE" "$KV" "${B_V:-NA}" "${B_AUC:-NA}" "${B_ACC:-NA}" | tee -a "$SWEEP_LOG"
    else
        printf "  %-22s | %-25s | %-3s | %-9s | %-9s | %s\n" \
            "$TAG" "$MODE" "$KV" "—" "—" "—" | tee -a "$SWEEP_LOG"
    fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "=== All $TOTAL runs done.  Log saved to $SWEEP_LOG ===" | tee -a "$SWEEP_LOG"

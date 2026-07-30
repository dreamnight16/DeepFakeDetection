#!/bin/bash
# ===========================================================================
# Sweep: Beta(2,5) asymmetric distribution on top of top+l1
#
#   Config 1: Beta(2,5)  — asymmetric, left-heavy, more fake injection into G_K
#   Config 2: λ = 1 − Beta(2,5)  — right-heavy, less fake injection
#
#   Base: lap_pyramid_label1_top (G_K mixed, label forced to 1/fake)
#   Fixed: gamma=1.0, v1 sampler (real_ratio=0.30)
# ===========================================================================
# Usage:
#   bash sweep_beta25_top_l1.sh              # single GPU
#   bash sweep_beta25_top_l1.sh 4            # 4-GPU DDP (torchrun)
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
LOG_DIR_BASE="./zhiyuanyan/logs/benchv2/icml25/beta25_top_l1_sweep"
SWEEP_LOG="sweep_beta25_top_l1.log"
TRAIN_DS="FaceForensics++"
VAL_DS="Celeb-DF-v2"
TEST_DS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"
NGPU=${1:-1}
GAMMA=1.0
ALPHA=2          # Beta α
BETA_B=5         # Beta β  →  Beta(2,5)

# ── Create patched train.py copy (safe for concurrent runs) ──────────────
TRAIN_PY="./training/train_beta25_top_l1_sweep.py"
cp ./training/train.py "$TRAIN_PY"
sed -i "s/^from trainer\..* import Trainer/from trainer.trainer_v2 import Trainer/" "$TRAIN_PY"
trap "rm -f $TRAIN_PY" EXIT

> "$SWEEP_LOG"

run_one() {
    local TAG=$1            # display tag
    local FLIP=$2           # "false" or "true" for mixup_beta_flip

    local SAFE_TAG=$(echo "$TAG" | sed 's/[][ \/]/_/g')
    local TRAIN_LOG="sweep_beta25_${SAFE_TAG}_train.log"
    echo "===== $TAG ====="

    TMP_YAML=$(mktemp /tmp/effort_beta25_XXXXXX.yaml)
    cp "$YAML" "$TMP_YAML"

    # ── Set sweep-specific log_dir ───────────────────────────────────────
    local LOG_DIR="${LOG_DIR_BASE}/${SAFE_TAG}"
    mkdir -p "$LOG_DIR"
    sed -i "s|^log_dir:.*|log_dir: ${LOG_DIR}|" "$TMP_YAML"

    # ── Mixup: top+l1 base + Beta(2,5) ───────────────────────────────────
    sed -i "s/^use_mixup:.*/use_mixup: true/"                         "$TMP_YAML"
    sed -i "s/^mixup_mode:.*/mixup_mode: lap_pyramid_label1_top/"     "$TMP_YAML"
    sed -i "s/^mixup_gamma:.*/mixup_gamma: ${GAMMA}/"                 "$TMP_YAML"
    sed -i "s/^mixup_alpha:.*/mixup_alpha: ${ALPHA}/"                 "$TMP_YAML"
    sed -i "s/^mixup_beta_b:.*/mixup_beta_b: ${BETA_B}/"              "$TMP_YAML"
    sed -i "s/^mixup_beta_flip:.*/mixup_beta_flip: ${FLIP}/"          "$TMP_YAML"

    # ── Sampler: v1 balance (real_ratio=0.30) ──────────────────────────────
    sed -i "s/^balance_sampler_v2:.*/balance_sampler_v2: false/"      "$TMP_YAML"
    sed -i "s/^use_balance_batch_sampler:.*/use_balance_batch_sampler: true/" "$TMP_YAML"
    sed -i "s/^sampler_real_ratio:.*/sampler_real_ratio: 0.30/"       "$TMP_YAML"

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
# 2 experiments: Beta(2,5) × flip on top+l1
# ═══════════════════════════════════════════════════════════════════════════
# Format: "TAG" "beta_flip"
RUNS=(
    "top_l1_Beta25              false"
    "top_l1_Beta25_flip          true"
)

TOTAL=${#RUNS[@]}
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Beta(2,5) on top+l1 — $TOTAL configs"                      | tee -a "$SWEEP_LOG"
echo "  Base: lap_pyramid_label1_top (G_K mixed, label=1)"         | tee -a "$SWEEP_LOG"
echo "  Run 1: λ ~ Beta(2,5)       → q ≈ 0.714 (more fake)"       | tee -a "$SWEEP_LOG"
echo "  Run 2: λ = 1 − Beta(2,5)   → q ≈ 0.286 (less fake)"       | tee -a "$SWEEP_LOG"
echo "  Fixed: γ=$GAMMA  a=$ALPHA  b=$BETA_B  trainer=trainer_v2" | tee -a "$SWEEP_LOG"
echo "  Sampler: v1 balance (real_ratio=0.30)"                      | tee -a "$SWEEP_LOG"
echo "  GPU mode: NGPU=$NGPU"                                       | tee -a "$SWEEP_LOG"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"                       | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

for i in "${!RUNS[@]}"; do
    idx=$((i + 1))
    read TAG FLIP <<< "${RUNS[$i]}"
    run_one "[$idx/$TOTAL] $TAG" "$FLIP"
    echo "" | tee -a "$SWEEP_LOG"
done

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Summary — Beta(2,5) on top+l1  ($(date '+%Y-%m-%d %H:%M'))" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
printf "  %-30s | %-10s | %-9s | %-9s | %s\n" \
    "config" "beta_flip" "video_auc" "auc" "acc" | tee -a "$SWEEP_LOG"
echo "  --------------------------------|------------|-----------|-----------|------" | tee -a "$SWEEP_LOG"

for RUN_LINE in "${RUNS[@]}"; do
    read TAG FLIP <<< "$RUN_LINE"
    LINE=$(grep "^${TAG} |" "$SWEEP_LOG" 2>/dev/null | head -1)
    if [ -n "$LINE" ]; then
        B_V=$(echo "$LINE"   | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
        B_AUC=$(echo "$LINE" | sed 's/.*auc=\([0-9.]*\).*/\1/')
        B_ACC=$(echo "$LINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
        printf "  %-30s | %-10s | %-9s | %-9s | %s\n" \
            "$TAG" "$FLIP" "${B_V:-NA}" "${B_AUC:-NA}" "${B_ACC:-NA}" | tee -a "$SWEEP_LOG"
    else
        printf "  %-30s | %-10s | %-9s | %-9s | %s\n" \
            "$TAG" "$FLIP" "—" "—" "—" | tee -a "$SWEEP_LOG"
    fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "=== All $TOTAL runs done.  Log saved to $SWEEP_LOG ===" | tee -a "$SWEEP_LOG"

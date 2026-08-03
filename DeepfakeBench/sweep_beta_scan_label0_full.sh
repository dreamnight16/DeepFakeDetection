#!/bin/bash
# ===========================================================================
# Sweep: Beta Distribution Scan on label0_full
#
#   Symmetric  Beta(α,α): α ∈ {0.5, 1.0, 2.0, 5.0}
#   Asymmetric Beta(α,β): (α,β) ∈ {(2,5), (5,2), (1,5), (5,1)}
#
#   Base: lap_pyramid_label0_full (all Laplacian mixed, RF label→0)
#   Fixed: gamma=1.0, v1 sampler (real_ratio=0.30), trainer_v2
# ===========================================================================
# Usage:
#   bash sweep_beta_scan_label0_full.sh              # single GPU
#   bash sweep_beta_scan_label0_full.sh 4            # 4-GPU DDP (torchrun)
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
LOG_DIR_BASE="./zhiyuanyan/logs/benchv2/icml25/beta_scan_label0_full"
SWEEP_LOG="sweep_beta_scan_label0_full.log"
TRAIN_DS="FaceForensics++"
VAL_DS="Celeb-DF-v2"
TEST_DS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"
NGPU=${1:-1}
GAMMA=1.0
MODE="lap_pyramid_label0_full"

# ── Create patched train.py copy ──────────────────────────────────────────
TRAIN_PY="./training/train_beta_scan_label0_full_sweep.py"
cp ./training/train.py "$TRAIN_PY"
sed -i "s/^from trainer\..* import Trainer/from trainer.trainer_v2 import Trainer/" "$TRAIN_PY"
trap "rm -f $TRAIN_PY" EXIT

> "$SWEEP_LOG"

run_one() {
    local TAG=$1            # display tag
    local ALPHA=$2          # Beta α
    local BETA_B=$3         # Beta β ("null" → symmetric)
    local FLIP=${4:-false}  # beta_flip (reserved, not used)

    local SAFE_TAG=$(echo "$TAG" | sed 's/[][ \/]/_/g')
    local TRAIN_LOG="sweep_beta_scan_${SAFE_TAG}_train.log"
    echo "===== $TAG ====="

    TMP_YAML=$(mktemp /tmp/effort_beta_scan_XXXXXX.yaml)
    cp "$YAML" "$TMP_YAML"

    # ── Set sweep-specific log_dir ───────────────────────────────────────
    local LOG_DIR="${LOG_DIR_BASE}/${SAFE_TAG}"
    mkdir -p "$LOG_DIR"
    sed -i "s|^log_dir:.*|log_dir: ${LOG_DIR}|" "$TMP_YAML"

    # ── Mixup: label0_full + Beta params ──────────────────────────────────
    sed -i "s/^use_mixup:.*/use_mixup: true/"                         "$TMP_YAML"
    sed -i "s/^mixup_mode:.*/mixup_mode: ${MODE}/"                    "$TMP_YAML"
    sed -i "s/^mixup_gamma:.*/mixup_gamma: ${GAMMA}/"                 "$TMP_YAML"
    sed -i "s/^mixup_alpha:.*/mixup_alpha: ${ALPHA}/"                 "$TMP_YAML"
    sed -i "s/^mixup_beta_flip:.*/mixup_beta_flip: ${FLIP}/"          "$TMP_YAML"

    if [ "$BETA_B" = "null" ]; then
        sed -i "s/^mixup_beta_b:.*/mixup_beta_b: null/"               "$TMP_YAML"
    else
        sed -i "s/^mixup_beta_b:.*/mixup_beta_b: ${BETA_B}/"          "$TMP_YAML"
    fi

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

    # Extract avg metrics
    V_AUC=$(awk '/^video_auc:/{v=$2} END{print v}' "$EVAL_LOG")
    AUC=$(awk   '/^auc:/{v=$2} END{print v}' "$EVAL_LOG")
    ACC=$(awk   '/^acc:/{v=$2} END{print v}' "$EVAL_LOG")

    echo "$TAG | video_auc=${V_AUC:-NA} | auc=${AUC:-NA} | acc=${ACC:-NA}" | tee -a "$SWEEP_LOG"
    rm -f "$TMP_YAML"
}

# ═══════════════════════════════════════════════════════════════════════════
# 8 configs: 4 symmetric + 4 asymmetric
# ═══════════════════════════════════════════════════════════════════════════
# Format: "TAG" "alpha" "beta_b" "flip"
#   beta_b="null" → symmetric Beta(α,α)
#   beta_b=<int>  → asymmetric Beta(α,β)
RUNS=(
    # ── Symmetric Beta(α,α) ──────────────────────────────────────────────
    "sym_a0.5              0.5   null   false"
    "sym_a1.0              1.0   null   false"
    "sym_a2.0              2.0   null   false"
    "sym_a5.0              5.0   null   false"

    # ── Asymmetric Beta(α,β) ─────────────────────────────────────────────
    "asym_a2b5             2.0   5      false"
    "asym_a5b2             5.0   2      false"
    "asym_a1b5             1.0   5      false"
    "asym_a5b1             5.0   1      false"
)

TOTAL=${#RUNS[@]}
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Beta Distribution Scan on label0_full — $TOTAL configs"    | tee -a "$SWEEP_LOG"
echo "  Symmetric:  α ∈ {0.5, 1.0, 2.0, 5.0}"                     | tee -a "$SWEEP_LOG"
echo "  Asymmetric: (α,β) ∈ {(2,5), (5,2), (1,5), (5,1)}"         | tee -a "$SWEEP_LOG"
echo "  Base: ${MODE}"                                              | tee -a "$SWEEP_LOG"
echo "  Fixed: γ=$GAMMA  sampler=v1 (real_ratio=0.30)"             | tee -a "$SWEEP_LOG"
echo "  GPU mode: NGPU=$NGPU"                                       | tee -a "$SWEEP_LOG"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"                       | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

for i in "${!RUNS[@]}"; do
    idx=$((i + 1))
    read TAG ALPHA BETA_B FLIP <<< "${RUNS[$i]}"
    run_one "[$idx/$TOTAL] $TAG" "$ALPHA" "$BETA_B" "$FLIP"
    echo "" | tee -a "$SWEEP_LOG"
done

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Summary — Beta Scan label0_full  ($(date '+%Y-%m-%d %H:%M'))" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
printf "  %-16s | %5s | %5s | %7s | %-9s | %-9s | %s\n" \
    "config" "α" "β" "E[λ]" "video_auc" "auc" "acc" | tee -a "$SWEEP_LOG"
echo "  ------------------|-------|-------|---------|-----------|-----------|------" | tee -a "$SWEEP_LOG"

declare -A EXPECTED_LAMBDA
EXPECTED_LAMBDA=(
    ["sym_a0.5"]="—"
    ["sym_a1.0"]="0.500"
    ["sym_a2.0"]="0.500"
    ["sym_a5.0"]="0.500"
    ["asym_a2b5"]="0.286"
    ["asym_a5b2"]="0.714"
    ["asym_a1b5"]="0.167"
    ["asym_a5b1"]="0.833"
)

for RUN_LINE in "${RUNS[@]}"; do
    read TAG ALPHA BETA_B FLIP <<< "$RUN_LINE"
    B_DISPLAY="${BETA_B}"
    E_LAMBDA="${EXPECTED_LAMBDA[$TAG]:-—}"
    LINE=$(grep "^${TAG} |" "$SWEEP_LOG" 2>/dev/null | head -1)
    if [ -n "$LINE" ]; then
        B_V=$(echo "$LINE"   | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
        B_AUC=$(echo "$LINE" | sed 's/.*auc=\([0-9.]*\).*/\1/')
        B_ACC=$(echo "$LINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
        printf "  %-16s | %5s | %5s | %7s | %-9s | %-9s | %s\n" \
            "$TAG" "$ALPHA" "$B_DISPLAY" "$E_LAMBDA" "${B_V:-NA}" "${B_AUC:-NA}" "${B_ACC:-NA}" | tee -a "$SWEEP_LOG"
    else
        printf "  %-16s | %5s | %5s | %7s | %-9s | %-9s | %s\n" \
            "$TAG" "$ALPHA" "$B_DISPLAY" "$E_LAMBDA" "—" "—" "—" | tee -a "$SWEEP_LOG"
    fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "=== All $TOTAL runs done.  Log saved to $SWEEP_LOG ===" | tee -a "$SWEEP_LOG"

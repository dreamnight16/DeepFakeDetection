#!/bin/bash
# ===========================================================================
# Master Experiment Runner — Pyramid Mixup Full Matrix
# ===========================================================================
# Runs all 6 experiment groups, collects testall results,
# and generates a master summary table.
#
# Usage:
#   bash run_all_experiments.sh              # single GPU, sequential
#   bash run_all_experiments.sh 4            # 4-GPU DDP
#
# All results saved to: ./experiment_results/master_sweep_YYYYMMDD_HHMMSS/
# ===========================================================================
set -euo pipefail

NGPU=${1:-1}
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULT_DIR="./experiment_results/master_sweep_${TIMESTAMP}"
MASTER_LOG="${RESULT_DIR}/MASTER_SUMMARY.log"
TRAIN_DS="FaceForensics++"
VAL_DS="Celeb-DF-v2"
TEST_DS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"

mkdir -p "$RESULT_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# Helper: extract per-dataset metrics from a testall eval log
# ═══════════════════════════════════════════════════════════════════════════
parse_metrics() {
    local EVAL_LOG=$1
    local PREFIX=${2:-""}
    local DS_ORDER=("WDF" "FFIW" "Celeb-DF-v2" "DeepFakeDetection" "DFDC" "DFDCP" "DeeperForensics-1.0")

    # Average metrics (last occurrence in testall output)
    local V_AUC=$(awk '/^video_auc:/{v=$2} END{print v}' "$EVAL_LOG")
    local AUC=$(awk   '/^auc:/{v=$2} END{print v}' "$EVAL_LOG")
    local ACC=$(awk   '/^acc:/{v=$2} END{print v}' "$EVAL_LOG")

    echo "${PREFIX}video_auc=${V_AUC:-NA} auc=${AUC:-NA} acc=${ACC:-NA}"

    # Per-dataset
    for ds in "${DS_ORDER[@]}"; do
        local ds_v_auc=$(awk "/^dataset: ${ds}\$/{found=1} found && /^video_auc:/{print \$2; found=0}" "$EVAL_LOG")
        local ds_auc=$(awk   "/^dataset: ${ds}\$/{found=1} found && /^auc:/{print \$2; found=0}" "$EVAL_LOG")
        local ds_acc=$(awk   "/^dataset: ${ds}\$/{found=1} found && /^acc:/{print \$2; found=0}" "$EVAL_LOG")
        echo "${PREFIX}ds_${ds}: video_auc=${ds_v_auc:-NA} auc=${ds_auc:-NA} acc=${ds_acc:-NA}"
    done
}

# ═══════════════════════════════════════════════════════════════════════════
# Group 0: BASELINE — original lap_pyramid + v1 sampler (real_ratio=0.30)
# ═══════════════════════════════════════════════════════════════════════════
run_baseline() {
    echo "" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"
    echo "  BASELINE: Original Pyramid Mixup (lap_pyramid, soft CE)" | tee -a "$MASTER_LOG"
    echo "  Sampler: v1, real_ratio=0.30, γ=1.0, α=5.0" | tee -a "$MASTER_LOG"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"

    local B_DIR="${RESULT_DIR}/baseline_pyramid_v1_rr30"
    mkdir -p "$B_DIR"

    TMP_YAML=$(mktemp /tmp/effort_baseline_XXXXXX.yaml)
    cp ./training/config/detector/effort.yaml "$TMP_YAML"

    # Configure: lap_pyramid + v1 sampler (real_ratio=0.30)
    local B_LOG_DIR="${B_DIR}/logs"
    mkdir -p "$B_LOG_DIR"
    sed -i "s|^log_dir:.*|log_dir: ${B_LOG_DIR}|" "$TMP_YAML"
    sed -i "s/^use_mixup:.*/use_mixup: true/"                         "$TMP_YAML"
    sed -i "s/^mixup_mode:.*/mixup_mode: lap_pyramid/"                "$TMP_YAML"
    sed -i "s/^mixup_gamma:.*/mixup_gamma: 1.0/"                      "$TMP_YAML"
    sed -i "s/^mixup_alpha:.*/mixup_alpha: 5.0/"                      "$TMP_YAML"
    sed -i "s/^balance_sampler_v2:.*/balance_sampler_v2: false/"      "$TMP_YAML"
    sed -i "s/^use_balance_batch_sampler:.*/use_balance_batch_sampler: true/" "$TMP_YAML"
    sed -i "s/^sampler_real_ratio:.*/sampler_real_ratio: 0.30/"       "$TMP_YAML"

    if [ "$NGPU" -gt 1 ]; then
        sed -i "s/^train_batchSize:.*/train_batchSize: $((32 * NGPU))/" "$TMP_YAML"
    fi

    # Train
    echo "  [baseline] Training..." | tee -a "$MASTER_LOG"
    local TRAIN_LOG="${B_DIR}/train.log"
    if [ "$NGPU" -gt 1 ]; then
        torchrun --nproc_per_node=$NGPU ./training/train.py \
            --ddp --detector_path "$TMP_YAML" \
            --train_dataset "$TRAIN_DS" --test_dataset "$VAL_DS" \
            > "$TRAIN_LOG" 2>&1 || { echo "  [baseline] TRAIN_ERROR" | tee -a "$MASTER_LOG"; rm -f "$TMP_YAML"; return; }
    else
        python3 ./training/train.py \
            --detector_path "$TMP_YAML" \
            --train_dataset "$TRAIN_DS" --test_dataset "$VAL_DS" \
            > "$TRAIN_LOG" 2>&1 || { echo "  [baseline] TRAIN_ERROR" | tee -a "$MASTER_LOG"; rm -f "$TMP_YAML"; return; }
    fi

    # Find checkpoint
    CKPT=$(ls -td "${B_LOG_DIR}"/effort_*/test/avg/ckpt_best.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "  [baseline] CKPT_NOT_FOUND" | tee -a "$MASTER_LOG"
        rm -f "$TMP_YAML"
        return
    fi

    # Evaluate with testall
    echo "  [baseline] testall evaluation..." | tee -a "$MASTER_LOG"
    local EVAL_LOG="${B_DIR}/testall.log"
    python3 testall.py \
        --detector_path "$TMP_YAML" \
        --weights_path "$CKPT" \
        --test_datasets $TEST_DS \
        > "$EVAL_LOG" 2>&1

    # Per-dataset summary
    echo "  [baseline] per-dataset:" | tee -a "$MASTER_LOG"
    awk '/^dataset: /{ds=$0} /^(auc|video_auc|acc):/{printf "    %s  %s\n", ds, $0}' "$EVAL_LOG" | tee -a "$MASTER_LOG"

    V_AUC=$(awk '/^video_auc:/{v=$2} END{print v}' "$EVAL_LOG")
    AUC=$(awk   '/^auc:/{v=$2} END{print v}' "$EVAL_LOG")
    ACC=$(awk   '/^acc:/{v=$2} END{print v}' "$EVAL_LOG")
    echo "  [baseline] video_auc=${V_AUC:-NA} | auc=${AUC:-NA} | acc=${ACC:-NA}" | tee -a "$MASTER_LOG"
    rm -f "$TMP_YAML"
    echo "  [Baseline] Done: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
}

# ═══════════════════════════════════════════════════════════════════════════
# Group 1: 2×2 Factorial — label × scope
# ═══════════════════════════════════════════════════════════════════════════
run_group1() {
    echo "" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"
    echo "  Group 1/5: 2×2 Factorial (label × scope)" | tee -a "$MASTER_LOG"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"

    local G1_DIR="${RESULT_DIR}/g1_2x2_factorial"
    mkdir -p "$G1_DIR"

    # Run the sweep script (modify LOG_DIR_BASE to point to our result dir)
    local G1_SCRIPT="${G1_DIR}/sweep_pyramid_label_variants.sh"
    cp ./sweep_pyramid_label_variants.sh "$G1_SCRIPT"
    sed -i "s|^LOG_DIR_BASE=.*|LOG_DIR_BASE=\"${G1_DIR}/logs\"|" "$G1_SCRIPT"
    sed -i "s|^SWEEP_LOG=.*|SWEEP_LOG=\"${G1_DIR}/sweep.log\"|" "$G1_SCRIPT"

    (cd "$(dirname "$0")" && bash "$G1_SCRIPT" "$NGPU")

    # Parse results
    local SWEEP_LOG="${G1_DIR}/sweep.log"
    if [ -f "$SWEEP_LOG" ]; then
        echo "  [G1] Results:" | tee -a "$MASTER_LOG"
        grep "| video_auc=" "$SWEEP_LOG" | while read line; do
            echo "    $line" | tee -a "$MASTER_LOG"
        done
    fi
    echo "  [G1] Done: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
}

# ═══════════════════════════════════════════════════════════════════════════
# Group 2: Beta(2,5) on top + label=1
# ═══════════════════════════════════════════════════════════════════════════
run_group2() {
    echo "" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"
    echo "  Group 2/5: Beta(2,5) on top+l1" | tee -a "$MASTER_LOG"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"

    local G2_DIR="${RESULT_DIR}/g2_beta25_top_l1"
    mkdir -p "$G2_DIR"

    local G2_SCRIPT="${G2_DIR}/sweep_beta25_top_l1.sh"
    cp ./sweep_beta25_top_l1.sh "$G2_SCRIPT"
    sed -i "s|^LOG_DIR_BASE=.*|LOG_DIR_BASE=\"${G2_DIR}/logs\"|" "$G2_SCRIPT"
    sed -i "s|^SWEEP_LOG=.*|SWEEP_LOG=\"${G2_DIR}/sweep.log\"|" "$G2_SCRIPT"

    (cd "$(dirname "$0")" && bash "$G2_SCRIPT" "$NGPU")

    local SWEEP_LOG="${G2_DIR}/sweep.log"
    if [ -f "$SWEEP_LOG" ]; then
        echo "  [G2] Results:" | tee -a "$MASTER_LOG"
        grep "| video_auc=" "$SWEEP_LOG" | while read line; do
            echo "    $line" | tee -a "$MASTER_LOG"
        done
    fi
    echo "  [G2] Done: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
}

# ═══════════════════════════════════════════════════════════════════════════
# Group 3: Full scope — label=0/1
# ═══════════════════════════════════════════════════════════════════════════
run_group3() {
    echo "" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"
    echo "  Group 3/5: Full Scope (label0/1 × full)" | tee -a "$MASTER_LOG"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"

    local G3_DIR="${RESULT_DIR}/g3_full_scope"
    mkdir -p "$G3_DIR"

    local G3_SCRIPT="${G3_DIR}/sweep_pyramid_label_full.sh"
    cp ./sweep_pyramid_label_full.sh "$G3_SCRIPT"
    sed -i "s|^LOG_DIR_BASE=.*|LOG_DIR_BASE=\"${G3_DIR}/logs\"|" "$G3_SCRIPT"
    sed -i "s|^SWEEP_LOG=.*|SWEEP_LOG=\"${G3_DIR}/sweep.log\"|" "$G3_SCRIPT"

    (cd "$(dirname "$0")" && bash "$G3_SCRIPT" "$NGPU")

    local SWEEP_LOG="${G3_DIR}/sweep.log"
    if [ -f "$SWEEP_LOG" ]; then
        echo "  [G3] Results:" | tee -a "$MASTER_LOG"
        grep "| video_auc=" "$SWEEP_LOG" | while read line; do
            echo "    $line" | tee -a "$MASTER_LOG"
        done
    fi
    echo "  [G3] Done: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
}

# ═══════════════════════════════════════════════════════════════════════════
# Group 4: Pyramid Loss Ablation (Exp1: soft CE, Exp2: hard CE)
# ═══════════════════════════════════════════════════════════════════════════
run_group4() {
    echo "" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"
    echo "  Group 4/5: Pyramid Loss Ablation" | tee -a "$MASTER_LOG"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"

    local G4_DIR="${RESULT_DIR}/g4_loss_ablation"
    mkdir -p "$G4_DIR"

    python3 experiments/pyramid_loss_ablation.py \
        --pyramid_mode lap_pyramid \
        --train_dataset "$TRAIN_DS" \
        --val_dataset "$VAL_DS" \
        --test_datasets $TEST_DS \
        --output_dir "$G4_DIR" \
        --alpha 5.0 --gamma 1.0 --num_levels 3 \
        --sampler v1 --sampler_real_ratio 0.30 \
        --n_epochs 10 \
        2>&1 | tee "${G4_DIR}/run.log"

    # Parse testall results from each experiment's testall log
    echo "  [G4] Results:" | tee -a "$MASTER_LOG"
    for exp_dir in "$G4_DIR"/exp*; do
        local exp_name=$(basename "$exp_dir")
        local testall_log="${exp_dir}/testall.log"
        if [ -f "$testall_log" ]; then
            local V_AUC=$(awk '/^video_auc:/{v=$2} END{print v}' "$testall_log")
            local AUC=$(awk   '/^auc:/{v=$2} END{print v}' "$testall_log")
            local ACC=$(awk   '/^acc:/{v=$2} END{print v}' "$testall_log")
            echo "    ${exp_name} | video_auc=${V_AUC:-NA} | auc=${AUC:-NA} | acc=${ACC:-NA}" | tee -a "$MASTER_LOG"
        fi
    done
    echo "  [G4] Done: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
}

# ═══════════════════════════════════════════════════════════════════════════
# Group 5: RR+FF Pyramid (RF stripped vs not generated)
# ═══════════════════════════════════════════════════════════════════════════
run_group5() {
    echo "" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"
    echo "  Group 5/5: RR+FF Pyramid Mixup" | tee -a "$MASTER_LOG"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
    echo "============================================================" | tee -a "$MASTER_LOG"

    local G5_DIR="${RESULT_DIR}/g5_rrff_pyramid"
    mkdir -p "$G5_DIR"

    python3 experiments/pyramid_rrff_experiment.py \
        --train_dataset "$TRAIN_DS" \
        --val_dataset "$VAL_DS" \
        --test_datasets $TEST_DS \
        --output_dir "$G5_DIR" \
        --alpha 5.0 --gamma 1.0 --num_levels 3 \
        --sampler_real_ratio 0.30 \
        --n_epochs 10 \
        2>&1 | tee "${G5_DIR}/run.log"

    # Parse testall results
    echo "  [G5] Results:" | tee -a "$MASTER_LOG"
    for grp_dir in "$G5_DIR"/g*; do
        local grp_name=$(basename "$grp_dir")
        local testall_log="${grp_dir}/testall.log"
        if [ -f "$testall_log" ]; then
            local V_AUC=$(awk '/^video_auc:/{v=$2} END{print v}' "$testall_log")
            local AUC=$(awk   '/^auc:/{v=$2} END{print v}' "$testall_log")
            local ACC=$(awk   '/^acc:/{v=$2} END{print v}' "$testall_log")
            echo "    ${grp_name} | video_auc=${V_AUC:-NA} | auc=${AUC:-NA} | acc=${ACC:-NA}" | tee -a "$MASTER_LOG"
        fi
    done
    echo "  [G5] Done: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
}

# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
echo "============================================================" | tee "$MASTER_LOG"
echo "  Master Experiment Runner — Pyramid Mixup Full Matrix"     | tee -a "$MASTER_LOG"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"                      | tee -a "$MASTER_LOG"
echo "  Result dir: $RESULT_DIR"                                   | tee -a "$MASTER_LOG"
echo "  GPU mode: NGPU=$NGPU"                                      | tee -a "$MASTER_LOG"
echo "============================================================" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "  Groups:" | tee -a "$MASTER_LOG"
echo "    BASELINE  Original lap_pyramid (soft CE)      —  1 config"  | tee -a "$MASTER_LOG"
echo "    1/5  2×2 Factorial (label × scope)          —  4 configs" | tee -a "$MASTER_LOG"
echo "    2/5  Beta(2,5) on top+l1                     —  2 configs" | tee -a "$MASTER_LOG"
echo "    3/5  Full Scope (label0/1)                   —  2 configs" | tee -a "$MASTER_LOG"
echo "    4/5  Pyramid Loss Ablation (soft vs hard CE)  —  2 configs" | tee -a "$MASTER_LOG"
echo "    5/5  RR+FF Pyramid (strip vs rrff)            —  2 configs" | tee -a "$MASTER_LOG"
echo "         Total: 1 baseline + 12 configs" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

run_baseline
run_group1
run_group2
run_group3
run_group4
run_group5

# ═══════════════════════════════════════════════════════════════════════════
# Master Summary Table
# ═══════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "######################################################################" | tee -a "$MASTER_LOG"
echo "  MASTER SUMMARY TABLE" | tee -a "$MASTER_LOG"
echo "  Completed: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "######################################################################" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"

printf "  %-4s | %-30s | %-25s | %9s | %9s | %9s\n" \
    "Grp" "Experiment" "Config" "video_auc" "auc" "acc" | tee -a "$MASTER_LOG"
echo "  -----|--------------------------------|---------------------------|-----------|-----------|-----------" | tee -a "$MASTER_LOG"

# ── Baseline row (from G4 Exp1 = lap_pyramid soft CE) ─────────────────
BASELINE_V_AUC=$(grep "exp1_original.*video_auc=" "${RESULT_DIR}/g4_loss_ablation/run.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
BASELINE_AUC=$(grep "exp1_original.*auc=" "${RESULT_DIR}/g4_loss_ablation/run.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')
BASELINE_ACC=$(grep "exp1_original.*acc=" "${RESULT_DIR}/g4_loss_ablation/run.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')
printf "  %-4s | %-30s | %-25s | %9s | %9s | %9s\n" \
    "BL" "BASELINE (G4 Exp1)" "lap_pyramid soft CE" \
    "${BASELINE_V_AUC:-—}" "${BASELINE_AUC:-—}" "${BASELINE_ACC:-—}" | tee -a "$MASTER_LOG"
echo "  -----|--------------------------------|---------------------------|-----------|-----------|-----------" | tee -a "$MASTER_LOG"

print_row() {
    local grp=$1; local exp=$2; local config=$3
    local v_auc=$4; local auc=$5; local acc=$6
    printf "  %-4s | %-30s | %-25s | %9s | %9s | %9s\n" \
        "$grp" "$exp" "$config" "${v_auc:-—}" "${auc:-—}" "${acc:-—}" | tee -a "$MASTER_LOG"
}

# G1 rows
print_row "G1" "2x2 factorial" "label0_top"     "$(grep "label0_GK-only.*video_auc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" "$(grep "label0_GK-only.*auc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" "$(grep "label0_GK-only.*acc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"
print_row "G1" "" "label0_bottom"  "$(grep "label0_L0-only.*video_auc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" "$(grep "label0_L0-only.*auc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" "$(grep "label0_L0-only.*acc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"
print_row "G1" "" "label1_top"     "$(grep "label1_GK-only.*video_auc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" "$(grep "label1_GK-only.*auc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" "$(grep "label1_GK-only.*acc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"
print_row "G1" "" "label1_bottom"  "$(grep "label1_L0-only.*video_auc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" "$(grep "label1_L0-only.*auc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" "$(grep "label1_L0-only.*acc=" "${RESULT_DIR}/g1_2x2_factorial/sweep.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"

# G2 rows
print_row "G2" "Beta(2,5) top+l1" "Beta(2,5)"      "$(grep "top_l1_Beta25 .*video_auc=" "${RESULT_DIR}/g2_beta25_top_l1/sweep.log" 2>/dev/null | grep -v flip | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" "$(grep "top_l1_Beta25 .*auc=" "${RESULT_DIR}/g2_beta25_top_l1/sweep.log" 2>/dev/null | grep -v flip | sed 's/.*auc=\([0-9.]*\).*/\1/')" "$(grep "top_l1_Beta25 .*acc=" "${RESULT_DIR}/g2_beta25_top_l1/sweep.log" 2>/dev/null | grep -v flip | sed 's/.*acc=\([0-9.]*\).*/\1/')"
print_row "G2" "" "1-Beta(2,5)"    "$(grep "top_l1_Beta25_flip.*video_auc=" "${RESULT_DIR}/g2_beta25_top_l1/sweep.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" "$(grep "top_l1_Beta25_flip.*auc=" "${RESULT_DIR}/g2_beta25_top_l1/sweep.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" "$(grep "top_l1_Beta25_flip.*acc=" "${RESULT_DIR}/g2_beta25_top_l1/sweep.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"

# G3 rows
print_row "G3" "Full scope" "label0_full"  "$(grep "label0_full.*video_auc=" "${RESULT_DIR}/g3_full_scope/sweep.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" "$(grep "label0_full.*auc=" "${RESULT_DIR}/g3_full_scope/sweep.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" "$(grep "label0_full.*acc=" "${RESULT_DIR}/g3_full_scope/sweep.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"
print_row "G3" "" "label1_full"  "$(grep "label1_full.*video_auc=" "${RESULT_DIR}/g3_full_scope/sweep.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" "$(grep "label1_full.*auc=" "${RESULT_DIR}/g3_full_scope/sweep.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" "$(grep "label1_full.*acc=" "${RESULT_DIR}/g3_full_scope/sweep.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"

# G4 rows
print_row "G4" "Loss ablation" "Exp1_soft_CE" \
    "$(grep "exp1_original.*video_auc=" "${RESULT_DIR}/g4_loss_ablation/run.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" \
    "$(grep "exp1_original.*auc=" "${RESULT_DIR}/g4_loss_ablation/run.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" \
    "$(grep "exp1_original.*acc=" "${RESULT_DIR}/g4_loss_ablation/run.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"
print_row "G4" "" "Exp2_hard_CE" \
    "$(grep "exp2_stripped.*video_auc=" "${RESULT_DIR}/g4_loss_ablation/run.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" \
    "$(grep "exp2_stripped.*auc=" "${RESULT_DIR}/g4_loss_ablation/run.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" \
    "$(grep "exp2_stripped.*acc=" "${RESULT_DIR}/g4_loss_ablation/run.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"

# G5 rows
print_row "G5" "RR+FF pyramid" "G1_RF_stripped" \
    "$(grep "g1_rf_stripped.*video_auc=" "${RESULT_DIR}/g5_rrff_pyramid/run.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" \
    "$(grep "g1_rf_stripped.*auc=" "${RESULT_DIR}/g5_rrff_pyramid/run.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" \
    "$(grep "g1_rf_stripped.*acc=" "${RESULT_DIR}/g5_rrff_pyramid/run.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"
print_row "G5" "" "G2_RF_not_generated" \
    "$(grep "g2_rf_not_generated.*video_auc=" "${RESULT_DIR}/g5_rrff_pyramid/run.log" 2>/dev/null | sed 's/.*video_auc=\([0-9.]*\).*/\1/')" \
    "$(grep "g2_rf_not_generated.*auc=" "${RESULT_DIR}/g5_rrff_pyramid/run.log" 2>/dev/null | sed 's/.*auc=\([0-9.]*\).*/\1/')" \
    "$(grep "g2_rf_not_generated.*acc=" "${RESULT_DIR}/g5_rrff_pyramid/run.log" 2>/dev/null | sed 's/.*acc=\([0-9.]*\).*/\1/')"

echo "" | tee -a "$MASTER_LOG"
echo "######################################################################" | tee -a "$MASTER_LOG"
echo "  ALL EXPERIMENTS COMPLETE" | tee -a "$MASTER_LOG"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MASTER_LOG"
echo "  Results: $RESULT_DIR" | tee -a "$MASTER_LOG"
echo "  Summary: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "######################################################################" | tee -a "$MASTER_LOG"

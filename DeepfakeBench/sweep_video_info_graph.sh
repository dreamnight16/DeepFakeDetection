#!/bin/bash
# ===========================================================================
# Sweep: Video-Level Information Graph Detectors — Ablation
# ===========================================================================
# Three registered detectors (mi_video, gt_video, gnn_video) with ablation
# variants controlled via YAML flags. Follows exact same pattern as
# sweep_trajectory_mixup.sh et al.
#
# Detector  | Variants                                   | Config key
# ----------|--------------------------------------------|------------------
# mi_video  | MI-T, MI-S, MI-F, MI-All                   | mi_temporal/spatial/frequency
# gt_video  | GT-Temporal, GT-Spatial, GT-Full           | gt_temporal/spatial/full
# gnn_video | MLP, GCN, GAT, STGCN                       | gnn_variant
#
# Usage:
#   bash sweep_video_info_graph.sh              # single GPU
#   bash sweep_video_info_graph.sh 4            # 4-GPU DDP (torchrun)
# ===========================================================================
set -euo pipefail

YAML="./training/config/detector/effort.yaml"
LOG_DIR_BASE="./zhiyuanyan/logs/benchv2/icml25/video_info_graph_sweep"
SWEEP_LOG="sweep_video_info_graph.log"
TRAIN_DS="FaceForensics++"
VAL_DS="Celeb-DF-v2"
TEST_DS="WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0"
NGPU=${1:-1}
CLIP_SIZE=8
EPOCHS=10

# ── Create patched train.py copy (safe for concurrent runs) ──────────────
TRAIN_PY="./training/train_video_sweep.py"
cp ./training/train.py "$TRAIN_PY"
# Use trainer_v2 (same as other sweeps)
sed -i "s/^from trainer\..* import Trainer/from trainer.trainer_v2 import Trainer/" "$TRAIN_PY"
trap "rm -f $TRAIN_PY" EXIT

> "$SWEEP_LOG"

run_one() {
    local TAG=$1       # display tag
    local MODEL=$2     # model_name: mi_video | gt_video | gnn_video
    local SED_CMDS=$3  # semicolon-separated sed commands for variant flags

    local SAFE_TAG=$(echo "$TAG" | sed 's/[][ \/]/_/g')
    local TRAIN_LOG="sweep_video_${SAFE_TAG}_train.log"
    echo "===== $TAG ====="

    TMP_YAML=$(mktemp /tmp/effort_video_XXXXXX.yaml)
    cp "$YAML" "$TMP_YAML"

    # ── Set sweep-specific log_dir ───────────────────────────────────────
    local LOG_DIR="${LOG_DIR_BASE}/${SAFE_TAG}"
    mkdir -p "$LOG_DIR"
    sed -i "s|^log_dir:.*|log_dir: ${LOG_DIR}|" "$TMP_YAML"

    # ── Model name ───────────────────────────────────────────────────────
    sed -i "s/^model_name:.*/model_name: ${MODEL}/" "$TMP_YAML"

    # ── Video mode ───────────────────────────────────────────────────────
    sed -i "s/^video_mode:.*/video_mode: true/"  "$TMP_YAML"
    sed -i "s/^clip_size:.*/clip_size: ${CLIP_SIZE}/" "$TMP_YAML"

    # ── Disable mixup (video experiments don't use it) ────────────────────
    sed -i "s/^use_mixup:.*/use_mixup: false/" "$TMP_YAML"

    # ── Simpler sampler for video clips ───────────────────────────────────
    sed -i "s/^balance_sampler_v2:.*/balance_sampler_v2: false/" "$TMP_YAML"
    sed -i "s/^use_balance_batch_sampler:.*/use_balance_batch_sampler: false/" "$TMP_YAML"

    # ── Epochs ───────────────────────────────────────────────────────────
    sed -i "s/^nEpochs:.*/nEpochs: ${EPOCHS}/" "$TMP_YAML"

    # ── Variant-specific flags ────────────────────────────────────────────
    if [ -n "$SED_CMDS" ]; then
        IFS=';' read -ra CMDS <<< "$SED_CMDS"
        for cmd in "${CMDS[@]}"; do
            [ -z "$cmd" ] && continue
            eval "sed -i \"$cmd\" \"$TMP_YAML\""
        done
    fi

    # ── Adjust batch size for video (T=8 frames → high memory) ────────────
    if [ "$MODEL" = "gnn_video" ]; then
        sed -i "s/^train_batchSize:.*/train_batchSize: 4/" "$TMP_YAML"
        sed -i "s/^test_batchSize:.*/test_batchSize: 4/" "$TMP_YAML"
    else
        sed -i "s/^train_batchSize:.*/train_batchSize: 8/" "$TMP_YAML"
        sed -i "s/^test_batchSize:.*/test_batchSize: 8/" "$TMP_YAML"
    fi

    if [ "$NGPU" -gt 1 ]; then
        local bs=$(grep '^train_batchSize:' "$TMP_YAML" | awk '{print $2}')
        sed -i "s/^train_batchSize:.*/train_batchSize: $((bs * NGPU))/" "$TMP_YAML"
    fi

    # ── Train ────────────────────────────────────────────────────────────
    echo "  [train] log -> $TRAIN_LOG"
    if [ "$NGPU" -gt 1 ]; then
        torchrun --nproc_per_node=$NGPU "$TRAIN_PY" \
            --ddp \
            --detector_path "$TMP_YAML" \
            --train_dataset "$TRAIN_DS" \
            --test_dataset "$VAL_DS" \
            > "$TRAIN_LOG" 2>&1 || {
                echo "$TAG | TRAIN_ERROR" | tee -a "$SWEEP_LOG"
                rm -f "$TMP_YAML"; return
            }
    else
        python3 "$TRAIN_PY" \
            --detector_path "$TMP_YAML" \
            --train_dataset "$TRAIN_DS" \
            --test_dataset "$VAL_DS" \
            > "$TRAIN_LOG" 2>&1 || {
                echo "$TAG | TRAIN_ERROR" | tee -a "$SWEEP_LOG"
                rm -f "$TMP_YAML"; return
            }
    fi

    # ── Find checkpoint ──────────────────────────────────────────────────
    CKPT=$(ls -td "${LOG_DIR}"/${MODEL}_*/test/avg/ckpt_best.pth 2>/dev/null | head -1)
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

    # Extract avg metrics
    V_AUC=$(awk '/^video_auc:/{v=$2} END{print v}' "$EVAL_LOG")
    AUC=$(awk   '/^auc:/{v=$2} END{print v}' "$EVAL_LOG")
    ACC=$(awk   '/^acc:/{v=$2} END{print v}' "$EVAL_LOG")

    echo "$TAG | video_auc=${V_AUC:-NA} | auc=${AUC:-NA} | acc=${ACC:-NA}" | tee -a "$SWEEP_LOG"
    rm -f "$TMP_YAML"
}

# ═══════════════════════════════════════════════════════════════════════════
# Experiment matrix
# ═══════════════════════════════════════════════════════════════════════════
# Format: "TAG" "MODEL" "SED_CMDS"
RUNS=(
    # ── MI-D ablations ───────────────────────────────────────────────────
    "MI-T                 mi_video    s/^mi_temporal:.*/mi_temporal: true/;s/^mi_spatial:.*/mi_spatial: false/;s/^mi_frequency:.*/mi_frequency: false/"
    "MI-S                 mi_video    s/^mi_temporal:.*/mi_temporal: false/;s/^mi_spatial:.*/mi_spatial: true/;s/^mi_frequency:.*/mi_frequency: false/"
    "MI-F                 mi_video    s/^mi_temporal:.*/mi_temporal: false/;s/^mi_spatial:.*/mi_spatial: false/;s/^mi_frequency:.*/mi_frequency: true/"
    "MI-All               mi_video    s/^mi_temporal:.*/mi_temporal: true/;s/^mi_spatial:.*/mi_spatial: true/;s/^mi_frequency:.*/mi_frequency: true/"

    # ── GT-D ablations ───────────────────────────────────────────────────
    "GT-Temporal          gt_video    s/^gt_temporal:.*/gt_temporal: true/;s/^gt_spatial:.*/gt_spatial: false/;s/^gt_full:.*/gt_full: false/"
    "GT-Spatial           gt_video    s/^gt_temporal:.*/gt_temporal: false/;s/^gt_spatial:.*/gt_spatial: true/;s/^gt_full:.*/gt_full: false/"
    "GT-Full              gt_video    s/^gt_temporal:.*/gt_temporal: true/;s/^gt_spatial:.*/gt_spatial: true/;s/^gt_full:.*/gt_full: true/"

    # ── GNN-D ablations ──────────────────────────────────────────────────
    "GNN-MLP              gnn_video   s/^gnn_variant:.*/gnn_variant: mlp/"
    "GNN-GCN              gnn_video   s/^gnn_variant:.*/gnn_variant: gcn/"
    "GNN-GAT              gnn_video   s/^gnn_variant:.*/gnn_variant: gat/"
)

TOTAL=${#RUNS[@]}
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Video Info-Graph Detector Ablation — $TOTAL configs"        | tee -a "$SWEEP_LOG"
echo "  Train: $TRAIN_DS   Val: $VAL_DS   Test: $TEST_DS"           | tee -a "$SWEEP_LOG"
echo "  Clip: T=$CLIP_SIZE   Epochs: $EPOCHS   GPU: NGPU=$NGPU"     | tee -a "$SWEEP_LOG"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"                       | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

for i in "${!RUNS[@]}"; do
    idx=$((i + 1))
    read TAG MODEL SED_CMDS <<< "${RUNS[$i]}"
    run_one "[$idx/$TOTAL] $TAG" "$MODEL" "$SED_CMDS"
    echo "" | tee -a "$SWEEP_LOG"
done

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Summary — Video Info-Graph  ($(date '+%Y-%m-%d %H:%M'))"  | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
printf "  %-20s | %-10s | %-9s | %-9s | %s\n" \
    "config" "model" "video_auc" "auc" "acc" | tee -a "$SWEEP_LOG"
echo "  ----------------------|------------|-----------|-----------|------" | tee -a "$SWEEP_LOG"

for RUN_LINE in "${RUNS[@]}"; do
    read TAG MODEL SED_CMDS <<< "$RUN_LINE"
    LINE=$(grep "^${TAG} |" "$SWEEP_LOG" 2>/dev/null | head -1)
    if [ -n "$LINE" ]; then
        B_V=$(echo "$LINE"   | sed 's/.*video_auc=\([0-9.]*\).*/\1/')
        B_AUC=$(echo "$LINE" | sed 's/.*auc=\([0-9.]*\).*/\1/')
        B_ACC=$(echo "$LINE" | sed 's/.*acc=\([0-9.]*\).*/\1/')
        printf "  %-20s | %-10s | %-9s | %-9s | %s\n" \
            "$TAG" "$MODEL" "${B_V:-NA}" "${B_AUC:-NA}" "${B_ACC:-NA}" | tee -a "$SWEEP_LOG"
    else
        printf "  %-20s | %-10s | %-9s | %-9s | %s\n" \
            "$TAG" "$MODEL" "—" "—" "—" | tee -a "$SWEEP_LOG"
    fi
done

echo "" | tee -a "$SWEEP_LOG"
echo "=== All $TOTAL runs done.  Log saved to $SWEEP_LOG ===" | tee -a "$SWEEP_LOG"

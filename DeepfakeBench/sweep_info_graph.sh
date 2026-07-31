#!/bin/bash
# ===========================================================================
# Sweep: Information Graph Theory — Standalone Validation
# ===========================================================================
# Validates three theoretical hypotheses on frozen CLIP ViT-L/14 features.
#
# All methods evaluated with the SAME Logistic Regression (5-fold CV, C-tuned)
# for fair comparison — no learned parameters beyond LR weights.
#
# 10 methods compared:
#   ┌──────────────────────────┬───────┬──────────────────────────────────┐
#   │  Method                  │  Dim  │  Description                     │
#   ├──────────────────────────┼───────┼──────────────────────────────────┤
#   │  CLIP CLS (*)            │  768  │  Baseline: CLS token             │
#   │  Token mean (*)          │  768  │  Baseline: mean of patches       │
#   │  MI stats                │    5  │  M: mean, std, ‖M‖_F, skew, kurt│
#   │  MI SVD                  │   20  │  top-20 singular values of M     │
#   │  MI eigen                │   10  │  top-10 eigenvalues of (M+M^T)/2 │
#   │  Spectrum (MI)           │   23  │  λ_low+λ_high+gap+H_G on MI graph│
#   │  Spectrum (cos)          │   23  │  ablation: cosine adjacency      │
#   │  GNN embedding           │  128  │  GCN on MI graph → mean pool     │
#   │  MI+Spec                 │   28  │  concat MI stats + MI spectrum   │
#   │  Spectrum (rand)         │   23  │  sanity check: random adjacency  │
#   └──────────────────────────┴───────┴──────────────────────────────────┘
#   (*) = baseline
#
# Protocol:
#   Train:  FF++ c23  → feature extraction + PCA fit + LR train
#   Test A: FF++ c23  → in-domain evaluation
#   Test B: Celeb-DF  → cross-domain generalization
#
# Theory:
#   P_r(G) ≠ P_f(G)
#   I_r(x_i; x_j) ≠ I_f(x_i; x_j)
#   Λ_r ≠ Λ_f
#
# Usage:
#   bash sweep_info_graph.sh              # single GPU, default settings
#   bash sweep_info_graph.sh 4            # 4-GPU DDP not needed (CPU-bound LR)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Experiment script ─────────────────────────────────────────────────────
EXP_PY="${PROJECT_ROOT}/experiments/validate_info_graph.py"

YAML="${PROJECT_ROOT}/DeepfakeBench/training/config/detector/effort.yaml"
SWEEP_LOG="sweep_info_graph.log"
TRAIN_DS="FaceForensics++"
TEST_DS="Celeb-DF-v2 FaceForensics++"
NGPU=${1:-1}

# ── Experiment parameters ─────────────────────────────────────────────────
MAX_TRAIN=2000          # balanced: 1000 real + 1000 fake
MAX_TEST=1000           # balanced: 500 real + 500 fake
BATCH_SIZE=16
PCA_DIM=32
GNN_K=10
SEED=1024

> "$SWEEP_LOG"

run_one() {
    local TAG=$1            # display tag
    local EXTRA_ARGS=${2:-} # extra CLI args (--clip_layer N, --output_attentions)

    local SAFE_TAG=$(echo "$TAG" | sed 's/[][ \/]/_/g')
    local EXP_LOG="sweep_infograph_${SAFE_TAG}.log"
    echo "===== $TAG ====="

    # ── Run the standalone experiment ─────────────────────────────────────
    echo "  [run] log -> $EXP_LOG"
    python3 "$EXP_PY" \
        --detector_path "$YAML" \
        --train_dataset $TRAIN_DS \
        --test_datasets $TEST_DS \
        --max_train_samples "$MAX_TRAIN" \
        --max_test_samples "$MAX_TEST" \
        --batch_size "$BATCH_SIZE" \
        --pca_dim "$PCA_DIM" \
        --gnn_k "$GNN_K" \
        --seed "$SEED" \
        $EXTRA_ARGS \
        > "$EXP_LOG" 2>&1 || {
            echo "$TAG | RUN_ERROR" | tee -a "$SWEEP_LOG"
            return
        }

    # ── Extract results per dataset ───────────────────────────────────────
    # The script prints a table after "=== Test dataset: <name>"
    # Parse AUC values for each method × dataset

    echo "  [results]" | tee -a "$SWEEP_LOG"

    # Extract per-dataset results using awk
    # Format:   Method                       AUC     Acc     F1      AP
    for DS in $TEST_DS; do
        echo "    dataset: $DS" | tee -a "$SWEEP_LOG"
        awk -v ds="$DS" '
            BEGIN { in_section=0 }
            /Test dataset: / {
                if ($3 == ds) in_section=1; else in_section=0
            }
            in_section && /^  [A-Z]/ {
                # Line format: "  Method_name (*)    0.XXXX  0.XXXX  ..."
                method=$1
                for (i=2; i<=NF && method !~ /^[A-Z]/; i++) method=method" "$i
                # Remove (*) marker for display
                gsub(/ \(\*\)/, "", method)
                auc=$(NF-3)
                acc=$(NF-2)
                f1=$(NF-1)
                ap=$NF
                if (auc ~ /^[0-9]/) printf "      %-30s  auc=%-8s acc=%-8s f1=%-8s ap=%s\n", method, auc, acc, f1, ap
            }
        ' "$EXP_LOG" | tee -a "$SWEEP_LOG"
    done

    # Show any warnings
    if grep -q "WARNING\|ERROR" "$EXP_LOG"; then
        echo "  [warnings]:" | tee -a "$SWEEP_LOG"
        grep "WARNING\|ERROR" "$EXP_LOG" | sed 's/^/    /' | tee -a "$SWEEP_LOG"
    fi

    echo "$TAG | DONE" | tee -a "$SWEEP_LOG"
}

# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Information Graph Theory — Standalone Validation"          | tee -a "$SWEEP_LOG"
echo "  Train: $TRAIN_DS  ($MAX_TRAIN samples)"                   | tee -a "$SWEEP_LOG"
echo "  Test:  $TEST_DS  ($MAX_TEST samples each)"                | tee -a "$SWEEP_LOG"
echo "  PCA:  D→$PCA_DIM   GNN k=$GNN_K   seed=$SEED"            | tee -a "$SWEEP_LOG"
echo "  Classifier: Logistic Regression (5-fold CV, C-tuned)"     | tee -a "$SWEEP_LOG"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"                       | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

# ═══════════════════════════════════════════════════════════════════════════
# Experiment matrix
# ═══════════════════════════════════════════════════════════════════════════
# Format: "TAG" "EXTRA_ARGS"
RUNS=(
    # ── Layer sweep: test MI/Spectrum at different ViT depths ─────────────
    "last_layer             "
    "layer_4                --clip_layer 4"
    "layer_6                --clip_layer 6"
    "layer_8                --clip_layer 8"
    "layer_10               --clip_layer 10"
    # ── Attention graph: use CLIP's own attention as adjacency ────────────
    "last_layer+attn        --output_attentions"
)

TOTAL=${#RUNS[@]}
echo "  Experiment matrix ($TOTAL configs):" | tee -a "$SWEEP_LOG"
echo "    - last_layer:     baseline (layer 12)" | tee -a "$SWEEP_LOG"
echo "    - layer_{4,6,8,10}: intermediate layer tokens" | tee -a "$SWEEP_LOG"
echo "    - +attn:          CLIP self-attention as graph adjacency" | tee -a "$SWEEP_LOG"
echo "" | tee -a "$SWEEP_LOG"

for i in "${!RUNS[@]}"; do
    idx=$((i + 1))
    read TAG EXTRA <<< "${RUNS[$i]}"
    run_one "[$idx/$TOTAL] $TAG" "$EXTRA"
    echo "" | tee -a "$SWEEP_LOG"
done

# ═══════════════════════════════════════════════════════════════════════════
# Summary — cross-dataset comparison
# ═══════════════════════════════════════════════════════════════════════════
echo "" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"
echo "  Summary — Information Graph Theory  ($(date '+%Y-%m-%d %H:%M'))" | tee -a "$SWEEP_LOG"
echo "============================================================" | tee -a "$SWEEP_LOG"

# Build per-dataset summary
for DS in $TEST_DS; do
    echo "" | tee -a "$SWEEP_LOG"
    echo "  ── $DS ──" | tee -a "$SWEEP_LOG"
    printf "  %-28s | %-8s | %-8s | %-8s | %-8s\n" \
        "Method" "AUC" "Acc" "F1" "AP" | tee -a "$SWEEP_LOG"
    echo "  ------------------------------|----------|----------|----------|----------" | tee -a "$SWEEP_LOG"

    # Parse from the experiment log
    for EXP_LOG in sweep_infograph_*.log; do
        [ -f "$EXP_LOG" ] || continue
        awk -v ds="$DS" '
            BEGIN { in_section=0 }
            /Test dataset: / {
                if ($3 == ds || ($3 " " $4) == ds) in_section=1; else in_section=0
            }
            in_section && /^  [A-Z]/ {
                # Extract method name (may contain spaces)
                line=$0
                sub(/^  /, "", line)
                # Split: method name ends at last 4 numeric columns
                n=split(line, a, /  +/)
                if (n >= 5) {
                    auc=a[n-3]
                    acc=a[n-2]
                    f1=a[n-1]
                    ap=a[n]
                    method=""
                    for (i=1; i<=n-4; i++) method=method (i>1?" ":"") a[i]
                    gsub(/ \(\*\)/, "", method)
                    if (auc ~ /^[0-9]/)
                        printf "  %-28s | %8s | %8s | %8s | %8s\n", method, auc, acc, f1, ap
                }
            }
        ' "$EXP_LOG" | tee -a "$SWEEP_LOG"
    done
done

echo "" | tee -a "$SWEEP_LOG"
echo "=== Done.  Log saved to $SWEEP_LOG ===" | tee -a "$SWEEP_LOG"

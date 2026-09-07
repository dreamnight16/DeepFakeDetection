"""
G18-2 — evidence-token-count sweep for the LFEQ read-out head.

Follows G18 exactly (frozen CLIP ViT-L/14 + LoRA, FF++ train, Celeb-DF-v2
best-ckpt selection, ``sampler_real_ratio=0.30`` v1 balance sampler, no mixup)
and changes exactly ONE variable across the whole sequence: the number of
learnable evidence tokens K in ``LearnableForgeryEvidenceQuery``.

    K1    K=1      fusion=0.5
    K2    K=2      fusion=0.5
    K4    K=4      fusion=0.5
    K8    K=8      fusion=0.5   <- config-identical to G18 L1
    K16   K=16     fusion=0.5
    K32   K=32     fusion=0.5

The LFEQ structure, the loss scalars (hidden=256, depth=2, heads=8,
dropout=0.1, lfeq_evidence_weight=1.0, lfeq_diversity_weight=0.01) and the
fusion_weight=0.5 are all pinned to the G18 L1 primary config.  So this is a
pure K sweep at the fused (global+evidence) setting.

Why K is a clean ablation (not a capacity confound): the EvidenceQueryBlock
self/cross-attention weights and the shared ``evidence_head`` are dimension-
based, NOT sequence-length-based — so the trainable-weight count is ~invariant
to K.  The only term that scales with K is the learnable ``evidence_tokens``
embedding (K x hidden = K x 256 extra params: 256 vs 8192 across the sweep),
which is negligible next to the ~2.37M LFEQ head.  The sweep therefore isolates
the NUMBER of evidence queries, not capacity.  (G18 report §6 already flags K as
a pinned default — this sequence closes that gap.)

Test = the SAME 7 cross-domain sets + a separate FaceForensics++ in-domain
column.  Cross-domain aggregate = mean(Celeb-DF-v2, DFDC) (G16 discipline; NEVER
the 7-set average and NEVER including FF++).  G = AUC_FFpp - AUC_cross.

── ISOLATION ────────────────────────────────────────────────────────────────
Every arm is run fully in isolation so no two arms can interfere:
  * per-arm output / log / ckpt / results  dirs  (no shared namespace);
  * per-arm config  (each set ONLY its own lfeq_num_evidence_tokens; fusion and
    every other lfeq_* scalar are pinned across arms);
  * no cross-arm checkpoint reuse  (no init_ckpt / warm-start);
  * train_model() and evaluate_model() each submit a FRESH Python subprocess,
    so every arm gets its own CUDA context released on process exit;
  * torch.cuda.empty_cache() after each arm (evaluate_model already
    `del model; torch.cuda.empty_cache()`).

Usage:
    python experiments/run_g18_2_evidence_sweep.py               # all 6 arms
    python experiments/run_g18_2_evidence_sweep.py --arms K1 K32 # subset
    python experiments/run_g18_2_evidence_sweep.py --n_epochs 10 --seed 2048
"""
import os
import sys
import argparse
import json

import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
_deepfake_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _current_dir)
sys.path.insert(0, _deepfake_dir)

from experiment_utils import build_config, train_model, evaluate_model

TRAIN_DS = 'FaceForensics++'
VAL_DS = 'Celeb-DF-v2'

# The 7 cross-domain test sets (G15-consistent).  FF++ is NOT in this list: it is
# the in-domain training set, reported as a SEPARATE column (see IN_DOMAIN_DS).
CROSS_DS = ['WDF', 'FFIW', 'Celeb-DF-v2', 'DeepFakeDetection',
            'DFDC', 'DFDCP', 'DeeperForensics-1.0']
IN_DOMAIN_DS = 'FaceForensics++'

# Full evaluation list = cross-domain 7 + in-domain FF++ column.
TEST_DS = CROSS_DS + [IN_DOMAIN_DS]

# Cross-domain aggregate = mean(Celeb-DF-v2, DFDC) — G16 discipline.
CROSS_METRIC_DS = ['Celeb-DF-v2', 'DFDC']

# LFEQ structural + loss scalars pinned to the G18 L1 primary config.  The sweep
# varies ONLY lfeq_num_evidence_tokens.
LFEQ_PINNED = dict(
    lfeq_hidden_dim=256, lfeq_depth=2, lfeq_num_heads=8, lfeq_dropout=0.1,
    lfeq_fusion_weight=0.5, lfeq_evidence_weight=1.0,
    lfeq_diversity_weight=0.01,
)


# ── G18-2 arms: evidence-token-count sweep at fusion=0.5 ────────────────────
G18_2_ARMS = [
    {'name': 'K1',  'lfeq_num_evidence_tokens': 1},
    {'name': 'K2',  'lfeq_num_evidence_tokens': 2},
    {'name': 'K4',  'lfeq_num_evidence_tokens': 4},
    {'name': 'K8',  'lfeq_num_evidence_tokens': 8},
    {'name': 'K16', 'lfeq_num_evidence_tokens': 16},
    {'name': 'K32', 'lfeq_num_evidence_tokens': 32},
]


def run_one(exp, args):
    """Train + eval a single arm in its own isolated dir. Returns summary dict."""
    exp_id = f"G18-2/{exp['name']}"
    output_dir = os.path.join(args.output_dir, exp['name'])
    os.makedirs(output_dir, exist_ok=True)
    # Per-arm isolated log dir — ckpts and run logs never collide across arms.
    log_dir = os.path.join(args.output_dir, 'logs', exp['name'])

    # Every arm uses the SAME pinned LFEQ config; only K varies.
    kwargs = dict(
        use_mixup=False, mixup_loss_strip=False,
        sampler_real_ratio=args.sampler_real_ratio,
        model_name='effort_lfeq',
        log_dir=log_dir, train_dataset=TRAIN_DS, test_dataset=VAL_DS,
        n_epochs=args.n_epochs,
        **LFEQ_PINNED,
        lfeq_num_evidence_tokens=exp['lfeq_num_evidence_tokens'],
    )

    config = build_config(**kwargs)
    config['manualSeed'] = args.seed      # deterministic per-run seed

    ckpt = train_model(config, TRAIN_DS, VAL_DS)
    if ckpt is None:
        print(f"[{exp_id}] TRAIN FAILED")
        return {'exp_name': exp_id, 'status': 'TRAIN_FAILED'}

    # Eval config mirrors train config exactly so test.py rebuilds the model and
    # dataset identically (arch_keys propagate model_name + lfeq_* to testall,
    # including lfeq_num_evidence_tokens for strict-load).
    config_eval = build_config(**{**kwargs, 'n_epochs': 0,
                                  'test_dataset': TEST_DS,
                                  'for_training': False})
    config_eval['manualSeed'] = args.seed
    summary = evaluate_model(config_eval, ckpt, TEST_DS, TRAIN_DS,
                             output_dir, exp_id)
    summary['ckpt'] = ckpt
    summary['status'] = 'OK'
    summary['exp_name'] = exp_id
    summary['lfeq_num_evidence_tokens'] = exp['lfeq_num_evidence_tokens']
    return summary


def _cross_and_gap(summary):
    """Return (cross_auc, in_auc, gap) from a summary, or (None, None, None)."""
    ta = summary.get('testall', {})
    cross_aucs = [ta[d]['video_auc'] for d in CROSS_METRIC_DS
                  if d in ta and 'video_auc' in ta[d]]
    cross = float(np.mean(cross_aucs)) if cross_aucs else None
    in_auc = ta.get(IN_DOMAIN_DS, {}).get('video_auc') \
        if IN_DOMAIN_DS in ta else None
    gap = (in_auc - cross) if (in_auc is not None and cross is not None) else None
    return cross, in_auc, gap


def _print_table(results):
    cols = TEST_DS
    hdr = (f"\n  {'Arm':<8s} | {'K':>3s} |"
           + " | ".join(f"{d[:11]:>11s}" for d in cols)
           + f" | {'AUC_cross':>10s} | {'In(+FF++)':>10s} | {'G':>7s}")
    sep = ("  " + "-"*8 + " | " + "-"*3 + " |"
           + " | ".join("-"*11 for _ in cols)
           + " | " + "-"*10 + " | " + "-"*10 + " | " + "-"*7)
    print(hdr)
    print(sep)
    for r in results:
        exp_id = r['exp_name']
        ta = r.get('testall', {})
        k = r.get('lfeq_num_evidence_tokens', '?')
        if r.get('status') != 'OK':
            print(f"  {str(exp_id):<8s} | {str(k):>3s} | "
                  + " | ".join('     N/A    ' for _ in cols)
                  + " |    N/A    |    N/A    |   N/A")
            continue
        cells = []
        for d in cols:
            v = ta.get(d, {}).get('video_auc') if d in ta else None
            cells.append(f"  {v:.4f}  " if v is not None else "     N/A    ")
        cross, in_auc, gap = _cross_and_gap(r)
        cross_s = f"  {cross:.4f}  " if cross is not None else "    N/A    "
        in_s = f"  {in_auc:.4f}  " if in_auc is not None else "    N/A    "
        gap_s = f"  {gap:.4f}  " if gap is not None else "   N/A  "
        print(f"  {str(exp_id):<8s} | {str(k):>3s} | "
              + " | ".join(cells) + f" | {cross_s:>10s} | {in_s:>10s} | {gap_s:>7s}")
    print(f"\n  AUC_cross = mean({', '.join(CROSS_METRIC_DS)}) — G16 discipline. "
          f"In(+FF++) = the in-domain FaceForensics++ column; G = In − AUC_cross. "
          f"Celeb-DF-v2 is partly selection-circular (it is the best-ckpt val set); "
          f"DFDC is the cleanest cross-domain read.  Seed = {args.seed} single-run "
          f"(any positive claim needs ≥3 distinct seeds, G15 discipline).")


def main():
    ap = argparse.ArgumentParser(description='G18-2 LFEQ evidence-token sweep runner')
    ap.add_argument('--arms', nargs='+', default=None,
                    help='Subset of arm ids (e.g. K1 K32). Default: all.')
    ap.add_argument('--output_dir', type=str,
                    default='./experiment_results/g18_2_evidence_sweep',
                    help='Root output dir (per-arm subdirs created under it)')
    ap.add_argument('--n_epochs', type=int, default=10)
    ap.add_argument('--sampler_real_ratio', type=float, default=0.30)
    ap.add_argument('--seed', type=int, default=1024,
                    help='manualSeed for training (deterministic per-run)')
    args = ap.parse_args()

    all_arms = list(G18_2_ARMS)
    if args.arms:
        wanted = set(args.arms)
        all_arms = [e for e in all_arms if e['name'] in wanted]

    print(f"{'='*78}\n  G18-2 LFEQ evidence-token sweep — {len(all_arms)} arms\n"
          f"  test={len(TEST_DS)} sets ({len(CROSS_DS)} cross-domain + {IN_DOMAIN_DS} in-domain)\n"
          f"  cross=mean({', '.join(CROSS_METRIC_DS)})\n"
          f"  sampler_real_ratio={args.sampler_real_ratio}  seed={args.seed}\n"
          f"  output={args.output_dir}\n{'='*78}")

    results = []
    for i, exp in enumerate(all_arms):
        exp_id = f"G18-2/{exp['name']}"
        print(f"\n[{'='*62}]")
        print(f"  [{i+1}/{len(all_arms)}] {exp_id}  "
              f"model=effort_lfeq  fusion={LFEQ_PINNED['lfeq_fusion_weight']}  "
              f"K={exp['lfeq_num_evidence_tokens']}")
        print(f"[{'='*62}]")
        r = run_one(exp, args)
        r['model_name'] = 'effort_lfeq'
        results.append(r)
        results_path = os.path.join(args.output_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    _print_table(results)

    print(f"\n  Full per-arm per-dataset metrics: {args.output_dir}/all_results.json")
    print(f"  Note: each arm's frame-level confusion matrix + KDE score plots "
          f"live under {args.output_dir}/<arm>/.  Between-arm CUDA cache is "
          f"cleared and each train/test runs in a fresh subprocess, so arms "
          f"cannot interfere.")


if __name__ == '__main__':
    main()

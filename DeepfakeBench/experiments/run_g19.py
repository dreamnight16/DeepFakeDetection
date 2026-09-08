"""
G19 — token read-out head swipe: mean / per-token / concat (three arms).

Follows the G6_baseline protocol (frozen CLIP ViT-L/14 + LoRA, FF++ train,
Celeb-DF-v2 best-ckpt selection, ``sampler_real_ratio=0.30`` v1 balance sampler,
no mixup) and changes exactly ONE dimension: the classification read-out over the
LFEQ query tokens.  All three arms share the SAME query-transformer body (K=8,
hidden=256, depth=2, heads=8, dropout=0.1 — parameter-identical to G18 L1) and
differ ONLY in how the K+1 query tokens collapse to [B,2]:

    A    'effort_lfeq_mean'         mean( queries )       -> Linear(256,2)   shared
    B    'effort_lfeq_per_token'    mean_i( Linear_i(q_i) )-> K+1 independent Linear(256,2)
    C    'effort_lfeq_concat'       flatten( queries )    -> Linear(256*9,2) wide

No hard-argmax, no fusion_weight, no evidence/diversity loss in any arm.  A is
the capacity-freeze anchor; B and C each carry ~9x the head capacity (capacity
mis-matched ON PURPOSE and reported — it's the degree of freedom being isolated).

Anchors (NOT re-run; same seed/session, cross-validation from G18-2 §6.5):
    L1   effort_lfeq   K=8 fusion=0.5  full LFEQ read-out  7-test avg ≈ 0.93158
    B0   effort        pooler->linear  baseline read-out   7-test avg ≈ 0.93663

Test = the SAME 7 cross-domain sets + a separate FaceForensics++ in-domain column.
Cross-domain aggregate = mean(Celeb-DF-v2, DFDC) (G16 discipline; NEVER the 7-set
average and NEVER including FF++).  G = AUC_FFpp - AUC_cross.

── ISOLATION ────────────────────────────────────────────────────────────────
Each arm runs fully isolated: its own output / log / ckpt / results dirs, its own
config (model_name + lfeq_* keys), no warm-start / init_ckpt, and each
train_model() / evaluate_model() submits a FRESH Python subprocess so CUDA is
released on exit; torch.cuda.empty_cache() between arms.

Usage:
    python experiments/run_g19.py                     # A + B + C, n_epochs=10, seed=1024
    python experiments/run_g19.py --seed 2048
    python experiments/run_g19.py --only A B          # run a subset of arms
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
CROSS_DS = ['WDF', 'FFIW', 'Celeb-DF-v2', 'DeepFakeDetection',
            'DFDC', 'DFDCP', 'DeeperForensics-1.0']
IN_DOMAIN_DS = 'FaceForensics++'
TEST_DS = CROSS_DS + [IN_DOMAIN_DS]
CROSS_METRIC_DS = ['Celeb-DF-v2', 'DFDC']

# G19 structural keys: identical query-transformer body to G18 L1, K=8 optimal.
# readout_mode is pinned by the detector subclass (model_name), NOT a config key.
LFEQ_STRUCT_KWARGS = dict(
    lfeq_hidden_dim=256, lfeq_num_evidence_tokens=8,
    lfeq_depth=2, lfeq_num_heads=8, lfeq_dropout=0.1,
    lfeq_fusion_weight=0.5, lfeq_evidence_weight=1.0, lfeq_diversity_weight=0.01,
)

# The three G19 arms: (exp_id, subdir, module_name).
ARMS = [
    ('G19/A', 'A', 'effort_lfeq_mean'),
    ('G19/B', 'B', 'effort_lfeq_per_token'),
    ('G19/C', 'C', 'effort_lfeq_concat'),
]


def run_one(args, arm):
    """Train + eval ONE G19 arm in its own isolated dir. Returns summary dict."""
    exp_id, subdir, model_name = arm
    output_dir = os.path.join(args.output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs', subdir)

    kwargs = dict(
        use_mixup=False, mixup_loss_strip=False,
        sampler_real_ratio=args.sampler_real_ratio,
        model_name=model_name,
        log_dir=log_dir, train_dataset=TRAIN_DS, test_dataset=VAL_DS,
        n_epochs=args.n_epochs,
        **{k: (args.num_evidence_tokens if k == 'lfeq_num_evidence_tokens' else v)
           for k, v in LFEQ_STRUCT_KWARGS.items()},
    )

    config = build_config(**kwargs)
    config['manualSeed'] = args.seed

    ckpt = train_model(config, TRAIN_DS, VAL_DS)
    if ckpt is None:
        print(f"[{exp_id}] TRAIN FAILED")
        return {'exp_name': exp_id, 'model_name': model_name, 'status': 'TRAIN_FAILED'}

    # Eval config mirrors train config exactly; arch_keys propagate model_name +
    # lfeq_* so test.py rebuilds the SAME read-out head (strict load).
    config_eval = build_config(**{**kwargs, 'n_epochs': 0,
                                  'test_dataset': TEST_DS,
                                  'for_training': False})
    config_eval['manualSeed'] = args.seed
    summary = evaluate_model(config_eval, ckpt, TEST_DS, TRAIN_DS,
                             output_dir, exp_id)
    summary['ckpt'] = ckpt
    summary['status'] = 'OK'
    summary['exp_name'] = exp_id
    summary['model_name'] = model_name
    return summary


def _cross_and_gap(summary):
    ta = summary.get('testall', {})
    cross_aucs = [ta[d]['video_auc'] for d in CROSS_METRIC_DS
                  if d in ta and 'video_auc' in ta[d]]
    cross = float(np.mean(cross_aucs)) if cross_aucs else None
    in_auc = ta.get(IN_DOMAIN_DS, {}).get('video_auc') if IN_DOMAIN_DS in ta else None
    gap = (in_auc - cross) if (in_auc is not None and cross is not None) else None
    return cross, in_auc, gap


def _print_table(results, seed):
    cols = TEST_DS
    hdr = (f"\n  {'Arm':<8s} | {'Model':<20s} |"
           + " | ".join(f"{d[:11]:>11s}" for d in cols)
           + f" | {'AUC_cross':>10s} | {'In(+FF++)':>10s} | {'G':>7s}")
    sep = ("  " + "-"*8 + " | " + "-"*20 + " |"
           + " | ".join("-"*11 for _ in cols)
           + " | " + "-"*10 + " | " + "-"*10 + " | " + "-"*7)
    print(hdr)
    print(sep)
    for r in results:
        exp_id = r['exp_name']
        ta = r.get('testall', {})
        if r.get('status') != 'OK':
            cells = ['     N/A    ' for _ in cols]
            cross_s = in_s = gap_s = "    N/A    "
            model = r.get('status', '?')
        else:
            cells = []
            for d in cols:
                v = ta.get(d, {}).get('video_auc') if d in ta else None
                cells.append(f"  {v:.4f}  " if v is not None else "     N/A    ")
            cross, in_auc, gap = _cross_and_gap(r)
            cross_s = f"  {cross:.4f}  " if cross is not None else "    N/A    "
            in_s = f"  {in_auc:.4f}  " if in_auc is not None else "    N/A    "
            gap_s = f"  {gap:.4f}  " if gap is not None else "   N/A  "
            model = r.get('model_name', '?')
        print(f"  {str(exp_id):<8s} | {model:<20s} | "
              + " | ".join(cells) + f" | {cross_s:>10s} | {in_s:>10s} | {gap_s:>7s}")
    print(f"\n  AUC_cross = mean({', '.join(CROSS_METRIC_DS)}) — G16 discipline. "
          f"In(+FF++) = in-domain FaceForensics++ column; G = In − AUC_cross. "
          f"Celeb-DF-v2 is partly selection-circular (best-ckpt val set); DFDC is "
          f"the cleanest cross-domain read.  Seed = {seed} single-run (a "
          f"positive claim needs ≥3 distinct seeds, G15 discipline).")
    print(f"\n  Head capacity: A ≈ 514 params (shared), B ≈ {256*2*(8+1):,} "
          f"params (K+1 independent), C ≈ {256*(8+1)*2:,} params (concat). "
          f"B and C mis-match A by ~9x — this is the degree of freedom isolated.")


def main():
    ap = argparse.ArgumentParser(description='G19 token read-out swipe (A/B/C)')
    ap.add_argument('--output_dir', type=str,
                    default='./experiment_results/g19_readout',
                    help='Root output dir (arm subdir under it)')
    ap.add_argument('--n_epochs', type=int, default=10)
    ap.add_argument('--sampler_real_ratio', type=float, default=0.30)
    ap.add_argument('--seed', type=int, default=1024)
    ap.add_argument('--num_evidence_tokens', type=int, default=8,
                    help='K for the query-transformer body (G19 uses K=8 optimal)')
    ap.add_argument('--only', nargs='*', default=None,
                    choices=['A', 'B', 'C'],
                    help='Run only these arms (default: all three)')
    args = ap.parse_args()

    arms = ARMS if not args.only else [a for a in ARMS if a[1] in args.only]

    print(f"{'='*78}\n  G19 token read-out swipe — {', '.join(a[1] for a in arms)}\n"
          f"  test={len(TEST_DS)} sets ({len(CROSS_DS)} cross-domain + {IN_DOMAIN_DS} in-domain)\n"
          f"  cross=mean({', '.join(CROSS_METRIC_DS)})\n"
          f"  K={args.num_evidence_tokens}  hidden=256 depth=2 heads=8\n"
          f"  sampler_real_ratio={args.sampler_real_ratio}  seed={args.seed}\n"
          f"  output={args.output_dir}\n{'='*78}")

    results = []
    for arm in arms:
        r = run_one(args, arm)
        results.append(r)
        torch_cuda_empty()
        results_path = os.path.join(args.output_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    _print_table(results, args.seed)
    print(f"\n  Full per-arm per-dataset metrics: {args.output_dir}/all_results.json")


def torch_cuda_empty():
    """Best-effort CUDA release between arms (no-op if no CUDA / not imported)."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == '__main__':
    main()

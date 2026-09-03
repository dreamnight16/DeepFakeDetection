"""
G18 — LFEQ read-out head vs baseline read-out (single sequence, isolated arms).

Reproduces the G6_baseline protocol (frozen CLIP ViT-L/14 + LoRA, FF++ train,
Celeb-DF-v2 best-ckpt selection, ``sampler_real_ratio=0.30`` v1 balance sampler,
no mixup) and changes exactly one variable per arm: the classification read-out.

    B0   anchor  'effort'        pooler->linear read-out (baseline, ~0.9366)
    L1   primary 'effort_lfeq'   LFEQ head, fusion_weight=0.5 (global+evidence)
    L2           'effort_lfeq'   LFEQ head, fusion_weight=1.0 (global token only)
    L3           'effort_lfeq'   LFEQ head, fusion_weight=0.0 (evidence token only)

Test = the SAME 7 cross-domain sets + a separate FaceForensics++ in-domain column.
Cross-domain aggregate = mean(Celeb-DF-v2, DFDC) (G16 discipline; NEVER the 7-set
average and NEVER including FF++).  G = AUC_FFpp - AUC_cross.

── ISOLATION ────────────────────────────────────────────────────────────────
Every arm is run fully in isolation so no two arms can interfere:
  * per-arm output / log / ckpt / results  dirs  (no shared namespace);
  * per-arm config  (effort vs effort_lfeq; each lfeq arm sets ONLY its own
    lfeq_fusion_weight — the two fusion values are never mixed on one arm);
  * no cross-arm checkpoint reuse  (no init_ckpt / warm-start; every arm trains
    its own ckpt, the B0 anchor is trained in-session, never referenced);
  * train_model() and evaluate_model() each submit a FRESH Python subprocess,
    so every arm gets its own CUDA context released on process exit;
  * torch.cuda.empty_cache() after each arm (evaluate_model already
    `del model; torch.cuda.empty_cache()`).

Usage:
    python experiments/run_g18_lfeq.py                  # all 4 arms
    python experiments/run_g18_lfeq.py --arms L1 L3     # subset
    python experiments/run_g18_lfeq.py --n_epochs 10 --seed 2048
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


# ── G18 arms ────────────────────────────────────────────────────────────────
G18_ARMS = [
    # B0 — baseline anchor.  Reproduces the effort RGB cross ≈ 0.928
    # (video_auc 7-test avg ≈ 0.9366) in-session.
    {'name': 'B0', 'model_name': 'effort'},
    # L1 — PRIMARY: LFEQ read-out, global+evidence fused (fusion=0.5).
    {'name': 'L1', 'model_name': 'effort_lfeq', 'lfeq_fusion_weight': 0.5},
    # L2 — LFEQ global decision token only (evidence branch weight 0).
    {'name': 'L2', 'model_name': 'effort_lfeq', 'lfeq_fusion_weight': 1.0},
    # L3 — LFEQ hard maximum-evidence branch only (global weight 0).
    {'name': 'L3', 'model_name': 'effort_lfeq', 'lfeq_fusion_weight': 0.0},
]


def run_one(exp, args):
    """Train + eval a single arm in its own isolated dir. Returns summary dict."""
    exp_id = f"G18/{exp['name']}"
    output_dir = os.path.join(args.output_dir, exp['name'])
    os.makedirs(output_dir, exist_ok=True)
    # Per-arm isolated log dir — ckpts and run logs never collide across arms.
    log_dir = os.path.join(args.output_dir, 'logs', exp['name'])

    # Build config with ONLY this arm's params.  B0 sets no lfeq_* keys; each
    # L-arm sets only its own lfeq_fusion_weight (never another arm's value).
    kwargs = dict(
        use_mixup=False, mixup_loss_strip=False,
        sampler_real_ratio=args.sampler_real_ratio,
        model_name=exp['model_name'],
        log_dir=log_dir, train_dataset=TRAIN_DS, test_dataset=VAL_DS,
        n_epochs=args.n_epochs,
    )
    if exp['model_name'] == 'effort_lfeq':
        kwargs['lfeq_fusion_weight'] = exp['lfeq_fusion_weight']

    config = build_config(**kwargs)
    config['manualSeed'] = args.seed      # deterministic per-run seed

    ckpt = train_model(config, TRAIN_DS, VAL_DS)
    if ckpt is None:
        print(f"[{exp_id}] TRAIN FAILED")
        return {'exp_name': exp_id, 'status': 'TRAIN_FAILED'}

    # Eval config mirrors train config exactly so test.py rebuilds the model and
    # dataset identically (arch_keys propagate model_name + lfeq_* to testall).
    config_eval = build_config(**{**kwargs, 'n_epochs': 0,
                                  'test_dataset': TEST_DS,
                                  'for_training': False})
    config_eval['manualSeed'] = args.seed
    summary = evaluate_model(config_eval, ckpt, TEST_DS, TRAIN_DS,
                             output_dir, exp_id)
    summary['ckpt'] = ckpt
    summary['status'] = 'OK'
    summary['exp_name'] = exp_id
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
    hdr = (f"\n  {'Arm':<8s} | {'Model':<13s} |"
           + " | ".join(f"{d[:11]:>11s}" for d in cols)
           + f" | {'AUC_cross':>10s} | {'In(+FF++)':>10s} | {'G':>7s}")
    sep = ("  " + "-"*8 + " | " + "-"*13 + " |"
           + " | ".join("-"*11 for _ in cols)
           + " | " + "-"*10 + " | " + "-"*10 + " | " + "-"*7)
    print(hdr)
    print(sep)
    for r in results:
        exp_id = r['exp_name']
        ta = r.get('testall', {})
        cells = []
        if r.get('status') != 'OK':
            cells = ['     N/A    ' for _ in cols]
            cross_s = in_s = gap_s = "    N/A    "
        else:
            for d in cols:
                v = ta.get(d, {}).get('video_auc') if d in ta else None
                cells.append(f"  {v:.4f}  " if v is not None else "     N/A    ")
            cross, in_auc, gap = _cross_and_gap(r)
            cross_s = f"  {cross:.4f}  " if cross is not None else "    N/A    "
            in_s = f"  {in_auc:.4f}  " if in_auc is not None else "    N/A    "
            gap_s = f"  {gap:.4f}  " if gap is not None else "   N/A  "
        model = r.get('model_name', '?') if r.get('status') == 'OK' else r.get('status', '?')
        print(f"  {str(exp_id):<8s} | {model:<13s} | "
              + " | ".join(cells) + f" | {cross_s:>10s} | {in_s:>10s} | {gap_s:>7s}")
    print(f"\n  AUC_cross = mean({', '.join(CROSS_METRIC_DS)}) — G16 discipline. "
          f"In(+FF++) = the in-domain FaceForensics++ column; G = In − AUC_cross. "
          f"Celeb-DF-v2 is partly selection-circular (it is the best-ckpt val set); "
          f"DFDC is the cleanest cross-domain read.  Seed = {args.seed} single-run "
          f"(any positive claim needs ≥3 distinct seeds, G15 discipline).")


def main():
    ap = argparse.ArgumentParser(description='G18 LFEQ read-out sequence runner')
    ap.add_argument('--arms', nargs='+', default=None,
                    help='Subset of arm ids (e.g. L1 L3). Default: all.')
    ap.add_argument('--output_dir', type=str,
                    default='./experiment_results/g18_lfeq',
                    help='Root output dir (per-arm subdirs created under it)')
    ap.add_argument('--n_epochs', type=int, default=10)
    ap.add_argument('--sampler_real_ratio', type=float, default=0.30)
    ap.add_argument('--seed', type=int, default=1024,
                    help='manualSeed for training (deterministic per-run)')
    args = ap.parse_args()

    all_arms = list(G18_ARMS)
    if args.arms:
        wanted = set(args.arms)
        all_arms = [e for e in all_arms if e['name'] in wanted]

    print(f"{'='*78}\n  G18 LFEQ read-out — {len(all_arms)} arms\n"
          f"  test={len(TEST_DS)} sets ({len(CROSS_DS)} cross-domain + {IN_DOMAIN_DS} in-domain)\n"
          f"  cross=mean({', '.join(CROSS_METRIC_DS)})\n"
          f"  sampler_real_ratio={args.sampler_real_ratio}  seed={args.seed}\n"
          f"  output={args.output_dir}\n{'='*78}")

    results = []
    for i, exp in enumerate(all_arms):
        exp_id = f"G18/{exp['name']}"
        print(f"\n[{'='*62}]")
        print(f"  [{i+1}/{len(all_arms)}] {exp_id}  model={exp['model_name']}"
              + (f"  fusion={exp['lfeq_fusion_weight']}"
                 if exp['model_name'] == 'effort_lfeq' else ""))
        print(f"[{'='*62}]")
        r = run_one(exp, args)
        r['model_name'] = exp['model_name']
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

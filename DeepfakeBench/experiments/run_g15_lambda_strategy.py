"""
G15 supplement (urgent) — detection-time two-head average (Strategy-3)
+ lambda_max sweep over {0.1, 0.5, 2, 10}.

For each lambda_max we train the G15 PRIMARY ``no_init`` effort_maxev ONCE, then
evaluate that EXACT checkpoint under BOTH detection-time score rules:

    'cls'  — CLS head alone  (doc Section 16; the default G15 score)
    'avg'  — Strategy-3 two-head average  0.5*P_cls + 0.5*max_i q_{i,1}

Because ``max_evidence_inference`` only alters the inference forward (no
weight-shape change), the same checkpoint can be scored under both rules, so this
is a clean within-lambda_max ablation : the SAME trained model, two score
functions.  Averaging the heads is what the user asked for ("检测时用两个分类头
的平均"); it is gated on ``inference=True`` so the training loss is unchanged.

Isolation: per-lambda_max  output / log / ckpt  dirs; each train and each of the
two evals runs in a fresh subprocess (train.py / testall.py), so arms cannot
interfere; torch.cuda.empty_cache() + explicit model free after each eval
(evaluate_model already does ``del model; torch.cuda.empty_cache()``).

Usage:
    python3 experiments/run_g15_lambda_strategy.py                        # all 4 lambdas
    python3 experiments/run_g15_lambda_strategy.py --lambdas 0.5 2        # subset
    python3 experiments/run_g15_lambda_strategy.py --cls_feature pooler_output
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

# The 7 cross-domain test sets (G15-consistent).  FF++ is NOT here: it is the
# in-domain training set, reported as a SEPARATE column (see IN_DOMAIN_DS).
CROSS_DS = ['WDF', 'FFIW', 'Celeb-DF-v2', 'DeepFakeDetection',
            'DFDC', 'DFDCP', 'DeeperForensics-1.0']
IN_DOMAIN_DS = 'FaceForensics++'
TEST_DS = CROSS_DS + [IN_DOMAIN_DS]

# Cross-domain aggregate = mean(Celeb-DF-v2, DFDC) — G16 discipline.
CROSS_METRIC_DS = ['Celeb-DF-v2', 'DFDC']

LAMBDA_MAXS = [0.1, 0.5, 2, 10]
INFERENCE_MODES = ('cls', 'avg')   # Strategy-3 runs both, so 'avg' is always scored


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


def run_one(lambda_max, args):
    """Train effort_maxev (no_init) once at lambda_max, then eval the SAME ckpt
    under both detection-time rules ('cls' and 'avg'). Returns a results dict."""
    lam_str = f"{lambda_max:g}"
    exp_id = f"G15s_lmax{lam_str}"
    output_dir = os.path.join(args.output_dir, 'G15_supplement', f"lmax{lam_str}")
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs', 'G15_supplement', f"lmax{lam_str}")

    kwargs = dict(
        use_mixup=False, mixup_loss_strip=False,
        sampler_real_ratio=args.sampler_real_ratio,
        model_name='effort_maxev',
        max_evidence_lambda=lambda_max,
        max_evidence_eps=1e-8,
        max_evidence_cls_feature=args.cls_feature,
        max_evidence_inference='cls',   # train-time rule (irrelevant to loss)
        log_dir=log_dir,
        train_dataset=TRAIN_DS, test_dataset=VAL_DS,
        n_epochs=args.n_epochs,
    )
    config = build_config(**kwargs)

    ckpt = train_model(config, TRAIN_DS, VAL_DS)
    if ckpt is None:
        print(f"[{exp_id}] TRAIN FAILED")
        return {'exp_name': exp_id, 'status': 'TRAIN_FAILED', 'lambda_max': lambda_max}

    results = {'exp_name': exp_id, 'lambda_max': lambda_max,
               'ckpt': ckpt, 'status': 'OK'}
    for mode in INFERENCE_MODES:
        # Same ckpt, different detection-time rule.  eval config mirrors train
        # config exactly except max_evidence_inference (a forward-only flag, no
        # weight-shape change -> the ckpt strict-loads under both rules).
        kwargs_eval = {**kwargs, 'n_epochs': 0, 'test_dataset': TEST_DS,
                       'for_training': False, 'max_evidence_inference': mode}
        config_eval = build_config(**kwargs_eval)
        eval_dir = os.path.join(output_dir, mode)
        os.makedirs(eval_dir, exist_ok=True)
        summary = evaluate_model(config_eval, ckpt, TEST_DS, TRAIN_DS, eval_dir,
                                 f"{exp_id}/{mode}")
        results[mode] = summary
    return results


def _print_table(results):
    print(f"\n  G15 supplement — detection-time heads: 'cls' (CLS alone) vs 'avg' "
          f"(Strategy-3 two-head average).  cross = mean({', '.join(CROSS_METRIC_DS)}); "
          f"In = {IN_DOMAIN_DS} column;  same ckpt per lambda_max, two score rules.\n")
    # Uniform column widths: lambda_max 9, rule 4, each cross dset 10, then the
    # trailing In+FF++ / cross / G at 8/8/6.  Every row (incl. TRAIN_FAILED) uses
    # the identical " | " separators so columns always line up.
    cross_w = 10
    trail = [('In+FF++', 8), ('cross', 8), ('G', 6)]
    hdr_cells = [f"{d[:cross_w]:>{cross_w}s}" for d in CROSS_DS]
    hdr_trail = [f"{name:>{w}s}" for name, w in trail]
    print(f"  {'lambda_max':>9s} | {'rule':>4s} | " +
          " | ".join(hdr_cells) + " | " + " | ".join(hdr_trail))
    print("  " + "-"*9 + " | " + "-"*4 + " | " +
          " | ".join("-"*cross_w for _ in CROSS_DS) + " | " +
          " | ".join("-"*w for _, w in trail))
    for r in results:
        lam_s = f"{r['lambda_max']:g}"
        if r.get('status') != 'OK':
            print(f"  {lam_s:>9s} | {'FAIL':>4s} | " +
                  " | ".join(f"{'N/A':>{cross_w}s}" for _ in CROSS_DS) +
                  " | " + " | ".join(f"{'N/A':>{w}s}" for _, w in trail))
            continue
        for mode in INFERENCE_MODES:
            ta = r.get(mode, {}).get('testall', {})
            cells = [
                f"{ta[d]['video_auc']:.4f}"
                if (ta and d in ta and 'video_auc' in ta[d]) else "N/A"
                for d in CROSS_DS
            ]
            cross, in_auc, gap = _cross_and_gap(r.get(mode, {})) if ta else (None, None, None)
            cross_s = f"{cross:.4f}" if cross is not None else "N/A"
            in_s = f"{in_auc:.4f}" if in_auc is not None else "N/A"
            gap_s = f"{gap:.4f}" if gap is not None else "N/A"
            print(f"  {lam_s:>9s} | {mode:>4s} | " +
                  " | ".join(f"{c:>{cross_w}s}" for c in cells) +
                  f" | {in_s:>8s} | {cross_s:>8s} | {gap_s:>6s}")
    print(f"\n  Strategy-3 'avg' blurs CLS-head score with the max-fake-evidence "
          f"patch head (0.5/0.5).  'cls' = doc Section 16 (CLS alone) — the G15 "
          f"default.  A positive strategy effect = 'avg' > 'cls' at the same "
          f"lambda_max; still subject to the ≥3-seed rule for any non-null claim.")


def main():
    ap = argparse.ArgumentParser(description='G15 supplement: lambda_max sweep + '
                                             'Strategy-3 two-head average')
    ap.add_argument('--lambdas', type=float, nargs='+', default=LAMBDA_MAXS,
                    help='lambda_max values to train+evaluate (default %s)' % LAMBDA_MAXS)
    ap.add_argument('--cls_feature', type=str, default='raw_token',
                    choices=('raw_token', 'pooler_output'),
                    help="CLS-head input: 'raw_token' (G15 PRIMARY, doc §3/§4.1) "
                         "or 'pooler_output' (anchor-aligned). Default: raw_token.")
    ap.add_argument('--output_dir', type=str, default='./experiment_results/g15_supplement',
                    help='Root output dir (per-lambda_max subdirs under it)')
    ap.add_argument('--n_epochs', type=int, default=10)
    ap.add_argument('--sampler_real_ratio', type=float, default=0.30)
    args = ap.parse_args()

    print(f"{'='*78}\n  G15 supplement — lambda_max sweep {args.lambdas} "
          f"({args.cls_feature} CLS head)\n"
          f"  test={len(TEST_DS)} sets ({len(CROSS_DS)} cross-domain + "
          f"{IN_DOMAIN_DS} in-domain), cross=mean({', '.join(CROSS_METRIC_DS)})\n"
          f"  output={args.output_dir}\n{'='*78}")

    results = []
    for i, lam in enumerate(args.lambdas):
        print(f"\n[{'='*62}]\n  [{i+1}/{len(args.lambdas)}] lambda_max={lam:g}\n"
              f"[{'='*62}]")
        r = run_one(lam, args)
        results.append(r)
        with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

    _print_table(results)
    print(f"\n  Full per-arm per-dataset metrics: "
          f"{args.output_dir}/all_results.json\n"
          f"  Note: each lambda_max trains ONCE and is scored twice (cls/avg) on "
          f"the same ckpt; each train/eval runs in a fresh subprocess.")


if __name__ == '__main__':
    main()

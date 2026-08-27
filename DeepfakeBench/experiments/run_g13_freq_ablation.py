"""
G13 — data-side frequency-band isolation ablation.

Question this protocol answers (no model change, controlled single variable):
    "Which frequency range of the input image carries the deepfake signal?"

Keeps only ONE radial FFT band of each input image (Low / Mid-Low / Mid-High /
High) and reconstructs it back to 3-channel RGB inside the dataset (see
training/dataset/utils/freq_band.py).  The model, training protocol, data split
and augmentation are IDENTICAL across all experiments — only the input transform
differs — so RGB vs the four bands is a clean single-variable ablation.

Use_mixup is forced OFF for all experiments: the mixup pipeline does its own
frequency-domain blending, which would confound a frequency-isolation study.

Round-1 protocol (cross-domain focus):
    Train   : FaceForensics++
    Validate: Celeb-DF-v2            (cross-domain best-ckpt selection)
    Test    : FF++ / Celeb-DF-v2 / DFDC

Usage:
    python3 experiments/run_g13_freq_ablation.py                          # RGB + 4 bands
    python3 experiments/run_g13_freq_ablation.py --names RGB High         # subset
    python3 experiments/run_g13_freq_ablation.py --full                   # evaluate all 7 test datasets
    python3 experiments/run_g13_freq_ablation.py --norm none              # raw band amplitude (control)
    python3 experiments/run_g13_freq_ablation.py --energy-match --norm none   # 2nd G13 experiment variant
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
# Round-1 cross-domain test set (the user's stated focus).  --full for the 7.
ROUND1_TEST_DS = ['FaceForensics++', 'Celeb-DF-v2', 'DFDC']
FULL_TEST_DS = ['WDF', 'FFIW', 'Celeb-DF-v2', 'DeepFakeDetection',
                'DFDC', 'DFDCP', 'DeeperForensics-1.0']

EXPERIMENTS = [
    {'name': 'RGB',      'freq_ablation': None},
    {'name': 'Low',      'freq_ablation': 'low'},
    {'name': 'Mid-Low',  'freq_ablation': 'mid_low'},
    {'name': 'Mid-High', 'freq_ablation': 'mid_high'},
    {'name': 'High',     'freq_ablation': 'high'},
]


def run_one(exp, args):
    """Train + eval a single frequency-band config. Returns summary dict."""
    exp_name = exp['name']
    output_dir = os.path.join(args.output_dir, 'bands', exp_name)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs', exp_name)

    config = build_config(
        # Frequency isolation must not be confounded by FFT-domain mixup.
        pyramid_mode='lap_pyramid', use_mixup=False, mixup_loss_strip=False,
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
        freq_ablation=exp['freq_ablation'],
        freq_norm=args.norm,
        freq_energy_match=args.energy_match,
        freq_after_aug=not args.before_aug,
        log_dir=log_dir,
        train_dataset=TRAIN_DS, test_dataset=VAL_DS,
        n_epochs=args.n_epochs,
    )

    ckpt = train_model(config, TRAIN_DS, VAL_DS)
    if ckpt is None:
        print(f"[{exp_name}] TRAIN FAILED")
        return {'exp_name': exp_name, 'status': 'TRAIN_FAILED',
                'freq_ablation': exp['freq_ablation']}

    test_ds = FULL_TEST_DS if args.full_test else args.test_datasets
    config_eval = build_config(
        pyramid_mode='lap_pyramid', use_mixup=False, mixup_loss_strip=False,
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
        freq_ablation=exp['freq_ablation'],
        freq_norm=args.norm,
        freq_energy_match=args.energy_match,
        freq_after_aug=not args.before_aug,
        n_epochs=0, train_dataset=TRAIN_DS, test_dataset=test_ds,
        for_training=False,
    )
    summary = evaluate_model(config_eval, ckpt, test_ds, TRAIN_DS,
                             output_dir, exp_name)
    summary['ckpt'] = ckpt
    summary['freq_ablation'] = exp['freq_ablation']
    summary['freq_norm'] = args.norm
    summary['freq_energy_match'] = args.energy_match
    summary['status'] = 'OK'
    return summary


def main():
    parser = argparse.ArgumentParser(description='G13 frequency-band ablation')
    parser.add_argument('--names', nargs='+', default=None,
                        help='Subset of experiment names (e.g. RGB High)')
    parser.add_argument('--output_dir', type=str,
                        default='./experiment_results/g13_freq_ablation',
                        help='Output directory for all results')
    parser.add_argument('--num_levels', type=int, default=3)
    parser.add_argument('--sampler_real_ratio', type=float, default=0.30)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--test_datasets', nargs='+', default=ROUND1_TEST_DS,
                        help='Test datasets (default: the round-1 cross-domain trio)')
    parser.add_argument('--full', dest='full_test', action='store_true',
                        help='Evaluate on the full 7-dataset list')
    parser.add_argument('--norm', choices=['minmax', 'none'], default='minmax',
                        help="Band reconstruction normalisation: 'minmax' (G13 "
                             "default, removes the energy/contrast shortcut) or "
                             "'none' (raw amplitude, diagnostic control)")
    parser.add_argument('--energy-match', dest='energy_match', action='store_true',
                        help='Second G13 experiment: equalise band L2 energy '
                             '(combine with --norm none; see freq_band.py caveat)')
    parser.add_argument('--before-aug', dest='before_aug', action='store_true',
                        help='Filter BEFORE augmentation (literal load→filter→'
                             'augment reading). Default False: filter AFTER '
                             'augmentation so the model input is strictly in-band.')
    args = parser.parse_args()

    # G13 honest guard: energy-match is a no-op under the default minmax norm
    # (the per-image min-max stretch re-absorbs the global L2 scale), so running
    # it as-is records freq_energy_match=True while changing nothing.  Round-1
    # uses minmax + energy_match=False by design; only run energy-match with
    # --norm none (and even then it is the round-2 variant, deferred).
    if args.energy_match and args.norm == 'minmax':
        print("  [WARN] --energy-match is a NO-OP under --norm minmax (the "
              "per-image min-max stretch re-absorbs the global L2 scale). "
              "Run with --norm none, or leave it off for round-1.")

    exp_list = EXPERIMENTS
    if args.names:
        exp_list = [e for e in EXPERIMENTS if e['name'] in args.names]

    total = len(exp_list)
    test_display = FULL_TEST_DS if args.full_test else ROUND1_TEST_DS
    print(f"{'='*72}\n  G13 Frequency-Band Ablation — {total} bands\n"
          f"  norm={args.norm}  energy_match={args.energy_match}  "
          f"filter_after_aug={not args.before_aug}\n"
          f"  test={test_display}\n{'='*72}")

    results = []
    for i, exp in enumerate(exp_list):
        exp_name = exp['name']
        print(f"\n[{'='*60}]")
        print(f"  [{i+1}/{total}] {exp_name}  freq_ablation={exp['freq_ablation']}")
        print(f"[{'='*60}]")
        r = run_one(exp, args)
        results.append(r)
        results_path = os.path.join(args.output_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}\n  SUMMARY TABLE  — per-dataset video_auc  "
          f"(anchor: G6_baseline no-mixup plain RGB video_auc = 0.9366, 7-test avg)\n{'='*80}")
    cols = FULL_TEST_DS if args.full_test else args.test_datasets
    cross_ds = [d for d in cols if d != TRAIN_DS]
    hdr = (f"  {'Band':<10s} | {'Status':<9s} | "
           + " | ".join(f"{d[:13]:>13s}" for d in cols)
           + f" | {'AUC_cross':>9s} | {'avg':>8s}")
    sep = (f"  {'-'*10} | {'-'*9} | "
           + " | ".join('-'*13 for _ in cols)
           + f" | {'-'*9} | {'-'*8}")
    print(hdr)
    print(sep)
    for r in results:
        status = r.get('status', '?')
        ta = r.get('testall', {})
        cells = []
        for d in cols:
            if d in ta and 'video_auc' in ta[d]:
                cells.append(f"  {ta[d]['video_auc']:.4f}  ")
            else:
                cells.append("     N/A    ")
        cross_aucs = [ta[d]['video_auc'] for d in cross_ds
                      if d in ta and 'video_auc' in ta[d]]
        cross = f"  {np.mean(cross_aucs):.4f}  " if cross_aucs else "    N/A    "
        avg = ta.get('average', {}).get('video_auc', 'N/A')
        avg = f"  {avg:.4f}  " if isinstance(avg, (int, float)) else "  'N/A'  "
        print(f"  {r['exp_name']:<10s} | {status:<9s} | "
              + " | ".join(cells) + f" | {cross:>9s} | {avg:>8s}")
    print(f"\n  AUC_cross = mean video_auc over {cross_ds} "
          f"(excludes in-domain {TRAIN_DS}); Celeb-DF-v2 column is the same "
          f"objective used for best-ckpt selection, so it is partly "
          f"selection-circular — DFDC is the cleanest cross-domain read.")
    print(f"\n  Per-dataset video_auc also stored in: "
          f"{args.output_dir}/all_results.json")
    print(f"Results saved to: {args.output_dir}/all_results.json")


if __name__ == '__main__':
    main()

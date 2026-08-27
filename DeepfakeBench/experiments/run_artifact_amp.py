"""
Artifact-Amplified Pyramid Mixup — 3-tier ablation (amp_max ∈ {1, 2, 3}).

RF samples inject the fake artifact direction L_f − L_r at strength
a ~ Uniform(0, amp_max), with soft label y = a/(1+a); coarse structure G_K
stays 100% real. Compare against the already-run lap_pyramid baseline
(G4_exp1_soft_ce / G6_pyramid_only).

Usage:
    python3 experiments/run_artifact_amp.py                 # all 3
    python3 experiments/run_artifact_amp.py --amps 1 2      # specific tiers
"""
import os
import sys
import argparse
import json

_current_dir = os.path.dirname(os.path.abspath(__file__))
_deepfake_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _current_dir)
sys.path.insert(0, _deepfake_dir)

from experiment_utils import build_config, train_model, evaluate_model

TRAIN_DS = 'FaceForensics++'
VAL_DS = 'Celeb-DF-v2'
TEST_DS = ['WDF', 'FFIW', 'Celeb-DF-v2', 'DeepFakeDetection',
           'DFDC', 'DFDCP', 'DeeperForensics-1.0']

EXPERIMENTS = [
    {'name': 'artifact_amp_a1', 'amp_max': 1.0},
    {'name': 'artifact_amp_a2', 'amp_max': 2.0},
    {'name': 'artifact_amp_a3', 'amp_max': 3.0},
]


def run_one(exp, args):
    """Train + eval a single amp_max tier. Returns summary dict."""
    exp_name = exp['name']
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs', exp_name)

    config = build_config(
        pyramid_mode='artifact_amp',
        use_mixup=True,
        mixup_loss_strip=False,
        mixup_alpha=args.alpha, mixup_gamma=args.gamma,
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
        log_dir=log_dir,
        train_dataset=TRAIN_DS, test_dataset=VAL_DS,
        n_epochs=args.n_epochs,
    )
    config['amp_max'] = exp['amp_max']

    ckpt = train_model(config, TRAIN_DS, VAL_DS)
    if ckpt is None:
        print(f"[{exp_name}] TRAIN FAILED")
        return {'exp_name': exp_name, 'status': 'TRAIN_FAILED',
                'amp_max': exp['amp_max']}

    config_eval = build_config(
        pyramid_mode='artifact_amp',
        use_mixup=True,
        mixup_loss_strip=False,
        mixup_alpha=args.alpha, mixup_gamma=args.gamma,
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
        n_epochs=0, train_dataset=TRAIN_DS, test_dataset=TEST_DS,
        for_training=False,
    )
    summary = evaluate_model(config_eval, ckpt, TEST_DS, TRAIN_DS,
                             output_dir, exp_name)
    summary['ckpt'] = ckpt
    summary['amp_max'] = exp['amp_max']
    summary['status'] = 'OK'
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Artifact-amplified pyramid mixup ablation')
    parser.add_argument('--amps', nargs='+', type=float, default=None,
                        help='amp_max values to run (default: 1.0 2.0 3.0)')
    parser.add_argument('--output_dir', type=str,
                        default='./experiment_results/artifact_amp_sweep',
                        help='Output directory for all results')
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--num_levels', type=int, default=3)
    parser.add_argument('--sampler_real_ratio', type=float, default=0.30)
    parser.add_argument('--n_epochs', type=int, default=10)
    args = parser.parse_args()

    exp_list = EXPERIMENTS
    if args.amps:
        exp_list = [e for e in EXPERIMENTS if e['amp_max'] in args.amps]

    total = len(exp_list)
    print(f"{'='*70}\n  Artifact-Amp Ablation — {total} configs\n{'='*70}")

    results = []
    for i, exp in enumerate(exp_list):
        exp_name = exp['name']
        print(f"\n[{'='*60}]")
        print(f"  [{i+1}/{total}] {exp_name}  amp_max={exp['amp_max']}")
        print(f"[{'='*60}]")
        r = run_one(exp, args)
        results.append(r)
        results_path = os.path.join(args.output_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}\n  SUMMARY TABLE\n{'='*70}")
    hdr = (f"  {'Exp':<22s} | {'Status':<12s} | {'testall_vAUC':>12s} | "
           f"{'testall_AUC':>12s} | {'testall_ACC':>12s} | {'frame_ACC':>10s}")
    sep = (f"  {'-'*22} | {'-'*12} | {'-'*12} | {'-'*12} | "
           f"{'-'*12} | {'-'*10}")
    print(hdr)
    print(sep)
    for r in results:
        status = r.get('status', '?')
        ta = r.get('testall', {}).get('average', {})
        print(f"  {r['exp_name']:<22s} | {status:<12s} | "
              f"{str(ta.get('video_auc', 'N/A')):>12s} | "
              f"{str(ta.get('auc', 'N/A')):>12s} | "
              f"{str(ta.get('acc', 'N/A')):>12s} | "
              f"{str(r.get('test_acc_avg', 'N/A')):>10}")

    print(f"\nResults saved to: {args.output_dir}/all_results.json")


if __name__ == '__main__':
    main()

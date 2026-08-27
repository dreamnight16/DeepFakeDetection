"""
Frequency-split input ablation — NO mixup, structure + texture as separate channels.

Decomposes the normalized image into low-freq (structure) + high-freq (texture),
stacks them as extra input channels, and projects back to 3 channels via a
trainable 1x1 stem before the frozen CLIP backbone (see effort_detector.py).

Compare against G6_baseline (no mixup, plain RGB): video_auc = 0.9366.

Usage:
    python3 experiments/run_freq_split.py                 # default pool=4
    python3 experiments/run_freq_split.py --pools 2 4 8   # sweep cutoff
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
    {'name': 'freq_split_p4', 'freq_split_pool': 4},
]


def run_one(exp, args):
    """Train + eval a single freq-split config. Returns summary dict."""
    exp_name = exp['name']
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, 'logs', exp_name)

    config = build_config(
        pyramid_mode='lap_pyramid',   # irrelevant: use_mixup=False
        use_mixup=False,
        mixup_loss_strip=False,
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
        log_dir=log_dir,
        train_dataset=TRAIN_DS, test_dataset=VAL_DS,
        n_epochs=args.n_epochs,
    )
    config['use_freq_split'] = True
    config['freq_split_pool'] = exp['freq_split_pool']

    ckpt = train_model(config, TRAIN_DS, VAL_DS)
    if ckpt is None:
        print(f"[{exp_name}] TRAIN FAILED")
        return {'exp_name': exp_name, 'status': 'TRAIN_FAILED',
                'freq_split_pool': exp['freq_split_pool']}

    config_eval = build_config(
        pyramid_mode='lap_pyramid', use_mixup=False, mixup_loss_strip=False,
        lap_num_levels=args.num_levels, sampler_real_ratio=args.sampler_real_ratio,
        n_epochs=0, train_dataset=TRAIN_DS, test_dataset=TEST_DS,
        for_training=False,
    )
    config_eval['use_freq_split'] = True
    config_eval['freq_split_pool'] = exp['freq_split_pool']

    summary = evaluate_model(config_eval, ckpt, TEST_DS, TRAIN_DS,
                             output_dir, exp_name)
    summary['ckpt'] = ckpt
    summary['freq_split_pool'] = exp['freq_split_pool']
    summary['status'] = 'OK'
    return summary


def main():
    parser = argparse.ArgumentParser(description='Frequency-split input ablation')
    parser.add_argument('--pools', nargs='+', type=int, default=None,
                        help='freq_split_pool values to run (default: 4)')
    parser.add_argument('--output_dir', type=str,
                        default='./experiment_results/freq_split_sweep',
                        help='Output directory for all results')
    parser.add_argument('--num_levels', type=int, default=3)
    parser.add_argument('--sampler_real_ratio', type=float, default=0.30)
    parser.add_argument('--n_epochs', type=int, default=10)
    args = parser.parse_args()

    exp_list = EXPERIMENTS
    if args.pools:
        exp_list = [{'name': f'freq_split_p{p}', 'freq_split_pool': p}
                    for p in args.pools]

    total = len(exp_list)
    print(f"{'='*70}\n  Frequency-Split Ablation — {total} configs\n{'='*70}")

    results = []
    for i, exp in enumerate(exp_list):
        exp_name = exp['name']
        print(f"\n[{'='*60}]")
        print(f"  [{i+1}/{total}] {exp_name}  freq_split_pool={exp['freq_split_pool']}")
        print(f"[{'='*60}]")
        r = run_one(exp, args)
        results.append(r)
        results_path = os.path.join(args.output_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}\n  SUMMARY TABLE (baseline G6_baseline video_auc = 0.9366)\n{'='*70}")
    hdr = (f"  {'Exp':<18s} | {'Status':<12s} | {'testall_vAUC':>12s} | "
           f"{'testall_AUC':>12s} | {'testall_ACC':>12s} | {'frame_ACC':>10s}")
    sep = (f"  {'-'*18} | {'-'*12} | {'-'*12} | {'-'*12} | "
           f"{'-'*12} | {'-'*10}")
    print(hdr)
    print(sep)
    for r in results:
        status = r.get('status', '?')
        ta = r.get('testall', {}).get('average', {})
        print(f"  {r['exp_name']:<18s} | {status:<12s} | "
              f"{str(ta.get('video_auc', 'N/A')):>12s} | "
              f"{str(ta.get('auc', 'N/A')):>12s} | "
              f"{str(ta.get('acc', 'N/A')):>12s} | "
              f"{str(r.get('test_acc_avg', 'N/A')):>10}")

    print(f"\nResults saved to: {args.output_dir}/all_results.json")


if __name__ == '__main__':
    main()

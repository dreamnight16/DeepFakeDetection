"""
Unified experiment runner — 16-config pyramid + trajectory mixup matrix.

G1–G5 use the pyramid mixup family; G6 is the Diffusion Trajectory Mixup
(DTP-Mixup) 2×2 ablation (trajectory × pyramid).  All experiments share:
BalanceBatchSampler v1 (real_ratio=0.30), mixup_alpha=5.0,
mixup_gamma=1.0, lap_num_levels=3, n_epochs=10.

Usage:
    python3 experiments/run_experiments.py                          # all 16
    python3 experiments/run_experiments.py --groups G1 G3           # specific
    python3 experiments/run_experiments.py --groups G6              # trajectory ablation
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

# ═══════════════════════════════════════════════════════════════════════════
# Experiment definitions — 12 configs across 5 groups
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # ── G1: 2x2 label(0/1) x scope(top/bottom) ────────────────────────────
    {'group': 'G1', 'name': 'label0_top',    'mode': 'lap_pyramid_label0_top',    'strip': False},
    {'group': 'G1', 'name': 'label0_bottom', 'mode': 'lap_pyramid_label0_bottom', 'strip': False},
    {'group': 'G1', 'name': 'label1_top',    'mode': 'lap_pyramid_label1_top',    'strip': False},
    {'group': 'G1', 'name': 'label1_bottom', 'mode': 'lap_pyramid_label1_bottom', 'strip': False},

    # ── G2: Beta(2,5) on label1_top ───────────────────────────────────────
    {'group': 'G2', 'name': 'beta25',       'mode': 'lap_pyramid_label1_top', 'strip': False,
     'alpha': 2, 'beta_b': 5, 'beta_flip': False},
    {'group': 'G2', 'name': 'beta25_flip',  'mode': 'lap_pyramid_label1_top', 'strip': False,
     'alpha': 2, 'beta_b': 5, 'beta_flip': True},

    # ── G3: Full scope label 0/1 ───────────────────────────────────────────
    {'group': 'G3', 'name': 'label0_full',  'mode': 'lap_pyramid_label0_full', 'strip': False},
    {'group': 'G3', 'name': 'label1_full',  'mode': 'lap_pyramid_label1_full', 'strip': False},

    # ── G4: Pyramid loss ablation ──────────────────────────────────────────
    {'group': 'G4', 'name': 'exp1_soft_ce',  'mode': 'lap_pyramid', 'strip': False},
    {'group': 'G4', 'name': 'exp2_strip_rf', 'mode': 'lap_pyramid', 'strip': True},

    # ── G5: RR+FF pyramid (on strip-RF basis) ──────────────────────────────
    {'group': 'G5', 'name': 'g1_rf_stripped',      'mode': 'lap_pyramid_all',  'strip': True},
    {'group': 'G5', 'name': 'g2_rf_not_generated', 'mode': 'lap_pyramid_rrff', 'strip': False},

    # ── G6: Diffusion Trajectory Mixup (2×2 trajectory × pyramid) ──────────
    # real_ratio=0.30 (same as G1–G5) → pyramid_only is the direct对照 anchor
    # to G4_exp1_soft_ce (identical mode/hyperparams), so trajectory_pyramid
    # is directly comparable to the whole G1–G5 matrix.
    # Re-run under a fresh namespace — do NOT mix with the old buggy sweep:
    #   python3 experiments/run_experiments.py --groups G6 \
    #       --output_dir ./experiment_results/trajectory_mixup_sweep_v2_correct
    {'group': 'G6', 'name': 'baseline',           'mode': 'original',           'use_mixup': False, 'strip': False},
    {'group': 'G6', 'name': 'pyramid_only',       'mode': 'lap_pyramid',        'use_mixup': True,  'strip': False},
    {'group': 'G6', 'name': 'trajectory_only',    'mode': 'trajectory',         'use_mixup': True,  'strip': False,
     'traj_t_min': 50, 'traj_t_max': 700, 'traj_T': 1000},
    {'group': 'G6', 'name': 'trajectory_pyramid', 'mode': 'trajectory_pyramid', 'use_mixup': True,  'strip': False,
     'traj_t_min': 50, 'traj_t_max': 700, 'traj_T': 1000},
]


def run_one(exp, args):
    """Train + eval a single experiment config. Returns summary dict."""
    exp_name = f"{exp['group']}_{exp['name']}"
    output_dir = os.path.join(args.output_dir, exp['group'], exp['name'])
    os.makedirs(output_dir, exist_ok=True)

    exp_alpha = exp.get('alpha', args.alpha)
    exp_use_mixup = exp.get('use_mixup', True)
    traj_kwargs = {k: exp[k] for k in ('traj_t_min', 'traj_t_max', 'traj_T') if k in exp}

    log_dir = os.path.join(args.output_dir, 'logs', exp['group'], exp['name'])
    config = build_config(
        pyramid_mode=exp['mode'],
        use_mixup=exp_use_mixup,
        mixup_loss_strip=exp['strip'],
        mixup_alpha=exp_alpha, mixup_gamma=args.gamma,
        mixup_beta_b=exp.get('beta_b'), mixup_beta_flip=exp.get('beta_flip', False),
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
        **traj_kwargs,
        log_dir=log_dir,
        train_dataset=TRAIN_DS, test_dataset=VAL_DS,
        n_epochs=args.n_epochs,
    )

    # Train
    ckpt = train_model(config, TRAIN_DS, VAL_DS)
    if ckpt is None:
        print(f"[{exp_name}] TRAIN FAILED")
        return {'exp_name': exp_name, 'status': 'TRAIN_FAILED'}

    # Eval
    config_eval = build_config(
        pyramid_mode=exp['mode'],
        use_mixup=exp_use_mixup,
        mixup_loss_strip=exp['strip'],
        mixup_alpha=exp_alpha,
        mixup_beta_b=exp.get('beta_b'), mixup_beta_flip=exp.get('beta_flip', False),
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
        **traj_kwargs,
        n_epochs=0, train_dataset=TRAIN_DS, test_dataset=TEST_DS,
        for_training=False,
    )
    summary = evaluate_model(config_eval, ckpt, TEST_DS, TRAIN_DS, output_dir, exp_name)
    summary['ckpt'] = ckpt
    summary['status'] = 'OK'
    return summary


def main():
    parser = argparse.ArgumentParser(description='Unified pyramid mixup experiment runner')
    parser.add_argument('--groups', nargs='+', default=None,
                        help='Which groups to run (default: all)')
    parser.add_argument('--output_dir', type=str,
                        default='./experiment_results/master_sweep',
                        help='Output directory for all results')
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--num_levels', type=int, default=3)
    parser.add_argument('--sampler_real_ratio', type=float, default=0.30)
    parser.add_argument('--n_epochs', type=int, default=10)
    args = parser.parse_args()

    exp_list = EXPERIMENTS
    if args.groups:
        exp_list = [e for e in EXPERIMENTS if e['group'] in args.groups]

    total = len(exp_list)
    print(f"{'='*70}\n  Unified Experiment Runner — {total} configs\n{'='*70}")

    results = []
    for i, exp in enumerate(exp_list):
        exp_name = f"{exp['group']}_{exp['name']}"
        print(f"\n[{'='*60}]")
        print(f"  [{i+1}/{total}] {exp_name}")
        print(f"  mode={exp['mode']}  strip={exp['strip']}"
              + (f"  beta=({args.alpha},{exp['beta_b']}) flip={exp['beta_flip']}"
                 if 'beta_b' in exp else ""))
        print(f"[{'='*60}]")
        r = run_one(exp, args)
        results.append(r)
        # Save incremental results
        results_path = os.path.join(args.output_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*70}\n  SUMMARY TABLE\n{'='*70}")
    hdr = f"  {'Exp':<28s} | {'Status':<12s} | {'testall_vAUC':>12s} | {'testall_AUC':>12s} | {'testall_ACC':>12s} | {'frame_ACC':>10s}"
    sep = f"  {'-'*28} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*10}"
    print(hdr)
    print(sep)
    for r in results:
        status = r.get('status', '?')
        ta = r.get('testall', {}).get('average', {})
        print(f"  {r['exp_name']:<28s} | {status:<12s} | "
              f"{str(ta.get('video_auc','N/A')):>12s} | {str(ta.get('auc','N/A')):>12s} | "
              f"{str(ta.get('acc','N/A')):>12s} | {str(r.get('test_acc_avg','N/A')):>10}")

    print(f"\nResults saved to: {args.output_dir}/all_results.json")


if __name__ == '__main__':
    main()

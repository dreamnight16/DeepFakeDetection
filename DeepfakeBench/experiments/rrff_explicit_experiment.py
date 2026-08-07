"""
RR+FF Explicit Pairing Experiment — real_ratio Sweep
=====================================================

Three groups comparing real:fake ratio in RR+FF explicit pairing:

  Group 1 (ratio_0.30): 30% real, 70% fake
  Group 2 (ratio_0.50): 50% real, 50% fake
  Group 3 (ratio_0.70): 70% real, 30% fake

All groups use rrff_explicit mixup:
  - RR: circular-shift pair reals, Laplacian pyramid mixup (full scope), label=0
  - FF: circular-shift pair fakes, Laplacian pyramid mixup (full scope), label=1
  - RF: not produced
  - Sampler: v1 (BalanceBatchSampler, balance_sampler_v2=false)

Outputs per group:
  - testall.py evaluation (video_auc, auc, acc per dataset)
  - Frame-level accuracy + confusion matrix (per dataset + average)
  - Score distributions: KDE plots for Real vs Fake, test set + train set

Both training + eval and eval-only modes supported.

Usage:
    # Full experiment (train + eval):
    python3 experiments/rrff_explicit_experiment.py \
        --train_dataset FaceForensics++ \
        --val_dataset Celeb-DF-v2 \
        --output_dir ./experiment_results/rrff_explicit

    # Eval-only (with existing checkpoints):
    python3 experiments/rrff_explicit_experiment.py \
        --ckpt_r030 /path/to/ratio_0.30_ckpt.pth \
        --ckpt_r050 /path/to/ratio_0.50_ckpt.pth \
        --ckpt_r070 /path/to/ratio_0.70_ckpt.pth \
        --eval_only

    # Single group:
    python3 experiments/rrff_explicit_experiment.py --skip_r050 --skip_r070 ...

Author: experiment
"""
import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ── Path setup (same as train.py) ───────────────────────────────────────────
_current_dir = os.path.dirname(os.path.abspath(__file__))
_deepfake_dir = os.path.dirname(_current_dir)
_training_dir = os.path.join(_deepfake_dir, 'training')
sys.path.insert(0, _training_dir)
sys.path.insert(0, _deepfake_dir)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, accuracy_score

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════════
#  Config helpers
# ═══════════════════════════════════════════════════════════════════════════════

DETECTOR_YAML = os.path.join(_training_dir, 'config', 'detector', 'effort.yaml')
TRAIN_YAML  = os.path.join(_training_dir, 'config', 'train_config.yaml')
TEST_YAML   = os.path.join(_training_dir, 'config', 'test_config.yaml')


def build_config(sampler_real_ratio=0.50,
                 mixup_alpha=5.0,
                 mixup_gamma=1.0,
                 lap_num_levels=3,
                 log_dir=None,
                 train_dataset=None,
                 test_dataset=None,
                 n_epochs=10,
                 for_training=True):
    """Build merged config dict with experiment parameters.

    for_training=True  → uses train_config.yaml  (training paths + labels)
    for_training=False → uses test_config.yaml   (eval paths + comprehensive labels)
    """
    with open(DETECTOR_YAML, 'r') as f:
        config = yaml.safe_load(f)
    base_yaml = TRAIN_YAML if for_training else TEST_YAML
    with open(base_yaml, 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)

    # ── Auto-detect dataset_json_folder (portability) ────────────────────
    _json_folder = os.path.join(_deepfake_dir, 'preprocessing', 'dataset_json')
    if os.path.isdir(_json_folder):
        config['dataset_json_folder'] = _json_folder

    # ── Mixup: rrff_explicit ─────────────────────────────────────────────
    config['use_mixup'] = True
    config['mixup_mode'] = 'rrff_explicit'
    config['mixup_alpha'] = mixup_alpha
    config['mixup_gamma'] = mixup_gamma
    config['mix_domain'] = 'rgb'
    config['lap_num_levels'] = lap_num_levels
    config['mixup_loss_strip'] = False

    # ── Sampler: v1 (BalanceBatchSampler) with real_ratio ────────────────
    config['balance_sampler_v2'] = False
    config['use_balance_batch_sampler'] = True
    config['sampler_real_ratio'] = sampler_real_ratio

    # ── Paths ────────────────────────────────────────────────────────────
    if log_dir is not None:
        config['log_dir'] = log_dir
    if train_dataset is not None:
        config['train_dataset'] = train_dataset if isinstance(train_dataset, list) else [train_dataset]
    if test_dataset is not None:
        config['test_dataset'] = test_dataset if isinstance(test_dataset, list) else [test_dataset]

    config['nEpochs'] = n_epochs
    config['dry_run'] = False
    config['ddp'] = False
    config['local_rank'] = 0
    config['save_ckpt'] = True
    config['save_feat'] = True
    config['save_avg'] = True
    config['workers'] = 4
    config['train_batchSize'] = 32
    config['test_batchSize'] = 32
    config['lmdb'] = False
    config['start_epoch'] = 0

    return config


def save_temp_yaml(config):
    """Write config to a temporary YAML file, return path."""
    fd, path = tempfile.mkstemp(suffix='.yaml', prefix='effort_rrff_explicit_')
    os.close(fd)
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Training  (reuses train.py as subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(config, train_dataset, val_dataset):
    """Train EffortDetector; return path to best checkpoint."""
    yaml_path = save_temp_yaml(config)

    train_py = os.path.join(_training_dir, 'train.py')
    cmd = [
        sys.executable, train_py,
        '--detector_path', yaml_path,
        '--train_dataset', train_dataset,
        '--test_dataset', val_dataset,
    ]
    print(f"[train] {' '.join(cmd)}")
    sys.stdout.flush()

    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        print(f"[train] WARNING: train.py exited with code {proc.returncode}")
        os.unlink(yaml_path)
        return None

    log_dir = config['log_dir']
    ckpt_candidates = list(Path(log_dir).glob('effort_*/test/avg/ckpt_best.pth'))
    if not ckpt_candidates:
        print(f"[train] WARNING: no checkpoint found under {log_dir}")
        os.unlink(yaml_path)
        return None

    ckpt = str(sorted(ckpt_candidates, key=os.path.getmtime)[-1])
    print(f"[train] best ckpt: {ckpt}")
    os.unlink(yaml_path)
    return ckpt


# ═══════════════════════════════════════════════════════════════════════════════
#  testall.py evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def run_testall(ckpt_path, test_datasets, log_path):
    """Run testall.py on checkpoint; return per-dataset metrics dict."""
    testall_py = os.path.join(_deepfake_dir, 'testall.py')

    TMP_YAML = tempfile.mkstemp(suffix='.yaml', prefix='effort_testall_')[1]
    with open(DETECTOR_YAML, 'r') as f:
        yaml_config = yaml.safe_load(f)
    with open(TEST_YAML, 'r') as f:
        test_config = yaml.safe_load(f)
    yaml_config.update(test_config)
    _json_folder = os.path.join(_deepfake_dir, 'preprocessing', 'dataset_json')
    if os.path.isdir(_json_folder):
        yaml_config['dataset_json_folder'] = _json_folder
    with open(TMP_YAML, 'w') as f:
        yaml.dump(yaml_config, f)

    cmd = [
        sys.executable, testall_py,
        '--detector_path', TMP_YAML,
        '--weights_path', ckpt_path,
        '--test_datasets',
    ] + test_datasets

    print(f"  [testall] {' '.join(cmd)}")
    sys.stdout.flush()

    with open(log_path, 'w') as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    os.unlink(TMP_YAML)

    # Parse per-dataset results
    metrics = {}
    with open(log_path, 'r') as log_f:
        current_ds = None
        for line in log_f:
            line = line.strip()
            if line.startswith('dataset:'):
                current_ds = line.split('dataset:')[1].strip()
                metrics[current_ds] = {}
            elif current_ds and line.startswith('acc:'):
                metrics[current_ds]['acc'] = float(line.split(':')[1].strip())
            elif current_ds and line.startswith('auc:'):
                metrics[current_ds]['auc'] = float(line.split(':')[1].strip())
            elif current_ds and line.startswith('video_auc:'):
                metrics[current_ds]['video_auc'] = float(line.split(':')[1].strip())

    # Parse average block (last occurrence)
    with open(log_path, 'r') as log_f:
        for line in log_f:
            line = line.strip()
            if line.startswith('acc:'):
                metrics['average'] = metrics.get('average', {})
                metrics['average']['acc'] = float(line.split(':')[1].strip())
            elif line.startswith('auc:'):
                metrics['average'] = metrics.get('average', {})
                metrics['average']['auc'] = float(line.split(':')[1].strip())
            elif line.startswith('video_auc:'):
                metrics['average'] = metrics.get('average', {})
                metrics['average']['video_auc'] = float(line.split(':')[1].strip())

    if proc.returncode != 0:
        print(f"  [testall] WARNING: exited with code {proc.returncode}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Frame-level evaluation (confusion matrix + KDE)
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(config, ckpt_path):
    """Load EffortDetector from config + checkpoint."""
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    new_weights = {k.replace('module.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(new_weights, strict=True)
    model.eval()
    return model


def get_data_loader(config, dataset_name, mode='test'):
    """Build a DataLoader for a single dataset."""
    cfg = config.copy()
    cfg['test_dataset'] = dataset_name
    ds = DeepfakeAbstractBaseDataset(config=cfg, mode=mode)
    loader = DataLoader(
        ds,
        batch_size=cfg['test_batchSize'],
        shuffle=False,
        num_workers=int(cfg.get('workers', 4)),
        collate_fn=ds.collate_fn,
    )
    return loader


def get_train_loader(config):
    """Build a DataLoader for training data — NO augmentations."""
    cfg = config.copy()
    cfg['use_data_augmentation'] = False
    ds = DeepfakeAbstractBaseDataset(config=cfg, mode='train')
    loader = DataLoader(
        ds,
        batch_size=cfg['test_batchSize'],
        shuffle=False,
        num_workers=int(cfg.get('workers', 4)),
        collate_fn=ds.collate_fn,
    )
    return loader


@torch.no_grad()
def collect_predictions(model, data_loader):
    """Run model on data_loader; return (probs, labels) as numpy arrays."""
    probs_list = []
    labels_list = []
    for data_dict in tqdm(data_loader, desc='eval', leave=False):
        data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)
        for k in list(data_dict.keys()):
            if data_dict[k] is not None and k != 'name':
                data_dict[k] = data_dict[k].to(DEVICE)
        pred = model(data_dict, inference=True)
        probs_list.append(pred['prob'].cpu().numpy())
        labels_list.append(data_dict['label'].cpu().numpy())
    return np.concatenate(probs_list), np.concatenate(labels_list)


def compute_metrics(probs, labels):
    """Compute accuracy, confusion matrix from predictions."""
    preds = (probs > 0.5).astype(int)
    acc = float(accuracy_score(labels, preds))
    cm = sk_confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'acc': acc,
        'tn': int(tn), 'fp': int(fp),
        'fn': int(fn), 'tp': int(tp),
    }


def print_results(exp_name, train_probs, train_labels,
                  test_probs_dict, test_labels_dict, testall_metrics, output_dir):
    """Print testall metrics, confusion matrix, and save KDE plots."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {exp_name}")
    print(f"{'='*70}")

    # ── testall.py metrics ──────────────────────────────────────────────
    if testall_metrics:
        print(f"\n  ── testall.py (avg per dataset) ──")
        avg_m = testall_metrics.get('average', {})
        print(f"  video_auc: {avg_m.get('video_auc', 'N/A')}")
        print(f"  auc:       {avg_m.get('auc', 'N/A')}")
        print(f"  acc:       {avg_m.get('acc', 'N/A')}")

    # ── Frame-level train set ───────────────────────────────────────────
    if len(train_probs) > 0:
        train_metrics = compute_metrics(train_probs, train_labels)
        print(f"\n  ── Train Set (frame-level) ──")
        print(f"  Accuracy:       {train_metrics['acc']:.4f}")
        print(f"  Confusion Matrix (actual \\ predicted):")
        print(f"           Pred=0   Pred=1")
        print(f"  Real  |  {train_metrics['tn']:6d}   {train_metrics['fp']:6d}")
        print(f"  Fake  |  {train_metrics['fn']:6d}   {train_metrics['tp']:6d}")

    # ── Frame-level test sets ───────────────────────────────────────────
    print(f"\n  ── Test Sets (frame-level) ──")
    print(f"  {'Dataset':<25s} | {'Acc':>8s} | {'TN':>6s} | {'FP':>6s} | {'FN':>6s} | {'TP':>6s}")
    print(f"  {'-'*25} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")

    all_acc = []
    total_tn = total_fp = total_fn = total_tp = 0
    for ds in sorted(test_probs_dict.keys()):
        m = compute_metrics(test_probs_dict[ds], test_labels_dict[ds])
        all_acc.append(m['acc'])
        total_tn += m['tn']; total_fp += m['fp']
        total_fn += m['fn']; total_tp += m['tp']
        print(f"  {ds:<25s} | {m['acc']:8.4f} | {m['tn']:6d} | {m['fp']:6d} | {m['fn']:6d} | {m['tp']:6d}")

    if len(all_acc) > 1:
        avg_acc = np.mean(all_acc)
        print(f"  {'-'*25} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")
        print(f"  {'average':<25s} | {avg_acc:8.4f} | {total_tn:6d} | {total_fp:6d} | {total_fn:6d} | {total_tp:6d}")

    _plot_score_distributions(
        exp_name, output_dir,
        train_probs, train_labels,
        test_probs_dict, test_labels_dict,
    )


def _plot_score_distributions(exp_name, output_dir,
                               train_probs, train_labels,
                               test_probs_dict, test_labels_dict):
    """Generate KDE plots for train + test score distributions."""
    safe_name = exp_name.replace(' ', '_').replace('/', '_')
    n_test = len(test_probs_dict)
    has_train = len(train_probs) > 0
    n_cols = n_test + (1 if has_train else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    xs = np.linspace(0, 1, 500)
    col_idx = 0

    if has_train:
        ax = axes[col_idx]
        _plot_one_dist(ax, train_probs, train_labels, 'Train Set', 'tab:blue', xs)
        col_idx += 1

    colors = plt.cm.tab10.colors
    for ds in sorted(test_probs_dict.keys()):
        ax = axes[col_idx]
        c = colors[col_idx % len(colors)]
        _plot_one_dist(ax, test_probs_dict[ds], test_labels_dict[ds], ds, c, xs)
        col_idx += 1

    fig.suptitle(f"Score Distributions — {exp_name}\n(Real vs Fake, per Dataset)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"score_dist_{safe_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Score distribution plot saved to: {out_path}")


def _plot_one_dist(ax, probs, labels, title, color, xs):
    """Plot KDE curves for real vs fake on a single axis."""
    for lbl, lname, ls in [(0, 'Real', '--'), (1, 'Fake', '-')]:
        subset = probs[labels == lbl]
        if len(subset) >= 2:
            kde = gaussian_kde(subset, bw_method=0.08)
            ys = kde(xs)
            ax.plot(xs, ys, color=color, linewidth=2, linestyle=ls, label=lname)
            ax.fill_between(xs, ys, alpha=0.12, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Fake Probability', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)


# ═══════════════════════════════════════════════════════════════════════════════
#  Single group runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_group(group_name, real_ratio, output_dir, ckpt_path, args):
    """Train + testall + frame-level eval for a single experiment group.

    Returns True on success, False on failure.
    """
    exp_dir = os.path.join(output_dir, group_name)
    os.makedirs(exp_dir, exist_ok=True)

    if ckpt_path:
        ckpt = ckpt_path
    else:
        log_dir = os.path.join(output_dir, f'logs_{group_name}')
        config_train = build_config(
            sampler_real_ratio=real_ratio,
            mixup_alpha=args.alpha,
            mixup_gamma=args.gamma,
            lap_num_levels=args.num_levels,
            log_dir=log_dir,
            train_dataset=args.train_dataset,
            test_dataset=args.val_dataset,
            n_epochs=args.n_epochs,
        )
        print(f"\n{'='*60}")
        print(f"  Training: {group_name}")
        print(f"  Mode: rrff_explicit  real_ratio={real_ratio}")
        print(f"{'='*60}")
        ckpt = train_model(config_train, args.train_dataset, args.val_dataset)
        if ckpt is None:
            print(f"[{group_name}] ERROR: Training failed.")
            return False

    # ── testall.py eval ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  testall.py: {group_name}")
    print(f"  Checkpoint: {ckpt}")
    print(f"{'='*60}")

    testall_log = os.path.join(exp_dir, 'testall.log')
    testall_metrics = run_testall(ckpt, args.test_datasets, testall_log)

    # Print per-dataset testall results
    if testall_metrics:
        print(f"\n  ── testall per-dataset ──")
        print(f"  {'Dataset':<25s} | {'video_auc':>9s} | {'auc':>9s} | {'acc':>9s}")
        print(f"  {'-'*25} | {'-'*9} | {'-'*9} | {'-'*9}")
        for ds in sorted(testall_metrics.keys()):
            m = testall_metrics[ds]
            print(f"  {ds:<25s} | {m.get('video_auc', 'N/A'):>9} | {m.get('auc', 'N/A'):>9} | {m.get('acc', 'N/A'):>9}")

    # ── Frame-level eval ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Frame-level eval: {group_name}")
    print(f"{'='*60}")

    config_eval = build_config(
        sampler_real_ratio=real_ratio,
        mixup_alpha=args.alpha,
        n_epochs=0,
        train_dataset=args.train_dataset,
        test_dataset=args.test_datasets,
        for_training=False,
    )
    model = load_model(config_eval, ckpt)

    test_probs = {}
    test_labels = {}
    for ds in args.test_datasets:
        print(f"  Testing on: {ds}")
        loader = get_data_loader(config_eval, ds, mode='test')
        probs, labels = collect_predictions(model, loader)
        test_probs[ds] = probs
        test_labels[ds] = labels

    print(f"  Collecting train set scores...")
    train_loader = get_train_loader(config_eval)
    train_probs, train_labels = collect_predictions(model, train_loader)

    print_results(group_name,
                  train_probs, train_labels,
                  test_probs, test_labels,
                  testall_metrics, exp_dir)

    del model
    torch.cuda.empty_cache()
    return True


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='RR+FF Explicit Mixup — real_ratio sweep (3 groups)')
    parser.add_argument('--train_dataset', type=str, default='FaceForensics++')
    parser.add_argument('--val_dataset', type=str, default='Celeb-DF-v2')
    parser.add_argument('--test_datasets', nargs='+',
                        default=['WDF', 'FFIW', 'Celeb-DF-v2', 'DeepFakeDetection',
                                 'DFDC', 'DFDCP', 'DeeperForensics-1.0'])
    parser.add_argument('--output_dir', type=str,
                        default='./experiment_results/rrff_explicit')
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--num_levels', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=10)

    # Eval-only mode
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--ckpt_r030', type=str, default=None,
                        help='Checkpoint for ratio_0.30')
    parser.add_argument('--ckpt_r050', type=str, default=None,
                        help='Checkpoint for ratio_0.50')
    parser.add_argument('--ckpt_r070', type=str, default=None,
                        help='Checkpoint for ratio_0.70')

    # Skip individual groups
    parser.add_argument('--skip_r030', action='store_true',
                        help='Skip ratio_0.30')
    parser.add_argument('--skip_r050', action='store_true',
                        help='Skip ratio_0.50')
    parser.add_argument('--skip_r070', action='store_true',
                        help='Skip ratio_0.70')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════
    #  Group definitions: 3 real_ratio variants
    # ═══════════════════════════════════════════════════════════════════════

    groups = [
        {
            'name': 'ratio_0.30',
            'real_ratio': 0.30,
            'ckpt_arg': args.ckpt_r030,
            'skip': args.skip_r030,
            'desc': '30% real, 70% fake — RR sparse, FF dominant',
        },
        {
            'name': 'ratio_0.50',
            'real_ratio': 0.50,
            'ckpt_arg': args.ckpt_r050,
            'skip': args.skip_r050,
            'desc': '50% real, 50% fake — RR = FF balanced',
        },
        {
            'name': 'ratio_0.70',
            'real_ratio': 0.70,
            'ckpt_arg': args.ckpt_r070,
            'skip': args.skip_r070,
            'desc': '70% real, 30% fake — RR dominant, FF sparse',
        },
    ]

    print("=" * 70)
    print("  RR+FF Explicit Mixup — real_ratio Sweep")
    print(f"  Mixup mode:  rrff_explicit (within-class pyramid, no RF)")
    print(f"  Train dataset: {args.train_dataset}")
    print(f"  Val dataset:   {args.val_dataset}")
    print(f"  Test datasets: {', '.join(args.test_datasets)}")
    print(f"  α={args.alpha}  γ={args.gamma}  levels={args.num_levels}")
    print(f"  Sampler: v1 balance (balance_sampler_v2=false)")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 70)
    print()
    for g in groups:
        status = "SKIP" if g['skip'] else "RUN"
        print(f"  [{status}] {g['name']}: real_ratio={g['real_ratio']}  —  {g['desc']}")
    print()

    for g in groups:
        if g['skip']:
            print(f"\n[SKIP] {g['name']}")
            continue

        ckpt_path = g['ckpt_arg'] if args.eval_only else None
        ok = run_group(
            group_name=g['name'],
            real_ratio=g['real_ratio'],
            output_dir=args.output_dir,
            ckpt_path=ckpt_path,
            args=args,
        )
        if not ok:
            print(f"[{g['name']}] FAILED, continuing...")

    print("\n" + "=" * 70)
    print("  All groups complete.")
    print(f"  Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()

"""
Shared experiment utilities for pyramid mixup experiments.
Reuses train.py + testall.py as subprocesses, plus direct model loading for
frame-level confusion matrix and KDE score distributions.
"""
import os
import sys
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

_current_dir = os.path.dirname(os.path.abspath(__file__))
_deepfake_dir = os.path.dirname(_current_dir)
_training_dir = os.path.join(_deepfake_dir, 'training')
sys.path.insert(0, _training_dir)
sys.path.insert(0, _deepfake_dir)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as sk_cm, accuracy_score
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETECTOR_YAML = os.path.join(_training_dir, 'config', 'detector', 'effort.yaml')
TRAIN_YAML  = os.path.join(_training_dir, 'config', 'train_config.yaml')
TEST_YAML   = os.path.join(_training_dir, 'config', 'test_config.yaml')


# ── Config helpers ─────────────────────────────────────────────────────────

def build_config(pyramid_mode='lap_pyramid',
                 use_mixup=True,
                 mixup_loss_strip=False,
                 mixup_alpha=5.0, mixup_gamma=1.0,
                 mixup_beta_b=None, mixup_beta_flip=False,
                 lap_num_levels=3,
                 sampler_real_ratio=0.30,
                 traj_t_min=None, traj_t_max=None, traj_T=None,
                 rank_margin=None, rank_loss_weight=None,
                 rank_softplus=None, rank_alpha=None,
                 model_name='effort',
                 aepa_mode=None, aepa_lambda_f=1.0, aepa_eps=1e-8,
                 aepa_init_ckpt=None,
                 max_evidence_lambda=1.0, max_evidence_eps=1e-8,
                 max_evidence_init_ckpt=None, max_evidence_cls_feature='raw_token',
                 max_evidence_inference='cls',
                 freq_ablation=None, freq_norm='minmax', freq_energy_match=False,
                 freq_after_aug=True,
                 residual_ablation=None, residual_sigma=2.0, residual_alpha=4.0,
                 residual_fft_r0=0.65, residual_shuffle=False,
                 comp_freq_bands=None, comp_freq_norm='train_rms', comp_freq_rms=0.5,
                 comp_band_sigma=None,
                 comp_fuse='gate', comp_gate_hidden=32,
                 comp_lambda_freq=1.0, comp_lambda_max=1.0, comp_cls_feature='pooler_output',
                 # G18 (LFEQ read-out head): structural + loss scalars, threaded
                 # through build_config and into arch_keys so testall rebuilds
                 # the model with identical head/weights/fusion (strict load).
                 lfeq_hidden_dim=256, lfeq_num_evidence_tokens=8,
                 lfeq_depth=2, lfeq_num_heads=8, lfeq_dropout=0.1,
                 lfeq_fusion_weight=0.5, lfeq_evidence_weight=1.0,
                 lfeq_diversity_weight=0.01,
                 log_dir=None,
                 train_dataset=None, test_dataset=None,
                 n_epochs=10, for_training=True):
    """Build merged config dict with experiment parameters."""
    with open(DETECTOR_YAML, 'r') as f:
        config = yaml.safe_load(f)
    base = TRAIN_YAML if for_training else TEST_YAML
    with open(base, 'r') as f:
        config.update(yaml.safe_load(f))
    json_folder = os.path.join(_deepfake_dir, 'preprocessing', 'dataset_json')
    if os.path.isdir(json_folder):
        config['dataset_json_folder'] = json_folder

    config['model_name'] = model_name
    config['use_mixup'] = use_mixup
    config['mixup_mode'] = pyramid_mode

    # G13 (data-side frequency-band isolation): these keys are read by
    # DeepfakeAbstractBaseDataset.__getitem__ and do NOT touch the model
    # architecture.  They are set unconditionally so the dataset never sees a
    # missing key.  freq_ablation=None (RGB baseline) skips filtering entirely.
    config['freq_ablation'] = freq_ablation
    config['freq_norm'] = freq_norm
    config['freq_energy_match'] = freq_energy_match
    config['freq_after_aug'] = freq_after_aug

    # G17-2 (data-side real-noise evidence isolation): read by __getitem__ and
    # do NOT touch the model architecture (the ``effort`` observer is unchanged —
    # only the input differs).  Set unconditionally so the dataset never sees a
    # missing key; residual_ablation=None (RGB baseline) leaves the input as-is.
    config['residual_ablation'] = residual_ablation
    config['residual_sigma'] = residual_sigma
    config['residual_alpha'] = residual_alpha
    config['residual_fft_r0'] = residual_fft_r0
    config['residual_shuffle'] = residual_shuffle

    # AEPA (G12) has no mixup: its patch-level asymmetric loss replaces the
    # CLS readout entirely.  Force mixup off so the trainer skips augmentation.
    if model_name == 'effort_aepa':
        config['use_mixup'] = False
        config['mixup_mode'] = 'none'
        if aepa_mode is not None:
            config['aepa_mode'] = aepa_mode
        config['aepa_lambda_f'] = aepa_lambda_f
        config['aepa_eps'] = aepa_eps
        if aepa_init_ckpt is not None:
            config['aepa_init_ckpt'] = aepa_init_ckpt

    # Max-fake-evidence selection loss (G15) also has no mixup: it keeps the
    # CLS branch and adds a patch-level local loss (L_cls + lambda_max * L_max).
    if model_name == 'effort_maxev':
        config['use_mixup'] = False
        config['mixup_mode'] = 'none'
        config['max_evidence_lambda'] = max_evidence_lambda
        config['max_evidence_eps'] = max_evidence_eps
        config['max_evidence_cls_feature'] = max_evidence_cls_feature
        # Detection-time score rule (G15 supplement Strategy-3): 'cls' (CLS head
        # alone, default §16) or 'avg' (0.5*P_cls + 0.5*max_i q_{i,1}, gated on
        # inference=True).  Forward-time behaviour only (no weight-shape change),
        # but carried through arch_keys so testall rebuilds the model with the
        # SAME rule that training/eval intended.
        config['max_evidence_inference'] = max_evidence_inference
        if max_evidence_init_ckpt is not None:
            config['max_evidence_init_ckpt'] = max_evidence_init_ckpt

    # G17-1 (model-side dual-line gated fusion) also has no mixup: the two lines
    # already share a frozen backbone and adding global mixup would smear the
    # per-line evidence heads.  Force it off so the trainer skips augmentation.
    if model_name == 'effort_dualcomp':
        config['use_mixup'] = False
        config['mixup_mode'] = 'none'
        config['comp_freq_bands'] = comp_freq_bands if comp_freq_bands else ['low']
        config['comp_freq_norm'] = comp_freq_norm
        config['comp_freq_rms'] = comp_freq_rms
        config['comp_fuse'] = comp_fuse
        config['comp_gate_hidden'] = comp_gate_hidden
        config['comp_lambda_freq'] = comp_lambda_freq
        config['comp_lambda_max'] = comp_lambda_max
        config['comp_cls_feature'] = comp_cls_feature
        # Fixed per-band RMS scalars over the training set (G17-1 §2 train-set
        # stats).  Non-weight, but MUST be carried so train/val/eval/testall all
        # rebuild the F line with an identical normalisation (a silent per-image
        # fallback would diverge the eval-time F view from training).
        config['comp_band_sigma'] = comp_band_sigma

    # G18 (LFEQ read-out head) also has no mixup: the LFEQ query-transformer
    # reads raw patch tokens and a global pixel mixup would smear the per-patch
    # attention the evidence queries are specialised on.  Force it off.  We also
    # force margin_loss_mode='off' — the LFEQ feat is the hidden-dim decision
    # feature (256), NOT the 1024-dim pooler feature the asymmetric center loss
    # is built around; leaving margin_loss on would crash on a shape mismatch.
    if model_name == 'effort_lfeq':
        config['use_mixup'] = False
        config['mixup_mode'] = 'none'
        config['margin_loss_mode'] = 'off'
        config['lfeq_hidden_dim'] = lfeq_hidden_dim
        config['lfeq_num_evidence_tokens'] = lfeq_num_evidence_tokens
        config['lfeq_depth'] = lfeq_depth
        config['lfeq_num_heads'] = lfeq_num_heads
        config['lfeq_dropout'] = lfeq_dropout
        config['lfeq_fusion_weight'] = lfeq_fusion_weight
        config['lfeq_evidence_weight'] = lfeq_evidence_weight
        config['lfeq_diversity_weight'] = lfeq_diversity_weight
    config['mixup_alpha'] = mixup_alpha
    config['mixup_gamma'] = mixup_gamma
    config['mix_domain'] = 'rgb'
    config['lap_num_levels'] = lap_num_levels
    config['mixup_loss_strip'] = mixup_loss_strip
    if mixup_beta_b is not None:
        config['mixup_beta_b'] = mixup_beta_b
    config['mixup_beta_flip'] = mixup_beta_flip
    if traj_t_min is not None:
        config['traj_t_min'] = traj_t_min
    if traj_t_max is not None:
        config['traj_t_max'] = traj_t_max
    if traj_T is not None:
        config['traj_T'] = traj_T
    if rank_margin is not None:
        config['rank_margin'] = rank_margin
    if rank_loss_weight is not None:
        config['rank_loss_weight'] = rank_loss_weight
    if rank_softplus is not None:
        config['rank_softplus'] = rank_softplus
    if rank_alpha is not None:
        config['rank_alpha'] = rank_alpha

    config['balance_sampler_v2'] = False
    config['use_balance_batch_sampler'] = True
    config['sampler_real_ratio'] = sampler_real_ratio

    if log_dir is not None:
        config['log_dir'] = log_dir
    if train_dataset is not None:
        config['train_dataset'] = [train_dataset] if isinstance(train_dataset, str) else train_dataset
    if test_dataset is not None:
        config['test_dataset'] = [test_dataset] if isinstance(test_dataset, str) else test_dataset

    config['nEpochs'] = n_epochs
    config['ddp'] = False
    config['local_rank'] = 0
    config['save_ckpt'] = True
    config['save_feat'] = True
    config['save_avg'] = True
    return config


def save_temp_yaml(config, prefix='effort_exp_'):
    fd, path = tempfile.mkstemp(suffix='.yaml', prefix=prefix)
    os.close(fd)
    with open(path, 'w') as f:
        yaml.dump(config, f)
    return path


# ── Training (subprocess) ──────────────────────────────────────────────────

def train_model(config, train_dataset, val_dataset):
    """Train EffortDetector via train.py subprocess; return best checkpoint path."""
    yaml_path = save_temp_yaml(config)
    train_py = os.path.join(_training_dir, 'train.py')
    cmd = [sys.executable, train_py, '--detector_path', yaml_path,
           '--train_dataset', train_dataset, '--test_dataset', val_dataset]
    print(f"[train] {' '.join(cmd)}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        print(f"[train] WARNING: exit code {proc.returncode}")
        os.unlink(yaml_path)
        return None
    log_dir = config['log_dir']
    ckpt_candidates = list(Path(log_dir).glob('effort_*/test/avg/ckpt_best.pth'))
    if not ckpt_candidates:
        print(f"[train] WARNING: no checkpoint under {log_dir}")
        os.unlink(yaml_path)
        return None
    ckpt = str(sorted(ckpt_candidates, key=os.path.getmtime)[-1])
    print(f"[train] best ckpt: {ckpt}")
    os.unlink(yaml_path)
    return ckpt


# ── testall.py evaluation (subprocess) ─────────────────────────────────────

def run_testall(ckpt_path, test_datasets, log_path, extra_config=None):
    """Run testall.py on checkpoint; return per-dataset metrics dict.

    extra_config: optional dict of model-architecture config keys (e.g.
    use_freq_split / freq_split_pool) that are not present in the detector YAML
    but change the model structure. Merged into the test config so test.py
    rebuilds the model identically to training (else strict checkpoint loading
    fails with unexpected keys like 'stem.weight').
    """
    testall_py = os.path.join(_deepfake_dir, 'testall.py')
    TMP = tempfile.mkstemp(suffix='.yaml', prefix='effort_testall_')[1]
    with open(DETECTOR_YAML, 'r') as f:
        yc = yaml.safe_load(f)
    with open(TEST_YAML, 'r') as f:
        yc.update(yaml.safe_load(f))
    if extra_config:
        yc.update(extra_config)
    json_folder = os.path.join(_deepfake_dir, 'preprocessing', 'dataset_json')
    if os.path.isdir(json_folder):
        yc['dataset_json_folder'] = json_folder
    with open(TMP, 'w') as f:
        yaml.dump(yc, f)
    cmd = [sys.executable, testall_py, '--detector_path', TMP,
           '--weights_path', ckpt_path, '--test_datasets'] + test_datasets
    print(f"  [testall] {' '.join(cmd)}")
    sys.stdout.flush()
    with open(log_path, 'w') as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    os.unlink(TMP)
    metrics = {}
    with open(log_path, 'r') as lf:
        current_ds = None
        for line in lf:
            line = line.strip()
            if line.startswith('dataset:'):
                current_ds = line.split('dataset:')[1].strip()
                metrics[current_ds] = {}
            elif current_ds:
                for k in ['acc', 'auc', 'video_auc']:
                    if line.startswith(f'{k}:'):
                        metrics[current_ds][k] = float(line.split(':')[1].strip())
    with open(log_path, 'r') as lf:
        for line in lf:
            line = line.strip()
            for k in ['acc', 'auc', 'video_auc']:
                if line.startswith(f'{k}:'):
                    metrics.setdefault('average', {})[k] = float(line.split(':')[1].strip())
    return metrics


# ── Model loading and frame-level evaluation ───────────────────────────────

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
    return DataLoader(ds, batch_size=cfg['test_batchSize'], shuffle=False,
                      num_workers=int(cfg.get('workers', 4)),
                      collate_fn=ds.collate_fn)


def get_train_loader(config):
    """Build a DataLoader for training data — no augmentations."""
    cfg = config.copy()
    cfg['use_data_augmentation'] = False
    ds = DeepfakeAbstractBaseDataset(config=cfg, mode='train')
    return DataLoader(ds, batch_size=cfg['test_batchSize'], shuffle=False,
                      num_workers=int(cfg.get('workers', 4)),
                      collate_fn=ds.collate_fn)


def compute_comp_band_sigma(config, bands, num_batches=4):
    """Estimate the fixed per-band RMS scalars for the G17-1 model-side F line.

    ``bands`` is the list of ``comp_freq_bands`` the dualcomp model forwards.  For
    each band we average the per-image std of the band-passed reconstruction over
    a few no-aug training batches, so the scalar is a train-set statistic (G17-1
    §2: a fixed scalar, NOT per-image min-max or per-image re-scaling), which
    preserves inter-image amplitude differences on the frequency line.

    The images come from ``get_train_loader`` (CLIP-normalised, no augmentation)
    so the F-line band view the model sees at train/val/eval is byte-identical to
    the one calibrated here — the "三处一致" requirement.

    Returns ``{band: float}``.  The caller passes this dict as ``comp_band_sigma``
    into ``build_config`` so both the training and eval configs (and testall, via
    ``arch_keys``) carry the SAME scalars.
    """
    bands = list(bands)
    # Lazy import so this helper does not pull the detector module at import time.
    from detectors.effort_detector_dualcomp import band_rms_scalar
    loader = get_train_loader(config)
    sums = {b: 0.0 for b in bands}
    counts = {b: 0 for b in bands}
    seen = 0
    for data_dict in loader:
        images = data_dict['image']
        # band_rms_scalar returns {band: mean per-image std over this batch}.
        for b, s in band_rms_scalar(images, bands).items():
            sums[b] += s
            counts[b] += 1
        seen += 1
        if seen >= num_batches:
            break
    sigma = {b: (sums[b] / counts[b]) if counts[b] else 0.5 for b in bands}
    return sigma


@torch.no_grad()
def collect_predictions(model, data_loader):
    """Run model on data_loader; return (probs, labels) as numpy arrays."""
    probs_list, labels_list = [], []
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
    cm = sk_cm(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {'acc': acc, 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


# ── Full evaluation runner ─────────────────────────────────────────────────

def evaluate_model(config, ckpt_path, test_datasets, train_dataset, output_dir, exp_name):
    """Full eval: testall + frame-level confusion matrix + KDE plots.
    Returns summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(config, ckpt_path)

    # Frame-level test
    test_probs, test_labels = {}, {}
    for ds in test_datasets:
        loader = get_data_loader(config, ds, mode='test')
        test_probs[ds], test_labels[ds] = collect_predictions(model, loader)

    # Frame-level train
    train_loader = get_train_loader(config)
    train_probs, train_labels = collect_predictions(model, train_loader)

    # testall
    testall_log = os.path.join(output_dir, 'testall.log')
    # Pass model-architecture + data-side keys that are not in the detector
    # YAML so test.py rebuilds both the model and the dataset identically
    # (e.g. use_freq_split adds a stem layer, effort_aepa changes the model
    # class and adds patch/evidence heads, effort_maxev reads
    # max_evidence_cls_feature to pick the CLS-branch input (raw_token vs
    # pooler_output), freq_ablation makes __getitem__ band-pass the input).
    # max_evidence_init_ckpt is deliberately NOT propagated: it is only used to
    # seed the patch head in __init__ and is overwritten by the checkpoint's own
    # patch_head on strict load, so carrying it would just re-torch.load at test.
    arch_keys = ('use_freq_split', 'freq_split_pool', 'model_name',
                 'aepa_mode', 'aepa_lambda_f', 'aepa_eps',
                 'max_evidence_cls_feature', 'max_evidence_lambda', 'max_evidence_eps',
                 'max_evidence_inference',
                 'freq_ablation', 'freq_norm', 'freq_energy_match', 'freq_after_aug',
                 # G17-2 data-side residual keys (dataset-only, carried so test
                 # rebuilds the residual input identically to training).
                 'residual_ablation', 'residual_sigma', 'residual_alpha',
                 'residual_fft_r0', 'residual_shuffle',
                 # G17-1 model-side dual-line gated-fusion structural keys.
                 'comp_freq_bands', 'comp_freq_norm', 'comp_freq_rms', 'comp_band_sigma',
                 'comp_fuse', 'comp_gate_hidden', 'comp_lambda_freq', 'comp_lambda_max',
                 'comp_cls_feature',
                 # G18 LFEQ read-out head structural keys (the module shape + the
                 # fusion weight change the model structure; without them test.py
                 # would rebuild the head with the default fusion=0.5 and strict
                 # load would fail on the constructor re-init).
                 'lfeq_hidden_dim', 'lfeq_num_evidence_tokens', 'lfeq_depth',
                 'lfeq_num_heads', 'lfeq_dropout', 'lfeq_fusion_weight',
                 'lfeq_evidence_weight', 'lfeq_diversity_weight')
    extra_config = {k: config[k] for k in arch_keys if k in config}
    testall_metrics = run_testall(ckpt_path, test_datasets, testall_log,
                                  extra_config=extra_config)

    del model
    torch.cuda.empty_cache()

    # Print results
    print_results(exp_name, train_probs, train_labels, test_probs, test_labels,
                  testall_metrics, output_dir)

    # Return summary dict
    summary = {'exp_name': exp_name}
    if testall_metrics:
        summary['testall'] = testall_metrics
    train_m = compute_metrics(train_probs, train_labels)
    summary['train_acc'] = train_m['acc']
    summary['test_acc'] = {}
    for ds in test_datasets:
        m = compute_metrics(test_probs[ds], test_labels[ds])
        summary['test_acc'][ds] = m['acc']
    summary['test_acc_avg'] = float(np.mean(list(summary['test_acc'].values())))

    # Score mean/std per set (G13: record more than AUC — these reveal whether a
    # band's lower AUC comes from a collapsed/reversed score spread or a genuine
    # separation gap).
    if len(train_probs) > 0:
        summary['train_score'] = {'mean': float(train_probs.mean()),
                                  'std': float(train_probs.std())}
    else:
        summary['train_score'] = {'mean': None, 'std': None}
    summary['test_score'] = {}
    for ds in test_datasets:
        p = test_probs[ds]
        summary['test_score'][ds] = {'mean': float(p.mean()), 'std': float(p.std())}
    return summary


# ── Output ─────────────────────────────────────────────────────────────────

def print_results(exp_name, train_probs, train_labels,
                  test_probs_dict, test_labels_dict, testall_metrics, output_dir):
    """Print metrics and save score distribution plots."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}\n  {exp_name}\n{'='*70}")
    if testall_metrics:
        avg = testall_metrics.get('average', {})
        print(f"\n  -- testall (avg) --")
        for k in ['video_auc', 'auc', 'acc']:
            print(f"  {k}: {avg.get(k, 'N/A')}")
    if len(train_probs) > 0:
        m = compute_metrics(train_probs, train_labels)
        print(f"\n  -- Train --\n  Acc: {m['acc']:.4f}")
        print(f"  CM: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")
        print(f"  Score mean/std: {train_probs.mean():.3f} ± {train_probs.std():.3f}")
        print(f"\n  -- Test Sets --")
        print(f"  {'Dataset':<25s} | {'Acc':>8s} | {'TN':>6s} | {'FP':>6s} | {'FN':>6s} | {'TP':>6s}")
        print(f"  {'-'*25} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")
    all_acc, total_tn, total_fp, total_fn, total_tp = [], 0, 0, 0, 0
    for ds in sorted(test_probs_dict.keys()):
        p = test_probs_dict[ds]
        m = compute_metrics(p, test_labels_dict[ds])
        all_acc.append(m['acc'])
        total_tn += m['tn']; total_fp += m['fp']
        total_fn += m['fn']; total_tp += m['tp']
        print(f"  {ds:<25s} | {m['acc']:8.4f} | {m['tn']:6d} | {m['fp']:6d} | {m['fn']:6d} | {m['tp']:6d}")
        print(f"  {'':<25s} |  score mean/std: {p.mean():.3f} ± {p.std():.3f}")
    if len(all_acc) > 1:
        print(f"  {'-'*25} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")
        print(f"  {'average':<25s} | {np.mean(all_acc):8.4f} | {total_tn:6d} | {total_fp:6d} | {total_fn:6d} | {total_tp:6d}")
    _plot_score_distributions(exp_name, output_dir, train_probs, train_labels,
                               test_probs_dict, test_labels_dict)


def _plot_score_distributions(exp_name, output_dir, train_probs, train_labels,
                               test_probs_dict, test_labels_dict):
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
        _plot_one_dist(axes[col_idx], train_probs, train_labels, 'Train Set', 'tab:blue', xs)
        col_idx += 1
    colors = plt.cm.tab10.colors
    for ds in sorted(test_probs_dict.keys()):
        _plot_one_dist(axes[col_idx], test_probs_dict[ds], test_labels_dict[ds], ds,
                       colors[col_idx % len(colors)], xs)
        col_idx += 1
    fig.suptitle(f"Score Distributions — {exp_name}\n(Real vs Fake, per Dataset)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"score_dist_{safe_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Score plot: {out_path}")


def _plot_one_dist(ax, probs, labels, title, color, xs):
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

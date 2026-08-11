# Experiment Refactor — Pyramid Mixup 12-Config Matrix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor all post-7.30 experiment code, fix `loss_mask` support in trainer, replace 5 sweep scripts + 2 experiment .py files with a single unified experiment runner, and collect all 12 results into one summary file.

**Architecture:** A single Python experiment runner (`experiments/run_experiments.py`) reads a declarative experiment matrix, calls `train.py` + `testall.py` as subprocesses, and evaluates each checkpoint via direct model loading (confusion matrix + KDE score distributions). The `loss_mask` mechanism is added to mixup functions and `get_losses` so RF pairs can be excluded from loss without being removed from training.

**Tech Stack:** Python 3, PyTorch, existing DeepfakeBench training infrastructure, bash for master runner.

## Global Constraints

- All experiments use BalanceBatchSampler v1 (real_ratio=0.30), mixup_alpha=5.0, mixup_gamma=1.0, lap_num_levels=3, n_epochs=10
- Train dataset: FaceForensics++, Val dataset: Celeb-DF-v2
- Test datasets: WDF FFIW Celeb-DF-v2 DeepFakeDetection DFDC DFDCP DeeperForensics-1.0
- RR/FF must be pixel-space mixup (not pyramid) in base `lap_pyramid_mixup`
- RF pairs participate in forward pass (data augmentation) but can be excluded from loss via `loss_mask`
- Minimal code changes — reuse existing verified infrastructure where possible

---

## File Structure

```
DeepfakeBench/
├── training/
│   ├── trainer/
│   │   └── trainer_v2.py                    # [MODIFY] Add loss_mask to mixup dispatch
│   └── detectors/
│       └── effort_detector.py               # [MODIFY] Use loss_mask in get_losses
├── experiments/
│   ├── experiment_utils.py                  # [CREATE] Shared train/eval utilities
│   └── run_experiments.py                   # [CREATE] Unified experiment runner
├── run_all_experiments.sh                   # [REWRITE] Master runner → calls Python runner
└── sweep_*.sh                               # [DELETE] All 5 old sweep scripts removed
```

### Task 1: Add `loss_mask` to `lap_pyramid_mixup`

**Files:**
- Modify: `DeepfakeBench/training/trainer/trainer_v2.py:475-619`

**Interfaces:**
- Produces: `lap_pyramid_mixup` now returns `(mixed_x, mixed_y, mixed_label, loss_mask)` where `loss_mask` is a [N] float32 tensor, all 1.0 (RF participates in loss by default)

- [ ] **Step 1: Add loss_mask to return values**

In `lap_pyramid_mixup` (around line 602-619), after the existing combine logic, add `loss_mask` construction:

```python
# ── loss_mask: which samples participate in loss ────────────────────────
mask_parts = []
if n_rr > 0:
    mask_parts.append(torch.ones(n_rr, device=x.device, dtype=torch.float32))
if n_ff > 0:
    mask_parts.append(torch.ones(n_ff, device=x.device, dtype=torch.float32))
if n_rf > 0:
    mask_parts.append(torch.ones(n_rf, device=x.device, dtype=torch.float32))
loss_mask = torch.cat(mask_parts, dim=0) if mask_parts else mixed_y.new_zeros(0)
```

Change return to:
```python
return mixed_x, mixed_y, mixed_label, loss_mask
```

### Task 2: Add `loss_mask` to `lap_pyramid_label_mixup`

**Files:**
- Modify: `DeepfakeBench/training/trainer/lap_pyramid_label_variants.py:160-169`

**Interfaces:**
- Produces: Same as Task 1 — returns 4-tuple with `loss_mask`

- [ ] **Step 1: Add loss_mask construction and return**

After line 158 (`rf_label = ...`), before combine, compute `loss_mask`:

```python
# ── loss_mask ──────────────────────────────────────────────────────────
mask_parts = []
if rr_n > 0:
    mask_parts.append(torch.ones(rr_n, device=x.device, dtype=torch.float32))
if ff_n > 0:
    mask_parts.append(torch.ones(ff_n, device=x.device, dtype=torch.float32))
if n_rf > 0:
    mask_parts.append(torch.ones(n_rf, device=x.device, dtype=torch.float32))
loss_mask = torch.cat(mask_parts, dim=0) if mask_parts else x.new_zeros(0)[:,0,0,0]
```

Change return statement (line 169) to:
```python
return mixed_x, mixed_y, mixed_label, loss_mask
```

### Task 3: Add `loss_mask` to `lap_pyramid_all_mixup`

**Files:**
- Modify: `DeepfakeBench/training/trainer/trainer_v2.py:784-937`

**Interfaces:**
- Produces: `lap_pyramid_all_mixup` returns 4-tuple with `loss_mask`

- [ ] **Step 1: Add loss_mask construction and return**

After line 935 (`mixed_label = ...`), before return:

```python
# ── loss_mask ──────────────────────────────────────────────────────────
mask_parts = []
if n_rr > 0:
    mask_parts.append(torch.ones(n_rr, device=x.device, dtype=torch.float32))
if n_ff > 0:
    mask_parts.append(torch.ones(n_ff, device=x.device, dtype=torch.float32))
if n_rf > 0:
    mask_parts.append(torch.ones(n_rf, device=x.device, dtype=torch.float32))
loss_mask = torch.cat(mask_parts, dim=0) if mask_parts else mixed_y.new_zeros(0)
```

Change return to:
```python
return mixed_x, mixed_y, mixed_label, loss_mask
```

### Task 4: Add `loss_mask` to `lap_pyramid_rrff_mixup` and `rrff_explicit_mixup`

**Files:**
- Modify: `DeepfakeBench/training/trainer/trainer_v2.py:940-1063` (rrff) and `688-781` (explicit)

**Interfaces:**
- Produces: Both functions return 4-tuple with `loss_mask` (all 1.0 — no RF to mask)

- [ ] **Step 1: lap_pyramid_rrff_mixup**

After line 1061 (`mixed_label = ...`):
```python
loss_mask = torch.ones_like(mixed_y) if mixed_y.numel() > 0 else mixed_y.new_zeros(0)
```
Return as 4-tuple.

- [ ] **Step 2: rrff_explicit_mixup**

After line 779:
```python
loss_mask = torch.ones_like(mixed_y) if mixed_y.numel() > 0 else mixed_y.new_zeros(0)
```
Return as 4-tuple.

### Task 5: Update mixup dispatch in training loop

**Files:**
- Modify: `DeepfakeBench/training/trainer/trainer_v2.py:1289-1386`

**Interfaces:**
- Consumes: 4-tuple returns from mixup functions
- Produces: `data_dict['loss_mask']` passed through to `get_losses`

- [ ] **Step 1: Update all mixup call sites to unpack loss_mask**

Each mixup call currently does:
```python
data_dict['image'], data_dict['label_soft'], data_dict['label'] = some_mixup(...)
```

Change ALL of them to:
```python
data_dict['image'], data_dict['label_soft'], data_dict['label'], data_dict['loss_mask'] = some_mixup(...)
```

This applies to lines handling:
- `lap_pyramid` (line 1321)
- `lap_pyramid_label*` (line 1337, already in `lap_pyramid_label_variants.py`)
- `lap_pyramid_all` (line 1348)
- `lap_pyramid_rrff` (line 1356)
- `rrff_explicit` (line 1364)
- `asymmetric_mixup` (line 1315 — this returns 2-tuple, needs to wrap: `loss_mask = torch.ones_like(mixed_y)`)

- [ ] **Step 2: Handle asymmetric_mixup (returns 2-tuple)**

For the `asymmetric_mixup` call (line 1315):
```python
data_dict['image'], data_dict['label_soft'] = asymmetric_mixup(...)
data_dict['loss_mask'] = torch.ones(data_dict['image'].size(0),
                                     device=data_dict['image'].device,
                                     dtype=torch.float32)
```

- [ ] **Step 3: Handle v2 sampler / rf_pair_mixup**

For `rf_pair_mixup` (lines 1308-1313):
```python
data_dict['image'], data_dict['label_soft'], data_dict['label'], data_dict['loss_mask'] = \
    rf_pair_mixup(...)
```

- [ ] **Step 4: Replace mixup_loss_strip logic**

Replace old strip logic (lines 1378-1385):
```python
# ── Strip RF from loss: zero out loss_mask for RF pairs ─────
if self.config.get('mixup_loss_strip', False):
    y_soft = data_dict['label_soft']
    # RF samples: ỹ ∈ (0, 1), i.e. not exactly 0 or 1
    is_rf = ~((y_soft <= 1e-6) | (y_soft >= 1.0 - 1e-6))
    data_dict['loss_mask'] = data_dict['loss_mask'].clone()
    data_dict['loss_mask'][is_rf] = 0.0
    data_dict.pop('label_soft', None)
```

This way RF images still pass through the model (forward pass) but their loss contribution is zeroed.

### Task 6: Add `loss_mask` support in `get_losses`

**Files:**
- Modify: `DeepfakeBench/training/detectors/effort_detector.py:225-291`

**Interfaces:**
- Consumes: `data_dict['loss_mask']` (when present)
- Produces: Masked `overall` loss

- [ ] **Step 1: Apply loss_mask in soft-label CE path**

In `get_losses`, after computing `per_sample` (line 234), apply mask:

```python
if 'loss_mask' in data_dict:
    mask = data_dict['loss_mask']
    per_sample = per_sample * mask
    loss = per_sample.sum() / mask.sum().clamp(min=1)
else:
    loss = per_sample.mean()
```

Replace lines 244-265 (the existing soft-label loss computation) with:
```python
if 'label_soft' in data_dict:
    log_probs = F.log_softmax(pred, dim=1)
    y_soft = data_dict['label_soft']
    per_sample = -(y_soft * log_probs[:, 1] +
                   (1.0 - y_soft) * log_probs[:, 0])

    if data_dict.get('mixup_selection') == 'mean':
        K = data_dict['mixup_k']
        n_rf = data_dict.get('n_rf', 0)
        if n_rf > 0:
            per_sample[-n_rf:] = per_sample[-n_rf:].view(-1, K).mean(dim=1)
        else:
            per_sample = per_sample.view(-1, K).mean(dim=1)

    # ── Apply loss_mask (zero out RF if mixup_loss_strip) ──
    if 'loss_mask' in data_dict:
        mask = data_dict['loss_mask']
        per_sample = per_sample * mask
        loss = per_sample.sum() / mask.sum().clamp(min=1)
    else:
        loss = per_sample.mean()
```

### Task 7: Update `rf_pair_mixup` return signature

**Files:**
- Modify: `DeepfakeBench/training/trainer/trainer_v2.py:622-686` (rf_pair_mixup)

**Interfaces:**
- Produces: Returns 4-tuple `(mixed_x, mixed_y, mixed_label, loss_mask)`

- [ ] **Step 1: Add loss_mask to rf_pair_mixup**

At end of `rf_pair_mixup`:
```python
loss_mask = torch.ones(N, device=x.device, dtype=torch.float32)
return rf_x, rf_y, rf_label, loss_mask
```

### Task 8: Create shared experiment utilities

**Files:**
- Create: `DeepfakeBench/experiments/experiment_utils.py`

**Interfaces:**
- Produces: `build_config()`, `train_model()`, `run_testall()`, `load_model()`, `collect_predictions()`, `compute_metrics()`, `plot_score_distributions()`, `evaluate_model()`

- [ ] **Step 1: Create the module**

```python
"""
Shared experiment utilities for pyramid mixup experiments.
Reuses train.py + testall.py as subprocesses, plus direct model loading for
frame-level confusion matrix and KDE score distributions.
"""
import os, sys, tempfile, subprocess, argparse
from pathlib import Path
import numpy as np, yaml
import matplotlib; matplotlib.use('Agg')
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

# ─────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────

def build_config(pyramid_mode='lap_pyramid',
                 mixup_loss_strip=False,
                 mixup_alpha=5.0, mixup_gamma=1.0,
                 mixup_beta_b=None, mixup_beta_flip=False,
                 lap_num_levels=3,
                 sampler_real_ratio=0.30,
                 log_dir=None,
                 train_dataset=None, test_dataset=None,
                 n_epochs=10, for_training=True):
    """Build merged config dict."""
    with open(DETECTOR_YAML, 'r') as f:
        config = yaml.safe_load(f)
    base = TRAIN_YAML if for_training else TEST_YAML
    with open(base, 'r') as f:
        config.update(yaml.safe_load(f))
    json_folder = os.path.join(_deepfake_dir, 'preprocessing', 'dataset_json')
    if os.path.isdir(json_folder):
        config['dataset_json_folder'] = json_folder

    config['use_mixup'] = True
    config['mixup_mode'] = pyramid_mode
    config['mixup_alpha'] = mixup_alpha
    config['mixup_gamma'] = mixup_gamma
    config['mix_domain'] = 'rgb'
    config['lap_num_levels'] = lap_num_levels
    config['mixup_loss_strip'] = mixup_loss_strip
    if mixup_beta_b is not None:
        config['mixup_beta_b'] = mixup_beta_b
    config['mixup_beta_flip'] = mixup_beta_flip

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

# ─────────────────────────────────────────────────────────────────────
# Training (subprocess)
# ─────────────────────────────────────────────────────────────────────

def train_model(config, train_dataset, val_dataset):
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

# ─────────────────────────────────────────────────────────────────────
# testall.py evaluation (subprocess)
# ─────────────────────────────────────────────────────────────────────

def run_testall(ckpt_path, test_datasets, log_path):
    testall_py = os.path.join(_deepfake_dir, 'testall.py')
    TMP = tempfile.mkstemp(suffix='.yaml', prefix='effort_testall_')[1]
    with open(DETECTOR_YAML, 'r') as f:
        yc = yaml.safe_load(f)
    with open(TEST_YAML, 'r') as f:
        yc.update(yaml.safe_load(f))
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

# ─────────────────────────────────────────────────────────────────────
# Model loading and frame-level evaluation
# ─────────────────────────────────────────────────────────────────────

def load_model(config, ckpt_path):
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
    cfg = config.copy()
    cfg['test_dataset'] = dataset_name
    ds = DeepfakeAbstractBaseDataset(config=cfg, mode=mode)
    return DataLoader(ds, batch_size=cfg['test_batchSize'], shuffle=False,
                      num_workers=int(cfg.get('workers', 4)),
                      collate_fn=ds.collate_fn)

def get_train_loader(config):
    cfg = config.copy()
    cfg['use_data_augmentation'] = False
    ds = DeepfakeAbstractBaseDataset(config=cfg, mode='train')
    return DataLoader(ds, batch_size=cfg['test_batchSize'], shuffle=False,
                      num_workers=int(cfg.get('workers', 4)),
                      collate_fn=ds.collate_fn)

@torch.no_grad()
def collect_predictions(model, data_loader):
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
    preds = (probs > 0.5).astype(int)
    acc = float(accuracy_score(labels, preds))
    cm = sk_cm(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {'acc': acc, 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

# ─────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────

def evaluate_model(config, ckpt_path, test_datasets, train_dataset, output_dir, exp_name):
    """Full eval: testall + frame-level confusion matrix + KDE plots."""
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
    testall_metrics = run_testall(ckpt_path, test_datasets, testall_log)

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
    summary['test_acc_avg'] = np.mean(list(summary['test_acc'].values()))
    return summary

# ─────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────

def print_results(exp_name, train_probs, train_labels,
                  test_probs_dict, test_labels_dict, testall_metrics, output_dir):
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
    print(f"\n  -- Test Sets --")
    print(f"  {'Dataset':<25s} | {'Acc':>8s} | {'TN':>6s} | {'FP':>6s} | {'FN':>6s} | {'TP':>6s}")
    print(f"  {'-'*25} | {'-'*8} | {'-'*6} | {'-'*6} | {'-'*6} | {'-'*6}")
    all_acc, total_tn, total_fp, total_fn, total_tp = [], 0, 0, 0, 0
    for ds in sorted(test_probs_dict.keys()):
        m = compute_metrics(test_probs_dict[ds], test_labels_dict[ds])
        all_acc.append(m['acc'])
        total_tn += m['tn']; total_fp += m['fp']
        total_fn += m['fn']; total_tp += m['tp']
        print(f"  {ds:<25s} | {m['acc']:8.4f} | {m['tn']:6d} | {m['fp']:6d} | {m['fn']:6d} | {m['tp']:6d}")
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
```

### Task 9: Create unified experiment runner

**Files:**
- Create: `DeepfakeBench/experiments/run_experiments.py`

**Interfaces:**
- Consumes: `experiment_utils` module
- Produces: All 12 configs trained + evaluated, results JSON

- [ ] **Step 1: Create the runner script with the experiment matrix**

```python
"""
Unified experiment runner — 12-config pyramid mixup matrix.
Usage:
    python3 experiments/run_experiments.py                          # all 12 configs
    python3 experiments/run_experiments.py --groups G1 G3           # specific groups
    python3 experiments/run_experiments.py --eval_only --ckpt_dir /path/to/ckpts
"""
import os, sys, argparse, json
from pathlib import Path

_current_dir = os.path.dirname(os.path.abspath(__file__))
_deepfake_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _current_dir)
sys.path.insert(0, _deepfake_dir)

from experiment_utils import *

TRAIN_DS = 'FaceForensics++'
VAL_DS = 'Celeb-DF-v2'
TEST_DS = ['WDF', 'FFIW', 'Celeb-DF-v2', 'DeepFakeDetection', 'DFDC', 'DFDCP', 'DeeperForensics-1.0']

# ═══════════════════════════════════════════════════════════════════════════
# Experiment definitions
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # ── G1: 2×2 label × scope ───────────────────────────────────────────
    {'group': 'G1', 'name': 'label0_top',    'mode': 'lap_pyramid_label0_top',    'strip': False},
    {'group': 'G1', 'name': 'label0_bottom', 'mode': 'lap_pyramid_label0_bottom', 'strip': False},
    {'group': 'G1', 'name': 'label1_top',    'mode': 'lap_pyramid_label1_top',    'strip': False},
    {'group': 'G1', 'name': 'label1_bottom', 'mode': 'lap_pyramid_label1_bottom', 'strip': False},

    # ── G2: Beta(2,5) on label1_top ─────────────────────────────────────
    {'group': 'G2', 'name': 'beta25',       'mode': 'lap_pyramid_label1_top', 'strip': False,
     'beta_b': 5, 'beta_flip': False},
    {'group': 'G2', 'name': 'beta25_flip',  'mode': 'lap_pyramid_label1_top', 'strip': False,
     'beta_b': 5, 'beta_flip': True},

    # ── G3: Full scope label 0/1 ─────────────────────────────────────────
    {'group': 'G3', 'name': 'label0_full',  'mode': 'lap_pyramid_label0_full', 'strip': False},
    {'group': 'G3', 'name': 'label1_full',  'mode': 'lap_pyramid_label1_full', 'strip': False},

    # ── G4: Loss ablation ────────────────────────────────────────────────
    {'group': 'G4', 'name': 'exp1_soft_ce',  'mode': 'lap_pyramid', 'strip': False},
    {'group': 'G4', 'name': 'exp2_strip_rf', 'mode': 'lap_pyramid', 'strip': True},

    # ── G5: RR+FF pyramid (on strip-RF basis) ────────────────────────────
    {'group': 'G5', 'name': 'g1_rf_stripped',     'mode': 'lap_pyramid_all',  'strip': True},
    {'group': 'G5', 'name': 'g2_rf_not_generated','mode': 'lap_pyramid_rrff', 'strip': False},
]


def run_one(exp, args):
    """Train + eval a single experiment config."""
    exp_name = f"{exp['group']}_{exp['name']}"
    output_dir = os.path.join(args.output_dir, exp['group'], exp['name'])
    os.makedirs(output_dir, exist_ok=True)

    # Build training config
    log_dir = os.path.join(args.output_dir, 'logs', exp['group'], exp['name'])
    config = build_config(
        pyramid_mode=exp['mode'],
        mixup_loss_strip=exp['strip'],
        mixup_alpha=args.alpha, mixup_gamma=args.gamma,
        mixup_beta_b=exp.get('beta_b'), mixup_beta_flip=exp.get('beta_flip', False),
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
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
        pyramid_mode=exp['mode'], mixup_loss_strip=exp['strip'],
        mixup_alpha=args.alpha,
        mixup_beta_b=exp.get('beta_b'), mixup_beta_flip=exp.get('beta_flip', False),
        lap_num_levels=args.num_levels,
        sampler_real_ratio=args.sampler_real_ratio,
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
    parser.add_argument('--output_dir', type=str, default='./experiment_results/master_sweep',
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
        print(f"\n[{'='*60}]\n  [{i+1}/{total}] {exp_name}\n[{'='*60}]")
        r = run_one(exp, args)
        results.append(r)
        # Save incremental results
        results_path = os.path.join(args.output_dir, 'all_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*70}\n  SUMMARY TABLE\n{'='*70}")
    print(f"{'Exp':<25s} | {'Status':<12s} | {'testall_vAUC':>12s} | {'testall_AUC':>12s} | {'testall_ACC':>12s} | {'frame_ACC':>10s}")
    print(f"{'-'*25} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*10}")
    for r in results:
        status = r.get('status', '?')
        ta = r.get('testall', {}).get('average', {})
        print(f"{r['exp_name']:<25s} | {status:<12s} | "
              f"{str(ta.get('video_auc','N/A')):>12s} | {str(ta.get('auc','N/A')):>12s} | "
              f"{str(ta.get('acc','N/A')):>12s} | {r.get('train_acc','N/A'):>10}")

    print(f"\nResults saved to: {args.output_dir}/all_results.json")


if __name__ == '__main__':
    main()
```

### Task 10: Rewrite master runner shell script

**Files:**
- Rewrite: `DeepfakeBench/run_all_experiments.sh`

- [ ] **Step 1: Simple wrapper around the Python runner**

```bash
#!/bin/bash
# ===========================================================================
# Master Experiment Runner — Pyramid Mixup Full Matrix (5 groups, 12 configs)
# ===========================================================================
# Usage:
#   bash run_all_experiments.sh              # single GPU, sequential
#   bash run_all_experiments.sh 4            # 4-GPU DDP (not yet supported in runner)
#
# All results: ./experiment_results/master_sweep_YYYYMMDD_HHMMSS/
#   all_results.json  — JSON with all metrics
#   MASTER_SUMMARY.log — text summary
# ===========================================================================
set -uo pipefail

NGPU=${1:-1}
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULT_DIR="./experiment_results/master_sweep_${TIMESTAMP}"

echo "============================================================"
echo "  Master Experiment Runner"
echo "  Result dir: $RESULT_DIR"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

mkdir -p "$RESULT_DIR"

if [ "$NGPU" -gt 1 ]; then
    echo "[WARN] Multi-GPU not yet supported in unified runner. Using single GPU."
fi

python3 experiments/run_experiments.py \
    --output_dir "$RESULT_DIR" \
    --alpha 5.0 --gamma 1.0 --num_levels 3 \
    --sampler_real_ratio 0.30 --n_epochs 10 \
    2>&1 | tee "${RESULT_DIR}/MASTER_SUMMARY.log"

echo ""
echo "============================================================"
echo "  Done — $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results: $RESULT_DIR"
echo "============================================================"
```

### Task 11: Remove old sweep scripts

**Files:**
- Delete: `DeepfakeBench/sweep_pyramid_label_variants.sh`
- Delete: `DeepfakeBench/sweep_beta25_top_l1.sh`
- Delete: `DeepfakeBench/sweep_pyramid_label_full.sh`
- Delete: `DeepfakeBench/sweep_beta_scan_label0_full.sh`
- Delete: `DeepfakeBench/sweep_pyramid_sampler.sh`
- Delete: `DeepfakeBench/experiments/pyramid_loss_ablation.py`
- Delete: `DeepfakeBench/experiments/pyramid_rrff_experiment.py`
- Delete: `DeepfakeBench/experiments/rrff_explicit_experiment.py`
- Delete: `DeepfakeBench/experiments/pyramid_loss_ablation.sh`
- Delete: `DeepfakeBench/experiments/pyramid_rrff_experiment.sh`

- [ ] **Step 1: Remove old files**

```bash
cd DeepfakeBench
rm -f sweep_pyramid_label_variants.sh sweep_beta25_top_l1.sh \
      sweep_pyramid_label_full.sh sweep_beta_scan_label0_full.sh \
      sweep_pyramid_sampler.sh
rm -f experiments/pyramid_loss_ablation.py experiments/pyramid_rrff_experiment.py \
      experiments/rrff_explicit_experiment.py \
      experiments/pyramid_loss_ablation.sh experiments/pyramid_rrff_experiment.sh
```

### Task 12: Verification

- [ ] **Step 1: Verify import chain**

```bash
cd DeepfakeBench && python3 -c "from experiments.experiment_utils import build_config, train_model; print('OK')"
```

- [ ] **Step 2: Verify dry-run experiment runner (config building only)**

```bash
cd DeepfakeBench && python3 -c "
from experiments.run_experiments import EXPERIMENTS
for e in EXPERIMENTS:
    print(f\"  {e['group']}_{e['name']}: mode={e['mode']} strip={e['strip']}\")
"
```

Expected output: all 12 configs listed.

- [ ] **Step 3: Verify mixup import and loss_mask signature**

```bash
cd DeepfakeBench && python3 -c "
from trainer.trainer_v2 import lap_pyramid_mixup
import inspect
sig = inspect.signature(lap_pyramid_mixup)
print('lap_pyramid_mixup params:', list(sig.parameters.keys()))
"
```

Verify output shows existing parameters unchanged.

- [ ] **Step 4: Run a smoke test (train 1 epoch on 1 small dataset)**

```bash
cd DeepfakeBench && python3 experiments/run_experiments.py \
    --groups G4 --output_dir ./experiment_results/smoke_test \
    --alpha 5.0 --gamma 1.0 --num_levels 3 \
    --sampler_real_ratio 0.30 --n_epochs 1
```

Expected: One training epoch completes, `all_results.json` written with 2 entries (exp1_soft_ce, exp2_strip_rf).

### Task 13: Commit

- [ ] **Step 1: Stage and commit**

```bash
git add DeepfakeBench/training/trainer/trainer_v2.py \
        DeepfakeBench/training/trainer/lap_pyramid_label_variants.py \
        DeepfakeBench/training/detectors/effort_detector.py \
        DeepfakeBench/experiments/experiment_utils.py \
        DeepfakeBench/experiments/run_experiments.py \
        DeepfakeBench/run_all_experiments.sh
git rm DeepfakeBench/sweep_pyramid_label_variants.sh \
       DeepfakeBench/sweep_beta25_top_l1.sh \
       DeepfakeBench/sweep_pyramid_label_full.sh \
       DeepfakeBench/sweep_beta_scan_label0_full.sh \
       DeepfakeBench/sweep_pyramid_sampler.sh \
       DeepfakeBench/experiments/pyramid_loss_ablation.py \
       DeepfakeBench/experiments/pyramid_rrff_experiment.py \
       DeepfakeBench/experiments/rrff_explicit_experiment.py \
       DeepfakeBench/experiments/pyramid_loss_ablation.sh \
       DeepfakeBench/experiments/pyramid_rrff_experiment.sh
git commit -m "refactor: replace 5 sweep scripts + 2 experiment files with unified runner

- Add loss_mask to all mixup functions (lap_pyramid, label variants,
  all_mixup, rrff_mixup, rf_pair_mixup, rrff_explicit)
- Update training loop to unpack loss_mask from mixup calls
- Replace mixup_loss_strip's image-filtering with loss_mask-based RF exclusion
- Add loss_mask support in effort_detector.get_losses
- Create experiments/experiment_utils.py with shared train/eval utilities
- Create experiments/run_experiments.py — single runner for all 12 configs
- Rewrite run_all_experiments.sh as thin wrapper
- Remove old sweep scripts (*.sh) and experiment python files
- RF pairs now participate in forward pass (data augmentation) even when
  excluded from loss via mask"
```

#!/usr/bin/env python3
"""
Standalone Validation: Information Graph Theory for Deepfake Detection
======================================================================

Phase-1 theoretical validation — standalone analysis using frozen CLIP.
Does NOT modify the EffortDetector training pipeline.

Validates three hypotheses:
  1. MI between CLIP patch tokens discriminates real from fake
  2. Graph Laplacian spectrum differs between real and fake
  3. GNN embeddings on information graphs capture discriminative structure

All features are evaluated with the SAME Logistic Regression classifier
(5-fold CV, C tuned via grid search) for fair comparison.

Protocol
--------
  Train:  FF++ c23 (1000 real + 1000 fake)  → feature extraction + PCA fit
  Test A: FF++ c23 (500 real + 500 fake)     → in-domain evaluation
  Test B: Celeb-DF v2 (500 real + 500 fake)  → cross-domain generalization

Feature sets compared (10 methods):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  #  Method                   Dim   Source                          │
  ├─────────────────────────────────────────────────────────────────────┤
  │  1  CLIP CLS (baseline)      768   CLS token                       │
  │  2  Token mean (baseline)    768   mean of patch tokens             │
  │  3  MI stats                   5   M: mean, std, ‖M‖_F, skew, kurt │
  │  4  MI SVD                    20   top-20 singular values of M     │
  │  5  MI eigen                  10   top-10 eigenvalues of (M+M^T)/2 │
  │  6  Spectrum (MI graph)       23   λ_low(10)+λ_high(10)+gap+H_G    │
  │  7  Spectrum (cos graph)      23   ablation: use cosine adjacency  │
  │  8  GNN embedding            128   InfoGraphGNN global pooling     │
  │  9  MI+Spec (concat)       5+23   combined features                │
  │ 10  Spectrum (random graph)   23   sanity check: random adjacency  │
  └─────────────────────────────────────────────────────────────────────┘

Usage
-----
  python experiments/validate_info_graph.py \
      --detector_path DeepfakeBench/training/config/detector/effort.yaml \
      --clip_model_path /path/to/clip-vit-large-patch14 \
      --train_dataset FaceForensics++ \
      --test_datasets Celeb-DF-v2 FaceForensics++ \
      --max_train_samples 2000 --max_test_samples 1000 \
      --batch_size 16 --device cuda

Author: personal experiment
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

# ── scikit-learn ──────────────────────────────────────────────────────────
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    average_precision_score,
)

# ── PyTorch ───────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from transformers import CLIPModel

# ── Project root detection ────────────────────────────────────────────────
# Works regardless of whether the script is called from project root,
# DeepfakeBench/, or any subdirectory.
_SCRIPT_FILE = os.path.abspath(__file__)
_SCRIPT_DIR  = os.path.dirname(_SCRIPT_FILE)
_project_root = os.path.dirname(_SCRIPT_DIR)  # experiments/ → project root
_training_dir = os.path.join(_project_root, 'DeepfakeBench', 'training')
sys.path.insert(0, _training_dir)

# Default config paths (relative to project root)
_DEFAULT_DETECTOR_YAML = os.path.join(_project_root, 'DeepfakeBench', 'training',
                                       'config', 'detector', 'effort.yaml')
_DEFAULT_TRAIN_CONFIG  = os.path.join(_project_root, 'DeepfakeBench', 'training',
                                       'config', 'train_config.yaml')
_DEFAULT_TEST_CONFIG   = os.path.join(_project_root, 'DeepfakeBench', 'training',
                                       'config', 'test_config.yaml')
_DEFAULT_CLIP_MODEL    = os.path.join(_project_root, 'DeepfakeBench', 'training',
                                       'models--openai--clip-vit-large-patch14')
_DEFAULT_OUTPUT_DIR    = os.path.join(_project_root, 'experiments', 'results')

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

# ── Direct import to avoid chain-import through detectors/__init__.py
#     (which requires tensorboard) ──────────────────────────────────────────
import importlib.util as _importlib_util

_igt_path = os.path.join(_training_dir, 'detectors', 'info_graph_theory.py')
_igt_spec = _importlib_util.spec_from_file_location('info_graph_theory', _igt_path)
_igt = _importlib_util.module_from_spec(_igt_spec)
_igt_spec.loader.exec_module(_igt)

MutualInformationAnalyzer = _igt.MutualInformationAnalyzer
GraphSpectralAnalyzer   = _igt.GraphSpectralAnalyzer
InfoGraphGNN            = _igt.InfoGraphGNN
top_k_adjacency         = _igt.top_k_adjacency
build_cosine_adjacency  = _igt.build_cosine_adjacency
build_random_adjacency  = _igt.build_random_adjacency

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Information Graph Theory — standalone validation')
    p.add_argument('--detector_path', type=str,
                   default=_DEFAULT_DETECTOR_YAML)
    p.add_argument('--train_config', type=str,
                   default=_DEFAULT_TRAIN_CONFIG)
    p.add_argument('--test_config', type=str,
                   default=_DEFAULT_TEST_CONFIG)
    p.add_argument('--clip_model_path', type=str,
                   default=_DEFAULT_CLIP_MODEL)
    p.add_argument('--train_dataset', type=str, nargs='+',
                   default=['FaceForensics++'])
    p.add_argument('--test_datasets', type=str, nargs='+',
                   default=['Celeb-DF-v2', 'FaceForensics++'])
    p.add_argument('--max_train_samples', type=int, default=2000)
    p.add_argument('--max_test_samples', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--pca_dim', type=int, default=32)
    p.add_argument('--gnn_k', type=int, default=10,
                   help='top-k edges per node for GNN sparsification')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=1024)
    p.add_argument('--output_dir', type=str, default=_DEFAULT_OUTPUT_DIR)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Config & dataset setup
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(args):
    """Merge detector + train + test configs."""
    with open(args.detector_path, 'r') as f:
        cfg = yaml.safe_load(f)
    with open(args.train_config, 'r') as f:
        cfg2 = yaml.safe_load(f)
    cfg.update({k: v for k, v in cfg2.items() if k not in cfg})
    with open(args.test_config, 'r') as f:
        cfg3 = yaml.safe_load(f)
    cfg.update({k: v for k, v in cfg3.items() if k not in cfg})
    return cfg


def build_dataloader(cfg, mode, dataset_name, max_samples=None):
    """Build a DataLoader for a given dataset and mode.

    Args:
        cfg: merged config dict
        mode: 'train' or 'test'
        dataset_name: str or list[str]
        max_samples: cap the number of samples (None = use all)

    Returns:
        DataLoader
    """
    cfg = deepcopy(cfg)
    if mode == 'train':
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]
        cfg['train_dataset'] = dataset_name
    else:
        cfg['test_dataset'] = dataset_name

    ds = DeepfakeAbstractBaseDataset(config=cfg, mode=mode)

    if max_samples is not None and len(ds) > max_samples:
        # Stratified subsample
        n_real = max_samples // 2
        n_fake = max_samples - n_real
        labels = np.array(ds.label_list)
        real_idx = np.where(labels == 0)[0]
        fake_idx = np.where(labels == 1)[0]
        rng = np.random.RandomState(cfg.get('manualSeed', 1024))
        sel_real = rng.choice(real_idx, min(n_real, len(real_idx)), replace=False)
        sel_fake = rng.choice(fake_idx, min(n_fake, len(fake_idx)), replace=False)
        sel = np.concatenate([sel_real, sel_fake])
        rng.shuffle(sel)
        ds = Subset(ds, sel.tolist())

    loader = DataLoader(
        ds,
        batch_size=cfg['test_batchSize'] if mode == 'test' else cfg['train_batchSize'],
        shuffle=False,
        num_workers=min(int(cfg.get('workers', 4)), 4),
        collate_fn=DeepfakeAbstractBaseDataset.collate_fn,
        drop_last=False,
    )
    return loader


# ═══════════════════════════════════════════════════════════════════════════════
# CLIP feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_clip_features(clip_model, dataloader, device):
    """Extract patch tokens + CLS token from frozen CLIP ViT-L/14.

    Returns:
        dict:
            patch_tokens: [N, 196, 768]   patch tokens (CLS excluded)
            cls_token:    [N, 768]         CLS token
            labels:       [N]              binary labels (0=real, 1=fake)
    """
    clip_model.eval()
    clip_model.to(device)

    all_patches = []
    all_cls = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Extracting CLIP features', leave=False):
        images = batch['image'].to(device)
        labels = batch['label']
        # Map label to binary: 0=real, 1=fake
        labels = torch.where(labels != 0, torch.tensor(1), torch.tensor(0))

        # Handle multi-crop: take the full image (index 0)
        if len(images.shape) == 5:
            images = images[:, 0, :, :, :]  # [B, 3, H, W]

        # Forward through vision model with hidden states
        outputs = clip_model.vision_model(
            images,
            output_hidden_states=True,
        )

        # Handle both tuple output (older transformers) and dict output
        if isinstance(outputs, (tuple, list)):
            last_hidden_state = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs[0]

        # Last hidden state: [B, 197, 768]  (1 CLS + 196 patches)
        # CLS token (index 0)
        cls_tok = last_hidden_state[:, 0, :]           # [B, 768]

        # Patch tokens (indices 1..196)
        patch_tok = last_hidden_state[:, 1:, :]        # [B, 196, 768]

        all_patches.append(patch_tok.cpu())
        all_cls.append(cls_tok.cpu())
        all_labels.append(labels)

    return {
        'patch_tokens': torch.cat(all_patches, dim=0),   # [N, 196, 768]
        'cls_token':    torch.cat(all_cls, dim=0),        # [N, 768]
        'labels':       torch.cat(all_labels, dim=0),     # [N]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PCA fitting & projection
# ═══════════════════════════════════════════════════════════════════════════════

def fit_pca(patch_tokens, n_components=32):
    """Fit PCA on pooled training patch tokens.

    X_PCA ∈ R^{(N_train × 196) × 768}  — flattened patch tokens.

    Returns:
        pca: fitted sklearn PCA object
    """
    N, n_patches, D = patch_tokens.shape
    flat = patch_tokens.reshape(-1, D).numpy()  # [N*196, 768]
    logger.info(f'Fitting PCA({D}→{n_components}) on {flat.shape[0]} patch tokens')
    pca = PCA(n_components=n_components, random_state=1024)
    pca.fit(flat)
    logger.info(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}')
    return pca


def project_pca(patch_tokens, pca):
    """Project patch tokens through fitted PCA.

    Args:
        patch_tokens: [N, 196, 768] torch tensor
        pca: fitted sklearn PCA

    Returns:
        z: [N, 196, d] projected tokens (torch tensor)
    """
    N, n_patches, D = patch_tokens.shape
    flat = patch_tokens.reshape(-1, D).numpy()
    proj = pca.transform(flat)
    return torch.from_numpy(proj.reshape(N, n_patches, -1).astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════════════
# Feature computation — one function per method
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_features(
    train_patches, train_cls,
    test_patches_dict, test_cls_dict,
    pca,
    device='cpu',
    gnn_k=10,
):
    """Compute all feature sets for all data splits.

    Args:
        train_patches:       [N_train, 196, 768]
        train_cls:           [N_train, 768]
        test_patches_dict:   {dataset_name: [N_test, 196, 768]}
        test_cls_dict:       {dataset_name: [N_test, 768]}
        pca:                 fitted sklearn PCA
        device:              torch device
        gnn_k:               top-k for GNN adjacency

    Returns:
        dict: {method_name: {
            'train': np.ndarray,  # [N_train, F]
            'test':  {name: np.ndarray},  # [N_test, F]
        }}
    """
    logger.info('Projecting patch tokens via PCA...')
    z_train = project_pca(train_patches, pca).to(device)
    z_test = {k: project_pca(v, pca).to(device)
              for k, v in test_patches_dict.items()}

    # Move raw tokens to device for GNN
    train_patches_gpu = train_patches.to(device)
    test_patches_gpu = {k: v.to(device) for k, v in test_patches_dict.items()}

    B_train = train_patches.shape[0]

    # ── Analyzer instances ─────────────────────────────────────────────────
    mi_analyzer = MutualInformationAnalyzer()
    spec_analyzer = GraphSpectralAnalyzer()

    # ── GNN model (untrained — we train LR on its frozen random embeddings
    #     to verify the graph structure ITSELF carries information) ──────────
    token_dim = train_patches_gpu.shape[2]  # 1024 for ViT-L/14, 768 for ViT-B/16
    gnn = InfoGraphGNN(in_dim=token_dim, hidden_dim=256, out_dim=128, dropout=0.2)
    gnn.to(device)
    gnn.eval()

    features = {}

    # ═══════════════════════════════════════════════════════════════════════
    # Method 1: CLIP CLS (baseline)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info('[1/10] CLIP CLS baseline')
    features['CLIP CLS'] = {
        'train': train_cls.numpy(),
        'test':  {k: v.numpy() for k, v in test_cls_dict.items()},
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Method 2: Token mean (baseline)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info('[2/10] Token mean baseline')
    features['Token mean'] = {
        'train': train_patches.mean(dim=1).numpy(),
        'test':  {k: v.mean(dim=1).numpy()
                  for k, v in test_patches_dict.items()},
    }

    # ── Compute MI matrices in batches ─────────────────────────────────────
    def compute_mi_batched(z, batch_size=32):
        """Compute MI features batch-wise to manage memory."""
        B = z.shape[0]
        mi_list = []
        for i in range(0, B, batch_size):
            z_b = z[i:i + batch_size]
            M_b = mi_analyzer.compute_mi_matrix(z_b)
            feat_b = mi_analyzer.extract_features(M_b)
            mi_list.append(feat_b)
        # Merge across batches
        merged = {}
        for key in mi_list[0]:
            merged[key] = torch.cat([f[key].cpu() for f in mi_list], dim=0)
        return merged

    def compute_spec_batched(A, batch_size=32):
        """Compute spectral features batch-wise."""
        B = A.shape[0]
        spec_list = []
        for i in range(0, B, batch_size):
            A_b = A[i:i + batch_size]
            L_b = spec_analyzer.compute_laplacian(A_b)
            feat_b = spec_analyzer.extract_features(L_b)
            spec_list.append(feat_b)
        merged = {}
        for key in spec_list[0]:
            merged[key] = torch.cat([f[key].cpu() for f in spec_list], dim=0)
        return merged

    # ── Compute MI features for all splits ─────────────────────────────────
    logger.info('[3-5] Computing MI features...')
    mi_train = compute_mi_batched(z_train)
    mi_test = {k: compute_mi_batched(v) for k, v in z_test.items()}

    # ═══════════════════════════════════════════════════════════════════════
    # Method 3: MI stats
    # ═══════════════════════════════════════════════════════════════════════
    features['MI stats'] = {
        'train': mi_train['mi_stats'].numpy(),
        'test':  {k: v['mi_stats'].numpy() for k, v in mi_test.items()},
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Method 4: MI SVD
    # ═══════════════════════════════════════════════════════════════════════
    features['MI SVD'] = {
        'train': mi_train['mi_svd'].numpy(),
        'test':  {k: v['mi_svd'].numpy() for k, v in mi_test.items()},
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Method 5: MI eigen
    # ═══════════════════════════════════════════════════════════════════════
    features['MI eigen'] = {
        'train': mi_train['mi_eigen'].numpy(),
        'test':  {k: v['mi_eigen'].numpy() for k, v in mi_test.items()},
    }

    # ── Laplacian spectral features ────────────────────────────────────────

    # ═══════════════════════════════════════════════════════════════════════
    # Method 6: Spectrum (MI graph) — PRIMARY
    # ═══════════════════════════════════════════════════════════════════════
    logger.info('[6/10] Spectrum (MI graph)')
    mi_matrices_train = mi_train['mi_matrix'].to(device)
    spec_mi_train = compute_spec_batched(mi_matrices_train)

    spec_mi_test = {}
    for k, v in mi_test.items():
        spec_mi_test[k] = compute_spec_batched(v['mi_matrix'].to(device))

    features['Spectrum (MI)'] = {
        'train': spec_mi_train['f_spec'].numpy(),
        'test':  {k: v['f_spec'].numpy() for k, v in spec_mi_test.items()},
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Method 7: Spectrum (cosine graph) — ABLATION
    # ═══════════════════════════════════════════════════════════════════════
    logger.info('[7/10] Spectrum (cosine graph)')
    cos_train = build_cosine_adjacency(train_patches_gpu)
    spec_cos_train = compute_spec_batched(cos_train)

    spec_cos_test = {}
    for k, v in test_patches_gpu.items():
        cos = build_cosine_adjacency(v)
        spec_cos_test[k] = compute_spec_batched(cos)

    features['Spectrum (cos)'] = {
        'train': spec_cos_train['f_spec'].numpy(),
        'test':  {k: v['f_spec'].numpy() for k, v in spec_cos_test.items()},
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Method 8: GNN embedding
    # ═══════════════════════════════════════════════════════════════════════
    logger.info('[8/10] GNN embedding')
    mi_matrices_train_cpu = mi_matrices_train.cpu()
    A_sparse_train = top_k_adjacency(mi_matrices_train_cpu, k=gnn_k).to(device)

    with torch.no_grad():
        z_gnn_train = gnn(train_patches_gpu, A_sparse_train).cpu().numpy()

    z_gnn_test = {}
    for k, v in test_patches_gpu.items():
        A_test = top_k_adjacency(mi_test[k]['mi_matrix'], k=gnn_k).to(device)
        with torch.no_grad():
            z_gnn_test[k] = gnn(v, A_test).cpu().numpy()

    features['GNN embedding'] = {
        'train': z_gnn_train,
        'test':  z_gnn_test,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Method 9: MI + Spectrum (concat)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info('[9/10] MI stats + Spectrum (concat)')
    features['MI+Spec'] = {
        'train': np.concatenate([
            mi_train['mi_stats'].numpy(),
            spec_mi_train['f_spec'].numpy(),
        ], axis=1),
        'test': {
            k: np.concatenate([
                mi_test[k]['mi_stats'].numpy(),
                spec_mi_test[k]['f_spec'].numpy(),
            ], axis=1)
            for k in mi_test
        },
    }

    # ═══════════════════════════════════════════════════════════════════════
    # Method 10: Spectrum (random graph) — SANITY CHECK
    # ═══════════════════════════════════════════════════════════════════════
    logger.info('[10/10] Spectrum (random graph) — sanity check')
    rand_A = build_random_adjacency(N=196, density=0.05,
                                    device=mi_matrices_train.device)
    rand_A_train = rand_A.expand(B_train, -1, -1)
    spec_rand_train = compute_spec_batched(rand_A_train)

    spec_rand_test = {}
    for k, v in test_patches_gpu.items():
        rand_A_exp = rand_A.expand(v.shape[0], -1, -1)
        spec_rand_test[k] = compute_spec_batched(rand_A_exp)

    features['Spectrum (rand)'] = {
        'train': spec_rand_train['f_spec'].numpy(),
        'test':  {k: v['f_spec'].numpy() for k, v in spec_rand_test.items()},
    }

    return features


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_features(train_feat, train_labels, test_feat, test_labels):
    """Train Logistic Regression (5-fold CV, C tuned) and evaluate.

    Returns:
        dict: {auc, acc, f1, ap, best_C}
    """
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat)
    X_test = scaler.transform(test_feat)

    # Grid search over C
    lr = LogisticRegression(max_iter=2000, solver='lbfgs', random_state=1024)
    params = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    gs = GridSearchCV(lr, params, cv=5, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train, train_labels)

    best = gs.best_estimator_
    probs = best.predict_proba(X_test)[:, 1]
    preds = best.predict(X_test)

    return {
        'auc':    roc_auc_score(test_labels, probs),
        'acc':    accuracy_score(test_labels, preds),
        'f1':     f1_score(test_labels, preds),
        'ap':     average_precision_score(test_labels, probs),
        'best_C': gs.best_params_['C'],
    }


def run_all_evaluations(features, train_labels, test_labels_dict):
    """Evaluate all methods on all test datasets.

    Returns:
        results: {method: {test_name: {auc, acc, f1, ap, best_C}}}
    """
    results = defaultdict(dict)

    for method_name, feat_dict in tqdm(features.items(),
                                       desc='Evaluating methods'):
        X_train = feat_dict['train']
        y_train = train_labels

        for test_name, X_test in feat_dict['test'].items():
            y_test = test_labels_dict[test_name]
            if len(X_test) == 0:
                logger.warning(f'  {method_name} on {test_name}: no data, skip')
                continue
            try:
                metrics = evaluate_features(X_train, y_train, X_test, y_test)
            except Exception as e:
                logger.error(f'  {method_name} on {test_name}: {e}')
                metrics = {'auc': 0, 'acc': 0, 'f1': 0, 'ap': 0, 'best_C': 0}
            results[method_name][test_name] = metrics

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════════

def print_results_table(results, test_datasets):
    """Pretty-print comparison table."""
    print('\n' + '=' * 90)
    print('  Information Graph Theory — Validation Results')
    print('=' * 90)

    for ds_name in test_datasets:
        if ds_name not in next(iter(results.values())):
            continue
        print(f'\n  Test dataset: {ds_name}')
        print(f'  {"Method":<28s} {"AUC":>8s} {"Acc":>8s} {"F1":>8s} {"AP":>8s}')
        print(f'  {"-"*28} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')

        # Sort by AUC descending for readability
        sorted_methods = sorted(
            results.items(),
            key=lambda kv: kv[1].get(ds_name, {}).get('auc', 0),
            reverse=True,
        )

        for method_name, ds_results in sorted_methods:
            m = ds_results.get(ds_name, {})
            auc = m.get('auc', 0)
            acc = m.get('acc', 0)
            f1  = m.get('f1', 0)
            ap  = m.get('ap', 0)

            # Mark baseline methods
            marker = ' (*)' if 'CLIP CLS' in method_name or 'Token mean' in method_name else ''

            print(f'  {method_name+marker:<28s} '
                  f'{auc:>8.4f} {acc:>8.4f} {f1:>8.4f} {ap:>8.4f}')

    print('\n' + '=' * 90)
    print('  (*) = baseline methods')
    print('=' * 90 + '\n')


def save_results(results, args):
    """Save results to JSON."""
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(args.output_dir, f'info_graph_results_{timestamp}.json')

    # Convert to serializable format
    serializable = {}
    for method, ds_dict in results.items():
        serializable[method] = {}
        for ds, metrics in ds_dict.items():
            serializable[method][ds] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f'Results saved to {path}')
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # ── Load config ─────────────────────────────────────────────────────────
    cfg = load_config(args)
    cfg['train_batchSize'] = args.batch_size
    cfg['test_batchSize'] = args.batch_size

    # ── Load frozen CLIP ────────────────────────────────────────────────────
    logger.info(f'Loading CLIP from {args.clip_model_path}')
    clip_model = CLIPModel.from_pretrained(args.clip_model_path)
    for p in clip_model.vision_model.parameters():
        p.requires_grad = False

    # ── Build dataloaders ───────────────────────────────────────────────────
    logger.info(f'Train dataset: {args.train_dataset}')
    train_loader = build_dataloader(
        cfg, 'train', args.train_dataset,
        max_samples=args.max_train_samples,
    )

    test_loaders = {}
    for ds_name in args.test_datasets:
        logger.info(f'Test dataset: {ds_name}')
        test_loaders[ds_name] = build_dataloader(
            cfg, 'test', ds_name,
            max_samples=args.max_test_samples,
        )

    # ── Extract CLIP features ───────────────────────────────────────────────
    logger.info('Extracting CLIP features from train set...')
    train_feat = extract_clip_features(clip_model, train_loader, device)
    logger.info(f'Train: {train_feat["patch_tokens"].shape[0]} samples')

    test_feat = {}
    for ds_name, loader in test_loaders.items():
        logger.info(f'Extracting CLIP features from {ds_name}...')
        test_feat[ds_name] = extract_clip_features(clip_model, loader, device)
        logger.info(f'  {ds_name}: {test_feat[ds_name]["patch_tokens"].shape[0]} samples')

    # ── Fit PCA on pooled train patches ────────────────────────────────────
    pca = fit_pca(train_feat['patch_tokens'], n_components=args.pca_dim)

    # ── Compute all features ───────────────────────────────────────────────
    test_patches = {k: v['patch_tokens'] for k, v in test_feat.items()}
    test_cls = {k: v['cls_token'] for k, v in test_feat.items()}

    all_features = compute_all_features(
        train_patches=train_feat['patch_tokens'],
        train_cls=train_feat['cls_token'],
        test_patches_dict=test_patches,
        test_cls_dict=test_cls,
        pca=pca,
        device=device,
        gnn_k=args.gnn_k,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────
    train_labels = train_feat['labels'].numpy()
    test_labels_dict = {k: v['labels'].numpy() for k, v in test_feat.items()}

    results = run_all_evaluations(all_features, train_labels, test_labels_dict)

    # ── Print & save ───────────────────────────────────────────────────────
    print_results_table(results, args.test_datasets)
    save_results(results, args)

    logger.info('Done.')


if __name__ == '__main__':
    main()

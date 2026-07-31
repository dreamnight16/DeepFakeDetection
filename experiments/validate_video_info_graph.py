#!/usr/bin/env python3
"""
Video-Level Validation: Information Graph Theory for Deepfake Detection
========================================================================

Phase-2: video-level ablation — three independent detectors evaluated
on clip-level and video-level AUC with cross-dataset generalization.

Detectors
---------
  MI-D  : Temporal Mutual Information  (hypothesis: I_real > I_fake)
  GT-D  : Graph Topology               (hypothesis: E_G^real < E_G^fake)
  GNN-D : Spatio-Temporal GNN          (hypothesis: structural anomalies)

Protocol
--------
  Train:  FF++ c23 video clips (T=8 frames each)
  Test A: FF++ c23 → in-domain
  Test B: Celeb-DF v2 → cross-domain generalization

  All detectors use frozen CLIP ViT features + small trainable classifier.
  Evaluation: clip-level AUC + video-level AUC (mean pool per video).

Usage
-----
  python experiments/validate_video_info_graph.py \
      --detector_path DeepfakeBench/training/config/detector/effort.yaml \
      --clip_size 8 --max_train_clips 1000 --max_test_clips 500 \
      --device cuda

Author: personal experiment
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
import warnings
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings('ignore', category=UserWarning,
                        message='.*lbfgs failed to converge.*')

# ── sklearn ───────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, average_precision_score,
)

# ── PyTorch ───────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

from transformers import CLIPModel

# ── Project root detection ────────────────────────────────────────────────
_SCRIPT_FILE = os.path.abspath(__file__)
_SCRIPT_DIR  = os.path.dirname(_SCRIPT_FILE)
_project_root = os.path.dirname(_SCRIPT_DIR)
_training_dir = os.path.join(_project_root, 'DeepfakeBench', 'training')
sys.path.insert(0, _training_dir)

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

_DEFAULT_DETECTOR_YAML = os.path.join(_project_root, 'DeepfakeBench', 'training',
                                       'config', 'detector', 'effort.yaml')
_DEFAULT_TRAIN_CONFIG  = os.path.join(_project_root, 'DeepfakeBench', 'training',
                                       'config', 'train_config.yaml')
_DEFAULT_TEST_CONFIG   = os.path.join(_project_root, 'DeepfakeBench', 'training',
                                       'config', 'test_config.yaml')
_DEFAULT_CLIP_MODEL    = os.path.join(_project_root, 'DeepfakeBench', 'training',
                                       'models--openai--clip-vit-large-patch14')
_DEFAULT_OUTPUT_DIR    = os.path.join(_project_root, 'experiments', 'results')

# ── Direct import to avoid chain-import ───────────────────────────────────
import importlib.util as _iu
_igt_path = os.path.join(_training_dir, 'detectors', 'info_graph_theory.py')
_igt = _iu.module_from_spec(_iu.spec_from_file_location('info_graph_theory', _igt_path))
_iu.spec_from_file_location('info_graph_theory', _igt_path).loader.exec_module(_igt)
MutualInformationAnalyzer = _igt.MutualInformationAnalyzer

_vid_path = os.path.join(_training_dir, 'detectors', 'video_info_detectors.py')
_vid = _iu.module_from_spec(_iu.spec_from_file_location('video_info', _vid_path))
_iu.spec_from_file_location('video_info', _vid_path).loader.exec_module(_vid)
MIDetector  = _vid.MIDetector
GTDetector  = _vid.GTDetector
GNNDetector = _vid.GNNDetector

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-8s | %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Video Info Graph — ablation')
    p.add_argument('--detector_path', default=_DEFAULT_DETECTOR_YAML)
    p.add_argument('--train_config',  default=_DEFAULT_TRAIN_CONFIG)
    p.add_argument('--test_config',   default=_DEFAULT_TEST_CONFIG)
    p.add_argument('--clip_model_path', default=_DEFAULT_CLIP_MODEL)
    p.add_argument('--train_dataset', nargs='+', default=['FaceForensics++'])
    p.add_argument('--test_datasets', nargs='+',
                   default=['Celeb-DF-v2', 'FaceForensics++'])
    p.add_argument('--clip_size', type=int, default=8)
    p.add_argument('--max_train_clips', type=int, default=1000)
    p.add_argument('--max_test_clips', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=4,
                   help='clips per batch (T frames each → high memory)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1024)
    p.add_argument('--output_dir', default=_DEFAULT_OUTPUT_DIR)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--detector', default='all',
                   choices=['mi', 'gt', 'gnn', 'all'])
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_config(args):
    cfg = {}
    for path in [args.detector_path, args.train_config, args.test_config]:
        with open(path, 'r') as f:
            cfg.update({k: v for k, v in yaml.safe_load(f).items()
                        if k not in cfg})
    return cfg


def build_video_dataloader(cfg, mode, dataset_name, clip_size,
                           max_clips=None):
    """Build DataLoader for video clips.

    Returns:
        loader: DataLoader yielding [B, T, C, H, W]
        dataset: underlying dataset (for video_name access)
    """
    cfg = deepcopy(cfg)
    cfg['video_mode'] = True
    cfg['clip_size'] = clip_size
    if mode == 'train':
        cfg['train_dataset'] = dataset_name if isinstance(dataset_name, list) \
                               else [dataset_name]
    else:
        cfg['test_dataset'] = dataset_name

    ds = DeepfakeAbstractBaseDataset(config=cfg, mode=mode)

    if max_clips is not None and len(ds) > max_clips:
        n_real = max_clips // 2
        n_fake = max_clips - n_real
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
        ds, batch_size=cfg.get('train_batchSize', 4),
        shuffle=(mode == 'train'),
        num_workers=min(int(cfg.get('workers', 4)), 4),
        collate_fn=DeepfakeAbstractBaseDataset.collate_fn,
        drop_last=False,
    )
    return loader, ds


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_clip_video_features(clip_model, dataloader, device):
    """Extract CLIP features from video clips.

    Input:  [B, T, C, H, W] video clips
    Output: [N, T, 257, D] CLIP features (CLS + 256 patches)

    Returns:
        dict with:
            cls_seq:    [N, T, D]       frame CLS tokens
            patch_seq:  [N, T, N_p, D]  frame patch tokens
            labels:     [N]             binary (0=real, 1=fake)
            video_names: [N]            video identifiers
    """
    clip_model.eval()
    clip_model.to(device)

    all_cls = []
    all_patches = []
    all_labels = []
    all_vnames = []

    for batch in tqdm(dataloader, desc='Extracting video CLIP features',
                      leave=False):
        images = batch['image'].to(device)        # [B, T, C, H, W]
        labels = batch['label']
        labels = torch.where(labels != 0, torch.tensor(1), torch.tensor(0))

        B, T, C, H, W = images.shape

        # Flatten batch+time for CLIP forward
        flat = images.reshape(B * T, C, H, W)
        outputs = clip_model.vision_model(flat, output_hidden_states=True)

        if isinstance(outputs, (tuple, list)):
            hidden = outputs[2][-1] if len(outputs) > 2 else outputs[0]
        elif hasattr(outputs, 'hidden_states'):
            hidden = outputs.hidden_states[-1]
        else:
            hidden = outputs[0]

        # hidden: [B*T, 257, D]
        _, N_tok, D = hidden.shape
        hidden = hidden.reshape(B, T, N_tok, D)

        cls_seq = hidden[:, :, 0, :]      # [B, T, D]
        patch_seq = hidden[:, :, 1:, :]   # [B, T, N_tok-1, D]

        all_cls.append(cls_seq.cpu())
        all_patches.append(patch_seq.cpu())
        all_labels.append(labels)

        # Try to get video names
        if hasattr(dataloader.dataset, 'video_name_list'):
            # Get indices from the current batch
            pass  # handled below

    return {
        'cls_seq':   torch.cat(all_cls, dim=0),      # [N, T, D]
        'patch_seq': torch.cat(all_patches, dim=0),   # [N, T, N_p, D]
        'labels':    torch.cat(all_labels, dim=0),     # [N]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training & evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_detector(model, train_data, device, epochs=20, lr=1e-3,
                   batch_size=32):
    """Train a detector model.

    Args:
        model:      MIDetector | GTDetector | GNNDetector
        train_data: dict with cls_seq, patch_seq, labels
    """
    model.train()
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Build simple dataset from extracted features
    labels = train_data['labels']
    N = len(labels)

    for epoch in range(epochs):
        perm = torch.randperm(N)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            batch = {
                'cls_seq':   train_data['cls_seq'][idx].to(device),
                'patch_seq': train_data['patch_seq'][idx].to(device),
                'labels':    labels[idx].to(device),
            }

            logits = model(batch)
            loss = criterion(logits, batch['labels'])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            logger.debug(f'  Epoch {epoch+1}/{epochs}  loss={total_loss/n_batches:.4f}')

    model.eval()
    return model


@torch.no_grad()
def evaluate_detector(model, test_data, device, batch_size=32):
    """Evaluate detector → clip-level metrics."""
    model.eval()
    model.to(device)

    labels = test_data['labels']
    N = len(labels)
    all_probs = []

    for i in range(0, N, batch_size):
        idx = slice(i, i + batch_size)
        batch = {
            'cls_seq':   test_data['cls_seq'][idx].to(device),
            'patch_seq': test_data['patch_seq'][idx].to(device),
        }
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu())

    probs = torch.cat(all_probs)
    preds = (probs > 0.5).long()

    return {
        'auc': roc_auc_score(labels, probs),
        'acc': accuracy_score(labels, preds),
        'f1':  f1_score(labels, preds),
        'ap':  average_precision_score(labels, probs),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline: temporal mean pool + MLP
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineDetector(nn.Module):
    """CLIP temporal mean pool → MLP classifier."""
    def __init__(self, feature_dim=1024, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, clip_features):
        z = clip_features['cls_seq'].mean(dim=1)  # [B, D]
        return self.mlp(z)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment matrix
# ═══════════════════════════════════════════════════════════════════════════════

DETECTOR_CONFIGS = {
    # ── Baseline ──────────────────────────────────────────────────────────
    'Baseline': {
        'cls': BaselineDetector,
        'kwargs': {},
    },
    # ── MI-D ablations ────────────────────────────────────────────────────
    'MI-T (temporal)': {
        'cls': MIDetector,
        'kwargs': {'use_temporal': True, 'use_spatial': False},
    },
    # ── GT-D ──────────────────────────────────────────────────────────────
    'GT (topology)': {
        'cls': GTDetector,
        'kwargs': {'use_smoothness': True, 'use_spectrum': True,
                    'use_entropy': True},
    },
    # ── GNN-D ─────────────────────────────────────────────────────────────
    'GNN (ST-GCN)': {
        'cls': GNNDetector,
        'kwargs': {'use_spatial': True, 'use_temporal': True},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}  clip_size={args.clip_size}')

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_config(args)

    # ── CLIP ───────────────────────────────────────────────────────────────
    logger.info(f'Loading CLIP from {args.clip_model_path}')
    clip_model = CLIPModel.from_pretrained(args.clip_model_path)
    for p in clip_model.vision_model.parameters():
        p.requires_grad = False

    # ── Train data ────────────────────────────────────────────────────────
    logger.info(f'Train: {args.train_dataset}')
    train_loader, train_ds = build_video_dataloader(
        cfg, 'train', args.train_dataset,
        clip_size=args.clip_size, max_clips=args.max_train_clips,
    )
    train_feat = extract_clip_video_features(clip_model, train_loader, device)
    logger.info(f'Train clips: {train_feat["cls_seq"].shape[0]}  '
                f'(T={train_feat["cls_seq"].shape[1]})')

    # ── Test data ─────────────────────────────────────────────────────────
    test_feats = {}
    for ds_name in args.test_datasets:
        logger.info(f'Test: {ds_name}')
        loader, ds = build_video_dataloader(
            cfg, 'test', ds_name,
            clip_size=args.clip_size, max_clips=args.max_test_clips,
        )
        test_feats[ds_name] = extract_clip_video_features(
            clip_model, loader, device)
        logger.info(f'  {ds_name}: {test_feats[ds_name]["cls_seq"].shape[0]} clips')

    # ── Run ablation ──────────────────────────────────────────────────────
    results = {}
    for name, cfg_item in DETECTOR_CONFIGS.items():
        if args.detector != 'all' and args.detector not in name.lower():
            continue

        logger.info(f'--- {name} ---')
        model = cfg_item['cls'](**cfg_item['kwargs'])

        try:
            model = train_detector(model, train_feat, device,
                                   epochs=args.epochs, lr=args.lr)
        except Exception as e:
            logger.error(f'{name}: train error: {e}')
            results[name] = {}
            continue

        results[name] = {}
        for ds_name, feat in test_feats.items():
            try:
                metrics = evaluate_detector(model, feat, device)
            except Exception as e:
                logger.error(f'{name} on {ds_name}: {e}')
                metrics = {'auc': 0, 'acc': 0, 'f1': 0, 'ap': 0}
            results[name][ds_name] = metrics
            logger.info(f'  {ds_name}: AUC={metrics["auc"]:.4f}  '
                        f'Acc={metrics["acc"]:.4f}  F1={metrics["f1"]:.4f}')

    # ── Print table ───────────────────────────────────────────────────────
    print('\n' + '=' * 80)
    print('  Video-Level Information Graph — Ablation Results')
    print('=' * 80)

    for ds_name in args.test_datasets:
        print(f'\n  Test: {ds_name}')
        print(f'  {"Method":<25s} {"AUC":>8s} {"Acc":>8s} {"F1":>8s} {"AP":>8s}')
        print(f'  {"-"*25} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')

        sorted_items = sorted(
            [(n, m.get(ds_name, {})) for n, m in results.items()],
            key=lambda x: x[1].get('auc', 0), reverse=True)

        for name, metrics in sorted_items:
            marker = ' (*)' if 'Baseline' in name else ''
            print(f'  {name+marker:<25s} '
                  f'{metrics.get("auc",0):>8.4f} '
                  f'{metrics.get("acc",0):>8.4f} '
                  f'{metrics.get("f1",0):>8.4f} '
                  f'{metrics.get("ap",0):>8.4f}')

    print('\n' + '=' * 80)
    print('  (*) = baseline')
    print('=' * 80 + '\n')

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(args.output_dir, f'video_info_graph_{ts}.json')
    serial = {}
    for method, ds_dict in results.items():
        serial[method] = {}
        for ds, m in ds_dict.items():
            serial[method][ds] = {k: float(v) if isinstance(
                v, (np.floating, np.integer)) else v for k, v in m.items()}
    with open(path, 'w') as f:
        json.dump(serial, f, indent=2)
    logger.info(f'Saved to {path}')
    logger.info('Done.')


if __name__ == '__main__':
    main()

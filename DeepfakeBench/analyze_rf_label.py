#!/usr/bin/env python3
"""
analyze_rf_label.py — 金字塔 RF 混合样本的能量接地软标签 e_f 分布

研究问题(γ=1.0 时 RF 软标签 y = e_f = fake 残差能量占比):
  金字塔 RF 混合样本的软标签是否整体 < 0.5(偏真)?
  这直接解释 G1/G3 的 label 消融:为什么硬标签置 0(real) 比置 1(fake) 好。

原理:
  RF 样本 = real 结构锚(G_K 全 real) + 部分 fake 的 Laplacian 残差。
  软标签 y = 1 - (1 - e_f)^γ,其中
    e_f = Σ ω_k·q²·E_f / (Σ ω_k[(1-q)²·E_r + q²·E_f] + ε)
  γ=1.0 时 y = e_f,即"fake 残差能量占混合后总残差能量的比例"。

结论判读:
  - e_f 整体 < 0.5 → 软标签偏真 → 硬标签 0 是更好的近似(印证 label0 > label1)
  - e_f 整体 > 0.5 → 软标签偏假,但 label0 仍赢 → 说明能量标签"漏掉了 real base",
    正确标签应更偏真,能量标签的方向本身有问题(更有意思的发现)

用法(在训练服务器上,从 DeepfakeBench/ 目录):
  python3 analyze_rf_label.py --n_batches 50 --alpha 5.0 --gamma 1.0 \
      --num_levels 3 --out rf_label_hist.png
"""
import argparse
import os
import sys

import numpy as np

_current = os.path.dirname(os.path.abspath(__file__))
_deepfake = _current
_training = os.path.join(_deepfake, 'training')
_experiments = os.path.join(_deepfake, 'experiments')
sys.path.insert(0, _training)
sys.path.insert(0, _deepfake)
sys.path.insert(0, _experiments)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from trainer.trainer_v2 import lap_pyramid_mixup
from experiment_utils import build_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_batches', type=int, default=50,
                    help='number of train batches to run mixup on')
    ap.add_argument('--alpha', type=float, default=5.0)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--num_levels', type=int, default=3)
    ap.add_argument('--train_dataset', type=str, default='FaceForensics++')
    ap.add_argument('--out', type=str, default='rf_label_hist.png')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')

    config = build_config(
        pyramid_mode='lap_pyramid',
        use_mixup=True,
        mixup_alpha=args.alpha,
        mixup_gamma=args.gamma,
        lap_num_levels=args.num_levels,
        sampler_real_ratio=0.30,
        train_dataset=args.train_dataset,
        test_dataset='Celeb-DF-v2',
        n_epochs=1,
        for_training=True,
    )
    config['use_data_augmentation'] = False

    ds = DeepfakeAbstractBaseDataset(config=config, mode='train')
    loader = DataLoader(
        ds, batch_size=int(config.get('test_batchSize', 32)),
        shuffle=True, num_workers=int(config.get('workers', 4)),
        collate_fn=ds.collate_fn,
    )

    e_f_list = []
    n_rf_total = 0
    for i, data_dict in enumerate(tqdm(loader, desc='mixup')):
        if i >= args.n_batches:
            break
        x = data_dict['image'].to(device)
        y = data_dict['label'].to(device)
        y_bin = torch.where(y != 0, 1, 0)  # binarize, as the trainer does

        mixed_x, mixed_y, mixed_label, loss_mask = lap_pyramid_mixup(
            x, y_bin, alpha=args.alpha, gamma=args.gamma,
            num_levels=args.num_levels)

        # RF samples: soft label strictly interior (0, 1)
        rf = (mixed_y > 1e-6) & (mixed_y < 1.0 - 1e-6)
        n_rf = int(rf.sum())
        if n_rf == 0:
            continue
        n_rf_total += n_rf
        y_rf = mixed_y[rf].cpu().numpy()
        # back out e_f from y = 1 - (1 - e_f)^gamma
        e_f = 1.0 - (1.0 - y_rf) ** (1.0 / args.gamma)
        e_f_list.append(e_f)

    if n_rf_total == 0:
        print('ERROR: no RF pairs found — check dataset / label_dict.')
        return

    e_f = np.concatenate(e_f_list)
    print(f'\n{"=" * 62}')
    print(f'  RF soft-label (e_f) distribution  —  n={n_rf_total}')
    print(f'{"=" * 62}')
    print(f'  mean          = {e_f.mean():.4f}')
    print(f'  median        = {np.median(e_f):.4f}')
    print(f'  std           = {e_f.std():.4f}')
    print(f'  P(e_f < 0.5)  = {(e_f < 0.5).mean():.4f}   ← 偏真占比')
    print(f'  P(e_f < 0.25) = {(e_f < 0.25).mean():.4f}')
    print(f'  P(e_f > 0.75) = {(e_f > 0.75).mean():.4f}')

    real_biased = e_f.mean() < 0.5
    verdict = ('REAL-biased → 硬标签 0 是更好近似(label0 > label1 成立)'
               if real_biased else
               'FAKE-biased → 能量标签漏了 real base,方向可能反了,需进一步查')
    print(f'\n  结论: e_f 整体 {"< 0.5" if real_biased else "> 0.5"}')
    print(f'        → {verdict}')

    np.save('rf_e_f.npy', e_f)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    plt.hist(e_f, bins=50, range=(0, 1), color='tab:blue',
             alpha=0.75, edgecolor='white')
    plt.axvline(0.5, color='red', linestyle='--', label='balanced 0.5')
    plt.axvline(e_f.mean(), color='orange', linestyle='-',
                label=f'mean={e_f.mean():.3f}')
    plt.xlabel('e_f  (fake residual energy fraction)')
    plt.ylabel('count')
    plt.title(f'RF soft-label dist (gamma={args.gamma}, alpha={args.alpha}, '
              f'n={n_rf_total})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'\n  saved: {args.out}, rf_e_f.npy')


if __name__ == '__main__':
    main()

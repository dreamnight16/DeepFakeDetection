#!/usr/bin/env python3
"""
analyze_rf_label.py — 金字塔 RF 混合样本的能量接地软标签 e_f 分布

研究问题:
  金字塔 RF 混合样本的软标签是否整体 < 0.5(偏真)?
  这直接解释 G1/G3 的 label 消融:为什么硬标签置 0(real) 比置 1(fake) 好。

原理:
  RF 样本 = real 结构锚(G_K 全 real) + 部分 fake 的 Laplacian 残差。
  软标签 y = 1 - (1 - e_f)^γ,其中
    e_f = Σ ω_k·q²·E_f / (Σ ω_k[(1-q)²·E_r + q²·E_f] + ε)

  本脚本额外计算两个版本(γ=1 时 y = e_f,直接比较 e_f):
    e_f_residual —— 训练现用版:只对 Laplacian 残差求和(漏掉 G_K)。
                    与 trainer_v2.lap_pyramid_mixup 完全一致,用于对照。
    e_f_full     —— 把 G_K(全真粗结构)作为 real 项加入分母,
                    且用"每像素能量密度"(÷每样本像素数)消除跨层分辨率差异。
                    这能定量回答"label0 赢是不是因为真结构主导"。

判读:
  - e_f_full << e_f_residual(≈0.5) → 真结构能量主导 → 混合样本本质偏真
    → 硬标签 0 是更好近似(定量印证 label0 > label1)。
  - 两者接近 → 粗结构能量不占优,label0 赢另有原因,需再查。

用法(在训练服务器上,从 DeepfakeBench/ 目录):
  python3 analyze_rf_label.py --n_batches 50 --alpha 5.0 --gamma 1.0 \
      --num_levels 3 --out rf_label_hist.png
"""
import argparse
import os
import sys
from collections import defaultdict

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
from trainer.trainer_v2 import build_gaussian_pyramid, build_laplacian_pyramid
from experiment_utils import build_config


def _rf_energy_breakdown(x, y, alpha, num_levels, omega=None):
    """复刻 lap_pyramid_mixup 的 RF 分支,返回逐样本能量诊断(不重建图像)。

    与 trainer_v2.lap_pyramid_mixup 保持同一配对与公式,额外输出:
      E_G:  G_K(全真粗结构)的逐样本总能量 ‖G_K‖²
      N_G:  每样本粗结构元素数(通道×像素,用于求能量密度)
      E_r/E_f: 各层逐样本残差总能量
      N_k:  各层每样本元素数
      omega: 残差版用到的层级权重
      q_val: 本批 fake 注入强度 q = 1-λ

    Returns: dict 或 None(本批无 RF 对时)。
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    q_val = 1.0 - lam

    index = torch.randperm(x.size(0), device=x.device)
    y_a = y.float()
    y_b = y[index].float()

    rf_mask = (y_a == 0) & (y_b == 1)
    fr_mask = (y_a == 1) & (y_b == 0)
    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])
    fake_pos = torch.cat([index[rf_idx], fr_idx])
    n_rf = len(real_pos)

    if n_rf == 0:
        return None

    x_r = x[real_pos]
    x_f = x[fake_pos]

    gpyr_r = build_gaussian_pyramid(x_r, num_levels)
    gpyr_f = build_gaussian_pyramid(x_f, num_levels)
    G_K = gpyr_r[-1]

    lap_r = build_laplacian_pyramid(gpyr_r)
    lap_f = build_laplacian_pyramid(gpyr_f)

    # 与训练 lap_pyramid_mixup 相同的默认递减权重
    if omega is None:
        omega = [float(num_levels - i) for i in range(num_levels)]
        s = sum(omega)
        omega = [w / s for w in omega]

    E_r, E_f, N_k = [], [], []
    for k in range(num_levels):
        L_r = lap_r[k]
        L_f = lap_f[k]
        E_r.append((L_r ** 2).reshape(n_rf, -1).sum(dim=1))
        E_f.append((L_f ** 2).reshape(n_rf, -1).sum(dim=1))
        N_k.append(L_r.numel() // n_rf)  # 每样本 通道×像素 元素数

    E_G = (G_K ** 2).reshape(n_rf, -1).sum(dim=1)
    N_G = G_K.numel() // n_rf

    return dict(q_val=q_val, E_r=E_r, E_f=E_f, E_G=E_G,
                N_k=N_k, N_G=N_G, omega=omega, n_rf=n_rf)


def _compute_e_f(bd, epsilon=1e-8):
    """由能量诊断计算 residual-only 与 full(含 G_K)两个 e_f。

    Returns: (e_f_residual, e_f_full) —— 均为逐样本 [n_rf] torch.Tensor。
    """
    q = bd['q_val']
    E_r, E_f = bd['E_r'], bd['E_f']
    E_G, N_G = bd['E_G'], bd['N_G']
    N_k, omega = bd['N_k'], bd['omega']
    K = len(E_r)

    # ── residual-only: 与训练 lap_pyramid_mixup 完全一致 ──
    num = sum(omega[k] * (q ** 2) * E_f[k] for k in range(K))
    den = sum(omega[k] * ((1.0 - q) ** 2 * E_r[k] + (q ** 2) * E_f[k])
              for k in range(K))
    e_f_residual = num / (den + epsilon)

    # ── full: 加入 G_K(全真)为 real 项,统一用每像素能量密度(scale-invariant)──
    # 密度 = 总能量 / 每样本元素数,消除各层分辨率差异;权重统一取 1。
    dG = E_G / N_G
    num_full = sum((q ** 2) * (E_f[k] / N_k[k]) for k in range(K))
    den_full = dG + sum((1.0 - q) ** 2 * (E_r[k] / N_k[k])
                        + (q ** 2) * (E_f[k] / N_k[k]) for k in range(K))
    e_f_full = num_full / (den_full + epsilon)

    return e_f_residual, e_f_full


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

    e_f_res_list = []
    e_f_full_list = []
    level_stats = defaultdict(lambda: {'E_r': [], 'E_f': [], 'd_r': [], 'd_f': []})
    base_stats = {'E_G': [], 'd_G': []}
    n_rf_total = 0

    for i, data_dict in enumerate(tqdm(loader, desc='mixup')):
        if i >= args.n_batches:
            break
        x = data_dict['image'].to(device)
        y = data_dict['label'].to(device)
        y_bin = torch.where(y != 0, 1, 0)  # binarize, as the trainer does

        bd = _rf_energy_breakdown(x, y_bin, alpha=args.alpha,
                                  num_levels=args.num_levels)
        if bd is None:
            continue
        n_rf_total += bd['n_rf']
        e_f_res, e_f_full = _compute_e_f(bd)
        e_f_res_list.append(e_f_res.cpu().numpy())
        e_f_full_list.append(e_f_full.cpu().numpy())

        for k in range(args.num_levels):
            E_r = bd['E_r'][k].cpu().numpy()
            E_f = bd['E_f'][k].cpu().numpy()
            N_k = bd['N_k'][k]
            level_stats[k]['E_r'].append(E_r)
            level_stats[k]['E_f'].append(E_f)
            level_stats[k]['d_r'].append(E_r / N_k)
            level_stats[k]['d_f'].append(E_f / N_k)
        base_stats['E_G'].append(bd['E_G'].cpu().numpy())
        base_stats['d_G'].append((bd['E_G'] / bd['N_G']).cpu().numpy())

    if n_rf_total == 0:
        print('ERROR: no RF pairs found — check dataset / label_dict.')
        return

    e_f_res = np.concatenate(e_f_res_list)
    e_f_full = np.concatenate(e_f_full_list)

    # ── 能量量级(每样本均值) ──────────────────────────────────────────────
    d_G = np.concatenate(base_stats['d_G']).mean()
    E_G = np.concatenate(base_stats['E_G']).mean()
    sum_d_r = 0.0
    level_lines = []
    for k in range(args.num_levels):
        E_r = np.concatenate(level_stats[k]['E_r']).mean()
        E_f = np.concatenate(level_stats[k]['E_f']).mean()
        d_r = np.concatenate(level_stats[k]['d_r']).mean()
        d_f = np.concatenate(level_stats[k]['d_f']).mean()
        sum_d_r += d_r
        level_lines.append(
            f'      L{k}: E_r={E_r:.3e}  E_f={E_f:.3e}   '
            f'密度 d_r={d_r:.4f}  d_f={d_f:.4f}')

    print(f'\n{"=" * 62}')
    print(f'  RF energy soft-label (e_f)  —  n={n_rf_total}')
    print(f'{"=" * 62}')

    print(f'  [1] residual-only e_f (训练现用, 漏 G_K)')
    print(f'      mean          = {e_f_res.mean():.4f}')
    print(f'      median        = {np.median(e_f_res):.4f}')
    print(f'      std           = {e_f_res.std():.4f}')
    print(f'      P(e_f < 0.5)  = {(e_f_res < 0.5).mean():.4f}')

    print(f'  [2] full e_f (含 G_K, 每像素密度)')
    print(f'      mean          = {e_f_full.mean():.4f}')
    print(f'      median        = {np.median(e_f_full):.4f}')
    print(f'      std           = {e_f_full.std():.4f}')
    print(f'      P(e_f < 0.5)  = {(e_f_full < 0.5).mean():.4f}')

    print(f'  --- 能量量级 (每样本均值) ---')
    print(f'      G_K 粗结构: 总能量={E_G:.3e}  密度={d_G:.4f}')
    for line in level_lines:
        print(line)
    ratio = d_G / (sum_d_r + 1e-12)
    print(f'      密度比  d_G / Σ_k d_r = {ratio:.3f}   '
          f'({"粗结构主导" if ratio > 1.0 else "残差主导"})')

    gap = e_f_res.mean() - e_f_full.mean()
    if e_f_full.mean() < 0.5 and gap > 0.1:
        verdict = (f'真结构能量主导:full e_f 比 residual 低 {gap:.3f},'
                   f'混合样本本质偏真 → 硬标签 0 是更好近似(印证 label0 > label1)')
    elif e_f_full.mean() < 0.5:
        verdict = (f'full e_f 略偏真(低 {gap:.3f}),方向与 label0>label1 一致,'
                   f'但幅度不大,建议看直方图再下结论')
    else:
        verdict = 'full e_f 仍 ≥ 0.5:粗结构不占优,label0 赢另有原因,需再查'
    print(f'\n  结论: {verdict}')

    np.save('rf_e_f.npy', e_f_res)          # 保持旧文件名(residual 版)
    np.save('rf_e_f_full.npy', e_f_full)    # 新增 full 版

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    plt.hist(e_f_res, bins=50, range=(0, 1), color='tab:blue',
             alpha=0.6, edgecolor='white', label=f'residual (mean={e_f_res.mean():.3f})')
    plt.hist(e_f_full, bins=50, range=(0, 1), color='tab:green',
             alpha=0.6, edgecolor='white', label=f'full+G_K (mean={e_f_full.mean():.3f})')
    plt.axvline(0.5, color='red', linestyle='--', label='balanced 0.5')
    plt.axvline(e_f_res.mean(), color='tab:blue', linestyle='-', alpha=0.7)
    plt.axvline(e_f_full.mean(), color='tab:green', linestyle='-', alpha=0.7)
    plt.xlabel('e_f  (fake residual energy fraction)')
    plt.ylabel('count')
    plt.title(f'RF soft-label dist (gamma={args.gamma}, alpha={args.alpha}, '
              f'n={n_rf_total})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'\n  saved: {args.out}, rf_e_f.npy, rf_e_f_full.npy')


if __name__ == '__main__':
    main()

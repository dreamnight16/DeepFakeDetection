"""
Ordinary pixel-space mixup for ALL pair types (RR/FF/RF), with cross-class
(RF/FR) soft label forced to 0 (real).

Ablation purpose (G7): isolate whether "label RF as real" (which won in
G3_label0_full) still holds when the mixing itself is ordinary pixel-space
rather than Laplacian pyramid.

    G3_label0_full : RF pyramid mixup  + label 0
    G7 pixel_label0: RF pixel mixup    + label 0   ← this file
    G6_baseline    : no mixup

Label rule (standard mixup, then override cross-class to 0):
    RR (real+real)   -> 0
    FF (fake+fake)   -> 1
    RF/FR (cross)    -> 0   (instead of standard 1-lam / lam)
"""
import numpy as np
import torch


def pixel_mixup_label0(x, y, alpha=1.0, gamma=5.0):
    # gamma unused (kept for API consistency with the trainer dispatch).
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)

    index = torch.randperm(x.size(0), device=x.device)
    y_a = y.float()
    y_b = y[index].float()

    # Pixel-space mixup for ALL pairs (RR / FF / RF / FR).
    mixed_x = lam_t * x + (1.0 - lam_t) * x[index]

    # Label: only fake+fake -> 1; real+real and cross-class -> 0.
    ff_mask = (y_a == 1) & (y_b == 1)
    mixed_y = torch.where(ff_mask, torch.ones_like(y_a), torch.zeros_like(y_a))

    mixed_label = mixed_y.long()
    loss_mask = torch.ones_like(mixed_y)
    return mixed_x, mixed_y, mixed_label, loss_mask

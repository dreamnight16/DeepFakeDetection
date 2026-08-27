"""G10: HiMix-style RF pixel mixup with hard fake labels (append mode).

The base batch is untouched — original images keep their hard labels and the
normal CE supervision. For each real image in the batch, one real+fake pixel
mixture is generated and appended to the batch with hard label fake (=1):

    X_mix = (1−λ)·X_real + λ·X_fake,   λ ~ Beta(α,α) per pair  (α=0.1 → bimodal)

This mirrors the MDA module of HiMix (arXiv:2604.27903): near-real mixtures
(λ→0) become hard "fake" samples that expand the fake decision region toward
the real manifold, and near-fake mixtures (λ→1) act as mild fake augmentation.
"""
import numpy as np
import torch


def build_pixel_rf_hardfake(x, y, alpha=0.1):
    """Build n_real RF pixel-mixes (one per real anchor), hard-labeled fake.

    Args:
        x:     [N, C, H, W] batch images
        y:     [N] labels (0=real, 1=fake)
        alpha: Beta(α,α) parameter for per-pair λ (0.1 → bimodal)

    Returns:
        dict with image [n_real, C, H, W], label (all 1, long),
        label_soft (all 1.0, float) — or None when the batch contains no
        real or no fake image.
    """
    real_mask = (y == 0)
    fake_mask = (y == 1)
    x_r = x[real_mask]
    x_f = x[fake_mask]
    n_real = x_r.size(0)
    n_fake = x_f.size(0)
    if n_real == 0 or n_fake == 0:
        return None

    # Fake partners: sample without replacement when possible, else with
    if n_fake >= n_real:
        partners = x_f[torch.randperm(n_fake, device=x.device)[:n_real]]
    else:
        idx = torch.randint(0, n_fake, (n_real,), device=x.device)
        partners = x_f[idx]

    # Per-pair λ ~ Beta(α,α) — bimodal for α=0.1
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=n_real).astype(np.float32)
    else:
        lam = np.full(n_real, 0.5, dtype=np.float32)
    lam_v = torch.tensor(lam, dtype=torch.float32, device=x.device).view(n_real, 1, 1, 1)

    mixed = (1.0 - lam_v) * x_r + lam_v * partners
    return {
        'image': mixed,
        'label': torch.ones(n_real, dtype=y.dtype, device=x.device),
        'label_soft': torch.ones(n_real, dtype=torch.float32, device=x.device),
    }

"""G11-1: RF pixel mixup labeled real (0), append mode (base batch untouched).

Same structure as G10's ``pixel_rf_hardfake``, but the appended real+fake pixel
mixes are hard-labeled REAL (0) instead of fake. This is the "label0" variant
that G1/G3 showed to preserve the real anchor and avoid the false-positive
explosion caused by labeling real-structure samples as fake.

    X_mix = (1−λ)·X_real + λ·X_fake,   λ ~ Beta(α,α)   (default α=5.0)

The base batch keeps its normal hard CE; the n_real extra mixes are appended
with hard label real (0).
"""
import numpy as np
import torch


def build_pixel_rf_label0(x, y, alpha=5.0):
    """Build n_real RF pixel-mixes (one per real anchor), hard-labeled real.

    Args:
        x:     [N, C, H, W] batch images
        y:     [N] labels (0=real, 1=fake)
        alpha: Beta(α,α) parameter for per-pair λ

    Returns:
        dict with image [n_real, C, H, W], label (all 0, long),
        label_soft (all 0.0, float) — or None when the batch contains no
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

    # Per-pair λ ~ Beta(α,α)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=n_real).astype(np.float32)
    else:
        lam = np.full(n_real, 0.5, dtype=np.float32)
    lam_v = torch.tensor(lam, dtype=torch.float32, device=x.device).view(n_real, 1, 1, 1)

    mixed = (1.0 - lam_v) * x_r + lam_v * partners
    return {
        'image': mixed,
        'label': torch.zeros(n_real, dtype=y.dtype, device=x.device),
        'label_soft': torch.zeros(n_real, dtype=torch.float32, device=x.device),
    }

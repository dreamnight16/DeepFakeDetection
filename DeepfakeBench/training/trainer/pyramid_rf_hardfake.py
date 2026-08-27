"""G10-pyramid: HiMix-style RF Laplacian-pyramid mixup with hard fake labels.

Same MDA framework as ``pixel_rf_hardfake`` (base batch untouched, ``n_real``
real+fake transitional samples appended, hard-labeled fake, per-pair mixing
coefficient ``q ~ Beta(alpha, alpha)`` with ``alpha=0.1`` → bimodal), but the
mixing operation is the project's Laplacian pyramid instead of naive pixel
interpolation: the real image's coarse structure ``G_K`` is preserved and only
the residual bands ``L_k`` are blended with the fake partner.

This combines HiMix's MDA recipe (arXiv:2604.27903, Eq. (1) hard-fake labeling
of near-real / near-fake transitional samples) with the project's core prior
that "fake = real structure + forged residual".  ``q ~ Beta(0.1, 0.1)`` yields
two regimes: q→0 keeps most real residual (near-real), q→1 injects mostly fake
residual (near-fake), both hard-labeled fake to expand the fake decision region.
"""
import numpy as np
import torch

# Reuse the pyramid helpers from trainer_v2 (module-level, no circular import)
from trainer.trainer_v2 import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_lap,
)


def build_pyramid_rf_hardfake(x, y, alpha=0.1, num_levels=3):
    """Build ``n_real`` RF Laplacian-pyramid mixes (one per real anchor), hard-labeled fake.

    Args:
        x:          [N, C, H, W] batch images
        y:          [N] labels (0=real, 1=fake)
        alpha:      Beta(α,α) parameter for the per-pair fake-injection strength
                    q (0.1 → bimodal: near-real or near-fake transitional samples)
        num_levels: K = number of Laplacian pyramid levels

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

    # Per-pair fake-injection strength q ~ Beta(α,α) — bimodal for α=0.1
    if alpha > 0:
        q = np.random.beta(alpha, alpha, size=n_real).astype(np.float32)
    else:
        q = np.full(n_real, 0.5, dtype=np.float32)
    q_t = torch.tensor(q, dtype=torch.float32, device=x.device).view(n_real, 1, 1, 1)

    # Laplacian pyramid: keep real coarse structure, blend residual bands
    gpyr_r = build_gaussian_pyramid(x_r, num_levels)
    gpyr_f = build_gaussian_pyramid(partners, num_levels)
    G_K = gpyr_r[-1]                              # coarse structure from real
    lap_r = build_laplacian_pyramid(gpyr_r)
    lap_f = build_laplacian_pyramid(gpyr_f)

    lap_mixed = [(1.0 - q_t) * lap_r[k] + q_t * lap_f[k] for k in range(num_levels)]
    mixed = reconstruct_from_lap(G_K, lap_mixed)
    # Clamp to the real anchor's value range (suppress reconstruction artefacts)
    mixed = torch.clamp(mixed,
                        x_r.amin(dim=(-3, -2, -1), keepdim=True),
                        x_r.amax(dim=(-3, -2, -1), keepdim=True))

    return {
        'image': mixed,
        'label': torch.ones(n_real, dtype=y.dtype, device=x.device),
        'label_soft': torch.ones(n_real, dtype=torch.float32, device=x.device),
    }

"""G11-3: RF Laplacian-pyramid mixup with soft energy-grounded label, append mode.

Takes the RF branch of the original ``lap_pyramid_mixup`` — coarse structure
from the real anchor, residual bands blended, soft label ỹ = 1−(1−e_f)^γ — but
keeps the base batch untouched (hard CE) and APPENDS the pyramid-mixed RF
samples instead of replacing the real samples.

No hard label is forced: the appended samples use the continuous energy-grounded
soft label (a "hard" label is derived only by thresholding for the acc metric).
"""
import numpy as np
import torch

from trainer.trainer_v2 import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_lap,
)


def build_pyramid_rf_soft(x, y, alpha=5.0, gamma=1.0, num_levels=3, epsilon=1e-8):
    """Build n_real RF pyramid-mixes with soft energy-grounded labels.

    Args:
        x:          [N, C, H, W] batch images
        y:          [N] labels (0=real, 1=fake)
        alpha:      Beta(α,α) parameter for the per-pair fake-injection strength q
        gamma:      asymmetry exponent for the soft label ỹ = 1−(1−e_f)^γ
        num_levels: K = number of Laplacian pyramid levels
        epsilon:    numerical stability constant

    Returns:
        dict with image [n_real, C, H, W], label (thresholded soft, long),
        label_soft (continuous in [0,1]) — or None when the batch contains no
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

    # Per-pair fake-injection strength q ~ Beta(α,α)
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

    # Default importance weights: ω₀ > ω₁ > ω₂ > …  (finer → higher)
    omega = [float(num_levels - i) for i in range(num_levels)]
    s = sum(omega)
    omega = [w / s for w in omega]

    lap_mixed = []
    num_terms = []
    den_terms = []
    for k in range(num_levels):
        L_r = lap_r[k]
        L_f = lap_f[k]
        L_mix = (1.0 - q_t) * L_r + q_t * L_f
        lap_mixed.append(L_mix)

        E_r = (L_r ** 2).reshape(n_real, -1).sum(dim=1)
        E_f = (L_f ** 2).reshape(n_real, -1).sum(dim=1)
        w_k = omega[k]
        num_terms.append(w_k * (q_t ** 2) * E_f)
        den_terms.append(w_k * ((1.0 - q_t) ** 2 * E_r + (q_t ** 2) * E_f))

    # Fake evidence e_f ∈ [0, 1] and energy-grounded soft label
    e_f = sum(num_terms) / (sum(den_terms) + epsilon)
    soft_y = 1.0 - (1.0 - e_f) ** gamma

    mixed = reconstruct_from_lap(G_K, lap_mixed)
    mixed = torch.clamp(mixed,
                        x_r.amin(dim=(-3, -2, -1), keepdim=True),
                        x_r.amax(dim=(-3, -2, -1), keepdim=True))

    return {
        'image': mixed,
        'label': (soft_y >= 0.5).long(),   # hard label only for the acc metric
        'label_soft': soft_y,
    }

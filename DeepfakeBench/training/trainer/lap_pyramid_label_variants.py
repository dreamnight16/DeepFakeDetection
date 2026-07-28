"""
Laplacian Pyramid Mixup — Label & Level Ablation Variants
==========================================================
Four experimental variants for ablating soft labels and pyramid levels:

  Variant 1 — Label=1 (lap_pyramid_label_1):
    Standard pyramid mixup (real base + fake injection), but soft labels
    are always 1 (all fake) regardless of injection proportion.
    RR/FF: pixel-space mixup, soft label = 1.

  Variant 2 — Label=0 (lap_pyramid_label_0):
    Standard pyramid mixup (real base + fake injection), but soft labels
    are always 0 (all real) regardless of injection proportion.
    RR/FF: pixel-space mixup, soft label = 0.

  Variant 3 — Top-Only (lap_pyramid_top_only):
    Only the coarsest Gaussian level G_K (pyramid top) is mixed;
    all Laplacian residuals come from the real image.
    Soft label from G_K-level injection-energy proportion.

  Variant 4 — Bottom-Only (lap_pyramid_bottom_only):
    Only the finest Laplacian level L_0 (pyramid bottom) is mixed;
    G_K and higher Laplacian levels (L_1, L_2, …) from the real image.
    Soft label from L_0-level injection-energy proportion.

All four use the v1 balance-batch sampler (use_balance_batch_sampler=true,
sampler_real_ratio=0.5) via randperm — RR, FF, and RF pairs are all produced.

mixup_mode values:
    lap_pyramid_label_1
    lap_pyramid_label_0
    lap_pyramid_top_only
    lap_pyramid_bottom_only

Author: personal experiment
"""

import numpy as np
import torch

# Reuse pyramid helpers from trainer_v2 (module-level, no circular import)
from trainer.trainer_v2 import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    reconstruct_from_lap,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _default_omega(num_levels):
    """Default level importance weights: finer level → higher weight."""
    omega = [float(num_levels - i) for i in range(num_levels)]
    s = sum(omega)
    return [w / s for w in omega]


# ═══════════════════════════════════════════════════════════════════════════════
# Variant 1: Label = 1  (all mixed samples labeled as fake)
# ═══════════════════════════════════════════════════════════════════════════════

def lap_pyramid_label_1(x, y, alpha=1.0, gamma=5.0, num_levels=3,
                         omega=None, epsilon=1e-8):
    """Pyramid mixup: RF pairs → soft label forced to 1 (fake).

    RR/FF: standard pixel-space mixup, label 0/1 respectively.
    RF:   pyramid mixup (real base + fake injection), soft_y = 1.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)

    index = torch.randperm(x.size(0), device=x.device)
    y_a = y.float()
    y_b = y[index].float()

    rr_mask = (y_a == 0) & (y_b == 0)
    ff_mask = (y_a == 1) & (y_b == 1)
    rf_mask = (y_a == 0) & (y_b == 1)
    fr_mask = (y_a == 1) & (y_b == 0)

    # rr: pixel-space, soft label = 0 (normal)
    rr_n = rr_mask.sum().item()
    if rr_n > 0:
        rr_x = lam_t * x[rr_mask] + (1.0 - lam_t) * x[index[rr_mask]]
        rr_y = torch.zeros(rr_n, device=x.device)
        rr_label = torch.zeros(rr_n, dtype=y.dtype, device=x.device)
    else:
        rr_x = torch.empty(0, *x.shape[1:], device=x.device)
        rr_y = torch.empty(0, device=x.device)
        rr_label = torch.empty(0, dtype=y.dtype, device=x.device)

    # ff: pixel-space, soft label = 1 (normal)
    ff_n = ff_mask.sum().item()
    if ff_n > 0:
        ff_x = lam_t * x[ff_mask] + (1.0 - lam_t) * x[index[ff_mask]]
        ff_y = torch.ones(ff_n, device=x.device)
        ff_label = torch.ones(ff_n, dtype=y.dtype, device=x.device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=x.device)
        ff_y = torch.empty(0, device=x.device)
        ff_label = torch.empty(0, dtype=y.dtype, device=x.device)

    # rf: all cross-class pairs (Q1 merge), pyramid mixup → soft label = 1
    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])
    fake_pos = torch.cat([index[rf_idx], fr_idx])
    n_rf = len(real_pos)

    if n_rf > 0:
        x_r = x[real_pos]
        x_f = x[fake_pos]

        gpyr_r = build_gaussian_pyramid(x_r, num_levels)
        gpyr_f = build_gaussian_pyramid(x_f, num_levels)
        G_K = gpyr_r[-1]                       # real coarse structure
        lap_r = build_laplacian_pyramid(gpyr_r)
        lap_f = build_laplacian_pyramid(gpyr_f)

        q_val = 1.0 - lam

        if omega is None:
            omega = _default_omega(num_levels)

        lap_mixed = []
        for k in range(num_levels):
            L_mix = (1.0 - q_val) * lap_r[k] + q_val * lap_f[k]
            lap_mixed.append(L_mix)

        rf_x = reconstruct_from_lap(G_K, lap_mixed)
        rf_x = torch.clamp(rf_x,
                           x_r.amin(dim=(-3, -2, -1), keepdim=True),
                           x_r.amax(dim=(-3, -2, -1), keepdim=True))
        # ── KEY: RF soft label forced to 1 ────────────────────────────────
        rf_y = torch.ones(n_rf, device=x.device)
    else:
        rf_x = torch.empty(0, *x.shape[1:], device=x.device)
        rf_y = torch.empty(0, device=x.device)

    # Combine
    parts_x = [t for t in [rr_x, ff_x, rf_x] if t.numel() > 0]
    parts_y = [t for t in [rr_y, ff_y, rf_y] if t.numel() > 0]
    mixed_x = torch.cat(parts_x, dim=0)
    mixed_y = torch.cat(parts_y, dim=0)

    # Hard labels (anchor class)
    label_parts = []
    if rr_n > 0:
        label_parts.append(torch.zeros(rr_n, dtype=y.dtype, device=x.device))
    if ff_n > 0:
        label_parts.append(torch.ones(ff_n, dtype=y.dtype, device=x.device))
    if n_rf > 0:
        label_parts.append(torch.ones(n_rf, dtype=y.dtype, device=x.device))
    mixed_label = torch.cat(label_parts, dim=0) if label_parts else y[:0]

    return mixed_x, mixed_y, mixed_label


# ═══════════════════════════════════════════════════════════════════════════════
# Variant 2: Label = 0  (all mixed samples labeled as real)
# ═══════════════════════════════════════════════════════════════════════════════

def lap_pyramid_label_0(x, y, alpha=1.0, gamma=5.0, num_levels=3,
                         omega=None, epsilon=1e-8):
    """Pyramid mixup: RF pairs → soft label forced to 0 (real).

    RR/FF: standard pixel-space mixup, label 0/1 respectively.
    RF:   pyramid mixup (real base + fake injection), soft_y = 0.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)

    index = torch.randperm(x.size(0), device=x.device)
    y_a = y.float()
    y_b = y[index].float()

    rr_mask = (y_a == 0) & (y_b == 0)
    ff_mask = (y_a == 1) & (y_b == 1)
    rf_mask = (y_a == 0) & (y_b == 1)
    fr_mask = (y_a == 1) & (y_b == 0)

    # rr: pixel-space, soft label = 0 (normal)
    rr_n = rr_mask.sum().item()
    if rr_n > 0:
        rr_x = lam_t * x[rr_mask] + (1.0 - lam_t) * x[index[rr_mask]]
        rr_y = torch.zeros(rr_n, device=x.device)
        rr_label = torch.zeros(rr_n, dtype=y.dtype, device=x.device)
    else:
        rr_x = torch.empty(0, *x.shape[1:], device=x.device)
        rr_y = torch.empty(0, device=x.device)
        rr_label = torch.empty(0, dtype=y.dtype, device=x.device)

    # ff: pixel-space, soft label = 1 (normal)
    ff_n = ff_mask.sum().item()
    if ff_n > 0:
        ff_x = lam_t * x[ff_mask] + (1.0 - lam_t) * x[index[ff_mask]]
        ff_y = torch.ones(ff_n, device=x.device)
        ff_label = torch.ones(ff_n, dtype=y.dtype, device=x.device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=x.device)
        ff_y = torch.empty(0, device=x.device)
        ff_label = torch.empty(0, dtype=y.dtype, device=x.device)

    # rf: all cross-class pairs (Q1 merge), pyramid mixup → soft label = 0
    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])
    fake_pos = torch.cat([index[rf_idx], fr_idx])
    n_rf = len(real_pos)

    if n_rf > 0:
        x_r = x[real_pos]
        x_f = x[fake_pos]

        gpyr_r = build_gaussian_pyramid(x_r, num_levels)
        gpyr_f = build_gaussian_pyramid(x_f, num_levels)
        G_K = gpyr_r[-1]
        lap_r = build_laplacian_pyramid(gpyr_r)
        lap_f = build_laplacian_pyramid(gpyr_f)

        q_val = 1.0 - lam

        if omega is None:
            omega = _default_omega(num_levels)

        lap_mixed = []
        for k in range(num_levels):
            L_mix = (1.0 - q_val) * lap_r[k] + q_val * lap_f[k]
            lap_mixed.append(L_mix)

        rf_x = reconstruct_from_lap(G_K, lap_mixed)
        rf_x = torch.clamp(rf_x,
                           x_r.amin(dim=(-3, -2, -1), keepdim=True),
                           x_r.amax(dim=(-3, -2, -1), keepdim=True))
        # ── KEY: RF soft label forced to 0 ────────────────────────────────
        rf_y = torch.zeros(n_rf, device=x.device)
    else:
        rf_x = torch.empty(0, *x.shape[1:], device=x.device)
        rf_y = torch.empty(0, device=x.device)

    # Combine
    parts_x = [t for t in [rr_x, ff_x, rf_x] if t.numel() > 0]
    parts_y = [t for t in [rr_y, ff_y, rf_y] if t.numel() > 0]
    mixed_x = torch.cat(parts_x, dim=0)
    mixed_y = torch.cat(parts_y, dim=0)

    # Hard labels (anchor class)
    label_parts = []
    if rr_n > 0:
        label_parts.append(torch.zeros(rr_n, dtype=y.dtype, device=x.device))
    if ff_n > 0:
        label_parts.append(torch.ones(ff_n, dtype=y.dtype, device=x.device))
    if n_rf > 0:
        label_parts.append(torch.zeros(n_rf, dtype=y.dtype, device=x.device))
    mixed_label = torch.cat(label_parts, dim=0) if label_parts else y[:0]

    return mixed_x, mixed_y, mixed_label


# ═══════════════════════════════════════════════════════════════════════════════
# Variant 3: Top-Only  (mix only coarsest Gaussian G_K)
# ═══════════════════════════════════════════════════════════════════════════════

def lap_pyramid_top_only(x, y, alpha=1.0, gamma=5.0, num_levels=3,
                          omega=None, epsilon=1e-8):
    """Pyramid mixup where ONLY the coarsest Gaussian level G_K is mixed.

    All Laplacian residual bands L_k come from the real image.
    Soft label is computed from the G_K-level injection-energy proportion:

        e_f = q²·‖G_K_f‖² / ((1−q)²·‖G_K_r‖² + q²·‖G_K_f‖² + ε)
        ỹ   = 1 − (1 − e_f)^γ

    RR/FF: standard pixel-space mixup, label 0/1 respectively.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)

    index = torch.randperm(x.size(0), device=x.device)
    y_a = y.float()
    y_b = y[index].float()

    rr_mask = (y_a == 0) & (y_b == 0)
    ff_mask = (y_a == 1) & (y_b == 1)
    rf_mask = (y_a == 0) & (y_b == 1)
    fr_mask = (y_a == 1) & (y_b == 0)

    # rr: pixel-space, label = 0
    rr_n = rr_mask.sum().item()
    if rr_n > 0:
        rr_x = lam_t * x[rr_mask] + (1.0 - lam_t) * x[index[rr_mask]]
        rr_y = torch.zeros(rr_n, device=x.device)
        rr_label = torch.zeros(rr_n, dtype=y.dtype, device=x.device)
    else:
        rr_x = torch.empty(0, *x.shape[1:], device=x.device)
        rr_y = torch.empty(0, device=x.device)
        rr_label = torch.empty(0, dtype=y.dtype, device=x.device)

    # ff: pixel-space, label = 1
    ff_n = ff_mask.sum().item()
    if ff_n > 0:
        ff_x = lam_t * x[ff_mask] + (1.0 - lam_t) * x[index[ff_mask]]
        ff_y = torch.ones(ff_n, device=x.device)
        ff_label = torch.ones(ff_n, dtype=y.dtype, device=x.device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=x.device)
        ff_y = torch.empty(0, device=x.device)
        ff_label = torch.empty(0, dtype=y.dtype, device=x.device)

    # rf: Q1 merge — only G_K mixed, all Laplacian from real
    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])
    fake_pos = torch.cat([index[rf_idx], fr_idx])
    n_rf = len(real_pos)

    if n_rf > 0:
        x_r = x[real_pos]
        x_f = x[fake_pos]

        gpyr_r = build_gaussian_pyramid(x_r, num_levels)
        gpyr_f = build_gaussian_pyramid(x_f, num_levels)

        q_val = 1.0 - lam

        # ── KEY: only G_K is mixed ───────────────────────────────────────
        G_K_r = gpyr_r[-1]                         # [n_rf, C, h, w]
        G_K_f = gpyr_f[-1]
        G_K_mixed = (1.0 - q_val) * G_K_r + q_val * G_K_f

        # All Laplacian residuals from real
        lap_r = build_laplacian_pyramid(gpyr_r)

        # ── Soft label: injection-energy proportion at G_K level only ───
        E_r = (G_K_r ** 2).reshape(n_rf, -1).sum(dim=1)     # [n_rf]
        E_f = (G_K_f ** 2).reshape(n_rf, -1).sum(dim=1)     # [n_rf]
        e_f = (q_val ** 2) * E_f / ((1.0 - q_val) ** 2 * E_r + (q_val ** 2) * E_f + epsilon)
        rf_y = 1.0 - (1.0 - e_f) ** gamma

        rf_x = reconstruct_from_lap(G_K_mixed, lap_r)
        rf_x = torch.clamp(rf_x,
                           x_r.amin(dim=(-3, -2, -1), keepdim=True),
                           x_r.amax(dim=(-3, -2, -1), keepdim=True))
        rf_label = rf_y
    else:
        rf_x = torch.empty(0, *x.shape[1:], device=x.device)
        rf_y = torch.empty(0, device=x.device)
        rf_label = torch.empty(0, dtype=y.dtype, device=x.device)

    # Combine
    parts_x = [t for t in [rr_x, ff_x, rf_x] if t.numel() > 0]
    parts_y = [t for t in [rr_y, ff_y, rf_y] if t.numel() > 0]
    parts_label = [t for t in [rr_label, ff_label, rf_label] if t.numel() > 0]

    mixed_x = torch.cat(parts_x, dim=0)
    mixed_y = torch.cat(parts_y, dim=0)
    mixed_label = torch.cat(parts_label, dim=0)

    return mixed_x, mixed_y, mixed_label


# ═══════════════════════════════════════════════════════════════════════════════
# Variant 4: Bottom-Only  (mix only finest Laplacian L_0)
# ═══════════════════════════════════════════════════════════════════════════════

def lap_pyramid_bottom_only(x, y, alpha=1.0, gamma=5.0, num_levels=3,
                             omega=None, epsilon=1e-8):
    """Pyramid mixup where ONLY the finest Laplacian level L_0 is mixed.

    G_K and higher Laplacian levels (L_1, L_2, …) all come from the real image.
    Soft label is computed from the L_0-level injection-energy proportion:

        e_f = q²·‖L_f_0‖² / ((1−q)²·‖L_r_0‖² + q²·‖L_f_0‖² + ε)
        ỹ   = 1 − (1 − e_f)^γ

    RR/FF: standard pixel-space mixup, label 0/1 respectively.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)

    index = torch.randperm(x.size(0), device=x.device)
    y_a = y.float()
    y_b = y[index].float()

    rr_mask = (y_a == 0) & (y_b == 0)
    ff_mask = (y_a == 1) & (y_b == 1)
    rf_mask = (y_a == 0) & (y_b == 1)
    fr_mask = (y_a == 1) & (y_b == 0)

    # rr: pixel-space, label = 0
    rr_n = rr_mask.sum().item()
    if rr_n > 0:
        rr_x = lam_t * x[rr_mask] + (1.0 - lam_t) * x[index[rr_mask]]
        rr_y = torch.zeros(rr_n, device=x.device)
        rr_label = torch.zeros(rr_n, dtype=y.dtype, device=x.device)
    else:
        rr_x = torch.empty(0, *x.shape[1:], device=x.device)
        rr_y = torch.empty(0, device=x.device)
        rr_label = torch.empty(0, dtype=y.dtype, device=x.device)

    # ff: pixel-space, label = 1
    ff_n = ff_mask.sum().item()
    if ff_n > 0:
        ff_x = lam_t * x[ff_mask] + (1.0 - lam_t) * x[index[ff_mask]]
        ff_y = torch.ones(ff_n, device=x.device)
        ff_label = torch.ones(ff_n, dtype=y.dtype, device=x.device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=x.device)
        ff_y = torch.empty(0, device=x.device)
        ff_label = torch.empty(0, dtype=y.dtype, device=x.device)

    # rf: Q1 merge — only L_0 mixed, all else from real
    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])
    fake_pos = torch.cat([index[rf_idx], fr_idx])
    n_rf = len(real_pos)

    if n_rf > 0:
        x_r = x[real_pos]
        x_f = x[fake_pos]

        gpyr_r = build_gaussian_pyramid(x_r, num_levels)
        gpyr_f = build_gaussian_pyramid(x_f, num_levels)
        G_K = gpyr_r[-1]                              # real coarse (unmixed)
        lap_r = build_laplacian_pyramid(gpyr_r)
        lap_f = build_laplacian_pyramid(gpyr_f)

        q_val = 1.0 - lam

        # ── KEY: only L_0 is mixed; L_1...L_{K-1} from real ──────────────
        L_0_mixed = (1.0 - q_val) * lap_r[0] + q_val * lap_f[0]
        lap_mixed = [L_0_mixed] + [lap_r[k] for k in range(1, num_levels)]

        # ── Soft label: injection-energy proportion at L_0 level only ────
        E_r_0 = (lap_r[0] ** 2).reshape(n_rf, -1).sum(dim=1)    # [n_rf]
        E_f_0 = (lap_f[0] ** 2).reshape(n_rf, -1).sum(dim=1)    # [n_rf]
        e_f = (q_val ** 2) * E_f_0 / ((1.0 - q_val) ** 2 * E_r_0 + (q_val ** 2) * E_f_0 + epsilon)
        rf_y = 1.0 - (1.0 - e_f) ** gamma

        rf_x = reconstruct_from_lap(G_K, lap_mixed)
        rf_x = torch.clamp(rf_x,
                           x_r.amin(dim=(-3, -2, -1), keepdim=True),
                           x_r.amax(dim=(-3, -2, -1), keepdim=True))
        rf_label = rf_y
    else:
        rf_x = torch.empty(0, *x.shape[1:], device=x.device)
        rf_y = torch.empty(0, device=x.device)
        rf_label = torch.empty(0, dtype=y.dtype, device=x.device)

    # Combine
    parts_x = [t for t in [rr_x, ff_x, rf_x] if t.numel() > 0]
    parts_y = [t for t in [rr_y, ff_y, rf_y] if t.numel() > 0]
    parts_label = [t for t in [rr_label, ff_label, rf_label] if t.numel() > 0]

    mixed_x = torch.cat(parts_x, dim=0)
    mixed_y = torch.cat(parts_y, dim=0)
    mixed_label = torch.cat(parts_label, dim=0)

    return mixed_x, mixed_y, mixed_label

"""
Artifact-Amplified Laplacian Pyramid Mixup
==========================================

RF samples inject the *fake artifact direction* L_f − L_r at controllable
strength a, instead of lap_pyramid_mixup's energy-grounded blend:

    L_mix = L_r + a·(L_f − L_r)   ==  (1−a)·L_r + a·L_f
    G_K   = G_r                    (coarse structure stays 100% real)
    y     = a / (1 + a)            (monotone, saturating soft label)

Motivation (from analyze_rf_label.py findings):
  - The mixed sample is ~97% real by energy (G_K dominates 10.85×), and
    E_f ≈ E_r in the residual band → the fake signal is weak AND the energy
    label collapses to ~0.5 (uninformative).
  - Amplifying a (allowing a > 1) strengthens the fake signal. The label
    a/(1+a) stays monotone for a ∈ [0, ∞), unlike q²/((1−q)²+q²) which is
    non-monotone beyond q = 1.

Pairing / RR / FF are identical to lap_pyramid_mixup; only the RF branch
differs. alpha drives RR/FF pixel mixing; gamma is kept for API compatibility
(unused).
"""
import numpy as np
import torch


def artifact_amp_mixup(x, y, alpha=1.0, gamma=5.0, num_levels=3,
                        amp_max=1.0):
    # Lazy import to avoid circular dependency (same pattern as trajectory_mixup).
    from trainer.trainer_v2 import (
        build_gaussian_pyramid,
        build_laplacian_pyramid,
        reconstruct_from_lap,
    )

    # RR/FF use a Beta(alpha, alpha) coefficient, unchanged from lap_pyramid_mixup.
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)

    index = torch.randperm(x.size(0), device=x.device)
    y_a = y.float()
    y_b = y[index].float()

    rr_mask = (y_a == 0) & (y_b == 0)   # real+real
    ff_mask = (y_a == 1) & (y_b == 1)   # fake+fake
    rf_mask = (y_a == 0) & (y_b == 1)   # real+fake
    fr_mask = (y_a == 1) & (y_b == 0)   # fake+real → merged into rf

    # ── RR: pixel-space mixup, label 0 ──────────────────────────────────────
    rr_x = lam_t * x[rr_mask] + (1.0 - lam_t) * x[index[rr_mask]]
    rr_y = torch.zeros(rr_mask.sum().item(), device=x.device)

    # ── FF: pixel-space mixup, label 1 ──────────────────────────────────────
    n_ff = ff_mask.sum().item()
    if n_ff > 0:
        ff_x = lam_t * x[ff_mask] + (1.0 - lam_t) * x[index[ff_mask]]
        ff_y = torch.ones(n_ff, device=x.device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=x.device)
        ff_y = torch.empty(0, device=x.device)

    # ── RF: artifact-amplified pyramid mixup ────────────────────────────────
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
        G_K = gpyr_r[-1]                       # coarse structure 100% real
        lap_r = build_laplacian_pyramid(gpyr_r)
        lap_f = build_laplacian_pyramid(gpyr_f)

        # Artifact injection strength: a ~ Uniform(0, amp_max), scalar per batch.
        a = float(np.random.uniform(0.0, amp_max))

        lap_mixed = []
        for k in range(num_levels):
            L_r = lap_r[k]
            L_f = lap_f[k]
            L_mix = L_r + a * (L_f - L_r)      # == (1−a)·L_r + a·L_f
            lap_mixed.append(L_mix)

        rf_x = reconstruct_from_lap(G_K, lap_mixed)
        rf_x = torch.clamp(rf_x,
                           x_r.amin(dim=(-3, -2, -1), keepdim=True),
                           x_r.amax(dim=(-3, -2, -1), keepdim=True))

        # Soft label: y = a/(1+a), monotone & saturating for a ∈ [0, ∞).
        y_soft = a / (1.0 + a)
        rf_y = torch.full((n_rf,), y_soft, device=x.device)
    else:
        rf_x = torch.empty(0, *x.shape[1:], device=x.device)
        rf_y = torch.empty(0, device=x.device)

    # ── Combine all pair types ──────────────────────────────────────────────
    parts_x = [t for t in [rr_x, ff_x, rf_x] if t.numel() > 0]
    parts_y = [t for t in [rr_y, ff_y, rf_y] if t.numel() > 0]
    mixed_x = torch.cat(parts_x, dim=0)
    mixed_y = torch.cat(parts_y, dim=0)

    # Hard labels: RF hard label = round(y) = (a >= 1).
    label_parts = []
    n_rr = rr_mask.sum().item()
    if n_rr > 0:
        label_parts.append(torch.zeros(n_rr, dtype=y.dtype, device=x.device))
    if n_ff > 0:
        label_parts.append(torch.ones(n_ff, dtype=y.dtype, device=x.device))
    if n_rf > 0:
        label_parts.append((rf_y >= 0.5).to(y.dtype))
    mixed_label = torch.cat(label_parts, dim=0) if label_parts else y[:0]

    # loss_mask: all ones (RF strip handled upstream via mixup_loss_strip).
    mask_parts = []
    if n_rr > 0:
        mask_parts.append(torch.ones(n_rr, device=x.device, dtype=torch.float32))
    if n_ff > 0:
        mask_parts.append(torch.ones(n_ff, device=x.device, dtype=torch.float32))
    if n_rf > 0:
        mask_parts.append(torch.ones(n_rf, device=x.device, dtype=torch.float32))
    loss_mask = torch.cat(mask_parts, dim=0) if mask_parts else mixed_y.new_zeros(0)

    return mixed_x, mixed_y, mixed_label, loss_mask

"""
Laplacian Pyramid Mixup Variants — Personal Experiments
========================================================
Three experimental variants extending lap_pyramid_mixup:

  Variant 1 — Fake-Base:
    Coarse structure G_K from the *fake* image; real Laplacian residuals
    are injected into the fake base.  Label measures remaining fake-ness.

  Variant 2 — Pyramid + Hardest K-Selection (real base):
    For each real anchor, K fake candidates are evaluated via full
    Laplacian-pyramid mixup; the one with highest model loss is kept.

  Variant 3 — Fake-Base + Hardest K-Selection:
    Combines variants 1+2: fake base with K-candidate hardest selection.

These are STANDALONE experiments — they do NOT modify any existing mixup
mode.  To use, set mixup_mode in the YAML config to one of:

    lap_pyramid_fake_base
    lap_pyramid_hardest
    lap_pyramid_fb_hardest

Author: personal experiment
"""

import numpy as np
import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════════
# Pyramid helpers  (self-contained — identical to trainer_v2 copies)
# ═══════════════════════════════════════════════════════════════════════════════

_PYR_KERNEL_CACHE = {}


def _get_pyr_kernel(channels, device):
    """Cached 5-tap binomial kernel [1,4,6,4,1]/16 (Burt & Adelson 1983)."""
    key = (channels, device.index if device.type == 'cuda' else -1)
    if key not in _PYR_KERNEL_CACHE:
        k = torch.tensor([1., 4., 6., 4., 1.], device=device) / 16.0
        _PYR_KERNEL_CACHE[key] = k
    return _PYR_KERNEL_CACHE[key]


def _pyr_down(x):
    """Gaussian pyramid reduce: separable binomial blur + downsample ×2."""
    N, C, H, W = x.shape
    k = _get_pyr_kernel(C, x.device)
    x_pad = F.pad(x, (2, 2, 2, 2), mode='reflect')
    k_h = k.view(1, 1, 1, 5).expand(C, 1, 1, 5)
    x_blur = F.conv2d(x_pad, k_h, groups=C)
    k_v = k.view(1, 1, 5, 1).expand(C, 1, 5, 1)
    x_blur = F.conv2d(x_blur, k_v, groups=C)
    return x_blur[:, :, ::2, ::2]


def _pyr_up(x, target_size):
    """Gaussian pyramid expand: upsample ×2 with bilinear interpolation."""
    return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)


def build_gaussian_pyramid(x, num_levels):
    """Build Gaussian pyramid: G_0 = x, G_{k+1} = pyr_down(G_k)."""
    pyramid = [x]
    for _ in range(num_levels):
        pyramid.append(_pyr_down(pyramid[-1]))
    return pyramid


def build_laplacian_pyramid(gpyr):
    """Build Laplacian pyramid: L_k = G_k - pyr_up(G_{k+1})."""
    lap = []
    for k in range(len(gpyr) - 1):
        G_k = gpyr[k]
        G_next_up = _pyr_up(gpyr[k + 1], G_k.shape[-2:])
        lap.append(G_k - G_next_up)
    return lap


def reconstruct_from_lap(G_K, lap_pyr):
    """Reconstruct image from coarsest Gaussian + Laplacian residuals (reversed)."""
    x = G_K
    for L_k in reversed(lap_pyr):
        x = _pyr_up(x, L_k.shape[-2:]) + L_k
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# Core: batched Laplacian-pyramid mixing for real+fake pairs
# ═══════════════════════════════════════════════════════════════════════════════

def _pyramid_rf_batch(x_r, x_f, lam_vals, gamma, num_levels, omega, epsilon,
                      base_is_real=True):
    """Core pyramid mixup for given real+fake pairs (batched, all levels at once).

    Args:
        x_r:       [N, C, H, W] real images  (anchor positions)
        x_f:       [N, C, H, W] fake images  (partner positions)
        lam_vals:  [N]  λ ∼ Beta(α,α) — mixing coefficient
        gamma:     asymmetry exponent for label
        num_levels: K pyramid levels
        omega:     level importance weights (default: decreasing → finer=higher)
        epsilon:   numerical stability
        base_is_real:
            True  → G_K from real,  inject fake residuals  (original)
            False → G_K from fake,  inject real residuals  (fake-base variant)

    Returns:
        mixed_x:  [N, C, H, W]  blended images
        soft_y:   [N]           soft labels ∈ [0, 1]
        e_inject: [N]           injection-energy proportion (for diagnostics)
    """
    N = x_r.size(0)
    device = x_r.device
    q_vals = 1.0 - lam_vals                    # injection strength  [N]

    # ── Build pyramids for all pairs ──────────────────────────────────────
    gpyr_r = build_gaussian_pyramid(x_r, num_levels)
    gpyr_f = build_gaussian_pyramid(x_f, num_levels)

    if base_is_real:
        G_K = gpyr_r[-1]                       # real coarse structure
        lap_base   = build_laplacian_pyramid(gpyr_r)   # real residuals
        lap_inject = build_laplacian_pyramid(gpyr_f)   # fake residuals injected
    else:
        G_K = gpyr_f[-1]                       # fake coarse structure
        lap_base   = build_laplacian_pyramid(gpyr_f)   # fake residuals
        lap_inject = build_laplacian_pyramid(gpyr_r)   # real residuals injected

    # ── Default importance weights: finer level → higher weight ────────────
    if omega is None:
        omega = [float(num_levels - i) for i in range(num_levels)]
        s = sum(omega)
        omega = [w / s for w in omega]

    # ── Mix level-by-level ────────────────────────────────────────────────
    lap_mixed = []
    num_terms = []
    den_terms = []

    q_sq   = q_vals ** 2                          # [N]
    q1_sq  = (1.0 - q_vals) ** 2                  # [N]
    q_t = q_vals.view(N, 1, 1, 1)                # [N, 1, 1, 1]

    for k in range(num_levels):
        L_base_k   = lap_base[k]
        L_inject_k = lap_inject[k]

        # Mixed residual:  (1−q)·L_base  +  q·L_inject
        L_mix_k = (1.0 - q_t) * L_base_k + q_t * L_inject_k
        lap_mixed.append(L_mix_k)

        # Per-sample residual energies
        E_base   = (L_base_k   ** 2).reshape(N, -1).sum(dim=1)   # [N]
        E_inject = (L_inject_k ** 2).reshape(N, -1).sum(dim=1)   # [N]

        w_k = omega[k]
        num_terms.append(w_k * q_sq   * E_inject)
        den_terms.append(w_k * (q1_sq * E_base + q_sq * E_inject))

    # Injection-energy proportion  e ∈ [0, 1]
    e_inject = sum(num_terms) / (sum(den_terms) + epsilon)        # [N]

    # ── Reconstruct ───────────────────────────────────────────────────────
    mixed_x = reconstruct_from_lap(G_K, lap_mixed)

    # Clamp to anchor's value range (suppress reconstruction artefacts)
    if base_is_real:
        anchor = x_r
    else:
        anchor = x_f
    mixed_x = torch.clamp(mixed_x,
                          anchor.amin(dim=(-3, -2, -1), keepdim=True),
                          anchor.amax(dim=(-3, -2, -1), keepdim=True))

    # ── Soft label ────────────────────────────────────────────────────────
    if base_is_real:
        # Real base + fake injection → deepfake-ness
        #   e_inject=0 (no fake) → 1−(1−0)^γ = 0  (real)
        #   e_inject=1 (all fake) → 1−0^γ = 1       (fake)
        soft_y = 1.0 - (1.0 - e_inject) ** gamma
    else:
        # Fake base + real injection → residual fake-ness
        #   e_inject=0 (no real) → (1−0)^γ = 1  (fully fake)
        #   e_inject=1 (all real) → 0^γ = 0     (real)
        soft_y = (1.0 - e_inject) ** gamma

    return mixed_x, soft_y, e_inject


# ═══════════════════════════════════════════════════════════════════════════════
# Variant 1: Fake-Base Laplacian Pyramid Mixup
# ═══════════════════════════════════════════════════════════════════════════════

def lap_pyramid_mixup_fake_base(x, y, alpha=1.0, gamma=5.0, num_levels=3,
                                 omega=None, epsilon=1e-8):
    """Fake-base Laplacian-pyramid residual mixup.

    The FAKE image provides the coarse structure G_K(x_f); REAL Laplacian
    residuals are injected.  This is the symmetric counterpart to the
    original lap_pyramid_mixup (which uses the real as base).

    Soft label:
        e_r = Σ ω_k q² ‖L_k(x_r)‖² / (Σ ω_k[(1−q)²‖L_k(x_f)‖² + q²‖L_k(x_r)‖²] + ε)
        ỹ   = (1 − e_r)^γ

    where q = 1−λ is the real residual injection strength (λ ∼ Beta(α,α)).

    Pairing (Q1: all cross-class pairs use fake as anchor):
      - real+real  → pixel-space mixup,  label = 0
      - fake+fake  → pixel-space mixup,  label = 1
      - real+fake  → fake-base pyramid mixup,  label = (1−e_r)^γ
      - fake+real  → merged into real+fake  (Q1)

    Args:
        x:          [N, C, H, W] images
        y:          [N] labels (0=real, 1=fake)
        alpha:      Beta(α,α) parameter for mixing strength λ
        gamma:      asymmetry exponent (γ>1 pushes labels toward extremes)
        num_levels: K = number of Laplacian pyramid levels
        omega:      importance weights [num_levels]; default = decreasing
        epsilon:    numerical stability constant

    Returns:
        mixed_x:     [N, C, H, W]
        mixed_y:     [N] soft labels ∈ [0, 1]
        mixed_label: [N] hard labels (anchor class, 0=real 1=fake)
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)

    index = torch.randperm(x.size(0), device=x.device)
    y_a = y.float()
    y_b = y[index].float()

    # ── Pair-type masks ───────────────────────────────────────────────────
    rr_mask = (y_a == 0) & (y_b == 0)          # real+real
    ff_mask = (y_a == 1) & (y_b == 1)          # fake+fake
    rf_mask = (y_a == 0) & (y_b == 1)          # real+fake → fake-base pyramid
    fr_mask = (y_a == 1) & (y_b == 0)          # fake+real → merged into rf

    # ── rr: real+real, pixel-space, label = 0 ──────────────────────────────
    rr_x = lam_t * x[rr_mask] + (1.0 - lam_t) * x[index[rr_mask]]
    rr_y = torch.zeros(rr_mask.sum().item(), device=x.device)

    # ── ff: fake+fake, pixel-space, label = 1 ──────────────────────────────
    n_ff = ff_mask.sum().item()
    if n_ff > 0:
        ff_x = lam_t * x[ff_mask] + (1.0 - lam_t) * x[index[ff_mask]]
        ff_y = torch.ones(n_ff, device=x.device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=x.device)
        ff_y = torch.empty(0, device=x.device)

    # ── rf: all cross-class pairs (Q1 merge), fake-base pyramid ────────────
    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    # Real anchor + fake partner (both directions merged per Q1)
    real_pos = torch.cat([rf_idx, index[fr_idx]])
    fake_pos = torch.cat([index[rf_idx], fr_idx])
    n_rf = len(real_pos)

    if n_rf > 0:
        x_r = x[real_pos]                    # real (inject into fake base)
        x_f = x[fake_pos]                    # fake (provides coarse structure)

        # Single shared λ for all rf pairs in this batch
        lam_vals = lam_t.expand(n_rf)
        rf_x, rf_y, _ = _pyramid_rf_batch(
            x_r, x_f, lam_vals, gamma, num_levels, omega, epsilon,
            base_is_real=False,               # ← fake is the base
        )
    else:
        rf_x = torch.empty(0, *x.shape[1:], device=x.device)
        rf_y = torch.empty(0, device=x.device)

    # ── Combine all pair types ────────────────────────────────────────────
    parts_x = [t for t in [rr_x, ff_x, rf_x] if t.numel() > 0]
    parts_y = [t for t in [rr_y, ff_y, rf_y] if t.numel() > 0]
    mixed_x = torch.cat(parts_x, dim=0)
    mixed_y = torch.cat(parts_y, dim=0)

    # Aligned hard labels (anchor class per mixed sample)
    label_parts = []
    n_rr = rr_mask.sum().item()
    if n_rr > 0:
        label_parts.append(torch.zeros(n_rr, dtype=y.dtype, device=x.device))
    if n_ff > 0:
        label_parts.append(torch.ones(n_ff, dtype=y.dtype, device=x.device))
    if n_rf > 0:
        # rf anchor is real → hard label = 0
        label_parts.append(torch.zeros(n_rf, dtype=y.dtype, device=x.device))
    mixed_label = torch.cat(label_parts, dim=0) if label_parts else y[:0]

    return mixed_x, mixed_y, mixed_label


# ═══════════════════════════════════════════════════════════════════════════════
# Variant 2: Pyramid + Hardest K-Selection  (real base)
# ═══════════════════════════════════════════════════════════════════════════════

def lap_pyramid_hardest_mixup(model, data_dict, K, alpha=1.0, gamma=5.0,
                               num_levels=3, omega=None, epsilon=1e-8,
                               selection='hardest'):
    """Pyramid mixup with K-candidate hardest selection for rf pairs.

    For each real anchor, K different fake partners are evaluated via full
    Laplacian-pyramid mixup.  The candidate producing the highest model loss
    is selected as the hardest training example.

    Pairing:
      - rr: pixel-space mixup, label = 0
      - ff: pixel-space mixup, label = 1
      - rf: K pyramid mixups → hardest selection  (real base, fake injected)

    Args:
        model:      detector model (used for hardness scoring)
        data_dict:  {'image': [N,C,H,W], 'label': [N], ...}
        K:          number of fake candidates per real anchor
        alpha:      Beta(α,α) parameter
        gamma:      asymmetry exponent
        num_levels: K pyramid levels
        omega:      level importance weights
        epsilon:    numerical stability
        selection:  'hardest' (max loss) | 'random' (uniform choice)

    Returns:
        data_dict with 'image', 'label', 'label_soft' updated
    """
    x, y = data_dict['image'], data_dict['label']
    real_idx = (y == 0).nonzero(as_tuple=True)[0]       # [R]
    fake_idx = (y == 1).nonzero(as_tuple=True)[0]       # [F]
    B = x.size(0)
    R, F_orig = len(real_idx), len(fake_idx)

    # ── Fallback: K≤1 or degenerate ────────────────────────────────────────
    if K <= 1 or R == 0 or F_orig == 0:
        mixed_x, mixed_y, mixed_label = _lap_pyramid_full_batch(
            x, y, alpha, gamma, num_levels, omega, epsilon, base_is_real=True)
        return {**data_dict, 'image': mixed_x, 'label': mixed_label,
                'label_soft': mixed_y}

    # ── 1. Randperm base (shared λ, same structure as lap_pyramid_mixup) ──
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)
    index = torch.randperm(B, device=x.device)

    rr = (y == 0) & (y[index] == 0)          # real+real
    ff = (y == 1) & (y[index] == 1)          # fake+fake
    rf = (y == 0) & (y[index] == 1)          # real+fake
    fr = (y == 1) & (y[index] == 0)          # fake+real → merged into rf

    # ── 2. rr: pixel-space, label = 0 ──────────────────────────────────────
    rr_x = lam_t * x[rr] + (1.0 - lam_t) * x[index[rr]]
    rr_soft = torch.zeros(rr.sum().item(), device=x.device)

    # ── 3. ff: pixel-space, label = 1 ──────────────────────────────────────
    n_ff = ff.sum().item()
    if n_ff > 0:
        ff_x = lam_t * x[ff] + (1.0 - lam_t) * x[index[ff]]
        ff_soft = torch.ones(n_ff, device=x.device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=x.device)
        ff_soft = torch.empty(0, device=x.device)

    # ── 4. rf: Q1 merge + K-candidate pyramid mixup ────────────────────────
    rf_idx = rf.nonzero(as_tuple=True)[0]
    fr_idx = fr.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])        # real anchors  [n_rf]
    fake_pos = torch.cat([index[rf_idx], fr_idx])        # fake partners [n_rf]
    n_rf = len(real_pos)

    if n_rf == 0:
        # No cross-class pairs — combine rr + ff only
        parts_x = [t for t in [rr_x, ff_x] if t.numel() > 0]
        parts_soft = [t for t in [rr_soft, ff_soft] if t.numel() > 0]
        parts_label = [
            torch.zeros(rr.sum().item(), dtype=y.dtype, device=x.device),
            torch.ones(n_ff, dtype=y.dtype, device=x.device) if n_ff > 0
            else torch.empty(0, dtype=y.dtype, device=x.device),
        ]
        new_x = torch.cat(parts_x, dim=0)
        new_label_soft = torch.cat(parts_soft, dim=0)
        new_label = torch.cat(parts_label, dim=0)
        return {**data_dict, 'image': new_x, 'label': new_label,
                'label_soft': new_label_soft}

    K_eff = min(K, F_orig)

    # K distinct fakes per rf-real (without replacement)
    cand_fake = torch.stack([
        fake_idx[torch.randperm(F_orig, device=x.device)[:K_eff]]
        for _ in range(n_rf)
    ], dim=1)                                                        # [K_eff, n_rf]

    # Per-(k,r) independent λ
    lam_kr = np.random.beta(alpha, alpha, size=(K_eff, n_rf)) if alpha > 0 \
        else np.ones((K_eff, n_rf))
    lam_t_kr = x.new_tensor(lam_kr).float()                          # [K_eff, n_rf]

    # ── Batch pyramid mixup for all K_eff × n_rf pairs ────────────────────
    x_real_rep = (x[real_pos]
                  .unsqueeze(0).expand(K_eff, -1, -1, -1, -1)
                  .reshape(K_eff * n_rf, *x.shape[1:]))
    x_fake_rep = x[cand_fake.reshape(-1)]

    mixed_kr, soft_val_kr, _ = _pyramid_rf_batch(
        x_real_rep, x_fake_rep,
        lam_t_kr.reshape(-1), gamma, num_levels, omega, epsilon,
        base_is_real=True,                     # real base, fake injected
    )
    soft_val_exp = soft_val_kr                                               # [K_eff * n_rf]

    # ── Selection (random or hardest) ─────────────────────────────────────
    if selection == 'random':
        best_k = torch.randint(0, K_eff, (n_rf,), device=x.device)
    else:  # hardest
        model_module = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            feat_kr = model_module.features({**data_dict, 'image': mixed_kr})
            pred_kr = model_module.classifier(feat_kr)
        log_p = F.log_softmax(pred_kr, dim=1)
        # Cross-entropy with soft labels: −[y·log(p₁) + (1−y)·log(p₀)]
        loss_kr = -(soft_val_exp * log_p[:, 1] +
                     (1.0 - soft_val_exp) * log_p[:, 0])
        best_k = loss_kr.view(K_eff, n_rf).argmax(dim=0)

    flat_idx = best_k * n_rf + torch.arange(n_rf, device=x.device)
    rf_x = mixed_kr[flat_idx]
    rf_soft = soft_val_exp[flat_idx]

    # ── Combine all ───────────────────────────────────────────────────────
    new_x = torch.cat([rr_x, ff_x, rf_x], dim=0)
    new_label_soft = torch.cat([rr_soft, ff_soft, rf_soft], dim=0)
    new_label = torch.cat([
        torch.zeros(rr.sum().item(), dtype=y.dtype, device=x.device),
        torch.ones(n_ff, dtype=y.dtype, device=x.device) if n_ff > 0
        else torch.empty(0, dtype=y.dtype, device=x.device),
        torch.zeros(n_rf, dtype=y.dtype, device=x.device),
    ], dim=0)

    return {**data_dict, 'image': new_x, 'label': new_label,
            'label_soft': new_label_soft}


# ═══════════════════════════════════════════════════════════════════════════════
# Variant 3: Fake-Base + Hardest K-Selection
# ═══════════════════════════════════════════════════════════════════════════════

def lap_pyramid_fake_base_hardest_mixup(model, data_dict, K, alpha=1.0,
                                         gamma=5.0, num_levels=3, omega=None,
                                         epsilon=1e-8, selection='hardest'):
    """Fake-base pyramid mixup with K-candidate hardest selection.

    Combines Variants 1+2:
      - Coarse structure G_K from the fake image (fake-base)
      - K fake candidates per real anchor → hardest selection via model loss

    Pairing:
      - rr: pixel-space mixup, label = 0
      - ff: pixel-space mixup, label = 1
      - rf: K fake-base pyramid mixups → hardest selection

    Args:
        model:      detector model (used for hardness scoring)
        data_dict:  {'image': [N,C,H,W], 'label': [N], ...}
        K:          number of fake candidates per real anchor
        alpha:      Beta(α,α) parameter
        gamma:      asymmetry exponent
        num_levels: K pyramid levels
        omega:      level importance weights
        epsilon:    numerical stability
        selection:  'hardest' (max loss) | 'random' (uniform choice)

    Returns:
        data_dict with 'image', 'label', 'label_soft' updated
    """
    x, y = data_dict['image'], data_dict['label']
    real_idx = (y == 0).nonzero(as_tuple=True)[0]
    fake_idx = (y == 1).nonzero(as_tuple=True)[0]
    B = x.size(0)
    R, F_orig = len(real_idx), len(fake_idx)

    # ── Fallback: K≤1 or degenerate ────────────────────────────────────────
    if K <= 1 or R == 0 or F_orig == 0:
        mixed_x, mixed_y, mixed_label = lap_pyramid_mixup_fake_base(
            x, y, alpha, gamma, num_levels, omega, epsilon)
        return {**data_dict, 'image': mixed_x, 'label': mixed_label,
                'label_soft': mixed_y}

    # ── 1. Randperm base ──────────────────────────────────────────────────
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    lam_t = torch.tensor(lam, dtype=torch.float32, device=x.device)
    index = torch.randperm(B, device=x.device)

    rr = (y == 0) & (y[index] == 0)
    ff = (y == 1) & (y[index] == 1)
    rf = (y == 0) & (y[index] == 1)
    fr = (y == 1) & (y[index] == 0)

    # ── 2. rr: pixel-space, label = 0 ──────────────────────────────────────
    rr_x = lam_t * x[rr] + (1.0 - lam_t) * x[index[rr]]
    rr_soft = torch.zeros(rr.sum().item(), device=x.device)

    # ── 3. ff: pixel-space, label = 1 ──────────────────────────────────────
    n_ff = ff.sum().item()
    if n_ff > 0:
        ff_x = lam_t * x[ff] + (1.0 - lam_t) * x[index[ff]]
        ff_soft = torch.ones(n_ff, device=x.device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=x.device)
        ff_soft = torch.empty(0, device=x.device)

    # ── 4. rf: Q1 merge + K-candidate FAKE-BASE pyramid mixup ──────────────
    rf_idx = rf.nonzero(as_tuple=True)[0]
    fr_idx = fr.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])
    fake_pos = torch.cat([index[rf_idx], fr_idx])
    n_rf = len(real_pos)

    if n_rf == 0:
        parts_x = [t for t in [rr_x, ff_x] if t.numel() > 0]
        parts_soft = [t for t in [rr_soft, ff_soft] if t.numel() > 0]
        parts_label = [
            torch.zeros(rr.sum().item(), dtype=y.dtype, device=x.device),
            torch.ones(n_ff, dtype=y.dtype, device=x.device) if n_ff > 0
            else torch.empty(0, dtype=y.dtype, device=x.device),
        ]
        new_x = torch.cat(parts_x, dim=0)
        new_label_soft = torch.cat(parts_soft, dim=0)
        new_label = torch.cat(parts_label, dim=0)
        return {**data_dict, 'image': new_x, 'label': new_label,
                'label_soft': new_label_soft}

    K_eff = min(K, F_orig)

    # K distinct fakes per rf-real
    cand_fake = torch.stack([
        fake_idx[torch.randperm(F_orig, device=x.device)[:K_eff]]
        for _ in range(n_rf)
    ], dim=1)                                                        # [K_eff, n_rf]

    # Per-(k,r) independent λ
    lam_kr = np.random.beta(alpha, alpha, size=(K_eff, n_rf)) if alpha > 0 \
        else np.ones((K_eff, n_rf))
    lam_t_kr = x.new_tensor(lam_kr).float()

    # ── Batch FAKE-BASE pyramid mixup for all K_eff × n_rf pairs ──────────
    x_real_rep = (x[real_pos]
                  .unsqueeze(0).expand(K_eff, -1, -1, -1, -1)
                  .reshape(K_eff * n_rf, *x.shape[1:]))
    x_fake_rep = x[cand_fake.reshape(-1)]

    # NOTE: x_r = real (provides injected residuals),
    #       x_f = fake (provides coarse structure G_K)
    mixed_kr, soft_val_kr, _ = _pyramid_rf_batch(
        x_real_rep, x_fake_rep,
        lam_t_kr.reshape(-1), gamma, num_levels, omega, epsilon,
        base_is_real=False,                    # ← fake is the base
    )
    soft_val_exp = soft_val_kr

    # ── Selection ─────────────────────────────────────────────────────────
    if selection == 'random':
        best_k = torch.randint(0, K_eff, (n_rf,), device=x.device)
    else:  # hardest
        model_module = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            feat_kr = model_module.features({**data_dict, 'image': mixed_kr})
            pred_kr = model_module.classifier(feat_kr)
        log_p = F.log_softmax(pred_kr, dim=1)
        loss_kr = -(soft_val_exp * log_p[:, 1] +
                     (1.0 - soft_val_exp) * log_p[:, 0])
        best_k = loss_kr.view(K_eff, n_rf).argmax(dim=0)

    flat_idx = best_k * n_rf + torch.arange(n_rf, device=x.device)
    rf_x = mixed_kr[flat_idx]
    rf_soft = soft_val_exp[flat_idx]

    # ── Combine all ───────────────────────────────────────────────────────
    new_x = torch.cat([rr_x, ff_x, rf_x], dim=0)
    new_label_soft = torch.cat([rr_soft, ff_soft, rf_soft], dim=0)
    new_label = torch.cat([
        torch.zeros(rr.sum().item(), dtype=y.dtype, device=x.device),
        torch.ones(n_ff, dtype=y.dtype, device=x.device) if n_ff > 0
        else torch.empty(0, dtype=y.dtype, device=x.device),
        torch.zeros(n_rf, dtype=y.dtype, device=x.device),
    ], dim=0)

    return {**data_dict, 'image': new_x, 'label': new_label,
            'label_soft': new_label_soft}


# ═══════════════════════════════════════════════════════════════════════════════
# Fallback helper: full-batch lap_pyramid_mixup (real base)
#   Used when hardest-mixup degenerates (K≤1, no real/fake pairs, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

def _lap_pyramid_full_batch(x, y, alpha, gamma, num_levels, omega, epsilon,
                             base_is_real=True):
    """Full-batch Laplacian pyramid mixup — used as fallback in hardest variants.

    This is logically identical to lap_pyramid_mixup / lap_pyramid_mixup_fake_base
    (depending on base_is_real) but returns (mixed_x, mixed_y, mixed_label).
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

    rr_x = lam_t * x[rr_mask] + (1.0 - lam_t) * x[index[rr_mask]]
    rr_y = torch.zeros(rr_mask.sum().item(), device=x.device)

    n_ff = ff_mask.sum().item()
    if n_ff > 0:
        ff_x = lam_t * x[ff_mask] + (1.0 - lam_t) * x[index[ff_mask]]
        ff_y = torch.ones(n_ff, device=x.device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=x.device)
        ff_y = torch.empty(0, device=x.device)

    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])
    fake_pos = torch.cat([index[rf_idx], fr_idx])
    n_rf = len(real_pos)

    if n_rf > 0:
        x_r = x[real_pos]
        x_f = x[fake_pos]
        lam_vals = lam_t.expand(n_rf)
        rf_x, rf_y, _ = _pyramid_rf_batch(
            x_r, x_f, lam_vals, gamma, num_levels, omega, epsilon,
            base_is_real=base_is_real,
        )
    else:
        rf_x = torch.empty(0, *x.shape[1:], device=x.device)
        rf_y = torch.empty(0, device=x.device)

    parts_x = [t for t in [rr_x, ff_x, rf_x] if t.numel() > 0]
    parts_y = [t for t in [rr_y, ff_y, rf_y] if t.numel() > 0]
    mixed_x = torch.cat(parts_x, dim=0)
    mixed_y = torch.cat(parts_y, dim=0)

    label_parts = []
    n_rr = rr_mask.sum().item()
    if n_rr > 0:
        label_parts.append(torch.zeros(n_rr, dtype=y.dtype, device=x.device))
    if n_ff > 0:
        label_parts.append(torch.ones(n_ff, dtype=y.dtype, device=x.device))
    if n_rf > 0:
        label_parts.append(torch.zeros(n_rf, dtype=y.dtype, device=x.device))
    mixed_label = torch.cat(label_parts, dim=0) if label_parts else y[:0]

    return mixed_x, mixed_y, mixed_label

"""
Diffusion Trajectory-Guided Mixup for Generalizable AIGC Detection
===================================================================

Implements the theoretical framework:

  - DDPM forward scheduler (linear schedule, T=1000)
  - Shared noise: same ε for real and fake at every timestep
  - Multi-step trajectory: K evenly-spaced timesteps t ∈ [t_min, t_max]
  - Cosine trajectory weight: λ_t = 0.5·(1 + cos(π·t/T))
  - Trajectory interpolation: z_t = λ_t·x_t^r + (1−λ_t)·x_t^f
  - Mean aggregation:  x_m = (1/K)·Σ_t z_t
  - Soft label:  y_m = 1 − (1/K)·Σ_t λ_t

Two modes:
  trajectory          — multi-step trajectory aggregation (pixel-space)
  trajectory_pyramid  — trajectory + Laplacian pyramid (per-step pyramid then aggregate)

Training strategy (domain consistency):
  RR → clean real,        label = 0
  FF → clean fake,        label = 1
  RF → trajectory-mixed,  soft label (hard label = round(soft))

The detector sees BOTH clean and trajectory-space samples in every batch,
preventing it from learning noise-level shortcuts.

These are STANDALONE experiments — they do NOT modify any existing mixup mode.
To use, set mixup_mode in the YAML config to:
    trajectory
    trajectory_pyramid

Config keys (in effort.yaml):
    traj_t_min:    50      # first trajectory timestep
    traj_t_max:    700     # last trajectory timestep
    traj_T:        1000    # total DDPM steps
    traj_num_steps: 14     # K = number of timesteps sampled (evenly spaced)

Author: personal experiment
"""

import math
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# DDPM Forward Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

class DDPMScheduler:
    """DDPM linear noise schedule (Ho et al. 2020).

    β ∈ [β_start, β_end], T timesteps.
    Precomputes ᾱ_t = ∏_{i=1}^t (1 − β_i) and its sqrt for fast q_sample.

    Default: T=1000, β_start=1e-4, β_end=0.02.
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 device='cpu'):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod)

    def to(self, device):
        """Move all buffers to *device*."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = \
            self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

    def q_sample(self, x_0, t, noise):
        """DDPM forward:  x_t = √ᾱ_t · x_0  +  √(1−ᾱ_t) · ε.

        Args:
            x_0:   [N, C, H, W] clean images
            t:     int — scalar timestep index (broadcast to all N samples)
            noise: [N, C, H, W] Gaussian noise ε ∼ N(0,I)

        Returns:
            x_t: [N, C, H, W] noised images
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise


# Global scheduler (lazy init, moved to correct device on first use)
_scheduler = None


def _get_scheduler(device='cpu'):
    global _scheduler
    if _scheduler is None:
        _scheduler = DDPMScheduler(device=device)
    elif str(_scheduler.betas.device) != str(device):
        _scheduler.to(device)
    return _scheduler


# ═══════════════════════════════════════════════════════════════════════════════
# Cosine Trajectory Mixing Weight
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lambda_t(t, T=1000):
    """Cosine schedule:  λ_t = 0.5 · (1 + cos(π · t / T)).

    Properties:
        t = 0   (clean image):  λ → 1  (preserve real structure)
        t = T   (pure  noise):  λ → 0  (let fake distribution dominate)

    Accepts scalar int; returns float.

    Args:
        t: int — timestep index
        T: int — total diffusion steps

    Returns:
        float — λ_t ∈ [0, 1]
    """
    return 0.5 * (1.0 + math.cos(math.pi * t / T))


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: build trajectory timesteps
# ═══════════════════════════════════════════════════════════════════════════════

def _build_traj_steps(t_min, t_max, num_steps, device):
    """K evenly-spaced integer timesteps in [t_min, t_max].

    Returns:
        steps: [K] LongTensor on *device*
    """
    return torch.linspace(t_min, t_max, num_steps, device=device).long()


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-step Diffusion Trajectory Aggregation Mixup
# ═══════════════════════════════════════════════════════════════════════════════

def trajectory_mixup(x, y, alpha=1.0, gamma=5.0,
                     t_min=50, t_max=700, T=1000,
                     num_traj_steps=14):
    """Multi-step Diffusion Trajectory Aggregation Mixup.

    Instead of a single random timestep, constructs a full trajectory
    at K evenly-spaced timesteps t ∈ [t_min, t_max], interpolates
    real and fake at each step, and aggregates via mean pooling:

        z_t  = λ_t·q_t(x_r, ε) + (1−λ_t)·q_t(x_f, ε)
        x_m  = (1/K)·Σ_t z_t
        y_m  = 1 − (1/K)·Σ_t λ_t

    Training strategy (domain consistency):
        RR → clean real,                label = 0
        FF → clean fake,                label = 1
        RF → trajectory-aggregated,     soft label, hard = round(soft)

    The detector sees both clean and trajectory-space samples,
    preventing noise-level shortcuts.

    Shared noise ε is identical across all timesteps for a given
    (real, fake) pair — this is the true DDPM forward trajectory.

    Args:
        x:              [N, C, H, W] images
        y:              [N] labels (0=real, 1=fake)
        alpha:          unused (API consistency)
        gamma:          unused (API consistency)
        t_min:          first trajectory timestep (default 50)
        t_max:          last trajectory timestep (default 700)
        T:              total DDPM steps (default 1000)
        num_traj_steps: K = number of timesteps (default 14, step≈50)

    Returns:
        mixed_x:     [M, C, H, W]  M = N (all pairs preserved)
        mixed_y:     [M] soft labels ∈ [0, 1]
        mixed_label: [M] hard labels (0/1)
    """
    B = x.size(0)
    device = x.device
    scheduler = _get_scheduler(device)

    # ── Randperm pairing ─────────────────────────────────────────────────
    index = torch.randperm(B, device=device)
    y_a = y.float()
    y_b = y[index].float()

    rr_mask = (y_a == 0) & (y_b == 0)          # real+real
    ff_mask = (y_a == 1) & (y_b == 1)          # fake+fake
    rf_mask = (y_a == 0) & (y_b == 1)          # real+fake
    fr_mask = (y_a == 1) & (y_b == 0)          # fake+real → merged into rf

    # ── RR: clean real, label = 0 ──────────────────────────────────────────
    rr_n = rr_mask.sum().item()
    if rr_n > 0:
        rr_x = x[rr_mask]
        rr_y = torch.zeros(rr_n, device=device)
        rr_label = torch.zeros(rr_n, dtype=y.dtype, device=device)
    else:
        rr_x = torch.empty(0, *x.shape[1:], device=device)
        rr_y = torch.empty(0, device=device)
        rr_label = torch.empty(0, dtype=y.dtype, device=device)

    # ── FF: clean fake, label = 1 ──────────────────────────────────────────
    ff_n = ff_mask.sum().item()
    if ff_n > 0:
        ff_x = x[ff_mask]
        ff_y = torch.ones(ff_n, device=device)
        ff_label = torch.ones(ff_n, dtype=y.dtype, device=device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=device)
        ff_y = torch.empty(0, device=device)
        ff_label = torch.empty(0, dtype=y.dtype, device=device)

    # ── RF+FR merged: multi-step trajectory aggregation ────────────────────
    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])       # real anchors  [n_rf]
    fake_pos = torch.cat([index[rf_idx], fr_idx])       # fake partners [n_rf]
    n_rf = len(real_pos)

    if n_rf > 0:
        x_r = x[real_pos]                               # [n_rf, C, H, W]
        x_f = x[fake_pos]                               # [n_rf, C, H, W]

        # ── K evenly-spaced timesteps ────────────────────────────────────
        steps = _build_traj_steps(t_min, t_max, num_traj_steps, device)

        # ── Shared noise seed (identical ε for all timesteps → true DDPM trajectory)
        noise = torch.randn_like(x_r)                   # [n_rf, C, H, W]

        # ── Accumulate across trajectory ─────────────────────────────────
        traj_accum = torch.zeros_like(x_r)              # Σ_t z_t
        lambda_sum = 0.0                                # Σ_t λ_t

        for t_val in steps:
            t_i = t_val.item()

            # DDPM forward with shared noise
            x_t_r = scheduler.q_sample(x_r, t_i, noise)
            x_t_f = scheduler.q_sample(x_f, t_i, noise)

            # Cosine weight at this timestep
            lam_t = compute_lambda_t(t_i, T)

            # Trajectory interpolation at step t
            z_t = lam_t * x_t_r + (1.0 - lam_t) * x_t_f

            traj_accum += z_t
            lambda_sum += lam_t

        # ── Mean aggregation ─────────────────────────────────────────────
        rf_x = traj_accum / num_traj_steps

        # ── Soft label: y_m = 1 − mean(λ_t) ──────────────────────────────
        lambda_mean = lambda_sum / num_traj_steps
        rf_y = torch.full((n_rf,), 1.0 - lambda_mean, device=device)

        # ── Hard label: threshold soft label at 0.5 ─────────────────────
        rf_label = (rf_y >= 0.5).long()
    else:
        rf_x = torch.empty(0, *x.shape[1:], device=device)
        rf_y = torch.empty(0, device=device)
        rf_label = torch.empty(0, dtype=y.dtype, device=device)

    # ── Combine all pair types ────────────────────────────────────────────
    parts_x = [t for t in [rr_x, ff_x, rf_x] if t.numel() > 0]
    parts_y = [t for t in [rr_y, ff_y, rf_y] if t.numel() > 0]
    parts_label = [t for t in [rr_label, ff_label, rf_label]
                   if t.numel() > 0]

    mixed_x = torch.cat(parts_x, dim=0)
    mixed_y = torch.cat(parts_y, dim=0)
    mixed_label = torch.cat(parts_label, dim=0)
    return mixed_x, mixed_y, mixed_label


# ═══════════════════════════════════════════════════════════════════════════════
# Pyramid Trajectory Aggregation Mixup
# ═══════════════════════════════════════════════════════════════════════════════

def pyramid_trajectory_mixup(x, y, alpha=1.0, gamma=5.0, num_levels=3,
                              omega=None, epsilon=1e-8,
                              t_min=50, t_max=700, T=1000,
                              num_traj_steps=14):
    """Multi-step Pyramid-aware Diffusion Trajectory Aggregation Mixup.

    Combines trajectory mixing (temporal dimension) with Laplacian pyramid
    mixing (spatial dimension).  At each trajectory timestep:

      1. DDPM forward x_t^r, x_t^f with shared noise
      2. Laplacian pyramid decomposition
      3. Per-level mixing: L_k^mix = λ_{t,k}·L_k^r + (1−λ_{t,k})·L_k^f
         where λ_{t,k} = λ_t · (1 − k/K)
      4. Coarsest Gaussian base from fake (λ_{t,K} = 0)
      5. Reconstruct to image space
      6. Accumulate across timesteps

    Then aggregate via mean pooling across the trajectory.

    Training strategy (domain consistency):
        RR → clean real,                label = 0
        FF → clean fake,                label = 1
        RF → pyramid-trajectory,        soft label, hard = round(soft)

    Args:
        x:              [N, C, H, W] images
        y:              [N] labels (0=real, 1=fake)
        alpha:          unused (API consistency)
        gamma:          unused (API consistency)
        num_levels:     K = number of Laplacian pyramid levels
        omega:          level importance weights [K+1]; default = decreasing
        epsilon:        numerical stability
        t_min:          first trajectory timestep (default 50)
        t_max:          last trajectory timestep (default 700)
        T:              total DDPM steps (default 1000)
        num_traj_steps: number of trajectory timesteps (default 14)

    Returns:
        mixed_x:     [M, C, H, W]
        mixed_y:     [M] soft labels ∈ [0, 1]
        mixed_label: [M] hard labels (0/1)
    """
    # Deferred import to avoid circular dependency at module level
    from trainer.trainer_v2 import (
        build_gaussian_pyramid,
        build_laplacian_pyramid,
        reconstruct_from_lap,
    )

    B = x.size(0)
    device = x.device
    scheduler = _get_scheduler(device)

    # ── Default omega: K Laplacian + 1 Gaussian base = K+1 weights ───────
    n_total = num_levels + 1
    if omega is None:
        omega_raw = [float(n_total - i) for i in range(n_total)]
        s = sum(omega_raw)
        omega = [w / s for w in omega_raw]
    elif len(omega) == num_levels:
        # Legacy: K weights → append base weight
        omega_base = omega[-1] * 0.5 if omega else 1.0 / n_total
        s = sum(omega) + omega_base
        omega = [w / s for w in omega] + [omega_base / s]

    # ── Randperm pairing ─────────────────────────────────────────────────
    index = torch.randperm(B, device=device)
    y_a = y.float()
    y_b = y[index].float()

    rr_mask = (y_a == 0) & (y_b == 0)
    ff_mask = (y_a == 1) & (y_b == 1)
    rf_mask = (y_a == 0) & (y_b == 1)
    fr_mask = (y_a == 1) & (y_b == 0)

    # ── RR: clean real, label = 0 ──────────────────────────────────────────
    rr_n = rr_mask.sum().item()
    if rr_n > 0:
        rr_x = x[rr_mask]
        rr_y = torch.zeros(rr_n, device=device)
        rr_label = torch.zeros(rr_n, dtype=y.dtype, device=device)
    else:
        rr_x = torch.empty(0, *x.shape[1:], device=device)
        rr_y = torch.empty(0, device=device)
        rr_label = torch.empty(0, dtype=y.dtype, device=device)

    # ── FF: clean fake, label = 1 ──────────────────────────────────────────
    ff_n = ff_mask.sum().item()
    if ff_n > 0:
        ff_x = x[ff_mask]
        ff_y = torch.ones(ff_n, device=device)
        ff_label = torch.ones(ff_n, dtype=y.dtype, device=device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=device)
        ff_y = torch.empty(0, device=device)
        ff_label = torch.empty(0, dtype=y.dtype, device=device)

    # ── RF+FR merged: pyramid trajectory aggregation ───────────────────────
    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])       # [n_rf]
    fake_pos = torch.cat([index[rf_idx], fr_idx])       # [n_rf]
    n_rf = len(real_pos)

    if n_rf > 0:
        x_r = x[real_pos]
        x_f = x[fake_pos]

        # ── K evenly-spaced timesteps ────────────────────────────────────
        steps = _build_traj_steps(t_min, t_max, num_traj_steps, device)

        # ── Shared noise seed ────────────────────────────────────────────
        noise = torch.randn_like(x_r)

        # ── Accumulate across trajectory ─────────────────────────────────
        traj_accum = torch.zeros_like(x_r)              # Σ_t (reconstructed image)
        lambda_bar_sum = 0.0                            # Σ_t λ̄_t

        for t_val in steps:
            t_i = t_val.item()

            # DDPM forward with shared noise
            x_t_r = scheduler.q_sample(x_r, t_i, noise)
            x_t_f = scheduler.q_sample(x_f, t_i, noise)

            # Cosine weight
            lam_t = compute_lambda_t(t_i, T)

            # ── Laplacian pyramid decomposition ─────────────────────────
            gpyr_r = build_gaussian_pyramid(x_t_r, num_levels)
            gpyr_f = build_gaussian_pyramid(x_t_f, num_levels)

            # Coarsest Gaussian from fake (λ_{t,K} = 0)
            G_K = gpyr_f[-1]

            lap_r = build_laplacian_pyramid(gpyr_r)
            lap_f = build_laplacian_pyramid(gpyr_f)

            # ── Per-level mixing ────────────────────────────────────────
            lap_mixed = []
            lambda_bar_t = 0.0                          # λ̄ at this timestep

            for k in range(num_levels):
                lambda_l = 1.0 - k / num_levels          # λ_l = 1 − l/L
                lambda_tl = lam_t * lambda_l             # λ_{t,l}

                L_mix = lambda_tl * lap_r[k] + (1.0 - lambda_tl) * lap_f[k]
                lap_mixed.append(L_mix)

                lambda_bar_t += omega[k] * lambda_tl

            # Gaussian base contribution: λ_{t,K} = 0 (explicit for clarity)
            lambda_bar_t += omega[num_levels] * lam_t * 0.0

            # ── Reconstruct to image space ──────────────────────────────
            img_t = reconstruct_from_lap(G_K, lap_mixed)

            # Clamp to noised-image range
            img_t = torch.clamp(img_t,
                                torch.min(x_t_r.amin(dim=(-3, -2, -1), keepdim=True),
                                          x_t_f.amin(dim=(-3, -2, -1), keepdim=True)),
                                torch.max(x_t_r.amax(dim=(-3, -2, -1), keepdim=True),
                                          x_t_f.amax(dim=(-3, -2, -1), keepdim=True)))

            traj_accum += img_t
            lambda_bar_sum += lambda_bar_t

        # ── Mean aggregation across trajectory ───────────────────────────
        rf_x = traj_accum / num_traj_steps

        # ── Soft label: y_m = 1 − mean(λ̄_t) ──────────────────────────────
        lambda_bar_mean = lambda_bar_sum / num_traj_steps
        rf_y = torch.full((n_rf,), 1.0 - lambda_bar_mean, device=device)

        # ── Hard label: threshold at 0.5 ─────────────────────────────────
        rf_label = (rf_y >= 0.5).long()
    else:
        rf_x = torch.empty(0, *x.shape[1:], device=device)
        rf_y = torch.empty(0, device=device)
        rf_label = torch.empty(0, dtype=y.dtype, device=device)

    # ── Combine ───────────────────────────────────────────────────────────
    parts_x = [t for t in [rr_x, ff_x, rf_x] if t.numel() > 0]
    parts_y = [t for t in [rr_y, ff_y, rf_y] if t.numel() > 0]
    parts_label = [t for t in [rr_label, ff_label, rf_label]
                   if t.numel() > 0]

    mixed_x = torch.cat(parts_x, dim=0)
    mixed_y = torch.cat(parts_y, dim=0)
    mixed_label = torch.cat(parts_label, dim=0)
    return mixed_x, mixed_y, mixed_label

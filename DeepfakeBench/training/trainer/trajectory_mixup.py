"""
Diffusion Trajectory-Guided Mixup for Generalizable AIGC Detection
===================================================================

Implements the theoretical framework:

  - DDPM forward scheduler (linear schedule, T=1000)
  - Shared noise: same ε for real and fake at every timestep
  - K discrete trajectory anchor points: t ∈ [t_min, t_max]
  - Cosine trajectory weight: λ_t = 0.5·(1 + cos(π·t/T))
  - Random trajectory state sampling: one t per batch, no pooling
  - Soft label: y_m = 1 − λ_t  (for the sampled t)

Key design choice — NO pooling across timesteps:
  Instead of aggregating all K trajectory states into one image (which
  would collapse the diffusion path into a single point), we randomly
  sample one state per training iteration.  Over many iterations, the
  detector sees the full trajectory {z_50, z_100, ..., z_700} and learns
  a continuous real→fake decision boundary along the diffusion path.

Two modes:
  trajectory          — random trajectory state (pixel-space)
  trajectory_pyramid  — random trajectory state + Laplacian pyramid

Training strategy (domain consistency):
  RR → clean real,        label = 0
  FF → clean fake,        label = 1
  RF → trajectory state,  soft label (hard = round(soft))

Config keys (in effort.yaml):
    traj_t_min:      50
    traj_t_max:      700
    traj_T:          1000
    traj_num_steps:  14        # K trajectory anchor points

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
            t:     int — scalar timestep (broadcast to all N samples)
            noise: [N, C, H, W] Gaussian noise ε ∼ N(0,I)

        Returns:
            x_t: [N, C, H, W] noised images
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise


_scheduler = None


def _get_scheduler(device='cpu'):
    global _scheduler
    if _scheduler is None:
        _scheduler = DDPMScheduler(device=device)
    elif str(_scheduler.betas.device) != str(device):
        _scheduler.to(device)
    return _scheduler


# ═══════════════════════════════════════════════════════════════════════════════
# Cosine Trajectory Weight
# ═══════════════════════════════════════════════════════════════════════════════

def compute_lambda_t(t, T=1000):
    """Cosine schedule:  λ_t = 0.5 · (1 + cos(π · t / T)).

    t = 0   (clean):  λ → 1  (preserve real structure)
    t = T   (noise):  λ → 0  (let fake distribution dominate)

    Args:
        t: int — timestep index
        T: int — total diffusion steps

    Returns:
        float — λ_t ∈ [0, 1]
    """
    return 0.5 * (1.0 + math.cos(math.pi * t / T))


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: sample a random trajectory state
# ═══════════════════════════════════════════════════════════════════════════════

def _sample_traj_state(t_min, t_max, T, device):
    """Sample one trajectory state uniformly: t ∼ U(t_min, t_max).

    Returns (timestep, lambda_t).
    """
    t = torch.randint(t_min, t_max + 1, (1,), device=device).item()
    lam = compute_lambda_t(t, T)
    return t, lam


# ═══════════════════════════════════════════════════════════════════════════════
# Random Trajectory State Mixup
# ═══════════════════════════════════════════════════════════════════════════════

def trajectory_mixup(x, y, alpha=1.0, gamma=5.0,
                     t_min=50, t_max=700, T=1000):
    """Random Diffusion Trajectory State Mixup.

    Each training iteration samples ONE timestep uniformly from the
    continuous trajectory: t ∼ U(t_min, t_max).  Both images are
    forward-diffused with shared noise to that state, then mixed:

        z_t = λ_t·q_t(x_r, ε) + (1−λ_t)·q_t(x_f, ε)
        y_m = 1 − λ_t

    Over many iterations the detector sees the full continuous
    trajectory distribution {z_t : t ∈ [50, 700]} and learns a
    continuous real→fake decision boundary along the diffusion path.

    Training strategy (domain consistency):
        RR → clean real,  label = 0
        FF → clean fake,  label = 1
        RF → trajectory state at t, soft label

    Args:
        x:      [N, C, H, W] images
        y:      [N] labels (0=real, 1=fake)
        alpha:  unused (API consistency)
        gamma:  unused (API consistency)
        t_min:  minimum timestep (default 50)
        t_max:  maximum timestep (default 700)
        T:      total DDPM steps (default 1000)

    Returns:
        mixed_x:     [M, C, H, W]
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

    # ── RF+FR merged: random trajectory state ──────────────────────────────
    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])
    fake_pos = torch.cat([index[rf_idx], fr_idx])
    n_rf = len(real_pos)

    if n_rf > 0:
        x_r = x[real_pos]                               # [n_rf, C, H, W]
        x_f = x[fake_pos]

        # ── Sample ONE trajectory state: t ∼ U(t_min, t_max) ────────────
        t, lam_t = _sample_traj_state(t_min, t_max, T, device)

        # Shared noise
        noise = torch.randn_like(x_r)

        # DDPM forward to the sampled state
        x_t_r = scheduler.q_sample(x_r, t, noise)
        x_t_f = scheduler.q_sample(x_f, t, noise)

        # Trajectory mixup at this state
        rf_x = lam_t * x_t_r + (1.0 - lam_t) * x_t_f

        # Soft label: y_m = 1 − λ_t  (for this state)
        rf_y = torch.full((n_rf,), 1.0 - lam_t, device=device)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Random Trajectory State + Pyramid Mixup
# ═══════════════════════════════════════════════════════════════════════════════

def pyramid_trajectory_mixup(x, y, alpha=1.0, gamma=5.0, num_levels=3,
                              omega=None, epsilon=1e-8,
                              t_min=50, t_max=700, T=1000):
    """Random Trajectory State + Laplacian Pyramid Mixup.

    Same random-state-sampling as trajectory_mixup, with pyramid:

      1. Sample ONE trajectory state: t ∼ U(t_min, t_max)
      2. DDPM forward with shared noise → x_t^r, x_t^f
      3. Laplacian pyramid decomposition
      4. Per-level mixing: L_k^mix = λ_{t,k}·L_k^r + (1−λ_{t,k})·L_k^f
         where λ_{t,k} = λ_t · (1 − k/K)
      5. Coarsest Gaussian base from fake (λ_{t,K} = 0)
      6. Reconstruct to image space
      7. Soft label: y_m = 1 − λ̄_t  where λ̄_t = Σ_{k=0}^K ω_k·λ_{t,k}

    Args:
        x:          [N, C, H, W] images
        y:          [N] labels (0=real, 1=fake)
        alpha:      unused (API consistency)
        gamma:      unused (API consistency)
        num_levels: K = number of Laplacian pyramid levels
        omega:      level weights [K+1]; default = decreasing
        epsilon:    numerical stability
        t_min:      minimum timestep (default 50)
        t_max:      maximum timestep (default 700)
        T:          total DDPM steps (default 1000)

    Returns:
        mixed_x, mixed_y, mixed_label
    """
    from trainer.trainer_v2 import (
        build_gaussian_pyramid,
        build_laplacian_pyramid,
        reconstruct_from_lap,
    )

    B = x.size(0)
    device = x.device
    scheduler = _get_scheduler(device)

    # ── Default omega: K Laplacian + 1 Gaussian base ─────────────────────
    n_total = num_levels + 1
    if omega is None:
        omega_raw = [float(n_total - i) for i in range(n_total)]
        s = sum(omega_raw)
        omega = [w / s for w in omega_raw]
    elif len(omega) == num_levels:
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

    # ── RR: clean real ─────────────────────────────────────────────────────
    rr_n = rr_mask.sum().item()
    if rr_n > 0:
        rr_x = x[rr_mask]
        rr_y = torch.zeros(rr_n, device=device)
        rr_label = torch.zeros(rr_n, dtype=y.dtype, device=device)
    else:
        rr_x = torch.empty(0, *x.shape[1:], device=device)
        rr_y = torch.empty(0, device=device)
        rr_label = torch.empty(0, dtype=y.dtype, device=device)

    # ── FF: clean fake ─────────────────────────────────────────────────────
    ff_n = ff_mask.sum().item()
    if ff_n > 0:
        ff_x = x[ff_mask]
        ff_y = torch.ones(ff_n, device=device)
        ff_label = torch.ones(ff_n, dtype=y.dtype, device=device)
    else:
        ff_x = torch.empty(0, *x.shape[1:], device=device)
        ff_y = torch.empty(0, device=device)
        ff_label = torch.empty(0, dtype=y.dtype, device=device)

    # ── RF+FR merged: random trajectory state + pyramid ────────────────────
    rf_idx = rf_mask.nonzero(as_tuple=True)[0]
    fr_idx = fr_mask.nonzero(as_tuple=True)[0]
    real_pos = torch.cat([rf_idx, index[fr_idx]])
    fake_pos = torch.cat([index[rf_idx], fr_idx])
    n_rf = len(real_pos)

    if n_rf > 0:
        x_r = x[real_pos]
        x_f = x[fake_pos]

        # ── Sample ONE trajectory state: t ∼ U(t_min, t_max) ────────────
        t, lam_t = _sample_traj_state(t_min, t_max, T, device)

        # Shared noise + DDPM forward
        noise = torch.randn_like(x_r)
        x_t_r = scheduler.q_sample(x_r, t, noise)
        x_t_f = scheduler.q_sample(x_f, t, noise)

        # ── Laplacian pyramid decomposition ──────────────────────────────
        gpyr_r = build_gaussian_pyramid(x_t_r, num_levels)
        gpyr_f = build_gaussian_pyramid(x_t_f, num_levels)

        # Coarsest Gaussian from fake (λ_{t,K} = λ_t·(1−K/K) = 0)
        G_K = gpyr_f[-1]

        lap_r = build_laplacian_pyramid(gpyr_r)
        lap_f = build_laplacian_pyramid(gpyr_f)

        # ── Per-level mixing ─────────────────────────────────────────────
        lap_mixed = []
        lambda_bar_t = 0.0

        for k in range(num_levels):
            lambda_l = 1.0 - k / num_levels              # λ_l = 1 − l/L
            lambda_tl = lam_t * lambda_l                 # λ_{t,l}

            L_mix = lambda_tl * lap_r[k] + (1.0 - lambda_tl) * lap_f[k]
            lap_mixed.append(L_mix)

            lambda_bar_t += omega[k] * lambda_tl

        # Gaussian base contribution (λ_{t,K} = 0, explicit)
        lambda_bar_t += omega[num_levels] * lam_t * 0.0

        # ── Reconstruct ──────────────────────────────────────────────────
        rf_x = reconstruct_from_lap(G_K, lap_mixed)
        rf_x = torch.clamp(rf_x,
                           torch.min(x_t_r.amin(dim=(-3, -2, -1), keepdim=True),
                                     x_t_f.amin(dim=(-3, -2, -1), keepdim=True)),
                           torch.max(x_t_r.amax(dim=(-3, -2, -1), keepdim=True),
                                     x_t_f.amax(dim=(-3, -2, -1), keepdim=True)))

        # Soft label: y_m = 1 − λ̄_t
        rf_y = torch.full((n_rf,), 1.0 - lambda_bar_t, device=device)
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

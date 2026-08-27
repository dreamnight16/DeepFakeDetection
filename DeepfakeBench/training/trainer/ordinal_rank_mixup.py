"""Ordinal Forensic Ranking Loss — G9 auxiliary loss helpers.

For each real image in the batch, one (x_r, x_f) pair is drawn (fake partners
sampled from the batch, with replacement when fakes are fewer than reals),
two mixing coefficients λ_a < λ_b are sampled, and the two pixel-mixed images

    x_λa = (1−λ_a)·x_r + λ_a·x_f
    x_λb = (1−λ_b)·x_r + λ_b·x_f

are constructed (2·n_real images in total). A hinge (or softplus) ranking
loss then pushes s(x_λb) − s(x_λa) ≥ m·(λ_b − λ_a), where s is the fake-class
logit of the classifier output (pre-softmax, per the spec's advice to avoid
probability saturation).

Spec: C:\\Users\\DreamNight\\Documents\\01My\\ai\\ordinal_forensic_ranking_loss.md
"""
import numpy as np
import torch
import torch.nn.functional as F


def build_ordinal_rank_pairs(x, y, alpha=1.0):
    """Build 2·n_real RF pixel-mixes (n_real pairs, one per real anchor).

    Args:
        x:     [N, C, H, W] batch images
        y:     [N] labels (0=real, 1=fake)
        alpha: Beta(α,α) parameter for λ sampling (sorted per-pair draws)

    Returns:
        dict with rank_xa, rank_xb [n_real, C, H, W] and rank_la, rank_lb
        [n_real], or None when the batch contains no real or no fake image.
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

    # λ_a < λ_b: two independent Beta(α,α) draws per pair, sorted
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=(n_real, 2)).astype(np.float32)
    else:
        lam = np.full((n_real, 2), 0.5, dtype=np.float32)
    la = np.min(lam, axis=1)
    lb = np.max(lam, axis=1)

    la_v = torch.tensor(la, dtype=torch.float32, device=x.device).view(n_real, 1, 1, 1)
    lb_v = torch.tensor(lb, dtype=torch.float32, device=x.device).view(n_real, 1, 1, 1)

    xa = (1.0 - la_v) * x_r + la_v * partners
    xb = (1.0 - lb_v) * x_r + lb_v * partners
    return {
        'rank_xa': xa,
        'rank_xb': xb,
        'rank_la': torch.tensor(la, dtype=torch.float32, device=x.device),
        'rank_lb': torch.tensor(lb, dtype=torch.float32, device=x.device),
    }


def ordinal_ranking_loss(score_a, score_b, lambda_a, lambda_b,
                         margin=1.0, softplus=False):
    """Ordinal forensic ranking loss, batch-averaged.

    Hinge (default):  max(0, m·Δλ − Δs)
    Softplus:         log(1 + exp(m·Δλ − Δs))

    Args:
        score_a:  [B] fake-class logits of x_λa
        score_b:  [B] fake-class logits of x_λb
        lambda_a: [B] smaller mixing coefficients
        lambda_b: [B] larger mixing coefficients
        margin:   base margin m
        softplus: use the smooth softplus variant instead of hinge

    Returns:
        scalar loss tensor
    """
    target_margin = margin * (lambda_b - lambda_a)
    score_difference = score_b - score_a
    if softplus:
        return F.softplus(target_margin - score_difference).mean()
    return F.relu(target_margin - score_difference).mean()

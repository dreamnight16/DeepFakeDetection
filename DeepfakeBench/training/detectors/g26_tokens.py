"""Asymmetric query supervision and fixed uncertainty gating for G26."""

import math

import torch
from torch.nn import functional as F


def smooth_max(log_odds: torch.Tensor, temperature: float) -> torch.Tensor:
    """Normalized log-sum-exp: equal queries retain their log odds for any K.

    This is a smooth surrogate for existence, not a probability that independent
    events occur. It lies between max(z) - temperature*log(K) and max(z).
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("MIL temperature must be finite and positive")
    if log_odds.ndim != 2 or log_odds.shape[1] < 1:
        raise ValueError("Expected evidence log odds shaped [batch, K>=1]")
    # Preserve float64 for numerical tests; promote half precision reductions.
    values = log_odds.float() if log_odds.dtype in (torch.float16, torch.bfloat16) else log_odds
    return temperature * (torch.logsumexp(values / temperature, dim=1)
                          - math.log(values.shape[1]))


def evidence_loss(log_odds: torch.Tensor, labels: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    """Real: every query real. Fake: at least one query supports fake.

    Real terms are averaged across queries, so adding K does not multiply their
    total weight. The fake surrogate is smooth but can still concentrate on a
    strong query; this does not guarantee diversity or prevent overfitting.
    """
    bag_logits = smooth_max(log_odds, temperature)
    all_real = F.softplus(log_odds.to(bag_logits.dtype)).mean(dim=1)
    exists_fake = F.softplus(-bag_logits)
    return torch.where(labels == 0, all_real, exists_fake)


def selective_fusion(cls_prob: torch.Tensor, evidence_prob: torch.Tensor,
                     width: float, max_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    """A fixed triangular gate around p_cls=0.5; no labels or learned routing.

    Outside the uncertainty band the score is exactly the main branch's score.
    The weight is at most 0.5 to keep the main branch at least equally weighted.
    """
    if not math.isfinite(width) or not 0 < width <= .5:
        raise ValueError("Gate width must be in (0, 0.5]")
    if not math.isfinite(max_weight) or not 0 <= max_weight <= .5:
        raise ValueError("Auxiliary max weight must be in [0, 0.5]")
    weight = max_weight * (1 - (cls_prob - .5).abs() / width).clamp(0, 1)
    fused = (1 - weight) * cls_prob + weight * evidence_prob
    return torch.where(weight > 0, fused, cls_prob), weight

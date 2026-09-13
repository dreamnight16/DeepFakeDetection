"""Video-probability-aligned objectives. No model construction or device setup."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def video_logits(frame_logits: Tensor) -> Tensor:
    """Return logit(mean_t softmax(z_t)[fake]) without probability clipping."""
    if (
        frame_logits.ndim < 2
        or frame_logits.shape[-1] != 2
        or frame_logits.shape[-2] == 0
    ):
        raise ValueError("expected [..., T>0, 2] frame logits")
    if not frame_logits.is_floating_point() or not torch.isfinite(frame_logits).all():
        raise ValueError("frame logits must be finite floating point values")
    z = frame_logits if frame_logits.dtype == torch.float64 else frame_logits.float()
    m = z[..., 1] - z[..., 0]
    return torch.logsumexp(F.logsigmoid(m), -1) - torch.logsumexp(F.logsigmoid(-m), -1)


def _check_video_shape(u: Tensor) -> None:
    if u.ndim != 3 or u.shape[-1] != 2 or u.shape[0] == 0 or u.shape[1] == 0:
        raise ValueError("expected [P>0,A>0,S=2] video logits")
    if not torch.isfinite(u).all():
        raise ValueError("nonfinite video logits")


def classification_loss(u: Tensor) -> Tensor:
    _check_video_shape(u)
    return (F.softplus(u[..., 0]).mean() + F.softplus(-u[..., 1]).mean()) * 0.5


def pairwise_losses(
    u: Tensor, permutation: Tensor | None = None, margin: float = 0.0
) -> Tensor:
    _check_video_shape(u)
    if not math.isfinite(margin):
        raise ValueError("nonfinite margin")
    real = u[..., 0]
    if permutation is not None:
        permutation = permutation.to(device=u.device, dtype=torch.long)
        if permutation.shape != (u.shape[0],) or not torch.equal(
            permutation.sort().values, torch.arange(u.shape[0], device=u.device)
        ):
            raise ValueError("permutation must be a bijection over P")
        real = real[permutation]
    return F.softplus(margin - u[..., 1] + real)


def group_risk(risks: Tensor, mode: str, temperature: float) -> tuple[Tensor, Tensor]:
    if risks.numel() == 0 or not torch.isfinite(risks).all():
        raise ValueError("group risks must be nonempty and finite")
    if mode == "mean":
        return risks.mean(), torch.full_like(risks, 1 / risks.numel())
    if mode != "group_softmax" or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid risk reduction/temperature")
    flat = risks.reshape(-1)
    value = temperature * (
        torch.logsumexp(flat / temperature, 0) - math.log(flat.numel())
    )
    return value, (flat / temperature).softmax(0).reshape_as(risks)

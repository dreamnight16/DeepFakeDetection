"""Compact E1002 feature and intervention readouts, isolated from frozen B0."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


def _positive_dimension(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_tensor(name: str, value: torch.Tensor, ndim: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional tensor")
    if not value.is_floating_point() or any(size < 1 for size in value.shape):
        raise ValueError(f"{name} must contain nonempty floating-point features")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def symmetric_covariance_normalize(covariance: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Smooth signed square-root followed by per-matrix Frobenius normalization.

    ``x / sqrt(abs(x) + eps)`` preserves sign and has a finite derivative at
    zero. Low-precision inputs use float32 arithmetic and an epsilon floor
    at their dtype's precision so the backward cast cannot overflow. An
    all-zero covariance remains zero, including one-token regions.
    """
    _finite_tensor("covariance", covariance, 3)
    if covariance.shape[-1] != covariance.shape[-2]:
        raise ValueError("covariance matrices must be square")
    if not isinstance(eps, Real) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    original_dtype = covariance.dtype
    if original_dtype in (torch.float16, torch.bfloat16):
        eps = max(eps, torch.finfo(original_dtype).eps)
        covariance = covariance.float()
    covariance = covariance * .5 + covariance.transpose(-1, -2) * .5
    signed_root = covariance / torch.sqrt(covariance.abs() + eps)
    norm = torch.linalg.vector_norm(signed_root, dim=(-2, -1), keepdim=True)
    return (signed_root / norm.clamp_min(eps)).to(original_dtype)


def sample_covariance(patches: torch.Tensor, projection: nn.Module | None = None,
                      normalize: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """Return centered sample covariance [B,R,R], optionally after projection.

    A one-token region has zero covariance. Otherwise the denominator is
    N-1. Projection precedes centering, so a projection bias cannot enter the
    covariance. This helper keeps gradients for a trainable projection; the
    readout detaches the B0 features at its own boundary. Half/bfloat16
    inputs produce float32 statistics, including under mixed precision.
    """
    _finite_tensor("patches", patches, 3)
    features = projection(patches) if projection is not None else patches
    _finite_tensor("projected patches", features, 3)
    if features.shape[:2] != patches.shape[:2]:
        raise ValueError("projection must preserve patch batch and token dimensions")
    with torch.autocast(device_type=features.device.type, enabled=False):
        statistics = features.float() if features.dtype in (torch.float16, torch.bfloat16) else features
        centered = statistics - statistics.mean(dim=1, keepdim=True)
        covariance = centered.transpose(1, 2).bmm(centered) / max(features.shape[1] - 1, 1)
    _finite_tensor("sample covariance", covariance, 3)
    if normalize:
        return symmetric_covariance_normalize(covariance, eps)
    return covariance


def _classifier(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))


class FeatureReadout(nn.Module):
    """CLS, patch mean, global covariance, or 2x2 regional covariance head."""

    def __init__(self, kind: str, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        if kind not in {"cls", "mean", "cov", "regional"}:
            raise ValueError(f"unknown feature readout kind: {kind}")
        self.kind = kind
        self.feature_dim = _positive_dimension("feature_dim", feature_dim)
        self.hidden_dim = _positive_dimension("hidden_dim", hidden_dim)
        if kind in {"cov", "regional"}:
            self.projection_dim = min(feature_dim, hidden_dim, 64)
            self.projection = nn.Linear(feature_dim, self.projection_dim, bias=False)
            indices = torch.triu_indices(self.projection_dim, self.projection_dim)
            self.register_buffer("covariance_indices", indices, persistent=False)
            self.covariance_bottleneck = nn.Sequential(
                nn.Linear(indices.shape[1], hidden_dim), nn.GELU())
            self.classifier = _classifier(hidden_dim * (4 if kind == "regional" else 1), hidden_dim)
        else:
            self.classifier = _classifier(feature_dim, hidden_dim)

    def _covariance_embedding(self, patches: torch.Tensor) -> torch.Tensor:
        covariance = sample_covariance(patches, self.projection, normalize=True)
        row, column = self.covariance_indices
        vector = covariance[:, row, column].to(self.covariance_bottleneck[0].weight.dtype)
        return self.covariance_bottleneck(vector)

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        _finite_tensor("cls", cls, 2)
        _finite_tensor("patches", patches, 3)
        if cls.shape[1] != self.feature_dim or patches.shape[2] != self.feature_dim:
            raise ValueError("CLS and patch dimensions must match feature_dim")
        if cls.shape[0] != patches.shape[0] or cls.device != patches.device or cls.dtype != patches.dtype:
            raise ValueError("CLS and patches must share batch size, device, and dtype")
        cls, patches = cls.detach(), patches.detach()
        if self.kind == "cls":
            embedding = cls
        elif self.kind == "mean":
            embedding = patches.mean(dim=1)
        elif self.kind == "cov":
            embedding = self._covariance_embedding(patches)
        else:
            side = math.isqrt(patches.shape[1])
            if side * side != patches.shape[1] or side % 2:
                raise ValueError("regional covariance requires an even square patch grid")
            half = side // 2
            grid = patches.reshape(patches.shape[0], side, side, self.feature_dim)
            regions = [grid[:, row:row + half, column:column + half].reshape(
                patches.shape[0], half * half, self.feature_dim)
                for row in (0, half) for column in (0, half)]
            embedding = torch.cat([self._covariance_embedding(region) for region in regions], dim=-1)
        logits = self.classifier(embedding)
        _finite_tensor("readout logits", logits, 2)
        return logits


class EnvironmentReadout(nn.Module):
    """Separate supervised forgery and known photometric-intervention branches."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = _positive_dimension("feature_dim", feature_dim)
        self.hidden_dim = _positive_dimension("hidden_dim", hidden_dim)
        self.forgery_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                            nn.Linear(hidden_dim, hidden_dim))
        self.nuisance_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        self.forgery_classifier = nn.Linear(hidden_dim, 2)
        self.nuisance_classifier = nn.Linear(hidden_dim, 4)

    def forward(self, cls: torch.Tensor) -> dict[str, torch.Tensor]:
        _finite_tensor("cls", cls, 2)
        if cls.shape[1] != self.feature_dim:
            raise ValueError("CLS dimension must match feature_dim")
        features = cls.detach()
        forgery, nuisance = self.forgery_encoder(features), self.nuisance_encoder(features)
        output = {"logits": self.forgery_classifier(forgery),
                  "nuisance_logits": self.nuisance_classifier(nuisance),
                  "forgery": forgery, "nuisance": nuisance}
        for name, value in output.items():
            _finite_tensor(name, value, 2)
        return output


def _class_labels(name: str, labels: torch.Tensor, batch: int, classes: int,
                  device: torch.device) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor) or labels.shape != (batch,):
        raise ValueError(f"{name} must contain one label per sample")
    if labels.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"{name} must be integer class labels")
    if labels.device != device or (labels < 0).any() or (labels >= classes).any():
        raise ValueError(f"{name} must be on the output device and in [0, {classes})")
    return labels.long()


def environment_loss(output: dict[str, torch.Tensor], labels: torch.Tensor,
                     nuisance_labels: torch.Tensor, orth_weight: float) -> dict[str, torch.Tensor]:
    """Both supervised CEs, plus normalized sample overlap and batch dependence.

    Weight zero is the supervised control. The nuisance targets identify the
    four applied interventions; they are not person or identity labels.
    ``cross_covariance`` is the mean squared sample covariance between the
    two unit-normalized embeddings. Batch size one contributes zero there.
    Low-precision losses use float32 arithmetic and a dtype-aware epsilon
    floor to keep tiny-embedding backward passes finite.
    """
    if isinstance(orth_weight, bool) or not isinstance(orth_weight, Real) or not math.isfinite(orth_weight) or orth_weight < 0:
        raise ValueError("orth_weight must be finite and nonnegative")
    if not isinstance(output, dict):
        raise ValueError("environment output must be a dictionary")
    for name in ("logits", "nuisance_logits", "forgery", "nuisance"):
        _finite_tensor(name, output.get(name), 2)
    logits, nuisance_logits = output["logits"], output["nuisance_logits"]
    forgery, nuisance = output["forgery"], output["nuisance"]
    batch = logits.shape[0]
    if logits.shape != (batch, 2) or nuisance_logits.shape != (batch, 4):
        raise ValueError("environment logits must be [B,2] and nuisance logits [B,4]")
    if forgery.shape != nuisance.shape or forgery.shape[0] != batch:
        raise ValueError("forgery and nuisance embeddings must share shape [B,D]")
    if any(value.device != logits.device or value.dtype != logits.dtype for value in output.values()):
        raise ValueError("environment tensors must share device and dtype")
    labels = _class_labels("labels", labels, batch, 2, logits.device)
    nuisance_labels = _class_labels("nuisance_labels", nuisance_labels, batch, 4, logits.device)
    eps = 1e-6
    if logits.dtype in (torch.float16, torch.bfloat16):
        eps = max(eps, torch.finfo(logits.dtype).eps)
        logits, nuisance_logits = logits.float(), nuisance_logits.float()
        forgery, nuisance = forgery.float(), nuisance.float()
    with torch.autocast(device_type=logits.device.type, enabled=False):
        forgery_loss = F.cross_entropy(logits, labels)
        nuisance_loss = F.cross_entropy(nuisance_logits, nuisance_labels)
        forgery_unit = F.normalize(forgery, p=2, dim=-1, eps=eps)
        nuisance_unit = F.normalize(nuisance, p=2, dim=-1, eps=eps)
        sample_cosine = (forgery_unit * nuisance_unit).sum(dim=-1).square().mean()
        forgery_centered = forgery_unit - forgery_unit.mean(dim=0, keepdim=True)
        nuisance_centered = nuisance_unit - nuisance_unit.mean(dim=0, keepdim=True)
        cross = forgery_centered.transpose(0, 1).matmul(nuisance_centered) / max(batch - 1, 1)
        cross_covariance = cross.square().mean()
        orthogonal = sample_cosine + cross_covariance
        overall = forgery_loss + nuisance_loss + orth_weight * orthogonal
    losses = {"overall": overall, "forgery": forgery_loss, "nuisance": nuisance_loss,
              "orthogonal": orthogonal, "sample_cosine": sample_cosine, "cross_covariance": cross_covariance}
    if any(not torch.isfinite(value).all() for value in losses.values()):
        raise ValueError("environment loss must be finite")
    return losses

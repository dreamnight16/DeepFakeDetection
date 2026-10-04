"""Torch-only E1005 objectives for explicitly supplied training minibatches.

Video scoring averages frame probabilities. Ranking uses the same score in
each view; its grouping is a current-minibatch risk proxy. This module has no
dataset loading, checkpoint selection, validation or test-data entry points.
"""

from collections.abc import Mapping
import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F


_INTEGER_DTYPES = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _working_precision(values: Tensor) -> Tensor:
    return values.float() if values.dtype in (torch.float16, torch.bfloat16) else values


def _validate_logits(logits: Tensor) -> None:
    if (not isinstance(logits, Tensor) or not logits.is_floating_point()
            or logits.ndim != 4 or logits.shape[-1] != 2
            or any(size == 0 for size in logits.shape[:3])):
        raise ValueError("logits must be nonempty floating [B,V,T,2]")


def _validate_mask(logits: Tensor, mask: Tensor) -> None:
    if (not isinstance(mask, Tensor) or mask.dtype != torch.bool
            or mask.shape != logits.shape[:3] or mask.device != logits.device):
        raise ValueError("mask must be boolean [B,V,T] on the logits device")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("mask must contain at least one valid frame in every video/view")
    if not bool(torch.isfinite(logits[mask]).all()):
        raise ValueError("valid logits must be finite")


def video_log_odds(logits: Tensor, mask: Tensor) -> Tensor:
    """Return logit(mean valid-frame fake probability), independently per view.

    The difference of two logsumexp reductions avoids probability clipping and
    sigmoid saturation. Padding may be nonfinite and receives zero gradient.
    Half precision is promoted before subtracting class logits.
    """
    _validate_logits(logits)
    _validate_mask(logits, mask)
    clean = torch.where(mask.unsqueeze(-1), _working_precision(logits), 0.)
    margin = clean[..., 1] - clean[..., 0]
    if not bool(torch.isfinite(margin[mask]).all()):
        raise ValueError("valid logit differences must be finite in working precision")
    positive = F.logsigmoid(margin).masked_fill(~mask, -torch.inf)
    negative = F.logsigmoid(-margin).masked_fill(~mask, -torch.inf)
    return torch.logsumexp(positive, dim=-1) - torch.logsumexp(negative, dim=-1)


def _integer_tensor(value, name: str, device, shape=None) -> Tensor:
    try:
        value = value if isinstance(value, Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be an integer tensor") from error
    if value.dtype not in _INTEGER_DTYPES or (shape is not None and value.shape != shape):
        raise ValueError(f"{name} must be an integer tensor with shape {shape}")
    return value.to(device=device, dtype=torch.long)


def _weight(value, name: str, positive=False) -> float:
    if (not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {requirement}")
    return float(value)


def _pair_indices(batch, labels: Tensor) -> tuple[Tensor, Tensor]:
    if "real_indices" not in batch or "fake_indices" not in batch:
        raise ValueError("ranking/keep require explicit real_indices and fake_indices")
    real = _integer_tensor(batch["real_indices"], "real_indices", labels.device)
    fake = _integer_tensor(batch["fake_indices"], "fake_indices", labels.device)
    for name, indices, label in (("real_indices", real, 0), ("fake_indices", fake, 1)):
        if (indices.ndim != 1 or indices.numel() == 0
                or bool(((indices < 0) | (indices >= labels.numel())).any())
                or indices.unique().numel() != indices.numel()):
            raise ValueError(f"{name} must contain unique in-range video rows")
        if not bool((labels[indices] == label).all()):
            raise ValueError(f"{name} disagree with binary labels")
        if indices.numel() != int((labels == label).sum()):
            raise ValueError(f"{name} must cover all current videos of its class")
    if real.numel() != fake.numel():
        raise ValueError("matched real_indices/fake_indices must have equal lengths")
    return real, fake


def _group_rank(raw: Tensor, methods, conditions, method_risk: bool,
                nuisance_risk: bool, tau: float):
    counts, risks = {}, {}
    if not method_risk and not nuisance_risk:
        counts["all"] = raw.numel()
        return raw.mean(), counts, {"all": float(raw.detach().mean())}
    keys = []
    if method_risk:
        keys.append(methods[:, None].expand_as(raw))
    if nuisance_risk:
        keys.append(conditions)
    group_ids = torch.stack(keys, dim=-1).reshape(-1, len(keys))
    flat = raw.reshape(-1)
    group_means = []
    for identity in torch.unique(group_ids, dim=0, sorted=True):
        selected = (group_ids == identity).all(dim=-1)
        parts = []
        if method_risk:
            parts.append(f"method:{int(identity[0])}")
        if nuisance_risk:
            parts.append(f"nuisance:{int(identity[-1])}")
        name = "/".join(parts)
        risk = flat[selected].mean()
        group_means.append(risk)
        counts[name] = int(selected.sum())
        risks[name] = float(risk.detach())
    stacked = torch.stack(group_means)
    # Actual groups only: absent methods/conditions never contribute zero risk.
    aggregate = tau * (torch.logsumexp(stacked / tau, dim=0) - math.log(len(group_means)))
    return aggregate, counts, risks


def _metric_losses(features: Tensor, mask: Tensor, labels: Tensor, zero: Tensor):
    if (not isinstance(features, Tensor) or not features.is_floating_point()
            or features.ndim != 4 or features.shape[:3] != mask.shape
            or features.shape[-1] == 0 or features.device != mask.device):
        raise ValueError("features must be floating [B,V,T,D] on the logits device")
    valid = _working_precision(features[mask])
    if not bool(torch.isfinite(valid).all()):
        raise ValueError("valid features must be finite")
    frame_labels = labels[:, None, None].expand_as(mask)[mask]
    # Scaling first prevents norm overflow/underflow for finite feature inputs.
    scale = valid.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(valid.dtype).tiny)
    scaled = valid / scale
    normalized = scaled / torch.linalg.vector_norm(scaled, dim=-1, keepdim=True).clamp_min(1e-12)
    pairs = torch.triu_indices(valid.shape[0], valid.shape[0], offset=1, device=valid.device)
    pair_count = pairs.shape[1]
    if pair_count == 0:
        return zero, zero, 0, 0
    distances = (normalized[pairs[0]] - normalized[pairs[1]]).square().sum(dim=-1)
    same_class = frame_labels[pairs[0]] == frame_labels[pairs[1]]
    same_count = int(same_class.sum())
    alignment = distances[same_class].mean() if same_count else zero
    uniformity = torch.logsumexp(-2. * distances, dim=0) - math.log(pair_count)
    return alignment, uniformity, same_count, pair_count


def _valid_spatial(values: Tensor, mask: Tensor, name: str) -> Tensor:
    if not isinstance(values, Tensor) or not values.is_floating_point() or values.device != mask.device:
        raise ValueError(f"{name} must be floating on the logits device")
    if values.ndim == 5 and values.shape[:3] == mask.shape:
        values = values.unsqueeze(3)
    elif values.ndim == 3:
        values = values.unsqueeze(1)
    if values.ndim == 6 and values.shape[:3] == mask.shape:
        valid = values[mask]
    elif values.ndim == 4:
        if values.shape[0] == mask.numel():
            valid = values[mask.reshape(-1)]
        elif values.shape[0] == int(mask.sum()):
            valid = values
        else:
            raise ValueError(f"{name} frame count must match full or valid frames")
    else:
        raise ValueError(f"{name} must be [B,V,T,1,H,W] or [N,1,H,W]")
    if valid.shape[1] != 1 or any(size == 0 for size in valid.shape[-2:]):
        raise ValueError(f"{name} must have one channel and nonempty spatial dimensions")
    if not bool(torch.isfinite(valid).all()):
        raise ValueError(f"valid {name} must be finite")
    return _working_precision(valid)


def compute_losses(output, batch, objective, teacher=None, keep_weight=.1, tau=.5,
                   rank_weight=1., metric=False, spatial_target=None):
    """Return differentiable loss terms and explicit minibatch diagnostics.

    ``output['logits']`` is [B,V,T,2]; ``batch['labels']`` is binary [B].
    ``valid_mask`` defaults to all frames valid for ordinary frame batches.
    Paired objectives require exhaustive ``real_indices``/``fake_indices``.
    ``shuffle_indices`` permutes positions in real_indices and must have no
    fixed point. K always uses the original exhaustive cross-video pairs.

    Optional metric features are L2-normalized valid frames, using all distinct
    same-class pairs for alignment and all distinct pairs for uniformity. These
    terms have fixed weights .1/.5. Spatial BCE has fixed weight .1 and uses area
    resized registered soft targets. Neither optional term reads dataset assets.

    ``keep_eligible_pairs`` counts distinct video pairs eligible in any view;
    ``keep_eligible_pair_views`` counts the observations averaged by K.
    """
    if not isinstance(output, Mapping) or not isinstance(batch, Mapping) or not isinstance(objective, Mapping):
        raise ValueError("output, batch and objective must be mappings")
    logits = output.get("logits")
    _validate_logits(logits)
    mask = batch.get("valid_mask", torch.ones(logits.shape[:3], dtype=torch.bool, device=logits.device))
    u = video_log_odds(logits, mask)
    if "labels" not in batch:
        raise ValueError("batch must contain binary labels")
    labels = _integer_tensor(batch["labels"], "labels", logits.device, (logits.shape[0],))
    if bool(((labels != 0) & (labels != 1)).any()):
        raise ValueError("labels must contain only real=0 or fake=1")
    classification = objective.get("classification", "video")
    rank_kind = objective.get("rank", "none")
    if classification not in ("frame", "video") or rank_kind not in ("none", "matched", "shuffled"):
        raise ValueError("classification must be frame/video and rank none/matched/shuffled")
    flags = {}
    for name in ("keep", "method_risk", "nuisance_risk", "independent_nuisance"):
        flag = objective.get(name, False)
        if not isinstance(flag, bool):
            raise ValueError(f"objective {name} must be boolean")
        flags[name] = flag
    if not isinstance(metric, bool):
        raise ValueError("metric must be boolean")
    if rank_kind == "none" and (flags["method_risk"] or flags["nuisance_risk"]):
        raise ValueError("group risks require a ranking objective")
    keep_weight = _weight(keep_weight, "keep_weight")
    rank_weight = _weight(rank_weight, "rank_weight")
    tau = _weight(tau, "tau", positive=True)

    methods, conditions = None, None
    if "methods" in batch:
        methods = _integer_tensor(batch["methods"], "methods", logits.device, labels.shape)
        if (bool((methods[labels == 0] != -1).any())
                or bool(((methods[labels == 1] < 0) | (methods[labels == 1] > 3)).any())):
            raise ValueError("methods must be -1 for real and 0..3 for fake videos")
    if "conditions" in batch:
        conditions = _integer_tensor(batch["conditions"], "conditions", logits.device, u.shape)
        if bool((conditions < 0).any()):
            raise ValueError("conditions must be nonnegative identifiers")
    if flags["method_risk"] and methods is None:
        raise ValueError("method risk requires current methods")
    if flags["nuisance_risk"] and conditions is None:
        raise ValueError("nuisance risk requires current conditions")

    zero = u.reshape(-1)[0] * 0.
    if classification == "frame":
        expanded_labels = labels[:, None, None].expand_as(mask)[mask]
        class_loss = F.cross_entropy(_working_precision(logits[mask]), expanded_labels)
    else:
        if not bool((labels == 0).any()) or not bool((labels == 1).any()):
            raise ValueError("balanced video classification requires both real and fake videos")
        class_loss = .5 * (F.softplus(u[labels == 0]).mean() + F.softplus(-u[labels == 1]).mean())

    diagnostics = {"valid_frames": int(mask.sum()), "video_views": u.numel(),
                   "rank_risk_scope": "current_minibatch", "rank_group_counts": {}, "rank_group_risks": {},
                   "keep_eligible_pairs": 0, "keep_eligible_pair_views": 0, "keep_total_pairs": 0,
                   "keep_total_pair_views": 0, "metric_alignment_pairs": 0, "metric_uniformity_pairs": 0,
                   "spatial_valid_frames": 0}
    rank_loss, keep_loss, alignment, uniformity, spatial = zero, zero, zero, zero, zero
    real, fake = None, None
    if rank_kind != "none" or flags["keep"]:
        real, fake = _pair_indices(batch, labels)
    if rank_kind != "none":
        partner = real
        if rank_kind == "shuffled":
            if "shuffle_indices" not in batch:
                raise ValueError("shuffled rank requires shuffle_indices")
            permutation = _integer_tensor(batch["shuffle_indices"], "shuffle_indices", logits.device, real.shape)
            positions = torch.arange(real.numel(), device=real.device)
            if (not torch.equal(permutation.sort().values, positions)
                    or bool((permutation == positions).any())):
                raise ValueError("shuffle_indices must be a permutation with no fixed points")
            partner = real[permutation]
        raw = F.softplus(u[partner] - u[fake])
        rank_loss, counts, risks = _group_rank(raw, methods[fake] if methods is not None else None,
                                              conditions[fake] if conditions is not None else None,
                                              flags["method_risk"], flags["nuisance_risk"], tau)
        diagnostics.update(rank_group_counts=counts, rank_group_risks=risks)
    if flags["keep"]:
        teacher_logits = teacher.get("logits") if isinstance(teacher, Mapping) else teacher
        if (not isinstance(teacher_logits, Tensor) or teacher_logits.shape != logits.shape
                or teacher_logits.device != logits.device):
            raise ValueError("keep requires teacher logits matching student shape/device")
        u0 = video_log_odds(teacher_logits.detach(), mask)
        d0 = u0[fake][None, :, :] - u0[real][:, None, :]
        student_margin = u[fake][None, :, :] - u[real][:, None, :]
        eligible = d0 > .5
        count = int(eligible.sum())
        if count:
            keep_loss = F.relu(d0[eligible].clamp_max(4.) - student_margin[eligible] - .1).square().mean()
        diagnostics.update(keep_eligible_pairs=int(eligible.any(dim=-1).sum()), keep_eligible_pair_views=count,
                           keep_total_pairs=real.numel() * fake.numel(), keep_total_pair_views=d0.numel())
    if metric:
        alignment, uniformity, alignment_pairs, uniformity_pairs = _metric_losses(
            output.get("features"), mask, labels, zero)
        diagnostics.update(metric_alignment_pairs=alignment_pairs, metric_uniformity_pairs=uniformity_pairs)
    if spatial_target is not None:
        mask_logits = _valid_spatial(output.get("mask_logits"), mask, "mask_logits")
        target = _valid_spatial(spatial_target, mask, "spatial_target").to(dtype=mask_logits.dtype)
        if bool(((target < 0) | (target > 1)).any()):
            raise ValueError("spatial_target values must lie in [0,1]")
        target = F.interpolate(target, size=mask_logits.shape[-2:], mode="area")
        spatial = F.binary_cross_entropy_with_logits(mask_logits, target)
        diagnostics["spatial_valid_frames"] = target.shape[0]
    overall = class_loss + rank_weight * rank_loss + keep_weight * keep_loss + .1 * alignment + .5 * uniformity + .1 * spatial
    return {"overall": overall, "loss": overall, "classification": class_loss, "rank": rank_loss,
            "keep": keep_loss, "metric_alignment": alignment, "metric_uniformity": uniformity,
            "spatial": spatial, "video_log_odds": u, "diagnostics": diagnostics}

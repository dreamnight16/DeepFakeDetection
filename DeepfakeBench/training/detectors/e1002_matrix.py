"""Independent warm-start matrix experiments for a fitted E0924 B0.

Each projection retains its original forward, including its fitted LoRA, and
adds a zero-initialized residual. Spectral bases use the effective fitted
matrix only at construction. These experiments are inspired by spectral
adaptation; the sparse coefficient arm is not a literal SVFT reproduction.
"""

import copy
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")
KINDS = ("lora", "svd_tail", "svft")


def _integer_setting(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _effective_weight(linear):
    """Return a detached logical [out,in] matrix without merging the module."""
    weight = linear.weight.detach()
    if getattr(linear, "fan_in_fan_out", False):
        weight = weight.T
    if not hasattr(linear, "lora_A") or getattr(linear, "r", 0) == 0:
        return weight
    if isinstance(linear, nn.Linear):  # loralib: A [r,in], B [out,r]
        if not getattr(linear, "merged", False):
            weight = weight + (linear.lora_B.detach() @ linear.lora_A.detach()) * linear.scaling
    elif not getattr(linear, "merge_weights", False):  # Effort's custom layout
        weight = weight + (linear.lora_A.detach() @ linear.lora_B.detach()).T * linear.scaling
    return weight


class MatrixResidualLinear(nn.Module):
    """Frozen original linear with an independently trainable matrix residual."""

    def __init__(self, original, kind, rank, preserve_top):
        super().__init__()
        self.original = original
        self.kind = kind
        self.rank = rank
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.preserve_top = min(preserve_top, min(self.in_features, self.out_features) - rank)
        for parameter in self.original.parameters():
            parameter.requires_grad_(False)
        # Prevent loralib train/eval from rewriting the copied stored weight.
        if isinstance(self.original, nn.Linear) and hasattr(self.original, "merge_weights"):
            self.original.merge_weights = False
        self.original.eval()
        device, dtype = original.weight.device, original.weight.dtype
        if kind in ("lora", "svd_tail"):
            self.down = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
            self.up = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        if kind in ("svd_tail", "svft"):
            effective = _effective_weight(original)
            if not torch.isfinite(effective).all():
                raise ValueError("The fitted projection weight must be finite")
            # CPU SVD does not support fp16/bfloat16. Keep fp64 when supplied.
            svd_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            with torch.no_grad():
                left, _, right_t = torch.linalg.svd(effective.to(svd_dtype), full_matrices=False)
            left, right_t = left.to(dtype), right_t.to(dtype)
            if kind == "svd_tail":
                self.register_buffer("leading_left", left[:, :self.preserve_top].contiguous())
                self.register_buffer("leading_right", right_t[:self.preserve_top].T.contiguous())
            else:
                self.register_buffer("left_basis", left.contiguous())
                self.register_buffer("right_basis", right_t.T.contiguous())
                dimension = left.shape[1]
                # rank cyclic diagonals of C in Delta W = U C V^T. This is a
                # sparse coefficient budget, not a bound on Delta W's rank.
                columns = (torch.arange(dimension, device=device)[:, None]
                           + torch.arange(rank, device=device)[None, :]) % dimension
                self.register_buffer("coefficient_columns", columns)
                self.coefficients = nn.Parameter(torch.zeros(dimension, rank, device=device, dtype=dtype))

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    def train(self, mode=True):
        super().train(mode)
        self.original.eval()
        return self

    def _factors(self):
        if self.kind == "svd_tail":
            # Complement projections on both sides keep leading singular
            # directions unchanged, even after optimizer weight decay.
            down = self.down - (self.down @ self.leading_right) @ self.leading_right.T
            up = self.up - self.leading_left @ (self.leading_left.T @ self.up)
            return down, up
        return self.down, self.up

    def update_weight(self):
        """Materialize only the new residual for spectral diagnostics."""
        if self.kind == "svft":
            dimension = self.left_basis.shape[1]
            coefficients = self.coefficients.new_zeros(dimension, dimension)
            coefficients = coefficients.scatter(1, self.coefficient_columns, self.coefficients)
            return self.left_basis @ coefficients @ self.right_basis.T
        down, up = self._factors()
        return (up @ down) / self.rank

    def forward(self, inputs):
        # Do not replace this with a linear on W_effective + Delta W: that
        # changes fitted LoRA's numerical evaluation order at initialization.
        original_output = self.original(inputs)
        if self.kind == "svft":
            projected = inputs @ self.right_basis
            mixed = (projected[..., self.coefficient_columns] * self.coefficients).sum(-1)
            residual = mixed @ self.left_basis.T
        else:
            down, up = self._factors()
            residual = F.linear(F.linear(inputs, down), up) / self.rank
        return original_output + residual


class MatrixAdaptation(nn.Module):
    """Deep-copy B0, train selected residuals and the copied classifier.

    The source model is never registered or retained. All copied backbone
    weights and fitted adapters are frozen; only the new suffix residuals and
    copied classifier enter ``trainable_parameters()``. Frozen backbone
    dropout remains in evaluation mode during adaptation.
    """

    def __init__(self, base, kind, rank=4, last_layers=4, preserve_top=32):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        rank = _integer_setting("rank", rank, 1)
        last_layers = _integer_setting("last_layers", last_layers, 1)
        preserve_top = _integer_setting("preserve_top", preserve_top, 0)
        if not hasattr(base, "backbone") or not hasattr(base.backbone, "encoder"):
            raise ValueError("B0 must expose backbone.encoder.layers")
        layers = base.backbone.encoder.layers
        if last_layers > len(layers):
            raise ValueError("last_layers exceeds the B0 encoder depth")
        if not hasattr(base, "head") or not isinstance(base.head, nn.Module):
            raise ValueError("B0 must expose a classifier head")
        for layer in layers[-last_layers:]:
            if not hasattr(layer, "self_attn"):
                raise ValueError("Selected encoder layer has no self_attn")
            for name in PROJECTIONS:
                projection = getattr(layer.self_attn, name, None)
                if (projection is None or not hasattr(projection, "weight")
                        or not hasattr(projection, "in_features") or not hasattr(projection, "out_features")):
                    raise ValueError(f"B0 attention must expose a linear {name}")
                if rank > min(projection.in_features, projection.out_features):
                    raise ValueError("rank exceeds the projection dimensions")
        self.settings = {"kind": kind, "rank": rank, "last_layers": last_layers,
                         "preserve_top": preserve_top, "residual_scaling": "1/rank" if kind != "svft" else "1",
                         "svft_pattern": "cyclic_coefficient_bands" if kind == "svft" else None}
        self.model = copy.deepcopy(base)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear) and hasattr(module, "merge_weights"):
                module.merge_weights = False
        for layer in self.model.backbone.encoder.layers[-last_layers:]:
            for name in PROJECTIONS:
                original = getattr(layer.self_attn, name)
                setattr(layer.self_attn, name, MatrixResidualLinear(original, kind, rank, preserve_top))
        for parameter in self.model.head.parameters():
            parameter.requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.model.backbone.eval()
        self.model.head.train(mode)
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, data, inference=False):
        return self.model(data, inference=inference)

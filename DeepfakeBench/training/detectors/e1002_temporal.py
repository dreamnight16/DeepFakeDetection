"""Small temporal readouts for frozen B0 features in E1002.

Differences refer to consecutive *sampled* frames. The selective diagonal
state-space readout is a torch-native experiment, not official Mamba.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    clean = torch.where(mask.unsqueeze(-1), values, 0.)
    return clean.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


class _SelectiveDiagonalSSM(nn.Module):
    """Causal diagonal recurrence with exact zero-order-hold discretization.

    Delta, B and C depend on the current projected frame. Positive delta and
    negative A make each transition a decay; a learned gate controls output.
    There is one state per hidden channel and no external scan dependency.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.delta = nn.Linear(hidden_dim, hidden_dim)
        self.input_coefficient = nn.Linear(hidden_dim, hidden_dim)
        self.output_coefficient = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.log_decay = nn.Parameter(torch.zeros(hidden_dim))
        self.skip = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        delta = F.softplus(self.delta(features)) + 1e-4
        coefficient_b = torch.sigmoid(self.input_coefficient(features))
        coefficient_c = torch.sigmoid(self.output_coefficient(features))
        gate = torch.sigmoid(self.gate(features))
        rate = self.log_decay.exp()  # A = -rate < 0
        state = features.new_zeros(features.shape[0], features.shape[2])
        outputs = []
        for index in range(features.shape[1]):
            exponent = -delta[:, index] * rate
            decay = exponent.exp()
            # Integral of exp(A*t) over [0, delta]; expm1 is stable near zero.
            gain = -torch.expm1(exponent) / rate
            candidate = decay * state + gain * coefficient_b[:, index] * features[:, index]
            valid = mask[:, index].unsqueeze(-1)
            state = torch.where(valid, candidate, state)
            output = (coefficient_c[:, index] * state + self.skip * features[:, index]) * gate[:, index]
            outputs.append(torch.where(valid, output, 0.))
        return torch.stack(outputs, dim=1)


class TemporalReadout(nn.Module):
    """Map detached [batch, time, feature] clips to binary logits.

    ``mask`` is a boolean [batch, time] tensor with a nonempty valid prefix in
    every row. Only valid inputs must be finite; arbitrary padding is ignored.
    ``mean`` averages learned frame projections. ``diff`` projects the feature
    mean and mean absolute/squared consecutive sampled-frame differences.
    ``tcn`` uses two masked width-three convolutions; ``ssm`` uses a causal
    selective diagonal state-space recurrence. Neither uses normalization
    across clips, so batch companions cannot change a clip's readout.
    """

    KINDS = ("mean", "diff", "tcn", "ssm")

    def __init__(self, kind: str, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        if not isinstance(kind, str) or kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}")
        for name, dimension in (("feature_dim", feature_dim), ("hidden_dim", hidden_dim)):
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.kind = kind
        self.feature_dim = feature_dim
        self.projection = nn.Linear(feature_dim * (3 if kind == "diff" else 1), hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)
        if kind == "tcn":
            self.convolutions = nn.ModuleList([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            ])
        elif kind == "ssm":
            self.ssm = _SelectiveDiagonalSSM(hidden_dim)

    def _validate_and_detach(self, features: Tensor, mask: Tensor) -> Tensor:
        if (not isinstance(features, Tensor) or features.ndim != 3
                or not torch.is_floating_point(features)
                or features.shape[-1] != self.feature_dim):
            raise ValueError("features must be floating [B,T,feature_dim]")
        if features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features must contain nonempty clips")
        if (not isinstance(mask, Tensor) or mask.dtype != torch.bool
                or mask.shape != features.shape[:2] or mask.device != features.device):
            raise ValueError("mask must be boolean [B,T] on the features device")
        if not bool(mask[:, 0].all()):
            raise ValueError("mask must contain a nonempty valid prefix for every clip")
        if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("mask must be prefix-valid")
        detached = features.detach()
        if not bool(torch.isfinite(detached[mask]).all()):
            raise ValueError("valid features must be finite")
        # Multiplication would leave padded NaN/Inf behind (NaN * 0 is NaN).
        return torch.where(mask.unsqueeze(-1), detached, 0.)

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        features = self._validate_and_detach(features, mask)
        if self.kind == "diff":
            pair_mask = mask[:, 1:] & mask[:, :-1]
            differences = torch.where(
                pair_mask.unsqueeze(-1), features[:, 1:] - features[:, :-1], 0.)
            summary = torch.cat((
                _masked_mean(features, mask),
                _masked_mean(differences.abs(), pair_mask),
                _masked_mean(differences.square(), pair_mask),
            ), dim=-1)
            pooled = F.gelu(self.projection(summary))
        else:
            hidden = F.gelu(self.projection(features))
            hidden = torch.where(mask.unsqueeze(-1), hidden, 0.)
            if self.kind == "tcn":
                hidden = hidden.transpose(1, 2)
                for convolution in self.convolutions:
                    hidden = F.gelu(convolution(hidden))
                    hidden = torch.where(mask.unsqueeze(1), hidden, 0.)
                hidden = hidden.transpose(1, 2)
            elif self.kind == "ssm":
                hidden = self.ssm(hidden, mask)
            pooled = _masked_mean(hidden, mask)
        return self.head(pooled)

"""G30: independent evidence/decision decoders reading an immutable B0.

No query enters CLIP. A temporary read-only hook captures the input of an
original encoder block; B0 itself executes its original forward unchanged.
Only ``auxiliary.parameters()`` belongs in the evidence optimizer. Checkpoints
contain the decoder alone and are bound to the SHA256 of the B0 checkpoint.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ReadOnlyDecoderBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, query, memory):
        readout = self.attention(self.query_norm(query), memory, memory, need_weights=False)[0]
        query = query + readout
        return query + self.ffn(self.output_norm(query))


class EvidenceDecoder(nn.Module):
    def __init__(self, memory_dim, num_tokens, hidden_dim=256, num_heads=4, depth=2):
        super().__init__()
        if min(memory_dim, num_tokens, hidden_dim, num_heads, depth) < 1 or hidden_dim % num_heads:
            raise ValueError("Require positive decoder dimensions and hidden_dim divisible by num_heads")
        self.queries = nn.Parameter(torch.empty(1, num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=.02)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.memory_projection = nn.Linear(memory_dim, hidden_dim)
        self.blocks = nn.ModuleList([ReadOnlyDecoderBlock(hidden_dim, num_heads) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(num_tokens)])

    def forward(self, memory):
        # Enforce the boundary even when used outside FrozenEvidenceSidecar.
        memory = self.memory_projection(self.memory_norm(memory.detach()))
        query = self.queries.expand(len(memory), -1, -1)
        for block in self.blocks:
            query = block(query, memory)
        query = self.output_norm(query)
        return torch.stack([head(query[:, i]) for i, head in enumerate(self.heads)], 1)


class DecisionToken(nn.Module):
    """One independent token reading cached scalar features, never a vision model."""

    def __init__(self, input_dim, hidden_dim=32, num_heads=4):
        super().__init__()
        if min(input_dim, hidden_dim, num_heads) < 1 or hidden_dim % num_heads:
            raise ValueError("Invalid decision-token dimensions")
        self.query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.feature_identity = nn.Parameter(torch.empty(1, input_dim, hidden_dim))
        nn.init.trunc_normal_(self.query, std=.02)
        nn.init.trunc_normal_(self.feature_identity, std=.02)
        self.value_projection = nn.Linear(1, hidden_dim)
        self.block = ReadOnlyDecoderBlock(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.feature_identity.shape[1]:
            raise ValueError("Decision features must match the saved input dimension")
        memory = self.value_projection(features.detach()[..., None]) + self.feature_identity
        query = self.block(self.query.expand(len(features), -1, -1), memory)
        return self.head(self.norm(query[:, 0]))[:, 0]


class FrozenEvidenceSidecar(nn.Module):
    def __init__(self, base, num_tokens=4, memory_layer=20, hidden_dim=256,
                 num_heads=4, depth=2, mil_temperature=.5, gate_width=.2,
                 aux_max_weight=.5, balance_weight=0., hard_weighting=False,
                 consistency_weight=0., hard_floor=.2):
        super().__init__()
        if not 0 <= memory_layer < len(base.backbone.encoder.layers):
            raise ValueError("memory_layer must name an existing B0 encoder block (zero based)")
        for name, value in (("mil_temperature", mil_temperature), ("gate_width", gate_width)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if gate_width > .5 or not 0 <= aux_max_weight <= .5 or not 0 <= hard_floor <= 1:
            raise ValueError("Invalid gate width/auxiliary weight/hard floor")
        if any(not math.isfinite(v) or v < 0 for v in (balance_weight, consistency_weight)):
            raise ValueError("Auxiliary loss weights must be finite and nonnegative")
        self.settings = dict(num_tokens=num_tokens, memory_layer=memory_layer, hidden_dim=hidden_dim,
                             num_heads=num_heads, depth=depth, mil_temperature=mil_temperature,
                             gate_width=gate_width, aux_max_weight=aux_max_weight,
                             balance_weight=balance_weight, hard_weighting=hard_weighting,
                             consistency_weight=consistency_weight, hard_floor=hard_floor)
        self.base = base
        self.base.requires_grad_(False)
        for parameter in self.base.parameters():
            parameter.grad = None
        self.base.eval()
        dim = base.backbone.embeddings.class_embedding.numel()
        with torch.random.fork_rng(devices=[]):
            self.auxiliary = EvidenceDecoder(dim, num_tokens, hidden_dim, num_heads, depth)

    def train(self, mode=True):
        # Never transiently put B0/LoRA in train mode (some LoRA backends merge).
        self.training = mode
        self.auxiliary.train(mode)
        self.base.eval()
        return self

    def forward(self, data, inference=False, auxiliary_enabled=True):
        if data["image"].ndim != 4:
            raise ValueError("E1001/G30 requires single-crop [B,C,H,W] images")
        self.base.eval()
        if not auxiliary_enabled:
            with torch.no_grad():
                return self.base(data, inference=True)
        captured = []

        def capture(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            captured.append(hidden.detach())

        layer = self.base.backbone.encoder.layers[self.settings["memory_layer"]]
        hook = layer.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with torch.no_grad():
                original = self.base(data, inference=True)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one original B0 encoder pass")
        # Patch memory only; auxiliary queries have their own attention/FFN.
        logits = self.auxiliary(captured[0][:, 1:])
        z = logits[..., 1] - logits[..., 0]
        temperature = self.settings["mil_temperature"]
        bag = temperature * (torch.logsumexp(z / temperature, -1) - math.log(z.shape[1]))
        p, e = original["prob"], bag.sigmoid()
        gate = (1 - (p - .5).abs() / self.settings["gate_width"]).clamp(0, 1)
        gate = self.settings["aux_max_weight"] * gate
        fused = (1 - gate) * p + gate * e
        readout = dict(global_logits=original["cls"], cls_prob=p, evidence_prob=e,
                       evidence_log_odds=z, gated_prob=fused, gate_weight=gate)
        log_probs = torch.stack((1 - fused, fused), -1).clamp_min(torch.finfo(fused.dtype).tiny).log()
        return {**original, "cls": log_probs, "prob": fused, "g30": readout}

    def losses(self, output, labels, view_output=None):
        values = output["g30"]
        z = values["evidence_log_odds"]
        if labels.ndim != 1 or len(labels) != len(z) or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("G30 requires hard binary labels aligned with the batch")
        t = self.settings["mil_temperature"]
        bag = t * (torch.logsumexp(z / t, -1) - math.log(z.shape[1]))
        auxiliary = torch.where(labels == 0, F.softplus(z).mean(-1), F.softplus(-bag))
        if self.settings["hard_weighting"]:
            uncertainty = (1 - (values["cls_prob"].detach() - .5).abs() /
                           self.settings["gate_width"]).clamp(0, 1)
            floor = self.settings["hard_floor"]
            auxiliary = auxiliary * (floor + (1 - floor) * uncertainty)
        fake = z[labels == 1]
        balance = ((fake.softmax(-1).mean(0) - 1 / z.shape[1]).square().sum()
                   if len(fake) else z.sum() * 0)
        consistency = z.sum() * 0
        if self.settings["consistency_weight"]:
            if view_output is None:
                raise ValueError("Consistency supervision requires a second view")
            consistency = (z.sigmoid() - view_output["g30"]["evidence_log_odds"].sigmoid()).square().mean()
        overall = (auxiliary.mean() + self.settings["balance_weight"] * balance +
                   self.settings["consistency_weight"] * consistency)
        # Main CE is a detached diagnostic only, never part of the objective.
        return dict(overall=overall, loss_evidence=auxiliary.mean(), loss_balance=balance,
                    loss_consistency=consistency,
                    loss_global=F.cross_entropy(values["global_logits"], labels).detach())

    def checkpoint(self, base_sha256):
        self._validate_hash(base_sha256)
        return {"version": 1, "family": "G30", "base_sha256": base_sha256,
                "settings": dict(self.settings),
                "state_dict": {key: value.detach().cpu().clone()
                               for key, value in self.auxiliary.state_dict().items()}}

    def load_checkpoint(self, artifact, base_sha256):
        self._validate_hash(base_sha256)
        if artifact.get("base_sha256") != base_sha256:
            raise ValueError("Auxiliary checkpoint belongs to a different B0 checkpoint")
        if (artifact.get("version") != 1 or artifact.get("family") != "G30" or
                artifact.get("settings") != self.settings):
            raise ValueError("Incompatible G30 auxiliary architecture/protocol")
        incoming, expected = artifact["state_dict"], self.auxiliary.state_dict()
        if incoming.keys() != expected.keys() or any(not isinstance(incoming[k], torch.Tensor) or
                incoming[k].shape != expected[k].shape or incoming[k].dtype != expected[k].dtype for k in expected):
            raise ValueError("Auxiliary checkpoint has unexpected parameters")
        if any(not torch.isfinite(value).all() for value in incoming.values()):
            raise ValueError("Auxiliary checkpoint parameters must be finite")
        self.auxiliary.load_state_dict(incoming, strict=True)

    @staticmethod
    def _validate_hash(value):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Expected a SHA256 identifying the B0 checkpoint")

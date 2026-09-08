"""LFEQ query-transformer body with a swap-table read-out head (G19 A/B/C).

This is G19's read-out head.  It keeps the EXACT LFEQ query attention (a
learnable decision token plus K evidence tokens, run through the same
self-attention / cross-attention / FFN blocks over the RAW patch tokens), but
replaces the documented LFEQ read-out (global_head / evidence_head /
hard-argmax selection / fusion_weight / diversity regulariser) with ONE of three
simple read-outs selected by ``readout_mode``.  The query-transformer body is
parameter-identical to ``LearnableForgeryEvidenceQuery`` at the same K /
hidden_dim / depth / num_heads / dropout — only the aggregation head differs,
so G19 attributes any change to the read-out, not to the query attention or the
frozen backbone.

The three modes differ ONLY in how the K+1 query tokens are collapsed to a
single [B, 2] logits (class convention 0 = real, 1 = fake):

    mean       ``head( mean_i(queries_i) )``        single Linear(hidden,2)   ~514 params
    per_token  ``mean_i( head_i(queries_i) )``      (K+1) INDEPENDENT Linear(hidden,2)  ~9*(K+1)/8x
    concat     ``head( [queries_1;...;queries_N] )``  Linear(hidden*(K+1),2)   ~9x

A is the capacity-freeze anchor (shared head, no per-position weights).  B and C
each carry ~9x the head capacity; this capacity mis-match is intentional and
reported, NOT matched across arms — it is exactly the degree of freedom this
experiment isolates.

NOTE the B trap: ``mean_i( W·q_i + b ) == W·mean_i(q_i) + b`` because a linear
map commutes with mean.  So a *shared* per-token linear collapses to ``mean``.
B therefore deliberately uses K+1 DISTINCT heads (``head_i``), which do NOT
collapse to A; it reads each token with its own classifier, then soft-averages
the per-token logits.  A and B stand for different training-time operations.

There is NO hard-argmax over the evidence tokens and NO fusion weighting.  Loss
is a single cross-entropy over ``logits`` (no evidence CE, no diversity term), so
the base detector's ``get_losses`` / ``get_train_metrics`` apply unchanged.

Class convention: 0 = real, 1 = fake.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .lfeq_module import EvidenceQueryBlock

_READOUT_MODES = ('mean', 'per_token', 'concat')


class TokenReadout(nn.Module):
    """LFEQ query attention + one of the G19 aggregation read-out heads."""

    def __init__(
        self,
        vit_dim: int,
        hidden_dim: int = 256,
        num_evidence_tokens: int = 8,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        readout_mode: str = 'mean',
    ) -> None:
        super().__init__()
        if readout_mode not in _READOUT_MODES:
            raise ValueError(f"readout_mode must be one of {_READOUT_MODES}, got {readout_mode!r}")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_evidence_tokens < 1:
            raise ValueError("num_evidence_tokens must be at least 1")
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.num_evidence_tokens = num_evidence_tokens
        self.readout_mode = readout_mode
        # Same patch projection / token set / block stack / output norm as LFEQ.
        self.patch_projection = (
            nn.Identity() if vit_dim == hidden_dim else nn.Linear(vit_dim, hidden_dim)
        )
        self.decision_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.evidence_tokens = nn.Parameter(torch.empty(1, num_evidence_tokens, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                EvidenceQueryBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        # G19 read-out head, picked by mode.  All modes drop the LFEQ head split
        # (no global_head / evidence_head) and the argmax/fusion/diversity logic.
        if readout_mode == 'mean':
            # Shared single head over the mean-pooled feature (capacity-freeze anchor).
            self.head = nn.Linear(hidden_dim, 2)
            self.heads = None
        elif readout_mode == 'per_token':
            # K+1 INDEPENDENT heads, one per query token; soft-averaged in logit
            # space.  Distinct from 'mean' (a shared linear would collapse to it).
            self.head = None
            self.heads = nn.ModuleList(
                [nn.Linear(hidden_dim, 2) for _ in range(num_evidence_tokens + 1)]
            )
        else:  # concat
            # Full-dimension read-out: flatten all K+1 tokens into one vector.
            self.head = nn.Linear(hidden_dim * (num_evidence_tokens + 1), 2)
            self.heads = None

        nn.init.trunc_normal_(self.decision_token, std=0.02)
        nn.init.trunc_normal_(self.evidence_tokens, std=0.02)
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _aggregate(self, queries: Tensor) -> Tensor:
        """Collapse [B, K+1, hidden] query tokens to [B, 2] logits."""
        if self.readout_mode == 'mean':
            return self.head(queries.mean(dim=1))
        if self.readout_mode == 'per_token':
            per = torch.stack([h(queries[:, i]) for i, h in enumerate(self.heads)], dim=1)
            return per.mean(dim=1)                       # soft-average in logit space
        # concat
        return self.head(queries.reshape(queries.size(0), -1))

    def _feature(self, queries: Tensor) -> Tensor:
        """The representation saved as ``feat`` (feat/embedding for save_feat)."""
        if self.readout_mode == 'concat':
            return queries.reshape(queries.size(0), -1)
        return queries.mean(dim=1)

    def forward(
        self,
        patch_tokens: Tensor,
        patch_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [B, N, vit_dim]")
        if patch_mask is not None:
            if patch_mask.shape != patch_tokens.shape[:2]:
                raise ValueError("patch_mask must have shape [B, N]")
            patch_mask = patch_mask.to(device=patch_tokens.device, dtype=torch.bool)
            if (~patch_mask).all(dim=1).any():
                raise ValueError("each sample must contain at least one valid patch")
            key_padding_mask = ~patch_mask
        else:
            key_padding_mask = None

        batch_size = patch_tokens.shape[0]
        patches = self.patch_projection(patch_tokens)
        queries = torch.cat(
            [
                self.decision_token.expand(batch_size, -1, -1),
                self.evidence_tokens.expand(batch_size, -1, -1),
            ],
            dim=1,
        )  # [B, K+1, hidden]

        for block in self.blocks:
            queries, _ = block(queries, patches, key_padding_mask)
        queries = self.output_norm(queries)

        logits = self._aggregate(queries)                 # [B, 2]
        probs = logits.softmax(dim=-1)                    # [B, 2]

        return {
            "logits": logits,                              # [B, 2]
            "probs": probs,                                # [B, 2]
            "prob": probs[:, 1],                           # [B]  fake probability
            "pooled": self._feature(queries),              # feat / embed
            "queries": queries,                            # [B, K+1, hidden]
            "prediction": probs.argmax(dim=-1),            # [B]
        }

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        labels: Tensor,
    ) -> Dict[str, Tensor]:
        """Single cross-entropy over the linear head.  No evidence/diversity terms."""
        labels = labels.long()
        loss = nn.functional.cross_entropy(outputs["logits"], labels)
        return {"loss": loss}

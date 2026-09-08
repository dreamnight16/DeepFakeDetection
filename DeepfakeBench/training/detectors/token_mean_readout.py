"""LFEQ query-transformer body with a single mean-pooled linear read-out (G19-A).

This is the G19 read-out head: it keeps the EXACT LFEQ query attention (a
learnable decision token plus K evidence tokens, run through the same
self-attention / cross-attention / FFN blocks over the RAW patch tokens), but
replaces the documented LFEQ read-out (global_head / evidence_head /
hard-argmax selection / fusion_weight / diversity regulariser) with ONE linear
head over the mean of ALL query tokens:

    queries = output_norm( block_stack( [decision; K evidence], patches ) )  # [B, K+1, D]
    pooled  = queries.mean(dim=1)        # mean over every token (incl decision) [B, D]
    logits  = Linear(D, 2)(pooled)       # single linear head, direct prediction  [B, 2]
    prob    = softmax(logits)[:, 1]

There is NO hard-argmax over the evidence tokens and NO fusion weighting.  Loss
is a single cross-entropy over ``logits`` (no evidence CE, no diversity term).

The query-transformer body is parameter-identical to
``LearnableForgeryEvidenceQuery`` at the same K / hidden_dim / depth / num_heads /
dropout — only the read-out differs.  This lets G19 attribute any change to the
read-out head, not to the query attention or the frozen backbone.

Class convention: 0 = real, 1 = fake.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .lfeq_module import EvidenceQueryBlock


class TokenMeanReadout(nn.Module):
    """LFEQ query attention + mean-pooled linear read-out (G19-A)."""

    def __init__(
        self,
        vit_dim: int,
        hidden_dim: int = 256,
        num_evidence_tokens: int = 8,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_evidence_tokens < 1:
            raise ValueError("num_evidence_tokens must be at least 1")
        if depth < 1:
            raise ValueError("depth must be at least 1")

        self.num_evidence_tokens = num_evidence_tokens
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
        # G19-A read-out: a single linear head (no global/evidence split).
        self.head = nn.Linear(hidden_dim, 2)

        nn.init.trunc_normal_(self.decision_token, std=0.02)
        nn.init.trunc_normal_(self.evidence_tokens, std=0.02)
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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

        # G19-A read-out: mean-pool ALL query tokens (decision + evidence), then
        # a single linear head.  No argmax, no fusion, no head split.
        pooled = queries.mean(dim=1)                  # [B, hidden]
        logits = self.head(pooled)                    # [B, 2]
        probs = logits.softmax(dim=-1)                # [B, 2]

        return {
            "logits": logits,                          # [B, 2]
            "probs": probs,                            # [B, 2]
            "prob": probs[:, 1],                       # [B]  fake probability
            "pooled": pooled,                          # [B, hidden]  (the feature / feat)
            "queries": queries,                        # [B, K+1, hidden]
            "prediction": probs.argmax(dim=-1),        # [B]
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

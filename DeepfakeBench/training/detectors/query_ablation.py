"""Query modules shared by the G22 layer-source and G23 lightweight ablations.

G22 uses the input to CLIP ViT's final encoder block (``hidden_states[-2]``)
instead of the final encoder output.  G23 keeps the final output but replaces
each full LFEQ query block with a cross-attention-only block.  Both experiments
keep K, hidden width, depth, head count, and mean read-out fixed to G19-A.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn


def select_vit_patch_tokens(
    backbone_output: Mapping[str, object], source: str
) -> Tensor:
    """Select non-CLS tokens from an explicit CLIP-ViT layer source."""
    if source == "final_output":
        tokens = backbone_output.get("last_hidden_state")
        if tokens is None:
            raise ValueError("backbone output does not contain last_hidden_state")
    elif source == "last_block_input":
        hidden_states = backbone_output.get("hidden_states")
        if hidden_states is None:
            raise ValueError(
                "G22 requires hidden_states; call the backbone with "
                "output_hidden_states=True"
            )
        if not isinstance(hidden_states, Sequence) or len(hidden_states) < 2:
            raise ValueError("G22 requires at least two ViT hidden states")
        tokens = hidden_states[-2]
    else:
        raise ValueError(f"unknown ViT token source: {source!r}")

    if not isinstance(tokens, Tensor) or tokens.ndim != 3:
        raise ValueError("selected ViT tokens must have shape [B, P+1, D]")
    return tokens[:, 1:, :]


class FullQueryBlock(nn.Module):
    """G19/LFEQ block: query self-attention, cross-attention, then FFN."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: Tensor,
        patches: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        q = self.norm_self(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        q = self.norm_cross_q(queries)
        kv = self.norm_cross_kv(patches)
        queries = queries + self.cross_attn(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )[0]
        return queries + self.ffn(self.norm_ffn(queries))


class CrossOnlyQueryBlock(nn.Module):
    """G23 block: retain only pre-normalized query-to-patch attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        queries: Tensor,
        patches: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        q = self.norm_cross_q(queries)
        kv = self.norm_cross_kv(patches)
        return queries + self.cross_attn(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )[0]


class QueryReadout(nn.Module):
    """Mean read-out over learnable queries with selectable query-block cost."""

    def __init__(
        self,
        vit_dim: int,
        hidden_dim: int = 256,
        num_evidence_tokens: int = 8,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        readout_mode: str = "mean",
        variant: str = "full",
    ) -> None:
        super().__init__()
        if readout_mode != "mean":
            raise ValueError("G22/G23 fix readout_mode='mean' for isolation")
        if variant not in ("full", "cross_only"):
            raise ValueError("variant must be 'full' or 'cross_only'")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_evidence_tokens < 1 or depth < 1:
            raise ValueError("num_evidence_tokens and depth must be positive")

        self.variant = variant
        self.patch_projection = (
            nn.Identity() if vit_dim == hidden_dim else nn.Linear(vit_dim, hidden_dim)
        )
        self.decision_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.evidence_tokens = nn.Parameter(
            torch.empty(1, num_evidence_tokens, hidden_dim)
        )
        block_type = FullQueryBlock if variant == "full" else CrossOnlyQueryBlock
        self.blocks = nn.ModuleList(
            [block_type(hidden_dim, num_heads, dropout) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
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
        self, patch_tokens: Tensor, patch_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must have shape [B, P, D]")
        key_padding_mask = None
        if patch_mask is not None:
            if patch_mask.shape != patch_tokens.shape[:2]:
                raise ValueError("patch_mask must have shape [B, P]")
            patch_mask = patch_mask.to(device=patch_tokens.device, dtype=torch.bool)
            if (~patch_mask).all(dim=1).any():
                raise ValueError("each sample must contain at least one valid patch")
            key_padding_mask = ~patch_mask

        batch_size = patch_tokens.shape[0]
        patches = self.patch_projection(patch_tokens)
        queries = torch.cat(
            (
                self.decision_token.expand(batch_size, -1, -1),
                self.evidence_tokens.expand(batch_size, -1, -1),
            ),
            dim=1,
        )
        for block in self.blocks:
            queries = block(queries, patches, key_padding_mask)
        queries = self.output_norm(queries)
        pooled = queries.mean(dim=1)
        logits = self.head(pooled)
        probs = logits.softmax(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "prob": probs[:, 1],
            "pooled": pooled,
            "queries": queries,
            "prediction": probs.argmax(dim=-1),
        }

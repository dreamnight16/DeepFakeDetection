"""Learnable Forgery Evidence Query (LFEQ) module.

The module consumes the final patch tokens of an arbitrary ViT and produces
image-level real/fake predictions. Class 0 denotes real and class 1 denotes
fake throughout this file.

NOTE (G18 vendor copy): taken from ``01My/ai/lfeq_module.py`` and vendored under
``training/detectors/`` so the ``effort_lfeq`` detector can import it via a
relative ``from .lfeq_module import ...`` (no cross-directory sys.path hack).
The ONLY addition vs the upstream copy is the ``'global_feature'`` return key
(= the decision-token feature after ``output_norm``), needed so the detector can
expose a ``feat`` vector.  It is additive and does not change the documented
API, the shapes of any existing key, the fusion rule, or the loss.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class EvidenceQueryBlock(nn.Module):
    """Self-attention over query tokens followed by cross-attention to patches."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)

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
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: Tensor,
        patch_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        q = self.norm_self(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]

        q = self.norm_cross_q(queries)
        kv = self.norm_cross_kv(patch_tokens)
        cross_out, attention = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        queries = queries + cross_out
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries, attention


class LearnableForgeryEvidenceQuery(nn.Module):
    """Classify ViT patch tokens using learnable forgery-evidence queries.

    Args:
        vit_dim: Channel dimension of input patch tokens.
        hidden_dim: Internal query dimension.
        num_evidence_tokens: Number of learnable evidence slots.
        depth: Number of query-transformer blocks.
        num_heads: Attention heads; must divide ``hidden_dim``.
        mlp_ratio: Expansion ratio in each block's FFN.
        dropout: Dropout used by attention and FFN layers.
        fusion_weight: Weight of the global branch in inference probability
            fusion. The evidence branch receives ``1 - fusion_weight``.

    Input:
        patch_tokens: Float tensor shaped ``[B, N, vit_dim]``.
        patch_mask: Optional bool tensor shaped ``[B, N]`` where True means a
            valid patch and False means padding.

    Class convention:
        0 = real, 1 = fake.
    """

    def __init__(
        self,
        vit_dim: int,
        hidden_dim: int = 256,
        num_evidence_tokens: int = 8,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        fusion_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_evidence_tokens < 1:
            raise ValueError("num_evidence_tokens must be at least 1")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if not 0.0 <= fusion_weight <= 1.0:
            raise ValueError("fusion_weight must be in [0, 1]")

        self.num_evidence_tokens = num_evidence_tokens
        self.fusion_weight = fusion_weight
        self.patch_projection = (
            nn.Identity()
            if vit_dim == hidden_dim
            else nn.Linear(vit_dim, hidden_dim)
        )

        # Token 0 is the global decision token. The remaining K tokens are
        # evidence queries used by the hard maximum-evidence branch.
        self.decision_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.evidence_tokens = nn.Parameter(
            torch.empty(1, num_evidence_tokens, hidden_dim)
        )
        self.blocks = nn.ModuleList(
            [
                EvidenceQueryBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.global_head = nn.Linear(hidden_dim, 2)
        self.evidence_head = nn.Linear(hidden_dim, 2)

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
        )

        attention = None
        for block in self.blocks:
            queries, attention = block(queries, patches, key_padding_mask)
        queries = self.output_norm(queries)

        decision_feature = queries[:, 0]
        evidence_features = queries[:, 1:]
        global_logits = self.global_head(decision_feature)
        evidence_logits = self.evidence_head(evidence_features)
        evidence_probs = evidence_logits.softmax(dim=-1)

        # Selection always follows the fake probability, independent of label.
        selected_index = evidence_probs[..., 1].argmax(dim=1)
        batch_index = torch.arange(batch_size, device=patch_tokens.device)
        selected_evidence_logits = evidence_logits[batch_index, selected_index]
        selected_evidence_probs = evidence_probs[batch_index, selected_index]

        global_probs = global_logits.softmax(dim=-1)
        fused_probs = (
            self.fusion_weight * global_probs
            + (1.0 - self.fusion_weight) * selected_evidence_probs
        )

        if attention is None:  # Only possible when depth == 0.
            raise RuntimeError("depth must be at least 1")

        return {
            "global_logits": global_logits,                    # [B, 2]
            "evidence_logits": evidence_logits,                # [B, K, 2]
            "selected_evidence_logits": selected_evidence_logits,  # [B, 2]
            "selected_evidence_index": selected_index,         # [B]
            "global_probs": global_probs,                      # [B, 2]
            "evidence_probs": evidence_probs,                  # [B, K, 2]
            "fused_probs": fused_probs,                        # [B, 2]
            "prediction": fused_probs.argmax(dim=-1),          # [B]
            "attention_maps": attention[:, 1:],                # [B, K, N]
            "decision_attention": attention[:, 0],             # [B, N]
            "evidence_features": evidence_features,            # [B, K, D]
            "global_feature": decision_feature,                # [B, D]  (G18 addition)
        }

    @staticmethod
    def attention_diversity_loss(attention_maps: Tensor) -> Tensor:
        """Mean pairwise cosine similarity between evidence attention maps."""
        if attention_maps.ndim != 3:
            raise ValueError("attention_maps must have shape [B, K, N]")
        k = attention_maps.shape[1]
        if k < 2:
            return attention_maps.new_zeros(())
        normalized = F.normalize(attention_maps, p=2, dim=-1, eps=1e-8)
        similarity = normalized @ normalized.transpose(1, 2)
        off_diagonal_sum = similarity.sum(dim=(1, 2)) - k
        return (off_diagonal_sum / (k * (k - 1))).mean()

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        labels: Tensor,
        evidence_weight: float = 1.0,
        diversity_weight: float = 0.01,
    ) -> Dict[str, Tensor]:
        """Compute global CE + hard maximum-evidence CE + diversity loss."""
        labels = labels.long()
        global_loss = F.cross_entropy(outputs["global_logits"], labels)
        evidence_loss = F.cross_entropy(
            outputs["selected_evidence_logits"], labels
        )
        diversity_loss = self.attention_diversity_loss(
            outputs["attention_maps"]
        )
        total = (
            global_loss
            + evidence_weight * evidence_loss
            + diversity_weight * diversity_loss
        )
        return {
            "loss": total,
            "global_loss": global_loss,
            "evidence_loss": evidence_loss,
            "diversity_loss": diversity_loss,
        }

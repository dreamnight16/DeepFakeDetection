"""G25 token injection using the existing CLIP encoder blocks and projections."""

import torch
from torch import nn
from torch.nn import functional as F


MASK_MODES = {
    "read_only": (False, False),
    "cls_only": (True, False),
    "patch_only": (False, True),
    "full": (True, True),
}


def make_attention_mask(hidden, original_length, mode):
    """Additive [B,1,Q,K] mask; original order is [CLS, patches, evidence]."""
    if mode not in MASK_MODES:
        raise ValueError(f"Unknown G25 attention mode: {mode}")
    batch, length, _ = hidden.shape
    if not 2 <= original_length < length:
        raise ValueError("Expected CLS, at least one patch, and evidence tokens")
    cls_sees, patch_sees = MASK_MODES[mode]
    mask = hidden.new_zeros((1, 1, length, length))
    blocked = torch.finfo(hidden.dtype).min
    if not cls_sees:
        mask[:, :, :1, original_length:] = blocked
    if not patch_sees:
        mask[:, :, 1:original_length, original_length:] = blocked
    return mask.expand(batch, -1, -1, -1)


def forward_with_evidence(vision, images, evidence_tokens, insert_layer, mode,
                          output_attentions=True):
    """Run CLIP with one late insertion, retaining gradients through ALL layers.

    The pretrained embeddings and original positions are unchanged. Evidence
    vectors enter the residual stream directly; each vector identifies its slot.
    Only final-layer attention weights are retained for the diversity loss.
    """
    layers = vision.encoder.layers
    if not 0 <= insert_layer < len(layers):
        raise ValueError(f"insert_layer must be in [0, {len(layers) - 1}]")
    hidden = vision.pre_layrnorm(vision.embeddings(images))
    batch, original_length, dim = hidden.shape
    if (evidence_tokens.ndim != 3 or evidence_tokens.shape[0] != 1
            or evidence_tokens.shape[1] < 1 or evidence_tokens.shape[2] != dim):
        raise ValueError("evidence_tokens must have shape [1, K>=1, hidden_size]")
    attention_mask = None
    attention = None
    for index, layer in enumerate(layers):
        if index == insert_layer:
            extra = evidence_tokens.to(dtype=hidden.dtype).expand(batch, -1, -1)
            hidden = torch.cat((hidden, extra), dim=1)
            attention_mask = make_attention_mask(hidden, original_length, mode)
        need_attention = output_attentions and index == len(layers) - 1
        result = layer(
            hidden, attention_mask=attention_mask, causal_attention_mask=None,
            output_attentions=need_attention,
        )
        hidden = result[0]
        if need_attention:
            if len(result) < 2 or result[1] is None:
                raise RuntimeError("G25 requires CLIP attention weights; use eager attention")
            # Average heads; keep only evidence-query -> patch-key attention.
            attention = result[1][:, :, original_length:, 1:original_length].mean(1)
    features = vision.post_layernorm(
        torch.cat((hidden[:, :1], hidden[:, original_length:]), dim=1)
    )
    return {"features": features, "attention_maps": attention,
            "last_hidden_state": hidden}


class EvidenceHeads(nn.Module):
    """One independent linear classifier per evidence position, never shared."""

    def __init__(self, dim, num_tokens):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, 2) for _ in range(num_tokens)])

    def forward(self, features):
        return torch.stack([head(features[:, i]) for i, head in enumerate(self.heads)], dim=1)


def score_tokens(global_logits, evidence_logits, fusion_weight):
    evidence_probs = evidence_logits.softmax(-1)
    selected = evidence_probs[..., 1].argmax(1)
    rows = torch.arange(global_logits.shape[0], device=global_logits.device)
    selected_logits = evidence_logits[rows, selected]
    fused = (fusion_weight * global_logits.softmax(-1)
             + (1 - fusion_weight) * evidence_probs[rows, selected])
    return {"global_logits": global_logits, "evidence_logits": evidence_logits,
            "selected_evidence_logits": selected_logits,
            "selected_evidence_index": selected, "fused_probs": fused}


def token_losses(outputs, labels, supervision, evidence_weight, diversity_weight):
    """Both variants use the same max-evidence score and diversity constraint."""
    if supervision not in ("max", "all"):
        raise ValueError("G25 supervision must be 'max' or 'all'")
    global_ce = F.cross_entropy(outputs["global_logits"], labels, reduction="none")
    if supervision == "max":
        evidence_ce = F.cross_entropy(
            outputs["selected_evidence_logits"], labels, reduction="none"
        )
    else:
        logits = outputs["evidence_logits"]
        targets = labels[:, None].expand(-1, logits.shape[1])
        evidence_ce = F.cross_entropy(
            logits.reshape(-1, 2), targets.reshape(-1), reduction="none"
        ).reshape_as(targets).mean(1)
    diversity = global_ce.new_zeros(())
    if diversity_weight:
        attention = outputs["attention_maps"]
        if attention is None:
            raise ValueError("Diversity loss requires evidence attention maps")
        k = attention.shape[1]
        if k > 1:
            normalized = F.normalize(attention, p=2, dim=-1, eps=1e-8)
            similarity = normalized @ normalized.transpose(1, 2)
            off_diagonal = ~torch.eye(k, device=attention.device, dtype=torch.bool)
            diversity = similarity[:, off_diagonal].mean()
    per_sample = global_ce + evidence_weight * evidence_ce
    real = per_sample[labels == 0]
    fake = per_sample[labels == 1]
    return {
        "overall": per_sample.mean() + diversity_weight * diversity,
        "real_loss": real.mean() if real.numel() else per_sample.new_zeros(()),
        "fake_loss": fake.mean() if fake.numel() else per_sample.new_zeros(()),
        "loss_global": global_ce.mean(), "loss_evidence": evidence_ce.mean(),
        "loss_diversity": diversity,
    }

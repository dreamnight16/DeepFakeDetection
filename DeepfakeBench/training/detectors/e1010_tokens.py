"""E1010: G18 queries or late CLIP tokens learning beside an immutable B0."""

import importlib
import math
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F


# Standalone experiment/test imports must not initialize every detector.
_package = __package__
if not _package:
    _package = "_e1010_detector_helpers"
    if _package not in sys.modules:
        package = types.ModuleType(_package)
        package.__path__ = [str(Path(__file__).resolve().parent)]
        sys.modules[_package] = package
g25 = importlib.import_module(f"{_package}.g25_tokens")
g25v2 = importlib.import_module(f"{_package}.g25v2_tokens")
g26 = importlib.import_module(f"{_package}.g26_tokens")
g30 = importlib.import_module(f"{_package}.g30_sidecar")
lfeq = importlib.import_module(f"{_package}.lfeq_module")


def forgery_likelihood_loss(patches, labels, mode, likelihood_temperature=.1,
                            contrastive_temperature=.1, max_patches=16):
    """Fake anchors contrast other fake images against the sampled batch pool.

    Prototypes and fake-likelihood weights are detached weak supervision, not
    local annotations. Every denominator includes all other sampled patches;
    positive patches belong to another fake image, never the anchor image.
    """
    if mode == "off":
        return patches.sum() * 0
    real, fake = labels == 0, labels == 1
    if not real.any() or int(fake.sum()) < 2:
        raise ValueError("Forgery likelihood requires at least one real and two fake images")
    count = min(max_patches, patches.shape[1])
    indices = torch.linspace(0, patches.shape[1] - 1, count, device=patches.device).round().long()
    sampled = patches[:, indices]
    if sampled.dtype in (torch.float16, torch.bfloat16):
        sampled = sampled.float()
    normalized = F.normalize(sampled, dim=-1)
    image_ids = torch.arange(len(labels), device=patches.device).repeat_interleave(count)
    flat = normalized.reshape(-1, normalized.shape[-1])
    similarities = flat @ flat.T / contrastive_temperature
    self_mask = torch.eye(len(flat), device=patches.device, dtype=torch.bool)
    log_prob = similarities - torch.logsumexp(
        similarities.masked_fill(self_mask, -torch.inf), dim=1, keepdim=True)
    fake_patches = fake.repeat_interleave(count)
    positive = (fake_patches[:, None] & fake_patches[None, :]
                & (image_ids[:, None] != image_ids[None, :]))
    anchor_losses = -(log_prob[fake_patches] * positive[fake_patches]).sum(1)
    anchor_losses = anchor_losses / positive[fake_patches].sum(1)
    weights = sampled.new_ones((int(fake.sum()), count))
    if mode == "weighted":
        with torch.no_grad():
            real_prototype = F.normalize(patches[real].detach().to(sampled.dtype).mean((0, 1)), dim=0)
            fake_prototype = F.normalize(patches[fake].detach().to(sampled.dtype).mean((0, 1)), dim=0)
            delta = normalized[fake] @ fake_prototype - normalized[fake] @ real_prototype
            weights = (delta / likelihood_temperature).sigmoid().clamp_min(1e-8)
            weights = weights / weights.mean(1, keepdim=True)
    return (anchor_losses.reshape_as(weights) * weights).mean()


class ForgeryAuxiliary(nn.Module):
    def __init__(self, memory_dim, settings):
        super().__init__()
        self.family = settings["token_family"]
        tokens = settings["num_tokens"]
        if self.family == "lfeq":
            self.lfeq = lfeq.LearnableForgeryEvidenceQuery(
                vit_dim=memory_dim, hidden_dim=settings["hidden_dim"],
                num_evidence_tokens=tokens, depth=settings["depth"],
                num_heads=settings["num_heads"], dropout=settings["lfeq_dropout"],
                fusion_weight=settings["lfeq_fusion_weight"])
            # Use the original G18 projection once, shared by FL and queries.
            self.memory_projection = self.lfeq.patch_projection
            if isinstance(self.memory_projection, nn.Identity):
                # Equal dimensions must still permit local supervision to learn.
                self.memory_projection = nn.Linear(memory_dim, memory_dim)
                nn.init.eye_(self.memory_projection.weight)
                nn.init.zeros_(self.memory_projection.bias)
            self.lfeq.patch_projection = nn.Identity()
        elif self.family == "decoder":
            decoder = g30.EvidenceDecoder(memory_dim, tokens, settings["hidden_dim"],
                                           settings["num_heads"], settings["depth"])
            self.memory_norm = decoder.memory_norm
            self.memory_projection = decoder.memory_projection
            self.queries = decoder.queries
            self.blocks = decoder.blocks
            self.output_norm = decoder.output_norm
            self.heads = decoder.heads
        else:
            self.memory_projection = nn.Linear(memory_dim, memory_dim)
            nn.init.eye_(self.memory_projection.weight)
            nn.init.zeros_(self.memory_projection.bias)
            self.evidence_tokens = nn.Parameter(torch.empty(1, tokens, memory_dim))
            nn.init.trunc_normal_(self.evidence_tokens, std=.02)
            self.evidence_heads = g25.EvidenceHeads(memory_dim, tokens)

    def forward(self, boundary, vision, settings):
        patches = boundary[:, 1:].detach()
        if self.family == "decoder":
            patches = self.memory_norm(patches)
        projected = self.memory_projection(patches)
        if self.family == "lfeq":
            outputs = self.lfeq(projected)
            return {"evidence_logits": outputs["evidence_logits"],
                    "evidence_prob": outputs["fused_probs"][:, 1],
                    "aux_decision_logits": outputs["global_logits"],
                    "attention_maps": outputs["attention_maps"],
                    "projected_patches": projected}
        if self.family == "decoder":
            query = self.queries.expand(len(projected), -1, -1)
            for block in self.blocks:
                query = block(query, projected)
            query = self.output_norm(query)
            logits = torch.stack([head(query[:, index]) for index, head in enumerate(self.heads)], 1)
            return {"evidence_logits": logits,
                    "evidence_prob": logits.softmax(-1)[..., 1].max(1).values,
                    "projected_patches": projected}
        original_length = boundary.shape[1]
        hidden = torch.cat((boundary[:, :1].detach(), projected,
                            self.evidence_tokens.to(projected.dtype).expand(len(boundary), -1, -1)), 1)
        mask = g25.make_attention_mask(hidden, original_length, settings["attention_mode"])
        attention = None
        for index in range(settings["memory_layer"], len(vision.encoder.layers)):
            need_attention = settings["diversity_weight"] > 0 and index == len(vision.encoder.layers) - 1
            outputs = g25v2.call_with_detached_parameters(
                vision.encoder.layers[index], hidden, attention_mask=mask, causal_attention_mask=None,
                output_attentions=need_attention)
            hidden = outputs[0]
            if need_attention:
                if len(outputs) < 2 or outputs[1] is None:
                    raise RuntimeError("E1010 auxiliary diversity requires eager CLIP attention")
                attention = outputs[1][:, :, original_length:, 1:original_length].mean(1)
        features = g25v2.call_with_detached_parameters(
            vision.post_layernorm, hidden[:, original_length:])
        logits = self.evidence_heads(features)
        return {"evidence_logits": logits,
                "evidence_prob": logits.softmax(-1)[..., 1].max(1).values,
                "attention_maps": attention,
                "projected_patches": projected}


class FrozenForgerySidecar(nn.Module):
    def __init__(self, base, family="lfeq", num_tokens=None, memory_layer=20,
                 hidden_dim=256, num_heads=None, depth=2, likelihood="off",
                 asymmetric=False, likelihood_weight=.14, likelihood_temperature=.1,
                 contrastive_temperature=.1, max_contrastive_patches=16,
                 mil_temperature=.5, gate_width=.2, aux_max_weight=.5,
                 diversity_weight=.01, memory_source=None, lfeq_dropout=.1,
                 lfeq_fusion_weight=.5, attention_mode="read_only", supervision="max"):
        super().__init__()
        if family not in ("lfeq", "aux", "decoder") or likelihood not in ("off", "uniform", "weighted"):
            raise ValueError("Invalid E1010 token family or likelihood mode")
        num_tokens = (8 if family == "lfeq" else 4) if num_tokens is None else num_tokens
        num_heads = (8 if family == "lfeq" else 4) if num_heads is None else num_heads
        expected_source = "final" if family == "lfeq" else "block_input"
        memory_source = expected_source if memory_source is None else memory_source
        if memory_source != expected_source:
            raise ValueError("G18 LFEQ requires final patches; late auxiliary tokens require block_input")
        if (min(num_tokens, hidden_dim, num_heads, depth, max_contrastive_patches) < 1
                or (family != "aux" and hidden_dim % num_heads)):
            raise ValueError("Require positive dimensions and hidden_dim divisible by num_heads")
        if family != "lfeq" and not 0 <= memory_layer < len(base.backbone.encoder.layers):
            raise ValueError("memory_layer must name an existing B0 encoder block")
        for name, value in (("likelihood_temperature", likelihood_temperature),
                            ("contrastive_temperature", contrastive_temperature),
                            ("mil_temperature", mil_temperature), ("gate_width", gate_width)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if gate_width > .5 or not math.isfinite(aux_max_weight) or not 0 <= aux_max_weight <= .5:
            raise ValueError("Invalid gate width or auxiliary fusion weight")
        if (not math.isfinite(lfeq_dropout) or not 0 <= lfeq_dropout < 1
                or any(not math.isfinite(w) or w < 0 for w in (likelihood_weight, diversity_weight))):
            raise ValueError("Dropout and loss weights must be finite and in range")
        if (not math.isfinite(lfeq_fusion_weight) or not 0 <= lfeq_fusion_weight <= 1
                or attention_mode not in g25.MASK_MODES or supervision not in ("max", "all")):
            raise ValueError("Invalid LFEQ fusion weight, attention mode or token supervision")
        self.settings = dict(token_family=family, num_tokens=num_tokens, memory_layer=memory_layer,
                             hidden_dim=hidden_dim, num_heads=num_heads, depth=depth,
                             likelihood=likelihood, asymmetric=bool(asymmetric),
                             likelihood_weight=likelihood_weight,
                             likelihood_temperature=likelihood_temperature,
                             contrastive_temperature=contrastive_temperature,
                             max_contrastive_patches=max_contrastive_patches,
                             mil_temperature=mil_temperature, gate_width=gate_width,
                             aux_max_weight=aux_max_weight,
                             diversity_weight=diversity_weight if family != "decoder" else 0.,
                             memory_source=memory_source, lfeq_dropout=lfeq_dropout,
                             lfeq_fusion_weight=lfeq_fusion_weight,
                             attention_mode=attention_mode, supervision=supervision,
                             consistency_weight=0.)
        self.base = base
        self.base.requires_grad_(False)
        for parameter in self.base.parameters():
            parameter.grad = None
        self.base.eval()
        dim = base.backbone.embeddings.class_embedding.numel()
        with torch.random.fork_rng(devices=[]):
            self.auxiliary = ForgeryAuxiliary(dim, self.settings)

    def train(self, mode=True):
        self.training = mode
        self.auxiliary.train(mode)
        self.base.eval()
        return self

    def forward(self, data, inference=False, auxiliary_enabled=True):
        if data["image"].ndim != 4:
            raise ValueError("E1010 requires single-crop [B,C,H,W] images")
        self.base.eval()
        if not auxiliary_enabled:
            with torch.no_grad():
                return self.base(data, inference=True)
        captured = []
        if self.settings["memory_source"] == "final":
            layer = self.base.backbone.encoder.layers[-1]

            def capture(module, args, result):
                captured.append(result[0].detach())

            hook = layer.register_forward_hook(capture)
        else:
            layer = self.base.backbone.encoder.layers[self.settings["memory_layer"]]

            def capture(module, args, kwargs):
                hidden = args[0] if args else kwargs["hidden_states"]
                captured.append(hidden.detach())

            hook = layer.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with torch.no_grad():
                original = self.base(data, inference=True)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one original B0 encoder pass")
        auxiliary = self.auxiliary(captured[0], self.base.backbone, self.settings)
        logits = auxiliary.pop("evidence_logits")
        z = logits[..., 1] - logits[..., 0]
        selected_index = logits.softmax(-1)[..., 1].argmax(1)
        p, e = original["prob"], auxiliary["evidence_prob"]
        fused, gate = g26.selective_fusion(p, e, self.settings["gate_width"], self.settings["aux_max_weight"])
        values = {**auxiliary, "global_logits": original["cls"], "cls_prob": p,
                  "evidence_log_odds": z, "selected_evidence_index": selected_index,
                  "gated_prob": fused, "gate_weight": gate}
        probabilities = torch.stack((1 - fused, fused), -1)
        return {**original, "cls": probabilities.clamp_min(torch.finfo(fused.dtype).tiny).log(),
                "prob": fused, "e1010": values}

    def losses(self, output, labels, view_output=None):
        values, settings = output["e1010"], self.settings
        z = values["evidence_log_odds"]
        if labels.ndim != 1 or len(labels) != len(z) or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("E1010 requires hard binary labels aligned with the batch")
        if settings["asymmetric"]:
            evidence = g26.evidence_loss(z, labels, settings["mil_temperature"]).mean()
        elif settings["token_family"] == "aux" and settings["supervision"] == "all":
            evidence = F.softplus(torch.where(labels[:, None] == 1, -z, z)).mean()
        else:
            # Preserve G18/G25's first-slot rule when fake probabilities tie.
            selected = z.gather(1, values["selected_evidence_index"][:, None]).squeeze(1)
            evidence = F.softplus(torch.where(labels == 1, -selected, selected)).mean()
        zero = z.sum() * 0
        decision = F.cross_entropy(values["aux_decision_logits"], labels) if settings["token_family"] == "lfeq" else zero
        diversity = (lfeq.LearnableForgeryEvidenceQuery.attention_diversity_loss(values["attention_maps"])
                     if settings["diversity_weight"] else zero)
        likelihood = forgery_likelihood_loss(
            values["projected_patches"], labels, settings["likelihood"],
            settings["likelihood_temperature"], settings["contrastive_temperature"],
            settings["max_contrastive_patches"])
        overall = decision + evidence + settings["diversity_weight"] * diversity + settings["likelihood_weight"] * likelihood
        return dict(overall=overall, loss_evidence=evidence, loss_likelihood=likelihood,
                    loss_aux_decision=decision, loss_diversity=diversity,
                    loss_global=F.cross_entropy(values["global_logits"], labels).detach())

    def checkpoint(self, base_sha256):
        g30.FrozenEvidenceSidecar._validate_hash(base_sha256)
        return {"version": 1, "family": "E1010", "base_sha256": base_sha256,
                "settings": dict(self.settings),
                "state_dict": {key: value.detach().cpu().clone()
                               for key, value in self.auxiliary.state_dict().items()}}

    def load_checkpoint(self, artifact, base_sha256):
        g30.FrozenEvidenceSidecar._validate_hash(base_sha256)
        if artifact.get("base_sha256") != base_sha256:
            raise ValueError("Auxiliary checkpoint belongs to a different B0 checkpoint")
        if (artifact.get("version") != 1 or artifact.get("family") != "E1010"
                or artifact.get("settings") != self.settings):
            raise ValueError("Incompatible E1010 auxiliary architecture/protocol")
        incoming, expected = artifact["state_dict"], self.auxiliary.state_dict()
        if incoming.keys() != expected.keys() or any(not isinstance(incoming[key], torch.Tensor)
                or incoming[key].shape != expected[key].shape or incoming[key].dtype != expected[key].dtype for key in expected):
            raise ValueError("Auxiliary checkpoint has unexpected parameters")
        if any(not torch.isfinite(value).all() for value in incoming.values()):
            raise ValueError("Auxiliary checkpoint parameters must be finite")
        self.auxiliary.load_state_dict(incoming, strict=True)

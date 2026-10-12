"""E1012: train LoRA with LFEQ, G25/G25v2 tokens, or an evidence decoder."""

import importlib
import math
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F


# Importing the standalone core must not register every project detector.
_package = __package__
if not _package:
    _package = "_e1012_detector_helpers"
    if _package not in sys.modules:
        package = types.ModuleType(_package)
        package.__path__ = [str(Path(__file__).resolve().parent)]
        sys.modules[_package] = package
g25 = importlib.import_module(f"{_package}.g25_tokens")
g25v2 = importlib.import_module(f"{_package}.g25v2_tokens")
g26 = importlib.import_module(f"{_package}.g26_tokens")
g30 = importlib.import_module(f"{_package}.g30_sidecar")
lfeq = importlib.import_module(f"{_package}.lfeq_module")
forgery_likelihood_loss = importlib.import_module(
    f"{_package}.e1010_tokens").forgery_likelihood_loss


class JointForgeryModel(nn.Module):
    def __init__(self, base, family="lfeq", num_tokens=None, memory_layer=20,
                 hidden_dim=256, num_heads=None, depth=2, likelihood="off",
                 asymmetric=False, likelihood_weight=.14, likelihood_temperature=.1,
                 contrastive_temperature=.1, max_contrastive_patches=16,
                 mil_temperature=.5, gate_width=.2, aux_max_weight=.5,
                 diversity_weight=.01, lfeq_dropout=.1, lfeq_fusion_weight=.5,
                 attention_mode="read_only", supervision="max", aux_grad_mode="joint",
                 memory_source=None):
        super().__init__()
        if family not in ("lfeq", "aux", "decoder") or likelihood not in ("off", "uniform", "weighted"):
            raise ValueError("Invalid E1012 family or likelihood")
        if aux_grad_mode not in ("joint", "isolated") or (family != "aux" and aux_grad_mode != "joint"):
            raise ValueError("Isolated auxiliary gradients are supported only for G25 tokens")
        if attention_mode not in g25.MASK_MODES or supervision not in ("max", "all"):
            raise ValueError("Invalid E1012 attention mask or supervision")
        expected_source = "final" if family == "lfeq" else "block_input"
        if memory_source not in (None, expected_source):
            raise ValueError("Memory source does not match the E1012 family")
        num_tokens = (8 if family == "lfeq" else 4) if num_tokens is None else num_tokens
        num_heads = (8 if family == "lfeq" else 4) if num_heads is None else num_heads
        if (min(num_tokens, hidden_dim, num_heads, depth, max_contrastive_patches) < 1
                or (family != "aux" and hidden_dim % num_heads)):
            raise ValueError("Require positive dimensions and hidden_dim divisible by num_heads")
        if family != "lfeq" and not 0 <= memory_layer < len(base.backbone.encoder.layers):
            raise ValueError("memory_layer must identify an existing CLIP block")
        if any(not math.isfinite(value) or value <= 0 for value in
               (likelihood_temperature, contrastive_temperature, mil_temperature)):
            raise ValueError("Loss temperatures must be finite and positive")
        if any(not math.isfinite(value) or value < 0 for value in (likelihood_weight, diversity_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")
        if (not math.isfinite(lfeq_dropout) or not 0 <= lfeq_dropout < 1
                or not math.isfinite(lfeq_fusion_weight) or not 0 <= lfeq_fusion_weight <= 1):
            raise ValueError("Invalid LFEQ dropout or fusion weight")
        if (not math.isfinite(gate_width) or not 0 < gate_width <= .5
                or not math.isfinite(aux_max_weight) or not 0 <= aux_max_weight <= .5):
            raise ValueError("Invalid decoder gate")
        self.settings = dict(family=family, num_tokens=num_tokens, memory_layer=memory_layer,
                             hidden_dim=hidden_dim, num_heads=num_heads, depth=depth,
                             likelihood=likelihood, asymmetric=bool(asymmetric),
                             likelihood_weight=likelihood_weight,
                             likelihood_temperature=likelihood_temperature,
                             contrastive_temperature=contrastive_temperature,
                             max_contrastive_patches=max_contrastive_patches,
                             mil_temperature=mil_temperature, gate_width=gate_width,
                             aux_max_weight=aux_max_weight,
                             diversity_weight=diversity_weight if family != "decoder" else 0.,
                             lfeq_dropout=lfeq_dropout, lfeq_fusion_weight=lfeq_fusion_weight,
                             attention_mode=attention_mode, supervision=supervision,
                             aux_grad_mode=aux_grad_mode, memory_source=expected_source)
        self.base = base
        dim = base.backbone.embeddings.class_embedding.numel()
        if family == "lfeq":
            self.base.backbone.embeddings.class_embedding.requires_grad_(False)
            self.base.head.requires_grad_(False)
            self.lfeq = lfeq.LearnableForgeryEvidenceQuery(
                dim, hidden_dim, num_tokens, depth, num_heads, dropout=lfeq_dropout,
                fusion_weight=lfeq_fusion_weight)
        elif family == "aux":
            self.base.backbone.embeddings.class_embedding.requires_grad_(True)
            self.base.head.requires_grad_(True)
            self.memory_projection = nn.Linear(dim, dim)
            nn.init.eye_(self.memory_projection.weight)
            nn.init.zeros_(self.memory_projection.bias)
            self.evidence_tokens = nn.Parameter(torch.empty(1, num_tokens, dim))
            nn.init.trunc_normal_(self.evidence_tokens, std=.02)
            self.evidence_heads = g25.EvidenceHeads(dim, num_tokens)
            if aux_grad_mode == "isolated":
                for module in base.backbone.modules():
                    dropout = module.p if isinstance(module, nn.Dropout) else getattr(module, "dropout", 0.)
                    if isinstance(dropout, (int, float)) and dropout != 0:
                        raise ValueError("Isolated G25 replay requires zero CLIP/LoRA dropout")
        else:
            self.base.head.requires_grad_(True)
            self.decoder = g30.EvidenceDecoder(dim, num_tokens, hidden_dim, num_heads, depth)

    def _aux_suffix(self, prefix, detached_parameters):
        vision, settings = self.base.backbone, self.settings
        original_length = prefix.shape[1]
        projected = self.memory_projection(prefix[:, 1:])
        hidden = torch.cat((prefix[:, :1], projected,
                            self.evidence_tokens.to(prefix.dtype).expand(len(prefix), -1, -1)), dim=1)
        mask = g25.make_attention_mask(hidden, original_length, settings["attention_mode"])
        attention = None
        for index in range(settings["memory_layer"], len(vision.encoder.layers)):
            layer = vision.encoder.layers[index]
            need_attention = settings["diversity_weight"] > 0 and index == len(vision.encoder.layers) - 1
            kwargs = dict(attention_mask=mask, causal_attention_mask=None, output_attentions=need_attention)
            with g25.eager_attention_for_weights(layer, need_attention):
                outputs = (g25v2.call_with_detached_parameters(layer, hidden, **kwargs)
                           if detached_parameters else layer(hidden, **kwargs))
            hidden = outputs[0]
            if need_attention:
                if len(outputs) < 2 or outputs[1] is None:
                    raise RuntimeError("E1012 G25 diversity requires CLIP attention weights")
                attention = outputs[1][:, :, original_length:, 1:original_length].mean(1)
        selected = torch.cat((hidden[:, :1], hidden[:, original_length:]), dim=1)
        features = (g25v2.call_with_detached_parameters(vision.post_layernorm, selected)
                    if detached_parameters else vision.post_layernorm(selected))
        return features, projected, attention

    def _aux_forward(self, images):
        vision = self.base.backbone
        prefix = vision.pre_layrnorm(vision.embeddings(images))
        for layer in vision.encoder.layers[:self.settings["memory_layer"]]:
            prefix = layer(prefix, attention_mask=None, causal_attention_mask=None,
                           output_attentions=False)[0]
        features, projected, attention = self._aux_suffix(prefix, False)
        evidence_features = features[:, 1:]
        if self.settings["aux_grad_mode"] == "isolated" and self.training and torch.is_grad_enabled():
            replay, projected, attention = self._aux_suffix(prefix.detach(), True)
            evidence_features = replay[:, 1:]
        return dict(global_logits=self.base.head(features[:, 0]),
                    evidence_logits=self.evidence_heads(evidence_features),
                    projected_patches=projected, attention_maps=attention, feat=features[:, 0])

    def _decoder_forward(self, data, inference):
        captured = []

        def capture(module, args, kwargs):
            captured.append(args[0] if args else kwargs["hidden_states"])

        hook = self.base.backbone.encoder.layers[self.settings["memory_layer"]].register_forward_pre_hook(
            capture, with_kwargs=True)
        try:
            original = self.base(data, inference=inference)
        finally:
            hook.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected one CLIP memory capture")
        # EvidenceDecoder.forward intentionally detaches memory for G30; E1012
        # reuses its operators directly so evidence losses can update LoRA.
        projected = self.decoder.memory_projection(self.decoder.memory_norm(captured[0][:, 1:]))
        query = self.decoder.queries.expand(len(projected), -1, -1)
        for block in self.decoder.blocks:
            query = block(query, projected)
        query = self.decoder.output_norm(query)
        logits = torch.stack([head(query[:, index]) for index, head in enumerate(self.decoder.heads)], dim=1)
        return dict(global_logits=original["cls"], evidence_logits=logits,
                    projected_patches=projected, attention_maps=None, feat=original["feat"])

    def forward(self, data, inference=False):
        images = data["image"]
        if images.ndim != 4:
            raise ValueError("E1012 core expects single-crop [B,C,H,W] images")
        family = self.settings["family"]
        if family == "decoder":
            values = self._decoder_forward(data, inference)
        else:
            prepared = self.base._prep_input(images) if hasattr(self.base, "_prep_input") else images
            if family == "aux":
                values = self._aux_forward(prepared)
            else:
                patches = self.base.backbone(prepared).last_hidden_state[:, 1:]
                result = self.lfeq(patches)
                values = dict(global_logits=result["global_logits"], evidence_logits=result["evidence_logits"],
                              projected_patches=self.lfeq.patch_projection(patches),
                              attention_maps=result["attention_maps"], feat=result["global_feature"])
        cls_prob = values["global_logits"].softmax(-1)[:, 1]
        evidence_probs = values["evidence_logits"].softmax(-1)[..., 1]
        index = evidence_probs.argmax(1)
        evidence_prob = evidence_probs.gather(1, index[:, None]).squeeze(1)
        if family == "decoder":
            primary, _ = g26.selective_fusion(cls_prob, evidence_prob,
                                              self.settings["gate_width"], self.settings["aux_max_weight"])
        else:
            weight = self.settings["lfeq_fusion_weight"] if family == "lfeq" else .5
            primary = weight * cls_prob + (1 - weight) * evidence_prob
        feature = values.pop("feat")
        values.update(cls_prob=cls_prob, evidence_prob=evidence_prob, primary_prob=primary,
                      selected_evidence_index=index)
        probabilities = torch.stack((1 - primary, primary), dim=-1)
        return dict(cls=probabilities.clamp_min(torch.finfo(primary.dtype).tiny).log(),
                    prob=primary, feat=feature, e1012=values)

    def losses(self, output, labels):
        values, settings = output["e1012"], self.settings
        logits = values["evidence_logits"]
        if labels.ndim != 1 or len(labels) != len(logits) or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("E1012 requires aligned hard binary labels")
        labels = labels.long()
        global_per_sample = F.cross_entropy(values["global_logits"], labels, reduction="none")
        if settings["asymmetric"]:
            evidence_per_sample = g26.evidence_loss(logits[..., 1] - logits[..., 0], labels,
                                                    settings["mil_temperature"])
        elif settings["family"] == "aux" and settings["supervision"] == "all":
            targets = labels[:, None].expand(-1, logits.shape[1])
            evidence_per_sample = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1),
                                                  reduction="none").reshape_as(targets).mean(1)
        else:
            rows = torch.arange(len(labels), device=labels.device)
            selected = logits[rows, values["selected_evidence_index"]]
            evidence_per_sample = F.cross_entropy(selected, labels, reduction="none")
        zero = logits.sum() * 0
        diversity = (lfeq.LearnableForgeryEvidenceQuery.attention_diversity_loss(values["attention_maps"])
                     if settings["diversity_weight"] else zero)
        # Validation loaders contain single-class batches; FL is a training-only
        # batch objective, not a condition for evaluating a saved detector.
        likelihood = (forgery_likelihood_loss(
            values["projected_patches"], labels, settings["likelihood"],
            settings["likelihood_temperature"], settings["contrastive_temperature"],
            settings["max_contrastive_patches"]) if self.training else zero)
        per_sample = global_per_sample + evidence_per_sample
        overall = per_sample.mean() + settings["diversity_weight"] * diversity + settings["likelihood_weight"] * likelihood
        return dict(overall=overall, loss_global=global_per_sample.mean(),
                    loss_evidence=evidence_per_sample.mean(), loss_likelihood=likelihood,
                    loss_diversity=diversity,
                    real_loss=per_sample[labels == 0].mean() if (labels == 0).any() else zero,
                    fake_loss=per_sample[labels == 1].mean() if (labels == 1).any() else zero)

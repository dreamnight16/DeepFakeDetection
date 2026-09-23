"""G26: freely attending evidence queries that cannot change the B0 update."""

import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

from detectors import DETECTOR
from .effort_detector import EffortDetector
from .g25_tokens import EvidenceHeads
from .g25v2_tokens import forward_isolated_evidence
from .g26_tokens import evidence_loss, selective_fusion, smooth_max


@DETECTOR.register_module(module_name="effort_g26")
class EffortDetectorG26(EffortDetector):
    def __init__(self, config=None):
        config = dict(config or {})
        config.update(full_train_head=True, margin_loss_mode="off")
        if config.get("use_mixup", False) or config.get("use_freq_split", False):
            raise ValueError("G26 requires RGB inputs and hard labels; disable mixup/frequency input")
        super().__init__(config)
        self.num_tokens = int(config.get("g26_num_tokens", 8))
        self.insert_layer = int(config.get("g26_insert_layer", 18))
        self.temperature = float(config.get("g26_mil_temperature", .5))
        self.evidence_weight = float(config.get("g26_evidence_weight", 1.0))
        self.gate_width = float(config.get("g26_gate_width", .2))
        self.aux_max_weight = float(config.get("g26_aux_max_weight", .5))
        self.score_mode = config.get("g26_score_mode", "gated")
        if self.num_tokens < 1 or not 0 <= self.insert_layer < len(self.backbone.encoder.layers):
            raise ValueError("Require K>=1 and an existing insertion block")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("g26_mil_temperature must be finite and positive")
        if not math.isfinite(self.evidence_weight) or self.evidence_weight < 0:
            raise ValueError("g26_evidence_weight must be finite and nonnegative")
        if not math.isfinite(self.gate_width) or not 0 < self.gate_width <= .5:
            raise ValueError("g26_gate_width must be in (0, 0.5]")
        if not math.isfinite(self.aux_max_weight) or not 0 <= self.aux_max_weight <= .5:
            raise ValueError("g26_aux_max_weight must be in [0, 0.5]")
        if self.score_mode not in ("gated", "cls", "evidence"):
            raise ValueError("g26_score_mode must be gated, cls, or evidence")

        # G26 keeps B0's original CLS frozen; only existing LoRA remains tunable.
        self.backbone.embeddings.class_embedding.requires_grad_(False)
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout) and module.p != 0:
                raise ValueError("G26 replay requires zero dropout")
            dropout = getattr(module, "dropout", None)
            if isinstance(dropout, (float, int)) and dropout != 0:
                raise ValueError("G26 replay requires zero attention dropout")
        for layer in self.backbone.encoder.layers:
            if "Flash" in type(layer.self_attn).__name__:
                raise ValueError("G26 requires a CLIP attention implementation supporting additive masks")

        dim = self.backbone.embeddings.class_embedding.numel()
        # Extra CPU initialization must not shift later sampler/augmentation RNG
        # relative to B0. The base backbone and head were already initialized.
        with torch.random.fork_rng(devices=[]):
            self.evidence_tokens = nn.Parameter(torch.empty(1, self.num_tokens, dim))
            nn.init.trunc_normal_(self.evidence_tokens, std=.02)
            self.evidence_heads = EvidenceHeads(dim, self.num_tokens)
        logging.getLogger(__name__).info(
            "G26 K=%d insert=%d suffix=%d tau=%g gate_width=%g aux_max=%g score=%s",
            self.num_tokens, self.insert_layer, len(self.backbone.encoder.layers) - self.insert_layer,
            self.temperature, self.gate_width, self.aux_max_weight, self.score_mode,
        )

    def _forward_images(self, images):
        encoded = forward_isolated_evidence(
            self.backbone, self._prep_input(images), self.evidence_tokens,
            self.insert_layer, "read_only", output_attentions=False, isolate_auxiliary=True,
        )
        features = encoded["features"][:, 0]
        global_logits = self.head(features)
        evidence_logits = self.evidence_heads(encoded["evidence_features"])
        log_odds = evidence_logits[..., 1] - evidence_logits[..., 0]
        bag_logits = smooth_max(log_odds, self.temperature)
        cls_prob = global_logits.softmax(-1)[:, 1]
        evidence_prob = bag_logits.sigmoid()
        fused, gate = selective_fusion(cls_prob, evidence_prob, self.gate_width, self.aux_max_weight)
        scores = {"gated": fused, "cls": cls_prob, "evidence": evidence_prob}
        prob = scores[self.score_mode]
        probabilities = torch.stack((1 - prob, prob), dim=-1)
        output = {"global_logits": global_logits, "evidence_log_odds": log_odds,
                  "cls_prob": cls_prob, "evidence_prob": evidence_prob,
                  "gated_prob": fused, "gate_weight": gate}
        return {"cls": probabilities.clamp_min(torch.finfo(prob.dtype).tiny).log(),
                "prob": prob, "feat": features, "g26": output}

    def forward(self, data_dict, inference=False):
        images = data_dict["image"]
        if images.ndim == 4:
            return self._forward_images(images)
        if not inference or images.ndim != 5:
            raise ValueError("G26 expects 4D images, or 5D crops during inference")
        batch, crops = images.shape[:2]
        result = self._forward_images(images.flatten(0, 1))
        # Use the main branch's TAA selection for ALL readouts, keeping the
        # original decision/crop outside the gate and branch comparisons paired.
        chosen = (result["g26"]["cls_prob"].reshape(batch, crops) - .5).abs().argmax(1)
        rows = torch.arange(batch, device=images.device)

        def select(value):
            return value.reshape(batch, crops, *value.shape[1:])[rows, chosen]

        return {key: ({name: select(value) for name, value in values.items()}
                      if isinstance(values, dict) else select(values))
                for key, values in result.items()}

    def get_losses(self, data_dict, pred_dict):
        if "label_soft" in data_dict:
            raise ValueError("G26 requires hard binary labels")
        labels, output = data_dict["label"], pred_dict["g26"]
        if labels.ndim != 1 or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("G26 labels must be a vector of 0=real, 1=fake")
        global_ce = F.cross_entropy(output["global_logits"], labels, reduction="none")
        auxiliary = evidence_loss(output["evidence_log_odds"], labels, self.temperature)
        # Never train on fused scores: this would let the gate change the main
        # gradient. Train evidence on every sample, not only ambiguous samples.
        per_sample = global_ce + self.evidence_weight * auxiliary
        real, fake = per_sample[labels == 0], per_sample[labels == 1]
        return {"overall": per_sample.mean(), "loss_global": global_ce.mean(),
                "loss_evidence": auxiliary.mean(),
                "real_loss": real.mean() if real.numel() else per_sample.new_zeros(()),
                "fake_loss": fake.mean() if fake.numel() else per_sample.new_zeros(())}

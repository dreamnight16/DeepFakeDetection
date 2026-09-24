"""G27: isolated expert-token ablations on the G26 architecture."""

import math

import torch
from torch.nn import functional as F

from detectors import DETECTOR
from .effort_detector_g26 import EffortDetectorG26
from .g25v2_tokens import forward_isolated_evidence
from .g26_tokens import evidence_loss
from .g27_tokens import balance_loss, difficulty_weights, isolated_view_features, photometric_view


@DETECTOR.register_module(module_name="effort_g27")
class EffortDetectorG27(EffortDetectorG26):
    def __init__(self, config=None):
        config = dict(config or {})
        # Translate only at the inheritance boundary; artifacts use G27 keys.
        parent = dict(config)
        defaults = dict(num_tokens=4, insert_layer=20, mil_temperature=.5,
                        evidence_weight=1., gate_width=.2, aux_max_weight=.5, score_mode="gated")
        for key, value in defaults.items():
            parent[f"g26_{key}"] = config.get(f"g27_{key}", value)
        super().__init__(parent)
        self.balance_weight = float(config.get("g27_balance_weight", 0.))
        self.consistency_weight = float(config.get("g27_consistency_weight", 0.))
        self.router_temperature = float(config.get("g27_router_temperature", 1.))
        self.hard_weighting = config.get("g27_hard_weighting", False)
        self.hard_floor = float(config.get("g27_hard_floor", .2))
        self.hard_width = float(config.get("g27_hard_width", .2))
        self.view_contrast = float(config.get("g27_view_contrast", .9))
        self.view_brightness = float(config.get("g27_view_brightness", .02))
        self.view_std = tuple(float(v) for v in config.get("std", (.26862954, .26130258, .27577711)))
        for name in ("balance_weight", "consistency_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"G27 {name} must be finite and nonnegative")
        if not math.isfinite(self.router_temperature) or self.router_temperature <= 0:
            raise ValueError("G27 router temperature must be finite and positive")
        if not isinstance(self.hard_weighting, bool):
            raise ValueError("G27 hard weighting must be a boolean")
        if not 0 <= self.hard_floor <= 1 or not 0 < self.hard_width <= .5:
            raise ValueError("G27 hard floor must be in [0,1], width in (0,.5]")
        if (not math.isfinite(self.view_contrast) or not 0 < self.view_contrast <= 1
                or not math.isfinite(self.view_brightness) or abs(self.view_brightness) > .1):
            raise ValueError("G27 view requires contrast in (0,1] and abs(brightness)<=.1")
        if len(self.view_std) != 3 or any(not math.isfinite(v) or v <= 0 for v in self.view_std):
            raise ValueError("G27 std must have three positive finite channels")

    def forward(self, data_dict, inference=False):
        result = super().forward(data_dict, inference=inference)
        output = result.pop("g26")
        result["g27"] = output
        if self.training and not inference and torch.is_grad_enabled() and self.consistency_weight > 0:
            images = photometric_view(data_dict["image"], self.view_std,
                                      self.view_contrast, self.view_brightness)
            features = isolated_view_features(self.backbone, self._prep_input(images),
                                              self.evidence_tokens, self.insert_layer)
            logits = self.evidence_heads(features)
            output["second_log_odds"] = logits[..., 1] - logits[..., 0]
        return result

    @torch.no_grad()
    def evidence_attention(self, images):
        """Bounded diagnostic snapshot, same input; no spatial-supervision claim."""
        return forward_isolated_evidence(
            self.backbone, self._prep_input(images), self.evidence_tokens, self.insert_layer,
            "read_only", output_attentions=True, isolate_auxiliary=True)["attention_maps"]

    def get_losses(self, data_dict, pred_dict):
        if "label_soft" in data_dict:
            raise ValueError("G27 requires hard binary labels")
        labels, output = data_dict["label"], pred_dict["g27"]
        z = output["evidence_log_odds"]
        if labels.shape != z.shape[:1] or not ((labels == 0) | (labels == 1)).all():
            raise ValueError("G27 labels must be a vector of 0=real, 1=fake")
        global_ce = F.cross_entropy(output["global_logits"], labels, reduction="none")
        evidence = evidence_loss(z, labels, self.temperature)
        weights = (difficulty_weights(output["cls_prob"], self.hard_width, self.hard_floor)
                   if self.hard_weighting else torch.ones_like(evidence))
        weighted = weights * evidence
        balance = balance_loss(z, labels, self.router_temperature)
        consistency = torch.zeros_like(evidence)
        if "second_log_odds" in output:
            # Symmetric gradients to both views' expert tokens/heads; all shared
            # backbone parameters are detached by their respective forward paths.
            consistency = (z.float().sigmoid() - output["second_log_odds"].float().sigmoid()).square().mean(1)
        elif self.training and torch.is_grad_enabled() and self.consistency_weight > 0:
            raise ValueError("G27 consistency training requires second view logits")
        auxiliary = self.evidence_weight * weighted + self.consistency_weight * consistency
        per_sample = global_ce + auxiliary
        balance_term = self.balance_weight * balance
        real, fake = per_sample[labels == 0], per_sample[labels == 1]
        return {"overall": per_sample.mean() + balance_term,
                "loss_global": global_ce.mean(), "loss_evidence": evidence.mean(),
                "loss_evidence_weighted": weighted.mean(), "loss_balance": balance,
                "loss_consistency": consistency.mean(),
                "loss_auxiliary": auxiliary.mean() + balance_term,
                "expert_weight_mean": weights.mean(),
                # Class diagnostics exclude the batch-level balance term.
                "real_loss": real.mean() if real.numel() else per_sample.new_zeros(()),
                "fake_loss": fake.mean() if fake.numel() else per_sample.new_zeros(())}

"""G25: train the pretrained CLS and insert independent evidence tokens late."""

import logging
import math

import torch
from torch import nn

from detectors import DETECTOR
from .effort_detector import EffortDetector
from .g25_tokens import (MASK_MODES, EvidenceHeads, forward_with_evidence,
                         score_tokens, token_losses)


@DETECTOR.register_module(module_name="effort_g25")
class EffortDetectorG25(EffortDetector):
    def __init__(self, config=None):
        config = dict(config or {})
        config["full_train_head"] = True
        config["margin_loss_mode"] = "off"
        super().__init__(config)
        self.num_tokens = int(config.get("g25_num_tokens", 4))
        self.insert_layer = int(config.get("g25_insert_layer", 20))
        self.mask_mode = config.get("g25_attention_mode", "read_only")
        self.supervision = config.get("g25_supervision", "max")
        self.fusion_weight = float(config.get("g25_fusion_weight", 0.5))
        self.evidence_weight = float(config.get("g25_evidence_weight", 1.0))
        self.diversity_weight = float(config.get("g25_diversity_weight", 0.01))
        if self.num_tokens < 0:
            raise ValueError("g25_num_tokens must be nonnegative (0 is the CLS-only control)")
        if self.mask_mode not in MASK_MODES or self.supervision not in ("max", "all"):
            raise ValueError("Invalid G25 attention mode or supervision")
        if not math.isfinite(self.fusion_weight) or not 0 <= self.fusion_weight <= 1:
            raise ValueError("g25_fusion_weight must be in [0, 1]")
        if any(not math.isfinite(w) or w < 0 for w in (self.evidence_weight, self.diversity_weight)):
            raise ValueError("G25 loss weights must be finite and nonnegative")
        if self.num_tokens and not 0 <= self.insert_layer < len(self.backbone.encoder.layers):
            raise ValueError("g25_insert_layer is outside the ViT encoder")
        if config.get("use_mixup", False) or config.get("use_freq_split", False):
            raise ValueError("G25 requires RGB inputs with mixup disabled")

        # Re-enable only the ORIGINAL CLS parameter, retaining its pretrained value.
        self.backbone.embeddings.class_embedding.requires_grad_(True)
        dim = self.backbone.embeddings.class_embedding.numel()
        if self.num_tokens:
            self.evidence_tokens = nn.Parameter(torch.empty(1, self.num_tokens, dim))
            nn.init.trunc_normal_(self.evidence_tokens, std=0.02)
            self.evidence_heads = EvidenceHeads(dim, self.num_tokens)
            # Modern CLIPAttention dispatch reads this config. Legacy SDPA falls
            # back to eager when final-layer attention weights are requested.
            for layer in self.backbone.encoder.layers:
                if "Flash" in type(layer.self_attn).__name__:
                    raise ValueError("G25 does not support FlashAttention; use eager CLIP")
                if hasattr(layer.self_attn, "config"):
                    layer.self_attn.config._attn_implementation = "eager"
        logging.getLogger(__name__).info(
            "G25 K=%d insert=%d mask=%s supervision=%s trainable=%d",
            self.num_tokens, self.insert_layer, self.mask_mode, self.supervision,
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def _forward_images(self, images, output_attentions=True):
        if not self.num_tokens:
            feat = self.backbone(self._prep_input(images))["pooler_output"]
            logits = self.head(feat)
            return {"cls": logits, "prob": logits.softmax(-1)[:, 1], "feat": feat}
        encoded = forward_with_evidence(
            self.backbone, self._prep_input(images), self.evidence_tokens,
            self.insert_layer, self.mask_mode,
            output_attentions=output_attentions and self.diversity_weight > 0,
        )
        features = encoded["features"]
        outputs = score_tokens(
            self.head(features[:, 0]), self.evidence_heads(features[:, 1:]), self.fusion_weight
        )
        outputs["attention_maps"] = encoded["attention_maps"]
        # Log probabilities make train metrics (softmax(cls)) match the scored
        # fused branch. The loss explicitly uses each branch's original logits.
        logits = outputs["fused_probs"].clamp_min(torch.finfo(features.dtype).tiny).log()
        return {"cls": logits, "prob": outputs["fused_probs"][:, 1],
                "feat": features[:, 0], "g25": outputs}

    def forward(self, data_dict, inference=False):
        images = data_dict["image"]
        if not (inference and images.ndim == 5):
            # Validation also calls get_losses, even with inference=True.
            return self._forward_images(images)
        batch, crops, channels, height, width = images.shape
        result = self._forward_images(images.reshape(-1, channels, height, width), False)
        probabilities = result["prob"].reshape(batch, crops)
        selected = (probabilities - 0.5).abs().argmax(1)
        rows = torch.arange(batch, device=images.device)
        return {"cls": result["cls"].reshape(batch, crops, 2)[rows, selected],
                "prob": probabilities[rows, selected],
                "feat": result["feat"].reshape(batch, crops, -1)[rows, selected]}

    def get_losses(self, data_dict, pred_dict):
        if "label_soft" in data_dict:
            raise ValueError("G25 losses require hard labels; disable mixup")
        if not self.num_tokens:
            return super().get_losses(data_dict, pred_dict)
        return token_losses(pred_dict["g25"], data_dict["label"], self.supervision,
                            self.evidence_weight, self.diversity_weight)

"""E1009 varies trainable parameters while retaining G25/G25v2 behavior."""

import logging

from torch import nn

from detectors import DETECTOR
from .effort_detector_g25v2 import EffortDetectorG25v2


@DETECTOR.register_module(module_name="effort_e1009")
class EffortDetectorE1009(EffortDetectorG25v2):
    def __init__(self, config=None):
        config = dict(config or {})
        tuning = config.get("e1009_tuning", "tokens")
        if tuning not in ("tokens", "late_lora", "layernorm", "all_lora"):
            raise ValueError("e1009_tuning must be 'tokens', 'late_lora', 'layernorm', or 'all_lora'")
        super().__init__(config)
        self.e1009_tuning = tuning

        # Retain pretrained CLIP and zero-initialized LoRA modules. Freezing
        # weights must still allow input gradients to reach CLS and new tokens.
        self.backbone.requires_grad_(False)
        self.backbone.embeddings.class_embedding.requires_grad_(True)
        if tuning in ("late_lora", "all_lora"):
            layers = self.backbone.encoder.layers
            if tuning == "late_lora":
                layers = layers[-4:]
            for layer in layers:
                for name, parameter in layer.named_parameters():
                    if name.endswith((".lora_A", ".lora_B")):
                        parameter.requires_grad_(True)
        elif tuning == "layernorm":
            for module in self.backbone.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)

        self.tuning_summary = {
            group: {"names": [], "numel": 0}
            for group in ("cls", "evidence_tokens", "heads", "lora", "layernorm")
        }
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == "backbone.embeddings.class_embedding":
                group = "cls"
            elif name == "evidence_tokens":
                group = "evidence_tokens"
            elif name.startswith(("head.", "evidence_heads.")):
                group = "heads"
            elif name.endswith((".lora_A", ".lora_B")):
                group = "lora"
            else:
                group = "layernorm"
            self.tuning_summary[group]["names"].append(name)
            self.tuning_summary[group]["numel"] += parameter.numel()
        logging.getLogger(__name__).info(
            "E1009 tuning=%s trainable=%d groups=%s", tuning,
            sum(group["numel"] for group in self.tuning_summary.values()), self.tuning_summary,
        )

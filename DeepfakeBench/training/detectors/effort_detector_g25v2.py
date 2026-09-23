"""G25v2 keeps the original G25 architecture and isolates auxiliary gradients."""

import torch
from torch import nn

from detectors import DETECTOR
from .effort_detector_g25 import EffortDetectorG25
from .g25_tokens import score_tokens
from .g25v2_tokens import forward_isolated_evidence


@DETECTOR.register_module(module_name="effort_g25v2")
class EffortDetectorG25v2(EffortDetectorG25):
    def __init__(self, config=None):
        config = dict(config or {})
        self.aux_grad_mode = config.get("g25v2_aux_grad_mode", "isolated")
        self.score_mode = config.get("g25v2_score_mode", "fused")
        if self.aux_grad_mode not in ("isolated", "joint"):
            raise ValueError("g25v2_aux_grad_mode must be 'isolated' or 'joint'")
        if self.score_mode not in ("fused", "cls", "evidence"):
            raise ValueError("g25v2_score_mode must be 'fused', 'cls', or 'evidence'")
        super().__init__(config)
        if not self.num_tokens and self.score_mode == "evidence":
            raise ValueError("Evidence-only scoring requires evidence tokens")
        # The replay must compute the same values as the main suffix. Existing
        # G25 uses zero CLIP/LoRA dropout; reject unsupported stochastic variants.
        if self.aux_grad_mode == "isolated":
            for module in self.backbone.modules():
                if isinstance(module, nn.Dropout) and module.p != 0:
                    raise ValueError("G25v2 isolated replay requires zero dropout")
                dropout = getattr(module, "dropout", None)
                if isinstance(dropout, (float, int)) and dropout != 0:
                    raise ValueError("G25v2 isolated replay requires zero attention dropout")

    def _forward_images(self, images, output_attentions=True):
        if not self.num_tokens:
            return super()._forward_images(images, output_attentions)
        encoded = forward_isolated_evidence(
            self.backbone, self._prep_input(images), self.evidence_tokens,
            self.insert_layer, self.mask_mode,
            output_attentions=output_attentions and self.diversity_weight > 0,
            isolate_auxiliary=self.aux_grad_mode == "isolated",
        )
        outputs = score_tokens(
            self.head(encoded["features"][:, 0]),
            self.evidence_heads(encoded["evidence_features"]), self.fusion_weight,
        )
        outputs["attention_maps"] = encoded["attention_maps"]
        if self.score_mode == "cls":
            probabilities = outputs["global_logits"].softmax(-1)
        elif self.score_mode == "evidence":
            probabilities = outputs["selected_evidence_logits"].softmax(-1)
        else:
            probabilities = outputs["fused_probs"]
        logits = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        return {"cls": logits, "prob": probabilities[:, 1],
                "feat": encoded["features"][:, 0], "g25": outputs}

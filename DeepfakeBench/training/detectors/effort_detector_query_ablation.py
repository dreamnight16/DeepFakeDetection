"""Detector integration for isolated G22 and G23 query ablations."""

import torch

from detectors import DETECTOR

from .effort_detector import EffortDetector
from .query_ablation import QueryReadout, select_vit_patch_tokens
from .token_readout import TokenReadout


class EffortDetectorQueryAblationBase(EffortDetector):
    token_source = "final_output"
    query_variant = "full"

    def __init__(self, config=None):
        config = config if config is not None else {}
        super().__init__(config)
        readout_type = TokenReadout if self.query_variant == "full" else QueryReadout
        readout_kwargs = dict(
            vit_dim=1024,
            hidden_dim=int(config.get("lfeq_hidden_dim", 256)),
            num_evidence_tokens=int(config.get("lfeq_num_evidence_tokens", 8)),
            depth=int(config.get("lfeq_depth", 2)),
            num_heads=int(config.get("lfeq_num_heads", 8)),
            dropout=float(config.get("lfeq_dropout", 0.1)),
            readout_mode="mean",
        )
        if readout_type is QueryReadout:
            readout_kwargs["variant"] = self.query_variant
        self.readout = readout_type(**readout_kwargs)
        for parameter in self.head.parameters():
            parameter.requires_grad = False

    def _backbone_output(self, images):
        return self.backbone(
            self._prep_input(images),
            output_hidden_states=self.token_source == "last_block_input",
        )

    def _forward_images(self, images):
        output = self._backbone_output(images)
        patches = select_vit_patch_tokens(output, self.token_source)
        result = self.readout(patches)
        return {
            "cls": result["logits"],
            "prob": result["prob"],
            "feat": result["pooled"],
        }

    def forward(self, data_dict, inference=False):
        images = data_dict["image"]
        if not (inference and images.ndim == 5):
            return self._forward_images(images)

        batch_size, crops, channels, height, width = images.shape
        result = self._forward_images(images.reshape(-1, channels, height, width))
        per_crop = result["prob"].view(batch_size, crops)
        crop_index = torch.abs(per_crop - 0.5).argmax(dim=1)
        batch_index = torch.arange(batch_size, device=images.device)
        logits = result["cls"].view(batch_size, crops, 2)
        features = result["feat"].view(batch_size, crops, -1)
        return {
            "cls": logits[batch_index, crop_index],
            "prob": per_crop[batch_index, crop_index],
            "feat": features[batch_index, crop_index],
        }


@DETECTOR.register_module(module_name="effort_g22_last_block_input")
class EffortDetectorG22(EffortDetectorQueryAblationBase):
    """G22: full G19-A query block over the final ViT block's input."""

    token_source = "last_block_input"
    query_variant = "full"


@DETECTOR.register_module(module_name="effort_g23_cross_only")
class EffortDetectorG23(EffortDetectorQueryAblationBase):
    """G23: cross-only query blocks over the final ViT output."""

    token_source = "final_output"
    query_variant = "cross_only"

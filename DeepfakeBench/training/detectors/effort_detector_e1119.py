"""E1119 detector registered with the original Trainer and checkpoint loader."""

from pathlib import Path

import torch
from torch import nn
from transformers import CLIPModel

from detectors import DETECTOR
from .effort_detector import EffortDetector, Linear, lora_lib
from e1119_model import (
    DETACH, PROJECTIONS, SPLIT, LoRA, Spectral, blocks, bypass,
    hidden, hooks, paired_forward, projections, relative_drift, run_block, selected_indices,
)


@DETECTOR.register_module(module_name="effort_e1119")
class EffortDetectorE1119(EffortDetector):
    def __init__(self, config):
        self.arm = config["e1119_arm"]
        super().__init__(config)
        if self.arm == "T02":
            self.register_buffer("aux_step", torch.zeros((), dtype=torch.long))

    def build_backbone(self, config):
        native_linear = lora_lib.Linear if self.use_loralib else Linear
        vision = CLIPModel.from_pretrained(config["clip_pretrained_path"]).vision_model
        vision.requires_grad_(False)
        if len(vision.encoder.layers) != 24 or vision.post_layernorm.normalized_shape != (1024,):
            raise ValueError("E1119 requires the original 24-layer CLIP ViT-L/14 vision tower")
        targets = {"B02": PROJECTIONS[2:], "B03": PROJECTIONS[:2]}.get(self.arm, PROJECTIONS)
        assets = config.get("e1119_assets")
        for name, base in list(projections(vision)):
            parent_name, key = name.rsplit(".", 1)
            layer = int(parent_name.split(".")[-2])
            # Consume exactly the native B0 initialization draws. Only install the
            # registered modules, so the head and subsequent loader RNG match B00.
            native = native_linear(base.in_features, base.out_features, r=4, lora_alpha=16,
                                   lora_dropout=0, merge_weights=False)
            if self.arm in ("B01", "M03") or key not in targets or (self.arm == "T03" and layer >= 16):
                continue
            if self.arm.startswith("M"):
                asset = torch.load(Path(assets) / f"{name}.pt", map_location="cpu", weights_only=True)
                replacement = Spectral(base, self.arm, asset)
            else:
                basis = None
                if self.arm.startswith("P"):
                    point = "out_proj" if key == "out_proj" else "q_proj"
                    basis = torch.load(Path(assets) / f"{parent_name}.{point}.pt",
                                       map_location="cpu", weights_only=True)
                route = None
                if key != "out_proj":
                    if layer - 1 in SPLIT.get(self.arm, ()):
                        route = "split"
                    elif layer in DETACH.get(self.arm, ()):
                        route = "detach"
                replacement = LoRA(base, native, 2 if self.arm == "B04" else 4, basis, route)
            setattr(vision.get_submodule(parent_name), key, replacement)
        if self.arm == "M03":
            for module in vision.modules():
                if isinstance(module, nn.LayerNorm):
                    module.requires_grad_(True)
        return vision

    def features(self, data_dict):
        if self.arm not in SPLIT:
            return super().features(data_dict)
        layers = blocks(self)
        forward = super().features
        return paired_forward(layers, SPLIT[self.arm], lambda: forward(data_dict))

    def forward(self, data_dict, inference=False):
        if inference or not self.training or self.arm not in ("T01", "T02"):
            return super().forward(data_dict, inference)
        entry = []
        with hooks() as handles:
            handles.append(blocks(self)[15].register_forward_hook(
                lambda _, args, output: entry.append(hidden(output).detach())))
            result = super().forward(data_dict, inference)
        labels = data_dict["label"]
        step = int(self.aux_step) if self.arm == "T02" else 0
        generator = torch.Generator().manual_seed(self.config["manualSeed"] + 1119 + step)
        selected = selected_indices(labels, self.arm == "T02", generator, step)
        if self.arm == "T02":
            self.aux_step.add_(1)
        result["segment_loss"] = (self.segment_drift(entry[0][selected]).mean()
                                  if len(selected) else result["cls"].new_zeros(()))
        return result

    def segment_drift(self, entry):
        segment = blocks(self)[16:]
        student = teacher = entry.detach()
        with torch.no_grad(), bypass(segment):
            for layer in segment:
                teacher = run_block(layer, teacher)
        for layer in segment:
            student = run_block(layer, student)
        return relative_drift(student, teacher)

    def get_losses(self, data_dict, pred_dict):
        losses = super().get_losses(data_dict, pred_dict)
        if not self.training:
            return losses
        if self.arm == "M02":
            with torch.autocast(device_type=pred_dict["cls"].device.type, enabled=False):
                orth, norm = zip(*(module.regularizers() for _, module in projections(self.backbone)))
                orth, norm = torch.stack(orth).mean(), torch.stack(norm).mean()
            losses.update(ce=losses["overall"], orth=orth, norm=norm,
                          orth_weighted=orth, norm_weighted=norm)
            losses["overall"] = losses["ce"] + orth + norm
        if "segment_loss" in pred_dict:
            losses["ce"] = losses["overall"]
            losses["segment_loss"] = pred_dict["segment_loss"]
            losses["overall"] = losses["ce"] + losses["segment_loss"]
        return losses

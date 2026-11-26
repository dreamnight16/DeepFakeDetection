"""E1124 adapters on the original CLIP detector and classification head."""

from contextvars import ContextVar
from pathlib import Path

import torch
import torch.nn.functional as F

from e1119_model import OFF, PROJECTIONS, REFERENCE, LoRA, blocks, paired_forward, projections
from . import DETECTOR
from .effort_detector import EffortDetector, Linear, lora_lib


FULL_READ = ContextVar("e1124_full_read", default=False)
ROUTES = {"L1": "real", "L2": "random", "L3": "mixed", "L4": "real", "L5": "real",
          "C1": "real", "C2": "real", "C3": "random"}
UPDATE = {"P128S": "P01", "V95": "V95", "V95R": "V95R", "W128": "weight",
          "W128LR": "weight", "N0": "N0", "N1": "N1", "N2": "N2", "N3": "N3", "N4": "N4",
          "C1": "N1", "C2": "N0", "C3": "N1", "X1": "N1"}


def project_columns(value, basis):
    return basis @ (basis.T @ value)


class ProjectedLoRA(LoRA):
    def __init__(self, base, native, rank, update, left, read_basis, rule):
        super().__init__(base, native, rank)
        self.register_buffer("update_basis", update)
        self.register_buffer("left_basis", left)
        self.register_buffer("read_basis", read_basis)
        self.rule = rule
        self.project_parameters()

    @torch.no_grad()
    def project_parameters(self):
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            if self.update_basis is not None:
                self.lora_A.sub_(project_columns(self.lora_A.float(), self.update_basis.float()))
            if self.left_basis is not None:
                self.lora_B.sub_(project_columns(self.lora_B.float().T, self.left_basis.float()).T)

    def removed_input(self, actual, reference):
        with torch.autocast(device_type=actual.device.type, enabled=False):
            delta = actual.float() - reference.float()
            basis = self.read_basis.float()
            removed = (delta @ basis) @ basis.T
            if self.rule == "L4":
                length = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                safe = torch.where(length > 0, length, torch.ones_like(length))
                removed = delta * (torch.linalg.vector_norm(removed, dim=-1, keepdim=True) / safe)
        return removed

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        if OFF.get():
            return base
        if self.read_basis is not None and not FULL_READ.get() and (self.training or self.rule != "L5"):
            removed = self.removed_input(x, REFERENCE.get()[self]).to(x.dtype)
            x = x + (removed.detach() - removed) if self.rule == "L5" else x - removed
        return base + self.delta(x)


def make_native(factory, base, rank):
    return factory(base.in_features, base.out_features, r=rank, lora_alpha=4 * rank,
                   lora_dropout=0, merge_weights=False)


def register_projection_step(model, optimizer):
    constrained = [module for _, module in projections(model.backbone)
                   if isinstance(module, ProjectedLoRA) and module.update_basis is not None]
    if constrained:
        def after_step(*_):
            for module in constrained:
                module.project_parameters()

        optimizer.register_step_post_hook(after_step)


@DETECTOR.register_module(module_name="effort_e1124")
class EffortDetectorE1124(EffortDetector):
    def build_backbone(self, config):
        from transformers import CLIPModel

        self.arm = config["e1124_arm"]
        vision = CLIPModel.from_pretrained(config["clip_pretrained_path"]).vision_model
        vision.requires_grad_(False)
        if len(vision.encoder.layers) != 24 or vision.post_layernorm.normalized_shape != (1024,):
            raise ValueError("E1124 requires the original CLIP ViT-L/14 vision tower")
        factory = lora_lib.Linear if self.use_loralib else Linear
        ranks = (6, 2, 4) if self.arm == "R2" else (2, 6, 4) if self.arm in ("R1", "X1") else (4, 4, 4)
        assets = config.get("e1124_assets", {})

        def load(kind, name):
            return torch.load(Path(assets[kind]) / f"{name}.pt", weights_only=True, map_location="cpu")

        for name, base in list(projections(vision)):
            parent, key = name.rsplit(".", 1)
            layer = int(parent.split(".")[-2])
            rank = ranks[layer // 8]
            # Consume native draws even when the adapter has another rank.
            if rank == 4:
                native = make_native(factory, base, rank)
            else:
                with torch.random.fork_rng(devices=[]):
                    native = make_native(factory, base, rank)
                make_native(factory, base, 4)
            point = f"{parent}.{'out_proj' if key == 'out_proj' else 'q_proj'}"
            update_kind = UPDATE.get(self.arm)
            update = load(update_kind, name if update_kind == "weight" else point) if update_kind else None
            left = None
            if update_kind == "weight":
                left = update["left"] if self.arm == "W128LR" else None
                update = update["right"]
            route = ROUTES.get(self.arm) if layer == 23 and key in PROJECTIONS[:3] else None
            read_basis = load(route, point) if route else None
            replacement = ProjectedLoRA(base, native, rank, update, left, read_basis, self.arm)
            setattr(vision.get_submodule(parent), key, replacement)
        if any(torch.count_nonzero(module.lora_B).item() for _, module in projections(vision)):
            raise ValueError("The initial E1124 adapters must leave the pretrained projections unchanged")
        return vision

    def features(self, data_dict):
        forward = super().features
        if self.arm in ROUTES and not FULL_READ.get() and (self.training or self.arm != "L5"):
            return paired_forward(blocks(self), (22,), lambda: forward(data_dict))
        return forward(data_dict)

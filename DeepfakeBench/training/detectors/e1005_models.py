"""Independent E1005 students, zero residuals, and raw CLIP controls.

This module has no dataset, detector-registry, or pretrained-download imports.
The supplied B0 and pristine pretrained model are construction inputs only;
every retained model is an independent copy. Video scoring belongs to the
runner/objective layer and remains mean per-frame probability.
"""

import copy
import importlib.util
import inspect
import json
import math
from numbers import Integral
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


SCOPES = ("H", "L", "J", "reset_head", "new_residual", "residual_head")
LOCAL_KINDS = ("local_swiglu", "shuffle_local", "late_local", "local_geglu")
DELTA_KINDS = ("pretrained_delta", "b0_delta", "ln_delta", "ln_metric")
PIXEL_KINDS = ("pixel224", "tamper", "boundary", "shuffled_mask", "interpolated448", "native448")
COLD_KINDS = ("cold_b0", "cold_video", "gend")


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _vision(module):
    if hasattr(module, "backbone"):
        module = module.backbone
    elif hasattr(module, "vision_model"):
        module = module.vision_model
    if not all(hasattr(module, name) for name in ("embeddings", "encoder", "pre_layrnorm", "post_layernorm")):
        raise ValueError("Model must expose a CLIP vision transformer")
    return module


def _dimension(vision):
    return int(vision.embeddings.class_embedding.numel())


def _image_size(vision):
    size = getattr(vision.config, "image_size", None)
    return _integer("CLIP image_size", size)


def _copy_frozen(module):
    """Copy LoRA without calling train/eval on the construction source."""
    result = copy.deepcopy(module)
    for child in result.modules():
        if isinstance(child, nn.Linear) and hasattr(child, "lora_A"):
            # A source using loralib's eval merging may arrive merged. Unmerge
            # only the copy before disabling subsequent train/eval rewriting.
            if getattr(child, "merged", False) and getattr(child, "merge_weights", False):
                child.train(True)
            if hasattr(child, "merge_weights"):
                child.merge_weights = False
    result.requires_grad_(False)
    for parameter in result.parameters():
        parameter.grad = None
    result.eval()
    return result


def _pristine_vision(base, pretrained):
    if pretrained is None:
        raise ValueError("This branch requires an explicitly supplied pristine pretrained CLIP")
    if pretrained is base:
        raise ValueError("pretrained must be pristine and independent of the fitted B0")
    supplied = _vision(pretrained)
    for module in supplied.modules():
        if hasattr(module, "lora_B") and torch.count_nonzero(module.lora_B.detach()).item():
            raise ValueError("pretrained contains fitted LoRA; supply pristine CLIP weights")
    vision = _copy_frozen(supplied)
    # A pristine Effort shell may already contain zero LoRA. D/R3 must read the
    # raw CLIP rather than retaining dormant fitted-adapter parameter names.
    for layer in vision.encoder.layers:
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            original = getattr(layer.self_attn, name)
            if not hasattr(original, "lora_A"):
                continue
            plain = nn.Linear(original.in_features, original.out_features,
                              bias=original.bias is not None,
                              device=original.weight.device, dtype=original.weight.dtype)
            with torch.no_grad():
                weight = original.weight.T if getattr(original, "fan_in_fan_out", False) else original.weight
                plain.weight.copy_(weight)
                if plain.bias is not None:
                    plain.bias.copy_(original.bias)
            plain.requires_grad_(False)
            setattr(layer.self_attn, name, plain)
    return vision


def _enable_layernorm(vision):
    for module in vision.modules():
        if isinstance(module, nn.LayerNorm):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)


def _linear_head(dim, device, dtype):
    return nn.Linear(dim, 2, device=device, dtype=dtype)


def _pooler(vision, images):
    result = vision(images)
    return result["pooler_output"] if isinstance(result, dict) else result.pooler_output


class _ColdLoRALinear(nn.Linear):
    """Effort's plain delta with the source factory's A initialization.

    The runner passes ``lora_initializer`` from the source Effort configuration:
    loralib uses Kaiming uniform, while Effort's custom backend uses N(0,.02).
    Both start with B=0 and alpha=16. The stored factor layout is conventional
    loralib for both; it does not change the underlying rank-four delta.
    """

    def __init__(self, original, rank=4, initializer="kaiming"):
        if initializer not in ("kaiming", "normal"):
            raise ValueError("lora_initializer must be kaiming or normal")
        super().__init__(original.in_features, original.out_features, bias=original.bias is not None,
                         device=original.weight.device, dtype=original.weight.dtype)
        self.r = rank
        self.scaling = 16 / rank
        self.lora_A = nn.Parameter(self.weight.new_empty(rank, self.in_features))
        self.lora_B = nn.Parameter(self.weight.new_zeros(self.out_features, rank))
        if initializer == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            nn.init.normal_(self.lora_A, std=.02)
        with torch.no_grad():
            self.weight.copy_(original.weight)
            if self.bias is not None:
                self.bias.copy_(original.bias)
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, inputs):
        return F.linear(inputs, self.weight, self.bias) + F.linear(F.linear(inputs, self.lora_A), self.lora_B) * self.scaling


class _RawClassifier(nn.Module):
    def __init__(self, vision, normalized=False):
        super().__init__()
        self.backbone = vision
        self.normalized = normalized
        reference = vision.embeddings.class_embedding
        self.head = _linear_head(_dimension(vision), reference.device, reference.dtype)

    def forward(self, data, inference=False):
        features = _pooler(self.backbone, data["image"])
        if self.normalized:
            features = F.normalize(features, dim=-1)
        logits = self.head(features)
        return {"cls": logits, "prob": logits.softmax(-1)[:, 1], "feat": features}


class LocalPatchAdapter(nn.Module):
    """Two bottleneck projections, gated activation, and spatial depthwise conv."""

    def __init__(self, hidden_dim, width=32, activation="silu", patch_count=None, shuffle=False, seed=1024):
        super().__init__()
        hidden_dim, width = _integer("hidden_dim", hidden_dim), _integer("adapter_width", width)
        if activation not in ("silu", "gelu"):
            raise ValueError("Local adapter activation must be silu or gelu")
        self.activation = activation
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, width)
        self.gate = nn.Linear(hidden_dim, width)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.up = nn.Linear(width, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        if shuffle:
            patch_count = _integer("patch_count", patch_count)
            generator = torch.Generator().manual_seed(_integer("shuffle_seed", seed, 0))
            permutation = torch.randperm(patch_count, generator=generator)
            self.register_buffer("permutation", permutation)
            self.register_buffer("inverse_permutation", permutation.argsort())
        else:
            self.register_buffer("permutation", None)
            self.register_buffer("inverse_permutation", None)

    def forward(self, patches):
        batch, count, _ = patches.shape
        side = math.isqrt(count)
        if side * side != count:
            raise ValueError("Local adapters require a square CLIP patch grid")
        normalized = self.norm(patches)
        gate = self.gate(normalized)
        gate = F.silu(gate) if self.activation == "silu" else F.gelu(gate)
        values = self.down(normalized) * gate
        if self.permutation is not None:
            if self.permutation.numel() != count:
                raise ValueError("CLIP patch count differs from the fixed shuffle grid")
            values = values[:, self.permutation]
        values = values.transpose(1, 2).reshape(batch, -1, side, side)
        values = self.spatial(values).flatten(2).transpose(1, 2)
        if self.inverse_permutation is not None:
            values = values[:, self.inverse_permutation]
        return patches + self.up(values)


class PixelBranch(nn.Module):
    """505,505 parameters at hidden_dim=1024 and two injection layers."""

    def __init__(self, hidden_dim, layers):
        super().__init__()
        blocks = []
        previous = 3
        for channels, stride in ((32, 2), (64, 2), (128, 2), (128, 1)):
            blocks.extend((nn.Conv2d(previous, channels, 3, stride=stride, padding=1, bias=False),
                           nn.GroupNorm(8, channels), nn.SiLU()))
            previous = channels
        self.encoder = nn.Sequential(*blocks)
        self.mask_head = nn.Conv2d(128, 1, 1)
        self.injections = nn.ModuleDict({str(layer): nn.Conv2d(128, hidden_dim, 1, bias=True) for layer in layers})
        for projection in self.injections.values():
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, images):
        features = self.encoder(images)
        return features, self.mask_head(features)


def _layers(options, name, defaults, depth):
    requested = options.get(name, defaults)
    if not isinstance(requested, (tuple, list)) or not requested:
        raise ValueError(f"{name} must be a nonempty list of zero-based block indices")
    layers = [_integer(name, layer, 0) for layer in requested]
    if len(set(layers)) != len(layers):
        raise ValueError(f"{name} contains duplicate block indices")
    if any(layer >= depth - 1 for layer in layers):
        raise ValueError(f"{name} requires a subsequent attention block; injection after the last block cannot influence CLS")
    return layers


def _matrix_model(base, options):
    path = Path(__file__).with_name("e1002_matrix.py")
    spec = importlib.util.spec_from_file_location("e1005_plain_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MatrixAdaptation(base, "lora", rank=_integer("rank", options.get("rank", 4)),
                                   last_layers=_integer("last_layers", options.get("last_layers", 4)),
                                   preserve_top=0)


class E1005Model(nn.Module):
    def __init__(self, base, spec, pretrained=None, options=None):
        super().__init__()
        if not isinstance(spec, dict) or (options is not None and not isinstance(options, dict)):
            raise ValueError("spec and options must be serializable dictionaries")
        options = copy.deepcopy(options or {})
        try:
            # Round-trip prevents live objects from entering an artifact signature.
            spec = json.loads(json.dumps(spec, allow_nan=False))
            options = json.loads(json.dumps(options, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ValueError("spec and options must be JSON serializable") from error
        kind = spec.get("kind", "warm")
        if kind in (None, "b0", "effort", "scope"):
            kind = "warm"
        if kind in SCOPES:
            spec["scope"], kind = kind, "warm"
        allowed = ("warm",) + LOCAL_KINDS + DELTA_KINDS + PIXEL_KINDS + COLD_KINDS
        if kind not in allowed:
            raise ValueError(f"Unsupported E1005 model kind: {kind}")
        self.kind = kind
        self.settings = {"spec": spec, "kind": kind, "options": options}
        self._new_prefixes = []
        self._layer_kwargs = []
        if kind == "warm":
            scope = spec.get("scope", "J")
            if scope not in SCOPES:
                raise ValueError(f"scope must be one of {SCOPES}")
            if not isinstance(getattr(base, "head", None), nn.Module):
                raise ValueError("B0 must expose its mature classifier head")
            if scope in ("new_residual", "residual_head"):
                matrix = _matrix_model(base, options)
                self.student = matrix.model
                self.settings["matrix"] = matrix.settings
                for parameter in self.student.head.parameters():
                    parameter.requires_grad_(scope == "residual_head")
            else:
                self.student = _copy_frozen(base)
                if scope == "reset_head":
                    original = self.student.head
                    if not isinstance(original, nn.Linear) or hasattr(original, "lora_A") or original.out_features != 2:
                        raise ValueError("reset_head requires the mature two-class plain linear head")
                    self.student.head = nn.Linear(original.in_features, original.out_features,
                                                  bias=original.bias is not None,
                                                  device=original.weight.device, dtype=original.weight.dtype)
                if scope in ("H", "J", "reset_head"):
                    self.student.head.requires_grad_(True)
                if scope in ("L", "J"):
                    count = 0
                    for name, parameter in self.student.backbone.named_parameters():
                        if name.endswith(("lora_A", "lora_B")):
                            parameter.requires_grad_(True)
                            count += 1
                    if not count:
                        raise ValueError("Scope L/J requires B0's existing fitted LoRA")
            self.settings["scope"] = scope
        elif kind in COLD_KINDS:
            vision = _pristine_vision(base, pretrained)
            if kind == "gend":
                _enable_layernorm(vision)
            else:
                rank = _integer("rank", options.get("rank", 4))
                initializer = options.get("lora_initializer", "kaiming")
                if initializer not in ("kaiming", "normal"):
                    raise ValueError("lora_initializer must be kaiming or normal")
                self.settings["lora_initializer"] = initializer
                for layer in vision.encoder.layers:
                    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                        original = getattr(layer.self_attn, name)
                        if rank > min(original.in_features, original.out_features):
                            raise ValueError("cold LoRA rank exceeds projection dimensions")
                        setattr(layer.self_attn, name, _ColdLoRALinear(original, rank, initializer))
            self.student = _RawClassifier(vision, normalized=kind == "gend")
            self.settings["l2_features"] = kind == "gend"
        elif kind in DELTA_KINDS:
            self.anchor = _copy_frozen(base)
            if kind == "b0_delta":
                self.branch = _copy_frozen(base)
                vision = _vision(self.branch)
            else:
                self.branch = _pristine_vision(base, pretrained)
                vision = self.branch
                if kind in ("ln_delta", "ln_metric"):
                    _enable_layernorm(vision)
            reference = vision.embeddings.class_embedding
            self.delta_head = nn.Linear(_dimension(vision), 1, device=reference.device, dtype=reference.dtype)
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
            self._new_prefixes.append("delta_head.")
            self.settings["l2_features"] = kind == "ln_metric"
        else:
            self.student = _copy_frozen(base)
            vision = _vision(self.student)
            depth, dim = len(vision.encoder.layers), _dimension(vision)
            reference = vision.embeddings.class_embedding
            if kind in LOCAL_KINDS:
                key, defaults = ("late_layers", [16, 20, 22]) if kind == "late_local" else ("local_layers", [4, 8, 12])
                selected = _layers(options, key, defaults, depth)
                width = _integer("adapter_width", options.get("adapter_width", 32))
                count = int(vision.embeddings.num_positions) - 1
                self.adapters = nn.ModuleDict({str(layer): LocalPatchAdapter(
                    dim, width, activation="gelu" if kind == "local_geglu" else "silu",
                    patch_count=count, shuffle=kind == "shuffle_local", seed=options.get("shuffle_seed", 1024))
                    for layer in selected}).to(device=reference.device, dtype=reference.dtype)
                self._new_prefixes.append("adapters.")
            else:
                selected = _layers(options, "pixel_layers", [8, 12], depth)
                self.pixel = PixelBranch(dim, selected).to(device=reference.device, dtype=reference.dtype)
                self._new_prefixes.append("pixel.")
                self._aux_size = _integer("aux_image_size", options.get("aux_image_size", 2 * _image_size(vision)))
                self.settings["aux_image_size"] = self._aux_size if kind in ("native448", "interpolated448") else None
            self.settings["layers"] = selected
            self.settings["patch_only"] = True
            for layer in vision.encoder.layers:
                parameters = inspect.signature(layer.forward).parameters
                self._layer_kwargs.append({name: value for name, value in {
                    "attention_mask": None, "causal_attention_mask": None, "output_attentions": False}.items()
                    if name in parameters})
        self.train(False)

    def train(self, mode=True):
        # Copied backbone dropout remains eval; affine LN and LoRA gradients
        # still flow. Avoid transient LoRA train/eval merge operations.
        self.training = mode
        if hasattr(self, "student"):
            self.student.eval()
            self.student.head.train(mode)
        if hasattr(self, "anchor"):
            self.anchor.eval()
            self.branch.eval()
            self.delta_head.train(mode)
        if hasattr(self, "adapters"):
            self.adapters.train(mode)
        if hasattr(self, "pixel"):
            self.pixel.train(mode)
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def snapshot_state(self):
        """Detached clones of trained values plus owned new-module buffers."""
        keys = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        keys.update(name for name, _ in self.named_buffers()
                    if any(name.startswith(prefix) for prefix in self._new_prefixes))
        return {name: value.detach().clone() for name, value in self.state_dict().items() if name in keys}

    def restore_snapshot(self, state):
        """Validate the complete small snapshot before changing any tensor."""
        expected = self.snapshot_state()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("Snapshot keys must exactly match trainable parameters and owned buffers")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError(f"Snapshot shape/dtype mismatch: {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Snapshot values must be finite: {name}")
        current = self.state_dict()
        with torch.no_grad():
            for name, value in state.items():
                current[name].copy_(value)

    load_snapshot = restore_snapshot

    def _injected(self, data):
        images = data["image"]
        vision = _vision(self.student)
        prepared = self.student._prep_input(images) if hasattr(self.student, "_prep_input") else images
        hidden = vision.pre_layrnorm(vision.embeddings(prepared))
        pixel_features, mask_logits = None, None
        if hasattr(self, "pixel"):
            pixel_images = images
            if self.kind in ("native448", "interpolated448"):
                pixel_images = data.get("aux_image")
                if (not isinstance(pixel_images, torch.Tensor) or pixel_images.ndim != 4
                        or pixel_images.shape != (len(images), 3, self._aux_size, self._aux_size)
                        or pixel_images.device != images.device or pixel_images.dtype != images.dtype):
                    raise ValueError("aux_image must be a same-batch, same-device native RGB crop at the configured resolution")
                if self.kind == "interpolated448":
                    # The caller registers B0's low-resolution input to this
                    # native crop. Reuse that exact tensor so a second resize
                    # kernel cannot introduce a different low-resolution signal.
                    pixel_images = F.interpolate(images, size=(self._aux_size, self._aux_size), mode="bilinear", align_corners=False)
            pixel_features, mask_logits = self.pixel(pixel_images)
        for index, layer in enumerate(vision.encoder.layers):
            output = layer(hidden, **self._layer_kwargs[index])
            hidden = output if isinstance(output, torch.Tensor) else output[0]
            key = str(index)
            if hasattr(self, "adapters") and key in self.adapters:
                hidden = torch.cat((hidden[:, :1], self.adapters[key](hidden[:, 1:])), dim=1)
            if pixel_features is not None and key in self.pixel.injections:
                side = math.isqrt(hidden.shape[1] - 1)
                if side * side != hidden.shape[1] - 1:
                    raise ValueError("Pixel injections require a square CLIP patch grid")
                delta = self.pixel.injections[key](pixel_features)
                delta = F.interpolate(delta, size=(side, side), mode="bilinear", align_corners=False)
                hidden = torch.cat((hidden[:, :1], hidden[:, 1:] + delta.flatten(2).transpose(1, 2)), dim=1)
        features = vision.post_layernorm(hidden[:, 0])
        logits = self.student.head(features)
        output = {"cls": logits, "prob": logits.softmax(-1)[:, 1], "feat": features}
        if mask_logits is not None:
            output["mask_logits"] = mask_logits
        return output

    def forward(self, data, inference=False):
        images = data.get("image") if isinstance(data, dict) else None
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("E1005 requires image[N,3,H,W]")
        reference = _vision(self.anchor if hasattr(self, "anchor") else self.student)
        if images.shape[-2:] != (_image_size(reference), _image_size(reference)):
            raise ValueError("image must match B0's original CLIP input resolution")
        if self.kind in LOCAL_KINDS + PIXEL_KINDS:
            return self._injected(data)
        if self.kind in DELTA_KINDS:
            with torch.no_grad():
                anchor = self.anchor(data, inference=True)
            if self.kind == "b0_delta":
                with torch.no_grad():
                    features = self.branch(data, inference=True)["feat"]
            elif self.kind == "pretrained_delta":
                with torch.no_grad():
                    features = _pooler(self.branch, images)
            else:
                features = _pooler(self.branch, images)
            if self.kind == "ln_metric":
                features = F.normalize(features, dim=-1)
            delta = self.delta_head(features)
            logits = anchor["cls"] + torch.cat((torch.zeros_like(delta), delta), dim=-1)
            return {"cls": logits, "prob": logits.softmax(-1)[:, 1], "feat": features}
        return self.student(data, inference=inference)


def build_model(base, spec, pretrained=None, options=None):
    """Build an independent student from a serializable E1005 arm spec."""
    return E1005Model(base, spec, pretrained=pretrained, options=options)

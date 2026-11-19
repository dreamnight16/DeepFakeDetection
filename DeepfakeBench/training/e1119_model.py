"""Model operators for E1119. The CLIP encoder and attention kernels are native."""

from contextlib import contextmanager
from contextvars import ContextVar
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")
SPLIT = {"I01": (22,), "I02": (14,), "I04": tuple(range(0, 24, 2))}
DETACH = {"I03": (23,), "I05": tuple(range(1, 24, 2))}
ASSET_SEED = 1119
EFFORT_REVISION = "96f5dea2b534d400cfd7003f053c7e93c8e16461"
OFF = ContextVar("e1119_adapter_off", default=False)
REFERENCE = ContextVar("e1119_reference", default=None)


def projections(vision):
    # This also preserves the native projection initialization order.
    for name, module in vision.named_modules():
        if name.rsplit(".", 1)[-1] in PROJECTIONS:
            yield name, module


@contextmanager
def setting(variable, value):
    token = variable.set(value)
    try:
        yield
    finally:
        variable.reset(token)


@contextmanager
def hooks():
    handles = []
    try:
        yield handles
    finally:
        for handle in reversed(handles):
            handle.remove()


def blocks(model):
    return model.backbone.encoder.layers


def hidden(output):
    return output[0] if isinstance(output, tuple) else output


def run_block(block, x):
    kwargs = {"attention_mask": None}
    if "causal_attention_mask" in inspect.signature(block.forward).parameters:
        kwargs["causal_attention_mask"] = None
    return hidden(block(x, **kwargs))


class LoRA(nn.Module):
    def __init__(self, base, native, rank, basis=None, route=None):
        super().__init__()
        self.weight = base.weight
        self.bias = base.bias
        self.use_loralib = hasattr(native, "fan_in_fan_out")
        a = native.lora_A.T if self.use_loralib else native.lora_A
        b = native.lora_B.T if self.use_loralib else native.lora_B
        self.lora_A = nn.Parameter(a[:, :rank].detach().contiguous())
        self.lora_B = nn.Parameter(b[:rank].detach().contiguous())
        self.scaling = native.scaling
        self.route = route
        self.register_buffer("basis", basis)

    def delta(self, x):
        a = self.lora_A
        if self.basis is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                a = a - self.basis @ (self.basis.T @ a)
        if self.use_loralib:
            return F.linear(F.linear(x, a.T), self.lora_B.T) * self.scaling
        return (x @ a @ self.lora_B) * self.scaling

    def update_weight(self):
        # Use the rank-sized input basis directly; no d-by-d projector.
        a = self.lora_A
        if self.basis is not None:
            a = a - self.basis @ (self.basis.T @ a)
        return self.scaling * (self.lora_B.T @ a.T)

    def fold(self):
        return self.weight + self.update_weight()

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        if OFF.get():
            return base
        if self.route == "detach" and self.training:
            return base + self.delta(x.detach())
        if self.route == "split":
            return base + self.delta(REFERENCE.get()[self])
        return base + self.delta(x)


class Spectral(nn.Module):
    def __init__(self, base, arm, asset):
        super().__init__()
        self.arm = arm
        self.bias = base.bias
        left, sigma, right_t = (asset[key] for key in ("u", "s", "vh"))
        if arm in ("M01", "M02"):
            self.u = nn.Parameter(left[:, -1].clone())
            self.v = nn.Parameter(right_t[-1].clone())
            self.s = nn.Parameter(sigma[-1].clone())
            self.register_buffer("weight", base.weight.detach() - sigma[-1] * torch.outer(self.u.detach(), self.v.detach()))
            if arm == "M02":
                self.register_buffer("u_main", left[:, :-1].contiguous())
                self.register_buffer("v_main_t", right_t[:-1].contiguous())
                # Compute this difference before casting to FP32 to avoid cancellation.
                self.register_buffer("norm_offset", self.weight.double().square().sum()
                                     .sub(base.weight.double().square().sum()).float().detach())
        else:
            self.weight = base.weight
            self.register_buffer("left", left)
            self.register_buffer("sigma", sigma)
            self.register_buffer("right_t", right_t)
            self.coeff = nn.Parameter(torch.zeros_like(sigma))

    def rotate(self, x):
        for shift, angles in enumerate(self.coeff.chunk(2)):
            first, second = x.roll(-shift, dims=-1).unflatten(-1, (-1, 2)).unbind(-1)
            c, s = angles.cos(), angles.sin()
            x = torch.stack((first * c - second * s, first * s + second * c), dim=-1)
            x = x.flatten(-2).roll(shift, dims=-1)
        return x

    def forward(self, x):
        if self.arm in ("M01", "M02"):
            return F.linear(x, self.weight, self.bias) + (x @ self.v).unsqueeze(-1) * (self.s * self.u)
        if self.arm == "M05":
            return F.linear(self.rotate(F.linear(x, self.right_t) * self.sigma), self.left, self.bias)
        residual = F.linear(x, self.right_t) * (self.sigma * self.coeff.expm1())
        return F.linear(x, self.weight, self.bias) + F.linear(residual, self.left)

    def regularizers(self):
        u2, v2 = self.u.square().sum(), self.v.square().sum()
        orth = 2 * ((self.u_main.T @ self.u).square().sum() +
                    (self.v_main_t @ self.v).square().sum())
        orth = orth + (u2 - 1).square() + (v2 - 1).square()
        norm = self.norm_offset + 2 * self.s * (self.u @ (self.weight @ self.v))
        return orth, (norm + self.s.square() * u2 * v2).abs()

    def fold(self):
        if self.arm in ("M01", "M02"):
            return self.weight + self.s * torch.outer(self.u, self.v)
        if self.arm == "M05":
            return self.left @ self.rotate((self.sigma[:, None] * self.right_t).T).T
        return self.weight + (self.left * (self.sigma * self.coeff.expm1())) @ self.right_t


def native_delta(module, x):
    if isinstance(module, LoRA):
        return module.delta(x)
    if hasattr(module, "fan_in_fan_out"):
        return F.linear(F.linear(x, module.lora_A), module.lora_B) * module.scaling
    return (x @ module.lora_A @ module.lora_B) * module.scaling


def base_only(module, args, output):
    return F.linear(args[0], module.weight, module.bias)


@contextmanager
def bypass(layers):
    # Native B00 is used verbatim. Its read-only interventions replace outputs
    # with the base projection, while custom LoRA implements its own switch.
    with hooks() as handles, setting(OFF, True):
        for layer in layers:
            for _, module in projections(layer):
                if hasattr(module, "lora_A") and not isinstance(module, LoRA):
                    handles.append(module.register_forward_hook(base_only))
        yield


def relative_drift(student, teacher):
    student, teacher = student.float(), teacher.float()
    return (student - teacher).square().mean((1, 2)) / (teacher.square().mean((1, 2)) + 1e-6)


def selected_indices(labels, mixed, generator, step):
    real = (labels == 0).nonzero().flatten()
    if not mixed or not len(real):
        return real
    fake = (labels == 1).nonzero().flatten()
    count = len(real)
    n_fake = min(len(fake), count // 2 + (count % 2) * (step % 2))
    real_order = torch.randperm(count, generator=generator)[:count - n_fake].to(real.device)
    fake_order = torch.randperm(len(fake), generator=generator)[:n_fake].to(fake.device)
    return torch.cat((real[real_order], fake[fake_order]))



def paired_forward(layers, indices, forward):
    references = {}
    with hooks() as handles, setting(REFERENCE, references):
        for idx in indices:
            next_block = layers[idx + 1]

            def capture(block, args, kwargs, next_block=next_block):
                with setting(OFF, True):
                    reference = hidden(block.forward(*args, **kwargs))
                normalized = next_block.layer_norm1(reference)
                for key in PROJECTIONS[:3]:
                    references[getattr(next_block.self_attn, key)] = normalized

            handles.append(layers[idx].register_forward_pre_hook(capture, with_kwargs=True))
        return forward()

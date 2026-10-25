"""Two additive B0 adapters with separate real/fake parameter gradients."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class DualLoRALinear(nn.Module):
    def __init__(self, original, real_rank, fake_rank):
        super().__init__()
        for rank in (real_rank, fake_rank):
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                raise ValueError("E1021 LoRA ranks must be positive integers")
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        # B0 uses alpha/r = 16/4. Keep it fixed when ranks differ.
        self.scaling = 4.0
        for branch, rank in (("real", real_rank), ("fake", fake_rank)):
            a = nn.Parameter(self.weight.new_empty(rank, self.in_features))
            b = nn.Parameter(self.weight.new_zeros(self.out_features, rank))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.register_parameter(f"lora_A_{branch}", a)
            self.register_parameter(f"lora_B_{branch}", b)

    def forward(self, x):
        real = F.linear(F.linear(x, self.lora_A_real), self.lora_B_real)
        fake = F.linear(F.linear(x, self.lora_A_fake), self.lora_B_fake)
        return F.linear(x, self.weight, self.bias) + self.scaling * (real + fake)


def replace_lora(backbone, real_rank, fake_rank):
    # B0/head initialization and subsequent sampling retain their original RNG.
    with torch.random.fork_rng(devices=[]):
        for name, module in list(backbone.named_modules()):
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                parent_name, _, child_name = name.rpartition(".")
                parent = backbone.get_submodule(parent_name)
                setattr(parent, child_name, DualLoRALinear(module, real_rank, fake_rank))


def lora_parameter_groups(backbone):
    real, fake = [], []
    for module in backbone.modules():
        if isinstance(module, DualLoRALinear):
            real.extend((module.lora_A_real, module.lora_B_real))
            fake.extend((module.lora_A_fake, module.lora_B_fake))
    return real, fake


def class_conditional_backward(losses, real_parameters, fake_parameters, shared_parameters, beta=0.0):
    if not 0.0 <= beta <= 1.0:
        raise ValueError("LoRA routing beta must be finite and between 0 and 1")
    real_parameters = tuple(real_parameters)
    fake_parameters = tuple(fake_parameters)
    shared_parameters = tuple(shared_parameters)
    all_parameters = real_parameters + fake_parameters + shared_parameters
    real_loss, fake_loss = losses["real_loss"], losses["fake_loss"]
    # Keep absent-class gradients as None under the original hard routing.
    if beta > 0:
        real_loss, fake_loss = real_loss + beta * fake_loss, beta * real_loss + fake_loss
    groups = ((real_loss, real_parameters),
              (fake_loss, fake_parameters),
              (losses["overall"], shared_parameters))
    for index, (loss, parameters) in enumerate(groups):
        # Single-class validation is valid; an absent class has no Adam update.
        if not loss.requires_grad:
            for parameter in parameters:
                parameter.grad = None
            continue
        # A head-only final call stops at the classifier and leaves the retained
        # encoder activations alive through the epoch's loss recorders. Visit all
        # trainable paths to free them, keeping only the shared-head gradients.
        targets = all_parameters if index == 2 else parameters
        gradients = torch.autograd.grad(loss, targets, retain_graph=index < 2)
        if index == 2:
            gradients = gradients[len(real_parameters) + len(fake_parameters):]
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient

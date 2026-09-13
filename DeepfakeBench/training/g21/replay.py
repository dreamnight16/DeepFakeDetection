"""Exact first-order full-window gradients with only one microbatch graph in memory."""

from __future__ import annotations

import torch

from .config import ARMS
from .losses import classification_loss, group_risk, pairwise_losses, video_logits
from .randomness import capture_rng, restore_rng


def forward_videos(model, images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 7 or images.shape[1:3] != (2, 2):
        raise ValueError("expected [P,A=2,S=2,T,C,H,W]")
    p, a, s, t, c, h, w = images.shape
    output = model({"image": images.reshape(-1, c, h, w)})
    logits = output["cls"]
    if logits.shape != (p * a * s * t, 2):
        raise ValueError("model did not return [F,2] logits")
    return video_logits(logits.reshape(p, a, s, t, 2))


def window_backward(
    model,
    windows: list[torch.Tensor],
    *,
    arm: str,
    permutations: list[torch.Tensor],
    margin: float,
    temperature: float,
    device: str | torch.device = "cpu",
) -> dict:
    """Accumulate gradients only. The caller owns zero_grad and optimizer.step.

    All four arms run both passes. A bypasses ranking and reports a zero pair loss.
    Loss values are the original objective, not the weighted gradient surrogate.
    """
    if arm not in ARMS or not windows or len(permutations) != len(windows):
        raise ValueError("invalid arm/window/permutations")
    if not model.training:
        raise ValueError("gradient replay requires train mode in both passes")
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    states, values, class_losses, risks = [], [], [], []
    initial_rng = capture_rng()
    try:
        with torch.no_grad():
            for x, permutation in zip(windows, permutations):
                states.append(capture_rng())
                u = forward_videos(model, x.to(device))
                values.append(u.detach().cpu())
                class_losses.append(classification_loss(u))
                risks.append(
                    pairwise_losses(
                        u, permutation if arm == "C" else None, margin
                    ).mean(0)
                    if arm != "A"
                    else torch.zeros(u.shape[1], device=u.device)
                )
        after_rng = capture_rng()
        if any(not torch.equal(buffers[name], b) for name, b in model.named_buffers()):
            raise ValueError("mutable model buffers are unsupported by two-pass replay")
        all_risks = torch.stack(risks)
        pair_value, weights = group_risk(all_risks, ARMS[arm][1], temperature)
        weights = weights.detach()
        loss_value = torch.stack(class_losses).mean() + pair_value * ARMS[arm][2]
        for j, (x, permutation) in enumerate(zip(windows, permutations)):
            restore_rng(states[j])
            u = forward_videos(model, x.to(device))
            if not torch.allclose(u.detach().cpu(), values[j], atol=1e-6, rtol=1e-5):
                raise ValueError(
                    "replayed outputs changed: RNG/state or input mismatch"
                )
            surrogate = classification_loss(u) / len(windows)
            if arm != "A":
                risk = pairwise_losses(
                    u, permutation if arm == "C" else None, margin
                ).mean(0)
                surrogate = surrogate + (weights[j] * risk).sum()
            surrogate.backward()
        restore_rng(after_rng)
    except BaseException:
        restore_rng(initial_rng)
        with torch.no_grad():
            for name, b in model.named_buffers():
                b.copy_(buffers[name])
        raise
    predictions = torch.cat(values)
    return {
        "loss": loss_value.item(),
        "classification_loss": torch.stack(class_losses).mean().item(),
        "pair_loss": pair_value.item(),
        "group_risks": all_risks.cpu().tolist(),
        "group_weights": weights.cpu().tolist(),
        "mean_real_probability": predictions[..., 0].sigmoid().mean().item(),
        "mean_fake_probability": predictions[..., 1].sigmoid().mean().item(),
        "matched_pair_accuracy": (predictions[..., 1] > predictions[..., 0])
        .float()
        .mean()
        .item(),
        "forward_images": 2
        * sum(x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3] for x in windows),
        "backward_images": sum(
            x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3] for x in windows
        ),
    }

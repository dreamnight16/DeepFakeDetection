"""G27 auxiliary objectives and a label-preserving photometric candidate view."""

import torch

from .g25_tokens import make_attention_mask
from .g25v2_tokens import call_with_detached_parameters


def balance_loss(log_odds, labels, temperature):
    """Balance soft responsibility over fake samples, without requiring hard roles.

    Identical uniform queries also minimize this loss. It is a load constraint,
    not evidence that specialization has emerged. Empty fake batches return zero.
    """
    values = log_odds.float() if log_odds.dtype in (torch.float16, torch.bfloat16) else log_odds
    fake = values[labels == 1]
    if not len(fake):
        return values.sum() * 0
    responsibility = (fake / temperature).softmax(-1).mean(0)
    return (responsibility - 1 / values.shape[1]).square().sum()


def difficulty_weights(cls_prob, width, floor):
    """Detached uncertainty proxy; confident errors retain floor supervision."""
    uncertainty = (1 - (cls_prob.detach() - .5).abs() / width).clamp(0, 1)
    return floor + (1 - floor) * uncertainty


def photometric_view(images, std, contrast, brightness):
    """Fixed mild contrast/brightness change in normalized RGB space.

    In pixel space: x' = contrast*x + (1-contrast)*spatial_mean(x) + brightness.
    No cropping, clipping or RNG consumption. Assumes per-channel affine input
    normalization with the supplied std; the normalization mean cancels out.
    This is a candidate invariance, not proof all forensic cues are preserved.
    """
    scale = images.new_tensor(std).reshape(1, 3, 1, 1)
    return contrast * images + (1 - contrast) * images.mean((-2, -1), keepdim=True) + brightness / scale


def isolated_view_features(vision, images, evidence_tokens, insert_layer):
    """Second view: no main-loss graph, detached shared weights, token gradients on."""
    with torch.no_grad():
        boundary = vision.pre_layrnorm(vision.embeddings(images))
        for layer in vision.encoder.layers[:insert_layer]:
            boundary = layer(boundary, attention_mask=None, causal_attention_mask=None,
                             output_attentions=False)[0]
    batch, length, _ = boundary.shape
    hidden = torch.cat((boundary, evidence_tokens.to(boundary.dtype).expand(batch, -1, -1)), 1)
    mask = make_attention_mask(hidden, length, "read_only")
    for layer in vision.encoder.layers[insert_layer:]:
        hidden = call_with_detached_parameters(
            layer, hidden, attention_mask=mask, causal_attention_mask=None, output_attentions=False)[0]
    return call_with_detached_parameters(vision.post_layernorm, hidden[:, length:])

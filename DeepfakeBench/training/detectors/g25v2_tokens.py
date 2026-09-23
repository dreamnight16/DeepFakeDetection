"""G25v2: reuse late CLIP blocks with an isolated auxiliary backward path."""

import torch

from .g25_tokens import make_attention_mask


def call_with_detached_parameters(module, *args, **kwargs):
    """Freeze parameter gradients for this call, retaining INPUT gradients.

    No weights are copied or registered and no requires_grad flags are changed.
    In particular, this must NOT be wrapped in no_grad: evidence tokens learn
    through the fixed attention/FFN operators of this auxiliary call.
    """
    # Lazy import keeps registering G25v2 from changing old detectors' imports.
    from torch.func import functional_call

    parameters = {name: value.detach() for name, value in module.named_parameters()}
    buffers = {name: value.detach() for name, value in module.named_buffers()}
    return functional_call(module, (parameters, buffers), args, kwargs, strict=True)


def forward_isolated_evidence(vision, images, evidence_tokens, insert_layer, mode,
                              output_attentions=True, isolate_auxiliary=True):
    """Same forward values as G25; only the auxiliary loss graph changes.

    The prefix runs once. With gradients enabled, the suffix runs a second time
    with detached prefix features AND detached parameters for evidence losses.
    The main CLS path retains normal LoRA/CLS/token gradients. At inference the
    single main pass suffices. CLIP and adapter dropout must be zero, as in G25.
    """
    layers = vision.encoder.layers
    if not 0 <= insert_layer < len(layers):
        raise ValueError("insert_layer must select an existing CLIP block")
    boundary = vision.pre_layrnorm(vision.embeddings(images))
    batch, original_length, dim = boundary.shape
    if (evidence_tokens.ndim != 3 or evidence_tokens.shape[0] != 1
            or evidence_tokens.shape[1] < 1 or evidence_tokens.shape[2] != dim):
        raise ValueError("evidence_tokens must have shape [1, K>=1, hidden_size]")
    for layer in layers[:insert_layer]:
        boundary = layer(boundary, attention_mask=None, causal_attention_mask=None,
                         output_attentions=False)[0]

    def suffix(prefix, detached_parameters):
        hidden = torch.cat((prefix, evidence_tokens.to(dtype=prefix.dtype).expand(batch, -1, -1)), dim=1)
        mask = make_attention_mask(hidden, original_length, mode)
        attention = None
        for index in range(insert_layer, len(layers)):
            layer = layers[index]
            need_attention = output_attentions and index == len(layers) - 1
            kwargs = dict(attention_mask=mask, causal_attention_mask=None,
                          output_attentions=need_attention)
            outputs = (call_with_detached_parameters(layer, hidden, **kwargs)
                       if detached_parameters else layer(hidden, **kwargs))
            hidden = outputs[0]
            if need_attention:
                if len(outputs) < 2 or outputs[1] is None:
                    raise RuntimeError("G25v2 requires CLIP attention weights; use eager attention")
                attention = outputs[1][:, :, original_length:, 1:original_length].mean(1)
        selected = torch.cat((hidden[:, :1], hidden[:, original_length:]), dim=1)
        features = (call_with_detached_parameters(vision.post_layernorm, selected)
                    if detached_parameters else vision.post_layernorm(selected))
        return features, attention

    features, attention = suffix(boundary, False)
    if isolate_auxiliary and torch.is_grad_enabled():
        auxiliary_features, attention = suffix(boundary.detach(), True)
    else:
        auxiliary_features = features
    return {"features": features, "evidence_features": auxiliary_features[:, 1:],
            "attention_maps": attention}

"""E1009 tuning scopes on real tiny CLIP + LoRA, without model downloads."""

import copy
import sys
import types

import pytest
import torch

import test_g25
from test_g25 import DETECTORS, load_file, tokens
from test_g25 import detector_class as original_detector_class
from test_g25v2 import models as legacy_models


@pytest.fixture
def models(legacy_models, monkeypatch):
    # Reuse the existing real CLIP/LoRA fixture with six blocks so that the
    # last-four control has both frozen early blocks and trainable late blocks.
    original_tiny_vision = test_g25.tiny_vision

    def six_layer_vision(*args, **kwargs):
        vision = original_tiny_vision(*args, **kwargs)
        vision.encoder.layers.extend(copy.deepcopy(layer) for layer in list(vision.encoder.layers))
        vision.config.num_hidden_layers = 6
        return vision

    monkeypatch.setattr(test_g25, "tiny_vision", six_layer_vision)
    base = types.ModuleType("g25_test_package.effort_detector_g25v2")
    base.EffortDetectorG25v2 = legacy_models[1]
    monkeypatch.setitem(sys.modules, base.__name__, base)
    revised = load_file("g25_test_package.effort_detector_e1009", DETECTORS / "effort_detector_e1009.py")
    return *legacy_models, revised.EffortDetectorE1009


def config(**changes):
    return {"g25_insert_layer": 2, "g25_num_tokens": 3,
            "g25_supervision": "all", **changes}


def expected_trainable(tuning, num_tokens=3):
    names = {"backbone.embeddings.class_embedding", "head.weight", "head.bias"}
    if num_tokens:
        names.add("evidence_tokens")
        names.update(f"evidence_heads.heads.{index}.{part}"
                     for index in range(num_tokens) for part in ("weight", "bias"))
    if tuning in ("late_lora", "all_lora"):
        layers = range(2, 6) if tuning == "late_lora" else range(6)
        names.update(f"backbone.encoder.layers.{index}.self_attn.{projection}.lora_{part}"
                     for index in layers
                     for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
                     for part in ("A", "B"))
    if tuning == "layernorm":
        names.update(f"backbone.{norm}.{part}"
                     for norm in ("pre_layrnorm", "post_layernorm")
                     for part in ("weight", "bias"))
        names.update(f"backbone.encoder.layers.{index}.layer_norm{norm}.{part}"
                     for index in range(6) for norm in (1, 2) for part in ("weight", "bias"))
    return names


@pytest.mark.parametrize("tuning", ["tokens", "late_lora", "layernorm", "all_lora"])
def test_only_requested_parameters_are_trainable_and_summary_is_exact(models, tuning):
    model = models[2](config(e1009_tuning=tuning))
    actual = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert actual == expected_trainable(tuning)
    assert len(model.backbone.encoder.layers) == 6
    assert set(model.tuning_summary) == {"cls", "evidence_tokens", "heads", "lora", "layernorm"}
    summary_names = []
    parameters = dict(model.named_parameters())
    for group in model.tuning_summary.values():
        summary_names.extend(group["names"])
        assert group["numel"] == sum(parameters[name].numel() for name in group["names"])
    assert len(summary_names) == len(set(summary_names))
    assert set(summary_names) == actual
    assert model.tuning_summary["cls"]["names"] == ["backbone.embeddings.class_embedding"]
    assert model.tuning_summary["evidence_tokens"]["names"] == ["evidence_tokens"]
    assert set(model.tuning_summary["heads"]["names"]) == {
        "head.weight", "head.bias",
        *(f"evidence_heads.heads.{index}.{part}"
          for index in range(3) for part in ("weight", "bias")),
    }
    for name, parameter in model.backbone.named_parameters():
        if name.endswith("lora_B"):
            assert torch.count_nonzero(parameter) == 0


@pytest.mark.parametrize("aux_mode", ["joint", "isolated"])
def test_main_scope_keeps_frozen_parameters_bitwise_unchanged_after_two_adam_steps(models, aux_mode):
    model = models[2](config(g25v2_aux_grad_mode=aux_mode)).train()
    trainable = expected_trainable("tokens")
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=5e-4)
    for _ in range(2):
        data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}
        optimizer.zero_grad(set_to_none=True)
        model.get_losses(data, model(data))["overall"].backward()
        for name, parameter in model.named_parameters():
            if name in trainable:
                assert parameter.grad is not None, name
                assert parameter.grad.abs().sum() > 0, name
            else:
                assert parameter.grad is None, name
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name not in trainable:
                assert torch.equal(parameter.detach(), before[name]), name
    for name, parameter in model.named_parameters():
        if name in trainable:
            assert not torch.equal(parameter.detach(), before[name]), name


@pytest.mark.parametrize("mode", tokens.MASK_MODES)
def test_isolated_auxiliary_loss_cannot_update_cls_or_any_layernorm(models, mode):
    model = models[2](config(e1009_tuning="layernorm", g25_attention_mode=mode))
    data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}
    losses = model.get_losses(data, model(data))
    (losses["loss_evidence"] + 0.01 * losses["loss_diversity"]).backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("backbone.", "head.")):
            assert parameter.grad is None, name
    assert model.backbone.embeddings.class_embedding.requires_grad
    assert model.backbone.pre_layrnorm.weight.requires_grad
    assert model.backbone.post_layernorm.weight.requires_grad
    assert model.evidence_tokens.grad.abs().sum() > 0
    assert all(head.weight.grad.abs().sum() > 0 for head in model.evidence_heads.heads)


@pytest.mark.parametrize("aux_mode", ["joint", "isolated"])
@pytest.mark.parametrize("mode", tokens.MASK_MODES)
def test_original_all_layer_lora_matches_g25_or_g25v2_values_losses_and_gradients(models, aux_mode, mode):
    settings = config(e1009_tuning="all_lora", g25v2_aux_grad_mode=aux_mode,
                      g25_attention_mode=mode)
    reference = models[0 if aux_mode == "joint" else 1](settings)
    revised = models[2](settings)
    revised.load_state_dict(reference.state_dict(), strict=True)
    assert list(revised.state_dict()) == list(reference.state_dict())
    data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}
    outputs = [model(data) for model in (reference, revised)]
    for key in ("cls", "prob", "feat", "g25"):
        torch.testing.assert_close(outputs[0][key], outputs[1][key])
    losses = [model.get_losses(data, output) for model, output in zip((reference, revised), outputs)]
    torch.testing.assert_close(losses[0], losses[1])
    for loss in losses:
        loss["overall"].backward()
    for name, parameter in revised.named_parameters():
        original = dict(reference.named_parameters())[name]
        assert parameter.requires_grad == original.requires_grad, name
        if original.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, original.grad)


@pytest.mark.parametrize("tuning", ["tokens", "late_lora", "layernorm", "all_lora"])
def test_strict_checkpoint_reload_preserves_scores_and_tuning_scope(models, tuning):
    settings = config(e1009_tuning=tuning)
    model = models[2](settings).eval()
    restored = models[2](settings).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    images = torch.randn(2, 3, 3, 8, 8)
    with torch.no_grad():
        torch.testing.assert_close(restored({"image": images}, inference=True),
                                   model({"image": images}, inference=True))
    assert {name for name, parameter in restored.named_parameters() if parameter.requires_grad} == expected_trainable(tuning)


def test_cls_only_control_has_no_evidence_parameters(models):
    model = models[2](config(g25_num_tokens=0))
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == expected_trainable("tokens", num_tokens=0)
    assert not any("evidence" in name for name in model.state_dict())


def test_unknown_tuning_scope_is_rejected(models):
    with pytest.raises(ValueError, match="e1009_tuning"):
        models[2](config(e1009_tuning="wrong"))

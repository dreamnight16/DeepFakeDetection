"""E1012 joint and isolated learning contracts on real tiny CLIP + LoRA."""

import copy
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch.nn import functional as F

from test_g30 import TinyB0


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e1012_tokens_test", ROOT / "training/detectors/e1012_tokens.py")
e1012 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e1012)

OBJECTIVES = [("off", False), ("uniform", False), ("weighted", False),
              ("off", True), ("uniform", True), ("weighted", True)]


def model(family="aux", **settings):
    return e1012.JointForgeryModel(
        TinyB0("sdpa"), family=family, num_tokens=3, memory_layer=1,
        hidden_dim=8, num_heads=4, depth=2, lfeq_dropout=0., **settings)


def batch():
    return {"image": torch.randn(4, 3, 8, 8)}, torch.tensor([0, 1, 1, 0])


@pytest.mark.parametrize("family", ["lfeq", "aux", "decoder"])
@pytest.mark.parametrize("likelihood,asymmetric", OBJECTIVES)
def test_all_supervision_arms_update_lora_without_unfreezing_pretrained_clip(family, likelihood, asymmetric):
    current = model(family, likelihood=likelihood, asymmetric=asymmetric).train()
    data, labels = batch()
    before = current.base.backbone.encoder.layers[0].self_attn.v_proj.lora_B.detach().clone()
    optimizer = torch.optim.Adam((p for p in current.parameters() if p.requires_grad), lr=.01)
    output = current(data)
    losses = current.losses(output, labels)
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["overall"].backward()
    assert current.base.backbone.encoder.layers[0].self_attn.v_proj.lora_B.grad.abs().sum() > 0
    assert current.base.backbone.embeddings.patch_embedding.weight.grad is None
    assert current.base.backbone.embeddings.position_embedding.weight.grad is None
    optimizer.step()
    assert not torch.equal(before, current.base.backbone.encoder.layers[0].self_attn.v_proj.lora_B)
    assert current.base.training


@pytest.mark.parametrize("mask", ["read_only", "cls_only", "patch_only", "full"])
@pytest.mark.parametrize("likelihood", ["off", "uniform", "weighted"])
def test_isolated_auxiliary_losses_do_not_reach_lora_cls_or_global_head(mask, likelihood):
    current = model(aux_grad_mode="isolated", attention_mode=mask, likelihood=likelihood,
                    asymmetric=True).train()
    data, labels = batch()
    losses = current.losses(current(data), labels)
    (losses["loss_evidence"] + .01 * losses["loss_diversity"]
     + .14 * losses["loss_likelihood"]).backward()
    assert all(parameter.grad is None for parameter in current.base.parameters())
    assert current.memory_projection.weight.grad.abs().sum() > 0
    assert current.evidence_tokens.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["joint", "isolated"])
def test_global_ce_updates_early_late_lora_original_cls_and_linear_head(mode):
    current = model(aux_grad_mode=mode).train()
    data, labels = batch()
    current.losses(current(data), labels)["loss_global"].backward()
    vision = current.base.backbone
    assert vision.embeddings.class_embedding.requires_grad
    assert vision.embeddings.class_embedding.grad.abs().sum() > 0
    assert current.base.head.weight.grad.abs().sum() > 0
    for index in (0, -1):
        assert vision.encoder.layers[index].self_attn.v_proj.lora_B.grad.abs().sum() > 0


def test_isolated_total_original_parameter_gradients_equal_global_ce_only():
    current = model(aux_grad_mode="isolated", attention_mode="full", likelihood="weighted").train()
    reference = copy.deepcopy(current)
    data, labels = batch()
    current.losses(current(data), labels)["overall"].backward()
    reference.losses(reference(data), labels)["loss_global"].backward()
    for (name, parameter), (_, expected) in zip(current.base.named_parameters(), reference.base.named_parameters()):
        if expected.grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, expected.grad)


@pytest.mark.parametrize("weight", [0., .5, 1.])
def test_lfeq_uses_original_readout_and_freezes_unused_base_head_and_cls(weight):
    current = model("lfeq", lfeq_fusion_weight=weight).eval()
    data, _ = batch()
    output = current(data)
    values = output["e1012"]
    decision = values["global_logits"].softmax(-1)[:, 1]
    evidence = values["evidence_logits"].softmax(-1)[..., 1].max(1).values
    torch.testing.assert_close(output["prob"], weight * decision + (1 - weight) * evidence)
    assert not current.base.backbone.embeddings.class_embedding.requires_grad
    assert not any(p.requires_grad for p in current.base.head.parameters())


def test_g25_scores_the_same_augmented_sequence_without_outer_b0_gate():
    current = model(attention_mode="full").eval()
    with torch.no_grad():
        current.memory_projection.weight.mul_(.4)
        current.memory_projection.bias.fill_(.2)
    data, _ = batch()
    output = current(data)
    values = output["e1012"]
    torch.testing.assert_close(output["prob"], .5 * values["cls_prob"] + .5 * values["evidence_prob"])
    original = current.base(data)
    assert not torch.allclose(values["global_logits"], original["cls"])
    assert values["attention_maps"].shape == (4, 3, 4)
    assert len(list(current.parameters())) == len({id(p) for p in current.parameters()})


def test_decoder_evidence_gradient_reaches_prefix_lora():
    current = model("decoder").train()
    data, labels = batch()
    current.losses(current(data), labels)["loss_evidence"].backward()
    assert current.base.backbone.encoder.layers[0].self_attn.v_proj.lora_B.grad.abs().sum() > 0
    assert current.base.head.weight.grad is None


@pytest.mark.parametrize("family", ["lfeq", "aux", "decoder"])
@pytest.mark.parametrize("label", [0, 1])
def test_single_class_validation_disables_batch_likelihood(family, label):
    current = model(family, likelihood="weighted").eval()
    data, _ = batch()
    labels = torch.full((4,), label)
    with torch.no_grad():
        output = current(data, inference=True)
        losses = current.losses(output, labels)
    assert losses["loss_likelihood"] == 0
    assert torch.isfinite(losses["overall"])


@pytest.mark.parametrize("family", ["lfeq", "aux", "decoder"])
def test_full_checkpoint_reload_preserves_primary_and_all_branch_scores(family, tmp_path):
    current = model(family, likelihood="uniform").train()
    data, labels = batch()
    optimizer = torch.optim.Adam(current.parameters(), lr=.01)
    current.losses(current(data), labels)["overall"].backward()
    optimizer.step()
    current.eval()
    path = tmp_path / "joint.pth"
    torch.save(current.state_dict(), path)
    restored = model(family, likelihood="uniform").eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        first, second = current(data), restored(data)
    for key in ("cls", "prob", "feat"):
        assert torch.equal(first[key], second[key])
    for key in ("cls_prob", "evidence_prob", "primary_prob"):
        assert torch.equal(first["e1012"][key], second["e1012"][key])
    assert any(name.startswith("base.backbone") for name in current.state_dict())


@pytest.mark.parametrize("family", ["lfeq", "aux", "decoder"])
def test_trainer_can_compute_validation_losses_for_the_selected_crop(monkeypatch, family):
    package = types.ModuleType("e1012_wrapper_test")
    package.__path__ = []
    base = types.ModuleType("e1012_wrapper_test.effort_detector")
    base.EffortDetector = lambda config: TinyB0("sdpa")
    registry = types.ModuleType("detectors")
    registry.DETECTOR = types.SimpleNamespace(register_module=lambda **kwargs: lambda cls: cls)
    metrics = types.ModuleType("metrics.base_metrics_class")
    metrics.calculate_metrics_for_train = lambda *args: None
    for name, module in ((package.__name__, package), (base.__name__, base),
                         ("e1012_wrapper_test.e1012_tokens", e1012),
                         ("detectors", registry), (metrics.__name__, metrics)):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "e1012_wrapper_test.effort_detector_e1012", ROOT / "training/detectors/effort_detector_e1012.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    current = wrapper.EffortDetectorE1012({"e1012_settings": {
        "family": family, "num_tokens": 3, "memory_layer": 1, "hidden_dim": 8,
        "num_heads": 4, "depth": 1, "likelihood": "weighted"}}).eval()
    data = {"image": torch.randn(2, 3, 3, 8, 8), "label": torch.tensor([0, 1])}
    with torch.no_grad():
        output = current(data, inference=True)
        losses = current.get_losses(data, output)
    assert torch.isfinite(losses["overall"])
    assert losses["loss_likelihood"] == 0
    assert output["e1012"]["global_logits"].shape == (2, 2)
    torch.testing.assert_close(output["prob"], output["e1012"]["primary_prob"])

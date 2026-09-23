"""G26 CPU contracts: asymmetric MIL, selective scoring, and B0 isolation."""

import ast
import math
import sys

import pytest
import torch
from torch.nn import functional as F

from test_g25 import DETECTORS, detector_class, load_file


def configure_lora_backend(monkeypatch, backend):
    if backend == "loralib":
        return
    import test_g25
    original = test_g25.tiny_vision
    tree = ast.parse((DETECTORS / "effort_detector.py").read_text(encoding="utf-8"))
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Linear")
    namespace = {"torch": torch, "nn": torch.nn, "F": F, "math": math}
    exec(compile(ast.Module(body=[source], type_ignores=[]), "actual_effort_Linear", "exec"), namespace)

    def custom_vision(lora=False, implementation="eager"):
        vision = original(lora=False, implementation=implementation)
        if lora:
            for layer in vision.encoder.layers:
                for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                    old = getattr(layer.self_attn, name)
                    new = namespace["Linear"](16, 16, r=4, lora_alpha=16)
                    with torch.no_grad():
                        new.weight.copy_(old.weight)
                        new.bias.copy_(old.bias)
                    new.weight.requires_grad_(False)
                    new.bias.requires_grad_(False)
                    setattr(layer.self_attn, name, new)
        return vision

    monkeypatch.setattr(test_g25, "tiny_vision", custom_vision)


@pytest.fixture
def g26(detector_class, monkeypatch):
    # Reuse only the mocked app shell; run real tiny CLIP/LoRA internally.
    for name in ("g25v2_tokens", "g26_tokens"):
        module = load_file(f"g25_test_package.{name}", DETECTORS / f"{name}.py")
        monkeypatch.setitem(sys.modules, module.__name__, module)
    module = load_file("g25_test_package.effort_detector_g26", DETECTORS / "effort_detector_g26.py")
    return module.EffortDetectorG26, sys.modules["g25_test_package.g26_tokens"]


def make_model(g26, **kwargs):
    return g26[0]({"g26_insert_layer": 1, "g26_num_tokens": 8, **kwargs})


def test_all_real_and_exist_fake_gradients(g26):
    helper = g26[1]
    z = torch.zeros(2, 8, requires_grad=True)
    losses = helper.evidence_loss(z, torch.tensor([0, 1]), 0.5)
    grad = torch.autograd.grad(losses.sum(), z)[0]
    assert (grad[0] > 0).all() and (grad[1] < 0).all()
    torch.testing.assert_close(grad[0], -grad[1])
    # One strong fake query suffices; others need not become fake.
    one_fake = torch.tensor([[10.] + [-10.] * 7])
    assert helper.evidence_loss(one_fake, torch.tensor([1]), .5).item() < .001
    assert helper.evidence_loss(one_fake, torch.tensor([0]), .5).item() > 1


@pytest.mark.parametrize("k", [1, 8, 32, 256])
def test_no_count_inflation_or_product_saturation(g26, k):
    helper = g26[1]
    z = torch.zeros(1, k, requires_grad=True)
    torch.testing.assert_close(helper.smooth_max(z, .5), torch.zeros(1))
    loss = helper.evidence_loss(z, torch.ones(1, dtype=torch.long), .5).sum()
    loss.backward()
    assert z.grad.abs().sum().item() == pytest.approx(.5)
    extreme = torch.tensor([[-1000.] * k, [1000.] * k], requires_grad=True)
    loss = helper.evidence_loss(extreme, torch.tensor([1, 0]), .5).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()
    assert (extreme.grad[0] < 0).all() and (extreme.grad[1] > 0).all()


def test_gate_only_changes_ambiguous_cls_scores(g26):
    helper = g26[1]
    cls = torch.tensor([.05, .25, .4, .5, .6, .75, .95])
    evidence = 1 - cls
    fused, gate = helper.selective_fusion(cls, evidence, .2, .5)
    assert torch.equal(fused[[0, 1, 5, 6]], cls[[0, 1, 5, 6]])
    torch.testing.assert_close(gate, torch.tensor([0., 0., .25, .5, .25, 0., 0.]))
    assert ((fused >= 0) & (fused <= 1)).all()
    assert torch.equal(helper.selective_fusion(cls, evidence, .2, 0)[0], cls)


@pytest.mark.parametrize("backend", ["loralib", "custom"])
def test_auxiliary_loss_cannot_update_any_original_parameter(g26, monkeypatch, backend):
    configure_lora_backend(monkeypatch, backend)
    model = make_model(g26).train()
    data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}
    model.get_losses(data, model(data))["loss_evidence"].backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("backbone.", "head.")):
            assert parameter.grad is None, name
    assert model.evidence_tokens.grad.abs().sum() > 0
    assert all(head.weight.grad.abs().sum() > 0 for head in model.evidence_heads.heads)
    assert not model.backbone.embeddings.class_embedding.requires_grad


@pytest.mark.parametrize("backend", ["loralib", "custom"])
def test_original_outputs_gradients_and_adam_updates_match_b0(g26, monkeypatch, backend):
    configure_lora_backend(monkeypatch, backend)
    cls = g26[0]
    base = cls.__bases__[0]({"full_train_head": True})
    expected_rng = torch.get_rng_state().clone()
    model = make_model(g26)
    # The test backbone resets RNG during construction, like the base above.
    assert torch.equal(expected_rng, torch.get_rng_state())
    base_parameters = dict(base.named_parameters())
    for name, parameter in model.named_parameters():
        if name in base_parameters:
            torch.testing.assert_close(parameter, base_parameters[name])
    optimizers = [torch.optim.Adam(m.parameters(), lr=2e-4, weight_decay=5e-4)
                  for m in (base, model)]
    for _ in range(2):
        images, labels = torch.randn(2, 3, 8, 8), torch.tensor([0, 1])
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        baseline_logits = base.head(base.backbone(images).pooler_output)
        data = {"image": images, "label": labels}
        output = model(data)
        torch.testing.assert_close(output["g26"]["global_logits"], baseline_logits)
        F.cross_entropy(baseline_logits, labels).backward()
        model.get_losses(data, output)["overall"].backward()
        for name, parameter in model.named_parameters():
            if name in base_parameters:
                reference = base_parameters[name]
                if reference.grad is None:
                    assert parameter.grad is None, name
                else:
                    torch.testing.assert_close(parameter.grad, reference.grad)
        for optimizer in optimizers:
            optimizer.step()
        for name, parameter in model.named_parameters():
            if name in base_parameters:
                torch.testing.assert_close(parameter, base_parameters[name])


@pytest.mark.parametrize("score_mode", ["gated", "cls", "evidence"])
def test_score_reload_inference_and_common_cls_multicrop(g26, score_mode):
    config = {"g26_score_mode": score_mode}
    model = make_model(g26, **config).train()
    images = torch.randn(2, 3, 3, 8, 8)
    train_output = model({"image": images.flatten(0, 1)})
    restored = make_model(g26, **config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        flat = restored({"image": images.flatten(0, 1)}, inference=True)
        multi = restored({"image": images}, inference=True)
    torch.testing.assert_close(flat["prob"], train_output["prob"])
    torch.testing.assert_close(flat["cls"].softmax(-1)[:, 1], flat["prob"])
    chosen = (flat["g26"]["cls_prob"].reshape(2, 3) - .5).abs().argmax(1)
    torch.testing.assert_close(multi["prob"], flat["prob"].reshape(2, 3)[torch.arange(2), chosen])
    assert torch.isfinite(restored.get_losses({"label": torch.tensor([0, 1])}, multi)["overall"])
    assert set(model.state_dict()) == set(restored.state_dict())


def test_token_perturbations_do_not_change_cls_and_attention_has_no_regions(g26):
    model = make_model(g26).eval()
    images = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        before = model({"image": images})["g26"]["cls_prob"]
        model.evidence_tokens.mul_(100)
        after = model({"image": images})["g26"]["cls_prob"]
        from g25_test_package.g25v2_tokens import forward_isolated_evidence
        encoded = forward_isolated_evidence(model.backbone, images, model.evidence_tokens,
                                            1, "read_only", output_attentions=True)
    torch.testing.assert_close(before, after)
    assert encoded["attention_maps"].shape == (2, 8, 4)
    assert (encoded["attention_maps"] > 0).all()


def test_six_block_suffix_with_sdpa(g26, monkeypatch):
    import copy
    import test_g25
    original = test_g25.tiny_vision

    def vision24(lora=False):
        vision = original(lora=lora, implementation="sdpa")
        vision.encoder.layers.extend([copy.deepcopy(vision.encoder.layers[-1]) for _ in range(21)])
        return vision

    monkeypatch.setattr(test_g25, "tiny_vision", vision24)
    model = make_model(g26, g26_insert_layer=18)
    images = torch.randn(2, 3, 8, 8)
    data = {"image": images, "label": torch.tensor([0, 1])}
    output = model(data)
    baseline_logits = model.head(model.backbone(images).pooler_output)
    torch.testing.assert_close(output["g26"]["global_logits"], baseline_logits)
    model.get_losses(data, output)["overall"].backward()
    assert model.evidence_tokens.grad.abs().sum() > 0
    assert model.backbone.encoder.layers[0].self_attn.v_proj.lora_B.grad.abs().sum() > 0
    assert model.backbone.encoder.layers[23].self_attn.v_proj.lora_B.grad.abs().sum() > 0


@pytest.mark.parametrize("change", [
    {"g26_num_tokens": 0}, {"g26_insert_layer": 3}, {"g26_mil_temperature": 0},
    {"g26_gate_width": 0}, {"g26_gate_width": .6}, {"g26_aux_max_weight": .6},
    {"g26_evidence_weight": -1}, {"g26_mil_temperature": float("nan")},
    {"g26_score_mode": "max"}, {"use_mixup": True}, {"use_freq_split": True},
])
def test_bad_configs_fail(g26, change):
    with pytest.raises(ValueError):
        make_model(g26, **change)

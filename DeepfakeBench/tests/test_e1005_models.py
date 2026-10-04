"""Real tiny CLIP contracts for E1005's independently trainable students."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import test_g25
from test_g26 import configure_lora_backend
from test_g30 import TinyB0


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e1005_models_test", ROOT / "training/detectors/e1005_models.py")
models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(models)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fitted_base():
    base = TinyB0("eager").eval()
    with torch.no_grad():
        for name, parameter in base.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.03)
    return base


def state_of(module):
    return {name: value.clone() for name, value in module.state_dict().items()}


def step(model, data):
    optimizer = torch.optim.SGD(model.trainable_parameters(), lr=.1)
    before = state_of(model)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(data)
    loss = F.cross_entropy(output["cls"], torch.tensor([0, 1]))
    if "mask_logits" in output:
        loss = loss + output["mask_logits"].square().mean()
    loss.backward()
    optimizer.step()
    return before, output


@pytest.mark.parametrize("backend", ["custom", "loralib"])
@pytest.mark.parametrize("scope", ["H", "L", "J", "new_residual", "residual_head"])
def test_scopes_start_at_fitted_b0_and_never_mutate_source(scope, backend, monkeypatch):
    configure_lora_backend(monkeypatch, backend)
    base = fitted_base()
    for parameter in base.parameters():
        parameter.grad = torch.ones_like(parameter)
    original_grads = {name: parameter.grad.clone() for name, parameter in base.named_parameters()}
    original_flags = {name: parameter.requires_grad for name, parameter in base.named_parameters()}
    original = state_of(base)
    data = {"image": torch.randn(2, 3, 8, 8)}
    expected = base(data)["cls"].detach()
    model = models.build_model(base, {"family": "A", "scope": scope},
                               options={"last_layers": 2, "rank": 4})
    assert torch.equal(model(data)["cls"], expected)
    before, output = step(model, data)
    assert not torch.equal(model(data)["cls"], output["cls"].detach())
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    for name, value in model.state_dict().items():
        if name not in trainable:
            assert torch.equal(value, before[name])
    if scope in ("H", "J", "residual_head"):
        assert any("head.weight" in name for name in trainable)
    if scope == "L":
        assert all(name.endswith(("lora_A", "lora_B")) for name in trainable)
    if scope == "new_residual":
        assert all(name.endswith(("down", "up")) for name in trainable)
    assert not base.training
    assert all(torch.equal(value, original[name]) for name, value in base.state_dict().items())
    assert original_flags == {name: p.requires_grad for name, p in base.named_parameters()}
    assert all(torch.equal(p.grad, original_grads[name]) for name, p in base.named_parameters())
    json.dumps(model.settings)


def test_reset_head_changes_only_same_size_linear_head():
    base = fitted_base()
    model = models.build_model(base, {"scope": "reset_head"})
    assert isinstance(model.student.head, nn.Linear)
    assert model.student.head.weight.shape == base.head.weight.shape
    assert not torch.equal(model.student.head.weight, base.head.weight)
    assert all(name.startswith("student.head.") for name, p in model.named_parameters() if p.requires_grad)


@pytest.mark.parametrize("kind", ["local_swiglu", "shuffle_local", "late_local", "local_geglu"])
def test_local_adapters_are_identity_then_affect_cls_without_appending_tokens(kind):
    base = fitted_base()
    original = state_of(base)
    data = {"image": torch.randn(2, 3, 8, 8)}
    expected = base(data)["cls"].detach()
    model = models.build_model(base, {"kind": kind},
                               options={"local_layers": [0, 1], "late_layers": [1], "adapter_width": 4})
    assert torch.equal(model(data)["cls"], expected)
    assert all(torch.count_nonzero(adapter.up.weight) == 0 for adapter in model.adapters.values())
    assert all(torch.count_nonzero(adapter.down.weight) > 0 for adapter in model.adapters.values())
    lengths = []
    handles = [layer.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
               for layer in model.student.backbone.encoder.layers]
    before, output = step(model, data)
    assert set(lengths) == {5}
    for handle in handles:
        handle.remove()
    assert not torch.equal(model(data)["cls"], output["cls"].detach())
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items()
               if name.startswith("adapters."))
    assert all(torch.equal(value, original[name]) for name, value in base.state_dict().items())


def test_patch_injection_after_last_layer_is_rejected_and_exact_budget():
    base = fitted_base()
    with pytest.raises(ValueError, match="subsequent|last"):
        models.build_model(base, {"kind": "local_swiglu"}, options={"local_layers": [2]})
    adapter = models.LocalPatchAdapter(1024, 32)
    assert sum(p.numel() for p in adapter.parameters()) == 101760
    pixel = models.PixelBranch(1024, [8, 12])
    assert sum(p.numel() for p in pixel.parameters()) == 505505


def test_student_path_keeps_gradients_through_frozen_later_attention():
    model = models.build_model(fitted_base(), {"kind": "local_swiglu"},
                               options={"local_layers": [0], "adapter_width": 4})
    data = {"image": torch.randn(2, 3, 8, 8)}
    step(model, data)  # zero up-projection opens the inner gradient path after step one
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(data)["cls"], torch.tensor([0, 1])).backward()
    assert model.adapters["0"].down.weight.grad.abs().sum() > 0
    assert model.adapters["0"].gate.weight.grad.abs().sum() > 0
    assert model.adapters["0"].spatial.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.student.parameters())


@pytest.mark.parametrize("kind", ["pretrained_delta", "ln_delta", "cold_b0", "gend"])
def test_pristine_effort_shell_is_accepted_without_raw_branch_lora(kind):
    raw = TinyB0("eager").eval()
    original = state_of(raw)
    model = models.build_model(fitted_base(), {"kind": kind}, pretrained=raw)
    if kind in ("pretrained_delta", "ln_delta", "gend"):
        branch = model.branch if hasattr(model, "branch") else model.student.backbone
        assert not any("lora_" in name for name, _ in branch.named_parameters())
    assert all(torch.equal(value, original[name]) for name, value in raw.state_dict().items())


@pytest.mark.parametrize("kind", ["pretrained_delta", "b0_delta", "ln_delta", "ln_metric"])
def test_delta_uses_independent_features_and_immutable_anchor(kind):
    base = fitted_base()
    pretrained = test_g25.tiny_vision().eval()
    original = state_of(base)
    raw_original = state_of(pretrained)
    data = {"image": torch.randn(2, 3, 8, 8)}
    model = models.build_model(base, {"kind": kind}, pretrained=pretrained)
    assert torch.equal(model(data)["cls"], base(data)["cls"])
    output = model(data)
    if kind == "ln_metric":
        torch.testing.assert_close(output["feat"].norm(dim=-1), torch.ones(2))
    if kind != "b0_delta":
        raw_features = pretrained(data["image"]).pooler_output
        if kind == "ln_metric":
            raw_features = F.normalize(raw_features, dim=-1)
        torch.testing.assert_close(output["feat"], raw_features)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert not any(name.startswith("anchor.") for name in trainable)
    assert any(name.startswith("delta_head.") for name in trainable)
    for name in trainable:
        if not name.startswith("delta_head."):
            assert kind in ("ln_delta", "ln_metric")
            assert "layernorm" in name or "layrnorm" in name or "layer_norm" in name
    step(model, data)
    assert all(torch.equal(value, original[name]) for name, value in base.state_dict().items())
    assert all(torch.equal(value, raw_original[name]) for name, value in pretrained.state_dict().items())
    assert not model.anchor.training


@pytest.mark.parametrize("kind", ["pretrained_delta", "ln_delta", "ln_metric", "cold_b0", "cold_video", "gend"])
def test_raw_branches_require_actual_pristine_supplied_model(kind):
    base = fitted_base()
    with pytest.raises(ValueError, match="pretrained"):
        models.build_model(base, {"kind": kind})
    with pytest.raises(ValueError, match="pristine|fitted|pretrained"):
        models.build_model(base, {"kind": kind}, pretrained=base)
    with pytest.raises(ValueError, match="pristine|fitted"):
        models.build_model(base, {"kind": kind}, pretrained=copy.deepcopy(base))


@pytest.mark.parametrize("kind", ["cold_b0", "cold_video", "gend"])
def test_cold_baselines_do_not_inherit_fitted_lora(kind):
    base = fitted_base()
    raw = test_g25.tiny_vision().eval()
    raw_state = state_of(raw)
    model = models.build_model(base, {"kind": kind}, pretrained=raw)
    data = {"image": torch.randn(2, 3, 8, 8)}
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    if kind == "gend":
        assert all("head" in name or "layernorm" in name or "layrnorm" in name or "layer_norm" in name
                   for name in trainable)
        assert not any("lora_" in name for name, p in model.named_parameters())
        torch.testing.assert_close(model(data)["feat"].norm(dim=-1), torch.ones(2))
    else:
        assert any("lora_B" in name for name in trainable)
        assert all("head" in name or name.endswith(("lora_A", "lora_B")) for name in trainable)
    step(model, data)
    assert all(torch.equal(value, raw_state[name]) for name, value in raw.state_dict().items())


@pytest.mark.parametrize("kind", ["pixel224", "tamper", "boundary", "shuffled_mask", "interpolated448", "native448"])
def test_pixel_injection_and_mask_gradients_preserve_source(kind):
    base = fitted_base()
    original = state_of(base)
    data = {"image": torch.randn(2, 3, 8, 8)}
    model = models.build_model(base, {"kind": kind}, options={"pixel_layers": [0, 1]})
    if kind in ("interpolated448", "native448"):
        with pytest.raises(ValueError, match="aux_image"):
            model(data)
        data["aux_image"] = torch.randn(2, 3, 16, 16)
        with pytest.raises(ValueError, match="aux_image"):
            model({**data, "aux_image": torch.randn(1, 3, 16, 16)})
    assert torch.equal(model(data)["cls"], base(data)["cls"])
    before, output = step(model, data)
    assert output["mask_logits"].ndim == 4
    assert model.pixel.mask_head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.pixel.injections.parameters())
    assert not torch.equal(model(data)["cls"], output["cls"].detach())
    assert all(torch.equal(value, original[name]) for name, value in base.state_dict().items())


def test_interpolated_control_uses_only_downsampled_detail():
    base = fitted_base()
    model = models.build_model(base, {"kind": "interpolated448"}, options={"pixel_layers": [0]})
    native = torch.randn(2, 3, 16, 16)
    low_resolution = torch.randn(2, 3, 8, 8)
    seen = []
    handle = model.pixel.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    model({"image": low_resolution, "aux_image": native})
    model({"image": low_resolution, "aux_image": native * 10})
    handle.remove()
    expected = F.interpolate(low_resolution, size=(16, 16), mode="bilinear", align_corners=False)
    assert torch.equal(seen[0], expected)
    assert torch.equal(seen[1], expected)


@pytest.mark.parametrize("initializer", ["kaiming", "normal"])
def test_cold_lora_initialization_matches_backend_and_first_update(initializer, monkeypatch):
    calls = []
    kaiming = nn.init.kaiming_uniform_
    normal = nn.init.normal_

    def observed_kaiming(tensor, *args, **kwargs):
        if tensor.shape == (4, 16):
            calls.append(("kaiming", kwargs.get("a")))
        return kaiming(tensor, *args, **kwargs)

    def observed_normal(tensor, *args, **kwargs):
        if tensor.shape == (4, 16):
            calls.append(("normal", kwargs.get("std")))
        return normal(tensor, *args, **kwargs)

    monkeypatch.setattr(nn.init, "kaiming_uniform_", observed_kaiming)
    monkeypatch.setattr(nn.init, "normal_", observed_normal)
    base = fitted_base()
    raw = test_g25.tiny_vision().eval()
    calls.clear()
    model = models.build_model(base, {"kind": "cold_b0"}, pretrained=raw,
                               options={"lora_initializer": initializer})
    expected_argument = 5 ** .5 if initializer == "kaiming" else .02
    assert calls == [(initializer, expected_argument)] * 12
    assert model.settings["lora_initializer"] == initializer
    before_a = {name: p.clone() for name, p in model.named_parameters() if name.endswith("lora_A")}
    assert all(torch.count_nonzero(p) == 0 for name, p in model.named_parameters() if name.endswith("lora_B"))
    assert all(p.scaling == 4 for p in model.student.backbone.modules() if hasattr(p, "lora_A"))
    step(model, {"image": torch.randn(2, 3, 8, 8)})
    assert all(torch.equal(p, before_a[name]) for name, p in model.named_parameters() if name.endswith("lora_A"))
    assert any(torch.count_nonzero(p) > 0 for name, p in model.named_parameters() if name.endswith("lora_B"))


def test_cold_lora_defaults_to_loralib_initializer_and_rejects_unknown():
    base = fitted_base()
    raw = test_g25.tiny_vision().eval()
    default = models.build_model(base, {"kind": "cold_video"}, pretrained=raw)
    assert default.settings["lora_initializer"] == "kaiming"
    with pytest.raises(ValueError, match="lora_initializer"):
        models.build_model(base, {"kind": "cold_b0"}, pretrained=raw,
                           options={"lora_initializer": "unsupported"})


def test_snapshots_clone_only_trainable_values_and_new_buffers():
    base = fitted_base()
    options = {"local_layers": [0], "adapter_width": 4}
    model = models.build_model(base, {"kind": "shuffle_local"}, options=options)
    data = {"image": torch.randn(2, 3, 8, 8)}
    step(model, data)
    snapshot = model.snapshot_state()
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable <= set(snapshot)
    assert all(name in trainable or name.startswith("adapters.") for name in snapshot)
    assert any(name.endswith("permutation") for name in snapshot)
    restored = models.build_model(base, {"kind": "shuffle_local"}, options=options)
    restored.load_snapshot(snapshot)
    assert torch.equal(restored(data)["cls"], model(data)["cls"])
    before = {name: value.clone() for name, value in snapshot.items()}
    with torch.no_grad():
        next(model.trainable_parameters()).add_(100)
    assert all(torch.equal(value, before[name]) for name, value in snapshot.items())
    corrupt = {**snapshot, "adapters.0.up.weight": torch.full_like(snapshot["adapters.0.up.weight"], float("nan"))}
    state = state_of(restored)
    with pytest.raises(ValueError, match="finite"):
        restored.load_snapshot(corrupt)
    assert all(torch.equal(value, state[name]) for name, value in restored.state_dict().items())

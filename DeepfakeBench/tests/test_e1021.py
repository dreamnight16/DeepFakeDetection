"""Dual-LoRA forward, class-conditioned gradients, and B0 scope contracts."""

import ast
import copy
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_g25 import tiny_vision


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    assert path.is_file(), f"Missing E1021 implementation: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def detector(monkeypatch):
    # Keep B0's real forward/loss methods; replace only pretrained loading/dimensions.
    tree = ast.parse((ROOT / "training/detectors/effort_detector.py").read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "EffortDetector")
    namespace = {"torch": torch, "F": F}
    for method in original.body:
        if isinstance(method, ast.FunctionDef) and method.name != "__init__":
            exec(compile(ast.Module(body=[method], type_ignores=[]), "B0", "exec"), namespace)

    class Base(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.backbone = tiny_vision(lora=True)
            self.head = nn.Linear(16, 2)
            self.loss_func = nn.CrossEntropyLoss()
            self.margin_loss_mode = "off"
            self.use_freq_split = False

    for name in ("forward", "features", "classifier", "_prep_input", "get_losses"):
        setattr(Base, name, namespace[name])
    package = types.ModuleType("e1021_test_package")
    package.__path__ = []
    base = types.ModuleType(f"{package.__name__}.effort_detector")
    base.EffortDetector = Base
    registry = types.ModuleType("detectors")
    registry.DETECTOR = types.SimpleNamespace(register_module=lambda **kw: lambda cls: cls)
    for item in (package, base, registry):
        monkeypatch.setitem(sys.modules, item.__name__, item)
    helper = load(f"{package.__name__}.e1021_lora", ROOT / "training/detectors/e1021_lora.py")
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    module = load(f"{package.__name__}.effort_detector_e1021",
                  ROOT / "training/detectors/effort_detector_e1021.py")
    return module.EffortDetectorE1021, Base, helper


def config(real=2, fake=6):
    return {"e1021_real_rank": real, "e1021_fake_rank": fake,
            "full_train_head": True, "use_mixup": False}


def data():
    return {"image": torch.randn(4, 3, 8, 8), "label": torch.tensor([0, 1, 1, 1])}


@pytest.mark.parametrize("ranks", [(2, 2), (4, 4), (2, 6), (6, 2)])
def test_initial_logits_head_rng_and_frozen_scope_match_b0(detector, ranks):
    cls, base_cls, _ = detector
    base = base_cls(config()).eval()
    expected_rng = torch.get_rng_state()
    model = cls(config(*ranks)).eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    batch = data()
    torch.testing.assert_close(model(batch)["cls"], base(batch)["cls"], rtol=0, atol=0)
    assert torch.equal(model.head.weight, base.head.weight)
    assert not model.backbone.embeddings.class_embedding.requires_grad
    for name, parameter in model.backbone.named_parameters():
        assert parameter.requires_grad == ("lora_" in name), name
    layer = model.backbone.encoder.layers[0].self_attn.q_proj
    assert layer.lora_A_real.shape == (ranks[0], 16)
    assert layer.lora_B_fake.shape == (16, ranks[1])


def test_both_branches_are_always_added_and_input_jacobian_is_complete(detector):
    _, _, helper = detector
    base = nn.Linear(5, 3)
    layer = helper.DualLoRALinear(base, 2, 6)
    with torch.no_grad():
        layer.lora_B_real.normal_()
        layer.lora_B_fake.normal_()
    inputs = torch.randn(2, 4, 5, requires_grad=True)
    weight = layer.weight + 4 * (layer.lora_B_real @ layer.lora_A_real
                                + layer.lora_B_fake @ layer.lora_A_fake)
    expected = F.linear(inputs, weight, layer.bias)
    actual = layer(inputs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), inputs)[0],
                               torch.autograd.grad(expected.sum(), inputs)[0])


def test_gradients_equal_separate_class_oracles_across_all_clip_layers(detector):
    cls, _, _ = detector
    model = cls(config()).train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B_" in name:
                parameter.normal_(std=.03)
    reference = copy.deepcopy(model)
    batch = data()
    reference_logits = reference(batch)["cls"]
    real = F.cross_entropy(reference_logits[:1], batch["label"][:1])
    fake = F.cross_entropy(reference_logits[1:], batch["label"][1:])
    expected = {}
    for name, parameter in reference.named_parameters():
        if parameter.requires_grad:
            objective = real if name.endswith(("lora_A_real", "lora_B_real")) else (
                fake if name.endswith(("lora_A_fake", "lora_B_fake")) else real + fake)
            expected[name] = torch.autograd.grad(objective, parameter, retain_graph=True)[0]
    losses = model.get_losses(batch, model(batch))
    torch.testing.assert_close(losses["overall"], real + fake)
    model.backward(losses)
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=2e-5, atol=2e-6)
        else:
            assert parameter.grad is None, name


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_validation_and_absent_class_adam_momentum(detector, label):
    cls, _, _ = detector
    model = cls(config())
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.01)
    batch = data()
    model.backward(model.get_losses(batch, model(batch)))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    batch["label"].fill_(label)
    absent = "fake" if label == 0 else "real"
    before = {name: p.detach().clone() for name, p in model.named_parameters()
              if name.endswith((f"lora_A_{absent}", f"lora_B_{absent}"))}
    losses = model.get_losses(batch, model(batch))
    assert torch.isfinite(losses["overall"])
    model.backward(losses)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in before:
            assert parameter.grad is None
            assert torch.equal(parameter, before[name])
    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model.get_losses(batch, model(batch))["overall"])


def test_unequal_rank_checkpoint_reload_preserves_inference_without_labels(detector):
    cls, _, _ = detector
    model = cls(config()).train()
    batch = data()
    model.backward(model.get_losses(batch, model(batch)))
    torch.optim.SGD(model.parameters(), lr=.01).step()
    restored = cls(config()).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    images = {"image": batch["image"].unsqueeze(1).repeat(1, 2, 1, 1, 1)}
    with torch.no_grad():
        assert torch.equal(model(images, inference=True)["prob"],
                           restored(images, inference=True)["prob"])

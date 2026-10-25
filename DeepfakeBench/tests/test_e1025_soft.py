"""Soft LoRA routing against class-loss gradient oracles on a tiny CLIP."""

import copy
import gc
import sys
import types
import weakref

import pytest
import torch
from torch.nn import functional as F

import test_e1021
from test_e1021 import ROOT, config, data, detector, load


def test_routing_helper_uses_beta_for_both_cross_class_gradients(detector):
    helper = detector[2]
    real, fake, head = (torch.nn.Parameter(torch.ones(())) for _ in range(3))
    real_loss = (real + 2 * fake + 3 * head).square()
    fake_loss = (4 * real + 5 * fake + 6 * head).square()
    helper.class_conditional_backward(
        {"real_loss": real_loss, "fake_loss": fake_loss, "overall": real_loss + fake_loss},
        [real], [fake], [head], beta=.25)
    torch.testing.assert_close(real.grad, torch.tensor(42.))
    torch.testing.assert_close(fake.grad, torch.tensor(156.))
    torch.testing.assert_close(head.grad, torch.tensor(216.))


@pytest.fixture
def soft_detector(detector, monkeypatch):
    cls, _, _ = detector
    base = types.ModuleType(f"{cls.__module__.rsplit('.', 1)[0]}.effort_detector_e1021")
    base.EffortDetectorE1021 = cls
    monkeypatch.setitem(sys.modules, base.__name__, base)
    module = load(f"{base.__name__.rsplit('.', 1)[0]}.effort_detector_e1025_soft",
                  ROOT / "training/detectors/effort_detector_e1025_soft.py")
    return module.EffortDetectorE1025Soft


def make_model(cls, beta=.25, ranks=(2, 6)):
    model = cls({**config(*ranks), "e1025_beta": beta}).train()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B_" in name:
                parameter.normal_(std=.03)
    return model


@pytest.mark.parametrize("beta", [0., .25, 1.])
@pytest.mark.parametrize("ranks", [(2, 2), (4, 4), (2, 6), (6, 2)])
def test_each_parameter_receives_the_specified_class_weight(soft_detector, beta, ranks):
    model = make_model(soft_detector, beta, ranks)
    reference = copy.deepcopy(model)
    batch = data()
    logits = reference(batch)["cls"]
    real = F.cross_entropy(logits[:1], batch["label"][:1])
    fake = F.cross_entropy(logits[1:], batch["label"][1:])
    expected = {}
    for suffixes, objective in ((("lora_A_real", "lora_B_real"), real + beta * fake),
                                (("lora_A_fake", "lora_B_fake"), beta * real + fake),
                                (("head.weight", "head.bias"), real + fake)):
        named = [(name, p) for name, p in reference.named_parameters() if name.endswith(suffixes)]
        gradients = torch.autograd.grad(objective, [p for _, p in named], retain_graph=True)
        expected.update((name, g) for (name, _), g in zip(named, gradients))
    losses = model.get_losses(batch, model(batch))
    torch.testing.assert_close(losses["overall"], real + fake)
    model.backward(losses)
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], rtol=2e-5, atol=2e-6)
        else:
            assert parameter.grad is None, name
            assert not parameter.requires_grad, name


def test_beta_one_matches_joint_class_mean_adam_step(soft_detector):
    model = make_model(soft_detector, beta=1.)
    reference = copy.deepcopy(model)
    batch = data()
    logits = reference(batch)["cls"]
    loss = (F.cross_entropy(logits[:1], batch["label"][:1])
            + F.cross_entropy(logits[1:], batch["label"][1:]))
    loss.backward()
    model.backward(model.get_losses(batch, model(batch)))
    for candidate in (model, reference):
        torch.optim.Adam(candidate.parameters(), lr=.0002, weight_decay=.0005).step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("beta", [0., .25, 1.])
@pytest.mark.parametrize("label", [0, 1])
def test_single_class_batches_preserve_hard_routing_and_update_soft_routes(soft_detector, beta, label):
    model = make_model(soft_detector, beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.01)
    batch = data()
    model.backward(model.get_losses(batch, model(batch)))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    batch["label"].fill_(label)
    absent = "fake" if label == 0 else "real"
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.backward(model.get_losses(batch, model(batch)))
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if beta == 0 and name.endswith((f"lora_A_{absent}", f"lora_B_{absent}")):
                assert parameter.grad is None, name
            else:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
                assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    if beta == 0:
        for name, parameter in model.named_parameters():
            if name.endswith((f"lora_A_{absent}", f"lora_B_{absent}")):
                assert torch.equal(parameter, before[name]), name


@pytest.mark.parametrize("beta", [-.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_invalid_beta_is_rejected(soft_detector, beta):
    with pytest.raises(ValueError, match="beta"):
        soft_detector({**config(), "e1025_beta": beta})


def test_default_beta_receives_quarter_of_other_class_gradient(soft_detector):
    explicit = make_model(soft_detector)
    implicit = soft_detector(config())
    implicit.load_state_dict(explicit.state_dict())
    batch = data()
    for model in (implicit, explicit):
        model.backward(model.get_losses(batch, model(batch)))
    for actual, expected in zip(implicit.parameters(), explicit.parameters()):
        if expected.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("beta", [0., .25, 1.])
def test_loss_history_does_not_retain_encoder_activations(soft_detector, monkeypatch, backend, beta):
    original_vision = test_e1021.tiny_vision
    monkeypatch.setattr(test_e1021, "tiny_vision",
                        lambda lora: original_vision(lora=lora, implementation=backend))
    model = make_model(soft_detector, beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0002)
    losses_history, saved = [], []

    class SavedTensor:
        def __init__(self, tensor):
            self.tensor = tensor.detach()

    def pack(tensor):
        item = SavedTensor(tensor)
        saved.append(weakref.ref(item))
        return item

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda item: item.tensor):
        for _ in range(8):
            batch = data()
            predictions = model(batch)
            losses = model.get_losses(batch, predictions)
            optimizer.zero_grad(set_to_none=True)
            model.backward(losses)
            optimizer.step()
            losses_history.append(losses)
            del losses, predictions
            gc.collect()
            live_bytes = sum(item().tensor.numel() * item().tensor.element_size()
                             for item in saved if item() is not None)
            assert live_bytes == 0, f"retained {live_bytes} activation bytes"

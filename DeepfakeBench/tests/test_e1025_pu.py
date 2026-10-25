"""nnPU numerical, gradient, and unchanged single-LoRA contracts."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

from test_e1021 import config, data, detector


ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    path = ROOT / "training/detectors" / filename
    assert path.is_file(), f"Missing E1025 implementation: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def loss_class():
    return load("e1025_losses_test", "e1025_losses.py").NNPULoss


@pytest.fixture
def pu_detector(detector, monkeypatch):
    package = "e1021_test_package"
    losses = load(f"{package}.e1025_losses", "e1025_losses.py")
    monkeypatch.setitem(sys.modules, losses.__name__, losses)
    module = load(f"{package}.effort_detector_e1025_pu", "effort_detector_e1025_pu.py")
    return module.EffortDetectorE1025PU, detector[1]


@pytest.mark.parametrize("prior", [.1, .3, .5])
@pytest.mark.parametrize("clipped", [False, True])
def test_loss_and_gradients_match_class_risks_for_unequal_batches(loss_class, prior, clipped):
    # Active risk gives each fake its ordinary negative-class gradient; clamped
    # risk removes the whole fake batch, rather than relabeling individual fakes.
    scores = torch.tensor([8., 9., -8., -9., -10.] if clipped
                          else [-1., .5, 1., -1., 0.],
                          dtype=torch.float64, requires_grad=True)
    logits = torch.stack((scores, torch.zeros_like(scores)), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 1])
    positive = prior * F.cross_entropy(logits[:2], labels[:2])
    negative = (F.cross_entropy(logits[2:], labels[2:])
                - prior * F.cross_entropy(logits[:2], torch.ones(2, dtype=torch.long)))
    assert bool(negative < 0) == clipped
    loss = loss_class(prior)(logits, labels)
    torch.testing.assert_close(loss, positive if clipped else positive + negative)
    expected = torch.cat((-prior * torch.sigmoid(-scores[:2]) / 2,
                          torch.zeros(3, dtype=scores.dtype))) if clipped else torch.cat((
                              torch.full((2,), -prior / 2, dtype=scores.dtype),
                              torch.sigmoid(scores[2:]) / 3))
    torch.testing.assert_close(torch.autograd.grad(loss, scores)[0], expected)


def test_class_means_do_not_depend_on_real_fake_batch_count(loss_class):
    logits = torch.tensor([[.4, -.4], [-.1, .1]], dtype=torch.float64)
    loss = loss_class(.3)
    expected = loss(logits, torch.tensor([0, 1]))
    duplicate = logits[[0, 0, 0, 1, 1]]
    torch.testing.assert_close(loss(duplicate, torch.tensor([0, 0, 0, 1, 1])), expected)


def test_extreme_logits_have_finite_loss_and_gradients(loss_class):
    logits = torch.tensor([[1000., -1000.], [-1000., 1000.],
                           [1000., -1000.], [-1000., 1000.]], requires_grad=True)
    value = loss_class()(logits, torch.tensor([0, 0, 1, 1]))
    assert torch.isfinite(value)
    assert torch.isfinite(torch.autograd.grad(value, logits)[0]).all()


@pytest.mark.parametrize("prior", [-.1, 0., 1., 1.1, float("nan"), float("inf")])
def test_invalid_prior_is_rejected(loss_class, prior):
    with pytest.raises(ValueError, match="prior"):
        loss_class(prior)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_training_is_rejected(loss_class, label):
    with pytest.raises(ValueError, match="Real|real"):
        loss_class()(torch.zeros(3, 2), torch.full((3,), label))


def test_detector_preserves_b0_initialization_rng_and_frozen_scope(pu_detector):
    cls, base_cls = pu_detector
    base = base_cls(config()).eval()
    expected_rng = torch.get_rng_state()
    model = cls({**config(), "e1025_prior": .3}).eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert model.state_dict().keys() == base.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, base.state_dict()[name]), name
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == ("lora_" in name or name.startswith("head.")), name
    assert not model.backbone.embeddings.class_embedding.requires_grad
    batch = data()
    with torch.no_grad():
        assert torch.equal(model(batch)["cls"], base(batch)["cls"])


@pytest.mark.parametrize("label", [0, 1])
def test_validation_uses_ce_for_single_class_batches(pu_detector, label):
    model = pu_detector[0]({**config(), "e1025_prior": .3}).eval()
    batch = data()
    batch["label"].fill_(label)
    with torch.no_grad():
        output = model(batch)
        losses = model.get_losses(batch, output)
    torch.testing.assert_close(losses["overall"], F.cross_entropy(output["cls"], batch["label"]))
    assert torch.isfinite(losses["real_loss"])
    assert torch.isfinite(losses["fake_loss"])
    model.train()
    with pytest.raises(ValueError, match="Real|real"):
        model.get_losses(batch, model(batch))


def test_training_uses_nnpu_and_keeps_ce_diagnostics(pu_detector, loss_class):
    model = pu_detector[0]({**config(), "e1025_prior": .1}).train()
    batch = data()
    output = model(batch)
    losses = model.get_losses(batch, output)
    torch.testing.assert_close(losses["overall"], loss_class(.1)(output["cls"], batch["label"]))
    torch.testing.assert_close(losses["real_loss"], F.cross_entropy(output["cls"][:1], batch["label"][:1]))
    torch.testing.assert_close(losses["fake_loss"], F.cross_entropy(output["cls"][1:], batch["label"][1:]))
    assert not torch.allclose(losses["overall"], F.cross_entropy(output["cls"], batch["label"]))


def test_all_single_lora_and_head_parameters_update_and_checkpoint_reloads(pu_detector):
    cls, _ = pu_detector
    model = cls({**config(), "e1025_prior": .1}).train()
    # B starts at zero: the first optimizer step activates A's gradients for
    # subsequent steps. Two steps verify both A and B without changing init.
    optimizer = torch.optim.SGD(model.parameters(), lr=.05)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    batch = data()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model.get_losses(batch, model(batch))["overall"].backward()
        optimizer.step()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert not torch.equal(parameter, before[name]), name
        else:
            assert parameter.grad is None, name
            assert torch.equal(parameter, before[name]), name
    restored = cls({**config(), "e1025_prior": .1}).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        images = {"image": batch["image"]}
        assert torch.equal(model(images, inference=True)["prob"],
                           restored(images, inference=True)["prob"])


@pytest.mark.parametrize("extra", [
    {"use_mixup": True}, {"optimizer_wrapper": "pcgrad"},
    {"optimizer": {"type": "sam"}}, {"full_train_head": False},
    {"margin_loss_mode": "add"}, {"use_freq_split": True}, {"rank_loss_weight": .2},
])
def test_unrequested_training_variants_are_rejected(pu_detector, extra):
    with pytest.raises(ValueError):
        pu_detector[0]({**config(), **extra})

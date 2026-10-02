"""E1002 temporal contracts on detached, ordered sampled-frame features."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
KINDS = ("mean", "diff", "tcn", "ssm")


def readout(kind, feature_dim=3, hidden_dim=8):
    spec = importlib.util.spec_from_file_location(
        "e1002_temporal_test", ROOT / "training/detectors/e1002_temporal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TemporalReadout(kind, feature_dim, hidden_dim)


@pytest.mark.parametrize("kind", KINDS)
def test_single_frame_has_finite_binary_logits(kind):
    model = readout(kind)
    logits = model(torch.randn(2, 1, 3), torch.ones(2, 1, dtype=torch.bool))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("kind", KINDS)
def test_padding_values_length_and_batch_companions_do_not_change_logits(kind):
    torch.manual_seed(1024)
    model = readout(kind).eval()
    clips = [torch.randn(1, length, 3) for length in (1, 3, 5)]
    expected = torch.cat([
        model(clip, torch.ones(1, clip.shape[1], dtype=torch.bool))
        for clip in clips
    ])
    for padded_length in (5, 8, 31):
        features = torch.full((3, padded_length, 3), float("nan"))
        features[:, -1, 0] = float("inf")
        features[:, -1, 1] = -1e30
        mask = torch.zeros(3, padded_length, dtype=torch.bool)
        for row, clip in enumerate(clips):
            features[row, :clip.shape[1]] = clip[0]
            mask[row, :clip.shape[1]] = True
        original = features.clone()
        torch.testing.assert_close(model(features, mask), expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(features, original, equal_nan=True)


@pytest.mark.parametrize("kind", KINDS)
def test_readout_trains_without_backpropagating_into_frozen_features(kind):
    torch.manual_seed(23)
    model = readout(kind)
    features = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    loss = nn.functional.cross_entropy(model(features, mask), torch.tensor([0, 1]))
    loss.backward()
    assert features.grad is None
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    optimizer.step()
    assert any(not torch.equal(parameter, before[name])
               for name, parameter in model.named_parameters())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_valid_features_are_rejected(kind, value):
    features = torch.zeros(1, 2, 3)
    features[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        readout(kind)(features, torch.tensor([[True, False]]))


@pytest.mark.parametrize("mask", [
    torch.tensor([[True, False, True]]),
    torch.tensor([[False, True, True]]),
    torch.tensor([[False, False, False]]),
    torch.tensor([[1, 1, 0]]),
    torch.ones(1, 2, dtype=torch.bool),
    torch.ones(1, 3, 1, dtype=torch.bool),
])
def test_invalid_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="mask|prefix|nonempty"):
        readout("mean")(torch.ones(1, 3, 3), mask)


@pytest.mark.parametrize("features,mask", [
    (torch.empty(1, 0, 3), torch.empty(1, 0, dtype=torch.bool)),
    (torch.empty(0, 3, 3), torch.empty(0, 3, dtype=torch.bool)),
    (torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.bool)),
    (torch.ones(1, 3), torch.ones(1, 3, dtype=torch.bool)),
    (torch.ones(1, 3, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool)),
])
def test_invalid_feature_shapes_and_types_are_rejected(features, mask):
    with pytest.raises(ValueError, match="features|nonempty"):
        readout("mean")(features, mask)


@pytest.mark.parametrize("kind,feature_dim,hidden_dim", [
    ("unknown", 3, 8), ("mean", 0, 8), ("ssm", 3, 0),
    ("tcn", 3.5, 8), ("diff", 3, True),
])
def test_invalid_settings_fail_before_training(kind, feature_dim, hidden_dim):
    with pytest.raises(ValueError):
        readout(kind, feature_dim, hidden_dim)


def test_mean_is_invariant_to_sampled_frame_order():
    torch.manual_seed(1024)
    model = readout("mean", 2)
    features = torch.tensor([[[0., 1.], [1., -1.], [2., 3.], [3., 0.]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    torch.testing.assert_close(model(features, mask), model(features[:, [0, 3, 1, 2]], mask))


@pytest.mark.parametrize("kind", ("diff", "tcn", "ssm"))
def test_temporal_readouts_can_distinguish_equal_mean_clips_in_different_orders(kind):
    torch.manual_seed(1024)
    model = readout(kind, 2).eval()
    features = torch.tensor([[[0., 1.], [1., -1.], [2., 3.], [3., 0.]]])
    permuted = features[:, [0, 3, 1, 2]]
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert not torch.allclose(model(features, mask), model(permuted, mask), rtol=1e-5, atol=1e-6)


def test_ssm_remains_finite_on_a_long_clip_with_large_finite_features():
    torch.manual_seed(1024)
    model = readout("ssm", 2)
    features = torch.full((1, 512, 2), 1000.)
    logits = model(features, torch.ones(1, 512, dtype=torch.bool))
    logits.square().mean().backward()
    assert torch.isfinite(logits).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())

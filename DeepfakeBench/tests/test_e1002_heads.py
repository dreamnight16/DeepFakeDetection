"""Structural readout contracts on detached, real tensor features."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e1002_heads_test", ROOT / "training/detectors/e1002_heads.py")
heads = importlib.util.module_from_spec(SPEC)
if SPEC.origin and Path(SPEC.origin).exists():
    SPEC.loader.exec_module(heads)


def test_sample_covariance_uses_centering_and_sample_denominator():
    assert hasattr(heads, "sample_covariance"), "sample covariance readout is missing"
    patches = torch.tensor([[[1., 0.], [3., 2.], [5., 4.]]])
    expected = torch.tensor([[[4., 4.], [4., 4.]]])
    assert torch.equal(heads.sample_covariance(patches), expected)
    assert torch.equal(heads.sample_covariance(patches + 100.), expected)


def test_covariance_applies_reduced_projection_before_statistics():
    assert hasattr(heads, "sample_covariance"), "projected covariance readout is missing"
    projection = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([[1., 2.]]))
    patches = torch.tensor([[[1., 0.], [3., 2.], [5., 4.]]])
    assert torch.equal(heads.sample_covariance(patches, projection), torch.tensor([[[36.]]]))


def test_symmetric_normalization_preserves_sign_and_zero_has_finite_gradients():
    assert hasattr(heads, "symmetric_covariance_normalize"), "stable covariance normalization is missing"
    covariance = torch.tensor([[[4., -4.], [-4., 4.]]])
    normalized = heads.symmetric_covariance_normalize(covariance)
    assert torch.allclose(normalized, torch.tensor([[[.5, -.5], [-.5, .5]]]))
    assert torch.equal(normalized, normalized.transpose(-1, -2))
    zero = torch.zeros(2, 3, 3, requires_grad=True)
    result = heads.symmetric_covariance_normalize(zero)
    result.sum().backward()
    assert torch.equal(result, torch.zeros_like(result))
    assert torch.isfinite(zero.grad).all()
    assert torch.equal(heads.sample_covariance(torch.ones(1, 1, 3), normalize=True), torch.zeros(1, 3, 3))


def test_half_zero_covariance_normalization_has_finite_backward():
    zero = torch.zeros(2, 3, 3, dtype=torch.float16, requires_grad=True)
    normalized = heads.symmetric_covariance_normalize(zero)
    normalized.sum().backward()
    assert torch.equal(normalized, torch.zeros_like(normalized))
    assert torch.isfinite(zero.grad).all()


@pytest.mark.parametrize("kind", ["cov", "regional"])
@pytest.mark.parametrize("value", [0., 1.])
def test_half_constant_patch_readouts_do_not_corrupt_projection_gradients(kind, value):
    model = heads.FeatureReadout(kind, 8, hidden_dim=4).half()
    logits = model(torch.full((2, 8), value, dtype=torch.float16),
                   torch.full((2, 16, 8), value, dtype=torch.float16))
    logits.sum().backward()
    assert torch.isfinite(logits).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_half_covariance_statistics_do_not_overflow_before_normalization():
    patches = torch.tensor([[[0.], [1000.], [2000.]]], dtype=torch.float16)
    covariance = heads.sample_covariance(patches)
    assert torch.isfinite(covariance).all()
    assert covariance.item() == pytest.approx(1_000_000.)


@pytest.mark.parametrize("kind", ["cls", "mean", "cov", "regional"])
def test_readouts_train_without_gradients_reaching_input(kind):
    assert hasattr(heads, "FeatureReadout"), "feature readout is missing"
    torch.manual_seed(7)
    model = heads.FeatureReadout(kind, 8, hidden_dim=4)
    cls = torch.randn(3, 8, requires_grad=True)
    patches = torch.randn(3, 16, 8, requires_grad=True)
    logits = model(cls, patches)
    assert logits.shape == (3, 2)
    F.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
    assert cls.grad is None and patches.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("kind", ["cov", "regional"])
def test_covariance_readouts_do_not_allocate_a_full_backbone_covariance_classifier(kind):
    assert hasattr(heads, "FeatureReadout"), "compact covariance readout is missing"
    model = heads.FeatureReadout(kind, 1024)
    assert sum(p.numel() for p in model.parameters()) < 300_000
    assert model(torch.randn(1, 1024), torch.randn(1, 16, 1024)).shape == (1, 2)


@pytest.mark.parametrize("kind", ["mean", "cov"])
def test_global_patch_readouts_are_permutation_invariant(kind):
    assert hasattr(heads, "FeatureReadout"), "patch readout is missing"
    model = heads.FeatureReadout(kind, 4, hidden_dim=4).eval()
    cls = torch.randn(2, 4)
    patches = torch.randn(2, 16, 4)
    assert torch.allclose(model(cls, patches), model(cls, patches[:, torch.randperm(16)]), atol=1e-6)


def test_regional_covariance_keeps_quadrant_location():
    assert hasattr(heads, "FeatureReadout"), "regional readout is missing"
    torch.manual_seed(11)
    model = heads.FeatureReadout("regional", 3, hidden_dim=4).eval()
    cls = torch.zeros(1, 3)
    patches = torch.randn(1, 16, 3)
    grid = patches.reshape(1, 4, 4, 3)
    moved = torch.cat((grid[:, 2:], grid[:, :2]), dim=1).reshape(1, 16, 3)
    assert not torch.allclose(model(cls, patches), model(cls, moved), atol=1e-6)


@pytest.mark.parametrize("tokens", [1, 3, 9, 15])
def test_regional_grid_rejects_non_even_square_patch_layout(tokens):
    assert hasattr(heads, "FeatureReadout"), "regional grid validation is missing"
    model = heads.FeatureReadout("regional", 4)
    with pytest.raises(ValueError, match="grid|square|even"):
        model(torch.zeros(2, 4), torch.zeros(2, tokens, 4))


@pytest.mark.parametrize("kind", ["cls", "mean", "cov", "regional"])
def test_readouts_reject_nonfinite_or_mismatched_features(kind):
    assert hasattr(heads, "FeatureReadout"), "readout validation is missing"
    model = heads.FeatureReadout(kind, 4)
    with pytest.raises(ValueError):
        model(torch.zeros(2, 3), torch.zeros(2, 4, 4))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 4), torch.full((2, 4, 4), float("nan")))


def test_environment_branches_and_supervision_train_without_input_gradients():
    assert hasattr(heads, "EnvironmentReadout"), "environment readout is missing"
    assert hasattr(heads, "environment_loss"), "environment objective is missing"
    torch.manual_seed(19)
    model = heads.EnvironmentReadout(8, hidden_dim=4)
    features = torch.randn(4, 8, requires_grad=True)
    output = model(features)
    assert output["logits"].shape == (4, 2)
    assert output["nuisance_logits"].shape == (4, 4)
    assert output["forgery"].shape == output["nuisance"].shape == (4, 4)
    losses = heads.environment_loss(output, torch.tensor([0, 1, 0, 1]), torch.arange(4), .5)
    assert all(torch.isfinite(value).all() for value in losses.values())
    losses["overall"].backward()
    assert features.grad is None
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def environment_output(forgery, nuisance):
    batch = forgery.shape[0]
    return {"logits": torch.zeros(batch, 2, requires_grad=True),
            "nuisance_logits": torch.zeros(batch, 4, requires_grad=True),
            "forgery": forgery, "nuisance": nuisance}


def test_environment_control_retains_both_supervised_objectives():
    assert hasattr(heads, "environment_loss"), "environment objective is missing"
    output = environment_output(torch.ones(2, 3), torch.ones(2, 3))
    losses = heads.environment_loss(output, torch.tensor([0, 1]), torch.tensor([1, 3]), 0.)
    assert losses["forgery"].item() == pytest.approx(.69314718056)
    assert losses["nuisance"].item() == pytest.approx(1.38629436112)
    assert torch.equal(losses["overall"], losses["forgery"] + losses["nuisance"])


def test_environment_orthogonality_detects_sample_overlap_and_batch_dependence():
    assert hasattr(heads, "environment_loss"), "orthogonality objective is missing"
    labels, nuisances = torch.tensor([0, 1]), torch.tensor([0, 1])
    same = torch.tensor([[1., 0.], [-1., 0.]])
    aligned = heads.environment_loss(environment_output(same, same), labels, nuisances, 1.)
    perpendicular = torch.tensor([[0., 1.], [0., -1.]])
    cross_only = heads.environment_loss(environment_output(same, perpendicular), labels, nuisances, 1.)
    assert aligned["sample_cosine"].item() == pytest.approx(1.)
    assert aligned["cross_covariance"].item() == pytest.approx(1.)
    assert cross_only["sample_cosine"].item() == pytest.approx(0.)
    assert cross_only["cross_covariance"].item() == pytest.approx(1.)
    assert torch.equal(aligned["overall"], aligned["forgery"] + aligned["nuisance"] + aligned["orthogonal"])


@pytest.mark.parametrize("batch", [1, 3])
def test_zero_environment_embeddings_produce_finite_loss_and_gradients(batch):
    assert hasattr(heads, "environment_loss"), "stable orthogonality objective is missing"
    forgery = torch.zeros(batch, 3, requires_grad=True)
    nuisance = torch.zeros(batch, 3, requires_grad=True)
    losses = heads.environment_loss(environment_output(forgery, nuisance), torch.zeros(batch, dtype=torch.long),
                                    torch.zeros(batch, dtype=torch.long), 1.)
    assert losses["orthogonal"].item() == 0.
    losses["overall"].backward()
    assert torch.isfinite(forgery.grad).all() and torch.isfinite(nuisance.grad).all()


def test_tiny_half_environment_embeddings_have_finite_orthogonality_gradients():
    forgery = torch.tensor([[1e-6, 1e-6, 0.], [1e-6, 1e-6, 0.]], dtype=torch.float16, requires_grad=True)
    nuisance = torch.tensor([[1e-6, 0., 1e-6], [1e-6, 0., 1e-6]], dtype=torch.float16, requires_grad=True)
    output = environment_output(forgery, nuisance)
    output["logits"] = torch.zeros(2, 2, dtype=torch.float16, requires_grad=True)
    output["nuisance_logits"] = torch.zeros(2, 4, dtype=torch.float16, requires_grad=True)
    losses = heads.environment_loss(output, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long), 1.)
    losses["overall"].backward()
    assert torch.isfinite(losses["overall"])
    assert torch.isfinite(forgery.grad).all() and torch.isfinite(nuisance.grad).all()


@pytest.mark.parametrize("labels,nuisances,weight", [
    (torch.tensor([0, 2]), torch.tensor([0, 1]), .1),
    (torch.tensor([0, 1]), torch.tensor([0, 4]), .1),
    (torch.tensor([0., 1.]), torch.tensor([0, 1]), .1),
    (torch.tensor([0, 1]), torch.tensor([0]), .1),
    (torch.tensor([0, 1]), torch.tensor([0, 1]), -.1),
    (torch.tensor([0, 1]), torch.tensor([0, 1]), float("nan")),
])
def test_environment_loss_rejects_invalid_supervision(labels, nuisances, weight):
    assert hasattr(heads, "environment_loss"), "environment loss validation is missing"
    with pytest.raises(ValueError):
        heads.environment_loss(environment_output(torch.ones(2, 3), torch.ones(2, 3)), labels, nuisances, weight)


def test_environment_rejects_nonfinite_outputs_including_nuisance_logits():
    assert hasattr(heads, "environment_loss"), "environment finite validation is missing"
    output = environment_output(torch.ones(2, 3), torch.ones(2, 3))
    output["nuisance_logits"] = torch.full((2, 4), float("inf"))
    with pytest.raises(ValueError, match="finite"):
        heads.environment_loss(output, torch.tensor([0, 1]), torch.tensor([0, 1]), .1)

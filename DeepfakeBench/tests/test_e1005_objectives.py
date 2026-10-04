"""Numerical and gradient contracts for E1005's training-only objectives."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "training/detectors/e1005_objectives.py"


def objectives():
    assert MODULE_PATH.is_file(), "E1005 objective implementation is missing"
    spec = importlib.util.spec_from_file_location("e1005_objectives_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode(margins=None, frames=1):
    if margins is None:
        margins = torch.tensor([-2., .3, -1., .8, 0., 1.3, 1., 1.8], dtype=torch.float64)
    margins = torch.as_tensor(margins, dtype=torch.float64)
    if margins.ndim == 1:
        margins = margins[:, None, None].expand(-1, 2, frames)
    logits = torch.stack((torch.zeros_like(margins), margins), dim=-1).clone().requires_grad_()
    batch = {
        "valid_mask": torch.ones(logits.shape[:3], dtype=torch.bool),
        "labels": torch.tensor([0, 1] * 4),
        "methods": torch.tensor([-1, 0, -1, 1, -1, 2, -1, 3]),
        "conditions": torch.tensor([[0, 3]] * 8),
        "real_indices": torch.tensor([0, 2, 4, 6]),
        "fake_indices": torch.tensor([1, 3, 5, 7]),
        "shuffle_indices": torch.tensor([1, 2, 3, 0]),
    }
    return {"logits": logits}, batch


def objective(**overrides):
    result = {"classification": "video", "rank": "none", "keep": False,
              "method_risk": False, "nuisance_risk": False}
    result.update(overrides)
    return result


def test_video_score_is_logit_of_probability_mean_not_mean_margin():
    margins = torch.tensor([math.log(.1 / .9), math.log(.8 / .2)], dtype=torch.float64)
    logits = torch.stack((torch.zeros_like(margins), margins), dim=-1)[None, None]
    result = objectives().video_log_odds(logits, torch.ones(1, 1, 2, dtype=torch.bool))
    assert result.item() == pytest.approx(math.log(.45 / .55), abs=1e-12)
    assert abs(result.item() - margins.mean().item()) > .1


def test_extreme_half_logits_remain_finite_without_probability_clipping():
    logits = torch.tensor([[[[-10000., 10000.]]]], dtype=torch.float16, requires_grad=True)
    score = objectives().video_log_odds(logits, torch.ones(1, 1, 1, dtype=torch.bool))
    assert score.item() == 20000.
    score.sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[[[-1., 1.]]]], dtype=torch.float16))


def test_padding_is_ignored_in_value_gradient_and_input_storage():
    logits = torch.tensor([[[[0., -2.], [0., 1.], [float("nan"), float("inf")]]]],
                          dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[[True, True, False]]])
    original = logits.detach().clone()
    score = objectives().video_log_odds(logits, mask)
    expected_probability = (1 / (1 + math.exp(2)) + 1 / (1 + math.exp(-1))) / 2
    assert score.item() == pytest.approx(math.log(expected_probability / (1 - expected_probability)))
    score.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 2].eq(0).all()
    torch.testing.assert_close(logits.detach(), original, equal_nan=True)


@pytest.mark.parametrize("invalid", ["shape", "integer", "nonfinite", "empty", "mask_shape", "mask_dtype"])
def test_video_score_rejects_invalid_valid_inputs(invalid):
    logits = torch.zeros(2, 1, 2, 2)
    mask = torch.ones(2, 1, 2, dtype=torch.bool)
    if invalid == "shape":
        logits = logits[..., :1]
    elif invalid == "integer":
        logits = logits.long()
    elif invalid == "nonfinite":
        logits[0, 0, 0, 0] = float("nan")
    elif invalid == "empty":
        mask[0] = False
    elif invalid == "mask_shape":
        mask = mask[:, :, :1]
    else:
        mask = mask.long()
    with pytest.raises(ValueError):
        objectives().video_log_odds(logits, mask)


def test_video_classification_balances_classes_even_with_unequal_counts():
    logits = torch.tensor([[[[0., 0.]]], [[[0., 0.]]], [[[0., 2.]]]], dtype=torch.float64)
    batch = {"labels": torch.tensor([0, 0, 1])}
    losses = objectives().compute_losses({"logits": logits}, batch, objective())
    expected = .5 * (math.log(2) + math.log1p(math.exp(-2)))
    assert losses["classification"].item() == pytest.approx(expected)
    assert losses["overall"].item() == pytest.approx(expected)
    assert losses["loss"] is losses["overall"]


def test_frame_classification_retains_sample_mean_and_valid_frame_exposure():
    logits = torch.tensor([[[[0., 0.], [float("nan"), 0.]]],
                           [[[0., 0.], [0., 0.]]],
                           [[[0., 2.], [0., 2.]]]], dtype=torch.float64, requires_grad=True)
    batch = {"labels": torch.tensor([0, 0, 1]),
             "valid_mask": torch.tensor([[[True, False]], [[True, True]], [[True, True]]])}
    losses = objectives().compute_losses({"logits": logits}, batch, objective(classification="frame"))
    expected = (3 * math.log(2) + 2 * math.log1p(math.exp(-2))) / 5
    assert losses["overall"].item() == pytest.approx(expected)
    losses["overall"].backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 1].eq(0).all()


def test_matched_rank_is_softplus_of_real_minus_fake_and_has_useful_gradients():
    output, batch = episode(torch.tensor([0., 1.] * 4))
    losses = objectives().compute_losses(output, batch, objective(rank="matched"))
    assert losses["rank"].item() == pytest.approx(math.log1p(math.exp(-1)))
    losses["rank"].backward()
    assert output["logits"].grad[batch["real_indices"], :, :, 1].gt(0).all()
    assert output["logits"].grad[batch["fake_indices"], :, :, 1].lt(0).all()


def test_keep_uses_all_cross_video_pairs_and_detaches_teacher():
    output, batch = episode(torch.zeros(8))
    teacher, _ = episode(torch.tensor([0., 1., 1., 2., 2., 3., 3., 4.]))
    losses = objectives().compute_losses(output, batch, objective(keep=True), teacher=teacher)
    # Positive teacher gaps have multiplicities 4,3,2,1 for gaps 1,2,3,4.
    assert losses["keep"].item() == pytest.approx(4.61)
    assert losses["overall"].item() == pytest.approx(math.log(2) + .461)
    assert losses["diagnostics"]["keep_eligible_pair_views"] == 20
    assert losses["diagnostics"]["keep_eligible_pairs"] == 10
    losses["overall"].backward()
    assert teacher["logits"].grad is None
    assert torch.isfinite(output["logits"].grad).all()


@pytest.mark.parametrize("teacher_gap,expected,count", [(.5, 0., 0), (100., 15.21, 32)])
def test_keep_gate_is_strict_and_teacher_margin_is_capped(teacher_gap, expected, count):
    output, batch = episode(torch.zeros(8))
    teacher, _ = episode(torch.tensor([0., teacher_gap] * 4))
    losses = objectives().compute_losses(output, batch, objective(keep=True), teacher=teacher)
    assert losses["keep"].item() == pytest.approx(expected)
    assert losses["diagnostics"]["keep_eligible_pair_views"] == count


def test_shuffle_changes_only_rank_pairing_and_preserves_keep_value_and_gradient():
    output, batch = episode()
    teacher, _ = episode(torch.tensor([-1., 2., -.3, 3., 1., 2., 1.2, 2.2]))
    module = objectives()
    matched = module.compute_losses(output, batch, objective(rank="matched", keep=True), teacher)
    shuffled = module.compute_losses(output, batch, objective(rank="shuffled", keep=True), teacher)
    assert not torch.isclose(matched["rank"], shuffled["rank"])
    torch.testing.assert_close(matched["classification"], shuffled["classification"])
    torch.testing.assert_close(matched["keep"], shuffled["keep"])
    gradient_m = torch.autograd.grad(matched["keep"], output["logits"], retain_graph=True)[0]
    gradient_s = torch.autograd.grad(shuffled["keep"], output["logits"])[0]
    torch.testing.assert_close(gradient_m, gradient_s)


def test_keep_is_invariant_to_batch_row_permutation():
    output, batch = episode()
    teacher, _ = episode(torch.tensor([-1., 2., -.3, 3., 1., 2., 1.2, 2.2]))
    original = objectives().compute_losses(output, batch, objective(keep=True), teacher)
    permutation = torch.tensor([5, 2, 7, 0, 1, 6, 3, 4])
    inverse = torch.argsort(permutation)
    shuffled_batch = {key: value[permutation] if key in {"valid_mask", "labels", "methods", "conditions"}
                      else value for key, value in batch.items()}
    shuffled_batch["real_indices"] = inverse[batch["real_indices"]]
    shuffled_batch["fake_indices"] = inverse[batch["fake_indices"]]
    permuted = objectives().compute_losses({"logits": output["logits"][permutation]}, shuffled_batch,
                                           objective(keep=True), {"logits": teacher["logits"][permutation]})
    torch.testing.assert_close(original["keep"], permuted["keep"])


@pytest.mark.parametrize("method_risk,nuisance_risk,expected_groups", [(True, False, 4), (False, True, 2), (True, True, 8)])
def test_group_risk_is_stable_smooth_max_of_current_group_means(method_risk, nuisance_risk, expected_groups):
    risks = [[.2, .3], [.4, .6], [.8, 1.2], [1.6, 2.4]]
    margins = torch.zeros(8, 2, 1, dtype=torch.float64)
    for pair, views in enumerate(risks):
        for view, risk in enumerate(views):
            margins[2 * pair + 1, view, 0] = -math.log(math.expm1(risk))
    output, batch = episode(margins)
    losses = objectives().compute_losses(output, batch, objective(rank="matched", method_risk=method_risk,
                                                                  nuisance_risk=nuisance_risk))
    expected_risks = ([.25, .5, 1., 2.] if method_risk and not nuisance_risk else
                      [.75, 1.125] if nuisance_risk and not method_risk else [.2, .3, .4, .6, .8, 1.2, 1.6, 2.4])
    expected = .5 * math.log(sum(math.exp(r / .5) for r in expected_risks) / len(expected_risks))
    assert losses["rank"].item() == pytest.approx(expected)
    assert len(losses["diagnostics"]["rank_group_counts"]) == expected_groups
    assert sum(losses["diagnostics"]["rank_group_counts"].values()) == 8
    losses["overall"].backward()
    assert torch.isfinite(output["logits"].grad).all()


def test_absent_method_groups_do_not_add_zero_risks():
    output, batch = episode(torch.tensor([0., -1000.] * 4))
    batch["methods"][batch["fake_indices"]] = 0
    losses = objectives().compute_losses(output, batch, objective(rank="matched", method_risk=True))
    assert losses["rank"].item() == pytest.approx(1000.)
    assert losses["diagnostics"]["rank_group_counts"] == {"method:0": 8}


def test_metric_l2_alignment_and_uniformity_have_hand_computed_values():
    features = torch.tensor([[2., 0.], [0., 3.], [-4., 0.], [0., -5.]], dtype=torch.float64)
    features = features[:, None, None].requires_grad_()
    output = {"logits": torch.zeros(4, 1, 1, 2, dtype=torch.float64), "features": features}
    losses = objectives().compute_losses(output, {"labels": torch.tensor([0, 0, 1, 1])},
                                         objective(classification="frame"), metric=True)
    uniformity = math.log((4 * math.exp(-4) + 2 * math.exp(-8)) / 6)
    assert losses["metric_alignment"].item() == pytest.approx(2.)
    assert losses["metric_uniformity"].item() == pytest.approx(uniformity)
    assert losses["overall"].item() == pytest.approx(math.log(2) + .1 * 2 + .5 * uniformity)
    assert losses["diagnostics"]["metric_alignment_pairs"] == 2
    assert losses["diagnostics"]["metric_uniformity_pairs"] == 6
    losses["overall"].backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0


def test_metric_without_legitimate_same_class_pair_is_zero_with_diagnostic():
    output = {"logits": torch.zeros(1, 1, 1, 2, requires_grad=True),
              "features": torch.tensor([[[[1., 0.]]]], requires_grad=True)}
    losses = objectives().compute_losses(output, {"labels": torch.tensor([0])},
                                         objective(classification="frame"), metric=True)
    assert losses["metric_alignment"].item() == 0.
    assert losses["metric_uniformity"].item() == 0.
    assert losses["diagnostics"]["metric_alignment_pairs"] == 0
    assert torch.isfinite(losses["overall"])


def test_zero_norm_metric_features_have_finite_zero_gradients():
    features = torch.zeros(4, 1, 1, 3, requires_grad=True)
    output = {"logits": torch.zeros(4, 1, 1, 2, requires_grad=True), "features": features}
    losses = objectives().compute_losses(output, {"labels": torch.tensor([0, 0, 1, 1])},
                                         objective(classification="frame"), metric=True)
    losses["overall"].backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.eq(0).all()


@pytest.mark.parametrize("scale", [1e30, 1e-30])
def test_metric_normalization_preserves_geometry_at_extreme_finite_scales(scale):
    module = objectives()
    features = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])[:, None, None]
    batch = {"labels": torch.tensor([0, 0, 1, 1])}
    reference = module.compute_losses({"logits": torch.zeros(4, 1, 1, 2), "features": features},
                                      batch, objective(classification="frame"), metric=True)
    scaled = module.compute_losses({"logits": torch.zeros(4, 1, 1, 2), "features": features * scale},
                                   batch, objective(classification="frame"), metric=True)
    torch.testing.assert_close(scaled["metric_alignment"], reference["metric_alignment"])
    torch.testing.assert_close(scaled["metric_uniformity"], reference["metric_uniformity"])


def test_metric_padding_is_ignored_and_valid_nonfinite_features_fail():
    output, batch = episode(frames=2)
    batch["valid_mask"][:, :, 1] = False
    features = torch.ones(8, 2, 2, 3, dtype=torch.float64)
    features[:, :, 1] = float("nan")
    output["features"] = features.requires_grad_()
    losses = objectives().compute_losses(output, batch, objective(), metric=True)
    losses["overall"].backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad[:, :, 1].eq(0).all()
    with torch.no_grad():
        features[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        objectives().compute_losses(output, batch, objective(), metric=True)


def test_spatial_bce_area_resizes_soft_registered_targets_and_backpropagates():
    output, batch = episode()
    mask_logits = torch.full((16, 1, 1, 1), .7, dtype=torch.float64, requires_grad=True)
    output["mask_logits"] = mask_logits
    target = torch.tensor([[[[[[0., 1.], [1., 0.]]]]]]).expand(8, 2, 1, 1, 2, 2)
    losses = objectives().compute_losses(output, batch, objective(), spatial_target=target)
    expected = math.log1p(math.exp(.7)) - .7 * .5
    assert losses["spatial"].item() == pytest.approx(expected)
    assert losses["overall"].item() == pytest.approx(losses["classification"].item() + .1 * expected)
    losses["overall"].backward()
    assert mask_logits.grad.gt(0).all()


def test_spatial_padding_is_ignored_for_flat_and_full_outputs():
    output, batch = episode(frames=2)
    batch["valid_mask"][:, :, 1] = False
    mask_logits = torch.zeros(8, 2, 2, 1, 1, 1, dtype=torch.float64)
    mask_logits[:, :, 1] = float("nan")
    output["mask_logits"] = mask_logits.requires_grad_()
    target = torch.zeros(8, 2, 2, 1, 2, 2)
    target[:, :, 1] = float("inf")
    full = objectives().compute_losses(output, batch, objective(), spatial_target=target)
    compact = objectives().compute_losses({**output, "mask_logits": mask_logits[batch["valid_mask"]]}, batch,
                                          objective(), spatial_target=target)
    assert full["spatial"].item() == pytest.approx(math.log(2))
    torch.testing.assert_close(full["spatial"], compact["spatial"])
    full["overall"].backward()
    assert torch.isfinite(mask_logits.grad).all()
    assert mask_logits.grad[:, :, 1].eq(0).all()


@pytest.mark.parametrize("failure", ["labels_float", "labels_range", "labels_shape", "methods", "indices",
                                      "shuffle_fixed", "shuffle_duplicate", "conditions", "teacher", "tau",
                                      "risk_without_rank", "spatial_range", "spatial_shape", "missing_features"])
def test_loss_inputs_fail_explicitly_instead_of_silently_changing_supervision(failure):
    output, batch = episode()
    config = objective(rank="matched", keep=True)
    teacher, _ = episode()
    kwargs = {"teacher": teacher}
    if failure == "labels_float":
        batch["labels"] = batch["labels"].float()
    elif failure == "labels_range":
        batch["labels"][0] = 2
    elif failure == "labels_shape":
        batch["labels"] = batch["labels"][:, None]
    elif failure == "methods":
        batch["methods"][0] = 0
    elif failure == "indices":
        batch["real_indices"] = torch.tensor([0, 2, 4, 7])
    elif failure == "shuffle_fixed":
        config["rank"] = "shuffled"
        batch["shuffle_indices"] = torch.arange(4)
    elif failure == "shuffle_duplicate":
        config["rank"] = "shuffled"
        batch["shuffle_indices"] = torch.tensor([1, 1, 3, 0])
    elif failure == "conditions":
        config["nuisance_risk"] = True
        batch["conditions"] = torch.zeros(8, 1, dtype=torch.long)
    elif failure == "teacher":
        kwargs["teacher"] = None
    elif failure == "tau":
        kwargs["tau"] = 0.
    elif failure == "risk_without_rank":
        config.update(rank="none", method_risk=True)
    elif failure.startswith("spatial"):
        output["mask_logits"] = torch.zeros(16, 1, 1, 1)
        kwargs["spatial_target"] = torch.full((16, 1, 2, 2), 1.1) if failure == "spatial_range" else torch.zeros(15, 1, 2, 2)
    else:
        kwargs["metric"] = True
    with pytest.raises(ValueError):
        objectives().compute_losses(output, batch, config, **kwargs)

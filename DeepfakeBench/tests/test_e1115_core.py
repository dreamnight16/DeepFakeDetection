"""E1115 mathematical identities, independent calibration, and sampling contracts."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from e1115.sampling import FixedStepBatchSampler, select_frames
from e1115.math import (adapt_model, fit_direction, moment_components,
                       real_statistics, score_model, spectral_weights)
from e1115.protocol import split_target_real


def test_fixed_steps_and_class_counts_with_cycling():
    labels = [0] * 17 + [1] * 50
    for ratio, count in [(.3, 10), (.5, 16), (.7, 22)]:
        sampler = FixedStepBatchSampler(labels, 32, ratio, 12, seed=3)
        batches = list(sampler)
        assert len(batches) == len(sampler) == 12
        assert all(sum(labels[i] == 0 for i in b) == count for b in batches)
        assert batches == list(FixedStepBatchSampler(labels, 32, ratio, 12, seed=3))
        sampler.set_epoch(1)
        assert batches != list(sampler)


def test_frame_coverage_preserves_legacy_and_changes_epochs():
    paths = [f"v/{i:03d}.png" for i in range(80)]
    assert select_frames(paths, 8, "legacy", 3, 0, "v") == paths[:8]
    a = select_frames(paths, 8, "epoch_stratified", 3, 0, "v")
    b = select_frames(paths, 8, "epoch_stratified", 3, 1, "v")
    assert len(set(a)) == 8 and a != b
    assert [int(Path(p).stem) // 10 for p in a] == list(range(8))


def test_mean_and_variation_identity_retains_cancelling_signal():
    delta = np.array([[[1., 0.], [-1., 0.]], [[2., 1.], [2., -1.]]])
    parts = moment_components(delta)
    np.testing.assert_allclose(parts["total"], parts["mean"] + parts["variation"])
    assert parts["variation"][0, 0] > 0


def test_real_constraint_changes_direction_and_soft_weights_keep_spectrum():
    B = np.diag([9., 4., 1.])
    covariance = np.diag([100., 1., 1.])
    plain = fit_direction(B, np.eye(3), covariance, 1, eta=0., weighting="equal")
    constrained = fit_direction(B, np.eye(3), covariance, 1, eta=10., weighting="equal")
    assert np.argmax(np.abs(plain["directions"][:, 0])) == 0
    assert np.argmax(np.abs(constrained["directions"][:, 0])) == 1
    w = spectral_weights(np.array([20., 5., 1.]), 2, "soft")
    assert w[0] > w[1] and w.mean() == pytest.approx(1.)


def test_target_fit_calibration_test_are_disjoint_and_nested():
    ids = np.array([f"r/{i}" for i in range(100)] + [f"f/{i}" for i in range(20)])
    labels = np.array([0] * 100 + [1] * 20)
    a = split_target_real(ids, labels, 10, 20, 1024, reserve_budget=30)
    b = split_target_real(ids, labels, 30, 20, 1024, reserve_budget=30)
    assert set(a["fit"]) <= set(b["fit"])
    assert a["calibration"] == b["calibration"]
    assert a["test"] == b["test"]
    for item in (a, b):
        assert not set(item["fit"]) & set(item["calibration"])
        assert not (set(item["fit"]) | set(item["calibration"])) & set(item["test"])
        assert all(v.startswith("r/") for v in item["fit"] + item["calibration"])


def test_adaptation_preserves_direction_and_fits_only_supplied_real_groups():
    rng = np.random.default_rng(4)
    source = [rng.normal(size=(4, 5)) for _ in range(20)]
    mu, covariance = real_statistics(source, .1)
    direction = fit_direction(np.diag([8., 5., 2., 1., .5]), np.eye(5), covariance, 2)
    model = {**direction, "mu": mu, "covariance": covariance, "real_weight": 0., "direction_weight": 1.}
    target = [rng.normal(size=(4, 5)) + 2 for _ in range(10)]
    adapted = adapt_model(model, target, "mean_cov", .5, .1)
    np.testing.assert_array_equal(adapted["directions"], model["directions"])
    np.testing.assert_array_equal(adapted["weights"], model["weights"])
    np.testing.assert_allclose(adapted["mu"], np.mean([x.mean(0) for x in target], 0))
    assert np.isfinite(score_model(np.stack([x[0] for x in target]), adapted)).all()

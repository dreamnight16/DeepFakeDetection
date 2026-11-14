"""Server-only contracts for the independent-frame, video-mean A revision."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from e1111 import audit as audit_module
from e1111 import pipeline
from e1111.features import load_audit
from e1111.metric import (FisherRequired, ProtocolError, Space, decompose, fit_space,
                         nested_d1, second_moment)
from e1111.protocol import digest, protocol


@pytest.fixture
def video_design(tmp_path, monkeypatch):
    sources = [f"{i:03d}" for i in range(36)]
    splits = {"fit": sources[:32], "sel": sources[32:34], "rad": sources[34:35],
              "theta": sources[35:], "beta": []}
    monkeypatch.setattr(audit_module, "source_splits", lambda _: splits)
    monkeypatch.setattr(audit_module, "_valid_image", lambda *args: True)
    data = {"FF-real": {"train": {"c23": {
        v: {"frames": [f"real/{v}/{t}.png" for t in (0, 100, 200, 300)]} for v in sources}}}}
    for j, method in enumerate(protocol()["methods"]):
        data[method] = {"train": {"c23": {
            f"{v}_999": {"frames": [f"{method}/{v}/{2 * t + 1}.png" for t in range(j + 2)]}
            for v in sources}}}
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"FaceForensics++": data}))
    audit_module.index_pairing(path, tmp_path / "mapping", statistic_mode="video_mean")
    report = audit_module.audit(path, tmp_path, tmp_path / "mapping/pair_map.json",
                                tmp_path / "audit", statistic_mode="video_mean")
    lookup = {}
    for groups in report["real_frames"].values():
        for source, rows in groups.items():
            for row in rows:
                lookup[row["path"]] = np.array([float(source) / 10, row["frame"] / 100])
    for bag in [*report["fit_videos"], *report["sel_videos"]]:
        j = protocol()["methods"].index(bag["method"])
        for row in bag["frames"]:
            lookup[row["fake_path"]] = np.array([float(bag["source"]) / 10 + j + 1, row["frame"] / 10])
    return path, report, lookup


def test_disjoint_saved_indices_work_without_changing_legacy_mode(video_design, tmp_path):
    path, report, _ = video_design
    assert report["P_D7_pass"] and report["N4"] == 32 and report["Ncell"] == 1
    assert report["K_pair"] is None and not report["fit_pairs"]
    assert report["retained_fake_frames"] == 32 * (2 + 3 + 4 + 5)
    assert {len(bag["frames"]) for bag in report["fit_videos"]} == {2, 3, 4, 5}
    assert load_audit(tmp_path / "audit/audit.json")["statistic_mode"] == "video_mean"
    # Identical input still fails the original exact-index protocol; its gate is retained.
    audit_module.index_pairing(path, tmp_path / "legacy-map", statistic_mode="paired_frame")
    old = audit_module.audit(path, tmp_path, tmp_path / "legacy-map/pair_map.json",
                             tmp_path / "legacy-audit", statistic_mode="paired_frame")
    assert old["N4"] == 0 and old["Ncell"] == 0 and not old["P_D7_pass"]
    with pytest.raises(ValueError, match="different statistic mode"):
        audit_module.audit(path, tmp_path, tmp_path / "mapping/pair_map.json",
                           tmp_path / "wrong-mode", statistic_mode="paired_frame")


def test_missing_video_frames_still_fail_availability_gate(video_design, tmp_path, monkeypatch):
    path, _, _ = video_design
    broken = {f"{i:03d}" for i in range(4)}
    monkeypatch.setattr(audit_module, "_valid_image", lambda root, path: path.split("/")[1] not in broken)
    result = audit_module.audit(path, tmp_path, tmp_path / "mapping/pair_map.json",
                                tmp_path / "bad-audit", statistic_mode="video_mean")
    assert result["N4"] == 28 and result["Ncell"] == .875
    assert not result["P_D7_pass"] and not result["fit_videos"]


def test_video_means_weight_sources_and_methods_not_frame_counts(video_design):
    _, report, lookup = video_design
    delta, held = pipeline.video_mean_features(report, lookup)
    assert delta.shape == (32, 4, 2)
    assert all(value.shape == (1, 2) for groups in held.values() for value in groups.values())
    first = report["fit_videos"][0]
    expected = np.mean([lookup[row["fake_path"]] for row in first["frames"]], axis=0)
    expected -= np.mean([lookup[row["path"]] for row in report["real_frames"]["fit"][first["source"]]], axis=0)
    np.testing.assert_allclose(delta[0, 0], expected)
    changed = copy.deepcopy(report)
    changed["fit_videos"][0]["frames"] *= 3
    again, _ = pipeline.video_mean_features(changed, lookup)
    np.testing.assert_allclose(again, delta)
    means, contrast = pipeline.video_real_controls(report, lookup)
    assert means.shape == contrast.shape == (32, 2)
    assert np.linalg.norm(contrast) > 0


def test_video_mean_four_part_identity_does_not_invent_frame_residual():
    delta = np.random.default_rng(1024).normal(size=(7, 4, 6))
    moment, parts, report = decompose(delta)
    assert set(parts) == {"mean", "method", "video", "inter"}
    np.testing.assert_allclose(moment, second_moment(delta))
    np.testing.assert_allclose(moment, sum(parts.values()), atol=1e-12)
    assert abs(report["rho_sum"] - 1) < 1e-12
    with pytest.raises(ProtocolError, match="frame-residual"):
        fit_space(delta, np.eye(6), np.eye(6), .1, "Fisher")


def test_video_mean_d1_refits_and_outer_holdout_does_not_choose_its_parameters():
    rng = np.random.default_rng(12)
    delta = rng.normal(size=(6, 4, 6))
    methods = protocol()["methods"]
    held = {m: {str(v): rng.normal(size=(1, 6)) for v in range(6)} for m in methods}
    settings = {**protocol("video_mean"), "ranks": [1, 2], "lambda_m": [0., .1], "bootstrap_repeats": 20}
    _, first = nested_d1(delta, np.eye(6), np.eye(6), held, settings=settings)
    altered = {**held, methods[0]: {v: value + 100 for v, value in held[methods[0]].items()}}
    _, second = nested_d1(delta, np.eye(6), np.eye(6), altered, settings=settings)
    assert first["inner_selection"][methods[0]] == second["inner_selection"][methods[0]]
    for outer_method, rows in first["inner_selection"].items():
        for candidate in rows:
            for fold in candidate["inner"]:
                assert len(fold["fit_methods"]) == 2
                assert outer_method not in fold["fit_methods"]
                assert fold["held_method"] not in fold["fit_methods"]


@pytest.mark.parametrize("numeric_failure", [False, True])
def test_s0_uses_video_d1_and_never_inherits_old_d7_or_fisher(video_design, tmp_path, monkeypatch, numeric_failure):
    _, audit, lookup = video_design
    monkeypatch.setattr(pipeline, "load_audit", lambda path: audit)
    monkeypatch.setattr(pipeline, "load_cache", lambda *args: (
        {"compression_deltas": np.eye(2)}, {}, lookup))
    monkeypatch.setattr(pipeline, "file_digest", lambda path: "fixture")
    monkeypatch.setattr(pipeline, "choose_tau", lambda *args: (.1, []))
    monkeypatch.setattr(pipeline, "real_statistics", lambda *args: (np.zeros(2), np.eye(2), np.eye(2)))
    calls = []
    def nested(delta, covariance, dc, held, family, settings):
        calls.append(family)
        assert delta.ndim == 3 and settings["statistic_mode"] == "video_mean"
        if numeric_failure:
            raise FisherRequired("recovery exhausted")
        return Space(np.eye(2), np.eye(2), np.ones(2), {}), {
            "passed": True, "final_rank": 1, "final_lambda_m": .1}
    monkeypatch.setattr(pipeline, "nested_d1", nested)
    out = tmp_path / "s0"
    result = pipeline.run_s0("unused-audit", "unused-cache", out)
    assert calls == ["B"]
    if numeric_failure:
        assert result["status"] == "S0_STOPPED" and not (out / "metric.npz").exists()
    else:
        assert result["status"] == "S0_PASS" and result["D7"]["passed"] is None
        assert "resid" not in result["D7"]["rho"]
        assert not result["full_mechanism_eligible"]
        with np.load(out / "metric.npz", allow_pickle=False) as arrays:
            identity = json.loads(str(arrays["metadata"]))
            assert identity["statistic_mode"] == "video_mean"
            assert identity["protocol_sha256"] == digest(protocol("video_mean"))

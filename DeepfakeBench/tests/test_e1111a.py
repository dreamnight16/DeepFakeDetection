"""Server-side contracts for the registered E1111A data and statistic protocol."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from e1111 import audit as audit_module
from e1111.audit import gate, source_splits
from e1111.features import load_audit
from e1111.metric import (ProtocolError, Space, _best, capture, choose_tau, conformal_threshold,
                         decompose, nested_d1, overlap, real_statistics, score, second_moment)
from e1111.pipeline import run_s0, video_means
from e1111.protocol import digest, protocol


def test_split_is_source_disjoint_and_fixed():
    ids = [f"{i:03d}" for i in range(719)]
    first = source_splits(ids)
    assert first == source_splits(list(reversed(ids)))
    assert {key: len(value) for key, value in first.items()} == protocol()["split_counts"]
    flattened = [s for group in first.values() for s in group]
    assert len(flattened) == len(set(flattened)) == 719


def test_hard_gate_and_zero_denominators(tmp_path):
    assert not gate(197, 1., 395) and not gate(198, .899, 395)
    assert gate(198, .9, 395)
    space = Space(np.eye(3), np.eye(3), np.ones(3), {})
    with pytest.raises(ProtocolError, match="zero trace"):
        capture(space, {"a": np.zeros((2, 3))}, 1)
    for name in ("D2", "D8"):
        result = overlap(space, np.zeros((3, 3)), 1, name)
        assert result["value"] is None and not result["passed"]
    space.diagnostics["positive_rank"] = 1
    with pytest.raises(ProtocolError, match="rank"):
        capture(space, {"a": np.ones((2, 3))}, 2)
    payload = {"P_D7_pass": False, "protocol_sha256": digest(protocol())}
    payload["audit_sha256"] = digest(payload)
    path = tmp_path / "failed-audit.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="P-D7"):
        load_audit(path)
    result = run_s0(path, tmp_path / "not-read.npz", tmp_path / "stopped")
    assert result["status"] == "S0_STOPPED" and "P-D7" in result["error"]
    assert not (tmp_path / "stopped/metric.npz").exists()


def test_five_part_identity_and_method_scatter():
    rng = np.random.default_rng(1024)
    delta = rng.normal(size=(7, 4, 3, 6)) + rng.normal(size=(7, 4, 1, 6))
    B, parts, report = decompose(delta)
    np.testing.assert_allclose(B, sum(parts.values()), atol=1e-12)
    assert abs(report["rho_sum"] - 1) < 1e-12
    assert report["relative_reconstruction_error"] < 1e-12
    mean_method = delta.mean(axis=(0, 2))
    np.testing.assert_allclose(parts["method"], second_moment(mean_method - mean_method.mean(axis=0)))
    with pytest.raises(ProtocolError):
        decompose(np.zeros((7, 4, 3, 6)))
    with pytest.raises(ProtocolError):
        decompose(delta[:, :, :1])


def test_tau_minimizes_source_equal_real_nll():
    rng = np.random.default_rng(8)
    fit = [rng.normal(size=(count, 5)) for count in (2, 8, 3, 6)]
    held = [rng.normal(size=(count, 5)) for count in (3, 7, 4)]
    tau, table = choose_tau(fit, held)
    assert tau == min(table, key=lambda row: (row["real_video_mean_nll"], row["tau"]))["tau"]
    mu, covariance, temporal = real_statistics(fit, tau)
    assert mu.shape == (5,) and covariance.shape == temporal.shape == (5, 5)
    # Replicating one video's entire frame list does not change its weight.
    changed = [np.repeat(fit[0], 5, axis=0), *fit[1:]]
    for a, b in zip(real_statistics(fit, tau), real_statistics(changed, tau)):
        np.testing.assert_allclose(a, b)


def test_dimension_normalization_and_deterministic_ties():
    # Raw capture increases, while capture per random dimension decreases.
    rows = [{"rank": r, "lambda_m": lam, "J": rho / (r / 12)}
            for r, rho in ((2, .4), (4, .5), (8, .6)) for lam in (0., .1)]
    assert _best(rows) == rows[0]
    assert _best([{"rank": 4, "lambda_m": .1, "J": 1.},
                  {"rank": 2, "lambda_m": .3, "J": 1.}])["rank"] == 2


@pytest.mark.parametrize("family", ["B", "Fisher"])
def test_nested_refits_exclude_both_methods_and_outer_holdout_cannot_choose_its_parameters(family):
    rng = np.random.default_rng(12)
    delta = rng.normal(size=(6, 4, 3, 6)) + rng.normal(size=(1, 4, 1, 6))
    methods = protocol()["methods"]
    held = {m: {str(v): rng.normal(size=(3, 6)) for v in range(6)} for m in methods}
    settings = {**protocol(), "ranks": [1, 2], "lambda_m": [0., .1], "bootstrap_repeats": 30}
    _, result = nested_d1(delta, np.eye(6), np.eye(6), held, family, settings)
    for outer_method, rows in result["inner_selection"].items():
        for candidate in rows:
            for fold in candidate["inner"]:
                assert len(fold["fit_methods"]) == 2
                assert outer_method not in fold["fit_methods"]
                assert fold["held_method"] not in fold["fit_methods"]
    # Held method zero is only external evaluation for its own outer fold.
    altered = {**held, methods[0]: {key: value * 10 + 100 for key, value in held[methods[0]].items()}}
    _, again = nested_d1(delta, np.eye(6), np.eye(6), altered, family, settings)
    assert result["inner_selection"][methods[0]] == again["inner_selection"][methods[0]]
    assert result["outer"][0]["rank"] == again["outer"][0]["rank"]
    assert result["outer"][0]["lambda_m"] == again["outer"][0]["lambda_m"]
    assert result["family"] == family
    if family == "Fisher":
        assert all(row["rank"] == 1 for row in result["outer"])
        assert result["final_rank"] == 1


def test_video_aggregation_calibration_and_patch_shape():
    model = {"mu": np.zeros(3), "white": np.eye(3), "transform": np.eye(3),
             "vectors": np.eye(3)[:, :1], "weights": np.ones(1)}
    patches = np.arange(2 * 256 * 3).reshape(2, 256, 3) / 100
    expected = score(patches.mean(axis=1), model, .5)
    actual = score(patches, model, .5)
    np.testing.assert_allclose(actual["D_J"], expected["D_J"])
    with pytest.raises(ProtocolError, match="Patch"):
        score(patches[:, :255], model, .5)
    with pytest.raises(ProtocolError, match="groups real and fake"):
        video_means(np.array([.2, .8]), ["same", "same"], np.array([0, 1]))
    assert conformal_threshold(np.arange(137), .1) == 124


def test_audit_uses_complete_sources_equal_frames_and_reports_discarding(tmp_path, monkeypatch):
    # Fixture is deliberately smaller than the production FF++ split. Only the
    # test's split function changes; production gate thresholds stay unchanged.
    ids = [f"{i:03d}" for i in range(36)]
    splits = {"fit": ids[:32], "sel": ids[32:34], "rad": ids[34:35], "theta": ids[35:], "beta": []}
    monkeypatch.setattr(audit_module, "source_splits", lambda _: splits)
    monkeypatch.setattr(audit_module, "_valid_image", lambda root, relative: True)
    methods = protocol()["methods"]
    data = {"FF-real": {"train": {"c23": {v: {"frames": [f"real/{v}/{t}.png" for t in (0, 10, 20)]}
                                               for v in ids}}}}
    rules = {}
    for method in methods:
        data[method] = {"train": {"c23": {
            f"{v}_999": {"frames": [f"{method}/{v}/{t}.png" for t in (
                (0, 10) if v == ids[0] and method == methods[0] else (0, 10, 20))]}
            for v in ids}}}
        rules[method] = {"verified": True, "reference_component": 0, "evidence": "fixture",
                         "same_original_index_verified": True, "crop_alignment_verified": True,
                         "crop_alignment_evidence": "fixture shared transform"}
    data_path, map_path = tmp_path / "data.json", tmp_path / "map.json"
    data_path.write_text(json.dumps({"FaceForensics++": data}))
    map_path.write_text(json.dumps({"schema_version": 1, "rules": rules,
                                   "shared_crop_receipt": "receipt.json",
                                   "shared_crop_receipt_sha256": "fixture"}))
    receipt = {"dataset_content_sha256": digest(data), "receipt_sha256": "fixture",
               "files_sha256": {path: "fixture" for branch in data.values()
                                 for entry in branch["train"]["c23"].values() for path in entry["frames"]}}
    monkeypatch.setattr(audit_module, "load_receipt", lambda *args: receipt)
    monkeypatch.setattr(audit_module, "check_pair", lambda *args: True)
    monkeypatch.setattr(audit_module, "file_digest", lambda path: "fixture")
    report = audit_module.audit(data_path, tmp_path, map_path, tmp_path / "audit")
    assert report["P_D7_pass"] and report["N4"] == 32 and report["Ncell"] == 1
    assert report["K_pair"] == 2 and len(report["fit_pairs"]) == 32 * 4 * 2
    assert report["discarded_common_pair_fraction"] > 0
    assert load_audit(tmp_path / "audit/audit.json")["audit_sha256"] == report["audit_sha256"]


def test_manual_crop_flags_cannot_replace_generated_receipt(tmp_path, monkeypatch):
    ids = [str(i) for i in range(32)]
    monkeypatch.setattr(audit_module, "source_splits",
                        lambda _: {"fit": ids, "sel": [], "theta": [], "rad": [], "beta": []})
    monkeypatch.setattr(audit_module, "_valid_image", lambda *args: True)
    data = {"FF-real": {"train": {"c23": {v: {"frames": [f"real/{v}/{t}.png" for t in (0, 1)]}
                                               for v in ids}}}}
    for method in protocol()["methods"]:
        data[method] = {"train": {"c23": {f"{v}_999": {"frames": [f"{method}/{v}/{t}.png" for t in (0, 1)]}
                                             for v in ids}}}
    mapping = {"schema_version": 1, "rules": {m: {"verified": True, "reference_component": 0,
               "evidence": "source only", "same_original_index_verified": True,
               "crop_alignment_verified": True, "crop_alignment_evidence": "manual assertion"}
               for m in protocol()["methods"]}}
    (tmp_path / "data.json").write_text(json.dumps({"FaceForensics++": data}))
    (tmp_path / "map.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="manual flags"):
        audit_module.audit(tmp_path / "data.json", tmp_path, tmp_path / "map.json", tmp_path / "result")


def test_s1_clusters_sources_and_refuses_blind_results(tmp_path, monkeypatch):
    from e1111 import pipeline
    monkeypatch.setattr(pipeline, "protocol", lambda: {**protocol(), "bootstrap_repeats": 10})
    names = ("D_R", "D_F", "D_J", "D_P", "C_random", "C_shuffle", "C_real")
    identity = {"family": "B", "gamma": 1.,
                "thresholds": {name: {"0.1": .5} for name in names}}
    folders = []
    for dataset in ("FaceForensics++", "Celeb-DF-v2"):
        folder = tmp_path / dataset
        folder.mkdir()
        report = {"identity": identity, "feature_identity": {
            "dataset_id": dataset, "role": "development", "cluster_provenance": "fixture"}}
        (folder / "metrics.json").write_text(json.dumps(report))
        values = {name: np.zeros(8) for name in names}
        values["D_J"] = np.ones(8)
        values["D_F"] = np.tile([0., 1.], 4)
        np.savez(folder / "video_scores.npz", label=np.ones(8, dtype=int),
                 cluster_id=np.repeat(["source-A", "source-B"], 4), **values)
        folders.append(folder)
    result = pipeline.attribution(folders, tmp_path / "summary")
    assert "source clusters" in result["bootstrap_unit"]
    assert set(result["datasets"]) == {"FaceForensics++", "Celeb-DF-v2"}
    path = folders[0] / "metrics.json"
    report = json.loads(path.read_text())
    report["feature_identity"]["role"] = "blind"
    path.write_text(json.dumps(report))
    with pytest.raises(ProtocolError, match="blind"):
        pipeline.attribution(folders, tmp_path / "forbidden")

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


def test_frozen_features_supports_native_backbone_without_return_dict(monkeypatch):
    import torch
    from types import SimpleNamespace
    from e1111 import features

    batches = []
    class NativeBackbone:
        def __call__(self, pixels, /):
            batches.append(len(pixels))
            tokens = torch.arange(257, dtype=torch.float32)[None, :, None].expand(len(pixels), 257, 1024).clone()
            tokens[:, 0] = 1e6  # CLS must be excluded from the pooled feature.
            tokens[:, 1:] += pixels[:, 0, 0, 0][:, None, None]
            return {"last_hidden_state": tokens}

    monkeypatch.setattr(features, "normalized_tensor", lambda rgb: torch.full(
        (3, 224, 224), float(rgb[0, 0, 0])))
    extractor = features.FrozenFeatures.__new__(features.FrozenFeatures)
    extractor.model = SimpleNamespace(backbone=NativeBackbone())
    extractor.device, extractor.batch_size = "cpu", 2
    values = extractor.encode(np.full((8, 8, 3), i, dtype=np.uint8) for i in range(3))
    assert batches == [2, 1] and values.shape == (3, 1024)
    np.testing.assert_allclose(values, np.broadcast_to(np.arange(3)[:, None] + 128.5, (3, 1024)))


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
    json.dumps(result, allow_nan=False)
    assert all(type(row["Dc_floor_dominated"]) is bool for row in result["fit_diagnostics"].values())
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
    for method in methods:
        data[method] = {"train": {"c23": {
            f"{v}_999": {"frames": [f"{method}/{v}/{t}.png" for t in (
                (0, 10) if v == ids[0] and method == methods[0] else (0, 10, 20))]}
            for v in ids}}}
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps({"FaceForensics++": data}))
    audit_module.index_pairing(data_path, tmp_path / "pairing")
    report = audit_module.audit(data_path, tmp_path, tmp_path / "pairing/pair_map.json", tmp_path / "audit")
    assert report["P_D7_pass"] and report["N4"] == 32 and report["Ncell"] == 1
    assert report["pairing_mode"] == "ffpp_official_target_original_index"
    assert report["K_pair"] == 2 and len(report["fit_pairs"]) == 32 * 4 * 2
    assert report["discarded_common_pair_fraction"] > 0
    assert load_audit(tmp_path / "audit/audit.json")["audit_sha256"] == report["audit_sha256"]


def test_index_pairing_uses_official_target_and_exact_saved_indices_with_two_frames(tmp_path):
    # All paths are metadata only; no G21 pixel scorer or images are needed.
    data = {"FF-real": {"train": {"c23": {
        v: {"frames": [f"real/{v}/{t}.png" for t in (20, 0, 10)]} for v in ("000", "999")}}}}
    for method in protocol()["methods"]:
        data[method] = {"train": {"c23": {
            "000_999": {"frames": [f"{method}/000_999/{t}.png" for t in (20, 100, 0)]},
            "998_000": {"frames": [f"{method}/998_000/{t}.png" for t in (0, 20)]},
            "999_000": {"frames": [f"{method}/999_000/{t}.png" for t in (10, 11)]},
            "bad": {"frames": []}}}}
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"FaceForensics++": data}))
    report = audit_module.index_pairing(path, tmp_path / "pairing")
    assert report["index_pairable_videos"] == 4 and report["excluded_videos"] == 12
    assert report["pixel_similarity_is_gate"] is False
    mapping = json.loads((tmp_path / "pairing/pair_map.json").read_text())
    for method in protocol()["methods"]:
        accepted = mapping["overrides"][f"{method}/000_999"]
        assert accepted["reference_component"] == 0
        assert accepted["frame_matches"] == [[0, 0], [20, 20]]
        assert mapping["overrides"][f"{method}/998_000"]["exclude_reason"] == "target_not_in_official_train_reals"
        assert mapping["overrides"][f"{method}/999_000"]["exclude_reason"] == "insufficient_saved_frame_intersection"
        assert mapping["overrides"][f"{method}/bad"]["excluded"]
    # A generated map must not silently apply to another metadata snapshot.
    mapping["dataset_sha256"] = "different metadata"
    (tmp_path / "wrong-map.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="different dataset metadata"):
        audit_module.audit(path, tmp_path, tmp_path / "wrong-map.json", tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


def test_unverified_source_is_rejected_even_with_a_crop_claim(tmp_path, monkeypatch):
    ids = [str(i) for i in range(32)]
    monkeypatch.setattr(audit_module, "source_splits",
                        lambda _: {"fit": ids, "sel": [], "theta": [], "rad": [], "beta": []})
    monkeypatch.setattr(audit_module, "_valid_image", lambda *args: True)
    data = {"FF-real": {"train": {"c23": {v: {"frames": [f"real/{v}/{t}.png" for t in (0, 1)]}
                                               for v in ids}}}}
    for method in protocol()["methods"]:
        data[method] = {"train": {"c23": {f"{v}_999": {"frames": [f"{method}/{v}/{t}.png" for t in (0, 1)]}
                                             for v in ids}}}
    mapping = {"schema_version": 1, "rules": {m: {"verified": False, "reference_component": 0,
               "evidence": "source only", "same_original_index_verified": True,
               "crop_alignment_verified": True, "crop_alignment_evidence": "manual assertion"}
               for m in protocol()["methods"]}}
    (tmp_path / "data.json").write_text(json.dumps({"FaceForensics++": data}))
    (tmp_path / "map.json").write_text(json.dumps(mapping))
    result = audit_module.audit(tmp_path / "data.json", tmp_path, tmp_path / "map.json", tmp_path / "result")
    assert not result["P_D7_pass"] and not result["fit_pairs"]
    assert {row["reason"] for row in result["mapping_exclusions"]} == {"source_mapping_unverified"}


def test_absolute_index_view_references_the_same_pngs_without_rewriting_them(tmp_path, monkeypatch):
    root = tmp_path / "images"
    methods = protocol()["methods"]
    data = {}
    for method, video in [("FF-real", "000"), *[(m, "000_001") for m in methods]]:
        paths = [root / method / video / f"{i:03d}.png" for i in (2, 0)]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"unchanged preprocessed pixels")
        data[method] = {"train": {"c23": {video: {"frames": [str(p) for p in paths]}}}}
    source = tmp_path / "original.json"
    source.write_text(json.dumps({"FaceForensics++": data}))
    original_bytes = source.read_bytes()
    monkeypatch.setattr(audit_module, "source_splits", lambda _: {"fit": ["000"]})
    report = audit_module.inventory(source, tmp_path / "inventory", data_root=root)
    normalized = json.loads(Path(report["dataset_json"]).read_text())["FaceForensics++"]
    for method, videos in data.items():
        video = next(iter(videos["train"]["c23"]))
        old = videos["train"]["c23"][video]
        new = normalized[method]["train"]["c23"][video]
        assert audit_module.frame_map(old, root) == audit_module.frame_map(new)
        assert new["frames"] == [f"{method}/{video}/002.png", f"{method}/{video}/000.png"]
        for path in new["frames"]:
            assert (root / path).read_bytes() == b"unchanged preprocessed pixels"
    assert source.read_bytes() == original_bytes


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

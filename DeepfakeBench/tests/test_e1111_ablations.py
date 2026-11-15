"""Server-only ablation isolation, snapshot and diagnostic contracts."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "experiments"))

from e1111 import ablations
from e1111 import ablation_diagnostics as diagnostics
from e1111.ablation_models import fit_variants, gap_diagnostic, score_models, spherical_kmeans
from e1111.benchmark import readout
from e1111.metric import ProtocolError, fit_space, inverse_sqrt, real_statistics, score
from e1111.protocol import protocol
from g21.io import digest, file_digest
from run_g25 import TEST_DS


@pytest.fixture(scope="module")
def design():
    rng = np.random.default_rng(1024)
    real = [rng.normal(size=(4 + i % 3, 10)) + .2 for i in range(12)]
    rad = [rng.normal(size=(4, 10)) + .2 for _ in range(12)]
    delta = rng.normal(size=(12, 4, 10)) + rng.normal(size=(1, 4, 10))
    mu, covariance, _ = real_statistics(real, .3)
    dc = np.eye(10) * .1
    space = fit_space(delta, covariance, dc, .1)
    main = {"mu": mu, "white": inverse_sqrt(covariance), "transform": space.transform,
            "vectors": space.vectors[:, :4], "weights": space.weights[:4]}
    for name in ("C_random", "C_shuffle", "C_real"):
        main[name + "_V"] = np.eye(10)[:, :4]
        main[name + "_w"] = np.ones(4)
    identity = {"family": "B", "rank": 4, "lambda_m": .1, "gamma": .5}
    settings = protocol("video_mean")
    models, specs, gap = fit_variants(main, identity, {"tau_R": .3}, settings, delta, real, rad, dc)
    return main, identity, settings, models, specs, gap, delta, real, rad, dc


def test_full_registered_matrix_and_single_factor_refits(design):
    main, identity, _, models, specs, gap, *_ = design
    required = {"M_style", "R_style", "C4_no_method_reg", "rank_2", "rank_4", "rank_8",
                "lambda_0", "lambda_0.1", "lambda_0.3", "lambda_1", "tau_0.1", "tau_0.3", "tau_0.5", "gamma_inf"}
    assert required <= set(models)
    assert len(models) == 36 and all(spec["status"] == "FITTED" for spec in specs.values())
    c4 = specs["C4_no_method_reg"]
    assert (c4["lambda_m"], c4["lambda_t"], c4["rank"], c4["tau_R"], c4["gamma"]) == (0., 0., 4, .3, .5)
    for name in ("rank_2", "rank_8"):
        assert specs[name]["lambda_m"] == identity["lambda_m"] and specs[name]["gamma"] == .5
    assert gap["selected_K"] in range(1, 9) and gap["fit_split"].startswith("D_rad")
    z = np.random.default_rng(8).normal(size=(21, 10))
    actual = score_models(z, models, batch_size=5)
    expected = score(z, main, .5)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual["C4_no_method_reg"], actual["lambda_0"])
    np.testing.assert_allclose(actual["gamma_0.0"], actual["D_R"])
    np.testing.assert_allclose(actual["gamma_inf"], actual["D_F"])
    np.testing.assert_allclose(actual["M_style"], 5 * actual["D_R"] + models["M_style"]["nll_constant"])


def test_source_equal_prototypes_and_deterministic_gap(design):
    _, _, _, _, _, gap, _, real, rad, _ = design
    centers, _ = spherical_kmeans(np.array([[1., 0.], [1., 0.], [0., 1.]]), 1, np.array([.25, .25, .5]))
    np.testing.assert_allclose(centers[0], [2 ** -.5, 2 ** -.5])
    with pytest.raises(ProtocolError, match="distinct"):
        spherical_kmeans(np.ones((4, 3)), 2)
    assert gap == gap_diagnostic(rad)
    assert len(real) == 12


def test_complete_suite_scores_all_arms_and_oracle_never_fits_fake_rows(tmp_path, monkeypatch, design):
    main, identity, settings, models, specs, gap, delta, real, rad, dc = design
    run = tmp_path / "run"
    run.mkdir()
    cache = run / "features.npz"
    cache.write_bytes(b"source-only feature fixture")
    config_path = run / "resolved_base_config.json"
    config = {"clip_pretrained_path": "clip", "dataset_json_folder": str(tmp_path)}
    config_path.write_text(json.dumps(config))
    manifest = {"base": {"config_path": str(tmp_path / "native/train_config.json"), "checkpoint": "B0.pth"}}
    audit = {"statistic_mode": "video_mean"}
    monkeypatch.setattr(ablations, "context", lambda *args, **kwargs: (
        run, manifest, config_path, config, audit, cache, {"compression_deltas": np.eye(10) * np.sqrt(.1 * 10)}, {}, main, identity, {"tau_R": .3}))
    monkeypatch.setattr(ablations, "arcface_path", lambda model: None)
    monkeypatch.setattr(ablations, "_real_groups", lambda audit, lookup, split: real if split == "fit" else rad)
    monkeypatch.setattr(ablations, "video_mean_features", lambda *args: (delta, {}))
    monkeypatch.setattr(ablations, "fit_variants", lambda *args: (dict(models), copy.deepcopy(specs), gap))
    monkeypatch.setattr(ablations, "complementarity", lambda *args: {"status": "COMPUTED"})
    monkeypatch.setattr(ablations, "identity_audit", lambda *args: {"status": "UNAVAILABLE_ARCFACE_WEIGHTS", "passed": None})
    rng = np.random.default_rng(12)
    theta = rng.normal(size=(44, 10))
    monkeypatch.setattr(ablations, "theta_features", lambda *args: (theta, np.repeat(np.arange(22).astype(str), 2)))
    z = rng.normal(size=(4, 10)) + 30
    z[2:] += 10000
    rows = {"labels": np.array([0, 0, 1, 1]), "path": np.array([f"p/{i}.png" for i in range(4)]),
            "video_id": np.array(["real", "real", "fake", "fake"]), "cls_prob": np.array([.1, .2, .8, .9]),
            "global_log_odds": np.array([-2., -1., 1., 2.])}
    reference_score = score(z, main, .5)["D_J"]
    native = {"status": "OK", "B0": {d: readout(rows, rows["cls_prob"], .5) for d in TEST_DS},
              "datasets": {d: readout(rows, reference_score, 1.) for d in TEST_DS},
              "feature_cache": {d: {} for d in TEST_DS}}
    monkeypatch.setattr(ablations, "evaluate", lambda *args, **kwargs: native)
    monkeypatch.setattr(ablations, "_read_test_rows", lambda *args: (rows, z))
    target_groups = []
    original_statistics = real_statistics
    def real_only(groups, tau):
        assert tau == .3 and len(groups) == 1
        np.testing.assert_array_equal(groups[0], z[:2])
        target_groups.append(groups)
        return original_statistics(groups, tau)
    monkeypatch.setattr(ablations, "real_statistics", real_only)
    result = ablations.run_suite(run, tmp_path / "suite", device="cpu")
    assert result["status"] == "OK" and len(result["arms"]) == 38 and len(target_groups) == len(TEST_DS)
    assert all(set(arm["datasets"]) == set(TEST_DS) for arm in result["arms"].values())
    assert result["arms"]["C5_oracle_Sigma_R"]["scope"] == "transductive_oracle_diagnostic_only"
    with np.load(tmp_path / "suite/models/oracle_WDF.npz", allow_pickle=False) as model:
        np.testing.assert_array_equal(model["mu"], main["mu"])
    assert (tmp_path / "suite/ablation_metrics.csv").is_file()
    assert (tmp_path / "suite/ablation_summary.csv").is_file()


def test_cached_test_matrix_rejects_partial_or_changed_snapshot(tmp_path):
    (tmp_path / "features").mkdir()
    (tmp_path / "exports").mkdir()
    metadata = tmp_path / "json"
    metadata.mkdir()
    native = tmp_path / "eval.json"
    native.write_text("{}")
    identity = {"feature_identity": {"checkpoint_sha256": "B0"}}
    report = {"status": "OK", "frozen_metric_identity": identity,
              "baseline_eval_config_sha256": file_digest(native), "test_source_sha256": ablations.test_source_identity(),
              "feature_cache": {}, "dataset_sha256": {}}
    rows = {"labels": np.array([0, 1]), "path": np.array(["a.png", "b.png"]), "video_id": np.array(["a", "b"]),
            "cls_prob": np.array([.1, .9]), "global_log_odds": np.array([-1., 1.])}
    for dataset in TEST_DS:
        (metadata / f"{dataset}.json").write_text("{}")
        np.save(tmp_path / "features" / f"{dataset}.npy", np.zeros((2, 1024), dtype=np.float32), allow_pickle=False)
        np.savez_compressed(tmp_path / "exports" / f"{dataset}.npz", **rows)
        report["dataset_sha256"][dataset] = file_digest(metadata / f"{dataset}.json")
        report["feature_cache"][dataset] = {"file": f"features/{dataset}.npy", "rows": 2,
            "sha256": file_digest(tmp_path / "features" / f"{dataset}.npy"),
            "export_sha256": file_digest(tmp_path / "exports" / f"{dataset}.npz"),
            "rows_sha256": digest({key: rows[key].tolist() for key in ("labels", "path", "video_id")})}
    path = tmp_path / "all_results.json"
    path.write_text(json.dumps(report))
    assert ablations.load_test_cache(tmp_path, identity, native, metadata)["status"] == "OK"
    _, z = ablations._read_test_rows(tmp_path, "WDF", report["feature_cache"]["WDF"])
    assert z.shape == (2, 1024)
    report["status"] = "RUNNING"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="incomplete"):
        ablations.load_test_cache(tmp_path, identity, native, metadata)
    report["status"] = "OK"
    path.write_text(json.dumps(report))
    np.save(tmp_path / "features/WDF.npy", np.ones((2, 1024)), allow_pickle=False)
    with pytest.raises(ValueError, match="changed"):
        ablations.load_test_cache(tmp_path, identity, native, metadata)


def test_d5_development_only_and_missing_identity_weights_are_explicit(tmp_path, monkeypatch):
    identity = {"thresholds": {name: {"0.1": .5} for name in ("D_R", "D_F", "D_J")}}
    for tag in ("dev_ff_scores", "dev_cdf_scores"):
        folder = tmp_path / tag
        folder.mkdir()
        (folder / "metrics.json").write_text(json.dumps({"identity": identity, "feature_identity": {"role": "development"}}))
        np.savez_compressed(folder / "video_scores.npz", label=np.array([0, 0, 1, 1]),
            cluster_id=np.array(["r0", "r1", "f0", "f1"]), D_R=np.array([.1, .6, .8, .2]),
            D_F=np.array([.1, .3, .3, .8]), D_J=np.array([.1, .3, .8, .8]))
    result = diagnostics.complementarity(tmp_path, identity, repeats=30)
    assert result["macro_error_jaccard"] == 0 and result["gain_vs_best_endpoint"] == .5
    (tmp_path / "dev_ff_scores/metrics.json").write_text(json.dumps({"identity": identity, "feature_identity": {"role": "blind"}}))
    with pytest.raises(ProtocolError, match="development"):
        diagnostics.complementarity(tmp_path, identity, repeats=10)
    monkeypatch.setattr(diagnostics, "arcface_path", lambda path: None)
    result = diagnostics.identity_audit(tmp_path, {})
    assert result["status"] == "UNAVAILABLE_ARCFACE_WEIGHTS" and result["passed"] is None


@pytest.mark.parametrize("matching", [False, True])
def test_missing_fit_cache_is_rebuilt_atomically_without_rewriting_completed_run(tmp_path, monkeypatch, matching):
    run = tmp_path / "completed"
    run.mkdir()
    checkpoint, config = tmp_path / "B0.pth", run / "resolved_base_config.json"
    checkpoint.write_bytes(b"original frozen B0")
    config.write_text("{}")
    expected = {"checkpoint_sha256": file_digest(checkpoint), "base_config_sha256": file_digest(config), "d8": True}
    manifest = {"base": {"checkpoint": str(checkpoint)}, "status": "DEVELOPMENT_COMPLETE"}
    manifest_path = run / "all_results.json"
    manifest_path.write_text(json.dumps(manifest))
    original_manifest = manifest_path.read_bytes()
    calls = []
    def extract(audit, cfg, ckpt, clip, output, **kwargs):
        assert audit == run / "audit/audit.json" and cfg == config and ckpt == str(checkpoint)
        assert clip == "original_clip" and kwargs == {"device": "cuda:0", "batch_size": 8, "d8": True}
        assert output != run / "features/features.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"reconstructed feature snapshot")
        calls.append(output)
    monkeypatch.setattr(ablations, "extract", extract)
    monkeypatch.setattr(ablations, "load_cache", lambda path, audit: ({}, expected if matching else {"changed": True}, {}))
    arguments = (run, tmp_path / "deleted-run/features/features.npz", manifest, config,
                 {"clip_pretrained_path": "original_clip"}, {}, {"feature_identity": expected}, "cuda:0")
    if matching:
        path = ablations.recover_fit_cache(*arguments)
        assert path == run / "features/features.npz" and path.read_bytes() == b"reconstructed feature snapshot"
        assert not calls[0].exists()
        assert json.loads((path.parent / "recovery.json").read_text())["feature_identity"] == expected
    else:
        with pytest.raises(ValueError, match="identity differs"):
            ablations.recover_fit_cache(*arguments)
        assert not (run / "features/features.npz").exists()
    assert len(calls) == 1 and manifest_path.read_bytes() == original_manifest


@pytest.mark.parametrize("changed", ["checkpoint_sha256", "base_config_sha256"])
def test_fit_cache_recovery_refuses_changed_original_inputs(tmp_path, monkeypatch, changed):
    checkpoint, config = tmp_path / "B0.pth", tmp_path / "base.json"
    checkpoint.write_bytes(b"B0")
    config.write_text("{}")
    expected = {"checkpoint_sha256": file_digest(checkpoint), "base_config_sha256": file_digest(config), "d8": False}
    expected[changed] = "different original input"
    def forbidden(*args, **kwargs):
        raise AssertionError("Changed frozen inputs must be rejected before model execution")
    monkeypatch.setattr(ablations, "extract", forbidden)
    with pytest.raises(ValueError, match="original B0 or resolved configuration changed"):
        ablations.recover_fit_cache(tmp_path, tmp_path / "missing.npz", {"base": {"checkpoint": str(checkpoint)}},
                                   config, {}, {}, {"feature_identity": expected}, "cuda:0")


def test_context_uses_already_recovered_local_cache_instead_of_missing_link(tmp_path, monkeypatch):
    (tmp_path / "features").mkdir()
    (tmp_path / "s0").mkdir()
    local = tmp_path / "features/features.npz"
    local.write_bytes(b"previously recovered cache")
    (tmp_path / "resolved_base_config.json").write_text("{}")
    (tmp_path / "s0/s0.json").write_text('{"tau_R": 0.3}')
    (tmp_path / "all_results.json").write_text(json.dumps({"stages": {"features": {"cache_path": str(tmp_path / "deleted/cache.npz")}}}))
    identity = {"audit_sha256": "same-audit", "feature_identity": {"frozen": True}, "thresholds": {}, "status": "S0_PASS"}
    monkeypatch.setattr(ablations, "load_audit", lambda path: {"audit_sha256": "same-audit"})
    monkeypatch.setattr(ablations, "load_metric", lambda path: ({}, identity))
    paths = []
    def load(path, audit):
        paths.append(path)
        return {}, identity["feature_identity"], {}
    monkeypatch.setattr(ablations, "load_cache", load)
    def forbidden(*args):
        raise AssertionError("An already recovered local cache must not be rebuilt")
    monkeypatch.setattr(ablations, "recover_fit_cache", forbidden)
    assert ablations.context(tmp_path)[5] == local and paths == [local]


def test_d6_uses_real_identity_embeddings_and_source_frozen_matching_cutoff(tmp_path, monkeypatch):
    from types import SimpleNamespace
    model = tmp_path / "w600k_r50.onnx"
    model.write_bytes(b"identity-model fixture; no network is executed")
    methods = protocol()["methods"]
    audit = {"statistic_mode": "video_mean", "data_root": str(tmp_path), "protocol": {"methods": methods},
             "real_frames": {"sel": {source: [{"frame": t, "path": f"real/{source}/{t}.png"} for t in (0, 10)]
                                      for source in ("000", "001")}}, "sel_videos": []}
    for method in methods:
        for source in ("000", "001"):
            audit["sel_videos"].append({"method": method, "source": source,
                "frames": [{"fake_path": f"fake/{method}/{source}/{t}.png"} for t in (0, 10)]})
    def encode(root, paths):
        z = np.zeros((len(paths), 512))
        for i, path in enumerate(paths):
            z[i, 1 if path.startswith("fake/") else 0] = 1.
        return z
    def extractor(path, mean, std):
        assert path == model and mean == std == 127.5
        return SimpleNamespace(encode=encode)
    monkeypatch.setattr(diagnostics, "ArcFace", extractor)
    (tmp_path / "dev_ff").mkdir()
    (tmp_path / "dev_ff_scores").mkdir()
    videos = [f"FaceForensics++/FF-real/{source}" for source in ("000", "001")]
    videos += [f"FaceForensics++/{method}/{source}_999" for method in methods for source in ("000", "001")]
    labels = [0, 0] + [1] * 8
    frame_paths = []
    for video in videos:
        prefix = "real" if "/FF-real/" in video else "fake"
        frame_paths.extend([f"{prefix}/{video}/{t}.png" for t in (0, 10)])
    np.savez_compressed(tmp_path / "dev_ff/features.npz", path=np.array(frame_paths),
                        video_id=np.repeat(videos, 2), label=np.repeat(labels, 2))
    np.savez_compressed(tmp_path / "dev_ff_scores/video_scores.npz", video_id=np.array(videos), label=np.array(labels),
                        D_R=np.array(labels), D_F=np.array(labels), D_J=np.array(labels))
    result = diagnostics.identity_audit(tmp_path, audit, model)
    assert result["identity_only_auc"] == 1. and result["alarm"]
    assert result["frames_per_identity_mean"] == 1 and result["matching_cosine_cutoff"] == 1.
    assert result["matched_development"]["fake_videos"] == 0
    assert result["matched_development"]["video_auc"] is None
    json.dumps(result, allow_nan=False)

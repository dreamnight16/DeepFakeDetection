"""E0924 protocol, source grouping, router and feature tests on synthetic inputs."""

import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    return tuple(importlib.import_module(name) for name in
                 ("e0924_protocol", "e0924_decision", "run_e0924"))


def metadata():
    root = {}
    for label in ("FF-real", "FF-DF"):
        root[label] = {}
        for split, nums in (("train", (1, 2)), ("val", tuple(range(10, 30))), ("test", (31, 32))):
            videos = {}
            for i in nums:
                video = f"{i:03}" if label == "FF-real" else f"{i:03}_{(i + 1 if i % 2 == 0 else i - 1):03}"
                if split != "val" and label != "FF-real":
                    video = f"{nums[0]:03}_{nums[1]:03}"
                videos[video] = {"label": label, "frames": [f"FF/{label}/frames/{video}/{j}.png" for j in range(3)]}
            root[label][split] = {"c23": videos}
    return {"FaceForensics++": root}


def test_plan_deduplicates_and_locks_source(modules):
    p = modules[0].plan()
    assert len(p["training"]) == 9 and len(p["decision_arms"]) == 5
    assert p["decision_source"] == "G27_Full"
    assert p["aliases"]["G27_G26"] == "G26_K4L20"


def test_partition_group_disjoint_reproducible_and_no_fallback(modules):
    partition = modules[0].calibration_partition
    p = partition(metadata())
    assert p == partition(metadata())
    fit = {v for r in p["records"]["fit"] for v in r["ids"]}
    holdout = {v for r in p["records"]["holdout"] for v in r["ids"]}
    assert not fit & holdout
    broken = metadata()
    broken["FaceForensics++"]["FF-real"].pop("val")
    with pytest.raises(ValueError, match="Missing"):
        partition(broken)
    broken = metadata()
    broken["FaceForensics++"]["FF-real"]["val"]["c23"]["001"] = {"label": "FF-real", "frames": ["new.png"]}
    with pytest.raises(ValueError, match="overlap"):
        partition(broken)


def test_spectral_color_features_are_finite_and_frequency_sensitive(modules):
    features = modules[1].image_features
    constant = torch.full((2, 3, 16, 16), .5)
    low = features(constant)
    checker = constant.clone()
    checker[:, :, ::2, ::2] = 1
    checker[:, :, 1::2, 1::2] = 1
    checker[:, :, ::2, 1::2] = 0
    checker[:, :, 1::2, ::2] = 0
    high = features(checker)
    assert torch.isfinite(low[0]).all() and torch.isfinite(low[1]).all()
    assert high[0][:, 2].mean() > low[0][:, 2].mean()
    assert torch.equal(constant, torch.full_like(constant, .5))


def samples():
    p = np.array([.2, .8, .2, .8, .3, .7, .3, .7])
    e = 1 - p
    return {"labels": np.array([0, 1, 1, 0, 0, 1, 1, 0]), "cls_prob": p,
            "evidence_prob": e, "gated_prob": .75*p+.25*e,
            "query_log_odds": np.stack((np.log(e/(1-e)), np.log(e/(1-e))), 1),
            "global_log_odds": np.log(p/(1-p)), "freq": np.arange(48).reshape(8, 6)/48,
            "color": np.arange(96).reshape(8, 12)/96,
            "video_id": np.array([f"v{i}" for i in range(8)])}


def test_router_training_reload_and_labels_never_used_for_scoring(modules, tmp_path):
    m = modules[1]
    data = samples()
    artifact = m.fit_router(data, "both", 16, seed=1024, steps=5, lr=.01)
    first = m.score_router(artifact, data, threshold=.7)
    changed = {**data, "labels": 1-data["labels"]}
    np.testing.assert_array_equal(first, m.score_router(artifact, changed, threshold=.7))
    path = tmp_path / "router.json"
    path.write_text(json.dumps(artifact))
    np.testing.assert_allclose(first, m.score_router(json.loads(path.read_text()), data, threshold=.7))
    all_cls = m.score_router(artifact, data, threshold=1.)
    np.testing.assert_array_equal(all_cls, data["cls_prob"])
    assert np.isfinite(first).all()
    assert "oracle_auc" not in m.metrics(data, first)
    assert m.metrics(data, first)["oracle_threshold_accuracy"] == 1.


def test_video_auc_groups_full_video_paths_and_label_conflicts_fail(modules):
    m = modules[1]
    data = samples()
    data["video_id"] = np.array(["real/001", "fake/001", "v2", "v3", "v4", "v5", "v6", "v7"])
    assert m.metrics(data, data["cls_prob"])["num_videos"] == 8
    data["video_id"][1] = data["video_id"][0]
    with pytest.raises(ValueError, match="labels"):
        m.metrics(data, data["cls_prob"])


def test_no_disagreement_fails_explicitly(modules):
    m = modules[1]
    data = samples()
    data["evidence_prob"] = data["cls_prob"].copy()
    with pytest.raises(ValueError, match="disagreement"):
        m.fit_router(data, "model", 0, seed=1024, steps=2, lr=.01)


def test_dry_run_has_no_ml_dependency():
    r = subprocess.run([sys.executable, "-S", str(ROOT / "experiments/run_e0924.py"), "--dry_run"],
                       capture_output=True, text=True, check=True)
    assert len(json.loads(r.stdout)["training"]) == 9


def test_auc_ties_and_direction(modules):
    a = modules[1].auc
    assert a([0, 1], [.5, .5]) == .5
    assert a([0, 1], [.1, .9]) == 1
    assert a([0, 1], [.9, .1]) == 0
    with pytest.raises(ValueError):
        a([1, 1], [.1, .9])


def test_signature_roundtrip(modules):
    runner = modules[2]
    args = runner.build_parser().parse_args([])
    sig = runner.signature(args, {"metadata_sha256": "fixture"})
    assert json.loads(json.dumps(sig)) == sig


def test_retry_pointer_is_reused_and_other_jobs_continue(modules, tmp_path, monkeypatch):
    from types import SimpleNamespace
    runner = modules[2]
    args = runner.build_parser().parse_args(["--skip_diagnostics"])
    ckpt = tmp_path / "retry.pth"
    ckpt.write_bytes(b"fixture")
    runner.write_json(tmp_path / "training_results.json", {"G27_Full": {
        "status": "OK", "ckpt": str(ckpt), "artifact_dir": "attempt_folder"}})
    seen = []

    def run_one(args, arm, folder, adapter):
        seen.append(arm)
        if arm == "B0":
            raise RuntimeError("expected fixture failure")
        return {"status": "OK", "ckpt": str(ckpt)}

    for module_name in ("run_g26", "run_g27"):
        monkeypatch.setattr(importlib.import_module(module_name), "run_one", run_one)
    results = runner.run_training(args, tmp_path, SimpleNamespace())
    assert len(seen) == 8 and "Full" not in seen
    assert results["G26_B0"]["status"] == "FAILED"
    assert results["G27_Full"]["artifact_dir"] == "attempt_folder"


def test_export_strict_reader_fails_on_missing_frame_and_preserves_ids(modules):
    from types import SimpleNamespace
    export = importlib.import_module("e0924_export")
    seen = []

    def load(path):
        seen.append(path)
        if path == "missing":
            raise FileNotFoundError(path)
        return torch.zeros(3, 16, 16)

    base = SimpleNamespace(image_list=["real/001/1.png", "missing"], label_list=[0, 1],
                           load_rgb=load, to_tensor=lambda x: x, normalize=lambda x: x)
    loader = iter(export.strict_loader(base, 1))
    assert next(loader)["video_id"] == ["real/001"]
    with pytest.raises(FileNotFoundError):
        next(loader)
    assert seen == ["real/001/1.png", "missing"]


def test_export_features_use_same_resized_tensor(modules):
    export = importlib.import_module("e0924_export")

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def forward(self, data, inference):
            n = len(data["image"])
            p = data["image"].mean((1, 2, 3))
            return {"g27": {"cls_prob": p, "evidence_prob": 1-p, "gated_prob": p,
                            "evidence_log_odds": torch.zeros(n, 4), "global_logits": torch.zeros(n, 2)}}

    images = torch.rand(2, 3, 16, 16)
    loader = [{"image": images, "label": torch.tensor([0, 1]),
               "path": ["a/1.png", "b/1.png"], "video_id": ["a", "b"]}]
    result = export.export_loader(Fake(), loader, {"mean": [0, 0, 0], "std": [1, 1, 1]})
    f, c = modules[1].image_features(images)
    np.testing.assert_allclose(result["freq"], f.numpy())
    np.testing.assert_allclose(result["color"], c.numpy())


def test_all_decision_arms_end_to_end_with_cached_synthetic_exports(modules, tmp_path):
    runner = modules[2]
    export = importlib.import_module("e0924_export")
    from run_g25 import TEST_DS

    args = runner.build_parser().parse_args(["--router_steps", "2"])
    ckpt = tmp_path / "fixture.pth"
    ckpt.write_bytes(b"not a real checkpoint; exporter is already cached")
    folder = tmp_path / "training/G27_Full"
    runner.write_json(folder / "eval_config.json", {})
    runner.write_json(tmp_path / "training_results.json", {"G27_Full": {
        "status": "OK", "ckpt": str(ckpt), "artifact_dir": str(folder)}})
    cache = tmp_path / "decision_exports"
    cache.mkdir()
    for name in ("fit", "holdout", *[f"test_{ds}" for ds in TEST_DS]):
        data = samples()
        data["path"] = np.array([f"{name}/{i}/0.png" for i in range(8)])
        data["video_id"] = np.array([f"{name}/{i}" for i in range(8)])
        np.savez_compressed(cache / f"{name}.npz", **data)
    runner.write_json(cache / "manifest.json", {"checkpoint_sha256": export.file_sha256(ckpt),
        "files": {p.name: export.file_sha256(p) for p in cache.glob("*.npz")}})
    result = runner.run_decisions(args, tmp_path, {"metadata_sha256": "fixture"}, None, None)
    assert result["status"] == "OK"
    report = runner.read_json(tmp_path / "decisions/results.json")
    assert len(report) == 8
    assert all(set(r["testall"]) == set(TEST_DS) for r in report.values())
    assert all(np.isfinite(r["AUC_cross"]) for r in report.values())
    # Checksum failures stop reuse before any fit or inference.
    (cache / "fit.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        runner.run_decisions(args, tmp_path, {"metadata_sha256": "fixture"}, None, None)


def test_runtime_adapter_and_actual_diagnostic_call_chain(modules, monkeypatch, tmp_path):
    import types
    runner = modules[2]
    utilities = types.ModuleType("experiment_utils")
    for name in ("train_model", "evaluate_model", "run_testall"):
        setattr(utilities, name, lambda *args, **kwargs: None)
    utilities.build_config = lambda **kwargs: kwargs
    calls = []

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def forward(self, data, inference):
            p = torch.tensor([.2, .8])
            return {"g26": {"cls_prob": p, "evidence_prob": p, "gated_prob": p,
                            "gate_weight": torch.zeros(2), "evidence_log_odds": torch.zeros(2, 4)}}

    def load(*args):
        calls.append("load")
        return Fake()

    def loader(*args):
        calls.append("loader")
        return [{"image": torch.zeros(2, 3, 16, 16), "label": torch.tensor([0, 1])}]

    utilities.load_model, utilities.get_data_loader = load, loader
    monkeypatch.setitem(sys.modules, "experiment_utils", utilities)
    args = runner.build_parser().parse_args(["--dataset_json_folder", str(tmp_path), "--rgb_root", "images"])
    adapter, _, _ = runner.runtime(args)
    config = adapter.build_config()
    config["manualSeed"] = 1024
    assert config["e0924_protocol"] is True and config["rgb_root_override"] == "images"
    diagnostics = importlib.import_module("g26_diagnostics")
    result = diagnostics.collect_routing_diagnostics(config, "ckpt", ["fake_dataset"], tmp_path / "routing", adapter)
    assert calls == ["load", "loader"] and result["fake_dataset"]["num_frames"] == 2


def test_real_subprocess_builders_propagate_e0924_roots(modules, tmp_path):
    """Execute actual utility functions with only the child launch intercepted."""
    import ast
    import os
    import tempfile
    import types
    import yaml

    tree = ast.parse((ROOT / "experiments/experiment_utils.py").read_text(encoding="utf-8"))
    function_names = ("save_temp_yaml", "train_model", "run_testall")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in function_names]
    detector = tmp_path / "detector.yaml"
    detector.write_text("model_name: effort\ndataset_json_folder: WRONG\n")
    test_yaml = tmp_path / "test.yaml"
    test_yaml.write_text("dataset_json_folder: ALSO_WRONG\n")
    log_dir = tmp_path / "logs"
    ckpt = log_dir / "effort_fixture/test/avg/ckpt_best.pth"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"fixture")
    seen = []

    def launch(cmd, **kwargs):
        seen.append(cmd)
        config = yaml.safe_load(Path(cmd[cmd.index("--detector_path")+1]).read_text())
        assert config["dataset_json_folder"] == "INTENDED"
        assert cmd[cmd.index("--dataset_json_folder")+1] == "INTENDED"
        assert cmd[cmd.index("--data_root")+1] == "IMAGES"
        return types.SimpleNamespace(returncode=0)

    namespace = {"os": os, "sys": sys, "Path": Path, "tempfile": tempfile, "yaml": yaml,
                 "subprocess": types.SimpleNamespace(run=launch, STDOUT=-2),
                 "_training_dir": str(ROOT / "training"), "_deepfake_dir": str(ROOT),
                 "DETECTOR_YAML": str(detector), "TEST_YAML": str(test_yaml)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "actual_subprocess_builders", "exec"), namespace)
    config = {"e0924_protocol": True, "dataset_json_folder": "INTENDED", "rgb_root_override": "IMAGES",
              "log_dir": str(log_dir)}
    assert namespace["train_model"](config, "FF", "CDF") == str(ckpt)
    namespace["run_testall"](str(ckpt), ["FF"], str(tmp_path / "testall.log"), config)
    assert len(seen) == 2
    # Execute train.main's actual config-merge prefix, stopping before logger/GPU code.
    train_tree = ast.parse((ROOT / "training/train.py").read_text(encoding="utf-8"))
    main = next(n for n in train_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "logger_path" for t in n.targets))
    import io

    def fake_open(path, mode):
        return io.StringIO("lmdb: false\ndry_run: false\ndataset_json_folder: WRONG\n")

    context = {"open": fake_open, "yaml": yaml, "args": types.SimpleNamespace(
        detector_path="x", local_rank=0, train_dataset=None, test_dataset=None, save_ckpt=True,
        save_feat=True, mixup_gamma=None, dataset_json_folder="INTENDED", data_root="IMAGES")}
    exec(compile(ast.Module(body=main.body[:end], type_ignores=[]), "actual_train_merge", "exec"), context)
    assert context["config"]["dataset_json_folder"] == "INTENDED"
    assert context["config"]["rgb_root_override"] == "IMAGES"


def test_interrupted_resume_preserves_later_retry_record(modules, tmp_path, monkeypatch):
    runner = modules[2]
    from types import SimpleNamespace
    ckpt = tmp_path / "later.pth"
    ckpt.write_bytes(b"fixture")
    runner.write_json(tmp_path / "training_results.json", {"G27_Full": {
        "status": "OK", "ckpt": str(ckpt), "artifact_dir": "attempts/success"}})
    calls = []

    def run_one(args, arm, folder, adapter):
        calls.append(arm)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return {"status": "TRAIN_FAILED"}

    monkeypatch.setattr(importlib.import_module("run_g26"), "run_one", run_one)
    with pytest.raises(KeyboardInterrupt):
        runner.run_training(runner.build_parser().parse_args([]), tmp_path, SimpleNamespace())
    saved = runner.read_json(tmp_path / "training_results.json")
    assert saved["G27_Full"]["artifact_dir"] == "attempts/success"


def test_main_resume_and_metadata_change_guard(modules, tmp_path, monkeypatch):
    runner = modules[2]
    from run_g25 import TEST_DS
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    for dataset in TEST_DS:
        (metadata_root / f"{dataset}.json").write_text(json.dumps(metadata() if dataset == "FaceForensics++" else {}))
    monkeypatch.setattr(runner, "runtime", lambda args: (None, None, {"fixture": True}))
    monkeypatch.setattr(runner, "run_training", lambda *args: {"fixture": {"status": "OK"}})
    arguments = ["--stages", "train", "--dataset_json_folder", str(metadata_root),
                 "--output_dir", str(tmp_path / "runs")]
    assert runner.main(arguments) == 0
    run, = (tmp_path / "runs").iterdir()
    assert runner.main([*arguments, "--resume", str(run)]) == 0
    (metadata_root / "DFDC.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="signature mismatch"):
        runner.main([*arguments, "--resume", str(run)])


def test_preflight_missing_metadata_fails_without_runtime(modules, tmp_path, monkeypatch):
    runner = modules[2]
    (tmp_path / "FaceForensics++.json").write_text(json.dumps(metadata()))
    called = []
    monkeypatch.setattr(runner, "runtime", lambda args: called.append(True))
    assert runner.main(["--preflight", "--dataset_json_folder", str(tmp_path)]) == 2
    assert called == []

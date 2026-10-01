"""E1001 identity, cached decisions, and paired AUC contracts."""

import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_e0924 import samples


ROOT = Path(__file__).resolve().parents[1]


def evaluation_metadata(dataset):
    from test_e0924 import metadata

    if dataset == "FaceForensics++":
        return metadata()
    labels = {"WDF": ("WDF_Real", "WDF_Fake"), "FFIW": ("FFIW_Real", "FFIW_Fake"),
              "Celeb-DF-v2": ("CelebDFv2_real", "CelebDFv2_fake"),
              "DeepFakeDetection": ("DFD_real", "DFD_fake"), "DFDC": ("DFDC_Real", "DFDC_Fake"),
              "DFDCP": ("DFDCP_Real", "DFDCP_FakeA"), "DeeperForensics-1.0": ("DF_real", "DF_fake")}
    root = {}
    for label in labels[dataset]:
        videos = {"001": {"label": label, "frames": [f"{dataset}/{label}/001/0.png"]}}
        root[label] = {"test": {"c23": videos} if dataset == "DeepFakeDetection" else videos}
    return {dataset: root}


@pytest.fixture
def preflight_source(tmp_path):
    import yaml
    from run_g25 import TEST_DS

    for dataset in TEST_DS:
        (tmp_path / f"{dataset}.json").write_text(json.dumps(evaluation_metadata(dataset)))
    training = yaml.safe_load((ROOT / "training/config/train_config.yaml").read_text())
    training.update(dataset_json_folder=str(tmp_path), compression="c23", train_batchSize=32,
                    model_name="effort", use_loralib=True, full_train_head=True,
                    mean=[.4, .5, .6], std=[.1, .2, .3], resolution=224)
    return {"config": training}


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    return importlib.import_module("run_e1001")


def test_preflight_adds_wdf_and_ffiw_labels_without_changing_b0_protocol(runner, preflight_source):
    original = json.loads(json.dumps(preflight_source["config"]))
    assert "WDF_Real" not in original["label_dict"]
    config, _, _ = runner.preflight(runner.build_parser().parse_args([]), preflight_source)
    for real, fake in (("WDF_Real", "WDF_Fake"), ("FFIW_Real", "FFIW_Fake")):
        assert config["label_dict"][real] == 0 and config["label_dict"][fake] == 1
    for key in ("model_name", "use_loralib", "full_train_head", "mean", "std", "resolution"):
        assert config[key] == original[key]
    assert preflight_source["config"] == original


def test_preflight_preserves_existing_fake_subclasses(runner, preflight_source):
    preflight_source["config"]["label_dict"]["FF-DF"] = 7
    config, _, _ = runner.preflight(runner.build_parser().parse_args([]), preflight_source)
    assert config["label_dict"]["FF-DF"] == 7


def test_preflight_rejects_a_label_with_conflicting_binary_meaning(runner, preflight_source):
    preflight_source["config"]["label_dict"]["WDF_Real"] = 1
    with pytest.raises(ValueError, match="WDF_Real"):
        runner.preflight(runner.build_parser().parse_args([]), preflight_source)


def test_preflight_rejects_unknown_metadata_labels_before_model_loading(runner, preflight_source):
    folder = Path(preflight_source["config"]["dataset_json_folder"])
    data = evaluation_metadata("WDF")
    data["WDF"]["WDF_Real"]["test"]["001"]["label"] = "WDF_unknown"
    (folder / "WDF.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="WDF_unknown"):
        runner.preflight(runner.build_parser().parse_args([]), preflight_source)


def test_dry_run_is_new_g30_and_reuses_existing_b0_without_ml():
    result = subprocess.run([sys.executable, "-S", str(ROOT / "experiments/run_e1001.py"),
                             "--dry_run"], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["experiment"] == "E1001" and plan["family"] == "G30"
    assert plan["base_training"] is False
    assert plan["default_arms"] == ["K1L20", "K4L20", "Full"]
    assert plan["decisions"] == ["linear", "mlp", "token"]


def test_source_requires_a_complete_b0_and_never_uses_g27(runner, tmp_path):
    folder = tmp_path / "training/G26_B0"
    folder.mkdir(parents=True)
    ckpt = tmp_path / "b0.pth"
    ckpt.write_bytes(b"real checkpoint identity fixture")
    (folder / "train_config.json").write_text(json.dumps({"model_name": "effort"}))
    (folder / "result.json").write_text(json.dumps({"status": "OK", "ckpt": str(ckpt)}))
    args = runner.build_parser().parse_args(["--base_run", str(tmp_path)])
    source = runner.resolve_source(args)
    assert source["checkpoint"] == str(ckpt.resolve())
    assert len(source["base_sha256"]) == 64
    (folder / "train_config.json").write_text(json.dumps({"model_name": "effort_g27"}))
    with pytest.raises(ValueError, match="effort"):
        runner.resolve_source(args)
    (folder / "result.json").write_text(json.dumps({"status": "TRAIN_FAILED", "ckpt": str(ckpt)}))
    with pytest.raises(ValueError, match="B0"):
        runner.resolve_source(args)


def test_source_follows_e0924_successful_retry_pointer(runner, tmp_path):
    folder = tmp_path / "attempts/G26_B0_retry"
    folder.mkdir(parents=True)
    checkpoint = folder / "b0.pth"
    checkpoint.write_bytes(b"B0 retry identity")
    (folder / "train_config.json").write_text(json.dumps({"model_name": "effort"}))
    (tmp_path / "training_results.json").write_text(json.dumps({"G26_B0": {
        "status": "OK", "ckpt": str(checkpoint), "artifact_dir": str(folder)}}))
    args = runner.build_parser().parse_args(["--base_run", str(tmp_path)])
    assert runner.resolve_source(args)["checkpoint"] == str(checkpoint)


@pytest.mark.parametrize("batch,ratio", [(2, .9), (32, .99)])
def test_sampling_rejects_rounded_zero_fake_batches(runner, batch, ratio):
    with pytest.raises(ValueError, match="both"):
        runner.validate_sampling(batch, ratio)


def test_token_router_fits_cached_data_reloads_and_never_uses_test_labels(runner, tmp_path):
    data = samples()
    snapshot = {k: v.copy() for k, v in data.items()}
    artifact = runner.fit_token_router(data, "a" * 64, "b" * 64, steps=3)
    identity = {"base_sha256": "a" * 64, "evidence_sha256": "b" * 64}
    first = runner.score_token_router(artifact, data, **identity)
    from torch import float64, full_like, load, save
    path = tmp_path / "decision.pth"
    save(artifact, path)
    loaded = load(path, weights_only=True)
    np.testing.assert_array_equal(first, runner.score_token_router(loaded, {**data, "labels": 1-data["labels"]}, **identity))
    np.testing.assert_array_equal(data["cls_prob"], runner.score_token_router(loaded, data, threshold=1., **identity))
    assert loaded["base_sha256"] == "a" * 64
    assert loaded["evidence_sha256"] == "b" * 64
    for key in data:
        np.testing.assert_array_equal(data[key], snapshot[key])
    with pytest.raises(ValueError, match="identity"):
        runner.score_token_router(loaded, data, base_sha256="c" * 64, evidence_sha256="b" * 64)
    corrupt = {**loaded, "state_dict": {**loaded["state_dict"], "query": loaded["state_dict"]["query"] * float("nan")}}
    with pytest.raises(ValueError, match="finite"):
        runner.score_token_router(corrupt, data, **identity)
    with pytest.raises(ValueError, match="normalization"):
        runner.score_token_router({**loaded, "std": [0.] * loaded["input_dim"]}, data, **identity)
    overflow = {**loaded, "state_dict": {**loaded["state_dict"],
                "head.bias": full_like(loaded["state_dict"]["head.bias"], 1e300, dtype=float64)}}
    with pytest.raises(ValueError, match="parameters"):
        runner.score_token_router(overflow, data, **identity)


def test_changed_base_outputs_fail_before_comparing_auc(runner):
    before = samples()
    after = {k: v.copy() for k, v in before.items()}
    after["path"] = before["video_id"].copy()
    before["path"] = before["video_id"].copy()
    runner.verify_paired_base(before, after)
    after["cls_prob"][0] += 1e-8
    with pytest.raises(RuntimeError, match="B0"):
        runner.verify_paired_base(before, after)


def test_primary_video_auc_uses_full_paths_not_colliding_basenames(runner):
    data = samples()
    data["video_id"] = np.array(["real/001", "fake/001", "v2", "v3", "v4", "v5", "v6", "v7"])
    report = runner.readout_metrics(data, data["cls_prob"])
    assert report["video_auc"] == report["video_auc_fullpath"]
    assert report["legacy_group_collisions"] == 1
    assert "video_auc_legacy" in report


def test_live_run_rejects_missing_source_instead_of_training_b0(runner):
    with pytest.raises(SystemExit):
        runner.main([])


@pytest.mark.parametrize("arm", ["K1L20", "Full"])
@pytest.mark.parametrize("use_lmdb", [False, True])
def test_complete_training_export_and_reload_preserves_standalone_b0(runner, tmp_path, monkeypatch, arm, use_lmdb):
    """Only the external app loader is replaced; CLIP, training and artifacts run."""
    import random
    from types import SimpleNamespace
    import torch
    from test_g30 import TinyB0
    from e0924_export import strict_loader
    from run_g25 import TEST_DS
    lmdb = pytest.importorskip("lmdb") if use_lmdb else None

    data_dir = tmp_path / "metadata"
    data_dir.mkdir()
    for dataset in TEST_DS:
        (data_dir / f"{dataset}.json").write_text(json.dumps(evaluation_metadata(dataset)))
    config = {"model_name": "effort", "dataset_json_folder": str(data_dir), "compression": "c23",
              "test_batchSize": 8, "train_batchSize": 4, "label_dict": {"FF-real": 0, "FF-DF": 1},
              "std": [1., 1., 1.]}
    config_path, checkpoint = tmp_path / "config.json", tmp_path / "b0.pth"
    config_path.write_text(json.dumps(config))
    original = TinyB0("eager").eval()
    with torch.no_grad():
        original.head.weight.zero_()
        original.head.bias.copy_(torch.tensor([0., 2.]))
    torch.save(original.state_dict(), checkpoint)

    class Frames:
        def __init__(self, cfg, dataset):
            self.config = cfg
            rows = [(f"{dataset}/{label}/video{i}/0.png", label) for label in (0, 1) for i in range(4)]
            random.shuffle(rows)  # mirrors the real metadata loader
            self.image_list, self.label_list = map(list, zip(*rows))
            if lmdb is not None:
                env_dir = tmp_path / "lmdb" / dataset
                env_dir.mkdir(parents=True, exist_ok=True)
                self.env = lmdb.open(str(env_dir), map_size=1024 * 1024)

        def load_rgb(self, path):
            if lmdb is not None:
                self.env.stat()  # reading after an early close is a real error
            seed = sum(path.encode())
            return torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(seed))

        def to_tensor(self, tensor):
            return tensor

        def normalize(self, tensor):
            return tensor

    def load_model(cfg, path):
        random.random()  # other model initialization must not change export ordering
        base = TinyB0("eager")
        base.load_state_dict(torch.load(path, weights_only=True), strict=True)
        base.config = cfg
        return base.eval()

    utilities = SimpleNamespace(load_model=load_model, DEVICE=torch.device("cpu"),
                                get_data_loader=lambda cfg, ds: SimpleNamespace(dataset=Frames(cfg, ds)))
    monkeypatch.setitem(sys.modules, "experiment_utils", utilities)

    def training_loader(cfg, ratio):
        source = Frames(cfg, "FaceForensics++")
        frames = strict_loader(source, 4)
        frames.source_dataset = source
        return frames

    monkeypatch.setattr(runner, "make_train_loader", training_loader)
    monkeypatch.setitem(runner.ARMS, arm, (4 if arm == "Full" else 1, 1, arm == "Full"))
    output_dir = tmp_path / "E1001"
    cli = ["--base_checkpoint", str(checkpoint), "--base_config", str(config_path),
           "--output_dir", str(output_dir), "--arms", arm, "--n_epochs", "1",
           "--hidden_dim", "16", "--depth", "1", "--router_steps", "2"]
    if arm != "Full":
        cli.append("--skip_decisions")
    code = runner.main(cli)
    assert code == 0
    root, = output_dir.iterdir()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["base_state_unchanged"] and manifest["base_file_unchanged"]
    result = json.loads((root / f"G30_{arm}/result.json").read_text())
    assert result["status"] == "OK" and result["base_scores_bitwise_identical"]
    for dataset in TEST_DS:
        with np.load(root / "B0" / f"{dataset}.npz") as before:
            with np.load(root / f"G30_{arm}/exports" / f"{dataset}.npz") as after:
                runner.verify_paired_base(before, after)
    if arm == "Full":
        decisions = json.loads((root / "all_results.json").read_text())["decisions"]
        assert set(decisions) == {"linear", "mlp", "token"}
        assert all(result["status"] in ("OK", "SKIPPED_NO_DISAGREEMENT") for result in decisions.values())


def test_decision_failure_is_recorded_without_losing_other_independent_heads(runner, tmp_path, monkeypatch):
    import e0924_decision
    fit = e0924_decision.fit_router

    def fail_linear(data, kind, hidden, **kwargs):
        if hidden == 0:
            raise RuntimeError("linear fit fixture failure")
        return fit(data, kind, hidden, **kwargs)

    monkeypatch.setattr(e0924_decision, "fit_router", fail_linear)
    args = runner.build_parser().parse_args(["--router_steps", "2"])
    data = samples()
    data["path"] = data["video_id"].copy()
    result = runner.run_decisions(args, tmp_path, {"fit": data, "holdout": data}, "a" * 64, "b" * 64)
    assert result["linear"]["status"] == "FAILED"
    assert "fixture failure" in result["linear"]["error"]
    assert result["mlp"]["status"] == result["token"]["status"] == "OK"
    assert json.loads((tmp_path / "decisions/linear_result.json").read_text())["status"] == "FAILED"


def test_cached_linear_router_also_rejects_wrong_identity_and_nonfinite_state(runner):
    from e0924_decision import fit_router
    data = samples()
    artifact = fit_router(data, "model", 0, steps=2)
    artifact.update(base_sha256="a" * 64, evidence_sha256="b" * 64)
    with pytest.raises(ValueError, match="identity"):
        runner.score_cached_router(artifact, data, base_sha256="c" * 64, evidence_sha256="b" * 64)
    artifact["layers"][0]["weight"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner.score_cached_router(artifact, data, base_sha256="a" * 64, evidence_sha256="b" * 64)

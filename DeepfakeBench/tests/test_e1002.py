"""E1002 protocol, strict video identities, and real tiny-CLIP integration."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    return importlib.import_module("run_e1002")


def test_dependency_free_dry_run_covers_four_routes_with_val_selection():
    result = subprocess.run([sys.executable, "-S", str(ROOT / "experiments/run_e1002.py"),
                             "--dry_run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["experiment"] == "E1002"
    assert len(plan["arms"]) == 13
    assert set(a["route"] for a in plan["arms"].values()) == {"readout", "matrix", "environment", "temporal"}
    assert plan["selection"] == "FaceForensics++ official val video_auc"


def test_clip_grouping_uses_numeric_indices_and_rejects_conflicting_labels(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    data = importlib.import_module("e1002_data")
    groups = data.group_video_rows([("x/v/10.png", 0), ("x/v/2.png", 0), ("x/v/1.png", 0)], 8)
    assert groups[0]["paths"] == ["x/v/1.png", "x/v/2.png", "x/v/10.png"]
    assert groups[0]["indices"] == [1, 2, 10]
    with pytest.raises(ValueError, match="label"):
        data.group_video_rows([("x/v/1.png", 0), ("x/v/2.png", 1)], 8)
    with pytest.raises(ValueError, match="index"):
        data.group_video_rows([("x/v/no_index.png", 0)], 8)


def test_clip_sampler_rejects_rounded_zero_fake_count(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    data = importlib.import_module("e1002_data")
    source = SimpleNamespace(image_list=["x/real/1.png", "x/fake/1.png"], label_list=[0, 1])
    with pytest.raises(ValueError, match="counts"):
        data.clip_loader(source, 2, 8, train_ratio=.9)


def test_temporal_sampling_budget_is_rejected_before_starting_any_arm(runner):
    with pytest.raises(SystemExit) as error:
        runner.main(["--dry_run", "--clip_batch_size", "2", "--sampler_real_ratio", ".9"])
    assert error.value.code == 2


def test_metric_threshold_is_calibrated_on_real_validation_videos(runner):
    rows = {"labels": np.array([0, 0, 0, 1, 1]),
            "video_id": np.array(["a", "a", "b", "c", "d"]),
            "path": np.array(["a/1", "a/2", "b/1", "c/1", "d/1"]),
            "cls_prob": np.array([.1, .3, .4, .8, .9])}
    threshold = runner.validation_threshold(rows, rows["cls_prob"], .05)
    assert threshold == pytest.approx(.4)
    report = runner.metrics(rows, rows["cls_prob"], threshold)
    assert report["video_auc"] == 1.
    assert report["video_fpr_at_val_threshold"] == 0.
    assert report["video_tpr_at_val_threshold"] == 1.
    with pytest.raises(ValueError, match="finite"):
        runner.metrics(rows, np.full(5, np.nan), threshold)
    assert runner.metrics(rows, np.full(5, .5), .5)["video_ap"] == .5


def test_checkpoint_rejects_wrong_base_identity_and_nonfinite_parameters(runner):
    import torch
    model = torch.nn.Linear(2, 2)
    artifact = runner.checkpoint_artifact(model, "CLS_HEAD", {"dim": 2}, "a" * 64)
    with pytest.raises(ValueError, match="identity"):
        runner.restore_artifact(model, artifact, "CLS_HEAD", {"dim": 2}, "b" * 64)
    artifact["state_dict"]["weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        runner.restore_artifact(model, artifact, "CLS_HEAD", {"dim": 2}, "a" * 64)


@pytest.mark.parametrize("fail_arm,use_lmdb", [(None, False), (None, True), ("CLS_HEAD", False)])
def test_all_arms_train_reload_export_without_mutating_source(runner, tmp_path, monkeypatch,
                                                            fail_arm, use_lmdb):
    import torch
    from test_g30 import TinyB0
    from test_e1001 import evaluation_metadata
    from run_g25 import TEST_DS
    from e0924_export import strict_loader
    lmdb = pytest.importorskip("lmdb") if use_lmdb else None

    torch.set_num_threads(1)
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    for name in TEST_DS:
        (metadata_dir / f"{name}.json").write_text(json.dumps(evaluation_metadata(name)))
    config = {"model_name": "effort", "dataset_json_folder": str(metadata_dir), "compression": "c23",
              "test_batchSize": 4, "train_batchSize": 4, "label_dict": {"FF-real": 0, "FF-DF": 1},
              "std": [1., 1., 1.], "mean": [0., 0., 0.]}
    config_path, ckpt = tmp_path / "config.json", tmp_path / "b0.pth"
    config_path.write_text(json.dumps(config))
    original = TinyB0("eager").eval()
    original.config = config
    torch.save(original.state_dict(), ckpt)
    before = runner.legacy.state_sha256(original)

    class Frames:
        def __init__(self, cfg, dataset):
            self.config = cfg
            rows = [(f"{dataset}/{label}/video{i}/{j}.png", label)
                    for label in (0, 1) for i in range(4) for j in (10, 2, 1)]
            self.image_list, self.label_list = map(list, zip(*rows))
            if lmdb is not None:
                database = tmp_path / "lmdb" / dataset
                database.mkdir(parents=True, exist_ok=True)
                self.env = lmdb.open(str(database), map_size=1024 * 1024)

        def load_rgb(self, path):
            if lmdb is not None:
                self.env.stat()
            return torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(sum(path.encode())))

        def to_tensor(self, x):
            return x

        def normalize(self, x):
            return x

    def load_model(cfg, path):
        model = TinyB0("eager")
        model.load_state_dict(torch.load(path, weights_only=True))
        model.config = cfg
        return model.eval()

    utilities = SimpleNamespace(DEVICE=torch.device("cpu"), load_model=load_model,
                                get_data_loader=lambda cfg, name: SimpleNamespace(dataset=Frames(cfg, name)))
    monkeypatch.setitem(sys.modules, "experiment_utils", utilities)

    def make_train(cfg, ratio):
        source = Frames(cfg, "FaceForensics++")
        loader = strict_loader(source, cfg["train_batchSize"])
        loader.source_dataset = source
        return loader

    monkeypatch.setattr(runner.legacy, "make_train_loader", make_train)
    real_make_model = runner.make_model
    if fail_arm is not None:
        def failed_make_model(base, feature_dim, arm, args, device):
            if arm == fail_arm:
                raise RuntimeError("controlled independent arm failure")
            return real_make_model(base, feature_dim, arm, args, device)
        monkeypatch.setattr(runner, "make_model", failed_make_model)
    out = tmp_path / "E1002"
    cli = ["--base_checkpoint", str(ckpt), "--base_config", str(config_path),
                             "--output_dir", str(out), "--n_epochs", "1", "--hidden_dim", "4",
                             "--rank", "2", "--matrix_layers", "1", "--preserve_top", "2",
                             "--clip_frames", "3", "--max_train_batches", "2"]
    if fail_arm:
        cli += ["--arms", "CLS_HEAD", "PATCH_MEAN"]
    exit_code = runner.main(cli)
    assert exit_code == (1 if fail_arm else 0)
    folder, = out.iterdir()
    report = json.loads((folder / "all_results.json").read_text())
    assert len(report["arms"]) == (2 if fail_arm else 13)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["base_state_unchanged"] and manifest["base_file_unchanged"]
    assert runner.legacy.state_sha256(original) == before
    for name, row in report["arms"].items():
        if name == fail_arm:
            assert row["status"] == "FAILED"
            assert "controlled independent arm failure" in row["error"]
            assert not (folder / name / "best.pth").exists()
            continue
        assert row["status"] == "OK"
        assert row["selected_epoch"] == 1
        assert row["selection_dataset"] == "FaceForensics++/val"
        assert row["trainable_parameters"] > 0
        assert row["optimizer_steps"] > 0
        assert set(row["datasets"]) == set(TEST_DS)
        assert (folder / name / "best.pth").is_file()
        with np.load(folder / name / "exports" / "WDF.npz") as values:
            assert np.isfinite(values["score"]).all()
            assert len(values["labels"]) == len(values["score"])

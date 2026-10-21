"""E1021 protocol, strict full-path readout, and persisted failures."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/run_e1021.py"


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    assert SCRIPT.is_file(), "E1021 runner has not been implemented"
    spec = importlib.util.spec_from_file_location("e1021_runner_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_builder():
    tree = ast.parse((ROOT / "experiments/experiment_utils.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_config")
    namespace = {"os": os, "yaml": yaml, "_deepfake_dir": str(ROOT),
                 "DETECTOR_YAML": str(ROOT / "training/config/detector/effort.yaml"),
                 "TRAIN_YAML": str(ROOT / "training/config/train_config.yaml"),
                 "TEST_YAML": str(ROOT / "training/config/test_config.yaml")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "config_builder", "exec"), namespace)
    return namespace["build_config"]


def test_dry_run_has_five_fresh_arms_and_needs_no_ml_dependencies(tmp_path):
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'transformers', 'experiment_utils'}:
        raise AssertionError('ML dependency imported during dry run: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[2], '--dry_run', '--output_dir', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    destination = tmp_path / "unused"
    process = subprocess.run([sys.executable, "-S", "-c", code, str(SCRIPT.parent), str(SCRIPT), str(destination)],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    plan = json.loads(process.stdout)
    assert list(plan["arms"]) == ["B0", "R2F2", "R4F4", "R2F6", "R6F2"]
    assert plan["arms"]["R2F6"]["e1021_real_rank"] == 2
    assert plan["arms"]["R2F6"]["e1021_fake_rank"] == 6
    assert plan["training"]["training_passes"] == 11
    assert plan["training"]["dual_objective"] == "mean(real CE) + mean(fake CE)"
    assert plan["training"]["baseline_objective"] == "mean(batch CE)"
    assert plan["lora_scaling"] == 4.0
    assert not destination.exists()


@pytest.mark.parametrize("arm,ranks", [("B0", None), ("R2F2", (2, 2)), ("R4F4", (4, 4)),
                                      ("R2F6", (2, 6)), ("R6F2", (6, 2))])
def test_actual_merged_config_preserves_b0_conditions(runner, tmp_path, arm, ranks):
    args = runner.build_parser().parse_args([])
    for training in (True, False):
        config = runner.build_arm_config(args, arm, tmp_path, config_builder(), training)
        assert config["model_name"] == ("effort" if arm == "B0" else "effort_e1021")
        if ranks is None:
            assert "e1021_real_rank" not in config and "e1021_fake_rank" not in config
        else:
            assert (config["e1021_real_rank"], config["e1021_fake_rank"]) == ranks
        assert config["optimizer"]["type"] == "adam"
        assert config["optimizer"]["adam"]["lr"] == .0002
        assert config["optimizer"]["adam"]["weight_decay"] == .0005
        assert config["manualSeed"] == 1024
        assert config["train_batchSize"] == 32
        assert config["compression"] == "c23"
        assert config["frame_num"] == {"train": 8, "test": 8}
        assert config["nEpochs"] == (10 if training else 0)
        assert config["sampler_real_ratio"] == .3
        assert config["full_train_head"] is True
        assert config["use_mixup"] is False
        assert config["multi_crop"] is False
        assert config["use_adaptive_threshold"] is False
        assert config["use_data_augmentation"] is training
        assert config["e0924_protocol"] is True
        assert config["train_dataset"] == ["FaceForensics++"]
        assert config["test_dataset"] == (["Celeb-DF-v2"] if training else runner.TEST_DS)


def test_custom_ranks_and_data_overrides_reach_both_configs(runner, tmp_path):
    args = runner.build_parser().parse_args([
        "--arms", "CUSTOM", "--real_rank", "3", "--fake_rank", "5", "--seed", "7", "--n_epochs", "2",
        "--batch_size", "40", "--dataset_json_folder", str(tmp_path / "metadata"),
        "--rgb_root", str(tmp_path / "images"), "--clip_pretrained_path", "/pretrained/clip"])
    runner.validate_arguments(runner.build_parser(), args)
    assert list(runner.plan(args)["arms"]) == ["CUSTOM"]
    for training in (True, False):
        config = runner.build_arm_config(args, "CUSTOM", tmp_path, config_builder(), training)
        assert (config["e1021_real_rank"], config["e1021_fake_rank"]) == (3, 5)
        assert config["train_batchSize"] == 40
        assert config["manualSeed"] == 7
        assert config["nEpochs"] == (2 if training else 0)
        assert config["dataset_json_folder"] == str((tmp_path / "metadata").resolve())
        assert config["rgb_root_override"] == str(tmp_path / "images")
        assert config["clip_pretrained_path"] == "/pretrained/clip"


@pytest.mark.parametrize("argv", [["--arms", "CUSTOM"], ["--real_rank", "3", "--fake_rank", "5"],
                                ["--arms", "CUSTOM", "--real_rank", "0", "--fake_rank", "4"],
                                ["--arms", "R2F2", "R2F2"], ["--batch_size", "0"],
                                ["--sampler_real_ratio", "0.999"], ["--sampler_real_ratio", "nan"],
                                ["--seed", "-1"], ["--n_epochs", "-1"]])
def test_invalid_protocol_is_rejected_before_dry_run(runner, argv):
    with pytest.raises(SystemExit) as error:
        runner.main(["--dry_run", *argv])
    assert error.value.code == 2


def scores():
    # Repeated basenames must remain different videos, including real/fake collisions.
    return {"labels": np.array([0, 0, 1, 1]),
            "path": np.array(["real/a/0.png", "real/b/0.png", "fake/a/0.png", "fake/b/0.png"]),
            "video_id": np.array(["real/a", "real/b", "fake/a", "fake/b"]),
            "cls_prob": np.array([.1, .2, .8, .9]), "global_log_odds": np.array([-2., -1., 1., 2.])}


@pytest.mark.parametrize("failure,expected", [(None, "OK"), ("train", "TRAIN_FAILED"),
                                               ("missing_checkpoint", "TRAIN_FAILED"),
                                               ("export", "EVAL_FAILED"), ("missing_dataset", "EVAL_FAILED")])
def test_run_persists_configs_fullpath_metrics_and_truthful_status(runner, monkeypatch, tmp_path, failure, expected):
    checkpoint = tmp_path / "selected.pth"
    checkpoint.write_bytes(b"selected checkpoint")

    def train(config, train_dataset, validation_dataset):
        assert config["model_name"] == "effort_e1021"
        assert (train_dataset, validation_dataset) == ("FaceForensics++", "Celeb-DF-v2")
        if failure == "train":
            return None
        return str(tmp_path / "missing.pth") if failure == "missing_checkpoint" else str(checkpoint)

    def export(config, selected, utilities):
        assert config["use_data_augmentation"] is False and selected == str(checkpoint)
        if failure == "export":
            raise ValueError("unreadable frame")
        datasets = runner.TEST_DS[:-1] if failure == "missing_dataset" else runner.TEST_DS
        return {dataset: scores() for dataset in datasets}

    monkeypatch.setattr(runner, "export_checkpoint", export)
    utilities = SimpleNamespace(build_config=config_builder(), train_model=train)
    args = runner.build_parser().parse_args(["--arms", "R2F6"])
    directory = tmp_path / "arm"
    result = runner.run_one(args, "R2F6", directory, utilities)
    assert result["status"] == expected
    assert json.loads((directory / "result.json").read_text())["status"] == expected
    assert (directory / "train_config.json").is_file() and (directory / "eval_config.json").is_file()
    if expected == "OK":
        metric = result["datasets"]["WDF"]
        assert metric["video_auc"] == metric["video_auc_fullpath"] == pytest.approx(1.)
        assert metric["num_videos"] == 4 and metric["legacy_group_collisions"] == 2
        assert metric["video_auc_legacy"] is None
        assert result["video_auc_seven_mean_fullpath"] == pytest.approx(1.)
        assert result["independent_video_auc_mean_fullpath"] == pytest.approx(1.)
        assert result["checkpoint_sha256"]
        with np.load(directory / "exports/WDF.npz", allow_pickle=False) as values:
            np.testing.assert_array_equal(values["video_id"], scores()["video_id"])
    else:
        assert result["error"]


def test_export_scores_real_strict_singleframe_loader_and_retains_full_paths(runner, tmp_path):
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))

        def forward(self, data, inference=False):
            score = data["image"].reshape(-1) * self.weight
            logits = torch.stack((-score, score), dim=1)
            return {"cls": logits, "prob": logits.softmax(1)[:, 1]}

    model = Model()
    original = SimpleNamespace(image_list=["real/same/0.png", "fake/same/0.png"], label_list=[0, 1],
                               load_rgb=lambda path: -.7 if path.startswith("real") else .7,
                               to_tensor=lambda image: torch.tensor([image]), normalize=lambda image: image)
    utilities = SimpleNamespace(load_model=lambda config, checkpoint: model,
                                get_data_loader=lambda config, dataset: SimpleNamespace(dataset=original))
    args = runner.build_parser().parse_args([])
    config = runner.build_arm_config(args, "R2F2", tmp_path, config_builder(), False)
    exported = runner.export_checkpoint(config, "selected.pth", utilities)
    assert set(exported) == set(runner.TEST_DS)
    assert exported["WDF"]["video_id"].tolist() == ["real/same", "fake/same"]
    assert exported["WDF"]["cls_prob"].tolist() == pytest.approx([.1978161, .8021839])
    assert not model.training

    def unreadable(path):
        raise OSError("bad image")

    original.load_rgb = unreadable
    with pytest.raises(OSError, match="bad image"):
        runner.export_checkpoint(config, "selected.pth", utilities)


def test_main_saves_manifest_and_failure_even_if_runtime_import_fails(runner, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "experiment_utils", None)
    output = tmp_path / "output"
    assert runner.main(["--arms", "B0", "--output_dir", str(output)]) == 1
    manifest_paths = list(output.glob("seed*/manifest.json"))
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["arguments"]["output_dir"] == str(output)
    assert manifest["error"]
    assert (manifest_paths[0].parent / "all_results.json").is_file()


def test_manifest_records_runtime_versions_and_metadata_identity(runner, monkeypatch, tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for dataset in runner.TEST_DS:
        (metadata / f"{dataset}.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "experiment_utils", SimpleNamespace(
        build_config=config_builder(), train_model=lambda *args: None))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="torch-test"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="transformers-test"))
    output = tmp_path / "output"
    assert runner.main(["--arms", "B0", "--output_dir", str(output),
                        "--dataset_json_folder", str(metadata)]) == 1
    manifest_path, = output.glob("seed*/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["torch"] == "torch-test"
    assert manifest["transformers"] == "transformers-test"
    assert set(manifest["dataset_sha256"]) == set(runner.TEST_DS)
    result = json.loads((manifest_path.parent / "all_results.json").read_text())
    assert result["arms"]["B0"]["status"] == "TRAIN_FAILED"

"""E1012 training protocol, explicit B0 pairing, and runner failure records."""

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
SCRIPT = ROOT / "experiments/run_e1012.py"


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    assert SCRIPT.is_file(), "E1012 runner has not been implemented"
    spec = importlib.util.spec_from_file_location("e1012_runner_test", SCRIPT)
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


def scores(cls=(.9, .9, .4), primary=(.1, .8, .2)):
    return {"labels": np.array([0, 1, 1]), "path": np.array(["real/a/0.png", "fake/b/0.png", "fake/c/0.png"]),
            "video_id": np.array(["real/a", "fake/b", "fake/c"]),
            "cls_prob": np.array(cls), "global_log_odds": np.array([1., 2., -1.]),
            "evidence_prob": np.array(primary), "primary_prob": np.array(primary)}


def test_dry_run_describes_full_matrix_without_importing_ml_or_creating_output(tmp_path):
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
    assert len(plan["arms"]) == 124
    assert sum(name.startswith("G1_") for name in plan["arms"]) == 18
    assert sum(name.startswith("G2_G25_") for name in plan["arms"]) == 48
    assert sum(name.startswith("G2_G25V2_") for name in plan["arms"]) == 48
    assert sum(name.startswith("G3_") for name in plan["arms"]) == 6
    assert sum(name.startswith("G4_") for name in plan["arms"]) == 3
    assert plan["training"]["training_passes"] == 11
    assert not destination.exists()


@pytest.mark.parametrize("arm,family,mode,mask,supervision,likelihood,asymmetric", [
    ("G1_L2_FL_MIL", "lfeq", "joint", "read_only", "max", "weighted", True),
    ("G2_G25_M01_UNIFORM", "aux", "joint", "patch_only", "max", "uniform", False),
    ("G2_G25V2_A10_FL_MIL", "aux", "isolated", "cls_only", "all", "weighted", True),
    ("G3_MIL", "decoder", "joint", "read_only", "max", "off", True),
])
def test_real_config_builder_preserves_historical_training_and_requested_model(runner, tmp_path, arm,
                                                                             family, mode, mask, supervision,
                                                                             likelihood, asymmetric):
    args = runner.build_parser().parse_args(["--seed", "2048", "--n_epochs", "3", "--batch_size", "40",
                                             "--dataset_json_folder", str(tmp_path / "metadata"),
                                             "--rgb_root", str(tmp_path / "images"),
                                             "--clip_pretrained_path", "/pretrained/clip"])
    for training in (True, False):
        config = runner.build_arm_config(args, arm, tmp_path, config_builder(), training)
        assert config["model_name"] == "effort_e1012"
        settings = config["e1012_settings"]
        assert (settings["family"], settings["aux_grad_mode"], settings["attention_mode"],
                settings["supervision"], settings["likelihood"], settings["asymmetric"]) == (
                    family, mode, mask, supervision, likelihood, asymmetric)
        assert config["optimizer"]["type"] == "adam"
        assert config["optimizer"]["adam"]["lr"] == .0002
        assert config["optimizer"]["adam"]["weight_decay"] == .0005
        assert config["use_data_augmentation"] is training
        assert config["nEpochs"] == (3 if training else 0)
        assert config["manualSeed"] == 2048
        assert config["train_batchSize"] == 40
        assert config["clip_pretrained_path"] == "/pretrained/clip"
        assert config["dataset_json_folder"] == str((tmp_path / "metadata").resolve())
        assert config["rgb_root_override"] == str(tmp_path / "images")
        assert config["use_mixup"] is False
        assert config["e0924_protocol"] is True
        assert config["train_dataset"] == ["FaceForensics++"]
        assert config["test_dataset"] == (["Celeb-DF-v2"] if training else runner.TEST_DS)


def test_partial_plan_still_includes_one_independent_baseline(runner):
    args = runner.build_parser().parse_args(["--arms", "G3_PLAIN", "G4_RANDOM"])
    assert list(runner.plan(args)["arms"]) == ["G0", "G3_PLAIN", "G4_RANDOM"]
    args.baseline_only = True
    assert list(runner.plan(args)["arms"]) == ["G0"]
    config = runner.build_arm_config(args, "G0", Path("unused"), config_builder(), True)
    assert config["model_name"] == "effort"
    assert "e1012_settings" not in config


def test_corrections_use_independent_b0_instead_of_joint_models_cls(runner):
    values = scores()
    reference = scores(cls=(.1, .2, .7))
    report = runner.paired_metrics(values, values["primary_prob"], reference)
    assert (report["corrected_frames"], report["harmed_frames"]) == (1, 1)
    assert report["real_fpr"] == 0
    assert report["reference"] == "independent_G0"
    reference["path"] = reference["path"][::-1]
    with pytest.raises(ValueError, match="path"):
        runner.paired_metrics(values, values["primary_prob"], reference)


@pytest.mark.parametrize("failure,expected", [(None, "OK"), ("train", "TRAIN_FAILED"),
                                               ("evaluate", "EVAL_FAILED"), ("export", "EVAL_FAILED")])
def test_run_saves_selected_checkpoint_and_truthful_stage_status(runner, monkeypatch, tmp_path, failure, expected):
    calls = []
    checkpoint = tmp_path / "selected.pth"
    checkpoint.write_bytes(b"test checkpoint")
    reference = {ds: scores(cls=(.1, .2, .7)) for ds in runner.TEST_DS}

    def train(config, train_dataset, val_dataset):
        calls.append(("train", config, train_dataset, val_dataset))
        return None if failure == "train" else str(checkpoint)

    def evaluate(config, ckpt, datasets, train_dataset, directory, name):
        assert ckpt == str(checkpoint)
        assert datasets == runner.TEST_DS
        if failure == "evaluate":
            raise RuntimeError("evaluation failed")
        return {"testall": {ds: {"video_auc": .9} for ds in datasets}}

    def export(config, ckpt, utilities):
        assert ckpt == str(checkpoint)
        if failure == "export":
            raise ValueError("bad frame")
        return {ds: scores() for ds in runner.TEST_DS}

    monkeypatch.setattr(runner, "export_checkpoint", export)
    args = runner.build_parser().parse_args(["--arms", "G3_PLAIN"])
    utilities = SimpleNamespace(build_config=config_builder(), train_model=train, evaluate_model=evaluate,
                                seed_evaluation=lambda seed: calls.append(("seed", seed)))
    result, exported = runner.run_one(args, "G3_PLAIN", tmp_path / "arm", utilities, reference)
    assert result["status"] == expected
    assert json.loads((tmp_path / "arm/result.json").read_text())["status"] == expected
    assert (tmp_path / "arm/train_config.json").exists()
    if expected == "OK":
        assert result["checkpoint_sha256"]
        assert result["independent_mean"]["primary"]["excluded"] == ["Celeb-DF-v2", "FaceForensics++"]
        with np.load(tmp_path / "arm/exports/WDF.npz", allow_pickle=False) as values:
            np.testing.assert_array_equal(values["baseline_prob"], reference["WDF"]["cls_prob"])
            np.testing.assert_array_equal(values["cls_prob"], scores()["cls_prob"])
        assert exported
    else:
        assert exported is None


def test_g4_command_binds_this_runs_b0_and_does_not_request_training(runner, tmp_path):
    args = runner.build_parser().parse_args(["--seed", "2048", "--dataset_json_folder", str(tmp_path / "metadata")])
    baseline = {"ckpt": str(tmp_path / "G0/selected.pth"),
                "config_path": str(tmp_path / "G0/train_config.json")}
    command = runner.prototype_command(args, tmp_path, baseline, ["G4_RANDOM", "G4_TOPK"])
    assert command[:2] == [sys.executable, str(ROOT / "experiments/run_e1010.py")]
    assert command[command.index("--base_checkpoint") + 1] == baseline["ckpt"]
    assert command[command.index("--base_config") + 1] == baseline["config_path"]
    assert "--base_run" not in command
    assert command[command.index("--output_dir") + 1] == str(tmp_path / "G4_diagnostics")
    assert command[-3:] == ["--arms", "G4_RANDOM", "G4_TOPK"]


def test_main_writes_serializable_manifest_and_dataset_identity(runner, monkeypatch, tmp_path):
    utilities = SimpleNamespace(build_config=config_builder(), train_model=lambda *args: None,
                                evaluate_model=lambda *args: None, load_model=lambda *args: None,
                                get_data_loader=lambda *args: None)
    monkeypatch.setitem(sys.modules, "experiment_utils", utilities)
    metadata_hashes = {name: "a" * 64 for name in runner.TEST_DS}

    def preflight(args, source, **kwargs):
        assert source["config"]["use_data_augmentation"] is True
        return source["config"], {}, metadata_hashes

    monkeypatch.setattr(runner.diagnostics, "preflight", preflight)
    monkeypatch.setattr(runner, "run_one", lambda *args: ({"status": "OK"}, {}))
    assert runner.main(["--baseline_only", "--output_dir", str(tmp_path / "output"),
                        "--dataset_json_folder", str(tmp_path / "metadata")]) == 0
    paths = list((tmp_path / "output").glob("seed*/manifest.json"))
    assert len(paths) == 1
    manifest = json.loads(paths[0].read_text())
    assert manifest["status"] == "OK"
    assert manifest["arguments"]["output_dir"] == str(tmp_path / "output")
    assert manifest["arguments"]["dataset_json_folder"] == str(tmp_path / "metadata")
    assert manifest["dataset_sha256"] == metadata_hashes


def test_preflight_uses_the_same_local_metadata_default_as_training(runner, monkeypatch):
    expected = str(ROOT / "preprocessing/dataset_json")
    assert Path(expected).is_dir()

    def preflight(args, source, **kwargs):
        assert source["config"]["dataset_json_folder"] == expected
        return source["config"], {}, {}

    monkeypatch.setattr(runner.diagnostics, "preflight", preflight)
    assert runner.main(["--preflight", "--baseline_only"]) == 0

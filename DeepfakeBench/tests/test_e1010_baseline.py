"""Fresh E1010 B0 preserves the E0924/G26 baseline protocol."""

import ast
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_fresh_baseline_helper_is_available_without_ml_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    assert importlib.util.find_spec("e1010_baseline") is not None


@pytest.fixture
def baseline(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    return importlib.import_module("e1010_baseline")


@pytest.fixture
def config_builder():
    # Execute the real builder without importing datasets or detector registries.
    path = ROOT / "experiments/experiment_utils.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "build_config")
    namespace = {"os": os, "yaml": yaml, "_deepfake_dir": str(ROOT),
                 "DETECTOR_YAML": str(ROOT / "training/config/detector/effort.yaml"),
                 "TRAIN_YAML": str(ROOT / "training/config/train_config.yaml"),
                 "TEST_YAML": str(ROOT / "training/config/test_config.yaml")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["build_config"]


def arguments(tmp_path, **values):
    return SimpleNamespace(seed=1024, n_epochs=1, sampler_real_ratio=.5, batch_size=4,
                           base_n_epochs=10, base_sampler_real_ratio=.3, base_batch_size=None,
                           dataset_json_folder=tmp_path / "metadata", rgb_root=None,
                           clip_pretrained_path=None, **values)


def test_fresh_config_keeps_legacy_baseline_training_conditions(baseline, config_builder, tmp_path):
    import run_g26

    args = arguments(tmp_path)
    expected_args = run_g26.build_parser().parse_args(["--arms", "B0"])
    for training in (True, False):
        expected = run_g26.build_arm_config(expected_args, "B0", tmp_path, config_builder, training)
        expected.update(e0924_protocol=True, dataset_json_folder=str(args.dataset_json_folder.resolve()))
        actual = baseline.build_baseline_config(args, tmp_path, config_builder, training)
        assert actual == expected
        assert actual["use_data_augmentation"] is True
        assert actual["manualSeed"] == 1024
        assert actual["use_balance_batch_sampler"] is True
        assert actual["balance_sampler_v2"] is False
        assert actual["sampler_real_ratio"] == .3
        assert actual["full_train_head"] is True
        assert actual["multi_crop"] is False
        assert actual["use_mixup"] is False
        assert actual["margin_loss_mode"] == "off"
        assert actual["optimizer"]["adam"]["lr"] == .0002
        assert actual["nEpochs"] == (10 if training else 0)
        assert not any("init_ckpt" in key for key in actual)


def test_base_budget_and_data_overrides_reach_train_and_evaluation(baseline, config_builder, tmp_path):
    args = arguments(tmp_path)
    args.seed, args.base_n_epochs, args.base_sampler_real_ratio = 7, 2, .4
    args.base_batch_size, args.rgb_root, args.clip_pretrained_path = 20, "images", "local_clip"
    for training in (True, False):
        actual = baseline.build_baseline_config(args, tmp_path, config_builder, training)
        assert actual["manualSeed"] == 7
        assert actual["sampler_real_ratio"] == .4
        assert actual["train_batchSize"] == 20
        assert actual["rgb_root_override"] == "images"
        assert actual["clip_pretrained_path"] == "local_clip"
        assert actual["dataset_json_folder"] == str(args.dataset_json_folder.resolve())
        assert actual["nEpochs"] == (2 if training else 0)


def test_unset_metadata_override_keeps_original_builder_path(baseline, tmp_path):
    args = arguments(tmp_path)
    args.dataset_json_folder = None
    config = baseline.build_baseline_config(
        args, tmp_path, lambda **kwargs: {**kwargs, "dataset_json_folder": "configured_metadata"})
    assert config["dataset_json_folder"] == "configured_metadata"


def utilities(config_builder, checkpoint, stage):
    def train(config, training_dataset, validation_dataset):
        assert training_dataset == "FaceForensics++"
        assert validation_dataset == "Celeb-DF-v2"
        assert config["use_data_augmentation"] is True
        if stage == "train_error":
            raise RuntimeError("training failed")
        if stage == "train_none":
            return None
        if stage != "missing_checkpoint":
            checkpoint.write_bytes(b"trained-baseline-test-fixture")
        return str(checkpoint)

    def evaluate(config, selected_checkpoint, datasets, training_dataset, folder, name):
        assert selected_checkpoint == str(checkpoint.resolve())
        assert config["e0924_protocol"] is True
        assert config["multi_crop"] is False
        if stage == "eval_error":
            raise RuntimeError("evaluation failed")
        values = {"WDF": .9, "FFIW": .8, "Celeb-DF-v2": .95,
                  "DeepFakeDetection": .7, "DFDC": .6, "DFDCP": .85,
                  "DeeperForensics-1.0": .75, "FaceForensics++": .99}
        return {"testall": {} if stage == "missing_metrics" else
                {dataset: {"video_auc": value, "auc": value - .01}
                 for dataset, value in values.items()}}

    return SimpleNamespace(build_config=config_builder, train_model=train,
                           evaluate_model=evaluate, seed_evaluation=lambda seed: None)


@pytest.mark.parametrize("relative_paths", [False, True])
def test_fresh_training_saves_reusable_config_and_separate_historical_metrics(
        baseline, config_builder, tmp_path, monkeypatch, relative_paths):
    monkeypatch.chdir(tmp_path)
    checkpoint, folder = ((Path("b0.pth"), Path("G0_training")) if relative_paths else
                          (tmp_path / "b0.pth", tmp_path / "G0_training"))
    result = baseline.prepare_baseline(arguments(tmp_path), folder,
                                       utilities(config_builder, checkpoint, "ok"))
    assert result["status"] == "OK"
    assert result["ckpt"] == str(checkpoint.resolve())
    assert result["config_path"] == str((folder / "train_config.json").resolve())
    config = json.loads(Path(result["config_path"]).read_text())
    assert config["use_data_augmentation"] is True
    assert config["nEpochs"] == 10
    assert config["test_dataset"] == ["Celeb-DF-v2"]
    report = json.loads(Path(result["reproduction_path"]).read_text())
    assert report["status"] == "OK"
    assert report["reproduction_status"] == "REFERENCE_NOT_PROVIDED"
    assert report["checkpoint_selection"]["dataset"] == "Celeb-DF-v2"
    assert report["checkpoint_selection"]["metric"] == "frame_auc"
    assert report["metric_protocol"] == "project legacy basename video grouping"
    assert report["video_auc_seven_mean"] == pytest.approx(.7928571428571428)
    assert report["AUC_cross"] == pytest.approx(.775)
    assert report["G"] == pytest.approx(.215)
    assert report["datasets"]["Celeb-DF-v2"]["video_auc"] == .95
    assert report["independent_datasets"] == ["WDF", "FFIW", "DeepFakeDetection", "DFDC", "DFDCP", "DeeperForensics-1.0"]
    assert report["independent_video_auc_mean"] == pytest.approx(.7666666666666667)
    assert json.loads((folder / "result.json").read_text())["config_path"] == result["config_path"]


@pytest.mark.parametrize("stage,expected", [
    ("train_none", "TRAIN_FAILED"), ("train_error", "TRAIN_FAILED"),
    ("missing_checkpoint", "TRAIN_FAILED"), ("eval_error", "EVAL_FAILED"),
    ("missing_metrics", "EVAL_FAILED"),
])
def test_failed_baseline_never_claims_reproduction(baseline, config_builder, tmp_path, stage, expected):
    result = baseline.prepare_baseline(arguments(tmp_path), tmp_path / "G0_training",
                                       utilities(config_builder, tmp_path / "b0.pth", stage))
    assert result["status"] == expected
    report = json.loads(Path(result["reproduction_path"]).read_text())
    assert report["status"] == expected
    assert report["reproduction_status"] == "NOT_EVALUATED"
    assert "video_auc_seven_mean" not in report
    assert "G" not in report

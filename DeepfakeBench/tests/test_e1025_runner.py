"""E1025 ablations preserve B0 conditions and strict video readout."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_e1021_runner import config_builder, scores


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/run_e1025.py"


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    assert SCRIPT.is_file(), "E1025 runner is missing"
    spec = importlib.util.spec_from_file_location("e1025_runner_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_dry_run_has_sixteen_arms_without_ml_or_side_effects(tmp_path):
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'transformers', 'experiment_utils'}:
        raise AssertionError('ML import during dry run: ' + name)
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
    protocol = json.loads(process.stdout)
    assert protocol["experiment"] == "E1025"
    arms = protocol["arms"]
    assert len(arms) == 16
    assert arms["B0"] == {"model_name": "effort"}
    dual = [value for value in arms.values() if value["model_name"] == "effort_e1025_soft"]
    assert {(v["e1021_real_rank"], v["e1021_fake_rank"], v["e1025_beta"]) for v in dual} == {
        (real, fake, beta) for real, fake in ((2, 2), (4, 4), (2, 6), (6, 2)) for beta in (0, .25, 1)}
    assert {v["e1025_prior"] for v in arms.values() if v["model_name"] == "effort_e1025_pu"} == {.1, .3, .5}
    assert protocol["training"]["training_passes"] == 11
    assert protocol["gradient_clipping"] is None
    assert not destination.exists()


def test_every_effective_config_retains_b0_training_conditions(runner, tmp_path):
    args = runner.build_parser().parse_args([])
    for arm, settings in runner.arm_settings(args).items():
        for training in (True, False):
            config = runner.build_arm_config(args, arm, tmp_path, config_builder(), training)
            assert all(config[key] == value for key, value in settings.items())
            assert config["optimizer"]["type"] == "adam"
            assert config["optimizer"]["adam"]["lr"] == .0002
            assert config["optimizer"]["adam"]["weight_decay"] == .0005
            assert config["manualSeed"] == 1024
            assert config["train_batchSize"] == 32
            assert config["sampler_real_ratio"] == .3
            assert config["compression"] == "c23"
            assert config["frame_num"] == {"train": 8, "test": 8}
            assert config["nEpochs"] == (10 if training else 0)
            assert config["full_train_head"] is True
            assert config["use_mixup"] is False
            assert config["multi_crop"] is False
            assert config["use_adaptive_threshold"] is False
            assert config["use_data_augmentation"] is training
            assert config["e0924_protocol"] is True


def test_family_rank_and_hyperparameter_selection_are_independent(runner, tmp_path):
    parser = runner.build_parser()
    args = parser.parse_args(["--arms", "SOFT", "PU", "--ranks", "R2F6", "--betas", ".5", "1",
                              "--priors", ".2", ".4", "--sampler_real_ratio", ".375",
                              "--batch_size", "40", "--n_epochs", "2", "--seed", "7",
                              "--clip_pretrained_path", "/local/clip", "--rgb_root", "/local/images",
                              "--dataset_json_folder", str(tmp_path)])
    runner.validate_arguments(parser, args)
    settings = runner.arm_settings(args)
    assert len(settings) == 4
    assert {v["e1025_beta"] for v in settings.values() if "e1025_beta" in v} == {.5, 1}
    for arm in settings:
        config = runner.build_arm_config(args, arm, tmp_path, config_builder(), True)
        assert config["manualSeed"] == 7 and config["nEpochs"] == 2
        assert config["train_batchSize"] == 40 and config["sampler_real_ratio"] == .375
        assert config["clip_pretrained_path"] == "/local/clip"
        assert config["rgb_root_override"] == "/local/images"
        assert config["dataset_json_folder"] == str(tmp_path.resolve())


@pytest.mark.parametrize("argv", [["--betas", "-0.1"], ["--betas", "1.1"], ["--betas", "nan"],
                                ["--priors", "0"], ["--priors", "1"], ["--priors", "nan"],
                                ["--betas", ".25", ".25"], ["--priors", ".3", ".3"],
                                ["--ranks", "R4F4", "R4F4"], ["--arms", "B0", "B0"],
                                ["--sampler_real_ratio", ".999"], ["--batch_size", "0"],
                                ["--n_epochs", "-1"], ["--seed", "-1"]])
def test_invalid_protocol_is_rejected_before_creating_output(runner, argv, tmp_path):
    destination = tmp_path / "unused"
    with pytest.raises(SystemExit) as error:
        runner.main(["--dry_run", "--output_dir", str(destination), *argv])
    assert error.value.code == 2
    assert not destination.exists()


@pytest.mark.parametrize("failure,expected", [(None, "OK"), ("train", "TRAIN_FAILED"),
                                               ("missing_checkpoint", "TRAIN_FAILED"),
                                               ("export", "EVAL_FAILED"), ("missing_dataset", "EVAL_FAILED")])
def test_real_runner_persists_configs_strict_metrics_and_failure_states(runner, monkeypatch, tmp_path, failure, expected):
    checkpoint = tmp_path / "selected.pth"
    checkpoint.write_bytes(b"selected checkpoint")

    def train(config, train_dataset, validation_dataset):
        assert config["model_name"] == "effort_e1025_pu" and config["e1025_prior"] == .3
        assert (train_dataset, validation_dataset) == ("FaceForensics++", "Celeb-DF-v2")
        if failure == "train":
            return None
        return str(tmp_path / "missing.pth") if failure == "missing_checkpoint" else str(checkpoint)

    def export(config, selected, utilities):
        if failure == "export":
            raise OSError("unreadable frame")
        return {dataset: scores() for dataset in (runner.TEST_DS[:-1] if failure == "missing_dataset" else runner.TEST_DS)}

    monkeypatch.setattr(runner, "export_checkpoint", export)
    args = runner.build_parser().parse_args(["--arms", "PU", "--priors", ".3"])
    arm, = runner.arm_settings(args)
    result = runner.run_one(args, arm, tmp_path / "arm", SimpleNamespace(build_config=config_builder(), train_model=train))
    assert result["status"] == expected
    assert json.loads((tmp_path / "arm/result.json").read_text())["status"] == expected
    assert (tmp_path / "arm/train_config.json").is_file()
    if expected == "OK":
        assert result["datasets"]["WDF"]["video_auc"] == 1.
        assert result["datasets"]["WDF"]["num_videos"] == 4
        assert result["datasets"]["WDF"]["legacy_group_collisions"] == 2
        assert result["independent_video_auc_mean_fullpath"] == 1.
        with np.load(tmp_path / "arm/exports/WDF.npz", allow_pickle=False) as exported:
            np.testing.assert_array_equal(exported["video_id"], scores()["video_id"])


def test_runtime_failure_still_persists_manifest(runner, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "experiment_utils", None)
    assert runner.main(["--arms", "B0", "--output_dir", str(tmp_path)]) == 1
    manifest_path, = tmp_path.glob("seed*/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "FAILED" and manifest["error"]
    assert (manifest_path.parent / "all_results.json").is_file()


def test_manifest_records_e1025_sources_versions_and_metadata(runner, monkeypatch, tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for dataset in runner.TEST_DS:
        (metadata / f"{dataset}.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "experiment_utils", SimpleNamespace(
        build_config=config_builder(), train_model=lambda *args: None))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="torch-test"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="transformers-test"))
    output = tmp_path / "output"
    assert runner.main(["--arms", "PU", "--priors", ".2", "--output_dir", str(output),
                        "--dataset_json_folder", str(metadata)]) == 1
    manifest_path, = output.glob("seed*/manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["torch"] == "torch-test" and manifest["transformers"] == "transformers-test"
    assert set(manifest["dataset_sha256"]) == set(runner.TEST_DS)
    assert "training/detectors/e1025_losses.py" in manifest["source_sha256"]
    assert "training/detectors/effort_detector_e1025_soft.py" in manifest["source_sha256"]
    assert "training/detectors/effort_detector_e1025_pu.py" in manifest["source_sha256"]
    results = json.loads((manifest_path.parent / "all_results.json").read_text())
    assert results["experiment"] == "E1025"
    assert results["arms"]["PU_P0.2"]["status"] == "TRAIN_FAILED"

"""E1009 protocol, lazy CLI, and saved run records without data or GPU."""

import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/run_e1009.py"
MATRIX = [
    ("G25_TOKENS_M00", "G25", "tokens", "M00", "read_only", "max"),
    ("G25_TOKENS_M10", "G25", "tokens", "M10", "cls_only", "max"),
    ("G25_TOKENS_M01", "G25", "tokens", "M01", "patch_only", "max"),
    ("G25_TOKENS_M11", "G25", "tokens", "M11", "full", "max"),
    ("G25_TOKENS_A00", "G25", "tokens", "A00", "read_only", "all"),
    ("G25_TOKENS_A10", "G25", "tokens", "A10", "cls_only", "all"),
    ("G25_TOKENS_A01", "G25", "tokens", "A01", "patch_only", "all"),
    ("G25_TOKENS_A11", "G25", "tokens", "A11", "full", "all"),
    ("G25V2_TOKENS_M00", "G25v2", "tokens", "M00", "read_only", "max"),
    ("G25V2_TOKENS_M10", "G25v2", "tokens", "M10", "cls_only", "max"),
    ("G25V2_TOKENS_M01", "G25v2", "tokens", "M01", "patch_only", "max"),
    ("G25V2_TOKENS_M11", "G25v2", "tokens", "M11", "full", "max"),
    ("G25V2_TOKENS_A00", "G25v2", "tokens", "A00", "read_only", "all"),
    ("G25V2_TOKENS_A10", "G25v2", "tokens", "A10", "cls_only", "all"),
    ("G25V2_TOKENS_A01", "G25v2", "tokens", "A01", "patch_only", "all"),
    ("G25V2_TOKENS_A11", "G25v2", "tokens", "A11", "full", "all"),
    ("G25_LATE_LORA_M01", "G25", "late_lora", "M01", "patch_only", "max"),
    ("G25_LATE_LORA_M11", "G25", "late_lora", "M11", "full", "max"),
    ("G25V2_LATE_LORA_M01", "G25v2", "late_lora", "M01", "patch_only", "max"),
    ("G25V2_LATE_LORA_M11", "G25v2", "late_lora", "M11", "full", "max"),
    ("G25_LAYERNORM_M01", "G25", "layernorm", "M01", "patch_only", "max"),
    ("G25_LAYERNORM_M11", "G25", "layernorm", "M11", "full", "max"),
    ("G25V2_LAYERNORM_M01", "G25v2", "layernorm", "M01", "patch_only", "max"),
    ("G25V2_LAYERNORM_M11", "G25v2", "layernorm", "M11", "full", "max"),
    ("G25_ALL_LORA_M01", "G25", "all_lora", "M01", "patch_only", "max"),
    ("G25_ALL_LORA_M11", "G25", "all_lora", "M11", "full", "max"),
    ("G25V2_ALL_LORA_M01", "G25v2", "all_lora", "M01", "patch_only", "max"),
    ("G25V2_ALL_LORA_M11", "G25v2", "all_lora", "M11", "full", "max"),
]
DATASETS = ["WDF", "FFIW", "Celeb-DF-v2", "DeepFakeDetection", "DFDC", "DFDCP",
            "DeeperForensics-1.0", "FaceForensics++"]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    assert SCRIPT.is_file(), "E1009 runner is missing"
    spec = importlib.util.spec_from_file_location("e1009_runner_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics():
    return {"testall": {dataset: {"video_auc": 0.9} for dataset in DATASETS}}


def test_actual_dry_run_has_all_28_arms_without_ml_imports(tmp_path):
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'transformers', 'experiment_utils'}:
        raise AssertionError('dry_run imported ML dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[2], '--dry_run', '--output_dir', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    output = tmp_path / "unused"
    process = subprocess.run(
        [sys.executable, "-S", "-c", code, str(SCRIPT.parent), str(SCRIPT), str(output)],
        text=True, capture_output=True, check=False,
    )
    assert process.returncode == 0, process.stderr
    settings = json.loads(process.stdout)
    assert list(settings) == [row[0] for row in MATRIX]
    assert not output.exists()
    assert settings["G25_TOKENS_M00"]["g25v2_aux_grad_mode"] == "joint"
    assert settings["G25V2_TOKENS_A11"]["g25v2_aux_grad_mode"] == "isolated"
    assert settings["G25V2_ALL_LORA_M11"]["e1009_tuning"] == "all_lora"


@pytest.mark.parametrize("arm,variant,tuning,token_arm,mask,supervision", MATRIX)
def test_train_and_eval_configs_preserve_requested_arm(runner, tmp_path, arm, variant,
                                                     tuning, token_arm, mask, supervision):
    args = runner.build_parser().parse_args([
        "--num_tokens", "7", "--insert_layer", "17", "--seed", "2048",
        "--n_epochs", "3", "--sampler_real_ratio", "0.4",
        "--clip_pretrained_path", "/pretrained/clip",
    ])
    metadata = runner.ARMS[arm]
    assert metadata["variant"] == variant
    assert metadata["tuning"] == tuning
    assert metadata["token_arm"] == token_arm
    expected = {
        "model_name": "effort_e1009", "e1009_tuning": tuning,
        "g25_num_tokens": 7, "g25_insert_layer": 17,
        "g25_attention_mode": mask, "g25_supervision": supervision,
        "g25_fusion_weight": 0.5, "g25_evidence_weight": 1.0,
        "g25_diversity_weight": 0.01,
        "g25v2_aux_grad_mode": "joint" if variant == "G25" else "isolated",
        "g25v2_score_mode": "fused", "manualSeed": 2048,
        "clip_pretrained_path": "/pretrained/clip", "sampler_real_ratio": 0.4,
        "full_train_head": True, "use_mixup": False, "mixup_mode": "none",
        "margin_loss_mode": "off", "use_freq_split": False,
        "use_texture_crop": False, "optimizer_wrapper": None, "rank_loss_weight": 0.0,
    }
    for training in (True, False):
        config = runner.build_arm_config(args, arm, tmp_path, lambda **kw: kw, training)
        assert {key: config[key] for key in expected} == expected
        assert config["n_epochs"] == (3 if training else 0)
        assert config["test_dataset"] == ("Celeb-DF-v2" if training else DATASETS)
        assert config["train_dataset"] == "FaceForensics++"
        assert config["testall_artifact_dir"] == str((tmp_path / "testall_artifacts").resolve())


@pytest.mark.parametrize("arm,aux_mode", [
    ("G25_TOKENS_A01", "joint"), ("G25V2_TOKENS_A01", "isolated"),
])
@pytest.mark.parametrize("failed_mode", [None, "cls", "evidence"])
def test_selected_checkpoint_readouts_and_configs_are_saved(runner, tmp_path, arm,
                                                           aux_mode, failed_mode):
    received_configs = {}
    readouts = []

    def train(config, train_dataset, val_dataset):
        received_configs["train"] = dict(config)
        assert (train_dataset, val_dataset) == ("FaceForensics++", "Celeb-DF-v2")
        return "selected_checkpoint.pth"

    def evaluate(config, checkpoint, datasets, train_dataset, output, exp_name):
        received_configs["eval"] = dict(config)
        assert checkpoint == "selected_checkpoint.pth"
        assert datasets == DATASETS
        return metrics()

    def testall(checkpoint, datasets, log, extra_config, artifact_dir):
        mode = extra_config["g25v2_score_mode"]
        readouts.append((mode, checkpoint, artifact_dir))
        assert checkpoint == "selected_checkpoint.pth"
        assert extra_config["model_name"] == "effort_e1009"
        assert extra_config["e1009_tuning"] == "tokens"
        assert extra_config["g25v2_aux_grad_mode"] == aux_mode
        if mode == failed_mode:
            raise RuntimeError("diagnostic failed")
        return metrics()["testall"]

    adapter = SimpleNamespace(build_config=lambda **kw: kw, train_model=train,
                              evaluate_model=evaluate, run_testall=testall,
                              seed_evaluation=lambda seed: None)
    args = runner.build_parser().parse_args([])
    folder = tmp_path / arm
    result = runner.run_one(args, arm, folder, adapter)
    expected_status = "OK" if failed_mode is None else "EVAL_FAILED"
    assert result["status"] == expected_status
    assert result["primary_status"] == "OK"
    assert result["variant"] == ("G25" if aux_mode == "joint" else "G25v2")
    assert result["tuning"] == "tokens"
    assert result["token_arm"] == "A01"
    assert list(result["readouts"]) == ["fused", "cls", "evidence"]
    assert [mode for mode, _, _ in readouts] == ["cls", "evidence"]
    assert len({path for _, _, path in readouts}) == 2
    for kind in ("train", "eval"):
        saved = json.loads((folder / f"{kind}_config.json").read_text())
        assert saved == received_configs[kind]
        assert saved["g25_attention_mode"] == "patch_only"
        assert saved["g25_supervision"] == "all"
    assert json.loads((folder / "result.json").read_text())["status"] == expected_status
    for mode in ("cls", "evidence"):
        saved = json.loads((folder / "readouts" / mode / "result.json").read_text())
        assert saved["status"] == ("EVAL_FAILED" if mode == failed_mode else "OK")
    with pytest.raises(FileExistsError):
        runner.run_one(args, arm, folder, adapter)


@pytest.mark.parametrize("stage", ["train_none", "train_raise", "eval_raise", "missing_metrics"])
def test_failures_save_result_and_do_not_report_success(runner, tmp_path, stage):
    def train(*args):
        if stage == "train_raise":
            raise RuntimeError("training error")
        return None if stage == "train_none" else "checkpoint.pth"

    def evaluate(*args):
        if stage == "eval_raise":
            raise RuntimeError("evaluation error")
        return {"testall": {}}

    adapter = SimpleNamespace(build_config=lambda **kw: kw, train_model=train,
                              evaluate_model=evaluate, seed_evaluation=lambda seed: None)
    args = runner.build_parser().parse_args(["--skip_readout_diagnostics"])
    folder = tmp_path / stage
    result = runner.run_one(args, "G25_TOKENS_M00", folder, adapter)
    expected = "TRAIN_FAILED" if stage.startswith("train") else "EVAL_FAILED"
    assert result["status"] == expected
    assert json.loads((folder / "result.json").read_text())["status"] == expected
    assert (folder / "train_config.json").is_file()
    assert (folder / "eval_config.json").is_file()


def test_dry_run_does_not_start_training(runner, tmp_path, monkeypatch, capsys):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert name != "experiment_utils", "dry_run loaded training utilities"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    assert runner.main(["--dry_run", "--output_dir", str(tmp_path / "unused")]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 28
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("argv", [
    ["--arms", "G25_TOKENS_M00", "G25_TOKENS_M00"], ["--num_tokens", "0"],
    ["--insert_layer", "24"], ["--sampler_real_ratio", "0"], ["--n_epochs", "-1"],
])
def test_invalid_protocol_arguments_are_rejected(runner, argv):
    with pytest.raises(SystemExit) as exc:
        runner.main([*argv, "--dry_run"])
    assert exc.value.code == 2


def test_main_records_manifest_and_continues_after_a_failed_arm(runner, tmp_path, monkeypatch):
    def train(config, *args):
        return None if config["g25v2_aux_grad_mode"] == "joint" else "checkpoint.pth"

    utilities = SimpleNamespace(build_config=lambda **kw: kw, train_model=train,
                                evaluate_model=lambda *args: metrics(),
                                run_testall=lambda *args, **kwargs: metrics()["testall"])
    monkeypatch.setitem(sys.modules, "experiment_utils", utilities)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None)))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="test-torch", manual_seed=lambda seed: None,
        cuda=SimpleNamespace(is_available=lambda: False),
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="test-transformers"))
    exit_code = runner.main([
        "--arms", "G25_TOKENS_M01", "G25V2_TOKENS_M01",
        "--skip_readout_diagnostics", "--output_dir", str(tmp_path),
    ])
    assert exit_code == 1
    roots = list(tmp_path.glob("seed1024_*"))
    assert len(roots) == 1
    root = roots[0]
    results = json.loads((root / "all_results.json").read_text())
    assert [row["status"] for row in results] == ["TRAIN_FAILED", "OK"]
    assert [row["arm"] for row in results] == ["G25_TOKENS_M01", "G25V2_TOKENS_M01"]
    assert list(results[1]["readouts"]) == ["fused"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["git_head"]) == 40
    assert manifest["arguments"]["arms"] == ["G25_TOKENS_M01", "G25V2_TOKENS_M01"]
    assert "experiments/run_e1009.py" in manifest["source_sha256"]
    assert "training/detectors/effort_detector_e1009.py" in manifest["source_sha256"]
    assert all(len(digest) == 64 for digest in manifest["source_sha256"].values())
    assert (root / "G25_TOKENS_M01" / "result.json").is_file()
    assert (root / "G25V2_TOKENS_M01" / "result.json").is_file()

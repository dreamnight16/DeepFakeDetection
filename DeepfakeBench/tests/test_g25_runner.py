"""Runner/config contracts, independent of torch, data, and pretrained weights."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_g25_test", ROOT / "experiments" / "run_g25.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_matrix_only_changes_mask_or_supervision():
    args = runner.build_parser().parse_args([])
    assert len(args.arms) == 10
    for code in runner.MASKS:
        maximum = runner.arm_settings(args, "M" + code)
        all_tokens = runner.arm_settings(args, "A" + code)
        assert maximum.pop("g25_supervision") == "max"
        assert all_tokens.pop("g25_supervision") == "all"
        assert maximum == all_tokens
    first = runner.arm_settings(args, "M00")
    first.pop("g25_attention_mode")
    for code in runner.MASKS:
        arm = runner.arm_settings(args, "M" + code)
        arm.pop("g25_attention_mode")
        assert arm == first


def test_training_eval_and_testall_keep_all_g25_settings(tmp_path):
    args = runner.build_parser().parse_args(["--num_tokens", "7", "--insert_layer", "17"])
    configs = [runner.build_arm_config(args, "A01", tmp_path, lambda **kw: kw, training)
               for training in (True, False)]
    for key, value in runner.arm_settings(args, "A01").items():
        assert configs[0][key] == configs[1][key] == value
    tree = ast.parse((ROOT / "experiments" / "experiment_utils.py").read_text(encoding="utf-8"))
    keys = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "arch_keys" for t in node.targets))
    assert set(runner.arm_settings(args, "A01")) <= set(keys)
    assert "clip_pretrained_path" in keys


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1, 1.1, True])
def test_incomplete_or_invalid_metrics_fail(value):
    metrics = {d: {"video_auc": 0.9} for d in runner.TEST_DS}
    metrics["DFDC"]["video_auc"] = value
    assert runner.validate_metrics({"testall": metrics})["status"] == "EVAL_FAILED"


def test_metrics_keep_cross_and_seven_mean_separate():
    metrics = {d: {"video_auc": 0.7} for d in runner.TEST_DS}
    metrics["DFDC"]["video_auc"] = 0.9
    result = runner.validate_metrics({"testall": metrics})
    assert result["status"] == "OK"
    assert result["AUC_cross"] == pytest.approx(0.8)
    assert result["video_auc_seven_mean"] == pytest.approx(5.1 / 7)


@pytest.mark.parametrize("stage", ["training_none", "training_raise", "evaluation_raise", "missing_metrics", "ok"])
def test_run_artifacts_and_failure_status(tmp_path, stage):
    def train(*args):
        if stage == "training_raise":
            raise RuntimeError("training error")
        return None if stage == "training_none" else "checkpoint.pth"

    def evaluate(*args):
        if stage == "evaluation_raise":
            raise RuntimeError("evaluation error")
        return {"testall": {} if stage == "missing_metrics" else
                {d: {"video_auc": 0.9} for d in runner.TEST_DS}}

    adapter = SimpleNamespace(build_config=lambda **kw: kw, train_model=train,
                              evaluate_model=evaluate, seed_evaluation=lambda seed: None)
    args = runner.build_parser().parse_args([])
    folder = tmp_path / stage
    result = runner.run_one(args, "M10", folder, adapter)
    expected = ("TRAIN_FAILED" if stage.startswith("training") else
                "OK" if stage == "ok" else "EVAL_FAILED")
    assert result["status"] == expected
    assert json.loads((folder / "result.json").read_text())["status"] == expected
    assert (folder / "train_config.json").exists()
    assert (folder / "eval_config.json").exists()
    with pytest.raises(FileExistsError):
        runner.run_one(args, "M10", folder, adapter)


def test_dry_run_and_invalid_args(capsys):
    assert runner.main(["--dry_run"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 10
    with pytest.raises(SystemExit):
        runner.main(["--arms", "M00", "M00", "--dry_run"])
    with pytest.raises(SystemExit):
        runner.main(["--insert_layer", "24", "--dry_run"])


def test_testall_command_preserves_seed_mask_and_artifact_path(tmp_path, monkeypatch):
    # Execute the real utility function without importing unrelated data/GPU
    # infrastructure, then inspect the exact subprocess YAML and arguments.
    source = (ROOT / "experiments" / "experiment_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_testall")
    detector_yaml = tmp_path / "detector.yaml"
    test_yaml = tmp_path / "test.yaml"
    detector_yaml.write_text("manualSeed: 1024\nmodel_name: effort\nuse_mixup: true\n")
    test_yaml.write_text("mode: test\n")
    namespace = dict(os=os, sys=sys, tempfile=tempfile, subprocess=subprocess, yaml=yaml,
                     _deepfake_dir=str(ROOT), DETECTOR_YAML=str(detector_yaml), TEST_YAML=str(test_yaml))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / "experiments" / "experiment_utils.py"), "exec"), namespace)

    # Use the actual evaluate_model config assembly, not a hand-written list.
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_model")
    start = next(i for i, node in enumerate(evaluate.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "arch_keys" for t in node.targets))
    args = runner.build_parser().parse_args(["--seed", "2048"])
    config = runner.build_arm_config(args, "A01", tmp_path, lambda **kw: kw, False)
    assembly = {"config": config}
    exec(compile(ast.Module(body=evaluate.body[start:start + 3], type_ignores=[]), "config_assembly", "exec"), assembly)

    def capture(command, **kwargs):
        config = yaml.safe_load(Path(command[command.index("--detector_path") + 1]).read_text())
        assert config["manualSeed"] == 2048
        assert config["g25_attention_mode"] == "patch_only"
        assert config["g25_supervision"] == "all"
        assert config["use_mixup"] is False
        assert command[command.index("--artifact_dir") + 1] == str(tmp_path / "artifacts")
        kwargs["stdout"].write("dataset: DFDC\nvideo_auc: 0.9\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", capture)
    result = namespace["run_testall"](
        "checkpoint.pth", ["DFDC"], str(tmp_path / "test.log"),
        assembly["extra_config"],
        str(tmp_path / "artifacts"),
    )
    assert result["DFDC"]["video_auc"] == 0.9

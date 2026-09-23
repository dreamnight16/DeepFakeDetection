"""G25v2 protocol and same-checkpoint readout diagnostics without GPU/data."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    spec = importlib.util.spec_from_file_location("g25v2_runner_test", ROOT / "experiments/run_g25v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(runner):
    return {"testall": {d: {"video_auc": 0.9} for d in runner.baseline.TEST_DS}}


def test_same_matrix_and_only_gradient_routing_changes(runner, tmp_path):
    args = runner.build_parser().parse_args([])
    assert len(args.arms) == 10
    for arm in args.arms:
        old = runner.baseline.arm_settings(args, arm)
        new = runner.arm_settings(args, arm)
        diff = {k for k in new if new[k] != old.get(k)}
        assert diff == (set() if arm == "B0" else
                        {"model_name", "g25v2_aux_grad_mode", "g25v2_score_mode"})
        train = runner.build_arm_config(args, arm, tmp_path, lambda **kw: kw, True)
        evaluation = runner.build_arm_config(args, arm, tmp_path, lambda **kw: kw, False)
        for key in new:
            assert train[key] == evaluation[key]
    tree = ast.parse((ROOT / "experiments/experiment_utils.py").read_text())
    keys = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "arch_keys" for t in n.targets))
    assert {"g25v2_aux_grad_mode", "g25v2_score_mode"} <= set(keys)


@pytest.mark.parametrize("failed_mode", [None, "cls", "evidence"])
def test_same_checkpoint_diagnostics_isolated_artifacts_and_failure_status(runner, tmp_path, failed_mode):
    calls = []

    def evaluate(*args):
        return metrics(runner)

    def testall(checkpoint, datasets, log, extra_config, artifact_dir):
        assert checkpoint == "one_selected_checkpoint.pth"
        mode = extra_config["g25v2_score_mode"]
        calls.append((mode, artifact_dir))
        assert extra_config["g25v2_aux_grad_mode"] == "isolated"
        if mode == failed_mode:
            raise RuntimeError("diagnostic failed")
        return metrics(runner)["testall"]

    adapter = SimpleNamespace(build_config=lambda **kw: kw,
                              train_model=lambda *args: "one_selected_checkpoint.pth",
                              evaluate_model=evaluate, run_testall=testall,
                              seed_evaluation=lambda seed: None)
    args = runner.build_parser().parse_args([])
    result = runner.run_one(args, "A01", tmp_path / "A01", adapter)
    assert result["status"] == ("OK" if failed_mode is None else "EVAL_FAILED")
    assert result["primary_status"] == "OK"
    assert [mode for mode, _ in calls] == ["cls", "evidence"]
    assert len({path for _, path in calls}) == 2
    assert set(result["readouts"]) == {"fused", "cls", "evidence"}
    assert json.loads((tmp_path / "A01/result.json").read_text())["status"] == result["status"]


def test_evaluation_only_config_uses_saved_architecture_and_validates_arm(runner, tmp_path):
    args = runner.build_parser().parse_args(["--arms", "M10"])
    source = runner.baseline.build_arm_config(args, "M10", tmp_path, lambda **kw: kw, False)
    source.update(g25_num_tokens=7, g25_insert_layer=17, manualSeed=2048)
    path = tmp_path / "eval_config.json"
    path.write_text(json.dumps(source))
    args.source_config = str(path)
    checkpoint = tmp_path / "logs" / "run" / "test" / "avg" / "ckpt_best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"dummy checkpoint; no model is loaded")
    args.checkpoint = str(checkpoint)
    record = {"arm": "M10", "seed": 2048, "settings": dict(source),
              "ckpt": str(checkpoint)}
    (tmp_path / "result.json").write_text(json.dumps(record))
    loaded = runner.load_source_config(args)
    assert loaded["g25_num_tokens"] == 7
    assert loaded["g25_insert_layer"] == 17
    assert loaded["manualSeed"] == 2048
    assert loaded["g25v2_aux_grad_mode"] == "joint"
    assert loaded["model_name"] == "effort_g25v2"
    args.checkpoint = str(tmp_path / "another_arm.pth")
    with pytest.raises(ValueError, match="Checkpoint path"):
        runner.load_source_config(args)
    args.checkpoint = str(checkpoint)
    args.arms = ["M00"]
    with pytest.raises(ValueError, match="match"):
        runner.load_source_config(args)
    args.arms = ["M10"]
    source.pop("g25_fusion_weight")
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="Incomplete"):
        runner.load_source_config(args)


def test_controls_do_not_invent_evidence_scores(runner, tmp_path):
    calls = []
    adapter = SimpleNamespace(run_testall=lambda *a, **kw: calls.append(kw) or metrics(runner)["testall"])
    primary = {"status": "OK", **metrics(runner)}
    result = runner.evaluate_readouts({"g25_num_tokens": 0}, "ckpt", tmp_path, adapter, primary)
    assert list(result) == ["fused"]
    assert calls == []


@pytest.mark.parametrize("failure", ["train", "evaluate", "missing_metrics"])
def test_failures_do_not_report_success(runner, tmp_path, failure):
    def train(*args):
        return None if failure == "train" else "ckpt"

    def evaluate(*args):
        if failure == "evaluate":
            raise RuntimeError("failed evaluation")
        return {"testall": {}}

    adapter = SimpleNamespace(build_config=lambda **kw: kw, train_model=train,
                              evaluate_model=evaluate, seed_evaluation=lambda seed: None)
    args = runner.build_parser().parse_args(["--skip_readout_diagnostics"])
    result = runner.run_one(args, "M00", tmp_path / "run", adapter)
    assert result["status"] == ("TRAIN_FAILED" if failure == "train" else "EVAL_FAILED")
    assert (tmp_path / "run/result.json").exists()


def test_dry_run_no_ml_imports_and_invalid_options(runner, capsys):
    assert runner.main(["--dry_run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["M00"]["g25v2_aux_grad_mode"] == "isolated"
    for argv in (["--checkpoint", "ckpt"], ["--num_tokens", "0"], ["--arms", "M00", "M00"]):
        with pytest.raises(SystemExit):
            runner.main([*argv, "--dry_run"])

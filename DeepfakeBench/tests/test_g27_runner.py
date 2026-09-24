"""G27 batch launch, failure accounting and diagnostic contracts."""

import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_g26_runner import ROOT, load_file


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    return load_file("g27_runner_test", ROOT / "experiments" / "run_g27.py")


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    return load_file("g27_diagnostic_test", ROOT / "experiments" / "g27_diagnostics.py")


def test_default_arms_single_factor_and_evaluation_propagation(runner, tmp_path):
    args = runner.build_parser().parse_args([])
    assert args.arms == ["B0", "G26", "E1", "E2", "E3", "Full"]
    settings = {arm: runner.arm_settings(args, arm) for arm in args.arms}
    control = settings["G26"]
    for arm, key in (("E1", "g27_balance_weight"), ("E2", "g27_hard_weighting"),
                     ("E3", "g27_consistency_weight")):
        assert {k for k in control if control[k] != settings[arm][k]} == {key}
    assert control["g27_num_tokens"] == 4 and control["g27_insert_layer"] == 20
    tree = ast.parse((ROOT / "experiments" / "experiment_utils.py").read_text(encoding="utf-8"))
    keys = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "arch_keys" for t in node.targets))
    assert set(settings["Full"]) <= set(keys)
    configs = [runner.build_arm_config(args, "Full", tmp_path, lambda **kw: kw, train)
               for train in (True, False)]
    assert all(all(c[k] == v for k, v in settings["Full"].items()) for c in configs)
    assert all(c["multi_crop"] is False and c["use_mixup"] is False for c in configs)
    evaluate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_model")
    start = next(i for i, n in enumerate(evaluate.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "arch_keys" for t in n.targets))
    end = next(i for i, n in enumerate(evaluate.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "testall_metrics" for t in n.targets))
    namespace = {"config": configs[1]}
    exec(compile(ast.Module(body=evaluate.body[start:end], type_ignores=[]), "config_contract", "exec"), namespace)
    assert namespace["extra_config"]["multi_crop"] is False
    assert all(namespace["extra_config"][k] == v for k, v in settings["Full"].items())


def test_dry_run_without_site_packages_and_custom_shape(runner, capsys):
    output = subprocess.run([sys.executable, "-S", str(ROOT / "experiments" / "run_g27.py"),
                             "--dry_run"], capture_output=True, text=True, check=True)
    assert len(json.loads(output.stdout)) == 6
    assert runner.main(["--dry_run", "--arms", "E1", "--num_tokens", "8", "--insert_layer", "16"]) == 0
    assert json.loads(capsys.readouterr().out)["E1"]["g27_num_tokens"] == 8


@pytest.mark.parametrize("args", [
    ["--balance_weight", "nan"], ["--consistency_weight", "-1"],
    ["--router_temperature", "0"], ["--hard_floor", "1.1"], ["--hard_width", "0"],
    ["--view_contrast", "inf"], ["--view_brightness", "nan"],
    ["--num_tokens", "0"], ["--insert_layer", "24"], ["--arms", "E1", "E1"],
    ["--attention_samples", "-1"], ["--gate_width", "nan"],
])
def test_invalid_args(runner, args):
    with pytest.raises(SystemExit):
        runner.main(["--dry_run", *args])


@pytest.mark.parametrize("stage", ["train_none", "train_error", "primary_error", "missing_metrics",
                                    "branch_error", "routing_error", "ok", "skip"])
def test_run_artifacts_and_failures(runner, tmp_path, stage):
    args = runner.build_parser().parse_args(["--skip_diagnostics"] if stage == "skip" else [])
    seen = []

    def train(*_):
        if stage == "train_error":
            raise RuntimeError("train failed")
        return None if stage == "train_none" else "same_checkpoint.pth"

    def primary(*_):
        if stage == "primary_error":
            raise RuntimeError("primary failed")
        return {"testall": {} if stage == "missing_metrics" else
                {d: {"video_auc": .9} for d in runner.baseline.TEST_DS}}

    def testall(checkpoint, datasets, log_path, extra_config, artifact_dir):
        seen.append(extra_config["g27_score_mode"])
        assert checkpoint == "same_checkpoint.pth"
        assert extra_config["g27_hard_weighting"] is True
        if stage == "branch_error":
            raise RuntimeError("branch failed")
        return {d: {"video_auc": .9} for d in datasets}

    def routing(*_):
        if stage == "routing_error":
            raise RuntimeError("routing failed")
        return {d: {"num_frames": 2} for d in runner.baseline.TEST_DS}

    utilities = SimpleNamespace(build_config=lambda **kw: kw, train_model=train, evaluate_model=primary,
                                run_testall=testall, seed_evaluation=lambda seed: None,
                                collect_routing_diagnostics=routing)
    folder = tmp_path / "Full"
    result = runner.run_one(args, "Full", folder, utilities)
    expected = "TRAIN_FAILED" if stage.startswith("train") else "OK" if stage in ("ok", "skip") else "EVAL_FAILED"
    assert result["status"] == expected
    assert json.loads((folder / "result.json").read_text())["status"] == expected
    if stage in ("branch_error", "routing_error", "ok"):
        assert seen == ["cls", "evidence"]
    with pytest.raises(FileExistsError):
        runner.run_one(args, "Full", folder, utilities)


def test_main_continues_failed_arm_and_records_manifest(runner, monkeypatch, tmp_path):
    import types

    utilities = types.ModuleType("experiment_utils")
    for name in ("build_config", "train_model", "evaluate_model", "run_testall", "load_model", "get_data_loader"):
        setattr(utilities, name, lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "experiment_utils", utilities)
    seen = []

    def run_one(args, arm, folder, adapter):
        seen.append(arm)
        return {"arm": arm, "status": "TRAIN_FAILED" if arm == "E1" else "OK"}

    monkeypatch.setattr(runner, "run_one", run_one)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="testsha"))
    assert runner.main(["--arms", "E1", "E2", "--output_dir", str(tmp_path)]) == 1
    assert seen == ["E1", "E2"]
    root, = tmp_path.iterdir()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["git_head"] == "testsha"
    assert all(len(v) == 64 for v in manifest["source_sha256"].values())
    assert list(manifest["arms"]) == seen
    assert len(json.loads((root / "all_results.json").read_text())) == 2


def test_responsibility_entropy_ties_correlation_and_confident_errors(diagnostic):
    labels = np.array([1, 1, 0, 0])
    cls = np.array([.1, .5, .9, .2])
    gate = np.zeros(4)
    z = np.zeros((4, 2))
    result = diagnostic.summarize_experts(labels, cls, cls, cls, gate, np.full((4, 2), .5), z, 1., .2)
    assert result["confident_cls_errors"] == 2
    assert result["fake_soft_load"] == [.5, .5]
    assert result["fake_winner_share"] == [.5, .5]  # ties do not credit expert zero
    assert result["fake_route_entropy"] == pytest.approx(np.log(2))
    assert result["query_correlation"] == [[None, None], [None, None]]
    assert result["fake_tie_rate"] == 1
    real = diagnostic.summarize_experts(np.zeros(4), cls, cls, cls, gate, np.full((4, 2), .5), z, 1., .2)
    assert real["fake_route_entropy"] is None


def test_same_forward_collector_saves_scores_and_bounded_attention(diagnostic, tmp_path):
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def forward(self, data, inference):
            cls = torch.tensor([.1, .9])
            return {"g27": {"cls_prob": cls, "evidence_prob": cls, "gated_prob": cls,
                            "gate_weight": torch.zeros(2), "evidence_log_odds": torch.zeros(2, 4)}}

        def evidence_attention(self, images):
            return torch.full((len(images), 4, 4), .1)

    data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 2])}
    utilities = SimpleNamespace(load_model=lambda *args: Model().eval(), seed_evaluation=lambda seed: None,
                                get_data_loader=lambda *args: [data, data])
    result = diagnostic.collect_routing_diagnostics(
        {"manualSeed": 1024, "g27_attention_samples": 3}, "ckpt", ["sample"], tmp_path / "routing", utilities)
    assert result["sample"]["num_frames"] == 4
    with np.load(tmp_path / "routing" / "sample.npz") as saved:
        assert saved["query_log_odds"].shape == (4, 4)
        assert saved["responsibility"].shape == (4, 4)
    with np.load(tmp_path / "routing" / "sample_attention.npz") as saved:
        assert saved["attention"].shape == (3, 4, 4)
        assert saved["frame_index"].tolist() == [0, 1, 2]

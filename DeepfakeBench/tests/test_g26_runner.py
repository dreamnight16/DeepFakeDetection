"""G26 runner and paired routing contracts without datasets or checkpoints."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    return load_file("g26_runner_test", ROOT / "experiments" / "run_g26.py")


diagnostics = load_file("g26_diagnostics_test", ROOT / "experiments" / "g26_diagnostics.py")


def test_defaults_and_config_propagation(runner, tmp_path):
    args = runner.build_parser().parse_args([])
    assert args.arms == ["B0", "K4L20", "K8L20", "K4L16", "K8L16"]
    assert args.num_tokens is None and args.insert_layer is None
    assert runner.arm_settings(args, "B0") == {"model_name": "effort"}
    args = runner.build_parser().parse_args(["--arms", "G26", "--num_tokens", "12", "--insert_layer", "16",
                                            "--gate_width", ".15", "--mil_temperature", ".7"])
    configs = [runner.build_arm_config(args, "G26", tmp_path, lambda **kw: kw, train)
               for train in (True, False)]
    tree = ast.parse((ROOT / "experiments" / "experiment_utils.py").read_text(encoding="utf-8"))
    keys = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "arch_keys" for t in node.targets))
    settings = runner.arm_settings(args, "G26")
    assert set(settings) <= set(keys)
    for key, value in settings.items():
        assert configs[0][key] == configs[1][key] == value
    for config in configs:
        assert config["multi_crop"] is False and config["use_mixup"] is False
        assert not any(key.startswith("g25") for key in config)
    assert configs[0]["test_dataset"] == runner.baseline.VAL_DS
    evaluate = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "evaluate_model")
    start = next(i for i, node in enumerate(evaluate.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "arch_keys" for t in node.targets))
    end = next(i for i, node in enumerate(evaluate.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "testall_metrics" for t in node.targets))
    namespace = {"config": configs[1]}
    exec(compile(ast.Module(body=evaluate.body[start:end], type_ignores=[]),
                 "actual_evaluate_config", "exec"), namespace)
    assert namespace["extra_config"]["multi_crop"] is False
    assert all(namespace["extra_config"][key] == value for key, value in settings.items())


def test_matrix_only_varies_count_and_position(runner, tmp_path):
    args = runner.build_parser().parse_args([])
    common = None
    for arm, shape in {"K4L20": (4, 20), "K8L20": (8, 20),
                       "K4L16": (4, 16), "K8L16": (8, 16)}.items():
        settings = runner.arm_settings(args, arm)
        assert (settings.pop("g26_num_tokens"), settings.pop("g26_insert_layer")) == shape
        if common is None:
            common = settings
        assert settings == common
        for training in (True, False):
            config = runner.build_arm_config(args, arm, tmp_path / arm, lambda **kw: kw, training)
            assert (config["g26_num_tokens"], config["g26_insert_layer"]) == shape
            assert config["manualSeed"] == 1024
            assert config["testall_artifact_dir"] == str((tmp_path / arm / "testall_artifacts").resolve())


def test_selected_pair_and_legacy_custom_shape(runner, capsys):
    assert runner.main(["--dry_run", "--arms", "K4L20", "K8L16"]) == 0
    assert list(json.loads(capsys.readouterr().out)) == ["K4L20", "K8L16"]
    assert runner.main(["--dry_run", "--arms", "G26"]) == 0
    legacy = json.loads(capsys.readouterr().out)["G26"]
    assert (legacy["g26_num_tokens"], legacy["g26_insert_layer"]) == (8, 18)
    assert runner.main(["--dry_run", "--arms", "B0", "G26", "--num_tokens", "12",
                        "--insert_layer", "16"]) == 0
    custom = json.loads(capsys.readouterr().out)["G26"]
    assert (custom["g26_num_tokens"], custom["g26_insert_layer"]) == (12, 16)


@pytest.mark.parametrize("arguments", [
    ["--num_tokens", "0"], ["--insert_layer", "24"], ["--mil_temperature", "nan"],
    ["--gate_width", "nan"], ["--aux_max_weight", ".6"], ["--evidence_weight", "inf"],
    ["--arms", "G26", "G26"],
    ["--num_tokens", "12"], ["--arms", "K4L20", "--insert_layer", "16"],
    ["--arms", "G26", "K8L16"], ["--arms", "B0", "--num_tokens", "4"],
])
def test_invalid_cli_fails_without_loading_ml(runner, arguments):
    with pytest.raises(SystemExit):
        runner.main(["--dry_run", *arguments])


def test_dry_run_works_without_site_packages():
    result = subprocess.run([sys.executable, "-S", str(ROOT / "experiments" / "run_g26.py"),
                             "--dry_run"], capture_output=True, text=True, check=True)
    matrix = json.loads(result.stdout)
    assert list(matrix) == ["B0", "K4L20", "K8L20", "K4L16", "K8L16"]
    assert matrix["K4L20"]["g26_num_tokens"] == 4
    assert matrix["K8L16"]["g26_insert_layer"] == 16


@pytest.mark.parametrize("stage", ["train_none", "train_error", "primary_error", "missing_metrics",
                                    "branch_error", "routing_error", "ok", "skip"])
@pytest.mark.parametrize("arm", ["G26", "K4L20"])
def test_artifacts_readouts_and_failures(runner, tmp_path, stage, arm):
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
                {dataset: {"video_auc": .9} for dataset in runner.baseline.TEST_DS}}

    def testall(checkpoint, datasets, log_path, extra_config, artifact_dir):
        seen.append(extra_config["g26_score_mode"])
        assert checkpoint == "same_checkpoint.pth"
        assert extra_config["g26_num_tokens"] == (8 if arm == "G26" else 4)
        assert extra_config["g26_gate_width"] == .2
        assert Path(artifact_dir).parent.name == extra_config["g26_score_mode"]
        if stage == "branch_error":
            raise RuntimeError("branch failed")
        return {dataset: {"video_auc": .9} for dataset in datasets}

    def routing(*_):
        if stage == "routing_error":
            raise RuntimeError("routing failed")
        return {dataset: {"num_frames": 2} for dataset in runner.baseline.TEST_DS}

    utilities = SimpleNamespace(build_config=lambda **kw: kw, train_model=train,
                                evaluate_model=primary, run_testall=testall,
                                seed_evaluation=lambda seed: None, collect_routing_diagnostics=routing)
    folder = tmp_path / arm
    result = runner.run_one(args, arm, folder, utilities)
    expected = "TRAIN_FAILED" if stage.startswith("train") else "OK" if stage in ("ok", "skip") else "EVAL_FAILED"
    assert result["status"] == expected
    assert json.loads((folder / "result.json").read_text())["status"] == expected
    assert (folder / "train_config.json").exists() and (folder / "eval_config.json").exists()
    if stage in ("ok", "routing_error", "branch_error", "missing_metrics"):
        assert seen == ["cls", "evidence"]
        assert (folder / "routing_diagnostics.json").exists()
    if stage == "skip":
        assert seen == [] and result["diagnostics_requested"] is False
    with pytest.raises(FileExistsError):
        runner.run_one(args, arm, folder, utilities)


def test_routing_counts_include_corrections_harm_and_no_triggers():
    labels = np.array([1, 1, 0, 0])
    cls = np.array([.45, .6, .4, .1])
    evidence = np.array([.9, .0, .9, .9])
    gate = np.array([.375, .25, .25, 0])
    fused = (1 - gate) * cls + gate * evidence
    queries = np.stack((evidence, evidence), axis=1)
    result = diagnostics.summarize_routing(labels, cls, evidence, fused, gate, queries)
    assert result["triggered_frames"] == 3
    assert result["corrected_frames"] == 1 and result["harmed_frames"] == 2
    assert result["net_corrected_frames"] == -1
    inactive = diagnostics.summarize_routing(labels, cls, evidence, cls, gate * 0, queries)
    assert inactive["trigger_rate"] == 0
    assert inactive["triggered_cls_accuracy"] is None
    with pytest.raises(ValueError, match="Inactive gate"):
        diagnostics.summarize_routing(labels, cls, evidence, fused, gate * 0, queries)


def test_collector_uses_same_forward_and_saves_query_scores(tmp_path):
    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, data, inference):
            cls = data["image"][:, 0]
            evidence = 1 - cls
            gate = torch.full_like(cls, .25)
            return {"g26": {"cls_prob": cls, "evidence_prob": evidence,
                            "gated_prob": .75 * cls + .25 * evidence, "gate_weight": gate,
                            "evidence_log_odds": torch.logit(evidence[:, None].expand(-1, 8))}}

    data = {"image": torch.tensor([[.4], [.6]]), "label": torch.tensor([0, 2])}
    utilities = SimpleNamespace(load_model=lambda *args: Model(), seed_evaluation=lambda seed: None,
                                get_data_loader=lambda *args: [data])
    output_dir = tmp_path / "routing"
    summary = diagnostics.collect_routing_diagnostics({"manualSeed": 1024}, "checkpoint", ["sample"],
                                                      output_dir, utilities)
    assert summary["sample"]["num_frames"] == 2
    with np.load(output_dir / "sample.npz") as result:
        assert result["query_prob"].shape == (2, 8)
        assert result["labels"].tolist() == [0, 1]

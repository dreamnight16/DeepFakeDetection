"""Keep E1009's tuning scope when evaluation launches testall."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("tuning", ["tokens", "late_lora", "layernorm", "all_lora"])
def test_evaluation_forwards_tuning_to_testall(tmp_path, tuning):
    # Exercise the real evaluator while replacing GPU/data/plot work. Loading
    # just this function avoids the unrelated detector/dataset import graph.
    tree = ast.parse((ROOT / "experiments/experiment_utils.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "evaluate_model")
    forwarded = {}

    def testall(checkpoint, datasets, log, extra_config, artifact_dir):
        forwarded.update(extra_config)
        return {"Celeb-DF-v2": {"video_auc": 0.9}}

    namespace = {
        "os": os, "np": np,
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        "load_model": lambda *args: object(),
        "get_data_loader": lambda *args, **kwargs: None,
        "get_train_loader": lambda *args: None,
        "collect_predictions": lambda *args: (np.array([0.1, 0.9]), np.array([0, 1])),
        "run_testall": testall,
        "print_results": lambda *args: None,
        "compute_metrics": lambda *args: {"acc": 1.0},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 "experiment_utils.py", "exec"), namespace)
    config = {"model_name": "effort_e1009", "e1009_tuning": tuning,
              "g25v2_aux_grad_mode": "isolated", "g25_num_tokens": 4}
    namespace["evaluate_model"](config, "selected.pth", ["Celeb-DF-v2"],
                                "FaceForensics++", str(tmp_path), "E1009")
    assert forwarded["e1009_tuning"] == tuning
    assert forwarded["model_name"] == "effort_e1009"
    assert forwarded["g25v2_aux_grad_mode"] == "isolated"

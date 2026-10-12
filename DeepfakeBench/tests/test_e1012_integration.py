"""Keep E1012's complete architecture when testall rebuilds a checkpoint."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("family,mode", [("lfeq", "joint"), ("aux", "joint"),
                                         ("aux", "isolated"), ("decoder", "joint")])
def test_testall_receives_complete_joint_model_settings(tmp_path, family, mode):
    tree = ast.parse((ROOT / "experiments/experiment_utils.py").read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "evaluate_model")
    forwarded = {}

    def testall(checkpoint, datasets, log, extra_config, artifact_dir):
        forwarded.update(extra_config)
        return {"Celeb-DF-v2": {"video_auc": .9}}

    namespace = {
        "os": os, "np": np,
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        "load_model": lambda *args: object(),
        "get_data_loader": lambda *args, **kwargs: None,
        "get_train_loader": lambda *args: None,
        "collect_predictions": lambda *args: (np.array([.1, .9]), np.array([0, 1])),
        "run_testall": testall,
        "print_results": lambda *args: None,
        "compute_metrics": lambda *args: {"acc": 1.},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 "experiment_utils.py", "exec"), namespace)
    settings = {"family": family, "aux_grad_mode": mode, "num_tokens": 3,
                "memory_layer": 1, "likelihood": "weighted", "asymmetric": True,
                "lfeq_fusion_weight": 0., "attention_mode": "patch_only"}
    config = {"model_name": "effort_e1012", "full_train_head": True,
              "e1012_settings": settings}
    namespace["evaluate_model"](config, "selected.pth", ["Celeb-DF-v2"],
                                "FaceForensics++", str(tmp_path), "E1012")
    assert forwarded["model_name"] == "effort_e1012"
    assert forwarded.get("e1012_settings") == settings
    assert forwarded.get("use_adaptive_threshold") is False

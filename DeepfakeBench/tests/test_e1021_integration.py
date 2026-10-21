"""Exercise the actual Trainer step and testall configuration forwarding."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from test_e1021 import config, data, detector


ROOT = Path(__file__).resolve().parents[1]


def function(path, name, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    if class_name:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == class_name)
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


@pytest.mark.parametrize("dual", [False, True])
def test_actual_trainer_step_uses_class_gradients_only_for_e1021(detector, dual):
    cls, base_cls, _ = detector
    model = cls(config()) if dual else base_cls(config())
    batch = data()
    output = model(batch)
    if dual:
        target = next(p for name, p in model.named_parameters() if name.endswith("lora_B_real"))
        expected = torch.autograd.grad(F.cross_entropy(output["cls"][:1], batch["label"][:1]), target)[0]
    else:
        target = model.head.weight
        expected = torch.autograd.grad(F.cross_entropy(output["cls"], batch["label"]), target)[0]
    method = function("training/trainer/trainer_v2.py", "train_step", "Trainer")
    namespace = {"DDP": DDP, "SAM": type("SAM", (), {}), "PCGrad": type("PCGrad", (), {})}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "trainer_v2.py", "exec"), namespace)
    trainer = SimpleNamespace(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=.01),
                              config={"optimizer": {"type": "adam"},
                                      "model_name": "effort_e1021" if dual else "effort"})
    before = target.detach().clone()
    losses, _ = namespace["train_step"](trainer, batch)
    assert torch.isfinite(losses["overall"])
    torch.testing.assert_close(target.grad, expected)
    assert not torch.equal(target, before)


def test_testall_reconstructs_unequal_ranks_and_original_lora_backend(tmp_path):
    forwarded = {}

    def testall(checkpoint, datasets, log, extra_config, artifact_dir):
        forwarded.update(extra_config)
        return {"Celeb-DF-v2": {"video_auc": .9}}

    namespace = {
        "os": os, "np": np,
        "torch": SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
        "load_model": lambda *args: object(), "get_data_loader": lambda *args, **kw: None,
        "get_train_loader": lambda *args: None,
        "collect_predictions": lambda *args: (np.array([.1, .9]), np.array([0, 1])),
        "run_testall": testall, "print_results": lambda *args: None,
        "compute_metrics": lambda *args: {"acc": 1.},
    }
    method = function("experiments/experiment_utils.py", "evaluate_model")
    exec(compile(ast.Module(body=[method], type_ignores=[]), "experiment_utils.py", "exec"), namespace)
    namespace["evaluate_model"]({"model_name": "effort_e1021", "full_train_head": True,
                                 "use_loralib": True, "e1021_real_rank": 2, "e1021_fake_rank": 6},
                                "selected.pth", ["Celeb-DF-v2"], "FaceForensics++", str(tmp_path), "E1021")
    assert forwarded["e1021_real_rank"] == 2
    assert forwarded["e1021_fake_rank"] == 6
    assert forwarded["use_loralib"] is True

"""Exercise the real Trainer step and evaluation config reconstruction."""

import ast
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from test_e1021 import config, data, detector
from test_e1021_integration import function
from test_e1025_pu import pu_detector
from test_e1025_soft import soft_detector


@pytest.mark.parametrize("name", ["effort", "effort_e1021", "effort_e1025_soft", "effort_e1025_pu"])
def test_real_trainer_updates_with_the_requested_objective(detector, soft_detector, pu_detector, name):
    cls = {"effort": detector[1], "effort_e1021": detector[0],
           "effort_e1025_soft": soft_detector, "effort_e1025_pu": pu_detector[0]}[name]
    model = cls({**config(), "e1025_beta": .25, "e1025_prior": .3}).train()
    batch = data()
    output = model(batch)
    logits, labels = output["cls"], batch["label"]
    real, fake = F.cross_entropy(logits[:1], labels[:1]), F.cross_entropy(logits[1:], labels[1:])
    if name in ("effort_e1021", "effort_e1025_soft"):
        target = next(p for key, p in model.named_parameters() if key.endswith("lora_B_real"))
        objective = real if name == "effort_e1021" else real + .25 * fake
    else:
        target = model.head.weight
        scores = logits[:, 0] - logits[:, 1]
        objective = (F.cross_entropy(logits, labels) if name == "effort" else
                     .3 * F.softplus(-scores[:1]).mean()
                     + (F.softplus(scores[1:]).mean() - .3 * F.softplus(scores[:1]).mean()).clamp(min=0))
    expected = torch.autograd.grad(objective, target)[0]
    namespace = {"DDP": DDP, "SAM": type("SAM", (), {}), "PCGrad": type("PCGrad", (), {})}
    method = function("training/trainer/trainer_v2.py", "train_step", "Trainer")
    exec(compile(ast.Module(body=[method], type_ignores=[]), "trainer_v2.py", "exec"), namespace)
    trainer = SimpleNamespace(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=.01),
                              config={"optimizer": {"type": "adam"}, "model_name": name})
    before = target.detach().clone()
    losses, _ = namespace["train_step"](trainer, batch)
    assert torch.isfinite(losses["overall"])
    torch.testing.assert_close(target.grad, expected)
    assert not torch.equal(target, before)


@pytest.mark.parametrize("settings", [
    {"model_name": "effort_e1025_soft", "e1021_real_rank": 2, "e1021_fake_rank": 6, "e1025_beta": .25},
    {"model_name": "effort_e1025_pu", "e1025_prior": .1},
])
def test_testall_receives_exact_training_hyperparameters(settings, tmp_path):
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
    namespace["evaluate_model"]({**settings, "full_train_head": True, "use_loralib": True},
                                "selected.pth", ["Celeb-DF-v2"], "FaceForensics++", str(tmp_path), "E1025")
    assert all(forwarded.get(key) == value for key, value in settings.items())
    assert forwarded["full_train_head"] and forwarded["use_loralib"]

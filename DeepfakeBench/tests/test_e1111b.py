"""Server-side E1111B formula, CE limit, gradients and model integration."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

from test_e1021 import config, data, detector
from test_e1025_pu import load

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def objective():
    return load("e1111_losses_test", "e1111_losses.py").nnPUCEWithReg


@pytest.fixture
def model_class(detector, monkeypatch):
    package = "e1021_test_package"
    for filename in ("e1025_losses", "effort_detector_e1025_pu", "e1111_losses",
                     "effort_detector_e1111b"):
        name = f"{package}.{filename}"
        module = load(name, f"{filename}.py")
        monkeypatch.setitem(sys.modules, name, module)
    return module.EffortDetectorE1111B, detector[1]


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1], [0, 1, 1]])
def test_zero_is_exact_ce_in_value_and_gradient(objective, labels):
    logits = torch.tensor([[.5, -1.], [2., 1.], [-.2, .3]], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor(labels)
    actual, expected = objective(lam=0)(logits, labels), F.cross_entropy(logits, labels)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0])


@pytest.mark.parametrize("clipped", [False, True])
def test_requested_formula_and_gradients(objective, clipped):
    logits = torch.tensor([[10., -10.], [9., -9.], [-8., 8.], [-9., 9.]] if clipped
                          else [[-.3, .3], [.2, -.2], [1., -1.], [.5, -.5]],
                          dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    negative = F.cross_entropy(logits[2:], labels[2:]) - .3 * F.cross_entropy(
        logits[:2], torch.ones(2, dtype=torch.long))
    assert bool(negative < 0) == clipped
    expected = F.cross_entropy(logits, labels) + .05 * negative.clamp(min=0)
    actual = objective(.3, .05)(logits, labels)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0],
                               torch.autograd.grad(expected, logits)[0])


def test_small_lambda_has_linear_influence_and_clamp_keeps_ce_gradient(objective):
    logits = torch.tensor([[-.5, .5], [.5, -.5], [1., -1.]], requires_grad=True)
    labels = torch.tensor([0, 1, 1])
    ce = F.cross_entropy(logits, labels)
    for lam in (.001, .01, .1):
        value = objective(.3, lam)(logits, labels)
        assert torch.isfinite(value)
        assert torch.isfinite(torch.autograd.grad(value, logits, retain_graph=True)[0]).all()
    assert torch.allclose(objective(.3, .01)(logits, labels) - ce,
                          .1 * (objective(.3, .1)(logits, labels) - ce))


@pytest.mark.parametrize("kwargs", [{"lam": -.1}, {"lam": float("nan")}, {"prior": 0},
                                  {"prior": float("inf")}])
def test_invalid_parameters_fail(objective, kwargs):
    with pytest.raises(ValueError):
        objective(**kwargs)


def test_missing_class_requires_explicit_ce_control(objective):
    with pytest.raises(ValueError, match="both"):
        objective()(torch.zeros(3, 2), torch.zeros(3, dtype=torch.long))


def test_model_preserves_b0_and_validation_ce(model_class):
    cls, baseline = model_class
    base = baseline(config()).eval()
    model = cls({**config(), "e1111_lam": 0, "e1025_prior": .3}).eval()
    assert model.state_dict().keys() == base.state_dict().keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, base.state_dict()[key])
    batch = data()
    with torch.no_grad():
        assert torch.equal(model(batch)["cls"], base(batch)["cls"])
        batch["label"].zero_()
        output = model(batch)
        assert torch.equal(model.get_losses(batch, output)["overall"],
                           F.cross_entropy(output["cls"], batch["label"]))


@pytest.mark.parametrize("lam", [0., .01])
def test_actual_trainer_step_and_checkpoint_reload(model_class, lam):
    import ast
    from types import SimpleNamespace
    from torch.nn.parallel import DistributedDataParallel as DDP
    from test_e1021_integration import function

    cls, _ = model_class
    model = cls({**config(), "e1111_lam": lam, "e1025_prior": .3}).train()
    batch = data()
    output = model(batch)
    logits = output["cls"]
    lp = F.log_softmax(logits, dim=1)[:, 1]
    objective_value = F.cross_entropy(logits, batch["label"]) + lam * (
        -lp[1:].mean() + .3 * lp[:1].mean()).clamp(min=0)
    expected = torch.autograd.grad(objective_value, model.head.weight)[0]
    namespace = {"DDP": DDP, "SAM": type("SAM", (), {}), "PCGrad": type("PCGrad", (), {})}
    method = function("training/trainer/trainer_v2.py", "train_step", "Trainer")
    exec(compile(ast.Module(body=[method], type_ignores=[]), "trainer", "exec"), namespace)
    trainer = SimpleNamespace(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=.01),
                              config={"optimizer": {"type": "adam"}, "model_name": "effort_e1111b"})
    losses, _ = namespace["train_step"](trainer, batch)
    assert torch.isfinite(losses["overall"])
    torch.testing.assert_close(model.head.weight.grad, expected)
    restored = cls({**config(), "e1111_lam": lam, "e1025_prior": .3}).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    with torch.no_grad():
        assert torch.equal(restored(batch)["cls"], model(batch)["cls"])


def test_runner_isolated_small_lambdas_and_original_config(monkeypatch, tmp_path):
    from test_e1021_runner import config_builder
    script = ROOT / "experiments/run_e1111b.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("e1111b_runner_test", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    parser = runner.build_parser()
    args = parser.parse_args([])
    runner.validate_arguments(parser, args)
    settings = runner.arm_settings(args)
    assert len(settings) == 5 and settings["B0"] == {"model_name": "effort"}
    assert {x["e1111_lam"] for x in settings.values() if "e1111_lam" in x} == {0., .01, .05, .1}
    for arm in settings:
        actual = runner.build_arm_config(args, arm, tmp_path, config_builder(), True)
        assert actual["manualSeed"] == 1024 and actual["train_batchSize"] == 32
        assert actual["frame_num"] == {"train": 8, "test": 8}
        assert actual["optimizer"]["adam"]["lr"] == .0002
        assert actual["optimizer"]["adam"]["weight_decay"] == .0005
        assert actual["model_name"] == settings[arm]["model_name"]
        if arm == "B0":
            assert "e1111_lam" not in actual and "e1025_prior" not in actual
        else:
            assert actual["e1025_prior"] == .3
    assert runner.plan(args)["branch"] == "E1111B"


def test_native_b0_config_is_exactly_the_existing_e1025_baseline(monkeypatch, tmp_path):
    import argparse
    from test_e1021_runner import config_builder

    script = ROOT / "experiments/run_e1111b.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("e1111_native_b0_test", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    parser = runner.build_parser()
    args = parser.parse_args(["--arms", "B0"])
    runner.validate_arguments(parser, args)
    assert runner.arm_settings(args) == {"B0": {"model_name": "effort"}}
    assert runner.plan(args)["arm_objectives"] == {"B0": "original batch CE"}
    old_args = argparse.Namespace(**{**vars(args), "arms": ["B0"]})
    for training in (True, False):
        actual = runner.build_arm_config(args, "B0", tmp_path, config_builder(), training)
        previous = runner.previous.build_arm_config(old_args, "B0", tmp_path, config_builder(), training)
        assert actual == previous
    only_regularizer = parser.parse_args(["--arms", "PUCE", "--lams", ".01"])
    runner.validate_arguments(parser, only_regularizer)
    assert list(runner.arm_settings(only_regularizer)) == ["PUCE_P0.3_L0.01"]
    with pytest.raises(SystemExit):
        runner.validate_arguments(parser, parser.parse_args(["--arms", "B0", "B0"]))

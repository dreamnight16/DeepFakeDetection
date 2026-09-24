"""G27 mathematical and tiny real CLIP/LoRA contracts; no GPU or downloads."""

import sys

import pytest
import torch
from torch.nn import functional as F

from test_g25 import DETECTORS, detector_class, load_file
from test_g26 import configure_lora_backend, g26


@pytest.fixture
def g27(g26, monkeypatch):
    monkeypatch.setitem(sys.modules, "g25_test_package.effort_detector_g26",
                        sys.modules[g26[0].__module__] if g26[0].__module__ in sys.modules
                        else load_file("g25_test_package.effort_detector_g26",
                                       DETECTORS / "effort_detector_g26.py"))
    helper = load_file("g25_test_package.g27_tokens", DETECTORS / "g27_tokens.py")
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    module = load_file("g25_test_package.effort_detector_g27", DETECTORS / "effort_detector_g27.py")
    return module.EffortDetectorG27, helper


def model_for(g27, **changes):
    return g27[0]({"g27_num_tokens": 4, "g27_insert_layer": 1, **changes})


def test_balance_excludes_real_and_empty_is_differentiable(g27):
    balance = g27[1].balance_loss
    z = torch.tensor([[9., -9.], [-9., 9.], [40., -40.]], requires_grad=True)
    labels = torch.tensor([1, 1, 0])
    torch.testing.assert_close(balance(z, labels, 1.), torch.tensor(0.))
    collapsed = z[[0, 0]]
    assert balance(collapsed, torch.ones(2, dtype=torch.long), 1.) > .49
    empty = balance(z, torch.zeros(3, dtype=torch.long), 1.)
    assert empty.item() == 0 and torch.isfinite(empty)
    empty.backward()
    assert torch.count_nonzero(z.grad) == 0
    # Uniform identical experts also satisfy balance: this is not proof of diversity.
    assert balance(torch.zeros(3, 4), labels, 1.).item() == 0


def test_difficulty_weight_is_detached_bounded_and_exact(g27):
    p = torch.tensor([.1, .3, .4, .5, .6, .7, .9], requires_grad=True)
    weights = g27[1].difficulty_weights(p, .2, .2)
    torch.testing.assert_close(weights, torch.tensor([.2, .2, .6, 1., .6, .2, .2]))
    assert not weights.requires_grad


def test_view_is_photometric_reproducible_and_preserves_rng(g27):
    images = torch.randn(3, 3, 8, 8)
    before = images.clone()
    rng = torch.get_rng_state().clone()
    view = g27[1].photometric_view(images, [1., 2., 4.], .9, .02)
    expected = .9 * images + .1 * images.mean((-2, -1), keepdim=True)
    expected += torch.tensor([.02, .01, .005])[None, :, None, None]
    torch.testing.assert_close(view, expected)
    assert torch.equal(images, before) and torch.equal(rng, torch.get_rng_state())
    assert not torch.equal(view, images)


@pytest.mark.parametrize("backend", ["loralib", "custom"])
def test_full_loss_protects_main_outputs_gradients_and_two_adam_steps(g27, monkeypatch, backend):
    configure_lora_backend(monkeypatch, backend)
    model = model_for(g27, g27_balance_weight=.1, g27_hard_weighting=True,
                      g27_consistency_weight=.1).train()
    base = model_for(g27).train()
    base.load_state_dict(model.state_dict())
    optimizers = [torch.optim.Adam(m.parameters(), lr=2e-4, weight_decay=5e-4)
                  for m in (base, model)]
    for _ in range(2):
        data = {"image": torch.randn(3, 3, 8, 8), "label": torch.tensor([0, 1, 1])}
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        logits = base.head(base.backbone(data["image"]).pooler_output)
        rng = torch.get_rng_state().clone()
        output = model(data)
        assert torch.equal(rng, torch.get_rng_state())
        torch.testing.assert_close(output["g27"]["global_logits"], logits)
        F.cross_entropy(logits, data["label"]).backward()
        losses = model.get_losses(data, output)
        assert losses["loss_consistency"] > 0
        losses["overall"].backward()
        reference = dict(base.named_parameters())
        for name, parameter in model.named_parameters():
            if name.startswith(("backbone.", "head.")):
                if reference[name].grad is None:
                    assert parameter.grad is None, name
                else:
                    torch.testing.assert_close(parameter.grad, reference[name].grad)
        assert model.evidence_tokens.grad.abs().sum() > 0
        assert all(h.weight.grad.abs().sum() > 0 for h in model.evidence_heads.heads)
        for optimizer in optimizers:
            optimizer.step()
        for name, parameter in model.named_parameters():
            if name.startswith(("backbone.", "head.")):
                torch.testing.assert_close(parameter, reference[name])


def test_zero_options_match_g26_and_auxiliary_is_isolated(g27, g26):
    old = g26[0]({"g26_num_tokens": 4, "g26_insert_layer": 1}).train()
    new = model_for(g27).train()
    new.load_state_dict(old.state_dict(), strict=True)
    data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}
    a, b = old(data), new(data)
    torch.testing.assert_close(a["prob"], b["prob"])
    torch.testing.assert_close(old.get_losses(data, a)["overall"], new.get_losses(data, b)["overall"])
    new.get_losses(data, b)["loss_auxiliary"].backward()
    assert all(p.grad is None for name, p in new.named_parameters()
               if name.startswith(("backbone.", "head.")))


@pytest.mark.parametrize("mode", ["gated", "cls", "evidence"])
def test_reload_multicrop_eval_no_second_view_and_attention(g27, mode):
    model = model_for(g27, g27_score_mode=mode, g27_consistency_weight=.1).eval()
    restored = model_for(g27, g27_score_mode=mode, g27_consistency_weight=.1).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    images = torch.randn(2, 3, 3, 8, 8)
    with torch.no_grad():
        flat = model({"image": images.flatten(0, 1)}, inference=True)
        multi = restored({"image": images}, inference=True)
        attention = model.evidence_attention(images[:, 0])
    assert "second_log_odds" not in multi["g27"]
    chosen = (flat["g27"]["cls_prob"].reshape(2, 3) - .5).abs().argmax(1)
    torch.testing.assert_close(multi["prob"], flat["prob"].reshape(2, 3)[torch.arange(2), chosen])
    assert attention.shape == (2, 4, 4) and torch.isfinite(attention).all()
    loss = model.get_losses({"label": torch.tensor([0, 1])}, multi)
    assert loss["loss_consistency"].item() == 0


@pytest.mark.parametrize("change", [
    {"g27_balance_weight": -1}, {"g27_balance_weight": float("nan")},
    {"g27_consistency_weight": float("inf")}, {"g27_router_temperature": 0},
    {"g27_hard_floor": 1.1}, {"g27_hard_width": 0}, {"g27_view_contrast": -1},
    {"g27_view_brightness": float("nan")}, {"std": [0., 1., 1.]},
    {"g27_hard_weighting": "false"},
])
def test_invalid_configs_fail(g27, change):
    with pytest.raises(ValueError):
        model_for(g27, **change)


def test_labels_and_training_second_view_contract(g27):
    model = model_for(g27, g27_consistency_weight=.1).train()
    output = model({"image": torch.randn(2, 3, 8, 8)})
    for data in ({"label": torch.tensor([0, 2])},
                 {"label": torch.tensor([0, 1]), "label_soft": torch.ones(2, 2)}):
        with pytest.raises(ValueError):
            model.get_losses(data, output)
    output["g27"].pop("second_log_odds")
    with pytest.raises(ValueError, match="second"):
        model.get_losses({"label": torch.tensor([0, 1])}, output)

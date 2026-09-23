"""G25v2 gradient isolation contracts on real tiny CLIP + LoRA, CPU only."""

import sys
import types

import pytest
import torch
from torch.nn import functional as F

from test_g25 import DETECTORS, load_file, tokens
from test_g25 import detector_class as original_detector_class


@pytest.fixture
def models(original_detector_class, monkeypatch):
    base = types.ModuleType("g25_test_package.effort_detector_g25")
    base.EffortDetectorG25 = original_detector_class
    monkeypatch.setitem(sys.modules, base.__name__, base)
    helper = load_file("g25_test_package.g25v2_tokens", DETECTORS / "g25v2_tokens.py")
    monkeypatch.setitem(sys.modules, helper.__name__, helper)
    revised = load_file("g25_test_package.effort_detector_g25v2", DETECTORS / "effort_detector_g25v2.py")
    return original_detector_class, revised.EffortDetectorG25v2


@pytest.mark.parametrize("mode", tokens.MASK_MODES)
@pytest.mark.parametrize("supervision", ["max", "all"])
def test_same_forward_and_loss_but_backbone_only_receives_cls_gradient(models, mode, supervision):
    old_class, new_class = models
    config = {"g25_insert_layer": 1, "g25_num_tokens": 3,
              "g25_attention_mode": mode, "g25_supervision": supervision}
    old = old_class(config).train()
    new = new_class(config).train()
    new.load_state_dict(old.state_dict(), strict=True)
    before = {name: p.detach().clone() for name, p in new.named_parameters()}
    data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}
    original = old(data)
    revised = new(data)
    torch.testing.assert_close(original["prob"], revised["prob"])
    torch.testing.assert_close(old.get_losses(data, original)["overall"],
                               new.get_losses(data, revised)["overall"])
    # A joint loss must have exactly the same original-parameter gradients as
    # CLS-only training of the SAME augmented forward (not an unmodified B0).
    F.cross_entropy(original["g25"]["global_logits"], data["label"]).backward()
    new.get_losses(data, revised)["overall"].backward()
    for name, parameter in new.named_parameters():
        torch.testing.assert_close(parameter.detach(), before[name])
        if name.startswith(("backbone.", "head.")):
            reference = dict(old.named_parameters())[name]
            if reference.grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, reference.grad)
    assert new.backbone.embeddings.class_embedding.requires_grad
    assert new.backbone.embeddings.class_embedding.grad.abs().sum() > 0
    assert new.evidence_tokens.grad.abs().sum() > 0
    assert new.backbone.encoder.layers[0].self_attn.v_proj.lora_B.grad.abs().sum() > 0
    assert new.backbone.encoder.layers[-1].self_attn.v_proj.lora_B.grad.abs().sum() > 0
    if supervision == "all":
        assert all(head.weight.grad.abs().sum() > 0 for head in new.evidence_heads.heads)


@pytest.mark.parametrize("mode", tokens.MASK_MODES)
def test_auxiliary_loss_cannot_update_any_original_parameter(models, mode):
    model = models[1]({"g25_insert_layer": 1, "g25_num_tokens": 3,
                       "g25_supervision": "all", "g25_attention_mode": mode})
    data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}
    losses = model.get_losses(data, model(data))
    (losses["loss_evidence"] + 0.01 * losses["loss_diversity"]).backward()
    for name, p in model.named_parameters():
        if name.startswith(("backbone.", "head.")):
            assert p.grad is None, name
    assert model.evidence_tokens.grad.abs().sum() > 0
    assert all(h.weight.grad.abs().sum() > 0 for h in model.evidence_heads.heads)


def test_read_only_adam_update_matches_cls_only_reference(models):
    old = models[0]({"g25_insert_layer": 1, "g25_num_tokens": 3})
    new = models[1]({"g25_insert_layer": 1, "g25_num_tokens": 3})
    new.load_state_dict(old.state_dict(), strict=True)
    optimizers = [torch.optim.Adam(m.parameters(), lr=2e-4, weight_decay=5e-4) for m in (old, new)]
    for _ in range(2):
        data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}
        for model, optimizer in zip((old, new), optimizers):
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = (F.cross_entropy(output["g25"]["global_logits"], data["label"])
                    if model is old else model.get_losses(data, output)["overall"])
            loss.backward()
            optimizer.step()
        for name, p in new.named_parameters():
            if name.startswith(("backbone.", "head.")):
                torch.testing.assert_close(p, dict(old.named_parameters())[name])


def test_joint_control_matches_old_gradients_and_no_new_parameters(models):
    config = {"g25_insert_layer": 1, "g25_num_tokens": 3, "g25_attention_mode": "full",
              "g25v2_aux_grad_mode": "joint"}
    old, new = models[0](config), models[1](config)
    new.load_state_dict(old.state_dict(), strict=True)
    assert list(old.state_dict()) == list(new.state_dict())
    data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}
    for model in (old, new):
        model.get_losses(data, model(data))["overall"].backward()
    for name, p in new.named_parameters():
        reference = dict(old.named_parameters())[name]
        if reference.grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, reference.grad)


@pytest.mark.parametrize("score_mode", ["fused", "cls", "evidence"])
def test_inference_matches_training_scores_and_multicrop_checkpoint(models, score_mode):
    config = {"g25_insert_layer": 1, "g25_num_tokens": 3, "g25v2_score_mode": score_mode}
    model = models[1](config)
    images = torch.randn(2, 3, 3, 8, 8)
    flat = images.reshape(-1, 3, 8, 8)
    train_score = model({"image": flat})["prob"].detach()
    restored = models[1](config).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        out = restored({"image": flat}, inference=True)
        torch.testing.assert_close(out["prob"], train_score)
        torch.testing.assert_close(out["cls"].softmax(-1)[:, 1], out["prob"])
        expected = (out["g25"]["global_logits"].softmax(-1)[:, 1] if score_mode == "cls" else
                    out["g25"]["selected_evidence_logits"].softmax(-1)[:, 1] if score_mode == "evidence" else
                    out["g25"]["fused_probs"][:, 1])
        torch.testing.assert_close(out["prob"], expected)
        multi = restored({"image": images}, inference=True)
        scores = out["prob"].reshape(2, 3)
        torch.testing.assert_close(multi["prob"], scores[torch.arange(2), (scores - .5).abs().argmax(1)])


def test_invalid_modes_rejected(models):
    for config in ({"g25v2_aux_grad_mode": "wrong"}, {"g25v2_score_mode": "wrong"}):
        with pytest.raises(ValueError):
            models[1]({"g25_insert_layer": 1, **config})

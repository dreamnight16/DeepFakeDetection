"""E1010 supervision and frozen-B0 contracts on a real tiny CLIP."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from test_g30 import TinyB0


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "e1010_tokens_test", ROOT / "training/detectors/e1010_tokens.py")
e1010 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e1010)


def sidecar(family="aux", **settings):
    return e1010.FrozenForgerySidecar(
        TinyB0("eager"), family=family, num_tokens=settings.pop("num_tokens", 4),
        memory_layer=1, hidden_dim=settings.pop("hidden_dim", 8), num_heads=4, depth=1, **settings)


@pytest.mark.parametrize("family", ["lfeq", "aux", "decoder"])
@pytest.mark.parametrize("likelihood", ["uniform", "weighted"])
def test_training_preserves_real_clip_b0_and_trains_projector(family, likelihood):
    torch.manual_seed(8)
    model = sidecar(family, likelihood=likelihood, asymmetric=True)
    data = {"image": torch.randn(4, 3, 8, 8)}
    labels = torch.tensor([0, 1, 0, 1])
    original = model.base(data, inference=True)
    state = {name: value.clone() for name, value in model.base.state_dict().items()}
    before = model.auxiliary.memory_projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.auxiliary.parameters(), lr=.01)
    for _ in range(2):
        model.train()
        assert not model.base.training
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        assert torch.equal(output["e1010"]["cls_prob"], original["prob"])
        assert torch.equal(output["e1010"]["global_logits"], original["cls"])
        losses = model.losses(output, labels)
        assert losses["loss_likelihood"] > 0
        losses["loss_likelihood"].backward(retain_graph=True)
        assert model.auxiliary.memory_projection.weight.grad.abs().sum() > 0
        optimizer.zero_grad(set_to_none=True)
        losses["overall"].backward()
        assert all(parameter.grad is None for parameter in model.base.parameters())
        optimizer.step()
    assert not torch.equal(before, model.auxiliary.memory_projection.weight)
    assert all(torch.equal(value, state[name]) for name, value in model.base.state_dict().items())
    disabled = model(data, auxiliary_enabled=False)
    assert all(torch.equal(disabled[key], original[key]) for key in ("cls", "prob", "feat"))


@pytest.mark.parametrize("family", ["lfeq", "aux", "decoder"])
def test_loss_switches_preserve_architecture_and_readouts(family):
    torch.manual_seed(3)
    reference = sidecar(family)
    data = {"image": torch.randn(3, 3, 8, 8)}
    reference.eval()
    original = reference(data)["e1010"]
    expected = reference.auxiliary.state_dict()
    for likelihood in ("off", "uniform", "weighted"):
        for asymmetric in (False, True):
            model = sidecar(family, likelihood=likelihood, asymmetric=asymmetric)
            model.base.load_state_dict(reference.base.state_dict())
            assert model.auxiliary.state_dict().keys() == expected.keys()
            model.auxiliary.load_state_dict(expected)
            model.eval()
            values = model(data)["e1010"]
            for key in ("evidence_log_odds", "evidence_prob", "gated_prob", "cls_prob"):
                assert torch.equal(values[key], original[key])


def test_asymmetric_loss_matches_all_real_and_exists_fake_contract():
    model = sidecar(asymmetric=True, diversity_weight=0)
    labels = torch.tensor([0, 1])
    z = torch.tensor([[1., -2., 3., 0.], [-2., 0., 2., -1.]], requires_grad=True)
    output = {"e1010": {"evidence_log_odds": z,
                          "global_logits": torch.zeros(2, 2),
                          "projected_patches": torch.zeros(2, 4, 16)}}
    actual = model.losses(output, labels)["loss_evidence"]
    bag = .5 * (torch.logsumexp(z[1] / .5, 0) - torch.log(torch.tensor(4.)))
    expected = (F.softplus(z[0]).mean() + F.softplus(-bag)) / 2
    assert torch.allclose(actual, expected)
    actual.backward()
    assert (z.grad[0] > 0).all()
    assert (z.grad[1] < 0).all()


@pytest.mark.parametrize("family", ["lfeq", "aux", "decoder"])
@pytest.mark.parametrize("likelihood", ["uniform", "weighted"])
def test_k1_likelihood_and_class_requirements(family, likelihood):
    model = sidecar(family, num_tokens=1, likelihood=likelihood, asymmetric=True)
    output = model({"image": torch.randn(3, 3, 8, 8)})
    losses = model.losses(output, torch.tensor([0, 1, 1]))
    assert all(torch.isfinite(loss).all() for loss in losses.values())
    with pytest.raises(ValueError, match="real.*two fake"):
        model.losses(output, torch.tensor([1, 1, 1]))
    with pytest.raises(ValueError, match="real.*two fake"):
        model.losses(output, torch.tensor([0, 0, 1]))


@pytest.mark.parametrize("labels", [torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.long)])
def test_likelihood_off_allows_single_class(labels):
    model = sidecar(likelihood="off", asymmetric=True)
    losses = model.losses(model({"image": torch.randn(3, 3, 8, 8)}), labels)
    assert torch.isfinite(losses["overall"])
    assert losses["loss_likelihood"] == 0


@pytest.mark.parametrize("family", ["lfeq", "aux", "decoder"])
def test_checkpoint_bound_to_base_family_and_settings(family):
    model = sidecar(family, likelihood="weighted", asymmetric=True)
    model.eval()
    data = {"image": torch.randn(3, 3, 8, 8)}
    original = model(data)["prob"].detach()
    artifact = model.checkpoint("a" * 64)
    assert artifact["family"] == "E1010"
    assert set(artifact["state_dict"]) == set(model.auxiliary.state_dict())
    with torch.no_grad():
        model.auxiliary.memory_projection.weight.add_(2)
    model.load_checkpoint(artifact, "a" * 64)
    assert torch.equal(model(data)["prob"], original)
    with pytest.raises(ValueError, match="B0"):
        model.load_checkpoint(artifact, "b" * 64)
    for change in ({"family": "G30"}, {"settings": {**artifact["settings"], "asymmetric": False}}):
        with pytest.raises(ValueError, match="E1010"):
            model.load_checkpoint({**artifact, **change}, "a" * 64)
    corrupt = copy.deepcopy(artifact)
    key = next(iter(corrupt["state_dict"]))
    corrupt["state_dict"][key].fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        model.load_checkpoint(corrupt, "a" * 64)


def test_aux_projection_starts_as_identity_and_base_never_sees_extra_tokens():
    model = sidecar("aux")
    projection = model.auxiliary.memory_projection
    assert torch.equal(projection.weight, torch.eye(16))
    assert torch.equal(projection.bias, torch.zeros(16))
    lengths = []
    handle = model.base.backbone.encoder.layers[-1].register_forward_pre_hook(
        lambda module, args: lengths.append(args[0].shape[1]))
    model({"image": torch.randn(3, 3, 8, 8)})
    handle.remove()
    # The later functional suffix is an independent call on the same operators.
    assert lengths == [5, 9]


@pytest.mark.parametrize("mode", ["read_only", "cls_only", "patch_only", "full"])
@pytest.mark.parametrize("supervision", ["max", "all"])
def test_original_aux_masks_and_supervision_preserve_base(mode, supervision):
    model = sidecar("aux", attention_mode=mode, supervision=supervision)
    data = {"image": torch.randn(3, 3, 8, 8)}
    baseline = model.base(data)
    output = model(data)
    assert torch.equal(output["e1010"]["cls_prob"], baseline["prob"])
    assert output["e1010"]["attention_maps"].shape == (3, 4, 4)
    historical = e1010.g25.forward_with_evidence(
        model.base.backbone, data["image"], model.auxiliary.evidence_tokens, 1, mode)
    historical_logits = model.auxiliary.evidence_heads(historical["features"][:, 1:])
    assert torch.equal(output["e1010"]["evidence_log_odds"],
                       historical_logits[..., 1] - historical_logits[..., 0])
    losses = model.losses(output, torch.tensor([0, 1, 1]))
    assert losses["loss_diversity"] >= 0
    losses["overall"].backward()
    assert all(parameter.grad is None for parameter in model.base.parameters())


def test_original_g18_structure_projection_and_score_are_preserved():
    model = sidecar("lfeq", lfeq_fusion_weight=.5)
    model.eval()
    data = {"image": torch.randn(3, 3, 8, 8)}
    patches = model.base.backbone(data["image"]).last_hidden_state[:, 1:].detach()
    reference = e1010.lfeq.LearnableForgeryEvidenceQuery(
        16, hidden_dim=8, num_evidence_tokens=4, depth=1, num_heads=4,
        dropout=.1, fusion_weight=.5).eval()
    state = dict(model.auxiliary.lfeq.state_dict())
    state.update({"patch_projection." + key: value
                  for key, value in model.auxiliary.memory_projection.state_dict().items()})
    reference.load_state_dict(state)
    expected = reference(patches)
    actual = model(data)["e1010"]
    assert torch.equal(actual["evidence_prob"], expected["fused_probs"][:, 1])
    assert torch.equal(actual["projected_patches"], reference.patch_projection(patches))


def test_same_dimension_lfeq_projection_keeps_values_and_receives_likelihood_gradient():
    model = sidecar("lfeq", hidden_dim=16, likelihood="weighted")
    projection = model.auxiliary.memory_projection
    assert torch.equal(projection.weight, torch.eye(16))
    assert torch.equal(projection.bias, torch.zeros(16))
    output = model({"image": torch.randn(3, 3, 8, 8)})
    losses = model.losses(output, torch.tensor([0, 1, 1]))
    losses["loss_likelihood"].backward()
    assert projection.weight.grad.abs().sum() > 0


def test_decoder_reuses_original_g30_operators_and_weights():
    model = sidecar("decoder")
    model.eval()
    data = {"image": torch.randn(3, 3, 8, 8)}
    captured = []
    hook = model.base.backbone.encoder.layers[1].register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach()))
    model.base(data)
    hook.remove()
    reference = e1010.g30.EvidenceDecoder(16, 4, hidden_dim=8, num_heads=4, depth=1)
    reference.load_state_dict(model.auxiliary.state_dict())
    expected_logits = reference(captured[0][:, 1:])
    actual = model(data)["e1010"]["evidence_log_odds"]
    assert torch.equal(actual, expected_logits[..., 1] - expected_logits[..., 0])


def test_max_all_supervision_difference_and_mil_duplicate_controls():
    labels = torch.tensor([0, 1])
    z = torch.tensor([[2., -2., 1., -1.], [2., -2., 1., -1.]])
    output = {"e1010": {"evidence_log_odds": z, "global_logits": torch.zeros(2, 2),
                          "selected_evidence_index": z.sigmoid().argmax(1),
                          "projected_patches": torch.zeros(2, 4, 16)}}
    plain_max = sidecar(supervision="max", diversity_weight=0).losses(output, labels)["loss_evidence"]
    plain_all = sidecar(supervision="all", diversity_weight=0).losses(output, labels)["loss_evidence"]
    assert not torch.equal(plain_max, plain_all)
    mil_max = sidecar(supervision="max", asymmetric=True, diversity_weight=0).losses(output, labels)["loss_evidence"]
    mil_all = sidecar(supervision="all", asymmetric=True, diversity_weight=0).losses(output, labels)["loss_evidence"]
    assert torch.equal(mil_max, mil_all)


@pytest.mark.parametrize("family", ["lfeq", "aux", "decoder"])
def test_max_supervision_preserves_original_probability_tie_selection(family):
    model = sidecar(family, diversity_weight=0)
    logits = torch.tensor([[[0., 20.], [0., 30.]]])
    selected = logits.softmax(-1)[..., 1].argmax(1)
    z = (logits[..., 1] - logits[..., 0]).requires_grad_()
    values = {"evidence_log_odds": z, "selected_evidence_index": selected,
              "global_logits": torch.zeros(1, 2), "aux_decision_logits": torch.zeros(1, 2),
              "projected_patches": torch.zeros(1, 4, 16)}
    loss = model.losses({"e1010": values}, torch.tensor([0]))["loss_evidence"]
    expected = F.cross_entropy(logits[:, 0], torch.tensor([0]))
    assert torch.equal(loss, expected)
    loss.backward()
    assert z.grad[0, 0] > 0 and z.grad[0, 1] == 0


@pytest.mark.parametrize("family", ["aux", "decoder"])
def test_forward_selects_first_token_when_fake_probabilities_saturate(family):
    model = sidecar(family, num_tokens=2, diversity_weight=0)
    heads = model.auxiliary.evidence_heads.heads if family == "aux" else model.auxiliary.heads
    with torch.no_grad():
        for head, margin in zip(heads, (20., 30.)):
            head.weight.zero_()
            head.bias.copy_(torch.tensor([0., margin]))
    output = model({"image": torch.randn(1, 3, 8, 8)})["e1010"]
    assert output["selected_evidence_index"].item() == 0


def test_aux_family_does_not_validate_unused_decoder_divisibility():
    model = e1010.FrozenForgerySidecar(
        TinyB0("eager"), family="aux", memory_layer=1, hidden_dim=15, num_heads=4)
    assert torch.isfinite(model({"image": torch.randn(1, 3, 8, 8)})["prob"]).all()


@pytest.mark.parametrize("settings", [
    {"family": "invalid"}, {"likelihood": "invalid"}, {"likelihood_weight": float("nan")},
    {"likelihood_temperature": 0}, {"contrastive_temperature": float("inf")},
    {"max_contrastive_patches": 0}, {"gate_width": 1}, {"aux_max_weight": .6},
])
def test_invalid_settings_rejected(settings):
    family = settings.pop("family", "aux")
    with pytest.raises(ValueError):
        sidecar(family, **settings)

"""G30 contracts on a real tiny CLIP: the sidecar cannot alter B0."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

import test_g25
from test_g26 import configure_lora_backend


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g30_sidecar_test", ROOT / "training/detectors/g30_sidecar.py")
g30 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g30)


class TinyB0(nn.Module):
    def __init__(self, implementation):
        super().__init__()
        self.backbone = test_g25.tiny_vision(lora=True, implementation=implementation)
        self.head = nn.Linear(16, 2)

    def forward(self, data, inference=False):
        features = self.backbone(data["image"]).pooler_output
        logits = self.head(features)
        return {"cls": logits, "prob": logits.softmax(-1)[:, 1], "feat": features}


@pytest.mark.parametrize("implementation", ["eager", "sdpa"])
@pytest.mark.parametrize("tokens,layer", [(1, 0), (4, 1), (8, 2)])
@pytest.mark.parametrize("backend", ["loralib", "custom"])
def test_training_and_query_perturbation_preserve_b0_bitwise(implementation, tokens, layer, backend, monkeypatch):
    configure_lora_backend(monkeypatch, backend)
    base = TinyB0(implementation).eval()
    with torch.no_grad():
        for name, parameter in base.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.01)  # emulate a fitted B0, not zero LoRA
    data = {"image": torch.randn(4, 3, 8, 8), "label": torch.tensor([0, 1, 0, 1])}
    with torch.no_grad():
        original = base(data, inference=True)
    state = {key: value.clone() for key, value in base.state_dict().items()}
    lengths = []
    handle = base.backbone.encoder.layers[-1].register_forward_pre_hook(
        lambda module, args: lengths.append(args[0].shape[1]))
    model = g30.FrozenEvidenceSidecar(base, num_tokens=tokens, memory_layer=layer,
                                    hidden_dim=16, num_heads=4, depth=1)
    optimizer = torch.optim.AdamW(model.auxiliary.parameters(), lr=.02, weight_decay=.1)
    query_before = model.auxiliary.queries.detach().clone()
    for _ in range(3):
        model.train()
        assert not base.training
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        assert torch.equal(output["g30"]["cls_prob"], original["prob"])
        model.losses(output, data["label"])["overall"].backward()
        assert all(p.grad is None and not p.requires_grad for p in base.parameters())
        optimizer.step()
    assert not torch.equal(query_before, model.auxiliary.queries)
    with torch.no_grad():
        model.auxiliary.queries.mul_(100)
    output = model(data)
    disabled = model(data, auxiliary_enabled=False)
    assert torch.equal(output["g30"]["global_logits"], original["cls"])
    for name in ("cls", "prob", "feat"):
        assert torch.equal(disabled[name], original[name])
    assert all(torch.equal(value, state[key]) for key, value in base.state_dict().items())
    assert set(lengths) == {5}  # original CLS + four patches, no auxiliary queries
    handle.remove()


def test_checkpoint_is_auxiliary_only_and_rejects_a_different_base(tmp_path):
    base = TinyB0("eager")
    model = g30.FrozenEvidenceSidecar(base, num_tokens=4, memory_layer=1,
                                    hidden_dim=16, num_heads=4, depth=1)
    images = {"image": torch.randn(2, 3, 8, 8)}
    original = model(images)["prob"].detach()
    artifact = model.checkpoint("a" * 64)
    assert set(artifact["state_dict"]) == set(model.auxiliary.state_dict())
    assert not any(key.startswith(("base.", "backbone.", "head.")) for key in artifact["state_dict"])
    path = tmp_path / "auxiliary.pth"
    torch.save(artifact, path)
    loaded = torch.load(path, weights_only=True)
    with torch.no_grad():
        model.auxiliary.queries.add_(2)
    model.load_checkpoint(loaded, "a" * 64)
    assert torch.equal(model(images)["prob"], original)
    before = {k: v.clone() for k, v in model.auxiliary.state_dict().items()}
    with pytest.raises(ValueError, match="B0"):
        model.load_checkpoint(loaded, "b" * 64)
    assert all(torch.equal(v, before[k]) for k, v in model.auxiliary.state_dict().items())
    corrupt = {**loaded, "state_dict": {**loaded["state_dict"], "queries": torch.full_like(loaded["state_dict"]["queries"], float("nan"))}}
    with pytest.raises(ValueError, match="finite"):
        model.load_checkpoint(corrupt, "a" * 64)
    assert all(torch.equal(v, before[k]) for k, v in model.auxiliary.state_dict().items())


def test_decoder_and_decision_token_detach_their_inputs():
    decoder = g30.EvidenceDecoder(16, 4, hidden_dim=16, num_heads=4, depth=1)
    memory = torch.randn(3, 5, 16, requires_grad=True)
    decoder(memory).square().mean().backward()
    assert memory.grad is None
    assert decoder.queries.grad.abs().sum() > 0
    decision = g30.DecisionToken(11, hidden_dim=16, num_heads=4)
    features = torch.randn(3, 11, requires_grad=True)
    decision(features).square().mean().backward()
    assert features.grad is None
    assert decision.query.grad.abs().sum() > 0


def test_full_auxiliary_losses_train_decoder_but_never_b0():
    base = TinyB0("eager")
    model = g30.FrozenEvidenceSidecar(base, num_tokens=4, memory_layer=1,
                                    hidden_dim=16, num_heads=4, depth=1,
                                    balance_weight=.1, hard_weighting=True, consistency_weight=.1)
    data = {"image": torch.randn(4, 3, 8, 8)}
    first, view = model(data), model({"image": data["image"] * .9})
    losses = model.losses(first, torch.tensor([0, 1, 0, 1]), view)
    assert losses["loss_consistency"] > 0
    losses["overall"].backward()
    assert model.auxiliary.memory_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in base.parameters())
    with pytest.raises(ValueError, match="view"):
        model.losses(first, torch.tensor([0, 1, 0, 1]))

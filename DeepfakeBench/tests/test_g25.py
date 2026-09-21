"""CPU contracts using a real randomly initialized tiny CLIP, no model download."""

import importlib.util
from pathlib import Path
import sys
import types

import loralib
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer


ROOT = Path(__file__).resolve().parents[1]
DETECTORS = ROOT / "training" / "detectors"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tokens = load_file("g25_tokens_test", DETECTORS / "g25_tokens.py")


def tiny_vision(lora=False, implementation="eager"):
    torch.manual_seed(42)
    config = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                              num_attention_heads=4, image_size=8, patch_size=4,
                              attention_dropout=0.0)
    config._attn_implementation = implementation
    vision = CLIPVisionTransformer(config)
    for parameter in vision.parameters():
        parameter.requires_grad_(False)
    if lora:
        for layer in vision.encoder.layers:
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                old = getattr(layer.self_attn, name)
                new = loralib.Linear(16, 16, r=4, lora_alpha=16, merge_weights=False)
                with torch.no_grad():
                    new.weight.copy_(old.weight)
                    new.bias.copy_(old.bias)
                new.bias.requires_grad_(False)
                setattr(layer.self_attn, name, new)
    return vision


@pytest.mark.parametrize("mode", tokens.MASK_MODES)
def test_attention_mask_direction_and_actual_weights(mode):
    hidden = torch.zeros(2, 8, 16)
    mask = tokens.make_attention_mask(hidden, 5, mode)
    cls_sees, patch_sees = tokens.MASK_MODES[mode]
    assert mask.shape == (2, 1, 8, 8)
    assert (mask[:, :, :, :5] == 0).all()
    assert (mask[:, :, 5:, :] == 0).all()
    assert bool((mask[:, :, :1, 5:] == 0).all()) == cls_sees
    assert bool((mask[:, :, 1:5, 5:] == 0).all()) == patch_sees
    vision = tiny_vision()
    result = vision.encoder.layers[0](torch.randn(2, 8, 16), mask, None, True)
    weights = result[1]
    if not cls_sees:
        assert torch.count_nonzero(weights[:, :, :1, 5:]) == 0
    if not patch_sees:
        assert torch.count_nonzero(weights[:, :, 1:5, 5:]) == 0


def test_read_only_preserves_original_outputs_and_blocks_evidence_perturbation():
    vision = tiny_vision().eval()
    images = torch.randn(2, 3, 8, 8)
    evidence = torch.randn(1, 3, 16)
    with torch.no_grad():
        baseline = vision(images)
        first = tokens.forward_with_evidence(vision, images, evidence, 1, "read_only")
        changed = tokens.forward_with_evidence(vision, images, evidence * 30, 1, "read_only")
        open_mask = tokens.forward_with_evidence(vision, images, evidence, 1, "full")
    torch.testing.assert_close(first["last_hidden_state"][:, :5], baseline.last_hidden_state)
    torch.testing.assert_close(first["features"][:, 0], baseline.pooler_output)
    torch.testing.assert_close(first["last_hidden_state"][:, :5], changed["last_hidden_state"][:, :5])
    assert not torch.allclose(open_mask["last_hidden_state"][:, :5], baseline.last_hidden_state)
    assert first["attention_maps"].shape == (2, 3, 4)


def test_legacy_sdpa_keeps_masks_and_returns_differentiable_attention():
    vision = tiny_vision(implementation="sdpa").eval()
    evidence = nn.Parameter(torch.randn(1, 3, 16) * 0.02)
    images = torch.randn(2, 3, 8, 8)
    baseline = vision(images).last_hidden_state
    output = tokens.forward_with_evidence(vision, images, evidence, 1, "read_only")
    torch.testing.assert_close(output["last_hidden_state"][:, :5], baseline)
    output["attention_maps"].square().sum().backward()
    assert evidence.grad.abs().sum() > 0


@pytest.mark.parametrize("supervision", ["max", "all"])
def test_cls_evidence_and_early_late_lora_receive_gradients(supervision):
    vision = tiny_vision(lora=True)
    vision.embeddings.class_embedding.requires_grad_(True)
    evidence = nn.Parameter(torch.randn(1, 3, 16) * 0.02)
    heads = tokens.EvidenceHeads(16, 3)
    global_head = nn.Linear(16, 2)
    encoded = tokens.forward_with_evidence(vision, torch.randn(2, 3, 8, 8), evidence, 1, "read_only")
    features = encoded["features"]
    output = tokens.score_tokens(global_head(features[:, 0]), heads(features[:, 1:]), 0.5)
    output["attention_maps"] = encoded["attention_maps"]
    losses = tokens.token_losses(output, torch.tensor([0, 1]), supervision, 1.0, 0.01)
    losses["overall"].backward()
    assert evidence.grad.abs().sum() > 0
    assert vision.embeddings.class_embedding.grad.abs().sum() > 0
    for layer in (vision.encoder.layers[0], vision.encoder.layers[-1]):
        assert layer.self_attn.v_proj.lora_B.grad.abs().sum() > 0
        # lora_B initializes at zero, so lora_A can have zero first-step gradient.
        assert layer.self_attn.v_proj.lora_A.grad is not None
        assert layer.self_attn.v_proj.weight.grad is None
    assert vision.embeddings.patch_embedding.weight.grad is None
    assert vision.embeddings.position_embedding.weight.grad is None
    assert len({head.weight.data_ptr() for head in heads.heads}) == 3
    if supervision == "all":
        assert all(head.weight.grad.abs().sum() > 0 for head in heads.heads)


def test_max_all_supervision_and_selection_are_exact():
    global_logits = torch.tensor([[1.0, -1.0], [-0.5, 0.5]], requires_grad=True)
    logits = torch.tensor([[[3., -3.], [-1., 1.], [1., -1.]],
                           [[2., -2.], [-2., 2.], [0., 0.]]], requires_grad=True)
    labels = torch.tensor([0, 1])
    result = tokens.score_tokens(global_logits, logits, 0.5)
    assert result["selected_evidence_index"].tolist() == [1, 1]
    for supervision in ("max", "all"):
        loss = tokens.token_losses(result, labels, supervision, 1., 0.)
        expected = (F.cross_entropy(logits[:, 1], labels) if supervision == "max" else
                    sum(F.cross_entropy(logits[:, i], labels) for i in range(3)) / 3)
        torch.testing.assert_close(loss["loss_evidence"], expected)
        grad = torch.autograd.grad(loss["overall"], logits, retain_graph=True)[0]
        assert bool(grad[:, 0].abs().sum() > 0) == (supervision == "all")
        assert torch.isfinite(loss["overall"])
    expected_score = 0.5 * global_logits.softmax(-1) + 0.5 * logits[:, 1].softmax(-1)
    torch.testing.assert_close(result["fused_probs"], expected_score)


@pytest.fixture
def detector_class(monkeypatch):
    # Stub only the application shell (dataset/metrics imports and pretrained
    # loading). The G25 detector and actual CLIP encoder run unchanged.
    package = types.ModuleType("g25_test_package")
    package.__path__ = []
    base_module = types.ModuleType("g25_test_package.effort_detector")

    class Base(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.backbone = tiny_vision(lora=True)
            self.head = nn.Linear(16, 2)

        def _prep_input(self, images):
            return images

        def get_losses(self, data, prediction):
            return {"overall": F.cross_entropy(prediction["cls"], data["label"])}

    base_module.EffortDetector = Base
    registry = types.ModuleType("detectors")
    registry.DETECTOR = types.SimpleNamespace(register_module=lambda **kwargs: lambda cls: cls)
    monkeypatch.setitem(sys.modules, "detectors", registry)
    monkeypatch.setitem(sys.modules, "g25_test_package", package)
    monkeypatch.setitem(sys.modules, base_module.__name__, base_module)
    monkeypatch.setitem(sys.modules, "g25_test_package.g25_tokens", tokens)
    module = load_file("g25_test_package.effort_detector_g25", DETECTORS / "effort_detector_g25.py")
    return module.EffortDetectorG25


def test_detector_checkpoint_reload_validation_loss_and_multicrop(detector_class):
    config = {"g25_insert_layer": 1, "g25_num_tokens": 3, "g25_supervision": "all"}
    model = detector_class(config).eval()
    images = torch.randn(2, 3, 3, 8, 8)
    with torch.no_grad():
        flat = model({"image": images.reshape(-1, 3, 8, 8)}, inference=True)
        multi = model({"image": images}, inference=True)
        per_crop = flat["prob"].reshape(2, 3)
        chosen = (per_crop - 0.5).abs().argmax(1)
        torch.testing.assert_close(multi["prob"], per_crop[torch.arange(2), chosen])
        torch.testing.assert_close(flat["cls"].softmax(-1)[:, 1], flat["prob"])
        losses = model.get_losses({"label": torch.tensor([0, 1, 0, 1, 0, 1])}, flat)
        assert torch.isfinite(losses["overall"])
        restored = detector_class(config).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored({"image": images}, inference=True)["prob"], multi["prob"])


def test_cls_only_control_preserves_pretrained_cls_and_has_no_extra_params(detector_class):
    baseline = tiny_vision()
    model = detector_class({"g25_num_tokens": 0})
    torch.testing.assert_close(model.backbone.embeddings.class_embedding, baseline.embeddings.class_embedding)
    assert model.backbone.embeddings.class_embedding.requires_grad
    assert not any("evidence" in name for name, _ in model.named_parameters())
    data = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}
    model.get_losses(data, model(data))["overall"].backward()
    assert model.backbone.embeddings.class_embedding.grad.abs().sum() > 0


@pytest.mark.parametrize("change", [{"g25_num_tokens": -1}, {"g25_insert_layer": 3},
                                    {"g25_attention_mode": "bad"}, {"g25_supervision": "bad"},
                                    {"g25_fusion_weight": 1.1}, {"g25_diversity_weight": -1}])
def test_invalid_configs_fail_closed(detector_class, change):
    with pytest.raises(ValueError):
        detector_class({"g25_insert_layer": 1, **change})

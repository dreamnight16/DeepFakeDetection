"""E1002 matrix updates preserve a fitted B0 and obey their spectral spaces."""

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from test_g25 import load_file
from test_g26 import configure_lora_backend
from test_g30 import TinyB0


ROOT = Path(__file__).resolve().parents[1]
KINDS = ("lora", "svd_tail", "svft")


def matrix_module():
    path = ROOT / "training/detectors/e1002_matrix.py"
    assert path.is_file(), "E1002 matrix adaptation is not implemented"
    return load_file("e1002_matrix_test", path)


def fitted_base(monkeypatch, backend, implementation="eager"):
    configure_lora_backend(monkeypatch, backend)
    base = TinyB0(implementation).eval()
    with torch.no_grad():
        for name, parameter in base.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.08)
    return base


def effective_weight(linear):
    """Independent expectation for the two actual B0 LoRA layouts."""
    weight = linear.weight.detach()
    if hasattr(linear, "lora_A") and not getattr(linear, "merged", False):
        if isinstance(linear, torch.nn.Linear):  # loralib subclasses nn.Linear
            weight = weight + (linear.lora_B @ linear.lora_A) * linear.scaling
        else:
            weight = weight + (linear.lora_A @ linear.lora_B).T * linear.scaling
    return weight


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("implementation", ("eager", "sdpa"))
@pytest.mark.parametrize("backend", ("loralib", "custom"))
def test_fitted_b0_is_bitwise_equal_at_start_and_immutable_after_training(
        monkeypatch, kind, implementation, backend):
    base = fitted_base(monkeypatch, backend, implementation)
    data = {"image": torch.randn(4, 3, 8, 8), "label": torch.tensor([0, 1, 0, 1])}
    original_state = {key: value.clone() for key, value in base.state_dict().items()}
    original_grad_flags = {key: parameter.requires_grad for key, parameter in base.named_parameters()}
    with torch.no_grad():
        original = base(data, inference=True)
    model = matrix_module().MatrixAdaptation(base, kind, rank=4, last_layers=2).eval()
    with torch.no_grad():
        initial = model(data, inference=True)
    for key in ("cls", "prob", "feat"):
        assert torch.equal(initial[key], original[key]), key
    residuals = [projection for projection in model.modules() if hasattr(projection, "update_weight")]
    assert all(torch.count_nonzero(projection.update_weight()) == 0 for projection in residuals)
    trainable = list(model.trainable_parameters())
    assert trainable and all(parameter.requires_grad for parameter in trainable)
    assert {id(p) for p in trainable} == {id(p) for p in model.parameters() if p.requires_grad}
    assert not {p.data_ptr() for p in model.parameters()} & {p.data_ptr() for p in base.parameters()}
    before = [parameter.detach().clone() for parameter in trainable]
    copied_head_before = model.model.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(trainable, lr=.02, weight_decay=.1)
    for _ in range(3):
        model.train()
        assert not base.training
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.cross_entropy(output["cls"], data["label"])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in base.parameters())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step()
    assert any(not torch.equal(old, parameter) for old, parameter in zip(before, trainable))
    assert not torch.equal(model.model.head.weight, copied_head_before)
    assert all(projection.update_weight().abs().sum() > 0 for projection in residuals)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all(torch.equal(value, original_state[key]) for key, value in base.state_dict().items())
    assert original_grad_flags == {key: p.requires_grad for key, p in base.named_parameters()}
    for key in ("cls", "prob", "feat"):
        assert torch.equal(base(data)[key], original[key])
    first = model.model.backbone.encoder.layers[0].self_attn.q_proj
    assert not hasattr(first, "update_weight")  # only the selected suffix changes
    for layer in model.model.backbone.encoder.layers[-2:]:
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            projection = getattr(layer.self_attn, name)
            assert all(not p.requires_grad for p in projection.original.parameters())


@pytest.mark.parametrize("backend", ("loralib", "custom"))
def test_svd_tail_preserves_the_fitted_effective_leading_singular_spaces(monkeypatch, backend):
    base = fitted_base(monkeypatch, backend)
    matrix = matrix_module().MatrixAdaptation(base, "svd_tail", rank=4, last_layers=1, preserve_top=32)
    original = base.backbone.encoder.layers[-1].self_attn.q_proj
    left, _, right_t = torch.linalg.svd(effective_weight(original), full_matrices=False)
    adapted = matrix.model.backbone.encoder.layers[-1].self_attn.q_proj
    # On a tiny dim16 backbone top32 clamps to top12, leaving rank4 room.
    assert adapted.preserve_top == 12
    with torch.no_grad():
        for parameter in adapted.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=.2)
    update = adapted.update_weight()
    assert update.abs().sum() > 0
    torch.testing.assert_close(left[:, :12].T @ update, torch.zeros(12, 16), atol=2e-6, rtol=0)
    torch.testing.assert_close(update @ right_t[:12].T, torch.zeros(16, 12), atol=2e-6, rtol=0)
    assert torch.linalg.matrix_rank(update, atol=1e-5) <= 4
    inputs = torch.randn(2, 5, 16)
    torch.testing.assert_close(adapted(inputs) - adapted.original(inputs), F.linear(inputs, update), atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("backend", ("loralib", "custom"))
def test_svft_updates_only_selected_fixed_singular_outer_products(monkeypatch, backend):
    base = fitted_base(monkeypatch, backend)
    matrix = matrix_module().MatrixAdaptation(base, "svft", rank=3, last_layers=1)
    original = base.backbone.encoder.layers[-1].self_attn.q_proj
    left, _, right_t = torch.linalg.svd(effective_weight(original), full_matrices=False)
    adapted = matrix.model.backbone.encoder.layers[-1].self_attn.q_proj
    with torch.no_grad():
        for parameter in adapted.parameters():
            if parameter.requires_grad:
                parameter.normal_(std=.2)
    update = adapted.update_weight()
    coefficients = left.T @ update @ right_t.T
    rows, columns = torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij")
    selected = (columns - rows).remainder(16) < 3
    assert coefficients[selected].abs().sum() > 0
    torch.testing.assert_close(coefficients[~selected], torch.zeros_like(coefficients[~selected]), atol=2e-6, rtol=0)
    inputs = torch.randn(2, 5, 16)
    torch.testing.assert_close(adapted(inputs) - adapted.original(inputs), F.linear(inputs, update), atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_trained_matrix_state_reloads_strictly(monkeypatch, kind, tmp_path):
    base = fitted_base(monkeypatch, "loralib")
    module = matrix_module()
    model = module.MatrixAdaptation(base, kind, rank=4, last_layers=2)
    optimizer = torch.optim.SGD(model.trainable_parameters(), lr=.1)
    data = {"image": torch.randn(4, 3, 8, 8)}
    F.cross_entropy(model(data)["cls"], torch.tensor([0, 1, 0, 1])).backward()
    optimizer.step()
    model.eval()
    expected = model(data, inference=True)
    path = tmp_path / "matrix.pth"
    torch.save(model.state_dict(), path)
    restored = module.MatrixAdaptation(base, kind, rank=4, last_layers=2).eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for key in ("cls", "prob", "feat"):
        assert torch.equal(restored(data, inference=True)[key], expected[key])
    assert restored.settings == model.settings


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("backend", ("loralib", "custom"))
def test_existing_merge_state_keeps_the_original_forward(monkeypatch, kind, backend):
    base = fitted_base(monkeypatch, backend)
    for module in base.modules():
        if hasattr(module, "merge_weights"):
            module.merge_weights = True
    base.train().eval()  # loralib now contains its fitted merged matrix
    data = {"image": torch.randn(3, 3, 8, 8)}
    original = base(data)
    state = {key: value.clone() for key, value in base.state_dict().items()}
    adapted = matrix_module().MatrixAdaptation(base, kind, rank=4, last_layers=2)
    for mode in (False, True, False):
        adapted.train(mode)
        for key in ("cls", "prob", "feat"):
            assert torch.equal(adapted(data)[key], original[key])
    assert all(torch.equal(value, state[key]) for key, value in base.state_dict().items())


@pytest.mark.parametrize("change", [
    {"kind": "unknown"}, {"rank": 0}, {"rank": -1}, {"rank": 17}, {"rank": 1.5},
    {"last_layers": 0}, {"last_layers": 4}, {"last_layers": 1.5},
    {"preserve_top": -1}, {"preserve_top": 1.5},
])
def test_invalid_matrix_settings_fail_before_touching_source(monkeypatch, change):
    base = fitted_base(monkeypatch, "loralib")
    state = {key: value.clone() for key, value in base.state_dict().items()}
    settings = {"kind": "lora", "rank": 4, "last_layers": 2, **change}
    with pytest.raises(ValueError):
        matrix_module().MatrixAdaptation(base, **settings)
    assert all(torch.equal(value, state[key]) for key, value in base.state_dict().items())

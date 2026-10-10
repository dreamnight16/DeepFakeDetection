"""G4: immutable B0, pixel attribution and train-only global prototypes."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from test_g30 import TinyB0


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "experiments/e1010_prototypes.py"


@pytest.fixture
def core():
    assert CORE_PATH.is_file(), "G4 prototype core has not been implemented"
    spec = importlib.util.spec_from_file_location("e1010_prototypes_test", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PatchBlock(nn.Module):
    def forward(self, hidden):
        return (hidden,)


class PixelB0(nn.Module):
    """Exact arithmetic attribution with CLIP's row-major patch layout."""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.embeddings = nn.Module()
        self.backbone.embeddings.patch_embedding = nn.Conv2d(3, 3, 2, stride=2, bias=False)
        self.backbone.encoder = nn.Module()
        self.backbone.encoder.layers = nn.ModuleList([PatchBlock()])
        self.anchor = nn.Parameter(torch.ones(()))
        self.seen = []

    def forward(self, data, inference=False):
        images = data["image"]
        self.seen.append(images.detach().clone())
        patches = F.avg_pool2d(images, 2, 2).flatten(2).transpose(1, 2)
        hidden = torch.cat((patches.mean(1, keepdim=True), patches), 1)
        hidden = self.backbone.encoder.layers[-1](hidden)[0]
        weights = images.new_tensor([1., -1., 2., .5])
        margin = (hidden[:, 1:, 0] * weights).sum(1) * self.anchor
        offset = images.mean((1, 2, 3)) * 3
        logits = torch.stack((offset - margin / 2, offset + margin / 2), 1)
        return {"cls": logits, "prob": logits.softmax(-1)[:, 1], "feat": hidden[:, 0]}


def image(values):
    return torch.tensor(values, dtype=torch.float64).reshape(1, 1, 2, 2).repeat_interleave(2, -1).repeat_interleave(2, -2).repeat(1, 3, 1, 1)


def batches():
    images = torch.cat((image([1, 2, 4, 8]), image([3, 5, 7, 11]),
                        image([2, 1, 3, 4]), image([4, 6, 8, 12])))
    return [{"image": images, "label": torch.tensor([0, 1, 0, 1]),
             "path": ["r0", "f0", "r1", "f1"]}]


def test_capture_last_block_keeps_real_clip_scores_parameters_and_gradient(core):
    torch.manual_seed(8)
    base = TinyB0("eager").eval()
    images = torch.randn(2, 3, 8, 8, requires_grad=True)
    state = {key: value.clone() for key, value in base.state_dict().items()}
    with torch.no_grad():
        expected = base({"image": images}, inference=True)
        patches = base.backbone(images).last_hidden_state[:, 1:]
    actual, captured = core.capture_patches(base, images)
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    assert torch.equal(captured, patches)
    assert not captured.requires_grad and not actual["cls"].requires_grad
    assert not base.training and all(parameter.grad is None for parameter in base.parameters())
    assert all(torch.equal(value, state[key]) for key, value in base.state_dict().items())
    assert len(base.backbone.encoder.layers[-1]._forward_hooks) == 0


def test_occlusion_is_pixel_mean_margin_row_major_and_memory_bounded(core):
    base = PixelB0().double()
    images = torch.cat((image([1, 2, 4, 8]), image([2, 4, 6, 9])))
    original, patches, contributions = core.occlusion_contributions(base, images, occlusion_batch_size=3)
    assert contributions.shape == (2, 4)
    assert patches.shape == (2, 4, 3)
    original_margin = original["cls"][:, 1] - original["cls"][:, 0]
    expected = []
    for frame in range(2):
        margins = []
        for index in range(4):
            masked = images[frame:frame + 1].clone()
            row, column = divmod(index, 2)
            masked[:, :, row * 2:(row + 1) * 2, column * 2:(column + 1) * 2] = images[frame:frame + 1].mean((-2, -1), keepdim=True)
            values = base({"image": masked})["cls"]
            margins.append(values[:, 1] - values[:, 0])
        expected.append(original_margin[frame] - torch.cat(margins))
    torch.testing.assert_close(contributions, torch.stack(expected), rtol=0, atol=1e-12)
    # First call is original B=2; the next three are bounded 3/3/2 occluded images.
    assert [len(value) for value in base.seen[:4]] == [2, 3, 3, 2]
    variants = torch.cat(base.seen[1:4])
    for flat_index, variant in enumerate(variants):
        frame, patch = divmod(flat_index, 4)
        row, column = divmod(patch, 2)
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[row * 2:(row + 1) * 2, column * 2:(column + 1) * 2] = True
        assert torch.equal(variant[:, ~mask], images[frame, :, ~mask])
        assert torch.equal(variant[:, mask], images[frame].mean((-2, -1))[:, None].expand(-1, 4))
    assert all(parameter.grad is None for parameter in base.parameters())


def test_real_clip_chunked_attribution_matches_single_patch_reference(core):
    base = TinyB0("eager").eval()
    images = torch.randn(2, 3, 8, 8)
    state = {key: value.clone() for key, value in base.state_dict().items()}
    original, patches, chunked = core.occlusion_contributions(base, images, 3)
    _, _, reference = core.occlusion_contributions(base, images, 1)
    torch.testing.assert_close(chunked, reference, rtol=1e-5, atol=2e-7)
    with torch.no_grad():
        assert torch.equal(original["prob"], base({"image": images})["prob"])
    assert patches.shape == (2, 4, 16)
    assert all(torch.equal(value, state[key]) for key, value in base.state_dict().items())


def test_prototypes_use_all_real_same_k_fake_and_actual_top_contributions(core):
    base = PixelB0().double()
    data = batches()
    artifacts, audit = core.build_prototypes(base, data, ["ALL", "RANDOM", "TOPK"], top_k=2,
                                            max_frames_per_class=2, seed=17, occlusion_batch_size=3)
    _, again = core.build_prototypes(PixelB0().double(), data, ["ALL", "RANDOM", "TOPK"], top_k=2,
                                     max_frames_per_class=2, seed=17, occlusion_batch_size=2)
    _, patches = core.capture_patches(base, data[0]["image"])
    labels = data[0]["label"].numpy()
    for method, artifact in artifacts.items():
        assert artifact["method"] == method
        assert artifact["real_prototype"].device.type == "cpu"
        torch.testing.assert_close(artifact["real_prototype"], patches[labels == 0].float().mean((0, 1)))
        assert artifact["counts"]["real_patches"] == 8
        assert artifact["counts"]["fake_patches"] == (8 if method == "ALL" else 4)
        assert np.array_equal(audit[method]["path"], ["r0", "f0", "r1", "f1"])
        assert np.array_equal(audit[method]["selected_index"], again[method]["selected_index"])
        fake = []
        for frame in np.flatnonzero(labels == 1):
            selection = audit[method]["selected_index"][frame]
            selection = selection[selection >= 0]
            assert len(set(selection)) == (4 if method == "ALL" else 2)
            fake.append(patches[frame, selection])
            if method == "TOPK":
                expected = np.argsort(-audit[method]["contributions"][frame], kind="stable")[:2]
                assert np.array_equal(selection, expected)
        torch.testing.assert_close(artifact["fake_prototype"], torch.cat(fake).float().mean(0))
        assert not artifact["real_prototype"].requires_grad


def test_all_random_skip_attribution_and_prototype_settings_validate(core, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ALL/RANDOM do not require occlusion")
    monkeypatch.setattr(core, "occlusion_contributions", forbidden)
    artifacts, audit = core.build_prototypes(PixelB0().double(), batches(), ["ALL", "RANDOM"], 2, 2)
    for method, artifact in artifacts.items():
        assert np.isnan(audit[method]["contributions"]).all()
        artifact["base_sha256"] = "a" * 64
        core.validate_artifact(artifact, "a" * 64, 4, 3, method, artifact["settings"])
        with pytest.raises(ValueError, match="B0"):
            core.validate_artifact(artifact, "b" * 64)
        with pytest.raises(ValueError, match="settings"):
            core.validate_artifact(artifact, expected_settings={})
        with pytest.raises(ValueError, match="method"):
            core.validate_artifact(artifact, expected_method="TOPK")
        corrupt = copy.deepcopy(artifact)
        corrupt["fake_prototype"][0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            core.validate_artifact(corrupt)


def test_negative_contributions_are_selected_and_recorded(core, monkeypatch):
    def negative(base, images, occlusion_batch_size=16, fill="mean"):
        output, patches = core.capture_patches(base, images)
        return output, patches, images.new_tensor([[-4., -3., -2., -1.]]).expand(len(images), -1)
    monkeypatch.setattr(core, "occlusion_contributions", negative)
    artifacts, audit = core.build_prototypes(PixelB0().double(), batches(), ["TOPK"], 2, 2)
    assert (audit["TOPK"]["selected_index"][[1, 3], :2] == [3, 2]).all()
    assert artifacts["TOPK"]["selectionmetadata"]["topk_nonpositive_fraction"] == 1.


def test_same_readout_matches_formula_has_no_label_dependency_and_reloads(core, tmp_path):
    artifacts, _ = core.build_prototypes(PixelB0().double(), batches(), ["ALL"], 2, 2)
    artifact = artifacts["ALL"]
    patches = torch.randn(3, 4, 3, requires_grad=True)
    path = tmp_path / "prototype.pth"
    torch.save(artifact, path)
    loaded = torch.load(path, weights_only=True)
    actual = core.score_prototypes(patches, loaded, temperature=.2, pool_top_k=2)
    z = F.normalize(patches.detach(), dim=-1)
    delta = z @ F.normalize(artifact["fake_prototype"], dim=0) - z @ F.normalize(artifact["real_prototype"], dim=0)
    expected = (delta / .2).sigmoid().topk(2, dim=1).values.mean(1)
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad and actual.shape == (3,)
    # Ground-truth labels only exist outside the scorer and cannot affect it.
    data = {"patches": patches, "labels": torch.tensor([0, 0, 1])}
    first = core.score_prototypes(data["patches"], loaded, .2, 2)
    data["labels"] = 1 - data["labels"]
    assert torch.equal(first, core.score_prototypes(data["patches"], loaded, .2, 2))


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"top_k": 5}, {"max_frames_per_class": 0},
                                  {"selected_methods": ["BAD"]}, {"selected_methods": []},
                                  {"fill": "zero"}, {"occlusion_batch_size": 0}])
def test_invalid_builder_configuration_is_rejected(core, kwargs):
    settings = dict(selected_methods=["ALL"], top_k=2, max_frames_per_class=2)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        core.build_prototypes(PixelB0().double(), batches(), **settings)


def test_builder_rejects_missing_class_budget_duplicate_path_and_bad_labels(core):
    data = batches()
    for change in ({"label": torch.zeros(4, dtype=torch.long)},
                   {"label": torch.tensor([0, 1, 2, 1])},
                   {"path": ["same"] * 4}):
        with pytest.raises(ValueError):
            core.build_prototypes(PixelB0().double(), [{**data[0], **change}], ["ALL"], 2, 4)
    with pytest.raises(ValueError, match="budget"):
        core.build_prototypes(PixelB0().double(), data, ["ALL"], 2, 1)


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_readout_temperature_is_rejected(core, temperature):
    artifacts, _ = core.build_prototypes(PixelB0().double(), batches(), ["ALL"], 2, 2)
    with pytest.raises(ValueError):
        core.score_prototypes(torch.randn(2, 4, 3), artifacts["ALL"], temperature, 2)


def test_shape_and_patch_geometry_mismatches_fail_without_clamping(core):
    base = PixelB0().double()
    with pytest.raises(ValueError, match="RGB"):
        core.capture_patches(base, torch.randn(1, 1, 4, 4))
    with pytest.raises(ValueError, match="grid"):
        core.occlusion_contributions(base, torch.randn(1, 3, 5, 4))
    base.backbone.embeddings.patch_embedding.stride = (1, 1)
    with pytest.raises(ValueError, match="nonoverlap"):
        core.occlusion_contributions(base, torch.randn(1, 3, 4, 4))
    artifacts, _ = core.build_prototypes(PixelB0().double(), batches(), ["ALL"], 2, 2)
    with pytest.raises(ValueError, match="K"):
        core.score_prototypes(torch.randn(2, 4, 3), artifacts["ALL"], pool_top_k=5)
    with pytest.raises(ValueError, match="shape"):
        core.score_prototypes(torch.randn(2, 4, 4), artifacts["ALL"], pool_top_k=2)

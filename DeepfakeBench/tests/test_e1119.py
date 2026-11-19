"""Targeted E1119 operator and protocol checks without model training or data."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "experiments"))

from e1119_model import LoRA, PROJECTIONS, Spectral, paired_forward, selected_indices
from e1119_assets import METHODS, matched_manifest
from e1119_eval import INDEPENDENT, means, video_rows
from e1119_report import auc_resamples
from run_e1119 import ARMS, ready_for, plan
from run_g25 import TEST_DS


def lora(base, rank=2, basis=None, route=None):
    width = base.in_features
    native = SimpleNamespace(lora_A=torch.randn(width, rank, dtype=base.weight.dtype),
                             lora_B=torch.randn(rank, width, dtype=base.weight.dtype), scaling=4)
    return LoRA(base.requires_grad_(False), native, rank, basis=basis, route=route)


def test_full_queue_and_snapshot_dependencies(tmp_path):
    jobs = plan()
    assert len(ARMS) == 23
    assert sum(key.startswith("train:") for key in jobs) == 46
    assert sum(key.startswith("eval:") for key in jobs) == 46
    assert jobs["train:B00:1024"] == []
    assert jobs["train:P01:1024"] == ["asset:activation"]
    assert jobs["train:P02:1025"] == ["asset:random"]
    assert jobs["train:M02:1025"] == ["asset:spectral"]
    key = "D01:B00:1024:epoch_0"
    dependencies = jobs[key]
    active = {dependencies[0]: {"status": "RUNNING", "folder": tmp_path}}
    assert not ready_for(key, dependencies, active)
    (tmp_path / "epoch_0.json").write_text("{}", encoding="utf-8")
    assert ready_for(key, dependencies, active)
    assert not ready_for("D01:B00:1024:best", dependencies, active)


def test_protected_basis_annihilates_update_and_folds():
    width = 8
    basis, _ = torch.linalg.qr(torch.randn(width, 3, dtype=torch.float64))
    model = lora(nn.Linear(width, width, dtype=torch.float64), basis=basis)
    x = torch.randn(4, 3, width, dtype=torch.float64, requires_grad=True)
    y = model(x)
    torch.testing.assert_close(y, nn.functional.linear(x, model.fold(), model.bias))
    torch.testing.assert_close(model.update_weight() @ basis, torch.zeros_like(basis), atol=1e-10, rtol=0)
    y.sum().backward()
    assert x.grad is not None and model.lora_A.grad is not None


@pytest.mark.parametrize("arm", ["M01", "M02", "M04", "M05", "M06"])
def test_spectral_initialization_gradient_and_fold(arm):
    width = 8
    base = nn.Linear(width, width).requires_grad_(False)
    x = torch.randn(4, 3, width)
    u, s, vh = torch.linalg.svd(base.weight)
    spectral = Spectral(base, arm, {"u": u, "s": s, "vh": vh})
    torch.testing.assert_close(spectral(x), base(x), atol=2e-6, rtol=1e-5)
    spectral(x).square().mean().backward()
    assert all(p.grad is not None for p in spectral.parameters() if p.requires_grad)
    with torch.no_grad():
        for p in spectral.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * .2)
    folded = spectral.fold()
    torch.testing.assert_close(spectral(x), nn.functional.linear(x, folded, base.bias))
    if arm == "M05":
        torch.testing.assert_close(folded.T @ folded, base.weight.T @ base.weight)
    if arm == "M02":
        orth, norm = spectral.regularizers()
        eye = torch.eye(width)
        left = torch.cat((u[:, :-1], spectral.u[:, None]), dim=1)
        right = torch.cat((vh[:-1], spectral.v[None]), dim=0)
        expected_orth = (left.T @ left - eye).square().sum() + (right @ right.T - eye).square().sum()
        torch.testing.assert_close(orth, expected_orth)
        torch.testing.assert_close(norm, (folded.square().sum() - base.weight.square().sum()).abs())


def test_detach_changes_only_the_delta_input_gradient():
    base = nn.Linear(4, 4, dtype=torch.float64)
    model = lora(base, route="detach")
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    detached = model(x)
    detached.sum().backward()
    input_grad, a_grad, b_grad = x.grad.clone(), model.lora_A.grad.clone(), model.lora_B.grad.clone()
    x.grad = None
    model.zero_grad()
    model.eval()
    normal = model(x)
    normal.sum().backward()
    torch.testing.assert_close(normal, detached)
    torch.testing.assert_close(input_grad, model.weight.sum(0).expand_as(x))
    torch.testing.assert_close(a_grad, model.lora_A.grad)
    torch.testing.assert_close(b_grad, model.lora_B.grad)
    assert not torch.allclose(x.grad, input_grad)


def test_split_uses_the_previous_base_block_and_keeps_its_input_gradient():
    class Block(nn.Module):
        def __init__(self, split):
            super().__init__()
            self.layer_norm1 = nn.LayerNorm(4)
            self.self_attn = nn.Module()
            for key in PROJECTIONS:
                mod = lora(nn.Linear(4, 4), route="split" if split and key != "out_proj" else None)
                setattr(self.self_attn, key, mod)

        def forward(self, x):
            n = self.layer_norm1(x)
            values = [getattr(self.self_attn, key)(n) for key in PROJECTIONS[:3]]
            return x + self.self_attn.out_proj(sum(values))

    odd, even = Block(False), Block(True)
    x = torch.randn(2, 3, 4, requires_grad=True)
    result = paired_forward((odd, even), (0,), lambda: even(odd(x)))
    from e1119_model import bypass
    with bypass((odd,)):
        u = odd(x)
    v = odd(x)
    actual, reference = even.layer_norm1(v), even.layer_norm1(u)
    deltas = list(even.self_attn.children())[:3]
    expected_values = [nn.functional.linear(actual, mod.weight, mod.bias) + mod.delta(reference) for mod in deltas]
    expected = v + even.self_attn.out_proj(sum(expected_values))
    torch.testing.assert_close(result, expected)
    g = torch.autograd.grad(result.square().sum(), (x, odd.self_attn.q_proj.lora_B), retain_graph=True)
    g_ref = torch.autograd.grad(expected.square().sum(), (x, odd.self_attn.q_proj.lora_B))
    for left, right in zip(g, g_ref):
        torch.testing.assert_close(left, right)


def test_aux_mixture_preserves_budget_and_local_rng():
    labels = torch.tensor([0] * 5 + [1] * 11)
    state = torch.get_rng_state()
    for step in range(2):
        idx = selected_indices(labels, True, torch.Generator().manual_seed(42 + step), step)
        assert len(idx.unique()) == 5
        assert int(labels[idx].sum()) == 2 + step
    assert torch.equal(state, torch.get_rng_state())


def test_matched_basis_budget_and_stratification(tmp_path):
    import json

    metadata, native = {}, []
    for method in ("FF-real", *METHODS):
        entries = {}
        for source in range(11):
            key = f"{source:03}" if method == "FF-real" else f"{source:03}_999"
            frames = [f"images/{method}/{key}/{i}.png" for i in range(12)]
            entries[key] = {"label": method, "frames": frames}
            native.extend(frames[:8])
        metadata[method] = {"train": {"c23": entries}}
    index = tmp_path / "ff.json"
    index.write_text(json.dumps({"FaceForensics++": metadata}), encoding="utf-8")
    rows = matched_manifest(index, native)
    assert len(rows) == 10
    assert sum(row["mixed"] == "real" for row in rows) == 5
    for method in METHODS:
        stratum = [row for row in rows if row["method"] == method]
        assert abs(sum(row["mixed"] == "real" for row in stratum) * 2 - len(stratum)) <= 1
    assert all(len(row["real"]["frames"]) == len(row["fake"]["frames"]) == 8 for row in rows)


def test_full_path_grouping_and_bootstrap_tied_ranks():
    frames = {"video_id": ["real/a", "real/a", "fake/a"], "labels": [0, 0, 1], "cls_prob": [.1, .3, .7]}
    rows = video_rows(frames)
    assert list(rows["video_id"]) == ["real/a", "fake/a"]
    np.testing.assert_allclose(rows["prob_fake"], [.2, .7])
    metrics = {dataset: {"video_auc_fullpath": .8} for dataset in TEST_DS}
    del metrics[INDEPENDENT[0]]
    assert means(metrics)["independent_video_auc_mean_fullpath"] is None
    y, scores = np.array([0, 0, 1, 1]), np.array([.1, .8, .8, .9])
    counts = np.array([[1, 1, 1, 1], [0, 2, 2, 0]])
    expected = [roc_auc_score(y, scores, sample_weight=row) for row in counts]
    np.testing.assert_allclose(auc_resamples(scores, y, counts), expected)

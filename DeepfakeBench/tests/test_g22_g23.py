"""CPU-only contract tests for the isolated G22/G23 query ablations."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "training" / "detectors" / "query_ablation.py"
DETECTOR_PATH = ROOT / "training" / "detectors" / "effort_detector_query_ablation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("query_ablation", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_g22_selects_last_block_input_not_final_output():
    module = _load_module()
    hidden_states = tuple(torch.full((2, 5, 8), float(i)) for i in range(4))

    selected = module.select_vit_patch_tokens(
        {"last_hidden_state": hidden_states[-1], "hidden_states": hidden_states},
        source="last_block_input",
    )

    assert torch.equal(selected, hidden_states[-2][:, 1:, :])
    assert not torch.equal(selected, hidden_states[-1][:, 1:, :])


def test_g22_rejects_missing_hidden_states():
    module = _load_module()
    final = torch.randn(2, 5, 8)

    try:
        module.select_vit_patch_tokens(
            {"last_hidden_state": final}, source="last_block_input"
        )
    except ValueError as exc:
        assert "hidden_states" in str(exc)
    else:
        raise AssertionError("G22 must fail closed when hidden states are unavailable")


def test_g22_detector_reuses_g19_mean_readout():
    source = DETECTOR_PATH.read_text(encoding="utf-8")
    assert "from .token_readout import TokenReadout" in source
    assert 'TokenReadout if self.query_variant == "full"' in source


def test_runner_fail_closed_and_run_isolation_contracts():
    runner = (ROOT / "experiments" / "query_ablation_runner.py").read_text(
        encoding="utf-8"
    )
    utilities = (ROOT / "experiments" / "experiment_utils.py").read_text(
        encoding="utf-8"
    )
    assert "strftime" in runner and "args.run_dir" in runner
    assert 'result["status"] = "EVAL_FAILED"' in runner
    assert "missing_video_auc" in runner
    assert "proc.returncode != 0" in utilities
    assert "testall failed with exit code" in utilities


def test_g23_lite_query_matches_shape_and_is_materially_smaller():
    module = _load_module()
    kwargs = dict(
        vit_dim=32,
        hidden_dim=16,
        num_evidence_tokens=3,
        depth=2,
        num_heads=4,
        dropout=0.0,
        readout_mode="mean",
    )
    full = module.QueryReadout(variant="full", **kwargs)
    lite = module.QueryReadout(variant="cross_only", **kwargs)
    patches = torch.randn(2, 7, 32)

    full_out = full(patches)
    lite_out = lite(patches)

    for out in (full_out, lite_out):
        assert out["logits"].shape == (2, 2)
        assert out["prob"].shape == (2,)
        assert out["queries"].shape == (2, 4, 16)

    full_params = sum(p.numel() for p in full.parameters())
    lite_params = sum(p.numel() for p in lite.parameters())
    assert lite_params < 0.5 * full_params, (lite_params, full_params)


def test_g23_cross_only_has_no_self_attention_or_ffn():
    module = _load_module()
    lite = module.QueryReadout(
        vit_dim=32,
        hidden_dim=16,
        num_evidence_tokens=3,
        depth=2,
        num_heads=4,
        dropout=0.0,
        readout_mode="mean",
        variant="cross_only",
    )

    names = tuple(name for name, _ in lite.named_parameters())
    assert any("cross_attn" in name for name in names)
    assert not any("self_attn" in name for name in names)
    assert not any("ffn" in name for name in names)


def test_patch_mask_blocks_masked_patch_content_and_rejects_all_masked():
    module = _load_module()
    lite = module.QueryReadout(
        vit_dim=16,
        hidden_dim=16,
        num_evidence_tokens=2,
        depth=1,
        num_heads=4,
        dropout=0.0,
        variant="cross_only",
    ).eval()
    patches = torch.randn(1, 4, 16)
    changed = patches.clone()
    changed[:, -1] = 10_000.0
    mask = torch.tensor([[True, True, True, False]])
    with torch.no_grad():
        first = lite(patches, mask)["logits"]
        second = lite(changed, mask)["logits"]
    assert torch.allclose(first, second, atol=1e-6)

    try:
        lite(patches, torch.zeros_like(mask))
    except ValueError as exc:
        assert "at least one valid patch" in str(exc)
    else:
        raise AssertionError("all-masked patch sets must be rejected")


def test_production_shape_parameter_budget_and_backward():
    module = _load_module()
    kwargs = dict(
        vit_dim=1024,
        hidden_dim=256,
        num_evidence_tokens=8,
        depth=2,
        num_heads=8,
        dropout=0.0,
        readout_mode="mean",
    )
    full = module.QueryReadout(variant="full", **kwargs)
    lite = module.QueryReadout(variant="cross_only", **kwargs)

    assert sum(p.numel() for p in full.parameters()) == 2_373_634
    assert sum(p.numel() for p in lite.parameters()) == 794_114

    patches = torch.randn(2, 16, 1024)
    labels = torch.tensor([0, 1])
    output = lite(patches)
    loss = torch.nn.functional.cross_entropy(output["logits"], labels)
    loss.backward()
    for name in (
        "patch_projection.weight",
        "decision_token",
        "evidence_tokens",
        "blocks.0.cross_attn.in_proj_weight",
        "head.weight",
    ):
        parameter = dict(lite.named_parameters())[name]
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


if __name__ == "__main__":
    tests = (
        test_g22_selects_last_block_input_not_final_output,
        test_g22_rejects_missing_hidden_states,
        test_g22_detector_reuses_g19_mean_readout,
        test_runner_fail_closed_and_run_isolation_contracts,
        test_g23_lite_query_matches_shape_and_is_materially_smaller,
        test_g23_cross_only_has_no_self_attention_or_ffn,
        test_patch_mask_blocks_masked_patch_content_and_rejects_all_masked,
        test_production_shape_parameter_budget_and_backward,
    )
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    print("ALL PASS")

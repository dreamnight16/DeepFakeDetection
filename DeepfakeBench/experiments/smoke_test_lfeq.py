"""Local verification for the LFEQ read-out head (G18).

This runs WITHOUT the heavy runtime deps (loralib / tensorboard / sklearn /
datasets), which only exist on the server.  It loads ``detectors/lfeq_module.py``
directly by file path (the module is pure torch) and exercises the novel piece
end-to-end: forward shapes, fusion arithmetic, hard-argmax selection, loss
composition, gradient flow, and the diversity regulariser — plus it static-checks
(``py_compile``) every G18 file for syntax and asserts the config-key names are
consistent between the detector, ``build_config``, and ``arch_keys``.

The detector's own forward can't be instantiated here (its base class pulls the
full training stack); instead we validate the module contract the detector relies
on (all returned keys it indexes) and the 5D TAA aggregation arithmetic inline.

Usage:
    python experiments/smoke_test_lfeq.py           # module + static checks
"""
import importlib.util
import os
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEEPFAKE = os.path.dirname(_HERE)
_MODULE_PATH = os.path.join(_DEEPFAKE, 'training', 'detectors', 'lfeq_module.py')


def load_module():
    spec = importlib.util.spec_from_file_location('lfeq_module', _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_patches(b=4, n=256, d=1024, seed=0):
    torch.manual_seed(seed)
    return torch.randn(b, n, d)


def test_forward_shapes(module, b=4, n=256, d=1024, evi=8, hidden=256, heads=8):
    lfeq = module.LearnableForgeryEvidenceQuery(
        vit_dim=d, hidden_dim=hidden, num_evidence_tokens=evi,
        depth=2, num_heads=heads, fusion_weight=0.5)
    patches = _make_patches(b=b, n=n, d=d, seed=0)
    out = lfeq(patches)
    assert out['global_logits'].shape == (b, 2), out['global_logits'].shape
    assert out['evidence_logits'].shape == (b, evi, 2), out['evidence_logits'].shape
    assert out['selected_evidence_logits'].shape == (b, 2), out['selected_evidence_logits'].shape
    assert out['fused_probs'].shape == (b, 2), out['fused_probs'].shape
    assert out['prediction'].shape == (b,), out['prediction'].shape
    assert out['attention_maps'].shape == (b, evi, n), out['attention_maps'].shape
    assert out['decision_attention'].shape == (b, n)
    assert out['evidence_features'].shape == (b, evi, hidden)
    assert out['global_feature'].shape == (b, hidden), out['global_feature'].shape
    # probs are valid probabilities
    assert torch.all(out['global_probs'] >= 0) and torch.all(out['global_probs'] <= 1)
    assert torch.all(out['evidence_probs'] >= 0) and torch.all(out['evidence_probs'] <= 1)
    assert torch.all(out['fused_probs'] >= 0) and torch.all(out['fused_probs'] <= 1)
    # detector depends on these exact keys:
    for k in ('global_logits', 'evidence_logits', 'selected_evidence_logits',
              'selected_evidence_index', 'global_probs', 'evidence_probs',
              'fused_probs', 'prediction', 'attention_maps', 'evidence_features',
              'global_feature'):
        assert k in out, f"detector relies on key {k} which is missing"
    print("  [ok] forward shapes + key contract")


def test_fusion_arithmetic(module):
    b, n, d = 3, 16, 1024
    for w in (0.0, 0.5, 1.0):
        lfeq = module.LearnableForgeryEvidenceQuery(
            vit_dim=d, hidden_dim=256, num_evidence_tokens=8,
            depth=2, num_heads=8, fusion_weight=w)
        out = lfeq(_make_patches(b=b, n=n, d=d, seed=1))
        # The module fuses with the softmax of the SELECTED evidence logits
        # (it keeps that as a local var, so recompute it from the returned key).
        sel_prob = out['selected_evidence_logits'].softmax(dim=-1)
        fused = w * out['global_probs'] + (1 - w) * sel_prob
        assert torch.allclose(out['fused_probs'], fused, atol=1e-7), f"fusion w={w} mismatch"
    print("  [ok] fusion_weight arithmetic (0 / 0.5 / 1)")


def test_hard_argmax_wiring(module):
    """Selected evidence token must be the max-fake-prob token, and its logits
    must be the ones used in the loss / fusion.  The selection is label-agnostic
    by construction: forward() takes only patches (no label argument)."""
    b, n, d = 3, 16, 1024
    lfeq = module.LearnableForgeryEvidenceQuery(
        vit_dim=d, hidden_dim=128, num_evidence_tokens=6,
        depth=1, num_heads=4)
    out = lfeq(torch.randn(b, n, d))
    expected_idx = out['evidence_probs'][..., 1].argmax(dim=1)          # [B]
    assert torch.equal(out['selected_evidence_index'], expected_idx), \
        "selected index != argmax of evidence fake prob"
    bidx = torch.arange(b)
    expected_logits = out['evidence_logits'][bidx, expected_idx]
    assert torch.allclose(out['selected_evidence_logits'], expected_logits, atol=1e-6), \
        "selected logits != logits of the max-fake token"
    # fused uses the softmax of those selected logits (recompute), for both classes
    w = lfeq.fusion_weight
    sel_prob = out['selected_evidence_logits'].softmax(dim=-1)
    expected_fused = w * out['global_probs'] + (1 - w) * sel_prob
    assert torch.allclose(out['fused_probs'], expected_fused, atol=1e-6)
    print("  [ok] hard-argmax selection wiring + fused-prob arithmetic (label-agnostic)")


def test_loss_composition_and_backward(module):
    b, n, d = 3, 32, 1024
    lfeq = module.LearnableForgeryEvidenceQuery(
        vit_dim=d, hidden_dim=256, num_evidence_tokens=8, depth=2, num_heads=8)
    patches = _make_patches(b=b, n=n, d=d, seed=7)
    labels = torch.tensor([0, 1, 1])
    out = lfeq(patches)
    li = lfeq.compute_loss(out, labels, evidence_weight=1.0, diversity_weight=0.01)
    assert torch.isfinite(li['loss']), "loss is NaN/Inf"
    expected = li['global_loss'] + 1.0 * li['evidence_loss'] + 0.01 * li['diversity_loss']
    assert torch.allclose(li['loss'], expected, atol=1e-6), "loss composition mismatch"
    # backward + gradient flow to the trainable read-out
    li['loss'].backward()
    grads = {
        'decision_token': lfeq.decision_token.grad,
        'evidence_tokens': lfeq.evidence_tokens.grad,
        'global_head.weight': lfeq.global_head.weight.grad,
        'evidence_head.weight': lfeq.evidence_head.weight.grad,
        'patch_projection.weight': lfeq.patch_projection.weight.grad,
        'block0.cross_attn.in_proj_weight': lfeq.blocks[0].cross_attn.in_proj_weight.grad,
    }
    for name, g in grads.items():
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, \
            f"no gradient flow to {name}"
    print("  [ok] loss composition + gradient flow to decision/evidence/attn")


def test_diversity_regulariser(module):
    b, n, d = 2, 16, 256
    lfeq = module.LearnableForgeryEvidenceQuery(
        vit_dim=d, hidden_dim=256, num_evidence_tokens=8, depth=1, num_heads=8)
    out = lfeq(torch.randn(b, n, d))
    div = lfeq.attention_diversity_loss(out['attention_maps'])
    assert torch.isfinite(div) and div >= 0, f"diversity invalid: {div}"
    print(f"  [ok] diversity regulariser finite and >=0 (value={div.item():.4f})")


def test_5d_aggregation_arithmetic(module):
    """Validate the argmax-confidence TAA aggregation the detector uses."""
    torch.manual_seed(0)
    b, ncrops = 2, 5
    lfeq = module.LearnableForgeryEvidenceQuery(
        vit_dim=1024, hidden_dim=256, num_evidence_tokens=8, depth=1, num_heads=8)
    per_crop = lfeq(torch.randn(b * ncrops, 16, 1024))['fused_probs'][:, 1].view(b, ncrops)
    conf = torch.abs(per_crop - 0.5)
    max_idx = torch.argmax(conf, dim=1)
    ar = torch.arange(b)
    final = per_crop[ar, max_idx]
    expected = torch.stack([per_crop[i, int(max_idx[i])] for i in range(b)])
    assert torch.allclose(final, expected), "TAA aggregation mismatch"
    print("  [ok] 5D argmax-confidence TAA aggregation arithmetic")


def py_compile_all():
    files = [
        os.path.join(_DEEPFAKE, 'training', 'detectors', 'lfeq_module.py'),
        os.path.join(_DEEPFAKE, 'training', 'detectors', 'effort_detector_lfeq.py'),
        os.path.join(_DEEPFAKE, 'training', 'detectors', '__init__.py'),
        os.path.join(_DEEPFAKE, 'experiments', 'experiment_utils.py'),
        os.path.join(_DEEPFAKE, 'experiments', 'run_g18_lfeq.py'),
        os.path.join(_DEEPFAKE, 'experiments', 'smoke_test_lfeq.py'),
    ]
    for f in files:
        r = subprocess.run([sys.executable, '-m', 'py_compile', f],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [x] py_compile FAILED: {os.path.basename(f)}\n{r.stderr}")
            return False
    print("  [ok] py_compile all G18 files")
    return True


def check_key_consistency():
    """Cross-check the lfeq_* keys are identical in detector reads, build_config,
    and arch_keys (static read + regex)."""
    import re
    det = open(os.path.join(_DEEPFAKE, 'training', 'detectors',
                            'effort_detector_lfeq.py'), encoding='utf-8').read()
    utils = open(os.path.join(_DEEPFAKE, 'experiments',
                              'experiment_utils.py'), encoding='utf-8').read()
    det_keys = set(re.findall(r"config\.get\('(lfeq_[a-z_]+)'", det))
    utils_keys = set(re.findall(r"config\['(lfeq_[a-z_]+)'\]", utils))
    arch_keys = set(re.findall(r"'(lfeq_[a-z_]+)'", utils))
    # the detector also reads config.get('lfeq_...') — union & compare
    missing_in_utils = det_keys - utils_keys
    missing_in_det = utils_keys - det_keys
    assert not missing_in_utils, f"setup keys used by detector but not set in build_config: {missing_in_utils}"
    assert not missing_in_det, f"setup keys set in build_config but not read by detector: {missing_in_det}"
    missing_in_arch = utils_keys - arch_keys
    assert not missing_in_arch, f"keys missing from arch_keys (testall strict-load): {missing_in_arch}"
    assert 'lfeq_fusion_weight' in arch_keys and 'lfeq_fusion_weight' in det_keys
    print("  [ok] lfeq_* keys consistent across detector / build_config / arch_keys")


if __name__ == '__main__':
    mod = load_module()
    ok = True
    print("G18 LFEQ verification (local, torch-only)\n" + "=" * 50)
    for fn in (test_forward_shapes, test_fusion_arithmetic,
               test_hard_argmax_wiring, test_loss_composition_and_backward,
               test_diversity_regulariser, test_5d_aggregation_arithmetic):
        try:
            fn(mod)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [x] {fn.__name__} FAILED: {e}")
    ok = py_compile_all() and ok
    try:
        check_key_consistency()
    except AssertionError as e:
        ok = False
        print(f"  [x] key consistency FAILED: {e}")
    print("=" * 50)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)

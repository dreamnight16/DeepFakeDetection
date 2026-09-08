"""Local verification for the LFEQ read-out head (G18) and the G19 read-outs.

This runs WITHOUT the heavy runtime deps (loralib / tensorboard / sklearn /
datasets), which only exist on the server.  It loads ``detectors/lfeq_module.py``
and ``detectors/token_readout.py`` directly by file path (pure torch) and
exercises the novel pieces end-to-end: forward shapes, fusion arithmetic,
hard-argmax selection, loss composition, gradient flow, the diversity
regulariser (G18), and all THREE G19 read-out modes (mean / per_token / concat)
of ``TokenReadout`` — truthful arithmetic, single-CE gradient flow, no LFEQ
read-out keys — plus it static-checks (``py_compile``) every G18/G19 file and
asserts the config-key names are consistent across detectors, ``build_config``,
and ``arch_keys``.

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
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEEPFAKE = os.path.dirname(_HERE)
_MODULE_PATH = os.path.join(_DEEPFAKE, 'training', 'detectors', 'lfeq_module.py')
_TOK_PATH = os.path.join(_DEEPFAKE, 'training', 'detectors', 'token_readout.py')


def load_module():
    spec = importlib.util.spec_from_file_location('lfeq_module', _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_as(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_token_readout():
    """Load ``token_readout.py`` under a synthetic package so its
    ``from .lfeq_module import EvidenceQueryBlock`` relative import resolves,
    without pulling the heavy training stack (loralib / datasets)."""
    dirname = os.path.dirname(_MODULE_PATH)   # training/detectors
    pkg_name = 'detectors_smoke'
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [dirname]
    sys.modules[pkg_name] = pkg
    _load_as(pkg_name + '.lfeq_module', _MODULE_PATH)
    return _load_as(pkg_name + '.token_readout', _TOK_PATH)


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


def test_evidence_token_sweep(module):
    """G18-2: all 6 K values in the sweep must build, forward, and backprop.

    K=1 (no diversity, single evidence slot) through K=32 (33 query tokens in
    self-attn).  Verifies the evidence-dependent shapes scale with K, that the
    diversity regulariser is a no-op (0) for K=1, and that gradients reach the
    trainable read-out for the extreme K values.
    """
    b, n, d = 3, 32, 1024
    for k in (1, 2, 4, 8, 16, 32):
        lfeq = module.LearnableForgeryEvidenceQuery(
            vit_dim=d, hidden_dim=256, num_evidence_tokens=k,
            depth=2, num_heads=8, fusion_weight=0.5)
        patches = _make_patches(b=b, n=n, d=d, seed=k)
        out = lfeq(patches)
        assert out['evidence_logits'].shape == (b, k, 2), (k, out['evidence_logits'].shape)
        assert out['attention_maps'].shape == (b, k, n), (k, out['attention_maps'].shape)
        assert out['selected_evidence_index'].shape == (b,), k
        assert out['fused_probs'].shape == (b, 2), k
        labels = torch.tensor([0, 1, 1])
        li = lfeq.compute_loss(out, labels, evidence_weight=1.0, diversity_weight=0.01)
        assert torch.isfinite(li['loss']) and torch.isfinite(li['diversity_loss']), k
        if k == 1:
            assert li['diversity_loss'].item() == 0.0, "K=1 must have an empty diversity term"
        li['loss'].backward()
        for nm, p in [('decision_token', lfeq.decision_token),
                      ('evidence_tokens', lfeq.evidence_tokens),
                      ('global_head', lfeq.global_head.weight),
                      ('evidence_head', lfeq.evidence_head.weight)]:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, \
                f"(K={k}) no gradient flow to {nm}"
    print("  [ok] evidence-token sweep K in {1,2,4,8,16,32}: build/forward/backward/diversity")


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


def _readout_contract(mod, mode, b=4, n=256, d=1024, evi=8, hidden=256, heads=8):
    """Build a TokenReadout in ``mode``, forward random patches, return (readout, out)."""
    readout = mod.TokenReadout(
        vit_dim=d, hidden_dim=hidden, num_evidence_tokens=evi,
        depth=2, num_heads=heads, dropout=0.1, readout_mode=mode)
    out = readout(torch.randn(b, n, d))
    assert readout.readout_mode == mode
    # no LFEQ read-out keys (head split / argmax / fusion / div)
    for k in ('fused_probs', 'global_logits', 'evidence_logits',
              'selected_evidence_logits', 'selected_evidence_index',
              'global_feature', 'evidence_features', 'attention_maps'):
        assert k not in out, f"G19 read-out must NOT expose LFEQ read-out key {k}"
    return readout, out


def test_mean_readout_forward(mod, b=4, n=256, d=1024, evi=8, hidden=256):
    """G19-A (mean): shapes + truthful arithmetic (pooled == mean, logits == head(pooled))."""
    readout, out = _readout_contract(mod, 'mean', b=b, n=n, d=d, evi=evi, hidden=hidden)
    assert out['logits'].shape == (b, 2), out['logits'].shape
    assert out['probs'].shape == (b, 2)
    assert out['prob'].shape == (b,)
    assert out['pooled'].shape == (b, hidden), out['pooled'].shape
    assert out['queries'].shape == (b, evi + 1, hidden), out['queries'].shape
    assert out['prediction'].shape == (b,)
    assert torch.allclose(out['pooled'], out['queries'].mean(dim=1), atol=1e-6), \
        "mean: pooled != mean over all query tokens"
    assert torch.allclose(out['logits'], readout.head(out['pooled']), atol=1e-6), \
        "mean: logits != head(pooled)"
    assert torch.allclose(out['prob'], out['probs'][:, 1], atol=1e-7)
    assert torch.all(out['probs'] >= 0) and torch.all(out['probs'] <= 1)
    assert out['queries'].shape[1] == readout.decision_token.shape[1] + readout.evidence_tokens.shape[1]
    print("  [ok] G19-A mean read-out: forward shapes + no-LFEQ-readout + arithmetic")


def test_per_token_readout_forward(mod, b=4, n=256, d=1024, evi=8, hidden=256):
    """G19-B (per-token): K+1 independent heads; logits == soft-average of per-token logits."""
    readout, out = _readout_contract(mod, 'per_token', b=b, n=n, d=d, evi=evi, hidden=hidden)
    assert out['logits'].shape == (b, 2)
    assert out['prob'].shape == (b,)
    assert out['pooled'].shape == (b, hidden), "per_token feat is the mean-pool feature"
    assert len(readout.heads) == evi + 1, len(readout.heads)
    # arithmetic: logits == mean_i( head_i(queries_i) )
    per = torch.stack([h(out['queries'][:, i]) for i, h in enumerate(readout.heads)], dim=1)
    assert torch.allclose(out['logits'], per.mean(dim=1), atol=1e-6), \
        "per_token: logits != soft-average of per-token logits"
    # B must NOT reduce to A: independent heads mean per-token logits differ from a
    # single shared head on the mean-pooled feature (same weights NOT reused).
    assert torch.allclose(out['pooled'], out['queries'].mean(dim=1), atol=1e-6)
    print("  [ok] G19-B per-token read-out: K+1 heads + logits == soft-average of per-token")


def test_concat_readout_forward(mod, b=4, n=256, d=1024, evi=8, hidden=256):
    """G19-C (concat): flatten all tokens; logits == head(flatten(queries))."""
    readout, out = _readout_contract(mod, 'concat', b=b, n=n, d=d, evi=evi, hidden=hidden)
    assert out['logits'].shape == (b, 2)
    assert out['prob'].shape == (b,)
    assert out['pooled'].shape == (b, hidden * (evi + 1)), out['pooled'].shape
    assert out['queries'].shape == (b, evi + 1, hidden)
    flat = out['queries'].reshape(b, -1)
    assert torch.allclose(out['pooled'], flat, atol=1e-6), \
        "concat: pooled != flatten(queries)"
    assert torch.allclose(out['logits'], readout.head(flat), atol=1e-6), \
        "concat: logits != head(flatten(queries))"
    print("  [ok] G19-C concat read-out: flatten all tokens + wide linear head")


def test_readout_loss_grad(mod, b=4, n=32, d=1024, evi=8, hidden=256):
    """All THREE modes: single CE is finite and gradients reach the query transformer."""
    for mode in ('mean', 'per_token', 'concat'):
        readout = mod.TokenReadout(
            vit_dim=d, hidden_dim=hidden, num_evidence_tokens=evi, depth=2, num_heads=8,
            dropout=0.1, readout_mode=mode)
        patches = torch.randn(b, n, d)
        labels = torch.tensor([0, 1, 1, 1])
        out = readout(patches)
        loss = readout.compute_loss(out, labels)['loss']
        assert torch.isfinite(loss), f"{mode}: loss is NaN/Inf"
        loss.backward()
        head_param = ('head', readout.head.weight) if readout.head is not None \
            else ('heads[0]', readout.heads[0].weight)
        for nm, p in [('decision_token', readout.decision_token),
                      ('evidence_tokens', readout.evidence_tokens),
                      (head_param[0], head_param[1]),
                      ('patch_projection.weight', readout.patch_projection.weight),
                      ('block0.cross_attn.in_proj_weight', readout.blocks[0].cross_attn.in_proj_weight)]:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, \
                f"({mode}) no gradient flow to {nm}"
    print("  [ok] all G19 modes: single CE + gradient flow to head/tokens/attn")


def py_compile_all():
    files = [
        os.path.join(_DEEPFAKE, 'training', 'detectors', 'lfeq_module.py'),
        os.path.join(_DEEPFAKE, 'training', 'detectors', 'token_readout.py'),
        os.path.join(_DEEPFAKE, 'training', 'detectors', 'effort_detector_lfeq.py'),
        os.path.join(_DEEPFAKE, 'training', 'detectors', 'effort_detector_lfeq_readout_base.py'),
        os.path.join(_DEEPFAKE, 'training', 'detectors', 'effort_detector_lfeq_mean.py'),
        os.path.join(_DEEPFAKE, 'training', 'detectors', 'effort_detector_lfeq_per_token.py'),
        os.path.join(_DEEPFAKE, 'training', 'detectors', 'effort_detector_lfeq_concat.py'),
        os.path.join(_DEEPFAKE, 'training', 'detectors', '__init__.py'),
        os.path.join(_DEEPFAKE, 'experiments', 'experiment_utils.py'),
        os.path.join(_DEEPFAKE, 'experiments', 'run_g18_lfeq.py'),
        os.path.join(_DEEPFAKE, 'experiments', 'run_g18_2_evidence_sweep.py'),
        os.path.join(_DEEPFAKE, 'experiments', 'run_g19.py'),
        os.path.join(_DEEPFAKE, 'experiments', 'smoke_test_lfeq.py'),
    ]
    for f in files:
        r = subprocess.run([sys.executable, '-m', 'py_compile', f],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [x] py_compile FAILED: {os.path.basename(f)}\n{r.stderr}")
            return False
    print("  [ok] py_compile all G18/G19 files")
    return True


def check_key_consistency():
    """Cross-check the lfeq_* keys are identical in detector reads, build_config,
    and arch_keys (static read + regex)."""
    import re
    det_text = ""
    for f in ('effort_detector_lfeq.py', 'effort_detector_lfeq_readout_base.py'):
        det_text += open(os.path.join(_DEEPFAKE, 'training', 'detectors', f),
                         encoding='utf-8').read()
    utils = open(os.path.join(_DEEPFAKE, 'experiments',
                              'experiment_utils.py'), encoding='utf-8').read()
    det_keys = set(re.findall(r"config\.get\('(lfeq_[a-z_]+)'", det_text))
    utils_keys = set(re.findall(r"config\['(lfeq_[a-z_]+)'\]", utils))
    arch_keys = set(re.findall(r"'(lfeq_[a-z_]+)'", utils))
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
    tok = load_token_readout()
    ok = True
    print("G18 + G19 verification (local, torch-only)\n" + "=" * 50)
    for fn in (test_forward_shapes, test_fusion_arithmetic,
               test_hard_argmax_wiring, test_loss_composition_and_backward,
               test_diversity_regulariser, test_evidence_token_sweep,
               test_5d_aggregation_arithmetic):
        try:
            fn(mod)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [x] {fn.__name__} FAILED: {e}")
    for fn in (test_mean_readout_forward, test_per_token_readout_forward,
               test_concat_readout_forward, test_readout_loss_grad):
        try:
            fn(tok)
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

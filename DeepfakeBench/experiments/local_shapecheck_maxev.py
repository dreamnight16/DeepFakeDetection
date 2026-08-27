"""Throwaway LOCAL shape/wiring check for EffortDetectorMaxEvidence.

Stubs server-only deps (loralib / tensorboard) and the backbone so the real
detector class can be instantiated and run forward + get_losses + backward here
(no CLIP, no datasets).  Verifies:
    * forward shapes: cls[B,2], patch_logits[B,N,2], fake_prob_map[B,N],
      max_index[B], prob[B]
    * prob == softmax(cls)[:,1]
    * get_losses returns overall/loss_cls/loss_max scalars & finite
    * backward: both head and patch_head get nonzero gradient (branches alive)
    * the patch_head has NO LayerNorm (ln removed)
"""
import os
import sys
import types

_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # DeepfakeBench
_train = os.path.join(_base, 'training')
for p in (_train, _base):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

# stubs for server-only deps
mod = types.ModuleType('torch.utils.tensorboard'); mod.SummaryWriter = object
sys.modules['torch.utils.tensorboard'] = mod
sys.modules['loralib'] = types.ModuleType('loralib')
# networks/efficientnetb4 imports this; provide a tolerated dummy
_ep = types.ModuleType('efficientnet_pytorch'); _ep.EfficientNet = object
sys.modules['efficientnet_pytorch'] = _ep

import torch.nn as nn
from detectors import DETECTOR
from detectors.effort_detector import EffortDetector


class StubBackbone(nn.Module):
    def __init__(self, N=16, D=1024):
        super().__init__()
        self.N, self.D = N, D
    def forward(self, x):
        B = x.shape[0]
        return {
            'last_hidden_state': torch.randn(B, self.N + 1, self.D),   # [B, N+1, D]
            'pooler_output': torch.randn(B, self.D),                    # [B, D]
        }


def run_one_mode(cls_feature):
    config = {
        'model_name': 'effort_maxev',
        'use_loralib': False,
        'full_train_head': True,          # head = nn.Linear(1024, 2)
        'max_evidence_lambda': 1.0,
        'max_evidence_eps': 1e-8,
        'max_evidence_cls_feature': cls_feature,
    }
    model = DETECTOR['effort_maxev'](config)
    model.train()
    return model


def main():
    # monkeypatch build_backbone so __init__ doesn't load the CLIP model
    EffortDetector.build_backbone = lambda self, config: StubBackbone()

    B, N = 4, 16
    images = torch.randn(B, 3, 224, 224)
    labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    for cls_feature in ('raw_token', 'pooler_output'):
        print(f"\n########## cls_feature = {cls_feature} ##########")
        model = run_one_mode(cls_feature)

        print("== architecture ==")
        print("  has patch_ln:", hasattr(model, 'patch_ln'))
        assert not hasattr(model, 'patch_ln'), "patch_ln present (LN not removed)"
        print("  head:", type(model.head).__name__, "patch_head:", type(model.patch_head).__name__)

        print("\n== forward ==")
        pred = model.forward({'image': images, 'label': labels}, inference=False)
        assert pred['cls'].shape == (B, 2), pred['cls'].shape
        assert pred['patch_logits'].shape == (B, N, 2), pred['patch_logits'].shape
        assert pred['fake_prob_map'].shape == (B, N), pred['fake_prob_map'].shape
        assert pred['max_index'].shape == (B,), pred['max_index'].shape
        assert pred['prob'].shape == (B,), pred['prob'].shape
        assert torch.allclose(pred['prob'], torch.softmax(pred['cls'], dim=-1)[:, 1], atol=1e-6)
        # selection picks the row-max fake prob
        gathered = pred['fake_prob_map'][torch.arange(B), pred['max_index']]
        assert torch.allclose(gathered, pred['fake_prob_map'].max(dim=1)[0], atol=1e-6)
        print("  shapes OK: cls", tuple(pred['cls'].shape),
              "patch_logits", tuple(pred['patch_logits'].shape),
              "max_index", tuple(pred['max_index'].shape))
        print("  prob == softmax(cls)[:,1]: OK")
        print("  selection == argmax fake-prob: OK")

        print("\n== get_losses ==")
        losses = model.get_losses({'image': images, 'label': labels}, pred)
        for k in ('overall', 'loss_cls', 'loss_max', 'real_loss', 'fake_loss'):
            v = losses[k]
            assert v.ndim == 0 and torch.isfinite(v), (k, v)
            print(f"  {k} = {float(v):.4f}")

        print("\n== backward (both heads alive) ==")
        model.zero_grad()
        losses['overall'].backward()
        g_head = model.head.weight.grad.norm().item() if model.head.weight.grad is not None else 0.0
        g_patch = model.patch_head.weight.grad.norm().item() if model.patch_head.weight.grad is not None else 0.0
        print(f"  grad|head| = {g_head:.6g}   grad|patch_head| = {g_patch:.6g}")
        assert g_head > 0, "CLS head gradient dead"
        assert g_patch > 0, "patch_head gradient dead"

    print("\nALL WIRING CHECKS PASSED (raw_token + pooler_output)")


if __name__ == '__main__':
    main()

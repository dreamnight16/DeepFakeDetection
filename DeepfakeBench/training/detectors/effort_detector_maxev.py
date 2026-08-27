"""Maximum Fake-Evidence Selection Loss detector (Experiment G15).

Implements the method in ``最大伪造证据选择损失.md``: a frozen CLIP ViT-L/14 +
LoRA backbone, a **kept** CLS classification head (Section 4), and an auxiliary
**patch-level local-evidence head** that is supervised only on the single patch
with the largest fake probability (Sections 5-7).

For an image x with N patch tokens h_i (i = 1..N) and a CLS feature h_cls.
By default (``max_evidence_cls_feature='raw_token'``) h_cls is the raw final-layer
CLS token ``last_hidden_state[:, 0]`` — strict per the method doc §3/§4.1. If set
to ``'pooler_output'``, h_cls is CLIP's ``pooler_output`` (= LayerNorm(CLS token)),
the exact feature the baseline CLS head is trained on, so the B0↔maxev CLS branch
is byte-identical (anchor-aligned supplementary variant):

    cls_logits = W_cls h_cls + b_cls                    (kept CLS head)
    a_i        = W_e h_i + b_e                          (shared patch evidence head)
    q_i        = softmax(a_i)                           (per-patch 2-way)

Selection (Section 6): pick the patch with the largest fake probability

    i*  = argmax_i q_{i,1}   == argmax_i (a_{i,1} - a_{i,0})   (monotone)

Total loss (Section 10):

    L = L_cls + lambda_max * L_max
      = CE(cls_logits, y) + lambda_max * CE(a_{i*}, y)

Non-symmetric behaviour (Sections 8-9): for real images (y=0) it drives the max
fake probability to 0, hence *all* patches toward 0 (universal); for fake images
(y=1) it drives the max fake probability to 1 — i.e. only requires *some* patch
to be fake (existential).  Inference (Section 16) uses the CLS branch as the
final score: prob = softmax(cls_logits)[:, 1].

Design notes for this implementation:
    * The patch evidence head is a plain linear layer applied directly to the
      raw patch tokens — NO LayerNorm (the "ln 直接去掉" instruction; the AEPA
      (G12) patch branch used a LayerNorm, this one does not).
    * The CLS head is trained from scratch (identical in both variants).
    * Two variants differ ONLY in how the patch head is initialised:
        - ``max_evidence_init_ckpt`` absent  ->  patch head random (default init)
        - ``max_evidence_init_ckpt`` set     ->  patch head seeded from a trained
          CLS-classifier checkpoint (W_e <- W_cls, b_e <- b_cls), the AEPA (G12)
          Section-3 warm-start.  The ckpt is only read for init; the trained
          weights are then loaded normally.
    * Caveat on the warm-start transfer — the source is the SAME in both variants:
      the warm-start W_cls always comes from the b0_cls_baseline checkpoint
      (model_name='effort'), whose head was trained on CLIP's pooler_output
      (base EffortDetector.features() returns pooler_output).  So in BOTH the
      'raw_token' and 'pooler_output' variants, a pooler-trained W_cls is copied
      onto a patch head that acts on raw patch tokens -> the identical
      pooler->raw input-distribution gap.  The max_evidence_cls_feature flag only
      changes which feature the maxev model's OWN CLS branch consumes during
      maxev training (_max_ev_forward); it does NOT change the warm-start source.
      G15 has no raw-token-trained CLS baseline that could serve as a
      domain-matched init source, so the earlier "better domain-matched transfer"
      reading is not realized.
        - default 'raw_token': the maxev CLS branch reads the raw final-layer CLS
          token, so it is no longer byte-identical to the b0 baseline (which uses
          CLIP's pooler_output).
        - 'pooler_output' (supplementary): the maxev CLS branch reads CLIP's
          pooler_output, keeping the b0↔maxev CLS branch identical (cleanest
          anchor control) but deviating from the doc's literal h_cls.
      Either way the no_init vs cls_init ablation is a clean single variable (both
      consume raw patch tokens; only init differs).  The warm-start should be read
      as "seed the patch readout from a pooler-trained CLS classifier" rather than
      a domain-matched transfer.  See the protocol doc's caveat section (§2/§5.2/§8).
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors import DETECTOR
from .effort_detector import EffortDetector

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='effort_maxev')
class EffortDetectorMaxEvidence(EffortDetector):
    """Max-fake-evidence-selection detector.

    Subclasses ``EffortDetector`` to reuse its CLIP + LoRA backbone *and* its
    CLS classification head (``self.head``), then adds the shared patch evidence
    head and the max-fake-evidence local loss.

    Config keys:
        max_evidence_lambda   : float, local-loss weight lambda_max (default 1.0)
        max_evidence_eps      : float, numerical stability (default 1e-8)
        max_evidence_init_ckpt: optional path to a trained CLS-classifier
                                checkpoint whose ``head.weight``/``head.bias``
                                seed the patch evidence head (W_e <- W_cls).
        max_evidence_cls_feature: 'raw_token' (default, strict per doc §3/§4.1)
                               or 'pooler_output' (supplementary, b0-anchor-aligned).
    """

    def __init__(self, config=None):
        config = config if config is not None else {}
        # super().__init__ builds the CLIP+LoRA backbone and the CLS head
        # (self.head), so the CLS branch matches the baseline reading exactly.
        super().__init__(config)

        hidden = 1024  # CLIP ViT-L/14 feature dimension

        # Shared patch evidence head (W_e, b_e).  Plain linear, NO LayerNorm:
        # applied directly to the raw patch tokens h_i (Section 5.1).
        self.patch_head = nn.Linear(hidden, 2, bias=True)

        self.lambda_max = float(config.get('max_evidence_lambda', 1.0))
        self.eps = float(config.get('max_evidence_eps', 1e-8))

        # Which CLIP output feeds the (kept) CLS classification head.
        #   raw_token    (default, PRIMARY — strict per the method doc §3/§4.1):
        #                 h_cls = last_hidden_state[:, 0]  (the final-layer CLS token)
        #   pooler_output (SUPPLEMENTARY — anchor-aligned to the ``effort`` baseline):
        #                 h_cls = pooler_output  (= LayerNorm(CLS token)), the exact
        #                 feature the b0_cls_baseline ``self.head`` was trained on, so
        #                 the b0<->maxev CLS branch is a byte-identical control.
        self.cls_feature = config.get('max_evidence_cls_feature', 'raw_token')

        # Optional warm-start: seed the patch head from a trained CLS classifier
        # (Section 3 of AEPA_method.md, here applied to the max-evidence head).
        init_ckpt = config.get('max_evidence_init_ckpt', None)
        if init_ckpt:
            self._init_patch_head_from_cls(init_ckpt)

    # ── Init helpers ────────────────────────────────────────────────────────

    def _init_patch_head_from_cls(self, ckpt_path: str) -> None:
        """Seed the patch evidence head from a CLS-classifier checkpoint.

        The trained CLS baseline stores its readout as ``head.weight`` (2x1024)
        and ``head.bias`` (2); the patch head has the same input/output dims,
        so we reuse the CLS head as the initial patch evidence head
        (W_e <- W_cls, b_e <- b_cls).
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        new = {k.replace('module.', ''): v for k, v in ckpt.items()}
        w = new.get('head.weight')
        b = new.get('head.bias')
        if w is None or b is None:
            raise KeyError(
                f"[maxev] checkpoint {ckpt_path} has no 'head.weight'/'head.bias' "
                f"(keys: {sorted(new.keys())[:8]} ...)"
            )
        with torch.no_grad():
            self.patch_head.weight.copy_(w)
            self.patch_head.bias.copy_(b)
        logger.info(f"[maxev] warm-started patch_head from CLS head: {ckpt_path}")

    # ── Core forward ────────────────────────────────────────────────────────

    def _max_ev_forward(self, images):
        """Forward images -> CLS logits, patch logits, and the selection.

        Returns a dict with:
            cls:            [B, 2]  CLS logits (argmax = class)
            prob:           [B]     CLS fake probability = softmax(cls)[:, 1]
                                    (Section 16: final detection score)
            feat:           [B, D]  CLS feature fed to the head (raw token, or
                                    pooler_output if cls_feature='pooler_output')
            cls_logits:     [B, 2]  alias of cls
            cls_prob:       [B]     alias of prob
            patch_logits:   [B, N, 2]  raw patch evidence-head logits
            patch_prob:     [B, N, 2]  softmax over the 2 classes per patch
            fake_prob_map:  [B, N]  per-patch fake probabilities q_{i,1}
            max_index:      [B]     argmax_i q_{i,1}  (i*)
        """
        out = self.backbone(self._prep_input(images))
        tokens = out['last_hidden_state']              # [B, N+1, D]
        # PRIMARY (strict per the doc §3/§4.1): h_cls = the final-layer CLS
        # token.  SUPPLEMENTARY: h_cls = pooler_output (= LayerNorm(CLS token)),
        # the exact feature the b0 baseline head was trained on, so the b0<->maxev
        # CLS branch is byte-identical.  The patch evidence head uses the raw
        # patch tokens in both cases (no LN, per the instruction).
        cls_feat = (out['pooler_output']
                    if self.cls_feature == 'pooler_output'
                    else tokens[:, 0, :])                # [B, D]
        patches = tokens[:, 1:, :]                     # [B, N, D] (raw, no LN)

        cls_logits = self.head(cls_feat)               # [B, 2]
        cls_prob = torch.softmax(cls_logits, dim=-1)[:, 1]

        # Patch evidence head applied directly to raw tokens (no LayerNorm).
        patch_logits = self.patch_head(patches)        # [B, N, 2]
        patch_prob = torch.softmax(patch_logits, dim=-1)   # [B, N, 2]
        fake_prob = patch_prob[..., 1]                 # [B, N]
        max_index = fake_prob.argmax(dim=1)            # [B]

        return {
            'cls': cls_logits,
            'prob': cls_prob,
            'feat': cls_feat,
            'cls_logits': cls_logits,
            'cls_prob': cls_prob,
            'patch_logits': patch_logits,
            'patch_prob': patch_prob,
            'fake_prob_map': fake_prob,
            'max_index': max_index,
        }

    def forward(self, data_dict, inference=False):
        images = data_dict['image']

        # Multi-crop test-time augmentation [B, n_crops, C, H, W].  Delegate to
        # the base forward so the CLS-branch ensemble (argmax-confidence patch
        # selection) is byte-identical to the baseline, keeping TTA reported-AUC
        # directly comparable.  Inference-only, so no patch_loss keys are needed.
        if inference and len(images.shape) == 5:
            return super().forward(data_dict, inference=True)

        return self._max_ev_forward(images)


    # ── Max-fake-evidence loss ──────────────────────────────────────────────

    def get_losses(self, data_dict, pred_dict):
        """L = L_cls + lambda_max * L_max (Section 10 / Section 15).

        ``L_cls`` is the standard CE on the CLS logits; ``L_max`` is the CE on
        the raw logits of the max-fake-probability patch only.  Using the raw
        (pre-softmax) logits of the selected patch, rather than the softmax
        probability, keeps the cross-entropy numerically stable (Section 15).

        Both losses are computed per-sample so real/fake can be decomposed the
        same way the base detector does (real_loss / fake_loss = per-class means).
        """
        label = data_dict['label']                     # [B] long, 0=real / 1=fake

        cls_logits = pred_dict['cls']                  # [B, 2]
        patch_logits = pred_dict['patch_logits']       # [B, N, 2]
        max_index = pred_dict['max_index']             # [B]

        # L_cls — image-level CE on the CLS branch.
        cls_ce = F.cross_entropy(cls_logits, label, reduction='none')   # [B]

        # Select the raw logits of the max-fake-probability patch, per image.
        batch_idx = torch.arange(patch_logits.size(0),
                                 device=patch_logits.device)
        selected_logits = patch_logits[batch_idx, max_index, :]        # [B, 2]

        # L_max — CE on the selected patch's logits only.
        max_ce = F.cross_entropy(selected_logits, label, reduction='none')  # [B]

        per_sample = cls_ce + self.lambda_max * max_ce              # [B]
        loss = per_sample.mean()

        mask_real = label == 0
        mask_fake = label == 1
        loss_real = per_sample[mask_real].mean() if mask_real.any() \
            else per_sample.new_tensor(0.0)
        loss_fake = per_sample[mask_fake].mean() if mask_fake.any() \
            else per_sample.new_tensor(0.0)

        return {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
            'loss_cls': cls_ce.mean(),
            'loss_max': max_ce.mean(),
        }

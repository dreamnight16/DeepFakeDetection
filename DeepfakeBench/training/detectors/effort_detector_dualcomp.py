"""Dual-line cross-complementary gated fusion detector (Experiment G17-1).

Part of the G17 sequence (G17-1 = model-side gated fusion; G17-2 = data-side
real-noise isolation, which keeps the ``effort`` observer unchanged and only
changes the input via ``freq_band.apply_residual``).

Method (G17-1_dual_complement_gate_protocol.md): a frozen CLIP ViT-L/14 + LoRA
backbone is shared by TWO parallel lines — an RGB line and a frequency-band line
— each with its own judge head and its own patch-level evidence head.  A small
gate MLP reads each line's judgment + evidence ``[p_R, e_R, p_F, e_F]`` and emits
fusion weight ``w``; the composite score is:

    p_final = w * p_R + (1 - w) * p_F

so an uncertain RGB line hands the decision to the frequency line and vice versa
("cross-complementary").  The gate is no-regret: it can collapse to w=1 (always
trust RGB), so the model can never be worse than the RGB-only decision while
remaining free to exploit frequency where RGB evidence is weak.

Trained FROM SCRATCH (no warm-start): the evidence heads and judge heads are
random-init.  This removes the G15 ``cls_init`` warm-start confound so we measure
the pure effect of adding the frequency line + gating.

Inference uses ``p_final`` (the composite score).  For multi-crop 5D TTA the
forward delegates to the base ``EffortDetector`` CLS branch (documented round-1
simplification: the gate under TTA runs on the RGB line only).

Config keys (all MUST be added to ``experiment_utils.arch_keys`` so testall.py
rebuilds the model identically, else strict checkpoint load fails):
    comp_freq_bands    : list[str]  band keys to forward on the F line (['low'])
    comp_freq_norm     : 'fixed_rms' | 'none'  F-line view normalisation
    comp_freq_rms      : float  fixed-RMS target for the F view (default 0.5)
    comp_fuse          : 'gate' (learned) | 'equal' (fixed w=0.5, control)
    comp_gate_hidden   : int  gate MLP hidden width (default 32)
    comp_lambda_freq   : float  weight on the per-line CLS losses (default 1.0)
    comp_lambda_max    : float  weight on the max-evidence loss (default 1.0)
    comp_cls_feature   : 'pooler_output' (anchor-aligned to ``effort``; default)
                         | 'raw_token' (doc-strict)
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora_lib

from detectors import DETECTOR
from .effort_detector import EffortDetector, lora

logger = logging.getLogger(__name__)

# G17-1 frequency bands (subset of freq_band.FREQ_BANDS) — pre-registered Low
# for the F line (G13's most-generalising band).  Only the radial *range* is used
# by the in-detector FFT; the data-side reconstruction/normalisation discipline
# of G13 applies to the G17-2 data-side arm, not to this model-side band view.
_FREQ_RANGES = {
    "low": (0.00, 0.15),
    "mid_low": (0.15, 0.35),
    "mid_high": (0.35, 0.65),
    "high": (0.65, 1.00),
}


@DETECTOR.register_module(module_name='effort_dualcomp')
class EffortDetectorDualComplement(EffortDetector):
    def __init__(self, config=None):
        config = config if config is not None else {}
        super().__init__(config)

        self.comp_bands = list(config.get('comp_freq_bands', ['low']))
        self.comp_norm = config.get('comp_freq_norm', 'fixed_rms')
        self.comp_rms = float(config.get('comp_freq_rms', 0.5))
        self.comp_fuse = config.get('comp_fuse', 'gate')          # 'gate' | 'equal'
        self.comp_gate_hidden = int(config.get('comp_gate_hidden', 32))
        self.comp_lambda_freq = float(config.get('comp_lambda_freq', 1.0))
        self.comp_lambda_max = float(config.get('comp_lambda_max', 1.0))
        self.comp_cls_feature = config.get('comp_cls_feature', 'pooler_output')

        hidden = 1024  # CLIP ViT-L/14 feature dimension
        LinearClass = lora_lib.Linear if self.use_loralib else lora.Linear

        # R judge = the base classifier (self.head), so the RGB line is
        # byte-identical to the ``effort`` anchor (A1's R line == A0).  F judge
        # is a matching LoRA-linear so the two lines are structurally comparable.
        self.head_f = LinearClass(
            in_features=hidden, out_features=2,
            r=2, lora_alpha=8, lora_dropout=0, merge_weights=False, bias=True
        )

        # Per-line patch evidence heads (plain linear, no LayerNorm — G12/G15).
        self.patch_head_r = nn.Linear(hidden, 2, bias=True)
        self.patch_head_f = nn.Linear(hidden, 2, bias=True)

        # Cross-complementary gate: reads [p_R, e_R, p_F, e_F] -> w in [0,1].
        self.gate = nn.Sequential(
            nn.Linear(4, self.comp_gate_hidden),
            nn.ReLU(),
            nn.Linear(self.comp_gate_hidden, 1),
        )

    # ── In-detector frequency-band view (torch, fixed-RMS) ─────────────────
    def _radial_mask(self, h, w, lo, hi):
        ys = torch.arange(h, dtype=torch.float32, device=self.last_device).view(h, 1)
        xs = torch.arange(w, dtype=torch.float32, device=self.last_device).view(1, w)
        dist = torch.sqrt((ys - h / 2.0) ** 2 + (xs - w / 2.0) ** 2)
        rmax = min(h, w) / 2.0
        rn = dist / rmax
        return ((rn >= lo) & (rn <= hi)).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    def _band_view(self, x, band):
        """Band-pass ``x`` ([B,C,H,W] CLIP-normalised) to a radial band, then
        fixed-RMS normalise (G13 §7.3: NOT per-image min-max, which re-stretches
        away the amplitude/contrast signal and is OOD for frozen CLIP)."""
        lo, hi = _FREQ_RANGES[band]
        xf = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        mask = self._radial_mask(x.shape[-2], x.shape[-1], lo, hi).to(x.dtype)
        rec = torch.real(torch.fft.ifft2(
            torch.fft.ifftshift(xf * mask, dim=(-2, -1)), dim=(-2, -1)))
        if self.comp_norm == 'fixed_rms':
            rms = rec.std(dim=(-3, -2, -1), keepdim=True) + 1e-6
            rec = rec / rms * self.comp_rms
        return rec

    def _line_out(self, images):
        out = self.backbone(self._prep_input(images))
        tokens = out['last_hidden_state']                    # [B, N+1, D]
        if self.comp_cls_feature == 'pooler_output':
            feat = out['pooler_output']                      # [B, D]
        else:  # 'raw_token'
            feat = tokens[:, 0, :]
        patches = tokens[:, 1:, :]
        return feat, patches

    # ── Core forward ────────────────────────────────────────────────────────
    def _dualcomp_forward(self, images):
        self.last_device = images.device

        feats = {}
        # RGB line.
        feat_r, patch_r = self._line_out(images)
        # Frequency lines: forward every band, aggregate logits + evidence below.
        for band in self.comp_bands:
            v = self._band_view(images, band)
            feat_f, patch_f = self._line_out(v)
            feats[band] = (feat_f, patch_f)

        # Aggregate the F line across bands by AVERAGING the 2-dim logits and the
        # per-patch logits (the same head asserts all bands, so logits are directly
        # comparable) — NOT by averaging probs (mean-of-probs ≠ prob-of-mean-logits).
        # Single band = the logits as-is.
        cls_f_all, patch_logits_f_all = [], []
        for band in self.comp_bands:
            feat_f, patch_f = feats[band]
            cls_f_all.append(self.head_f(feat_f))                       # [B, 2]
            if self.comp_lambda_max > 0:
                patch_logits_f_all.append(self.patch_head_f(patch_f))  # [B, N, 2]
        cls_f = torch.stack(cls_f_all, dim=0).mean(dim=0)               # [B, 2] aggregate F logits
        p_f = torch.softmax(cls_f, dim=-1)[:, 1]                        # [B] F fake-prob
        if self.comp_lambda_max > 0:
            patch_logits_f = torch.stack(patch_logits_f_all, dim=0).mean(dim=0)  # [B, N, 2]
            e_f = torch.softmax(patch_logits_f, dim=-1)[..., 1].max(dim=1).values
        else:
            patch_logits_f = None
            e_f = torch.zeros_like(p_f)

        # R judgement + evidence.
        cls_r = self.head(feat_r)                                       # [B, 2]
        p_r = torch.softmax(cls_r, dim=-1)[:, 1]
        if self.comp_lambda_max > 0:
            patch_logits_r = self.patch_head_r(patch_r)                 # [B, N, 2]
            e_r = torch.softmax(patch_logits_r, dim=-1)[..., 1].max(dim=1).values
        else:
            patch_logits_r = None
            e_r = torch.zeros_like(p_r)

        # Cross-complementary gate: read [p_R, e_R, p_F, e_F] -> w in [0,1].
        if self.comp_fuse == 'equal':
            w = torch.full_like(p_r, 0.5)
        else:
            g = self.gate(torch.stack([p_r, e_r, p_f, e_f], dim=1))
            w = torch.sigmoid(g.squeeze(-1))
        p_final = w * p_r + (1 - w) * p_f

        out = {
            'cls': cls_r,
            'prob': p_final,
            'feat': feat_r,
            'cls_r': cls_r,
            'cls_f': cls_f,
            'p_r': p_r,
            'p_f': p_f,
            'e_r': e_r,
            'e_f': e_f,
            'w': w,
        }
        if patch_logits_r is not None:
            out['patch_logits_r'] = patch_logits_r
            out['patch_logits_f'] = patch_logits_f
            out['fake_prob_r'] = torch.softmax(patch_logits_r, dim=-1)[..., 1]
            out['fake_prob_f'] = torch.softmax(patch_logits_f, dim=-1)[..., 1]
            out['max_index_r'] = out['fake_prob_r'].argmax(dim=1)
            out['max_index_f'] = out['fake_prob_f'].argmax(dim=1)
        return out

    def forward(self, data_dict, inference=False):
        images = data_dict['image']
        # Multi-crop TTA [B, n_crops, C, H, W]: delegate to the base CLS-branch
        # ensemble (documented round-1 simplification).
        if inference and len(images.shape) == 5:
            return super().forward(data_dict, inference=True)
        return self._dualcomp_forward(images)

    # ── Composite + per-line + max-evidence loss ────────────────────────────
    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label']                           # [B] 0=real/1=fake
        p_final = pred_dict['prob']                          # [B] composite score

        # Per-sample composite CE on p_final (the metric target).
        eps = 1e-8
        comp_ce = -(label * torch.log(p_final.clamp(eps, 1 - eps))
                    + (1 - label) * torch.log((1 - p_final).clamp(eps, 1 - eps)))

        # Per-line CLS CE on the 2-dim logits (numerically stable).
        cls_r_ce = F.cross_entropy(pred_dict['cls_r'], label, reduction='none')
        cls_f_ce = F.cross_entropy(pred_dict['cls_f'], label, reduction='none')

        # Max-fake-evidence per line (G15); 0 if lambda_max==0.
        max_ce_r = max_ce_f = 0.0
        if self.comp_lambda_max > 0:
            b = torch.arange(pred_dict['patch_logits_r'].size(0),
                             device=label.device)
            sel_r = pred_dict['patch_logits_r'][b, pred_dict['max_index_r']]
            sel_f = pred_dict['patch_logits_f'][b, pred_dict['max_index_f']]
            max_ce_r = F.cross_entropy(sel_r, label, reduction='none')
            max_ce_f = F.cross_entropy(sel_f, label, reduction='none')

        per_sample = (comp_ce
                      + self.comp_lambda_freq * (cls_r_ce + cls_f_ce)
                      + self.comp_lambda_max * (max_ce_r + max_ce_f))
        loss = per_sample.mean()

        mask_real = label == 0
        mask_fake = label == 1
        loss_real = per_sample[mask_real].mean() if mask_real.any() else per_sample.new_tensor(0.0)
        loss_fake = per_sample[mask_fake].mean() if mask_fake.any() else per_sample.new_tensor(0.0)

        return {
            'overall': loss,
            'real_loss': loss_real,
            'fake_loss': loss_fake,
            'loss_comp': comp_ce.mean(),
            'loss_cls_r': cls_r_ce.mean(),
            'loss_cls_f': cls_f_ce.mean(),
            'loss_max_r': (max_ce_r.mean() if self.comp_lambda_max > 0
                           else loss.new_tensor(0.0)),
            'loss_max_f': (max_ce_f.mean() if self.comp_lambda_max > 0
                           else loss.new_tensor(0.0)),
        }

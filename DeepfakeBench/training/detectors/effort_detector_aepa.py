"""Asymmetric Evidential Patch Aggregation (AEPA) detector for AIGI detection.

Implements Experiment G12 (see ``AEPA_method.md``): patch-level detection on a
frozen CLIP ViT-L/14 backbone with LoRA attention, replacing the CLS readout.

For an image x with N patch tokens h_i (i = 1..N):

    tilde_h_i = LN(h_i)                                     (patch normalization)
    p_i       = Softmax(W_p tilde_h_i + b_p)                (shared 2-way head)
    E_i       = Softplus(w_e^T tilde_h_i + b_e)             (shared scalar evidence)

    r_i = E_i p_iR / (2 + E_i)        real evidence mass   (b3 only)
    f_i = E_i p_iF / (2 + E_i)        fake evidence mass   (b3 only)

For b1/b2 (ablations, no evidence head) r_i = p_iR, f_i = p_iF.

Image-level fake quality and normalized score (Section 5.2 / 7):

    s_AEPA(x) = -1/N sum_i log(1 - f_i + eps)
    P_F(x)     = 1 - prod_i (1 - f_i) = 1 - exp(-N * s_AEPA)

Asymmetric loss (Section 6):

    real:  -1/N sum_i log(r_i + eps)          (universal aggregation)
    fake:  -log(1 - prod_i (1 - f_i) + eps)   (existential aggregation)
    L = mean_real + lambda_F * mean_fake

The CLS token still participates in self-attention but is not used for the
final score (Section 9).  Relative to the CLS baseline only a shared scalar
evidence head (d + 1 parameters) is added.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors import DETECTOR
from .effort_detector import EffortDetector

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='effort_aepa')
class EffortDetectorAEPA(EffortDetector):
    """Patch-level evidential detector.

    Subclasses ``EffortDetector`` only to reuse its CLIP + LoRA backbone
    builder; the CLS head, margin-loss machinery and mixup readout are replaced
    by the patch-level heads below.

    Config keys:
        aepa_mode      : 'b3_evidence' (default) | 'b2_mil' | 'b1_pool'
        aepa_lambda_f  : float, fake-branch loss weight (default 1.0)
        aepa_eps       : float, numerical stability (default 1e-8)
    """

    def __init__(self, config=None):
        config = config if config is not None else {}
        # Skip EffortDetector.__init__ (which builds a CLS head); only reuse
        # the backbone builder.  nn.Module.__init__ sets up the parameter
        # registries exactly as EffortDetector's own super().__init__ would.
        nn.Module.__init__(self)
        self.config = config
        self.use_loralib = config.get('use_loralib', False)
        self.backbone = EffortDetector.build_backbone(self, config)

        hidden = 1024  # CLIP ViT-L/14 feature dimension

        # Patch-level heads (shared across all N patches).
        self.patch_ln = nn.LayerNorm(hidden)
        self.patch_head = nn.Linear(hidden, 2, bias=True)     # W_p, b_p
        self.evidence_head = nn.Linear(hidden, 1, bias=True)  # w_e, b_e
        # Neutral evidence init: E = Softplus(0) = ln(2) ≈ 0.693 for every patch.
        # We deliberately do NOT start at E≈0: with the fixed non-saturating fake
        # loss -log(mean_i f_i) (see get_losses), abstaining (E→0 → f_i≈0) now gives
        # a LARGE loss instead of a small one, so there is no incentive to gate
        # evidence off.  Starting at a uniform moderate E keeps f_i in a trainable
        # band (f_i ≈ 0.26·p_iF on the warm-started head) — sane loss, no early
        # gradient explosion, and the head is free to move up or down from there.
        nn.init.zeros_(self.evidence_head.weight)
        nn.init.constant_(self.evidence_head.bias, 0.0)

        # Warm-start the patch head from a trained CLS head (Section 3:
        # W_p ← W_cls, b_p ← b_cls).  `aepa_init_ckpt` points at a B0
        # (CLS baseline) checkpoint whose `head.weight`/`head.bias` seed the
        # shared patch head.  This is what keeps B1/B2/B3 from starting at
        # p_iF ≈ 0.5 (which saturates the existential fake loss for N=256).
        init_ckpt = config.get('aepa_init_ckpt', None)
        if init_ckpt:
            self._init_patch_head_from_cls(init_ckpt)

        # Ablation mode + hyperparameters.
        self.aepa_mode = config.get('aepa_mode', 'b3_evidence')
        if self.aepa_mode not in ('b3_evidence', 'b2_mil', 'b1_pool'):
            raise ValueError(f"unknown aepa_mode: {self.aepa_mode}")
        self.use_evidence = self.aepa_mode == 'b3_evidence'
        self.lambda_f = float(config.get('aepa_lambda_f', 1.0))
        self.eps = float(config.get('aepa_eps', 1e-8))

        # OWTTT adaptive-threshold queue (kept for interface compatibility).
        self.prediction_queue = []

    def _init_patch_head_from_cls(self, ckpt_path: str) -> None:
        """Seed the shared patch head from a B0 CLS-head checkpoint (Section 3).

        B0 (the CLS baseline) stores its readout as ``head.weight`` (2×1024)
        and ``head.bias`` (2); the patch head has the same input/output dims,
        so the paper reuses the CLS head as the initial patch head.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        new = {k.replace('module.', ''): v for k, v in ckpt.items()}
        w = new.get('head.weight')
        b = new.get('head.bias')
        if w is None or b is None:
            raise KeyError(
                f"[aepa] checkpoint {ckpt_path} has no 'head.weight'/'head.bias' "
                f"(keys: {sorted(new.keys())[:8]} ...)"
            )
        with torch.no_grad():
            self.patch_head.weight.copy_(w)
            self.patch_head.bias.copy_(b)
        logger.info(f"[aepa] warm-started patch_head from CLS head: {ckpt_path}")

    # ── Core patch-evidence forward ────────────────────────────────────────

    def _patch_evidence(self, images):
        """Forward images -> per-patch masses and image-level fake quality.

        Returns a dict with:
            cls:      [B, 2]  (argmax = class, softmax[:, 1] monotonic in score)
            prob:     [B]     detection score (s_AEPA for b2/b3, mean p_F for b1)
            feat:     [B, D]  patch-pooled feature (diagnostic / saving)
            p_iR/p_iF:[B, N]  patch softmax directions
            E:        [B, N]  patch evidence strength (b3 only, else None)
            r, f:     [B, N]  real / fake evidence masses
            mean_pF:  [B]     mean patch fake probability (used by b1 loss)
            s_aepa:   [B]     normalized cumulative fake-evidence score
        """
        out = self.backbone(images)
        tokens = out['last_hidden_state']              # [B, N+1, D]
        patches = tokens[:, 1:, :]                     # [B, N, D] (drop CLS)
        h = self.patch_ln(patches)                     # [B, N, D]

        logits = self.patch_head(h)                    # [B, N, 2]
        p = torch.softmax(logits, dim=-1)              # [B, N, 2]
        p_iR = p[..., 0]                               # [B, N]
        p_iF = p[..., 1]                               # [B, N]

        if self.use_evidence:
            a = self.evidence_head(h).squeeze(-1)      # [B, N]
            E = F.softplus(a)                          # [B, N] >= 0
            denom = 2.0 + E
            r_i = E * p_iR / denom                     # [B, N]
            f_i = E * p_iF / denom                     # [B, N]
        else:
            E = None
            r_i = p_iR                                 # B2: no evidence gating
            f_i = p_iF

        N = f_i.size(1)
        log1mf = torch.log(1.0 - f_i + self.eps)       # [B, N]
        s_aepa = -log1mf.mean(dim=1)                   # [B] normalized fake score
        P_F = (1.0 - torch.exp(log1mf.sum(dim=1))).clamp(0.0, 1.0)  # [B]

        mean_pF = p_iF.mean(dim=1)                     # [B]

        # Detection score — Section 7 boxed final score, verbatim:
        #     s(x) = -(1/N) Σ_i log(1 - E_i p_iF/(2+E_i) + ε)
        # i.e. s_AEPA.  (P_F = 1 - exp(-N·s_AEPA) saturates to 1.0 in float32
        # for N=256 patches, collapsing AUC via ties, so the paper uses s_AEPA
        # directly for AUC.)  B1 uses the symmetric mean-pool score p̄_F.
        if self.aepa_mode == 'b1_pool':
            prob = mean_pF
        else:
            prob = s_aepa

        feat = patches.mean(dim=1)                     # [B, D] patch-pooled
        return {
            'cls': torch.stack([1.0 - prob, prob], dim=1),
            'prob': prob,
            'feat': feat,
            'p_iR': p_iR,
            'p_iF': p_iF,
            'E': E,
            'r': r_i,
            'f': f_i,
            'P_F': P_F,
            'mean_pF': mean_pF,
            's_aepa': s_aepa,
        }

    def forward(self, data_dict, inference=False):
        images = data_dict['image']

        # Multi-crop test input [B, n_crops, C, H, W] -> average crop scores.
        if len(images.shape) == 5:
            b, n, c, h, w = images.shape
            flat = images.reshape(-1, c, h, w)
            out = self._patch_evidence(flat)
            prob = out['prob'].view(b, n).mean(dim=1)
            feat = out['feat'].view(b, n, -1).mean(dim=1)
            return {
                'cls': torch.stack([1.0 - prob, prob], dim=1),
                'prob': prob,
                'feat': feat,
                # averaged over crops so get_losses stays well-defined.
                'r': out['r'].view(b, n, -1).mean(dim=2),
                'f': out['f'].view(b, n, -1).mean(dim=2),
                'P_F': out['P_F'].view(b, n).mean(dim=1),
                'mean_pF': out['mean_pF'].view(b, n).mean(dim=1),
            }

        return self._patch_evidence(images)

    # ── AbstractDetector-style helpers (unused by AEPA, kept coherent) ─────

    def features(self, data_dict):
        out = self.backbone(data_dict['image'])
        return out['last_hidden_state'][:, 1:, :].mean(dim=1)  # [B, D]

    def classifier(self, features):
        return self.patch_head(self.patch_ln(features))         # [B, 2]

    # ── Asymmetric evidential loss ─────────────────────────────────────────

    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label']

        if self.aepa_mode == 'b1_pool':
            # B1: symmetric mean-pooling cross-entropy.
            pbar_F = pred_dict['mean_pF']                       # [B]
            pbar_R = 1.0 - pbar_F
            log_R = torch.log(pbar_R + self.eps)
            log_F = torch.log(pbar_F + self.eps)
            y = label.float()
            ell = -(1.0 - y) * log_R - y * log_F               # [B]
        else:
            # B2/B3: universal (real) + existential (fake) aggregation.
            r_i = pred_dict['r']                               # [B, N]
            ell_R = -torch.log(r_i + self.eps).mean(dim=1)     # [B]
            # Fake (existential) loss — §5.2/§6.  Paper uses -log(P_F + ε) with
            # P_F = 1 - Π(1-f_i); for N=256 patches that product saturates to ~1
            # the moment mean f_i ≳ 0.02 (gradient -Π_{j≠i}(1-f_j)/P_F → 0), so
            # the fake branch never trains → collapse to all-real.  Measured on a
            # true batch (smoke test [5]): paper_logPF gives grad|patch_head| ≈
            # 1.9e-10 (dead), while -log(max f) gives 7.96.  But v3 showed max
            # MISALIGNMENT: the score s_AEPA = -(1/N)Σ log(1-f_i+ε) is a *mean*
            # over all N patches, so pushing the single max patch barely moves a
            # 256-patch mean → v3 B3 AUC collapsed to 0.795.
            # Use instead the dense, score-aligned surrogate
            #   ell_F = -log(mean_i f_i + ε).
            # It is dense (gradient covers every patch → balances the mean-pooled
            # real loss, no N:1 starvation), non-saturating, monotone in the mean
            # score (∂s_AEPA/∂f_i = (1/N)/(1-f_i) > 0), so the training objective
            # is aligned with the ranking metric actually reported (video_auc).
            f = pred_dict['f']                                 # [B, N]
            fmean = f.mean(dim=1)                              # [B]
            ell_F = -torch.log(fmean + self.eps)               # [B]

        mask_real = label == 0
        mask_fake = label == 1

        if self.aepa_mode == 'b1_pool':
            if mask_real.sum() > 0:
                loss_real = ell[mask_real].mean()
            else:
                loss_real = ell.new_tensor(0.0)
            if mask_fake.sum() > 0:
                loss_fake = ell[mask_fake].mean()
            else:
                loss_fake = ell.new_tensor(0.0)
        else:
            if mask_real.sum() > 0:
                loss_real = ell_R[mask_real].mean()
            else:
                loss_real = ell_R.new_tensor(0.0)
            if mask_fake.sum() > 0:
                loss_fake = ell_F[mask_fake].mean()
            else:
                loss_fake = ell_F.new_tensor(0.0)

        loss = loss_real + self.lambda_f * loss_fake
        return {'overall': loss, 'real_loss': loss_real, 'fake_loss': loss_fake}

"""Learnable Forgery Evidence Query read-out detector (Experiment G18).

Replaces the baseline pooler->linear read-out of the frozen CLIP ViT-L/14 + LoRA
backbone with the LFEQ module (``lfeq_module.py``): a learnable decision token
plus K learnable evidence tokens that cross-attend to the RAW patch tokens
(``last_hidden_state[:, 1:, :]``). The image-level score is

    prob = fusion_weight * P_global + (1 - fusion_weight) * P_selected_evidence

where ``P_selected_evidence`` is the fake probability of the single evidence
token with the LARGEST fake probability (hard argmax, label-independent — the
LFEQ maximum-evidence branch).  Exactly one input is changed versus the
``effort`` baseline (which reads ``pooler_output`` into a linear head): the
read-out.  The frozen backbone, LoRA, sampler, and eval protocol are untouched.

Config keys (all read from the merged config, threaded by ``build_config`` and
propagated to testall via ``arch_keys``):

    lfeq_hidden_dim            : query dim (default 256)
    lfeq_num_evidence_tokens   : K learnable evidence slots (default 8)
    lfeq_depth                 : query-transformer blocks (default 2)
    lfeq_num_heads             : attention heads (default 8)
    lfeq_dropout               : dropout (default 0.1)
    lfeq_fusion_weight         : global-branch weight in [0,1] (default 0.5)
    lfeq_evidence_weight       : lambda_evi on the max-evidence CE (default 1.0)
    lfeq_diversity_weight      : lambda_div on attention diversity (default 0.01)

Loss:  L = L_global + evi_weight * L_evi + div_weight * L_div.

Forward contract (must match the trainer / test.py / testall.py):
    cls  : [B,2] pre-softmax logits = LFEQ global_logits (train-metric / rank-loss)
    prob : [B]   score = fused_probs[:,1]  (the metric that testall reads)
    feat : [B,D] LFEQ global decision feature (used for save_feat)
    lfeq : the full LFEQ forward dict (consumed by get_losses)

5D multi-crop path: test.py hard-codes ``multi_crop=True``, so inference gets
``[B, n_crops, C, H, W]``.  Each crop is run through the backbone + LFEQ, then
aggregated with the base argmax-confidence TAA rule (the SAME aggregation the
baseline effort detector uses), so reported AUC stays directly comparable.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectors import DETECTOR
from .effort_detector import EffortDetector
from .lfeq_module import LearnableForgeryEvidenceQuery
from metrics.base_metrics_class import calculate_metrics_for_train

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='effort_lfeq')
class EffortDetectorLFEQ(EffortDetector):
    """Frozen CLIP-ViT-L/14 + LoRA backbone with an LFEQ read-out head."""

    def __init__(self, config=None):
        config = config if config is not None else {}
        # super().__init__ builds the frozen CLIP + LoRA backbone and the
        # baseline pooler->linear CLS head (self.head).  The LFEQ read-out
        # REPLACES that head; self.head is kept but frozen (never part of the
        # LFEQ score) so it can't be silently decayed by the optimizer.
        super().__init__(config)

        vit_dim = 1024  # CLIP ViT-L/14 feature dimension

        self.lfeq_hidden = int(config.get('lfeq_hidden_dim', 256))
        self.lfeq_num_evi = int(config.get('lfeq_num_evidence_tokens', 8))
        self.lfeq_depth = int(config.get('lfeq_depth', 2))
        self.lfeq_heads = int(config.get('lfeq_num_heads', 8))
        self.lfeq_dropout = float(config.get('lfeq_dropout', 0.1))
        self.lfeq_fusion_weight = float(config.get('lfeq_fusion_weight', 0.5))
        self.lfeq_evidence_weight = float(config.get('lfeq_evidence_weight', 1.0))
        self.lfeq_diversity_weight = float(config.get('lfeq_diversity_weight', 0.01))

        self.lfeq = LearnableForgeryEvidenceQuery(
            vit_dim=vit_dim,
            hidden_dim=self.lfeq_hidden,
            num_evidence_tokens=self.lfeq_num_evi,
            depth=self.lfeq_depth,
            num_heads=self.lfeq_heads,
            dropout=self.lfeq_dropout,
            fusion_weight=self.lfeq_fusion_weight,
        )

        # The inherited pooler->linear head is unused in the LFEQ path.  Freeze
        # it so the optimizer ignores the dead params (no silent weight decay).
        for p in self.head.parameters():
            p.requires_grad = False

    # ── Core forward ────────────────────────────────────────────────────────

    def _lfeq_forward(self, images):
        """4D image batch [B,C,H,W] -> LFEQ outputs + scored prob."""
        out = self.backbone(self._prep_input(images))
        tokens = out['last_hidden_state']              # [B, P+1, D]
        patches = tokens[:, 1:, :]                     # [B, P, D] (no CLS)
        lfeq = self.lfeq(patches)                      # LFEQ forward dict

        prob = lfeq['fused_probs'][:, 1]               # [B] fused score
        cls = lfeq['global_logits']                    # [B, 2]
        feat = lfeq['global_feature']                  # [B, D] decision token feat
        return {'cls': cls, 'prob': prob, 'feat': feat, 'lfeq': lfeq}

    def _lfeq_5d_forward(self, images):
        """5D multi-crop [B, n, C, H, W] -> LFEQ per crop + TAA aggregation.

        Mirrors the base effort 5D path: flatten crops, run backbone + LFEQ per
        crop, then aggregate with the argmax-confidence rule on the per-crop
        fused score, so reported AUC is directly comparable to the baseline.
        """
        b, n, c, h, w = images.shape
        flat = images.view(-1, c, h, w)                # [B*n, C, H, W]
        out = self.backbone(self._prep_input(flat))
        tokens = out['last_hidden_state']              # [B*n, P+1, D]
        patches = tokens[:, 1:, :]                     # [B*n, P, D]
        lfeq = self.lfeq(patches)

        per_crop = lfeq['fused_probs'][:, 1].view(b, n)   # [B, n]
        conf = torch.abs(per_crop - 0.5)
        max_idx = torch.argmax(conf, dim=1)            # [B]
        ar = torch.arange(b, device=images.device)
        final_prob = per_crop[ar, max_idx]             # [B]

        global_logits = lfeq['global_logits'].view(b, n, 2)
        global_feat = lfeq['global_feature'].view(b, n, -1)
        final_cls = global_logits[ar, max_idx, :]      # [B, 2]
        final_feat = global_feat[ar, max_idx, :]       # [B, D]

        # Inference-only: get_losses is never called, so no 'lfeq' key needed.
        return {'cls': final_cls, 'prob': final_prob, 'feat': final_feat}

    def forward(self, data_dict, inference=False):
        images = data_dict['image']

        # Multi-crop test-time augmentation [B, n_crops, C, H, W] (test.py
        # hard-codes multi_crop=True).  Only reached when inference=True AND the
        # input is 5D; training/val forward is always 4D.
        if inference and len(images.shape) == 5:
            return self._lfeq_5d_forward(images)

        return self._lfeq_forward(images)

    # ── Train-time metrics ──────────────────────────────────────────────────

    def get_train_metrics(self, data_dict, pred_dict):
        """Train log AUC/EER/AP must track the SCORED branch, not the global one.

        The LFEQ score is the FUSED branch (``pred_dict['prob']``).  The inherited
        base ``get_train_metrics`` feeds ``pred_dict['cls']`` (= the global decision
        token logits) into ``calculate_metrics_for_train``, which on arms with
        ``fusion_weight != 1`` (L1=0.5, L3=0.0) diverges from the fused score the
        ablation actually reports at test / that drives best-ckpt selection.  This
        is logging-only (ckpt selection reads the test fused ``prob``), but for a
        clean ablation we rebuild 2-col logits whose ``softmax[:,1] == prob`` so the
        train columns track the same score.
        """
        prob = pred_dict['prob']                            # [B] fused score

        if 'label_soft' in data_dict:
            label_for_acc = (data_dict['label_soft'] >= 0.5).long()
        else:
            label_for_acc = data_dict['label']
        pred_label = (prob > 0.5).long()
        correct = (pred_label == label_for_acc).sum().item()
        acc = correct / len(label_for_acc)

        eps = 1e-6
        p = prob.clamp(eps, 1 - eps)                        # guard 0/1 from softmax
        logits = torch.zeros(p.size(0), 2, device=prob.device)
        logits[:, 1] = (p / (1 - p)).log()                  # softmax[:,1] == p
        auc, eer, _, ap = calculate_metrics_for_train(label_for_acc.detach(), logits.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    # ── LFEQ loss ───────────────────────────────────────────────────────────

    def get_losses(self, data_dict, pred_dict):
        """L = L_global + evi_weight * L_evi + div_weight * L_div (LFEQ_README.md).

        The training loss is delegated to the DOCUMENTED ``self.lfeq.compute_loss``
        (global CE on the decision token + CE on the statistically-selected
        evidence token + attention-diversity regulariser) so the training objective
        can never drift from LFEQ_README.md.  Per-class real/fake values, plus the
        component logs, are derived here purely for the harness's logging — with
        no PCGrad (G18 ``optimizer_wrapper: null``) they are never backpropagated.
        """
        label = data_dict['label']                     # [B] long, 0=real / 1=fake

        lfeq = pred_dict['lfeq']
        # Documented training loss (exact). Uses the selected (max-fake) token.
        losses = self.lfeq.compute_loss(
            lfeq, label,
            evidence_weight=self.lfeq_evidence_weight,
            diversity_weight=self.lfeq_diversity_weight,
        )

        # Per-class decomposition (logging-only). matches the base detector.
        global_logits = lfeq['global_logits']          # [B, 2]
        selected_evi_logits = lfeq['selected_evidence_logits']   # [B, 2]
        global_ce = F.cross_entropy(global_logits, label, reduction='none')   # [B]
        evidence_ce = F.cross_entropy(selected_evi_logits, label, reduction='none')  # [B]
        per_sample = global_ce + self.lfeq_evidence_weight * evidence_ce      # [B]

        mask_real = label == 0
        mask_fake = label == 1
        loss_real = per_sample[mask_real].mean() if mask_real.any() \
            else per_sample.new_tensor(0.0)
        loss_fake = per_sample[mask_fake].mean() if mask_fake.any() \
            else per_sample.new_tensor(0.0)

        return {
            'overall': losses['loss'],
            'real_loss': loss_real,
            'fake_loss': loss_fake,
            'loss_global': losses['global_loss'],
            'loss_evidence': losses['evidence_loss'],
            'loss_diversity': losses['diversity_loss'],
        }

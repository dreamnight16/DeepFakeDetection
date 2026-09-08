"""Base detector for the G19 token read-out heads (shared A/B/C machinery).

Sibling of ``effort_detector_lfeq`` (G18).  It keeps the frozen CLIP ViT-L/14 +
LoRA backbone and the LFEQ query-transformer body (decision token + K evidence
token, self/cross-attention over the RAW patch tokens), but replaces the LFEQ
read-out (head split + hard-argmax + fusion + diversity loss) with a single
``TokenReadout`` selected by the ``readout_mode`` class attribute:

    'mean'      A   single Linear over the mean of ALL query tokens
    'per_token' B   (K+1) distinct Linear heads, soft-averaged logits
    'concat'    C   Linear over the full concatenated query vector

Common contract reported to the training loop:
    cls  : [B,2] pre-softmax logits = the (aggregated) single-head logits
    prob : [B]   score = softmax(logits)[:,1]  (metric testall reads)
    feat : [..]  the read-out representation (used for save_feat)

Exactly one input is changed vs the ``effort`` baseline: the read-out.  The
backbone, LoRA, sampler, and eval protocol are untouched.  Because each mode
produces a single [B,2] logits, the SCORE IS the single-head logits, so the
inherited ``get_losses`` (CE on ``cls``) and ``get_train_metrics`` (auc from
``cls``) apply unchanged — nothing to override.

Config keys (threaded by ``build_config`` and propagated to testall via
``arch_keys`` — the SAME keys the LFEQ detector uses, since the query-transformer
body is parameter-identical):
    lfeq_hidden_dim          : query dim (default 256)
    lfeq_num_evidence_tokens : K learnable evidence slots (default 8)
    lfeq_depth               : query-transformer blocks (default 2)
    lfeq_num_heads           : attention heads (default 8)
    lfeq_dropout             : dropout (default 0.1)

``readout_mode`` is NOT a config key: it is fixed per detector subclass and
selected via ``model_name`` at build time, so ``arch_keys`` (which carries
``model_name``) automatically rebuilds the identical head at test.

The 4D path is the training forward; the 5D path is the live testall inference
path (the eval batches are stacked as [B, n_crops, C, H, W]).  The 5D branch
mirrors the base TAA (argmax-confidence) aggregation on the aggregated score, so
reported AUC stays directly comparable to the ``effort`` baseline.
"""
import torch

from .effort_detector import EffortDetector
from .token_readout import TokenReadout


class EffortDetectorLFEQReadoutBase(EffortDetector):
    """Frozen CLIP-ViT-L/14 + LoRA backbone with a G19 token read-out.

    Concrete arms subclass this and set ``readout_mode`` ('mean' | 'per_token' |
    'concat'); each is registered under its own module_name by its subclass.
    This base is NOT registered — it is only the shared machinery.
    """

    readout_mode = 'mean'

    def __init__(self, config=None):
        config = config if config is not None else {}
        # super().__init__ builds the frozen CLIP + LoRA backbone and the
        # baseline pooler->linear CLS head (self.head).  The token read-out
        # REPLACES that head; self.head is kept but frozen so it can't silently
        # decay, exactly as in the LFEQ detector (it never contributes a score).
        super().__init__(config)

        vit_dim = 1024  # CLIP ViT-L/14 feature dimension

        self.lfeq_hidden = int(config.get('lfeq_hidden_dim', 256))
        self.lfeq_num_evi = int(config.get('lfeq_num_evidence_tokens', 8))
        self.lfeq_depth = int(config.get('lfeq_depth', 2))
        self.lfeq_heads = int(config.get('lfeq_num_heads', 8))
        self.lfeq_dropout = float(config.get('lfeq_dropout', 0.1))

        self.readout = TokenReadout(
            vit_dim=vit_dim,
            hidden_dim=self.lfeq_hidden,
            num_evidence_tokens=self.lfeq_num_evi,
            depth=self.lfeq_depth,
            num_heads=self.lfeq_heads,
            dropout=self.lfeq_dropout,
            readout_mode=self.readout_mode,
        )

        # The inherited pooler->linear head is unused in the token read-out path.
        for p in self.head.parameters():
            p.requires_grad = False

    def _mean_token_forward(self, images):
        """4D image batch [B,C,H,W] -> token read-out + scored prob."""
        out = self.backbone(self._prep_input(images))
        tokens = out['last_hidden_state']              # [B, P+1, D]
        patches = tokens[:, 1:, :]                     # [B, P, D] (no CLS)
        res = self.readout(patches)
        return {'cls': res['logits'], 'prob': res['prob'], 'feat': res['pooled']}

    def _mean_token_5d_forward(self, images):
        """5D multi-crop [B, n, C, H, W] -> token read-out + TAA aggregation.

        Mirrors the base effort 5D path (argmax-confidence), so reported AUC is
        directly comparable.  Reached only if a config ever enables multi-crop.
        """
        b, n, c, h, w = images.shape
        flat = images.view(-1, c, h, w)                # [B*n, C, H, W]
        out = self.backbone(self._prep_input(flat))
        tokens = out['last_hidden_state']              # [B*n, P+1, D]
        patches = tokens[:, 1:, :]                     # [B*n, P, D]
        res = self.readout(patches)

        per_crop = res['prob'].view(b, n)              # [B, n]
        conf = torch.abs(per_crop - 0.5)
        max_idx = torch.argmax(conf, dim=1)            # [B]
        ar = torch.arange(b, device=images.device)
        final_prob = per_crop[ar, max_idx]             # [B]

        logits = res['logits'].view(b, n, 2)
        pooled = res['pooled'].view(b, n, -1)
        final_cls = logits[ar, max_idx, :]             # [B, 2]
        final_feat = pooled[ar, max_idx, :]            # [B, D]

        return {'cls': final_cls, 'prob': final_prob, 'feat': final_feat}

    def forward(self, data_dict, inference=False):
        images = data_dict['image']

        if inference and len(images.shape) == 5:
            return self._mean_token_5d_forward(images)

        return self._mean_token_forward(images)

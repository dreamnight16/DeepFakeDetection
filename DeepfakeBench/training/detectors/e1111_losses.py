"""E1111B: ordinary batch CE plus the user-specified nonnegative PU risk."""

import math

from torch import nn
from torch.nn import functional as F


class nnPUCEWithReg(nn.Module):
    def __init__(self, prior=0.3, lam=0.5):
        super().__init__()
        if not math.isfinite(prior) or not 0 < prior < 1:
            raise ValueError("prior must be finite and in (0, 1)")
        if not math.isfinite(lam) or lam < 0:
            raise ValueError("lam must be finite and nonnegative")
        self.prior, self.lam = prior, lam

    def forward(self, logits, labels):
        if logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] == 0:
            raise ValueError("Expected nonempty binary logits with shape (N, 2)")
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError("Expected one hard label per sample")
        ce = F.cross_entropy(logits, labels)
        if self.lam == 0:
            return ce
        log_p_neg = F.log_softmax(logits, dim=1)[:, 1]
        real, unl = labels == 0, labels == 1
        if not real.any() or not unl.any():
            raise ValueError("PU regularization needs both Real and Generated in each training batch")
        neg_risk = -log_p_neg[unl].mean() + self.prior * log_p_neg[real].mean()
        return ce + self.lam * neg_risk.clamp(min=0)

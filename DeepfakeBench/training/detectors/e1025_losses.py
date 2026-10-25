"""E1025 logistic nnPU risk with the requested direct non-negative clamp."""

from torch import nn
from torch.nn import functional as F


class NNPULoss(nn.Module):
    def __init__(self, prior=.3):
        super().__init__()
        if not 0 < prior < 1:
            raise ValueError("nnPU prior must be in (0, 1)")
        # This is the real-class prior inside U, not the batch's real ratio.
        self.prior = prior

    def forward(self, logits, labels):
        scores = logits[:, 0] - logits[:, 1]
        real_scores = scores[labels == 0]
        gen_scores = scores[labels == 1]
        if real_scores.numel() == 0 or gen_scores.numel() == 0:
            raise ValueError("Each nnPU training batch must contain Real and Generated samples")
        positive_risk = self.prior * F.softplus(-real_scores).mean()
        negative_risk = (F.softplus(gen_scores).mean()
                         - self.prior * F.softplus(real_scores).mean())
        return positive_risk + negative_risk.clamp(min=0.)

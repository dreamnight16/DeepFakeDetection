"""Numerical contracts: real torch, tiny synthetic tensors, CPU only."""

import copy
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))


class TinyModel(torch.nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(3, 2)

    def forward(self, data):
        return {"cls": self.head(self.drop(data["image"].mean(dim=(-1, -2))))}


class NumericalContracts(unittest.TestCase):
    def test_video_logits_match_probability_mean_and_shift_invariance(self):
        from g21.losses import video_logits

        torch.manual_seed(7)
        z = torch.randn(4, 2, 2, 3, 2, dtype=torch.float64, requires_grad=True)
        u = video_logits(z)
        expected = torch.logit(torch.softmax(z, -1)[..., 1].mean(-1))
        torch.testing.assert_close(u, expected, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(video_logits(z + torch.randn_like(z[..., :1])), u)
        u.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_extremes_are_finite(self):
        from g21.losses import video_logits, classification_loss, pairwise_losses

        z = torch.tensor(
            [[[[[0.0, 1000.0], [0.0, 1000.0]], [[0.0, -1000.0], [0.0, -1000.0]]]]],
            requires_grad=True,
        )
        u = video_logits(z)
        loss = classification_loss(u) + pairwise_losses(u).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_ranking_both_sides_and_permutation(self):
        from g21.losses import pairwise_losses

        u = torch.zeros(4, 2, 2, requires_grad=True)
        pairwise_losses(u).sum().backward()
        self.assertTrue((u.grad[..., 0] > 0).all())
        self.assertTrue((u.grad[..., 1] < 0).all())
        torch.testing.assert_close(
            pairwise_losses(u), pairwise_losses(u, torch.arange(4))
        )

    def test_group_risk_equal_and_singleton(self):
        from g21.losses import group_risk

        r = torch.full((3, 2), 0.7, dtype=torch.float64, requires_grad=True)
        value, w = group_risk(r, "group_softmax", 0.5)
        torch.testing.assert_close(value, r.mean())
        torch.testing.assert_close(w, torch.full_like(r, 1 / 6))
        self.assertAlmostEqual(
            group_risk(r[:1, :1], "group_softmax", 0.5)[0].item(), 0.7
        )

    def test_replay_matches_full_graph_with_dropout_all_arms(self):
        from g21.losses import (
            video_logits,
            classification_loss,
            pairwise_losses,
            group_risk,
        )
        from g21.replay import window_backward

        torch.set_num_threads(1)
        for arm in "ABCD":
            torch.manual_seed(3)
            model = TinyModel(0.3)
            replay_model = copy.deepcopy(model)
            windows = [torch.randn(4, 2, 2, 2, 3, 4, 4) for _ in range(3)]
            perms = [torch.tensor([1, 2, 3, 0]) for _ in windows]
            rng = torch.get_rng_state()
            ce, risks = [], []
            for j, x in enumerate(windows):
                z = model({"image": x.reshape(-1, 3, 4, 4)})["cls"].reshape(
                    4, 2, 2, 2, 2
                )
                u = video_logits(z)
                ce.append(classification_loss(u))
                risks.append(
                    pairwise_losses(u, perms[j] if arm == "C" else None).mean(0)
                )
            expected = torch.stack(ce).mean()
            if arm != "A":
                expected = (
                    expected
                    + group_risk(
                        torch.stack(risks),
                        "group_softmax" if arm == "D" else "mean",
                        0.5,
                    )[0]
                )
            expected.backward()
            after_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            result = window_backward(
                replay_model,
                windows,
                arm=arm,
                permutations=perms,
                margin=0.0,
                temperature=0.5,
            )
            self.assertAlmostEqual(result["loss"], expected.item(), places=5)
            for a, b in zip(model.parameters(), replay_model.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4)
            self.assertTrue(torch.equal(after_rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()

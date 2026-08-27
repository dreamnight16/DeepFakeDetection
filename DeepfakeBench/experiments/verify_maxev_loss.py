"""Standalone pure-torch verification of the Maximum Fake-Evidence Selection Loss.

Replicates the algorithm from ``最大伪造证据选择损失.md`` (Section 15 PyTorch, and
the gradient behaviour claimed in Section 13) *without* the detector class, so it
runs anywhere torch is available (no CLIP / datasets / tensorboard).  It checks:

  [1] Selection correctness     : argmax_i q_{i,1} selects the highest fake prob.
  [2] Logit-margin equivalence  : argmax_i q_{i,1} == argmax_i (a_{i,1} - a_{i,0})
                                  (Section 15 last equation).
  [3] Loss formula (Sections 10/15): L = CE(cls,y) + lambda_max * CE(a_{i*},y), and
                                  a hard max over the patch softmax equals the
                                  same selection as over the logits.
  [4] Gradient sparsity (Section 13): backward of L_max sends gradient ONLY to the
                                  selected patch; non-selected patches get exactly 0.
  [5] Asymmetric directions (Sections 8-9): on a fake image the selected patch's
                                  fake logit is pushed UP (q_max -> 1); on a real
                                  image it is pushed DOWN (q_max -> 0).

Run (plain torch):
    python experiments/verify_maxev_loss.py
"""
import torch
import torch.nn.functional as F


def compute(cls_logits, patch_logits, labels, lambda_max=1.0):
    """The method verbatim (Section 15)."""
    loss_cls = F.cross_entropy(cls_logits, labels)

    patch_prob = torch.softmax(patch_logits, dim=-1)   # [B, N, 2]
    fake_prob = patch_prob[..., 1]                     # [B, N]
    max_index = fake_prob.max(dim=1)[1]                # [B]

    batch_index = torch.arange(patch_logits.size(0),
                               device=patch_logits.device)
    selected_logits = patch_logits[batch_index, max_index, :]   # [B, 2]
    loss_max = F.cross_entropy(selected_logits, labels)

    loss_total = loss_cls + lambda_max * loss_max
    return loss_total, loss_cls, loss_max, max_index, fake_prob


def main():
    torch.manual_seed(0)
    B, N = 8, 16
    ok = True

    def chk(cond, msg):
        nonlocal ok
        flag = 'PASS' if cond else 'FAIL'
        if not cond:
            ok = False
        print(f"    [{flag}] {msg}")
        return cond

    print("=" * 64)
    print("  Maximum Fake-Evidence Selection Loss — pure-torch verification")
    print("=" * 64)

    cls_logits = torch.randn(B, 2)
    patch_logits = torch.randn(B, N, 2, requires_grad=True)
    labels = torch.randint(0, 2, (B,)).long()

    total, loss_cls, loss_max, max_index, fake_prob = compute(
        cls_logits, patch_logits, labels)

    print("\n[1] SELECTION CORRECTNESS")
    gathered_fake = fake_prob[torch.arange(B), max_index]
    per_row_max = fake_prob.max(dim=1)[0]
    chk(torch.allclose(gathered_fake, per_row_max, atol=1e-6),
        f"selected fake prob == row max  (err="
        f"{float((gathered_fake - per_row_max).abs().max()):.2e})")
    print(f"    mean selected q_max = {float(gathered_fake.mean()):.4f}")

    print("\n[2] LOGIT-MARGIN == SOFTMAX-SELECTION EQUIVALENCE")
    margin = patch_logits[..., 1] - patch_logits[..., 0]   # [B, N]
    # monotone: softmax ratio q_{i,1}/q_{i,0} = exp(a1 - a0); equal argmaxes
    margin_idx = margin.argmax(dim=1)
    chk(torch.equal(margin_idx, max_index),
        f"argmax over (a1-a0) == argmax over q_{'1'}  "
        f"(mismatch={int((margin_idx != max_index).sum())})")

    # independence: reconstruction from a single selected patch matches direct CE
    print("\n[3] LOSS FORMULA")
    chk(total.ndim == 0 and torch.isfinite(total),
        f"L_total scalar & finite = {float(total):.4f} "
        f"(L_cls={float(loss_cls):.4f}, L_max={float(loss_max):.4f})")
    # L_max should be the CE on the SELECTED patch only (not a mean of all)
    batch_idx = torch.arange(B)
    sel = patch_logits[batch_idx, max_index]
    chk(torch.allclose(loss_max, F.cross_entropy(sel, labels), atol=1e-5),
        "L_max == CE(selected patch logits, y) exactly")

    print("\n[4] GRADIENT SPARSITY (Section 13)")
    patch_logits.retain_grad()
    total.backward()
    g = patch_logits.grad                       # [B, N, 2]
    # zero out the selected positions, count remaining non-zero
    g_sel = g[batch_idx, max_index, :]
    # build a mask: non-selected -> must be exactly 0
    mask = torch.zeros_like(g, dtype=torch.bool)
    mask[batch_idx, max_index, :] = True
    non_sel = g[~mask]
    chk(non_sel.abs().max().item() < 1e-6,
        f"non-selected patch gradients exactly ~0 (max |g| = {float(non_sel.abs().max()):.2e})")
    chk(g_sel.abs().sum().item() > 1e-6,
        "selected patch receives nonzero gradient")
    # sparsity ratio
    sparsity = (g.abs() > 0).float().mean().item()
    print(f"    frac of patch-logit entries with nonzero grad = {sparsity:.4f} "
          f"(expected ~{1.0 / N:.4f}, i.e. ~1 selected patch of N)")

    print("\n[5] ASYMMETRIC DIRECTIONS (Sections 8-9)")
    # L_cls depends only on the CLS logits, so dL patch_logits == dL_max / d patch.
    # Under gradient descent, dL/d a_i*,1 < 0 raises the fake logit (q_max -> 1,
    # fake y=1); dL/d a_i*,1 > 0 lowers it (q_max -> 0, real y=0).
    for y, name, want_up in [(1, 'fake (y=1)', True), (0, 'real (y=0)', False)]:
        lbl = torch.tensor([y]).long()
        pl = torch.randn(1, N, 2, requires_grad=True)
        cl = torch.randn(1, 2, requires_grad=True)
        L = compute(cl, pl, lbl)[0]
        L.backward()
        i_star = compute(cl, pl, lbl)[3][0]      # argmax_i q_{i,1}
        g = pl.grad[0, i_star, 1]                # dL / d a_{i*,fake}
        pushes_up = g < 0                        # negative grad => rises under descent
        direction = 'UP -> q_max->1' if pushes_up else 'DOWN -> q_max->0'
        chk(pushes_up == want_up,
            f"{name}: dL/d[a_i*,fake] = {float(g):+.4f}  => {direction}  "
            f"(expected {'UP' if want_up else 'DOWN'})")
    print("    (real image: loss pushes the selected-patch fake logit DOWN;")
    print("     fake image: loss pushes it UP — the non-symmetry of Sections 8-9.)")

    print("\n" + ("=" * 64))
    print("  RESULT: " + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    print("=" * 64)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

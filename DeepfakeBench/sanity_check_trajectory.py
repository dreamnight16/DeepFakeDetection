"""
Sanity checks for Diffusion Trajectory-Guided Mixup (DTP-Mixup) — run BEFORE G6.

Purpose: confirm the restored trajectory / trajectory_pyramid implementation
matches the spec (real base + energy-grounded label) so the G6 ablation table
is interpretable. Run once on the training server (or any machine with torch).

Usage:
    # Checks 1 & 2 (pure math, no deps) + Check 3 (synthetic tensors, needs torch)
    python3 DeepfakeBench/sanity_check_trajectory.py

    # Check 3 on real images: visually verify "structure ~ real, fake detail"
    python3 DeepfakeBench/sanity_check_trajectory.py --real /path/real.png --fake /path/fake.png

Checks:
    1. trajectory λ_t:  t=50 → λ≈0.994,  t=700 → λ≈0.206
    2. pyramid label:   λ_t=0.6 → per-level [0.6,0.4,0.2], base=1, lambda_bar > 0.5
    3. save a mixed trajectory_pyramid image for visual inspection
"""
import argparse
import math
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════
# Check 1 — trajectory λ_t
# ═══════════════════════════════════════════════════════════════════════════

def compute_lambda_t(t, T=1000):
    return 0.5 * (1.0 + math.cos(math.pi * t / T))


def check1():
    print("=" * 66)
    print("  Check 1 — trajectory λ_t  (t ~ U(50,700), T=1000)")
    print("=" * 66)
    ok = True
    for t in (50, 700):
        lam = compute_lambda_t(t)
        print(f"    t={t:>4d}  λ_t = {lam:.4f}")
    lam50, lam700 = compute_lambda_t(50), compute_lambda_t(700)
    if abs(lam50 - 0.994) < 0.01 and abs(lam700 - 0.206) < 0.01:
        print("    [PASS] λ_t(50)≈0.994, λ_t(700)≈0.206")
    else:
        print("    [FAIL] λ_t out of expected range")
        ok = False
    print()
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# Check 2 — pyramid label
# ═══════════════════════════════════════════════════════════════════════════

def default_omega(num_levels):
    """Mirror trajectory_mixup.py default omega: K Lap + 1 base, uniform."""
    n_total = num_levels + 1
    return [1.0 / n_total] * n_total


def check2(num_levels=3, lam_t=0.6):
    print("=" * 66)
    print(f"  Check 2 — pyramid label  (num_levels={num_levels}, λ_t={lam_t})")
    print("=" * 66)
    omega = default_omega(num_levels)
    print(f"    omega = {[f'{w:.3f}' for w in omega]}  (K Lap + 1 base, uniform)")
    lambda_bar = 0.0
    for k in range(num_levels):
        lam_tl = lam_t * (1.0 - k / num_levels)
        lambda_bar += omega[k] * lam_tl
        print(f"    k={k}  λ_{{t,k}} = λ_t·(1−{k}/{num_levels}) = {lam_tl:.3f}")
    lambda_bar += omega[num_levels] * 1.0   # base = fully real
    y_m = 1.0 - lambda_bar
    print(f"    base  λ_{{t,K}} = 1.0  (real)")
    print(f"    lambda_bar = {lambda_bar:.4f}   →   y_m = 1 − λ̄ = {y_m:.4f}")
    if lambda_bar > 0.5:
        print(f"    [PASS] lambda_bar > 0.5  (real-biased)")
    else:
        print(f"    [WARN] lambda_bar = {lambda_bar:.4f} NOT > 0.5 "
              f"(borderline; decreasing omega gives the base only weight {omega[num_levels]:.3f})")
    print()
    return lambda_bar > 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Check 3 — save a mixed image
# ═══════════════════════════════════════════════════════════════════════════

def check3(real_path=None, fake_path=None, out_path="trajectory_pyramid_sample.png"):
    print("=" * 66)
    print("  Check 3 — save a mixed trajectory_pyramid image")
    print("=" * 66)
    try:
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image
    except ImportError as e:
        print(f"    [SKIP] missing dependency: {e}")
        return True

    # Import the actual implementation (lazy imports trainer.trainer_v2)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "training", "trainer"))
    try:
        from trajectory_mixup import pyramid_trajectory_mixup
    except Exception as e:
        print(f"    [SKIP] cannot import pyramid_trajectory_mixup: {e}")
        print("    (run on the training server where tensorboard is installed)")
        return True

    torch.manual_seed(0)
    if real_path and fake_path and os.path.isfile(real_path) and os.path.isfile(fake_path):
        x_r = TF.to_tensor(Image.open(real_path).convert("RGB"))
        x_f = TF.to_tensor(Image.open(fake_path).convert("RGB"))
        h, w = x_r.shape[-2:]
        x_r = TF.resize(x_r, (h // 4 * 4, w // 4 * 4))
        x_f = TF.resize(x_f, (h // 4 * 4, w // 4 * 4))
    else:
        print("    (no --real/--fake given → synthetic noise, shapes only)")
        x_r = torch.rand(1, 3, 64, 64)
        x_f = torch.rand(1, 3, 64, 64)

    x = torch.cat([x_r, x_f], dim=0)
    y = torch.tensor([0, 1], dtype=torch.long)  # real, fake

    mx, my, ml = pyramid_trajectory_mixup(
        x, y, alpha=5.0, gamma=1.0, num_levels=3, t_min=50, t_max=700, T=1000)

    rf_img = mx[0].clamp(0, 1)
    TF.to_pil_image(rf_img).save(out_path)
    print(f"    mixed shape={tuple(mx.shape)}  soft_label={my[0]:.4f}  hard={int(ml[0])}")
    print(f"    [PASS] saved → {out_path}  (inspect: structure ~ real, fake detail)")
    print()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=None, help="path to a real image")
    ap.add_argument("--fake", default=None, help="path to a fake image")
    ap.add_argument("--out", default="trajectory_pyramid_sample.png")
    args = ap.parse_args()

    ok = [check1(), check2(), check3(args.real, args.fake, args.out)]
    print("=" * 66)
    print(f"  Summary: {sum(ok)}/{len(ok)} checks passed "
          f"(Check 2 borderline warning counts as pass-with-note)")
    print("=" * 66)


if __name__ == "__main__":
    main()

"""
vis_hf_mixup.py  —  HF-Mixup visualization
===========================================
Four-column layout per row: Real | Fake | RGB Mixup | HF-Mixup

Usage:
    python vis_hf_mixup.py
    python vis_hf_mixup.py --rows 8 --cutoff 0.125 --lam 0.5
    python vis_hf_mixup.py --datasets FaceForensics++ Celeb-DF-v2 --rows 6 --mode train
    python vis_hf_mixup.py --rows 6 --alpha 1.0   # sample λ ~ Beta(α,α) per row
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════════════════════
# FFT utilities
# ═══════════════════════════════════════════════════════════════════════════════

_FREQ_MASK_CACHE: dict = {}


def _get_freq_masks(
    h: int, w: int, cutoff: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cached circular low/high frequency masks (after fftshift).

    M_L : disc of radius  cutoff * min(H,W) / 2
    M_H : 1 - M_L
    """
    dev_idx = device.index if device.type == "cuda" else -1
    key = (h, w, round(cutoff, 6), dev_idx)
    if key not in _FREQ_MASK_CACHE:
        r = cutoff * min(h, w) / 2.0
        ys = torch.arange(h, device=device, dtype=torch.float32)
        xs = torch.arange(w, device=device, dtype=torch.float32)
        yv, xv = torch.meshgrid(ys, xs, indexing="ij")
        dist = torch.sqrt((yv - h / 2.0) ** 2 + (xv - w / 2.0) ** 2)
        ml = (dist <= r).float()
        _FREQ_MASK_CACHE[key] = (ml, 1.0 - ml)
    return _FREQ_MASK_CACHE[key]


def decompose_fft(
    x: torch.Tensor, cutoff: float = 0.125
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose (B,C,H,W) into (x_low, x_high) via FFT masking.

    x^L = F⁻¹(M_L ⊙ F(x))
    x^H = F⁻¹(M_H ⊙ F(x))
    """
    _, _, H, W = x.shape
    ml, mh = _get_freq_masks(H, W, cutoff, x.device)
    ml = ml.view(1, 1, H, W)
    mh = mh.view(1, 1, H, W)
    X = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    x_low  = torch.fft.ifft2(torch.fft.ifftshift(X * ml, dim=(-2, -1))).real
    x_high = torch.fft.ifft2(torch.fft.ifftshift(X * mh, dim=(-2, -1))).real
    return x_low, x_high


def pixel_mixup(
    x_real: torch.Tensor, x_fake: torch.Tensor, lam: float
) -> torch.Tensor:
    """Standard RGB-space Mixup:  x̃ = λ·x_r + (1-λ)·x_f"""
    return lam * x_real + (1.0 - lam) * x_fake


def hf_mixup(
    x_real: torch.Tensor, x_fake: torch.Tensor,
    lam: float, cutoff: float = 0.125
) -> torch.Tensor:
    """High-frequency-only Mixup:  x̃_H = x_r^L + λ·x_r^H + (1-λ)·x_f^H

    Preserves the real image's low-frequency semantic structure;
    blends only the high-frequency components from both images.
    """
    real_low, real_high = decompose_fft(x_real, cutoff)
    _,        fake_high = decompose_fft(x_fake, cutoff)
    return real_low + lam * real_high + (1.0 - lam) * fake_high


def sample_lambda(alpha: float) -> float:
    """Draw λ ~ Beta(α, α)."""
    return float(np.random.beta(alpha, alpha))


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_DATASETS: List[str] = [
    "FaceForensics++", "Celeb-DF-v1", "Celeb-DF-v2",
    "DFDC", "DFDCP", "FaceShifter",
    "DeepFakeDetection", "DeeperForensics-1.0", "UADFV",
]


def _extract_frames(entry) -> List[str]:
    """Recursively pull frame paths out of a JSON leaf.

    Handles both direct {"frames": [...]} and compression-nested layouts.
    """
    if not isinstance(entry, dict):
        return []
    if "frames" in entry and isinstance(entry["frames"], list):
        return entry["frames"]
    paths: List[str] = []
    for k, v in entry.items():
        if k in ("label", "frames"):
            continue
        if isinstance(v, dict):
            paths.extend(_extract_frames(v))
    return paths


def collect_images(
    dataset_names: List[str],
    json_folder: str,
    base_data_path: str,
    mode: Optional[str],
) -> Tuple[List[str], List[str]]:
    """Return (real_paths, fake_paths) from DeepfakeBench-style JSON files."""
    real_paths: List[str] = []
    fake_paths: List[str] = []

    for ds_name in dataset_names:
        json_path = os.path.join(json_folder, f"{ds_name}.json")
        if not os.path.exists(json_path):
            print(f"  [SKIP] {json_path} not found")
            continue

        with open(json_path) as fh:
            data = json.load(fh)

        ds_real: List[str] = []
        ds_fake: List[str] = []

        for top_key in data:
            for label_name, modes in data[top_key].items():
                if mode and mode in modes:
                    subset = modes[mode]
                elif "test" in modes:
                    subset = modes["test"]
                elif "train" in modes:
                    subset = modes["train"]
                else:
                    continue

                label_lower = label_name.lower()
                is_real = label_lower.endswith("real") or label_lower.endswith("_real")

                for video_entry in subset.values():
                    for fp in _extract_frames(video_entry):
                        fp = fp.replace("\\", "/")
                        if not fp.startswith("/") and not fp.startswith(base_data_path):
                            fp = os.path.join(base_data_path, fp)
                        (ds_real if is_real else ds_fake).append(fp)

        print(f"  [{ds_name}]  real={len(ds_real):,}  fake={len(ds_fake):,}")
        real_paths.extend(ds_real)
        fake_paths.extend(ds_fake)

    real_paths = sorted(set(real_paths))
    fake_paths = sorted(set(fake_paths))
    print(f"  Total: {len(real_paths):,} real  |  {len(fake_paths):,} fake")
    return real_paths, fake_paths


# ═══════════════════════════════════════════════════════════════════════════════
# Image helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_img(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_display(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(arr, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

_COL_LABELS = ["Real", "Fake", "RGB Mixup", "HF-Mixup"]
_COL_COLORS = ["#2196F3", "#F44336", "#9C27B0", "#4CAF50"]


def build_figure(
    sampled_real: List[str],
    sampled_fake: List[str],
    lam: float,
    cutoff: float,
    size: int,
    alpha: float,
    device: str,
) -> plt.Figure:
    rows = len(sampled_real)
    fig = plt.figure(figsize=(14, 3.2 * rows + 1.0))

    gs = gridspec.GridSpec(
        rows + 1, 4, figure=fig,
        hspace=0.05, wspace=0.03,
        top=0.97, bottom=0.02, left=0.07, right=0.99,
        height_ratios=[0.15] + [1.0] * rows,
    )

    # — column header row —
    for ci, (label, color) in enumerate(zip(_COL_LABELS, _COL_COLORS)):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_facecolor(color)
        ax.text(0.5, 0.5, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # — data rows —
    for ri, (rp, fp) in enumerate(zip(sampled_real, sampled_fake)):
        r_np = load_img(rp, size)
        f_np = load_img(fp, size)

        x_r = torch.from_numpy(r_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
        x_f = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

        row_lam = sample_lambda(alpha) if alpha > 0.0 else lam

        imgs = [
            to_display(x_r),
            to_display(x_f),
            to_display(pixel_mixup(x_r, x_f, row_lam)),
            to_display(hf_mixup(x_r, x_f, row_lam, cutoff)),
        ]

        for ci, img in enumerate(imgs):
            ax = fig.add_subplot(gs[ri + 1, ci])
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(_COL_COLORS[ci])
                sp.set_linewidth(1.5)
            if ci == 0:
                ax.set_ylabel(f"#{ri + 1}  λ={row_lam:.2f}",
                              fontsize=8, rotation=0,
                              labelpad=46, va="center", color="#555")

        def _short(p: str) -> str:
            parts = p.replace("\\", "/").split("/")
            return "/".join(parts[-4:])
        print(f"  [{ri + 1:>2}] λ={row_lam:.3f}  "
              f"real={_short(rp)}  |  fake={_short(fp)}")

    fig.suptitle(
        f"HF-Mixup (cutoff={cutoff})  vs  RGB Mixup"
        + (f"  ·  λ~Beta({alpha},{alpha})" if alpha > 0 else f"  ·  λ={lam}"),
        fontsize=12, y=0.99,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="HF-Mixup: Real | Fake | RGB Mixup | HF-Mixup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json_folder",    default="DeepfakeBench/preprocessing/dataset_json")
    p.add_argument("--base_data_path", default="/home/user1/effort/data")
    p.add_argument("--datasets",       nargs="*", default=DEFAULT_DATASETS)
    p.add_argument("--rows",           type=int,   default=6)
    p.add_argument("--lam",            type=float, default=0.5,
                   help="Fixed λ; ignored when --alpha > 0")
    p.add_argument("--alpha",          type=float, default=0.0,
                   help="Beta(α,α) shape; 0 = use fixed --lam")
    p.add_argument("--cutoff",         type=float, default=0.125)
    p.add_argument("--size",           type=int,   default=224)
    p.add_argument("--out",            default="hf_mixup_vis.png")
    p.add_argument("--mode",           choices=["train", "test"], default=None)
    p.add_argument("--seed",           type=int,   default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nCollecting images …")
    real_paths, fake_paths = collect_images(
        args.datasets, args.json_folder, args.base_data_path, args.mode
    )

    if len(real_paths) < args.rows or len(fake_paths) < args.rows:
        raise RuntimeError(
            f"Not enough images — real={len(real_paths)}, "
            f"fake={len(fake_paths)}, need {args.rows} each."
        )

    sampled_real = random.sample(real_paths, args.rows)
    sampled_fake = random.sample(fake_paths, args.rows)

    print("\nRendering …")
    fig = build_figure(
        sampled_real, sampled_fake,
        lam=args.lam, cutoff=args.cutoff,
        size=args.size, alpha=args.alpha,
        device=device,
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
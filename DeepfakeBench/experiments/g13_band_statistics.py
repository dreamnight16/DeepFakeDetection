"""
G13 — pre-training frequency-band statistics diagnostic.

Question this answers *before* spending compute on the 5-band round-1 ablation:

    On the distribution the model is actually trained/tested on (real face crops),
    (a) how much genuine spectral energy lives in each radial band, and
    (b) what do the reconstructed per-band inputs look like (mean / std / RMS)?

Purpose — separate two very different explanations for a low band AUC:

    Frequency problem   : the discriminative signal is simply absent from that
                          band (real info is thin there), OR
    Input-distribution problem : the band reconstruction is statistically
                          unnatural compared to the natural images the frozen
                          CLIP backbone was calibrated on, so the model receives
                          this band poorly for reasons unrelated to deepfake info.

The single most informative number is the **spectral energy fraction** computed on
the RAW crop (before any min-max stretch / clip): the share of the image's total
FFT energy that each band actually contains.  If e.g. the High band carries
~0.8% of a real face's energy, a High-band AUC collapse is plausibly an
information-scarcity / input-distribution effect, NOT evidence that high
frequencies are irrelevant to deepfakes.

It reuses ``DeepfakeAbstractBaseDataset`` (mode='test', no augmentation,
multi_crop off) purely to get the EXACT raw face crop the model sees —
``np.array(ds.load_rgb(path))`` — so the diagnostic is faithful to the real
input pipeline, with zero changes to the model or dataset code.

Usage (run on the data server):
    python3 experiments/g13_band_statistics.py                     # FF++ real crops, 1000 samples
    python3 experiments/g13_band_statistics.py --dataset Celeb-DF-v2 --n_samples 500
    python3 experiments/g13_band_statistics.py --norm none         # raw band amplitude
    python3 experiments/g13_band_statistics.py --band high         # only the High band stats
Output: ./experiment_results/g13_band_statistics/{dataset}/stats.json + a per-band
reconstruction grid + a spectral-energy bar chart.
"""
import os
import sys
import json
import argparse

import numpy as np

_current_dir = os.path.dirname(os.path.abspath(__file__))
_deepfake_dir = os.path.dirname(_current_dir)
sys.path.insert(0, _current_dir)
sys.path.insert(0, _deepfake_dir)

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.utils.freq_band import FREQ_BANDS, _band_mask, apply_freq_band

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _energy_fractions(x: np.ndarray) -> dict[str, float]:
    """Fraction of total FFT energy in each radial band, on the RAW crop.

    x: float64 [H, W, 3] in [0, 255] (natural range, not stretched).
    Computed on the raw spectrum (pre min-max / clip) so it reflects genuine
    signal content, not post-processing.  Returns {band: frac, 'covered': frac}
    where 'covered' is the sum of the 4 bands (== 1 minus the dropped corners).
    """
    h, w = x.shape[:2]
    xf = np.fft.fftshift(np.fft.fft2(x, axes=(0, 1)), axes=(0, 1))  # [H,W,3]
    e2 = (np.abs(xf) ** 2).sum(axis=2)                              # [H,W]
    e_total = float(e2.sum())
    if e_total < 1e-12:
        return {**{k: 0.0 for k in FREQ_BANDS}, 'covered': 0.0}
    out = {}
    for name, (lo, hi) in FREQ_BANDS.items():
        # Reuse the exact cached mask from freq_band so the band definition here
        # is byte-identical to apply_freq_band.
        m = _band_mask(h, w, lo, hi)
        out[name] = float((e2 * m).sum()) / e_total
    out['covered'] = sum(v for k, v in out.items() if k != 'covered')
    return out


def _recon_stats(x0: np.ndarray, band: str | None, norm: str) -> dict:
    """For one raw crop, apply the chosen band and return reconstructed stats."""
    if band is None:
        rec = x0
    else:
        rec = apply_freq_band(x0, band, norm=norm, energy_match=False)
    r = rec.astype(np.float32)
    return {
        'mean': float(r.mean()),
        'std': float(r.std()),
        'rms': float(np.sqrt((r ** 2).mean())),
    }


def _parse_arg_list(s: str | None) -> list[str]:
    if not s:
        return list(FREQ_BANDS.keys())
    return [x.strip() for x in s.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='G13 pre-training band statistics')
    parser.add_argument('--dataset', default='FaceForensics++',
                        help='Dataset whose REAL crops are sampled (default FF++)')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of real face crops to sample (default 1000)')
    parser.add_argument('--mode', choices=['train', 'test', 'val'], default='test',
                        help='Split to draw REAL crops from (default test)')
    parser.add_argument('--norm', choices=['minmax', 'none'], default='minmax',
                        help="Reconstruction norm to report match (default minmax, "
                             "the G13 round-1 default)")
    parser.add_argument('--bands', type=str, default=None,
                        help='Comma-separated band subset (e.g. low,high); default all')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str,
                        default='./experiment_results/g13_band_statistics')
    parser.add_argument('--resolution', type=int, default=224)
    args = parser.parse_args()

    bands = _parse_arg_list(args.bands)
    rng = np.random.default_rng(args.seed)

    # Build a minimal test config with freq_ablation=None so the dataset returns
    # the untouched natural crop (which load_rgb produces).  Reuses the project's
    # config plumbing to point at the right json + resolution.
    from experiment_utils import build_config
    config = build_config(
        pyramid_mode='lap_pyramid', use_mixup=False, mixup_loss_strip=False,
        freq_ablation=None, freq_norm=args.norm, freq_energy_match=False,
        freq_after_aug=True,
        n_epochs=0, train_dataset=args.dataset, test_dataset=args.dataset,
        for_training=False,
    )
    config['resolution'] = args.resolution

    ds = DeepfakeAbstractBaseDataset(config=config, mode=args.mode)

    # Collect REAL (label==0) sample indices; handle image-level entries that may
    # be a single-item list (one frame) per image path.
    real_idx = [i for i in range(len(ds.data_dict['label']))
                if int(ds.data_dict['label'][i]) == 0]
    if not real_idx:
        print(f"No real (label==0) samples in {args.dataset} mode={args.mode}")
        return
    n = min(args.n_samples, len(real_idx))
    idx = rng.choice(real_idx, size=n, replace=False)
    print(f"[stats] {args.dataset} mode={args.mode} real samples={len(real_idx)} "
          f"sampling {n}")

    # Stat accumulators.
    raw_stats = {k: [] for k in ['mean', 'std', 'rms']}             # Raw crop (no band)
    band_recon = {b: {k: [] for k in ['mean', 'std', 'rms']} for b in bands}
    band_energy = {b: [] for b in bands}
    covered_energy = []
    example_grid = {}  # band -> first reconstructed image (for the visual check)

    for ii, i in enumerate(idx):
        path = ds.data_dict['image'][i]
        if isinstance(path, list):
            path = path[0] if path else None
        if path is None:
            continue
        try:
            x0 = np.array(ds.load_rgb(path), dtype=np.float64)  # [res,res,3] uint8->float
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue

        for k_ in ['mean', 'std', 'rms']:
            raw_stats[k_].append(_recon_stats(x0, None, args.norm)[k_])

        e = _energy_fractions(x0)
        for b in bands:
            band_energy[b].append(e[b])
            rs = _recon_stats(x0, b, args.norm)
            for k_ in ['mean', 'std', 'rms']:
                band_recon[b][k_].append(rs[k_])
        covered_energy.append(e['covered'])

        if ii == 0:
            for b in bands:
                example_grid[b] = apply_freq_band(x0.astype(np.uint8), b,
                                                  norm=args.norm, energy_match=False)

    def _agg(vals):
        if not vals:
            return None
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    table = {}
    row_raw = {k: _agg(raw_stats[k]) for k in raw_stats}
    table['raw'] = row_raw
    for b in bands:
        table[b] = {
            'energy_frac': _agg(band_energy[b]),
            'recon': {k: _agg(band_recon[b][k]) for k in ['mean', 'std', 'rms']},
        }
    table['covered_energy_frac'] = _agg(covered_energy)

    # ---- Save stats ----
    out_dir = os.path.join(args.output_dir, args.dataset.replace('/', '_'))
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump({'dataset': args.dataset, 'mode': args.mode, 'n': n,
                   'norm': args.norm, 'bands': bands, 'table': table},
                  f, indent=2, default=str)
    print(f"[stats] saved to {stats_path}")

    # ---- Print report ----
    print(f"\n{'='*78}\n  Band statistics — {args.dataset} real crops "
          f"(n={n}, norm={args.norm})\n{'='*78}")
    print(f"  {'Band':<10s} | {'Energy frac':>11s} | {'recon mean':>10s} | "
          f"{'recon std':>10s} | {'recon RMS':>10s}")
    print(f"  {'-'*10} | {'-'*11} | {'-'*10} | {'-'*10} | {'-'*10}")
    for b in ['raw'] + bands:
        row = table[b] if b != 'raw' else table['raw']
        if b == 'raw':
            ef = '-'
            recon = row
        else:
            ef = f"{row['energy_frac']['mean']*100:.2f}%"
            recon = row['recon']
        print(f"  {b:<10s} | {ef:>11s} | "
              f"{(recon['mean']['mean'] if recon['mean'] else 'N/A'):>10.2f} | "
              f"{(recon['std']['mean'] if recon['std'] else 'N/A'):>10.2f} | "
              f"{(recon['rms']['mean'] if recon['rms'] else 'N/A'):>10.2f}")
    print(f"\n  Covered energy (sum of 4 bands): "
          f"{table['covered_energy_frac']['mean']*100:.2f}%  "
          f"(the rest is the dropped spectral corners)")
    print("\n  Interpretation:")
    print("    * energy_frac  = share of total face energy in that band (RAW, pre-stretch).")
    print("    * recon mean/std/RMS = what the model actually sees after min-max stretch.")
    print("    * A band with tiny energy_frac but large recon std after min-max means")
    print("      the stretch amplifies near-empty content to full contrast -> the model")
    print("      sees an unnatural/OOD input, not a signal-rich band.")

    # ---- Visual: per-band reconstruction grid (1 real face) ----
    if example_grid:
        fig, axes = plt.subplots(1, len(bands), figsize=(3.2 * len(bands), 3.6))
        if len(bands) == 1:
            axes = [axes]
        for ax, b in zip(axes, bands):
            ax.imshow(example_grid[b])
            ax.set_title(f"{b}", fontsize=12)
            ax.axis('off')
        fig.suptitle(f"Per-band reconstruction (1 real crop, norm={args.norm})\n"
                     f"{args.dataset} — RGB → Low → Mid-Low → Mid-High → High",
                     fontsize=13, y=1.02)
        fig.tight_layout()
        grid_path = os.path.join(out_dir, 'recon_grid.png')
        fig.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[stats] reconstruction grid: {grid_path}")

    # ---- Visual: spectral energy bar chart ----
    if bands:
        labels = bands
        means = [table[b]['energy_frac']['mean'] * 100 for b in bands]
        stds = [table[b]['energy_frac']['std'] * 100 for b in bands]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means, yerr=stds, capsize=4, color='#4c72b0', alpha=0.85)
        ax.set_ylabel('% of total face FFT energy')
        ax.set_title(f"Per-band spectral energy content — {args.dataset} real crops (n={n})")
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        echart = os.path.join(out_dir, 'energy_bars.png')
        fig.tight_layout()
        fig.savefig(echart, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[stats] energy bar chart: {echart}")


if __name__ == '__main__':
    main()

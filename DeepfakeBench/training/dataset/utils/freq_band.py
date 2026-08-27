"""
Data-side frequency-band isolation for the G13 ablation.

Keeps only one radial FFT band of an image and reconstructs it back to a
3-channel RGB image.  The model, training protocol, data split and augmentation
are all left completely unchanged — only this input transform differs across
experiments, so RGB vs Low vs Mid-Low vs Mid-High vs High is a clean
single-variable controlled ablation of "which frequency range carries the
deepfake signal".

Frame convention follows the project's existing frequency handling
(``trainer_v2._get_freq_masks``, mix_domain hf/lf): circular radius in
fftshifted coordinates, ``r = dist / (min(H, W) / 2)``, so ``r`` lives
in [0, 1] where 1 = the Nyquist radius.  A band is [lo, hi).

NOTE — this is NOT a byte-exact mirror of ``_get_freq_masks``.  That function
builds ``mask_high = 1 - mask_low`` over the full spectral rectangle, so its
high-frequency split *includes* the four corners (rₙ in (1, √2]).  The G13
band masks here are disk annuli restricted to rₙ <= 1, so the corners are
excluded from *every* band.  Consequences (intentional for round-1, kept for
consistency with the radial convention):

* The four G13 bands do NOT partition/reconstruct the original image exactly —
  corner content exists only in the unfiltered RGB baseline — so RGB vs a band
  is not a strict single-variable split (band-vs-band *is* clean, since all
  four share the same disk-annulus + reconstruction + min-max pipeline).
* Natural face spectra decay steeply at the corners, so the dropped energy is
  small in practice, but the High band is marginally under-counted (it excludes
  the very finest diagonal texture the corners carry).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# Normalized radial-frequency partitions chosen by the G13 protocol.
#   Low       : 0.00 ~ 0.15   (semantic structure / luminance)
#   Mid-Low   : 0.15 ~ 0.35
#   Mid-High  : 0.35 ~ 0.65
#   High      : 0.65 ~ 1.00   (fine texture / artifact)
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "low":      (0.00, 0.15),
    "mid_low":  (0.15, 0.35),
    "mid_high": (0.35, 0.65),
    "high":     (0.65, 1.00),
}

_MASK_CACHE: dict[tuple[int, int, float, float], np.ndarray] = {}


def _band_mask(h: int, w: int, lo: float, hi: float) -> np.ndarray:
    """Radial band-pass mask in fftshifted coordinates, cached per (h, w, lo, hi).

    Returns a binary [H, W] float mask (1 = keep frequency, 0 = remove) with the
    DC at the centre.  ``lo``/``hi`` are normalized radii in [0, 1] (Nyquist = 1).
    """
    key = (h, w, round(lo, 5), round(hi, 5))
    if key not in _MASK_CACHE:
        ys = np.arange(h, dtype=np.float64).reshape(h, 1)
        xs = np.arange(w, dtype=np.float64).reshape(1, w)
        dist = np.sqrt((ys - h / 2.0) ** 2 + (xs - w / 2.0) ** 2)
        rmax = min(h, w) / 2.0
        rn = dist / rmax  # [H, W], r ∈ [0, √2], 1.0 = Nyquist
        _MASK_CACHE[key] = ((rn >= lo) & (rn <= hi)).astype(np.float32)
    return _MASK_CACHE[key]


def apply_freq_band(
    image: np.ndarray,
    band: str | None,
    norm: str = "minmax",
    energy_match: bool = False,
) -> np.ndarray:
    """Band-pass ``image`` (uint8 [H, W, 3]) to a single radial bandwidth.

    ``band`` is one of :data:`FREQ_BANDS` keys, or ``None`` to return the input
    untouched (the RGB baseline).  The reconstructed image is returned as a
    uint8 [H, W, 3] array ready for the normal tensor/normalize pipeline.

    norm
        'minmax'  Per-image global min-max stretch to [0, 255].  Removes the
                  luminance/contrast gap which would otherwise let the model
                  score "this band has different energy" instead of deepfake
                  features (the shortcut the G13 protocol is designed to avoid).
                  This is the recommended / G13 default.
        'none'    Plain clip to [0, 255], keeping each band's raw energy
                  difference.  Diagnostic/control only — a model can exploit
                  the energy caveat, so do not compare it to 'minmax' bands.

    energy_match
        Scale the band-passed spectrum so its L2 norm equals the full image's,
        i.e. ``X' = X_band · ||X||₂ / ||X_band||₂``, making different bands
        comparable in total energy while preserving each band's relative
        spectral shape (the second G13 experiment, "reconstruction + energy
        matching").  Two honest caveats:

        * It is a single global scalar, so it does not change *which*
          frequencies are present — only their amplitude.
        * With ``norm='minmax'`` the per-image stretch that follows re-absorbs
          the global scale, so energy_match is effectively a no-op there.  With
          ``norm='none'`` a small band (mid/high) is amplified ~10-50× and the
          [0,255] clip saturates it, defeating the energy match.  For that
          reason G13 round-1 (RGB vs 4 bands) uses ``norm='minmax'`` and
          keeps ``energy_match=False``; a clean energy-matched variant would
          pre-standardise each band to a fixed RMS (see G13 protocol notes).
    """
    if band is None:
        return image

    lo, hi = FREQ_BANDS[band]
    x = image.astype(np.float64)
    x_fft = np.fft.fftshift(np.fft.fft2(x, axes=(0, 1)), axes=(0, 1))  # [H,W,3]
    mask = _band_mask(x.shape[0], x.shape[1], lo, hi)[..., None]       # [H,W,1]

    x_band = x_fft * mask  # broadcast [H,W,3] * [H,W,1]
    if energy_match:
        eps = 1e-12
        e_full = float(np.linalg.norm(x_fft))
        e_band = float(np.linalg.norm(x_band))
        x_band = x_band * (e_full / (e_band + eps))

    x_rec = np.real(
        np.fft.ifft2(np.fft.ifftshift(x_band, axes=(0, 1)), axes=(0, 1))
    )

    if norm == "minmax":
        lo_v, hi_v = float(x_rec.min()), float(x_rec.max())
        if hi_v - lo_v < 1e-6:
            out = x_rec - lo_v  # degenerate flat band → near-black
        else:
            out = (x_rec - lo_v) / (hi_v - lo_v) * 255.0
    else:  # 'none'
        out = x_rec.copy()

    return np.clip(np.round(out), 0, 255).astype(np.uint8)


# ── G17-2: real-noise (high-pass residual) evidence isolation ──────────────
# High-pass residual of an image, kept as a fixed-scale (NOT per-image min-max)
# 3-channel map.  The fixed 128 + alpha*N map preserves each image's residual
# AMPLITUDE statistics — the very object of study — whereas a per-image min-max
# stretch would re-normalise out the Real/Fake amplitude gap (G13 §6.5 lesson,
# G17-2 protocol §0/§1).  The observer (frozen CLIP + LoRA + head) is unchanged;
# only this input differs, so it is a clean data-side ablation.
_RNG = np.random.default_rng(1234)  # deterministic shuffle across the run


def _shuffle_spatial(arr: np.ndarray) -> np.ndarray:
    """Pixel-permute a [H, W, C] array with one shared spatial permutation.

    Preserves the per-channel histograms (hence mean/variance) exactly, but
    destroys spatial correlation / local structure.  Same permutation applied to
    every channel so per-pixel colour relationships are kept in the multiset —
    only the spatial layout changes (G17-2 arm 07 spatial-structure control).
    """
    h, w, c = arr.shape
    perm = _RNG.permutation(h * w)
    return arr.reshape(h * w, c)[perm].reshape(h, w, c)


def apply_residual(
    image: np.ndarray,
    mode: str = "gauss",
    sigma: float = 2.0,
    alpha: float = 4.0,
    r0: float = 0.65,
    shuffle: bool = False,
) -> np.ndarray:
    """High-pass residual of a uint8 [H, W, 3] image, fixed-mapped to [0, 255].

    mode
        'gauss'     N = I - GaussianBlur_sigma(I)      (smooth structure removed)
        'fft_high'  N = the radial-FFT band r > r0 reconstructed (G13-High linked)
        'none'      N = I (identity / control)
    shuffle
        If True, pixel-permute the residual (breaks spatial correlation while
        preserving amplitude statistics) — the spatial-structure control.
    """
    x = image.astype(np.float64)  # [H, W, 3]

    if mode == "gauss":
        low = np.stack([gaussian_filter(x[..., c], sigma) for c in range(x.shape[2])],
                       axis=-1)
        res = x - low
    elif mode == "fft_high":
        x_fft = np.fft.fftshift(np.fft.fft2(x, axes=(0, 1)), axes=(0, 1))
        mask = _band_mask(x.shape[0], x.shape[1], r0, 1.0)[..., None]
        res = np.real(np.fft.ifft2(np.fft.ifftshift(x_fft * mask, axes=(0, 1)),
                                   axes=(0, 1)))
    else:
        res = x

    if shuffle:
        res = _shuffle_spatial(res)

    out = 128.0 + alpha * res
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

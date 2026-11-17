"""Shared AUC readout plus honest video thresholds and baseline correction counts."""

import math
import numpy as np


def aggregate_video(rows, scores):
    ids, first, inverse, counts = np.unique(rows["video_id"], return_index=True, return_inverse=True, return_counts=True)
    labels = np.asarray(rows["labels"])[first]
    if not np.array_equal(labels[inverse], rows["labels"]):
        raise ValueError("Mixed labels within a full-path video")
    return ids, labels, np.bincount(inverse, weights=np.asarray(scores, dtype=np.float64)) / counts


def calibrate(real_video_scores, alpha):
    values = np.asarray(real_video_scores)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 < alpha < 1:
        raise ValueError("Invalid real calibration scores or alpha")
    index = math.ceil((len(values) + 1) * (1 - alpha))
    # No finite order statistic can support the requested operating point.
    return None if index > len(values) else float(np.sort(values)[index - 1])


def thresholds(rows, scores, alpha=(.1, .05)):
    _, labels, values = aggregate_video(rows, scores)
    if np.any(labels != 0):
        raise ValueError("Threshold calibration may only use real videos")
    return {str(a): calibrate(values, a) for a in alpha}


def fusion_score(baseline_prob, energy, source_real_energy, weight):
    p = np.asarray(baseline_prob)
    if not 0 <= weight <= 1:
        raise ValueError("Fusion weight must be in [0,1]")
    if weight == 0:
        return p.copy()
    reference = np.sort(np.asarray(source_real_energy))
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Need finite source real scores for fixed CDF scaling")
    rank = np.searchsorted(reference, energy, side="right") / (len(reference) + 1)
    return (1 - weight) * p + weight * rank


def subset(rows, mask):
    return {k: np.asarray(v)[mask] for k, v in rows.items()}


def evaluate_scores(rows, scores, fixed_thresholds, baseline_thresholds=None):
    from run_e1001 import readout_metrics
    result = readout_metrics(rows, scores)
    result.pop("corrected_frames", None)
    result.pop("harmed_frames", None)
    result["accuracy_scope"] = "frame score > .5 reference; energy scores are not probabilities"
    _, labels, video = aggregate_video(rows, scores)
    _, _, base = aggregate_video(rows, rows["cls_prob"])
    result["operating_points"] = {}
    for alpha, theta in fixed_thresholds.items():
        if theta is None:
            result["operating_points"][alpha] = {"status": "INSUFFICIENT_CALIBRATION"}
            continue
        pred = video > theta
        real, fake = labels == 0, labels == 1
        point = {"status": "OK", "threshold": theta,
                 "FPR": float(pred[real].mean()) if real.any() else None,
                 "TPR": float(pred[fake].mean()) if fake.any() else None,
                 "num_real_videos": int(real.sum()), "num_fake_videos": int(fake.sum())}
        from scipy.stats import beta
        for mask, key in ((real, "FPR"), (fake, "TPR")):
            n, k = int(mask.sum()), int(pred[mask].sum())
            point[key + "_CP95"] = ([0. if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
                                      1. if k == n else float(beta.ppf(.975, k + 1, n - k))] if n else None)
        bt = (baseline_thresholds or {}).get(alpha)
        if bt is not None:
            correct, base_correct = pred == labels, (base > bt) == labels
            point.update(corrected_videos=int((correct & ~base_correct).sum()),
                         harmed_videos=int((~correct & base_correct).sum()))
        result["operating_points"][alpha] = point
    return result

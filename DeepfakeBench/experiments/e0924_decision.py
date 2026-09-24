"""Small frozen-model decision heads; no external Jev API or visual encoder."""

import numpy as np


FREQ_NAMES = ["low_energy", "mid_energy", "high_energy", "log_high_low",
              "radial_log_slope", "horizontal_vertical_high_difference"]
COLOR_NAMES = ["y_mean", "cb_mean", "cr_mean", "y_std", "cb_std", "cr_std",
               "cb_high_fraction", "cr_high_fraction", "rgb_residual_correlation",
               "y_cb_edge_correlation", "y_cr_edge_correlation", "clipped_pixel_fraction"]


def image_features(rgb):
    """Deterministic statistics of the exact resized RGB crop scored by G27.

    Full 2D FFT, mean removed per channel, radius normalized to the diagonal
    Nyquist frequency. Bands are [0,.25), [.25,.65), [.65,1]. BT.601-style
    floating YCbCr is computed in [0,1] RGB; no band image enters CLIP.
    """
    import torch
    from torch.nn import functional as F

    rgb = rgb.float()
    if rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[-2:]) < 4 or not torch.isfinite(rgb).all():
        raise ValueError("Expected finite [N,3,H>=4,W>=4] RGB")
    if rgb.min() < -1e-5 or rgb.max() > 1.00001:
        raise ValueError("image_features expects denormalized RGB in [0,1]")
    y = .299*rgb[:, 0] + .587*rgb[:, 1] + .114*rgb[:, 2]
    color = torch.stack((y, .5+.564*(rgb[:, 2]-y), .5+.713*(rgb[:, 0]-y)), 1)
    h, w = rgb.shape[-2:]
    fy, fx = torch.meshgrid(torch.fft.fftfreq(h, device=rgb.device),
                            torch.fft.fftfreq(w, device=rgb.device), indexing="ij")
    radius = (fx.square()+fy.square()).sqrt() / (.5**.5)
    masks = [radius < .25, (radius >= .25) & (radius < .65), radius >= .65]

    def power(images):
        centered = images - images.mean((-2, -1), keepdim=True)
        return torch.fft.fft2(centered, norm="ortho").abs().square()

    energy = power(y[:, None])[:, 0]
    total = energy.sum((-2, -1)).clamp_min(1e-12)
    bands = torch.stack([energy[:, mask].sum(-1)/total for mask in masks], -1)
    high_mask_h = masks[2] & (fx.abs() >= fy.abs())
    high_mask_v = masks[2] & (fx.abs() < fy.abs())
    orientation = (energy[:, high_mask_h].sum(-1)-energy[:, high_mask_v].sum(-1))/total
    bin_energy = torch.stack([energy[:, (radius >= i/8) & (radius < (i+1)/8)].mean(-1)
                              for i in range(1, 7)], -1).clamp_min(1e-12)
    log_r = torch.log(torch.arange(1.5, 7, device=rgb.device)/8)
    log_r = log_r-log_r.mean()
    slope = (bin_energy.log()*log_r).sum(-1)/log_r.square().sum()
    freq = torch.cat((bands, torch.log((bands[:, 2]+1e-12)/(bands[:, 0]+1e-12))[:, None],
                      slope[:, None], orientation[:, None]), -1)
    chroma_power = power(color[:, 1:])
    chroma_high = chroma_power[:, :, masks[2]].sum(-1)/chroma_power.sum((-2, -1)).clamp_min(1e-12)
    residual = rgb-F.avg_pool2d(F.pad(rgb, (1, 1, 1, 1), mode="replicate"), 3, stride=1)

    def correlation(a, b):
        a, b = a.flatten(1), b.flatten(1)
        a, b = a-a.mean(1, keepdim=True), b-b.mean(1, keepdim=True)
        return (a*b).sum(1)/(a.square().sum(1)*b.square().sum(1)).sqrt().clamp_min(1e-12)

    residual_corr = sum(correlation(residual[:, a], residual[:, b]) for a, b in ((0, 1), (0, 2), (1, 2)))/3
    dx = color[:, :, :-1, 1:]-color[:, :, :-1, :-1]
    dy = color[:, :, 1:, :-1]-color[:, :, :-1, :-1]
    edge = (dx.square()+dy.square()).sqrt()
    clipping = ((rgb <= .01) | (rgb >= .99)).any(1).float().mean((-2, -1))
    stats = torch.cat((color.mean((-2, -1)), color.std((-2, -1), unbiased=False), chroma_high,
                       residual_corr[:, None], correlation(edge[:, 0], edge[:, 1])[:, None],
                       correlation(edge[:, 0], edge[:, 2])[:, None], clipping[:, None]), -1)
    if not torch.isfinite(freq).all() or not torch.isfinite(stats).all():
        raise ValueError("Nonfinite image statistics")
    return freq, stats


def feature_matrix(data, kind):
    z = np.asarray(data["query_log_odds"], dtype=np.float64)
    p, e = data["cls_prob"], data["evidence_prob"]
    q = np.exp(z-z.max(1, keepdims=True))
    q /= q.sum(1, keepdims=True)
    ordered = np.sort(z, axis=1)
    gap = ordered[:, -1]-ordered[:, -2] if z.shape[1] > 1 else np.zeros(len(z))
    entropy = -(q*np.log(np.maximum(q, 1e-12))).sum(1)
    model = np.column_stack((data["global_log_odds"], p, e, z, ordered[:, -1], gap, entropy, p-e))
    parts = [model]
    if kind in ("freq", "both"):
        parts.append(data["freq"])
    if kind in ("color", "both"):
        parts.append(data["color"])
    if kind not in ("model", "freq", "color", "both"):
        raise ValueError(f"Unknown feature kind {kind}")
    result = np.concatenate(parts, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite decision features")
    return result


def fit_router(data, kind, hidden, seed=1024, steps=300, lr=.01):
    """Train only on held-out-source disagreements; all scaling fits here only."""
    import torch
    from torch import nn

    if steps < 1 or not np.isfinite(lr) or lr <= 0 or hidden not in (0, 16):
        raise ValueError("Invalid router training budget/architecture")
    x = feature_matrix(data, kind)
    p, e, labels = data["cls_prob"], data["evidence_prob"], data["labels"]
    disagreement = (p > .5) != (e > .5)
    if not disagreement.any():
        raise ValueError("No disagreement samples available for router training")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Binary labels required")
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1
    x = np.clip((x-mean)/std, -10, 10)[disagreement]
    target = ((e[disagreement] > .5) == labels[disagreement]).astype(np.float32)
    # Equal total weight per video among disagreement frames.
    videos = data["video_id"][disagreement]
    _, inverse, counts = np.unique(videos, return_inverse=True, return_counts=True)
    weights = (1/counts[inverse]).astype(np.float32)
    weights /= weights.mean()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers = ([nn.Linear(x.shape[1], hidden), nn.Tanh(), nn.Linear(hidden, 1)]
                  if hidden else [nn.Linear(x.shape[1], 1)])
        model = nn.Sequential(*layers).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        tensor, truth, weight = torch.tensor(x, dtype=torch.float32), torch.tensor(target), torch.tensor(weights)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (nn.functional.binary_cross_entropy_with_logits(model(tensor)[:, 0], truth,
                                                                     reduction="none")*weight).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite router loss")
            loss.backward()
            optimizer.step()
        parameters = [{"weight": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()}
                      for layer in model if isinstance(layer, nn.Linear)]
    return {"version": 1, "kind": kind, "hidden": hidden, "seed": seed, "steps": steps,
            "lr": lr, "weight_decay": .001, "mean": mean.tolist(), "std": std.tolist(),
            "layers": parameters, "fit_frames": len(labels), "disagreement_frames": int(disagreement.sum()),
            "expert_correct_fraction": float(target.mean()), "last_loss": float(loss.detach()),
            "standardized_clip": 10., "video_weighting": "equal disagreement-video weight"}


def score_router(artifact, data, threshold=.7):
    """Binary trust-expert model; low confidence and agreeing branches keep CLS.

    Labels are not accessed. Threshold is fixed before looking at test results.
    A high-confidence trust-expert decision switches to its continuous score.
    """
    if not .5 <= threshold <= 1:
        raise ValueError("Reject threshold must be in [.5,1]")
    x = feature_matrix(data, artifact["kind"])
    x = np.clip((x-np.asarray(artifact["mean"]))/np.asarray(artifact["std"]), -10, 10)
    for i, layer in enumerate(artifact["layers"]):
        x = x@np.asarray(layer["weight"]).T+np.asarray(layer["bias"])
        if i+1 < len(artifact["layers"]):
            x = np.tanh(x)
    trust = 1/(1+np.exp(-np.clip(x[:, 0], -60, 60)))
    p, e = data["cls_prob"], data["evidence_prob"]
    use = (trust > threshold) & ((p > .5) != (e > .5))
    return np.where(use, e, p)


def auc(labels, scores):
    """Mann-Whitney AUC with average ranks for ties, no sklearn dependency."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("Invalid AUC input")
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == len(labels):
        raise ValueError("AUC requires both classes")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    _, start, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.repeat(start+(counts+1)/2, counts)
    return float((ranks[labels == 1].sum()-n_pos*(n_pos+1)/2)/(n_pos*(len(labels)-n_pos)))


def metrics(data, scores):
    labels = np.asarray(data["labels"])
    grouped = {}
    for video, label, score in zip(data["video_id"], labels, scores):
        grouped.setdefault(video, []).append((int(label), float(score)))
    video_labels, video_scores = [], []
    legacy = {}
    for rows in grouped.values():
        if len({r[0] for r in rows}) != 1:
            raise ValueError("Conflicting labels inside full-path video group")
        video_labels.append(rows[0][0])
        video_scores.append(np.mean([r[1] for r in rows]))
    for video, rows in grouped.items():
        legacy.setdefault(str(video).replace("\\", "/").rsplit("/", 1)[-1], []).extend(rows)
    # Preserve project's historical basename grouping under video_auc. Also
    # report collision-safe full-path grouping explicitly, never conflate them.
    legacy_labels = [int(np.mean([r[0] for r in rows])) for rows in legacy.values()]
    legacy_scores = [np.mean([r[1] for r in rows]) for rows in legacy.values()]
    cls_correct = (data["cls_prob"] > .5) == labels
    evidence_correct = (data["evidence_prob"] > .5) == labels
    correct = (scores > .5) == labels
    return {"frame_auc": auc(labels, scores), "video_auc": auc(legacy_labels, legacy_scores),
            "video_auc_fullpath": auc(video_labels, video_scores),
            "legacy_group_collisions": len(grouped)-len(legacy),
            "legacy_mixed_label_groups": sum(len({r[0] for r in rows}) > 1 for rows in legacy.values()),
            "num_frames": len(labels), "num_videos": len(grouped), "accuracy": float(correct.mean()),
            "corrected_frames": int((~cls_correct & correct).sum()),
            "harmed_frames": int((cls_correct & ~correct).sum()),
            "oracle_threshold_accuracy": float((cls_correct | evidence_correct).mean()),
            "oracle_note": "label-using threshold accuracy diagnostic, not an AUC upper bound"}

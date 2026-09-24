"""Same-input expert diagnostics; no thresholds or readouts fitted on test data."""

import math

import numpy as np

from g26_diagnostics import summarize_routing


def responsibilities(log_odds, temperature):
    values = np.asarray(log_odds, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Require finite [N,K] logits and positive router temperature")
    shifted = (values - values.max(1, keepdims=True)) / temperature
    scores = np.exp(shifted)
    return scores / scores.sum(1, keepdims=True)


def summarize_experts(labels, cls_prob, evidence_prob, gated_prob, gate_weight,
                      query_prob, query_log_odds, temperature, gate_width):
    result = summarize_routing(labels, cls_prob, evidence_prob, gated_prob, gate_weight, query_prob)
    labels, cls_prob = np.asarray(labels), np.asarray(cls_prob)
    z, probabilities = np.asarray(query_log_odds), np.asarray(query_prob)
    if z.shape != probabilities.shape:
        raise ValueError("Query logits and probabilities must align")
    q = responsibilities(z, temperature)
    fake = labels == 1
    result["router_temperature"] = temperature
    result["confident_cls_errors"] = int((((cls_prob > .5) != labels)
                                           & (np.abs(cls_prob - .5) >= gate_width)).sum())
    if fake.any():
        selected = q[fake]
        entropy = -(selected * np.log(np.maximum(selected, np.finfo(float).tiny))).sum(1)
        load = selected.mean(0)
        # Split exactly tied maxima instead of artificially crediting expert 0.
        winners = z[fake] == z[fake].max(1, keepdims=True)
        result.update(fake_soft_load=load.tolist(),
                      fake_winner_share=(winners / winners.sum(1, keepdims=True)).mean(0).tolist(),
                      fake_tie_rate=float((winners.sum(1) > 1).mean()),
                      fake_route_entropy=float(entropy.mean()),
                      fake_marginal_entropy=float(-(load * np.log(np.maximum(load, np.finfo(float).tiny))).sum()),
                      fake_route_information=float(-(load * np.log(np.maximum(load, np.finfo(float).tiny))).sum() - entropy.mean()))
    else:
        result.update({key: None for key in ("fake_soft_load", "fake_winner_share", "fake_tie_rate",
                                            "fake_route_entropy", "fake_marginal_entropy", "fake_route_information")})
    centered = probabilities.astype(np.float64) - probabilities.mean(0)
    norms = np.sqrt((centered ** 2).sum(0))
    denominator = norms[:, None] * norms[None, :]
    correlation = np.divide(centered.T @ centered, denominator,
                            out=np.full_like(denominator, np.nan), where=denominator > 0)
    result["query_correlation"] = [[float(v) if np.isfinite(v) else None for v in row]
                                   for row in correlation]
    return result


def collect_routing_diagnostics(config, checkpoint, datasets, output_dir, utilities):
    """Save aligned frame arrays and bounded attention snapshots on those inputs.

    Correlation is Pearson on query probabilities; undefined constant columns
    are null. Attention is averaged over heads, new-token rows to patch columns,
    without renormalizing away mass assigned to CLS/other evidence tokens.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    model = utilities.load_model(config, checkpoint)
    model.eval()
    device = next(model.parameters()).device
    temperature = config.get("g27_router_temperature", 1.)
    width = config.get("g27_gate_width", .2)
    limit = config.get("g27_attention_samples", 16)
    summary = {}
    try:
        with torch.no_grad():
            for dataset in datasets:
                utilities.seed_evaluation(config["manualSeed"])
                batches = {key: [] for key in ("labels", "cls_prob", "evidence_prob", "gated_prob",
                                               "gate_weight", "query_prob", "query_log_odds")}
                snapshots, sample_images, frame_indices = [], [], []
                offset, saved = 0, 0
                for data in utilities.get_data_loader(config, dataset):
                    images = data["image"].to(device)
                    labels = (data["label"] != 0).long()
                    output = model({"image": images}, inference=True)["g27"]
                    batches["labels"].append(labels.cpu().numpy())
                    for key in ("cls_prob", "evidence_prob", "gated_prob", "gate_weight"):
                        batches[key].append(output[key].cpu().numpy())
                    z = output["evidence_log_odds"]
                    batches["query_prob"].append(z.sigmoid().cpu().numpy())
                    batches["query_log_odds"].append(z.cpu().numpy())
                    count = min(limit - saved, len(images))
                    if count > 0:
                        if images.ndim != 4:
                            raise ValueError("G27 attention diagnostics require single-crop 4D images")
                        attention = model.evidence_attention(images[:count])
                        if not torch.isfinite(attention).all():
                            raise ValueError("Nonfinite diagnostic attention")
                        snapshots.append(attention.cpu().numpy())
                        sample_images.append(images[:count].cpu().numpy())
                        frame_indices.extend(range(offset, offset + count))
                        saved += count
                    offset += len(images)
                if not batches["labels"]:
                    raise ValueError(f"No diagnostic frames for {dataset}")
                values = {key: np.concatenate(parts) for key, parts in batches.items()}
                summary[dataset] = summarize_experts(**values, temperature=temperature, gate_width=width)
                np.savez_compressed(output_dir / f"{dataset}.npz", **values,
                                    responsibility=responsibilities(values["query_log_odds"], temperature))
                if snapshots:
                    np.savez_compressed(output_dir / f"{dataset}_attention.npz",
                                        attention=np.concatenate(snapshots),
                                        normalized_images=np.concatenate(sample_images),
                                        frame_index=np.asarray(frame_indices),
                                        labels=values["labels"][frame_indices])
                summary[dataset]["attention_frames"] = saved
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary

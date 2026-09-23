"""Paired, same-forward routing diagnostics; these never tune the gate."""

import numpy as np


def summarize_routing(labels, cls_prob, evidence_prob, gated_prob, gate_weight, query_prob):
    """Frame-level counts at threshold 0.5, plus per-query score statistics."""
    labels = np.asarray(labels)
    arrays = [np.asarray(value) for value in (cls_prob, evidence_prob, gated_prob, gate_weight)]
    query_prob = np.asarray(query_prob)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected a nonempty binary label vector")
    if any(value.shape != labels.shape for value in arrays):
        raise ValueError("Diagnostic vectors must refer to the same samples")
    if query_prob.ndim != 2 or query_prob.shape[0] != len(labels) or query_prob.shape[1] < 1:
        raise ValueError("Expected per-query probabilities [N, K]")
    for value in arrays + [query_prob]:
        if not np.isfinite(value).all() or not ((value >= 0) & (value <= 1)).all():
            raise ValueError("Diagnostic probabilities/weights must be finite and in [0,1]")
    cls_prob, evidence_prob, gated_prob, gate_weight = arrays
    active = gate_weight > 0
    cls_correct = (cls_prob > .5) == labels
    evidence_correct = (evidence_prob > .5) == labels
    gated_correct = (gated_prob > .5) == labels
    if not np.array_equal(gated_prob[~active], cls_prob[~active]):
        raise ValueError("Inactive gate changed the original score")
    corrected = int((~cls_correct & gated_correct).sum())
    harmed = int((cls_correct & ~gated_correct).sum())
    per_class = {}
    for label, name in ((0, "real"), (1, "fake")):
        scores = query_prob[labels == label]
        per_class[name] = {"count": len(scores),
                           "query_mean": scores.mean(0).tolist() if len(scores) else None,
                           "query_std": scores.std(0).tolist() if len(scores) else None}
    return {"num_frames": len(labels), "triggered_frames": int(active.sum()),
            "trigger_rate": float(active.mean()), "corrected_frames": corrected,
            "harmed_frames": harmed, "net_corrected_frames": corrected - harmed,
            "cls_accuracy": float(cls_correct.mean()),
            "evidence_accuracy": float(evidence_correct.mean()),
            "gated_accuracy": float(gated_correct.mean()),
            "triggered_cls_accuracy": float(cls_correct[active].mean()) if active.any() else None,
            "triggered_evidence_accuracy": float(evidence_correct[active].mean()) if active.any() else None,
            "triggered_gated_accuracy": float(gated_correct[active].mean()) if active.any() else None,
            "mean_gate_weight": float(gate_weight.mean()), "per_class": per_class}


def collect_routing_diagnostics(config, checkpoint, datasets, output_dir, utilities):
    """One extra pass per dataset; all branch scores share exactly the same input.

    This is separate from testall's video_auc, so stochastic augmentation cannot
    silently turn different forward passes into supposed paired evidence.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    model = utilities.load_model(config, checkpoint)
    device = next(model.parameters()).device
    summary = {}
    try:
        with torch.no_grad():
            for dataset in datasets:
                utilities.seed_evaluation(config["manualSeed"])
                loader = utilities.get_data_loader(config, dataset)
                batches = {key: [] for key in ("labels", "cls_prob", "evidence_prob", "gated_prob",
                                               "gate_weight", "query_prob")}
                for data in loader:
                    images = data["image"].to(device)
                    labels = (data["label"] != 0).long()
                    output = model({"image": images}, inference=True)["g26"]
                    batches["labels"].append(labels.cpu().numpy())
                    for key in ("cls_prob", "evidence_prob", "gated_prob", "gate_weight"):
                        batches[key].append(output[key].cpu().numpy())
                    batches["query_prob"].append(output["evidence_log_odds"].sigmoid().cpu().numpy())
                if not batches["labels"]:
                    raise ValueError(f"No diagnostic frames for {dataset}")
                values = {key: np.concatenate(parts) for key, parts in batches.items()}
                summary[dataset] = summarize_routing(**values)
                np.savez_compressed(output_dir / f"{dataset}.npz", **values)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary

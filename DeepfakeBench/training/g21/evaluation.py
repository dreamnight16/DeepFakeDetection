"""Small-fixture evaluator for server tests; production G21 uses legacy.py/testall.py."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

from .data import decode_rgb, normalized_tensor, selected_frames
from .io import atomic_json, write_jsonl


def binary_auc(scores: list[float], labels: list[int]) -> float:
    if len(scores) != len(labels) or set(labels) != {0, 1}:
        raise ValueError("AUC needs equally sized arrays and both classes")
    if any(not math.isfinite(p) for p in scores):
        raise ValueError("nonfinite AUC score")
    ordered = sorted(zip(scores, labels))
    pos_ranks = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        pos_ranks += sum(y for _, y in ordered[start:end]) * (start + 1 + end) / 2
        start = end
    positives = sum(labels)
    negatives = len(labels) - positives
    return (pos_ranks - positives * (positives + 1) / 2) / (positives * negatives)


def aggregate_predictions(
    records: list[dict], expected: list[dict]
) -> tuple[list[dict], dict]:
    def key(row):
        return row["dataset_id"], row["video_id"], row["frame_id"]

    expected_map = {key(r): r["label"] for r in expected}
    if len(expected_map) != len(expected):
        raise ValueError("duplicate expected frame identity")
    groups, seen = defaultdict(list), set()
    for row in records:
        k = key(row)
        p = row["fake_probability"]
        if k in seen or k not in expected_map or row["label"] != expected_map[k]:
            raise ValueError("duplicate/unexpected frame or label mismatch")
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid fake probability")
        seen.add(k)
        groups[k[:2]].append(row)
    if seen != expected_map.keys():
        raise ValueError("incomplete prediction manifest")
    videos = []
    for (dataset, identity), rows in sorted(groups.items()):
        labels = {row["label"] for row in rows}
        if len(labels) != 1:
            raise ValueError("video label conflict")
        videos.append(
            {
                "dataset_id": dataset,
                "video_id": identity,
                "label": rows[0]["label"],
                "fake_probability": math.fsum(row["fake_probability"] for row in rows)
                / len(rows),
                "n_frames": len(rows),
            }
        )
    if len({v["dataset_id"] for v in videos}) != 1:
        raise ValueError("evaluate one dataset at a time")
    metrics = {
        "metric": "video_auc",
        "video_auc": binary_auc(
            [v["fake_probability"] for v in videos], [v["label"] for v in videos]
        ),
        "auc": binary_auc(
            [v["fake_probability"] for v in records], [v["label"] for v in records]
        ),
        "n_videos": len(videos),
        "n_real": sum(v["label"] == 0 for v in videos),
        "n_fake": sum(v["label"] == 1 for v in videos),
        "n_frames": len(records),
        "aggregation": "mean_frame_probability_by_full_video_id",
    }
    return videos, metrics


def evaluate_model(
    model,
    videos: list[dict],
    *,
    root: str,
    frame_count: int,
    batch_size: int,
    device="cpu",
    output: str | Path | None = None,
    provenance: dict | None = None,
) -> dict:
    import torch

    expected = selected_frames(videos, frame_count)
    records = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(expected), batch_size):
                rows = expected[start : start + batch_size]
                images = torch.stack(
                    [normalized_tensor(decode_rgb(root, row["path"])) for row in rows]
                ).to(device)
                logits = model({"image": images})["cls"].float()
                if logits.shape != (len(rows), 2) or not torch.isfinite(logits).all():
                    raise ValueError("invalid evaluation logits")
                probs = logits.softmax(-1)[:, 1].cpu().tolist()
                for row, prob in zip(rows, probs):
                    records.append(
                        {
                            k: row[k]
                            for k in ("dataset_id", "video_id", "frame_id", "label")
                        }
                        | {"fake_probability": prob}
                    )
    finally:
        model.train(was_training)
    video_rows, metrics = aggregate_predictions(records, expected)
    metrics.update(provenance or {})
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_hash = (provenance or {}).get("checkpoint_sha256")
        for row in records:
            row["checkpoint_sha256"] = checkpoint_hash
        write_jsonl(output / "frame_predictions.jsonl", records)
        write_jsonl(output / "video_predictions.jsonl", video_rows)
        atomic_json(output / "metrics.json", metrics)
    return metrics


def summarize_datasets(results: dict) -> dict:
    seven = [
        "WDF",
        "FFIW",
        "Celeb-DF-v2",
        "DeepFakeDetection",
        "DFDC",
        "DFDCP",
        "DeeperForensics-1.0",
    ]

    def mean(names):
        return (
            sum(results[n]["video_auc"] for n in names) / len(names)
            if all(n in results for n in names)
            else None
        )

    return {
        "AUC_cross": mean(["Celeb-DF-v2", "DFDC"]),
        "mean7": mean(seven),
        "missing_for_mean7": [n for n in seven if n not in results],
    }

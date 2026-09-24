"""Frozen G27 score/image-stat export, with strict frame identity and no fallback."""

import hashlib
import json

import numpy as np

from e0924_decision import image_features, feature_matrix
from e0924_protocol import canonical_path


def strict_loader(base_dataset, batch_size, records=None):
    """Reuse project's RGB loader/normalization, fail on an unreadable frame.

    Ordinary dataset.__getitem__ randomly substitutes a frame on load failure;
    that would silently corrupt the path-to-score alignment needed here.
    """
    import torch

    if records is None:
        rows = [(path, label) for path, label in zip(base_dataset.image_list, base_dataset.label_list)]
    else:
        rows = [(path, base_dataset.config["label_dict"][record["label_name"]])
                for record in records for path in record["frames"]]
    if not rows or any(not isinstance(p, str) for p, _ in rows):
        raise ValueError("Export requires nonempty single-frame paths")
    if len({canonical_path(p) for p, _ in rows}) != len(rows):
        raise ValueError("Duplicate frame identities in export")

    class StrictFrames(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            path, label = rows[index]
            rgb = base_dataset.load_rgb(path)  # raises; never replace with a random sample
            tensor = base_dataset.normalize(base_dataset.to_tensor(rgb))
            path = canonical_path(path)
            return {"image": tensor, "label": int(label != 0), "path": path,
                    "video_id": path.rsplit("/", 1)[0]}

    # Single process preserves LMDB handles and deterministic row ordering.
    return torch.utils.data.DataLoader(StrictFrames(), batch_size=batch_size, shuffle=False, num_workers=0)


def export_loader(model, loader, config):
    import torch

    model.eval()
    device = next(model.parameters()).device
    batches = {key: [] for key in ("labels", "path", "video_id", "cls_prob", "evidence_prob",
                                   "gated_prob", "query_log_odds", "global_log_odds", "freq", "color")}
    with torch.no_grad():
        for data in loader:
            images = data["image"].to(device)
            output = model({"image": images}, inference=True)["g27"]
            rgb = images*images.new_tensor(config["std"])[None, :, None, None]
            rgb = rgb+images.new_tensor(config["mean"])[None, :, None, None]
            freq, color = image_features(rgb)
            batches["labels"].append(data["label"].cpu().numpy())
            batches["path"].append(np.asarray(data["path"]))
            batches["video_id"].append(np.asarray(data["video_id"]))
            for name in ("cls_prob", "evidence_prob", "gated_prob"):
                batches[name].append(output[name].cpu().numpy())
            batches["query_log_odds"].append(output["evidence_log_odds"].cpu().numpy())
            logits = output["global_logits"]
            batches["global_log_odds"].append((logits[:, 1]-logits[:, 0]).cpu().numpy())
            batches["freq"].append(freq.cpu().numpy())
            batches["color"].append(color.cpu().numpy())
    if not batches["labels"]:
        raise ValueError("Empty export loader")
    result = {key: np.concatenate(parts) for key, parts in batches.items()}
    feature_matrix(result, "both")  # fail closed on nonfinite feature output
    return result


def export_all(config, checkpoint, partition, output_dir, utilities, datasets, seed_evaluation):
    """Export calibration and test arrays from one checkpoint before any router fit."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=False)
    model = utilities.load_model(config, checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    summary = {}
    try:
        # Reuse an FF++ dataset instance only as an image loader (including LMDB).
        # Calibration rows come exclusively from official val metadata.
        seed_evaluation(config["manualSeed"])
        base = utilities.get_data_loader(config, "FaceForensics++").dataset
        for split in ("fit", "holdout"):
            loader = strict_loader(base, config["test_batchSize"], partition["records"][split])
            values = export_loader(model, loader, config)
            np.savez_compressed(output_dir / f"{split}.npz", **values)
            summary[split] = {"frames": len(values["labels"]), "videos": len(set(values["video_id"]))}
        del base
        for dataset in datasets:
            seed_evaluation(config["manualSeed"])
            base = utilities.get_data_loader(config, dataset).dataset
            values = export_loader(model, strict_loader(base, config["test_batchSize"]), config)
            np.savez_compressed(output_dir / f"test_{dataset}.npz", **values)
            summary[dataset] = {"frames": len(values["labels"]), "videos": len(set(values["video_id"]))}
            del base
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary


def load_export(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

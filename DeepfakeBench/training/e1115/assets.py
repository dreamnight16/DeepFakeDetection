"""Strict source statistics and native eight-frame evaluation feature exports."""

import json
from pathlib import Path, PurePosixPath

import numpy as np

from g21.io import atomic_json, digest, file_digest


def build_source_audit(metadata, rgb_root, output):
    from e1111.audit import inventory, index_pairing, audit
    output = Path(output)
    indexed = inventory(Path(metadata) / "FaceForensics++.json", output / "inventory",
                        data_root=rgb_root, statistic_mode="video_mean")
    index = Path(indexed["dataset_json"])
    index_pairing(index, output / "pairing", data_root=rgb_root, statistic_mode="video_mean")
    checked = audit(index, rgb_root, output / "pairing/pair_map.json", output / "audit",
                    statistic_mode="video_mean")
    # This is input validation, not S0/D1/S1 performance gating.
    if checked["invalid_paths"] or not checked["complete_sources"]:
        raise ValueError("Source input audit has unreadable images or no complete sources")
    data = json.loads(index.read_text(encoding="utf-8"))["FaceForensics++"]
    checked["parent_audit_sha256"] = checked.pop("audit_sha256")
    checked["e1115_matched"] = matched_candidates(data, checked, rgb_root)
    checked["audit_sha256"] = digest(checked)
    atomic_json(output / "e1115_source_manifest.json", checked)
    return checked, index


def matched_candidates(data, audit, rgb_root, readable=None):
    """Intersect original frame indices before selecting a common candidate list."""
    from e1111.audit import frame_map, _valid_image
    from g21.data import uniform_indices
    from .statistics import METHODS
    readable = readable or _valid_image
    bags = {(b["source"], b["method"]): b for b in audit["fit_videos"]}
    result = {}
    for source in audit["complete_sources"]:
        real = frame_map(data["FF-real"]["train"]["c23"][source])
        fake = {}
        for method in METHODS:
            path = bags[source, method]["frames"][0]["fake_path"].replace("\\", "/")
            key = PurePosixPath(path).parent.name
            fake[method] = frame_map(data[method]["train"]["c23"][key])
        common = sorted(set(real).intersection(*(set(x) for x in fake.values())))
        if len(common) < 2:
            continue
        selected = [common[i] for i in uniform_indices(len(common), 32)]
        paths = [real[t] for t in selected] + [fake[m][t] for m in METHODS for t in selected]
        if not all(readable(rgb_root, p) for p in paths):
            raise ValueError(f"Unreadable shared-index candidate for {source}; no silent sample replacement")
        result[source] = {"real": [{"frame": t, "path": real[t]} for t in selected],
                          "fake": {m: [{"frame": t, "fake_path": fake[m][t]} for t in selected] for m in METHODS}}
    return result


def _encode(extractor, images, count):
    import cv2
    import torch
    from g21.data import normalized_tensor
    features, probabilities, batch = [], [], []
    def flush():
        tensor = torch.stack(batch).to(extractor.device)
        with torch.no_grad():
            vision = extractor.model.backbone(extractor.model._prep_input(tensor))
            features.append(vision["last_hidden_state"][:, 1:].mean(1).float().cpu().numpy())
            probabilities.append(extractor.model.classifier(vision["pooler_output"]).softmax(1)[:, 1].cpu().numpy())
        batch.clear()
    for rgb in images:
        batch.append(normalized_tensor(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC)))
        if len(batch) == extractor.batch_size:
            flush()
    if batch:
        flush()
    z, p = np.concatenate(features), np.concatenate(probabilities)
    if len(z) != count or not np.isfinite(z).all() or not np.isfinite(p).all():
        raise ValueError("Source feature export has invalid rows")
    return z, p


def prepare_source(audit, index_path, config_path, checkpoint, clip, output, device, batch_size):
    from e1111.features import FrozenFeatures, read_rgb, jpeg_roundtrip
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    rgb_root = config["rgb_root_override"]
    data = json.loads(Path(index_path).read_text(encoding="utf-8-sig"))["FaceForensics++"]
    real_index = data["FF-real"]["train"]["c23"]
    # Fit can use 32 frames; calibration uses the native index's first-eight rule.
    calibration = {v: real_index[v]["frames"][:8] for v in audit["splits"]["theta"]}
    fit_real_rows = [row for group in audit["real_frames"]["fit"].values() for row in group]
    paths = sorted({x["path"] for x in fit_real_rows}
                   | {x["fake_path"] for b in audit["fit_videos"] for x in b["frames"]}
                   | {x for group in calibration.values() for x in group}
                   | {row["path"] for b in audit.get("e1115_matched", {}).values() for row in b["real"]}
                   | {row["fake_path"] for b in audit.get("e1115_matched", {}).values() for rows in b["fake"].values() for row in rows})
    extractor = FrozenFeatures(config_path, checkpoint, clip, device, batch_size)
    z, probabilities = _encode(extractor, (read_rgb(rgb_root, p) for p in paths), len(paths))
    lookup = dict(zip(paths, z))
    prob_lookup = dict(zip(paths, probabilities))
    dc, n = np.zeros((z.shape[1], z.shape[1])), 0
    # Accumulate compression moments by batches, without retaining four large copies.
    for begin in range(0, len(fit_real_rows), batch_size):
        rows = fit_real_rows[begin:begin + batch_size]
        reference = np.stack([lookup[row["path"]] for row in rows])
        for quality in (15, 30, 40, 50):
            compressed, _ = _encode(extractor, (jpeg_roundtrip(read_rgb(rgb_root, row["path"]), quality)
                                                 for row in rows), len(rows))
            delta = compressed.astype(np.float64) - reference
            dc += delta.T @ delta
            n += len(delta)
    dc /= n
    cal_paths = [x for group in calibration.values() for x in group]
    cal_rows = {"path": np.asarray(cal_paths), "labels": np.zeros(len(cal_paths), dtype=np.int64),
                "video_id": np.asarray([p.replace("\\", "/").rsplit("/", 1)[0] for p in cal_paths]),
                "cls_prob": np.asarray([prob_lookup[p] for p in cal_paths])}
    fit_real = [np.stack([lookup[row["path"]] for row in rows])
                for _, rows in sorted(audit["real_frames"]["fit"].items())]
    cal_z = np.stack([lookup[p] for p in cal_paths])
    np.savez_compressed(output / "fit_features.npz", paths=np.asarray(paths), z=z, compression=dc)
    np.savez_compressed(output / "calibration.npz", **cal_rows, z=cal_z)
    atomic_json(output / "manifest.json", {"status": "OK", "checkpoint_sha256": file_digest(checkpoint),
                "config_sha256": file_digest(config_path), "audit_sha256": audit["audit_sha256"],
                "features_sha256": file_digest(output / "fit_features.npz"),
                "calibration_sha256": file_digest(output / "calibration.npz"),
                "feature_identity": extractor.identity, "compression_observations": n,
                "calibration_frames": 8, "statistic_frames": 32,
                "source_split_scope": audit["split_scope"]})
    del extractor
    return lookup, fit_real, dc, cal_rows, cal_z


def export_native(config_path, checkpoint, clip, output, device, batch_size):
    """Same loader/readout rows as B0, same forward for class and patch features."""
    import torch
    import experiment_utils as utilities
    from e0924_export import strict_loader
    from e1111.features import FrozenFeatures
    from run_e1021 import seed_everything
    from run_g25 import TEST_DS
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "features").mkdir()
    (output / "exports").mkdir()
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    if config["frame_num"]["test"] != 8 or config.get("multi_crop"):
        raise ValueError("E1115 requires native eight-frame single-crop evaluation")
    extractor = FrozenFeatures(config_path, checkpoint, clip, device, batch_size)
    report = {"status": "RUNNING", "checkpoint_sha256": file_digest(checkpoint),
              "eval_config_sha256": file_digest(config_path), "feature_identity": extractor.identity,
              "datasets": {}}
    atomic_json(output / "manifest.json", report)
    for name in TEST_DS:
        print(f"E1115/export/{name}", flush=True)
        seed_everything(config["manualSeed"])
        dataset = utilities.get_data_loader(config, name).dataset
        file = output / "features" / f"{name}.npy"
        z = np.lib.format.open_memmap(file, mode="w+", dtype=np.float32, shape=(len(dataset.image_list), 1024))
        chunks = {k: [] for k in ("labels", "path", "video_id", "cls_prob", "global_log_odds")}
        cursor = 0
        try:
            with torch.no_grad():
                for batch in strict_loader(dataset, batch_size):
                    vision = extractor.model.backbone(extractor.model._prep_input(batch["image"].to(device)))
                    values = vision["last_hidden_state"][:, 1:].mean(1).float().cpu().numpy()
                    logits = extractor.model.classifier(vision["pooler_output"])
                    if not np.isfinite(values).all() or not torch.isfinite(logits).all():
                        raise ValueError(f"Nonfinite native outputs in {name}")
                    z[cursor:cursor + len(values)] = values
                    cursor += len(values)
                    for key in ("labels", "path", "video_id"):
                        chunks[key].append(batch["label"].cpu().numpy() if key == "labels" else np.asarray(batch[key]))
                    chunks["cls_prob"].append(logits.softmax(1)[:, 1].cpu().numpy())
                    chunks["global_log_odds"].append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            if cursor != len(z) or not cursor:
                raise ValueError("Native feature export row count mismatch")
            z.flush()
            rows = {k: np.concatenate(v) for k, v in chunks.items()}
            np.savez_compressed(output / "exports" / f"{name}.npz", **rows)
            report["datasets"][name] = {"rows": cursor, "features_sha256": file_digest(file),
                "exports_sha256": file_digest(output / "exports" / f"{name}.npz"),
                "rows_sha256": digest({k: rows[k].tolist() for k in ("labels", "path", "video_id")})}
            atomic_json(output / "manifest.json", report)
        finally:
            del z
            env = getattr(dataset, "env", None)
            if env is not None:
                env.close()
    report["status"] = "OK"
    atomic_json(output / "manifest.json", report)
    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def load_native(directory, dataset):
    directory = Path(directory)
    report = json.loads((directory / "manifest.json").read_text())
    if report["status"] != "OK":
        raise ValueError("Cannot use incomplete native feature exports")
    info = report["datasets"][dataset]
    path = directory / "exports" / f"{dataset}.npz"
    feature_path = directory / "features" / f"{dataset}.npy"
    if file_digest(path) != info["exports_sha256"] or file_digest(feature_path) != info["features_sha256"]:
        raise ValueError("Native cache bytes changed")
    with np.load(path, allow_pickle=False) as data:
        rows = {k: data[k] for k in data.files}
    if digest({k: rows[k].tolist() for k in ("labels", "path", "video_id")}) != info["rows_sha256"]:
        raise ValueError("Native row identity changed")
    return rows, np.load(feature_path, mmap_mode="r", allow_pickle=False)

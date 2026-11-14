"""Strict frozen-B0 global features; no legacy first-eight-frame loader."""

import json
from pathlib import Path

import numpy as np

from g21.data import normalized_tensor
from g21.io import file_digest
from g21.manifest import image_path

from .protocol import digest, implementation_digest, protocol


def load_audit(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    identity = value.pop("audit_sha256")
    settings = protocol(value.get("statistic_mode", "paired_frame"))
    if digest(value) != identity or value["protocol_sha256"] != digest(settings):
        raise ValueError("Audit content/protocol changed; rerun the audit")
    value["audit_sha256"] = identity
    if not value["P_D7_pass"]:
        raise ValueError("P-D7 is not passed; stop before feature fitting/S0")
    if value.get("audit_implementation_sha256") != implementation_digest("audit"):
        raise ValueError("Audit implementation changed; regenerate its verified manifest")
    return value


def read_rgb(root, relative):
    import cv2

    image = cv2.imdecode(np.frombuffer(image_path(root, relative).read_bytes(), dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image; no sample substitution: {relative}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class FrozenFeatures:
    def __init__(self, config_path, checkpoint, clip_path, device="cuda:0", batch_size=8):
        import torch
        import yaml
        import transformers
        import loralib
        import cv2
        import PIL
        from detectors import DETECTOR

        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
        if (config.get("model_name", "effort") != "effort"
                or not config.get("full_train_head", False)
                or config.get("use_freq_split", False)):
            raise ValueError("E1111A requires a plain original B0 configuration")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        config["clip_pretrained_path"] = str(clip_path)
        self.model = DETECTOR["effort"](config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(device)
        self.device, self.batch_size = device, batch_size
        self.label_dict = config["label_dict"]
        self.identity = {"checkpoint_sha256": file_digest(checkpoint),
                         "base_config_sha256": file_digest(config_path),
                         "feature": "last_hidden_state excluding CLS; mean of 256 patch tokens",
                         "preprocess": "RGB; cubic resize 224; CLIP normalization; no augmentation",
                         "feature_implementation_sha256": implementation_digest("feature"),
                         "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                      "numpy": np.__version__,
                                      "opencv": cv2.__version__, "Pillow": PIL.__version__,
                                      "loralib": getattr(loralib, "__version__", "unreported")},
                         "checkpoint_provenance": "developmental unless independently documented as clean"}

    def encode(self, images):
        import cv2
        import torch

        output, batch = [], []

        def flush():
            pixels = torch.stack(batch).to(self.device)
            with torch.no_grad():
                tokens = self.model.backbone(pixel_values=pixels, return_dict=True).last_hidden_state
                if tuple(tokens.shape[1:]) != (257, 1024):
                    raise ValueError("Expected CLIP ViT-L/14 last hidden state (N,257,1024)")
                output.append(tokens[:, 1:].mean(dim=1).cpu().float().numpy())
            batch.clear()

        for image in images:
            batch.append(normalized_tensor(cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)))
            if len(batch) == self.batch_size:
                flush()
        if batch:
            flush()
        if not output:
            raise ValueError("Empty feature request")
        features = np.concatenate(output)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite backbone features")
        return features


def jpeg_roundtrip(rgb, quality):
    import cv2

    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG re-encoding failed")
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG decoding failed")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def transformed(rgb, name):
    import cv2

    if name == "flip":
        return np.ascontiguousarray(rgb[:, ::-1])
    if name == "crop":
        h, w = rgb.shape[:2]
        return cv2.resize(rgb[h // 20:h - h // 20, w // 20:w - w // 20], (w, h),
                          interpolation=cv2.INTER_CUBIC)
    if name == "brightness":
        return np.clip(rgb.astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown image transform: {name}")


def extract(audit_path, config, checkpoint, clip, output, *, device="cuda:0", batch_size=8, d8=False):
    audit = load_audit(audit_path)
    settings = protocol(audit.get("statistic_mode", "paired_frame"))
    root = audit["data_root"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    extractor = FrozenFeatures(config, checkpoint, clip, device, batch_size)
    # D_theta is intentionally absent; read it only after gamma is frozen.
    real_rows = [row for split in ("fit", "sel", "rad")
                 for rows in audit["real_frames"][split].values() for row in rows]
    pairs = [*audit["fit_pairs"], *audit["sel_pairs"]]
    video_paths = {row["fake_path"] for bag in [*audit.get("fit_videos", []), *audit.get("sel_videos", [])]
                   for row in bag["frames"]}
    paths = sorted({r["path"] for r in real_rows} |
                   {r[key] for r in pairs for key in ("real_path", "fake_path")} | video_paths)
    z = extractor.encode(read_rgb(root, path) for path in paths)
    lookup = {path: vector for path, vector in zip(paths, z)}
    fit_rows = [(source, row) for source, rows in audit["real_frames"]["fit"].items() for row in rows]
    q = settings["jpeg_quality"]
    # Q is every valid (real fit source, selected frame, quality) combination.
    # A decode/transform failure is fatal; no quality or frame is dropped silently.
    compressed = extractor.encode(jpeg_roundtrip(read_rgb(root, row["path"]), quality)
                                  for _, row in fit_rows for quality in q)
    reference = np.repeat(np.stack([lookup[row["path"]] for _, row in fit_rows]), len(q), axis=0)
    arrays = {"paths": np.asarray(paths), "z": z, "compression_deltas": compressed - reference}
    if d8:
        names = ("flip", "crop", "brightness")
        values = extractor.encode(transformed(read_rgb(root, row["path"]), name)
                                  for _, row in fit_rows for name in names)
        arrays.update(transform_deltas=values - np.repeat(
            np.stack([lookup[row["path"]] for _, row in fit_rows]), len(names), axis=0),
            transform_source=np.asarray([source for source, _ in fit_rows for _ in names]),
            transform_name=np.asarray([name for _ in fit_rows for name in names]))
    metadata = {**extractor.identity, "audit_sha256": audit["audit_sha256"],
                "protocol_sha256": audit["protocol_sha256"], "Q_size": len(fit_rows) * len(q),
                "Q": "all valid fit-real (source,32-candidate frame,JPEG quality) combinations; equal weight",
                "d8": d8}
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return metadata


def load_cache(cache_path, audit):
    with np.load(cache_path, allow_pickle=False) as arrays:
        values = {key: arrays[key] for key in arrays.files}
    metadata = json.loads(str(values.pop("metadata")))
    if (metadata["audit_sha256"] != audit["audit_sha256"]
            or metadata["protocol_sha256"] != digest(protocol(audit.get("statistic_mode", "paired_frame")))):
        raise ValueError("Feature cache does not match the frozen audit/protocol")
    if metadata.get("feature_implementation_sha256") != implementation_digest("feature"):
        raise ValueError("Feature implementation changed; cached coordinates are obsolete")
    paths, z = values["paths"], values["z"]
    if (z.shape != (len(paths), 1024) or len(set(paths.tolist())) != len(paths)
            or not np.isfinite(z).all()):
        raise ValueError("Invalid global feature cache")
    return values, metadata, dict(zip(paths.tolist(), z))


def export_dataset(dataset_json, dataset_id, split, data_root, config, checkpoint, clip, output,
                   *, device="cuda:0", batch_size=8):
    """Export a chosen official split, never using the legacy eight-frame path."""
    from .audit import frame_map
    from g21.data import uniform_indices

    allowed_dev = {("FaceForensics++", "val"), ("Celeb-DF-v2", "train")}
    allowed_test = {(name, "test") for name in (
        "DeepFakeDetection", "DFDC", "DFDCP", "UADFV", "Celeb-DF-v2")}
    if (dataset_id, split) not in allowed_dev | allowed_test:
        raise ValueError("Only the preregistered development or blind-test splits can be exported")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8-sig"))[dataset_id]
    extractor = FrozenFeatures(config, checkpoint, clip, device, batch_size)
    rows = []
    lineage = {}
    if dataset_id == "FaceForensics++":
        # A_B names both source videos. Clustering both connected identities
        # avoids guessing which one is the cropped content reference.
        for category, branches in data.items():
            videos = branches.get(split, {}).get("c23", {})
            for video in videos:
                ids = video.split("_")
                if not all(value.isdigit() for value in ids):
                    raise ValueError(f"Unsupported FF++ lineage: {video}")
                for source in ids:
                    lineage.setdefault(source, source)
                def find(source):
                    while lineage[source] != source:
                        source = lineage[source]
                    return source
                roots = [find(source) for source in ids]
                for source in roots:
                    lineage[source] = min(roots)
    for category, branches in sorted(data.items()):
        if split not in branches:
            continue
        videos = branches[split]
        if "c23" in videos:
            videos = videos["c23"]
        for video, entry in sorted(videos.items()):
            if entry["label"] not in extractor.label_dict:
                raise ValueError(f"Unknown dataset label: {entry['label']}")
            frames = frame_map(entry, data_root)
            ordered = sorted(frames)
            for i in uniform_indices(len(ordered), protocol()["candidate_frames"]):
                path = frames[ordered[i]]
                if dataset_id == "FaceForensics++":
                    source = video.split("_")[0]
                    while lineage[source] != source:
                        source = lineage[source]
                    cluster = f"{dataset_id}/{split}/lineage/{source}"
                else:
                    cluster = f"{dataset_id}/{category}/{video}"
                rows.append((path, f"{dataset_id}/{category}/{video}",
                             int(extractor.label_dict[entry["label"]] != 0), cluster))
    if not rows or len({r[0] for r in rows}) != len(rows):
        raise ValueError("Empty/duplicate split export")
    z = extractor.encode(read_rgb(data_root, row[0]) for row in rows)
    metadata = {**extractor.identity, "dataset_id": dataset_id, "official_split": split,
                "role": "development" if (dataset_id, split) in allowed_dev else "blind",
                "dataset_sha256": file_digest(dataset_json), "candidate_frames": 32,
                "cluster_provenance": "FF++ connected source lineage" if lineage else
                    "video identity proxy; original-source lineage unavailable"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, z=z, path=np.asarray([r[0] for r in rows]),
                            video_id=np.asarray([r[1] for r in rows]),
                            label=np.asarray([r[2] for r in rows]),
                            cluster_id=np.asarray([r[3] for r in rows]),
                            feature_identity=np.asarray(json.dumps(metadata, sort_keys=True)))
    return metadata

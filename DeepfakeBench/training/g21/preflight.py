"""Read-only metadata/image preflight: no torch import or model forward."""

from __future__ import annotations

import json
from pathlib import Path

from .config import validate_config
from .data import decode_rgb
from .io import digest, file_digest
from .manifest import image_path, load_pairs, validate_training_groups


def inspect_inputs(
    cfg: dict, *, decode: bool = True, check_model: bool = True
) -> tuple[list, list, dict, dict]:
    validate_config(cfg, require_paths=True)
    paths, data = cfg["paths"], cfg["data"]
    if not Path(paths["data_root"]).is_dir():
        raise ValueError("data_root is not a directory")
    pairs = load_pairs(
        paths["train_manifest"], min_frames=data["frames_per_training_video"]
    )
    groups = validate_training_groups(
        pairs, data["train_methods"], data["pairs_per_microbatch"]
    )
    from .legacy import BENCH, legacy_config

    legacy_config(
        cfg
    )  # Check the existing YAML instead of inventing a new test pipeline.
    train_sources = {s for row in pairs for s in row["lineage_source_ids"]}
    files = {
        image_path(paths["data_root"], f[k]): "train"
        for row in pairs
        for f in row["frame_pairs"]
        for k in ("real_path", "fake_path")
    }
    manifests = {"train_pairs": file_digest(paths["train_manifest"])}
    evaluation_counts = {}
    for name in cfg["evaluation"]["test_datasets"]:
        index = Path(paths["dataset_json_folder"]) / f"{name}.json"
        manifests[name] = file_digest(index)
        raw = json.loads(index.read_text(encoding="utf-8-sig"))[name]
        count = 0
        for label, splits in raw.items():
            block = splits["test"]
            block = block.get("c23", block)
            for video_key, entry in block.items():
                if name == "FaceForensics++" and train_sources.intersection(
                    video_key.split("_")
                ):
                    raise ValueError(f"FF++ test lineage overlaps train: {video_key}")
                # The original frame loader currently takes the first N entries.
                # Preflight mirrors this for file checks; actual evaluation still calls that loader.
                for frame in entry["frames"][:8]:
                    normalized = frame.replace("\\", "/")
                    path = (
                        Path(normalized).resolve()
                        if Path(normalized).is_absolute()
                        else image_path(paths["data_root"], normalized)
                    )
                    if files.get(path) == "train":
                        raise ValueError(
                            f"evaluation path alias overlaps train: {path}"
                        )
                    files[path] = "evaluation"
                    count += 1
        if not count:
            raise ValueError(f"empty historical test dataset: {name}")
        evaluation_counts[name] = count
    content_hashes, inode_roles, hash_roles = {}, {}, {}
    for path, role in sorted(files.items()):
        if not path.is_file():
            raise ValueError(f"missing image: {path}")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino) if stat.st_ino else str(path)
        if identity in inode_roles and inode_roles[identity] != role:
            raise ValueError(f"cross-split file alias: {path}")
        inode_roles[identity] = role
        if decode:
            decode_rgb(str(path.parent), path.name)
        hashed = file_digest(path)
        if hashed in hash_roles and hash_roles[hashed] != role:
            raise ValueError(
                f"exact duplicated image content across train/evaluation: {path}"
            )
        hash_roles[hashed] = role
        content_hashes[str(path)] = hashed
    model_hashes = {}
    if check_model:
        model_root = Path(paths["clip_pretrained_path"])
        model_config = model_root / "config.json"
        config = json.loads(model_config.read_text(encoding="utf-8"))
        vision = config.get("vision_config", {})
        if any(
            vision.get(k) != v
            for k, v in {
                "hidden_size": 1024,
                "patch_size": 14,
                "image_size": 224,
                "num_hidden_layers": 24,
            }.items()
        ):
            raise ValueError("expected local CLIP ViT-L/14 configuration")
        weights = sorted(model_root.glob("*.safetensors")) + sorted(
            model_root.glob("pytorch_model*.bin")
        )
        if not weights or any(p.stat().st_size == 0 for p in weights):
            raise ValueError("missing local CLIP weights")
        model_hashes = {p.name: file_digest(p) for p in [model_config, *weights]}
    for name in ("effort.yaml", "train_config.yaml", "test_config.yaml"):
        path = (
            BENCH
            / "training/config"
            / ("detector/effort.yaml" if name == "effort.yaml" else name)
        )
        manifests[f"legacy/{name}"] = file_digest(path)
    report = {
        "status": "PREFLIGHT_OK",
        "model_loaded": False,
        "images_decoded": decode,
        "pair_count": len(pairs),
        "source_counts": groups,
        "evaluation_frames": evaluation_counts,
        "validation_dataset": "Celeb-DF-v2",
        "selection_metric": "auc",
        "report_metric": "video_auc",
        "selection_dataset_reused_in_test": True,
        "unique_image_files": len(files),
        "manifest_hashes": manifests,
        "image_content_digest": digest(content_hashes),
        "model_hashes": model_hashes,
    }
    report["input_digest"] = digest(
        {
            k: report[k]
            for k in ("manifest_hashes", "image_content_digest", "model_hashes")
        }
    )
    return pairs, None, {}, report

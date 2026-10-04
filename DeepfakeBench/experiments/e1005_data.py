"""E1005 full-metadata manifests and explicitly audited pair episodes.

Metadata operations use only the standard library. Torch, Pillow and optional
OpenCV are imported when constructing/reading tensor datasets. Filename lineage
and equal numeric indices define candidates, never verified time/face alignment.
"""

from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import re


METHODS = ("DF", "F2F", "FS", "NT")
NESTED_COMPRESSION = {"FaceForensics++", "DeepFakeDetection"}
NUISANCE_IDS = {"clean": 0, "jpeg75": 1, "jpeg95": 2, "resize": 3, "photometry": 4}


def canonical_path(value):
    """Normalize metadata separators while rejecting traversal and empty paths."""
    if not isinstance(value, (str, Path)):
        raise ValueError("Frame path must be a string or Path")
    text = str(value).replace("\\", "/")
    if not text or "\x00" in text or ".." in text.split("/"):
        raise ValueError(f"Unsafe canonical path containment: {value!r}")
    parts = [part for part in text.split("/") if part and part != "."]
    if not parts:
        raise ValueError("Empty canonical path")
    return ("/" if text.startswith("/") else "") + "/".join(parts)


def frame_index(path):
    match = re.search(r"(?:^|_)(\d+)$", PurePosixPath(canonical_path(path)).stem)
    if match is None:
        raise ValueError(f"Frame filename lacks a numeric index: {path}")
    return int(match.group(1))


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _videos(splits, split, dataset, compression, *, required=False):
    values = splits.get(split, {})
    if not isinstance(values, dict):
        raise ValueError(f"Invalid {dataset}/{split} metadata")
    if dataset in NESTED_COMPRESSION:
        if values and compression not in values:
            raise ValueError(f"Missing compression {compression} in {dataset}/{split}")
        values = values.get(compression, {})
    if not isinstance(values, dict):
        raise ValueError(f"Invalid videos in {dataset}/{split}/{compression}")
    if any(not isinstance(info, dict) or not isinstance(info.get("frames"), list)
           for info in values.values()):
        raise ValueError(f"Invalid video-keyed schema in {dataset}/{split}/{compression}")
    if required and not values:
        return {}
    return values


def _lineage(dataset, video, info):
    if dataset == "FaceForensics++":
        if not re.fullmatch(r"\d+(?:_\d+)?", video):
            raise ValueError(f"Invalid official FF++ target/source identifier: {video}")
        members = video.split("_")
        target, source = members[0], members[1] if len(members) == 2 else None
        if info.get("target_id", target) != target or info.get("source_id", source) != source:
            raise ValueError(f"Conflicting official target/source lineage: {video}")
        return target, source
    # Other datasets have no universal filename lineage convention.
    return info.get("target_id"), info.get("source_id")


def _audit_split_leaks(root, dataset, compression):
    sources, paths = {}, {}
    for split in ("train", "val", "test"):
        ids, frame_paths = set(), set()
        for splits in root.values():
            for video, info in _videos(splits, split, dataset, compression).items():
                target, source = _lineage(dataset, video, info)
                ids.update(str(value) for value in (target, source) if value is not None)
                frame_paths.update(canonical_path(path) for path in info.get("frames", []))
        sources[split], paths[split] = ids, frame_paths
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if paths[first] & paths[second]:
            raise ValueError(f"{dataset} {first}/{second} frame path overlap")
        if sources[first] & sources[second]:
            raise ValueError(f"{dataset} {first}/{second} source-ID overlap")


def _indexed_paths(paths, name):
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"Nonempty frame list required for {name}")
    canonical = [canonical_path(path) for path in paths]
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"Duplicate frame path in {name}")
    indexed = [(frame_index(path), path) for path in canonical]
    if len({index for index, _ in indexed}) != len(indexed):
        raise ValueError(f"Duplicate numeric frame index in {name}")
    return indexed


def build_manifest(metadata, dataset, split, label_dict, compression="c23",
                   sampling="uniform_metadata8", frames=8, *, role_scope="all_splits"):
    """Build independent full-path records from the complete dataset JSON.

    ``all_*`` is numerically sorted; legacy selection alone preserves source-list
    order. Labels become binary (configured zero versus positive method IDs).
    Source/path split audits inspect identities, without reading other splits'
    labels into the requested records. Explicit external ``selected_split`` role
    scope ignores unused mirrored metadata partitions. FF++ always audits all
    official splits. The hash binds the complete, unfiltered input metadata.
    """
    _positive_int(frames, "frames")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    if sampling not in {"uniform_metadata8", "legacy_prefix8"}:
        raise ValueError(f"Unknown sampling policy: {sampling}")
    if role_scope not in {"all_splits", "selected_split"}:
        raise ValueError("role_scope must be all_splits or selected_split")
    if dataset == "FaceForensics++" and role_scope != "all_splits":
        raise ValueError("FF++ requires all official splits to be audited")
    if isinstance(metadata, (str, Path)):
        metadata_bytes = Path(metadata).read_bytes()
        metadata = json.loads(metadata_bytes)
    else:
        metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    metadata_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    if not isinstance(metadata, dict) or dataset not in metadata:
        raise ValueError(f"Missing dataset metadata: {dataset}")
    root = metadata[dataset]
    if role_scope == "all_splits":
        _audit_split_leaks(root, dataset, compression)
    result, seen_paths, seen_videos = [], set(), set()
    for label_name, splits in root.items():
        for video, info in _videos(splits, split, dataset, compression, required=True).items():
            if info.get("label") != label_name:
                raise ValueError(f"Metadata label conflicts with group {label_name}/{video}")
            if label_name not in label_dict:
                raise ValueError(f"Label {label_name} missing from label_dict")
            configured_label = label_dict[label_name]
            if (isinstance(configured_label, bool) or not isinstance(configured_label, int)
                    or configured_label < 0):
                raise ValueError(f"Invalid configured label for {label_name}")
            original = _indexed_paths(info.get("frames"), f"{label_name}/{video}")
            ordered = sorted(original)
            video_dirs = {str(PurePosixPath(path).parent) for _, path in ordered}
            if len(video_dirs) != 1:
                raise ValueError(f"Frames cross full video identities: {video}")
            video_id = next(iter(video_dirs))
            if PurePosixPath(video_id).name != video:
                raise ValueError(f"Video key/path identity mismatch: {video}")
            if video_id in seen_videos:
                raise ValueError(f"Duplicate full video identity: {video_id}")
            all_paths = [path for _, path in ordered]
            if seen_paths.intersection(all_paths):
                raise ValueError(f"Duplicate frame path across video records: {video_id}")
            seen_videos.add(video_id)
            seen_paths.update(all_paths)
            if sampling == "legacy_prefix8":
                selected = original[:frames]
            elif len(ordered) <= frames:
                selected = ordered
            else:
                # Integer arithmetic implements floor(rank + .5) exactly.
                ranks = ([0] if frames == 1 else
                         [(2 * i * (len(ordered) - 1) + frames - 1) // (2 * (frames - 1))
                          for i in range(frames)])
                selected = [ordered[rank] for rank in ranks]
            target, source = _lineage(dataset, video, info)
            binary = int(configured_label != 0)
            if dataset == "FaceForensics++" and ((binary == 0) != (source is None)):
                raise ValueError(f"FF++ label/lineage conflict: {label_name}/{video}")
            mask_paths = info.get("masks", [])
            mask_map = {}
            if mask_paths:
                mask_map = dict(_indexed_paths(mask_paths, f"masks/{label_name}/{video}"))
                if not set(mask_map).issubset({index for index, _ in ordered}):
                    raise ValueError(f"Mask/RGB numeric index mismatch: {video}")
            method = label_name.removeprefix("FF-") if binary else None
            result.append({"video_id": video_id, "label": binary, "label_name": label_name,
                           "frames": [path for _, path in selected],
                           "indices": [index for index, _ in selected],
                           "all_frames": all_paths, "all_indices": [index for index, _ in ordered],
                           "method": method, "target_id": target, "source_id": source,
                           "masks": [mask_map.get(index) for index, _ in selected],
                           "all_masks": [mask_map.get(index) for index, _ in ordered],
                           "dataset": dataset, "split": split, "compression": compression,
                           "sampling": sampling, "role_scope": role_scope,
                           "metadata_sha256": metadata_sha256})
    if not result:
        raise ValueError(f"Empty manifest for {dataset}/{split}/{compression}")
    return sorted(result, key=lambda row: row["video_id"])


def build_pair_candidates(train_records, min_frames=2):
    """Official target-first FF++ candidates plus explicit exclusion coverage."""
    _positive_int(min_frames, "min_frames")
    if any(row.get("split") != "train" for row in train_records):
        raise ValueError("Pair candidates must come exclusively from train")
    reals = {}
    for row in train_records:
        if row["label"] == 0 and row.get("dataset") == "FaceForensics++":
            target = row["target_id"]
            if target in reals:
                raise ValueError(f"Ambiguous target real video: {target}")
            reals[target] = row
    pairs, exclusions, coverage = [], [], {}
    for fake in sorted(train_records, key=lambda row: row["video_id"]):
        if fake["label"] == 0:
            continue
        method = fake.get("method")
        counts = coverage.setdefault(method, {"candidates": 0, "included": 0, "excluded": 0})
        counts["candidates"] += 1
        real = reals.get(fake.get("target_id"))
        common = sorted(set(fake["all_indices"]) & set(real["all_indices"])) if real else []
        reason = None
        if fake.get("dataset") != "FaceForensics++" or method not in METHODS:
            reason = "unsupported_official_lineage"
        elif real is None:
            reason = "missing_target_real"
        elif len(common) < min_frames:
            reason = "insufficient_common_indices"
        if reason:
            counts["excluded"] += 1
            exclusions.append({"video_id": fake["video_id"], "method": method, "reason": reason,
                               "target_id": fake.get("target_id"), "source_id": fake.get("source_id"),
                               "common_indices": common})
            continue
        counts["included"] += 1
        pair_id = hashlib.sha256((real["video_id"] + "\n" + fake["video_id"]).encode()).hexdigest()
        pairs.append({"pair_id": pair_id, "method": method, "real": real, "fake": fake,
                      "common_indices": common})
    return {"pairs": pairs, "exclusions": exclusions, "coverage": coverage}


def verified_pairs(candidates, audit):
    """Validate explicit receipt identities, timestamps and target-face evidence.

    Receipt: {schema_version:1, receipt_id:str, pairs:[{pair_id, real_video_id,
    fake_video_id, target_id, source_id, common_indices, time_verified:true,
    target_face_verified:true, evidence:str}]}. An audited subset of candidate
    common indices is permitted; absent receipt entries remain unverified.
    """
    if isinstance(audit, (str, Path)):
        audit = json.loads(Path(audit).read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or not isinstance(audit.get("receipt_id"), str) or not audit["receipt_id"].strip():
        raise ValueError("Explicit audit receipt_id required")
    if audit.get("schema_version") != 1 or not isinstance(audit.get("pairs"), list):
        raise ValueError("Invalid audit receipt schema")
    pairs = candidates["pairs"] if isinstance(candidates, dict) else candidates
    by_id = {pair["pair_id"]: pair for pair in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("Duplicate candidate pair identity")
    result, seen = [], set()
    for entry in audit["pairs"]:
        pair_id = entry.get("pair_id")
        if pair_id in seen or pair_id not in by_id:
            raise ValueError("Unknown or duplicate audit pair identity")
        seen.add(pair_id)
        pair = by_id[pair_id]
        expected = {"real_video_id": pair["real"]["video_id"], "fake_video_id": pair["fake"]["video_id"],
                    "target_id": pair["fake"]["target_id"], "source_id": pair["fake"]["source_id"]}
        if any(entry.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Pair audit identity/lineage mismatch: {pair_id}")
        indices = entry.get("common_indices")
        if (not isinstance(indices, list) or len(indices) < 2
                or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
                or indices != sorted(set(indices)) or not set(indices).issubset(pair["common_indices"])):
            raise ValueError(f"Pair audit common-index mismatch: {pair_id}")
        if (entry.get("time_verified") is not True or entry.get("target_face_verified") is not True
                or not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip()):
            raise ValueError(f"Explicit time/target-face audit evidence required: {pair_id}")
        result.append({**pair, "common_indices": list(indices), "verified": True,
                       "audit_receipt_id": audit["receipt_id"], "audit_evidence": entry["evidence"]})
    return sorted(result, key=lambda pair: (METHODS.index(pair["method"]), pair["pair_id"]))


def _nuisance(kind, rng):
    values = {"kind": kind, "condition": NUISANCE_IDS[kind]}
    if kind.startswith("jpeg"):
        values["quality"] = int(kind[4:])
    elif kind == "resize":
        values["scale"] = .8
    elif kind == "photometry":
        values.update(brightness=rng.choice((-.02, .02)), contrast=.9)
    return values


def build_episodes(pairs, steps, seed=1024, frames=2):
    """Materialize train-only four-method episodes with distinct target sources.

    Each pair cycles shuffled indices in the first/second half of its *whole*
    audited intersection. Exposure-aware matching balances pairs within methods.
    Main nuisance is batch-global. C2 saves an independent fake-class schedule;
    every complete four-step block has identical per-class condition exposure.
    """
    _positive_int(steps, "steps")
    if frames != 2:
        raise ValueError("E1005 pair episodes require exactly two frames")
    if (not pairs or any(pair.get("verified") is not True or not pair.get("audit_receipt_id")
                         for pair in pairs)):
        raise ValueError("Episodes require explicitly verified pair receipts")
    if any(pair[role].get("split") != "train" for pair in pairs for role in ("real", "fake")):
        raise ValueError("Episodes may only contain train records")
    groups = {method: [pair for pair in pairs if pair["method"] == method] for method in METHODS}
    if any(not values for values in groups.values()):
        raise ValueError("Episodes require all four fake methods")
    if len({pair["fake"]["target_id"] for pair in pairs}) < 4:
        raise ValueError("Need four distinct target sources per episode")
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate verified pair identities")
    rng = random.Random(seed)
    exposures, queues = Counter(), {}
    kinds = ("jpeg75", "jpeg95", "resize", "photometry")
    independent_schedule = []
    while len(independent_schedule) < steps:
        block = list(kinds)
        rng.shuffle(block)
        independent_schedule.extend(block)

    def select_pairs():
        choices = {}
        for method, values in groups.items():
            ordered = list(values)
            rng.shuffle(ordered)
            choices[method] = sorted(ordered, key=lambda pair: exposures[pair["pair_id"]])
        order = sorted(METHODS, key=lambda method: len(choices[method]))

        def search(position, targets, selected):
            if position == 4:
                return selected
            method = order[position]
            # Choices with a duplicate target are interchangeable for feasibility.
            seen_targets = set()
            for pair in choices[method]:
                target = pair["fake"]["target_id"]
                if target in targets or target in seen_targets:
                    continue
                seen_targets.add(target)
                found = search(position + 1, targets | {target}, {**selected, method: pair})
                if found is not None:
                    return found
            return None

        selected = search(0, set(), {})
        if selected is None:
            raise ValueError("Cannot match four methods to distinct target sources")
        return [selected[method] for method in METHODS]

    def next_index(pair, half):
        key = (pair["pair_id"], half)
        common = pair["common_indices"]
        cut = len(common) // 2
        if len(common) < 2 or common != sorted(set(common)):
            raise ValueError("Invalid verified common intersection")
        if not queues.get(key):
            values = common[:cut] if half == 0 else common[cut:]
            queues[key] = list(values)
            rng.shuffle(queues[key])
        return queues[key].pop()

    episodes = []
    for step in range(steps):
        selected = select_pairs()
        sampled = []
        for pair in selected:
            exposures[pair["pair_id"]] += 1
            sampled.append({**pair, "indices": [next_index(pair, 0), next_index(pair, 1)]})
        permutation = list(range(4))
        while any(index == value for index, value in enumerate(permutation)):
            rng.shuffle(permutation)
        episodes.append({"step": step, "seed": seed, "pairs": sampled,
                         "shuffle_indices": permutation,
                         "nuisance": _nuisance(kinds[step % 4], rng),
                         "independent_nuisance": _nuisance(independent_schedule[step], rng)})
    return episodes


def _read_rgb(reader, path):
    import torch
    image = reader(path)
    if (not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3
            or min(image.shape[1:]) < 1 or not image.is_floating_point()
            or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1):
        raise ValueError(f"Reader must return finite RGB CHW floats in [0,1]: {path}")
    return image


def _normalizer(mean, std):
    import torch
    mean, std = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
    if mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("RGB mean/std must have three finite values and positive std")
    return mean[:, None, None], std[:, None, None]


class VideoDataset:
    """Strict padded clips; the default image output remains raw RGB [0,1]."""

    def __init__(self, records, reader, frames=8, mean=None, std=None):
        _positive_int(frames, "frames")
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be supplied together")
        self.records, self.reader, self.frames = records, reader, frames
        self.normalization = _normalizer(mean, std) if mean is not None else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        record = self.records[index]
        if not 0 < len(record["frames"]) <= self.frames:
            raise ValueError("Video frames exceed pad budget or are empty")
        if len(record["frames"]) != len(record["indices"]):
            raise ValueError("Video frame/index identity mismatch")
        images = torch.stack([_read_rgb(self.reader, path) for path in record["frames"]])
        if self.normalization is not None:
            mean, std = self.normalization
            images = (images - mean.to(images)) / std.to(images)
        count = len(images)
        padded = images.new_zeros((self.frames, *images.shape[1:]))
        padded[:count] = images
        indices = torch.full((self.frames,), -1, dtype=torch.long)
        indices[:count] = torch.tensor(record["indices"], dtype=torch.long)
        return {"image": padded, "mask": torch.arange(self.frames) < count,
                "indices": indices, "label": record["label"], "paths": list(record["frames"]),
                "video_id": record["video_id"]}


def collate_videos(rows):
    import torch
    if not rows:
        raise ValueError("Cannot collate empty videos")
    return {key: torch.stack([row[key] for row in rows]) for key in ("image", "mask", "indices")} | {
        "label": torch.tensor([row["label"] for row in rows], dtype=torch.long),
        "paths": [row["paths"] for row in rows], "video_id": [row["video_id"] for row in rows]}


def _transform(image, nuisance):
    import torch
    import torch.nn.functional as functional
    kind = nuisance["kind"]
    if kind == "photometry":
        return ((image - .5) * nuisance["contrast"] + .5 + nuisance["brightness"]).clamp(0, 1)
    if kind == "resize":
        height, width = image.shape[-2:]
        small = (max(1, math.floor(height * nuisance["scale"])),
                 max(1, math.floor(width * nuisance["scale"])))
        reduced = functional.interpolate(image[None], size=small, mode="bicubic", align_corners=False, antialias=True)
        return functional.interpolate(reduced, size=(height, width), mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)
    if kind.startswith("jpeg"):
        from PIL import Image
        pixels = (image.detach().cpu().permute(1, 2, 0) * 255).round().to(torch.uint8).contiguous()
        source = Image.frombytes("RGB", (image.shape[2], image.shape[1]), pixels.numpy().tobytes())
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", quality=nuisance["quality"])
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            data = torch.frombuffer(bytearray(decoded.convert("RGB").tobytes()), dtype=torch.uint8).to(image.dtype)
        return data.reshape(image.shape[1], image.shape[2], 3).permute(2, 0, 1).to(image.device) / 255
    if kind == "clean":
        return image
    raise ValueError(f"Unknown nuisance: {kind}")


class PairDataset:
    """One effective batch per episode: 8 videos x 2 views x 2 frames."""

    def __init__(self, episodes, reader, mean, std, independent_nuisance=False):
        self.episodes, self.reader = episodes, reader
        self.mean, self.std = _normalizer(mean, std)
        self.independent_nuisance = independent_nuisance

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        import torch
        episode = self.episodes[index]
        pairs = episode["pairs"]
        shuffle = episode["shuffle_indices"]
        if (len(pairs) != 4 or [pair["method"] for pair in pairs] != list(METHODS)
                or len({pair["fake"]["target_id"] for pair in pairs}) != 4
                or sorted(shuffle) != list(range(4))
                or any(position == value for position, value in enumerate(shuffle))):
            raise ValueError("Invalid pair episode method/source/shuffle contract")
        images, paths, video_ids, conditions = [], [], [], []
        for pair in pairs:
            if (pair.get("verified") is not True or not pair.get("audit_receipt_id")
                    or any(pair[role].get("split") != "train" for role in ("real", "fake"))
                    or pair["real"]["label"] != 0 or pair["fake"]["label"] != 1
                    or pair["real"]["target_id"] != pair["fake"]["target_id"]
                    or len(pair["indices"]) != 2 or len(set(pair["indices"])) != 2
                    or not set(pair["indices"]).issubset(pair["common_indices"])):
                raise ValueError("Pair episode requires verified train labels/indices/lineage")
            for role in ("real", "fake"):
                record = pair[role]
                lookup = dict(zip(record["all_indices"], record["all_frames"]))
                selected = [lookup[value] for value in pair["indices"]]
                raw = torch.stack([_read_rgb(self.reader, path) for path in selected])
                nuisance = (episode["independent_nuisance"] if self.independent_nuisance and role == "fake"
                            else episode["nuisance"])
                altered = torch.stack([_transform(image, nuisance) for image in raw])
                views = torch.stack((raw, altered))
                images.append((views - self.mean.to(views)) / self.std.to(views))
                paths.append(selected)
                video_ids.append(record["video_id"])
                conditions.append([0, nuisance["condition"]])
        image = torch.stack(images)
        if image.shape[:3] != (8, 2, 2):
            raise ValueError("Pair dataset must produce exact 8x2x2 image budget")
        return {"image": image, "valid_mask": torch.ones((8, 2, 2), dtype=torch.bool),
                "labels": torch.tensor([0, 1] * 4, dtype=torch.long),
                "methods": torch.tensor([value for method in range(4) for value in (-1, method)]),
                "conditions": torch.tensor(conditions, dtype=torch.long),
                "real_indices": torch.arange(0, 8, 2), "fake_indices": torch.arange(1, 8, 2),
                "shuffle_indices": torch.tensor(episode["shuffle_indices"], dtype=torch.long),
                "paths": paths, "video_id": video_ids,
                "pair_id": [pair["pair_id"] for pair in episode["pairs"]],
                "frame_indices": torch.tensor([pair["indices"] for pair in episode["pairs"] for _ in range(2)]),
                "nuisance": episode["nuisance"], "step": episode["step"]}


def collate_pairs(rows):
    if len(rows) != 1:
        raise ValueError("Pair episode collate requires DataLoader batch_size=1")
    return rows[0]


def _asset_path(root, path):
    root = Path(root).resolve()
    canonical = canonical_path(path)
    if re.match(r"^[A-Za-z]:", canonical):
        raise ValueError(f"Asset path containment violation: {path}")
    value = Path(canonical)
    resolved = (value if value.is_absolute() else root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Asset path containment violation: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def make_reader(rgb_root, resolution=224, backend="auto"):
    """Strict raw RGB reader. Prefer original cv2 INTER_CUBIC when available.

    Pillow BICUBIC fallback is recorded on ``reader.settings`` because its
    resampling differs numerically. Exact anchor comparisons should use the
    original source load_rgb callback or explicitly request backend='cv2'.
    Resolution=None preserves native pixels; callers audit crop registration.
    """
    import torch
    from PIL import Image
    if resolution is not None:
        if isinstance(resolution, int):
            _positive_int(resolution, "resolution")
            size = (resolution, resolution)
        elif len(resolution) == 2 and all(isinstance(value, int) and value > 0 for value in resolution):
            size = tuple(resolution)
        else:
            raise ValueError("resolution must be positive or a (height,width) pair")
    else:
        size = None
    if backend not in {"auto", "cv2", "pil"}:
        raise ValueError("reader backend must be auto, cv2 or pil")
    cv2 = None
    if backend != "pil":
        try:
            import cv2
        except ImportError:
            if backend == "cv2":
                raise

    def reader(path):
        resolved = _asset_path(rgb_root, path)
        if cv2 is not None:
            import numpy as np
            pixels = cv2.imdecode(np.fromfile(resolved, dtype=np.uint8), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError(f"Cannot decode RGB asset: {resolved}")
            pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            if size is not None:
                pixels = cv2.resize(pixels, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)
            return torch.from_numpy(pixels.copy()).permute(2, 0, 1).float() / 255
        with Image.open(resolved) as source:
            rgb = source.convert("RGB")
            if size is not None:
                rgb = rgb.resize((size[1], size[0]), Image.Resampling.BICUBIC)
            values = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).float()
            return values.reshape(rgb.height, rgb.width, 3).permute(2, 0, 1) / 255

    reader.settings = {"backend": "cv2" if cv2 is not None else "pil", "resolution": size,
                       "interpolation": "INTER_CUBIC" if cv2 is not None else "PIL_BICUBIC",
                       "raw_rgb": True, "backend_policy": backend,
                       "opencv_unavailable_fallback": backend == "auto" and cv2 is None}
    reader.preprocessing_sha256 = hashlib.sha256(json.dumps(reader.settings, sort_keys=True).encode()).hexdigest()
    return reader


image_reader = make_reader


def make_mask_reader(mask_root, resolution=None):
    """Decode declared masks strictly; no fabricated zero masks on errors."""
    import torch
    from PIL import Image
    if resolution is not None:
        _positive_int(resolution, "resolution")

    def reader(path):
        resolved = _asset_path(mask_root, path)
        with Image.open(resolved) as source:
            gray = source.convert("L")
            # Determine encoding before interpolation can discard extrema.
            denominator = 1 if max(gray.tobytes()) <= 1 else 255
            if resolution is not None:
                gray = gray.resize((resolution, resolution), Image.Resampling.NEAREST)
            values = torch.frombuffer(bytearray(gray.tobytes()), dtype=torch.uint8).float()
            # Binary PNGs may encode foreground as either 1 or 255. Do not
            # silently turn a legitimate 0/1 target into a 1/255 soft label.
            return values.reshape(1, gray.height, gray.width) / denominator

    return reader

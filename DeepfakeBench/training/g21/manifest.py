"""Explicit source/time mappings and collision-free video manifests."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from collections import defaultdict

from .io import read_jsonl


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"expected normalized relative path: {value!r}")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or str(p) != value:
        raise ValueError(f"unsafe/ambiguous relative path: {value!r}")
    return value


def image_path(root: str | Path, value: str) -> Path:
    root = Path(root).resolve()
    p = (root / relative_path(value)).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"path escapes data root: {value}")
    return p


def _identifier(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {name}")


def _lineage(value: object, required: bool = True) -> list[str]:
    if value is None and not required:
        return []
    if (
        not isinstance(value, list)
        or (required and not value)
        or not all(isinstance(x, str) and x.strip() for x in value)
    ):
        raise ValueError("lineage_source_ids must be a list of nonempty strings")
    if len(set(value)) != len(value):
        raise ValueError("duplicate lineage_source_ids")
    return value


def validate_pair(item: dict, min_frames: int = 2) -> None:
    if item.get("schema_version") != 1 or item.get("mapping_verified") is not True:
        raise ValueError("mapping/schema is not verified")
    for key in (
        "pair_id",
        "method",
        "content_reference_id",
        "real_video_id",
        "fake_video_id",
        "mapping_evidence",
    ):
        _identifier(item.get(key), key)
    if item.get("split") != "train":
        raise ValueError("pair manifest must be the train split")
    lineage = _lineage(item.get("lineage_source_ids"))
    if item["content_reference_id"] not in lineage:
        raise ValueError("reference missing from lineage")
    if item["real_video_id"] == item["fake_video_id"]:
        raise ValueError("same real and fake identity")
    frames = item.get("frame_pairs")
    if not isinstance(frames, list) or len(frames) < min_frames:
        raise ValueError("insufficient distinct frame pairs")
    for key in ("sample_id", "real_path", "fake_path"):
        values = [frame.get(key) for frame in frames]
        if any(not isinstance(v, str) or not v for v in values) or len(
            set(values)
        ) != len(values):
            raise ValueError(f"missing/duplicate {key}")
    for frame in frames:
        relative_path(frame["real_path"])
        relative_path(frame["fake_path"])
        if frame["real_path"] == frame["fake_path"]:
            raise ValueError("identical real/fake path")


def load_pairs(path: str | Path, min_frames: int = 2) -> list[dict]:
    rows = read_jsonl(path)
    seen = set()
    fake_ids = set()
    real_ids = {}
    reverse_real = {}
    path_owners = {}
    for item in rows:
        validate_pair(item, min_frames)
        if item["pair_id"] in seen or item["fake_video_id"] in fake_ids:
            raise ValueError("duplicate pair/fake video")
        source = item["content_reference_id"]
        if source in real_ids and real_ids[source] != item["real_video_id"]:
            raise ValueError("ambiguous source real identity")
        if (
            item["real_video_id"] in reverse_real
            and reverse_real[item["real_video_id"]] != source
        ):
            raise ValueError("real video aliased by different sources")
        reverse_real[item["real_video_id"]] = source
        for frame in item["frame_pairs"]:
            for side in ("real", "fake"):
                owner = (item[f"{side}_video_id"], side, source)
                path = frame[f"{side}_path"]
                if path in path_owners and path_owners[path] != owner:
                    raise ValueError("frame path aliased across video identities")
                path_owners[path] = owner
        real_ids[source] = item["real_video_id"]
        seen.add(item["pair_id"])
        fake_ids.add(item["fake_video_id"])
    return rows


def load_videos(path: str | Path) -> list[dict]:
    rows = read_jsonl(path)
    seen, paths = set(), set()
    for item in rows:
        if (
            item.get("schema_version") != 1
            or type(item.get("label")) is not int
            or item["label"] not in (0, 1)
        ):
            raise ValueError("invalid video schema/label")
        for key in ("dataset_id", "video_id", "split"):
            _identifier(item.get(key), key)
        identity = (item["dataset_id"], item["video_id"])
        if identity in seen:
            raise ValueError("duplicate video_id")
        seen.add(identity)
        frames = item.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("empty video")
        frame_ids = set()
        for frame in frames:
            _identifier(frame.get("frame_id"), "frame_id")
            p = relative_path(frame.get("path"))
            if frame["frame_id"] in frame_ids or p in paths:
                raise ValueError("duplicate frame_id/path")
            frame_ids.add(frame["frame_id"])
            paths.add(p)
        _lineage(item.get("lineage_source_ids"), required=item.get("split") == "val")
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("video AUC requires both classes")
    return rows


def validate_split_isolation(pairs: list[dict], videos: list[dict]) -> None:
    train_sources = {s for row in pairs for s in row["lineage_source_ids"]}
    other_sources = {
        s
        for row in videos
        for s in _lineage(row.get("lineage_source_ids"), required=True)
    }
    overlap = train_sources & other_sources
    train_paths = {
        f[k]
        for row in pairs
        for f in row["frame_pairs"]
        for k in ("real_path", "fake_path")
    }
    other_paths = {f["path"] for row in videos for f in row.get("frames", [])}
    if overlap or train_paths & other_paths:
        raise ValueError(
            f"split overlap: {sorted(overlap)}; path overlap={len(train_paths & other_paths)}"
        )


def validate_training_groups(pairs: list[dict], methods: list[str], count: int) -> dict:
    groups = defaultdict(set)
    for row in pairs:
        groups[row["method"]].add(row["content_reference_id"])
    if set(groups) != set(methods) or any(len(groups[m]) < count for m in methods):
        raise ValueError("insufficient pairs or unexpected training methods")
    return {m: len(groups[m]) for m in methods}

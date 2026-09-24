"""Dependency-light E0924 plan and fail-closed FF++ calibration partition."""

import hashlib
import re


TRAIN_JOBS = [("G26", "B0"), ("G26", "K4L20"), ("G26", "K8L20"),
              ("G26", "K4L16"), ("G26", "K8L16"),
              ("G27", "E1"), ("G27", "E2"), ("G27", "E3"), ("G27", "Full")]
DECISION_ARMS = {"G28_linear": ("model", 0), "G28_mlp": ("model", 16),
                 "G29_freq": ("freq", 16), "G29_color": ("color", 16),
                 "G29_both": ("both", 16)}


def plan(seed=1024, epochs=10):
    return {"experiment": "E0924", "seed": seed, "n_epochs": epochs,
            "training": [{"id": f"{family}_{arm}", "family": family, "arm": arm}
                         for family, arm in TRAIN_JOBS],
            "aliases": {"G27_B0": "G26_B0", "G27_G26": "G26_K4L20"},
            "decision_source": "G27_Full", "decision_arms": DECISION_ARMS,
            "calibration": "FF++ official val, source-ID connected components split 70/30",
            "selection": "base: CDF-v2 frame auc; router: fixed training budget and fixed reject threshold",
            "oracle": "threshold accuracy upper bound only, never an AUC bound"}


def canonical_path(value):
    return str(value).replace("\\", "/").removeprefix("./").rstrip("/")


def source_ids(video):
    """Conservative FF++ name grouping, not verified positive-pair lineage."""
    if not re.fullmatch(r"\d+(?:_\d+)?", video):
        raise ValueError(f"Cannot safely group FF++ video identifier: {video}")
    return set(video.split("_"))


def calibration_partition(metadata, compression="c23", seed=1024, frames=8):
    """Never substitute train/test if val is absent; check names and frame overlap.

    Union every identifier co-occurring in a validation filename, including
    across manipulation methods. This conservatively groups possible relatives;
    it is not a claim of verified source-target pairing or person-level identity.
    """
    if frames < 1:
        raise ValueError("frames must be positive")
    root = metadata["FaceForensics++"]
    records, split_ids, split_paths = [], {}, {}
    for split in ("train", "val", "test"):
        ids, paths = set(), set()
        for label, splits in root.items():
            videos = splits.get(split, {}).get(compression)
            if not videos:
                raise ValueError(f"Missing nonempty FF++ {label}/{split}/{compression}")
            for video, info in videos.items():
                participants = source_ids(video)
                ids.update(participants)
                all_frames = [canonical_path(p) for p in info["frames"]]
                paths.update(all_frames)
                if split == "val" and all_frames:
                    records.append({"video": video, "label_name": info["label"],
                                    "ids": sorted(participants), "frames": all_frames[:frames]})
        split_ids[split], split_paths[split] = ids, paths
    for a, b in (("train", "val"), ("val", "test"), ("train", "test")):
        if split_ids[a] & split_ids[b] or split_paths[a] & split_paths[b]:
            raise ValueError(f"FF++ {a}/{b} overlap: calibration would leak")
    parent = {key: key for key in split_ids["val"]}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for record in records:
        members = sorted(record["ids"])
        for key in members[1:]:
            a, b = find(members[0]), find(key)
            parent[max(a, b)] = min(a, b)
    components = {}
    for key in parent:
        components.setdefault(find(key), []).append(key)
    groups = sorted(components, key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError("Need at least two disjoint calibration components")
    cut = max(1, min(len(groups) - 1, int(.7 * len(groups))))
    fit_groups = set(groups[:cut])
    result = {"fit": [], "holdout": []}
    for record in records:
        group = find(record["ids"][0])
        record = {**record, "group": group}
        result["fit" if group in fit_groups else "holdout"].append(record)
    if any(not rows for rows in result.values()):
        raise ValueError("Empty calibration partition")
    return {"records": result, "components": components,
            "fit_groups": sorted(fit_groups), "holdout_groups": sorted(set(groups) - fit_groups),
            "seed": seed, "compression": compression,
            "grouping": "conservative filename-component grouping, not verified lineage"}

"""Build normalized manifests from explicit, user-verified FF++ mapping rules."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .io import atomic_json, file_digest, write_jsonl
from .manifest import load_pairs, relative_path

METHODS = ("FF-DF", "FF-F2F", "FF-FS", "FF-NT")
ALL_METHODS = METHODS


def mapping_template() -> dict:
    return {
        "schema_version": 1,
        "rules": {
            m: {
                "verified": False,
                "reference_component": None,
                "same_original_index_verified": False,
                "evidence": "",
            }
            for m in ALL_METHODS
        },
        "overrides": {},
    }


def _frames(entry: dict) -> dict[int, str]:
    out = {}
    for raw in entry["frames"]:
        p = relative_path(raw.replace("\\", "/"))
        index = int(PurePosixPath(p).stem)
        if index in out:
            raise ValueError(f"duplicate original frame index: {p}")
        out[index] = p
    return out


def _rule(mapping: dict, method: str, key: str) -> dict:
    rule = dict(mapping.get("rules", {}).get(method, {}))
    rule.update(mapping.get("overrides", {}).get(f"{method}/{key}", {}))
    if (
        rule.get("verified") is not True
        or not isinstance(rule.get("evidence"), str)
        or not rule["evidence"].strip()
    ):
        raise ValueError(f"unverified mapping rule: {method}/{key}")
    return rule


def _reference(mapping: dict, method: str, key: str) -> tuple[str, list[str], dict]:
    rule = _rule(mapping, method, key)
    ids = key.split("_")
    if len(ids) != 2 or not all(x.isdigit() for x in ids):
        raise ValueError(f"unsupported FF++ source identifiers: {key}")
    component = rule.get("reference_component")
    if type(component) is not int or component not in (0, 1):
        raise ValueError(
            f"reference_component must explicitly be 0 or 1: {method}/{key}"
        )
    return ids[component], sorted(set(ids)), rule


def build_ffpp(
    dataset_json: str | Path, pair_map: str | Path, output: str | Path
) -> dict:
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8-sig"))[
        "FaceForensics++"
    ]
    mapping = json.loads(Path(pair_map).read_text(encoding="utf-8-sig"))
    if mapping.get("schema_version") != 1:
        raise ValueError("unsupported mapping schema")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"manifest output already exists: {output}")
    pairs, exclusions = [], []
    real_train = data["FF-real"]["train"]["c23"]
    for method in METHODS:
        for key, entry in sorted(data[method]["train"]["c23"].items()):
            ref, lineage, rule = _reference(mapping, method, key)
            if any(x not in real_train for x in lineage):
                raise ValueError(
                    f"source missing from official train split: {method}/{key}"
                )
            real, fake = _frames(real_train[ref]), _frames(entry)
            matches = rule.get("frame_matches")
            if matches is None:
                if rule.get("same_original_index_verified") is not True:
                    raise ValueError(f"frame alignment not verified: {method}/{key}")
                matches = [[i, i] for i in sorted(real.keys() & fake.keys())]
            if not isinstance(matches, list) or any(
                not isinstance(x, list) or len(x) != 2 for x in matches
            ):
                raise ValueError(
                    "frame_matches must contain [real_index, fake_index] pairs"
                )
            matches = sorted(matches)
            if any(
                type(r) is not int
                or type(f) is not int
                or r not in real
                or f not in fake
                for r, f in matches
            ):
                raise ValueError("frame mapping references an absent original frame")
            if any(matches[i][1] >= matches[i + 1][1] for i in range(len(matches) - 1)):
                raise ValueError("frame alignment must be monotone and one-to-one")
            if len(matches) < 2:
                exclusions.append(
                    {
                        "method": method,
                        "fake_key": key,
                        "reason": "fewer than two matched frames",
                    }
                )
                continue
            pairs.append(
                {
                    "schema_version": 1,
                    "pair_id": f"ffpp/c23/{method}/{key}",
                    "split": "train",
                    "method": method,
                    "content_reference_id": ref,
                    "lineage_source_ids": lineage,
                    "real_video_id": f"ffpp/c23/FF-real/{ref}",
                    "fake_video_id": f"ffpp/c23/{method}/{key}",
                    "mapping_verified": True,
                    "mapping_evidence": rule["evidence"],
                    "alignment_mode": "explicit_indices"
                    if rule.get("frame_matches") is not None
                    else "verified_original_frame_index",
                    "frame_pairs": [
                        {
                            "sample_id": str(r),
                            "real_path": real[r],
                            "fake_path": fake[f],
                            "real_frame_index": r,
                            "fake_frame_index": f,
                        }
                        for r, f in matches
                    ],
                }
            )
    if {p["method"] for p in pairs} != set(METHODS):
        raise ValueError("a training method has no valid mapped pairs")
    output.mkdir(parents=True)
    write_jsonl(output / "pairs_train.jsonl", pairs)
    load_pairs(output / "pairs_train.jsonl")
    report = {
        "schema_version": 1,
        "protocol": "g18_g19_legacy_testall_v1",
        "pair_count": len(pairs),
        "validation_dataset": "Celeb-DF-v2 (original test loader)",
        "exclusions": exclusions,
        "pairs_by_method": {m: sum(p["method"] == m for p in pairs) for m in METHODS},
        "source_sha256": {
            "dataset_json": file_digest(dataset_json),
            "pair_map": file_digest(pair_map),
        },
    }
    atomic_json(output / "manifest_report.json", report)
    return report

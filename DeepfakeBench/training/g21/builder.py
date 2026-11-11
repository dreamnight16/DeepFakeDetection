"""Build normalized manifests from explicit, user-verified FF++ mapping rules."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .io import atomic_json, file_digest, write_jsonl
from .manifest import load_pairs, relative_path

METHODS = ("FF-DF", "FF-F2F", "FF-FS", "FF-NT")
ALL_METHODS = METHODS

EXCLUDED_KEY = "excluded"
EXCLUDE_REASON_KEY = "exclude_reason"
EXCLUDE_DETAIL_KEY = "exclude_detail"
EXPLICIT_EXCLUSION = "explicit_exclusion"
INSUFFICIENT_INTERSECTION = "insufficient_frame_intersection"
# The funnel always reports these machine-readable reasons, even at zero, so an
# operator can diff runs without the schema shifting underneath them.
EXCLUSION_REASONS = (EXPLICIT_EXCLUSION, INSUFFICIENT_INTERSECTION)


def mapping_template() -> dict:
    # No rule carries an exclusion marker: an unverified rule must fail loudly
    # rather than masquerade as an intentional opt-out.
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


def _merged_rule(mapping: dict, method: str, key: str) -> dict:
    """Merge a method-level rule with any per-video override before validation."""
    rule = dict(mapping.get("rules", {}).get(method, {}))
    rule.update(mapping.get("overrides", {}).get(f"{method}/{key}", {}))
    return rule


def _exclusion(rule: dict, method: str, key: str) -> dict | None:
    """Return a structured exclusion record when the merged rule opts out.

    The marker is explicit and opt-in and takes precedence over the verified
    gate, so a per-video override carrying it can exclude one video even when
    its method-level rule is verified. A rule that does not set the marker is
    still held to the verified/evidence contract by ``_require_verified``.
    """
    if rule.get(EXCLUDED_KEY) is not True:
        return None
    reason = rule.get(EXCLUDE_REASON_KEY)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(
            f"exclusion requires a nonempty {EXCLUDE_REASON_KEY}: {method}/{key}"
        )
    record = {
        "method": method,
        "fake_key": key,
        "reason_code": EXPLICIT_EXCLUSION,
        "reason": reason.strip(),
    }
    if EXCLUDE_DETAIL_KEY in rule:
        record["detail"] = rule[EXCLUDE_DETAIL_KEY]
    return record


def _require_verified(rule: dict, method: str, key: str) -> None:
    if (
        rule.get("verified") is not True
        or not isinstance(rule.get("evidence"), str)
        or not rule["evidence"].strip()
    ):
        raise ValueError(f"unverified mapping rule: {method}/{key}")


def _reference(rule: dict, method: str, key: str) -> tuple[str, list[str]]:
    ids = key.split("_")
    if len(ids) != 2 or not all(x.isdigit() for x in ids):
        raise ValueError(f"unsupported FF++ source identifiers: {key}")
    component = rule.get("reference_component")
    if type(component) is not int or component not in (0, 1):
        raise ValueError(
            f"reference_component must explicitly be 0 or 1: {method}/{key}"
        )
    return ids[component], sorted(set(ids))


def _funnel_bucket() -> dict:
    return {
        "fake_videos_seen": 0,
        "paired": 0,
        "excluded": 0,
        "excluded_by_reason": {},
    }


def _empty_funnel() -> dict:
    return {
        "overall": _funnel_bucket(),
        "by_method": {m: _funnel_bucket() for m in METHODS},
    }


def _record_seen(funnel: dict, method: str) -> None:
    funnel["overall"]["fake_videos_seen"] += 1
    funnel["by_method"][method]["fake_videos_seen"] += 1


def _record_paired(funnel: dict, method: str) -> None:
    funnel["overall"]["paired"] += 1
    funnel["by_method"][method]["paired"] += 1


def _record_excluded(funnel: dict, method: str, reason_code: str) -> None:
    for bucket in (funnel["overall"], funnel["by_method"][method]):
        bucket["excluded"] += 1
        reasons = bucket["excluded_by_reason"]
        reasons[reason_code] = reasons.get(reason_code, 0) + 1


def _finalize_funnel(funnel: dict) -> dict:
    buckets = [funnel["overall"], *(funnel["by_method"][m] for m in METHODS)]
    for bucket in buckets:
        present = bucket["excluded_by_reason"]
        bucket["excluded_by_reason"] = {
            code: present.get(code, 0)
            for code in sorted(set(present) | set(EXCLUSION_REASONS))
        }
    return funnel


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
    funnel = _empty_funnel()
    real_train = data["FF-real"]["train"]["c23"]
    for method in METHODS:
        for key, entry in sorted(data[method]["train"]["c23"].items()):
            _record_seen(funnel, method)
            rule = _merged_rule(mapping, method, key)
            # Precedence: an explicit exclusion marker wins over the verified
            # gate, so an opt-out is never mistaken for a missing verification.
            exclusion = _exclusion(rule, method, key)
            if exclusion is not None:
                exclusions.append(exclusion)
                _record_excluded(funnel, method, exclusion["reason_code"])
                continue
            _require_verified(rule, method, key)
            ref, lineage = _reference(rule, method, key)
            if any(x not in real_train for x in lineage):
                raise ValueError(
                    f"source missing from official train split: {method}/{key}"
                )
            real, fake = _frames(real_train[ref]), _frames(entry)
            explicit = rule.get("frame_matches") is not None
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
                record = {
                    "method": method,
                    "fake_key": key,
                    "reason_code": INSUFFICIENT_INTERSECTION,
                    "reason": "fewer than two matched frames",
                    "real_available": len(real),
                    "fake_available": len(fake),
                    "intersection": len(real.keys() & fake.keys()),
                }
                if explicit:
                    record["matched_moments"] = len(matches)
                exclusions.append(record)
                _record_excluded(funnel, method, INSUFFICIENT_INTERSECTION)
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
            _record_paired(funnel, method)
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
        "funnel": _finalize_funnel(funnel),
    }
    atomic_json(output / "manifest_report.json", report)
    return report

"""FF++ source splits and image-readable balanced pairing, independent of scores."""

from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import random

from g21.data import uniform_indices
from g21.io import atomic_json, file_digest
from g21.manifest import image_path, relative_path

from .protocol import digest, implementation_digest, protocol


def canonical_frame_path(raw, data_root=None):
    text = raw.replace("\\", "/")
    path = Path(text)
    if path.is_absolute():
        if data_root is None:
            raise ValueError("Absolute frame paths require the configured RGB data root")
        try:
            text = path.resolve().relative_to(Path(data_root).resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"Frame path is outside the configured RGB root: {raw}") from exc
    return relative_path(PurePosixPath(text).as_posix())


def frame_map(entry, data_root=None):
    result = {}
    for raw in entry["frames"]:
        path = canonical_frame_path(raw, data_root)
        frame = int(PurePosixPath(path).stem)
        if frame in result:
            raise ValueError(f"Duplicate frame index: {path}")
        result[frame] = path
    return result


def source_splits(sources, seed=1024, counts=None):
    counts = protocol()["split_counts"] if counts is None else counts
    sources = sorted(sources)
    if len(sources) != sum(counts.values()) or len(set(sources)) != len(sources):
        raise ValueError("FF++ train real source count does not match the frozen split")
    random.Random(seed).shuffle(sources)
    output, begin = {}, 0
    for name, count in counts.items():
        output[name] = sorted(sources[begin:begin + count])
        begin += count
    return output


def gate(n4, cell_coverage, n_fit):
    return n4 >= 30 and n4 >= .5 * n_fit and cell_coverage >= .9


def _index_paths(value, data_root):
    """Change only internal path spelling, never pixels, ordering or membership."""
    if isinstance(value, dict):
        return {key: [canonical_frame_path(p, data_root) if isinstance(p, str) else p for p in item]
                if key in ("frames", "masks", "landmarks") and isinstance(item, list)
                else _index_paths(item, data_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_index_paths(item, data_root) for item in value]
    return value


def inventory(dataset_json, output, *, data_root=None, statistic_mode="paired_frame"):
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8-sig"))["FaceForensics++"]
    settings = protocol(statistic_mode)
    split = source_splits(data["FF-real"]["train"]["c23"])
    report = {
        "status": "MAPPING_AND_IMAGES_REQUIRED", "P_D7_pass": False,
        "N4": None, "Ncell": None, "K_pair": None, "protocol": settings,
        "statistic_mode": statistic_mode,
        "protocol_sha256": digest(settings), "splits": split,
        "audit_implementation_sha256": implementation_digest("audit"),
        "dataset_sha256": digest(data),
        "indexed_train_videos": {method: len(data[method]["train"]["c23"])
                                 for method in ["FF-real", *settings["methods"]]},
        "note": "Index inventory only; existing DeepfakeBench crop files, no source/time correspondence yet verified.",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if data_root is not None:
        normalized = _index_paths(data, data_root)
        index_path = output / "dataset.json"
        atomic_json(index_path, {"FaceForensics++": normalized})
        report.update(dataset_json=str(index_path.resolve()), original_index_sha256=file_digest(dataset_json),
                      path_view="same original files; absolute paths expressed relative to RGB root")
    atomic_json(output / "inventory.json", report)
    atomic_json(output / "splits.json", split)
    return report


def _valid_image(root, relative):
    from PIL import Image

    try:
        with Image.open(image_path(root, relative)) as image:
            image.load()
            return image.width > 0 and image.height > 0
    except (OSError, ValueError):
        return False


def index_pairing(dataset_json, output, *, data_root=None, statistic_mode="paired_frame"):
    """Use official lineage, with temporal intersections only in the legacy mode."""
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8-sig"))["FaceForensics++"]
    methods = protocol(statistic_mode)["methods"]
    reals = data["FF-real"]["train"]["c23"]
    evidence = (
        "FF++ official filenames are <target sequence>_<source sequence>; "
        "target component 0 is the reference. "
        "https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md")
    if statistic_mode == "paired_frame":
        evidence += " Repository preprocessing preserves original cnt_frame; pairs use equal saved indices."
    mapping = {"schema_version": 1, "dataset_sha256": digest(data),
               "statistic_mode": statistic_mode,
               "pairing_mode": "ffpp_official_target_video_bags" if statistic_mode == "video_mean"
                               else "ffpp_official_target_original_index",
               "rules": {}, "overrides": {}}
    decisions = []
    for method in methods:
        for key, entry in sorted(data[method]["train"]["c23"].items()):
            ids = key.split("_")
            reason = None
            if len(ids) != 2 or not all(value.isdigit() for value in ids):
                reason = "unsupported_ffpp_filename"
            elif ids[0] not in reals:
                reason = "target_not_in_official_train_reals"
            else:
                real = frame_map(reals[ids[0]], data_root)
                fake = frame_map(entry, data_root)
                common = sorted(real.keys() & fake.keys())
                if statistic_mode == "video_mean" and min(len(real), len(fake)) < 2:
                    reason = "insufficient_independent_video_frames"
                elif statistic_mode == "paired_frame" and len(common) < 2:
                    reason = "insufficient_saved_frame_intersection"
            if reason:
                mapping["overrides"][f"{method}/{key}"] = {
                    "excluded": True, "exclude_reason": reason}
                decisions.append({"method": method, "video": key, "reason": reason})
                continue
            rule = {"verified": True, "reference_component": 0, "evidence": evidence}
            if statistic_mode == "paired_frame":
                rule.update(same_original_index_verified=True,
                            frame_matches=[[index, index] for index in common])
            mapping["overrides"][f"{method}/{key}"] = rule
            decisions.append({"method": method, "video": key, "reason": "index_pairable",
                              "reference": ids[0], "common_frames": len(common),
                              "real_frames": len(real), "fake_frames": len(fake)})
    report = {
        "status": "INDEX_PAIRING_READY", "pairing_mode": mapping["pairing_mode"],
        "index_pairable_videos": sum(row["reason"] == "index_pairable" for row in decisions),
        "excluded_videos": sum(row["reason"] != "index_pairable" for row in decisions),
        "statistic_mode": statistic_mode,
        "verification_scope": "official lineage and independent frame sets; no time alignment" if statistic_mode == "video_mean"
                              else "official lineage and equal saved frame indices; image availability is checked by audit",
        "pixel_similarity_is_gate": False, "decisions": decisions,
        "dataset_sha256": digest(data)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "pair_map.json", mapping)
    atomic_json(output / "pairing_audit.json", report)
    return report


def audit(dataset_json, data_root, pair_map, output, *, statistic_mode="paired_frame"):
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is absent: {data_root}")
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8-sig"))["FaceForensics++"]
    mapping = json.loads(Path(pair_map).read_text(encoding="utf-8-sig"))
    if mapping.get("schema_version") != 1:
        raise ValueError("Unsupported pair-map schema")
    if mapping.get("dataset_sha256") is not None and mapping["dataset_sha256"] != digest(data):
        raise ValueError("Pair map was generated from different dataset metadata")
    if mapping.get("statistic_mode", statistic_mode) != statistic_mode:
        raise ValueError("Pair map uses a different statistic mode; regenerate it")
    settings = protocol(statistic_mode)
    reals = data["FF-real"]["train"]["c23"]
    splits = source_splits(reals)
    source_split = {source: split for split, sources in splits.items() for source in sources}
    selected_real, checked = {}, {}
    invalid_paths = []

    def readable(path):
        if path not in checked:
            checked[path] = _valid_image(data_root, path)
            if not checked[path]:
                invalid_paths.append(path)
        return checked[path]

    for source, entry in sorted(reals.items()):
        frames = frame_map(entry, data_root)
        keys = sorted(frames)
        candidates = [keys[i] for i in uniform_indices(len(keys), settings["candidate_frames"])]
        selected_real[source] = {t: frames[t] for t in candidates if readable(frames[t])}

    cells = defaultdict(list)
    exclusions = []
    cross_split_lineage = []
    for method in settings["methods"]:
        for key, entry in sorted(data[method]["train"]["c23"].items()):
            rule = dict(mapping.get("rules", {}).get(method, {}))
            rule.update(mapping.get("overrides", {}).get(f"{method}/{key}", {}))
            ids = key.split("_")
            reason = None
            if rule.get("excluded") is True:
                reason = "explicit_exclusion"
            elif (rule.get("verified") is not True or not isinstance(rule.get("evidence"), str)
                  or not rule["evidence"].strip()
                  or type(rule.get("reference_component")) is not int
                  or rule["reference_component"] not in (0, 1) or len(ids) != 2):
                reason = "source_mapping_unverified"
            if reason:
                exclusions.append({"method": method, "video": key, "reason": reason})
                continue
            source = ids[rule["reference_component"]]
            if source not in reals:
                exclusions.append({"method": method, "video": key, "reason": "not_official_train_source"})
                continue
            other_sources = [value for value in ids if value != source]
            for other in other_sources:
                if source_split.get(other) != source_split[source]:
                    cross_split_lineage.append({
                        "method": method, "video": key, "content_reference_id": source,
                        "content_split": source_split[source], "other_source_id": other,
                        "other_split": source_split.get(other, "not_official_train"),
                    })
            fake = frame_map(entry, data_root)
            if statistic_mode == "video_mean":
                keys = sorted(fake)
                positions = [keys[i] for i in uniform_indices(len(keys), settings["candidate_frames"])]
                frames = {t: {"source": source, "split": source_split[source], "method": method,
                              "frame": t, "fake_path": fake[t],
                              "video_id": f"ffpp/c23/{method}/{key}"}
                          for t in positions if readable(fake[t])}
                cells[(source, method)].append(frames)
                continue
            matches = rule.get("frame_matches")
            if matches is None:
                if rule.get("same_original_index_verified") is not True:
                    exclusions.append({"method": method, "video": key, "reason": "time_alignment_unverified"})
                    continue
                matches = [[t, t] for t in sorted(selected_real[source].keys() & fake.keys())]
            if (not isinstance(matches, list)
                    or any(not isinstance(x, list) or len(x) != 2
                           or any(type(i) is not int for i in x) for x in matches)
                    or len({x[0] for x in matches}) != len(matches)
                    or len({x[1] for x in matches}) != len(matches)):
                raise ValueError(f"Ambiguous explicit frame matches: {method}/{key}")
            frames = {}
            for real_index, fake_index in matches:
                if real_index in selected_real[source] and fake_index in fake and readable(fake[fake_index]):
                    frames[real_index] = {
                        "source": source, "split": source_split[source], "method": method,
                        "frame": real_index, "real_path": selected_real[source][real_index],
                        "fake_path": fake[fake_index], "fake_frame": fake_index,
                        "video_id": f"ffpp/c23/{method}/{key}",
                    }
            cells[(source, method)].append(frames)
    # More than one mapped fake for a source/method is ambiguous; never choose
    # whichever video happened to produce better downstream statistics.
    valid = {}
    ambiguous = []
    for cell, values in cells.items():
        if len(values) == 1:
            valid[cell] = values[0]
        else:
            ambiguous.append({"source": cell[0], "method": cell[1], "videos": len(values)})

    fit = splits["fit"]
    methods = settings["methods"]
    counts = {v: {m: len(valid.get((v, m), {}))
                  if statistic_mode == "paired_frame" or len(selected_real[v]) >= 2 else 0
                  for m in methods} for v in fit}
    common = {v: sorted(set.intersection(*(set(valid.get((v, m), {})) for m in methods)))
              for v in fit}
    complete = sorted(v for v in fit if (all(n >= 2 for n in counts[v].values())
                                        if statistic_mode == "video_mean" else len(common[v]) >= 2))
    n4 = len(complete)
    coverage = sum(n >= 2 for row in counts.values() for n in row.values()) / (len(fit) * len(methods))
    passed = gate(n4, coverage, len(fit))
    k = min((len(common[v]) for v in complete), default=None) if statistic_mode == "paired_frame" else None
    balanced = []
    if passed and statistic_mode == "paired_frame":
        for v in complete:
            positions = [common[v][i] for i in uniform_indices(len(common[v]), k)]
            balanced.extend(valid[(v, m)][t] for m in methods for t in positions)
    sel_pairs = [frame for (v, _), frames in sorted(valid.items()) if v in splits["sel"]
                 for _, frame in sorted(frames.items())] if statistic_mode == "paired_frame" else []
    def video_bags(sources):
        return [{"source": v, "method": m, "video_id": next(iter(frames.values()))["video_id"],
                 "frames": [row for _, row in sorted(frames.items())]}
                for (v, m), frames in sorted(valid.items())
                if v in sources and len(selected_real[v]) >= 2 and len(frames) >= 2]
    fit_videos = video_bags(complete) if statistic_mode == "video_mean" and passed else []
    sel_videos = video_bags(splits["sel"]) if statistic_mode == "video_mean" else []
    total_valid = sum(n for row in counts.values() for n in row.values())
    complete_common = len(methods) * sum(len(common[v]) for v in complete)
    retained = len(balanced)
    report = {
        "status": "P_D7_PASS" if passed else "S0_PAUSED_P_D7", "P_D7_pass": passed,
        "N4": n4, "Ncell": coverage, "K_pair": k,
        "statistic_mode": statistic_mode,
        "retained_pairs": retained if statistic_mode == "paired_frame" else None,
        "fit_source_count": len(fit), "required_complete_sources": max(30, (len(fit) + 1) // 2),
        "n_vm": counts, "n_v": {v: min([len(selected_real[v]), *counts[v].values()])
                                if statistic_mode == "video_mean" else len(common[v]) for v in fit},
        "complete_sources": complete, "discarded_sources": sorted(set(fit) - set(complete)),
        "invalid_paths": invalid_paths, "mapping_exclusions": exclusions, "ambiguous_cells": ambiguous,
        "input_preprocessing": settings["input_preprocessing"],
        "crop_scope": settings["crop_scope"],
        "coverage_definition": settings["P_D7"].get("unit", "source x method with at least 2 same-index pairs"),
        "missing_cells_by_method": {m: sum(counts[v][m] == 0 for v in fit) for m in methods},
        "discarded_valid_pair_fraction": None if statistic_mode == "video_mean" or not total_valid else 1 - retained / total_valid,
        "discarded_common_pair_fraction": None if statistic_mode == "video_mean" or not complete_common else 1 - retained / complete_common,
        "protocol": settings, "protocol_sha256": digest(settings), "splits": splits,
        "audit_implementation_sha256": implementation_digest("audit"),
        "split_scope": "content-reference sources; not both FF++ source lineages",
        "cross_split_other_lineage": cross_split_lineage,
        "dataset_sha256": digest(data), "pair_map_sha256": digest(mapping),
        "pairing_mode": mapping.get("pairing_mode", "explicit_provided_map"),
        "data_root": str(data_root.resolve()), "fit_pairs": balanced, "sel_pairs": sel_pairs,
        "real_frames": {split: {v: [{"frame": t, "path": p}
                                    for t, p in sorted(selected_real[v].items())] for v in sources}
                        for split, sources in splits.items()},
    }
    if statistic_mode == "video_mean":
        report.update(fit_videos=fit_videos, sel_videos=sel_videos,
                      video_frame_counts={"real": {v: len(rows) for v, rows in selected_real.items()},
                                          "fake": {f"{v}/{m}": len(frames) for (v, m), frames in valid.items()}},
                      retained_fake_frames=sum(len(bag["frames"]) for bag in fit_videos),
                      frame_weighting="mean within each video, then equal source x method weight; no shared K")
    for split in ("fit", "sel", "theta", "rad"):
        empty = [v for v, frames in report["real_frames"][split].items() if not frames]
        if empty:
            report.update(status="S0_PAUSED_UNREADABLE_REAL", P_D7_pass=False,
                          unreadable_real_sources=empty)
    report["audit_sha256"] = digest(report)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "audit.json", report)
    atomic_json(output / "splits.json", splits)
    return report

"""FF++ source splits and image-verified balanced pairing, independent of scores."""

from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath
import random

from g21.data import uniform_indices
from g21.io import atomic_json, file_digest
from g21.manifest import image_path, relative_path

from .protocol import digest, implementation_digest, protocol
from .shared_crop import check_pair, load_receipt


def frame_map(entry):
    result = {}
    for raw in entry["frames"]:
        path = relative_path(raw.replace("\\", "/"))
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


def inventory(dataset_json, output):
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8-sig"))["FaceForensics++"]
    settings = protocol()
    split = source_splits(data["FF-real"]["train"]["c23"])
    report = {
        "status": "MAPPING_AND_IMAGES_REQUIRED", "P_D7_pass": False,
        "N4": None, "Ncell": None, "K_pair": None, "protocol": settings,
        "protocol_sha256": digest(settings), "splits": split,
        "audit_implementation_sha256": implementation_digest("audit"),
        "dataset_sha256": digest(data),
        "indexed_train_videos": {method: len(data[method]["train"]["c23"])
                                 for method in ["FF-real", *settings["methods"]]},
        "note": "Index inventory only. No source/time/crop correspondence or file availability verified.",
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
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


def audit(dataset_json, data_root, pair_map, output):
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root is absent: {data_root}")
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8-sig"))["FaceForensics++"]
    mapping = json.loads(Path(pair_map).read_text(encoding="utf-8-sig"))
    if mapping.get("schema_version") != 1:
        raise ValueError("Unsupported pair-map schema")
    if not mapping.get("shared_crop_receipt"):
        raise ValueError("Strict same-crop evidence is absent: generate shared crops from original videos; manual flags do not establish it")
    receipt = load_receipt(image_path(Path(pair_map).parent, mapping["shared_crop_receipt"]),
                           data_root, mapping.get("shared_crop_receipt_sha256"))
    if receipt["dataset_content_sha256"] != digest(data):
        raise ValueError("Shared crops and dataset metadata differ")
    settings = protocol()
    reals = data["FF-real"]["train"]["c23"]
    splits = source_splits(reals)
    source_split = {source: split for split, sources in splits.items() for source in sources}
    selected_real, checked = {}, {}
    invalid_paths = []
    crop_rejections = []

    def readable(path):
        if path not in checked:
            expected = receipt["files_sha256"].get(path)
            checked[path] = bool(expected and _valid_image(data_root, path)
                                 and file_digest(image_path(data_root, path)) == expected)
            if not checked[path]:
                invalid_paths.append(path)
        return checked[path]

    for source, entry in sorted(reals.items()):
        frames = frame_map(entry)
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
            fake = frame_map(entry)
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
                    if not check_pair(receipt, selected_real[source][real_index], fake[fake_index],
                                      source, real_index, fake_index):
                        crop_rejections.append({"method": method, "video": key,
                                                "real_frame": real_index, "fake_frame": fake_index})
                        continue
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
    counts = {v: {m: len(valid.get((v, m), {})) for m in methods} for v in fit}
    common = {v: sorted(set.intersection(*(set(valid.get((v, m), {})) for m in methods)))
              for v in fit}
    complete = sorted(v for v in fit if len(common[v]) >= 2)
    n4 = len(complete)
    coverage = sum(n >= 2 for row in counts.values() for n in row.values()) / (len(fit) * len(methods))
    passed = gate(n4, coverage, len(fit))
    k = min((len(common[v]) for v in complete), default=None)
    balanced = []
    if passed:
        for v in complete:
            positions = [common[v][i] for i in uniform_indices(len(common[v]), k)]
            balanced.extend(valid[(v, m)][t] for m in methods for t in positions)
    sel_pairs = [frame for (v, _), frames in sorted(valid.items()) if v in splits["sel"]
                 for _, frame in sorted(frames.items())]
    total_valid = sum(n for row in counts.values() for n in row.values())
    complete_common = len(methods) * sum(len(common[v]) for v in complete)
    retained = len(balanced)
    report = {
        "status": "P_D7_PASS" if passed else "S0_PAUSED_P_D7", "P_D7_pass": passed,
        "N4": n4, "Ncell": coverage, "K_pair": k, "retained_pairs": retained,
        "fit_source_count": len(fit), "required_complete_sources": max(30, (len(fit) + 1) // 2),
        "n_vm": counts, "n_v": {v: len(common[v]) for v in fit},
        "complete_sources": complete, "discarded_sources": sorted(set(fit) - set(complete)),
        "invalid_paths": invalid_paths, "mapping_exclusions": exclusions, "ambiguous_cells": ambiguous,
        "shared_crop_receipt_sha256": receipt["receipt_sha256"],
        "crop_frame_rejections": crop_rejections,
        "missing_cells_by_method": {m: sum(counts[v][m] == 0 for v in fit) for m in methods},
        "discarded_valid_pair_fraction": None if not total_valid else 1 - retained / total_valid,
        "discarded_common_pair_fraction": None if not complete_common else 1 - retained / complete_common,
        "protocol": settings, "protocol_sha256": digest(settings), "splits": splits,
        "audit_implementation_sha256": implementation_digest("audit"),
        "split_scope": "content-reference sources; not both FF++ source lineages",
        "cross_split_other_lineage": cross_split_lineage,
        "dataset_sha256": digest(data), "pair_map_sha256": digest(mapping),
        "data_root": str(data_root.resolve()), "fit_pairs": balanced, "sel_pairs": sel_pairs,
        "real_frames": {split: {v: [{"frame": t, "path": p}
                                    for t, p in sorted(selected_real[v].items())] for v in sources}
                        for split, sources in splits.items()},
    }
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

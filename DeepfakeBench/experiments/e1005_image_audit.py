"""Audit FF++ cropped images under the repository preprocessing contract.

The receipt establishes official target lineage, original numeric frame indices
and decodable files. It does not establish exact timestamps, face identity or
pixel registration between independently aligned real and manipulated crops.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re


METHODS = ("DF", "F2F", "FS", "NT")
OFFICIAL_LINEAGE_URL = "https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md"
PAIRING_MODE = "ffpp_original_frame_index"


class _AssetValidationError(ValueError):
    """Stable validation diagnostics authored by this module."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _canonical(value):
    if not isinstance(value, (str, Path)):
        raise _AssetValidationError("Invalid path containment: expected a string or Path")
    text = str(value).replace("\\", "/")
    if not text or "\x00" in text or ".." in text.split("/"):
        raise _AssetValidationError("Unsafe path containment")
    parts = [part for part in text.split("/") if part not in {"", "."}]
    if not parts:
        raise _AssetValidationError("Empty path containment")
    return ("/" if text.startswith("/") else "") + "/".join(parts)


def _resolved(root, value):
    canonical = _canonical(value)
    path = Path(canonical)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise _AssetValidationError("Asset path containment escapes rgb_root")
    return canonical, resolved


def _preprocessing_contract(path):
    raw = Path(path).read_bytes()
    tree = ast.parse(raw.decode("utf-8"))
    original_counter = any(
        isinstance(node, ast.For) and isinstance(node.target, ast.Name)
        and node.target.id == "cnt_frame" and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"
        and len(node.iter.args) == 1 and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "frame_count_org" for node in ast.walk(tree))
    indexed_saves = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        values = node.values
        if (len(values) == 2 and isinstance(values[0], ast.FormattedValue)
                and isinstance(values[0].value, ast.Name)
                and values[0].value.id == "cnt_frame"
                and isinstance(values[1], ast.Constant)):
            indexed_saves.add(values[1].value)
    # Do not run dlib/OpenCV or accept a script that renumbers saved crops.
    if not original_counter or not {".png", ".npy"}.issubset(indexed_saves):
        raise ValueError("Preprocessing original frame-index save contract is unavailable")
    crop = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                 and node.name == "img_align_crop"), None)
    warp, resize = {}, {}
    if crop is not None:
        for node in ast.walk(crop):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "cv2"
                    and node.args and isinstance(node.args[0], ast.Name)):
                role = node.args[0].id
                if node.func.attr == "warpAffine":
                    warp[role] = [ast.dump(value) for value in node.args[1:]]
                elif node.func.attr == "resize":
                    resize[role] = [ast.dump(value) for value in node.args[1:]]
    shared_mask_transform = (len(warp.get("img", [])) == 2
        and warp.get("img") == warp.get("mask") and bool(resize.get("img"))
        and resize.get("img") == resize.get("mask"))
    return {"script_sha256": hashlib.sha256(raw).hexdigest(),
        "contract": "cnt_frame ranges over original frame_count_org; frames and landmarks use cnt_frame filenames",
        "mask_contract": "optional masks use the same extract_aligned_face_dlib crop transform and cnt_frame filename",
        "code_contract_checked": True, "mask_contract_checked": shared_mask_transform}


def _derived(path, folder, suffix):
    canonical = _canonical(path)
    parts = canonical.split("/")
    if parts.count("frames") != 1:
        raise ValueError("Preprocessed frame path must contain one /frames/ component")
    parts[parts.index("frames")] = folder
    return str(PurePosixPath("/".join(parts)).with_suffix(suffix))


def _lookup(record):
    indices, paths = record["all_indices"], record["all_frames"]
    if (not isinstance(indices, list) or not isinstance(paths, list)
            or len(indices) != len(paths) or len(set(indices)) != len(indices)
            or any(isinstance(index, bool) or not isinstance(index, int) or index < 0
                   for index in indices)):
        raise ValueError("Invalid original frame-index/path mapping")
    for index, path in zip(indices, paths):
        stem = PurePosixPath(str(path).replace("\\", "/")).stem
        if not re.fullmatch(r"\d+", stem) or int(stem) != index:
            raise ValueError("Numeric original frame-index/path mismatch")
    return dict(zip(indices, paths))


def _lineage(pair):
    real, fake = pair["real"], pair["fake"]
    method = pair.get("method")
    if method not in METHODS or fake.get("method") != method:
        raise ValueError("Unsupported official FF++ method")
    if any(row.get("dataset") != "FaceForensics++" or row.get("split") != "train"
           for row in (real, fake)) or real.get("label") != 0 or fake.get("label") != 1:
        raise ValueError("Pair must be official FF++ training real/fake records")
    real_name = PurePosixPath(_canonical(real["video_id"])).name
    fake_name = PurePosixPath(_canonical(fake["video_id"])).name
    match = re.fullmatch(r"(\d+)_(\d+)", fake_name)
    if not re.fullmatch(r"\d+", real_name) or match is None:
        raise ValueError("Invalid official FF++ target_source filename")
    target, source = match.groups()
    if (real_name != target or real.get("target_id") != target
            or real.get("source_id") is not None or fake.get("target_id") != target
            or fake.get("source_id") != source):
        raise ValueError("Official target-first FF++ lineage mismatch")
    return _lookup(real), _lookup(fake)


def audit_preprocessed_pairs(candidates: dict, rgb_root: str | Path,
                            preprocessing_path: str | Path,
                            progress: Callable[[dict], None] | None = None) -> dict:
    """Return image-only eligible pairs, a stable report and discovered masks.

    All common original frame indices are checked. Unreadable RGB frames remove
    only their index; fewer than two surviving indices excludes the pair. Missing
    or malformed optional landmarks/masks produce diagnostics, never a fabricated
    face-identity assertion. File bytes are hashed once and decoded from those
    same bytes, with caching for real frames shared by several methods.
    """
    from PIL import Image

    root = Path(rgb_root).resolve()
    contract = _preprocessing_contract(preprocessing_path)
    if not isinstance(candidates, dict) or not isinstance(candidates.get("pairs"), list):
        raise ValueError("Pair-candidate dictionary with pairs required")
    pending = sorted(candidates["pairs"], key=lambda pair: (
        str(pair.get("method")), str(pair.get("pair_id"))))
    if len({pair["pair_id"] for pair in pending}) != len(pending):
        raise ValueError("Duplicate candidate pair identity")
    report = {"schema_version": 1, "pairing_mode": PAIRING_MODE,
        "verification_scope": "official lineage, equal original frame indices and file integrity only",
        "time_verified": False, "target_face_verified": False,
        "pixel_alignment_verified": False, "preprocessing": contract,
        "official_lineage_source": OFFICIAL_LINEAGE_URL,
        "candidate_exclusions": copy.deepcopy(candidates.get("exclusions", [])),
        "metadata_coverage": copy.deepcopy(candidates.get("coverage", {})),
        "frame_failures": [], "excluded_pairs": [], "accepted_pairs": [],
        "coverage": {method: {"candidates": 0, "included": 0, "excluded": 0,
                               "usable_common_indices": 0} for method in METHODS},
        "landmarks": {"present": 0, "valid": 0, "missing": 0, "invalid": 0},
        "masks": {"present": 0, "registered": 0, "missing": 0, "invalid": 0},
        "warnings": []}
    cache, registered_masks, result = {}, {}, []

    def inspect(kind, path):
        try:
            canonical = _canonical(path)
        except ValueError:
            canonical = str(path)
        key = (kind, canonical)
        if key in cache:
            return cache[key]
        detail = {"kind": kind, "path": canonical, "status": "invalid",
                  "sha256": None, "shape": None}
        cache[key] = detail
        try:
            _, resolved = _resolved(root, path)
            if not resolved.is_file():
                detail.update(status="missing", reason="File is missing")
                return detail
            raw = resolved.read_bytes()
            detail.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
            if kind == "landmarks":
                import numpy as np
                points = np.load(io.BytesIO(raw), allow_pickle=False)
                if not isinstance(points, np.ndarray):
                    if hasattr(points, "close"):
                        points.close()
                    raise _AssetValidationError("Landmarks require an NPY numeric array")
                detail["shape"] = list(points.shape)
                if (points.ndim != 2 or points.shape[0] < 5 or points.shape[1] != 2
                        or not np.issubdtype(points.dtype, np.number)
                        or np.iscomplexobj(points) or not np.isfinite(points).all()):
                    raise _AssetValidationError("Landmarks must be finite numeric >=5x2 coordinates")
                detail["dtype"] = str(points.dtype)
            else:
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    width, height = image.size
                    if kind == "mask":
                        gray_rgb = (image.mode == "RGB" and len({channel.tobytes()
                            for channel in image.split()}) == 1)
                        if image.mode not in {"1", "L"} and not gray_rgb:
                            raise _AssetValidationError("Mask requires grayscale/binary semantics")
                        pixels = image.convert("L").tobytes()
                        detail.update(shape=[height, width],
                            mask_encoding="binary_01" if max(pixels) <= 1 else "uint8_0255",
                            positive_fraction=sum(value != 0 for value in pixels) / len(pixels))
                    else:
                        image.convert("RGB").load()
                        detail["shape"] = [height, width, 3]
            detail["status"] = "valid"
        except (ValueError, OSError, EOFError, SyntaxError) as error:
            # Pillow embeds a BytesIO object address in some decode errors.
            # External exception strings must not change deterministic receipts.
            detail["reason"] = (str(error) if isinstance(error, _AssetValidationError)
                else f"{type(error).__name__}: invalid {kind} file: {canonical}")
        return detail

    def optional_landmark(frame):
        try:
            path = _derived(frame, "landmarks", ".npy")
        except ValueError as error:
            report["warnings"].append({"kind": "landmarks", "frame_path": frame,
                                       "reason": str(error)})
            return
        inspect("landmarks", path)

    def optional_mask(pair, index, frame, rgb):
        if not contract["mask_contract_checked"]:
            report["warnings"].append({"kind": "mask", "frame_path": frame,
                "reason": "Preprocessing shared image/mask transform contract is unavailable"})
            report["masks"]["invalid"] += 1
            return
        try:
            path = _derived(frame, "masks", ".png")
            fake = pair["fake"]
            mask_values = fake.get("all_masks", [])
            if mask_values and len(mask_values) != len(fake["all_indices"]):
                raise ValueError("Metadata mask/index mapping length mismatch")
            supplied = dict(zip(fake["all_indices"], mask_values)).get(index)
            if supplied is not None and _canonical(supplied) != path:
                raise ValueError("Metadata mask path disagrees with preprocessing /masks/ contract")
        except ValueError as error:
            report["warnings"].append({"kind": "mask", "frame_path": frame,
                                       "reason": str(error)})
            report["masks"]["invalid"] += 1
            return
        mask = inspect("mask", path)
        if mask["status"] != "valid":
            return
        if mask["shape"] != rgb["shape"][:2]:
            # Coordinates refer to this exact crop; resizing would conceal a mismatch.
            mask.update(status="invalid", reason="Mask/RGB crop dimensions disagree")
            return
        registered_masks[frame] = {"path": path, "coordinates_verified": True,
            "verification_mode": "preprocessing_contract",
            "evidence": "same-frame /masks/ crop from extract_aligned_face_dlib; script_sha256=" + contract["script_sha256"],
            "sha256": mask["sha256"], "shape": mask["shape"],
            "mask_encoding": mask["mask_encoding"],
            "positive_fraction": mask["positive_fraction"]}

    for number, pair in enumerate(pending, 1):
        method = pair.get("method")
        coverage = report["coverage"].setdefault(str(method),
            {"candidates": 0, "included": 0, "excluded": 0, "usable_common_indices": 0})
        coverage["candidates"] += 1
        reason, common, accepted = None, [], []
        try:
            real_lookup, fake_lookup = _lineage(pair)
            common = sorted(set(real_lookup) & set(fake_lookup))
            if common != pair.get("common_indices"):
                raise ValueError("Candidate common indices disagree with original-index intersection")
            if len(common) < 2:
                raise ValueError("Insufficient original common frame indices")
        except (ValueError, KeyError, TypeError) as error:
            reason = str(error)
        if reason is None:
            for index in common:
                good = True
                rgb_details = {}
                for role, lookup in (("real", real_lookup), ("fake", fake_lookup)):
                    path = lookup[index]
                    detail = inspect("rgb", path)
                    rgb_details[role] = detail
                    failure = detail.get("reason") if detail["status"] != "valid" else None
                    if failure is None:
                        try:
                            _, actual = _resolved(root, path)
                            _, expected = _resolved(root, pair[role]["video_id"])
                            if actual.parent != expected:
                                raise ValueError("RGB path/video identity mismatch")
                        except ValueError as error:
                            failure = str(error)
                    if failure is not None:
                        good = False
                        report["frame_failures"].append({"pair_id": pair["pair_id"],
                            "role": role, "index": index, "path": str(path),
                            "reason": failure})
                    else:
                        optional_landmark(path)
                if good:
                    accepted.append(index)
                    optional_mask(pair, index, fake_lookup[index], rgb_details["fake"])
            if len(accepted) < 2:
                reason = "insufficient_decodable_common_indices"
        if reason is not None:
            coverage["excluded"] += 1
            report["excluded_pairs"].append({"pair_id": pair["pair_id"], "method": method,
                "reason": reason, "common_indices": common, "usable_common_indices": accepted})
        else:
            coverage["included"] += 1
            coverage["usable_common_indices"] += len(accepted)
            row = copy.deepcopy(pair)
            row.update(common_indices=accepted, verified=True, pairing_mode=PAIRING_MODE,
                time_verified=False, target_face_verified=False, pixel_alignment_verified=False,
                audit_evidence="official target-first FF++ lineage; equal cnt_frame filenames; RGB bytes decoded and hashed; no timestamp or face-identity claim")
            result.append(row)
            report["accepted_pairs"].append({"pair_id": pair["pair_id"], "method": method,
                "real_video_id": pair["real"]["video_id"], "fake_video_id": pair["fake"]["video_id"],
                "target_id": pair["fake"]["target_id"], "source_id": pair["fake"]["source_id"],
                "common_indices": accepted})
        if progress is not None and (number == 1 or number % 50 == 0 or number == len(pending)):
            progress({"processed_pairs": number, "total_pairs": len(pending),
                "accepted_pairs": len(result), "excluded_pairs": len(report["excluded_pairs"]),
                "audited_files": len(cache), "frame_failures": len(report["frame_failures"])})
    report["audited_files"] = [cache[key] for key in sorted(cache)]
    for detail in report["audited_files"]:
        if detail["kind"] not in {"landmarks", "mask"}:
            continue
        counts = report["landmarks" if detail["kind"] == "landmarks" else "masks"]
        if detail["status"] == "missing":
            counts["missing"] += 1
        else:
            counts["present"] += 1
            counts["valid" if detail["kind"] == "landmarks" and detail["status"] == "valid"
                   else "registered" if detail["status"] == "valid" else "invalid"] += 1
        if detail["status"] == "invalid":
            report["warnings"].append({"kind": detail["kind"], "path": detail["path"],
                                       "reason": detail["reason"]})
    rgb_files = [value for value in cache.values() if value["kind"] == "rgb"]
    report["counts"] = {"candidate_pairs": len(pending), "accepted_pairs": len(result),
        "excluded_pairs": len(report["excluded_pairs"]), "audited_rgb_files": len(rgb_files),
        "valid_rgb_files": sum(value["status"] == "valid" for value in rgb_files),
        "failed_rgb_files": sum(value["status"] != "valid" for value in rgb_files)}
    report["warnings"].sort(key=lambda item: json.dumps(item, sort_keys=True))
    report["receipt_id"] = "e1005-images-" + _digest(report)
    for pair in result:
        pair["audit_receipt_id"] = report["receipt_id"]
    # Register only frames retained in eligible pairs, not orphaned failed pairs.
    eligible_frames = {dict(zip(pair["fake"]["all_indices"], pair["fake"]["all_frames"]))[index]
        for pair in result for index in pair["common_indices"]}
    registered_masks = {path: registered_masks[path]
        for path in sorted(eligible_frames & registered_masks.keys())}
    receipt = {"schema_version": 1, "receipt_id": report["receipt_id"] + "-assets",
        "registered_masks": registered_masks, "native_crops": {}}
    return {"pairs": result, "report": report, "asset_receipt": receipt}

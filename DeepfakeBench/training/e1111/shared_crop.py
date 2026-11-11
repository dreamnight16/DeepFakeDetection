"""Prepare strict real/fake crops with one real-reference affine per frame."""

from collections import defaultdict
import copy
import json
from pathlib import Path, PurePosixPath

from g21.data import uniform_indices
from g21.io import atomic_json, file_digest
from g21.manifest import image_path

from .protocol import digest, implementation_digest, protocol

MODE = "same_reference_affine_v1"


def raw_video_path(frame_path):
    path = PurePosixPath(frame_path.replace("\\", "/"))
    if path.parent.parent.name != "frames":
        raise ValueError(f"Unsupported preprocessed frame path: {frame_path}")
    return str(path.parent.parent.parent / "videos" / f"{path.parent.name}.mp4")


def reference_affine(points, size=256):
    """Use the existing preprocess.py ArcFace template, margin and scale."""
    import numpy as np
    from skimage.transform import SimilarityTransform

    points = np.asarray(points, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("Expected five finite real-reference landmarks")
    target = np.array([[30.2946, 51.6963], [65.5318, 51.5014],
                       [48.0252, 71.7366], [33.5493, 92.3655],
                       [62.7299, 92.2041]], dtype=np.float32)
    target[:, 0] += 8.
    target *= size / 112.
    margin = size * .3 / 2.
    target = (target + margin) * (size / (size + 2 * margin))
    transform = SimilarityTransform()
    if not transform.estimate(points.astype(np.float32), target):
        raise ValueError("Cannot estimate reference crop transform")
    matrix = transform.params[:2]
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-12:
        raise ValueError("Degenerate reference crop transform")
    return matrix


def landmark_estimator(predictor_path):
    import dlib
    import cv2
    import numpy as np

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))

    def estimate(frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb, 1)
        if not faces:
            raise ValueError("No real-reference face detected")
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = predictor(rgb, face)
        # Exactly the indices used by preprocessing/preprocess.py:get_keypts.
        return np.array([[shape.part(i).x, shape.part(i).y] for i in (37, 44, 30, 49, 55)])

    return estimate


def video_frames(path, indices):
    """Decode sequentially: the index is the original frame counter, not a seek."""
    import cv2

    remaining = set(indices)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open original video: {path}")
        frame_index = 0
        while remaining:
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"Original video ended before indices {sorted(remaining)}: {path}")
            if frame_index in remaining:
                remaining.remove(frame_index)
                yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def warp(frame, matrix, size=256):
    import cv2
    return cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _save(root, relative, crop):
    import cv2

    path = image_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise OSError(f"Cannot encode shared crop: {relative}")
    path.write_bytes(encoded.tobytes())
    return file_digest(path)


def prepare(dataset_json, pair_map, raw_root, predictor_path, output):
    from .audit import frame_map

    raw_root, predictor_path, output = Path(raw_root), Path(predictor_path), Path(output)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Strict shared cropping needs the original video root: {raw_root}")
    if not predictor_path.is_file():
        raise FileNotFoundError(f"Existing preprocessing landmark predictor is missing: {predictor_path}")
    data = json.loads(Path(dataset_json).read_text(encoding="utf-8-sig"))["FaceForensics++"]
    mapping = json.loads(Path(pair_map).read_text(encoding="utf-8-sig"))
    if mapping.get("schema_version") != 1:
        raise ValueError("Unsupported pairing map")
    reals = data["FF-real"]["train"]["c23"]
    # Refuse an image-only root before spending time detecting faces.
    raw_paths = {raw_video_path(entry["frames"][0]) for entry in reals.values() if entry["frames"]}
    if not any(image_path(raw_root, path).is_file() for path in raw_paths):
        raise FileNotFoundError("Original FF++ c23 videos are absent; independently cropped PNGs cannot establish a shared affine")
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    estimate = landmark_estimator(predictor_path)
    real_warps, pair_warps, hashes, failures, raw_hashes = {}, {}, {}, [], {}
    available = defaultdict(dict)

    def video_identity(relative):
        if relative not in raw_hashes:
            raw_hashes[relative] = file_digest(image_path(raw_root, relative))
        return raw_hashes[relative]

    for source, entry in sorted(reals.items()):
        frames = frame_map(entry)
        ordered = sorted(frames)
        if not ordered:
            failures.append({"source": source, "reason": "empty_real_index"})
            continue
        indices = [ordered[i] for i in uniform_indices(len(ordered), protocol()["candidate_frames"])]
        relative_video = raw_video_path(frames[indices[0]])
        try:
            raw_hash = video_identity(relative_video)
            for index, frame in video_frames(image_path(raw_root, relative_video), indices):
                try:
                    matrix = reference_affine(estimate(frame))
                    path = frames[index]
                    hashes[path] = _save(images, path, warp(frame, matrix))
                    record = {"matrix": matrix.tolist(), "matrix_digest": digest(matrix.tolist()),
                              "source": source, "frame": index, "raw_shape": list(frame.shape[:2]),
                              "raw_video": relative_video, "raw_video_sha256": raw_hash}
                    real_warps[path] = record
                    available[source][index] = (path, record)
                except ValueError as exc:
                    failures.append({"source": source, "frame": index, "reason": str(exc)})
        except OSError as exc:
            failures.append({"source": source, "reason": str(exc)})
        print(f"E1111A/shared_crop real {source}: {len(available[source])}/{len(indices)}", flush=True)

    result_map = copy.deepcopy(mapping)
    for method in protocol()["methods"]:
        for key, entry in sorted(data[method]["train"]["c23"].items()):
            rule = dict(mapping.get("rules", {}).get(method, {}))
            rule.update(mapping.get("overrides", {}).get(f"{method}/{key}", {}))
            if rule.get("excluded") is True or rule.get("verified") is not True:
                continue
            ids = key.split("_")
            component = rule.get("reference_component")
            if len(ids) != 2 or type(component) is not int or component not in (0, 1):
                raise ValueError(f"Invalid verified source map: {method}/{key}")
            source = ids[component]
            fake = frame_map(entry)
            matches = rule.get("frame_matches")
            if matches is None:
                if rule.get("same_original_index_verified") is not True:
                    raise ValueError(f"Unverified original time mapping: {method}/{key}")
                matches = [[i, i] for i in sorted(set(available[source]) & set(fake))]
            if (not isinstance(matches, list)
                    or any(not isinstance(x, list) or len(x) != 2
                           or any(type(i) is not int for i in x) for x in matches)
                    or len({x[0] for x in matches}) != len(matches)
                    or len({x[1] for x in matches}) != len(matches)):
                raise ValueError(f"Ambiguous original frame matches: {method}/{key}")
            requested = {f: r for r, f in matches if r in available[source] and f in fake}
            created = 0
            if requested:
                relative_video = raw_video_path(fake[min(requested)])
                try:
                    raw_hash = video_identity(relative_video)
                    for fake_index, frame in video_frames(image_path(raw_root, relative_video), requested):
                        real_index = requested[fake_index]
                        real_path, record = available[source][real_index]
                        if list(frame.shape[:2]) != record["raw_shape"]:
                            failures.append({"method": method, "video": key, "frame": fake_index,
                                             "reason": "raw_real_fake_geometry_mismatch"})
                            continue
                        import numpy as np
                        # There is exactly one M: reuse it rather than detecting
                        # or re-estimating any landmarks on the manipulated frame.
                        matrix = np.asarray(record["matrix"], dtype=np.float64)
                        fake_path = fake[fake_index]
                        hashes[fake_path] = _save(images, fake_path, warp(frame, matrix))
                        pair_warps[fake_path] = {"real_path": real_path, "source": source,
                                                "real_frame": real_index, "fake_frame": fake_index,
                                                "matrix_digest": record["matrix_digest"],
                                                "raw_shape": record["raw_shape"],
                                                "raw_video": relative_video, "raw_video_sha256": raw_hash}
                        created += 1
                except OSError as exc:
                    failures.append({"method": method, "video": key, "reason": str(exc)})
            update = result_map.setdefault("overrides", {}).setdefault(f"{method}/{key}", {})
            if created:
                update.update(crop_alignment_verified=True,
                              crop_alignment_evidence=f"{MODE}: matrix and crop hashes in shared_crop_receipt.json")
            else:
                update.update(excluded=True, exclude_reason="no_valid_shared_crop")
            print(f"E1111A/shared_crop {method}/{key}: {created}", flush=True)

    receipt = {"schema_version": 1, "mode": MODE, "output_root": str(images.resolve()),
               "producer_sha256": implementation_digest("crop"),
               "dataset_sha256": file_digest(dataset_json), "pair_map_sha256": file_digest(pair_map),
               "dataset_content_sha256": digest(data),
               "predictor_sha256": file_digest(predictor_path), "crop_size": 256,
               "real_warps": real_warps, "pair_warps": pair_warps, "files_sha256": hashes,
               "failures": failures, "raw_video_sha256": raw_hashes}
    receipt["receipt_sha256"] = digest(receipt)
    result_map["shared_crop_receipt"] = "shared_crop_receipt.json"
    result_map["shared_crop_receipt_sha256"] = receipt["receipt_sha256"]
    atomic_json(output / "shared_crop_receipt.json", receipt)
    atomic_json(output / "pair_map.json", result_map)
    atomic_json(output / "FaceForensics++.json", {"FaceForensics++": data})
    return {"status": "SHARED_CROP_PREPARED", "images": str(images.resolve()),
            "dataset_json": str((output / "FaceForensics++.json").resolve()),
            "pair_map": str((output / "pair_map.json").resolve()),
            "real_crops": len(real_warps), "fake_crops": len(pair_warps),
            "failures": len(failures), "receipt_sha256": receipt["receipt_sha256"]}


def load_receipt(path, data_root, expected_digest):
    receipt = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    identity = receipt.pop("receipt_sha256")
    if (digest(receipt) != identity or identity != expected_digest
            or receipt.get("mode") != MODE
            or receipt.get("producer_sha256") != implementation_digest("crop")
            or Path(receipt["output_root"]).resolve() != Path(data_root).resolve()):
        raise ValueError("Shared-crop receipt does not match its producer, images or pairing map")
    receipt["receipt_sha256"] = identity
    return receipt


def check_pair(receipt, real_path, fake_path, source, real_index, fake_index):
    pair = receipt["pair_warps"].get(fake_path)
    real = receipt["real_warps"].get(real_path)
    return bool(pair and real and pair["real_path"] == real_path
                and pair["source"] == real["source"] == source
                and pair["real_frame"] == real["frame"] == real_index
                and pair["fake_frame"] == fake_index
                and pair["raw_shape"] == real["raw_shape"]
                and pair["matrix_digest"] == real["matrix_digest"] == digest(real["matrix"]))

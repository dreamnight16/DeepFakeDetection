"""Stateless sampling plans and CPU-side, microbatch-shared RGB augmentation."""

from __future__ import annotations

from collections import defaultdict
import random

from .io import digest
from .manifest import image_path, validate_training_groups

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def uniform_indices(length: int, count: int) -> list[int]:
    if length < 1 or count < 1:
        raise ValueError("nonempty frame list and positive frame count required")
    count = min(length, count)
    if count == 1:
        return [length // 2]
    return [j * (length - 1) // (count - 1) for j in range(count)]


def _rng(seed: int, update: int, method: str, stream: str) -> random.Random:
    return random.Random(int(digest([seed, update, method, stream])[:16], 16))


def make_window(pairs: list[dict], cfg: dict, update: int) -> list[dict]:
    data, seed = cfg["data"], cfg["seed"]
    methods, count = data["train_methods"], data["pairs_per_microbatch"]
    validate_training_groups(pairs, methods, count)
    by_method = defaultdict(lambda: defaultdict(list))
    for row in pairs:
        by_method[row["method"]][row["content_reference_id"]].append(row)
    order = sorted(methods)
    _rng(seed, update, "all", "method_order").shuffle(order)
    window = []
    for method in order:
        rng = _rng(seed, update, method, "sources")
        sources = rng.sample(sorted(by_method[method]), count)
        selected = [
            rng.choice(sorted(by_method[method][s], key=lambda r: r["pair_id"]))
            for s in sources
        ]
        frames_rng = _rng(seed, update, method, "frames")
        frames = [
            sorted(
                frames_rng.sample(
                    uniform_indices(len(r["frame_pairs"]), data["training_frame_pool"]),
                    data["frames_per_training_video"],
                )
            )
            for r in selected
        ]
        aug_rng = _rng(seed, update, method, "augment")
        perm_rng = _rng(seed, update, method, "permutation")
        permutation = list(range(count))
        # Sattolo: a random single cycle, always a derangement for P >= 2.
        for i in range(count - 1, 0, -1):
            j = perm_rng.randrange(i)
            permutation[i], permutation[j] = permutation[j], permutation[i]
        plan = {
            "method": method,
            "pairs": selected,
            "frame_indices": frames,
            "flip": aug_rng.random() < data["horizontal_flip_probability"],
            "quality": aug_rng.randint(
                data["jpeg_quality_min"], data["jpeg_quality_max"]
            ),
            "permutation": permutation,
        }
        plan["digest"] = digest(plan)
        window.append(plan)
    return window


def decode_rgb(root: str, relative: str, size: int = 224):
    import cv2
    import numpy as np

    path = image_path(root, relative)
    # imdecode supports Unicode paths on Windows as well as Linux paths.
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if image is None:
        raise ValueError(f"cannot decode image: {relative}")
    return cv2.cvtColor(
        cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC),
        cv2.COLOR_BGR2RGB,
    )


def normalized_tensor(rgb):
    import numpy as np
    import torch

    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    return (x - x.new_tensor(MEAN)[:, None, None]) / x.new_tensor(STD)[:, None, None]


def render_plan(plan: dict, root: str, size: int = 224):
    import cv2
    import torch

    rows = []
    for record, indices in zip(plan["pairs"], plan["frame_indices"]):
        sides = []
        for side in ("real", "fake"):
            views = [[], []]
            for index in indices:
                path = record["frame_pairs"][index][f"{side}_path"]
                rgb = decode_rgb(root, path, size)
                if plan["flip"]:
                    rgb = rgb[:, ::-1].copy()
                ok, encoded = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, plan["quality"]],
                )
                if not ok:
                    raise ValueError(f"JPEG encoding failed: {path}")
                jpeg = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if jpeg is None:
                    raise ValueError(f"JPEG decoding failed: {path}")
                views[0].append(normalized_tensor(rgb))
                views[1].append(
                    normalized_tensor(cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB))
                )
            sides.append(torch.stack([torch.stack(v) for v in views]))
        rows.append(torch.stack(sides, dim=1))  # [A,S,T,C,H,W]
    return torch.stack(rows)


def selected_frames(videos: list[dict], count: int) -> list[dict]:
    selected = []
    for video in videos:
        for index in uniform_indices(len(video["frames"]), count):
            frame = video["frames"][index]
            selected.append(
                {
                    "dataset_id": video["dataset_id"],
                    "video_id": video["video_id"],
                    "frame_id": frame["frame_id"],
                    "label": video["label"],
                    "path": frame["path"],
                }
            )
    return selected

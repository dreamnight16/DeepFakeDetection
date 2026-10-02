"""Read-only B0 features and strict clips of existing numerically ordered frames."""

import math
from pathlib import PurePosixPath
import re

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from e0924_protocol import canonical_path


class FrozenFeatures(nn.Module):
    """Observe the original vision output without changing its token sequence."""

    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, images):
        captured = []

        def capture(module, args, output):
            hidden = (output.last_hidden_state if hasattr(output, "last_hidden_state")
                      else output["last_hidden_state"])
            captured.append(hidden[:, 1:].detach())

        self.base.eval()
        handle = self.base.backbone.register_forward_hook(capture)
        try:
            with torch.no_grad():
                original = self.base({"image": images}, inference=True)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError("Expected one original B0 vision forward")
        cls, patches = original["feat"].detach(), captured[0]
        if not all(torch.isfinite(x).all() for x in (cls, patches, original["prob"], original["cls"])):
            raise ValueError("B0 features/scores must be finite")
        return original, cls, patches


def frame_index(path):
    name = PurePosixPath(canonical_path(path)).stem
    match = re.search(r"(?:^|_)(\d+)$", name)
    if match is None:
        raise ValueError(f"Frame filename lacks a numeric index: {path}")
    return int(match.group(1))


def group_video_rows(rows, clip_frames):
    if not isinstance(clip_frames, int) or clip_frames < 1:
        raise ValueError("clip_frames must be positive")
    groups, seen = {}, set()
    for path, label in rows:
        path = canonical_path(path)
        if path in seen:
            raise ValueError(f"Duplicate frame identity: {path}")
        seen.add(path)
        video = path.rsplit("/", 1)[0]
        binary = int(label != 0)
        group = groups.setdefault(video, {"video_id": video, "label": binary, "rows": []})
        if group["label"] != binary:
            raise ValueError(f"Conflicting labels inside video: {video}")
        group["rows"].append((frame_index(path), path))
    if not groups:
        raise ValueError("Empty clip dataset")
    result = []
    for video in sorted(groups):
        group = groups[video]
        ordered = sorted(group.pop("rows"))
        if len({idx for idx, _ in ordered}) != len(ordered):
            raise ValueError(f"Duplicate numeric frame index inside {video}")
        if len(ordered) > clip_frames:
            # Uniform over available sampled frames, not adjacent original video frames.
            positions = [round(i * (len(ordered) - 1) / (clip_frames - 1))
                         for i in range(clip_frames)] if clip_frames > 1 else [0]
            ordered = [ordered[i] for i in positions]
        group.update(paths=[path for _, path in ordered], indices=[idx for idx, _ in ordered])
        result.append(group)
    return result


class StrictClips(Dataset):
    def __init__(self, source, clip_frames, records=None):
        self.source = source
        if records is None:
            if len(source.image_list) != len(source.label_list):
                raise ValueError("Misaligned frame paths/labels")
            rows = list(zip(source.image_list, source.label_list))
        else:
            rows = [(path, source.config["label_dict"][record["label_name"]])
                    for record in records for path in record["frames"]]
        self.groups = group_video_rows(rows, clip_frames)
        self.clip_frames = clip_frames

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[index]
        frames = [self.source.normalize(self.source.to_tensor(self.source.load_rgb(path)))
                  for path in group["paths"]]
        images = torch.stack(frames)
        if images.ndim != 4 or not torch.isfinite(images).all():
            raise ValueError("Clip images must be finite [T,C,H,W]")
        count = len(frames)
        padded = images.new_zeros((self.clip_frames, *images.shape[1:]))
        padded[:count] = images
        mask = torch.arange(self.clip_frames) < count
        indices = torch.full((self.clip_frames,), -1, dtype=torch.long)
        indices[:count] = torch.tensor(group["indices"])
        return {"image": padded, "mask": mask, "indices": indices, "label": group["label"],
                "video_id": group["video_id"], "paths": list(group["paths"])}


def collate_clips(batch):
    return {key: torch.stack([row[key] for row in batch]) for key in ("image", "mask", "indices")} | {
        "label": torch.tensor([row["label"] for row in batch], dtype=torch.long),
        "video_id": [row["video_id"] for row in batch], "paths": [row["paths"] for row in batch]}


def clip_loader(source, batch_size, clip_frames, records=None, train_ratio=None):
    dataset = StrictClips(source, clip_frames, records)
    kwargs = {"collate_fn": collate_clips, "num_workers": 0}
    if train_ratio is None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    else:
        # Load this standalone sampler without the dataset package's optional
        # detector/augmentation dependencies (also usable in tiny CLIP tests).
        import importlib.util
        from pathlib import Path
        from run_e1001 import validate_sampling
        validate_sampling(batch_size, train_ratio)
        sampler_path = Path(__file__).resolve().parents[1] / "training/dataset/balance_batch_sampler.py"
        spec = importlib.util.spec_from_file_location("e1002_balance_sampler", sampler_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if batch_size < 2 or not math.isfinite(train_ratio) or not 0 < train_ratio < 1:
            raise ValueError("Invalid balanced clip sampling budget")
        labels = [g["label"] for g in dataset.groups]
        if set(labels) != {0, 1}:
            raise ValueError("Clip training requires real and fake videos")
        sampler = module.BalanceBatchSampler(labels, max(1, batch_size // 2), real_ratio=train_ratio)
        loader = DataLoader(dataset, batch_sampler=sampler, **kwargs)
    loader.source_dataset = source
    return loader


def extract_clip_features(extractor, batch, device):
    images, mask = batch["image"].to(device), batch["mask"].to(device)
    original, cls, _ = extractor(images[mask])
    features = cls.new_zeros((*mask.shape, cls.shape[-1]))
    features[mask] = cls
    return original, features, mask

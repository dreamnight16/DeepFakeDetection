"""Strict registered training masks and shared native-crop E1005 datasets.

No inference dataset reads mask supervision. Native arms derive their global
image from the same native tensor used by the pixel branch, preserving frame IDs.
"""

import hashlib
import importlib.util
import json
from pathlib import Path


MASK_KINDS = {"tamper", "boundary", "shuffled_mask"}
NATIVE_KINDS = {"native448", "interpolated448"}
_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        spec = importlib.util.spec_from_file_location("e1005_assets_data", Path(__file__).with_name("e1005_data.py"))
        _DATA = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_DATA)
    return _DATA


def _receipt(value):
    if isinstance(value, (str, Path)):
        raw = Path(value).read_bytes()
        value = json.loads(raw)
    else:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or not isinstance(value.get("receipt_id"), str) or not value["receipt_id"].strip()):
        raise ValueError("Explicit registered asset receipt schema/id required")
    result = dict(value)
    for kind in ("registered_masks", "native_crops"):
        values = value.get(kind, {})
        if not isinstance(values, dict):
            raise ValueError(f"Asset receipt {kind} must be a path map")
        normalized = {}
        for path, entry in values.items():
            canonical = _data().canonical_path(path)
            if canonical in normalized or not isinstance(entry, dict):
                raise ValueError(f"Duplicate/invalid registered asset identity: {canonical}")
            normalized[canonical] = dict(entry)
        result[kind] = normalized
    return result, hashlib.sha256(raw).hexdigest()


def _registered(receipt, path, kind):
    canonical = _data().canonical_path(path)
    entry = receipt[kind].get(canonical)
    flags = (("coordinates_verified",) if kind == "registered_masks" else
             ("same_frame_verified", "crop_verified", "native_verified"))
    if (entry is None or any(entry.get(flag) is not True for flag in flags)
            or not isinstance(entry.get("evidence"), str) or not entry["evidence"].strip()
            or not entry.get("path")):
        raise ValueError(f"Missing registered {kind} coordinates/frame/crop evidence: {canonical}")
    return canonical, entry


def _image_details(root, asset_path, *, mask=False):
    from PIL import Image
    resolved = _data()._asset_path(root, asset_path)
    with Image.open(resolved) as image:
        image.load()  # Header existence alone does not establish decodability.
        width, height = image.size
        if mask:
            grayscale_rgb = False
            if image.mode == "RGB":
                red, green, blue = [channel.tobytes() for channel in image.split()]
                grayscale_rgb = red == green == blue
            if image.mode not in {"1", "L"} and not grayscale_rgb:
                raise ValueError(f"Registered mask must have explicit grayscale/binary semantics: {resolved}")
            gray = image.convert("L")
            pixels = gray.tobytes()
            positive_fraction = sum(value != 0 for value in pixels) / len(pixels)
        else:
            image.convert("RGB").load()
            positive_fraction = None
    return {"path": str(resolved), "width": width, "height": height,
            "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            "positive_fraction": positive_fraction,
            "mask_encoding": ("binary_01" if max(pixels) <= 1 else "uint8_0255") if mask else None}


def _train_paths(pairs):
    fake_paths, native_paths = set(), set()
    if not pairs:
        raise ValueError("No verified training pairs for conditional assets")
    for pair in pairs:
        if pair.get("verified") is not True or not pair.get("audit_receipt_id"):
            raise ValueError("Conditional assets require verified time/target-face pairs")
        if pair["real"]["label"] != 0 or pair["fake"]["label"] != 1:
            raise ValueError("Conditional asset pair labels must be real/fake")
        for role in ("real", "fake"):
            record = pair[role]
            if record.get("split") != "train":
                raise ValueError("Conditional asset supervision must be training only")
            lookup = dict(zip(record["all_indices"], record["all_frames"]))
            for index in pair["common_indices"]:
                if index not in lookup:
                    raise ValueError("Asset pair common indices disagree with frame identities")
                canonical = _data().canonical_path(lookup[index])
                native_paths.add(canonical)
                if role == "fake":
                    fake_paths.add(canonical)
    return sorted(fake_paths), native_paths


def _eval_paths(manifests):
    if isinstance(manifests, dict) and "frames" in manifests:
        return {_data().canonical_path(path) for path in manifests["frames"]}
    if isinstance(manifests, dict):
        values = manifests.values()
    elif isinstance(manifests, (list, tuple)):
        values = manifests
    else:
        raise ValueError("Evaluation manifests must contain video record lists")
    paths = set()
    for values_or_record in values:
        paths.update(_eval_paths(values_or_record))
    return paths


def _resolutions(global_resolution, native_resolution):
    _data()._positive_int(global_resolution, "global_resolution")
    _data()._positive_int(native_resolution, "native_resolution")
    if native_resolution <= global_resolution:
        raise ValueError("Native resolution must exceed global resolution")


def audit_assets(receipt, pairs, eval_manifests, rgb_root, global_resolution=224, native_resolution=448):
    """Audit every eligible train mask and all train/development/regression crops.

    Asset failures block their conditional family and include their exact reason;
    they do not fabricate fallback masks/crops. Native audit checks original asset
    dimensions before any interpolation. Empty individual fake masks are counted;
    a fully empty fake-mask collection cannot supply tamper supervision.
    """
    _resolutions(global_resolution, native_resolution)
    result = {"mask": {"eligible": False, "reason": "", "audited_frames": 0},
              "native": {"eligible": False, "reason": "", "audited_frames": 0},
              "audited_identities": {"mask": [], "native": []}}
    try:
        receipt, signature = _receipt(receipt)
        fake_paths, native_paths = _train_paths(pairs)
        native_paths.update(_eval_paths(eval_manifests))
    except (ValueError, OSError, KeyError) as error:
        for family in ("mask", "native"):
            result[family]["reason"] = str(error)
        return result
    result.update(receipt_id=receipt["receipt_id"], receipt_sha256=signature,
                  global_resolution=global_resolution, native_resolution=native_resolution)
    failures = {"mask": [], "native": []}
    for path in fake_paths:
        try:
            _, entry = _registered(receipt, path, "registered_masks")
            mask = _image_details(rgb_root, entry["path"], mask=True)
            rgb = _image_details(rgb_root, path)
            if (mask["width"], mask["height"]) != (rgb["width"], rgb["height"]):
                raise ValueError(f"Registered mask/RGB crop dimensions disagree: {path}")
            result["audited_identities"]["mask"].append({"frame_path": path, **mask,
                "rgb_sha256": rgb["sha256"], "evidence": entry["evidence"]})
        except (ValueError, OSError) as error:
            failures["mask"].append({"frame_path": path, "reason": str(error)})
    for path in sorted(native_paths):
        try:
            _, entry = _registered(receipt, path, "native_crops")
            native = _image_details(rgb_root, entry["path"])
            if min(native["width"], native["height"]) < native_resolution:
                raise ValueError(f"Native crop has insufficient original pixels ({native['width']}x{native['height']}): {path}")
            result["audited_identities"]["native"].append({"frame_path": path, **native,
                "evidence": entry["evidence"]})
        except (ValueError, OSError) as error:
            failures["native"].append({"frame_path": path, "reason": str(error)})
    masks = result["audited_identities"]["mask"]
    if masks and all(row["positive_fraction"] == 0 for row in masks):
        failures["mask"].append({"reason": "All fake training masks are empty; no positive tamper supervision"})
    for family in ("mask", "native"):
        audited = result["audited_identities"][family]
        result[family].update(eligible=bool(audited) and not failures[family], audited_frames=len(audited),
                              failures=failures[family], reason=failures[family][0]["reason"] if failures[family] else "verified")
    result["mask"]["empty_fraction"] = (sum(row["positive_fraction"] == 0 for row in masks) / len(masks) if masks else None)
    return result


def _native_reader(receipt, root, native_resolution):
    raw_reader = _data().make_reader(root, native_resolution, backend="pil")
    verified = set()

    def reader(path):
        canonical, entry = _registered(receipt, path, "native_crops")
        if canonical not in verified:
            detail = _image_details(root, entry["path"])
            if min(detail["width"], detail["height"]) < native_resolution:
                raise ValueError(f"Native crop has insufficient original pixels: {canonical}")
            verified.add(canonical)
        return raw_reader(entry["path"])

    reader.settings = {**raw_reader.settings, "asset_receipt_id": receipt["receipt_id"], "registered_native": True}
    return reader


def _downsample(images, resolution):
    import torch.nn.functional as functional
    leading = images.shape[:-3]
    result = functional.interpolate(images.reshape(-1, *images.shape[-3:]), size=(resolution, resolution),
                                    mode="bilinear", align_corners=False)
    return result.reshape(*leading, *result.shape[-3:])


class _NativeDataset:
    def __init__(self, source, global_resolution):
        self.source, self.global_resolution = source, global_resolution

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        row = dict(self.source[index])
        row["aux_image"] = row["image"]
        row["image"] = _downsample(row["aux_image"], self.global_resolution)
        return row


def _geometry(mask, nuisance):
    import torch.nn.functional as functional
    if nuisance["kind"] != "resize":
        return mask
    height, width = mask.shape[-2:]
    small = (max(1, int(height * nuisance["scale"])), max(1, int(width * nuisance["scale"])))
    return functional.interpolate(functional.interpolate(mask, small, mode="nearest"), (height, width), mode="nearest")


class _MaskDataset:
    def __init__(self, source, episodes, kind, receipt, root, resolution, independent_nuisance):
        self.source, self.episodes, self.kind = source, episodes, kind
        self.receipt, self.root, self.resolution = receipt, root, resolution
        self.independent_nuisance = independent_nuisance
        self.mask_reader = _data().make_mask_reader(root, resolution)
        self.checked = set()

    def __len__(self):
        return len(self.source)

    def _mask(self, path):
        canonical, entry = _registered(self.receipt, path, "registered_masks")
        if canonical not in self.checked:
            mask, rgb = _image_details(self.root, entry["path"], mask=True), _image_details(self.root, path)
            if (mask["width"], mask["height"]) != (rgb["width"], rgb["height"]):
                raise ValueError(f"Registered mask/RGB crop dimensions disagree: {path}")
            self.checked.add(canonical)
        return self.mask_reader(entry["path"])

    def __getitem__(self, index):
        import torch
        import torch.nn.functional as functional
        row = dict(self.source[index])
        episode = self.episodes[index]
        targets = []
        for video, paths in enumerate(row["paths"]):
            fake = bool(row["labels"][video])
            clean = (torch.stack([self._mask(path) for path in paths]) if fake else
                     torch.zeros((2, 1, self.resolution, self.resolution), dtype=row["image"].dtype))
            nuisance = (episode["independent_nuisance"] if self.independent_nuisance and fake else episode["nuisance"])
            views = torch.stack((clean, _geometry(clean, nuisance)))
            if self.kind == "boundary":
                shape = views.shape
                masks = views.reshape(-1, 1, self.resolution, self.resolution)
                views = (functional.max_pool2d(masks, 3, 1, 1) +
                         functional.max_pool2d(-masks, 3, 1, 1)).reshape(shape)
            elif self.kind == "shuffled_mask" and fake:
                for frame, path in enumerate(paths):
                    seed = int(hashlib.sha256(f"1024:{path}:{self.resolution}".encode()).hexdigest()[:15], 16)
                    generator = torch.Generator().manual_seed(seed)
                    permutation = torch.randperm(self.resolution ** 2, generator=generator)
                    views[:, frame] = views[:, frame].flatten(-2)[:, :, permutation].reshape(2, 1, self.resolution, self.resolution)
            targets.append(views)
        row["spatial_target"] = torch.stack(targets)
        return row


def asset_pair_dataset(episodes, reader, mean, std, kind, receipt, rgb_root,
                       global_resolution=224, native_resolution=448, independent_nuisance=False):
    """Wrap the exact E1005 episode stream with audited spatial/native assets."""
    if kind not in MASK_KINDS | NATIVE_KINDS | {"pixel224"}:
        raise ValueError(f"Unknown conditional asset kind: {kind}")
    if kind == "pixel224":
        return _data().PairDataset(episodes, reader, mean, std, independent_nuisance)
    receipt, _ = _receipt(receipt)
    if kind in NATIVE_KINDS:
        _resolutions(global_resolution, native_resolution)
        source = _data().PairDataset(episodes, _native_reader(receipt, rgb_root, native_resolution), mean, std,
                                     independent_nuisance)
        return _NativeDataset(source, global_resolution)
    source = _data().PairDataset(episodes, reader, mean, std, independent_nuisance)
    return _MaskDataset(source, episodes, kind, receipt, rgb_root, global_resolution, independent_nuisance)


def asset_video_dataset(records, reader, frames=8, mean=None, std=None, kind="pixel224", receipt=None,
                        rgb_root=None, global_resolution=224, native_resolution=448):
    """Inference is image-only even for arms trained with tamper/boundary masks."""
    if kind not in NATIVE_KINDS:
        return _data().VideoDataset(records, reader, frames, mean, std)
    _resolutions(global_resolution, native_resolution)
    receipt, _ = _receipt(receipt)
    source = _data().VideoDataset(records, _native_reader(receipt, rgb_root, native_resolution), frames, mean, std)
    return _NativeDataset(source, global_resolution)


def asset_collate_videos(rows):
    import torch
    row = _data().collate_videos(rows)
    if any("aux_image" in value for value in rows):
        if not all("aux_image" in value for value in rows):
            raise ValueError("Mixed native/non-native video batch")
        row["aux_image"] = torch.stack([value["aux_image"] for value in rows])
    return row

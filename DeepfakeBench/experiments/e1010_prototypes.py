"""E1010 G4: fixed global prototypes from frozen-B0 pixel attribution.

The caller supplies only its deterministically selected FF++ training frames.
No model is fitted here; inference never receives image or patch labels.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F


METHODS = ("ALL", "RANDOM", "TOPK")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _patch_grid(base, images):
    if (images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3
            or not images.is_floating_point()):
        raise ValueError("G4 requires nonempty floating-point RGB [B,3,H,W] images")
    if not torch.isfinite(images).all():
        raise ValueError("G4 images must be finite")
    embedding = base.backbone.embeddings.patch_embedding
    kernel, stride = tuple(embedding.kernel_size), tuple(embedding.stride)
    if (kernel != stride or tuple(embedding.padding) != (0, 0)
            or tuple(embedding.dilation) != (1, 1)):
        raise ValueError("G4 requires nonoverlapping CLIP patch embedding with no padding")
    height, width = images.shape[-2:]
    if height < kernel[0] or width < kernel[1] or height % kernel[0] or width % kernel[1]:
        raise ValueError("Image shape does not align with the CLIP patch grid")
    return height // kernel[0], width // kernel[1], kernel


def _logits(output, batch_size):
    logits = output["cls"]
    if logits.shape != (batch_size, 2) or not torch.isfinite(logits).all():
        raise ValueError("B0 must return finite [B,2] cls logits")
    return logits


@torch.no_grad()
def capture_patches(base, images):
    """Read final encoder patches from the unmodified original B0 forward."""
    rows, columns, _ = _patch_grid(base, images)
    base.eval()
    captured = []
    handle = base.backbone.encoder.layers[-1].register_forward_hook(
        lambda module, args, output: captured.append(output[0].detach()))
    try:
        output = base({"image": images}, inference=True)
    finally:
        handle.remove()
    _logits(output, len(images))
    if len(captured) != 1:
        raise ValueError("Expected exactly one final B0 encoder output")
    hidden = captured[0]
    if hidden.ndim != 3 or hidden.shape[:2] != (len(images), rows * columns + 1):
        raise ValueError("Final B0 patches do not match the RGB patch grid")
    patches = hidden[:, 1:]
    if not torch.isfinite(patches).all():
        raise ValueError("B0 patches must be finite")
    return output, patches


@torch.no_grad()
def occlusion_contributions(base, images, occlusion_batch_size=16, fill="mean"):
    """Original minus occluded fake-real logit margin, in row-major patch order.

    Occlusion replaces RGB pixels by each image/channel's spatial input mean.
    Only one bounded chunk of perturbed images is allocated at a time.
    """
    _positive_integer(occlusion_batch_size, "occlusion_batch_size")
    if fill != "mean":
        raise ValueError("G4 currently supports only image-channel mean pixel fill")
    rows, columns, kernel = _patch_grid(base, images)
    output, patches = capture_patches(base, images)
    patch_count = rows * columns
    logits = _logits(output, len(images))
    margin = logits[:, 1] - logits[:, 0]
    contributions = margin.new_empty((len(images), patch_count))
    means = images.mean((-2, -1), keepdim=True)
    total = len(images) * patch_count
    for start in range(0, total, occlusion_batch_size):
        stop = min(total, start + occlusion_batch_size)
        positions = torch.arange(start, stop, device=images.device)
        image_indices = positions // patch_count
        patch_indices = positions % patch_count
        variants = images.index_select(0, image_indices).clone()
        for index, patch in enumerate(patch_indices.tolist()):
            row, column = divmod(patch, columns)
            variants[index, :, row * kernel[0]:(row + 1) * kernel[0],
                     column * kernel[1]:(column + 1) * kernel[1]] = means[image_indices[index]]
        masked_logits = _logits(base({"image": variants}, inference=True), len(variants))
        masked_margin = masked_logits[:, 1] - masked_logits[:, 0]
        contributions.view(-1)[start:stop] = margin[image_indices] - masked_margin
    if not torch.isfinite(contributions).all():
        raise ValueError("Occlusion contributions must be finite")
    return output, patches, contributions


@torch.no_grad()
def build_prototypes(base, loader, selected_methods, top_k=16, max_frames_per_class=32,
                     seed=1024, occlusion_batch_size=16, fill="mean"):
    """Consume all preselected training records; do not select from test data.

    ALL uses every fake patch; RANDOM and TOPK use exactly K patches per fake
    frame. Every method shares the same all-real-patch mean and frame subset.
    """
    selected_methods = list(selected_methods)
    if (not selected_methods or len(set(selected_methods)) != len(selected_methods)
            or any(method not in METHODS for method in selected_methods)):
        raise ValueError("selected_methods must be a nonempty unique subset of ALL/RANDOM/TOPK")
    for name, value in (("top_k", top_k), ("max_frames_per_class", max_frames_per_class),
                        ("occlusion_batch_size", occlusion_batch_size)):
        _positive_integer(value, name)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if fill != "mean":
        raise ValueError("G4 currently supports only image-channel mean pixel fill")
    generator = torch.Generator().manual_seed(seed)
    device = next(base.parameters()).device
    sums = {method: None for method in selected_methods}
    fake_patch_counts = {method: 0 for method in selected_methods}
    audit = {method: {key: [] for key in ("path", "labels", "selected_index", "selected_count",
                                         "contributions", "baseline_prob", "baseline_margin")}
             for method in selected_methods}
    frame_counts = [0, 0]
    seen_paths = set()
    real_sum, real_patch_count = None, 0
    patch_count, feature_dim = None, None
    nonpositive, topk_selected_count = 0, 0
    for batch in loader:
        labels = torch.as_tensor(batch["label"]).detach().cpu()
        paths = list(batch["path"])
        if (labels.ndim != 1 or len(labels) != len(batch["image"]) or len(paths) != len(labels)
                or not ((labels == 0) | (labels == 1)).all()):
            raise ValueError("Prototype records require matching paths, images and binary labels")
        if any(not isinstance(path, str) or not path for path in paths):
            raise ValueError("Prototype records require nonempty string paths")
        if len(set(paths)) != len(paths) or seen_paths.intersection(paths):
            raise ValueError("Prototype training paths must be unique")
        seen_paths.update(paths)
        for label in (0, 1):
            frame_counts[label] += int((labels == label).sum())
            if frame_counts[label] > max_frames_per_class:
                raise ValueError("Prototype loader exceeds the preselected per-class frame budget")
        images = batch["image"].to(device)
        output, patches = capture_patches(base, images)
        _, current_patches, current_dim = patches.shape
        if patch_count is None:
            patch_count, feature_dim = current_patches, current_dim
            if top_k > patch_count:
                raise ValueError("top_k must not exceed the B0 patch count")
        elif (patch_count, feature_dim) != (current_patches, current_dim):
            raise ValueError("Prototype frames have inconsistent patch shape")
        fake_mask = labels == 1
        contributions = np.full((len(labels), patch_count), np.nan, dtype=np.float64)
        if "TOPK" in selected_methods and fake_mask.any():
            _, _, fake_contributions = occlusion_contributions(
                base, images[fake_mask.to(device)], occlusion_batch_size, fill)
            contributions[fake_mask.numpy()] = fake_contributions.double().cpu().numpy()
        patches_cpu = patches.float().cpu()
        real_patches = patches_cpu[labels == 0].reshape(-1, feature_dim)
        if len(real_patches):
            batch_real_sum = real_patches.sum(0)
            real_sum = batch_real_sum if real_sum is None else real_sum + batch_real_sum
            real_patch_count += len(real_patches)
        logits = _logits(output, len(images)).float().cpu()
        probabilities = output["prob"].float().cpu().numpy()
        if probabilities.shape != (len(images),) or not np.isfinite(probabilities).all():
            raise ValueError("B0 must return finite [B] fake probabilities")
        margins = (logits[:, 1] - logits[:, 0]).numpy()
        for method in selected_methods:
            indices = np.full((len(labels), patch_count), -1, dtype=np.int64)
            counts = np.empty(len(labels), dtype=np.int64)
            for frame, label in enumerate(labels.tolist()):
                if label == 0 or method == "ALL":
                    selected = torch.arange(patch_count)
                elif method == "RANDOM":
                    selected = torch.randperm(patch_count, generator=generator)[:top_k]
                else:
                    selected = torch.from_numpy(np.argsort(-contributions[frame], kind="stable")[:top_k].copy())
                    nonpositive += int((contributions[frame, selected.numpy()] <= 0).sum())
                    topk_selected_count += len(selected)
                indices[frame, :len(selected)] = selected.numpy()
                counts[frame] = len(selected)
                if label == 1:
                    selected_sum = patches_cpu[frame, selected].sum(0)
                    sums[method] = selected_sum if sums[method] is None else sums[method] + selected_sum
                    fake_patch_counts[method] += len(selected)
            values = {"path": np.asarray(paths), "labels": labels.long().numpy(),
                      "selected_index": indices, "selected_count": counts,
                      "contributions": contributions.copy(), "baseline_prob": probabilities,
                      "baseline_margin": margins}
            for key, value in values.items():
                audit[method][key].append(value)
    if min(frame_counts) < 1:
        raise ValueError("Global prototypes require at least one real and one fake training frame")
    for method in selected_methods:
        audit[method] = {key: np.concatenate(values) for key, values in audit[method].items()}
    settings = dict(prototype_scope="global", source_split="train", top_k=top_k,
                    max_frames_per_class=max_frames_per_class, seed=seed, fill=fill,
                    attribution="pixel_occlusion_logit_margin", occlusion_batch_size=occlusion_batch_size,
                    feature_source="last_encoder_output_patch", selected_methods=selected_methods)
    artifacts = {}
    for method in selected_methods:
        artifacts[method] = {
            "family": "E1010_G4", "method": method,
            "real_prototype": real_sum / real_patch_count,
            "fake_prototype": sums[method] / fake_patch_counts[method],
            "feature_dim": feature_dim, "patch_count": patch_count,
            "settings": dict(settings),
            "counts": {"real_frames": frame_counts[0], "fake_frames": frame_counts[1],
                       "real_patches": real_patch_count, "fake_patches": fake_patch_counts[method]},
            "selectionmetadata": {"path": audit[method]["path"].tolist(),
                                  "labels": audit[method]["labels"].tolist(),
                                  "topk_nonpositive_fraction": (nonpositive / topk_selected_count
                                                               if method == "TOPK" else None)}}
        validate_artifact(artifacts[method])
    return artifacts, audit


def validate_artifact(artifact, base_sha256=None, patch_count=None, feature_dim=None,
                      expected_method=None, expected_settings=None):
    """Reject incompatible/corrupt prototype files before test inference."""
    if artifact.get("family") != "E1010_G4" or artifact.get("method") not in METHODS:
        raise ValueError("Invalid E1010 G4 prototype method or family")
    if expected_method is not None and artifact["method"] != expected_method:
        raise ValueError("Prototype method differs from requested method")
    if base_sha256 is not None and artifact.get("base_sha256") != base_sha256:
        raise ValueError("Prototype artifact belongs to a different B0 checkpoint")
    settings = artifact.get("settings", {})
    if expected_settings is not None and settings != expected_settings:
        raise ValueError("Prototype settings differ from requested settings")
    if settings.get("prototype_scope") != "global" or settings.get("source_split") != "train":
        raise ValueError("G4 requires global prototypes constructed from training records")
    dim, count = artifact.get("feature_dim"), artifact.get("patch_count")
    _positive_integer(dim, "feature_dim")
    _positive_integer(count, "patch_count")
    _positive_integer(settings.get("top_k"), "top_k")
    if settings["top_k"] > count:
        raise ValueError("Prototype top_k exceeds patch count")
    if (patch_count is not None and count != patch_count
            or feature_dim is not None and dim != feature_dim):
        raise ValueError("Prototype patch shape differs from B0 patch shape")
    for name in ("real_prototype", "fake_prototype"):
        prototype = artifact.get(name)
        if (not isinstance(prototype, torch.Tensor) or prototype.shape != (dim,)
                or prototype.device.type != "cpu" or not prototype.is_floating_point()):
            raise ValueError("Prototype must be a CPU floating-point vector of the expected shape")
        if not torch.isfinite(prototype).all():
            raise ValueError("Prototype vectors must be finite")
    counts = artifact.get("counts", {})
    for name in ("real_frames", "fake_frames", "real_patches", "fake_patches"):
        _positive_integer(counts.get(name), name)
    if counts["real_patches"] != counts["real_frames"] * count:
        raise ValueError("Real prototype must contain all real patches")
    per_fake = count if artifact["method"] == "ALL" else settings["top_k"]
    if counts["fake_patches"] != counts["fake_frames"] * per_fake:
        raise ValueError("Fake prototype count differs from its method")
    return artifact


@torch.no_grad()
def score_prototypes(patches, artifact, temperature=.1, pool_top_k=16):
    """Same top-K mean sigmoid cosine-difference readout for every method."""
    validate_artifact(artifact)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Prototype temperature must be finite and positive")
    _positive_integer(pool_top_k, "pool_top_k")
    if (patches.ndim != 3 or patches.shape[1:] != (artifact["patch_count"], artifact["feature_dim"])
            or not patches.is_floating_point() or not torch.isfinite(patches).all()):
        raise ValueError("Test patches must have finite [B,P,D] prototype-compatible shape")
    if pool_top_k > patches.shape[1]:
        raise ValueError("Readout K must not exceed the patch count")
    normalized = F.normalize(patches.float(), dim=-1)
    real = F.normalize(artifact["real_prototype"].to(normalized), dim=0)
    fake = F.normalize(artifact["fake_prototype"].to(normalized), dim=0)
    likelihood = ((normalized @ fake - normalized @ real) / temperature).sigmoid()
    return likelihood.topk(pool_top_k, dim=1).values.mean(1)

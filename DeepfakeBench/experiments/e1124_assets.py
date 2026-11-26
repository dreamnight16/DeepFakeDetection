"""Fit fixed E1124 bases using only the original FF++ training input."""

import hashlib
import json
from pathlib import Path

import torch

from e0924_export import strict_loader
from e0924_protocol import canonical_path
from e1119_assets import frozen_vision, matched_manifest, save_asset
from e1119_model import hooks, projections
from g21.io import atomic_json, digest, file_digest
from run_e1119 import read
from run_g25 import TRAIN_DS


METHODS = ("FF-DF", "FF-F2F", "FF-FS", "FF-NT")
RECEIVER = "encoder.layers.23.self_attn.q_proj"


def random_basis(width, rank, tag, device):
    generator = torch.Generator().manual_seed(int(digest([1024, tag])[:16], 16))
    x = torch.randn(width, rank, generator=generator, dtype=torch.float64).to(device)
    return torch.linalg.qr(x, mode="reduced").Q


def top_basis(cov, count):
    values, vectors = torch.linalg.eigh(cov)
    threshold = values[-1] * (cov.shape[0] * torch.finfo(values.dtype).eps)
    if not torch.isfinite(values).all() or values[-count] <= threshold or values[-count] <= 0:
        raise ValueError(f"Fewer than {count} positive directions for the registered basis")
    return vectors[:, -count:].flip(1)


def anchors(folder, original, files):
    complete = read(original / "complete.json")["sha256"]
    for kind in ("P01", "P04"):
        paths = {key for key in complete if key.startswith(kind + "/")}
        if len(paths) != 48:
            raise ValueError(f"Expected 48 reference bases for {kind}")
        for key in sorted(paths):
            source = original / f"{key}.pt"
            if file_digest(source) != complete[key]:
                raise ValueError(f"Reference basis digest differs: {source}")
            basis = torch.load(source, map_location="cpu", weights_only=True)
            if basis.shape != (1024, 128):
                raise ValueError(f"Unexpected shape for {source}")
            save_asset(folder / kind, source.stem, basis, files)
    save_asset(folder / "random", RECEIVER, random_basis(1024, 128, "route", "cpu").float(), files)


def fitted_real95(folder, vision, dataset, config, points, batch_size, device, original, files):
    historical = read(original / "references.json")
    index = Path(config["dataset_json_folder"]) / f"{TRAIN_DS}.json"
    manifest = matched_manifest(index, dataset.image_list)
    if historical["sources"] != manifest:
        raise ValueError("The original E1119 activation reference inputs changed")
    records = [row[side] for row in manifest for side in ("real", "fake")]
    moments = {point: torch.zeros(1024, 1024, device=device) for point in points}
    count = 0
    with hooks() as handles:
        for point, module in points.items():
            def collect(_, args, point=point):
                x = args[0][real].float().flatten(0, 1)
                moments[point].addmm_(x.T, x)

            handles.append(module.register_forward_pre_hook(collect))
        for batch in strict_loader(dataset, batch_size, records):
            real = (batch["label"] == 0).to(device)
            count += int(real.sum()) * 257
            vision(batch["image"].to(device))

    metadata = {}
    for point, moment in moments.items():
        values, vectors = torch.linalg.eigh((moment / count).double())
        spectrum = values.flip(0).clamp_min(0)
        reference = historical["projections"][point]["P01"]
        old_values = spectrum.new_tensor(reference["eigenvalues"]).clamp_min(0)
        if count != reference["tokens"] or not torch.allclose(
                spectrum / spectrum.sum(), old_values / old_values.sum(), rtol=.02, atol=1e-5):
            raise ValueError(f"E1119 reference spectrum differs at {point}")
        old_basis = torch.load(original / "P01" / f"{point}.pt", weights_only=True).to(vectors)
        overlap = (vectors[:, -128:].T @ old_basis).square().sum().item() / 128
        if overlap < .99:
            raise ValueError(f"E1119 top-128 subspace differs at {point}: {overlap}")
        cumulative = spectrum.cumsum(0) / spectrum.sum()
        k = int(torch.searchsorted(cumulative, .95)) + 1
        metadata[point] = {"rank": k, "tokens": count, "top128_overlap": overlap,
                           "energy": float(cumulative[k - 1])}
        save_asset(folder / "V95", point, vectors[:, -k:].flip(1).float().cpu(), files)
        save_asset(folder / "V95R", point, random_basis(1024, k, f"V95R:{point}", device).float().cpu(), files)
    atomic_json(folder / "spectra.json", metadata)


def frame_map(clip):
    return {int(Path(canonical_path(path)).stem): path for path in clip["frames"][:8]}


def pair_index(index):
    entries = json.loads(index.read_text(encoding="utf-8-sig"))[TRAIN_DS]
    return {key: entries[key]["train"]["c23"] for key in ("FF-real", *METHODS)}


def registered_manifest(index, references_sha256):
    clips = pair_index(index)
    groups = {}
    for fake in sorted(set.intersection(*(set(clips[method]) for method in METHODS))):
        names = fake.split("_")
        if len(names) != 2 or names[0] not in clips["FF-real"]:
            continue
        rows = [clips["FF-real"][names[0]], *(clips[method][fake] for method in METHODS)]
        frames = sorted(set.intersection(*(set(frame_map(row)) for row in rows)))
        if len(frames) >= 2:
            group = "_".join(sorted(names))
            groups.setdefault(group, []).append({"real": names[0], "fake": fake,
                                                 "common_numeric_frame_ids": frames})
    keys = sorted(groups, key=lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest())
    fit = len(keys) * 4 // 5
    return {"schema": 1, "index_sha256": file_digest(index), "methods": list(METHODS),
            "activation_references_sha256": references_sha256, "groups": groups,
            "fit_group_keys": keys[:fit], "holdout_group_keys": keys[fit:]}


def paired_paths(manifest, index, native):
    fit, holdout = (manifest[key] for key in ("fit_group_keys", "holdout_group_keys"))
    keys = fit + holdout
    groups = manifest["groups"]
    ids = [{i for name in side for i in name.split("_")} for side in (fit, holdout)]
    if len(fit) != 67 or len(holdout) != 17 or len(keys) != len(set(keys)) or set(keys) != set(groups) or ids[0] & ids[1]:
        raise ValueError("The registered E1124 fit/holdout source partition is invalid")
    directories = pair_index(index)
    paths, lookup, records, sides = {}, [], [], []
    for side in (fit, holdout):
        frame_counts = []
        for group in side:
            for pair in groups[group]:
                clips = [directories["FF-real"][pair["real"]]]
                clips.extend(directories[key][pair["fake"]] for key in METHODS)
                maps = list(map(frame_map, clips))
                if sorted(pair["fake"].split("_")) != group.split("_") or pair["fake"].split("_")[0] != pair["real"]:
                    raise ValueError(f"Pair does not belong to its declared source group: {pair}")
                frames = pair["common_numeric_frame_ids"]
                if len(frames) < 2 or frames != sorted(set.intersection(*(set(m) for m in maps))):
                    raise ValueError(f"Registered frame intersection changed: {pair}")
                frame_counts.append(len(frames))
                for number in frames:
                    row = []
                    for clip, mapping in zip(clips, maps):
                        path = mapping[number]
                        canonical = canonical_path(path)
                        if canonical not in native:
                            raise ValueError(f"Pair references a frame outside the native training set: {path}")
                        if canonical not in paths:
                            paths[canonical] = len(records)
                            records.append({"label_name": clip["label"], "frames": [path]})
                        row.append(paths[canonical])
                    lookup.append(row)
        sides.append(frame_counts)
    if [(len(counts), sum(counts)) for counts in sides] != [(134, 348), (33, 87)]:
        raise ValueError("The registered frame or source counts changed")
    return records, lookup, sides


def weighted_moments(samples, counts):
    real, fake = samples[:, 0], samples[:, 1:]
    center = fake.mean(dim=1)
    weight = samples.new_tensor([1 / (len(counts) * n) for n in counts for _ in range(n)]).sqrt()
    d = ((fake - center[:, None]) * (weight / 2)[:, None, None]).flatten(0, 1)
    g = (center - real) * weight[:, None]
    r = real * weight[:, None]
    return d.T @ d, g.T @ g, r.T @ r


def fit_pairs(folder, vision, dataset, config, points, batch_size, device, manifest_path, files):
    manifest = read(manifest_path)
    index = Path(config["dataset_json_folder"]) / f"{TRAIN_DS}.json"
    if manifest["index_sha256"] != file_digest(index) or tuple(manifest["methods"]) != METHODS:
        raise ValueError("The registered paired frame manifest does not match FF++")
    records, indices, counts = paired_paths(manifest, index, set(map(canonical_path, dataset.image_list)))
    collected = {point: [] for point in points}
    with hooks() as handles:
        for point, module in points.items():
            def collect(_, args, point=point):
                x = args[0]
                if x.shape[1:] != (257, 1024):
                    raise ValueError(f"Unexpected CLIP token shape: {x.shape}")
                collected[point].append(x[:, 1:].float().mean(1).cpu())

            handles.append(module.register_forward_pre_hook(collect))
        for batch in strict_loader(dataset, batch_size, records):
            vision(batch["image"].to(device))

    index_rows = torch.tensor(indices, device=device)
    split = sum(counts[0])
    diagnostics = {}
    for point in points:
        samples = torch.cat(collected.pop(point)).to(device=device, dtype=torch.float64)[index_rows]
        method, common, real = weighted_moments(samples[:split], counts[0])
        hold_method, hold_common, _ = weighted_moments(samples[split:], counts[1])
        common_basis = top_basis(common, 4)
        control_basis = random_basis(1024, 4, f"N4:{point}", device)
        identity = torch.eye(1024, dtype=method.dtype, device=device)
        bases = {"N0": random_basis(1024, 16, f"N0:{point}", device),
                 "N1": top_basis(method, 16), "N2": top_basis(real, 16)}
        for kind, preserved in (("N3", common_basis), ("N4", control_basis)):
            h = identity - preserved @ preserved.T
            bases[kind] = top_basis(h @ method @ h, 16)
            if (preserved.T @ bases[kind]).norm().item() > 1e-6:
                raise ValueError(f"Preserved and restricted directions overlap: {point}/{kind}")
        diagnostics[point] = {}
        for kind, basis in bases.items():
            orth = (basis.T @ basis - identity[:16, :16]).norm().item()
            if orth > 1e-6:
                raise ValueError(f"Nonorthogonal basis: {point}/{kind}")
            diagnostics[point][kind] = {
                "orthogonality_error": orth,
                "method_coverage": None if hold_method.trace() == 0 else float((basis * (hold_method @ basis)).sum() / hold_method.trace()),
                "common_coverage": None if hold_common.trace() == 0 else float((basis * (hold_common @ basis)).sum() / hold_common.trace()),
                "overlap_preserved": (float((common_basis.T @ basis).norm()) if kind == "N3" else
                                      float((control_basis.T @ basis).norm()) if kind == "N4" else None),
            }
            save_asset(folder / kind, point, basis.float().cpu(), files)
    atomic_json(folder / "held_out_coverage.json", diagnostics)


@torch.no_grad()
def build(kind, folder, config, settings, batch_size, device="cuda:0"):
    import experiment_utils as utilities

    files = {}
    original = Path(settings["activation"])
    manifest_path = Path(settings["pair_manifest"])
    if kind == "anchors":
        anchors(folder, original, files)
    else:
        vision = frozen_vision(config, device)
        if kind == "spectral":
            for name, module in projections(vision):
                left, _, right_t = torch.linalg.svd(module.weight.float(), full_matrices=False)
                save_asset(folder / "weight", name,
                           {"left": left[:, :128].contiguous().cpu(), "right": right_t[:128].T.contiguous().cpu()}, files)
        else:
            points = {name: mod for name, mod in projections(vision) if name.endswith(("q_proj", "out_proj"))}
            dataset = utilities.get_train_loader(config).dataset
            if kind == "real95":
                fitted_real95(folder, vision, dataset, config, points, batch_size, device, original, files)
            elif kind == "paired":
                fit_pairs(folder, vision, dataset, config, points, batch_size, device, manifest_path, files)
            else:
                raise ValueError(f"Unknown E1124 asset: {kind}")
    atomic_json(folder / "complete.json", {"kind": kind, "sha256": files,
                                          "source_material": settings["source_material"]})

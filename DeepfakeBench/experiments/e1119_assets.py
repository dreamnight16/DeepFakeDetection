"""Fit fixed E1119 model assets from the original CLIP and FF++ train frames."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import torch

from e1119_model import ASSET_SEED, EFFORT_REVISION, hooks, projections
from e0924_export import strict_loader
from e0924_protocol import canonical_path
from g21.io import atomic_json, digest, file_digest
from run_g25 import TRAIN_DS


METHODS = ("FF-DF", "FF-F2F", "FF-FS", "FF-NT")
K = 128


def frozen_vision(config, device):
    from transformers import CLIPModel

    return CLIPModel.from_pretrained(config["clip_pretrained_path"]).vision_model.to(device).eval()


def haar(rows, cols, tag, device):
    seed = int(digest([ASSET_SEED, tag])[:16], 16)
    generator = torch.Generator().manual_seed(seed)
    sample = torch.randn((rows, cols), generator=generator, dtype=torch.float32).to(device)
    q, r = torch.linalg.qr(sample, mode="reduced")
    return q * torch.where(r.diagonal() < 0, -1, 1)


def matched_manifest(index, native_frames):
    methods = json.loads(index.read_text(encoding="utf-8-sig"))[TRAIN_DS]
    native_frames = {canonical_path(p) for p in native_frames}

    def with_frames(entries):
        for video, entry in sorted(entries.items()):
            frames = [p for p in entry["frames"] if canonical_path(p) in native_frames]
            if len(frames) == 8:
                yield video, {"label_name": entry["label"], "frames": frames}

    reals = dict(with_frames(methods["FF-real"]["train"]["c23"]))
    fakes = defaultdict(dict)
    for method in METHODS:
        for video, entry in with_frames(methods[method]["train"]["c23"]):
            ids = video.split("_")
            if len(ids) == 2 and all(x.isdigit() for x in ids) and ids[0] in reals:
                fakes[ids[0]].setdefault(method, (video, entry))
    ranked = sorted(fakes, key=lambda source: hashlib.sha256(source.encode("utf-8")).hexdigest())
    ranked = ranked[:len(ranked) // 2 * 2]
    if not ranked:
        raise ValueError("E1119 protected bases require at least two matched FF++ sources with eight frames")
    rows = []
    for i, source in enumerate(ranked):
        method = next(METHODS[(i + j) % len(METHODS)] for j in range(len(METHODS))
                      if METHODS[(i + j) % len(METHODS)] in fakes[source])
        video, fake = fakes[source][method]
        rows.append({"source": source, "method": method, "fake_video": video,
                     "real": reals[source], "fake": fake})
    extras_are_real = True
    for method in METHODS:
        stratum = [row for row in rows if row["method"] == method]
        for pos, row in enumerate(stratum):
            row["mixed"] = "real" if (pos % 2 == 0) == extras_are_real else "fake"
        if len(stratum) % 2:
            extras_are_real = not extras_are_real
    return rows


def save_asset(folder, name, value, files):
    path = folder / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)
    files[f"{folder.name}/{name}"] = file_digest(path)


@torch.no_grad()
def build_assets(kind, folder, config, batch_size, device):
    import experiment_utils as utilities

    vision = frozen_vision(config, device)
    files = {}
    if kind == "spectral":
        for name, module in projections(vision):
            u, s, vh = torch.linalg.svd(module.weight.float(), full_matrices=False)
            save_asset(folder / "svd", name, {"u": u.cpu(), "s": s.cpu(), "vh": vh.cpu()}, files)
            width = len(s)
            u = haar(width, width, f"{name}:left", device)
            vh = haar(width, width, f"{name}:right", device).T.contiguous()
            save_asset(folder / "haar", name, {"u": u.cpu(), "s": s.cpu(), "vh": vh.cpu()}, files)
    else:
        points = {name: mod for name, mod in projections(vision) if name.endswith(("q_proj", "out_proj"))}
        if kind == "random":
            for name, mod in points.items():
                save_asset(folder / "P02", name, haar(mod.in_features, K, f"{name}:protected", device).cpu(), files)
        else:
            dataset = utilities.get_train_loader(config).dataset
            index = Path(config["dataset_json_folder"]) / f"{TRAIN_DS}.json"
            manifest = matched_manifest(index, dataset.image_list)
            fit_activation_bases(folder, vision, dataset, manifest, points, batch_size, device, files)
    report = {"kind": kind, "seed": ASSET_SEED, "sha256": files}
    if kind == "spectral":
        report["effort_source_revision"] = EFFORT_REVISION
    atomic_json(folder / "complete.json", report)


def fit_activation_bases(folder, vision, dataset, manifest, points, batch_size, device, files):
    selected = {canonical_path(p) for row in manifest for p in row[row["mixed"]]["frames"]}
    records = [row[side] for row in manifest for side in ("real", "fake")]
    loader = strict_loader(dataset, batch_size, records)
    moments = {name: torch.zeros((3, mod.in_features, mod.in_features), device=device)
               for name, mod in points.items()}
    token_counts = Counter()
    masks = None
    with hooks() as handles:
        for name, module in points.items():
            def accumulate(_, args, point=name):
                x = args[0].float()
                for index, mask in enumerate(masks):
                    if len(mask):
                        tokens = x.index_select(0, mask).flatten(0, 1)
                        moments[point][index].addmm_(tokens.T, tokens)
                        token_counts[(point, index)] += len(tokens)

            handles.append(module.register_forward_pre_hook(accumulate))
        for batch in loader:
            labels = batch["label"]
            mixed = torch.tensor([p in selected for p in batch["path"]])
            masks = tuple(mask.nonzero().flatten().to(device) for mask in (labels == 0, labels == 1, mixed))
            vision(batch["image"].to(device))
    spectra = {}
    for point, values in moments.items():
        spectra[point] = {}
        bases = {}
        for index, arm in enumerate(("P01", "P03", "P04")):
            count = token_counts[(point, index)]
            eigenvalues, basis = torch.linalg.eigh((values[index] / count).double())
            basis = basis[:, -K:].flip(1).float().cpu().contiguous()
            bases[arm] = basis
            spectra[point][arm] = {"tokens": count, "eigenvalues": eigenvalues.flip(0).cpu().tolist(),
                                   "basis_norm": basis.norm().item()}
            save_asset(folder / arm, point, basis, files)
        spectra[point]["P01_P03_overlap"] = (bases["P01"].T @ bases["P03"]).square().sum().item() / K
    atomic_json(folder / "references.json", {
        "sources": manifest, "count_per_arm": len(manifest), "frames_per_video": 8,
        "methods": dict(Counter(row["method"] for row in manifest)),
        "mixed_by_method_and_label": dict(Counter(f'{row["method"]}:{row["mixed"]}' for row in manifest)),
        "projections": spectra,
    })

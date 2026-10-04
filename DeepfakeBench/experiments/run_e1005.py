"""E1005 / G31--G36: source-controlled training and baseline-paired analysis."""

import argparse
from copy import deepcopy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import e1005_data as data_api
import e1005_protocol as protocol
import e1005_reporting as reporting
import run_e1001 as legacy

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


class Progress:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()

    def update(self, stage, gid, dataset="", completed=0, total=0, **details):
        value = {"experiment": "E1005", "stage": stage, "g_id": gid, "dataset": dataset,
                 "completed": completed, "total": total, "elapsed_seconds": time.monotonic() - self.started,
                 "utc_time": datetime.datetime.now(datetime.timezone.utc).isoformat(), **details}
        write_json(self.root / "progress.json", value)
        extra = " ".join(f"{key}={entry}" for key, entry in details.items())
        message = f"[{value['utc_time']}] {stage} {gid} {dataset} {completed}/{total} {extra}".strip()
        print(message, flush=True)
        with (self.root / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")
            stream.flush()


def core_module(name):
    key = f"e1005_{name}_runtime"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "training/detectors" / f"e1005_{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def snapshot_artifact(model, base_sha256, signature, step):
    legacy.core_module().FrozenEvidenceSidecar._validate_hash(base_sha256)
    legacy.core_module().FrozenEvidenceSidecar._validate_hash(signature)
    return {"version": 1, "experiment": "E1005", "base_sha256": base_sha256,
            "training_signature": signature, "step": step, "settings": deepcopy(model.settings),
            "state": {key: value.detach().cpu().clone() for key, value in model.snapshot_state().items()}}


def restore_artifact(model, artifact, base_sha256, signature):
    if (artifact.get("version") != 1 or artifact.get("experiment") != "E1005" or
            artifact.get("base_sha256") != base_sha256 or artifact.get("training_signature") != signature):
        raise ValueError("Checkpoint identity does not match E1005 source/training signature")
    model.restore_snapshot(artifact["state"])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_run", type=Path)
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--base_config", type=Path)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1005")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--arms", nargs="+", help="G31--G36 family, full G ID, or original design alias")
    parser.add_argument("--steps", type=int, default=5750)
    parser.add_argument("--base_selected_steps", type=int, help="Audited inherited B0 optimizer budget")
    parser.add_argument("--cold_steps", type=int, help="Explicit cold-start total budget; provenance is recorded")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--train_frames", type=int, default=8)
    parser.add_argument("--eval_frames", type=int, default=8)
    parser.add_argument("--eval_videos", type=int, default=4)
    parser.add_argument("--eval_every", type=int, default=575)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--cold_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=.01)
    parser.add_argument("--keep_weight", type=float, default=.1)
    parser.add_argument("--rank_weight", type=float, default=1.)
    parser.add_argument("--risk_tau", type=float, default=.5)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--last_layers", type=int, default=4)
    parser.add_argument("--adapter_width", type=int, default=32)
    parser.add_argument("--local_layers", nargs="+", type=int, default=[4, 8, 12])
    parser.add_argument("--late_layers", nargs="+", type=int, default=[16, 20, 22])
    parser.add_argument("--pixel_layers", nargs="+", type=int, default=[8, 12])
    parser.add_argument("--native_resolution", type=int, default=448)
    parser.add_argument("--pair_audit", type=Path)
    parser.add_argument("--asset_audit", type=Path)
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root", type=Path)
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--reader_backend", choices=("auto", "cv2", "pil"), default="auto")
    parser.add_argument("--val_fpr", type=float, default=.05)
    parser.add_argument("--bootstrap_repeats", type=int, default=2000)
    parser.add_argument("--repeat_runs", type=int, default=1)
    parser.add_argument("--no_evaluation", action="store_true", help="Train/select only; never access regression scores")
    parser.add_argument("--allow_nondeterministic", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--audit_metadata", action="store_true", help="No checkpoint/model needed; write pairing candidates")
    return parser


def _config(args, source=None):
    import yaml
    if source is not None:
        config = deepcopy(source["config"])
    elif args.base_config:
        config = json.loads(args.base_config.read_text())
    else:
        config = yaml.safe_load((ROOT / "training/config/train_config.yaml").read_text())
    evaluation = yaml.safe_load((ROOT / "training/config/test_config.yaml").read_text())
    config.setdefault("label_dict", {})
    for name, value in evaluation["label_dict"].items():
        if name in config["label_dict"] and (config["label_dict"][name] != 0) != (value != 0):
            raise ValueError(f"Conflicting binary label mapping: {name}")
        config["label_dict"].setdefault(name, value)
    config["dataset_json_folder"] = str(args.dataset_json_folder or config.get(
        "dataset_json_folder", ROOT / "preprocessing/dataset_json"))
    config["rgb_root_override"] = str(args.rgb_root or config.get("rgb_root_override", "/home/user1/effort/data"))
    if args.clip_pretrained_path:
        config["clip_pretrained_path"] = args.clip_pretrained_path
    config["multi_crop"] = False
    for name in ("mean", "std"):
        if name not in config or len(config[name]) != 3 or not all(math.isfinite(float(x)) for x in config[name]):
            raise ValueError(f"Original B0 RGB {name} must contain three finite entries")
    if any(value <= 0 for value in config["std"]):
        raise ValueError("RGB std must be positive")
    return config


def prepare_data(args, config):
    folder = Path(config["dataset_json_folder"])
    labels, compression = config["label_dict"], config.get("compression", "c23")
    ff_path = folder / "FaceForensics++.json"
    print("E1005 metadata: FaceForensics++ train/val with full split audit", file=sys.stderr, flush=True)
    train = data_api.build_manifest(ff_path, "FaceForensics++", "train", labels, compression,
                                    frames=args.train_frames)
    source_val = data_api.build_manifest(ff_path, "FaceForensics++", "val", labels, compression,
                                         frames=args.eval_frames)
    manifests = {"source_val": source_val}
    metadata_hashes = {"FaceForensics++": legacy.file_sha256(ff_path)}
    unavailable = []
    for dataset in ["Celeb-DF-v2", *protocol.REGRESSION_DATASETS]:
        path = folder / f"{dataset}.json"
        if args.audit_metadata and not path.is_file():
            unavailable.append(dataset)
            continue
        print(f"E1005 metadata: {dataset} selected test split", file=sys.stderr, flush=True)
        metadata_hashes[dataset] = legacy.file_sha256(path)
        manifests[dataset] = data_api.build_manifest(path, dataset, "test", labels, compression,
                                    frames=args.eval_frames, role_scope="selected_split")
    legacy_val = data_api.build_manifest(ff_path, "FaceForensics++", "val", labels, compression,
                                        sampling="legacy_prefix8", frames=args.eval_frames)
    candidates = data_api.build_pair_candidates(train)
    pairs = data_api.verified_pairs(candidates, args.pair_audit) if args.pair_audit else []
    return {"train": train, "manifests": manifests, "legacy_val": legacy_val,
            "metadata_sha256": metadata_hashes, "candidates": candidates, "pairs": pairs,
            "pair_audit_sha256": legacy.file_sha256(args.pair_audit) if args.pair_audit else None,
            "unavailable_metadata": unavailable}


def training_settings(args, config, options, budget):
    return {"seed": args.seed, "steps": budget, "train_frames": args.train_frames,
            "eval_frames": args.eval_frames, "lr": args.lr, "cold_lr": args.cold_lr,
            "weight_decay": args.weight_decay, "keep_weight": args.keep_weight,
            "rank_weight": args.rank_weight, "risk_tau": args.risk_tau, "options": options,
            "original_augmentation": bool(config.get("use_data_augmentation", False)),
            "real_ratio": float(config.get("sampler_real_ratio", .3)),
            "deterministic_algorithms": not args.allow_nondeterministic}


def evaluation_steps(budget, every):
    return sorted({0, budget, *[value for value in (100, 250, 575) if value < budget],
                   *range(every, budget, every)})


def _augmentation(config, seed):
    if not config.get("use_data_augmentation", False):
        return lambda image: image
    from types import SimpleNamespace
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    transform = DeepfakeAbstractBaseDataset.init_data_aug_method(SimpleNamespace(config=config))
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)

    def augment(image):
        import torch
        pixels = (image.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
        value = transform(image=pixels)["image"].copy()
        return torch.from_numpy(value).permute(2, 0, 1).float() / 255
    return augment


def frame_batches(records, reader, config, steps, seed):
    import numpy as np
    import torch
    path = ROOT / "training/dataset/balance_batch_sampler.py"
    spec = importlib.util.spec_from_file_location("e1005_frame_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [(frame, record["label"]) for record in records for frame in record["frames"]]
    sampler = module.BalanceBatchSampler([label for _, label in rows], 16,
                                        real_ratio=float(config.get("sampler_real_ratio", .3)))
    augment = _augmentation(config, seed)
    mean = torch.tensor(config["mean"])[:, None, None]
    std = torch.tensor(config["std"])[:, None, None]
    count = 0
    while count < steps:
        for indices in sampler:
            selected = [rows[index] for index in indices]
            images = torch.stack([(augment(reader(path)) - mean) / std for path, _ in selected])
            yield {"image": images[:, None, None], "valid_mask": torch.ones((len(selected), 1, 1), dtype=torch.bool),
                   "labels": torch.tensor([label for _, label in selected], dtype=torch.long)}
            count += 1
            if count == steps:
                break


def packed_forward(model, batch, device):
    import torch
    images, mask = batch["image"].to(device), batch["valid_mask"].to(device)
    inputs = {"image": images[mask]}
    if "aux_image" in batch:
        inputs["aux_image"] = batch["aux_image"].to(device)[mask]
    values = model(inputs, inference=False)
    logits = values["cls"].new_zeros((*mask.shape, 2))
    logits[mask] = values["cls"]
    result = {"logits": logits}
    if "feat" in values:
        features = values["feat"].new_zeros((*mask.shape, values["feat"].shape[-1]))
        features[mask] = values["feat"]
        result["features"] = features
    if "mask_logits" in values:
        result["mask_logits"] = values["mask_logits"]
    return result


def _eval_loader(records, reader, config, args, native=False, asset_receipt=None):
    from torch.utils.data import DataLoader
    if native:
        import e1005_assets as assets
        dataset = assets.asset_video_dataset(records, reader, args.eval_frames, config["mean"], config["std"],
                        "native448", asset_receipt, config["rgb_root_override"], config["resolution"], args.native_resolution)
        collate = assets.asset_collate_videos
    else:
        dataset = data_api.VideoDataset(records, reader, args.eval_frames, config["mean"], config["std"])
        collate = data_api.collate_videos
    return DataLoader(dataset, batch_size=args.eval_videos, shuffle=False, num_workers=0, collate_fn=collate)


def export_scores(model, base, records, reader, config, args, device, progress, gid, dataset,
                  native=False, asset_receipt=None):
    import numpy as np
    import torch
    loader = _eval_loader(records, reader, config, args, native, asset_receipt)
    progress.update("baseline" if model is None else "export", gid, dataset, 0, len(loader))
    if model is not None:
        model.eval()
    base.eval()
    chunks = {key: [] for key in ("labels", "path", "video_id", "cls_prob", "global_log_odds", "score")}
    observed = 0
    with torch.no_grad():
        for index, batch in enumerate(loader, 1):
            mask = batch["mask"].to(device)
            inputs = {"image": batch["image"].to(device)[mask]}
            if "aux_image" in batch:
                inputs["aux_image"] = batch["aux_image"].to(device)[mask]
            original = base(inputs, inference=True)
            output = original if model is None else model(inputs, inference=True)
            scores = output["cls"].softmax(-1)[:, 1]
            if not all(torch.isfinite(value).all() for value in (scores, original["cls"], original["prob"])):
                raise ValueError("Nonfinite export scores")
            lengths = mask.sum(-1).cpu().tolist()
            paths = [path for paths in batch["paths"] for path in paths]
            videos = [video for video, count in zip(batch["video_id"], lengths) for _ in range(count)]
            labels = [int(label) for label, count in zip(batch["label"], lengths) for _ in range(count)]
            if len(paths) != len(scores):
                raise ValueError("Export frame identities do not align with valid scores")
            chunks["labels"].append(np.asarray(labels))
            chunks["path"].append(np.asarray(paths))
            chunks["video_id"].append(np.asarray(videos))
            chunks["cls_prob"].append(original["prob"].cpu().numpy())
            chunks["global_log_odds"].append((original["cls"][:, 1] - original["cls"][:, 0]).cpu().numpy())
            chunks["score"].append(scores.cpu().numpy())
            observed += len(scores)
            if index == 1 or index % args.log_every == 0 or index == len(loader):
                progress.update("baseline" if model is None else "export", gid, dataset,
                                index, len(loader), frames=observed)
    if not chunks["labels"]:
        raise ValueError("Empty evaluation manifest")
    values = {key: np.concatenate(parts) for key, parts in chunks.items()}
    if len(set(values["path"])) != len(values["path"]):
        raise ValueError("Duplicate exported frame identities")
    reporting.video_arrays(values)
    return values


def train_arm(spec, base, pristine, prepared, reader, config, args, options, source, signature,
              budget, folder, references, progress, device, asset_receipt=None):
    import torch
    model = core_module("models").build_model(base, spec, pristine, options).to(device)
    parameters = list(model.trainable_parameters())
    if not parameters:
        raise ValueError("Arm has no trainable parameters")
    cold = spec["requires_pristine"] and spec["family"] == "G33"
    lr = args.cold_lr if cold else args.lr
    optimizer = (torch.optim.Adam(parameters, lr=1e-5, weight_decay=0.) if spec["kind"] == "gend" else
                 torch.optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay))
    if spec["requires_pairs"]:
        import e1005_assets as assets
        episodes = data_api.build_episodes(prepared["pairs"], budget, args.seed)
        if spec["kind"] in assets.MASK_KINDS | assets.NATIVE_KINDS | {"pixel224"}:
            dataset = assets.asset_pair_dataset(episodes, reader, config["mean"], config["std"], spec["kind"],
                        asset_receipt, config["rgb_root_override"], config["resolution"], args.native_resolution,
                        independent_nuisance=spec.get("independent_nuisance", False))
        else:
            dataset = data_api.PairDataset(episodes, reader, config["mean"], config["std"],
                                          independent_nuisance=spec.get("independent_nuisance", False))
        batches = (dataset[index] for index in range(budget))
    else:
        batches = frame_batches(prepared["train"], reader, config, budget, args.seed)
    native = spec["requires_native"]
    scope = "native" if native else "standard"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    scheduled = set(evaluation_steps(budget, args.eval_every))
    history = []
    started = time.perf_counter()
    totals = {}

    def evaluate(step):
        source_values = export_scores(model, base, prepared["manifests"]["source_val"], reader, config,
                        args, device, progress, spec["id"], "FF++/val", native, asset_receipt)
        dev_values = export_scores(model, base, prepared["manifests"]["Celeb-DF-v2"], reader, config,
                        args, device, progress, spec["id"], "CDF/development", native, asset_receipt)
        legacy.verify_paired_base(references[scope]["source_val"], source_values)
        legacy.verify_paired_base(references[scope]["Celeb-DF-v2"], dev_values)
        artifact = snapshot_artifact(model, source["base_sha256"], signature, step)
        path = checkpoints / f"step{step:07}.pth"
        temporary = path.with_suffix(".pth.tmp")
        torch.save(artifact, temporary)
        temporary.replace(path)
        row = {"step": step, "source": reporting.metrics(source_values),
               "development": reporting.metrics(dev_values), "checkpoint": str(path),
               "checkpoint_sha256": legacy.file_sha256(path)}
        history.append(row)
        write_json(folder / "history.json", history)
        progress.update("checkpoint", spec["id"], "S3", step, budget,
                        video_auc=row["development"]["video_auc"], file=path.name)

    evaluate(0)
    for step, batch in enumerate(batches, 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if spec["kind"] == "gend":
            warm = min(575, budget)
            current_lr = (1e-5 + (3e-4 - 1e-5) * step / warm if step <= warm else
                          1e-5 + .5 * (3e-4 - 1e-5) * (1 + math.cos(math.pi * (step - warm) / max(1, budget - warm))))
            for group in optimizer.param_groups:
                group["lr"] = current_lr
        output = packed_forward(model, batch, device)
        teacher = None
        if spec["objective"]["keep"]:
            with torch.no_grad():
                teacher = packed_forward(base, batch, device)
        objective_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                           for key, value in batch.items() if key not in ("image", "aux_image")}
        losses = core_module("objectives").compute_losses(output, objective_batch, spec["objective"], teacher,
                        keep_weight=args.keep_weight, tau=args.risk_tau, rank_weight=args.rank_weight,
                        metric=spec["metric"], spatial_target=objective_batch.get("spatial_target"))
        if any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all() for value in losses.values()):
            raise ValueError("Nonfinite E1005 loss")
        losses["overall"].backward()
        if any(parameter.requires_grad or parameter.grad is not None for parameter in base.parameters()):
            raise RuntimeError("B0 isolation failed: base received gradients/became trainable")
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise ValueError("Nonfinite E1005 gradients")
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1 and key != "loss":
                totals[key] = totals.get(key, 0.) + float(value.detach())
        if step == 1 or step % args.log_every == 0 or step == budget:
            elapsed = time.perf_counter() - started
            progress.update("train", spec["id"], "FF++/train", step, budget,
                            loss=round(float(losses["overall"].detach()), 6),
                            eta_seconds=round(elapsed / step * (budget - step), 1))
        if step in scheduled:
            evaluate(step)
    selectors = reporting.select_steps(history)
    primary = next(row for row in history if row["step"] == selectors["S3"])
    restore_artifact(model, torch.load(primary["checkpoint"], map_location=device, weights_only=True),
                     source["base_sha256"], signature)
    result = {"id": spec["id"], "family": spec["family"], "design_id": spec["design_id"],
              "status": "OK", "spec": spec, "training_signature": signature,
              "trainable_parameters": sum(parameter.numel() for parameter in parameters),
              "optimizer_steps": budget, "selected_step": selectors["S3"], "selectors": selectors,
              "primary_auc": primary["development"]["video_auc"], "selection_dataset": "Celeb-DF-v2/development",
              "checkpoint": primary["checkpoint"], "checkpoint_sha256": primary["checkpoint_sha256"],
              "training_seconds": time.perf_counter() - started,
              "losses": {key: value / budget for key, value in totals.items()},
              "settings": model.settings}
    write_json(folder / "result.json", result)
    return result, model


def _source_budget(args, source):
    if args.base_selected_steps is not None:
        return args.base_selected_steps, {"kind": "explicit_audited_override", "completed_updates": args.base_selected_steps}
    import re
    candidates = sorted(Path(source["config_path"]).parent.glob("logs/**/training.log"))
    matched = []
    for path in candidates:
        current, selected = None, None
        for line in path.read_text(errors="replace").splitlines():
            value = re.search(r"dataset:\s*avg\s+step:\s*(\d+)", line)
            if value:
                current = int(value.group(1))
            if ("Checkpoint saved to" in line and str(source["checkpoint"]) in line and
                    current is not None):
                selected = current + 1  # Original train loop logs a zero-based optimizer iteration.
        if selected is not None:
            matched.append((selected, path))
    if len(matched) == 1:
        selected, path = matched[0]
        return selected, {"kind": "matching_checkpoint_log_zero_based_step_plus_one", "completed_updates": selected,
                          "path": str(path), "sha256": legacy.file_sha256(path)}
    return None, {"kind": "unknown", "note": "provide --base_selected_steps to match cold total supervision budget"}


def _run_identity(args, source, prepared, reader, options, budget_info, asset_receipt):
    files = [ROOT / "experiments" / name for name in ("run_e1005.py", "e1005_protocol.py", "e1005_data.py",
                  "e1005_assets.py", "e1005_reporting.py", "run_e1001.py", "run_e1002.py")]
    files += [ROOT / "training/detectors" / name for name in ("e1005_models.py", "e1005_objectives.py", "e1002_matrix.py")]
    files += [ROOT / "training/dataset" / name for name in ("abstract_dataset.py", "balance_batch_sampler.py")]
    files.append(ROOT / "training/detectors/effort_detector.py")
    return {"base_sha256": source["base_sha256"], "config_sha256": source["config_sha256"],
            "metadata_sha256": prepared["metadata_sha256"], "pair_audit_sha256": prepared["pair_audit_sha256"],
            "asset_audit_hash": json_hash(asset_receipt) if asset_receipt else None,
            "reader": reader.settings, "options": options, "seed": args.seed,
            "warm_steps": args.steps, "cold_steps": args.cold_steps, "budget_provenance": budget_info,
            "eval_frames": args.eval_frames, "train_frames": args.train_frames,
            "training": {key: getattr(args, key) for key in ("lr", "cold_lr", "weight_decay", "keep_weight",
                "rank_weight", "risk_tau", "eval_every", "allow_nondeterministic")},
            "source_sha256": {str(path.relative_to(ROOT)): legacy.file_sha256(path) for path in files}}


def _load_export(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def cached_baseline(path, compute, manifest, root, resume):
    import numpy as np
    path, root = Path(path), Path(root)
    key = str(path.relative_to(root))
    hashes = manifest.setdefault("baseline_sha256", {})
    if resume and path.is_file():
        if hashes.get(key) != legacy.file_sha256(path):
            raise ValueError(f"Resume baseline cache identity changed: {key}")
        values = _load_export(path)
    else:
        values = compute()
        if not np.array_equal(values["score"], values["cls_prob"]):
            raise ValueError("Original baseline score differs from B0 probability")
        path.parent.mkdir(parents=True, exist_ok=True)
        legacy.save_npz(path, values)
        hashes[key] = legacy.file_sha256(path)
        write_json(root / "manifest.json", manifest)
    if not np.array_equal(values["score"], values["cls_prob"]):
        raise ValueError("Original baseline score differs from cached B0 probability")
    return values


def load_or_initialize_results(root, resume):
    path = Path(root) / "all_results.json"
    results = (json.loads(path.read_text()) if resume and path.is_file() else
               {"experiment": "E1005", "families": protocol.FAMILIES, "arms": {}, "evaluation": {}})
    write_json(path, results)
    return results


def checked_asset_audit(root, asset_status, resume):
    path = Path(root) / "asset_audit_result.json"
    if resume and path.is_file() and json.loads(path.read_text()) != asset_status:
        raise ValueError("Resume asset content/audit identity changed")
    write_json(path, asset_status)


def _eligible(spec, prepared, asset_status, leader, inherited_steps):
    if spec["requires_pairs"] and not prepared["pairs"]:
        return "NOT_ELIGIBLE_PAIR_AUDIT", "verified time/target-face pair audit receipt missing"
    if spec["depends_on_leader"] and leader is None:
        return "NOT_ELIGIBLE_PREREQUISITE", "no successful G32 development leader"
    if spec["requires_mask"] and not asset_status["mask"]["eligible"]:
        return "NOT_ELIGIBLE_ASSET", asset_status["mask"]["reason"]
    if spec["requires_native"] and not asset_status["native"]["eligible"]:
        return "NOT_ELIGIBLE_ASSET", asset_status["native"]["reason"]
    return None, None


def run(args, source, config, prepared):
    import numpy as np
    import PIL
    import torch
    import transformers
    import experiment_utils as utilities
    import e1005_assets as assets
    device = utilities.DEVICE
    if not args.allow_nondeterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    legacy.seed_everything(args.seed)
    reader = data_api.make_reader(config["rgb_root_override"], config["resolution"], args.reader_backend)
    inherited_steps, budget_info = _source_budget(args, source)
    cold_budget = args.cold_steps if args.cold_steps is not None else args.steps + (inherited_steps or 0)
    asset_receipt = json.loads(args.asset_audit.read_text()) if args.asset_audit else None
    options = {"rank": args.rank, "last_layers": args.last_layers, "adapter_width": args.adapter_width,
               "local_layers": args.local_layers, "late_layers": args.late_layers, "pixel_layers": args.pixel_layers,
               "aux_image_size": args.native_resolution, "shuffle_seed": args.seed,
               "lora_initializer": "kaiming" if config.get("use_loralib", False) else "normal"}
    identity = _run_identity(args, source, prepared, reader, options, budget_info, asset_receipt)
    specs = protocol.resolve_arms(args.arms)
    print(f"E1005 model: loading immutable B0 {source['checkpoint']}", file=sys.stderr, flush=True)
    base = utilities.load_model(config, source["checkpoint"]).to(device)
    base.requires_grad_(False).eval()
    for parameter in base.parameters():
        parameter.grad = None
    identity["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__,
                           "numpy": np.__version__, "pillow": PIL.__version__, "device": str(device)}
    from importlib.metadata import PackageNotFoundError, version
    for package in ("opencv-python", "opencv-python-headless", "albumentations", "loralib"):
        try:
            identity["runtime"][package] = version(package)
        except PackageNotFoundError:
            identity["runtime"][package] = None
    identity["base_architecture"] = {"vision": base.backbone.config.to_dict(),
                      "attention": getattr(base.backbone.config, "_attn_implementation", None)}
    pretrained = None
    if any(spec["requires_pristine"] for spec in specs):
        print("E1005 model: loading pristine CLIP reference", file=sys.stderr, flush=True)
        legacy.seed_everything(args.seed)
        pretrained = utilities.DETECTOR["effort"](config).to(device).eval()
        pretrained.requires_grad_(False)
        identity["pristine"] = {"state_sha256": legacy.state_sha256(pretrained.backbone),
                       "vision": pretrained.backbone.config.to_dict(),
                       "attention": getattr(pretrained.backbone.config, "_attn_implementation", None)}
    else:
        identity["pristine"] = None
    # Manifest JSON round-trips integer mapping keys and tuples. Compare the
    # canonical persisted form, including HF configuration id2label mappings.
    identity = json.loads(json.dumps(identity, sort_keys=True))
    root = args.resume or args.output_dir / f"seed{args.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    old = None
    if args.resume:
        old = json.loads((root / "manifest.json").read_text())
        if old["identity"] != identity:
            raise ValueError("Resume source/config/data/code/training identity changed")
    else:
        root.mkdir(parents=True, exist_ok=False)
    progress = Progress(root)
    progress.update("initialize", "E1005", completed=0, total=50, output=str(root))
    manifest = {"experiment": "E1005", "identity": identity, "source": source, "protocol": protocol.plan(
                args.arms, args.seed, args.steps), "budget_provenance": budget_info,
                "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                "device": str(device), "config": config, "status": "RUNNING",
                "baseline_sha256": deepcopy(old.get("baseline_sha256", {})) if old else {}}
    write_json(root / "manifest.json", manifest)
    write_json(root / "runtime_config.json", config)
    write_json(root / "pair_candidates.json", prepared["candidates"])
    write_json(root / "verified_pairs.json", prepared["pairs"])
    write_json(root / "input_manifests.json", prepared["manifests"])
    write_json(root / "experiment_catalog.json", protocol.plan(seed=args.seed, steps=args.steps))
    initial_hash = legacy.state_sha256(base)
    manifest["base_state_sha256_before"] = initial_hash
    write_json(root / "manifest.json", manifest)
    results = load_or_initialize_results(root, args.resume)
    references = {"standard": {}, "native": {}}
    try:
        if asset_receipt and prepared["pairs"]:
            progress.update("asset_audit", "G36", completed=0, total=1)
            asset_status = assets.audit_assets(asset_receipt, prepared["pairs"], prepared["manifests"],
                            config["rgb_root_override"], config["resolution"], args.native_resolution)
        else:
            asset_status = {"mask": {"eligible": False, "reason": "registered training mask receipt unavailable"},
                            "native": {"eligible": False, "reason": "registered full-panel native crop receipt unavailable"}}
        checked_asset_audit(root, asset_status, args.resume)
        for scope in ["standard", *(["native"] if asset_status["native"]["eligible"] else [])]:
            folder = root / ("B0_uniform" if scope == "standard" else "B0_native")
            folder.mkdir(exist_ok=True)
            for name in ("source_val", "Celeb-DF-v2"):
                path = folder / f"{name}.npz"
                values = cached_baseline(path, lambda name=name, scope=scope:
                      export_scores(None, base, prepared["manifests"][name], reader, config, args, device,
                                    progress, "B0", name, scope == "native", asset_receipt),
                      manifest, root, args.resume)
                references[scope][name] = values
        prefix = root / "B0_legacy_prefix8"
        prefix.mkdir(exist_ok=True)
        cached_baseline(prefix / "source_val.npz", lambda:
                       export_scores(None, base, prepared["legacy_val"], reader, config, args, device,
                                     progress, "B0_legacy", "FF++/val"), manifest, root, args.resume)

        for position, template in enumerate(specs, 1):
            leader_id = reporting.choose_champion(results["arms"], {"G32"})
            leader = results["arms"][leader_id]["spec"] if leader_id else None
            gid = template["id"]
            status, reason = _eligible(template, prepared, asset_status, leader, inherited_steps)
            if status:
                result = {**template, "status": status, "reason": reason}
                results["arms"][gid] = result
                write_json(root / gid / "result.json", result)
                write_json(root / "all_results.json", results)
                progress.update("ineligible", gid, completed=position, total=len(specs), reason=reason)
                continue
            try:
                spec = protocol.materialize_arm(template, leader)
            except ValueError as exc:
                if not str(exc).startswith("NOT_APPLICABLE"):
                    raise
                result = {**template, "status": "NOT_APPLICABLE", "reason": str(exc)}
                results["arms"][gid] = result
                write_json(root / gid / "result.json", result)
                continue
            budget = cold_budget if spec["family"] == "G33" else args.steps
            settings = training_settings(args, config, options, budget)
            signature = protocol.training_signature(spec, settings, identity)
            prior = results["arms"].get(gid)
            if args.resume and prior and prior.get("status") in ("OK", "REUSED"):
                if prior.get("training_signature") != signature or legacy.file_sha256(prior["checkpoint"]) != prior["checkpoint_sha256"]:
                    raise ValueError(f"Resume completed-arm identity changed: {gid}")
                progress.update("reuse", gid, completed=position, total=len(specs))
                continue
            identical = next((row for row in results["arms"].values() if row.get("status") in ("OK", "REUSED")
                              and row.get("training_signature") == signature), None)
            if identical:
                result = {**deepcopy(identical), "id": gid, "family": spec["family"], "design_id": spec["design_id"],
                          "spec": spec, "status": "REUSED", "same_training_as": identical["id"]}
                results["arms"][gid] = result
                write_json(root / gid / "result.json", result)
                continue
            progress.update("build_model", gid, completed=position, total=len(specs), budget=budget)
            model = None
            try:
                legacy.seed_everything(args.seed)
                result, model = train_arm(spec, base, pretrained, prepared, reader, config, args, options,
                                source, signature, budget, root / gid, references, progress, device, asset_receipt)
                result["total_budget_matched"] = (spec["family"] != "G33" or
                                    (inherited_steps is not None and budget >= args.steps + inherited_steps))
            except Exception as exc:
                result = {"id": gid, "family": template["family"], "design_id": template["design_id"],
                          "spec": spec, "status": "FAILED", "training_signature": signature,
                          "error": f"{type(exc).__name__}: {exc}"}
                progress.update("failed", gid, completed=position, total=len(specs), error=result["error"])
                if "B0 isolation" in str(exc):
                    raise
            finally:
                results["arms"][gid] = result
                write_json(root / gid / "result.json", result)
                write_json(root / "all_results.json", results)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if legacy.state_sha256(base) != initial_hash:
                    raise RuntimeError("B0 isolation failed: original model state changed")
                if legacy.file_sha256(source["checkpoint"]) != source["base_sha256"]:
                    raise RuntimeError("B0 isolation failed: original checkpoint file changed")

        champion = reporting.choose_champion(results["arms"], {"G32", "G33", "G35", "G36"})
        lock = {"experiment": "E1005", "champion": champion, "identity_hash": json_hash(identity),
                "selection": "S3 development video_auc", "source_ids": identity,
                "config": results["arms"][champion]["spec"] if champion else None}
        old_lock = root / "champion_lock.json"
        if args.resume and old_lock.is_file() and json.loads(old_lock.read_text())["champion"] != champion:
            raise ValueError("Previously locked champion cannot be changed using later results")
        if args.no_evaluation:
            write_json(root / "development_leader.json", lock)
        else:
            write_json(old_lock, lock)
        if champion and not args.no_evaluation:
            evaluate_final(args, root, results, lock, base, pretrained, prepared, reader, config, options,
                           source, references, progress, device, asset_receipt, manifest)
        reporting.write_analysis_csv(root / "analysis.csv", results["arms"], results.get("evaluation", {}))
        failures = [gid for gid, row in results["arms"].items() if row["status"] == "FAILED"]
        blocked = [gid for gid, row in results["arms"].items() if row["status"].startswith("NOT_ELIGIBLE")]
        manifest["status"] = ("FAILED_ARMS" if failures else
                              ("PARTIAL_PREREQUISITES" if blocked else "COMPLETE_ELIGIBLE_ARMS"))
        results["coverage"] = {"requested": len(specs), "failed": len(failures), "blocked": len(blocked),
                      "trained": sum(row["status"] == "OK" for row in results["arms"].values()),
                      "reused": sum(row["status"] == "REUSED" for row in results["arms"].values()),
                      "not_applicable": sum(row["status"] == "NOT_APPLICABLE" for row in results["arms"].values())}
        results["status"] = manifest["status"]
        write_json(root / "all_results.json", results)
        progress.update("complete", "E1005", completed=len(specs), total=len(specs),
                        failed=len(failures), blocked=len(blocked), champion=champion or "none", results=str(root / "all_results.json"))
        return 1 if failures else 0
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        results["run_error"] = manifest["error"]
        write_json(root / "all_results.json", results)
        progress.update("fatal", "E1005", error=manifest["error"])
        raise
    finally:
        manifest["base_state_sha256_after"] = legacy.state_sha256(base)
        manifest["base_state_unchanged"] = manifest["base_state_sha256_after"] == initial_hash
        manifest["base_file_unchanged"] = legacy.file_sha256(source["checkpoint"]) == source["base_sha256"]
        write_json(root / "manifest.json", manifest)


ORDINARY_CONTROLS = frozenset({"G32_H_V", "G32_L_V", "G32_J_V", "G33_MATCHED_B0", "G33_VIDEO_B0", "G33_GEND"})


def ordinary_comparison(exports, champion, ordinary_ids, repeats, seed):
    """Report the strongest fixed reference; never use this to select a champion."""
    available = sorted((set(ordinary_ids) & exports.keys()) - {champion})
    if not available:
        return None
    for gid in available:
        for dataset in protocol.REGRESSION_DATASETS:
            legacy.verify_paired_base(exports[champion][dataset], exports[gid][dataset])
    reference = min(available, key=lambda gid: (
        -sum(reporting.metrics(exports[gid][dataset])["video_auc"]
             for dataset in protocol.REGRESSION_DATASETS), gid))
    comparison = reporting.paired_macro_bootstrap(
        {dataset: (exports[reference][dataset], exports[champion][dataset])
         for dataset in protocol.REGRESSION_DATASETS}, repeats, seed)
    return {**comparison, "ordinary_baseline_id": reference,
            "champion_is_reference_recipe": champion in ordinary_ids,
            "selection": "strongest predeclared ordinary reference on fixed historical panel; champion unchanged"}


def final_selections(chosen, successful, champion, base, pristine, prepared, reader, config,
                     args, options, source, references, progress, device, asset_receipt):
    """Lock same-input ordinary checkpoints on development before regression export."""
    import torch
    champion_native = successful[champion]["spec"]["requires_native"]
    selections = {}
    for gid in sorted(chosen):
        row = successful[gid]
        native = row["spec"]["requires_native"] or (champion_native and gid in ORDINARY_CONTROLS)
        selection = {key: row[key] for key in ("selected_step", "checkpoint", "checkpoint_sha256")}
        selection["input_scope"] = "native" if native else "standard"
        if champion_native and gid in ORDINARY_CONTROLS:
            # Retarget the fixed ordinary trajectory to the identical global224
            # pixels derived from audited native crops. Only CDF chooses S3.
            model = core_module("models").build_model(base, row["spec"], pristine, options).to(device)
            history = json.loads((Path(row["checkpoint"]).parent.parent / "history.json").read_text())
            matched_history = []
            for point in history:
                if legacy.file_sha256(point["checkpoint"]) != point["checkpoint_sha256"]:
                    raise ValueError("Ordinary trajectory checkpoint identity changed")
                restore_artifact(model, torch.load(point["checkpoint"], map_location=device, weights_only=True),
                                 source["base_sha256"], row["training_signature"])
                values = export_scores(model, base, prepared["manifests"]["Celeb-DF-v2"], reader, config,
                                       args, device, progress, gid, "CDF/same-input-control", True, asset_receipt)
                legacy.verify_paired_base(references["native"]["Celeb-DF-v2"], values)
                matched_history.append({"step": point["step"], "video_auc": reporting.metrics(values)["video_auc"],
                                        "checkpoint": point["checkpoint"], "checkpoint_sha256": point["checkpoint_sha256"]})
            best = min(matched_history, key=lambda point: (-point["video_auc"], point["step"]))
            selection.update(selected_step=best["step"], checkpoint=best["checkpoint"],
                             checkpoint_sha256=best["checkpoint_sha256"], development_video_auc=best["video_auc"],
                             same_input_development_history=matched_history,
                             selection="S3 on champion's native-derived global input; fixed ordinary trajectory")
            del model
        selections[gid] = selection
    return selections


def evaluate_final(args, root, results, lock, base, pristine, prepared, reader, config, options,
                   source, references, progress, device, asset_receipt, manifest):
    import torch
    champion = lock["champion"]
    successful = {gid: row for gid, row in results["arms"].items() if row["status"] in ("OK", "REUSED")}
    chosen = {champion, *[gid for gid in ("G32_H_V", "G32_L_V", "G32_J_V", "G33_MATCHED_B0",
                                "G33_VIDEO_B0", "G33_GEND") if gid in successful]}
    family = successful[champion]["family"]
    if family in ("G35", "G36"):
        chosen.update(gid for gid, row in successful.items() if row["family"] == family)
    chosen.update(gid for gid, row in successful.items() if row["family"] == "G34")
    lock["final_export_ids"] = sorted(chosen)
    lock["final_selection"] = final_selections(chosen, successful, champion, base, pristine, prepared,
                          reader, config, args, options, source, references, progress, device, asset_receipt)
    write_json(root / "champion_lock.json", lock)
    final_cache = {"standard": {}, "native": {}}
    scopes = {lock["final_selection"][gid]["input_scope"] for gid in chosen}
    for scope in scopes:
        folder = root / ("B0_uniform" if scope == "standard" else "B0_native")
        for dataset in protocol.REGRESSION_DATASETS:
            path = folder / f"{dataset}.npz"
            values = cached_baseline(path, lambda dataset=dataset, scope=scope:
                       export_scores(None, base, prepared["manifests"][dataset], reader, config, args, device,
                                     progress, "B0", dataset, scope == "native", asset_receipt),
                       manifest, root, args.resume)
            final_cache[scope][dataset] = values
    evaluation = {"champion": champion, "reports": {}, "export_ids": sorted(chosen),
                  "panel": "historically examined six-domain regression, not fresh confirmation"}
    exports = {}
    for position, gid in enumerate(sorted(chosen), 1):
        row, spec = successful[gid], successful[gid]["spec"]
        selection = lock["final_selection"][gid]
        scope = selection["input_scope"]
        model = core_module("models").build_model(base, spec, pristine, options).to(device)
        if legacy.file_sha256(selection["checkpoint"]) != selection["checkpoint_sha256"]:
            raise ValueError("Final checkpoint identity changed")
        restore_artifact(model, torch.load(selection["checkpoint"], map_location=device, weights_only=True),
                         source["base_sha256"], row["training_signature"])
        folder = root / gid / "exports"
        folder.mkdir(parents=True, exist_ok=True)
        source_values = export_scores(model, base, prepared["manifests"]["source_val"], reader, config, args,
                    device, progress, gid, "FF++/threshold", scope == "native", asset_receipt)
        legacy.verify_paired_base(references[scope]["source_val"], source_values)
        threshold = reporting.calibration_threshold(source_values, args.val_fpr)
        baseline_threshold = reporting.calibration_threshold(references[scope]["source_val"], args.val_fpr)
        legacy.save_npz(folder / "source_val.npz", source_values)
        evaluation["reports"][gid], exports[gid] = {}, {}
        for dataset in protocol.REGRESSION_DATASETS:
            values = export_scores(model, base, prepared["manifests"][dataset], reader, config, args, device,
                                   progress, gid, dataset, scope == "native", asset_receipt)
            before = final_cache[scope][dataset]
            report = reporting.compare_exports(before, values, threshold, baseline_threshold)
            legacy.save_npz(folder / f"{dataset}.npz", values)
            evaluation["reports"][gid][dataset] = report
            exports[gid][dataset] = values
        row["final_metrics"] = evaluation["reports"][gid]
        row["final_selection"] = selection
        row["export_sha256"] = {path.name: legacy.file_sha256(path) for path in folder.glob("*.npz")}
        write_json(root / gid / "result.json", row)
        results["evaluation"] = evaluation
        write_json(root / "all_results.json", results)
        progress.update("final_evaluation", gid, completed=position, total=len(chosen))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    champion_spec = successful[champion]["spec"]
    scope = "native" if champion_spec["requires_native"] else "standard"
    comparisons = {dataset: (final_cache[scope][dataset], exports[champion][dataset])
                   for dataset in protocol.REGRESSION_DATASETS}
    evaluation["bootstrap"] = reporting.paired_macro_bootstrap(comparisons, args.bootstrap_repeats, args.seed)
    evaluation["reproduction"] = {"passed": False, "reason": "same-config repeat not performed"}
    repeat_reports = []
    for index in range(args.repeat_runs):
        progress.update("reproduce", champion, completed=index, total=args.repeat_runs)
        legacy.seed_everything(args.seed)
        budget = successful[champion]["optimizer_steps"]
        repeat_folder = root / "reproductions" / f"repeat{index + 1}" / champion
        repeated, model = train_arm(champion_spec, base, pristine, prepared, reader, config, args, options,
                        source, successful[champion]["training_signature"], budget, repeat_folder,
                        references, progress, device, asset_receipt)
        repeated_exports = {}
        for dataset in protocol.REGRESSION_DATASETS:
            values = export_scores(model, base, prepared["manifests"][dataset], reader, config, args, device,
                                   progress, champion, f"repeat/{dataset}", scope == "native", asset_receipt)
            legacy.verify_paired_base(final_cache[scope][dataset], values)
            repeated_exports[dataset] = values
            repeat_export_dir = repeat_folder / "exports"
            repeat_export_dir.mkdir(exist_ok=True)
            legacy.save_npz(repeat_export_dir / f"{dataset}.npz", values)
        comparison = reporting.compare_reproduction(exports[champion], repeated_exports,
                        successful[champion]["selected_step"], repeated["selected_step"])
        comparison["identity_hash"] = lock["identity_hash"]
        comparison["repeat_index"] = index + 1
        write_json(repeat_folder / "comparison.json", comparison)
        repeat_reports.append(comparison)
        del model
    if repeat_reports:
        evaluation["reproduction"] = {"passed": all(row["passed"] for row in repeat_reports),
                                       "repeats": repeat_reports, "same_seed": args.seed}
    required_controls = ORDINARY_CONTROLS
    formal = (args.steps >= 5750 and required_controls.issubset(chosen) and
              all(successful[gid].get("total_budget_matched", False) for gid in required_controls))
    evaluation["ordinary_comparison"] = ordinary_comparison(exports, champion, required_controls,
                                                            args.bootstrap_repeats, args.seed)
    evaluation["assessment"] = reporting.assess_breakthrough(evaluation["reports"][champion], evaluation["bootstrap"],
                           evaluation["reproduction"], formal_budget=formal,
                           ordinary_comparison=evaluation["ordinary_comparison"])
    evaluation.update(evaluation["assessment"])
    evaluation["formal_budget"] = formal
    evaluation["notes"] = ["Only development scores choose the champion; regression controls cannot change it.",
                            "Repeat-run reproducibility does not estimate cross-seed stability."]
    results["evaluation"] = evaluation
    write_json(root / "evaluation.json", evaluation)
    write_json(root / "all_results.json", results)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    values = (args.steps, args.train_frames, args.eval_frames, args.eval_videos, args.eval_every, args.log_every,
              args.rank, args.last_layers, args.adapter_width, args.native_resolution, args.bootstrap_repeats)
    if min(values) < 1 or args.repeat_runs < 0 or (args.cold_steps is not None and args.cold_steps < 1):
        parser.error("Positive budgets/dimensions and nonnegative repeat count required")
    if args.base_selected_steps is not None and args.base_selected_steps < 0:
        parser.error("base_selected_steps must be nonnegative")
    if not 0 <= args.seed < 2 ** 32 or any(min(group) < 0 or len(set(group)) != len(group)
            for group in (args.local_layers, args.late_layers, args.pixel_layers)):
        parser.error("Valid seed and unique nonnegative layer indices required")
    if any(not math.isfinite(value) or value <= 0 for value in (args.lr, args.cold_lr, args.risk_tau)):
        parser.error("Finite positive learning rates/risk_tau required")
    if any(not math.isfinite(value) or value < 0 for value in (args.weight_decay, args.keep_weight, args.rank_weight)):
        parser.error("Finite nonnegative loss/decay weights required")
    if not 0 < args.val_fpr < 1:
        parser.error("val_fpr must be between zero and one")
    try:
        protocol.resolve_arms(args.arms)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(protocol.plan(args.arms, args.seed, args.steps), indent=2, ensure_ascii=False))
        return 0
    print("E1005 preflight: resolving source and full metadata manifests", file=sys.stderr, flush=True)
    if args.resume and not args.base_run and not args.base_checkpoint:
        saved = json.loads((args.resume / "manifest.json").read_text())
        args.base_checkpoint = Path(saved["source"]["checkpoint"])
        args.base_config = Path(saved["source"]["config_path"])
    try:
        source = None if args.audit_metadata else legacy.resolve_source(args)
        config = _config(args, source)
        prepared = prepare_data(args, config)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.audit_metadata or args.preflight:
        folder = args.output_dir / "audit"
        write_json(folder / "input_manifests.json", prepared["manifests"])
        write_json(folder / "pair_candidates.json", prepared["candidates"])
        template = {"schema_version": 1, "receipt_id": "REQUIRES_TIME_AND_TARGET_FACE_VERIFICATION", "pairs": [
            {"pair_id": row["pair_id"], "real_video_id": row["real"]["video_id"],
             "fake_video_id": row["fake"]["video_id"], "target_id": row["fake"]["target_id"],
             "source_id": row["fake"]["source_id"], "common_indices": row["common_indices"],
             "time_verified": False, "target_face_verified": False, "evidence": ""}
            for row in prepared["candidates"]["pairs"]]}
        write_json(folder / "pair_audit_template.json", template)
        print(json.dumps({"experiment": "E1005", "status": "PAIR_CANDIDATES_ONLY" if prepared["unavailable_metadata"] else "METADATA_OK", "audit_dir": str(folder),
             "verified_pairs": len(prepared["pairs"]), "candidate_coverage": prepared["candidates"]["coverage"],
             "unavailable_external_metadata": prepared["unavailable_metadata"],
             "base_sha256": source["base_sha256"] if source else None,
             "note": "Candidates are not verified alignment; model/images/CUDA not validated here."}, indent=2))
        return 0
    try:
        return run(args, source, config, prepared)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"E1005 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

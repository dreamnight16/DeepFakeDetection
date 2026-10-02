"""E1002: four structural routes against one immutable, previously fitted B0."""

import argparse
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import run_e1001 as legacy

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "CLS_HEAD": ("readout", "cls"), "PATCH_MEAN": ("readout", "mean"),
    "COV_GLOBAL": ("readout", "cov"), "COV_REGIONAL": ("readout", "regional"),
    "MATRIX_LORA": ("matrix", "lora"), "MATRIX_SVD_TAIL": ("matrix", "svd_tail"),
    "MATRIX_SVFT": ("matrix", "svft"), "ENV_CONTROL": ("environment", "control"),
    "ENV_ORTH": ("environment", "orth"), "TEMP_MEAN": ("temporal", "mean"),
    "TEMP_DIFF": ("temporal", "diff"), "TEMP_TCN": ("temporal", "tcn"),
    "TEMP_SSM": ("temporal", "ssm"),
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_run", type=Path)
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--base_config", type=Path)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1002")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--seeds", nargs="+", type=int, help="Run one independent batch per seed")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--matrix_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=.01)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--matrix_layers", type=int, default=4)
    parser.add_argument("--preserve_top", type=int, default=32)
    parser.add_argument("--orth_weight", type=float, default=.1)
    parser.add_argument("--clip_frames", type=int, default=8)
    parser.add_argument("--clip_batch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--sampler_real_ratio", type=float, default=.3)
    parser.add_argument("--val_fpr", type=float, default=.05)
    parser.add_argument("--max_train_batches", type=int, help="Explicit smoke-test cap per epoch")
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root")
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser


def plan(args):
    return {"experiment": "E1002", "arms": {name: {"route": ARMS[name][0], "kind": ARMS[name][1]}
            for name in args.arms}, "base_training": False,
            "selection": "FaceForensics++ official val video_auc",
            "base_selection_caveat": "Reused B0 was previously selected using CDF-v2",
            "matrix": "warm-start independent B0 copies; new zero-initialized constrained updates",
            "environment": "supervised photometric intervention proxy; not measured person identity",
            "temporal": "ordered available sampled frames; torch-native selective SSM, not official Mamba",
            "input": "inherited single RGB normalization, no random augmentation; explicit environment views",
            "budget": {"epochs": args.n_epochs, "seeds": args.seeds or [args.seed],
                       "max_train_batches": args.max_train_batches,
                       "hidden_dim": args.hidden_dim, "rank": args.rank,
                       "matrix_layers": args.matrix_layers, "preserve_top": args.preserve_top},
            "threshold": {"fixed": .5, "val_real_fpr_target": args.val_fpr},
            "isolation": "B0 state and file hashes, paired score/identity checks; independent artifacts",
            "capacity_matching": "counts and measured budgets reported; parameter matching not assumed"}


def core_module(name):
    key = f"e1002_{name}_runtime"
    if key not in sys.modules:
        path = ROOT / "training/detectors" / f"e1002_{name}.py"
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def checkpoint_artifact(model, arm, settings, base_sha256):
    legacy.core_module().FrozenEvidenceSidecar._validate_hash(base_sha256)
    return {"version": 1, "experiment": "E1002", "arm": arm, "settings": settings,
            "base_sha256": base_sha256, "state_dict": {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}


def restore_artifact(model, artifact, arm, settings, base_sha256):
    import torch
    if (artifact.get("base_sha256") != base_sha256 or artifact.get("version") != 1 or
            artifact.get("experiment") != "E1002" or artifact.get("arm") != arm or
            artifact.get("settings") != settings):
        raise ValueError("Checkpoint identity/settings do not match E1002 arm and B0")
    incoming, expected = artifact.get("state_dict", {}), model.state_dict()
    if incoming.keys() != expected.keys() or any(not isinstance(incoming[key], torch.Tensor) or
            incoming[key].shape != expected[key].shape or incoming[key].dtype != expected[key].dtype
            for key in expected):
        raise ValueError("Checkpoint parameter keys/shapes/dtypes do not match")
    if any(not torch.isfinite(value).all() for value in incoming.values()):
        raise ValueError("Checkpoint parameters must be finite")
    model.load_state_dict(incoming, strict=True)


def video_rows(data, scores):
    import numpy as np
    labels, scores = np.asarray(data["labels"]), np.asarray(scores)
    if (labels.ndim != 1 or len(labels) == 0 or scores.shape != labels.shape or
            len(data["video_id"]) != len(labels) or not np.isin(labels, (0, 1)).all()):
        raise ValueError("Misaligned binary scores/video identities")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities")
    groups = {}
    for video, label, score in zip(data["video_id"], labels, scores):
        groups.setdefault(str(video), []).append((int(label), float(score)))
    if any(len({y for y, _ in rows}) != 1 for rows in groups.values()):
        raise ValueError("Conflicting labels within video")
    y = np.array([rows[0][0] for rows in groups.values()])
    p = np.array([np.mean([p for _, p in rows]) for rows in groups.values()])
    if set(y) != {0, 1}:
        raise ValueError("Metrics require both real and fake videos")
    return y, p


def validation_threshold(data, scores, target_fpr):
    import numpy as np
    if not math.isfinite(target_fpr) or not 0 < target_fpr < 1:
        raise ValueError("Invalid validation FPR target")
    labels, probs = video_rows(data, scores)
    return float(np.quantile(probs[labels == 0], 1 - target_fpr, method="higher"))


def metrics(data, scores, threshold):
    import numpy as np
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid frozen validation threshold")
    y, p = video_rows(data, scores)
    result = legacy.readout_metrics(data, scores)
    # Integrate precision at complete score-tie groups, never arbitrary tie order.
    order = np.argsort(-p, kind="stable")
    sorted_y, sorted_p = y[order], p[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_p)), len(p) - 1]
    positives = sorted_y.cumsum()[endpoints]
    precision = positives / (endpoints + 1)
    recall_increments = np.diff(np.r_[0, positives]) / y.sum()
    result.update(video_ap=float((precision * recall_increments).sum()), val_threshold=threshold,
                  frame_fpr=float((np.asarray(scores)[data["labels"] == 0] > .5).mean()),
                  frame_fnr=float((np.asarray(scores)[data["labels"] == 1] <= .5).mean()),
                  video_fpr=float((p[y == 0] > .5).mean()),
                  video_fnr=float((p[y == 1] <= .5).mean()),
                  video_fpr_at_val_threshold=float((p[y == 0] > threshold).mean()),
                  video_tpr_at_val_threshold=float((p[y == 1] > threshold).mean()))
    return result


def make_model(base, feature_dim, arm, args, device):
    route, kind = ARMS[arm]
    if route == "matrix":
        model = core_module("matrix").MatrixAdaptation(base, kind, rank=args.rank,
                    last_layers=args.matrix_layers, preserve_top=args.preserve_top)
    elif route == "temporal":
        model = core_module("temporal").TemporalReadout(kind, feature_dim, args.hidden_dim)
    elif route == "environment":
        model = core_module("heads").EnvironmentReadout(feature_dim, args.hidden_dim)
    else:
        model = core_module("heads").FeatureReadout(kind, feature_dim, args.hidden_dim)
    return model.to(device)


def interventions(images, config):
    import torch
    # Original, mild brightness +/- and mild contrast are known nuisance targets.
    targets = torch.randint(4, (len(images),), device=images.device)
    std = images.new_tensor(config["std"])[None, :, None, None]
    altered = images.clone()
    for target, delta in ((1, .02), (2, -.02)):
        selected = targets == target
        altered[selected] = images[selected] + delta / std
    selected = targets == 3
    altered[selected] = .9 * images[selected] + .1 * images[selected].mean((-2, -1), keepdim=True)
    return altered, targets


def arm_logits(model, extractor, data, route, device):
    from e1002_data import extract_clip_features
    if route == "temporal":
        original, features, mask = extract_clip_features(extractor, data, device)
        return original, model(features, mask), mask
    images = data["image"].to(device)
    original, cls, patches = extractor(images)
    if route == "matrix":
        logits = model({"image": images}, inference=True)["cls"]
    elif route == "environment":
        logits = model(cls)["logits"]
    else:
        logits = model(cls, patches)
    return original, logits, None


def export_scores(model, extractor, loader, route, device):
    import numpy as np
    import torch
    if model is not None:
        model.eval()
    chunks = {key: [] for key in ("labels", "path", "video_id", "cls_prob", "global_log_odds", "score")}
    with torch.no_grad():
        for data in loader:
            if model is None:
                if route == "temporal":
                    from e1002_data import extract_clip_features
                    original, _, mask = extract_clip_features(extractor, data, device)
                else:
                    original, _, _ = extractor(data["image"].to(device))
                    mask = None
                score = original["prob"]
            else:
                original, logits, mask = arm_logits(model, extractor, data, route, device)
                score = logits.softmax(-1)[:, 1]
            if mask is not None:
                lengths = mask.sum(-1).tolist()
                labels = [int(label) for label, n in zip(data["label"], lengths) for _ in range(n)]
                paths = [path for group in data["paths"] for path in group]
                videos = [video for video, n in zip(data["video_id"], lengths) for _ in range(n)]
                if model is not None:
                    score = score.repeat_interleave(mask.sum(-1))
            else:
                labels, paths, videos = data["label"].numpy(), data["path"], data["video_id"]
            chunks["labels"].append(np.asarray(labels))
            chunks["path"].append(np.asarray(paths))
            chunks["video_id"].append(np.asarray(videos))
            chunks["cls_prob"].append(original["prob"].cpu().numpy())
            logits = original["cls"]
            chunks["global_log_odds"].append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            chunks["score"].append(score.cpu().numpy())
    if not chunks["labels"]:
        raise ValueError("Empty score export")
    result = {key: np.concatenate(parts) for key, parts in chunks.items()}
    video_rows(result, result["score"])
    if len(set(result["path"])) != len(result["path"]):
        raise ValueError("Duplicate exported frame identities")
    return result


def synchronize(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_arm(model, extractor, training, validation, reference, arm, args, config, folder, source, device):
    import torch
    from torch.nn import functional as F
    from e1002_data import extract_clip_features
    route = ARMS[arm][0]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.matrix_lr if route == "matrix" else args.lr,
                                 weight_decay=args.weight_decay)
    best, history, total_steps = -1., [], 0
    settings = {"route": route, "kind": ARMS[arm][1], "feature_dim": model_feature_dim(model, extractor),
                "hidden_dim": args.hidden_dim, "rank": args.rank, "matrix_layers": args.matrix_layers,
                "preserve_top": args.preserve_top, "clip_frames": args.clip_frames,
                "orth_weight": args.orth_weight if arm == "ENV_ORTH" else 0.}
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(args.n_epochs):
        model.train()
        steps, totals = 0, {}
        for data in training:
            optimizer.zero_grad(set_to_none=True)
            labels = data["label"].to(device)
            if route == "matrix":
                output = model({"image": data["image"].to(device)})
                losses = {"overall": F.cross_entropy(output["cls"], labels)}
            elif route == "temporal":
                _, features, mask = extract_clip_features(extractor, data, device)
                losses = {"overall": F.cross_entropy(model(features, mask), labels)}
            else:
                images = data["image"].to(device)
                _, cls, patches = extractor(images)
                if route == "environment":
                    altered, targets = interventions(images, config)
                    _, second_cls, _ = extractor(altered)
                    output = model(torch.cat((cls, second_cls)))
                    losses = core_module("heads").environment_loss(output, labels.repeat(2),
                                torch.cat((torch.zeros_like(targets), targets)),
                                args.orth_weight if arm == "ENV_ORTH" else 0.)
                else:
                    losses = {"overall": F.cross_entropy(model(cls, patches), labels)}
            if not all(torch.isfinite(loss).all() for loss in losses.values()):
                raise ValueError("Nonfinite training loss")
            losses["overall"].backward()
            if any(p.requires_grad or p.grad is not None for p in extractor.base.parameters()):
                raise RuntimeError("B0 isolation failed: base became trainable or received gradients")
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("Nonfinite model gradients")
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            steps += 1
            total_steps += 1
            for key, loss in losses.items():
                totals[key] = totals.get(key, 0.) + float(loss.detach())
            if steps % 100 == 0:
                print(f"  {arm} epoch={epoch + 1} step={steps} loss={float(losses['overall'].detach()):.6f}", flush=True)
            if args.max_train_batches is not None and steps >= args.max_train_batches:
                break
        if not steps:
            raise ValueError("Empty training loader")
        values = export_scores(model, extractor, validation, route, device)
        legacy.verify_paired_base(reference, values)
        report = metrics(values, values["score"], .5)
        row = {"epoch": epoch + 1, "optimizer_steps": steps,
               "losses": {key: total / steps for key, total in totals.items()}, "validation": report}
        history.append(row)
        if report["video_auc"] > best:
            best = report["video_auc"]
            temporary = folder / "best.pth.tmp"
            import copy
            torch.save(checkpoint_artifact(model, arm, copy.deepcopy(settings), source["base_sha256"]), temporary)
            temporary.replace(folder / "best.pth")
        legacy.write_json(folder / "history.json", history)
        print(f"  {arm} epoch={epoch + 1} val_video_auc={report['video_auc']:.6f}", flush=True)
    restore_artifact(model, torch.load(folder / "best.pth", map_location=device, weights_only=True),
                     arm, settings, source["base_sha256"])
    model.eval()
    synchronize(device)
    return {"settings": settings, "selected_epoch": max(history, key=lambda row: row["validation"]["video_auc"])["epoch"],
            "trainable_parameters": sum(p.numel() for p in parameters), "optimizer_steps": total_steps,
            "train_and_validation_seconds": time.perf_counter() - start,
            "selection_dataset": "FaceForensics++/val", "checkpoint": str(folder / "best.pth")}


def model_feature_dim(model, extractor):
    return int(extractor.base.backbone.config.hidden_size)


def run_seed(args, source, config, partition, metadata_hashes):
    import numpy as np
    import torch
    import transformers
    import experiment_utils as utilities
    from e0924_export import strict_loader
    from e1002_data import FrozenFeatures, clip_loader
    from run_g25 import TEST_DS
    from run_e0924 import source_hashes

    legacy.seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    root = args.output_dir / f"seed{args.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    hashes.update({str(path.relative_to(ROOT)): legacy.file_sha256(path)
                   for path in (Path(__file__), ROOT / "experiments/e1002_data.py",
                                ROOT / "experiments/run_e1001.py")})
    manifest = {"protocol": plan(args), "source": source, "dataset_sha256": metadata_hashes,
                "source_sha256": hashes, "arguments": {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}, "python": sys.version, "torch": torch.__version__,
                "transformers": transformers.__version__, "device": str(utilities.DEVICE)}
    legacy.write_json(root / "manifest.json", manifest)
    legacy.write_json(root / "runtime_config.json", config)
    legacy.write_json(root / "calibration_partition.json", partition)
    base = utilities.load_model(config, source["checkpoint"])
    extractor = FrozenFeatures(base).to(utilities.DEVICE)
    baseline_hash = legacy.state_sha256(base)
    manifest["base_state_sha256_before"] = baseline_hash
    results = {"experiment": "E1002", "B0": {}, "arms": {}}
    records = [*partition["records"]["fit"], *partition["records"]["holdout"]]
    # Prefix sorting by path stabilizes validation batches independent of partition RNG.
    records.sort(key=lambda row: tuple(row["frames"]))
    references = {"frame": {}, "temporal": {}}
    print(f"E1002 B0: {source['checkpoint']}\nResults: {root}", flush=True)

    def loader(dataset, temporal=False, validation=False):
        data_source = utilities.get_data_loader(config, dataset).dataset
        selected = records if validation else None
        if temporal:
            return clip_loader(data_source, args.clip_batch_size, args.clip_frames, selected)
        frames = strict_loader(data_source, config["test_batchSize"], selected)
        frames.source_dataset = data_source
        return frames

    scopes = ["frame"]
    if any(ARMS[name][0] == "temporal" for name in args.arms):
        scopes.append("temporal")
    try:
        for scope in scopes:
            folder = root / f"B0_{scope}"
            folder.mkdir()
            results["B0"][scope] = {}
            for dataset in ["val", *TEST_DS]:
                legacy.seed_everything(args.seed)
                frames = loader("FaceForensics++" if dataset == "val" else dataset,
                                temporal=scope == "temporal", validation=dataset == "val")
                try:
                    values = export_scores(None, extractor, frames, scope, utilities.DEVICE)
                finally:
                    legacy.close_loader(frames)
                references[scope][dataset] = values
                legacy.save_npz(folder / f"{dataset}.npz", values)
            val = references[scope]["val"]
            threshold = validation_threshold(val, val["cls_prob"], args.val_fpr)
            for dataset, values in references[scope].items():
                results["B0"][scope][dataset] = metrics(values, values["cls_prob"], threshold)
        legacy.write_json(root / "all_results.json", results)
        for arm in args.arms:
            legacy.seed_everything(args.seed)
            route, kind = ARMS[arm]
            scope = "temporal" if route == "temporal" else "frame"
            folder = root / arm
            folder.mkdir()
            result = {"status": "FAILED", "route": route, "kind": kind, "base_sha256": source["base_sha256"]}
            model, training, validation = None, None, None
            arm_started = time.perf_counter()
            try:
                synchronize(utilities.DEVICE)
                build_started = time.perf_counter()
                model = make_model(base, base.backbone.config.hidden_size, arm, args, utilities.DEVICE)
                synchronize(utilities.DEVICE)
                result["model_build_seconds"] = time.perf_counter() - build_started
                training = legacy.make_train_loader(config, args.sampler_real_ratio)
                source_dataset = training.source_dataset
                if route == "temporal":
                    training = clip_loader(source_dataset, args.clip_batch_size, args.clip_frames,
                                           train_ratio=args.sampler_real_ratio)
                    validation = clip_loader(source_dataset, args.clip_batch_size, args.clip_frames, records)
                else:
                    validation = strict_loader(source_dataset, config["test_batchSize"], records)
                # Val and train share one owned LMDB environment. Opening two
                # separate dataset objects for the same path fails in python-lmdb.
                validation.source_dataset = None
                result.update(train_arm(model, extractor, training, validation, references[scope]["val"],
                                        arm, args, config, folder, source, utilities.DEVICE))
                val = export_scores(model, extractor, validation, route, utilities.DEVICE)
                legacy.verify_paired_base(references[scope]["val"], val)
                threshold = validation_threshold(val, val["score"], args.val_fpr)
                result.update(val_threshold=threshold, validation=metrics(val, val["score"], threshold), datasets={})
                export_dir = folder / "exports"
                export_dir.mkdir()
                legacy.save_npz(export_dir / "val.npz", val)
                # Close val/train LMDB environments before opening test datasets.
                legacy.close_loader(validation)
                validation = None
                legacy.close_loader(training)
                training = None
                for dataset in TEST_DS:
                    legacy.seed_everything(args.seed)
                    frames = loader(dataset, scope == "temporal")
                    try:
                        values = export_scores(model, extractor, frames, route, utilities.DEVICE)
                    finally:
                        legacy.close_loader(frames)
                    legacy.verify_paired_base(references[scope][dataset], values)
                    legacy.save_npz(export_dir / f"{dataset}.npz", values)
                    result["datasets"][dataset] = metrics(values, values["score"], threshold)
                if legacy.state_sha256(base) != baseline_hash:
                    raise RuntimeError("B0 isolation failed: original state changed")
                result.update(status="OK", base_scores_bitwise_identical=True,
                              checkpoint_sha256=legacy.file_sha256(folder / "best.pth"),
                              export_sha256={p.name: legacy.file_sha256(p) for p in export_dir.glob("*.npz")},
                              score_semantics="broadcast clip score" if route == "temporal" else "frame score")
            except Exception as exc:
                result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                if "B0 isolation" in str(exc):
                    raise
            finally:
                for frames in (training, validation):
                    if frames is not None:
                        legacy.close_loader(frames)
                results["arms"][arm] = result
                result["total_seconds"] = time.perf_counter() - arm_started
                legacy.write_json(folder / "result.json", result)
                legacy.write_json(root / "all_results.json", results)
                del model
                if utilities.DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
                if legacy.state_sha256(base) != baseline_hash:
                    raise RuntimeError("B0 isolation failed: original state changed")
                if legacy.file_sha256(source["checkpoint"]) != source["base_sha256"]:
                    raise RuntimeError("B0 isolation failed: original checkpoint file changed")
            detail = f" — {result['error']}" if "error" in result else ""
            print(f"{arm}: {result['status']}{detail}", flush=True)
    finally:
        manifest["base_state_sha256_after"] = legacy.state_sha256(base)
        manifest["base_state_unchanged"] = manifest["base_state_sha256_after"] == baseline_hash
        manifest["base_file_unchanged"] = legacy.file_sha256(source["checkpoint"]) == source["base_sha256"]
        legacy.write_json(root / "manifest.json", manifest)
    print(f"Results: {root / 'all_results.json'}", flush=True)
    return 0 if manifest["base_state_unchanged"] and manifest["base_file_unchanged"] and all(
        row["status"] == "OK" for row in results["arms"].values()) else 1


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    positive = (args.n_epochs, args.hidden_dim, args.rank, args.matrix_layers, args.clip_frames, args.clip_batch_size)
    if min(positive) < 1 or args.preserve_top < 0 or (args.batch_size is not None and args.batch_size < 2):
        parser.error("Require positive dimensions/budgets and nonnegative preserve_top")
    if args.clip_batch_size < 2 or (args.max_train_batches is not None and args.max_train_batches < 1):
        parser.error("clip_batch_size>=2 and max_train_batches>=1 required")
    if any(not math.isfinite(v) or v <= 0 for v in (args.lr, args.matrix_lr)):
        parser.error("Learning rates must be finite and positive")
    if any(not math.isfinite(v) or v < 0 for v in (args.weight_decay, args.orth_weight)):
        parser.error("Loss weights must be finite and nonnegative")
    if not 0 < args.sampler_real_ratio < 1 or not 0 < args.val_fpr < 1:
        parser.error("Sampling ratio and val_fpr must be between zero and one")
    if len(set(args.arms)) != len(args.arms) or (args.seeds and len(set(args.seeds)) != len(args.seeds)):
        parser.error("Arms and seeds must be unique")
    if any(ARMS[name][0] == "temporal" for name in args.arms):
        try:
            legacy.validate_sampling(args.clip_batch_size, args.sampler_real_ratio)
        except ValueError as exc:
            parser.error(f"Invalid clip sampling: {exc}")
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return 0
    seeds = args.seeds or [args.seed]
    codes = []
    for seed in seeds:
        args.seed = seed
        try:
            source = legacy.resolve_source(args)
            config, partition, hashes = legacy.preflight(args, source)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
        if args.preflight:
            print(json.dumps({"experiment": "E1002", "seed": seed, "status": "OK",
                              "base_sha256": source["base_sha256"], "metadata_sha256": hashes,
                              "note": "source/metadata only; images and ML runtime unverified"}, indent=2))
            codes.append(0)
        else:
            codes.append(run_seed(args, source, config, partition, hashes))
    return max(codes)


if __name__ == "__main__":
    sys.exit(main())

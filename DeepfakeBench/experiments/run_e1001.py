"""E1001/G30: train independent evidence/decision tokens on an existing frozen B0."""

import argparse
import datetime
from functools import partial
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ARMS = {"K1L20": (1, 20, False), "K4L20": (4, 20, False),
        "K8L20": (8, 20, False), "K4L16": (4, 16, False), "Full": (4, 20, True)}
DEFAULT_ARMS = ["K1L20", "K4L20", "Full"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_run", type=Path, help="Existing E0924 run containing training/G26_B0")
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--base_config", type=Path)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1001")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(DEFAULT_ARMS))
    parser.add_argument("--decision_source", choices=list(ARMS), default="Full")
    parser.add_argument("--skip_decisions", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--aux_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=.01)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch_size", type=int, help="Auxiliary train batch size; defaults to B0 config")
    parser.add_argument("--sampler_real_ratio", type=float, default=.3)
    parser.add_argument("--router_steps", type=int, default=300)
    parser.add_argument("--router_lr", type=float, default=.01)
    parser.add_argument("--reject_threshold", type=float, default=.7)
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root")
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Check source/data identities without ML imports")
    return parser


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def resolve_source(args):
    if args.base_run:
        if args.base_checkpoint or args.base_config:
            raise ValueError("Use --base_run OR both --base_checkpoint and --base_config")
        folder = args.base_run / "training/G26_B0"
        pointer = args.base_run / "training_results.json"
        result = json.loads(pointer.read_text()).get("G26_B0") if pointer.is_file() else None
        if result is None:
            result = json.loads((folder / "result.json").read_text())
        elif result.get("artifact_dir"):
            folder = Path(result["artifact_dir"])
        if result.get("status") != "OK" or not result.get("ckpt"):
            raise ValueError("The source E0924 B0 must have a successful checkpoint")
        checkpoint, config_path = Path(result["ckpt"]), folder / "train_config.json"
    elif args.base_checkpoint and args.base_config:
        checkpoint, config_path = args.base_checkpoint, args.base_config
    else:
        raise ValueError("Require --base_run, or both --base_checkpoint and --base_config; B0 is never retrained")
    config = json.loads(config_path.read_text())
    if config.get("model_name") != "effort":
        raise ValueError("G30 requires the original effort B0, not a G26/G27 checkpoint")
    if any(config.get(key) for key in ("multi_crop", "video_mode", "use_freq_split", "use_mixup", "use_texture_crop")):
        raise ValueError("E1001 requires RGB single-frame B0 without mixup/texture crops")
    if config.get("freq_ablation") or config.get("residual_ablation"):
        raise ValueError("E1001 requires the original RGB B0 input protocol")
    return {"checkpoint": str(checkpoint.resolve()), "config_path": str(config_path.resolve()),
            "base_sha256": file_sha256(checkpoint), "config_sha256": file_sha256(config_path), "config": config}


def arm_settings(args, arm):
    tokens, layer, full = ARMS[arm]
    return dict(num_tokens=tokens, memory_layer=layer, hidden_dim=args.hidden_dim,
                num_heads=args.num_heads, depth=args.depth, balance_weight=.1 if full else 0.,
                hard_weighting=full, consistency_weight=.1 if full else 0.)


def plan(args):
    return {"experiment": "E1001", "family": "G30", "base_training": False,
            "default_arms": list(DEFAULT_ARMS), "arms": {a: arm_settings(args, a) for a in args.arms},
            "decisions": [] if args.skip_decisions else ["linear", "mlp", "token"],
            "decision_source": args.decision_source,
            "boundary": "original B0 forward -> detached patch memory -> separate cross-attention decoder",
            "base": "reuse one existing B0 checkpoint; freeze all ViT/LoRA/head weights",
            "selection": "CDF-v2 gated frame AUC after each epoch; no base checkpoint selection",
            "calibration": "FF++ official val source components 70/30; fixed router budget/threshold",
            "primary_video_auc": "mean frame scores per full video path; legacy basename AUC also reported",
            "training_input": "strict single RGB frames, original resize/normalization, no random augmentation",
            "isolation": "exact base score comparison on every export; state hash before/after auxiliary training"}


def core_module():
    # Load only this independent torch module, without detectors/__init__ and its
    # dataset/network dependencies. Cached decision fitting needs no CLIP loader.
    name = "e1001_g30_sidecar"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "training/detectors/g30_sidecar.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def fit_token_router(data, base_sha256, evidence_sha256, steps=300, lr=.01, seed=1024):
    import numpy as np
    import torch
    from e0924_decision import feature_matrix

    if steps < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid decision-token training budget")
    core = core_module()
    core.FrozenEvidenceSidecar._validate_hash(base_sha256)
    core.FrozenEvidenceSidecar._validate_hash(evidence_sha256)
    x = feature_matrix(data, "model")
    p, e, labels = data["cls_prob"], data["evidence_prob"], data["labels"]
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Decision calibration requires binary labels")
    disagreement = (p > .5) != (e > .5)
    if not disagreement.any():
        raise ValueError("No disagreement samples for decision-token training")
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1
    x = np.clip((x - mean) / std, -10, 10)[disagreement]
    target = ((e[disagreement] > .5) == labels[disagreement]).astype(np.float32)
    _, inverse, counts = np.unique(data["video_id"][disagreement], return_inverse=True, return_counts=True)
    weights = (1 / counts[inverse]).astype(np.float32)
    weights /= weights.mean()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = core.DecisionToken(x.shape[1]).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=.001)
        inputs, truth, weight = torch.tensor(x, dtype=torch.float32), torch.tensor(target), torch.tensor(weights)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(inputs), truth,
                                                                         reduction="none") * weight).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite decision-token loss")
            loss.backward()
            optimizer.step()
    return {"version": 1, "family": "G30_decision_token", "base_sha256": base_sha256,
            "evidence_sha256": evidence_sha256, "input_dim": x.shape[1], "hidden_dim": 32, "num_heads": 4,
            "mean": mean.tolist(), "std": std.tolist(), "seed": seed, "steps": steps, "lr": lr,
            "weight_decay": .001, "fit_frames": len(labels), "disagreement_frames": int(disagreement.sum()),
            "video_weighting": "equal disagreement-video weight", "last_loss": float(loss.detach()),
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def score_token_router(artifact, data, threshold=.7, *, base_sha256, evidence_sha256):
    import numpy as np
    import torch
    from e0924_decision import feature_matrix

    if not .5 <= threshold <= 1 or artifact.get("version") != 1 or artifact.get("family") != "G30_decision_token":
        raise ValueError("Invalid decision-token artifact/threshold")
    core = core_module()
    core.FrozenEvidenceSidecar._validate_hash(base_sha256)
    core.FrozenEvidenceSidecar._validate_hash(evidence_sha256)
    if artifact.get("base_sha256") != base_sha256 or artifact.get("evidence_sha256") != evidence_sha256:
        raise ValueError("Decision-token checkpoint identity does not match B0/evidence")
    x = feature_matrix(data, "model")
    mean, std = np.asarray(artifact["mean"]), np.asarray(artifact["std"])
    if (mean.shape != (x.shape[1],) or std.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("Invalid decision-token normalization")
    if any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in artifact["state_dict"].values()):
        raise ValueError("Decision-token parameters must be finite tensors")
    x = np.clip((x - mean) / std, -10, 10)
    with torch.random.fork_rng(devices=[]):
        model = core_module().DecisionToken(artifact["input_dim"], artifact["hidden_dim"], artifact["num_heads"]).cpu()
        incoming, expected = artifact["state_dict"], model.state_dict()
        if incoming.keys() != expected.keys() or any(incoming[k].shape != expected[k].shape or
                incoming[k].dtype != expected[k].dtype for k in expected):
            raise ValueError("Decision-token checkpoint has unexpected parameters")
        model.load_state_dict(incoming, strict=True)
        model.eval()
        with torch.no_grad():
            batches = []
            for start in range(0, len(x), 2048):
                logits = model(torch.tensor(x[start:start + 2048], dtype=torch.float32))
                if not torch.isfinite(logits).all():
                    raise ValueError("Decision-token logits must be finite")
                batches.append(logits.sigmoid().numpy())
            trust = np.concatenate(batches)
    if not np.isfinite(trust).all():
        raise ValueError("Decision-token trust must be finite")
    p, e = data["cls_prob"], data["evidence_prob"]
    use = (trust > threshold) & ((p > .5) != (e > .5))
    return np.where(use, e, p)


def score_cached_router(artifact, data, threshold=.7, *, base_sha256, evidence_sha256):
    import numpy as np
    from e0924_decision import feature_matrix, score_router

    core = core_module()
    core.FrozenEvidenceSidecar._validate_hash(base_sha256)
    core.FrozenEvidenceSidecar._validate_hash(evidence_sha256)
    if artifact.get("base_sha256") != base_sha256 or artifact.get("evidence_sha256") != evidence_sha256:
        raise ValueError("Cached router identity does not match B0/evidence")
    x = feature_matrix(data, artifact["kind"])
    mean, std = np.asarray(artifact["mean"]), np.asarray(artifact["std"])
    if (mean.shape != (x.shape[1],) or std.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("Invalid cached router normalization")
    if any(not np.isfinite(layer[key]).all() for layer in artifact["layers"] for key in ("weight", "bias")):
        raise ValueError("Cached router parameters must be finite")
    scores = score_router(artifact, data, threshold)
    if not np.isfinite(scores).all():
        raise ValueError("Cached router scores must be finite")
    return scores


def verify_paired_base(before, after):
    import numpy as np

    for key in ("labels", "path", "video_id", "cls_prob", "global_log_odds"):
        if not np.array_equal(before[key], after[key]):
            raise RuntimeError(f"B0 isolation failed: paired {key} changed")


def readout_metrics(data, scores):
    import numpy as np
    from e0924_decision import auc

    labels, scores = np.asarray(data["labels"]), np.asarray(scores)
    if labels.ndim != 1 or scores.shape != labels.shape or len(data["video_id"]) != len(labels):
        raise ValueError("Misaligned frame identities/scores")
    groups = {}
    for video, label, score in zip(data["video_id"], labels, scores):
        groups.setdefault(str(video), []).append((int(label), float(score)))
    if any(len({label for label, _ in rows}) != 1 for rows in groups.values()):
        raise ValueError("Conflicting labels inside a full-path video group")
    video_auc = auc([rows[0][0] for rows in groups.values()],
                    [np.mean([score for _, score in rows]) for rows in groups.values()])
    legacy = {}
    for video, rows in groups.items():
        legacy.setdefault(video.replace("\\", "/").rsplit("/", 1)[-1], []).extend(rows)
    try:
        legacy_auc = auc([int(np.mean([label for label, _ in rows])) for rows in legacy.values()],
                         [np.mean([score for _, score in rows]) for rows in legacy.values()])
    except ValueError:
        legacy_auc = None
    cls_correct, correct = (data["cls_prob"] > .5) == labels, (scores > .5) == labels
    return {"frame_auc": auc(labels, scores), "video_auc": video_auc, "video_auc_fullpath": video_auc,
            "video_auc_legacy": legacy_auc, "legacy_group_collisions": len(groups) - len(legacy),
            "legacy_mixed_label_groups": sum(len({label for label, _ in rows}) > 1 for rows in legacy.values()),
            "num_frames": len(labels), "num_videos": len(groups), "accuracy": float(correct.mean()),
            "corrected_frames": int((~cls_correct & correct).sum()), "harmed_frames": int((cls_correct & ~correct).sum())}


def export_scores(model, loader, auxiliary_enabled=True):
    import numpy as np
    import torch

    model.eval()
    device = next(model.parameters()).device
    batches = {key: [] for key in ("labels", "path", "video_id", "cls_prob", "global_log_odds")}
    if auxiliary_enabled:
        batches.update({key: [] for key in ("evidence_prob", "gated_prob", "query_log_odds")})
    with torch.no_grad():
        for data in loader:
            inputs = {"image": data["image"].to(device)}
            if auxiliary_enabled:
                output = model(inputs, inference=True)["g30"]
                p, logits = output["cls_prob"], output["global_logits"]
                for name in ("evidence_prob", "gated_prob"):
                    batches[name].append(output[name].cpu().numpy())
                batches["query_log_odds"].append(output["evidence_log_odds"].cpu().numpy())
            else:
                output = model(inputs, inference=True)
                p, logits = output["prob"], output["cls"]
            batches["labels"].append(data["label"].cpu().numpy())
            batches["path"].append(np.asarray(data["path"]))
            batches["video_id"].append(np.asarray(data["video_id"]))
            batches["cls_prob"].append(p.cpu().numpy())
            batches["global_log_odds"].append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    if not batches["labels"]:
        raise ValueError("Empty score export")
    result = {key: np.concatenate(parts) for key, parts in batches.items()}
    if any(not np.isfinite(v).all() for k, v in result.items() if k not in ("path", "video_id")):
        raise ValueError("Nonfinite score export")
    return result


def save_npz(path, data):
    import numpy as np

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(path)


def state_sha256(model):
    import torch

    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def seed_everything(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preflight(args, source):
    from e0924_protocol import calibration_partition
    from run_g25 import TEST_DS

    config = dict(source["config"])
    config.update(dataset_json_folder=str((args.dataset_json_folder or Path(config["dataset_json_folder"])).resolve()),
                  train_dataset=["FaceForensics++"], manualSeed=args.seed, multi_crop=False,
                  use_data_augmentation=False)
    if args.rgb_root:
        config["rgb_root_override"] = args.rgb_root
    if args.clip_pretrained_path:
        config["clip_pretrained_path"] = args.clip_pretrained_path
    if args.batch_size:
        config["train_batchSize"] = args.batch_size
    validate_sampling(config["train_batchSize"], args.sampler_real_ratio)
    hashes = {}
    for dataset in TEST_DS:
        hashes[dataset] = file_sha256(Path(config["dataset_json_folder"]) / f"{dataset}.json")
    metadata = json.loads((Path(config["dataset_json_folder"]) / "FaceForensics++.json").read_text())
    partition = calibration_partition(metadata, compression=config["compression"], seed=args.seed, frames=8)
    if any({r["label_name"] == "FF-real" for r in rows} != {True, False}
           for rows in partition["records"].values()):
        raise ValueError("Both calibration partitions require real and fake videos")
    return config, partition, hashes


def validate_sampling(batch_size, ratio):
    if batch_size < 2 or not 0 < ratio < 1:
        raise ValueError("Balanced training requires batch_size>=2 and ratio in (0,1)")
    effective = 2 * (batch_size // 2)
    real = max(1, round(effective * ratio))
    if real >= effective:
        raise ValueError("Rounded sampling counts must include both real and fake frames")


def make_train_loader(config, ratio):
    import numpy as np
    from torch.utils.data import DataLoader
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    from dataset.balance_batch_sampler import BalanceBatchSampler
    from e0924_export import strict_loader

    validate_sampling(config["train_batchSize"], ratio)
    base = DeepfakeAbstractBaseDataset(config=dict(config), mode="train")
    frames = strict_loader(base, config["train_batchSize"]).dataset
    labels = (np.asarray(base.label_list) != 0).astype(int)
    sampler = BalanceBatchSampler(labels, max(1, config["train_batchSize"] // 2), real_ratio=ratio)
    loader = DataLoader(frames, batch_sampler=sampler, num_workers=0)
    loader.source_dataset = base
    return loader


def close_loader(loader):
    """Release an owned LMDB environment before opening the next same dataset."""
    dataset = getattr(loader, "source_dataset", None)
    environment = getattr(dataset, "env", None)
    if environment is not None:
        environment.close()


def train_auxiliary(model, loader, validation_loader, reference, args, checkpoint, base_sha256):
    import torch

    optimizer = torch.optim.AdamW(model.auxiliary.parameters(), lr=args.aux_lr, weight_decay=args.weight_decay)
    device = next(model.parameters()).device
    best, history = -1., []
    for epoch in range(args.n_epochs):
        model.train()
        totals, steps = {}, 0
        for data in loader:
            inputs = {"image": data["image"].to(device)}
            labels = data["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            view = None
            if model.settings["consistency_weight"]:
                images = inputs["image"]
                scale = images.new_tensor(model.base.config["std"])[None, :, None, None]
                altered = .9 * images + .1 * images.mean((-2, -1), keepdim=True) + .02 / scale
                view = model({"image": altered})
            losses = model.losses(output, labels, view)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise ValueError("Nonfinite auxiliary training loss")
            losses["overall"].backward()
            if any(p.grad is not None or p.requires_grad for p in model.base.parameters()):
                raise RuntimeError("B0 isolation failed: trainable base/received gradient")
            optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.) + float(value.detach())
            steps += 1
            if steps % 100 == 0:
                print(f"  epoch={epoch + 1} step={steps}/{len(loader)} aux_loss={float(losses['overall'].detach()):.6f}", flush=True)
        if not steps:
            raise ValueError("Empty auxiliary training loader")
        values = export_scores(model, validation_loader)
        verify_paired_base(reference, values)
        metric = readout_metrics(values, values["gated_prob"])
        history.append({"epoch": epoch + 1, "steps": steps,
                        "losses": {k: v / steps for k, v in totals.items()}, "validation": metric})
        if metric["frame_auc"] > best:
            best = metric["frame_auc"]
            temporary = checkpoint.with_suffix(".pth.tmp")
            torch.save(model.checkpoint(base_sha256), temporary)
            temporary.replace(checkpoint)
        write_json(checkpoint.parent / "history.json", history)
        print(f"  epoch={epoch + 1} CDF_frame_auc={metric['frame_auc']:.6f} best={best:.6f}", flush=True)
    model.load_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True), base_sha256)
    return history


def run_decisions(args, root, exported, base_sha256, evidence_sha256):
    import torch
    from e0924_decision import fit_router

    results = {}
    folder = root / "decisions"
    folder.mkdir()
    for name, hidden in (("linear", 0), ("mlp", 16), ("token", None)):
        try:
            if name == "token":
                artifact = fit_token_router(exported["fit"], base_sha256, evidence_sha256,
                                            steps=args.router_steps, lr=args.router_lr, seed=args.seed)
                temporary = folder / "token.pth.tmp"
                torch.save(artifact, temporary)
                temporary.replace(folder / "token.pth")
                score = partial(score_token_router, base_sha256=base_sha256, evidence_sha256=evidence_sha256)
            else:
                artifact = fit_router(exported["fit"], "model", hidden, seed=args.seed,
                                      steps=args.router_steps, lr=args.router_lr)
                artifact.update(base_sha256=base_sha256, evidence_sha256=evidence_sha256)
                write_json(folder / f"{name}.json", artifact)
                score = partial(score_cached_router, base_sha256=base_sha256, evidence_sha256=evidence_sha256)
            report = {"status": "OK", "base_sha256": base_sha256, "evidence_sha256": evidence_sha256,
                      "threshold": args.reject_threshold, "datasets": {}}
            for dataset, data in exported.items():
                values = score(artifact, data, threshold=args.reject_threshold)
                report["datasets"][dataset] = readout_metrics(data, values)
                save_npz(folder / f"{name}_{dataset}.npz", {"scores": values, "path": data["path"],
                                                           "video_id": data["video_id"], "labels": data["labels"]})
        except Exception as exc:
            # No disagreements is a valid base-only outcome, not a fitted router.
            if isinstance(exc, ValueError) and "No disagreement" in str(exc):
                report = {"status": "SKIPPED_NO_DISAGREEMENT", "reason": str(exc)}
            else:
                report = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        results[name] = report
        write_json(folder / f"{name}_result.json", report)
        write_json(folder / "all_results.json", results)
        print(f"G30/decision_{name}: {report['status']}", flush=True)
    return results


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.n_epochs < 1 or args.router_steps < 1 or min(args.hidden_dim, args.num_heads, args.depth) < 1
            or args.hidden_dim % args.num_heads or (args.batch_size is not None and args.batch_size < 2)):
        parser.error("Require positive training budgets/dimensions and batch_size>=2")
    if any(not math.isfinite(v) or v <= 0 for v in (args.aux_lr, args.router_lr)):
        parser.error("Learning rates must be finite and positive")
    if (not math.isfinite(args.weight_decay) or args.weight_decay < 0 or
            not 0 < args.sampler_real_ratio < 1 or not .5 <= args.reject_threshold <= 1):
        parser.error("Invalid weight decay, sampling ratio or reject threshold")
    if len(args.arms) != len(set(args.arms)) or (not args.skip_decisions and args.decision_source not in args.arms):
        parser.error("Arms must be unique and include the declared decision_source")
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return 0
    try:
        source = resolve_source(args)
        config, partition, metadata_hashes = preflight(args, source)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.preflight:
        print(json.dumps({"status": "OK", "base_sha256": source["base_sha256"], "dataset_sha256": metadata_hashes,
                          "note": "source/metadata only; image files and ML runtime have not been checked"}, indent=2))
        return 0

    import torch
    import transformers
    import experiment_utils as utilities
    from e0924_export import strict_loader
    from run_e0924 import source_hashes
    from run_g25 import TEST_DS, VAL_DS

    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = args.output_dir / f"seed{args.seed}_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    hashes["experiments/run_e1001.py"] = file_sha256(Path(__file__))
    manifest = {"protocol": plan(args), "arguments": vars(args), "source": source,
                "dataset_sha256": metadata_hashes, "source_sha256": hashes,
                "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__}
    # Path arguments are serialized as strings, not Python reprs.
    manifest["arguments"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(root / "manifest.json", manifest)
    write_json(root / "calibration_partition.json", partition)
    write_json(root / "runtime_config.json", config)
    base = utilities.load_model(config, source["checkpoint"])
    base.requires_grad_(False)
    base.eval()
    state_before = state_sha256(base)
    manifest["base_state_sha256_before"] = state_before
    write_json(root / "manifest.json", manifest)
    print(f"E1001/G30 frozen B0: {source['checkpoint']}\nResults: {root}", flush=True)

    def loader(dataset, records=None):
        dataset_object = utilities.get_data_loader(config, dataset).dataset
        frames = strict_loader(dataset_object, config["test_batchSize"], records)
        frames.source_dataset = dataset_object
        return frames

    seed_everything(args.seed)
    validation_loader = loader(VAL_DS)

    def paired_loader(dataset):
        if dataset == VAL_DS:
            return validation_loader
        if dataset in ("fit", "holdout"):
            return loader("FaceForensics++", partition["records"][dataset])
        return loader(dataset)

    b0_dir = root / "B0"
    b0_dir.mkdir()
    references, b0_results = {}, {}
    # Cache all standalone B0 readouts before attaching any auxiliary module.
    for dataset in [*TEST_DS, "fit", "holdout"]:
        seed_everything(args.seed)
        frames = paired_loader(dataset)
        try:
            values = export_scores(base, frames, auxiliary_enabled=False)
        finally:
            if frames is not validation_loader:
                close_loader(frames)
        save_npz(b0_dir / f"{dataset}.npz", values)
        references[dataset] = values
        b0_results[dataset] = readout_metrics(values, values["cls_prob"])
    write_json(b0_dir / "result.json", {"status": "OK", "reused_checkpoint": source["checkpoint"], "datasets": b0_results})
    results = {"experiment": "E1001", "family": "G30", "B0": b0_results, "arms": {}, "decisions": {}}
    write_json(root / "all_results.json", results)
    for arm in args.arms:
        folder = root / f"G30_{arm}"
        folder.mkdir()
        seed_everything(args.seed)
        model = core_module().FrozenEvidenceSidecar(base, **arm_settings(args, arm)).to(utilities.DEVICE)
        result = {"status": "TRAIN_FAILED", "settings": model.settings, "base_sha256": source["base_sha256"]}
        print(f"G30/{arm}: {model.settings}", flush=True)
        try:
            checkpoint = folder / "auxiliary_best.pth"
            training_loader = make_train_loader(config, args.sampler_real_ratio)
            try:
                history = train_auxiliary(model, training_loader, validation_loader,
                                          references[VAL_DS], args, checkpoint, source["base_sha256"])
            finally:
                close_loader(training_loader)
            if state_sha256(base) != state_before:
                raise RuntimeError("B0 isolation failed: base state changed during auxiliary training")
            result.update(status="EVAL_FAILED", checkpoint=str(checkpoint),
                          selected_epoch=max(history, key=lambda row: row["validation"]["frame_auc"])["epoch"],
                          base_state_unchanged=True, datasets={})
            exported = {}
            export_dir = folder / "exports"
            export_dir.mkdir()
            for dataset in references:
                seed_everything(args.seed)
                frames = paired_loader(dataset)
                try:
                    values = export_scores(model, frames)
                finally:
                    if frames is not validation_loader:
                        close_loader(frames)
                verify_paired_base(references[dataset], values)
                save_npz(export_dir / f"{dataset}.npz", values)
                exported[dataset] = values
                result["datasets"][dataset] = {name: readout_metrics(values, values[key]) for name, key in
                                                (("B0", "cls_prob"), ("evidence", "evidence_prob"), ("gated", "gated_prob"))}
            result.update(status="OK", base_scores_bitwise_identical=True,
                          export_sha256={p.name: file_sha256(p) for p in export_dir.glob("*.npz")})
            if arm == args.decision_source and not args.skip_decisions:
                results["decisions"] = run_decisions(args, root, exported, source["base_sha256"], file_sha256(checkpoint))
        except Exception as exc:
            result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, RuntimeError) and "B0 isolation" in str(exc):
                write_json(folder / "result.json", result)
                raise  # fail closed: later arms cannot use a changed baseline
        finally:
            write_json(folder / "result.json", result)
            results["arms"][arm] = result
            write_json(root / "all_results.json", results)
        print(f"G30/{arm}: {result['status']}", flush=True)
        del model
    manifest["base_state_sha256_after"] = state_sha256(base)
    manifest["base_state_unchanged"] = manifest["base_state_sha256_after"] == state_before
    manifest["base_file_unchanged"] = file_sha256(source["checkpoint"]) == source["base_sha256"]
    write_json(root / "manifest.json", manifest)
    close_loader(validation_loader)
    print(f"Results: {root / 'all_results.json'}", flush=True)
    return 0 if manifest["base_state_unchanged"] and manifest["base_file_unchanged"] and all(
        row["status"] == "OK" for row in results["arms"].values()) and all(
        row["status"] in ("OK", "SKIPPED_NO_DISAGREEMENT") for row in results["decisions"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())

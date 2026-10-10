"""E1010: auxiliary-only forgery likelihood and asymmetric token supervision."""

import argparse
import copy
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys

import run_e1001 as frozen
from run_g25 import TEST_DS, VAL_DS


ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT_DS = [dataset for dataset in TEST_DS if dataset not in (VAL_DS, "FaceForensics++")]
OBJECTIVES = {"PLAIN": ("off", False), "UNIFORM": ("uniform", False),
              "FL": ("weighted", False), "MIL": ("off", True),
              "UNIFORM_MIL": ("uniform", True), "FL_MIL": ("weighted", True)}
MASKS = {"00": "read_only", "10": "cls_only", "01": "patch_only", "11": "full"}
STRUCTURES = {f"G1_{name}": {"family": "lfeq", "lfeq_fusion_weight": weight}
              for name, weight in (("L1", .5), ("L2", 1.), ("L3", 0.))}
STRUCTURES.update({f"G2_{prefix}{code}": {"family": "aux", "attention_mode": mask, "supervision": supervision}
                   for prefix, supervision in (("M", "max"), ("A", "all")) for code, mask in MASKS.items()})
STRUCTURES["G3"] = {"family": "decoder"}
ARMS = {f"{structure}_{objective}": {**settings, "likelihood": likelihood, "asymmetric": asymmetric}
        for structure, settings in STRUCTURES.items()
        for objective, (likelihood, asymmetric) in OBJECTIVES.items()}
ARMS.update({f"G4_{method}": {"family": "prototype", "method": method}
             for method in ("ALL", "RANDOM", "TOPK")})

# Public E1001 helpers preserve the source checkpoint and strict input protocol.
preflight = frozen.preflight
make_train_loader = frozen.make_train_loader
verify_paired_base = frozen.verify_paired_base
state_sha256 = frozen.state_sha256
file_sha256 = frozen.file_sha256
seed_everything = frozen.seed_everything
write_json = frozen.write_json
save_npz = frozen.save_npz
close_loader = frozen.close_loader


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_run", type=Path, help="Existing E0924 or E1010 run with a successful trained B0")
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--base_config", type=Path)
    parser.add_argument("--baseline_only", action="store_true", help="Train/reuse and evaluate B0, then stop")
    parser.add_argument("--base_n_epochs", type=int, default=10,
                        help="Historical B0 nEpochs (10 means epoch 0-10, eleven passes)")
    parser.add_argument("--base_sampler_real_ratio", type=float, default=.3)
    parser.add_argument("--base_batch_size", type=int, help="B0 training batch override; default historical config")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1010")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--aux_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=.01)
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="G1/G3 decoder width; G2 uses B0's suffix width")
    parser.add_argument("--num_heads", type=int,
                        help="G1/G3 head override (defaults 8/4); G2 inherits B0 suffix attention")
    parser.add_argument("--depth", type=int, default=2,
                        help="G1/G3 decoder blocks; G2 inherits B0's remaining blocks")
    parser.add_argument("--num_tokens", type=int, help="Override token count for all three families")
    parser.add_argument("--lfeq_num_tokens", type=int, default=8)
    parser.add_argument("--aux_num_tokens", type=int, default=4)
    parser.add_argument("--decoder_num_tokens", type=int, default=4)
    parser.add_argument("--lfeq_dropout", type=float, default=.1)
    parser.add_argument("--memory_layer", type=int, default=20,
                        help="G2 insertion block and G3 memory block; G1 reads final patches")
    parser.add_argument("--batch_size", type=int, help="Auxiliary batch size; defaults to B0 config")
    parser.add_argument("--sampler_real_ratio", type=float, default=.3)
    parser.add_argument("--likelihood_temperature", type=float, default=.1)
    parser.add_argument("--likelihood_weight", type=float, default=.14)
    parser.add_argument("--contrastive_temperature", type=float, default=.1)
    parser.add_argument("--max_contrastive_patches", type=int, default=16)
    parser.add_argument("--mil_temperature", type=float, default=.5)
    parser.add_argument("--gate_width", type=float, default=.2)
    parser.add_argument("--aux_max_weight", type=float, default=.5)
    parser.add_argument("--diversity_weight", type=float, default=.01)
    parser.add_argument("--prototype_top_k", type=int, default=16,
                        help="Selected fake patches per training frame for RANDOM/TOPK")
    parser.add_argument("--prototype_pool_top_k", type=int, default=16,
                        help="Identical test-time top-K patch pooling for all G4 methods")
    parser.add_argument("--prototype_temperature", type=float, default=.1)
    parser.add_argument("--prototype_frames_per_class", type=int, default=32,
                        help="Maximum unique FF++ train frames per binary class for G4")
    parser.add_argument("--occlusion_batch_size", type=int, default=16,
                        help="Maximum perturbed images in one G4 attribution forward")
    parser.add_argument("--occlusion_fill", choices=["mean"], default="mean")
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root")
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Source and metadata checks without ML imports")
    return parser


def arm_settings(args, arm):
    objective = ARMS[arm]
    if objective["family"] == "prototype":
        return {**objective, "top_k": args.prototype_top_k,
                "pool_top_k": args.prototype_pool_top_k, "temperature": args.prototype_temperature,
                "max_frames_per_class": args.prototype_frames_per_class, "seed": args.seed,
                "occlusion_batch_size": args.occlusion_batch_size, "fill": args.occlusion_fill,
                "gate_width": args.gate_width, "aux_max_weight": args.aux_max_weight}
    lfeq = objective["family"] == "lfeq"
    defaults = {"lfeq": args.lfeq_num_tokens, "aux": args.aux_num_tokens, "decoder": args.decoder_num_tokens}
    tokens = args.num_tokens if args.num_tokens is not None else defaults[objective["family"]]
    heads = args.num_heads if args.num_heads is not None else (8 if lfeq else 4)
    return {"lfeq_fusion_weight": .5, "attention_mode": "read_only", "supervision": "max",
            **objective, "num_tokens": tokens, "memory_layer": args.memory_layer,
            "memory_source": "final" if lfeq else "block_input", "hidden_dim": args.hidden_dim,
            "num_heads": heads, "depth": args.depth, "lfeq_dropout": args.lfeq_dropout,
            "likelihood_temperature": args.likelihood_temperature, "likelihood_weight": args.likelihood_weight,
            "contrastive_temperature": args.contrastive_temperature,
            "max_contrastive_patches": args.max_contrastive_patches,
            "mil_temperature": args.mil_temperature, "gate_width": args.gate_width,
            "aux_max_weight": args.aux_max_weight,
            "diversity_weight": 0. if objective["family"] == "decoder" else args.diversity_weight}


def plan(args):
    mil_aliases = {f"G2_A{code}_{objective}": f"G2_M{code}_{objective}"
                   for code in MASKS for objective in ("MIL", "UNIFORM_MIL", "FL_MIL")}
    fresh = not (args.base_run or args.base_checkpoint or args.base_config)
    arms = [] if args.baseline_only else args.arms
    return {"experiment": "E1010", "groups": {"G0": "B0", "G1": "lfeq", "G2": "aux", "G3": "decoder", "G4": "prototype"},
            "base_training": fresh, "default_arms": list(ARMS),
            "arms": {arm: arm_settings(args, arm) for arm in arms}, "decisions": [],
            "base": "train a fresh original B0 when no source is supplied, otherwise reuse; freeze before all auxiliary stages",
            "baseline": {"protocol": "E0924/G26_B0", "mode": "train" if fresh else "reuse",
                         "baseline_only": args.baseline_only, "nEpochs": args.base_n_epochs,
                         "training_passes": args.base_n_epochs + 1,
                         "sampler_real_ratio": args.base_sampler_real_ratio,
                         "train_batch_override": args.base_batch_size,
                         "selection": "CDF-v2 frame AUC",
                         "augmentation": "original B0 training augmentation; disabled only after training",
                         "reproduction": "historical basename metrics and separate strict full-path G0 metrics"},
            "G1_source": "G18 LFEQ queries and original readout on final B0 patches",
            "G2_source": "G25 four attention masks and max/all supervision in an independent frozen B0 suffix",
            "G3_source": "G30 independent evidence decoder reading block-input patches",
            "prototype_probe": {"training": False, "prototype_scope": "global_shared_train_only",
                                "source": "same seeded unique FF++ train frame subset for all three methods",
                                "max_frames_per_class": args.prototype_frames_per_class,
                                "top_k": args.prototype_top_k, "pool_top_k": args.prototype_pool_top_k,
                                "temperature": args.prototype_temperature,
                                "attribution": "pixel occlusion: original minus masked fake-real logit margin",
                                "fill": args.occlusion_fill, "occlusion_batch_size": args.occlusion_batch_size,
                                "cost": "TOPK adds P masked-image forwards per selected fake training frame; none at test",
                                "iteration": False},
            "equivalent_MIL_objectives": mil_aliases,
            "local_analysis": "train-time batch prototype likelihood on auxiliary projected patches only",
            "selection": "CDF-v2 gated frame AUC; one selected auxiliary checkpoint per arm",
            "primary_video_auc": "mean frame score per full video path; legacy basename also reported",
            "independent_datasets": list(INDEPENDENT_DS),
            "excluded_from_independent_mean": [VAL_DS, "FaceForensics++"],
            "training_input": "B0 resize/normalization; strict single RGB frames; no random augmentation",
            "comparison_boundary": "family structure preserved; frozen fitted B0 changes historical training conditions"}


def validate_sampling(batch_size, ratio, require_two_fake=False):
    frozen.validate_sampling(batch_size, ratio)
    effective = 2 * (batch_size // 2)
    real = max(1, round(effective * ratio))
    if require_two_fake and effective - real < 2:
        raise ValueError("Local contrastive learning requires at least two fake images and one real per batch")


def resolve_source(args):
    if args.base_run and (args.base_run / "G0/training/result.json").is_file():
        if args.base_checkpoint or args.base_config:
            raise ValueError("Use --base_run OR both --base_checkpoint and --base_config")
        folder = args.base_run / "G0/training"
        result = json.loads((folder / "result.json").read_text())
        if result.get("status") != "OK" or not result.get("ckpt"):
            raise ValueError("The E1010 source B0 must have successful training and evaluation")
        direct = copy.copy(args)
        direct.base_run = None
        direct.base_checkpoint = Path(result["ckpt"])
        direct.base_config = folder / "train_config.json"
        return frozen.resolve_source(direct)
    return frozen.resolve_source(args)


def planned_baseline_config(args):
    """Read input metadata settings for preflight, without creating an ML model.

    This preview is never used for training. The runtime uses the historical
    G26 builder so its augmentation, optimizer and model settings are preserved.
    """
    import yaml

    config = yaml.safe_load((ROOT / "training/config/detector/effort.yaml").read_text())
    config.update(yaml.safe_load((ROOT / "training/config/train_config.yaml").read_text()))
    local_metadata = ROOT / "preprocessing/dataset_json"
    if args.dataset_json_folder:
        config["dataset_json_folder"] = str(args.dataset_json_folder.resolve())
    elif local_metadata.is_dir():
        config["dataset_json_folder"] = str(local_metadata)
    if args.base_batch_size is not None:
        config["train_batchSize"] = args.base_batch_size
    config.update(model_name="effort", use_mixup=False, multi_crop=False,
                  use_texture_crop=False, use_freq_split=False)
    return config


def core_module():
    name = "e1010_auxiliary_core"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "training/detectors/e1010_tokens.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def readout_metrics(data, scores):
    import numpy as np

    report = frozen.readout_metrics(data, scores)
    real = np.asarray(data["labels"]) == 0
    report["real_fpr"] = float((np.asarray(scores)[real] > .5).mean()) if real.any() else None
    report["real_fpr_threshold"] = .5
    report["num_real_frames"] = int(real.sum())
    return report


def independent_mean(datasets, readout):
    return {"video_auc_fullpath_mean": sum(datasets[dataset][readout]["video_auc_fullpath"]
                                           for dataset in INDEPENDENT_DS) / len(INDEPENDENT_DS),
            "datasets": list(INDEPENDENT_DS), "excluded": [VAL_DS, "FaceForensics++"]}


def prototype_records(config, args):
    """Select a reproducible unique frame pool exclusively from FF++ train."""
    from e0924_protocol import canonical_path

    path = Path(config["dataset_json_folder"]) / "FaceForensics++.json"
    metadata = json.loads(path.read_text())["FaceForensics++"]
    candidates = {0: [], 1: []}
    seen = set()
    for splits in metadata.values():
        videos = splits["train"][config["compression"]]
        for video in videos.values():
            name = video["label"]
            label = int(config["label_dict"][name] != 0)
            for frame in video["frames"]:
                frame = canonical_path(frame)
                if frame in seen:
                    raise ValueError("Duplicate FF++ training frame in prototype selection")
                seen.add(frame)
                candidates[label].append((frame, name))
    generator = random.Random(args.seed)
    records, counts = [], {}
    for label, key in ((0, "real"), (1, "fake")):
        pool = sorted(candidates[label])
        if not pool:
            raise ValueError("Prototype selection requires real and fake FF++ training frames")
        generator.shuffle(pool)
        selected = pool[:args.prototype_frames_per_class]
        counts[key] = len(selected)
        records.extend({"label_name": name, "frames": [frame]} for frame, name in selected)
    return records, {"split": "train", "seed": args.seed, "counts": counts,
                     "max_frames_per_class": args.prototype_frames_per_class,
                     "selection": "sort unique paths, seeded shuffle per class, take budget",
                     "metadata_sha256": file_sha256(path), "records": records}


def export_prototype_scores(base, loader, artifact, args, base_sha256):
    import numpy as np
    import torch
    import e1010_prototypes as prototypes

    prototypes.validate_artifact(artifact, base_sha256)
    base.eval()
    device = next(base.parameters()).device
    arrays = {key: [] for key in ("labels", "path", "video_id", "cls_prob", "global_log_odds",
                                  "prototype_prob", "gated_prob")}
    with torch.no_grad():
        for batch in loader:
            output, patches = prototypes.capture_patches(base, batch["image"].to(device))
            p = output["prob"]
            e = prototypes.score_prototypes(patches, artifact, args.prototype_temperature,
                                             args.prototype_pool_top_k)
            weight = args.aux_max_weight * (1 - (p - .5).abs() / args.gate_width).clamp(0, 1)
            fused = torch.where(weight > 0, (1 - weight) * p + weight * e, p)
            for key, value in (("cls_prob", p), ("prototype_prob", e), ("gated_prob", fused),
                               ("global_log_odds", output["cls"][:, 1] - output["cls"][:, 0])):
                arrays[key].append(value.cpu().numpy())
            arrays["labels"].append(batch["label"].cpu().numpy())
            arrays["path"].append(np.asarray(batch["path"]))
            arrays["video_id"].append(np.asarray(batch["video_id"]))
    if not arrays["labels"]:
        raise ValueError("Empty prototype score export")
    result = {key: np.concatenate(parts) for key, parts in arrays.items()}
    if any(not np.isfinite(value).all() for key, value in result.items() if key not in ("path", "video_id")):
        raise ValueError("Nonfinite prototype score export")
    return result


def export_scores(model, loader, auxiliary_enabled=True):
    import numpy as np
    import torch

    model.eval()
    device = next(model.parameters()).device
    scored_model = model if auxiliary_enabled else getattr(model, "base", model)
    arrays = {key: [] for key in ("labels", "path", "video_id", "cls_prob", "global_log_odds")}
    if auxiliary_enabled:
        arrays.update({key: [] for key in ("evidence_prob", "gated_prob", "query_log_odds")})
    with torch.no_grad():
        for batch in loader:
            output = scored_model({"image": batch["image"].to(device)}, inference=True)
            if auxiliary_enabled:
                values = output["e1010"]
                prob, logits = values["cls_prob"], values["global_logits"]
                for key in ("evidence_prob", "gated_prob"):
                    arrays[key].append(values[key].cpu().numpy())
                arrays["query_log_odds"].append(values["evidence_log_odds"].cpu().numpy())
            else:
                prob, logits = output["prob"], output["cls"]
            arrays["labels"].append(batch["label"].cpu().numpy())
            arrays["path"].append(np.asarray(batch["path"]))
            arrays["video_id"].append(np.asarray(batch["video_id"]))
            arrays["cls_prob"].append(prob.cpu().numpy())
            arrays["global_log_odds"].append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    if not arrays["labels"]:
        raise ValueError("Empty E1010 export")
    result = {key: np.concatenate(parts) for key, parts in arrays.items()}
    if any(not np.isfinite(value).all() for key, value in result.items() if key not in ("path", "video_id")):
        raise ValueError("Nonfinite E1010 export")
    return result


def train_auxiliary(model, loader, validation_loader, reference, args, checkpoint, base_sha256,
                    base_state_sha256):
    import torch

    optimizer = torch.optim.AdamW((p for p in model.auxiliary.parameters() if p.requires_grad),
                                 lr=args.aux_lr, weight_decay=args.weight_decay)
    device = next(model.parameters()).device
    best, history = -1., []
    for epoch in range(args.n_epochs):
        model.train()
        totals, steps = {}, 0
        for batch in loader:
            labels = batch["label"].to(device)
            if model.settings["likelihood"] != "off" and ((labels == 0).sum() < 1 or (labels == 1).sum() < 2):
                raise ValueError("Local contrastive batch requires at least one real and two fake images")
            optimizer.zero_grad(set_to_none=True)
            output = model({"image": batch["image"].to(device)})
            losses = model.losses(output, labels)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise ValueError("Nonfinite E1010 training loss")
            losses["overall"].backward()
            if any(parameter.requires_grad or parameter.grad is not None for parameter in model.base.parameters()):
                raise RuntimeError("B0 isolation failed: trainable baseline or auxiliary gradient reached B0")
            optimizer.step()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.) + float(value.detach())
            steps += 1
            if steps % 100 == 0:
                print(f"  epoch={epoch + 1} step={steps}/{len(loader)} aux_loss={float(losses['overall'].detach()):.6f}", flush=True)
        if not steps:
            raise ValueError("Empty E1010 training loader")
        if state_sha256(model.base) != base_state_sha256:
            raise RuntimeError("B0 isolation failed: baseline state changed during auxiliary training")
        values = export_scores(model, validation_loader)
        verify_paired_base(reference, values)
        metric = readout_metrics(values, values["gated_prob"])
        history.append({"epoch": epoch + 1, "steps": steps,
                        "losses": {key: value / steps for key, value in totals.items()}, "validation": metric})
        if metric["frame_auc"] > best:
            best = metric["frame_auc"]
            temporary = checkpoint.with_suffix(".pth.tmp")
            torch.save(model.checkpoint(base_sha256), temporary)
            temporary.replace(checkpoint)
        write_json(checkpoint.parent / "history.json", history)
        print(f"  epoch={epoch + 1} CDF_gated_frame_auc={metric['frame_auc']:.6f} best={best:.6f}", flush=True)
    model.load_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True), base_sha256)
    return history


def _validate_arguments(parser, args):
    arms = [] if args.baseline_only else args.arms
    auxiliary_arms = [arm for arm in arms if ARMS[arm]["family"] != "prototype"]
    prototype_arms = [arm for arm in arms if ARMS[arm]["family"] == "prototype"]
    if not (args.base_run or args.base_checkpoint or args.base_config):
        if (args.base_n_epochs < 0 or not 0 < args.base_sampler_real_ratio < 1
                or (args.base_batch_size is not None and args.base_batch_size < 2)):
            parser.error("Require base_n_epochs>=0, a valid B0 sampling ratio and base_batch_size>=2")
    if len(args.arms) != len(set(args.arms)):
        parser.error("Do not repeat arms within a run")
    if args.seed < 0 or not 0 < args.gate_width <= .5 or not 0 <= args.aux_max_weight <= .5:
        parser.error("Invalid seed or fixed gate")
    if prototype_arms and (min(args.prototype_top_k, args.prototype_pool_top_k,
                              args.prototype_frames_per_class, args.occlusion_batch_size) < 1
                           or not math.isfinite(args.prototype_temperature) or args.prototype_temperature <= 0):
        parser.error("Prototype budgets and temperature must be positive and finite")
    if not auxiliary_arms:
        return
    if (args.n_epochs < 1 or min(args.hidden_dim, args.depth, args.lfeq_num_tokens, args.aux_num_tokens, args.decoder_num_tokens,
                               args.max_contrastive_patches) < 1 or args.memory_layer < 0
            or (args.num_tokens is not None and args.num_tokens < 1)
            or (args.num_heads is not None and args.num_heads < 1)
            or (args.batch_size is not None and args.batch_size < 2)):
        parser.error("Require positive budgets/dimensions and a nonnegative memory_layer")
    for arm in auxiliary_arms:
        if ARMS[arm]["family"] != "aux" and args.hidden_dim % arm_settings(args, arm)["num_heads"]:
            parser.error("hidden_dim must be divisible by each selected family's num_heads")
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.aux_lr, args.likelihood_temperature, args.contrastive_temperature, args.mil_temperature)):
        parser.error("Learning rate and temperatures must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in
           (args.weight_decay, args.likelihood_weight, args.diversity_weight)):
        parser.error("Loss weights and weight_decay must be finite and nonnegative")
    if (not 0 < args.sampler_real_ratio < 1 or not 0 <= args.lfeq_dropout < 1
            or not 0 < args.gate_width <= .5 or not 0 <= args.aux_max_weight <= .5):
        parser.error("Invalid sampling ratio, dropout, or fixed gate")
    if len(args.arms) != len(set(args.arms)):
        parser.error("Do not repeat arms within a run")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return 0
    fresh_baseline = not (args.base_run or args.base_checkpoint or args.base_config)
    active_arms = [] if args.baseline_only else args.arms
    try:
        has_auxiliary = any(ARMS[arm]["family"] != "prototype" for arm in active_arms)
        if fresh_baseline:
            source = {"kind": "fresh_pretrained_planned", "config": planned_baseline_config(args)}
            preview_args = copy.copy(args)
            preview_args.batch_size = None
            preview_args.sampler_real_ratio = args.base_sampler_real_ratio
            config, partition, metadata_hashes = preflight(preview_args, source, validate_train_sampling=True)
        else:
            source = resolve_source(args)
            config, partition, metadata_hashes = preflight(args, source, validate_train_sampling=has_auxiliary)
        if has_auxiliary:
            auxiliary_batch = args.batch_size if args.batch_size is not None else config["train_batchSize"]
            validate_sampling(auxiliary_batch, args.sampler_real_ratio,
                              any(ARMS[arm].get("likelihood", "off") != "off" for arm in active_arms))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.preflight:
        print(json.dumps({"status": "OK", "base_sha256": source.get("base_sha256"),
                          "base_training": fresh_baseline,
                          "dataset_sha256": metadata_hashes,
                          "note": "source/metadata only; images, checkpoint loading, and ML runtime not checked"}, indent=2))
        return 0

    import torch
    import transformers
    import experiment_utils as utilities
    from e0924_export import strict_loader
    from run_e0924 import source_hashes

    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = args.output_dir / f"seed{args.seed}_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    for path in ("experiments/run_e1010.py", "experiments/run_e1001.py", "training/detectors/e1010_tokens.py",
                 "experiments/run_g18_lfeq.py", "training/detectors/effort_detector_lfeq.py",
                 "training/detectors/lfeq_module.py", "experiments/e1010_prototypes.py",
                 "experiments/e1010_baseline.py"):
        hashes[path] = file_sha256(ROOT / path)
    if (ROOT / "experiments/E1010_README.md").is_file():
        hashes["experiments/E1010_README.md"] = file_sha256(ROOT / "experiments/E1010_README.md")
    manifest = {"status": "RUNNING", "protocol": plan(args), "source": source,
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "dataset_sha256": metadata_hashes, "source_sha256": hashes,
                "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                "budget": {"epochs": args.n_epochs, "aux_lr": args.aux_lr, "weight_decay": args.weight_decay,
                           "sampler_real_ratio": args.sampler_real_ratio, "train_batch_size": config["train_batchSize"]},
                "scope": "original B0 training/reuse phase, then frozen ViT/LoRA/CLS/head for all auxiliary stages"}
    results = {"experiment": "E1010", "G0": {"status": "NOT_RUN"},
               "arms": {arm: {"status": "NOT_RUN", "settings": arm_settings(args, arm)} for arm in active_arms}}
    write_json(root / "manifest.json", manifest)
    write_json(root / "all_results.json", results)
    write_json(root / "runtime_config.json", config)
    write_json(root / "calibration_partition.json", partition)
    baseline_training = None
    if fresh_baseline:
        from e1010_baseline import prepare_baseline

        try:
            print("E1010/G0: train fresh B0 using E0924/G26_B0 protocol", flush=True)
            baseline_training = prepare_baseline(args, root / "G0/training", utilities)
            manifest["baseline_training"] = baseline_training
            if baseline_training["status"] != "OK":
                manifest.update(status="BASE_TRAIN_FAILED" if baseline_training["status"] == "TRAIN_FAILED" else "BASE_EVAL_FAILED")
                results["G0"] = {"status": "FAILED", "baseline_training": baseline_training}
                write_json(root / "G0/result.json", results["G0"])
                write_json(root / "manifest.json", manifest)
                write_json(root / "all_results.json", results)
                return 1
            direct = copy.copy(args)
            direct.base_run = None
            direct.base_checkpoint = Path(baseline_training["ckpt"])
            direct.base_config = Path(baseline_training["config_path"])
            source = frozen.resolve_source(direct)
            config, partition, metadata_hashes = preflight(args, source, validate_train_sampling=has_auxiliary)
            if has_auxiliary:
                validate_sampling(config["train_batchSize"], args.sampler_real_ratio,
                                  any(ARMS[arm].get("likelihood", "off") != "off" for arm in active_arms))
            manifest.update(source=source, dataset_sha256=metadata_hashes)
            manifest["budget"]["train_batch_size"] = config["train_batchSize"]
            write_json(root / "runtime_config.json", config)
            write_json(root / "calibration_partition.json", partition)
            write_json(root / "manifest.json", manifest)
        except Exception as exc:
            manifest.update(status="BASE_PREPARATION_FAILED", error=f"{type(exc).__name__}: {exc}")
            results["G0"] = {"status": "FAILED", "baseline_training": baseline_training, "error": manifest["error"]}
            write_json(root / "manifest.json", manifest)
            write_json(root / "all_results.json", results)
            return 1
    try:
        base = utilities.load_model(config, source["checkpoint"])
    except Exception as exc:
        manifest.update(status="BASE_LOAD_FAILED", error=f"{type(exc).__name__}: {exc}")
        results["G0"] = {"status": "FAILED", "error": manifest["error"]}
        write_json(root / "manifest.json", manifest)
        write_json(root / "all_results.json", results)
        return 1
    base.requires_grad_(False)
    base.eval()
    state_before = state_sha256(base)
    manifest["base_state_sha256_before"] = state_before
    write_json(root / "manifest.json", manifest)
    print(f"E1010 frozen B0: {source['checkpoint']}\nResults: {root}", flush=True)

    def loader(dataset):
        seed_everything(args.seed)
        original = utilities.get_data_loader(config, dataset).dataset
        frames = strict_loader(original, config["test_batchSize"])
        frames.source_dataset = original
        return frames

    validation_loader = None
    aborted = False
    try:
        validation_loader = loader(VAL_DS)
        references, baseline_metrics = {}, {}
        baseline_folder = root / "G0"
        baseline_folder.mkdir(exist_ok=True)
        for dataset in TEST_DS:
            frames = validation_loader if dataset == VAL_DS else loader(dataset)
            try:
                values = export_scores(base, frames, auxiliary_enabled=False)
            finally:
                if frames is not validation_loader:
                    close_loader(frames)
            references[dataset] = values
            save_npz(baseline_folder / f"{dataset}.npz", values)
            baseline_metrics[dataset] = readout_metrics(values, values["cls_prob"])
        results["G0"] = {"status": "OK", "reused_checkpoint": source["checkpoint"], "datasets": baseline_metrics,
                         "independent_mean": independent_mean({ds: {"B0": value} for ds, value in baseline_metrics.items()}, "B0")}
        if baseline_training is not None:
            results["G0"]["baseline_training"] = baseline_training
        write_json(baseline_folder / "result.json", results["G0"])
        write_json(root / "all_results.json", results)
        prototype_data = None
        prototype_setup_error = None
        prototype_methods = [ARMS[arm]["method"] for arm in active_arms if ARMS[arm]["family"] == "prototype"]
        for arm in active_arms:
            folder = root / arm
            folder.mkdir()
            seed_everything(args.seed)
            result = {"status": "TRAIN_FAILED", "settings": arm_settings(args, arm),
                      "base_sha256": source["base_sha256"]}
            print(f"E1010/{arm}: {result['settings']}", flush=True)
            try:
                is_prototype = ARMS[arm]["family"] == "prototype"
                if is_prototype:
                    import e1010_prototypes as prototypes

                    result["training"] = False
                    if prototype_setup_error is not None:
                        raise ValueError(f"Shared prototype construction failed: {prototype_setup_error}")
                    if prototype_data is None:
                        try:
                            if max(args.prototype_top_k, args.prototype_pool_top_k) > base.backbone.embeddings.num_patches:
                                raise ValueError("Prototype selection/readout K exceeds B0 patch count")
                            records, selection = prototype_records(config, args)
                            original = utilities.get_data_loader(config, "FaceForensics++").dataset
                            training_frames = strict_loader(original, config["test_batchSize"], records)
                            training_frames.source_dataset = original
                            try:
                                artifacts, audit = prototypes.build_prototypes(
                                    base, training_frames, prototype_methods,
                                    top_k=args.prototype_top_k, max_frames_per_class=args.prototype_frames_per_class,
                                    seed=args.seed, occlusion_batch_size=args.occlusion_batch_size, fill=args.occlusion_fill)
                            finally:
                                close_loader(training_frames)
                            if state_sha256(base) != state_before:
                                raise RuntimeError("B0 isolation failed: baseline changed during prototype construction")
                            write_json(root / "G4_selection.json", selection)
                            prototype_data = artifacts, audit, selection
                        except Exception as exc:
                            prototype_setup_error = f"{type(exc).__name__}: {exc}"
                            raise
                    artifacts, audit, selection = prototype_data
                    method = ARMS[arm]["method"]
                    artifact = {**artifacts[method], "base_sha256": source["base_sha256"],
                                "selection_sha256": file_sha256(root / "G4_selection.json"),
                                "readout_settings": result["settings"]}
                    prototypes.validate_artifact(artifact, source["base_sha256"],
                                                 expected_method=method, expected_settings=artifacts[method]["settings"])
                    checkpoint = folder / "prototype.pth"
                    temporary = checkpoint.with_suffix(".pth.tmp")
                    torch.save(artifact, temporary)
                    temporary.replace(checkpoint)
                    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    prototypes.validate_artifact(artifact, source["base_sha256"],
                                                 expected_method=method, expected_settings=artifacts[method]["settings"])
                    if artifact["readout_settings"] != result["settings"]:
                        raise ValueError("Prototype readout settings changed during reload")
                    save_npz(folder / "selection.npz", audit[method])
                    result.update(status="EVAL_FAILED", checkpoint=str(checkpoint), datasets={},
                                  prototype_counts=artifact["counts"], selection=selection,
                                  attribution=artifact["selectionmetadata"],
                                  selection_sha256=artifact["selection_sha256"],
                                  checkpoint_sha256=file_sha256(checkpoint))
                    readouts = (("B0", "cls_prob"), ("prototype", "prototype_prob"), ("gated", "gated_prob"))
                else:
                    model = core_module().FrozenForgerySidecar(base, **result["settings"]).to(utilities.DEVICE)
                    checkpoint = folder / "auxiliary_best.pth"
                    training_loader = make_train_loader(config, args.sampler_real_ratio)
                    try:
                        history = train_auxiliary(model, training_loader, validation_loader, references[VAL_DS],
                                                  args, checkpoint, source["base_sha256"], state_before)
                    finally:
                        close_loader(training_loader)
                    result.update(status="EVAL_FAILED", checkpoint=str(checkpoint),
                                  selected_epoch=max(history, key=lambda row: row["validation"]["frame_auc"])["epoch"],
                                  datasets={})
                    readouts = (("B0", "cls_prob"), ("evidence", "evidence_prob"), ("gated", "gated_prob"))
                if state_sha256(base) != state_before:
                    raise RuntimeError("B0 isolation failed: baseline state changed during auxiliary training")
                result["base_state_unchanged"] = True
                exports = folder / "exports"
                exports.mkdir()
                for dataset in TEST_DS:
                    frames = validation_loader if dataset == VAL_DS else loader(dataset)
                    try:
                        values = (export_prototype_scores(base, frames, artifact, args, source["base_sha256"])
                                  if is_prototype else export_scores(model, frames))
                    finally:
                        if frames is not validation_loader:
                            close_loader(frames)
                    verify_paired_base(references[dataset], values)
                    save_npz(exports / f"{dataset}.npz", values)
                    result["datasets"][dataset] = {name: readout_metrics(values, values[key]) for name, key in readouts}
                    if state_sha256(base) != state_before:
                        raise RuntimeError("B0 isolation failed: baseline changed during score export")
                result.update(status="OK", base_scores_bitwise_identical=True,
                              independent_mean={name: independent_mean(result["datasets"], name)
                                                for name, _ in readouts},
                              export_sha256={path.name: file_sha256(path) for path in exports.glob("*.npz")})
            except Exception as exc:
                result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                aborted = (isinstance(exc, RuntimeError) and "B0 isolation" in str(exc)) or state_sha256(base) != state_before
            finally:
                write_json(folder / "result.json", result)
                results["arms"][arm] = result
                write_json(root / "all_results.json", results)
            print(f"E1010/{arm}: {result['status']}", flush=True)
            if aborted:
                break
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if results["G0"]["status"] != "OK":
            results["G0"] = {"status": "FAILED", "error": manifest["error"]}
    finally:
        if validation_loader is not None:
            close_loader(validation_loader)
        manifest["base_state_sha256_after"] = state_sha256(base)
        manifest["base_state_unchanged"] = manifest["base_state_sha256_after"] == state_before
        manifest["base_file_unchanged"] = file_sha256(source["checkpoint"]) == source["base_sha256"]
        ok = (results["G0"]["status"] == "OK" and manifest["base_state_unchanged"] and manifest["base_file_unchanged"]
              and all(result["status"] == "OK" for result in results["arms"].values()) and not manifest.get("error"))
        manifest["status"] = "OK" if ok else "FAILED"
        write_json(root / "manifest.json", manifest)
        write_json(root / "all_results.json", results)
    print(f"Results: {root / 'all_results.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

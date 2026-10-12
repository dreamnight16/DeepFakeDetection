"""E1012: independently initialized joint training under the historical protocol."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import run_e1010 as diagnostics
import run_g25 as baseline


ROOT = Path(__file__).resolve().parents[1]
TEST_DS = baseline.TEST_DS
INDEPENDENT_DS = [ds for ds in TEST_DS if ds not in (baseline.VAL_DS, baseline.TRAIN_DS)]
ARMS = {"G0": {"family": "baseline"}}
for name, settings in diagnostics.ARMS.items():
    if name.startswith("G2_"):
        for variant, mode in (("G25", "joint"), ("G25V2", "isolated")):
            ARMS[f"G2_{variant}_{name[3:]}"] = {"source_arm": name, "aux_grad_mode": mode, **settings}
    else:
        ARMS[name] = {"source_arm": name, "aux_grad_mode": "joint", **settings}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1012")
    parser.add_argument("--n_epochs", type=int, default=10, help="Historical epoch 0..N inclusive (10 means 11 passes)")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--sampler_real_ratio", type=float, default=.3)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_tokens", type=int)
    parser.add_argument("--lfeq_num_tokens", type=int, default=8)
    parser.add_argument("--aux_num_tokens", type=int, default=4)
    parser.add_argument("--decoder_num_tokens", type=int, default=4)
    parser.add_argument("--memory_layer", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lfeq_dropout", type=float, default=.1)
    parser.add_argument("--likelihood_temperature", type=float, default=.1)
    parser.add_argument("--likelihood_weight", type=float, default=.14)
    parser.add_argument("--contrastive_temperature", type=float, default=.1)
    parser.add_argument("--max_contrastive_patches", type=int, default=16)
    parser.add_argument("--mil_temperature", type=float, default=.5)
    parser.add_argument("--gate_width", type=float, default=.2)
    parser.add_argument("--aux_max_weight", type=float, default=.5)
    parser.add_argument("--diversity_weight", type=float, default=.01)
    parser.add_argument("--prototype_top_k", type=int, default=16)
    parser.add_argument("--prototype_pool_top_k", type=int, default=16)
    parser.add_argument("--prototype_temperature", type=float, default=.1)
    parser.add_argument("--prototype_frames_per_class", type=int, default=32)
    parser.add_argument("--occlusion_batch_size", type=int, default=16)
    parser.add_argument("--occlusion_fill", choices=["mean"], default="mean")
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root")
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Metadata/config checks only; no image or model execution")
    return parser


def arm_settings(args, arm):
    metadata = ARMS[arm]
    if arm == "G0":
        return {"family": "baseline", "model_name": "effort"}
    settings = diagnostics.arm_settings(args, metadata["source_arm"])
    if metadata["family"] != "prototype":
        settings["aux_grad_mode"] = metadata["aux_grad_mode"]
    return settings


def active_arms(args):
    return ["G0"] + ([] if args.baseline_only else [arm for arm in args.arms if arm != "G0"])


def plan(args):
    return {"experiment": "E1012", "default_arms": list(ARMS),
            "arms": {arm: arm_settings(args, arm) for arm in active_arms(args)},
            "initialization": "fresh CLIP pretrained independently per trained arm; no detector warm start",
            "training": {"optimizer": "Adam", "lr": .0002, "weight_decay": .0005,
                         "nEpochs": args.n_epochs, "training_passes": args.n_epochs + 1,
                         "sampler_real_ratio": args.sampler_real_ratio,
                         "batch_size": args.batch_size or 32, "augmentation": "original B0 augmentation",
                         "selection": "CDF-v2 primary frame AUC at historical trainer validation intervals"},
            "primary_readout": {"G1": "LFEQ L1/L2/L3", "G2": "0.5 CLS + 0.5 max evidence",
                                "G3": "fixed uncertainty gate", "G4": "fixed prototypes from this run's G0"},
            "metric_protocol": "strict single-frame exports; full video path primary, legacy basename separately",
            "independent_datasets": INDEPENDENT_DS,
            "excluded_from_independent_mean": [baseline.VAL_DS, baseline.TRAIN_DS],
            "correction_reference": "independent_G0, paired by labels/path/video_id",
            "G4": "E1010 prototype diagnostics only; no second B0 training or auxiliary training"}


def build_arm_config(args, arm, folder, build_config, for_training):
    config = baseline.build_arm_config(args, "B0", folder, build_config, for_training)
    config.update(model_name="effort" if arm == "G0" else "effort_e1012",
                  e0924_protocol=True, multi_crop=False, use_adaptive_threshold=False,
                  use_data_augmentation=for_training)
    if arm != "G0":
        config["e1012_settings"] = arm_settings(args, arm)
    if args.batch_size is not None:
        config["train_batchSize"] = args.batch_size
    if args.dataset_json_folder is not None:
        config["dataset_json_folder"] = str(args.dataset_json_folder.resolve())
    if args.rgb_root is not None:
        config["rgb_root_override"] = args.rgb_root
    return config


def paired_metrics(values, probabilities, reference):
    import numpy as np

    for key in ("labels", "path", "video_id"):
        if not np.array_equal(values[key], reference[key]):
            raise ValueError(f"Independent B0 pairing changed {key}")
    paired = {**values, "cls_prob": reference["cls_prob"]}
    report = diagnostics.readout_metrics(paired, probabilities)
    report["reference"] = "independent_G0"
    return report


def export_checkpoint(config, checkpoint, utilities):
    import numpy as np
    import torch
    from e0924_export import strict_loader

    model = utilities.load_model(config, checkpoint)
    device = next(model.parameters()).device
    exported = {}
    try:
        for dataset in TEST_DS:
            diagnostics.seed_everything(config["manualSeed"])
            original = utilities.get_data_loader(config, dataset).dataset
            frames = strict_loader(original, config["test_batchSize"])
            frames.source_dataset = original
            arrays = {key: [] for key in ("labels", "path", "video_id", "cls_prob", "global_log_odds", "primary_prob")}
            if config["model_name"] != "effort":
                arrays["evidence_prob"] = []
            try:
                with torch.no_grad():
                    for batch in frames:
                        output = model({"image": batch["image"].to(device)}, inference=True)
                        values = output.get("e1012")
                        cls_prob = values["cls_prob"] if values is not None else output["prob"]
                        logits = values["global_logits"] if values is not None else output["cls"]
                        arrays["labels"].append(batch["label"].cpu().numpy())
                        arrays["path"].append(np.asarray(batch["path"]))
                        arrays["video_id"].append(np.asarray(batch["video_id"]))
                        arrays["cls_prob"].append(cls_prob.cpu().numpy())
                        arrays["primary_prob"].append(output["prob"].cpu().numpy())
                        arrays["global_log_odds"].append((logits[:, 1] - logits[:, 0]).cpu().numpy())
                        if values is not None:
                            arrays["evidence_prob"].append(values["evidence_prob"].cpu().numpy())
                if not arrays["labels"]:
                    raise ValueError(f"Empty E1012 export: {dataset}")
                result = {key: np.concatenate(parts) for key, parts in arrays.items()}
                if any(not np.isfinite(value).all() for key, value in result.items() if key not in ("path", "video_id")):
                    raise ValueError(f"Nonfinite E1012 export: {dataset}")
                exported[dataset] = result
            finally:
                diagnostics.close_loader(frames)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return exported


def run_one(args, arm, folder, utilities, reference=None):
    folder.mkdir(parents=True, exist_ok=False)
    result = {"arm": arm, "status": "TRAIN_FAILED", "settings": arm_settings(args, arm),
              "initialization": "fresh CLIP pretrained; no detector warm start"}
    exported = None
    try:
        training = build_arm_config(args, arm, folder, utilities.build_config, True)
        evaluation = build_arm_config(args, arm, folder, utilities.build_config, False)
        config_path = folder / "train_config.json"
        diagnostics.write_json(config_path, training)
        diagnostics.write_json(folder / "eval_config.json", evaluation)
        checkpoint = utilities.train_model(training, baseline.TRAIN_DS, baseline.VAL_DS)
        if checkpoint is None:
            result["error"] = "Training did not produce a selected checkpoint"
            return result, None
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
        result.update(status="EVAL_FAILED", ckpt=str(Path(checkpoint).resolve()),
                      config_path=str(config_path.resolve()), checkpoint_sha256=diagnostics.file_sha256(checkpoint))
        utilities.seed_evaluation(args.seed)
        legacy = utilities.evaluate_model(evaluation, checkpoint, TEST_DS, baseline.TRAIN_DS,
                                          str(folder), f"E1012/{arm}")
        baseline.validate_metrics(legacy)
        result["historical_evaluation"] = legacy
        if legacy["status"] != "OK":
            raise ValueError("Historical evaluation is missing valid dataset video AUCs")
        exported = export_checkpoint(evaluation, checkpoint, utilities)
        if set(exported) != set(TEST_DS):
            raise ValueError("Strict export is missing requested datasets")
        reference = exported if reference is None and arm == "G0" else reference
        if reference is None:
            raise ValueError("Joint arms require this run's independent G0 exports")
        global_readout = "decision" if ARMS[arm]["family"] == "lfeq" else "CLS"
        readouts = [("B0", "cls_prob")] if arm == "G0" else [
            (global_readout, "cls_prob"), ("evidence", "evidence_prob"), ("primary", "primary_prob")]
        result["datasets"] = {}
        destination = folder / "exports"
        destination.mkdir()
        for dataset, values in exported.items():
            result["datasets"][dataset] = {name: paired_metrics(values, values[key], reference[dataset])
                                           for name, key in readouts}
            values["baseline_prob"] = reference[dataset]["cls_prob"].copy()
            diagnostics.save_npz(destination / f"{dataset}.npz", values)
        result.update(status="OK", independent_mean={name: diagnostics.independent_mean(result["datasets"], name)
                                                     for name, _ in readouts},
                      export_sha256={p.name: diagnostics.file_sha256(p) for p in destination.glob("*.npz")})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        exported = None
    finally:
        diagnostics.write_json(folder / "result.json", result)
    return result, exported


def prototype_command(args, root, result, arms):
    command = [sys.executable, str(ROOT / "experiments/run_e1010.py"),
               "--base_checkpoint", result["ckpt"], "--base_config", result["config_path"],
               "--output_dir", str(root / "G4_diagnostics"), "--seed", str(args.seed)]
    for key in ("prototype_top_k", "prototype_pool_top_k", "prototype_temperature", "prototype_frames_per_class",
                "occlusion_batch_size", "occlusion_fill", "gate_width", "aux_max_weight"):
        command += [f"--{key}", str(getattr(args, key))]
    for key in ("dataset_json_folder", "rgb_root", "clip_pretrained_path"):
        if getattr(args, key) is not None:
            command += [f"--{key}", str(getattr(args, key))]
    return command + ["--arms", *arms]


def run_prototypes(args, root, result, arms):
    folder = root / "G4_diagnostics"
    folder.mkdir()
    with (folder / "run.log").open("w") as handle:
        process = subprocess.run(prototype_command(args, root, result, arms), cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    paths = list(folder.glob("seed*/all_results.json"))
    if len(paths) != 1:
        raise ValueError(f"G4 subprocess exit {process.returncode}: expected one result record; see {folder / 'run.log'}")
    payload = json.loads(paths[0].read_text())
    records = {}
    for arm in arms:
        record = payload.get("arms", {}).get(arm)
        if not isinstance(record, dict) or record.get("status") not in ("OK", "FAILED", "EVAL_FAILED"):
            raise ValueError(f"G4 has no final result for {arm}")
        records[arm] = {**record, "diagnostic_run": str(paths[0].parent.resolve()),
                        "source": "fixed prototypes on this E1012 run's independent G0", "training": False}
    if process.returncode != 0 and all(record["status"] == "OK" for record in records.values()):
        raise RuntimeError(f"G4 subprocess failed after arm evaluation; see {folder / 'run.log'}")
    return records


def validate_arguments(parser, args):
    if len(args.arms) != len(set(args.arms)):
        parser.error("Do not repeat arms")
    if args.n_epochs < 0 or args.seed < 0 or not 0 < args.sampler_real_ratio < 1:
        parser.error("Invalid epochs, seed, or sampling ratio")
    counts = (args.lfeq_num_tokens, args.aux_num_tokens, args.decoder_num_tokens, args.hidden_dim, args.depth,
              args.max_contrastive_patches, args.prototype_top_k, args.prototype_pool_top_k,
              args.prototype_frames_per_class, args.occlusion_batch_size)
    if min(counts) < 1 or (args.num_tokens is not None and args.num_tokens < 1) or (args.num_heads is not None and args.num_heads < 1):
        parser.error("Token counts, dimensions, heads and prototype budgets must be positive")
    if not 0 <= args.memory_layer < 24 or not 0 <= args.lfeq_dropout < 1:
        parser.error("Invalid insertion layer or dropout")
    if any(not math.isfinite(value) or value <= 0 for value in (
            args.likelihood_temperature, args.contrastive_temperature, args.mil_temperature, args.prototype_temperature)):
        parser.error("Temperatures must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in (args.likelihood_weight, args.diversity_weight)):
        parser.error("Loss weights must be finite and nonnegative")
    if not 0 < args.gate_width <= .5 or not 0 <= args.aux_max_weight <= .5:
        parser.error("Invalid fixed gate")
    try:
        diagnostics.validate_sampling(args.batch_size or 32, args.sampler_real_ratio,
                                      any(ARMS[arm].get("likelihood", "off") != "off" for arm in active_arms(args)))
    except ValueError as exc:
        parser.error(str(exc))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return 0
    if args.preflight:
        from types import SimpleNamespace

        preview = SimpleNamespace(**vars(args), base_batch_size=args.batch_size)
        config = diagnostics.planned_baseline_config(preview)
        _, _, hashes = diagnostics.preflight(args, {"config": config}, validate_train_sampling=False)
        print(json.dumps({"status": "OK", "dataset_sha256": hashes, "protocol": plan(args),
                          "note": "Metadata only; images, model loading and GPU execution were not checked"}, indent=2))
        return 0
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = args.output_dir.resolve() / f"seed{args.seed}_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = {"status": "RUNNING", "protocol": plan(args), "arguments": arguments, "python": sys.version}
    results = {"experiment": "E1012", "G0": {"status": "NOT_RUN"}, "arms": {}}
    try:
        import torch
        import transformers
        import experiment_utils as utilities
        from types import SimpleNamespace
        from run_e0924 import source_hashes

        adapter = SimpleNamespace(build_config=utilities.build_config, train_model=utilities.train_model,
                                  evaluate_model=utilities.evaluate_model, seed_evaluation=diagnostics.seed_everything,
                                  load_model=utilities.load_model, get_data_loader=utilities.get_data_loader)
        paths = ["experiments/run_e1012.py", "experiments/run_e1010.py", "experiments/run_e1001.py",
                 "experiments/e1010_baseline.py", "experiments/e1010_prototypes.py",
                 "training/detectors/e1012_tokens.py", "training/detectors/effort_detector_e1012.py"]
        preview = build_arm_config(args, "G0", root / "G0", adapter.build_config, True)
        _, _, metadata_hashes = diagnostics.preflight(args, {"config": preview}, validate_train_sampling=False)
        manifest.update(source_sha256={**source_hashes(), **{p: diagnostics.file_sha256(ROOT / p) for p in paths}},
                        dataset_sha256=metadata_hashes,
                        torch=torch.__version__, transformers=transformers.__version__)
        diagnostics.write_json(root / "manifest.json", manifest)
        result, reference = run_one(args, "G0", root / "G0", adapter)
        results["G0"] = result
        diagnostics.write_json(root / "all_results.json", results)
        if result["status"] != "OK":
            raise RuntimeError("Independent G0 training/evaluation failed; joint comparisons cannot run")
        prototype_arms = []
        for arm in active_arms(args)[1:]:
            if ARMS[arm]["family"] == "prototype":
                prototype_arms.append(arm)
                continue
            print(f"E1012/{arm}: {arm_settings(args, arm)}", flush=True)
            result, _ = run_one(args, arm, root / arm, adapter, reference)
            results["arms"][arm] = result
            diagnostics.write_json(root / "all_results.json", results)
            print(f"E1012/{arm}: {result['status']}", flush=True)
        if prototype_arms:
            try:
                results["arms"].update(run_prototypes(args, root, results["G0"], prototype_arms))
            except Exception as exc:
                results["arms"].update({arm: {"status": "FAILED", "training": False,
                                            "error": f"{type(exc).__name__}: {exc}"} for arm in prototype_arms})
        successful = results["G0"]["status"] == "OK" and all(row["status"] == "OK" for row in results["arms"].values())
        manifest["status"] = "OK" if successful else "FAILED"
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        diagnostics.write_json(root / "manifest.json", manifest)
        diagnostics.write_json(root / "all_results.json", results)
    print(f"E1012/{manifest['status']}: {root}", flush=True)
    return 0 if manifest["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""E1021: fresh B0 and real/fake LoRA branches with class-specific gradients."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import run_g25 as baseline


ROOT = Path(__file__).resolve().parents[1]
TEST_DS = baseline.TEST_DS
INDEPENDENT_DS = [dataset for dataset in baseline.CROSS_DS if dataset != baseline.VAL_DS]
ARMS = {"B0": None, "R2F2": (2, 2), "R4F4": (4, 4), "R2F6": (2, 6), "R6F2": (6, 2)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=[*ARMS, "CUSTOM"], default=list(ARMS))
    parser.add_argument("--real_rank", type=int, help="Real-branch rank for --arms CUSTOM")
    parser.add_argument("--fake_rank", type=int, help="Fake-branch rank for --arms CUSTOM")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1021")
    parser.add_argument("--n_epochs", type=int, default=10, help="Epochs 0..N inclusive")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--sampler_real_ratio", type=float, default=.3)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root")
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def arm_settings(args, arm):
    if arm == "B0":
        return {"model_name": "effort"}
    real_rank, fake_rank = (args.real_rank, args.fake_rank) if arm == "CUSTOM" else ARMS[arm]
    return {"model_name": "effort_e1021", "e1021_real_rank": real_rank, "e1021_fake_rank": fake_rank}


def validate_arguments(parser, args):
    if len(set(args.arms)) != len(args.arms):
        parser.error("Do not repeat arms")
    if "CUSTOM" in args.arms:
        if args.real_rank is None or args.fake_rank is None or min(args.real_rank, args.fake_rank) < 1:
            parser.error("CUSTOM requires positive --real_rank and --fake_rank")
    elif args.real_rank is not None or args.fake_rank is not None:
        parser.error("Use --arms CUSTOM to specify real/fake ranks")
    batch_size = args.batch_size if args.batch_size is not None else 32
    if args.n_epochs < 0 or args.seed < 0 or batch_size < 2 or not 0 < args.sampler_real_ratio < 1:
        parser.error("Require nonnegative epochs/seed, batch_size >= 2, and sampling ratio in (0, 1)")
    sampled_batch_size = 2 * max(1, batch_size // 2)
    if max(1, round(sampled_batch_size * args.sampler_real_ratio)) >= sampled_batch_size:
        parser.error("The balanced sampler must include both real and fake samples")


def plan(args):
    return {"experiment": "E1021", "arms": {arm: arm_settings(args, arm) for arm in args.arms},
            "initialization": "fresh CLIP pretrained independently per arm; no detector warm start",
            "lora_scaling": 4.0,
            "training": {"optimizer": "Adam", "lr": .0002, "weight_decay": .0005,
                         "seed": args.seed, "nEpochs": args.n_epochs, "training_passes": args.n_epochs + 1,
                         "batch_size": args.batch_size if args.batch_size is not None else 32,
                         "sampler_real_ratio": args.sampler_real_ratio,
                         "dataset": baseline.TRAIN_DS, "compression": "c23", "frames_per_video": 8,
                         "baseline_objective": "mean(batch CE)",
                         "dual_objective": "mean(real CE) + mean(fake CE)",
                         "checkpoint_selection": "Celeb-DF-v2 frame AUC at historical validation intervals"},
            "trainable": {"B0": "original rank-4 LoRA and full shared head",
                          "dual": "real/fake LoRA and full shared head; class-specific branch gradients",
                          "frozen": "original CLIP weights and CLS embedding"},
            "metric_protocol": "strict single-frame export; video_auc uses full parent path; legacy basename separate",
            "datasets": TEST_DS, "independent_datasets": INDEPENDENT_DS,
            "excluded_from_independent_mean": [baseline.VAL_DS, baseline.TRAIN_DS]}


def build_arm_config(args, arm, folder, build_config, for_training):
    config = baseline.build_arm_config(args, "B0", folder, build_config, for_training)
    config.update(arm_settings(args, arm))
    config.update(e0924_protocol=True, multi_crop=False, use_adaptive_threshold=False,
                  use_data_augmentation=for_training)
    if args.batch_size is not None:
        config["train_batchSize"] = args.batch_size
    if args.dataset_json_folder is not None:
        config["dataset_json_folder"] = str(args.dataset_json_folder.resolve())
    if args.rgb_root is not None:
        config["rgb_root_override"] = args.rgb_root
    return config


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def export_checkpoint(config, checkpoint, utilities):
    import numpy as np
    import torch
    from e0924_export import strict_loader

    model = utilities.load_model(config, checkpoint)
    model.eval()
    device = next(model.parameters()).device
    exported = {}
    try:
        for dataset in TEST_DS:
            seed_everything(config["manualSeed"])
            original = utilities.get_data_loader(config, dataset).dataset
            frames = strict_loader(original, config["test_batchSize"])
            arrays = {key: [] for key in ("labels", "path", "video_id", "cls_prob", "global_log_odds")}
            try:
                with torch.no_grad():
                    for batch in frames:
                        output = model({"image": batch["image"].to(device)}, inference=True)
                        logits = output["cls"]
                        arrays["labels"].append(batch["label"].cpu().numpy())
                        arrays["path"].append(np.asarray(batch["path"]))
                        arrays["video_id"].append(np.asarray(batch["video_id"]))
                        arrays["cls_prob"].append(output["prob"].cpu().numpy())
                        arrays["global_log_odds"].append((logits[:, 1] - logits[:, 0]).cpu().numpy())
                if not arrays["labels"]:
                    raise ValueError(f"Empty strict export: {dataset}")
                values = {key: np.concatenate(parts) for key, parts in arrays.items()}
                if any(not np.isfinite(value).all() for key, value in values.items()
                       if key not in ("path", "video_id")):
                    raise ValueError(f"Nonfinite strict export: {dataset}")
                exported[dataset] = values
            finally:
                env = getattr(original, "env", None)
                if env is not None:
                    env.close()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return exported


def run_one(args, arm, folder, utilities):
    from e0924_export import file_sha256
    from run_e1001 import readout_metrics
    import numpy as np

    folder.mkdir(parents=True, exist_ok=False)
    result = {"arm": arm, "settings": arm_settings(args, arm), "status": "TRAIN_FAILED",
              "initialization": "fresh CLIP pretrained; no detector warm start"}
    try:
        training = build_arm_config(args, arm, folder, utilities.build_config, True)
        evaluation = build_arm_config(args, arm, folder, utilities.build_config, False)
        baseline.write_json(folder / "train_config.json", training)
        baseline.write_json(folder / "eval_config.json", evaluation)
        checkpoint = utilities.train_model(training, baseline.TRAIN_DS, baseline.VAL_DS)
        if checkpoint is None:
            raise RuntimeError("Training did not produce a selected checkpoint")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint}")
        result.update(status="EVAL_FAILED", ckpt=str(Path(checkpoint).resolve()),
                      checkpoint_sha256=file_sha256(checkpoint))
        exported = export_checkpoint(evaluation, checkpoint, utilities)
        if set(exported) != set(TEST_DS):
            raise ValueError("Strict export is missing requested datasets")
        destination = folder / "exports"
        destination.mkdir()
        result["datasets"] = {}
        for dataset in TEST_DS:
            values = exported[dataset]
            metrics = readout_metrics(values, values["cls_prob"])
            result["datasets"][dataset] = {key: value for key, value in metrics.items()
                                           if key not in ("corrected_frames", "harmed_frames")}
            np.savez_compressed(destination / f"{dataset}.npz", **values)
        metrics = result["datasets"]
        result.update(status="OK", metric_protocol="video_auc=full-path video; video_auc_legacy=basename",
                      video_auc_seven_mean_fullpath=sum(metrics[d]["video_auc"] for d in baseline.CROSS_DS) / len(baseline.CROSS_DS),
                      independent_video_auc_mean_fullpath=sum(metrics[d]["video_auc"] for d in INDEPENDENT_DS) / len(INDEPENDENT_DS),
                      independent_datasets=INDEPENDENT_DS,
                      excluded_from_independent_mean=[baseline.VAL_DS, baseline.TRAIN_DS])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        baseline.write_json(folder / "result.json", result)
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return 0
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = args.output_dir.resolve() / f"seed{args.seed}_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "RUNNING", "protocol": plan(args), "arguments": vars(args), "python": sys.version}
    results = {"experiment": "E1021", "arms": {arm: {"status": "NOT_RUN"} for arm in args.arms}}
    baseline.write_json(root / "manifest.json", manifest)
    try:
        import experiment_utils as utilities
        import torch
        import transformers

        manifest.update(torch=torch.__version__, transformers=transformers.__version__)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        paths = ["experiments/run_e1021.py", "experiments/run_g25.py", "experiments/experiment_utils.py",
                 "experiments/e0924_export.py", "experiments/run_e1001.py",
                 "training/detectors/effort_detector_e1021.py", "training/detectors/e1021_lora.py",
                 "training/detectors/effort_detector.py", "training/detectors/__init__.py",
                 "training/trainer/trainer_v2.py", "training/train.py",
                 "training/config/detector/effort.yaml", "training/config/train_config.yaml",
                 "training/config/test_config.yaml"]
        manifest.update(git_head=revision.stdout.strip() if revision.returncode == 0 else None,
                        source_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths})
        preview = build_arm_config(args, args.arms[0], root / args.arms[0], utilities.build_config, True)
        metadata = Path(preview["dataset_json_folder"])
        manifest["dataset_sha256"] = {dataset: hashlib.sha256((metadata / f"{dataset}.json").read_bytes()).hexdigest()
                                      for dataset in TEST_DS}
        baseline.write_json(root / "manifest.json", manifest)
        for arm in args.arms:
            print(f"E1021/{arm}: {arm_settings(args, arm)}", flush=True)
            results["arms"][arm] = run_one(args, arm, root / arm, utilities)
            baseline.write_json(root / "all_results.json", results)
            print(f"E1021/{arm}: {results['arms'][arm]['status']}", flush=True)
        manifest["status"] = "OK" if all(row["status"] == "OK" for row in results["arms"].values()) else "FAILED"
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        baseline.write_json(root / "manifest.json", manifest)
        baseline.write_json(root / "all_results.json", results)
    print(f"Results: {root / 'all_results.json'}")
    return 0 if manifest["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())

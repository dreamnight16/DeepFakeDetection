"""E1025: dual-LoRA soft-gradient routing and single-LoRA nnPU ablations."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import run_e1021 as previous


ROOT = previous.ROOT
TEST_DS = previous.TEST_DS
INDEPENDENT_DS = previous.INDEPENDENT_DS
RANKS = {name: ranks for name, ranks in previous.ARMS.items() if ranks is not None}
export_checkpoint = previous.export_checkpoint


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=["B0", "SOFT", "PU"], default=["B0", "SOFT", "PU"],
                        help="Experiment families; SOFT includes beta=0 and beta=1 controls")
    parser.add_argument("--ranks", nargs="+", choices=list(RANKS), default=list(RANKS))
    parser.add_argument("--betas", nargs="+", type=float, default=[0., .25, 1.])
    parser.add_argument("--priors", nargs="+", type=float, default=[.1, .3, .5])
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1025")
    parser.add_argument("--n_epochs", type=int, default=10, help="Epochs 0..N inclusive")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--sampler_real_ratio", type=float, default=.3)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root")
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def validate_arguments(parser, args):
    for name in ("arms", "ranks", "betas", "priors"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"Do not repeat {name}")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in args.betas):
        parser.error("betas must be finite and in [0, 1]")
    if any(not math.isfinite(value) or not 0 < value < 1 for value in args.priors):
        parser.error("priors must be finite and in (0, 1)")
    batch_size = args.batch_size if args.batch_size is not None else 32
    if args.n_epochs < 0 or args.seed < 0 or batch_size < 2 or not 0 < args.sampler_real_ratio < 1:
        parser.error("Require nonnegative epochs/seed, batch_size >= 2, and sampling ratio in (0, 1)")
    sampled_batch_size = 2 * max(1, batch_size // 2)
    if max(1, round(sampled_batch_size * args.sampler_real_ratio)) >= sampled_batch_size:
        parser.error("The balanced sampler must include both real and fake samples")


def arm_settings(args):
    settings = {}
    for family in args.arms:
        if family == "B0":
            settings["B0"] = {"model_name": "effort"}
        elif family == "SOFT":
            for rank in args.ranks:
                real, fake = RANKS[rank]
                for beta in args.betas:
                    tag = str(beta).removesuffix(".0")
                    settings[f"{rank}_B{tag}"] = {"model_name": "effort_e1025_soft", "e1021_real_rank": real,
                                                      "e1021_fake_rank": fake, "e1025_beta": beta}
        else:
            for prior in args.priors:
                settings[f"PU_P{prior}"] = {"model_name": "effort_e1025_pu", "e1025_prior": prior}
    return settings


def plan(args):
    baseline_args = argparse.Namespace(**{**vars(args), "arms": ["B0"]})
    protocol = previous.plan(baseline_args)
    protocol.update(experiment="E1025", arms=arm_settings(args), gradient_clipping=None)
    protocol["trainable"]["dual"] = "real/fake LoRA and full shared head; opposite-class gradients scaled by beta"
    protocol["trainable"]["PU"] = "original rank-4 LoRA and full shared head; nnPU replaces training CE"
    protocol["training"].update(
        soft_routing="real branch: Lreal + beta*Lfake; fake branch: beta*Lreal + Lfake; head: Lreal + Lfake",
        nnpu="prior*mean(softplus(-s_real)) + clamp(mean(softplus(s_gen))-prior*mean(softplus(s_real)), min=0); s=z_real-z_fake",
        nnpu_prior="positive class prior within U; independent of sampler_real_ratio",
        nnpu_validation="CE for validation loss logging, unchanged frame-AUC checkpoint selection")
    return protocol


def build_arm_config(args, arm, folder, build_config, for_training):
    config = previous.build_arm_config(args, "B0", folder, build_config, for_training)
    config.update(arm_settings(args)[arm])
    return config


def run_one(args, arm, folder, utilities):
    from e0924_export import file_sha256
    from run_e1001 import readout_metrics
    import numpy as np

    folder.mkdir(parents=True, exist_ok=False)
    result = {"arm": arm, "settings": arm_settings(args)[arm], "status": "TRAIN_FAILED",
              "initialization": "fresh CLIP pretrained; no detector warm start"}
    try:
        training = build_arm_config(args, arm, folder, utilities.build_config, True)
        evaluation = build_arm_config(args, arm, folder, utilities.build_config, False)
        previous.baseline.write_json(folder / "train_config.json", training)
        previous.baseline.write_json(folder / "eval_config.json", evaluation)
        checkpoint = utilities.train_model(training, previous.baseline.TRAIN_DS, previous.baseline.VAL_DS)
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
                      video_auc_seven_mean_fullpath=sum(metrics[d]["video_auc"] for d in previous.baseline.CROSS_DS) / len(previous.baseline.CROSS_DS),
                      independent_video_auc_mean_fullpath=sum(metrics[d]["video_auc"] for d in INDEPENDENT_DS) / len(INDEPENDENT_DS),
                      independent_datasets=INDEPENDENT_DS,
                      excluded_from_independent_mean=[previous.baseline.VAL_DS, previous.baseline.TRAIN_DS])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        previous.baseline.write_json(folder / "result.json", result)
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
    results = {"experiment": "E1025", "arms": {arm: {"status": "NOT_RUN"} for arm in arm_settings(args)}}
    previous.baseline.write_json(root / "manifest.json", manifest)
    try:
        import experiment_utils as utilities
        import torch
        import transformers

        manifest.update(torch=torch.__version__, transformers=transformers.__version__)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        paths = ["experiments/run_e1025.py", "experiments/run_e1021.py", "experiments/run_g25.py",
                 "experiments/experiment_utils.py", "experiments/e0924_export.py", "experiments/run_e1001.py",
                 "training/detectors/effort_detector_e1025_soft.py", "training/detectors/effort_detector_e1025_pu.py",
                 "training/detectors/e1025_losses.py", "training/detectors/effort_detector_e1021.py",
                 "training/detectors/e1021_lora.py", "training/detectors/effort_detector.py",
                 "training/detectors/__init__.py", "training/trainer/trainer_v2.py", "training/train.py",
                 "training/config/detector/effort.yaml", "training/config/train_config.yaml",
                 "training/config/test_config.yaml"]
        manifest.update(git_head=revision.stdout.strip() if revision.returncode == 0 else None,
                        source_sha256={path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths})
        first_arm = next(iter(results["arms"]))
        preview = build_arm_config(args, first_arm, root / first_arm, utilities.build_config, True)
        metadata = Path(preview["dataset_json_folder"])
        manifest["dataset_sha256"] = {dataset: hashlib.sha256((metadata / f"{dataset}.json").read_bytes()).hexdigest()
                                      for dataset in TEST_DS}
        previous.baseline.write_json(root / "manifest.json", manifest)
        for arm in results["arms"]:
            print(f"E1025/{arm}: {arm_settings(args)[arm]}", flush=True)
            results["arms"][arm] = run_one(args, arm, root / arm, utilities)
            previous.baseline.write_json(root / "all_results.json", results)
            print(f"E1025/{arm}: {results['arms'][arm]['status']}", flush=True)
        manifest["status"] = "OK" if all(row["status"] == "OK" for row in results["arms"].values()) else "FAILED"
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        previous.baseline.write_json(root / "manifest.json", manifest)
        previous.baseline.write_json(root / "all_results.json", results)
    print(f"Results: {root / 'all_results.json'}")
    return 0 if manifest["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())

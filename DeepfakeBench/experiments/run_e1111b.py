"""E1111B: E1025 single-LoRA CE + lambda * nonnegative PU-risk ablation."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import run_e1025 as previous


ROOT = previous.ROOT
BASE = previous.previous


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=float, default=.3)
    parser.add_argument("--lams", type=float, nargs="+", default=[0., .01, .05, .1])
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1111/E1111B")
    parser.add_argument("--n_epochs", type=int, default=10, help="E1025 epochs 0..N inclusive")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--sampler_real_ratio", type=float, default=.3)
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root")
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def validate_arguments(parser, args):
    if not math.isfinite(args.prior) or not 0 < args.prior < 1:
        parser.error("prior must be finite and in (0, 1)")
    if (len(set(args.lams)) != len(args.lams)
            or any(not math.isfinite(x) or x < 0 for x in args.lams)):
        parser.error("lams must be distinct, finite and nonnegative")
    BASE.validate_arguments(parser, argparse.Namespace(**vars(args), arms=["B0"],
                                                     real_rank=None, fake_rank=None))


def arm_settings(args):
    return {f"CE_L0" if lam == 0 else f"PUCE_P{args.prior:g}_L{lam:g}":
            {"model_name": "effort_e1111b", "e1025_prior": args.prior, "e1111_lam": lam}
            for lam in args.lams}


def plan(args):
    protocol = BASE.plan(argparse.Namespace(**vars(args), arms=["B0"]))
    protocol.update(experiment="E1111", branch="E1111B", arms=arm_settings(args),
                    reference_experiment="E1025 single-LoRA B0/PU",
                    objective="CE(logits,labels) + lam*clamp(-mean(logp1[U])+prior*mean(logp1[R]),min=0)",
                    lam_zero="exact batch CE, including its gradients",
                    prior_meaning="real-class prior inside U, independent of sampler_real_ratio",
                    validation_loss="CE; same historical Celeb-DF-v2 frame-AUC checkpoint selection",
                    gradient_clipping=None)
    protocol["trainable"] = {"model": "original rank-4 LoRA and full head",
                            "frozen": "original CLIP weights and CLS embedding"}
    return protocol


def build_arm_config(args, arm, folder, build_config, for_training):
    config = BASE.build_arm_config(args, "B0", folder, build_config, for_training)
    config.update(arm_settings(args)[arm])
    return config


def run_one(args, arm, folder, utilities):
    import numpy as np
    from e0924_export import file_sha256
    from run_e1001 import readout_metrics

    folder.mkdir(parents=True, exist_ok=False)
    result = {"arm": arm, "settings": arm_settings(args)[arm], "status": "TRAIN_FAILED",
              "initialization": "fresh CLIP pretrained; no detector warm start"}
    try:
        training = build_arm_config(args, arm, folder, utilities.build_config, True)
        evaluation = build_arm_config(args, arm, folder, utilities.build_config, False)
        BASE.baseline.write_json(folder / "train_config.json", training)
        BASE.baseline.write_json(folder / "eval_config.json", evaluation)
        checkpoint = utilities.train_model(training, BASE.baseline.TRAIN_DS, BASE.baseline.VAL_DS)
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(f"No selected checkpoint: {checkpoint}")
        result.update(status="EVAL_FAILED", ckpt=str(Path(checkpoint).resolve()),
                      checkpoint_sha256=file_sha256(checkpoint))
        exported = previous.export_checkpoint(evaluation, checkpoint, utilities)
        if set(exported) != set(previous.TEST_DS):
            raise ValueError("Strict export did not return every requested dataset")
        destination = folder / "exports"
        destination.mkdir()
        result["datasets"] = {}
        for dataset in previous.TEST_DS:
            values = exported[dataset]
            metrics = readout_metrics(values, values["cls_prob"])
            result["datasets"][dataset] = {key: value for key, value in metrics.items()
                                           if key not in ("corrected_frames", "harmed_frames")}
            np.savez_compressed(destination / f"{dataset}.npz", **values)
        metrics = result["datasets"]
        result.update(status="OK", metric_protocol="full-path video_id; frame mean",
                      independent_datasets=previous.INDEPENDENT_DS,
                      independent_video_auc_mean_fullpath=sum(
                          metrics[d]["video_auc"] for d in previous.INDEPENDENT_DS) / len(previous.INDEPENDENT_DS),
                      video_auc_seven_mean_fullpath=sum(
                          metrics[d]["video_auc"] for d in BASE.baseline.CROSS_DS) / len(BASE.baseline.CROSS_DS),
                      excluded_from_independent_mean=[BASE.baseline.VAL_DS, BASE.baseline.TRAIN_DS])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        BASE.baseline.write_json(folder / "result.json", result)
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return 0
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir.resolve() / f"seed{args.seed}_{stamp}_{os.getpid()}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "RUNNING", "protocol": plan(args), "arguments": vars(args),
                "python": sys.version}
    results = {"experiment": "E1111", "branch": "E1111B",
               "arms": {arm: {"status": "NOT_RUN"} for arm in arm_settings(args)}}
    try:
        import experiment_utils as utilities
        import torch
        import transformers

        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True, check=False)
        sources = ["experiments/run_e1111b.py", "experiments/run_e1025.py",
                   "experiments/run_e1021.py", "experiments/run_g25.py",
                   "experiments/experiment_utils.py", "experiments/e0924_export.py",
                   "experiments/run_e1001.py", "training/detectors/effort_detector.py",
                   "training/detectors/effort_detector_e1025_pu.py",
                   "training/detectors/e1025_losses.py", "training/detectors/e1111_losses.py",
                   "training/detectors/effort_detector_e1111b.py", "training/detectors/__init__.py",
                   "training/train.py", "training/trainer/trainer_v2.py",
                   "training/dataset/abstract_dataset.py", "training/config/train_config.yaml",
                   "training/config/test_config.yaml", "training/config/detector/effort.yaml"]
        manifest.update(torch=torch.__version__, transformers=transformers.__version__,
                        git_head=revision.stdout.strip() if revision.returncode == 0 else None,
                        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources})
        first = next(iter(results["arms"]))
        preview = build_arm_config(args, first, output / first, utilities.build_config, True)
        metadata = Path(preview["dataset_json_folder"])
        manifest["dataset_sha256"] = {
            dataset: hashlib.sha256((metadata / f"{dataset}.json").read_bytes()).hexdigest()
            for dataset in previous.TEST_DS}
        BASE.baseline.write_json(output / "manifest.json", manifest)
        for arm in results["arms"]:
            print(f"E1111B/{arm}: {arm_settings(args)[arm]}", flush=True)
            results["arms"][arm] = run_one(args, arm, output / arm, utilities)
            BASE.baseline.write_json(output / "all_results.json", results)
            print(f"E1111B/{arm}: {results['arms'][arm]['status']}", flush=True)
        manifest["status"] = "OK" if all(x["status"] == "OK" for x in results["arms"].values()) else "FAILED"
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        BASE.baseline.write_json(output / "manifest.json", manifest)
        BASE.baseline.write_json(output / "all_results.json", results)
    print(f"Results: {output / 'all_results.json'}")
    return 0 if manifest["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent G25 runs: B0, trainable-CLS control, and 4 masks x 2 losses."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


TRAIN_DS = "FaceForensics++"
VAL_DS = "Celeb-DF-v2"
CROSS_DS = ["WDF", "FFIW", VAL_DS, "DeepFakeDetection", "DFDC", "DFDCP", "DeeperForensics-1.0"]
TEST_DS = CROSS_DS + [TRAIN_DS]
MASKS = {"00": "read_only", "10": "cls_only", "01": "patch_only", "11": "full"}
ARMS = {"B0": {"model_name": "effort"},
        "C0": {"model_name": "effort_g25", "g25_num_tokens": 0}}
for prefix, supervision in (("M", "max"), ("A", "all")):
    for code, mask in MASKS.items():
        ARMS[prefix + code] = {"model_name": "effort_g25",
                               "g25_attention_mode": mask, "g25_supervision": supervision}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--output_dir", default="./experiment_results/g25")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--sampler_real_ratio", type=float, default=0.30)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--insert_layer", type=int, default=20)
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def arm_settings(args, arm):
    settings = dict(ARMS[arm])
    if settings["model_name"] == "effort_g25":
        settings = {"g25_num_tokens": args.num_tokens, "g25_insert_layer": args.insert_layer,
                    "g25_attention_mode": "read_only", "g25_supervision": "max",
                    "g25_fusion_weight": 0.5, "g25_evidence_weight": 1.0,
                    "g25_diversity_weight": 0.01, **settings}
    return settings


def build_arm_config(args, arm, run_dir, build_config, for_training):
    settings = arm_settings(args, arm)
    config = build_config(
        use_mixup=False, mixup_loss_strip=False,
        sampler_real_ratio=args.sampler_real_ratio, model_name=settings["model_name"],
        log_dir=str(run_dir / "logs"), train_dataset=TRAIN_DS,
        test_dataset=VAL_DS if for_training else TEST_DS,
        n_epochs=args.n_epochs if for_training else 0, for_training=for_training,
    )
    config.update(settings)
    config.update(manualSeed=args.seed, full_train_head=True, use_mixup=False,
                  mixup_mode="none", margin_loss_mode="off", use_freq_split=False,
                  use_texture_crop=False, optimizer_wrapper=None, rank_loss_weight=0.0)
    config["testall_artifact_dir"] = str((run_dir / "testall_artifacts").resolve())
    if args.clip_pretrained_path:
        config["clip_pretrained_path"] = args.clip_pretrained_path
    return config


def write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)


def validate_metrics(result):
    metrics = result.get("testall", {})
    missing = []
    for dataset in TEST_DS:
        value = metrics.get(dataset, {}).get("video_auc")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            missing.append(dataset)
    if missing:
        result.update(status="EVAL_FAILED", missing_video_auc=missing)
    else:
        cross = (metrics[VAL_DS]["video_auc"] + metrics["DFDC"]["video_auc"]) / 2
        result.update(status="OK", AUC_cross=cross,
                      video_auc_seven_mean=sum(metrics[d]["video_auc"] for d in CROSS_DS) / len(CROSS_DS),
                      G=metrics[TRAIN_DS]["video_auc"] - cross)
    return result


def run_one(args, arm, run_dir, utilities):
    run_dir.mkdir(parents=True, exist_ok=False)
    training = build_arm_config(args, arm, run_dir, utilities.build_config, True)
    evaluation = build_arm_config(args, arm, run_dir, utilities.build_config, False)
    write_json(run_dir / "train_config.json", training)
    write_json(run_dir / "eval_config.json", evaluation)
    result = {"exp_name": f"G25/{arm}", "arm": arm, "seed": args.seed,
              "settings": arm_settings(args, arm), "status": "TRAIN_FAILED"}
    try:
        checkpoint = utilities.train_model(training, TRAIN_DS, VAL_DS)
        if checkpoint is None:
            return result
        result.update(ckpt=checkpoint, status="EVAL_FAILED")
        # Give direct evaluation the same initial RNG state across arms,
        # including when stochastic multi-crop is enabled in the base config.
        utilities.seed_evaluation(args.seed)
        result.update(utilities.evaluate_model(
            evaluation, checkpoint, TEST_DS, TRAIN_DS, str(run_dir), f"G25/{arm}"
        ))
        validate_metrics(result)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        write_json(run_dir / "result.json", result)
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_tokens < 1 or not 0 <= args.insert_layer < 24 or args.n_epochs < 0:
        parser.error("Require num_tokens >= 1, 0 <= insert_layer < 24, n_epochs >= 0")
    if not 0 < args.sampler_real_ratio < 1:
        parser.error("sampler_real_ratio must be in (0, 1)")
    if len(args.arms) != len(set(args.arms)):
        parser.error("Do not repeat arms within one run")
    if args.dry_run:
        print(json.dumps({arm: arm_settings(args, arm) for arm in args.arms}, indent=2))
        return 0

    # Lazy import keeps --dry_run/--help usable without ML dependencies.
    import random
    import numpy as np
    import torch
    import experiment_utils as utilities

    def seed_evaluation(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Adapter local to this runner; existing experiment utilities stay unchanged.
    from types import SimpleNamespace
    adapter = SimpleNamespace(build_config=utilities.build_config,
                              train_model=utilities.train_model,
                              evaluate_model=utilities.evaluate_model,
                              seed_evaluation=seed_evaluation)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(args.output_dir) / f"seed{args.seed}_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=False)
    source_root = Path(__file__).resolve().parents[1]
    source_paths = ["experiments/run_g25.py", "experiments/experiment_utils.py",
                    "training/detectors/g25_tokens.py", "training/detectors/effort_detector_g25.py",
                    "training/detectors/effort_detector.py", "training/detectors/__init__.py",
                    "training/config/detector/effort.yaml", "testall.py", "training/test.py"]
    # HEAD alone does not identify uncommitted experiment code.
    source_hashes = {p: hashlib.sha256((source_root / p).read_bytes()).hexdigest()
                     for p in source_paths}
    write_json(root / "manifest.json", {"arguments": vars(args),
               "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
               "source_sha256": source_hashes,
               "torch": torch.__version__, "python": sys.version,
               "arms": {arm: arm_settings(args, arm) for arm in args.arms}})
    results = []
    for arm in args.arms:
        print(f"G25/{arm}: {arm_settings(args, arm)}", flush=True)
        result = run_one(args, arm, root / arm, adapter)
        results.append(result)
        write_json(root / "all_results.json", results)
        print(f"G25/{arm}: {result['status']}", flush=True)
    print(f"Results: {root / 'all_results.json'}")
    return 0 if all(result["status"] == "OK" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())

"""E1009: fresh CLIP initialization with G25/G25v2 tokens and tuning controls."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import run_g25 as baseline
from run_g25v2 import evaluate_readouts


ARMS = {}
for variant in ("G25", "G25v2"):
    for token_arm in baseline.ARMS:
        if token_arm not in ("B0", "C0"):
            ARMS[f"{variant.upper()}_TOKENS_{token_arm}"] = {
                "variant": variant, "tuning": "tokens", "token_arm": token_arm,
            }
for tuning in ("late_lora", "layernorm", "all_lora"):
    for variant in ("G25", "G25v2"):
        for token_arm in ("M01", "M11"):
            ARMS[f"{variant.upper()}_{tuning.upper()}_{token_arm}"] = {
                "variant": variant, "tuning": tuning, "token_arm": token_arm,
            }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--output_dir", default="./experiment_results/E1009")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--sampler_real_ratio", type=float, default=0.30)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--insert_layer", type=int, default=20)
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_readout_diagnostics", action="store_true",
                        help="Skip CLS/evidence scoring of the SAME selected checkpoint")
    return parser


def arm_settings(args, arm):
    metadata = ARMS[arm]
    settings = baseline.arm_settings(args, metadata["token_arm"])
    settings.update(model_name="effort_e1009", e1009_tuning=metadata["tuning"],
                    g25v2_aux_grad_mode="joint" if metadata["variant"] == "G25" else "isolated",
                    g25v2_score_mode="fused")
    return settings


def build_arm_config(args, arm, run_dir, build_config, for_training):
    config = baseline.build_arm_config(
        args, ARMS[arm]["token_arm"], run_dir, build_config, for_training,
    )
    config.update(arm_settings(args, arm))
    return config


def run_one(args, arm, run_dir, utilities):
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {"exp_name": f"E1009/{arm}", "arm": arm, **ARMS[arm], "seed": args.seed,
              "settings": arm_settings(args, arm), "status": "TRAIN_FAILED"}
    try:
        training = build_arm_config(args, arm, run_dir, utilities.build_config, True)
        evaluation = build_arm_config(args, arm, run_dir, utilities.build_config, False)
        baseline.write_json(run_dir / "train_config.json", training)
        baseline.write_json(run_dir / "eval_config.json", evaluation)
        checkpoint = utilities.train_model(training, baseline.TRAIN_DS, baseline.VAL_DS)
        if checkpoint is None:
            return result
        result.update(ckpt=checkpoint, status="EVAL_FAILED")
        utilities.seed_evaluation(args.seed)
        primary = utilities.evaluate_model(
            evaluation, checkpoint, baseline.TEST_DS, baseline.TRAIN_DS,
            str(run_dir), result["exp_name"],
        )
        baseline.validate_metrics(primary)
        result.update(primary)
        result["primary_status"] = primary["status"]
        result["readouts"] = evaluate_readouts(
            evaluation, checkpoint, run_dir, utilities, primary=primary,
            diagnostics=not args.skip_readout_diagnostics,
        )
        if any(readout["status"] != "OK" for readout in result["readouts"].values()):
            result["status"] = "EVAL_FAILED"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["status"] = "EVAL_FAILED" if "ckpt" in result else "TRAIN_FAILED"
    finally:
        baseline.write_json(run_dir / "result.json", result)
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

    # Keep --help and --dry_run usable with the standard-library Python alone.
    import random
    import numpy as np
    import torch
    import transformers
    import experiment_utils as utilities
    from types import SimpleNamespace

    def seed_evaluation(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    adapter = SimpleNamespace(build_config=utilities.build_config, train_model=utilities.train_model,
                              evaluate_model=utilities.evaluate_model, run_testall=utilities.run_testall,
                              seed_evaluation=seed_evaluation)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(args.output_dir) / f"seed{args.seed}_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root,
                              capture_output=True, text=True, check=False)
    paths = ["experiments/run_e1009.py", "experiments/run_g25.py", "experiments/run_g25v2.py",
             "experiments/experiment_utils.py", "training/detectors/effort_detector_e1009.py",
             "training/detectors/effort_detector_g25.py", "training/detectors/effort_detector_g25v2.py",
             "training/detectors/g25_tokens.py", "training/detectors/g25v2_tokens.py",
             "training/detectors/effort_detector.py", "training/detectors/__init__.py",
             "training/config/detector/effort.yaml", "training/config/train_config.yaml",
             "training/config/test_config.yaml", "training/dataset/abstract_dataset.py",
             "training/train.py", "training/test.py", "testall.py"]
    baseline.write_json(root / "manifest.json", {
        "arguments": vars(args), "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {path: hashlib.sha256((source_root / path).read_bytes()).hexdigest()
                          for path in paths},
        "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
        "arms": {arm: {**ARMS[arm], "settings": arm_settings(args, arm)} for arm in args.arms},
    })
    results = []
    for arm in args.arms:
        print(f"E1009/{arm}: {arm_settings(args, arm)}", flush=True)
        result = run_one(args, arm, root / arm, adapter)
        results.append(result)
        baseline.write_json(root / "all_results.json", results)
        print(f"E1009/{arm}: {result['status']}", flush=True)
    print(f"Results: {root / 'all_results.json'}")
    return 0 if all(result["status"] == "OK" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())

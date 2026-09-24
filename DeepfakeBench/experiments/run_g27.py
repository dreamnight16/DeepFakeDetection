"""G27 isolated expert ablations: B0, G26 control, E1/E2/E3 and Full."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import run_g25 as baseline


DEFAULT_ARMS = ("B0", "G26", "E1", "E2", "E3", "Full")
ARMS = DEFAULT_ARMS


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument("--output_dir", default="./experiment_results/g27")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--sampler_real_ratio", type=float, default=.30)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--insert_layer", type=int, default=20)
    parser.add_argument("--balance_weight", type=float, default=.1)
    parser.add_argument("--router_temperature", type=float, default=1.)
    parser.add_argument("--hard_floor", type=float, default=.2)
    parser.add_argument("--hard_width", type=float, default=.2)
    parser.add_argument("--consistency_weight", type=float, default=.1)
    parser.add_argument("--view_contrast", type=float, default=.9)
    parser.add_argument("--view_brightness", type=float, default=.02)
    parser.add_argument("--attention_samples", type=int, default=16,
                        help="First N frames per dataset for attention snapshots; 0 disables")
    parser.add_argument("--mil_temperature", type=float, default=.5)
    parser.add_argument("--evidence_weight", type=float, default=1.)
    parser.add_argument("--gate_width", type=float, default=.2)
    parser.add_argument("--aux_max_weight", type=float, default=.5)
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--skip_diagnostics", action="store_true",
                        help="Skip branch testall and paired routing diagnostics; run primary evaluation only")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def arm_settings(args, arm):
    if arm == "B0":
        return {"model_name": "effort"}
    if arm not in ARMS:
        raise ValueError(f"Unknown G27 arm: {arm}")
    num_tokens, insert_layer = args.num_tokens, args.insert_layer
    return {"model_name": "effort_g27", "g27_num_tokens": num_tokens,
            "g27_balance_weight": args.balance_weight if arm in ("E1", "Full") else 0.,
            "g27_hard_weighting": arm in ("E2", "Full"),
            "g27_consistency_weight": args.consistency_weight if arm in ("E3", "Full") else 0.,
            "g27_router_temperature": args.router_temperature,
            "g27_hard_floor": args.hard_floor, "g27_hard_width": args.hard_width,
            "g27_view_contrast": args.view_contrast, "g27_view_brightness": args.view_brightness,
            "g27_insert_layer": insert_layer, "g27_mil_temperature": args.mil_temperature,
            "g27_evidence_weight": args.evidence_weight, "g27_gate_width": args.gate_width,
            "g27_aux_max_weight": args.aux_max_weight, "g27_score_mode": "gated"}


def build_arm_config(args, arm, run_dir, build_config, for_training):
    # Reuse only the B0 protocol builder, then add G27's independently named keys.
    config = baseline.build_arm_config(args, "B0", run_dir, build_config, for_training)
    config.update(arm_settings(args, arm))
    config["multi_crop"] = False
    config["g27_attention_samples"] = args.attention_samples
    return config


def evaluate_diagnostics(config, checkpoint, run_dir, utilities):
    readouts = {}
    for mode in ("cls", "evidence"):
        folder = run_dir / "readouts" / mode
        folder.mkdir(parents=True, exist_ok=False)
        evaluation = dict(config)
        evaluation["g27_score_mode"] = mode
        evaluation["testall_artifact_dir"] = str((folder / "artifacts").resolve())
        baseline.write_json(folder / "eval_config.json", evaluation)
        result = {"score_mode": mode, "ckpt": checkpoint, "status": "EVAL_FAILED"}
        try:
            result["testall"] = utilities.run_testall(
                checkpoint, baseline.TEST_DS, str(folder / "testall.log"),
                extra_config=evaluation, artifact_dir=evaluation["testall_artifact_dir"],
            )
            baseline.validate_metrics(result)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        baseline.write_json(folder / "result.json", result)
        readouts[mode] = result
    routing = {"status": "EVAL_FAILED", "ckpt": checkpoint, "unit": "frame",
               "threshold": .5, "same_forward": True}
    try:
        routing["datasets"] = utilities.collect_routing_diagnostics(
            config, checkpoint, baseline.TEST_DS, run_dir / "routing", utilities)
        if set(routing["datasets"]) != set(baseline.TEST_DS):
            raise ValueError("Incomplete routing diagnostics")
        routing["status"] = "OK"
    except Exception as exc:
        routing["error"] = f"{type(exc).__name__}: {exc}"
    baseline.write_json(run_dir / "routing_diagnostics.json", routing)
    return readouts, routing


def run_one(args, arm, run_dir, utilities):
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {"exp_name": f"G27/{arm}", "arm": arm, "seed": args.seed,
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
        primary = utilities.evaluate_model(evaluation, checkpoint, baseline.TEST_DS,
                                           baseline.TRAIN_DS, str(run_dir), f"G27/{arm}")
        baseline.validate_metrics(primary)
        result.update(primary)
        result["primary_status"] = primary["status"]
        result["diagnostics_requested"] = arm != "B0" and not args.skip_diagnostics
        if result["diagnostics_requested"]:
            result["readouts"], result["routing"] = evaluate_diagnostics(
                evaluation, checkpoint, run_dir, utilities)
            if (result["routing"]["status"] != "OK"
                    or any(r["status"] != "OK" for r in result["readouts"].values())):
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
    if ((args.num_tokens is not None and args.num_tokens < 1)
            or (args.insert_layer is not None and not 0 <= args.insert_layer < 24)
            or args.n_epochs < 0):
        parser.error("Require K>=1, 0<=insert_layer<24, n_epochs>=0")
    if args.attention_samples < 0:
        parser.error("attention_samples must be nonnegative")
    for name in ("balance_weight", "consistency_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name} must be finite and nonnegative")
    if not math.isfinite(args.router_temperature) or args.router_temperature <= 0:
        parser.error("router_temperature must be finite and positive")
    if not 0 <= args.hard_floor <= 1 or not 0 < args.hard_width <= .5:
        parser.error("Require hard_floor in [0,1], hard_width in (0,.5]")
    if (not math.isfinite(args.view_contrast) or not 0 < args.view_contrast <= 1
            or not math.isfinite(args.view_brightness) or abs(args.view_brightness) > .1):
        parser.error("Require view_contrast in (0,1], abs(view_brightness)<=.1")
    if not 0 < args.sampler_real_ratio < 1 or len(args.arms) != len(set(args.arms)):
        parser.error("Require sampler_real_ratio in (0,1) and unique arms")
    if not math.isfinite(args.mil_temperature) or args.mil_temperature <= 0:
        parser.error("mil_temperature must be finite and positive")
    if not math.isfinite(args.evidence_weight) or args.evidence_weight < 0:
        parser.error("evidence_weight must be finite and nonnegative")
    if not 0 < args.gate_width <= .5 or not 0 <= args.aux_max_weight <= .5:
        parser.error("Require gate_width in (0,0.5] and aux_max_weight in [0,0.5]")
    if args.dry_run:
        print(json.dumps({arm: arm_settings(args, arm) for arm in args.arms}, indent=2))
        return 0

    import random
    from types import SimpleNamespace

    import numpy as np
    import torch
    import transformers
    import experiment_utils as utilities
    from g27_diagnostics import collect_routing_diagnostics

    def seed_evaluation(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    adapter = SimpleNamespace(build_config=utilities.build_config, train_model=utilities.train_model,
                              evaluate_model=utilities.evaluate_model, run_testall=utilities.run_testall,
                              load_model=utilities.load_model, get_data_loader=utilities.get_data_loader,
                              collect_routing_diagnostics=collect_routing_diagnostics,
                              seed_evaluation=seed_evaluation)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = Path(args.output_dir) / f"seed{args.seed}_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    paths = ["experiments/run_g27.py", "experiments/g27_diagnostics.py", "experiments/run_g25.py",
             "experiments/experiment_utils.py", "training/detectors/effort_detector_g27.py",
             "training/detectors/g27_tokens.py", "training/detectors/g25v2_tokens.py",
             "training/detectors/effort_detector_g26.py", "training/detectors/g26_tokens.py",
             "experiments/g26_diagnostics.py",
             "training/detectors/g25_tokens.py", "training/detectors/effort_detector.py",
             "training/detectors/__init__.py", "training/config/detector/effort.yaml",
             "training/train.py", "training/test.py", "testall.py"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root,
                              capture_output=True, text=True, check=False)
    baseline.write_json(root / "manifest.json", {
        "arguments": vars(args), "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {p: hashlib.sha256((source_root / p).read_bytes()).hexdigest() for p in paths},
        "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
        "arms": {arm: arm_settings(args, arm) for arm in args.arms},
        "selection": "Celeb-DF-v2 frame auc of primary score; diagnostics use same checkpoint",
        "gate_calibration": "fixed defaults, not fitted or tuned on test data",
        "control": "G26 arm uses G27 with all new losses off; CPU equivalence tested",
        "view": "fixed contrast 0.9/brightness +0.02 by default; full batch each training step",
        "attention": "bounded first frames, separate deterministic forward on same tensor; not localization truth",
    })
    results = []
    for arm in args.arms:
        print(f"G27/{arm}: {arm_settings(args, arm)}", flush=True)
        result = run_one(args, arm, root / arm, adapter)
        results.append(result)
        baseline.write_json(root / "all_results.json", results)
        print(f"G27/{arm}: {result['status']}", flush=True)
    print(f"Results: {root / 'all_results.json'}")
    return 0 if all(r["status"] == "OK" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

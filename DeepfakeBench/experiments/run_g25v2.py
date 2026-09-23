"""G25v2: isolate evidence-loss gradients; retain G25's 10-arm protocol."""

import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

import run_g25 as baseline


def build_parser():
    parser = baseline.build_parser()
    parser.description = __doc__
    parser.set_defaults(output_dir="./experiment_results/g25v2")
    parser.add_argument("--aux_grad_mode", choices=["isolated", "joint"], default="isolated")
    parser.add_argument("--skip_readout_diagnostics", action="store_true",
                        help="Skip CLS-only/evidence-only scoring of the SAME selected checkpoint")
    parser.add_argument("--checkpoint", help="Evaluation only; requires --source_config and one --arms entry")
    parser.add_argument("--source_config", help="Saved G25/G25v2 eval_config.json for --checkpoint")
    return parser


def arm_settings(args, arm):
    settings = baseline.arm_settings(args, arm)
    if settings["model_name"] == "effort_g25":
        settings.update(model_name="effort_g25v2", g25v2_aux_grad_mode=args.aux_grad_mode,
                        g25v2_score_mode="fused")
    return settings


def build_arm_config(args, arm, run_dir, build_config, for_training):
    config = baseline.build_arm_config(args, arm, run_dir, build_config, for_training)
    config.update(arm_settings(args, arm))
    return config


def evaluate_readouts(config, checkpoint, run_dir, utilities, primary=None, diagnostics=True):
    """Same checkpoint, fixed scoring rules; never select a winner on test sets.

    Reuse the primary fused evaluation when already available. Diagnostic
    branches use testall directly, avoiding redundant training-set inference.
    """
    modes = ["fused"]
    if diagnostics and config.get("g25_num_tokens", 0) > 0:
        modes += ["cls", "evidence"]
    results = {}
    for mode in modes:
        if mode == "fused" and primary is not None:
            results[mode] = dict(primary)
            continue
        output = run_dir / "readouts" / mode
        output.mkdir(parents=True, exist_ok=False)
        evaluation = dict(config)
        evaluation["g25v2_score_mode"] = mode
        evaluation["testall_artifact_dir"] = str((output / "artifacts").resolve())
        baseline.write_json(output / "eval_config.json", evaluation)
        result = {"status": "EVAL_FAILED", "score_mode": mode, "ckpt": checkpoint}
        try:
            metrics = utilities.run_testall(
                checkpoint, baseline.TEST_DS, str(output / "testall.log"),
                extra_config=evaluation, artifact_dir=evaluation["testall_artifact_dir"],
            )
            result["testall"] = metrics
            baseline.validate_metrics(result)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        baseline.write_json(output / "result.json", result)
        results[mode] = result
    return results


def run_one(args, arm, run_dir, utilities):
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {"exp_name": f"G25v2/{arm}", "arm": arm, "seed": args.seed,
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
                                            baseline.TRAIN_DS, str(run_dir), f"G25v2/{arm}")
        baseline.validate_metrics(primary)
        result.update(primary)
        result["primary_status"] = primary["status"]
        result["readouts"] = evaluate_readouts(
            evaluation, checkpoint, run_dir, utilities, primary=primary,
            diagnostics=not args.skip_readout_diagnostics,
        )
        if any(r["status"] != "OK" for r in result["readouts"].values()):
            result["status"] = "EVAL_FAILED"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        # Exceptions after primary evaluation must not leave a false OK status.
        result["status"] = "EVAL_FAILED" if "ckpt" in result else "TRAIN_FAILED"
    finally:
        baseline.write_json(run_dir / "result.json", result)
    return result


def load_source_config(args):
    """Load the checkpoint's actual saved config instead of guessing K/mask."""
    source_path = Path(args.source_config).resolve()
    config = json.loads(source_path.read_text(encoding="utf-8"))
    if config.get("model_name") not in ("effort_g25", "effort_g25v2"):
        raise ValueError("source_config must belong to G25 or G25v2")
    required = ("g25_num_tokens", "g25_insert_layer", "g25_attention_mode", "g25_supervision",
                "g25_fusion_weight", "g25_evidence_weight", "g25_diversity_weight", "manualSeed")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Incomplete source config: {missing}")
    expected = baseline.ARMS[args.arms[0]]
    if args.arms[0] == "B0" or any(config.get(k) != v for k, v in expected.items() if k != "model_name"):
        raise ValueError("--arms must match the checkpoint's mask/supervision/control")
    if args.arms[0] != "C0" and config["g25_num_tokens"] < 1:
        raise ValueError("A token arm requires g25_num_tokens >= 1")
    # Mask/loss settings do not change weight shapes: strict state_dict loading
    # alone cannot detect a checkpoint accidentally paired with another arm.
    record_path = source_path.parent / "result.json"
    if not record_path.is_file():
        raise ValueError("Keep the source arm's result.json beside source_config")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("arm") != args.arms[0] or record.get("seed") != config["manualSeed"]:
        raise ValueError("Source result.json arm/seed does not match source_config")
    architecture_keys = [key for key in required if key != "manualSeed"]
    if any(record.get("settings", {}).get(key) != config[key] for key in architecture_keys):
        raise ValueError("Source result.json settings do not match source_config")
    log_dir = PurePosixPath(str(config.get("log_dir", "")).replace("\\", "/"))
    saved_checkpoint = PurePosixPath(str(record.get("ckpt", "")).replace("\\", "/"))
    try:
        suffix = saved_checkpoint.relative_to(log_dir)
    except ValueError as exc:
        raise ValueError("Source checkpoint is outside the recorded log_dir") from exc
    if log_dir.name != "logs" or ".." in suffix.parts or suffix.suffix != ".pth":
        raise ValueError("Unsupported source checkpoint layout; preserve the arm directory")
    expected_checkpoint = source_path.parent / "logs" / Path(*suffix.parts)
    if Path(args.checkpoint).resolve() != expected_checkpoint.resolve():
        raise ValueError("Checkpoint path does not match the source arm's result.json")
    if not expected_checkpoint.is_file():
        raise ValueError("Recorded checkpoint file is missing")
    config["model_name"] = "effort_g25v2"
    # No backward pass runs in evaluation; record the source's training mode.
    config.setdefault("g25v2_aux_grad_mode", "joint")
    config["g25v2_score_mode"] = "fused"
    if args.clip_pretrained_path:
        config["clip_pretrained_path"] = args.clip_pretrained_path
    return config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_tokens < 1 or not 0 <= args.insert_layer < 24 or args.n_epochs < 0:
        parser.error("Require num_tokens >= 1, 0 <= insert_layer < 24, n_epochs >= 0")
    if not 0 < args.sampler_real_ratio < 1 or len(args.arms) != len(set(args.arms)):
        parser.error("Require sampler_real_ratio in (0,1) and unique arms")
    if bool(args.checkpoint) != bool(args.source_config) or (args.checkpoint and len(args.arms) != 1):
        parser.error("Evaluation-only requires --checkpoint, --source_config, and exactly one arm")
    try:
        source = load_source_config(args) if args.checkpoint else None
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(source if source is not None else
                         {arm: arm_settings(args, arm) for arm in args.arms}, indent=2))
        return 0

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
    seed = source["manualSeed"] if source is not None else args.seed
    root = Path(args.output_dir) / f"seed{seed}_{stamp}_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    paths = ["experiments/run_g25v2.py", "experiments/run_g25.py", "experiments/experiment_utils.py",
             "training/detectors/g25v2_tokens.py", "training/detectors/effort_detector_g25v2.py",
             "training/detectors/g25_tokens.py", "training/detectors/effort_detector_g25.py",
             "training/detectors/effort_detector.py", "training/config/detector/effort.yaml",
             "training/train.py", "training/test.py", "testall.py"]
    baseline.write_json(root / "manifest.json", {
        "arguments": vars(args), "effective_seed": seed,
        "source_sha256": {p: hashlib.sha256((source_root / p).read_bytes()).hexdigest() for p in paths},
        "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
        "evaluation_only": source is not None,
    })
    if source is not None:
        readouts = evaluate_readouts(source, args.checkpoint, root, adapter,
                                     diagnostics=not args.skip_readout_diagnostics)
        result = {"exp_name": f"G25v2/diagnose/{args.arms[0]}", "ckpt": args.checkpoint,
                  "source_config": args.source_config, "readouts": readouts,
                  "status": "OK" if all(r["status"] == "OK" for r in readouts.values()) else "EVAL_FAILED"}
        baseline.write_json(root / "all_results.json", [result])
        print(f"Results: {root / 'all_results.json'}")
        return 0 if result["status"] == "OK" else 1
    results = []
    for arm in args.arms:
        print(f"G25v2/{arm}: {arm_settings(args, arm)}", flush=True)
        result = run_one(args, arm, root / arm, adapter)
        results.append(result)
        baseline.write_json(root / "all_results.json", results)
        print(f"G25v2/{arm}: {result['status']}", flush=True)
    print(f"Results: {root / 'all_results.json'}")
    return 0 if all(r["status"] == "OK" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

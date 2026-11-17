"""E1115: full fixed-budget training and frozen metric/adaptation ablations."""

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from e1115.protocol import TRAIN_ARMS, plan


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=["sampling", "geometry", "adaptation", "baseline"],
                        default=["sampling", "geometry", "adaptation", "baseline"])
    parser.add_argument("--train_arms", nargs="+", choices=list(TRAIN_ARMS), default=list(TRAIN_ARMS))
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1115")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=10, help="Inclusive native epochs 0..N")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--feature_batch_size", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, help="Default: native .3-ratio sampler length")
    parser.add_argument("--sampler_real_ratio", type=float, default=.3)
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root")
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base_run", type=Path)
    parser.add_argument("--base_checkpoint", type=Path)
    parser.add_argument("--base_config", type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=[10, 30, 100])
    parser.add_argument("--calibration_videos", type=int, default=30,
                        help="Additional trusted real videos, separate from fitting budgets")
    parser.add_argument("--adaptation_repeats", type=int, default=3)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def experiment_plan(args):
    return {**plan(), "groups": args.groups, "train_arms": {k: TRAIN_ARMS[k] for k in args.train_arms},
            "training_seed": args.seed, "training_epochs_inclusive": [0, args.n_epochs],
            "batch_size": args.batch_size, "steps_per_epoch": args.steps_per_epoch or "native ratio .3 sampler length",
            "adaptation_fit_budgets": args.budgets, "calibration_real_videos": args.calibration_videos,
            "adaptation_repeats": args.adaptation_repeats, "development_gate": False,
            "posthoc_on_each_trained_arm": True,
            "checkpoint_selection": "native Celeb-DF-v2 frame AUC; same fixed evaluation schedule"}


def validate(parser, args):
    for name in ("groups", "train_arms", "budgets"):
        if len(getattr(args, name)) != len(set(getattr(args, name))):
            parser.error(f"Duplicate {name}")
    if (args.n_epochs < 0 or args.seed < 0 or args.batch_size < 2 or args.batch_size % 2
            or args.feature_batch_size < 1 or min(args.budgets) < 1
            or args.calibration_videos < 1 or args.adaptation_repeats < 1):
        parser.error("Invalid training or target real budget parameters")
    if args.sampler_real_ratio != .3:
        parser.error("Native reference ratio is .3; .5/.7 are separate registered arms")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 2:
        parser.error("steps_per_epoch must be at least 2")
    if "sampling" in args.groups and any((args.base_run, args.base_checkpoint, args.base_config)):
        parser.error("Omit sampling to run post-hoc experiments on a supplied frozen checkpoint")
    if "sampling" not in args.groups and not (args.base_run or (args.base_checkpoint and args.base_config)) and not args.dry_run:
        parser.error("Post-hoc groups need --base_run or --base_checkpoint with --base_config")


def build_training_config(args, arm, folder, utilities, steps):
    from run_e1021 import build_arm_config
    spec = TRAIN_ARMS[arm]
    training = build_arm_config(args, "B0", folder, utilities.build_config, True)
    evaluation = build_arm_config(args, "B0", folder, utilities.build_config, False)
    training.update(sampler_real_ratio=spec["ratio"], e1115_strict_images=True,
                    e1115_class_balanced_loss=spec["loss"] == "class_mean")
    if spec["fixed_budget"]:
        training.update(e1115_steps_per_epoch=steps, e1115_frame_policy=spec["frames"])
    evaluation.update(e1115_strict_images=True)
    return training, evaluation


def write_combined_tables(report, output):
    """Aggregate backbones, retaining full-test versus adapted-subset scope."""
    summary_fields = ["training_arm", "method", "scope", "status", "independent_video_auc_mean_fullpath", "video_auc_seven_mean_fullpath"]
    metric_fields = ["training_arm", "method", "scope", "dataset", "frame_auc", "video_auc_fullpath", "num_frames", "num_videos"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as sf, (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as mf:
        summary, metrics = csv.DictWriter(sf, fieldnames=summary_fields), csv.DictWriter(mf, fieldnames=metric_fields)
        summary.writeheader()
        metrics.writeheader()
        for training_arm, item in report["posthoc"].items():
            if not item.get("output"):
                continue
            directory = Path(item["output"])
            result_path = directory / "comparisons/all_results.json"
            if not result_path.is_file():
                result_path = directory / "all_results.json"
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            arms = result.get("arms", {"B0": {"scope": "source_only", "status": result["status"], "datasets": result.get("datasets", {})}})
            for name, arm in arms.items():
                base = {"training_arm": training_arm, "method": name, "scope": arm["scope"]}
                summary.writerow({**base, **{k: arm.get(k) for k in summary_fields[3:]}})
                for dataset, values in arm.get("datasets", {}).items():
                    metrics.writerow({**base, "dataset": dataset, **{k: values.get(k) for k in metric_fields[4:]}})


def frozen_configs(args, source_config):
    import copy
    import yaml
    from run_g25 import TEST_DS
    config = copy.deepcopy(source_config)
    labels = yaml.safe_load((ROOT / "training/config/test_config.yaml").read_text(encoding="utf-8"))["label_dict"]
    for name, value in labels.items():
        if name in config["label_dict"] and (config["label_dict"][name] != 0) != (value != 0):
            raise ValueError(f"Conflicting binary evaluation label: {name}")
        config["label_dict"].setdefault(name, value)
    config.update(dataset_json_folder=str(args.dataset_json_folder), rgb_root_override=args.rgb_root,
                  clip_pretrained_path=args.clip_pretrained_path, e1115_strict_images=True)
    evaluation = copy.deepcopy(config)
    evaluation.update(use_data_augmentation=False, test_dataset=TEST_DS, multi_crop=False)
    return config, evaluation


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(parser, args)
    if args.dry_run:
        print(json.dumps(experiment_plan(args), ensure_ascii=False, indent=2))
        return 0
    from e1111_paths import normalize_arguments
    from g21.io import atomic_json, file_digest
    # Resolve a supplied baseline before applying defaults: retain its actual paths.
    resolved = None
    if "sampling" not in args.groups:
        from e1111_runner import resolve_source
        resolved = resolve_source(args)
    normalize_arguments(args, ROOT, resolved["config"] if resolved else None)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir / f"seed{args.seed}_{stamp}_{os.getpid()}"
    output.mkdir(parents=True, exist_ok=False)
    report = {"experiment": "E1115", "status": "RUNNING", "protocol": experiment_plan(args),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "training": {}, "posthoc": {}, "output": str(output.resolve())}
    files = [*Path(ROOT / "training/e1115").glob("*.py"), Path(__file__),
             ROOT / "training/train.py", ROOT / "training/dataset/abstract_dataset.py",
             ROOT / "training/dataset/balance_batch_sampler.py", ROOT / "training/detectors/effort_detector.py"]
    files.extend(ROOT / p for p in ["experiments/experiment_utils.py", "experiments/run_e1021.py",
                 "experiments/run_e1111b.py", "experiments/run_g25.py", "experiments/e0924_export.py",
                 "experiments/run_e1001.py", "experiments/e0924_decision.py", "experiments/e1111_paths.py",
                 "training/trainer/trainer_v2.py", "training/e1111/features.py", "training/e1111/audit.py",
                 "training/e1111/protocol.py", "training/g21/io.py", "training/g21/data.py", "training/g21/manifest.py"])
    report["source_sha256"] = {str(p.relative_to(ROOT)): file_digest(p) for p in files}
    atomic_json(output / "all_results.json", report)
    try:
        import torch
        import experiment_utils as utilities
        from run_e1021 import build_arm_config, seed_everything
        from run_e1111b import check_input_files
        from run_g25 import TRAIN_DS, VAL_DS, TEST_DS
        from e1115.assets import build_source_audit, export_native, prepare_source
        from e1115.suite import run_suite, _means
        if args.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(args.device))
        source_records = []
        if "sampling" in args.groups:
            if args.device != "cuda:0":
                raise ValueError("Native training uses cuda:0; choose physical GPU through CUDA_VISIBLE_DEVICES")
            preview = build_arm_config(args, "B0", output / "preflight", utilities.build_config, True)
            eval_preview = build_arm_config(args, "B0", output / "preflight", utilities.build_config, False)
            report["input_preflight"] = check_input_files(preview, eval_preview, utilities)
            seed_everything(args.seed)
            labels = utilities.get_train_loader(preview).dataset.label_list
            from dataset.balance_batch_sampler import BalanceBatchSampler
            native_steps = len(BalanceBatchSampler(labels, args.batch_size // 2, real_ratio=.3))
            steps = args.steps_per_epoch or native_steps
            if "B0_NATIVE" in args.train_arms and steps != native_steps:
                raise ValueError("With B0_NATIVE included, steps_per_epoch must equal native length")
            report["fixed_steps_per_epoch"] = steps
            for arm in args.train_arms:
                print(f"E1115/train/{arm}", flush=True)
                folder = output / "training" / arm
                folder.mkdir(parents=True)
                training, evaluation = build_training_config(args, arm, folder, utilities, steps)
                atomic_json(folder / "train_config.json", training)
                atomic_json(folder / "eval_config.json", evaluation)
                item = {"status": "RUNNING", "settings": TRAIN_ARMS[arm], "steps_per_epoch": steps}
                report["training"][arm] = item
                atomic_json(output / "all_results.json", report)
                checkpoint = utilities.train_model(training, TRAIN_DS, VAL_DS)
                if checkpoint is None or not Path(checkpoint).is_file():
                    item.update(status="TRAIN_FAILED", error="Native training returned no selected checkpoint")
                    atomic_json(folder / "result.json", item)
                    atomic_json(output / "all_results.json", report)
                    continue
                item.update(status="OK", ckpt=str(Path(checkpoint).resolve()), checkpoint_sha256=file_digest(checkpoint))
                atomic_json(folder / "result.json", item)
                source_records.append((arm, folder / "train_config.json", folder / "eval_config.json", checkpoint))
                atomic_json(output / "all_results.json", report)
        else:
            config, evaluation = frozen_configs(args, resolved["config"])
            report["source_checkpoint_training_seed"] = config["manualSeed"]
            base = output / "base"
            base.mkdir()
            atomic_json(base / "train_config.json", config)
            atomic_json(base / "eval_config.json", evaluation)
            source_records = [("FROZEN_B0", base / "train_config.json", base / "eval_config.json", resolved["checkpoint"])]
        needs_metric = any(g in args.groups for g in ("geometry", "adaptation", "baseline"))
        audit, index = build_source_audit(args.dataset_json_folder, args.rgb_root, output / "source") if needs_metric else (None, None)
        anchor = None
        source_records.sort(key=lambda item: item[0] != "B0_NATIVE")
        for arm, config, evaluation, checkpoint in source_records:
            try:
                destination = output / "posthoc" / arm
                destination.mkdir(parents=True)
                seed_everything(args.seed)
                native = export_native(evaluation, checkpoint, args.clip_pretrained_path,
                                       destination / "native8", args.device, args.feature_batch_size)
                if needs_metric:
                    source = prepare_source(audit, index, config, checkpoint, args.clip_pretrained_path,
                                            destination / "source_features", args.device, args.feature_batch_size)
                    result = run_suite(audit, source, native, destination / "comparisons", groups=args.groups,
                                       seed=args.seed, budgets=args.budgets, calibration_count=args.calibration_videos,
                                       repeats=args.adaptation_repeats, anchor=anchor)
                    if arm == "B0_NATIVE":
                        anchor = {"native": native, "calibration": destination / "source_features/calibration.npz"}
                    del source
                else:
                    from e1115.assets import load_native
                    from e1115.evaluation import evaluate_scores
                    result = {"status": "OK", "arms": {"B0": {"status": "SCORED", "scope": "source_only", "datasets": {}}}}
                    for dataset in TEST_DS:
                        rows, z = load_native(native, dataset)
                        result["arms"]["B0"]["datasets"][dataset] = evaluate_scores(rows, rows["cls_prob"], {})
                        del z
                    _means(result)
                    atomic_json(destination / "all_results.json", result)
                report["posthoc"][arm] = {"status": result["status"], "output": str(destination.resolve())}
            except Exception as exc:
                report["posthoc"][arm] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
            atomic_json(output / "all_results.json", report)
        statuses = [x["status"] for x in [*report["training"].values(), *report["posthoc"].values()]]
        if not source_records or any(s not in ("OK", "COMPLETE_WITH_UNAVAILABLE") for s in statuses):
            report["status"] = "COMPLETED_WITH_FAILURES"
        else:
            report["status"] = "COMPLETE_WITH_UNAVAILABLE" if "COMPLETE_WITH_UNAVAILABLE" in statuses else "OK"
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        atomic_json(output / "all_results.json", report)
        write_combined_tables(report, output)
    print(f"E1115 results: {output}", flush=True)
    return 0 if report["status"] in ("OK", "COMPLETE_WITH_UNAVAILABLE") else 1


if __name__ == "__main__":
    sys.exit(main())

"""E0924: nine unique G26/G27 jobs followed by frozen G28/G29 decision heads."""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from e0924_protocol import DECISION_ARMS, TRAIN_JOBS, calibration_partition, plan


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Check metadata only, no ML imports or image inference")
    parser.add_argument("--output_dir", default=str(ROOT / "experiment_results" / "E0924"))
    parser.add_argument("--resume", type=Path, help="Existing E0924 run; successful stages are reused after signature checks")
    parser.add_argument("--stages", nargs="+", choices=("train", "decisions"), default=["train", "decisions"])
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--dataset_json_folder", default=str(ROOT / "preprocessing" / "dataset_json"))
    parser.add_argument("--rgb_root")
    parser.add_argument("--clip_pretrained_path")
    parser.add_argument("--attention_samples", type=int, default=16)
    parser.add_argument("--skip_diagnostics", action="store_true")
    parser.add_argument("--router_steps", type=int, default=300)
    parser.add_argument("--router_lr", type=float, default=.01)
    parser.add_argument("--reject_threshold", type=float, default=.7)
    return parser


def preflight(args):
    from run_g25 import TEST_DS

    path = Path(args.dataset_json_folder) / "FaceForensics++.json"
    partition = calibration_partition(read_json(path), seed=args.seed, frames=8)
    partition["metadata_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    partition["dataset_sha256"] = {}
    missing = [dataset for dataset in TEST_DS
               if not (Path(args.dataset_json_folder) / f"{dataset}.json").is_file()]
    if missing:
        raise ValueError("Missing evaluation metadata (no fallback permitted): " + ", ".join(missing))
    for dataset in TEST_DS:
        metadata_path = Path(args.dataset_json_folder) / f"{dataset}.json"
        partition["dataset_sha256"][dataset] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    for split, records in partition["records"].items():
        if {r["label_name"] == "FF-real" for r in records} != {False, True}:
            raise ValueError(f"Calibration {split} must contain real and fake videos")
    return partition


def source_hashes():
    paths = [*sorted((ROOT / "training" / "detectors").glob("*.py")),
             *sorted((ROOT / "training" / "dataset").rglob("*.py")),
             *sorted((ROOT / "training" / "trainer").rglob("*.py")),
             *sorted((ROOT / "training" / "metrics").rglob("*.py")),
             *sorted((ROOT / "training" / "config").rglob("*.yaml")),
             ROOT / "training/train.py", ROOT / "training/test.py", ROOT / "testall.py"]
    for pattern in ("run_g25.py", "run_g26.py", "run_g27.py", "g26_diagnostics.py", "g27_diagnostics.py",
                    "experiment_utils.py", "e0924_*.py", "run_e0924.py"):
        paths.extend((ROOT / "experiments").glob(pattern))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths))}


def signature(args, partition):
    values = {"protocol": plan(args.seed, args.n_epochs), "sources": source_hashes(),
            "metadata_sha256": partition["metadata_sha256"],
            "dataset_sha256": partition.get("dataset_sha256", {}),
            "dataset_json_folder": str(Path(args.dataset_json_folder).resolve()),
            "router_steps": args.router_steps, "router_lr": args.router_lr,
            "reject_threshold": args.reject_threshold, "rgb_root": args.rgb_root,
            "clip_pretrained_path": args.clip_pretrained_path,
            "skip_diagnostics": args.skip_diagnostics, "attention_samples": args.attention_samples}
    # Tuples in the plan become arrays on disk; compare the serialized schema.
    return json.loads(json.dumps(values))


def runtime(args):
    import random
    from types import SimpleNamespace
    import numpy as np
    import torch
    import transformers
    import experiment_utils as utilities

    def seed_evaluation(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def build_config(**kwargs):
        config = utilities.build_config(**kwargs)
        config["e0924_protocol"] = True
        config["dataset_json_folder"] = str(Path(args.dataset_json_folder).resolve())
        if args.rgb_root:
            config["rgb_root_override"] = args.rgb_root
        return config

    adapter = SimpleNamespace(build_config=build_config, train_model=utilities.train_model,
                              evaluate_model=utilities.evaluate_model, run_testall=utilities.run_testall,
                              load_model=utilities.load_model, get_data_loader=utilities.get_data_loader,
                              seed_evaluation=seed_evaluation)
    versions = {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__}
    return adapter, utilities, versions


def run_training(args, root, adapter):
    import run_g26
    import run_g27
    import g26_diagnostics
    import g27_diagnostics

    saved_results = (read_json(root / "training_results.json")
                     if (root / "training_results.json").exists() else {})
    results = dict(saved_results)
    for family, arm in TRAIN_JOBS:
        name = f"{family}_{arm}"
        result_file = root / "training" / name / "result.json"
        if name in saved_results or result_file.exists():
            old = saved_results.get(name) or read_json(result_file)
            if old.get("status") == "OK" and old.get("ckpt") and Path(old["ckpt"]).is_file():
                results[name] = old
                continue
        # Retain failed/incomplete attempts in place for auditing.
        folder = root / "training" / name
        if folder.exists():
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            folder = root / "attempts" / f"{name}_{stamp}"
        runner = run_g26 if family == "G26" else run_g27
        cli = ["--arms", arm, "--n_epochs", str(args.n_epochs), "--seed", str(args.seed)]
        if args.clip_pretrained_path:
            cli += ["--clip_pretrained_path", args.clip_pretrained_path]
        if args.skip_diagnostics:
            cli += ["--skip_diagnostics"]
        if family == "G27":
            cli += ["--attention_samples", str(args.attention_samples)]
        parsed = runner.build_parser().parse_args(cli)
        adapter.collect_routing_diagnostics = (g26_diagnostics if family == "G26" else g27_diagnostics).collect_routing_diagnostics
        print(f"E0924 {name} -> {folder}", flush=True)
        try:
            result = runner.run_one(parsed, arm, folder, adapter)
            result["artifact_dir"] = str(folder.resolve())
        except Exception as exc:
            result = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "artifact_dir": str(folder)}
        results[name] = result
        # Latest-result pointer, including retry directories, is persisted each job.
        write_json(root / "training_results.json", results)
    write_json(root / "training_results.json", results)
    return results


def run_decisions(args, root, partition, utilities, seed_evaluation):
    from e0924_export import export_all, file_sha256, load_export
    from e0924_decision import fit_router, metrics, score_router, FREQ_NAMES, COLOR_NAMES
    from run_g25 import TEST_DS, CROSS_DS, VAL_DS
    import numpy as np

    results = read_json(root / "training_results.json")
    source = results.get("G27_Full", {})
    if source.get("primary_status", source.get("status")) != "OK" or not source.get("ckpt"):
        raise ValueError("G27_Full primary result is not ready; no substitution with another arm")
    checkpoint = source["ckpt"]
    folder = Path(source.get("artifact_dir", root / "training" / "G27_Full"))
    config = read_json(folder / "eval_config.json")
    export_root = root / "decision_exports"
    checkpoint_sha = file_sha256(checkpoint)
    export_manifest = export_root / "manifest.json"
    if export_manifest.exists():
        export_info = read_json(export_manifest)
        if export_info["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("Export checkpoint mismatch; use a new E0924 run")
        for name, checksum in export_info["files"].items():
            if file_sha256(export_root / name) != checksum:
                raise ValueError(f"Corrupt decision export: {name}")
    else:
        if export_root.exists():
            # A partial export is never silently reused.
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            export_root.rename(root / f"decision_exports_incomplete_{stamp}")
        summary = export_all(config, checkpoint, partition, export_root, utilities, TEST_DS, seed_evaluation)
        write_json(export_manifest, {"checkpoint": checkpoint, "checkpoint_sha256": checkpoint_sha,
                                    "datasets": summary, "freq_names": FREQ_NAMES, "color_names": COLOR_NAMES,
                                    "files": {p.name: file_sha256(p) for p in export_root.glob("*.npz")}})
    fit, holdout = load_export(export_root / "fit.npz"), load_export(export_root / "holdout.npz")
    if set(fit["path"]) & set(holdout["path"]) or set(fit["video_id"]) & set(holdout["video_id"]):
        raise ValueError("Calibration fit/holdout overlap")
    # Fit artifacts and record holdout results BEFORE reading final test arrays.
    models = {}
    decision_dir = root / "decisions"
    decision_dir.mkdir(exist_ok=True)
    for name, (kind, hidden) in DECISION_ARMS.items():
        artifact = fit_router(fit, kind, hidden, args.seed, args.router_steps, args.router_lr)
        artifact.update(checkpoint_sha256=checkpoint_sha, reject_threshold=args.reject_threshold,
                        calibration_metadata_sha256=partition["metadata_sha256"])
        models[name] = artifact
        write_json(decision_dir / f"{name}_router.json", artifact)
        write_json(decision_dir / f"{name}_holdout.json",
                   metrics(holdout, score_router(artifact, holdout, args.reject_threshold)))
    report = {name: {"status": "OK", "testall": {}} for name in ("CLS", "evidence", "D0_fixed", *models)}
    for dataset in TEST_DS:
        data = load_export(export_root / f"test_{dataset}.npz")
        if set(data["path"]) & (set(fit["path"]) | set(holdout["path"])):
            raise ValueError(f"Calibration/test frame overlap in {dataset}")
        scores = {"CLS": data["cls_prob"], "evidence": data["evidence_prob"], "D0_fixed": data["gated_prob"]}
        scores.update({name: score_router(artifact, data, args.reject_threshold) for name, artifact in models.items()})
        for name, predictions in scores.items():
            report[name]["testall"][dataset] = metrics(data, predictions)
        np.savez_compressed(decision_dir / f"{dataset}_scores.npz", labels=data["labels"],
                            video_id=data["video_id"], path=data["path"], **scores)
    for item in report.values():
        scores = item["testall"]
        item["AUC_cross"] = (scores[VAL_DS]["video_auc"]+scores["DFDC"]["video_auc"])/2
        item["video_auc_seven_mean"] = sum(scores[d]["video_auc"] for d in CROSS_DS)/len(CROSS_DS)
        item["AUC_cross_fullpath"] = (scores[VAL_DS]["video_auc_fullpath"]+scores["DFDC"]["video_auc_fullpath"])/2
        item["video_auc_seven_mean_fullpath"] = sum(scores[d]["video_auc_fullpath"] for d in CROSS_DS)/len(CROSS_DS)
        item["metric_protocol"] = "video_auc: project legacy basename grouping; *_fullpath: collision-safe grouping"
    write_json(decision_dir / "results.json", report)
    return {"status": "OK", "results": str(decision_dir / "results.json"),
            "checkpoint_sha256": checkpoint_sha}


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_epochs < 0 or args.router_steps < 1 or args.attention_samples < 0:
        parser.error("Invalid epoch/step/attention budget")
    if not math.isfinite(args.router_lr) or args.router_lr <= 0 or not .5 <= args.reject_threshold <= 1:
        parser.error("Require finite positive router_lr and reject_threshold in [.5,1]")
    if args.dry_run:
        print(json.dumps(plan(args.seed, args.n_epochs), indent=2))
        return 0
    try:
        partition = preflight(args)
    except (OSError, KeyError, ValueError) as exc:
        print(f"E0924 preflight FAILED: {exc}", file=sys.stderr)
        return 2
    if args.preflight:
        print(json.dumps({"status": "OK", "metadata_sha256": partition["metadata_sha256"],
                          "fit_videos": len(partition["records"]["fit"]),
                          "holdout_videos": len(partition["records"]["holdout"]),
                          "note": "metadata only; image files, dependencies and GPU not checked"}, indent=2))
        return 0
    expected = signature(args, partition)
    if args.resume:
        root = args.resume.resolve()
        if read_json(root / "manifest.json")["signature"] != expected:
            raise ValueError("Resume signature mismatch (source/config/data changed); start a new run")
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root = Path(args.output_dir).resolve() / f"seed{args.seed}_{stamp}_{os.getpid()}"
        root.mkdir(parents=True, exist_ok=False)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        write_json(root / "manifest.json", {"signature": expected, "arguments": vars(args),
                                            "git_head": revision.stdout.strip() if revision.returncode == 0 else None})
        write_json(root / "calibration_partition.json", partition)
    summary = {"status": "RUNNING", "root": str(root)}
    write_json(root / "status.json", summary)
    try:
        adapter, utilities, versions = runtime(args)
        write_json(root / "runtime.json", versions)
        if "train" in args.stages:
            summary["training"] = run_training(args, root, adapter)
        elif (root / "training_results.json").exists():
            summary["training"] = read_json(root / "training_results.json")
        if "decisions" in args.stages:
            try:
                summary["decisions"] = run_decisions(args, root, partition, utilities, adapter.seed_evaluation)
            except Exception as exc:
                summary["decisions"] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        failed = any(r.get("status") != "OK" for r in summary.get("training", {}).values())
        failed |= summary.get("decisions", {}).get("status", "OK") != "OK"
        summary["status"] = "FAILED" if failed else "OK"
    except Exception as exc:
        summary.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    write_json(root / "status.json", summary)
    print(f"E0924 {summary['status']}: {root}")
    return 0 if summary["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())

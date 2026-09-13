"""One isolated G21 A/B/C/D matrix: same data, complete windows and compute budget."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from g21.config import ARMS, config_digest, load_config, resolve_arm
from g21.io import atomic_json, source_digest


def aggregate(run_dir: str | Path) -> dict:
    root = Path(run_dir).resolve()
    plan = json.loads((root / "run_plan.json").read_text(encoding="utf-8"))
    if (
        not plan.get("arms")
        or any(a not in ARMS for a in plan["arms"])
        or len(set(plan["arms"])) != len(plan["arms"])
        or not plan.get("seeds")
        or any(type(s) is not int or not 0 <= s < 2**32 for s in plan["seeds"])
        or len(set(plan["seeds"])) != len(plan["seeds"])
    ):
        raise ValueError("invalid aggregate run plan")
    records = []
    for seed in plan["seeds"]:
        initial = set()
        budgets = set()
        for arm in plan["arms"]:
            path = root / f"seed_{seed}" / arm / "result.json"
            row = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {"status": "MISSING", "exp_name": f"G21/{arm}", "seed": seed}
            )
            if row.get("status") in ("OK", "TRAINED"):
                if (
                    row.get("input_digest") != plan["input_digest"]
                    or row.get("protocol") != plan["protocol"]
                    or row.get("code_digest") != plan["code_digest"]
                    or row.get("config_digest")
                    != plan["arm_config_digests"][f"{seed}/{arm}"]
                    or row.get("seed") != seed
                    or row.get("exp_name") != f"G21/{arm}"
                ):
                    raise ValueError("completed result input/protocol mismatch")
                initial.add(row["initial_state_digest"])
                budgets.add(
                    (
                        row["completed_updates"],
                        row["counters"]["forward_images"],
                        row["counters"]["backward_images"],
                    )
                )
            records.append(row)
        if len(initial) > 1:
            raise ValueError(f"initial model weights differ across arms at seed {seed}")
        if len(budgets) > 1:
            raise ValueError(f"compute budgets differ across arms at seed {seed}")
    summary = {}
    for arm in plan["arms"]:
        rows = [
            r
            for r in records
            if r["exp_name"] == f"G21/{arm}" and r.get("status") == "OK"
        ]
        datasets = sorted({d for r in rows for d in r.get("testall", {})})
        summary[arm] = {}
        for dataset in datasets:
            values = [
                r["testall"][dataset]["video_auc"]
                for r in rows
                if dataset in r["testall"]
            ]
            summary[arm][dataset] = {
                "n_seeds": len(values),
                "mean_video_auc": statistics.mean(values),
                "seed_std": statistics.stdev(values) if len(values) > 1 else None,
            }
    result = {
        "protocol": plan["protocol"],
        "records": records,
        "seed_summary": summary,
        "all_completed": all(r["status"] in ("OK", "TRAINED") for r in records),
    }
    atomic_json(root / "all_results.json", records)
    atomic_json(root / "aggregate.json", result)
    return result


def run_matrix(args) -> int:
    from g21.preflight import inspect_inputs

    cfg = load_config(args.config)
    if len(set(args.arms)) != len(args.arms) or any(a not in ARMS for a in args.arms):
        raise ValueError("arms must be distinct members of A B C D")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("duplicate seeds are not independent runs")
    for arm in args.arms:
        for seed in args.seeds:
            resolve_arm(cfg, arm, seed)
    _, _, _, report = inspect_inputs(cfg)
    if args.resume and not args.run_dir:
        raise ValueError("--resume requires an explicit --run-dir")
    generated = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    )
    root = Path(args.run_dir or Path(cfg["paths"]["output_root"]) / generated).resolve()
    plan = {
        "schema_version": 1,
        "experiment": "G21",
        "protocol": cfg["protocol"],
        "arms": args.arms,
        "seeds": args.seeds,
        "config_digest": config_digest(cfg),
        "input_digest": report["input_digest"],
        "code_digest": source_digest(),
        "arm_config_digests": {
            f"{seed}/{arm}": config_digest(resolve_arm(cfg, arm, seed))
            for seed in args.seeds
            for arm in args.arms
        },
    }
    if args.resume:
        old = json.loads((root / "run_plan.json").read_text(encoding="utf-8"))
        if old != plan:
            raise ValueError(
                "run plan changed; resume cannot change data/config/arms/seeds"
            )
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "run_plan.json", plan)
        atomic_json(root / "config.base.json", cfg)
        atomic_json(root / "preflight.json", report)
    failures = False
    for seed in args.seeds:
        for arm in args.arms:
            out = root / f"seed_{seed}" / arm
            result_path = out / "result.json"
            if args.resume and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    result.get("status") in ("OK", "TRAINED")
                    and (out / "checkpoints/last.pth").exists()
                ):
                    if (
                        result.get("config_digest")
                        != plan["arm_config_digests"][f"{seed}/{arm}"]
                        or result.get("input_digest") != plan["input_digest"]
                        or result.get("code_digest") != plan["code_digest"]
                    ):
                        raise ValueError(
                            "completed arm cannot be reused with different config/data/code"
                        )
                    continue
            arm_cfg = resolve_arm(cfg, arm, seed)
            config_path = root / "configs" / f"{seed}_{arm}.json"
            atomic_json(config_path, arm_cfg)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "train",
                "--config",
                str(config_path),
                "--output",
                str(out),
            ]
            if args.resume and out.exists():
                command.append("--resume")
            log = root / f"worker_{seed}_{arm}.log"
            print(f"G21/{arm} seed={seed}: {log}", flush=True)
            with log.open("a" if args.resume else "w", encoding="utf-8") as stream:
                process = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=False
                )
            if process.returncode != 0:
                failures = True
                out.mkdir(parents=True, exist_ok=True)
                if not result_path.exists():
                    atomic_json(
                        result_path,
                        {
                            "status": "TRAIN_FAILED",
                            "exp_name": f"G21/{arm}",
                            "seed": seed,
                            "returncode": process.returncode,
                            "worker_log": str(log),
                        },
                    )
            aggregate(root)
    final = aggregate(root)
    print(
        json.dumps(
            {"run_dir": str(root), "all_completed": final["all_completed"]},
            ensure_ascii=False,
        )
    )
    return 1 if failures or not final["all_completed"] else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--config", required=True)
    pre.add_argument("--output")
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--arms", nargs="+", default=list(ARMS))
    run.add_argument("--seeds", nargs="+", type=int, default=[1024])
    run.add_argument("--run-dir")
    run.add_argument("--resume", action="store_true")
    train = sub.add_parser("train", help="isolated worker, normally invoked by run")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument(
        "--config", help="required for a plain historical state_dict checkpoint"
    )
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--data-root")
    evaluate.add_argument("--device")
    agg = sub.add_parser("aggregate")
    agg.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            from g21.preflight import inspect_inputs

            _, _, _, result = inspect_inputs(load_config(args.config))
            if args.output:
                atomic_json(args.output, result)
        elif args.command == "run":
            return run_matrix(args)
        elif args.command == "train":
            from g21.engine import train_arm

            result = train_arm(
                load_config(args.config), args.output, resume=args.resume
            )
        elif args.command == "evaluate":
            from g21.engine import evaluate_checkpoint

            result = evaluate_checkpoint(
                args.checkpoint,
                args.output,
                config=load_config(args.config) if args.config else None,
                data_root=args.data_root,
                device=args.device,
            )
        else:
            result = aggregate(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print(
            "G21 interrupted; resume from an existing complete last.pth",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

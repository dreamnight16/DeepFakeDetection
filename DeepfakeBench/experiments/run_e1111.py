"""Run the original B0 once, then E1111A and PUCE on one or two GPUs."""

import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from e1111_paths import normalize_arguments, relocated

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu_a", "--gpu-a", type=int, default=0, help="Physical CUDA device ordinal for A")
    parser.add_argument("--gpu_b", "--gpu-b", type=int, default=0, help="Physical CUDA device ordinal for B0/B")
    parser.add_argument("--base_run", type=Path, help="Reuse a successful new E1111 native-B0 run directory")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1111/all")
    parser.add_argument("--seed", type=int, choices=[1024], default=1024)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sampler_real_ratio", type=float, default=.3)
    parser.add_argument("--prior", type=float, default=.3)
    parser.add_argument("--lams", type=float, nargs="+", default=[0., .01, .05, .1])
    parser.add_argument("--feature_batch_size", type=int, default=8)
    parser.add_argument("--dataset_json_folder", type=Path)
    parser.add_argument("--rgb_root", type=Path)
    parser.add_argument("--clip_pretrained_path", type=Path)
    parser.add_argument("--pair_map", type=Path)
    parser.add_argument("--skip_d8", action="store_true")
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    return parser


def validate_arguments(parser, args):
    import run_e1111b

    if min(args.gpu_a, args.gpu_b) < 0 or args.feature_batch_size < 1:
        parser.error("GPU indices must be nonnegative; feature_batch_size must be positive")
    # Reuse the exact validation of the existing training runner.
    run_e1111b.validate_arguments(parser, argparse.Namespace(**{
        **vars(args), "arms": ["B0", "PUCE"]}))


def training_command(args, arms, output):
    command = [sys.executable, "-u", str(SCRIPTS / "run_e1111b.py"), "--arms", arms,
               "--output_dir", str(output), "--seed", str(args.seed),
               "--n_epochs", str(args.n_epochs), "--batch_size", str(args.batch_size),
               "--sampler_real_ratio", str(args.sampler_real_ratio),
               "--prior", str(args.prior), "--lams", *map(str, args.lams)]
    for key in ("dataset_json_folder", "rgb_root", "clip_pretrained_path"):
        if getattr(args, key) is not None:
            command.extend([f"--{key}", str(getattr(args, key))])
    return command


def a_command(args, base_run, output):
    command = [sys.executable, "-u", str(SCRIPTS / "run_e1111a.py"),
               "--base_run", str(base_run), "--output_dir", str(output),
               "--seed", str(args.seed), "--batch_size", str(args.feature_batch_size),
               "--device", "cuda:0"]
    for key in ("dataset_json_folder", "rgb_root", "clip_pretrained_path", "pair_map"):
        if getattr(args, key) is not None:
            command.extend([f"--{key}", str(getattr(args, key))])
    if args.skip_d8:
        command.append("--skip_d8")
    return command


def plan(args):
    return {
        "experiment": "E1111", "baseline": "original effort + CE; no warm start",
        "B0": {"action": "reuse" if args.base_run else "retrain once",
               "gpu": args.gpu_b, "base_run": str(args.base_run) if args.base_run else None},
        "schedule_after_B0": "parallel" if args.gpu_a != args.gpu_b else "A then B, sequential",
        "A": {"gpu": args.gpu_a, "base": "the same successful native B0",
              "command": a_command(args, args.base_run or "<new_B0_run>", "<output>/A")},
        "B": {"gpu": args.gpu_b, "arms": "PUCE only; does not train B0 again",
              "command": training_command(args, "PUCE", "<output>/B")},
        "B0_command": None if args.base_run else training_command(args, "B0", "<output>/B0"),
        "note": "GPU arguments are physical CUDA ordinals, replacing the outer mask; each child sees its card as cuda:0",
    }


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def owning_default(relative_file, key, manifest):
    """Read an omitted legacy path default without importing any ML code."""
    path = ROOT / relative_file
    if manifest.get("source_sha256", {}).get(relative_file) != file_sha256(path):
        raise ValueError(f"Cannot infer the old B0's omitted {key}: owning source changed")
    values = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == key):
            try:
                value = ast.literal_eval(node.args[1])
            except (ValueError, TypeError):
                continue
            if isinstance(value, str):
                values.add(value)
    if len(values) != 1:
        raise ValueError(f"Cannot uniquely resolve {key} from the B0 owning source")
    return values.pop()


def validate_base_run(path, args):
    path = Path(relocated(path)).resolve()
    report = json.loads((path / "all_results.json").read_text(encoding="utf-8-sig"))
    baseline = report.get("arms", {}).get("B0", {})
    if (report.get("experiment") != "E1111" or report.get("branch") != "E1111B"
            or baseline.get("status") != "OK"
            or baseline.get("settings") != {"model_name": "effort"}):
        raise ValueError("base_run must contain a successful original E1111 B0, not CE_L0/PUCE")
    checkpoint = Path(relocated(baseline["ckpt"]))
    config = json.loads((path / "B0/train_config.json").read_text(encoding="utf-8-sig"))
    expected = {"model_name": "effort", "manualSeed": args.seed, "nEpochs": args.n_epochs,
                "train_batchSize": args.batch_size, "sampler_real_ratio": args.sampler_real_ratio,
                "compression": "c23", "frame_num": {"train": 8, "test": 8}}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Reused B0 differs from requested training conditions: {key}")
    if file_sha256(checkpoint) != baseline.get("checkpoint_sha256"):
        raise ValueError("B0 checkpoint SHA does not match its result")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8-sig"))
    metadata = Path(relocated(args.dataset_json_folder or config["dataset_json_folder"])).resolve()
    data_hashes = manifest.get("dataset_sha256", {})
    if "FaceForensics++" not in data_hashes:
        raise ValueError("B0 manifest has no FF++ dataset identity")
    for dataset, expected_sha in data_hashes.items():
        if file_sha256(metadata / f"{dataset}.json") != expected_sha:
            raise ValueError(f"Requested metadata differs from B0: {dataset}")
    effective = {"dataset_json_folder": metadata}
    for argument, config_key, owner in (
        ("rgb_root", "rgb_root_override", "training/dataset/abstract_dataset.py"),
        ("clip_pretrained_path", "clip_pretrained_path", "training/detectors/effort_detector.py"),
    ):
        original = config.get(config_key) or owning_default(owner, config_key, manifest)
        original = Path(relocated(original)).resolve()
        supplied = getattr(args, argument)
        if supplied is not None and Path(relocated(supplied)).resolve() != original:
            raise ValueError(f"Requested {argument} differs from B0; use its original path or retrain B0")
        if not original.is_dir():
            raise FileNotFoundError(f"B0 {argument} is absent: {original}")
        effective[argument] = original
    # Both A and B receive these same effective paths, including on a reused B0.
    for key, value in effective.items():
        setattr(args, key, value)
    return path


def only_child_run(folder):
    reports = sorted(Path(folder).glob("seed*/all_results.json"))
    if len(reports) != 1:
        raise ValueError(f"Expected exactly one child run under {folder}, found {len(reports)}")
    return reports[0].parent.resolve()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def stop_process(process):
    if os.name == "posix":
        def group_alive():
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False
        try:
            os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while group_alive() and time.monotonic() < deadline:
                process.poll()  # Reap the leader, but keep checking its descendants.
                time.sleep(.1)
            if group_alive():
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        except ProcessLookupError:
            process.poll()
        return
    if process.poll() is not None:
        return
    try:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    except ProcessLookupError:
        pass


def run(args):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = args.output_dir.resolve() / f"seed{args.seed}_{stamp}_{os.getpid()}"
    output.mkdir(parents=True, exist_ok=False)
    state = {"experiment": "E1111", "status": "RUNNING", "plan": plan(args),
             "output": str(output), "jobs": {}}
    active, previous_handlers = {}, {}

    def save():
        write_json(output / "all_results.json", state)

    def launch(name, command, gpu):
        log = output / f"{name}.log"
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                           LOCAL_RANK="0", RANK="0", WORLD_SIZE="1", LOCAL_WORLD_SIZE="1")
        with log.open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=os.name == "posix")
        active[name] = process
        state["jobs"][name] = {"status": "RUNNING", "gpu": gpu, "pid": process.pid,
                               "command": command, "log": str(log)}
        print(f"E1111/{name}: PID={process.pid}, GPU={gpu}, log={log}", flush=True)
        save()

    def collect(name):
        code = active[name].poll()
        if code is None:
            return False
        if code != 0:
            stop_process(active[name])  # A killed runner may still have a train.py in its PGID.
        state["jobs"][name].update(status="OK" if code == 0 else "FAILED", exit_code=code)
        print(f"E1111/{name}: exit={code}", flush=True)
        active.pop(name)
        save()
        return True

    def wait_for(names):
        while any(name in active for name in names):
            for name in names:
                if name in active:
                    collect(name)
            if any(name in active for name in names):
                time.sleep(2)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    print(f"E1111 output: {output}", flush=True)
    save()
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        if args.base_run:
            base_run = validate_base_run(args.base_run, args)
            state["jobs"]["B0"] = {"status": "REUSED", "base_run": str(base_run)}
            save()
        else:
            launch("B0", training_command(args, "B0", output / "B0"), args.gpu_b)
            wait_for(["B0"])
            if state["jobs"]["B0"]["exit_code"] != 0:
                state["status"] = "B0_FAILED"
                return 1
            base_run = validate_base_run(only_child_run(output / "B0"), args)
            state["jobs"]["B0"]["base_run"] = str(base_run)
            save()
        state["plan"] = plan(args)
        state["effective_paths"] = {name: str(getattr(args, name)) for name in (
            "dataset_json_folder", "rgb_root", "clip_pretrained_path")}
        save()
        # A receives this newly trained (or explicitly reused) native B0.
        launch("A", a_command(args, base_run, output / "A"), args.gpu_a)
        if args.gpu_a == args.gpu_b:
            wait_for(["A"])
        launch("B", training_command(args, "PUCE", output / "B"), args.gpu_b)
        wait_for(["A", "B"])
        for name in ("A", "B"):
            try:
                child = only_child_run(output / name)
                state["jobs"][name]["run_dir"] = str(child)
                report = json.loads((child / "all_results.json").read_text(encoding="utf-8-sig"))
                if "arms" in report:
                    arms = {arm: row.get("status") for arm, row in report["arms"].items()}
                    state["jobs"][name]["arms"] = arms
                    state["jobs"][name]["experiment_status"] = (
                        "OK" if arms and all(value == "OK" for value in arms.values()) else "FAILED")
                else:
                    state["jobs"][name]["experiment_status"] = report.get("status")
                expected = "DEVELOPMENT_COMPLETE" if name == "A" else "OK"
                state["jobs"][name]["result_valid"] = (
                    state["jobs"][name]["experiment_status"] == expected)
                if not state["jobs"][name]["result_valid"]:
                    state["jobs"][name]["status"] = "FAILED"
            except (OSError, ValueError) as exc:
                state["jobs"][name].update(status="FAILED", result_valid=False, report_error=str(exc))
        success = all(state["jobs"][name]["exit_code"] == 0
                      and state["jobs"][name]["result_valid"] for name in ("A", "B"))
        state["status"] = "OK" if success else "COMPLETED_WITH_FAILURES"
        return 0 if success else 1
    except KeyboardInterrupt:
        for name, process in list(active.items()):
            stop_process(process)
            state["jobs"][name]["status"] = "INTERRUPTED"
            active.pop(name)
        state["status"] = "INTERRUPTED"
        return 130
    except Exception as exc:
        for name, process in list(active.items()):
            stop_process(process)
            state["jobs"][name]["status"] = "STOPPED"
            active.pop(name)
        state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        save()
        print(f"E1111 summary: {output / 'all_results.json'}", flush=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    # On a reused B0, retain its effective paths until validate_base_run can
    # verify and relocate them. Fresh runs use the current project/data roots.
    if not args.base_run:
        normalize_arguments(args, ROOT)
    else:
        args.base_run = Path(relocated(args.base_run))
        args.output_dir = Path(relocated(args.output_dir))
    if args.dry_run:
        print(json.dumps(plan(args), ensure_ascii=False, indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

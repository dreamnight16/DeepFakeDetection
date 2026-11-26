"""Queue all 46 E1119 trainings and their dependent read-only analyses."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from e1111_paths import input_file, normalize_arguments
from g21.io import atomic_json, digest, file_digest
from run_g25 import TEST_DS, TRAIN_DS, VAL_DS


ARMS = ("B00", "B01", "B02", "B03", "B04", "M01", "M02", "M03", "M04", "M05", "M06",
        "P01", "P02", "P03", "P04", "T01", "T02", "T03", "I01", "I02", "I03", "I04", "I05")
SEEDS = (1024, 1025)
PROTECTED = ("P01", "P02", "P03", "P04")
SEGMENT = ("T01", "T02")
SPECTRAL = ("M01", "M02", "M04", "M05", "M06")
MECHANISM = ("B00", *PROTECTED, *SEGMENT, "I01", "I02", "I03", "I04", "I05")
DRIFT = ("B00", *PROTECTED, *SEGMENT, "T03")


def plan():
    jobs = {f"asset:{name}": [] for name in ("spectral", "random", "activation")}
    jobs["D01:CLIP"] = []
    for arm in ARMS:
        for seed in SEEDS:
            tag = f"{arm}:{seed}"
            train = f"train:{tag}"
            dependency = ("spectral" if arm in SPECTRAL else "random" if arm == "P02" else
                          "activation" if arm in PROTECTED else None)
            jobs[train] = [f"asset:{dependency}"] if dependency else []
            jobs[f"eval:{tag}"] = [train]
            if arm in DRIFT:
                jobs[f"D03:{tag}"] = [train]
            if arm in MECHANISM:
                jobs[f"mechanism:{tag}"] = [train]
            if arm == "B00":
                for point in ("epoch_0", "epoch_5", "epoch_10", "best"):
                    jobs[f"D01:{tag}:{point}"] = [train]
                    jobs[f"D02:{tag}:{point}"] = [train]
    return jobs


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output_dir", type=Path, default=ROOT / "experiment_results/E1119")
    cli.add_argument("--dataset_json_folder", type=Path)
    cli.add_argument("--rgb_root")
    cli.add_argument("--clip_pretrained_path")
    cli.add_argument("--gpus", help="Comma-separated physical CUDA device IDs. Default: all visible GPUs.")
    cli.add_argument("--feature_batch_size", type=int, default=8, help="Batch size for the offline covariance pass only")
    cli.add_argument("--historical_b0", type=Path, help="Optional historical seed-1024 baseline and provenance JSON")
    cli.add_argument("--plan", action="store_true", help="Print the dependency graph without running an experiment")
    cli.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    return cli


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def input_identity(args, training, evaluation):
    import experiment_utils as utilities
    from run_e1021 import seed_everything

    clip = Path(args.clip_pretrained_path)
    rgb = Path(args.rgb_root)
    if not clip.is_dir() or not rgb.is_dir():
        raise FileNotFoundError(f"Expected local CLIP and RGB directories, got {clip} and {rgb}")
    clip_files = [p for p in clip.rglob("*") if p.is_file() and ".git" not in p.parts and (
        p.suffix in (".safetensors", ".bin") or p.name == "config.json" or p.name.endswith(".index.json"))]
    if not any(p.suffix in (".safetensors", ".bin") for p in clip_files):
        raise FileNotFoundError(f"No local CLIP model weights under {clip}")
    indices = {}
    for name in TEST_DS:
        path = args.dataset_json_folder / f"{name}.json"
        indices[name] = file_digest(path) if path.is_file() else None

    jobs = [(f"{TRAIN_DS}:train", training, None),
            *((name, evaluation, name) for name in TEST_DS)]
    rgb_hashes, sections = {}, {}
    for tag, config, dataset in jobs:
        try:
            seed_everything(config["manualSeed"])
            source = (utilities.get_train_loader(config) if dataset is None
                      else utilities.get_data_loader(config, dataset)).dataset
        except Exception as exc:
            if dataset in (None, VAL_DS):
                raise
            sections[tag] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        rows = []
        for raw, label in sorted(zip(source.image_list, source.label_list)):
            path = input_file(raw, rgb)
            if path not in rgb_hashes:
                rgb_hashes[path] = file_digest(path) if path.is_file() else None
            rows.append((str(raw).replace(chr(92), "/"), int(label), rgb_hashes[path]))
        sections[tag] = {"sha256": digest(rows), "frames": len(rows),
                         "missing_files": sum(row[-1] is None for row in rows)}
    hashes = {"dataset_json_sha256": indices, "native_rgb": sections,
              "clip_sha256": {str(p.relative_to(clip)): file_digest(p) for p in sorted(clip_files)}}
    return {"input_identity": digest(hashes), **hashes,
            "paths": {"rgb": str(rgb), "metadata": str(args.dataset_json_folder), "clip": str(clip)}}


def source_identity():
    owned = [*Path(__file__).parent.glob("*e1119*.py"),
             *(ROOT / "training/detectors").glob("*e1119*.py"),
             ROOT / "training/e1119_record.py", ROOT / "training/e1119_model.py"]
    inherited = [
        "training/train.py", "training/trainer/trainer_v2.py", "training/detectors/effort_detector.py",
        "training/dataset/abstract_dataset.py", "training/dataset/balance_batch_sampler.py",
        "training/config/detector/effort.yaml", "training/config/train_config.yaml",
        "training/config/test_config.yaml", "training/metrics/utils.py",
        "experiments/run_e1115.py", "experiments/run_g25.py", "experiments/run_e1021.py",
        "experiments/run_e1001.py", "experiments/experiment_utils.py", "experiments/e0924_export.py",
        "experiments/e0924_decision.py", "experiments/e0924_protocol.py", "experiments/e1111_paths.py",
    ]
    return {str(p.relative_to(ROOT)): file_digest(p) for p in sorted(owned + [ROOT / p for p in inherited])}


def worker_train(spec, destination, training):
    import yaml

    input_config = destination / "train_config.json"
    with (destination / "train.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(training, stream, allow_unicode=True)
    run = subprocess.run([sys.executable, str(ROOT / "training/train.py"),
                          "--detector_path", str(destination / "train.yaml"),
                          "--train_dataset", TRAIN_DS, "--test_dataset", VAL_DS,
                          "--dataset_json_folder", training["dataset_json_folder"],
                          "--data_root", training["rgb_root_override"]], cwd=ROOT)
    if run.returncode:
        raise RuntimeError(f"training/train.py exited with code {run.returncode}")
    best = read(destination / "best.json")
    for path in (Path(best["checkpoint"]), *(destination / f"epoch_{i}.pth" for i in (0, 5, 10))):
        if not path.is_file():
            raise FileNotFoundError(path)
    epochs = [read(destination / f"epoch_{i}.json") for i in range(11)]
    atomic_json(destination / "training.json", {
        **best, "checkpoint_sha256": file_digest(best["checkpoint"]),
        "training_wall_seconds": sum(epoch["wall_seconds"] for epoch in epochs),
        "peak_allocated_bytes": max(epoch["peak_allocated_bytes"] for epoch in epochs),
        "config_sha256": file_digest(input_config),
        "asset_manifest_sha256": (file_digest(spec["asset_folder"] / "complete.json")
                                  if spec.get("asset_folder") else None),
    })


def configs(spec, destination, *, model_override=None):
    root = Path(spec["run"])
    base = root / "baseline" / str(spec.get("seed", 1024))
    training, evaluation = (read(base / f"{phase}.json") for phase in ("train", "eval"))
    model = {}
    arm = spec.get("arm")
    if model_override is not None:
        model = model_override
    elif arm and arm != "B00":
        model.update(model_name="effort_e1119", e1119_arm=arm)
    if spec.get("asset_folder"):
        kind = spec["asset_folder"] = Path(spec["asset_folder"])
        subfolder = ("haar" if arm == "M06" else "svd") if arm in SPECTRAL else arm
        model["e1119_assets"] = str(kind / subfolder)
    if spec["kind"] == "train":
        for phase, cfg in (("train", training), ("eval", evaluation)):
            updated = {**cfg, **model}
            if phase == "train":
                updated.update(log_dir=str(destination / "native_logs"), e1119_record_dir=str(destination))
            atomic_json(destination / f"baseline_{phase}.json", cfg)
            atomic_json(destination / f"{phase}_config.json", updated)
            atomic_json(destination / f"{phase}_diff.json", {
                key: {"source": cfg.get(key), "actual": value}
                for key, value in updated.items() if key not in cfg or value != cfg[key]
            })
        return read(destination / "train_config.json"), read(destination / "eval_config.json")
    return training, {**evaluation, **model}


def worker(spec_path):
    spec = read(spec_path)
    destination = spec_path.parent
    try:
        import torch

        torch.cuda.set_device(0)
        kind = spec["kind"]
        training, evaluation = configs(spec, destination)
        result = {}
        if kind == "train":
            worker_train(spec, destination, training)
        elif kind == "asset":
            from e1119_assets import build_assets

            build_assets(spec["point"], destination, training, spec["feature_batch_size"], "cuda:0")
        else:
            from types import SimpleNamespace

            import experiment_utils as utilities
            from e1119_assets import frozen_vision
            from e1119_eval import evaluate

            if spec.get("training_folder"):
                training_folder = Path(spec["training_folder"])
                evaluation = read(training_folder / "eval_config.json")
                point = spec.get("point", "best")
                checkpoint = (read(training_folder / "training.json")["checkpoint"] if point == "best"
                              else training_folder / f"{point}.pth")
                model = utilities.load_model(evaluation, checkpoint)
                atomic_json(destination / "checkpoint.json", {"path": str(checkpoint), "sha256": file_digest(checkpoint)})
            else:
                model = SimpleNamespace(backbone=frozen_vision(evaluation, "cuda:0"))
                model.parameters = model.backbone.parameters
            if kind == "eval":
                result = evaluate(evaluation, model, destination, spec["arm"])
            else:
                from e1119_analysis import d01, d02, d03, mechanism, write_analysis

                if kind == "D01":
                    result = d01(model, training, evaluation, destination)
                elif kind == "D02":
                    result = d02(model, evaluation, destination)
                elif kind == "D03":
                    result = d03(model, frozen_vision(evaluation, "cuda:0"), evaluation)
                else:
                    result = mechanism(model, spec["arm"], evaluation)
                write_analysis(destination, result)
        unavailable = [name for name, row in result.get("datasets", {}).items() if row["status"] != "OK"]
        atomic_json(destination / "task.json", {"status": "PARTIAL" if unavailable else "OK",
                                                "unavailable": unavailable})
    except Exception as exc:
        traceback.print_exc()
        atomic_json(destination / "task.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


def task_dir(root, key):
    kind, arm, *parts = key.split(":")
    if kind in ("train", "eval"):
        return root / f"seed{parts[0]}" / arm / kind
    return root / "analyses" / kind / "_".join([arm, *parts]) if kind != "asset" else root / "assets" / arm


def task_status(root, key):
    folders = sorted(task_dir(root, key).glob("attempt_*"), reverse=True)
    for folder in folders:
        record = folder / "task.json"
        if record.is_file() and read(record)["status"] == "OK":
            if key.startswith(("D01:B00:", "D02:B00:")):
                requested = read(folder / "request.json")["training_folder"]
                current = task_status(root, "train:B00:" + key.split(":")[2])
                if current.get("folder") != Path(requested):
                    continue
            return {"status": "OK", "folder": folder}
    return {"status": "PENDING"}


def new_task(root, key, dependencies, jobs, args):
    destination = task_dir(root, key) / datetime.now().strftime("attempt_%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True)
    kind, arm, *parts = key.split(":")
    spec = {"kind": kind, "run": str(root)}
    if kind == "asset":
        spec.update(point=arm, feature_batch_size=args.feature_batch_size)
    elif arm != "CLIP":
        spec.update(arm=arm, seed=int(parts[0]))
        if kind != "train":
            train = jobs[f"train:{arm}:{parts[0]}"]["folder"]
            spec["training_folder"] = str(train)
            if len(parts) == 2:
                spec["point"] = parts[1]
        elif dependencies:
            spec["asset_folder"] = str(jobs[dependencies[0]]["folder"])
    atomic_json(destination / "request.json", spec)
    return destination


def save_queue(folder, jobs):
    atomic_json(folder / "queue.json", {
        key: {k: str(v) if isinstance(v, Path) else v for k, v in state.items()} for key, state in jobs.items()
    })


def ready_for(key, dependencies, jobs):
    if all(jobs[dep]["status"] == "OK" for dep in dependencies):
        return True
    # A completed snapshot can feed a probe while subsequent epochs train.
    point = key.split(":")[-1]
    if key.startswith(("D01:", "D02:")) and point.startswith("epoch_"):
        train = jobs[dependencies[0]]
        return train["status"] == "RUNNING" and (train["folder"] / f"{point}.json").is_file()
    return False


def schedule(folder, gpus, args, *, graph=None, initial=None, entrypoint=__file__, task_factory=new_task,
             validate_cached=None):
    if graph is None:
        graph = plan()
    jobs = {key: task_status(folder, key) for key in graph}
    jobs.update(initial or {})
    if validate_cached is not None:
        for key, dependencies in graph.items():
            if jobs[key]["status"] == "OK" and not validate_cached(key, dependencies, jobs):
                jobs[key] = {"status": "PENDING"}
    slots = {}
    try:
        while True:
            for gpu, (key, process) in list(slots.items()):
                if (code := process.poll()) is None:
                    continue
                destination = jobs[key]["folder"]
                status = destination / "task.json"
                state = read(status) if status.is_file() else {"status": "FAILED", "error": f"Worker exit {code}"}
                jobs[key].update(state)
                del slots[gpu]
                print(f"{key} {jobs[key]['status']}", flush=True)
            for key, dependencies in graph.items():
                if jobs[key]["status"] == "PENDING" and any(jobs[dep]["status"] in ("FAILED", "BLOCKED")
                                                             for dep in dependencies):
                    jobs[key].update(status="BLOCKED", dependencies=dependencies)
            free = [gpu for gpu in gpus if gpu not in slots]
            ready = [key for key, dependencies in graph.items()
                     if jobs[key]["status"] == "PENDING" and ready_for(key, dependencies, jobs)]
            for gpu, key in zip(free, ready):
                destination = task_factory(folder, key, graph[key], jobs, args)
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1"}
                with (destination / "worker.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen([sys.executable, str(Path(entrypoint).resolve()),
                                                "--worker", str(destination / "request.json")],
                                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=os.name != "nt")
                jobs[key].update(status="RUNNING", folder=destination)
                slots[gpu] = (key, process)
                print(f"{key} GPU {gpu} -> {destination}", flush=True)
            save_queue(folder, jobs)
            if not slots and not ready:
                break
            time.sleep(2)
    finally:
        for _, process in slots.values():
            if process.poll() is None:
                if os.name == "nt":
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
        for _, process in slots.values():
            process.wait()
    return jobs


@contextmanager
def lock(folder):
    # Admit one scheduler per input/config identity.
    with (folder / "scheduler.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if args.plan:
        print(json.dumps({"experiment": "E1119", "training_runs": 46, "seeds": SEEDS,
                          "epochs": list(range(11)), "test_datasets": TEST_DS, "jobs": plan()}, indent=2))
        return 0
    if args.feature_batch_size < 1:
        cli.error("--feature_batch_size must be positive")
    import torch
    import experiment_utils as utilities
    from e1119_report import report
    from run_e1115 import build_training_config

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = args.gpus or visible or ",".join(map(str, range(torch.cuda.device_count())))
    gpus = [device.strip() for device in gpus.split(",") if device.strip()]
    if not gpus or gpus == ["-1"] or len(set(gpus)) != len(gpus):
        cli.error("E1119 requires at least one unique CUDA GPU")
    normalize_arguments(args, ROOT)
    args.output_dir = args.output_dir.resolve()
    args.rgb_root = str(Path(args.rgb_root).resolve())
    args.clip_pretrained_path = str(Path(args.clip_pretrained_path).resolve())
    baseline = {}
    for seed in SEEDS:
        args.seed, args.n_epochs, args.batch_size, args.sampler_real_ratio = seed, 10, 32, .3
        baseline[seed] = build_training_config(args, "B0_NATIVE", args.output_dir, utilities, steps=None)
    if args.historical_b0:
        history = read(args.historical_b0)
        if history.get("seed") != 1024 or not history.get("source"):
            cli.error("--historical_b0 expects a seed-1024 record with a source")
    print("Hashing the original image lists, dataset indices, and CLIP weights.", flush=True)
    material = input_identity(args, *baseline[1024])
    sources = source_identity()
    identity = digest([material, sources, baseline, args.feature_batch_size])
    folder = args.output_dir / identity[:20]
    folder.mkdir(parents=True, exist_ok=True)
    with lock(folder):
        atomic_json(folder / "inputs.json", material)
        atomic_json(folder / "source_sha256.json", sources)
        atomic_json(folder / "plan.json", plan())
        for seed, (training, evaluation) in baseline.items():
            atomic_json(folder / "baseline" / str(seed) / "train.json", training)
            atomic_json(folder / "baseline" / str(seed) / "eval.json", evaluation)
        jobs = schedule(folder, gpus, args)
        report(folder, ARMS, SEEDS, jobs, args.historical_b0)
    print(f"E1119 results: {folder}", flush=True)
    return int(any(job["status"] != "OK" for job in jobs.values()))


if __name__ == "__main__":
    sys.exit(main())

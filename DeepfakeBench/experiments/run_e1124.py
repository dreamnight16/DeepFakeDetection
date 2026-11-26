"""Run all 23 E1124 configurations at seed 1024 on the original E1119 data flow."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import traceback
from zipfile import BadZipFile

import run_e1119 as native

from g21.io import atomic_json, digest, file_digest
from run_g25 import TEST_DS, TRAIN_DS, VAL_DS


ROOT = native.ROOT
SEED = 1024
ARMS = ("B00", "P128F", "P128S", "V95", "V95R", "W128", "W128LR", "N0", "N1", "N2",
        "N3", "N4", "R1", "R2", "L1", "L2", "L3", "L4", "L5", "C1", "C2", "C3", "X1")
ROUTED = ("L1", "L2", "L3", "L4", "L5", "C1", "C2", "C3")
ASSET_DIRS = {
    "anchors": {"P01": "P01", "real": "P01", "mixed": "P04", "random": "random"},
    "real95": {"V95": "V95", "V95R": "V95R"},
    "paired": {f"N{i}": f"N{i}" for i in range(5)},
    "spectral": {"weight": "weight"},
}
HISTORICAL_ARMS = {"B00": "B00", "P128F": "P01"}
PROTOCOL_KEYS = ("head_initial_sha256", "loader_rng_sha256", "train_list_order_sha256",
                 "steps_per_epoch", "total_steps", "real_per_batch", "fake_per_batch")


def plan():
    jobs = {f"asset:{name}": [] for name in ASSET_DIRS}
    jobs["asset:real95"] = ["asset:anchors"]
    for arm in ARMS:
        assets = []
        if arm in ("P128F", "P128S", *ROUTED):
            assets.append("asset:anchors")
        if arm in ("V95", "V95R"):
            assets.append("asset:real95")
        if arm in ("W128", "W128LR"):
            assets.append("asset:spectral")
        if arm.startswith(("N", "C")) or arm == "X1":
            assets.append("asset:paired")
        tag = f"{arm}:{SEED}"
        jobs[f"train:{tag}"] = assets
        jobs[f"eval:{tag}"] = [f"train:{tag}"]
        if arm in ROUTED:
            jobs[f"mechanism:{tag}"] = [f"train:{tag}", f"eval:{tag}"]
    return jobs


def new_task(folder, key, dependencies, jobs, args):
    destination = native.new_task(folder, key, dependencies, jobs, args)
    request = destination / "request.json"
    spec = native.read(request)
    if spec["kind"] == "train":
        spec.pop("asset_folder", None)
        spec["asset_folders"] = {dep.split(":")[1]: str(jobs[dep]["folder"]) for dep in dependencies}
    if spec["kind"] == "mechanism":
        spec["evaluation_folder"] = str(jobs[dependencies[1]]["folder"])
    atomic_json(request, spec)
    return destination


def validate_cached(key, dependencies, jobs):
    job = jobs[key]
    if any(jobs[dep]["status"] != "OK" for dep in dependencies):
        return False
    if job.get("reused_from") or key.startswith("asset:"):
        return True
    try:
        request = native.read(job["folder"] / "request.json")
        if key.startswith("train:"):
            expected = {dep.split(":")[1]: str(jobs[dep]["folder"]) for dep in dependencies}
            return request["asset_folders"] == expected
        training = jobs[dependencies[0]]["folder"]
        checkpoint = native.read(job["folder"] / "checkpoint.json")
        if (Path(request["training_folder"]) != training or
                checkpoint["sha256"] != native.read(training / "training.json")["checkpoint_sha256"]):
            return False
        return not key.startswith("mechanism:") or Path(request["evaluation_folder"]) == jobs[dependencies[1]]["folder"]
    except (OSError, ValueError, KeyError):
        return False


def training_model(spec):
    arm = spec["arm"]
    if arm == "B00":
        return {}
    folders = {name: Path(path) for name, path in spec["asset_folders"].items()}
    if arm == "P128F":
        return {"model_name": "effort_e1119", "e1119_arm": "P01",
                "e1119_assets": str(folders["anchors"] / "P01")}
    assets = {kind: str(folder / leaf) for name, folder in folders.items()
              for kind, leaf in ASSET_DIRS[name].items()}
    return {"model_name": "effort_e1124", "e1124_arm": arm, "e1124_assets": assets}


def worker(request):
    destination = request.parent
    spec = native.read(request)
    run = Path(spec["run"])
    try:
        import torch

        torch.cuda.set_device(0)
        kind = spec["kind"]
        if kind == "train":
            model = training_model(spec)
            training, _ = native.configs(spec, destination, model_override=model)
            native.worker_train(spec, destination, training)
            manifests = {name: file_digest(Path(path) / "complete.json")
                         for name, path in spec["asset_folders"].items()}
            record = native.read(destination / "training.json")
            atomic_json(destination / "training.json", {**record, "asset_manifest_sha256_by_kind": manifests})
        elif kind == "asset":
            from e1124_assets import build

            training = native.read(run / "baseline" / str(SEED) / "train.json")
            build(spec["point"], destination, training, native.read(run / "references.json"),
                  spec["feature_batch_size"])
        else:
            import experiment_utils as utilities
            from e1119_eval import evaluate
            from e1124_analysis import fold_check, model_diagnostics, route_diagnostics

            trained = Path(spec["training_folder"])
            config = native.read(trained / "eval_config.json")
            record = native.read(trained / "training.json")
            checkpoint = Path(record["checkpoint"])
            checkpoint_sha = file_digest(checkpoint)
            if checkpoint_sha != record["checkpoint_sha256"]:
                raise ValueError(f"Training checkpoint changed: {checkpoint}")
            model = utilities.load_model(config, checkpoint)
            atomic_json(destination / "checkpoint.json", {"path": str(checkpoint), "sha256": checkpoint_sha})
            if kind == "eval":
                arm = spec["arm"]
                refs = (22,) if arm in ROUTED and arm != "L5" else ()
                result = evaluate(config, model, destination, arm, reference_blocks=refs)
                model_diagnostics(model, destination)
                if not refs and result["status"] == "OK":
                    fold_check(model, config, destination)
            else:
                result = route_diagnostics(model, config, destination, Path(spec["evaluation_folder"]))
            missing = [key for key, row in result["datasets"].items() if row["status"] != "OK"]
            if missing:
                atomic_json(destination / "task.json", {"status": "PARTIAL", "unavailable": missing})
                return 0
        atomic_json(destination / "task.json", {"status": "OK"})
    except Exception as exc:
        traceback.print_exc()
        atomic_json(destination / "task.json", {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


def archive_path(root, queue, key):
    item = queue[key]
    if item["status"] != "OK":
        raise ValueError(f"{key} did not finish in E1119")
    return native.task_dir(root, key) / Path(item["folder"]).name


def historical_jobs(root, queue, baseline, activation):
    import numpy as np

    reused = {}
    baseline_protocol, baseline_rows = None, {}
    for arm, old in HISTORICAL_ARMS.items():
        tag, old_tag = f"{arm}:{SEED}", f"{old}:{SEED}"
        try:
            train = archive_path(root, queue, f"train:{old_tag}")
            evaluation = archive_path(root, queue, f"eval:{old_tag}")
            for phase, expected in zip(("train", "eval"), baseline):
                previous = native.read(train / f"baseline_{phase}.json")
                if ({k: v for k, v in expected.items() if k != "log_dir"} !=
                        {k: v for k, v in previous.items() if k != "log_dir"}):
                    raise ValueError(f"native {phase} configuration differs")
            old_config = native.read(train / "eval_config.json")
            if arm == "P128F":
                if old_config["e1119_arm"] != "P01" or Path(old_config["e1119_assets"]).parent.name != activation.name:
                    raise ValueError("original P01 reference is a different activation attempt")
            training = native.read(train / "training.json")
            if training["selection_dataset"] != VAL_DS:
                raise ValueError("checkpoint selection dataset differs")
            if native.read(evaluation / "checkpoint.json")["sha256"] != training["checkpoint_sha256"]:
                raise ValueError("training and evaluation checkpoint records disagree")
            result = native.read(evaluation / "evaluation.json")
            if result["status"] != "OK" or set(result["datasets"]) != set(TEST_DS):
                raise ValueError("the eight-dataset evaluation is incomplete")
            protocol = native.read(train / "training_protocol.json")
            if (tuple(protocol[k] for k in PROTOCOL_KEYS[3:]) != (575, 6325, 10, 22) or
                    protocol["trainable_count"] != 788482):
                raise ValueError("archived training and optimizer budget differs")
            alignment = tuple(protocol[k] for k in PROTOCOL_KEYS)
            if arm != "B00" and alignment != baseline_protocol:
                raise ValueError("archived training protocol cannot be aligned to a reusable B00")
            identities = {}
            for dataset in TEST_DS:
                exports = (
                    ("videos", ("video_id", "label", "frame_count"), ("prob_fake",)),
                    ("frames", ("labels", "path", "video_id"), ("cls_prob", "global_log_odds")),
                )
                for level, identifiers, prediction in exports:
                    size = result["datasets"][dataset][f"num_{level}"]
                    with np.load(evaluation / dataset / f"{level}.npz", allow_pickle=False) as handle:
                        arrays = {key: handle[key] for key in identifiers}
                        if any(len(row) != size for row in arrays.values()) or any(
                                len(handle[key]) != size or not np.isfinite(handle[key]).all()
                                for key in prediction):
                            raise ValueError(f"{dataset} {level} export is incomplete")
                        identity = tuple(sha256(arrays[key].tobytes()).hexdigest() for key in identifiers)
                        if arm != "B00" and baseline_rows[dataset, level] != identity:
                            raise ValueError(f"{dataset} {level} rows differ from archived B00")
                        identities[dataset, level] = identity
            if arm == "B00":
                baseline_protocol, baseline_rows = alignment, identities
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            print(f"{arm} will train and evaluate under E1124: {exc}", flush=True)
            continue
        reused[f"train:{tag}"] = {"status": "OK", "folder": train, "reused_from": old_tag}
        reused[f"eval:{tag}"] = {"status": "OK", "folder": evaluation, "reused_from": old_tag}
    return reused


def main(argv=None):
    cli = native.parser()
    cli.description = __doc__
    cli.set_defaults(output_dir=ROOT / "experiment_results/E1124")
    cli.add_argument("--e1119-root", type=Path, help="E1119 run folder containing inputs.json, queue.json and its original activation bases")
    cli.add_argument("--pair-manifest", type=Path, help="Optional registered manifest to verify against the native FF++ index")
    args = cli.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    graph = plan()
    if args.plan:
        print(json.dumps({"experiment": "E1124", "seed": SEED, "arms": ARMS, "test_datasets": TEST_DS,
                          "jobs": graph}, indent=2))
        return 0
    if args.e1119_root is None:
        cli.error("A completed --e1119-root activation archive is required")
    if args.pair_manifest is not None and not args.pair_manifest.is_file():
        cli.error("The supplied --pair-manifest does not exist")
    if args.historical_b0 is not None:
        cli.error("E1124 obtains historical B00 from --e1119-root")
    if args.feature_batch_size < 1:
        cli.error("--feature_batch_size must be positive")

    import torch
    import experiment_utils as utilities
    from e1124_analysis import summarize
    from e1124_assets import registered_manifest
    from run_e1115 import build_training_config

    devices = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "") or ",".join(map(str, range(torch.cuda.device_count())))
    gpus = [gpu.strip() for gpu in devices.split(",") if gpu.strip()]
    if not torch.cuda.is_available() or not gpus or "-1" in gpus or len(set(gpus)) != len(gpus):
        cli.error("E1124 requires at least one unique CUDA GPU")
    native.normalize_arguments(args, ROOT)
    for key in ("rgb_root", "clip_pretrained_path"):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    args.dataset_json_folder = args.dataset_json_folder.resolve()
    args.output_dir = args.output_dir.resolve()
    args.seed, args.n_epochs, args.batch_size, args.sampler_real_ratio = SEED, 10, 32, .3
    baseline = build_training_config(args, "B0_NATIVE", args.output_dir, utilities, steps=None)
    print("Verifying the original training and eight evaluation inputs.", flush=True)
    material = native.input_identity(args, *baseline)
    historical = args.e1119_root.resolve()
    if native.read(historical / "inputs.json")["input_identity"] != material["input_identity"]:
        raise ValueError("E1124 must use the same FF++ and eight test inputs and CLIP snapshot as E1119")
    queue = native.read(historical / "queue.json")
    activation = archive_path(historical, queue, "asset:activation")
    manifest = registered_manifest(args.dataset_json_folder / f"{TRAIN_DS}.json",
                                   file_digest(activation / "references.json"))
    if args.pair_manifest is not None:
        previous = native.read(args.pair_manifest)
        if any(previous.get(key) != value for key, value in manifest.items()):
            raise ValueError("Supplied pair manifest differs from the registered native frame split")

    source_material = {"activation_references_sha256": file_digest(activation / "references.json"),
                       "activation_assets_sha256": file_digest(activation / "complete.json"),
                       "pair_manifest_digest": digest(manifest)}
    sources = native.source_identity()
    for path in (list((ROOT / "experiments").glob("*e1124*.py")) +
                 [ROOT / "training/detectors/effort_detector_e1124.py", ROOT / "training/detectors/__init__.py"]):
        sources[str(path.relative_to(ROOT))] = file_digest(path)
    identity = digest([material, sources, baseline, args.feature_batch_size, source_material])
    folder = args.output_dir / identity[:20]
    folder.mkdir(parents=True, exist_ok=True)
    with native.lock(folder):
        atomic_json(folder / "inputs.json", material)
        atomic_json(folder / "source_sha256.json", sources)
        atomic_json(folder / "plan.json", graph)
        pair_manifest = folder / "pair_manifest.json"
        atomic_json(pair_manifest, manifest)
        atomic_json(folder / "references.json", {"activation": str(activation),
                    "pair_manifest": str(pair_manifest), "source_material": source_material})
        for phase, config in zip(("train", "eval"), baseline):
            atomic_json(folder / "baseline" / str(SEED) / f"{phase}.json", config)
        reused = historical_jobs(historical, queue, baseline, activation)
        jobs = native.schedule(folder, gpus, args, graph=graph, initial=reused, entrypoint=__file__,
                               task_factory=new_task, validate_cached=validate_cached)
        ok = summarize(folder, jobs, ARMS, SEED, PROTOCOL_KEYS)
    print(f"E1124 results: {folder}", flush=True)
    return int(not ok)


if __name__ == "__main__":
    sys.exit(main())

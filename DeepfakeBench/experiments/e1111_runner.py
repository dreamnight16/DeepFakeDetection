"""Complete E1111A server runner, using the existing frozen-B0 convention."""

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
from e1111_paths import normalize_arguments, relocated

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_RUN = ROOT / "experiment_results/E0924/seed1024_20260930_110618_353460_2771190"


def resolve_source(args):
    import run_e1001 as previous

    source_args = argparse.Namespace(
        base_run=Path(relocated(args.base_run)) if args.base_run else None,
        base_checkpoint=Path(relocated(args.base_checkpoint)) if args.base_checkpoint else None,
        base_config=Path(relocated(args.base_config)) if args.base_config else None)
    if not any((source_args.base_run, source_args.base_checkpoint, source_args.base_config)):
        source_args.base_run = DEFAULT_BASE_RUN
    if source_args.base_run and (source_args.base_checkpoint or source_args.base_config):
        raise ValueError("Use --base_run OR --base_checkpoint with --base_config")
    run_path = Path(source_args.base_run) if source_args.base_run else None
    report_path = run_path / "all_results.json" if run_path else None
    if report_path and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        if report.get("experiment") == "E1111" and report.get("branch") == "E1111B":
            baseline = report.get("arms", {}).get("B0", {})
            if baseline.get("status") != "OK" or baseline.get("settings", {}).get("model_name") != "effort":
                raise ValueError("The source E1111 B0 must be a successful original-CE run")
            source_args = argparse.Namespace(base_run=None, base_checkpoint=Path(relocated(baseline["ckpt"])),
                                            base_config=run_path / "B0/train_config.json")
            source = previous.resolve_source(source_args)
            if source["base_sha256"] != baseline.get("checkpoint_sha256"):
                raise ValueError("Retrained B0 checkpoint hash differs from its result")
            source["origin"] = "E1111B/B0: independently retrained original CE"
            return source
    if run_path:
        # The existing E0924 --base_run convention can also be moved intact.
        folder = run_path / "training/G26_B0"
        pointer = run_path / "training_results.json"
        result_path = folder / "result.json"
        result = json.loads(pointer.read_text(encoding="utf-8-sig")).get("G26_B0") if pointer.is_file() else None
        if result is None and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
        if result is not None:
            if result.get("status") != "OK" or not result.get("ckpt"):
                raise ValueError("The source E0924 B0 must have a successful checkpoint")
            if result.get("artifact_dir"):
                folder = Path(relocated(result["artifact_dir"]))
            source_args = argparse.Namespace(base_run=None,
                                            base_checkpoint=Path(relocated(result["ckpt"])),
                                            base_config=folder / "train_config.json")
    return previous.resolve_source(source_args)


def resolve_inputs(args):
    import yaml
    from e1111.shared_crop import raw_video_path
    from e1111.audit import source_splits
    from e1111.protocol import protocol
    from g21.manifest import image_path

    if args.seed != protocol()["seed"]:
        raise ValueError("This frozen E1111A protocol uses seed=1024")
    source = resolve_source(args)
    config = copy.deepcopy(source["config"])
    normalize_arguments(args, ROOT, config)
    labels = yaml.safe_load((ROOT / "training/config/test_config.yaml").read_text(encoding="utf-8"))["label_dict"]
    for name, value in labels.items():
        if name in config["label_dict"] and (config["label_dict"][name] != 0) != (value != 0):
            raise ValueError(f"Conflicting binary label: {name}")
        config["label_dict"].setdefault(name, value)
    local_metadata = ROOT / "preprocessing/dataset_json"
    metadata = Path(args.dataset_json_folder or (
        local_metadata if local_metadata.is_dir() else config["dataset_json_folder"]))
    data_root = Path(args.rgb_root or config.get("rgb_root_override", "/home/user1/effort/data"))
    raw_root = Path(args.raw_root or data_root)
    clip = Path(args.clip_pretrained_path or config.get(
        "clip_pretrained_path", ROOT / "training/models--openai--clip-vit-large-patch14"))
    predictor = Path(args.predictor_path)
    for path in (metadata / "FaceForensics++.json", metadata / "Celeb-DF-v2.json", predictor):
        if not path.is_file():
            raise FileNotFoundError(f"Required E1111A input is missing: {path}")
    for path in (data_root, raw_root, clip):
        if not path.is_dir():
            raise FileNotFoundError(f"Required E1111A directory is missing: {path}")
    data = json.loads((metadata / "FaceForensics++.json").read_text(encoding="utf-8-sig"))["FaceForensics++"]
    reals = data["FF-real"]["train"]["c23"]
    splits = source_splits(reals)
    required_sources = {v for split, values in splits.items() if split != "beta" for v in values}
    missing_real = []
    for v in sorted(required_sources):
        frames = reals[v]["frames"]
        if not frames:
            missing_real.append(v)
        elif not image_path(raw_root, raw_video_path(frames[0])).is_file():
            missing_real.append(v)
    if missing_real:
        raise FileNotFoundError(
            f"Strict shared cropping requires original c23 videos under {raw_root}; "
            f"{len(missing_real)} required real videos are missing, examples={missing_real[:8]}. "
            "Use --raw_root for the existing original-video root; cropped PNGs cannot replace these videos.")
    raw_fake_counts = {method: sum(bool(entry["frames"]) and image_path(
        raw_root, raw_video_path(entry["frames"][0])).is_file()
        for entry in data[method]["train"]["c23"].values()) for method in protocol()["methods"]}
    if any(count == 0 for count in raw_fake_counts.values()):
        raise FileNotFoundError(f"An original manipulated-video method is absent under {raw_root}: {raw_fake_counts}")
    config.update(dataset_json_folder=str(metadata.resolve()), rgb_root_override=str(data_root.resolve()),
                  clip_pretrained_path=str(clip.resolve()))
    return {"source": source, "config": config, "metadata": metadata.resolve(),
            "data_root": data_root.resolve(), "raw_root": raw_root.resolve(), "clip": clip.resolve(),
            "predictor": predictor.resolve(), "raw_fake_video_counts": raw_fake_counts}


def run(args):
    from g21.io import atomic_json, file_digest
    from e1111.audit import audit, inventory
    from e1111.features import extract, export_dataset
    from e1111.pipeline import run_s0, calibrate, score_export, attribution
    from e1111.protocol import protocol
    from e1111.shared_crop import prepare
    from g21.pairing import run_pairing

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output_dir) / f"seed{args.seed}_{stamp}_{os.getpid()}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"experiment": "E1111", "branch": "E1111A", "status": "RUNNING",
                "stage": "resolve_inputs", "protocol": protocol(), "stages": {},
                "results_dir": str(output.resolve()),
                "runner_sha256": file_digest(__file__)}

    def save():
        atomic_json(output / "all_results.json", manifest)

    def stage(name, function, *values, **keywords):
        manifest["stage"] = name
        print(f"E1111A/{name}: RUNNING", flush=True)
        save()
        result = function(*values, **keywords)
        manifest["stages"][name] = {key: result[key] for key in (
            "status", "N4", "Ncell", "K_pair", "real_crops", "fake_crops", "failures",
            "passed", "rho_sum", "receipt_sha256", "usable_pairs") if key in result}
        print(f"E1111A/{name}: {manifest['stages'][name]}", flush=True)
        save()
        return result

    print(f"E1111A run: {output.resolve()}", flush=True)
    save()
    try:
        inputs = resolve_inputs(args)
        source = inputs["source"]
        manifest["base"] = {key: source[key] for key in ("checkpoint", "config_path", "base_sha256", "config_sha256")}
        manifest["raw_fake_video_counts"] = inputs["raw_fake_video_counts"]
        config_path = output / "resolved_base_config.json"
        atomic_json(config_path, inputs["config"])
        if args.preflight:
            manifest.update(status="PREFLIGHT_OK", stage="preflight")
            save()
            return manifest
        ff_json = inputs["metadata"] / "FaceForensics++.json"
        stage("inventory", inventory, ff_json, output / "inventory")
        if args.pair_map:
            pair_map = args.pair_map
        else:
            stage("content_pairing", run_pairing, ff_json, inputs["data_root"], output / "content_pairing")
            pair_map = output / "content_pairing/pair_map.json"
        crops = stage("shared_crop", prepare, ff_json, pair_map, inputs["raw_root"],
                      inputs["predictor"], output / "shared_crop")
        audited = stage("audit", audit, crops["dataset_json"], crops["images"],
                        crops["pair_map"], output / "audit")
        if not audited["P_D7_pass"]:
            manifest.update(status=audited["status"], stage="audit")
            save()
            return manifest
        audit_path = output / "audit/audit.json"
        cache_path = output / "features/features.npz"
        stage("features", extract, audit_path, config_path, source["checkpoint"], inputs["clip"],
              cache_path, device=args.device, batch_size=args.batch_size, d8=not args.skip_d8)
        s0 = stage("s0", run_s0, audit_path, cache_path, output / "s0")
        if s0["status"] not in ("S0_PASS", "FISHER_D1_PASS_PENDING_R2"):
            manifest.update(status=s0["status"], stage="s0")
            save()
            return manifest
        stage("calibrate", calibrate, audit_path, cache_path, output / "s0/metric.npz",
              config_path, source["checkpoint"], inputs["clip"], output / "calibrated",
              device=args.device, batch_size=args.batch_size)
        metric_path = output / "calibrated/calibrated_metric.npz"
        evaluations = []
        for dataset, split, tag in (("FaceForensics++", "val", "dev_ff"),
                                    ("Celeb-DF-v2", "train", "dev_cdf")):
            destination = output / tag
            stage(tag, export_dataset, inputs["metadata"] / f"{dataset}.json", dataset, split,
                  inputs["data_root"], config_path, source["checkpoint"], inputs["clip"],
                  destination / "features.npz", device=args.device, batch_size=args.batch_size)
            scores = output / f"{tag}_scores"
            stage(f"{tag}_scores", score_export, destination / "features.npz", metric_path, scores)
            evaluations.append(scores)
        attribution_result = stage("attribution", attribution, evaluations, output / "attribution")
        manifest.update(status="DEVELOPMENT_COMPLETE" if attribution_result["passed"] else "S1_FAILED",
                        stage="complete", results_dir=str(output.resolve()),
                        claim_status="development experiment; S3 identity audit and formal blind tests remain separate")
    except Exception as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        save()
    print(f"E1111A results: {output / 'all_results.json'}", flush=True)
    return manifest

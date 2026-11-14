"""Frozen A scores on the native B0/B test rows, transforms and AUC readout."""

import copy
import datetime
import json
from pathlib import Path

import numpy as np

from g21.data import MEAN, STD
from g21.io import atomic_json, file_digest
from .features import FrozenFeatures
from .metric import score
from .pipeline import load_metric


def reference_alignment(values, path):
    """Check actual archived B0 rows and predictions, allowing the known move."""
    from e0924_protocol import canonical_path
    from e1111_paths import relocated

    with np.load(path, allow_pickle=False) as reference:
        for key in ("labels", "path", "video_id"):
            actual, expected = values[key], reference[key]
            if key != "labels":
                actual = [canonical_path(relocated(str(p))) for p in actual]
                expected = [canonical_path(relocated(str(p))) for p in expected]
            if not np.array_equal(actual, expected):
                raise ValueError(f"Native B0 test {key} differs from saved export: {path}")
        for key in ("cls_prob", "global_log_odds"):
            if (values[key].shape != reference[key].shape
                    or not np.allclose(values[key], reference[key], rtol=1e-5, atol=1e-6)):
                raise ValueError(f"Native B0 {key} differs from saved export: {path}")
    return {"status": "MATCHED", "sha256": file_digest(path)}


def readout(values, scores, threshold):
    from run_e1001 import readout_metrics

    result = readout_metrics(values, scores)
    # A energy is not a probability. Keep exactly the shared AUC calculation,
    # but use its already frozen decision threshold for thresholded accuracy.
    result["accuracy"] = float(((np.asarray(scores) > threshold) == values["labels"]).mean())
    result["accuracy_threshold"] = float(threshold)
    return {key: value for key, value in result.items()
            if key not in ("corrected_frames", "harmed_frames")}


def evaluate(metric_path, base_config, checkpoint, clip, evaluation_config, output,
             *, device="cuda:0", reference_exports=None):
    import torch
    import experiment_utils as utilities
    from e0924_export import strict_loader
    from run_e1021 import seed_everything
    from run_g25 import CROSS_DS, TEST_DS, TRAIN_DS, VAL_DS

    model, identity = load_metric(metric_path)
    if identity.get("status") not in ("S0_PASS", "FISHER_D1_PASS_PENDING_R2") or "thresholds" not in identity:
        raise ValueError("Unified test requires a frozen, calibrated S0/D1 model")
    base = json.loads(Path(base_config).read_text(encoding="utf-8-sig"))
    config = json.loads(Path(evaluation_config).read_text(encoding="utf-8-sig"))
    if (config.get("model_name") != "effort" or config.get("frame_num", {}).get("test") != 8
            or any(config.get(key) for key in ("multi_crop", "video_mode", "use_freq_split", "use_adaptive_threshold"))):
        raise ValueError("Use the original B0 single-frame, 8-frame evaluation configuration")
    if (config.get("resolution") != 224 or not np.allclose(config.get("mean", []), MEAN)
            or not np.allclose(config.get("std", []), STD)):
        raise ValueError("B0 test preprocessing differs from the fitted A feature coordinates")
    config = copy.deepcopy(config)
    config.update(dataset_json_folder=base["dataset_json_folder"], rgb_root_override=base["rgb_root_override"],
                  clip_pretrained_path=str(clip), use_data_augmentation=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    print(f"E1111A benchmark: {output.resolve()}", flush=True)
    atomic_json(output / "eval_config.json", config)
    extractor = FrozenFeatures(base_config, checkpoint, clip, device, config["test_batchSize"])
    for key in ("checkpoint_sha256", "base_config_sha256", "feature", "preprocess",
                "feature_implementation_sha256", "versions"):
        if extractor.identity[key] != identity["feature_identity"][key]:
            raise ValueError(f"Unified-test feature identity differs from the fitted model: {key}")
    result = {"experiment": "E1111", "branch": "E1111A", "status": "RUNNING",
              "primary_detector": "D_J", "metric_protocol": "full-path video_id; frame mean",
              "test_protocol": "native B0/B saved eval_config + get_data_loader(test) + strict_loader; 8 frames",
              "score_protocol": "raw frozen A energy; no sigmoid, sign selection or test-set tuning",
              "accuracy_protocol": "frame accuracy; A uses the frozen 32-frame D_theta video threshold as reference only, not an 8-frame FPR operating point; B0 uses probability > 0.5",
              "frozen_metric_identity": identity, "metric_sha256": file_digest(metric_path),
              "benchmark_implementation_sha256": file_digest(__file__),
              "baseline_eval_config_sha256": file_digest(evaluation_config),
              "datasets": {}, "detectors": {}, "B0": {}, "reference_alignment": {},
              "dataset_sha256": {name: file_digest(Path(config["dataset_json_folder"]) / f"{name}.json")
                                 for name in TEST_DS},
              "results_dir": str(output.resolve())}
    atomic_json(output / "all_results.json", result)
    exports = output / "exports"
    exports.mkdir()
    try:
        for dataset in TEST_DS:
            print(f"E1111A/benchmark/{dataset}: RUNNING", flush=True)
            seed_everything(config["manualSeed"])
            original = utilities.get_data_loader(config, dataset).dataset
            batches = {key: [] for key in ("labels", "path", "video_id", "cls_prob", "global_log_odds")}
            try:
                with torch.no_grad():
                    for batch in strict_loader(original, config["test_batchSize"]):
                        images = batch["image"].to(device)
                        vision = extractor.model.backbone(extractor.model._prep_input(images))
                        tokens = vision["last_hidden_state"]
                        if tuple(tokens.shape[1:]) != (257, 1024):
                            raise ValueError("Unified test expects CLIP (N,257,1024) tokens")
                        features = tokens[:, 1:].mean(dim=1).cpu().float().numpy()
                        logits = extractor.model.classifier(vision["pooler_output"])
                        batches["labels"].append(batch["label"].cpu().numpy())
                        batches["path"].append(np.asarray(batch["path"]))
                        batches["video_id"].append(np.asarray(batch["video_id"]))
                        batches["cls_prob"].append(logits.softmax(dim=1)[:, 1].cpu().numpy())
                        batches["global_log_odds"].append((logits[:, 1] - logits[:, 0]).cpu().numpy())
                        for name, values in score(features, model, identity["gamma"]).items():
                            batches.setdefault(name, []).append(values)
                if not batches["labels"]:
                    raise ValueError(f"Empty native test export: {dataset}")
                values = {key: np.concatenate(parts) for key, parts in batches.items()}
                if any(not np.isfinite(v).all() for key, v in values.items() if key not in ("path", "video_id")):
                    raise ValueError(f"Nonfinite unified test scores: {dataset}")
                if reference_exports is not None:
                    result["reference_alignment"][dataset] = reference_alignment(
                        values, Path(reference_exports) / f"{dataset}.npz")
                else:
                    result["reference_alignment"][dataset] = {"status": "SHARED_LOADER_NO_ARCHIVED_REFERENCE"}
                np.savez_compressed(exports / f"{dataset}.npz", **values)
                detectors = [name for name in values if name.startswith(("D_", "C_"))]
                result["detectors"][dataset] = {
                    name: readout(values, values[name], identity["thresholds"][name]["0.1"]) for name in detectors}
                result["datasets"][dataset] = result["detectors"][dataset]["D_J"]
                result["B0"][dataset] = readout(values, values["cls_prob"], .5)
                atomic_json(output / "all_results.json", result)
                print(f"E1111A/benchmark/{dataset}: {result['datasets'][dataset]}", flush=True)
            finally:
                env = getattr(original, "env", None)
                if env is not None:
                    env.close()
        independent = [name for name in CROSS_DS if name != VAL_DS]
        result.update(status="OK", independent_datasets=independent,
                      excluded_from_independent_mean=[VAL_DS, TRAIN_DS],
                      video_auc_seven_mean_fullpath=sum(result["datasets"][d]["video_auc"] for d in CROSS_DS) / len(CROSS_DS),
                      independent_video_auc_mean_fullpath=sum(result["datasets"][d]["video_auc"] for d in independent) / len(independent),
                      frame_auc_seven_mean=sum(result["datasets"][d]["frame_auc"] for d in CROSS_DS) / len(CROSS_DS),
                      independent_frame_auc_mean=sum(result["datasets"][d]["frame_auc"] for d in independent) / len(independent))
    except Exception as exc:
        result.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(output / "all_results.json", result)
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"E1111A benchmark results: {output / 'all_results.json'}", flush=True)
    return result


def evaluate_run(run_dir, output=None, *, latest=False, device="cuda:0"):
    run_dir = Path(run_dir)
    if latest:
        candidates = sorted(run_dir.glob("seed*/calibrated/calibrated_metric.npz"))
        if not candidates:
            raise FileNotFoundError("No calibrated A run is available yet; let S0/calibration finish first")
        run_dir = candidates[-1].parents[1]
    manifest = json.loads((run_dir / "all_results.json").read_text(encoding="utf-8-sig"))
    base = json.loads((run_dir / "resolved_base_config.json").read_text(encoding="utf-8-sig"))
    source = manifest["base"]
    native = Path(source["config_path"]).parent
    if output is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = run_dir / f"benchmark8_{stamp}"
    return evaluate(run_dir / "calibrated/calibrated_metric.npz", run_dir / "resolved_base_config.json",
                    source["checkpoint"], base["clip_pretrained_path"], native / "eval_config.json", output,
                    device=device, reference_exports=native / "exports" if (native / "exports").is_dir() else None)

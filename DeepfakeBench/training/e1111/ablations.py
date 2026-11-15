"""Complete registered A ablation matrix on one frozen feature snapshot."""

import csv
import datetime
import json
from pathlib import Path

import numpy as np

from g21.io import atomic_json, digest, file_digest
from .ablation_models import fit_variants, from_space, score_models
from .ablation_diagnostics import complementarity, identity_audit, arcface_path
from .benchmark import evaluate, readout, test_source_identity
from .features import FrozenFeatures, extract, load_audit, load_cache, read_rgb
from .metric import ProtocolError, conformal_threshold, fit_space, real_statistics, second_moment
from .pipeline import _real_groups, load_metric, paired_features, video_mean_features, video_means
from .protocol import protocol


def recover_fit_cache(run_dir, missing_path, manifest, config_path, config, audit, identity, device):
    """Reconstruct a missing linked cache without changing the completed A run."""
    expected = identity["feature_identity"]
    checkpoint = manifest["base"]["checkpoint"]
    if (file_digest(config_path) != expected["base_config_sha256"]
            or file_digest(checkpoint) != expected["checkpoint_sha256"]):
        raise ValueError("Cannot recover the fitted cache: original B0 or resolved configuration changed")
    destination = Path(run_dir) / "features/features.npz"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temporary = destination.with_name(f"features.recover_{stamp}.npz")
    print(f"E1111A/fit_cache: MISSING {missing_path}; rebuilding from the original frozen B0", flush=True)
    extract(Path(run_dir) / "audit/audit.json", config_path, checkpoint, config["clip_pretrained_path"],
            temporary, device=device, batch_size=8, d8=expected["d8"])
    metadata = load_cache(temporary, audit)[1]
    if metadata != expected:
        raise ValueError("Recovered feature identity differs from the original fitted model; refusing reuse")
    if destination.exists():
        raise FileExistsError(f"Another cache already exists; refusing overwrite: {destination}")
    temporary.replace(destination)
    atomic_json(destination.parent / "recovery.json", {
        "missing_reference": str(missing_path), "recovered_cache": str(destination.resolve()),
        "feature_identity": metadata, "cache_sha256": file_digest(destination),
        "verification": "same frozen B0/config/audit/implementation/versions; old cache bytes unavailable",
    })
    print(f"E1111A/fit_cache: RECOVERED {destination.resolve()}", flush=True)
    return destination


def context(run_dir, latest=False, device="cuda:0"):
    run_dir = Path(run_dir)
    if latest:
        candidates = sorted(run_dir.glob("seed*/calibrated/calibrated_metric.npz"))
        if not candidates:
            raise FileNotFoundError("No calibrated A run exists yet")
        run_dir = candidates[-1].parents[1]
    manifest = json.loads((run_dir / "all_results.json").read_text(encoding="utf-8-sig"))
    config_path = run_dir / "resolved_base_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    audit = load_audit(run_dir / "audit/audit.json")
    cache_path = Path(manifest.get("stages", {}).get("features", {}).get("cache_path", run_dir / "features/features.npz"))
    model, identity = load_metric(run_dir / "calibrated/calibrated_metric.npz")
    if audit["audit_sha256"] != identity["audit_sha256"]:
        raise ValueError("Ablation run and fitted model use different audits")
    if "thresholds" not in identity or identity["status"] not in ("S0_PASS", "FISHER_D1_PASS_PENDING_R2"):
        raise ValueError("Ablations require an already calibrated S0/D1 model")
    if not cache_path.is_file():
        local_cache = run_dir / "features/features.npz"
        cache_path = local_cache if local_cache.is_file() else recover_fit_cache(
            run_dir, cache_path, manifest, config_path, config, audit, identity, device)
    arrays, metadata, lookup = load_cache(cache_path, audit)
    if metadata != identity["feature_identity"]:
        raise ValueError("Ablation fitted model and cached features differ")
    s0 = json.loads((run_dir / "s0/s0.json").read_text(encoding="utf-8-sig"))
    return run_dir, manifest, config_path, config, audit, cache_path, arrays, lookup, model, identity, s0


def save_model(path, model, specification):
    arrays = {key: value for key, value in model.items() if isinstance(value, np.ndarray)}
    metadata = {key: value for key, value in model.items() if key not in arrays}
    metadata["specification"] = specification
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.asarray(json.dumps(metadata, allow_nan=False)))


def load_test_cache(directory, identity, native_config, metadata_root):
    from run_g25 import TEST_DS
    directory = Path(directory)
    report = json.loads((directory / "all_results.json").read_text(encoding="utf-8-sig"))
    if report.get("status") != "OK" or set(report.get("feature_cache", {})) != set(TEST_DS):
        raise ValueError("Test feature cache is incomplete; interrupted cache files cannot be reused")
    if (report["frozen_metric_identity"] != identity
            or report["baseline_eval_config_sha256"] != file_digest(native_config)
            or report.get("test_source_sha256") != test_source_identity()):
        raise ValueError("Test cache has different B0, feature implementation or native test configuration")
    for dataset in TEST_DS:
        item = report["feature_cache"][dataset]
        if (report["dataset_sha256"][dataset] != file_digest(Path(metadata_root) / f"{dataset}.json")
                or item["sha256"] != file_digest(directory / item["file"])
                or item["export_sha256"] != file_digest(directory / "exports" / f"{dataset}.npz")):
            raise ValueError(f"Test feature snapshot changed: {dataset}")
    return report


def _read_test_rows(directory, dataset, info):
    directory = Path(directory)
    with np.load(directory / "exports" / f"{dataset}.npz", allow_pickle=False) as values:
        rows = {key: values[key] for key in values.files
                if key in ("labels", "path", "video_id", "cls_prob", "global_log_odds") or key.startswith(("D_", "C_"))}
    if digest({key: rows[key].tolist() for key in ("labels", "path", "video_id")}) != info["rows_sha256"]:
        raise ValueError("Cached test feature row identities differ")
    z = np.load(directory / info["file"], mmap_mode="r", allow_pickle=False)
    if z.shape != (len(rows["labels"]), 1024) or len(z) != info["rows"]:
        raise ValueError("Test feature matrix has wrong shape")
    return rows, z


def theta_features(audit, config_path, checkpoint, clip, identity, output, device):
    import torch
    rows = [(source, row) for source, frames in audit["real_frames"]["theta"].items() for row in frames]
    extractor = FrozenFeatures(config_path, checkpoint, clip, device, 8)
    try:
        if any(extractor.identity[key] != identity["feature_identity"][key] for key in extractor.identity):
            raise ValueError("Ablation calibration uses a different feature extractor")
        z = extractor.encode(read_rgb(audit["data_root"], row["path"]) for _, row in rows)
    finally:
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    np.save(output / "theta_features.npy", z, allow_pickle=False)
    atomic_json(output / "theta_rows.json", [{"source": source, **row} for source, row in rows])
    return z, np.asarray([source for source, _ in rows])


def calibrate_models(z, sources, models, alphas):
    thresholds = {}
    for name, values in score_models(z, models).items():
        _, videos, _ = video_means(values, sources)
        thresholds[name] = {str(alpha): conformal_threshold(videos, alpha) for alpha in alphas}
    return thresholds


def _write_tables(output, report):
    from run_g25 import TEST_DS
    metric_fields = ["frame_auc", "video_auc", "video_auc_legacy", "accuracy", "num_frames", "num_videos"]
    fields = ["arm", "dataset", "scope", "status", "reason", *metric_fields]
    with (output / "ablation_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm, item in report["arms"].items():
            for dataset in TEST_DS:
                metrics = item.get("datasets", {}).get(dataset, {})
                reason = item.get("reason", item.get("undefined_datasets", {}).get(dataset, ""))
                writer.writerow({"arm": arm, "dataset": dataset, "scope": item["scope"],
                                 "status": "SCORED" if metrics else item["status"], "reason": reason,
                                 **{key: metrics.get(key) for key in metric_fields}})
    fields = ["arm", "scope", "status", "video_auc_seven_mean_fullpath", "independent_video_auc_mean_fullpath",
              "frame_auc_seven_mean", "independent_frame_auc_mean"]
    with (output / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for arm, item in report["arms"].items():
            writer.writerow({"arm": arm, **{key: item.get(key) for key in fields[1:]}})


def run_suite(run_dir, output=None, *, latest=False, device="cuda:0", test_feature_cache=None,
              arcface_model=None, arcface_input_mean=127.5, arcface_input_std=127.5):
    from run_g25 import CROSS_DS, TEST_DS, VAL_DS
    values = context(run_dir, latest, device=device)
    run_dir, manifest, config_path, config, audit, cache_path, arrays, lookup, main, identity, s0 = values
    arcface_path(arcface_model)  # An explicitly supplied missing file is an input error.
    if output is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = run_dir / f"ablations_{stamp}"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    (output / "scores").mkdir()
    settings = protocol(audit.get("statistic_mode", "paired_frame"))
    report = {"experiment": "E1111", "branch": "E1111A", "version": "E1111A-ablation-v1", "status": "RUNNING",
              "results_dir": str(output.resolve()), "source_run": str(run_dir.resolve()), "arms": {}, "diagnostics": {},
              "plan": {"rank": settings["ranks"], "lambda_m": settings["lambda_m"], "tau_R": settings["tau_R"],
                       "gamma": settings["gamma"], "prototype_K": list(range(1, 9)), "seed": settings["seed"],
                       "lambda_t": 0., "selection": "single-factor interventions; all other main parameters fixed; no test selection",
                       "prototype_selection": "D_rad gap before D_fit prototypes or target evaluation",
                       "oracle": "target real labels used only for C5 Sigma_R; source mu/rank/lambda/tau/gamma unchanged; not a guaranteed upper bound"},
              "accuracy_protocol": "reference frame accuracy using each arm's own 32-frame D_theta video threshold; not an 8-frame FPR guarantee",
              "fit_cache_sha256": file_digest(cache_path), "frozen_metric_identity": identity,
              "implementation_sha256": {name: file_digest(Path(__file__).with_name(name))
                                         for name in ("ablations.py", "ablation_models.py", "ablation_diagnostics.py")},
              "test_used_for_selection": False}
    atomic_json(output / "all_results.json", report)
    print(f"E1111A ablations: {output.resolve()}", flush=True)
    try:
        fit_real, rad_real = _real_groups(audit, lookup, "fit"), _real_groups(audit, lookup, "rad")
        delta, _ = (video_mean_features if audit.get("statistic_mode") == "video_mean" else paired_features)(audit, lookup)
        dc = second_moment(arrays["compression_deltas"])
        models, specs, d4 = fit_variants(main, identity, s0, settings, delta, fit_real, rad_real, dc)
        report["diagnostics"]["D4_gap"] = d4
        for name in ("D1", "D2", "D3", "D7", "D8"):
            report["diagnostics"][name] = s0.get(name, {"status": "NOT_RECORDED_IN_SOURCE_RUN"})
        report["arms"] = {name: {**spec, "datasets": {}} for name, spec in specs.items()}
        for name, model in models.items():
            save_model(output / "models" / f"{name}.npz", model, specs[name])
        atomic_json(output / "source_only_models_frozen.json", specs)
        print("E1111A/ablations/D5: RUNNING", flush=True)
        report["diagnostics"]["D5_complementarity"] = complementarity(run_dir, identity, settings["bootstrap_repeats"], settings["seed"])
        print("E1111A/ablations/D6: RUNNING", flush=True)
        try:
            report["diagnostics"]["D6_identity"] = identity_audit(run_dir, audit, arcface_model, arcface_input_mean, arcface_input_std)
        except Exception as exc:
            report["diagnostics"]["D6_identity"] = {"status": "FAILED_ARCFACE_DIAGNOSTIC", "passed": None,
                                                        "error": f"{type(exc).__name__}: {exc}"}
        print(f"E1111A/ablations/D6: {report['diagnostics']['D6_identity']['status']}", flush=True)
        native = Path(manifest["base"]["config_path"]).parent
        print("E1111A/ablations/calibration: RUNNING", flush=True)
        theta, theta_sources = theta_features(audit, config_path, manifest["base"]["checkpoint"], config["clip_pretrained_path"], identity, output, device)
        thresholds = calibrate_models(theta, theta_sources, models, settings["alpha"])
        atomic_json(output / "thresholds.json", thresholds)
        atomic_json(output / "all_results.json", report)
        if test_feature_cache is None:
            test_feature_cache = output / "benchmark8"
            benchmark = evaluate(run_dir / "calibrated/calibrated_metric.npz", config_path,
                manifest["base"]["checkpoint"], config["clip_pretrained_path"], native / "eval_config.json", test_feature_cache,
                device=device, reference_exports=native / "exports" if (native / "exports").is_dir() else None, cache_features=True)
        else:
            benchmark = load_test_cache(test_feature_cache, identity, native / "eval_config.json", config["dataset_json_folder"])
        report["benchmark"] = benchmark
        report["test_feature_cache"] = str(Path(test_feature_cache).resolve())
        report["arms"]["B0"] = {"status": "SCORED", "scope": "source_only", "datasets": benchmark["B0"]}
        report["arms"]["C5_oracle_Sigma_R"] = {"status": "RUNNING", "scope": "transductive_oracle_diagnostic_only", "datasets": {}}
        for dataset in TEST_DS:
            print(f"E1111A/ablations/evaluate/{dataset}: {len(models)} source-only arms + oracle", flush=True)
            rows, z = _read_test_rows(test_feature_cache, dataset, benchmark["feature_cache"][dataset])
            active = dict(models)
            try:
                real_groups = [np.asarray(z[(rows["video_id"] == video) & (rows["labels"] == 0)])
                               for video in np.unique(rows["video_id"][rows["labels"] == 0])]
                _, covariance, _ = real_statistics(real_groups, s0["tau_R"])
                space = fit_space(delta, covariance, dc, identity["lambda_m"], identity["family"])
                oracle = from_space(space, covariance, main["mu"], identity["rank"], identity["gamma"], f"oracle_{dataset}")
                active["C5_oracle_Sigma_R"] = oracle
                save_model(output / "models" / f"oracle_{dataset}.npz", oracle, report["plan"]["oracle"])
                thresholds["C5_oracle_Sigma_R"] = calibrate_models(theta, theta_sources, {"C5_oracle_Sigma_R": oracle}, settings["alpha"])["C5_oracle_Sigma_R"]
                report["arms"]["C5_oracle_Sigma_R"].setdefault("thresholds_by_dataset", {})[dataset] = thresholds["C5_oracle_Sigma_R"]
                report["arms"]["C5_oracle_Sigma_R"].setdefault("target_real_videos", {})[dataset] = len(real_groups)
            except (ProtocolError, np.linalg.LinAlgError) as exc:
                report["arms"]["C5_oracle_Sigma_R"].setdefault("undefined_datasets", {})[dataset] = str(exc)
            scores = score_models(z, active)
            # Preserve the exact frozen native-benchmark scores, including ties.
            # Compact matrix association is used for the new refitted variants.
            for name in ("D_R", "D_F", "D_J", "D_P", "C_random", "C_shuffle", "C_real"):
                if name in scores and name in rows:
                    if not np.allclose(scores[name], rows[name], rtol=1e-8, atol=1e-8):
                        raise ValueError(f"Compact scoring differs from the frozen benchmark: {name}")
                    scores[name] = rows[name]
            np.savez_compressed(output / "scores" / f"{dataset}.npz", labels=rows["labels"], path=rows["path"], video_id=rows["video_id"], **scores)
            for name, raw in scores.items():
                report["arms"][name]["datasets"][dataset] = readout(rows, raw, thresholds[name]["0.1"])
            for key in ("frame_auc", "video_auc"):
                if not np.isclose(report["arms"]["D_J"]["datasets"][dataset][key], benchmark["datasets"][dataset][key], atol=1e-10, rtol=0):
                    raise ValueError("Compact ablation scoring changed the frozen main-model AUC")
            atomic_json(output / "all_results.json", report)
        independent = [name for name in CROSS_DS if name != VAL_DS]
        for name, item in report["arms"].items():
            if set(item["datasets"]) != set(TEST_DS):
                if item["status"] != "UNDEFINED":
                    item["status"] = "UNDEFINED_PARTIAL_DATASETS"
                continue
            item.update(status="SCORED", video_auc_seven_mean_fullpath=float(np.mean([item["datasets"][d]["video_auc"] for d in CROSS_DS])),
                        independent_video_auc_mean_fullpath=float(np.mean([item["datasets"][d]["video_auc"] for d in independent])),
                        frame_auc_seven_mean=float(np.mean([item["datasets"][d]["frame_auc"] for d in CROSS_DS])),
                        independent_frame_auc_mean=float(np.mean([item["datasets"][d]["frame_auc"] for d in independent])))
        report.update(status="OK", independent_datasets=independent,
                      limitations=["R3 closed; R4 optimization outside this registered experiment",
                                   "Oracle uses target real labels and is excluded from source-only claims",
                                   "Single-Gaussian M-style is affine in D_R, not independent evidence"])
        report["coverage"] = {"requested_arms": len(report["arms"]),
            "scored_arms": [name for name, item in report["arms"].items() if item["status"] == "SCORED"],
            "undefined_arms": {name: item.get("reason", item.get("undefined_datasets"))
                               for name, item in report["arms"].items() if item["status"] != "SCORED"},
            "optional_identity_status": report["diagnostics"]["D6_identity"]["status"]}
        if report["diagnostics"]["D6_identity"]["status"] == "FAILED_ARCFACE_DIAGNOSTIC":
            report["status"] = "DIAGNOSTIC_FAILED"
        _write_tables(output, report)
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_json(output / "all_results.json", report)
    print(f"E1111A ablation results: {output / 'all_results.json'}", flush=True)
    return report

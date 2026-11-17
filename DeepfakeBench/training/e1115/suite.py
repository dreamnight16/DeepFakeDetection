"""Complete frozen-metric, trusted-real adaptation, and baseline comparison suite."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from g21.io import atomic_json, file_digest
from .assets import load_native
from .evaluation import evaluate_scores, fusion_score, subset, thresholds
from .math import adapt_model, fit_direction, prepare_model, real_statistics, score_model
from .protocol import metric_specs, plan, split_target_real, stable_seed
from .statistics import build_moments


def save_model(path, model, specification):
    arrays = {k: v for k, v in model.items() if isinstance(v, np.ndarray)}
    metadata = {k: v for k, v in model.items() if k not in arrays}
    np.savez_compressed(path, **arrays, metadata=np.asarray(json.dumps({**metadata, "specification": specification})))


def fit_models(audit, lookup, fit_real, dc):
    moments, diagnostic = build_moments(audit, lookup)
    mu, covariance = real_statistics(fit_real, .1)
    models, specs = {}, {}
    for name, specification in metric_specs().items():
        spec = dict(specification)
        try:
            B = moments[spec["statistic"]]
            if spec.get("legacy"):
                d = len(mu)
                epsilon = 1e-6 * max(np.trace(dc), np.trace(covariance)) / d
                values, vectors = eigh((B + B.T) / 2, dc + epsilon * np.eye(d))
                order = np.argsort(values)[::-1]
                values, vectors = values[order], vectors[:, order]
                positive = values[values > np.finfo(float).eps * d * max(abs(values).max(), 1e-300)]
                if len(positive) < spec["rank"]:
                    raise ValueError("Legacy metric has insufficient positive rank")
                direction = {"directions": vectors[:, :spec["rank"]],
                             "weights": np.minimum(positive[:spec["rank"]], np.quantile(positive, .99)),
                             "raw_eigenvalues": positive[:spec["rank"]]}
            else:
                direction = fit_direction(B, dc if spec["compression"] else np.zeros_like(dc),
                                          covariance, spec["rank"], spec["eta"], spec["weighting"])
            models[name] = prepare_model({**direction, **spec, "mu": mu, "covariance": covariance})
            specs[name] = {**spec, "status": "FITTED", "scope": "source_only"}
        except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
            specs[name] = {**spec, "status": "UNAVAILABLE", "scope": "source_only", "reason": str(exc)}
    if "FULL" not in models:
        raise ValueError(f"Full metric could not be fitted: {specs['FULL']}")
    return models, specs, diagnostic


def _means(report):
    from run_g25 import CROSS_DS, TEST_DS, TRAIN_DS, VAL_DS
    independent = [x for x in CROSS_DS if x != VAL_DS]
    for item in report["arms"].values():
        datasets = item.get("datasets", {})
        for names, field in [(independent, "independent_video_auc_mean_fullpath"),
                             (CROSS_DS, "video_auc_seven_mean_fullpath")]:
            # Never silently average the successful subset of an incomplete matrix.
            item[field] = float(np.mean([datasets[d]["video_auc_fullpath"] for d in names])) if all(d in datasets for d in names) else None
        item["scored_datasets"] = list(datasets)
    report["independent_datasets"] = independent
    report["excluded_from_independent_mean"] = [TRAIN_DS, VAL_DS]
    report["datasets"] = TEST_DS


def _csv(report, output):
    fields = ["arm", "scope", "dataset", "frame_auc", "video_auc_fullpath", "num_frames", "num_videos", "status"]
    with (output / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for arm, item in report["arms"].items():
            for dataset, values in item.get("datasets", {}).items():
                writer.writerow({"arm": arm, "scope": item["scope"], "dataset": dataset, "status": "SCORED",
                                 **{key: values.get(key) for key in fields[3:-1]}})
    fields = ["arm", "scope", "status", "independent_video_auc_mean_fullpath", "video_auc_seven_mean_fullpath"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for arm, item in report["arms"].items():
            writer.writerow({"arm": arm, **{key: item.get(key) for key in fields[1:]}})


def run_suite(audit, source, native_dir, output, *, groups, seed=1024, budgets=(10, 30, 100),
              calibration_count=30, repeats=3, anchor=None):
    from run_g25 import TEST_DS
    lookup, fit_real, dc, cal_rows, cal_z = source
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("models", "scores", "adaptation_splits", "adaptation_scores"):
        (output / name).mkdir()
    models, specifications, diagnostics = fit_models(audit, lookup, fit_real, dc)
    if "geometry" not in groups:
        models = {name: models[name] for name in ("FULL", "VM_LEGACY") if name in models}
        specifications = {name: specifications[name] for name in models}
    report = {"experiment": "E1115", "status": "RUNNING", "groups": groups,
              "protocol": plan(), "source_statistics": diagnostics, "arms": {},
              "development_gate": False, "test_used_for_selection": False,
              "native_manifest_sha256": file_digest(Path(native_dir) / "manifest.json"),
              "baseline_mutated": False, "adaptation_unavailable": [],
              "source_calibration_scope": "empirical: theta held out from metric fit, not from backbone training",
              "target_calibration_scope": "held-out real videos; validity requires representative exchangeable calibration/test reals"}
    source_thresholds, source_cal_scores = {}, {}
    baseline_thresholds = thresholds(cal_rows, cal_rows["cls_prob"])
    anchor_cal, anchor_thresholds = None, None
    if anchor is not None and "baseline" in groups:
        with np.load(anchor["calibration"], allow_pickle=False) as archive:
            anchor_cal = {k: archive[k] for k in ("labels", "path", "video_id", "cls_prob")}
        for key in ("labels", "path", "video_id"):
            if not np.array_equal(anchor_cal[key], cal_rows[key]):
                raise ValueError("Original B0 calibration rows differ from candidate rows")
        anchor_thresholds = thresholds(anchor_cal, anchor_cal["cls_prob"])
        report["anchor"] = {"native": str(anchor["native"]),
                            "manifest_sha256": file_digest(Path(anchor["native"]) / "manifest.json")}
    for name, model in models.items():
        save_model(output / "models" / f"{name}.npz", model, specifications[name])
        source_cal_scores[name] = score_model(cal_z, model)
        source_thresholds[name] = thresholds(cal_rows, source_cal_scores[name])
    report["arms"] = {name: {**spec, "datasets": {}} for name, spec in specifications.items()}
    report["arms"]["B0"] = {"scope": "source_only", "status": "RUNNING", "datasets": {}}
    # CDF scaling is fitted on source fitting videos, separately from theta calibration.
    fit_z = np.concatenate(fit_real)
    cdf_reference = score_model(fit_z, models["FULL"])
    del fit_z
    fusion_thresholds = {}
    if "baseline" in groups:
        for weight in plan()["fusion_weights"]:
            name = f"B0_PLUS_FULL_{weight:g}"
            fused = fusion_score(cal_rows["cls_prob"], source_cal_scores["FULL"], cdf_reference, weight)
            fusion_thresholds[name] = thresholds(cal_rows, fused)
            report["arms"][name] = {"scope": "source_only", "status": "RUNNING", "weight": weight, "datasets": {}}
        if anchor_cal is not None:
            report["arms"]["ANCHOR_B0"] = {"scope": "source_only", "status": "RUNNING", "datasets": {}}
            for weight in plan()["fusion_weights"]:
                name = f"ANCHOR_PLUS_FULL_{weight:g}"
                fused = fusion_score(anchor_cal["cls_prob"], source_cal_scores["FULL"], cdf_reference, weight)
                fusion_thresholds[name] = thresholds(cal_rows, fused)
                report["arms"][name] = {"scope": "source_only", "status": "RUNNING", "weight": weight, "datasets": {}}
    atomic_json(output / "source_models_frozen.json", {"specifications": specifications,
                "thresholds": source_thresholds, "baseline_thresholds": baseline_thresholds,
                "fusion_thresholds": fusion_thresholds})
    atomic_json(output / "all_results.json", report)
    for dataset in TEST_DS:
        print(f"E1115/full-matrix/{dataset}", flush=True)
        rows, z = load_native(native_dir, dataset)
        anchor_prob = None
        if anchor_cal is not None:
            anchor_rows, anchor_z = load_native(anchor["native"], dataset)
            for key in ("labels", "path", "video_id"):
                if not np.array_equal(anchor_rows[key], rows[key]):
                    raise ValueError(f"Original B0 native rows differ: {dataset}/{key}")
            anchor_prob = anchor_rows["cls_prob"]
            del anchor_z
            anchor_eval = {**rows, "cls_prob": anchor_prob}
            report["arms"]["ANCHOR_B0"]["datasets"][dataset] = evaluate_scores(anchor_eval, anchor_prob, anchor_thresholds, anchor_thresholds)
        base = evaluate_scores(rows, rows["cls_prob"], baseline_thresholds, baseline_thresholds)
        report["arms"]["B0"]["datasets"][dataset] = base
        scores = {}
        for name, model in models.items():
            scores[name] = score_model(z, model)
            report["arms"][name]["datasets"][dataset] = evaluate_scores(rows, scores[name], source_thresholds[name], baseline_thresholds)
        for weight in plan()["fusion_weights"] if "baseline" in groups else []:
            name = f"B0_PLUS_FULL_{weight:g}"
            scores[name] = fusion_score(rows["cls_prob"], scores["FULL"], cdf_reference, weight)
            if weight == 0 and not np.array_equal(scores[name], rows["cls_prob"]):
                raise ValueError("Zero-weight baseline preservation failed")
            report["arms"][name]["datasets"][dataset] = evaluate_scores(rows, scores[name], fusion_thresholds[name], baseline_thresholds)
        if anchor_prob is not None:
            for weight in plan()["fusion_weights"]:
                name = f"ANCHOR_PLUS_FULL_{weight:g}"
                scores[name] = fusion_score(anchor_prob, scores["FULL"], cdf_reference, weight)
                if weight == 0 and not np.array_equal(scores[name], anchor_prob):
                    raise ValueError("Original B0 zero-weight preservation failed")
                report["arms"][name]["datasets"][dataset] = evaluate_scores(anchor_eval, scores[name], fusion_thresholds[name], anchor_thresholds)
        np.savez_compressed(output / "scores" / f"{dataset}.npz", **scores)
        if "adaptation" in groups:
            _adapt_dataset(dataset, rows, z, models, scores, source_thresholds, baseline_thresholds,
                           report, output, seed, budgets, calibration_count, repeats, anchor_prob, anchor_thresholds)
        atomic_json(output / "all_results.json", report)
        del z
    for item in report["arms"].values():
        if item.get("datasets"):
            item["status"] = "SCORED" if len(item["datasets"]) == len(TEST_DS) else "PARTIAL_DATASET_AVAILABILITY"
    _means(report)
    unavailable = [k for k, v in report["arms"].items() if v["status"] != "SCORED"]
    report.update(status="COMPLETE_WITH_UNAVAILABLE" if unavailable or report["adaptation_unavailable"] else "OK",
                  coverage={"requested_source_arms": len(specifications), "unavailable_arms": unavailable,
                            "target_unavailable_count": len(report["adaptation_unavailable"])})
    atomic_json(output / "all_results.json", report)
    _csv(report, output)
    return report


def _adapt_dataset(dataset, rows, z, models, source_scores, source_thresholds, baseline_thresholds,
                   report, output, seed, budgets, calibration_count, repeats, anchor_prob=None, anchor_thresholds=None):
    real_ids = np.unique(rows["video_id"][rows["labels"] == 0])
    feasible = [b for b in budgets if len(real_ids) >= b + calibration_count + 20]
    for budget in set(budgets) - set(feasible):
        report["adaptation_unavailable"].append({"dataset": dataset, "budget": budget,
            "reason": "insufficient distinct real videos for fit + calibration + 20 held-out; budget not reduced"})
    if not feasible:
        return
    for repeat in range(repeats):
        split_seed = stable_seed(seed, f"target:{dataset}:{repeat}")
        for budget in feasible:
            partition = split_target_real(rows["video_id"], rows["labels"], budget,
                                          calibration_count, split_seed, reserve_budget=max(feasible))
            tag = f"N{budget}_R{repeat}"
            atomic_json(output / "adaptation_splits" / f"{dataset}_{tag}.json", partition)
            fit_groups = [np.asarray(z[rows["video_id"] == v]) for v in partition["fit"]]
            cal_mask = np.isin(rows["video_id"], partition["calibration"])
            test_mask = np.isin(rows["video_id"], partition["test"])
            cal, test = subset(rows, cal_mask), subset(rows, test_mask)
            if np.any(cal["labels"] != 0):
                raise ValueError("Target calibration contains fake labels")
            target_base_thresholds = thresholds(cal, cal["cls_prob"])
            prefix = f"TARGET_{tag}"
            saved = {"native_row_indices": np.flatnonzero(test_mask)}
            def record(name, values, theta):
                key = f"{prefix}_{name}"
                uses_fit = any(name.endswith("_" + mode) for mode in ("mean", "cov", "mean_cov"))
                uses_calibration = name.endswith("_threshold") or name.endswith("TARGET_THRESHOLD") or uses_fit
                item = report["arms"].setdefault(key, {"scope": "trusted_target_real", "status": "RUNNING", "datasets": {},
                     "reserved_fit_budget": budget, "fit_real_videos": budget if uses_fit else 0,
                     "calibration_real_videos": calibration_count if uses_calibration else 0, "repeat": repeat})
                item["datasets"][dataset] = evaluate_scores(test, values, theta, target_base_thresholds)
                saved[name] = values
            record("B0_SOURCE_THRESHOLD", test["cls_prob"], baseline_thresholds)
            record("B0_TARGET_THRESHOLD", test["cls_prob"], target_base_thresholds)
            record("FULL_SOURCE", source_scores["FULL"][test_mask], source_thresholds["FULL"])
            if anchor_prob is not None:
                anchor_cal = {**cal, "cls_prob": anchor_prob[cal_mask]}
                record("ANCHOR_SOURCE_THRESHOLD", anchor_prob[test_mask], anchor_thresholds)
                record("ANCHOR_TARGET_THRESHOLD", anchor_prob[test_mask], thresholds(anchor_cal, anchor_cal["cls_prob"]))
            for base_name in ("FULL", "VM_LEGACY"):
                if base_name not in models:
                    continue
                for mode in plan()["adaptation_modes"]:
                    adapted = adapt_model(models[base_name], fit_groups, mode, .5, .1)
                    cal_score = score_model(z[cal_mask], adapted)
                    theta = thresholds(cal, cal_score)
                    values = score_model(z, adapted)[test_mask]
                    name = f"{base_name}_{mode}"
                    record(name, values, theta)
                    save_model(output / "models" / f"{dataset}_{tag}_{name}.npz", adapted,
                               {"scope": "trusted_target_real", "split_file": f"{dataset}_{tag}.json", "mode": mode})
            np.savez_compressed(output / "adaptation_scores" / f"{dataset}_{tag}.npz", **saved)

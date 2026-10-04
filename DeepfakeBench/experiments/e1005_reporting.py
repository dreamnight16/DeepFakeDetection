"""E1005 development-only selection and video-level, paired ranking reports."""

from collections.abc import Mapping
import csv
import math
from numbers import Real
import os
from pathlib import Path
import tempfile


def select_steps(history):
    if not history or any(not isinstance(row.get("step"), int) or row["step"] < 0 for row in history):
        raise ValueError("Nonempty valid step history required")
    ordered = sorted(history, key=lambda row: row["step"])
    if len({row["step"] for row in ordered}) != len(ordered):
        raise ValueError("Duplicate evaluation step")
    paths = {"S1": ("source", "frame_auc"), "S2": ("source", "video_auc"),
             "S3": ("development", "video_auc")}
    result = {}
    for selector, (section, metric) in paths.items():
        if any(not math.isfinite(row[section][metric]) for row in ordered):
            raise ValueError("Selector metrics must be finite")
        result[selector] = max(ordered, key=lambda row: row[section][metric])["step"]
    result["S4"] = ordered[-1]["step"]
    return result


def choose_champion(results, allowed_families=None):
    candidates = []
    for gid, row in results.items():
        if row.get("status") not in ("OK", "REUSED") or (allowed_families and gid.split("_", 1)[0] not in allowed_families):
            continue
        auc = row.get("primary_auc")
        if not isinstance(auc, (int, float)) or not math.isfinite(auc):
            raise ValueError(f"Invalid development score for {gid}")
        candidates.append(((-auc, row["selected_step"], row["trainable_parameters"], gid), gid))
    return min(candidates)[1] if candidates else None


def ranking_changes(labels, baseline, candidate, block_size=1024):
    import numpy as np
    from e0924_decision import auc
    y, before, after = np.asarray(labels), np.asarray(baseline), np.asarray(candidate)
    if y.ndim != 1 or before.shape != y.shape or after.shape != y.shape or not np.isin(y, (0, 1)).all():
        raise ValueError("Aligned binary video ranking arrays required")
    if not np.isfinite(before).all() or not np.isfinite(after).all() or block_size < 1:
        raise ValueError("Finite ranking scores and positive block size required")
    real, fake = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if not len(real) or not len(fake):
        raise ValueError("Ranking changes require both classes")
    repair = damage = 0.
    for start in range(0, len(fake), block_size):
        indices = fake[start:start + block_size]
        old_difference = before[indices, None] - before[real][None, :]
        new_difference = after[indices, None] - after[real][None, :]
        old = (old_difference > 0).astype(float) + .5 * (old_difference == 0)
        new = (new_difference > 0).astype(float) + .5 * (new_difference == 0)
        change = new - old
        repair += np.maximum(change, 0).sum()
        damage += np.maximum(-change, 0).sum()
    denominator = len(real) * len(fake)
    result = {"repair": float(repair / denominator), "damage": float(damage / denominator),
              "delta_auc": float(auc(y, after) - auc(y, before)), "num_video_pairs": denominator}
    if abs(result["repair"] - result["damage"] - result["delta_auc"]) > 1e-10:
        raise RuntimeError("Ranking decomposition does not match AUC difference")
    return result


def video_arrays(values, score_key="score"):
    import numpy as np
    import run_e1002 as metrics_module
    y, p = metrics_module.video_rows(values, values[score_key])
    return y, p


def metrics(values, threshold=.5):
    import run_e1002 as metrics_module
    return metrics_module.metrics(values, values["score"], threshold)


def calibration_threshold(values, target_fpr=.05):
    import run_e1002 as metrics_module
    return metrics_module.validation_threshold(values, values["score"], target_fpr)


def compare_exports(before, after, threshold, baseline_threshold):
    import numpy as np
    import run_e1001 as legacy
    legacy.verify_paired_base(before, after)
    y, old = video_arrays(before)
    y_after, new = video_arrays(after)
    if not np.array_equal(y, y_after):
        raise ValueError("Paired video labels differ")
    report = metrics(after, threshold)
    report["ranking_changes"] = ranking_changes(y, old, new)
    report["baseline"] = metrics(before, baseline_threshold)
    report["shared_baseline_threshold"] = metrics(after, baseline_threshold)
    return report


def paired_macro_bootstrap(exports, repeats=2000, seed=1024):
    """Stratified paired video resampling within each fixed reporting domain.

    Source-cluster resampling requires separately audited source groups. This
    fallback explicitly makes a video-independence approximation; frame/pair
    counts are never used as independent units.
    """
    import numpy as np
    from e0924_decision import auc
    if repeats < 1 or not exports:
        raise ValueError("Positive bootstrap budget and datasets required")
    arrays = []
    for before, after in exports.values():
        y, old = video_arrays(before)
        _, new = video_arrays(after)
        if old.shape != new.shape:
            raise ValueError("Misaligned bootstrap arrays")
        arrays.append((y, old, new, np.flatnonzero(y == 0), np.flatnonzero(y == 1)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        deltas = []
        for y, old, new, real, fake in arrays:
            indices = np.r_[rng.choice(real, len(real)), rng.choice(fake, len(fake))]
            deltas.append(auc(y[indices], new[indices]) - auc(y[indices], old[indices]))
        draws.append(np.mean(deltas))
    return {"mean_delta": float(np.mean([auc(y, new) - auc(y, old) for y, old, new, _, _ in arrays])),
            "ci95": np.quantile(draws, [.025, .975]).tolist(), "repeats": repeats, "seed": seed,
            "unit": "stratified paired videos", "source_clustering": "unavailable: video-independence approximation"}


def _finite_number(value, name, probability=False):
    if (not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
            or (probability and not 0 <= value <= 1)):
        raise ValueError(f"{name} must be finite" + (" and in [0,1]" if probability else ""))
    return float(value)


def assess_breakthrough(dataset_reports, bootstrap, reproduction, formal_budget=True, ordinary_comparison=None):
    """Apply predeclared historical-six-domain AUC and separate FPR gates.

    Inputs are JSON-style reports from compare_exports. The caller establishes
    identical inputs and valid source/config receipts. The full breakthrough
    gate also needs a comparison against the strongest predeclared ordinary
    recipe on the champion's same inputs. This comparison cannot change the
    development-selected champion. A new-method increment requires a strictly
    positive mean and CI95 lower bound and excludes ordinary recipe champions.
    """
    from e1005_protocol import REGRESSION_DATASETS

    if not isinstance(dataset_reports, Mapping) or set(dataset_reports) != set(REGRESSION_DATASETS):
        raise ValueError("Breakthrough assessment requires the exact six regression dataset names")
    if not isinstance(formal_budget, bool):
        raise ValueError("formal_budget must be boolean")
    if reproduction is not None and not isinstance(reproduction, Mapping):
        raise ValueError("reproduction must be a report mapping or None")
    auc_deltas, fpr_deltas = {}, {}
    candidate_aucs, baseline_aucs = [], []
    for dataset in REGRESSION_DATASETS:
        row = dataset_reports[dataset]
        if not isinstance(row, Mapping) or not isinstance(row.get("baseline"), Mapping):
            raise ValueError(f"{dataset} requires a candidate and baseline report")
        candidate = _finite_number(row.get("video_auc"), f"{dataset} video_auc", probability=True)
        baseline = _finite_number(row["baseline"].get("video_auc"), f"{dataset} baseline video_auc", probability=True)
        fpr = _finite_number(row.get("video_fpr_at_val_threshold"), f"{dataset} calibrated video FPR", probability=True)
        old_fpr = _finite_number(row["baseline"].get("video_fpr_at_val_threshold"),
                                 f"{dataset} baseline calibrated video FPR", probability=True)
        auc_deltas[dataset], fpr_deltas[dataset] = candidate - baseline, fpr - old_fpr
        candidate_aucs.append(candidate)
        baseline_aucs.append(baseline)
    lower = None
    if bootstrap is not None:
        if not isinstance(bootstrap, Mapping):
            raise ValueError("bootstrap must be a report mapping or None")
        interval = bootstrap.get("ci95")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("bootstrap requires a finite ordered ci95 interval")
        lower = _finite_number(interval[0], "bootstrap ci95 lower")
        upper = _finite_number(interval[1], "bootstrap ci95 upper")
        if lower > upper:
            raise ValueError("bootstrap ci95 must be ordered")
    ordinary = None
    if ordinary_comparison is not None:
        if not isinstance(ordinary_comparison, Mapping):
            raise ValueError("ordinary_comparison must be a report mapping or None")
        reference_id = ordinary_comparison.get("ordinary_baseline_id")
        if not isinstance(reference_id, str) or not reference_id.strip():
            raise ValueError("ordinary comparison requires an ordinary_baseline_id")
        reference_delta = _finite_number(ordinary_comparison.get("mean_delta"), "ordinary comparison mean_delta")
        reference_interval = ordinary_comparison.get("ci95")
        if not isinstance(reference_interval, (list, tuple)) or len(reference_interval) != 2:
            raise ValueError("ordinary comparison requires a finite ordered ci95 interval")
        reference_lower = _finite_number(reference_interval[0], "ordinary comparison ci95 lower")
        reference_upper = _finite_number(reference_interval[1], "ordinary comparison ci95 upper")
        if reference_lower > reference_upper:
            raise ValueError("ordinary comparison ci95 must be ordered")
        reference_recipe = ordinary_comparison.get("champion_is_reference_recipe", False)
        if not isinstance(reference_recipe, bool):
            raise ValueError("champion_is_reference_recipe must be boolean")
        ordinary = {"ordinary_baseline_id": reference_id, "mean_delta": reference_delta,
                    "ci95": [reference_lower, reference_upper], "champion_is_reference_recipe": reference_recipe}
    mean_delta = math.fsum(auc_deltas.values()) / 6
    nondecreasing = sum(delta >= 0 for delta in auc_deltas.values())
    minimum_delta = min(auc_deltas.values())
    # Only absorb floating subtraction roundoff at inclusive decimal gates.
    tolerance = 1e-12
    gates = {"mean6_gain": mean_delta >= .005 - tolerance,
             "four_nondecreasing_domains": nondecreasing >= 4,
             "maximum_domain_drop": minimum_delta >= -.01 - tolerance,
             "positive_paired_ci95_lower": lower is not None and lower > 0}
    auc_breakthrough = all(gates.values())
    fpr_controlled = max(fpr_deltas.values()) <= .02 + tolerance
    repeat_verified = reproduction is not None and reproduction.get("passed") is True
    can_claim_base_improvement = auc_breakthrough and repeat_verified and formal_budget
    ordinary_not_better = ordinary is not None and ordinary["mean_delta"] >= 0
    method_increment_supported = (ordinary is not None and ordinary["mean_delta"] > 0
                                  and ordinary["ci95"][0] > 0 and not ordinary["champion_is_reference_recipe"])
    can_claim_breakthrough = can_claim_base_improvement and ordinary_not_better
    limitations = ["Historical six-domain regression panel; no independent new-source confirmation.",
                   "Same-seed reproduction does not estimate cross-seed variability."]
    reasons = {"mean6_gain": "Mean6 AUC gain is below 0.005.",
               "four_nondecreasing_domains": "Fewer than four domains have nonnegative AUC changes.",
               "maximum_domain_drop": "A domain AUC drop exceeds 0.01.",
               "positive_paired_ci95_lower": "Paired macro bootstrap CI95 lower bound is missing or not positive."}
    limitations.extend(reasons[name] for name, passed in gates.items() if not passed)
    if not repeat_verified:
        limitations.append("Same-seed reproduction has not passed verification.")
    if not formal_budget:
        limitations.append("Formal training budget is not established; this run cannot support a breakthrough claim.")
    if not fpr_controlled:
        limitations.append("Calibrated cross-domain FPR increases by more than 0.02 in at least one domain.")
    if bootstrap is not None and bootstrap.get("source_clustering"):
        limitations.append(str(bootstrap["source_clustering"]))
    if ordinary is None:
        limitations.append("Predeclared ordinary-baseline comparison is missing; no full breakthrough claim is supported.")
    elif not ordinary_not_better:
        limitations.append("The strongest predeclared ordinary baseline exceeds the champion; B0 improvement is distinct.")
    if ordinary is not None:
        if ordinary["champion_is_reference_recipe"]:
            limitations.append("The champion is an ordinary reference recipe; no new-method increment is claimed.")
        elif not method_increment_supported:
            limitations.append("Ordinary-baseline comparison does not support a strictly positive new-method increment.")
    return {"auc_breakthrough": auc_breakthrough, "cross_domain_fpr_controlled": fpr_controlled,
            "can_claim_breakthrough": can_claim_breakthrough,
            "base_improvement_supported": auc_breakthrough, "can_claim_base_improvement": can_claim_base_improvement,
            "method_increment_supported": method_increment_supported,
            "can_claim_method_increment": can_claim_breakthrough and method_increment_supported,
            "ordinary_reference_not_better": ordinary_not_better, "ordinary_comparison": ordinary,
            "limitations": limitations, "mean6_video_auc": math.fsum(candidate_aucs) / 6,
            "baseline_mean6_video_auc": math.fsum(baseline_aucs) / 6,
            "mean6_delta_auc": mean_delta, "nondecreasing_domains": nondecreasing,
            "minimum_domain_delta_auc": minimum_delta, "dataset_delta_auc": auc_deltas,
            "dataset_delta_fpr_at_val_threshold": fpr_deltas, "auc_gates": gates,
            "reproduction_verified": repeat_verified, "formal_budget": formal_budget}


def _validated_repeat_export(values, name):
    import numpy as np

    required = ("labels", "path", "video_id", "cls_prob", "global_log_odds", "score")
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be an export mapping")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"{name} missing paired identity/B0 arrays: {missing}")
    arrays = {key: np.asarray(values[key]) for key in required}
    labels = arrays["labels"]
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{name} requires nonempty binary paired labels")
    for key, values_array in arrays.items():
        if values_array.shape != labels.shape:
            raise ValueError(f"{name} paired {key} must match labels shape")
        if key in ("score", "cls_prob", "global_log_odds"):
            if values_array.dtype.kind not in "buif" or not np.isfinite(values_array).all():
                raise ValueError(f"{name} {key} must be finite real numeric arrays")
            if key != "global_log_odds" and ((values_array < 0) | (values_array > 1)).any():
                raise ValueError(f"{name} {key} must contain probabilities in [0,1]")
        elif key in ("path", "video_id") and any(not str(item) for item in values_array):
            raise ValueError(f"{name} paired {key} must contain nonempty identities")
    return arrays


def _bitwise_array_equal(first, second):
    return first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes()


def compare_reproduction(reference_exports, repeated_exports, reference_selected_step, repeated_selected_step):
    """Compare same-seed prediction arrays after verifying paired identities.

    ``passed`` is the numerical gate (macro AUC difference <=0.0005 and each
    domain <=0.001). Source/config/manifests receipts are checked by the caller.
    A different selected step is recorded independently; bitwise agreement is
    stricter than passing the numerical gate and is reported without inference.
    """
    import numpy as np
    from e0924_decision import auc

    if (not isinstance(reference_exports, Mapping) or not isinstance(repeated_exports, Mapping)
            or not reference_exports or set(reference_exports) != set(repeated_exports)
            or any(not isinstance(name, str) for name in reference_exports)):
        raise ValueError("Reproduction requires matching nonempty dataset export mappings")
    for name, step in (("reference_selected_step", reference_selected_step),
                       ("repeated_selected_step", repeated_selected_step)):
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    datasets, auc_differences = {}, {}
    for dataset in sorted(reference_exports):
        before = _validated_repeat_export(reference_exports[dataset], f"{dataset} reference")
        after = _validated_repeat_export(repeated_exports[dataset], f"{dataset} repeat")
        for key in ("labels", "path", "video_id"):
            if not np.array_equal(before[key], after[key]):
                raise ValueError(f"{dataset} paired identity changed: {key}")
        for key in ("cls_prob", "global_log_odds"):
            if not _bitwise_array_equal(before[key], after[key]):
                raise ValueError(f"{dataset} paired B0 output changed: {key}")
        y, first_scores = video_arrays(before)
        repeated_y, repeat_scores = video_arrays(after)
        if not np.array_equal(y, repeated_y):
            raise ValueError(f"{dataset} paired video labels changed")
        first_auc, repeat_auc = auc(y, first_scores), auc(y, repeat_scores)
        delta = repeat_auc - first_auc
        auc_differences[dataset] = delta
        datasets[dataset] = {
            "reference_video_auc": first_auc, "repeated_video_auc": repeat_auc, "delta_video_auc": delta,
            "bitwise_equal": _bitwise_array_equal(before["score"], after["score"]),
            "max_abs_difference": float(np.max(np.abs(after["score"].astype(np.float64)
                                                          - before["score"].astype(np.float64)))),
            "num_frames": len(before["labels"]), "num_videos": len(y)}
    macro_delta = math.fsum(auc_differences.values()) / len(datasets)
    tolerance = 1e-12
    passed = (abs(macro_delta) <= .0005 + tolerance
              and all(abs(delta) <= .001 + tolerance for delta in auc_differences.values()))
    return {"passed": passed, "bitwise_equal": all(row["bitwise_equal"] for row in datasets.values()),
            "max_abs_difference": max(row["max_abs_difference"] for row in datasets.values()),
            "macro_auc_difference": macro_delta, "individual_auc_differences": auc_differences,
            "datasets": datasets, "dataset_names": sorted(datasets), "identity_verified": True,
            "reference_selected_step": reference_selected_step, "repeated_selected_step": repeated_selected_step,
            "selected_step_consistent": reference_selected_step == repeated_selected_step,
            "scope": "same-seed repeat; no cross-seed variability estimate"}


def write_analysis_csv(path, results, evaluation):
    """Atomically extract arm/domain analysis rows from the JSON source reports."""
    if not isinstance(results, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("results and evaluation must be JSON-style mappings")
    reports = evaluation.get("reports", {})
    assessment = evaluation.get("assessment", {})
    if not isinstance(reports, Mapping) or not isinstance(assessment, Mapping):
        raise ValueError("evaluation reports and assessment must be mappings")
    fields = ["g_id", "family", "design_id", "status", "selected_step", "trainable_parameters", "primary_auc",
              "dataset", "video_auc", "baseline_video_auc", "delta_video_auc", "video_fpr_at_val_threshold",
              "baseline_video_fpr_at_val_threshold", "delta_video_fpr_at_val_threshold", "ranking_repair",
              "ranking_damage", "ranking_delta_auc", "champion", "auc_breakthrough",
              "cross_domain_fpr_controlled", "can_claim_breakthrough"]
    rows = []
    for gid in sorted(results):
        result = results[gid]
        if not isinstance(gid, str) or not isinstance(result, Mapping):
            raise ValueError("results must map arm names to result mappings")
        design_id = result.get("design_id")
        if design_id is None:
            from e1005_protocol import resolve_arms
            try:
                design_id = resolve_arms([gid])[0]["design_id"]
            except ValueError:
                design_id = ""
        shared = {key: result.get(key, "") for key in ("status", "selected_step", "trainable_parameters", "primary_auc")}
        is_champion = gid == evaluation.get("champion")
        shared.update(g_id=gid, family=result.get("family", gid.split("_", 1)[0]), design_id=design_id,
                      champion=is_champion)
        for key in ("auc_breakthrough", "cross_domain_fpr_controlled", "can_claim_breakthrough"):
            shared[key] = assessment.get(key, "") if is_champion else ""
        domain_reports = reports.get(gid, {})
        if not isinstance(domain_reports, Mapping):
            raise ValueError(f"{gid} domain reports must be mappings")
        if not domain_reports:
            rows.append(shared)
        for dataset in sorted(domain_reports):
            report = domain_reports[dataset]
            if not isinstance(report, Mapping) or not isinstance(report.get("baseline"), Mapping):
                raise ValueError(f"{gid}/{dataset} requires candidate and baseline metric mappings")
            baseline = report["baseline"]
            ranking = report.get("ranking_changes", {})
            if not isinstance(ranking, Mapping):
                raise ValueError(f"{gid}/{dataset} ranking changes must be a mapping")
            row = {**shared, "dataset": dataset, "video_auc": report.get("video_auc", ""),
                   "baseline_video_auc": baseline.get("video_auc", ""),
                   "video_fpr_at_val_threshold": report.get("video_fpr_at_val_threshold", ""),
                   "baseline_video_fpr_at_val_threshold": baseline.get("video_fpr_at_val_threshold", ""),
                   "ranking_repair": ranking.get("repair", ""), "ranking_damage": ranking.get("damage", ""),
                   "ranking_delta_auc": ranking.get("delta_auc", "")}
            for metric in ("video_auc", "video_fpr_at_val_threshold"):
                candidate, old = report.get(metric), baseline.get(metric)
                row[f"delta_{metric}"] = (candidate - old if isinstance(candidate, Real) and isinstance(old, Real) else "")
            rows.append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return str(path)

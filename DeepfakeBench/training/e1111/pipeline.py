"""Server-side S0, final calibration and strict video-score export."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import scipy

from g21.io import atomic_json, file_digest

from .features import FrozenFeatures, load_audit, load_cache, read_rgb
from .metric import (FisherRequired, ProtocolError, choose_tau, conformal_threshold, control_spectrum, decompose, inverse_sqrt,
                     nested_d1, overlap, real_statistics, score, second_moment)
from .protocol import digest, implementation_digest, protocol


def _real_groups(audit, lookup, split):
    return [np.stack([lookup[r["path"]] for r in rows])
            for _, rows in sorted(audit["real_frames"][split].items())]


def paired_features(audit, lookup):
    cells = defaultdict(dict)
    for row in audit["fit_pairs"]:
        key = (row["source"], row["method"])
        if row["frame"] in cells[key]:
            raise ProtocolError("Duplicate balanced cell frame")
        cells[key][row["frame"]] = lookup[row["fake_path"]] - lookup[row["real_path"]]
    sources = audit["complete_sources"]
    delta = np.stack([np.stack([np.stack([x for _, x in sorted(cells[(v, m)].items())])
                               for m in protocol()["methods"]]) for v in sources])
    if delta.shape[:3] != (audit["N4"], 4, audit["K_pair"]):
        raise ProtocolError("Cache/manifest does not form the registered balanced tensor")
    held = {m: defaultdict(list) for m in protocol()["methods"]}
    for row in audit["sel_pairs"]:
        if row["split"] != "sel" or row["source"] not in audit["splits"]["sel"]:
            raise ProtocolError("D1 validation contains non-D_sel_FF++ data")
        held[row["method"]][row["source"]].append(lookup[row["fake_path"]] - lookup[row["real_path"]])
    return delta, {m: {s: np.stack(x) for s, x in groups.items()} for m, groups in held.items()}


def video_mean_features(audit, lookup):
    """One difference of independently averaged video features per source/method."""
    methods = protocol("video_mean")["methods"]
    real = {source: np.mean([lookup[row["path"]] for row in rows], axis=0)
            for split in ("fit", "sel") for source, rows in audit["real_frames"][split].items()}
    def difference(bag):
        if len(bag["frames"]) < 2:
            raise ProtocolError("Video-mean bag contains fewer than two frames")
        return np.mean([lookup[row["fake_path"]] for row in bag["frames"]], axis=0) - real[bag["source"]]
    cells = {}
    for bag in audit["fit_videos"]:
        key = (bag["source"], bag["method"])
        if key in cells or bag["source"] not in audit["complete_sources"]:
            raise ProtocolError("Duplicate or unexpected fit video-mean cell")
        cells[key] = difference(bag)
    if len(cells) != audit["N4"] * len(methods):
        raise ProtocolError("Video-mean design is not complete across four methods")
    delta = np.stack([np.stack([cells[(v, m)] for m in methods]) for v in audit["complete_sources"]])
    held = {m: {} for m in methods}
    for bag in audit["sel_videos"]:
        source, method = bag["source"], bag["method"]
        if source not in audit["splits"]["sel"] or source in held[method]:
            raise ProtocolError("Duplicate or non-D_sel video-mean validation cell")
        held[method][source] = difference(bag)[None, :]
    return delta, held


def video_real_controls(audit, lookup):
    """A full real mean for shuffling, and a nontrivial two-bag real-only control."""
    means, contrasts = [], []
    for source in audit["complete_sources"]:
        rows = sorted(audit["real_frames"]["fit"][source], key=lambda row: row["frame"])
        values = np.stack([lookup[row["path"]] for row in rows])
        if len(values) < 2:
            raise ProtocolError("Real-only video-mean control requires at least two frames")
        first, second = np.array_split(values, 2)
        means.append(values.mean(axis=0))
        contrasts.append(first.mean(axis=0) - second.mean(axis=0))
    return np.stack(means), np.stack(contrasts)


def run_s0(audit_path, cache_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"experiment": "E1111", "branch": "E1111A", "status": "S0_STOPPED",
              "protocol": protocol(), "local_execution": False}
    try:
        audit = load_audit(audit_path)
        statistic_mode = audit.get("statistic_mode", "paired_frame")
        settings = protocol(statistic_mode)
        video_mode = statistic_mode == "video_mean"
        report.update(protocol=settings, statistic_mode=statistic_mode)
        arrays, metadata, lookup = load_cache(cache_path, audit)
        fit_real, sel_real = _real_groups(audit, lookup, "fit"), _real_groups(audit, lookup, "sel")
        tau, tau_table = choose_tau(fit_real, sel_real)
        mu, covariance, temporal = real_statistics(fit_real, tau)
        delta, held = (video_mean_features if video_mode else paired_features)(audit, lookup)
        B, _, d7 = decompose(delta)
        dc = second_moment(arrays["compression_deltas"])
        family = "B" if video_mode or d7["rho"]["resid"] < .8 else "Fisher"
        report.update(P_D7={"passed": True, "N4": audit["N4"], "Ncell": audit["Ncell"],
                            "K_pair": audit["K_pair"]}, tau_R=tau, tau_selection=tau_table,
                      D7={**d7, "passed": None if video_mode else family == "B",
                          "status": "DESCRIPTIVE_VIDEO_MEAN_FOUR_PART" if video_mode else "PAIRED_FRAME_FIVE_PART"}, family=family,
                      audit_sha256=audit["audit_sha256"], feature_identity=metadata,
                      cache_sha256=file_digest(cache_path))
        # Fisher calls the same nested procedure on Fisher-specific statistics
        # in every two/three-method fold, never inheriting a B-subspace D1 pass.
        try:
            space, d1 = nested_d1(delta, covariance, dc, held, family, settings)
        except FisherRequired as exc:
            if video_mode or family == "Fisher":
                raise
            report["numeric_fisher_trigger"] = str(exc)
            family = "Fisher"
            report["family"] = family
            space, d1 = nested_d1(delta, covariance, dc, held, family, settings)
        rank = d1["final_rank"]
        d2 = overlap(space, temporal, rank, "D2_temporal")
        d8 = {}
        if "transform_deltas" in arrays:
            for name in sorted(set(arrays["transform_name"].tolist())):
                sources = sorted(set(arrays["transform_source"].tolist()))
                moment = np.zeros_like(covariance)
                for source in sources:
                    mask = (arrays["transform_name"] == name) & (arrays["transform_source"] == source)
                    moment += second_moment(arrays["transform_deltas"][mask])
                d8[name] = overlap(space, moment / len(sources), rank, f"D8_{name}")
        eigenvalues = np.maximum(np.linalg.eigvalsh(B)[-32:], 0)
        effective_rank = float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())
        status = ("S0_PASS" if family == "B" else "FISHER_D1_PASS_PENDING_R2") if d1["passed"] else "D1_FAILED_REGULARIZATION_ONLY"
        report.update(status=status, D1=d1, D2=d2, D8=d8 or {"status": "NOT_REQUESTED"},
                      D3={"status": "DESCRIPTIVE_ONLY", "R3_enabled": False},
                      D4={"status": "NOT_RUN", "note": "Non-gating gap-statistic diagnostic"},
                      B_top32_effective_rank=effective_rank,
                      full_mechanism_eligible=bool(not video_mode and family == "B" and d1["passed"] and d2["passed"]),
                      claim_scope="video-average differences; no framewise cancellation or original D7 mechanism claim" if video_mode
                                  else "legacy paired-frame differences",
                      claim_status="S1/S2/S3 not yet evaluated; no mechanism conclusion",
                      numeric_diagnostics=space.diagnostics)
        model = {"mu": mu, "white": inverse_sqrt(covariance), "transform": space.transform,
                 "vectors": space.vectors[:, :rank], "weights": space.weights[:rank]}
        # Controls share Sigma_R, D_nuis, r and the v coordinates. Only the
        # direction source/spectrum changes. QR is used solely to generate the
        # random orthogonal control, never to replace the fitted metric.
        rng = np.random.default_rng(settings["seed"])
        random_basis, _ = np.linalg.qr(rng.normal(size=(len(mu), rank)), mode="reduced")
        model.update(C_random_V=random_basis, C_random_w=rng.permutation(space.weights[:rank]))
        if video_mode:
            real_tensor, real_difference = video_real_controls(audit, lookup)
        else:
            real_by_source = {}
            for row in audit["fit_pairs"]:
                if row["method"] == settings["methods"][0]:
                    real_by_source.setdefault(row["source"], {})[row["frame"]] = lookup[row["real_path"]]
            real_tensor = np.stack([np.stack([x for _, x in sorted(real_by_source[v].items())])
                                    for v in audit["complete_sources"]])
            real_difference = real_tensor - np.roll(real_tensor, 1, axis=1)
        order = rng.permutation(len(real_tensor))
        permutation = np.empty(len(order), dtype=int)
        permutation[order] = np.roll(order, 1)
        shuffled = delta + real_tensor[:, None] - real_tensor[permutation, None]
        numerators = {"C_shuffle": second_moment(shuffled), "C_real": second_moment(real_difference)}
        if family == "Fisher":
            numerators["C_shuffle"] = decompose(shuffled)[1]["method"]
            # A method-mean Fisher control cannot be learned from true-only
            # differences replicated across four identical method slots.
            numerators.pop("C_real")
        omitted_controls = {}
        for name, numerator in numerators.items():
            try:
                vectors, weights = control_spectrum(space, numerator, rank)
                model.update({f"{name}_V": vectors, f"{name}_w": weights})
            except ProtocolError as exc:
                omitted_controls[name] = str(exc)
        if family == "Fisher":
            omitted_controls["C_real"] = "Undefined method-mean numerator for real-only Fisher control"
        report["omitted_controls"] = omitted_controls
        identity = {"audit_sha256": audit["audit_sha256"], "protocol_sha256": digest(settings),
                    "statistic_mode": statistic_mode,
                    "metric_implementation_sha256": implementation_digest("metric"),
                    "feature_identity": metadata, "status": status, "family": family,
                    "rank": rank, "lambda_m": d1["final_lambda_m"], "tau_R": tau,
                    "fit_versions": {"numpy": np.__version__, "scipy": scipy.__version__},
                    "omitted_controls": omitted_controls}
        with (output / "metric.npz").open("xb") as stream:
            np.savez_compressed(stream, **model, metadata=np.asarray(json.dumps(identity, sort_keys=True)))
    except (ValueError, OSError, KeyError, np.linalg.LinAlgError) as exc:
        report.update(status="S0_STOPPED", full_mechanism_eligible=False,
                      error=f"{type(exc).__name__}: {exc}")
    atomic_json(output / "s0.json", report)
    return report


def load_metric(path):
    with np.load(path, allow_pickle=False) as arrays:
        model = {key: arrays[key] for key in arrays.files if key != "metadata"}
        identity = json.loads(str(arrays["metadata"]))
    if identity["protocol_sha256"] != digest(protocol(identity.get("statistic_mode", "paired_frame"))):
        raise ProtocolError("Metric protocol is obsolete")
    if identity.get("metric_implementation_sha256") != implementation_digest("metric"):
        raise ProtocolError("Metric implementation changed; refit rather than reuse obsolete results")
    return model, identity


def video_means(scores, video_ids, labels=None):
    groups = defaultdict(list)
    for index, name in enumerate(video_ids):
        groups[str(name)].append(index)
    names, values, truth = [], [], []
    for name, indices in sorted(groups.items()):
        names.append(name)
        values.append(float(np.mean(scores[indices])))
        if labels is not None:
            unique = set(np.asarray(labels)[indices].tolist())
            if len(unique) != 1:
                raise ProtocolError("Video ID groups real and fake samples together")
            truth.append(unique.pop())
    return np.asarray(names), np.asarray(values), np.asarray(truth) if labels is not None else None


def calibrate(audit_path, cache_path, metric_path, config, checkpoint, clip, output,
              *, device="cuda:0", batch_size=8):
    audit = load_audit(audit_path)
    model, identity = load_metric(metric_path)
    if identity["status"] not in ("S0_PASS", "FISHER_D1_PASS_PENDING_R2"):
        raise ProtocolError("S0/D1 not passed; no automatic formal continuation")
    if identity["audit_sha256"] != audit["audit_sha256"]:
        raise ProtocolError("Metric and audit differ")
    _, cache_metadata, lookup = load_cache(cache_path, audit)
    if cache_metadata != identity["feature_identity"]:
        raise ProtocolError("Metric and selection cache differ")
    real_groups = _real_groups(audit, lookup, "sel")
    if audit.get("statistic_mode") == "video_mean":
        fake_groups = [np.stack([lookup[row["fake_path"]] for row in bag["frames"]])
                       for bag in audit["sel_videos"]]
    else:
        fake_groups = [np.stack([lookup[row["fake_path"]] for row in audit["sel_pairs"]
                                if row["video_id"] == video])
                       for video in sorted({r["video_id"] for r in audit["sel_pairs"]})]
    if not fake_groups:
        raise ProtocolError("No D_sel fake videos are available for gamma selection")
    table = []
    for gamma in protocol()["gamma"]:
        real = [score(x, model, gamma)["D_J"].mean() for x in real_groups]
        fake = [score(x, model, gamma)["D_J"].mean() for x in fake_groups]
        threshold = conformal_threshold(real, .1)
        table.append({"gamma": gamma, "selection_video_TPR": float(np.mean(np.asarray(fake) > threshold))})
    maximum = max(x["selection_video_TPR"] for x in table)
    winner = min((x for x in table if x["selection_video_TPR"] >= maximum - 1e-12),
                 key=lambda x: float("inf") if x["gamma"] == "inf" else x["gamma"])
    gamma = winner["gamma"]
    # Gamma is now frozen. Only here are D_theta images read.
    extractor = FrozenFeatures(config, checkpoint, clip, device, batch_size)
    if any(extractor.identity[key] != cache_metadata[key] for key in extractor.identity):
        raise ProtocolError("Calibration feature extractor differs from S0")
    theta_rows = [(source, row) for source, rows in audit["real_frames"]["theta"].items() for row in rows]
    z = extractor.encode(read_rgb(audit["data_root"], row["path"]) for _, row in theta_rows)
    frame_scores = score(z, model, gamma)
    thresholds = {}
    for detector, values in frame_scores.items():
        _, videos, _ = video_means(values, [source for source, _ in theta_rows])
        thresholds[detector] = {str(alpha): conformal_threshold(videos, alpha)
                               for alpha in protocol()["alpha"]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity.update(gamma=gamma, thresholds=thresholds, gamma_selection=table,
                    metric_sha256=file_digest(metric_path), claim_status="calibrated; R2/controls not yet evaluated")
    with (output / "calibrated_metric.npz").open("xb") as stream:
        np.savez_compressed(stream, **model, metadata=np.asarray(json.dumps(identity, sort_keys=True)))
    atomic_json(output / "calibration.json", identity)
    return identity


def score_export(feature_file, metric_path, output):
    model, identity = load_metric(metric_path)
    if "thresholds" not in identity:
        raise ProtocolError("Metric is not yet calibrated")
    with np.load(feature_file, allow_pickle=False) as arrays:
        features, video_ids, labels = arrays["z"], arrays["video_id"], arrays["label"]
        cluster_ids = arrays["cluster_id"]
        supplied_identity = json.loads(str(arrays["feature_identity"]))
    expected = identity["feature_identity"]
    for key in ("checkpoint_sha256", "base_config_sha256", "feature", "preprocess",
                "feature_implementation_sha256", "versions"):
        if supplied_identity[key] != expected[key]:
            raise ProtocolError(f"Evaluation feature identity mismatch: {key}")
    if (len(features) != len(video_ids) or len(features) != len(labels)
            or len(features) != len(cluster_ids) or not set(labels.tolist()) <= {0, 1}):
        raise ProtocolError("Feature export length/label mismatch")
    clusters_by_video = defaultdict(set)
    for video, cluster in zip(video_ids, cluster_ids):
        clusters_by_video[str(video)].add(str(cluster))
    if any(len(values) != 1 for values in clusters_by_video.values()):
        raise ProtocolError("A video has conflicting source clusters")
    from scipy.stats import beta
    from sklearn.metrics import roc_auc_score

    frame = score(features, model, identity["gamma"])
    report = {"detectors": {}, "identity": identity, "feature_identity": supplied_identity,
              "scope": "evaluation only; no parameter updates"}
    for detector, values in frame.items():
        names, videos, truth = video_means(values, video_ids, labels)
        metrics = {"num_videos": len(names)}
        if len(set(truth.tolist())) == 2:
            metrics["AUC"] = float(roc_auc_score(truth, videos))
        n_real = int((truth == 0).sum())
        for alpha in protocol()["alpha"]:
            key = str(alpha)
            if n_real < 50 or (n_real < 100 and alpha == .05):
                continue
            threshold = identity["thresholds"][detector][key]
            false = int((videos[truth == 0] > threshold).sum())
            ci = [0. if false == 0 else float(beta.ppf(.025, false, n_real - false + 1)),
                  1. if false == n_real else float(beta.ppf(.975, false + 1, n_real - false))]
            metrics[key] = {"threshold": threshold, "FPR": false / n_real, "FPR_CP95": ci}
            if (truth == 1).any():
                metrics[key]["TPR"] = float((videos[truth == 1] > threshold).mean())
        report["detectors"][detector] = metrics
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "video_scores.npz", video_id=names, label=truth,
                        cluster_id=np.asarray([next(iter(clusters_by_video[name])) for name in names]),
                        **{name: video_means(values, video_ids)[1] for name, values in frame.items()})
    atomic_json(output / "metrics.json", report)
    return report


def attribution(evaluation_dirs, output):
    """Paired, dataset-stratified cluster bootstrap on development sources only."""
    datasets = []
    cluster_provenance = {}
    metric_identity = None
    for directory in map(Path, evaluation_dirs):
        report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        supplied = report["feature_identity"]
        if supplied["role"] != "development":
            raise ProtocolError("S1 attribution may not read blind-test results")
        if metric_identity is None:
            metric_identity = report["identity"]
        elif metric_identity != report["identity"]:
            raise ProtocolError("Development evaluations used different frozen metrics")
        with np.load(directory / "video_scores.npz", allow_pickle=False) as values:
            truth = values["label"]
            needed = {"D_R", "D_F", "D_J", "D_P", "C_random", "C_shuffle"}
            if metric_identity["family"] == "B":
                needed.add("C_real")
            if not needed <= set(values.files):
                raise ProtocolError("S1 controls are incomplete; no attribution pass")
            fake = truth == 1
            if not fake.any():
                raise ProtocolError("S1 requires development fake videos")
            predictions = {name: values[name][fake] > metric_identity["thresholds"][name]["0.1"]
                           for name in needed}
            datasets.append((supplied["dataset_id"], predictions, values["cluster_id"][fake]))
            cluster_provenance[supplied["dataset_id"]] = supplied["cluster_provenance"]
    if {name for name, _, _ in datasets} != {"FaceForensics++", "Celeb-DF-v2"} or len(datasets) != 2:
        raise ProtocolError("Formal S1 requires both FF++ val and CDF-v2 train, once each")
    rng = np.random.default_rng(protocol()["seed"])
    deltas = []
    for _ in range(protocol()["bootstrap_repeats"]):
        tpr = defaultdict(list)
        for _, predictions, clusters in datasets:
            unique = sorted(set(clusters.tolist()))
            draw_clusters = rng.choice(unique, len(unique), replace=True)
            draw = np.concatenate([np.flatnonzero(clusters == source) for source in draw_clusters])
            for name, accepted in predictions.items():
                tpr[name].append(float(accepted[draw].mean()))
        mean = {name: np.mean(values) for name, values in tpr.items()}
        deltas.append(mean["D_J"] - max(mean["D_R"], mean["D_F"]))
    tpr = {name: float(np.mean([predictions[name].mean() for _, predictions, _ in datasets]))
           for name in datasets[0][1]}
    lower = float(np.quantile(deltas, .05))
    random_not_reproduced = tpr["D_F"] > 0 and tpr["C_random"] < .9 * tpr["D_F"]
    gamma = metric_identity["gamma"]
    result = {"experiment": "E1111", "branch": "E1111A", "stage": "S1",
              "datasets": [name for name, _, _ in datasets], "macro_TPR_at_0_10": tpr,
              "delta_vs_best_endpoint": tpr["D_J"] - max(tpr["D_R"], tpr["D_F"]),
              "delta_one_sided_lower_95": lower,
              "bootstrap_unit": "paired source clusters, stratified by dataset",
              "cluster_provenance": cluster_provenance,
              "CI_scope": "CDF-v2 uses video proxies where source lineage is unavailable",
              "passed": bool(lower > .01 and random_not_reproduced),
              "random_control_not_reproduced": random_not_reproduced,
              "S2_interior_gamma": bool(gamma != "inf" and float(gamma) > 0),
              "S2_note": "rho_cap is independent of gamma; report D1 separately from the TPR(gamma) curve",
              "S3_identity_audit": "NOT_EVALUATED", "claim_status": "pending S3; no full-mechanism claim",
              "family": metric_identity["family"], "frozen_metric_identity": metric_identity}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "attribution.json", result)
    return result

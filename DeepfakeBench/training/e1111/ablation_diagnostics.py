"""Development-only complementarity and optional actual ArcFace diagnostics."""

import json
from pathlib import Path

import numpy as np

from .audit import canonical_frame_path
from .features import read_rgb
from .metric import ProtocolError
from .ablation_models import unit


def complementarity(run_dir, identity, repeats=1000, seed=1024):
    run_dir = Path(run_dir)
    datasets = []
    for tag in ("dev_ff_scores", "dev_cdf_scores"):
        metadata = json.loads((run_dir / tag / "metrics.json").read_text(encoding="utf-8"))
        if metadata["identity"] != identity or metadata["feature_identity"]["role"] != "development":
            raise ProtocolError("D5 only accepts matching frozen-model development results")
        with np.load(run_dir / tag / "video_scores.npz", allow_pickle=False) as saved:
            labels, clusters = saved["label"], saved["cluster_id"]
            accepted = {name: saved[name] > identity["thresholds"][name]["0.1"]
                        for name in ("D_R", "D_F", "D_J")}
            errors = {name: decision != labels for name, decision in accepted.items()}
            union = int((errors["D_R"] | errors["D_F"]).sum())
            jaccard = float((errors["D_R"] & errors["D_F"]).sum() / union) if union else None
            keep = labels == 1
            if not keep.any():
                raise ProtocolError("D5 requires development fake videos")
            datasets.append((tag, {name: values[keep] for name, values in accepted.items()}, clusters[keep], jaccard))
    tpr = {name: float(np.mean([values[name].mean() for _, values, _, _ in datasets]))
           for name in ("D_R", "D_F", "D_J")}
    rng, draws = np.random.default_rng(seed), []
    for _ in range(repeats):
        terms = {name: [] for name in tpr}
        for _, values, clusters, _ in datasets:
            unique = np.unique(clusters)
            chosen = rng.choice(unique, len(unique), replace=True)
            indices = np.concatenate([np.flatnonzero(clusters == source) for source in chosen])
            for name in tpr:
                terms[name].append(float(values[name][indices].mean()))
        mean = {name: float(np.mean(value)) for name, value in terms.items()}
        draws.append(mean["D_J"] - max(mean["D_R"], mean["D_F"]))
    lower = float(np.quantile(draws, .05))
    jac = [value for _, _, _, value in datasets if value is not None]
    macro = float(np.mean(jac)) if len(jac) == len(datasets) else None
    return {"status": "COMPUTED", "alpha": .1, "macro_TPR": tpr,
            "D3_descriptive_pi_in": {tag: float(1 - values["D_J"].mean()) for tag, values, _, _ in datasets},
            "D3_band": "NOT_APPLICABLE_R3_DISABLED; no invented bandwidth or pass/fail",
            "error_jaccard_by_dataset": {tag: value for tag, _, _, value in datasets},
            "macro_error_jaccard": macro, "gain_vs_best_endpoint": tpr["D_J"] - max(tpr["D_R"], tpr["D_F"]),
            "gain_one_sided_lower_95": lower,
            "passes_table_rule": bool(macro is not None and macro < .8 and lower > 0),
            "passes_one_point_margin": bool(macro is not None and macro < .8 and lower > .01),
            "bootstrap_repeats": repeats, "seed": seed,
            "scope": "frozen D_J versus frozen endpoints; development only; dataset-stratified paired source clusters; no new fusion weight selection"}


def arcface_path(explicit=None):
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    root = Path(__file__).resolve().parents[2]
    candidates = [Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx",
                  root / "preprocessing/w600k_r50.onnx", root / "training/w600k_r50.onnx"]
    return next((path for path in candidates if path.is_file()), None)


class ArcFace:
    def __init__(self, path, mean=127.5, std=127.5):
        import cv2
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError("ArcFace normalization std must be positive")
        self.net, self.mean, self.std = cv2.dnn.readNetFromONNX(str(path)), mean, std

    def encode(self, root, paths, batch_size=32):
        import cv2
        vectors = []
        for start in range(0, len(paths), batch_size):
            images = [read_rgb(root, canonical_frame_path(path, root)) for path in paths[start:start + batch_size]]
            # read_rgb already yields RGB; official ArcFace BGR->RGB swap is unnecessary here.
            blob = cv2.dnn.blobFromImages(images, 1 / self.std, (112, 112), (self.mean,) * 3, swapRB=False)
            self.net.setInput(blob)
            features = self.net.forward().reshape(len(images), -1)
            if features.shape[1] != 512:
                raise ValueError("Expected an ArcFace recognition model with 512-dimensional output")
            vectors.append(unit(features))
        return np.concatenate(vectors)


def identity_audit(run_dir, audit, model_path=None, input_mean=127.5, input_std=127.5):
    from e0924_decision import auc
    from g21.io import file_digest

    model_path = arcface_path(model_path)
    if model_path is None:
        return {"status": "UNAVAILABLE_ARCFACE_WEIGHTS", "passed": None,
                "scope": "optional D6 per 01 section 1.4; identity confounding remains uncontrolled",
                "required_input": "512-D ArcFace ONNX recognition weights, e.g. w600k_r50.onnx; never CLIP as identity"}
    if audit.get("statistic_mode") != "video_mean":
        return {"status": "NOT_APPLICABLE_TO_LEGACY_MODE", "passed": None}
    extractor = ArcFace(model_path, input_mean, input_std)
    paths = sorted({row["path"] for rows in audit["real_frames"]["sel"].values() for row in rows}
                   | {row["fake_path"] for bag in audit["sel_videos"] for row in bag["frames"]})
    embeddings = dict(zip(paths, extractor.encode(audit["data_root"], paths)))
    q = min([len(rows) // 2 for rows in audit["real_frames"]["sel"].values() if len(rows) >= 2]
            + [len(bag["frames"]) // 2 for bag in audit["sel_videos"]])
    if q < 1:
        raise ProtocolError("D6 needs two nonempty genuine bags")
    real_means, genuine = {}, []
    for source, rows in audit["real_frames"]["sel"].items():
        rows = sorted(rows, key=lambda row: row["frame"])
        if len(rows) < 2:
            continue
        values = np.stack([embeddings[row["path"]] for row in rows])
        first, second = values[:q], values[-q:]
        real_means[source] = unit(first.mean(axis=0))
        genuine.append(float(unit(first.mean(axis=0)) @ unit(second.mean(axis=0))))
    if not genuine:
        raise ProtocolError("D6 has no genuine D_sel two-bag comparisons")
    cutoff = float(np.quantile(genuine, .05))
    methods = {}
    all_fake = []
    for method in audit["protocol"]["methods"]:
        similarities = []
        for bag in audit["sel_videos"]:
            if bag["method"] == method and bag["source"] in real_means:
                fake = unit(np.mean([embeddings[row["fake_path"]] for row in bag["frames"][:q]], axis=0))
                similarities.append(float(fake @ real_means[bag["source"]]))
        if not similarities:
            raise ProtocolError(f"D6 has no development identity pairs for {method}")
        all_fake.extend(similarities)
        identity_auc = auc([0] * len(genuine) + [1] * len(similarities),
                           [1 - value for value in [*genuine, *similarities]])
        methods[method] = {"pairs": len(similarities), "cosine_median": float(np.median(similarities)),
                           "cosine_q05": float(np.quantile(similarities, .05)), "identity_only_auc": identity_auc,
                           "alarm": identity_auc > .65}
    identity_auc = auc([0] * len(genuine) + [1] * len(all_fake), [1 - value for value in [*genuine, *all_fake]])
    result = {"status": "COMPUTED", "identity_only_auc": identity_auc,
              "alarm": bool(identity_auc > .65 or any(row["alarm"] for row in methods.values())),
              "methods": methods, "genuine_pairs": len(genuine), "matching_cosine_cutoff": cutoff,
              "frames_per_identity_mean": q,
              "cutoff_source": "D_sel real genuine comparisons q05; frozen before FF++ val matching",
              "recognition_weights_sha256": file_digest(model_path),
              "input": {"size": 112, "channels": "RGB", "mean": input_mean, "std": input_std,
                        "alignment": "existing benchmark crops, resized; no new face alignment"},
              "claim": "identity-cue diagnostic, not causal attribution or proof of no identity leakage"}
    run_dir = Path(run_dir)
    with np.load(run_dir / "dev_ff/features.npz", allow_pickle=False) as values:
        dev_paths, videos, labels = values["path"], values["video_id"], values["label"]
        z = extractor.encode(audit["data_root"], dev_paths.tolist())
    means = {video: unit(z[videos == video][:q].mean(axis=0)) for video in np.unique(videos)
             if int((videos == video).sum()) >= q}
    real = {video.rsplit("/", 1)[-1]: vector for video, vector in means.items()
            if np.all(labels[videos == video] == 0)}
    accepted, coverage = set(), {}
    for video, vector in means.items():
        if np.all(labels[videos == video] == 0):
            accepted.add(video)
            continue
        method, name = video.split("/")[-2:]
        coverage.setdefault(method, {"total": 0, "matched": 0, "missing_reference": 0})["total"] += 1
        reference = real.get(name.split("_")[0])
        if reference is None:
            coverage[method]["missing_reference"] += 1
        elif float(vector @ reference) >= cutoff:
            accepted.add(video)
            coverage[method]["matched"] += 1
    with np.load(run_dir / "dev_ff_scores/video_scores.npz", allow_pickle=False) as scores:
        keep = np.isin(scores["video_id"], sorted(accepted))
        truth = scores["label"][keep]
        matched_auc = {name: auc(truth, scores[name][keep]) for name in ("D_R", "D_F", "D_J")} if len(set(truth.tolist())) == 2 else None
        result["matched_development"] = {"scope": "FF++ official val only; cutoff frozen on D_sel genuine",
            "coverage_by_method": coverage, "real_videos": int((truth == 0).sum()), "fake_videos": int((truth == 1).sum()),
            "original_real_videos": int((scores["label"] == 0).sum()),
            "original_fake_videos": int((scores["label"] == 1).sum()),
            "video_auc": matched_auc}
        if matched_auc is not None:
            result["matched_development"]["auc_gain_vs_best_endpoint"] = matched_auc["D_J"] - max(matched_auc["D_R"], matched_auc["D_F"])
            frozen = json.loads((run_dir / "dev_ff_scores/metrics.json").read_text(encoding="utf-8"))["identity"]
            fake = keep & (scores["label"] == 1)
            tpr = {name: float((scores[name][fake] > frozen["thresholds"][name]["0.1"]).mean())
                   for name in ("D_R", "D_F", "D_J")}
            gain = tpr["D_J"] - max(tpr["D_R"], tpr["D_F"])
            result["matched_development"].update(TPR_at_alpha_0_1=tpr, gain_vs_best_endpoint=gain,
                                                 point_gain_nonpositive=bool(gain <= 0))
            if "cluster_id" in scores.files:
                clusters = scores["cluster_id"][fake]
                accepted = {name: scores[name][fake] > frozen["thresholds"][name]["0.1"]
                            for name in ("D_R", "D_F", "D_J")}
                unique = np.unique(clusters)
                rng, gains = np.random.default_rng(audit["protocol"].get("seed", 1024)), []
                for _ in range(audit["protocol"].get("bootstrap_repeats", 1000)):
                    draw = np.concatenate([np.flatnonzero(clusters == value) for value in rng.choice(unique, len(unique), replace=True)])
                    terms = {name: float(values[draw].mean()) for name, values in accepted.items()}
                    gains.append(terms["D_J"] - max(terms["D_R"], terms["D_F"]))
                result["matched_development"]["gain_one_sided_lower_95"] = float(np.quantile(gains, .05))
    return result

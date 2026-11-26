"""E1124 mechanism measurements and reporting on the original evaluation exports."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from detectors.effort_detector_e1124 import FULL_READ, ProjectedLoRA
from e1119_analysis import VideoMeans, add_ratios, energy
from e1119_eval import native_loader, video_rows, INDEPENDENT
from e1119_model import PROJECTIONS, REFERENCE, LoRA, blocks, hooks, paired_forward, projections, setting
from e1119_report import BOOTSTRAPS, PRIMARY, difference, paired_samples, table
from g21.io import atomic_json
from run_e1001 import export_scores, readout_metrics, save_npz
from run_e1119 import read
from run_g25 import TEST_DS, TRAIN_DS, VAL_DS


METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
CONTRASTS = {
    "C1_interaction": (("C1", "L1"), ("B00", "N1")),
    "X1_interaction": (("X1", "R1"), ("B00", "N1")),
}
PAIRS = (
    ("P128F", "B00"), ("P128S", "P128F"), ("V95", "P128S"), ("V95", "V95R"),
    ("W128", "P128S"), ("W128LR", "W128"), ("W128LR", "B00"),
    ("N1", "B00"), ("N1", "N0"), ("N1", "N2"), ("N3", "N1"), ("N3", "N4"),
    ("R1", "B00"), ("R1", "R2"), ("R2", "B00"), ("L1", "B00"), ("L1", "L2"),
    ("L1", "L3"), ("L1", "L4"), ("L1", "L5"), ("L5", "B00"), ("C1", "B00"),
    ("C1", "L1"), ("C1", "N1"), ("C1", "C2"), ("C1", "C3"), ("X1", "B00"),
    ("X1", "R1"), ("X1", "N1"),
)
FRAME_KEYS = ("labels", "path", "video_id")
VIDEO_KEYS = ("video_id", "label", "frame_count")


def npz(path):
    with np.load(path, allow_pickle=False) as handle:
        return dict(handle)


def matched_methods(values):
    probability = video_rows(values)
    log_odds = video_rows({**values, "cls_prob": values["global_log_odds"]})["prob_fake"]
    real, fake = {}, {}
    for path, label, prob, logit in zip(probability["video_id"], probability["label"], probability["prob_fake"], log_odds):
        parts = str(path).replace(chr(92), "/").split("/")
        clip = parts[-1]
        if label:
            method = parts[-4]
            if method not in METHODS:
                raise ValueError(f"Unexpected FF++ method path: {path}")
            fake.setdefault(clip, {})[method] = (prob, logit)
        else:
            real[clip] = (prob, logit)
    names = [clip for clip in sorted(fake) if set(fake[clip]) == set(METHODS) and clip.split("_")[0] in real]
    if len(names) != 140:
        raise ValueError(f"Expected 140 complete FF++ test pairs, found {len(names)}")
    positive = np.asarray([[fake[clip][method] for method in METHODS] for clip in names])
    negative = np.asarray([real[clip.split("_")[0]] for clip in names])
    result = {}
    for index, kind in enumerate(("probability", "mean_frame_log_odds")):
        scores = positive[:, :, index]
        genuine = negative[:, index]
        aucs = {method: float(roc_auc_score(np.repeat([0, 1], len(names)),
                                           np.concatenate([genuine, scores[:, i]])))
                for i, method in enumerate(METHODS)}
        gap = float(np.mean(scores.mean(1) - genuine))
        spread = float(np.ptp(scores, axis=1).mean())
        result[kind] = {"method_auc": aucs, "mean_method_auc": float(np.mean(list(aucs.values()))),
                        "worst_method_auc": min(aucs.values()), "fake_minus_real_mean": gap,
                        "within_source_method_range": spread, "range_over_gap": spread / gap if gap != 0 else None}
    return {"pairs": len(names), **result}


@torch.no_grad()
def model_diagnostics(model, destination):
    result = {}
    for name, module in projections(model.backbone):
        if not isinstance(module, ProjectedLoRA):
            continue
        update = module.update_basis
        route = module.read_basis
        left = module.left_basis
        if update is None and route is None:
            continue
        weight = module.update_weight().float()
        norm = weight.norm() + 1e-8
        result[name] = {
            "update_basis_rank": None if update is None else update.shape[1],
            "update_right_residual": None if update is None else float((weight @ update.float()).norm() / norm),
            "update_left_residual": None if left is None else float((left.float().T @ weight).norm() / norm),
            "update_read_overlap": None if update is None or route is None else
            float((update.float().T @ route.float()).square().sum() / min(update.shape[1], route.shape[1])),
        }
    if result:
        atomic_json(destination / "projection_diagnostics.json", result)


@torch.no_grad()
def fold_check(model, config, destination):
    model.eval()
    batch = next(iter(native_loader(config, VAL_DS)))
    inputs = {"image": batch["image"].to(next(model.parameters()).device)}
    normal = model(inputs, inference=True)
    with hooks() as handles:
        for _, module in projections(model.backbone):
            if isinstance(module, LoRA):
                folded = module.fold()
            elif hasattr(module, "lora_A"):
                a, b = module.lora_A, module.lora_B
                update = b @ a if hasattr(module, "fan_in_fan_out") else b.T @ a.T
                folded = module.weight + module.scaling * update
            else:
                continue

            def replace(mod, args, output, weight=folded):
                return torch.nn.functional.linear(args[0], weight, mod.bias)

            handles.append(module.register_forward_hook(replace))
        merged = model(inputs, inference=True)
    errors = {key: float((normal[key] - merged[key]).abs().max()) for key in ("cls", "prob")}
    atomic_json(destination / "fold_check.json", {"dataset": VAL_DS, "frames": len(batch["label"]),
                                                "max_absolute_error": errors})
    if not torch.allclose(normal["prob"], merged["prob"], rtol=1e-4, atol=1e-4):
        raise ValueError(f"Folded model probabilities differ on original validation frames: {errors}")


@torch.no_grad()
def route_diagnostics(model, config, destination, original):
    device = next(model.parameters()).device
    layers = blocks(model)
    model.eval()
    result = {}
    for dataset in TEST_DS:
        try:
            measures = VideoMeans()
            values = {}
            with hooks() as handles:
                for name in PROJECTIONS[:3]:
                    mod = getattr(layers[23].self_attn, name)

                    def response(module, args, name=name):
                        actual = args[0]
                        reference = REFERENCE.get()[module]
                        removed = module.removed_input(actual, reference).to(actual.dtype)
                        diff = energy(module.delta(removed)).cpu().numpy()
                        values[name + ":potential_removed_energy"] = diff
                        values[name + ":actual_removed_energy"] = np.zeros_like(diff) if model.arm == "L5" else diff
                        values[name + ":full_read_energy"] = energy(module.delta(actual)).cpu().numpy()

                    handles.append(mod.register_forward_pre_hook(response))
                for batch in native_loader(config, dataset):
                    inputs = {"image": batch["image"].to(device)}

                    def forward():
                        return model(inputs, inference=True)

                    if model.arm == "L5":
                        paired_forward(layers, (22,), forward)
                    else:
                        forward()
                    measures.add(batch, values)

            source_values = npz(original / dataset / "frames.npz")
            if model.arm == "L5":
                full_values = source_values
            else:
                with setting(FULL_READ, True):
                    full_values = export_scores(model, native_loader(config, dataset), auxiliary_enabled=False)
                if not all(np.array_equal(source_values[key], full_values[key]) for key in FRAME_KEYS):
                    raise ValueError("The full-read intervention changed native frame rows")
            folder = destination / dataset
            folder.mkdir(exist_ok=True)
            save_npz(folder / "full_read_videos.npz", video_rows(full_values))
            normal = read(original / dataset / "metrics.json")
            intervention = readout_metrics(full_values, full_values["cls_prob"])
            stats = measures.result()
            for prefix in ("potential", "actual"):
                add_ratios(stats, f":{prefix}_removed_energy", ":full_read_energy", f":{prefix}_ratio", 1e-8)
            result[dataset] = {"status": "OK", "normal_video_auc": normal["video_auc_fullpath"],
                               "full_read_video_auc": intervention["video_auc_fullpath"],
                               "delta_full_minus_normal_pp": difference(intervention["video_auc_fullpath"],
                                                                         normal["video_auc_fullpath"]),
                               "measurements": stats}
            if dataset == TRAIN_DS:
                result[dataset]["normal_matched"] = matched_methods(source_values)
                result[dataset]["full_read_matched"] = matched_methods(full_values)
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            result[dataset] = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    record = {"datasets": result}
    atomic_json(destination / "analysis.json", record)
    return record


def interval(jobs, seed, terms, point_estimates):
    by_dataset = []
    for i, name in enumerate(INDEPENDENT):
        combined = np.zeros(BOOTSTRAPS)
        for first, second in terms:
            a, b = (jobs[f"eval:{arm}:{seed}"] for arm in (first, second))
            if a["status"] != "OK" or b["status"] != "OK":
                raise ValueError(f"Missing complete evaluations for {first}/{second}")
            scores = [npz(job["folder"] / name / "videos.npz") for job in (a, b)]
            combined += paired_samples(*scores, np.random.default_rng(1024 + i))
        by_dataset.append(combined)
    lo, hi = np.percentile(100 * np.mean(by_dataset, axis=0), [2.5, 97.5])
    estimate = sum(point_estimates[a] - point_estimates[b] for a, b in terms)
    return {"status": "OK", "delta_pp": 100 * estimate, "low_pp": float(lo), "high_pp": float(hi)}


def identities(path, keys):
    from hashlib import sha256

    with np.load(path, allow_pickle=False) as handle:
        return {key: sha256(handle[key].tobytes()).hexdigest() for key in keys}


def summarize(folder, jobs, arms, seed, keys):
    rows, datasets, checks, matched, intervals = [], [], [], [], []
    estimates = {}
    anchor_train = jobs[f"train:B00:{seed}"].get("folder")
    baseline = read(anchor_train / "training_protocol.json") if anchor_train else {}
    anchor_eval = jobs[f"eval:B00:{seed}"].get("folder")
    reference_frames, reference_videos = {}, {}
    if jobs[f"eval:B00:{seed}"]["status"] == "OK":
        for name in TEST_DS:
            reference_frames[name] = identities(anchor_eval / name / "frames.npz", FRAME_KEYS)
            reference_videos[name] = identities(anchor_eval / name / "videos.npz", VIDEO_KEYS)

    for arm in arms:
        train = jobs[f"train:{arm}:{seed}"]
        evaluation = jobs[f"eval:{arm}:{seed}"]
        a, b = train.get("folder"), evaluation.get("folder")
        protocol = read(a / "training_protocol.json") if a and (a / "training_protocol.json").is_file() else {}
        result = read(b / "evaluation.json") if b and (b / "evaluation.json").is_file() else {}
        summary = read(a / "training.json") if a and (a / "training.json").is_file() else {}
        estimates[arm] = result.get(PRIMARY)
        rows.append({"arm": arm, "seed": seed, "train_status": train["status"],
                     "evaluation_status": evaluation["status"], "train_dir": str(a or ""),
                     "eval_dir": str(b or ""), "reused_from": evaluation.get("reused_from"),
                     "trainable_parameters": protocol.get("trainable_count"),
                     PRIMARY: estimates[arm], "delta_same_seed_B00_pp": difference(estimates[arm], estimates["B00"]),
                     "video_auc_seven_mean_fullpath": result.get("video_auc_seven_mean_fullpath"),
                     "missing_datasets": ",".join(result.get("missing_datasets", TEST_DS)),
                     **{key: summary.get(key) for key in ("epoch", "step", "frame_auc", "checkpoint_sha256",
                                                          "training_wall_seconds", "peak_allocated_bytes")}})
        validation = {key: key in protocol and key in baseline and protocol[key] == baseline[key] for key in keys}
        validation["expected_training_budget"] = (
            tuple(protocol.get(k) for k in keys[3:]) == (575, 6325, 10, 22)
            and protocol.get("trainable_count") == 788482)
        for name in TEST_DS:
            ds = result.get("datasets", {}).get(name, {})
            datasets.append({"arm": arm, "seed": seed, "dataset": name, **ds})
            validation[name] = (ds.get("status") == "OK" and name in reference_frames
                                and identities(b / name / "frames.npz", FRAME_KEYS) == reference_frames[name]
                                and identities(b / name / "videos.npz", VIDEO_KEYS) == reference_videos[name])
        checks.append({"arm": arm, "seed": seed, **validation})
        if result.get("datasets", {}).get(TRAIN_DS, {}).get("status") == "OK":
            matched.append({"arm": arm, "seed": seed, **matched_methods(npz(b / TRAIN_DS / "frames.npz"))})
    table(folder / "runs.csv", rows)
    table(folder / "main_seed1024.csv", rows)
    table(folder / "datasets.csv", datasets)
    table(folder / "protocol_alignment.csv", checks)
    table(folder / "analyses.csv", [{"task": key, "status": job["status"], "folder": str(job.get("folder", ""))}
                                    for key, job in jobs.items() if key.startswith(("asset:", "mechanism:"))])
    atomic_json(folder / "matched_ffpp.json", matched)

    comparisons = {f"{a}-{b}": ((a, b),) for a, b in PAIRS} | CONTRASTS
    for name, terms in comparisons.items():
        row = {"seed": seed, "contrast": name, "terms": str(terms), "resamples": BOOTSTRAPS}
        try:
            row.update(interval(jobs, seed, terms, estimates))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row.update(status="UNAVAILABLE", error=str(exc))
        intervals.append(row)
    table(folder / "paired_bootstrap.csv", intervals)
    aligned = all(all(v for k, v in row.items() if k not in ("arm", "seed")) for row in checks)
    complete = (len(datasets) == 184 and all(job["status"] == "OK" for job in jobs.values())
                and aligned and all(row["status"] == "OK" for row in intervals))
    atomic_json(folder / "completion.json", {"status": "COMPLETE" if complete else "INCOMPLETE",
                "protocol_aligned": aligned, "configured_evaluations": len(arms),
                "completed_dataset_rows": sum(row.get("status") == "OK" for row in datasets),
                "incomplete_tasks": [key for key, job in jobs.items() if job["status"] != "OK"]})
    return complete

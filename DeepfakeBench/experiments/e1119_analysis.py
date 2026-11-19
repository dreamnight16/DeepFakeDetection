"""Frozen-checkpoint probes, interventions, and mechanism measurements."""

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

from e1119_model import PROJECTIONS, blocks, bypass, hidden, hooks, native_delta, projections, run_block
from e1119_eval import native_loader, video_rows
from g21.io import atomic_json
from run_e1001 import export_scores, readout_metrics, save_npz
from run_g25 import TEST_DS, TRAIN_DS


PROBE_LAYERS = (4, 8, 12, 16, 20, 24)
DRIFT_LAYERS = (8, 16, 24)


def dump_cls(model, loader, layers):
    with hooks() as handles:
        batch_features = {}
        for depth in layers:
            def capture(_, args, output, depth=depth):
                batch_features[depth] = hidden(output)[:, 0].float().cpu().numpy()

            handles.append(model.backbone.encoder.layers[depth - 1].register_forward_hook(capture))
        device = next(model.parameters()).device
        for batch in loader:
            model.backbone(batch["image"].to(device))
            yield batch, batch_features


@torch.no_grad()
def d01(model, training, evaluation, folder):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def write_split(name, train=False):
        config = training if train else evaluation
        loader = native_loader(config, name, training=train)
        subfolder = folder / ("FFPP_train" if train else name)
        subfolder.mkdir(parents=True, exist_ok=True)
        labels, paths, videos = [], [], []
        arrays = {}
        offset = 0
        for batch, features in dump_cls(model, loader, PROBE_LAYERS):
            n = len(batch["label"])
            if not arrays:
                arrays = {depth: np.lib.format.open_memmap(subfolder / f"block_{depth}.npy",
                                                           mode="w+", dtype=np.float32,
                                                           shape=(len(loader.dataset), feat.shape[-1]))
                          for depth, feat in features.items()}
            for depth, feat in features.items():
                arrays[depth][offset:offset + n] = feat
            labels.extend(batch["label"].tolist())
            paths.extend(batch["path"])
            videos.extend(batch["video_id"])
            offset += n
        for array in arrays.values():
            array.flush()
        return subfolder, {"labels": np.asarray(labels), "path": np.asarray(paths),
                           "video_id": np.asarray(videos)}

    source, train_rows = write_split(TRAIN_DS, train=True)
    probes = {}
    for depth in PROBE_LAYERS:
        train_z = np.load(source / f"block_{depth}.npy", mmap_mode="r")
        probes[depth] = make_pipeline(StandardScaler(), LogisticRegression(
            C=1, max_iter=1000, solver="lbfgs", fit_intercept=True, penalty="l2"))
        probes[depth].fit(train_z, train_rows["labels"])
    scores = {}
    for name in TEST_DS:
        try:
            path, rows = write_split(name)
            metrics = {}
            for depth, probe in probes.items():
                probability = probe.predict_proba(np.load(path / f"block_{depth}.npy", mmap_mode="r"))[:, 1]
                values = {**rows, "cls_prob": probability}
                save_npz(path / f"block_{depth}_videos.npz", video_rows(values))
                metrics[str(depth)] = readout_metrics(values, probability)
            scores[name] = {"status": "OK", "layers": metrics}
        except Exception as exc:
            scores[name] = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    return {"datasets": scores}


@torch.no_grad()
def d02(model, evaluation, folder):
    result = {}
    for name in TEST_DS:
        dest = folder / name
        dest.mkdir(parents=True, exist_ok=True)
        try:
            baseline = export_scores(model, native_loader(evaluation, name), auxiliary_enabled=False)
            score = readout_metrics(baseline, baseline["cls_prob"])
            save_npz(dest / "original_frames.npz", baseline)
            save_npz(dest / "original_videos.npz", video_rows(baseline))
            arms = {"original": score}
            for start in (0, 8, 16):
                with bypass(blocks(model)[start:start + 8]):
                    values = export_scores(model, native_loader(evaluation, name), auxiliary_enabled=False)
                for key in ("path", "video_id", "labels"):
                    if not np.array_equal(values[key], baseline[key]):
                        raise ValueError(f"Intervention changed row alignment: {name}/{key}")
                label = f"off_{start + 1}_{start + 8}"
                metrics = readout_metrics(values, values["cls_prob"])
                arms[label] = {**metrics, **{f"delta_{key}_pp": 100 * (metrics[key] - score[key])
                                              for key in ("frame_auc", "video_auc_fullpath")}}
                save_npz(dest / f"{label}_frames.npz", values)
                save_npz(dest / f"{label}_videos.npz", video_rows(values))
            result[name] = {"status": "OK", "interventions": arms}
        except Exception as exc:
            result[name] = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    return {"datasets": result}


class VideoMeans:
    """Pool per-frame scalars to videos, then videos to class means."""

    def __init__(self):
        self.sums = defaultdict(dict)
        self.counts = defaultdict(dict)

    def add(self, batch, values):
        for key, vector in values.items():
            for video, label, value in zip(batch["video_id"], batch["label"].tolist(), vector):
                group = (int(label), video)
                sums, counts = self.sums[key], self.counts[key]
                sums[group] = sums.get(group, 0.0) + float(value)
                counts[group] = counts.get(group, 0) + 1

    def result(self):
        result = {}
        for key, sums in self.sums.items():
            result[key] = {}
            for label, name in ((0, "real"), (1, "fake")):
                values = [value / self.counts[key][group] for group, value in sums.items() if group[0] == label]
                result[key][name] = {"videos": len(values), "mean": float(np.mean(values)) if values else None}
        return result


@torch.no_grad()
def d03(model, reference, evaluation):
    result = {}
    for name in TEST_DS:
        try:
            loader = native_loader(evaluation, name)
            metrics = VideoMeans()
            # Both towers consume the exact same tensor in the same loader batch.
            with hooks() as handles:
                reference_features, fitted_features = {}, {}
                for tower, features in ((reference, reference_features), (model.backbone, fitted_features)):
                    for depth in DRIFT_LAYERS:
                        def capture(_, args, output, depth=depth, features=features):
                            features[depth] = hidden(output)[:, 0].float()

                        handles.append(tower.encoder.layers[depth - 1].register_forward_hook(capture))
                device = next(model.parameters()).device
                for batch in loader:
                    images = batch["image"].to(device)
                    reference(images)
                    model({"image": images}, inference=True)
                    distances = {str(depth): (1 - F.cosine_similarity(fitted_features[depth],
                                                                       reference_features[depth])).cpu().numpy()
                                 for depth in DRIFT_LAYERS}
                    metrics.add(batch, distances)
            result[name] = {"status": "OK", "layers": metrics.result()}
        except Exception as exc:
            result[name] = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    return {"datasets": result}


def energy(x):
    return x.float().square().sum(-1).mean(-1)


def add_ratios(metrics, numerator, denominator, suffix, epsilon):
    for name in [key for key in metrics if key.endswith(numerator)]:
        point = name.removesuffix(numerator)
        for label, entry in metrics[name].items():
            num = entry["mean"]
            den = metrics[point + denominator][label]["mean"]
            metrics.setdefault(point + suffix, {})[label] = (
                num / (den + epsilon) if num is not None and den is not None and den + epsilon > 0 else None)


@torch.no_grad()
def mechanism(model, arm, evaluation):
    device = next(model.parameters()).device
    is_protected = arm.startswith("P")
    is_segment = arm in ("T01", "T02")
    layers = blocks(model)
    fixed = {}
    if is_protected:
        for name, mod in projections(model.backbone):
            delta = mod.update_weight()
            fixed[name] = ((delta @ mod.basis).norm() / (delta.norm() + 1e-8)).item()

    output = {}
    for name in TEST_DS:
        try:
            metrics = VideoMeans()
            with hooks() as handles:
                inputs, outputs, quantities = {}, {}, {}
                if is_protected:
                    for point, mod in projections(model.backbone):
                        if not point.endswith(("q_proj", "out_proj")):
                            continue

                        def cover(module, args, point=point):
                            quantities[point + ":protected_energy"] = energy(args[0] @ module.basis)
                            quantities[point + ":input_energy"] = energy(args[0])

                        handles.append(mod.register_forward_pre_hook(cover))
                else:
                    indices = (15,) if is_segment else tuple(range(0, 24, 2))
                    for idx in indices:
                        def remember(_, args, output, idx=idx):
                            inputs[idx], outputs[idx] = args[0], hidden(output)

                        handles.append(layers[idx].register_forward_hook(remember))

                for batch in native_loader(evaluation, name):
                    model({"image": batch["image"].to(device)}, inference=True)
                    if is_segment:
                        quantities["segment_drift"] = model.segment_drift(outputs[15])
                    elif not is_protected:
                        for idx in indices:
                            bound, following = layers[idx], layers[idx + 1]
                            # Preserve the actual output captured above when the reference hook fires.
                            actual = following.layer_norm1(outputs[idx])
                            with bypass((bound,)):
                                reference = following.layer_norm1(run_block(bound, inputs[idx]))
                            for key in PROJECTIONS[:3]:
                                mod = getattr(following.self_attn, key)
                                point = f"{idx + 1}->{idx + 2}:{key}"
                                quantities[point + ":difference_energy"] = energy(native_delta(mod, actual - reference))
                                quantities[point + ":actual_energy"] = energy(native_delta(mod, actual))
                    matrix = torch.stack(tuple(quantities.values())).cpu().numpy()
                    metrics.add(batch, dict(zip(quantities, matrix)))
                    inputs.clear()
                    outputs.clear()
            measured = metrics.result()
            add_ratios(measured, ":protected_energy", ":input_energy", ":coverage", 0)
            add_ratios(measured, ":difference_energy", ":actual_energy", ":difference_ratio", 1e-8)
            output[name] = {"status": "OK", "measurements": measured}
        except Exception as exc:
            output[name] = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
    return {"direct_protected_residual": fixed, "datasets": output}


def write_analysis(folder, result):
    atomic_json(folder / "analysis.json", result)

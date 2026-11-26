"""Native score export and inference artifacts for E1119."""

from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

from e1119_model import LoRA, SPLIT, Spectral, projections
from e0924_export import strict_loader
from g21.io import atomic_json
from run_e1001 import export_scores, readout_metrics, save_npz
from run_e1021 import seed_everything
from run_g25 import CROSS_DS, TEST_DS, VAL_DS


INDEPENDENT = [name for name in CROSS_DS if name != VAL_DS]


def native_loader(config, dataset, training=False):
    import experiment_utils as utilities

    seed_everything(config["manualSeed"])
    source = (utilities.get_train_loader(config) if training
              else utilities.get_data_loader(config, dataset)).dataset
    return strict_loader(source, config["test_batchSize"])


def video_rows(values):
    groups = defaultdict(list)
    labels = {}
    for video, score, label in zip(values["video_id"], values["cls_prob"], values["labels"]):
        groups[video].append(float(score))
        if labels.setdefault(video, int(label)) != label:
            raise ValueError(f"Inconsistent labels for video {video}")
    return {"video_id": np.asarray(list(groups)),
            "label": np.asarray([labels[video] for video in groups]),
            "prob_fake": np.asarray([np.mean(scores) for scores in groups.values()]),
            "frame_count": np.asarray([len(scores) for scores in groups.values()])}


def means(datasets):
    def average(names, key="video_auc_fullpath"):
        values = [datasets.get(name, {}).get(key) for name in names]
        return float(np.mean(values)) if all(v is not None for v in values) else None

    return {"video_auc_seven_mean_fullpath": average(CROSS_DS),
            "independent_video_auc_mean_fullpath": average(INDEPENDENT),
            "AUC_cross_fullpath": average([VAL_DS, "DFDC"]),
            "AUC_cross_legacy": average([VAL_DS, "DFDC"], "video_auc_legacy")}


class TimedDetector:
    """Keep the historical exporter and time its actual model calls on device."""

    def __init__(self, model):
        self.model = model
        self.seconds = 0.0
        self.batch_seconds = None

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def __call__(self, inputs, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = self.model(inputs, **kwargs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.seconds += elapsed
        if self.batch_seconds is None:
            self.batch_seconds = elapsed
            self.batch_size = len(inputs["image"])
        return output


@torch.no_grad()
def export_folded(model, folder, arm, reference_blocks=None):
    blocks = SPLIT.get(arm, ()) if reference_blocks is None else reference_blocks
    if blocks:
        atomic_json(folder / "deployment.json", {"representation": "paired_detector_state",
                    "reference_blocks": [i + 1 for i in blocks],
                    "block_evaluations": 24 + len(blocks)})
        return
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()
             if key.startswith(("backbone.", "head."))}
    adaptive = {}
    for name, mod in projections(model.backbone):
        prefix = f"backbone.{name}."
        if isinstance(mod, (LoRA, Spectral)):
            weight = mod.fold()
            if isinstance(mod, Spectral) and hasattr(mod, "coeff"):
                values = mod.coeff if arm == "M05" else mod.coeff.exp()
                adaptive[name] = values.detach().cpu().tolist()
        elif hasattr(mod, "lora_A"):
            a, b = mod.lora_A, mod.lora_B
            update = b @ a if hasattr(mod, "fan_in_fan_out") else b.T @ a.T
            weight = mod.weight + mod.scaling * update
        else:
            continue
        for suffix in list(mod.state_dict()):
            if suffix not in ("weight", "bias"):
                state.pop(prefix + suffix)
        state[prefix + "weight"] = weight.detach().cpu()
    path = folder / "folded.pth"
    scratch = path.with_suffix(".tmp")
    torch.save(state, scratch)
    scratch.replace(path)
    metadata = {"representation": "native_clip_vision_and_linear_head", "block_evaluations": 24}
    if adaptive:
        metadata["angles" if arm == "M05" else "gains"] = adaptive
    if arm == "M05":
        metadata["zero_based_pairs"] = [[[i, i + 1] for i in range(0, 1024, 2)],
                                         [[i, (i + 1) % 1024] for i in range(1, 1024, 2)]]
    atomic_json(folder / "deployment.json", metadata)


def evaluate(config, model, folder, arm, *, reference_blocks=None):
    datasets = {}
    if arm in ("M01", "M02"):
        spectral = {}
        with torch.no_grad():
            for name, mod in projections(model.backbone):
                if arm == "M02":
                    orth, norm = mod.regularizers()
                else:
                    asset = torch.load(Path(config["e1119_assets"]) / f"{name}.pt", weights_only=True)
                    u = asset["u"][:, :-1].to(mod.u.device)
                    v = asset["vh"][:-1].to(mod.v.device)
                    orth = (2 * ((u.T @ mod.u).square().sum() + (v @ mod.v).square().sum())
                            + (mod.u.square().sum() - 1).square() + (mod.v.square().sum() - 1).square())
                    norm = (mod.fold().double().square().sum()
                            - asset["s"].double().square().sum().to(mod.u.device)).abs()
                spectral[name] = {"orth": orth.item(), "norm": norm.item()}
        atomic_json(folder / "matrix_diagnostics.json", {"projections": spectral,
                    **{key + "_mean": float(np.mean([v[key] for v in spectral.values()]))
                       for key in ("orth", "norm")}})
    export_folded(model, folder, arm, reference_blocks)
    for name in TEST_DS:
        dest = folder / name
        dest.mkdir(parents=True, exist_ok=True)
        try:
            loader = native_loader(config, name)
            torch.cuda.reset_peak_memory_stats()
            timer = TimedDetector(model)
            started = time.perf_counter()
            values = export_scores(timer, loader, auxiliary_enabled=False)
            wall = time.perf_counter() - started
            videos = video_rows(values)
            save_npz(dest / "frames.npz", values)
            save_npz(dest / "videos.npz", videos)
            metrics = readout_metrics(values, values["cls_prob"])
            metrics.update(status="OK", forward_seconds=timer.seconds,
                           wall_seconds=wall, wall_seconds_per_video=wall / len(videos["video_id"]),
                           forward_seconds_per_video=timer.seconds / len(videos["video_id"]),
                           first_batch_seconds=timer.batch_seconds, first_batch_size=timer.batch_size,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated())
        except Exception as exc:
            metrics = {"status": "UNAVAILABLE", "error": f"{type(exc).__name__}: {exc}"}
        atomic_json(dest / "metrics.json", metrics)
        datasets[name] = metrics
    missing = [name for name in TEST_DS if datasets[name]["status"] != "OK"]
    result = {"status": "COMPLETE_WITH_UNAVAILABLE" if missing else "OK",
              "datasets": datasets, "missing_datasets": missing, **means(datasets)}
    atomic_json(folder / "evaluation.json", result)
    return result

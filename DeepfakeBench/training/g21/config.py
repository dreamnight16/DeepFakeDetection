"""G21 configuration is independent of the historical YAML merge machinery."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .io import digest

ARMS = {
    "A": ("none", "mean", 0.0),
    "B": ("matched", "mean", 1.0),
    "C": ("shuffled", "mean", 1.0),
    "D": ("matched", "group_softmax", 1.0),
}
TEST_DATASETS = [
    "WDF",
    "FFIW",
    "Celeb-DF-v2",
    "DeepFakeDetection",
    "DFDC",
    "DFDCP",
    "DeeperForensics-1.0",
    "FaceForensics++",
]
DEFAULT_CONFIG = {
    "schema_version": 1,
    "experiment": "G21",
    "protocol": "g18_g19_legacy_testall_v1",
    "seed": 1024,
    "arm": None,
    "paths": {
        "data_root": None,
        "clip_pretrained_path": None,
        "train_manifest": None,
        "dataset_json_folder": None,
        "output_root": "./experiment_results/g21",
    },
    "model": {
        "use_loralib": True,
        "full_train_head": True,
        "use_freq_split": False,
        "margin_loss_mode": "off",
    },
    "data": {
        "train_methods": ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"],
        "pairs_per_microbatch": 4,
        "frames_per_training_video": 2,
        "training_frame_pool": 8,
        "evaluation_frames_per_video": 8,
        "resolution": 224,
        "views": ["reference", "jpeg"],
        "horizontal_flip_probability": 0.5,
        "jpeg_quality_min": 70,
        "jpeg_quality_max": 100,
    },
    "loss": {
        "pairing": "matched",
        "pair_mode": "mean",
        "pair_lambda": 1.0,
        "pair_margin": 0.0,
        "robust_temperature": 0.5,
    },
    "optimizer": {
        "lr": 0.0002,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0005,
    },
    "training": {
        "device": "cuda:0",
        "precision": "fp32",
        "max_updates": 6000,
        "val_every_updates": 200,
        "save_last_every_updates": 200,
        "evaluation_batch_size": 32,
        "selection_metric": "auc",
        "gradient_mode": "two_pass_replay",
        "deterministic": True,
    },
    "evaluation": {
        "backend": "legacy_testall",
        "validation_dataset": "Celeb-DF-v2",
        "test_datasets": TEST_DATASETS,
    },
}


def _merge(base: dict, values: dict, prefix: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in values.items():
        if key not in base:
            raise ValueError(f"unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"expected mapping: {prefix}{key}")
            out[key] = _merge(base[key], value, prefix + key + ".")
        else:
            out[key] = copy.deepcopy(value)
    return out


def validate_config(cfg: dict, require_paths: bool = False) -> None:
    if cfg["schema_version"] != 1 or cfg["experiment"] != "G21":
        raise ValueError("unsupported G21 schema/experiment")
    if cfg["protocol"] != "g18_g19_legacy_testall_v1":
        raise ValueError("G21 must retain the G18/G19 testing protocol")
    data, tr, loss = cfg["data"], cfg["training"], cfg["loss"]
    methods = data["train_methods"]
    if len(methods) != 4 or set(methods) != {"FF-DF", "FF-F2F", "FF-FS", "FF-NT"}:
        raise ValueError("training must include all four FF++ methods, including FF-NT")
    if cfg["evaluation"] != DEFAULT_CONFIG["evaluation"]:
        raise ValueError(
            "test datasets, CDF validation and legacy backend must match G18/G19"
        )
    for name in (
        "pairs_per_microbatch",
        "frames_per_training_video",
        "training_frame_pool",
        "evaluation_frames_per_video",
        "resolution",
    ):
        if type(data[name]) is not int or data[name] < 1:
            raise ValueError(f"invalid data.{name}")
    if data["pairs_per_microbatch"] < 2 or data["frames_per_training_video"] < 2:
        raise ValueError("at least two distinct sources and time points are required")
    if data["training_frame_pool"] < data["frames_per_training_video"]:
        raise ValueError("training frame pool is smaller than T")
    if data["views"] != ["reference", "jpeg"] or data["resolution"] != 224:
        raise ValueError("G21 uses 224 RGB and reference/jpeg views")
    if not (0 <= data["horizontal_flip_probability"] <= 1):
        raise ValueError("invalid flip probability")
    if not (1 <= data["jpeg_quality_min"] <= data["jpeg_quality_max"] <= 100):
        raise ValueError("invalid JPEG range")
    for name in (
        "max_updates",
        "val_every_updates",
        "save_last_every_updates",
        "evaluation_batch_size",
    ):
        if type(tr[name]) is not int or tr[name] < 1:
            raise ValueError(f"invalid training.{name}")
    if tr["selection_metric"] != "auc" or tr["gradient_mode"] != "two_pass_replay":
        raise ValueError(
            "all G21 arms require historical auc selection and a matched two-pass window"
        )
    if data["evaluation_frames_per_video"] != 8 or tr["evaluation_batch_size"] != 32:
        raise ValueError("historical evaluation uses 8 frames and batch size 32")
    if tr["precision"] != "fp32" or type(tr["deterministic"]) is not bool:
        raise ValueError("only explicit FP32 deterministic settings are supported")
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    if cfg["model"] != DEFAULT_CONFIG["model"]:
        raise ValueError("G21 must retain the declared CLIP-LoRA pooler model")
    if loss["pairing"] not in ("none", "matched", "shuffled") or loss[
        "pair_mode"
    ] not in ("mean", "group_softmax"):
        raise ValueError("invalid pairing/reduction")
    for name in ("pair_lambda", "pair_margin", "robust_temperature"):
        if not math.isfinite(loss[name]) or loss[name] < 0:
            raise ValueError(f"invalid loss.{name}")
    if loss["robust_temperature"] <= 0:
        raise ValueError("temperature must be positive")
    opt = cfg["optimizer"]
    if any(
        not math.isfinite(opt[k]) or opt[k] < 0 for k in ("lr", "eps", "weight_decay")
    ):
        raise ValueError("invalid optimizer settings")
    if (
        opt["lr"] == 0
        or opt["eps"] == 0
        or len(opt["betas"]) != 2
        or not all(0 <= x < 1 for x in opt["betas"])
    ):
        raise ValueError("invalid Adam parameters")
    if require_paths:
        for key in (
            "data_root",
            "clip_pretrained_path",
            "train_manifest",
            "dataset_json_folder",
            "output_root",
        ):
            if not isinstance(cfg["paths"][key], str) or not cfg["paths"][key]:
                raise ValueError(f"missing paths.{key}")


def resolve_arm(cfg: dict, arm: str, seed: int) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    out = copy.deepcopy(cfg)
    pairing, mode, weight = ARMS[arm]
    out.update(arm=arm, seed=seed)
    out["loss"].update(pairing=pairing, pair_mode=mode, pair_lambda=weight)
    validate_config(out)
    return out


def config_digest(cfg: dict) -> str:
    return digest(cfg)


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        values = json.loads(text)
    else:
        import yaml

        values = yaml.safe_load(text)
    if not isinstance(values, dict):
        raise ValueError("configuration must be a mapping")
    cfg = _merge(DEFAULT_CONFIG, values)
    for key, value in cfg["paths"].items():
        if isinstance(value, str):
            cfg["paths"][key] = str((path.parent / value).resolve())
    validate_config(cfg)
    return cfg

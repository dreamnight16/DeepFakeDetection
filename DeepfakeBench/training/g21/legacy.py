"""G18/G19 evaluation compatibility. Scores use the existing loaders and metrics."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import subprocess
import sys

from .config import TEST_DATASETS
from .io import atomic_json

BENCH = Path(__file__).resolve().parents[2]


def legacy_config(cfg: dict, *, validation: bool = False) -> dict:
    import yaml

    with (BENCH / "training/config/detector/effort.yaml").open(
        encoding="utf-8"
    ) as stream:
        result = yaml.safe_load(stream)
    filename = "train_config.yaml" if validation else "test_config.yaml"
    with (BENCH / "training/config" / filename).open(encoding="utf-8") as stream:
        result.update(yaml.safe_load(stream))
    expected = {
        "resolution": 224,
        "test_batchSize": 32,
        "multi_crop": False,
        "use_adaptive_threshold": False,
        "full_train_head": True,
        "use_loralib": True,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"historical evaluation config drift: {key}={result.get(key)!r}"
            )
    if (
        result.get("frame_num", {}).get("test") != 8
        or result.get("compression") != "c23"
    ):
        raise ValueError("historical frame count/compression changed")
    result.update(
        model_name="effort",
        use_mixup=False,
        use_freq_split=False,
        margin_loss_mode="off",
        clip_pretrained_path=cfg["paths"]["clip_pretrained_path"],
        rgb_root_override=cfg["paths"]["data_root"],
        dataset_json_folder=cfg["paths"]["dataset_json_folder"],
        manualSeed=cfg["seed"],
        test_dataset=["Celeb-DF-v2"],
        metric_scoring="auc",
    )
    return result


def validation_loader(cfg: dict):
    from torch.utils.data import DataLoader
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset

    config = legacy_config(cfg, validation=True)
    config["test_dataset"] = "Celeb-DF-v2"
    dataset = DeepfakeAbstractBaseDataset(config=config, mode="test")
    return DataLoader(
        dataset,
        batch_size=config["test_batchSize"],
        shuffle=False,
        num_workers=int(config["workers"]),
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )


def validate_model(model, loader, device) -> dict:
    import numpy as np
    import torch
    from metrics.utils import get_test_metrics

    predictions, labels = [], []
    previous = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                output = model({"image": images}, inference=True)
                predictions.extend(output["prob"].detach().cpu().tolist())
                labels.extend(torch.where(batch["label"] != 0, 1, 0).tolist())
    finally:
        model.train(previous)
    if len(predictions) != len(loader.dataset.data_dict["image"]):
        raise ValueError("historical validation output length mismatch")
    values = get_test_metrics(
        np.asarray(predictions), np.asarray(labels), loader.dataset.data_dict["image"]
    )
    return {k: float(v) for k, v in values.items() if k not in ("pred", "label")}


def parse_testall_log(text: str) -> dict:
    results, current = {}, None
    for line in text.splitlines():
        line = line.strip()
        if (
            "[WARNING] test.py exited with code" in line
            or "[WARNING] No metrics parsed" in line
        ):
            raise ValueError(f"legacy testall reported a failed dataset: {line}")
        if line.startswith("dataset:"):
            name = line.split(":", 1)[1].strip()
            current = (
                name
                if name in TEST_DATASETS
                else "average"
                if name.startswith("average (")
                else None
            )
            if current is not None:
                results[current] = {}
        elif current is not None:
            match = re.fullmatch(r"(acc|auc|video_auc):\s*([-+0-9.eE]+)", line)
            if match:
                value = float(match[2])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("invalid legacy metric")
                results[current][match[1]] = value
    for name in [*TEST_DATASETS, "average"]:
        if set(results.get(name, {})) != {"acc", "auc", "video_auc"}:
            raise ValueError(f"incomplete legacy testall metrics: {name}")
    return results


def run_testall(cfg: dict, weights_path: str | Path, output: str | Path) -> dict:
    import yaml

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "test_config.resolved.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(legacy_config(cfg), stream, sort_keys=False)
    command = [
        sys.executable,
        str(BENCH / "testall.py"),
        "--detector_path",
        str(config_path),
        "--weights_path",
        str(Path(weights_path).resolve()),
        "--test_datasets",
        *TEST_DATASETS,
        "--artifact_dir",
        str(output / "artifacts"),
        "--dataset_json_folder",
        cfg["paths"]["dataset_json_folder"],
        "--data_root",
        cfg["paths"]["data_root"],
    ]
    environment = dict(os.environ)
    requested = cfg["training"]["device"]
    if requested.startswith("cuda"):
        index = int(requested.split(":")[1]) if ":" in requested else 0
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        environment["CUDA_VISIBLE_DEVICES"] = (
            visible.split(",")[index] if visible is not None else str(index)
        )
    else:
        raise ValueError(
            "historical test.py requires CUDA; use the server GPU environment"
        )
    log = output / "testall.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=BENCH,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"legacy testall failed ({result.returncode}); inspect {log}"
        )
    metrics = parse_testall_log(log.read_text(encoding="utf-8", errors="replace"))
    atomic_json(output / "testall_metrics.json", metrics)
    return metrics

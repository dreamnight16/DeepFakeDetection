"""Opt-in records on the native training path."""

import json
from pathlib import Path
import random

import numpy as np
import torch

from g21.io import atomic_json, digest


def train_started(config, model, loader, optimizer):
    trainable = {name: list(value.shape) for name, value in model.named_parameters() if value.requires_grad}
    names = {id(value): name for name, value in model.named_parameters()}
    groups = [{**{k: v for k, v in group.items() if k != "params"},
               "params": [names[id(value)] for value in group["params"]]}
              for group in optimizer.param_groups]
    folder = Path(config["e1119_record_dir"])
    atomic_json(folder / "resolved_train_config.json", config)
    source = json.loads((folder / "baseline_train.json").read_text(encoding="utf-8"))
    atomic_json(folder / "resolved_train_diff.json", {
        key: {"source": source.get(key), "actual": value}
        for key, value in config.items() if key not in source or source[key] != value
    })
    algorithm, state, position, has_gauss, cached_gauss = np.random.get_state()
    loader_rng = [torch.get_rng_state().tolist(), random.getstate(),
                  [algorithm, state.tolist(), position, has_gauss, cached_gauss]]
    atomic_json(folder / "training_protocol.json", {
        "trainable": trainable,
        "trainable_count": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "optimizer_groups": groups,
        "head_initial_sha256": digest({k: v.detach().tolist() for k, v in model.head.state_dict().items()}),
        "loader_rng_sha256": digest(loader_rng),
        "train_list_order_sha256": digest(list(zip(loader.dataset.image_list, loader.dataset.label_list))),
        "steps_per_epoch": len(loader),
        "total_steps": len(loader) * (config["nEpochs"] - config["start_epoch"] + 1),
        "real_per_batch": loader.batch_sampler.n_real,
        "fake_per_batch": loader.batch_sampler.n_fake,
        "device": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
    })


def epoch_finished(config, model, epoch, seconds, steps):
    folder = Path(config["e1119_record_dir"])
    if epoch in (0, 5, 10):
        path = folder / f"epoch_{epoch}.pth"
        scratch = path.with_suffix(".tmp")
        torch.save(model.state_dict(), scratch)
        scratch.replace(path)
    atomic_json(folder / f"epoch_{epoch}.json", {"epoch": epoch, "steps": steps,
                "wall_seconds": seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})


def best_selected(config, log_dir, dataset, epoch, iteration, step, score):
    atomic_json(Path(config["e1119_record_dir"]) / "best.json", {
        "checkpoint": str(Path(log_dir, "test", dataset, "ckpt_best.pth").resolve()),
        "selection_dataset": config["test_dataset"][0], "frame_auc": float(score),
        "epoch": epoch, "iteration": iteration, "step": step,
    })

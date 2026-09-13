"""Isolated single-GPU full-window G21 training, resume and evaluation."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
import uuid

import torch

from .config import ARMS, config_digest, validate_config
from .data import make_window, render_plan
from .evaluation import evaluate_model, summarize_datasets
from .io import append_jsonl, atomic_json, atomic_text, file_digest, source_digest
from .preflight import inspect_inputs
from .randomness import capture_rng, restore_rng, seed_all
from .replay import window_backward


def code_digest() -> str:
    return source_digest()


def build_model(cfg: dict):
    # Lazy import: preflight/manifest tools never import the detector registry.
    from detectors.effort_detector import EffortDetector

    options = dict(cfg["model"])
    options["clip_pretrained_path"] = cfg["paths"]["clip_pretrained_path"]
    return EffortDetector(options)


def state_digest(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def output_lock(output: Path):
    lock = output / ".running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "host": platform.node()}, stream)
    try:
        yield
    finally:
        lock.unlink()


def _truncate_log(path: Path, completed: int) -> None:
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        rows = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1 and not raw.endswith("\n"):
                    break  # A killed writer may leave an incomplete final append.
                raise
        atomic_text(
            path,
            "".join(
                json.dumps(r, ensure_ascii=False) + "\n"
                for r in rows
                if r["update"] <= completed
            ),
        )


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for package in (
        "torch",
        "numpy",
        "transformers",
        "loralib",
        "opencv-python",
        "opencv-python-headless",
    ):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            out[package] = None
    return out


def train_arm(
    cfg: dict,
    output: str | Path,
    *,
    resume: bool = False,
    model_factory=None,
    stop_after: int | None = None,
) -> dict:
    """Train one arm; injected tiny models/stop_after are for server-side tests only."""
    validate_config(cfg, require_paths=True)
    arm = cfg["arm"]
    if arm not in ARMS:
        raise ValueError("resolved arm is required")
    actual = (
        cfg["loss"]["pairing"],
        cfg["loss"]["pair_mode"],
        cfg["loss"]["pair_lambda"],
    )
    if actual != ARMS[arm]:
        raise ValueError("loss configuration does not match the declared arm")
    output = Path(output).resolve()
    if not resume and output.exists():
        raise FileExistsError(f"new arm output already exists: {output}")
    if resume and not (output / "checkpoints/last.pth").is_file():
        raise ValueError("resume requires this arm's last.pth")
    output.mkdir(parents=True, exist_ok=resume)
    with output_lock(output):
        try:
            return _train_locked(cfg, output, resume, model_factory, stop_after)
        except BaseException as exc:
            atomic_json(
                output / "result.json",
                {
                    "exp_name": f"G21/{arm}",
                    "seed": cfg["seed"],
                    "status": "INTERRUPTED"
                    if isinstance(exc, KeyboardInterrupt)
                    else "TRAIN_FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise


def _train_locked(cfg, output, resume, model_factory, stop_after):
    pairs, val, tests, inputs = inspect_inputs(cfg, check_model=model_factory is None)
    configuration_hash, implementation_hash = config_digest(cfg), code_digest()
    if not resume:
        atomic_json(output / "config.resolved.json", cfg)
        atomic_json(output / "preflight.json", inputs)
    device = torch.device(cfg["training"]["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("requested CUDA device is unavailable")
        torch.cuda.set_device(device)
    seed_all(cfg["seed"], cfg["training"]["deterministic"])
    model = (model_factory or build_model)(cfg).to(device)
    initial_digest = state_digest(model)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.Adam(
        [p for _, p in trainable],
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
        weight_decay=opt_cfg["weight_decay"],
    )
    if model_factory is None:
        from .legacy import validation_loader

        val_loader = validation_loader(cfg)
    completed, best, best_update, best_video = 0, None, None, None
    counters = {"forward_images": 0, "backward_images": 0}
    if resume:
        checkpoint = torch.load(
            output / "checkpoints/last.pth", map_location="cpu", weights_only=True
        )
        for key, expected in [
            ("config_digest", configuration_hash),
            ("input_digest", inputs["input_digest"]),
            ("code_digest", implementation_hash),
            ("initial_state_digest", initial_digest),
        ]:
            if checkpoint[key] != expected:
                raise ValueError(f"RESUME_MISMATCH: {key}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed, best, best_update = (
            checkpoint["completed_updates"],
            checkpoint["best_selection_auc"],
            checkpoint["best_update"],
        )
        best_video = checkpoint["best_video_auc"]
        counters = checkpoint["counters"]
        restore_rng(checkpoint["rng_state"])
        for name in ("train.jsonl", "validation.jsonl", "data_plan_digests.jsonl"):
            _truncate_log(output / name, completed)
        if best_update is not None:
            best_path = output / "checkpoints/best.pth"
            if not best_path.exists():
                raise ValueError("resume best checkpoint is missing")
            saved_best = torch.load(best_path, map_location="cpu", weights_only=True)
            if saved_best["completed_updates"] != best_update:
                # A newer best may have been saved after the last periodic last.
                # Keep a best snapshot inside last to recover the old selection.
                if "best_state_dict" not in checkpoint:
                    raise ValueError("resume best/last selection mismatch")
                saved_best = dict(
                    checkpoint,
                    model_state_dict=checkpoint["best_state_dict"],
                    completed_updates=best_update,
                )
                save_checkpoint(best_path, saved_best)
        best_state = checkpoint.get("best_state_dict")
    else:
        best_state = None
        atomic_json(
            output / "provenance.json",
            {
                "versions": _versions(),
                "code_digest": implementation_hash,
                "input_digest": inputs["input_digest"],
                "initial_state_digest": initial_digest,
                "trainable_parameters": {n: p.numel() for n, p in trainable},
                "model_factory": "production"
                if model_factory is None
                else "injected_test_model",
            },
        )
    tr, loss = cfg["training"], cfg["loss"]
    limit = (
        min(tr["max_updates"], stop_after)
        if stop_after is not None
        else tr["max_updates"]
    )

    def pack(include_best=False):
        payload = {
            "schema_version": 1,
            "config": cfg,
            "config_digest": configuration_hash,
            "input_digest": inputs["input_digest"],
            "code_digest": implementation_hash,
            "initial_state_digest": initial_digest,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": capture_rng(),
            "completed_updates": completed,
            "best_selection_auc": best,
            "best_video_auc": best_video,
            "best_update": best_update,
            "counters": dict(counters),
        }
        if include_best:
            payload["best_state_dict"] = best_state
        return payload

    for index in range(completed, limit):
        started = time.monotonic()
        plans = make_window(pairs, cfg, index)
        windows = [render_plan(p, cfg["paths"]["data_root"]) for p in plans]
        permutations = [torch.tensor(p["permutation"]) for p in plans]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logs = window_backward(
            model,
            windows,
            arm=cfg["arm"],
            permutations=permutations,
            margin=loss["pair_margin"],
            temperature=loss["robust_temperature"],
            device=device,
        )
        gradients = [p.grad for _, p in trainable if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError("NONFINITE or missing trainable gradients")
        logs["gradient_norm"] = (
            torch.stack([g.detach().float().norm() for g in gradients]).norm().item()
        )
        if not math.isfinite(logs["gradient_norm"]):
            raise ValueError("NONFINITE gradient norm")
        optimizer.step()
        completed = index + 1
        for key in counters:
            counters[key] += logs[key]
        logs.update(
            update=completed,
            methods=[p["method"] for p in plans],
            jpeg_quality=[p["quality"] for p in plans],
            seconds=time.monotonic() - started,
        )
        append_jsonl(output / "train.jsonl", logs)
        append_jsonl(
            output / "data_plan_digests.jsonl",
            {"update": completed, "digests": [p["digest"] for p in plans]},
        )
        if completed % tr["val_every_updates"] == 0 or completed == tr["max_updates"]:
            rng = capture_rng()
            if model_factory is None:
                from .legacy import validate_model

                metrics = validate_model(model, val_loader, device)
            else:
                metrics = evaluate_model(
                    model,
                    val,
                    root=cfg["paths"]["data_root"],
                    frame_count=cfg["data"]["evaluation_frames_per_video"],
                    batch_size=tr["evaluation_batch_size"],
                    device=device,
                )
            restore_rng(rng)
            append_jsonl(output / "validation.jsonl", dict(metrics, update=completed))
            if best is None or metrics["auc"] > best:
                best, best_update, best_video = (
                    metrics["auc"],
                    completed,
                    metrics["video_auc"],
                )
                best_state = {
                    n: p.detach().cpu().clone() for n, p in model.state_dict().items()
                }
                save_checkpoint(output / "checkpoints/best.pth", pack())
        if completed % tr["save_last_every_updates"] == 0 or completed == limit:
            save_checkpoint(output / "checkpoints/last.pth", pack(include_best=True))
        print(
            f"G21/{cfg['arm']} update={completed}/{tr['max_updates']} loss={logs['loss']:.6f} best_CDF_auc={best} selected_video_auc={best_video}",
            flush=True,
        )
    if completed < tr["max_updates"]:
        result = {
            "status": "INTERRUPTED",
            "exp_name": f"G21/{cfg['arm']}",
            "completed_updates": completed,
        }
    else:
        best_path = output / "checkpoints/best.pth"
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        checkpoint_hash = file_digest(best_path)
        test_results = {}
        training_best_path = best_path
        if model_factory is None:
            from .legacy import run_testall

            best_path = output / "checkpoints/best_testall.pth"
            save_checkpoint(best_path, checkpoint["model_state_dict"])
            checkpoint_hash = file_digest(best_path)
            del model, optimizer
            trainable.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            test_results = run_testall(
                cfg, best_path, output / "evaluation" / uuid.uuid4().hex[:12]
            )
        else:
            for name, videos in tests.items():
                test_results[name] = evaluate_model(
                    model,
                    videos,
                    root=cfg["paths"]["data_root"],
                    frame_count=cfg["data"]["evaluation_frames_per_video"],
                    batch_size=tr["evaluation_batch_size"],
                    device=device,
                    output=output / "evaluation" / name,
                    provenance={
                        "checkpoint_sha256": checkpoint_hash,
                        "protocol": cfg["protocol"],
                    },
                )
        result = {
            "status": "OK" if test_results else "TRAINED",
            "exp_name": f"G21/{cfg['arm']}",
            "seed": cfg["seed"],
            "protocol": cfg["protocol"],
            "completed_updates": completed,
            "best_update": best_update,
            "validation_auc": best,
            "validation_video_auc": best_video,
            "ckpt": str(best_path),
            "training_ckpt": str(training_best_path),
            "checkpoint_sha256": checkpoint_hash,
            "initial_state_digest": initial_digest,
            "input_digest": inputs["input_digest"],
            "code_digest": implementation_hash,
            "config_digest": configuration_hash,
            "selection_metric": "auc",
            "report_metric": "video_auc",
            "test_backend": "legacy_testall",
            "testall": test_results,
            "counters": counters,
            **summarize_datasets(test_results),
        }
    atomic_json(output / "result.json", result)
    return result


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    output: str | Path,
    *,
    config: dict | None = None,
    data_root: str | None = None,
    device: str | None = None,
) -> dict:
    if Path(output).exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = config or checkpoint.get("config")
    if cfg is None:
        raise ValueError("plain historical weights require --config")
    validate_config(cfg, require_paths=True)
    if data_root is not None:
        cfg["paths"]["data_root"] = data_root
    if device is not None:
        cfg["training"]["device"] = device
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    weights = checkpoint.get(
        "model_state_dict", checkpoint.get("state_dict", checkpoint)
    )
    raw_path = output / "weights.pth"
    save_checkpoint(raw_path, weights)
    from .legacy import run_testall

    return run_testall(cfg, raw_path, output / "testall")

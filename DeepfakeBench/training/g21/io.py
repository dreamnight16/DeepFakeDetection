"""Strict, deterministic metadata and atomic output helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import tempfile
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_digest() -> str:
    folder = Path(__file__).resolve().parent
    bench = folder.parents[1]
    files = list(folder.glob("*.py")) + list((bench / "experiments").glob("*g21*.py"))
    files += [
        bench / name
        for name in (
            "testall.py",
            "training/test.py",
            "training/detectors/effort_detector.py",
            "training/dataset/abstract_dataset.py",
            "training/metrics/utils.py",
        )
    ]
    return digest({str(p.relative_to(bench)): file_digest(p) for p in sorted(files)})


def atomic_json(path: str | Path, value: Any) -> None:
    atomic_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def atomic_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(item)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
    )


def append_jsonl(path: str | Path, row: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

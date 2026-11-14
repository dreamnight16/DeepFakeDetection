"""Server-only shared-affine preparation and complete-runner contracts."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from e1111 import shared_crop
from e1111.protocol import protocol


@pytest.mark.parametrize("different_geometry", [False, True])
def test_one_real_affine_is_reused_for_all_fake_crops(tmp_path, monkeypatch, different_geometry):
    import cv2
    from e1111.audit import frame_map

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    predictor = tmp_path / "predictor.dat"
    predictor.write_bytes(b"fixture predictor")
    methods = protocol()["methods"]
    real_paths = {v: [f"FaceForensics++/original_sequences/youtube/c23/frames/{v}/{i:03d}.png"
                      for i in (0, 2)] for v in ("001", "002")}
    data = {"FF-real": {"train": {"c23": {v: {"frames": paths} for v, paths in real_paths.items()}}}}
    frames_by_video = {}
    raw = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    for source, paths in real_paths.items():
        frames_by_video[shared_crop.raw_video_path(paths[0])] = {i: raw.copy() for i in (0, 2)}
    rules = {}
    for method in methods:
        paths = [f"FaceForensics++/manipulated_sequences/{method}/c23/frames/001_002/{i:03d}.png"
                 for i in (0, 2)]
        data[method] = {"train": {"c23": {"001_002": {"frames": paths}}}}
        image = np.flip(raw, axis=1).copy()
        if different_geometry and method == methods[0]:
            image = image[:-1]
        frames_by_video[shared_crop.raw_video_path(paths[0])] = {i: image for i in (0, 2)}
        rules[method] = {"verified": True, "reference_component": 0, "evidence": "fixture",
                         "same_original_index_verified": True}
    for relative in frames_by_video:
        path = raw_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture raw video")
    matrix = np.array([[.8, 0., 7.], [0., .8, 10.]])
    estimated_on_real = []
    monkeypatch.setattr(shared_crop, "landmark_estimator", lambda path: lambda frame: np.zeros((5, 2)))

    def estimate(points):
        estimated_on_real.append(points)
        return matrix.copy()

    monkeypatch.setattr(shared_crop, "reference_affine", estimate)
    monkeypatch.setattr(shared_crop, "video_frames", lambda path, indices: (
        (i, frames_by_video[str(path.relative_to(raw_root)).replace("\\", "/")][i]) for i in sorted(indices)))
    dataset_json, pair_map = tmp_path / "dataset.json", tmp_path / "map.json"
    dataset_json.write_text(json.dumps({"FaceForensics++": data}))
    pair_map.write_text(json.dumps({"schema_version": 1, "rules": rules}))
    out = tmp_path / "prepared"
    result = shared_crop.prepare(dataset_json, pair_map, raw_root, predictor, out)
    assert len(estimated_on_real) == 4  # two real videos x two frames; never fake detection
    receipt = shared_crop.load_receipt(out / "shared_crop_receipt.json", out / "images", result["receipt_sha256"])
    assert len(receipt["pair_warps"]) == (6 if different_geometry else 8)
    for fake_path, record in receipt["pair_warps"].items():
        real_path = record["real_path"]
        assert shared_crop.check_pair(receipt, real_path, fake_path, "001", record["real_frame"], record["fake_frame"])
        actual_real = cv2.imdecode(np.frombuffer((out / "images" / real_path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        actual_fake = cv2.imdecode(np.frombuffer((out / "images" / fake_path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(actual_real, shared_crop.warp(raw, matrix))
        expected_fake = frames_by_video[record["raw_video"]][record["fake_frame"]]
        np.testing.assert_array_equal(actual_fake, shared_crop.warp(expected_fake, matrix))
    if different_geometry:
        assert any(row["reason"] == "raw_real_fake_geometry_mismatch" for row in receipt["failures"])
    first_fake = next(iter(receipt["pair_warps"]))
    receipt["pair_warps"][first_fake]["matrix_digest"] = "different affine"
    record = receipt["pair_warps"][first_fake]
    assert not shared_crop.check_pair(receipt, record["real_path"], first_fake, "001",
                                      record["real_frame"], record["fake_frame"])
    assert frame_map(data["FF-real"]["train"]["c23"]["001"])


def load_runner(monkeypatch):
    path = ROOT / "experiments/e1111_runner.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("e1111_full_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standard_inputs_require_no_original_videos_or_predictor(tmp_path, monkeypatch):
    runner = load_runner(monkeypatch)
    root, metadata, clip = (tmp_path / name for name in ("images", "metadata", "clip"))
    for path in (root, metadata, clip):
        path.mkdir()
    for name in ("FaceForensics++", "Celeb-DF-v2"):
        (metadata / f"{name}.json").write_text("{}")
    source = {"config": {"label_dict": {}, "dataset_json_folder": str(metadata),
                         "rgb_root_override": str(root), "clip_pretrained_path": str(clip)}}
    monkeypatch.setattr(runner, "resolve_source", lambda args: source)
    args = argparse.Namespace(seed=1024, dataset_json_folder=None, rgb_root=None,
                              clip_pretrained_path=None)
    result = runner.resolve_inputs(args)
    assert result["data_root"] == root.resolve()
    assert "raw_root" not in result and "predictor" not in result
    assert not list(root.rglob("*.mp4"))


@pytest.mark.parametrize("gate_pass", [False, True])
def test_full_runner_stops_before_model_work_when_gate_fails(tmp_path, monkeypatch, gate_pass):
    from e1111 import audit, features, pipeline

    runner = load_runner(monkeypatch)
    calls = []
    source = {"checkpoint": "B0.pth", "config_path": "B0.json", "base_sha256": "b0",
              "config_sha256": "cfg"}
    inputs = {"source": source, "config": {}, "metadata": tmp_path, "data_root": tmp_path, "clip": tmp_path}
    monkeypatch.setattr(runner, "resolve_inputs", lambda args: inputs)

    def record(name, result):
        def run(*args, **kwargs):
            calls.append(name)
            return result
        return run

    monkeypatch.setattr(audit, "inventory", record("inventory", {"dataset_json": str(tmp_path / "index.json")}))
    monkeypatch.setattr(audit, "index_pairing", record("pairing", {}))
    def forbidden_crop(*args, **kwargs):
        raise AssertionError("Default A must not generate new crops or require original videos")
    monkeypatch.setattr(shared_crop, "prepare", forbidden_crop)
    def audit_same_inputs(dataset, data_root, *args, **kwargs):
        assert Path(dataset) == tmp_path / "index.json"
        assert data_root == inputs["data_root"]
        assert kwargs["statistic_mode"] == "video_mean"
        return record("audit", {"P_D7_pass": gate_pass,
                      "status": "P_D7_PASS" if gate_pass else "S0_PAUSED_P_D7"})()
    monkeypatch.setattr(audit, "audit", audit_same_inputs)
    monkeypatch.setattr(features, "extract", record("features", {}))
    monkeypatch.setattr(pipeline, "run_s0", record("s0", {"status": "S0_PASS"}))
    monkeypatch.setattr(pipeline, "calibrate", record("calibrate", {}))
    monkeypatch.setattr(features, "export_dataset", record("export", {}))
    monkeypatch.setattr(pipeline, "score_export", record("score", {}))
    monkeypatch.setattr(pipeline, "attribution", record("attribution", {"passed": True}))
    args = argparse.Namespace(output_dir=tmp_path / "run", seed=1024, preflight=False,
                              pair_map=None, device="cpu", batch_size=8, skip_d8=False)
    result = runner.run(args)
    if not gate_pass:
        assert calls == ["inventory", "pairing", "audit"]
        assert result["status"] == "S0_PAUSED_P_D7"
    else:
        assert calls == ["inventory", "pairing", "audit", "features", "s0", "calibrate",
                         "export", "score", "export", "score", "attribution"]
        assert result["status"] == "DEVELOPMENT_COMPLETE"


def test_a_accepts_successful_native_e1111_b0_without_using_the_old_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace

    runner = load_runner(monkeypatch)
    baseline = {"status": "OK", "settings": {"model_name": "effort"},
                "ckpt": str(tmp_path / "new_B0.pth"), "checkpoint_sha256": "fresh_sha"}
    report = {"experiment": "E1111", "branch": "E1111B", "arms": {"B0": baseline}}
    (tmp_path / "all_results.json").write_text(json.dumps(report))
    seen = []

    def resolve(args):
        seen.append(args)
        return {"base_sha256": "fresh_sha", "config": {"model_name": "effort"}}

    monkeypatch.setitem(sys.modules, "run_e1001", SimpleNamespace(resolve_source=resolve))
    args = argparse.Namespace(base_run=tmp_path, base_checkpoint=None, base_config=None)
    source = runner.resolve_source(args)
    assert seen[0].base_run is None
    assert seen[0].base_checkpoint == tmp_path / "new_B0.pth"
    assert seen[0].base_config == tmp_path / "B0/train_config.json"
    assert source["origin"].startswith("E1111B/B0")
    baseline["settings"]["model_name"] = "effort_e1111b"
    (tmp_path / "all_results.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="original-CE"):
        runner.resolve_source(args)


def test_legacy_e0924_embedded_paths_follow_the_same_relocation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import e1111_paths

    runner = load_runner(monkeypatch)
    moved = tmp_path / "moved"
    monkeypatch.setattr(e1111_paths, "NEW_ROOT", str(moved))
    run = moved / "effort/old_run"
    run.mkdir(parents=True)
    (run / "training_results.json").write_text(json.dumps({"G26_B0": {
        "status": "OK", "ckpt": "/home/user1/effort/weights/B0.pth",
        "artifact_dir": "/home/user1/effort/artifact"}}))
    seen = []
    monkeypatch.setitem(sys.modules, "run_e1001", SimpleNamespace(
        resolve_source=lambda args: seen.append(args) or {"config": {"model_name": "effort"}}))
    args = argparse.Namespace(base_run=Path("/home/user1/effort/old_run"),
                              base_checkpoint=None, base_config=None)
    runner.resolve_source(args)
    assert seen[0].base_run is None
    assert seen[0].base_checkpoint == moved / "effort/weights/B0.pth"
    assert seen[0].base_config == moved / "effort/artifact/train_config.json"

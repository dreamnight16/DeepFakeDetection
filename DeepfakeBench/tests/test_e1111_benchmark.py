"""Server-only contracts for native-loader, same-row A/B0 benchmark metrics."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "experiments"))

from e1111 import benchmark
from g21.data import MEAN, STD
from run_e1001 import readout_metrics
from run_g25 import TEST_DS


def rows():
    return {"labels": np.array([0, 0, 1, 1]),
            "path": np.array([f"/mnt/data2/effort/data/{v}/{t}.png" for v in ("real/a", "fake/b") for t in (0, 1)]),
            "video_id": np.array(["/mnt/data2/effort/data/real/a"] * 2 + ["/mnt/data2/effort/data/fake/b"] * 2),
            "cls_prob": np.array([.1, .2, .8, .9]), "global_log_odds": np.array([-2., -1., 1., 2.])}


def test_same_auc_function_and_raw_video_mean_without_sigmoid():
    values, energy = rows(), np.array([0., 10., 4., 4.])
    actual = benchmark.readout(values, energy, 6.)
    expected = readout_metrics(values, energy)
    for name in ("frame_auc", "video_auc", "video_auc_fullpath", "video_auc_legacy", "num_frames", "num_videos"):
        assert actual[name] == expected[name]
    # Sigmoid before video averaging would reverse this ranking.
    assert actual["video_auc"] == 0. and actual["accuracy"] == .25
    assert "corrected_frames" not in actual and "harmed_frames" not in actual


@pytest.mark.parametrize("changed", [None, "labels", "path", "video_id", "cls_prob", "global_log_odds"])
def test_archived_native_identity_and_scores_are_checked(tmp_path, changed):
    values = rows()
    saved = {key: value.copy() for key, value in values.items()}
    for key in ("path", "video_id"):
        saved[key] = np.array([p.replace("/mnt/data2", "/home/user1") for p in saved[key]])
    if changed is not None:
        saved[changed][0] = "different" if changed in ("path", "video_id") else 1 - saved[changed][0]
    path = tmp_path / "reference.npz"
    np.savez_compressed(path, **saved)
    if changed:
        with pytest.raises(ValueError, match="differs"):
            benchmark.reference_alignment(values, path)
    else:
        assert benchmark.reference_alignment(values, path)["status"] == "MATCHED"


def test_eight_native_datasets_same_tensors_and_primary_metrics(tmp_path, monkeypatch):
    import torch
    import run_e1021

    metadata = tmp_path / "json"
    metadata.mkdir()
    for dataset in TEST_DS:
        (metadata / f"{dataset}.json").write_text("{}")
    base_path, eval_path, metric_path = (tmp_path / name for name in ("base.json", "eval.json", "metric.npz"))
    base_path.write_text(json.dumps({"dataset_json_folder": str(metadata), "rgb_root_override": str(tmp_path)}))
    config = {"model_name": "effort", "frame_num": {"train": 8, "test": 8}, "resolution": 224,
              "mean": list(MEAN), "std": list(STD), "test_batchSize": 2, "manualSeed": 1024}
    eval_path.write_text(json.dumps(config))
    metric_path.write_bytes(b"frozen calibrated metric fixture")
    feature_identity = {name: name for name in ("checkpoint_sha256", "base_config_sha256", "feature", "preprocess",
                                               "feature_implementation_sha256", "versions")}
    identity = {"status": "S0_PASS", "gamma": .25, "feature_identity": feature_identity,
                "thresholds": {name: {"0.1": 6.} for name in ("D_R", "D_F", "D_J", "D_P")}}
    monkeypatch.setattr(benchmark, "load_metric", lambda path: ({}, identity))
    calls, tensor_rows = [], []
    def data_loader(cfg, dataset):
        assert cfg["frame_num"]["test"] == 8 and not cfg["use_data_augmentation"]
        assert cfg["test_batchSize"] == 2
        calls.append(dataset)
        entry = rows()
        image_values = {path: i for i, path in enumerate(entry["path"])}
        original = SimpleNamespace(image_list=entry["path"].tolist(), label_list=entry["labels"].tolist(),
                                   load_rgb=lambda path: np.full((2, 2, 3), image_values[path], dtype=np.uint8),
                                   to_tensor=lambda rgb: torch.full((3, 224, 224), float(rgb[0, 0, 0])),
                                   normalize=lambda tensor: tensor)
        return SimpleNamespace(dataset=original)
    monkeypatch.setitem(sys.modules, "experiment_utils", SimpleNamespace(get_data_loader=data_loader))
    seeds = []
    monkeypatch.setattr(run_e1021, "seed_everything", lambda seed: seeds.append(seed))
    def backbone(images):
        numbers = images[:, 0, 0, 0]
        tensor_rows.extend(numbers.tolist())
        tokens = numbers[:, None, None].expand(-1, 257, 1024).clone()
        tokens[:, 0] = 1e6
        return {"last_hidden_state": tokens, "pooler_output": numbers[:, None]}
    class Extractor:
        def __init__(self, *args):
            self.identity = feature_identity
            self.model = SimpleNamespace(backbone=backbone, _prep_input=lambda images: images,
                                         classifier=lambda pool: torch.cat((-pool, pool), dim=1))
    monkeypatch.setattr(benchmark, "FrozenFeatures", Extractor)
    def score(features, model, gamma):
        assert gamma == .25
        np.testing.assert_array_equal(features, np.broadcast_to(features[:, :1], features.shape))
        indices = features[:, 0].astype(int)
        return {name: np.array([0., 10., 4., 4.])[indices] for name in ("D_R", "D_F", "D_J", "D_P")}
    monkeypatch.setattr(benchmark, "score", score)
    references = tmp_path / "references"
    references.mkdir()
    reference = rows()
    number = torch.arange(4, dtype=torch.float32)
    reference["cls_prob"] = torch.stack((-number, number), dim=1).softmax(dim=1)[:, 1].numpy()
    reference["global_log_odds"] = (2 * number).numpy()
    for dataset in TEST_DS:
        np.savez_compressed(references / f"{dataset}.npz", **reference)
    result = benchmark.evaluate(metric_path, base_path, "B0.pth", "clip", eval_path,
                                tmp_path / "result", device="cpu", reference_exports=references)
    assert result["status"] == "OK" and calls == TEST_DS and seeds == [1024] * len(TEST_DS)
    assert tensor_rows == [0., 1., 2., 3.] * len(TEST_DS)
    assert set(result["datasets"]) == set(TEST_DS)
    assert result["video_auc_seven_mean_fullpath"] == result["independent_video_auc_mean_fullpath"] == 0.
    assert "Celeb-DF-v2" not in result["independent_datasets"] and len(result["independent_datasets"]) == 6
    assert all(value["status"] == "MATCHED" for value in result["reference_alignment"].values())
    with np.load(tmp_path / "result/exports/WDF.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["path"], reference["path"])
        np.testing.assert_array_equal(saved["labels"], reference["labels"])
        np.testing.assert_array_equal(saved["D_J"], [0., 10., 4., 4.])


def test_latest_benchmark_reuses_only_an_existing_calibrated_run(tmp_path, monkeypatch):
    for name in ("seed1024_20261114_170000", "seed1024_20261114_180000"):
        run = tmp_path / name
        (run / "calibrated").mkdir(parents=True)
        (run / "calibrated/calibrated_metric.npz").write_bytes(b"fixture")
        (run / "resolved_base_config.json").write_text(json.dumps({"clip_pretrained_path": "clip"}))
        (run / "all_results.json").write_text(json.dumps({"base": {
            "config_path": str(tmp_path / "native/B0/train_config.json"), "checkpoint": "B0.pth"}}))
    calls = []
    def evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "OK"}
    monkeypatch.setattr(benchmark, "evaluate", evaluate)
    assert benchmark.evaluate_run(tmp_path, latest=True)["status"] == "OK"
    assert calls[0][0][0] == tmp_path / "seed1024_20261114_180000/calibrated/calibrated_metric.npz"

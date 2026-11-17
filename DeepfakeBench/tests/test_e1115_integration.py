"""Complete synthetic-data flow and real training-hook contracts; no GPU/data training."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "experiments"))

from e1115 import suite
from e1115.assets import matched_candidates
from e1115.math import prepare_model, score_model
from e1115.sampling import FixedStepBatchSampler, refresh_epoch
from e1115.statistics import METHODS


def fixture_data():
    rng = np.random.default_rng(31)
    audit = {"complete_sources": [str(i) for i in range(8)], "real_frames": {"fit": {}}, "fit_videos": []}
    lookup, reals = {}, []
    for source in audit["complete_sources"]:
        x = rng.normal(size=(8, 12)).astype(np.float32)
        reals.append(x)
        audit["real_frames"]["fit"][source] = []
        for i, vector in enumerate(x):
            p = f"r/{source}/{i}"
            lookup[p] = vector
            audit["real_frames"]["fit"][source].append({"frame": i, "path": p})
        for method in METHODS:
            frames = []
            for i in range(8):
                p = f"{method}/{source}/{i}"
                lookup[p] = (x[i] + rng.normal(size=12) + .3).astype(np.float32)
                frames.append({"frame": i, "fake_path": p})
            audit["fit_videos"].append({"source": source, "method": method, "frames": frames})
    cal_z = rng.normal(size=(60, 12)).astype(np.float32)
    cal = {"labels": np.zeros(60, dtype=int), "path": np.array([f"cal/{i}" for i in range(60)]),
           "video_id": np.repeat([f"cal/{i}" for i in range(30)], 2), "cls_prob": rng.uniform(.05, .4, 60)}
    return audit, (lookup, reals, np.eye(12) * .1, cal, cal_z)


def test_all_models_all_datasets_and_target_controls(tmp_path, monkeypatch):
    from run_g25 import TEST_DS
    audit, source = fixture_data()
    native = tmp_path / "native"
    native.mkdir()
    (native / "manifest.json").write_text('{}')
    rng = np.random.default_rng(4)
    rows = {"labels": np.repeat([0] * 60 + [1] * 20, 2),
            "path": np.array([f"p/{i}" for i in range(160)]),
            "video_id": np.repeat([f"v/{i}" for i in range(80)], 2),
            "cls_prob": np.linspace(.02, .98, 160)}
    z = rng.normal(size=(160, 12)).astype(np.float32)
    original_p = rows["cls_prob"].copy()
    monkeypatch.setattr(suite, "load_native", lambda *_: (rows, z))
    target_fit_calls = []
    original_adapt = suite.adapt_model
    def checked_adapt(model, real_groups, *args):
        assert all(any(np.array_equal(g, z[i:i + 2]) for i in range(0, 120, 2)) for g in real_groups)
        target_fit_calls.append(len(real_groups))
        return original_adapt(model, real_groups, *args)
    monkeypatch.setattr(suite, "adapt_model", checked_adapt)
    anchor_cal = tmp_path / "anchor_cal.npz"
    np.savez_compressed(anchor_cal, **source[3])
    report = suite.run_suite(audit, source, native, tmp_path / "out", groups=["geometry", "adaptation", "baseline"],
                             budgets=[5, 10, 100], calibration_count=20, repeats=1,
                             anchor={"native": native, "calibration": anchor_cal})
    assert report["status"] == "COMPLETE_WITH_UNAVAILABLE"
    assert len(report["adaptation_unavailable"]) == len(TEST_DS)
    assert {5, 10} == set(target_fit_calls)
    for name, item in report["arms"].items():
        assert item["status"] == "SCORED", (name, item.get("reason"))
        assert set(item["datasets"]) == set(TEST_DS)
    assert report["arms"]["B0"]["datasets"] == report["arms"]["B0_PLUS_FULL_0"]["datasets"]
    assert report["arms"]["ANCHOR_B0"]["datasets"] == report["arms"]["ANCHOR_PLUS_FULL_0"]["datasets"]
    np.testing.assert_array_equal(rows["cls_prob"], original_p)
    split_dir = tmp_path / "out/adaptation_splits"
    for dataset in TEST_DS:
        a = json.loads((split_dir / f"{dataset}_N5_R0.json").read_text())
        b = json.loads((split_dir / f"{dataset}_N10_R0.json").read_text())
        assert a["test"] == b["test"] and a["calibration"] == b["calibration"]
        assert set(a["fit"]) <= set(b["fit"])
    assert (tmp_path / "out/summary.csv").is_file()


def test_original_index_intersection_precedes_independent_sampling():
    data = {m: {"train": {"c23": {}}} for m in ["FF-real", *METHODS]}
    data["FF-real"]["train"]["c23"]["a"] = {"frames": [f"real/a/{i}.png" for i in range(6)]}
    audit = {"complete_sources": ["a"], "fit_videos": []}
    for method in METHODS:
        data[method]["train"]["c23"]["a_b"] = {"frames": [f"{method}/a_b/{i}.png" for i in range(3, 9)]}
        # Previously independently selected rows deliberately exclude shared frames.
        audit["fit_videos"].append({"source": "a", "method": method,
                                    "frames": [{"frame": 8, "fake_path": f"{method}/a_b/8.png"}]})
    aligned = matched_candidates(data, audit, ".", readable=lambda *_: True)
    assert [x["frame"] for x in aligned["a"]["real"]] == [3, 4, 5]
    assert all([x["frame"] for x in rows] == [3, 4, 5] for rows in aligned["a"]["fake"].values())


def test_normalized_real_and_direction_energy_have_comparable_scale():
    rng = np.random.default_rng(9)
    model = prepare_model({"mu": np.zeros(16), "covariance": np.eye(16), "directions": np.eye(16)[:, :2],
                           "weights": np.ones(2), "real_weight": 0., "direction_weight": 1.})
    x = rng.normal(size=(20000, 16))
    assert score_model(x, model).mean() == pytest.approx(1., abs=.03)
    real_model = prepare_model({**model, "real_weight": 1., "direction_weight": 0.})
    assert score_model(x, real_model).mean() == pytest.approx(1., abs=.03)


def test_actual_detector_loss_hook_preserves_default_and_eval_gradients():
    import torch
    from torch.nn import functional as F
    # Execute the actual method without importing unrelated detector dependencies.
    tree = ast.parse((ROOT / "training/detectors/effort_detector.py").read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "EffortDetector")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "get_losses")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_get_losses", "exec"), namespace)
    owner = SimpleNamespace(config={}, training=True, loss_func=torch.nn.CrossEntropyLoss(), margin_loss_mode="off")
    labels = torch.tensor([0, 1, 1, 1])
    logits = torch.tensor([[2., 0.], [1., 0.], [0., 2.], [1., 3.]], requires_grad=True)
    def get():
        return namespace["get_losses"](owner, {"label": labels}, {"cls": logits})["overall"]
    torch.testing.assert_close(get(), F.cross_entropy(logits, labels))
    owner.config["e1115_class_balanced_loss"] = True
    expected = .5 * (F.cross_entropy(logits[:1], labels[:1]) + F.cross_entropy(logits[1:], labels[1:]))
    torch.testing.assert_close(torch.autograd.grad(get(), logits)[0], torch.autograd.grad(expected, logits)[0])
    owner.training = False
    torch.testing.assert_close(get(), F.cross_entropy(logits, labels))


def test_epoch_refresh_keeps_label_counts_and_rng_state():
    import random
    def collect(_):
        return [f"e{ds.config['e1115_epoch']}/{i}" for i in range(40)], [0] * 20 + [1] * 20, []
    ds = SimpleNamespace(config={"e1115_frame_policy": "epoch_stratified", "manualSeed": 1,
                                 "train_dataset": ["FF++"]}, label_list=[0] * 20 + [1] * 20,
                         collect_img_and_label_for_one_dataset=collect)
    sampler = FixedStepBatchSampler(ds.label_list, 32, .5, 4)
    state = random.getstate()
    refresh_epoch(ds, sampler, 3)
    assert random.getstate() == state and ds.image_list[0] == "e3/0" and sampler.epoch == 3


def test_actual_loader_hook_uses_fixed_budget_and_nonpersistent_workers():
    import torch
    tree = ast.parse((ROOT / "training/train.py").read_text(encoding="utf-8"))
    method = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "prepare_training_data")
    class Dataset:
        def __init__(self, config, mode):
            self.label_list = [0] * 40 + [1] * 80
        def __len__(self):
            return len(self.label_list)
        def __getitem__(self, index):
            return self.label_list[index]
        @staticmethod
        def collate_fn(values):
            return values
    namespace = {"torch": torch, "DeepfakeAbstractBaseDataset": Dataset}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_prepare_training_data", "exec"), namespace)
    loader = namespace["prepare_training_data"]({"e1115_steps_per_epoch": 7, "train_batchSize": 32,
             "sampler_real_ratio": .7, "manualSeed": 1024, "workers": 0})
    batches = list(loader)
    assert len(batches) == 7 and all(x.count(0) == 22 for x in batches)
    assert loader.persistent_workers is False


def test_registered_training_configs_keep_native_evaluation(tmp_path):
    from run_e1115 import build_parser, build_training_config
    from e1115.protocol import TRAIN_ARMS
    args = build_parser().parse_args([])
    utility = SimpleNamespace(build_config=lambda **kw: {"frame_num": {"train": 8, "test": 8},
                            "train_batchSize": 32, "test_batchSize": 32, **kw})
    for arm, spec in TRAIN_ARMS.items():
        train, test = build_training_config(args, arm, tmp_path / arm, utility, 575)
        assert train["sampler_real_ratio"] == spec["ratio"]
        assert train.get("e1115_steps_per_epoch") == (575 if spec["fixed_budget"] else None)
        assert test["frame_num"]["test"] == 8 and test["model_name"] == "effort"
        assert "e1115_frame_policy" not in test and "e1115_class_balanced_loss" not in test
        assert not train["use_mixup"] and train["full_train_head"]


def test_legacy_closed_form_matches_existing_metric(monkeypatch):
    from e1111.metric import fit_space, inverse_sqrt, real_statistics as old_stats, score
    audit, source = fixture_data()
    lookup, reals, dc, _, _ = source
    delta = []
    bags = {(b["source"], b["method"]): b for b in audit["fit_videos"]}
    for s in audit["complete_sources"]:
        real = np.mean([lookup[x["path"]].astype(float) for x in audit["real_frames"]["fit"][s]], 0)
        delta.append([np.mean([lookup[x["fake_path"]].astype(float) for x in bags[s, m]["frames"]], 0) - real for m in METHODS])
    mu, cov, _ = old_stats(reals, .1)
    space = fit_space(np.asarray(delta), cov, dc, 0.)
    original = {"mu": mu, "white": inverse_sqrt(cov), "transform": space.transform,
                "vectors": space.vectors[:, :2], "weights": space.weights[:2]}
    models, _, _ = suite.fit_models(audit, lookup, reals, dc)
    x = np.random.default_rng(4).normal(size=(20, 12))
    np.testing.assert_allclose(score_model(x, models["VM_LEGACY"]), score(x, original, 2.)["D_J"], rtol=1e-9, atol=1e-9)


def test_frozen_train_config_adds_test_labels_without_mutating_original():
    from run_e1115 import build_parser, frozen_configs
    args = build_parser().parse_args([])
    original = {"label_dict": {"FF-real": 0, "FF-DF": 1}, "frame_num": {"train": 8, "test": 8}, "manualSeed": 1024}
    train, test = frozen_configs(args, original)
    assert test["label_dict"]["WDF_Real"] == 0 and test["label_dict"]["WDF_Fake"] != 0
    assert "FFIW_Real" in test["label_dict"] and "WDF_Real" not in original["label_dict"]
    with pytest.raises(ValueError, match="Conflicting"):
        frozen_configs(args, {**original, "label_dict": {"WDF_Real": 1}})


def test_combined_tables_support_sampling_only_and_preserve_scope(tmp_path):
    import csv
    from run_e1115 import write_combined_tables
    from run_g25 import TEST_DS
    folder = tmp_path / "sampling"
    folder.mkdir()
    result = {"status": "OK", "arms": {"B0": {"status": "SCORED", "scope": "source_only",
              "datasets": {d: {"video_auc_fullpath": .9, "frame_auc": .8, "num_frames": 16, "num_videos": 2} for d in TEST_DS}}}}
    suite._means(result)
    (folder / "all_results.json").write_text(json.dumps(result))
    write_combined_tables({"posthoc": {"REAL70": {"output": str(folder), "status": "OK"}}}, tmp_path)
    with (tmp_path / "summary.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["training_arm"] == "REAL70" and rows[0]["scope"] == "source_only"
    assert float(rows[0]["independent_video_auc_mean_fullpath"]) == pytest.approx(.9)

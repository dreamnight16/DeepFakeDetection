"""E1005 stable G IDs, selector isolation, reporting, and integration contracts."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


class ProtocolContracts(unittest.TestCase):
    def test_dependency_free_catalog_exposes_all_numbered_routes(self):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "experiments/run_e1005.py"),
                                 "--dry_run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        catalog = json.loads(result.stdout)
        self.assertEqual(catalog["experiment"], "E1005")
        arms = catalog["arms"]
        self.assertEqual(len(arms), 50)
        self.assertEqual({row["family"] for row in arms.values()},
                         {"G31", "G32", "G33", "G34", "G35", "G36"})
        row = arms["G32_J_MKGN"]
        self.assertEqual(row["scope"], "J")
        self.assertEqual(row["objective"]["rank"], "matched")
        self.assertTrue(row["objective"]["keep"])
        self.assertTrue(row["objective"]["method_risk"])
        self.assertTrue(row["objective"]["nuisance_risk"])
        self.assertEqual(row["design_id"], "B_J_MKGN")
        self.assertEqual(catalog["seed"], 1024)

    def test_design_aliases_resolve_to_the_same_immutable_arm(self):
        module = importlib.import_module("e1005_protocol")
        by_g = module.resolve_arms(["G32_J_MKGN"])
        by_alias = module.resolve_arms(["B_J_MKGN"])
        self.assertEqual(by_g, by_alias)
        by_alias[0]["objective"]["keep"] = False
        self.assertTrue(module.resolve_arms(["G32_J_MKGN"])[0]["objective"]["keep"])
        with self.assertRaises(ValueError):
            module.resolve_arms(["G32_J_MKGN", "B_J_MKGN"])

    def test_champion_uses_development_only_and_stable_tie_rules(self):
        module = importlib.import_module("e1005_reporting")
        rows = {
            "G32_H_M": {"status": "OK", "primary_auc": .9, "selected_step": 100,
                          "trainable_parameters": 10, "historical_mean6": .999},
            "G32_J_M": {"status": "OK", "primary_auc": .95, "selected_step": 250,
                          "trainable_parameters": 20, "historical_mean6": .7},
            "G33_GEND": {"status": "FAILED", "primary_auc": 1., "selected_step": 0,
                          "trainable_parameters": 1},
        }
        self.assertEqual(module.choose_champion(rows), "G32_J_M")
        rows["G32_H_M"]["primary_auc"] = .95
        self.assertEqual(module.choose_champion(rows), "G32_H_M")

    def test_selector_reuses_one_trajectory_and_keeps_earliest_tie(self):
        module = importlib.import_module("e1005_reporting")
        history = [
            {"step": 0, "source": {"frame_auc": .7, "video_auc": .8}, "development": {"video_auc": .9}},
            {"step": 100, "source": {"frame_auc": .9, "video_auc": .85}, "development": {"video_auc": .88}},
            {"step": 250, "source": {"frame_auc": .9, "video_auc": .95}, "development": {"video_auc": .9}},
        ]
        chosen = module.select_steps(history)
        self.assertEqual(chosen, {"S1": 100, "S2": 250, "S3": 0, "S4": 250})

    def test_ranking_damage_is_pairwise_auc_not_threshold_accuracy(self):
        module = importlib.import_module("e1005_reporting")
        report = module.ranking_changes([0, 0, 1, 1], [.1, .4, .3, .8], [.5, .5, .5, .5])
        self.assertAlmostEqual(report["repair"], .125)
        self.assertAlmostEqual(report["damage"], .375)
        self.assertAlmostEqual(report["delta_auc"], -.25)

    def test_dynamic_ablation_only_changes_its_declared_factor(self):
        module = importlib.import_module("e1005_protocol")
        leader = module.resolve_arms(["G32_J_MKGN"])[0]
        shuffled = module.materialize_arm(module.resolve_arms(["G34_SHUFFLE"])[0], leader)
        self.assertEqual(shuffled["scope"], "J")
        self.assertEqual(shuffled["objective"]["rank"], "shuffled")
        self.assertTrue(shuffled["objective"]["keep"])
        self.assertTrue(shuffled["objective"]["method_risk"])
        self.assertEqual(leader["objective"]["rank"], "matched")


if __name__ == "__main__":
    unittest.main()


def test_small_snapshot_is_identity_bound_and_never_contains_frozen_backbone(tmp_path):
    import pytest
    import torch
    from test_g30 import TinyB0
    runner = importlib.import_module("run_e1005")
    spec = importlib.import_module("e1005_protocol").resolve_arms(["G31_HEAD"])[0]
    models = runner.core_module("models")
    model = models.build_model(TinyB0("eager"), spec)
    artifact = runner.snapshot_artifact(model, "a" * 64, "b" * 64, 0)
    assert not any("backbone" in key for key in artifact["state"])
    with pytest.raises(ValueError, match="identity"):
        runner.restore_artifact(model, artifact, "c" * 64, "b" * 64)
    artifact["state"][next(iter(artifact["state"]))].fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        runner.restore_artifact(model, artifact, "a" * 64, "b" * 64)


def test_progress_is_persistent_during_baseline_export(tmp_path, capsys):
    runner = importlib.import_module("run_e1005")
    progress = runner.Progress(tmp_path)
    progress.update("baseline", "B0", "WDF", 1, 8, frames=32)
    assert "baseline" in capsys.readouterr().out
    assert "WDF" in (tmp_path / "run.log").read_text()
    value = json.loads((tmp_path / "progress.json").read_text())
    assert value["completed"] == 1 and value["total"] == 8
    assert value["frames"] == 32


def test_all_baseline_cache_scopes_reject_changed_bytes_and_changed_score(tmp_path):
    import numpy as np
    import pytest
    runner = importlib.import_module("run_e1005")
    values = {"labels": np.array([0, 1]), "path": np.array(["a/1", "b/1"]),
              "video_id": np.array(["a", "b"]), "cls_prob": np.array([.1, .9]),
              "global_log_odds": np.array([-2., 2.]), "score": np.array([.1, .9])}
    manifest = {"baseline_sha256": {}}
    path = tmp_path / "B0_uniform/WDF.npz"
    path.parent.mkdir()
    runner.cached_baseline(path, lambda: values, manifest, tmp_path, False)
    drift = {**values, "score": 1 - values["score"]}
    runner.legacy.save_npz(path, drift)
    with pytest.raises(ValueError, match="identity"):
        runner.cached_baseline(path, lambda: values, manifest, tmp_path, True)
    manifest["baseline_sha256"][str(path.relative_to(tmp_path))] = runner.legacy.file_sha256(path)
    with pytest.raises(ValueError, match="probability"):
        runner.cached_baseline(path, lambda: values, manifest, tmp_path, True)


def test_initial_results_can_be_created_when_interruption_precedes_first_arm(tmp_path):
    runner = importlib.import_module("run_e1005")
    results = runner.load_or_initialize_results(tmp_path, resume=True)
    assert results["experiment"] == "E1005" and results["arms"] == {}
    assert json.loads((tmp_path / "all_results.json").read_text()) == results


def test_asset_audit_rejects_content_drift_before_overwriting_resume_evidence(tmp_path):
    import pytest
    runner = importlib.import_module("run_e1005")
    old = {"audited_identities": {"mask": [{"sha256": "a" * 64}], "native": []}}
    runner.checked_asset_audit(tmp_path, old, resume=False)
    path = tmp_path / "asset_audit_result.json"
    saved = path.read_bytes()
    with pytest.raises(ValueError, match="asset.*identity"):
        runner.checked_asset_audit(tmp_path, {"audited_identities": {"mask": [{"sha256": "b" * 64}], "native": []}},
                                   resume=True)
    assert path.read_bytes() == saved
    runner.checked_asset_audit(tmp_path, old, resume=True)


def test_ordinary_comparison_uses_the_strongest_reference_without_switching_champion():
    import numpy as np
    runner = importlib.import_module("run_e1005")
    def export(score):
        return {"labels": np.array([0, 0, 1, 1]), "path": np.array(["a/0", "b/0", "c/0", "d/0"]),
                "video_id": np.array(["a", "b", "c", "d"]), "cls_prob": np.array([.1, .4, .3, .8]),
                "global_log_odds": np.array([-2., -.4, -.8, 1.]), "score": np.array(score)}
    protocol = importlib.import_module("e1005_protocol")
    exports = {gid: {dataset: export(scores) for dataset in protocol.REGRESSION_DATASETS}
               for gid, scores in {"champion": [.1, .4, .3, .8], "weak": [.5, .5, .5, .5],
                                   "strong": [.1, .2, .8, .9]}.items()}
    comparison = runner.ordinary_comparison(exports, "champion", {"weak", "strong"}, 8, 1024)
    assert comparison["ordinary_baseline_id"] == "strong"
    assert comparison["mean_delta"] == -.25
    assert not comparison["champion_is_reference_recipe"]
    reference_champion = runner.ordinary_comparison(exports, "strong", {"weak", "strong"}, 8, 1024)
    assert reference_champion["ordinary_baseline_id"] == "weak"
    assert reference_champion["champion_is_reference_recipe"]


import pytest


@pytest.mark.parametrize("native_champion", [False, True])
def test_all_numbered_families_use_real_tiny_clip_training_and_exports(tmp_path, monkeypatch, native_champion):
    import copy
    import hashlib
    import numpy as np
    from PIL import Image
    import torch
    from types import SimpleNamespace
    from test_g30 import TinyB0
    from run_g25 import TEST_DS
    runner = importlib.import_module("run_e1005")
    data = importlib.import_module("e1005_data")
    torch.set_num_threads(1)
    rgb_root, metadata_root = tmp_path / "rgb", tmp_path / "metadata"
    rgb_root.mkdir()
    metadata_root.mkdir()
    masks, native = {}, {}

    def frame(path, fake=False):
        value = (sum(path.encode()) % 128) + 64
        pixels = np.full((8, 8, 3), value, dtype=np.uint8)
        asset = rgb_root / path
        asset.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(asset)
        high = rgb_root / "native" / path
        high.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.repeat(np.repeat(pixels, 2, 0), 2, 1)).save(high)
        native[path] = {"path": str(high.relative_to(rgb_root)), "same_frame_verified": True,
                        "crop_verified": True, "native_verified": True, "evidence": "synthetic native fixture"}
        if fake:
            mask_path = rgb_root / "mask" / path
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            Image.fromarray(mask).save(mask_path)
            masks[path] = {"path": str(mask_path.relative_to(rgb_root)), "coordinates_verified": True,
                           "evidence": "registered synthetic fixture"}

    ff = {}
    labels = {"FF-real": 0, **{f"FF-{method}": 1 for method in data.METHODS}}
    for label in labels:
        ff[label] = {}
        for split, start in (("train", 0), ("val", 100), ("test", 200)):
            videos = {}
            for offset in range(6):
                target, source = f"{start + offset:03}", f"{start + (offset + 1) % 6:03}"
                name = target if label == "FF-real" else f"{target}_{source}"
                paths = [f"FaceForensics++/{label}/{name}/{index}.png" for index in (0, 3, 10, 20)]
                for path in paths:
                    frame(path, label != "FF-real")
                videos[name] = {"label": label, "frames": paths}
            ff[label][split] = {"c23": videos}
    (metadata_root / "FaceForensics++.json").write_text(json.dumps({"FaceForensics++": ff}))
    for dataset in TEST_DS:
        if dataset == "FaceForensics++":
            continue
        root = {}
        for binary in (0, 1):
            label = f"{dataset}_{binary}"
            labels[label] = binary
            videos = {}
            for offset in range(2):
                name = f"video{offset}"
                paths = [f"{dataset}/{label}/{name}/{index}.png" for index in (0, 10)]
                for path in paths:
                    frame(path, bool(binary))
                videos[name] = {"label": label, "frames": paths}
            root[label] = {"test": {"c23": videos} if dataset == "DeepFakeDetection" else videos}
        (metadata_root / f"{dataset}.json").write_text(json.dumps({dataset: root}))
    train = data.build_manifest({"FaceForensics++": ff}, "FaceForensics++", "train", labels)
    candidates = data.build_pair_candidates(train)
    audit = {"schema_version": 1, "receipt_id": "synthetic-pair-audit", "pairs": []}
    for pair in candidates["pairs"]:
        audit["pairs"].append({"pair_id": pair["pair_id"], "real_video_id": pair["real"]["video_id"],
            "fake_video_id": pair["fake"]["video_id"], "target_id": pair["fake"]["target_id"],
            "source_id": pair["fake"]["source_id"], "common_indices": pair["common_indices"],
            "time_verified": True, "target_face_verified": True, "evidence": "aligned synthetic episode"})
    pair_path = tmp_path / "pair_audit.json"
    pair_path.write_text(json.dumps(audit))
    asset_path = tmp_path / "asset_audit.json"
    asset_path.write_text(json.dumps({"schema_version": 1, "receipt_id": "synthetic-assets",
                                     "registered_masks": masks, "native_crops": native}))
    config = {"model_name": "effort", "full_train_head": True, "use_loralib": True,
              "dataset_json_folder": str(metadata_root), "label_dict": labels, "compression": "c23",
              "resolution": 8, "mean": [0., 0., 0.], "std": [1., 1., 1.],
              "train_batchSize": 32, "test_batchSize": 32, "use_data_augmentation": False}
    config_path, checkpoint = tmp_path / "base.json", tmp_path / "B0.pth"
    config_path.write_text(json.dumps(config))
    original = TinyB0("eager").eval()
    original.config = config
    torch.save(original.state_dict(), checkpoint)
    original_state = runner.legacy.state_sha256(original)

    def load_model(cfg, path):
        model = TinyB0("eager")
        model.load_state_dict(torch.load(path, weights_only=True))
        model.config = cfg
        return model.eval()

    def pristine(cfg):
        model = TinyB0("eager")
        model.config = cfg
        return model.eval()

    utilities = SimpleNamespace(DEVICE=torch.device("cpu"), load_model=load_model,
                                DETECTOR={"effort": pristine})
    monkeypatch.setitem(sys.modules, "experiment_utils", utilities)
    output = tmp_path / "E1005"
    choose = runner.reporting.choose_champion
    if native_champion:
        def native_choice(rows, allowed_families=None):
            if allowed_families and "G36" in allowed_families:
                return "G36_NATIVE_448"
            return choose(rows, allowed_families)
        monkeypatch.setattr(runner.reporting, "choose_champion", native_choice)
    command = ["--base_checkpoint", str(checkpoint), "--base_config", str(config_path),
                       "--output_dir", str(output), "--rgb_root", str(rgb_root),
                       "--pair_audit", str(pair_path), "--asset_audit", str(asset_path),
                       "--steps", "1", "--cold_steps", "1", "--base_selected_steps", "0",
                       "--eval_every", "1", "--eval_frames", "2", "--train_frames", "2",
                       "--reader_backend", "pil", "--local_layers", "0", "1",
                       "--late_layers", "1", "--pixel_layers", "0", "1", "--last_layers", "1",
                       "--adapter_width", "4", "--rank", "2", "--native_resolution", "16",
                       "--bootstrap_repeats", "8", "--repeat_runs", "1"]
    code = runner.main(command)
    assert code == 0
    folder, = output.iterdir()
    result = json.loads((folder / "all_results.json").read_text())
    assert set(row["family"] for row in result["arms"].values()) == {
        "G31", "G32", "G33", "G34", "G35", "G36"}
    assert len(result["arms"]) == 50
    assert all(row["status"] == "OK" for gid, row in result["arms"].items() if gid.startswith("G32_"))
    assert (folder / "champion_lock.json").is_file()
    assert json.loads((folder / "manifest.json").read_text())["base_state_unchanged"]
    assert runner.legacy.state_sha256(original) == original_state
    for gid, row in result["arms"].items():
        if row["status"] != "OK":
            assert row["status"] in {"NOT_APPLICABLE", "REUSED"}
            continue
        artifact = torch.load(row["checkpoint"], weights_only=True)
        assert artifact["base_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert artifact["training_signature"] == row["training_signature"]
    assert result["evaluation"]["champion"] == json.loads((folder / "champion_lock.json").read_text())["champion"]
    assert not result["evaluation"]["can_claim_breakthrough"], "smoke budgets cannot support a performance claim"
    assert result["evaluation"]["reproduction"]["passed"]
    assert result["evaluation"]["assessment"]["ordinary_comparison"]["ordinary_baseline_id"]
    if native_champion:
        selections = json.loads((folder / "champion_lock.json").read_text())["final_selection"]
        assert all(selections[gid]["input_scope"] == "native" for gid in runner.ORDINARY_CONTROLS)
    assert runner.main(command + ["--resume", str(folder)]) == 0
    assert json.loads((folder / "all_results.json").read_text())["evaluation"]["reproduction"]["passed"]
    if not native_champion:
        partial_output = tmp_path / "no_pair_receipt"
        partial = command.copy()
        partial[partial.index("--output_dir") + 1] = str(partial_output)
        for flag in ("--pair_audit", "--asset_audit"):
            index = partial.index(flag)
            del partial[index:index + 2]
        assert runner.main(partial + ["--arms", "G31", "G32_H_V", "G33", "--no_evaluation"]) == 0
        partial_folder, = partial_output.iterdir()
        partial_result = json.loads((partial_folder / "all_results.json").read_text())
        assert partial_result["status"] == "PARTIAL_PREREQUISITES"
        assert partial_result["arms"]["G32_H_V"]["status"] == "NOT_ELIGIBLE_PAIR_AUDIT"
        assert partial_result["arms"]["G33_VIDEO_B0"]["status"] == "NOT_ELIGIBLE_PAIR_AUDIT"
        assert partial_result["arms"]["G33_MATCHED_B0"]["status"] == "OK"
        train_arm = runner.train_arm
        def interrupted(spec, *args, **kwargs):
            if spec["id"] == "G31_HEAD":
                raise RuntimeError("synthetic interrupted arm")
            return train_arm(spec, *args, **kwargs)
        monkeypatch.setattr(runner, "train_arm", interrupted)
        failed_output = tmp_path / "failed_arm"
        failed = partial.copy()
        failed[failed.index("--output_dir") + 1] = str(failed_output)
        assert runner.main(failed + ["--arms", "G31", "--no_evaluation"]) == 1
        failed_folder, = failed_output.iterdir()
        failed_result = json.loads((failed_folder / "all_results.json").read_text())
        assert failed_result["arms"]["G31_HEAD"]["status"] == "FAILED"
        assert failed_result["arms"]["G31_LORA"]["status"] == "OK"
        monkeypatch.setattr(runner, "train_arm", train_arm)
        assert runner.main(failed + ["--arms", "G31", "--no_evaluation", "--resume", str(failed_folder)]) == 0
        assert json.loads((failed_folder / "all_results.json").read_text())["arms"]["G31_HEAD"]["status"] == "OK"

"""E1010 protocol, B0 isolation, selection, and real tiny-model training."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_e1010_baseline import config_builder


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/run_e1010.py"
SUPERVISION_NAMES = ["PLAIN", "UNIFORM", "FL", "MIL", "UNIFORM_MIL", "FL_MIL"]
ARM_NAMES = ([f"G1_{structure}_{objective}" for structure in ("L1", "L2", "L3") for objective in SUPERVISION_NAMES]
             + [f"G2_{structure}_{objective}" for structure in ("M00", "M10", "M01", "M11", "A00", "A10", "A01", "A11")
                for objective in SUPERVISION_NAMES]
             + [f"G3_{objective}" for objective in SUPERVISION_NAMES]
             + ["G4_ALL", "G4_RANDOM", "G4_TOPK"])


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    assert SCRIPT.is_file(), "E1010 runner has not been implemented"
    spec = importlib.util.spec_from_file_location("e1010_runner_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_lists_all_three_families_without_ml_dependencies(tmp_path):
    process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--dry_run",
                              "--output_dir", str(tmp_path / "unused")],
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    protocol = json.loads(process.stdout)
    assert protocol["experiment"] == "E1010" and protocol["base_training"] is True
    assert protocol["baseline"]["nEpochs"] == 10
    assert protocol["baseline"]["training_passes"] == 11
    assert list(protocol["arms"]) == ARM_NAMES
    assert len(protocol["arms"]) == 75
    assert protocol["groups"] == {"G0": "B0", "G1": "lfeq", "G2": "aux", "G3": "decoder", "G4": "prototype"}
    assert protocol["arms"]["G1_L1_PLAIN"]["num_tokens"] == 8
    assert protocol["arms"]["G1_L1_PLAIN"]["num_heads"] == 8
    assert protocol["arms"]["G1_L1_PLAIN"]["memory_source"] == "final"
    assert protocol["arms"]["G1_L1_PLAIN"]["lfeq_dropout"] == .1
    assert protocol["arms"]["G1_L2_PLAIN"]["lfeq_fusion_weight"] == 1.
    assert protocol["arms"]["G1_L3_PLAIN"]["lfeq_fusion_weight"] == 0.
    assert protocol["arms"]["G2_M00_PLAIN"]["num_tokens"] == 4
    assert protocol["arms"]["G2_M00_PLAIN"]["num_heads"] == 4
    assert protocol["arms"]["G2_M00_PLAIN"]["memory_source"] == "block_input"
    assert protocol["arms"]["G2_A11_PLAIN"]["attention_mode"] == "full"
    assert protocol["arms"]["G2_A11_PLAIN"]["supervision"] == "all"
    assert protocol["arms"]["G3_PLAIN"]["family"] == "decoder"
    assert protocol["arms"]["G4_TOPK"]["family"] == "prototype"
    assert protocol["arms"]["G4_TOPK"]["max_frames_per_class"] == 32
    assert protocol["arms"]["G4_TOPK"]["top_k"] == 16
    assert protocol["decisions"] == []
    assert protocol["independent_datasets"] == ["WDF", "FFIW", "DeepFakeDetection",
                                                "DFDC", "DFDCP", "DeeperForensics-1.0"]
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("prefix,family", [("G1_L1", "lfeq"), ("G1_L2", "lfeq"), ("G1_L3", "lfeq"),
                                          ("G2_M00", "aux"), ("G2_M10", "aux"), ("G2_M01", "aux"), ("G2_M11", "aux"),
                                          ("G2_A00", "aux"), ("G2_A10", "aux"), ("G2_A01", "aux"), ("G2_A11", "aux"),
                                          ("G3", "decoder")])
def test_ablation_arms_preserve_structure_and_only_change_local_objective(runner, prefix, family):
    args = runner.build_parser().parse_args(["--num_tokens", "3", "--memory_layer", "17",
                                           "--hidden_dim", "64", "--num_heads", "4",
                                           "--depth", "1"])
    wanted = {"PLAIN": ("off", False), "UNIFORM": ("uniform", False),
              "FL": ("weighted", False), "MIL": ("off", True), "UNIFORM_MIL": ("uniform", True),
              "FL_MIL": ("weighted", True)}
    structures = []
    for suffix, (likelihood, asymmetric) in wanted.items():
        settings = runner.arm_settings(args, f"{prefix}_{suffix}")
        assert settings["family"] == family
        assert settings["likelihood"] == likelihood and settings["asymmetric"] is asymmetric
        assert settings["num_tokens"] == 3 and settings["memory_layer"] == 17
        assert settings["hidden_dim"] == 64 and settings["depth"] == 1
        structures.append({key: value for key, value in settings.items()
                           if key not in ("likelihood", "asymmetric")})
    assert all(settings == structures[0] for settings in structures)


@pytest.mark.parametrize("argv", [
    ["--n_epochs", "0"], ["--aux_lr", "nan"], ["--aux_lr", "0"],
    ["--weight_decay", "-1"], ["--hidden_dim", "15"], ["--num_tokens", "0"],
    ["--lfeq_num_tokens", "0"], ["--aux_num_tokens", "0"], ["--lfeq_dropout", "1"],
    ["--memory_layer", "-1"], ["--sampler_real_ratio", "1"],
    ["--likelihood_temperature", "0"], ["--likelihood_weight", "nan"],
    ["--contrastive_temperature", "-1"], ["--max_contrastive_patches", "0"],
    ["--arms", "G1_L1_FL", "G1_L1_FL"], ["--gate_width", "0.8"],
    ["--aux_max_weight", "0.6"], ["--diversity_weight", "-0.1"],
])
def test_invalid_protocol_is_rejected_before_dry_run(runner, argv):
    with pytest.raises(SystemExit):
        runner.main(["--dry_run", *argv])


def test_likelihood_sampling_requires_other_fake_image_with_exact_sampler_rounding(runner):
    runner.validate_sampling(4, .3, require_two_fake=True)
    runner.validate_sampling(3, .3, require_two_fake=False)
    with pytest.raises(ValueError, match="two fake"):
        runner.validate_sampling(3, .3, require_two_fake=True)
    with pytest.raises(ValueError, match="two fake"):
        runner.validate_sampling(4, .7, require_two_fake=True)


def test_readout_metrics_keep_fullpath_identity_and_report_real_false_positives(runner):
    data = {"labels": np.array([0, 0, 1, 1]),
            "video_id": np.array(["real/a", "real/b", "fake/a", "fake/b"]),
            "cls_prob": np.array([.6, .1, .9, .2])}
    scores = np.array([.4, .7, .8, .9])
    result = runner.readout_metrics(data, scores)
    assert result["video_auc_fullpath"] == pytest.approx(1.0)
    assert result["video_auc"] == result["video_auc_fullpath"]
    assert result["num_videos"] == 4 and result["legacy_group_collisions"] == 2
    assert result["legacy_mixed_label_groups"] == 2
    assert result["real_fpr"] == .5
    assert result["corrected_frames"] == 2 and result["harmed_frames"] == 1


def test_disabled_auxiliary_export_reads_original_b0_even_with_a_sidecar(runner):
    import torch
    from test_g30 import TinyB0

    base = TinyB0("eager").eval()
    model = runner.core_module().FrozenForgerySidecar(
        base, family="decoder", memory_layer=1, hidden_dim=8, num_heads=4, depth=1)
    batch = {"image": torch.randn(4, 3, 8, 8), "label": torch.tensor([0, 1, 0, 1]),
             "path": [f"frame/{i}/0.png" for i in range(4)],
             "video_id": [f"frame/{i}" for i in range(4)]}
    reference = runner.export_scores(base, [batch], auxiliary_enabled=False)
    actual = runner.export_scores(model, [batch], auxiliary_enabled=False)
    runner.verify_paired_base(reference, actual)
    assert set(actual) == set(reference)


def test_independent_mean_excludes_selection_and_training_datasets(runner):
    datasets = {dataset: {"gated": {"video_auc_fullpath": .7}}
                for dataset in runner.TEST_DS}
    datasets["Celeb-DF-v2"]["gated"]["video_auc_fullpath"] = 1.0
    datasets["FaceForensics++"]["gated"]["video_auc_fullpath"] = .1
    report = runner.independent_mean(datasets, "gated")
    assert report["video_auc_fullpath_mean"] == pytest.approx(.7)
    assert len(report["datasets"]) == 6
    assert report["excluded"] == ["Celeb-DF-v2", "FaceForensics++"]


@pytest.fixture
def tiny_source(runner, tmp_path, monkeypatch, request):
    import random
    import torch
    from test_g30 import TinyB0
    from test_g26 import configure_lora_backend
    from test_e1001 import evaluation_metadata
    from e0924_export import strict_loader

    configure_lora_backend(monkeypatch, getattr(request, "param", "custom"))

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for dataset in runner.TEST_DS:
        (metadata / f"{dataset}.json").write_text(json.dumps(evaluation_metadata(dataset)))
    config = {"model_name": "effort", "dataset_json_folder": str(metadata), "compression": "c23",
              "test_batchSize": 4, "train_batchSize": 4,
              "label_dict": {"FF-real": 0, "FF-DF": 1}, "std": [1., 1., 1.]}
    config_path, checkpoint = tmp_path / "config.json", tmp_path / "b0.pth"
    config_path.write_text(json.dumps(config))
    base = TinyB0("eager").eval()
    torch.save(base.state_dict(), checkpoint)

    class Frames:
        def __init__(self, cfg, dataset):
            self.config = cfg
            rows = [(f"{dataset}/{label}/video{i}/0.png", label) for label in (0, 1) for i in range(4)]
            random.shuffle(rows)
            self.image_list, self.label_list = map(list, zip(*rows))

        def load_rgb(self, path):
            return torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(sum(path.encode())))

        def to_tensor(self, value):
            return value

        def normalize(self, value):
            return value

    def load_model(cfg, path):
        result = TinyB0("eager")
        result.load_state_dict(torch.load(path, weights_only=True), strict=True)
        result.config = cfg
        return result.eval()

    utilities = SimpleNamespace(load_model=load_model, DEVICE=torch.device("cpu"),
                                get_data_loader=lambda cfg, ds: SimpleNamespace(dataset=Frames(cfg, ds)))
    monkeypatch.setitem(sys.modules, "experiment_utils", utilities)

    def training_loader(cfg, ratio):
        from torch.utils.data import DataLoader
        source = Frames(cfg, "FaceForensics++")
        dataset = strict_loader(source, 4).dataset
        # Balanced mixed batches, independent of the shuffled source ordering.
        rows = [(i, dataset[i]["label"]) for i in range(len(dataset))]
        real, fake = [[i for i, label in rows if label == value] for value in (0, 1)]
        batches = [[real[0], fake[0], fake[1], fake[2]], [real[1], fake[0], fake[1], fake[3]]]
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=0)
        loader.source_dataset = source
        return loader

    monkeypatch.setattr(runner, "make_train_loader", training_loader)
    return checkpoint, config_path


@pytest.mark.parametrize("arm,tiny_source", [("G1_L1_FL_MIL", "custom"), ("G2_M00_FL_MIL", "custom"),
                                            ("G3_FL_MIL", "custom"), ("G2_M11_FL_MIL", "loralib"),
                                            ("G2_A11_FL_MIL", "custom")], indirect=["tiny_source"])
def test_main_trains_reloads_and_exports_auxiliary_without_changing_b0(runner, tiny_source,
                                                                    tmp_path, arm):
    import torch
    checkpoint, config_path = tiny_source
    output = tmp_path / "E1010"
    assert runner.main(["--base_checkpoint", str(checkpoint), "--base_config", str(config_path),
                        "--output_dir", str(output), "--arms", arm, "--n_epochs", "1",
                        "--num_tokens", "2", "--memory_layer", "1", "--hidden_dim", "8",
                        "--num_heads", "4", "--depth", "1", "--max_contrastive_patches", "2"]) == 0
    root, = output.iterdir()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["base_state_unchanged"] and manifest["base_file_unchanged"]
    assert manifest["protocol"]["base_training"] is False
    assert "training/detectors/e1010_tokens.py" in manifest["source_sha256"]
    assert "experiments/run_e1010.py" in manifest["source_sha256"]
    assert "experiments/run_e1001.py" in manifest["source_sha256"]
    assert manifest["status"] == "OK"
    result = json.loads((root / arm / "result.json").read_text())
    assert result["status"] == "OK" and result["base_scores_bitwise_identical"]
    assert result["selected_epoch"] == 1
    assert len(result["independent_mean"]["gated"]["datasets"]) == 6
    artifact = torch.load(root / arm / "auxiliary_best.pth", weights_only=True)
    assert artifact["base_sha256"] == manifest["source"]["base_sha256"]
    assert not any(key.startswith("base.") for key in artifact["state_dict"])
    for dataset in runner.TEST_DS:
        with np.load(root / "G0" / f"{dataset}.npz") as before:
            with np.load(root / arm / "exports" / f"{dataset}.npz") as after:
                runner.verify_paired_base(before, after)
                assert "query_log_odds" in after.files
        for name in ("B0", "evidence", "gated"):
            assert "real_fpr" in result["datasets"][dataset][name]
    all_results = json.loads((root / "all_results.json").read_text())
    assert list(all_results["arms"]) == [arm]
    assert all_results["arms"][arm]["status"] == "OK"
    assert all_results["G0"]["status"] == "OK"


def test_drift_aborts_before_training_later_arms(runner, tiny_source, tmp_path, monkeypatch):
    import torch
    checkpoint, config_path = tiny_source
    model_class = runner.core_module().FrozenForgerySidecar
    original_losses = model_class.losses

    def corrupt_base(model, output, labels):
        losses = original_losses(model, output, labels)
        with torch.no_grad():
            model.base.head.bias[1].add_(1)
        return losses

    monkeypatch.setattr(model_class, "losses", corrupt_base)
    output = tmp_path / "drift"
    code = runner.main(["--base_checkpoint", str(checkpoint), "--base_config", str(config_path),
                        "--output_dir", str(output), "--arms", "G1_L1_PLAIN", "G2_M00_PLAIN",
                        "--n_epochs", "1", "--memory_layer", "1", "--hidden_dim", "8",
                        "--num_heads", "4", "--depth", "1"])
    assert code == 1
    root, = output.iterdir()
    results = json.loads((root / "all_results.json").read_text())
    assert results["arms"]["G1_L1_PLAIN"]["status"] == "FAILED"
    assert "B0 isolation" in results["arms"]["G1_L1_PLAIN"]["error"]
    assert results["arms"]["G2_M00_PLAIN"]["status"] == "NOT_RUN"
    assert not (root / "G2_M00_PLAIN").exists()


def test_wrong_config_and_corrupt_checkpoint_do_not_report_success(runner, tiny_source, tmp_path):
    checkpoint, config_path = tiny_source
    config = json.loads(config_path.read_text())
    config["model_name"] = "effort_g26"
    config_path.write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        runner.main(["--base_checkpoint", str(checkpoint), "--base_config", str(config_path), "--preflight"])
    config["model_name"] = "effort"
    config_path.write_text(json.dumps(config))
    checkpoint.write_bytes(b"not a torch checkpoint")
    output = tmp_path / "broken"
    assert runner.main(["--base_checkpoint", str(checkpoint), "--base_config", str(config_path),
                        "--output_dir", str(output), "--arms", "G1_L1_PLAIN"]) == 1
    root, = output.iterdir()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "BASE_LOAD_FAILED"
    assert json.loads((root / "all_results.json").read_text())["G0"]["status"] == "FAILED"


def test_g4_train_pool_is_seeded_unique_and_excludes_val_test(runner, tiny_source):
    _, config_path = tiny_source
    config = json.loads(config_path.read_text())
    args = runner.build_parser().parse_args(["--prototype_frames_per_class", "2"])
    records, info = runner.prototype_records(config, args)
    assert runner.prototype_records(config, args) == (records, info)
    paths = [path for record in records for path in record["frames"]]
    assert len(paths) == len(set(paths)) == 4
    assert info["split"] == "train" and info["counts"] == {"real": 2, "fake": 2}
    assert all("/001/" in path or "/002/" in path or "/001_002/" in path for path in paths)


def test_g4_only_does_not_require_auxiliary_training(runner, capsys):
    assert runner.main(["--dry_run", "--arms", "G4_ALL", "G4_RANDOM", "G4_TOPK",
                        "--n_epochs", "0", "--hidden_dim", "15"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["prototype_probe"]["training"] is False
    assert report["prototype_probe"]["prototype_scope"] == "global_shared_train_only"


def test_g2_ignores_divisibility_of_unused_decoder_dimensions(runner, capsys):
    assert runner.main(["--dry_run", "--arms", "G2_M00_PLAIN", "--hidden_dim", "15"]) == 0
    assert json.loads(capsys.readouterr().out)["arms"]["G2_M00_PLAIN"]["family"] == "aux"


def test_g4_end_to_end_shared_train_pool_reloads_and_preserves_b0(runner, tiny_source, tmp_path, monkeypatch):
    import torch

    checkpoint, config_path = tiny_source
    config = json.loads(config_path.read_text())
    config["train_batchSize"] = 1  # G4 is a prototype probe, without balanced training.
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "make_train_loader", lambda *a: pytest.fail("G4 must not train auxiliary models"))
    output = tmp_path / "G4"
    code = runner.main(["--base_checkpoint", str(checkpoint), "--base_config", str(config_path),
                        "--output_dir", str(output), "--arms", "G4_ALL", "G4_RANDOM", "G4_TOPK",
                        "--prototype_top_k", "2", "--prototype_pool_top_k", "2",
                        "--prototype_frames_per_class", "1", "--occlusion_batch_size", "3",
                        "--n_epochs", "0", "--sampler_real_ratio", "1"])
    assert code == 0
    root, = output.iterdir()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["base_state_unchanged"] and manifest["base_file_unchanged"]
    assert "experiments/e1010_prototypes.py" in manifest["source_sha256"]
    selected = json.loads((root / "G4_selection.json").read_text())
    assert selected["split"] == "train" and selected["counts"] == {"real": 1, "fake": 1}
    paths = []
    for arm in ("G4_ALL", "G4_RANDOM", "G4_TOPK"):
        artifact = torch.load(root / arm / "prototype.pth", weights_only=True)
        assert artifact["base_sha256"] == manifest["source"]["base_sha256"]
        assert artifact["counts"]["real_patches"] == 4
        assert artifact["counts"]["fake_patches"] == (4 if arm == "G4_ALL" else 2)
        with np.load(root / arm / "selection.npz", allow_pickle=False) as audit:
            paths.append(audit["path"].copy())
        result = json.loads((root / arm / "result.json").read_text())
        assert result["status"] == "OK" and result["training"] is False
        for dataset in runner.TEST_DS:
            with np.load(root / "G0" / f"{dataset}.npz") as before:
                with np.load(root / arm / "exports" / f"{dataset}.npz") as after:
                    runner.verify_paired_base(before, after)
                    assert "prototype_prob" in after.files
            assert set(result["datasets"][dataset]) == {"B0", "prototype", "gated"}
    assert all(np.array_equal(paths[0], path) for path in paths[1:])


@pytest.mark.parametrize("flag,value", [("--prototype_top_k", "0"), ("--prototype_pool_top_k", "0"),
                                        ("--prototype_frames_per_class", "0"), ("--occlusion_batch_size", "0"),
                                        ("--prototype_temperature", "nan")])
def test_invalid_g4_configuration_fails_before_loading(runner, flag, value):
    with pytest.raises(SystemExit):
        runner.main(["--dry_run", "--arms", "G4_ALL", flag, value])


def test_baseline_only_plan_has_no_auxiliary_groups_and_no_ml(tmp_path):
    process = subprocess.run([sys.executable, "-S", str(SCRIPT), "--dry_run", "--baseline_only"],
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    plan = json.loads(process.stdout)
    assert plan["base_training"] is True and plan["arms"] == {}
    assert plan["baseline"]["protocol"] == "E0924/G26_B0"


@pytest.fixture
def fresh_training(tiny_source, runner, config_builder, monkeypatch):
    import torch
    from test_g30 import TinyB0

    _, config_path = tiny_source
    source_config = json.loads(config_path.read_text())
    utilities = sys.modules["experiment_utils"]
    utilities.build_config = config_builder
    calls = []

    def train(config, train_dataset, validation_dataset):
        calls.append(dict(config))
        assert config["use_data_augmentation"] is True
        assert config["use_mixup"] is False and config["full_train_head"] is True
        assert config["sampler_real_ratio"] == .3 and config["nEpochs"] == 0
        model = TinyB0("eager")
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=2e-4)
        logits = model({"image": torch.randn(4, 3, 8, 8)})["cls"]
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
        optimizer.step()
        checkpoint = Path(config["log_dir"]) / "tiny_trained_b0.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint)
        return str(checkpoint)

    def evaluate(config, checkpoint, datasets, train_dataset, folder, exp_name):
        assert config["e0924_protocol"] is True
        return {"testall": {ds: {"video_auc": .8, "auc": .79} for ds in runner.TEST_DS}}

    utilities.train_model = train
    utilities.evaluate_model = evaluate
    return Path(source_config["dataset_json_folder"]), utilities, calls


def test_fresh_baseline_only_trains_then_exports_both_metric_protocols(runner, fresh_training, tmp_path):
    metadata, _, calls = fresh_training
    output = tmp_path / "fresh"
    assert runner.main(["--baseline_only", "--dataset_json_folder", str(metadata),
                        "--base_n_epochs", "0", "--base_batch_size", "4",
                        "--n_epochs", "0", "--sampler_real_ratio", "1", "--output_dir", str(output)]) == 0
    root, = output.iterdir()
    assert len(calls) == 1
    training = json.loads((root / "G0/training/train_config.json").read_text())
    runtime = json.loads((root / "runtime_config.json").read_text())
    assert training["use_data_augmentation"] is True and runtime["use_data_augmentation"] is False
    report = json.loads((root / "G0/training/reproduction.json").read_text())
    assert report["video_auc_seven_mean"] == pytest.approx(.8)
    assert report["reproduction_status"] == "REFERENCE_NOT_PROVIDED"
    assert report["epoch_loop"]["training_passes"] == 1
    result = json.loads((root / "all_results.json").read_text())
    assert result["arms"] == {} and result["G0"]["status"] == "OK"
    assert result["G0"]["baseline_training"]["status"] == "OK"
    assert "video_auc_fullpath" in result["G0"]["datasets"]["DFDC"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["protocol"]["base_training"] is True
    assert manifest["base_state_unchanged"] and manifest["base_file_unchanged"]
    assert "experiments/e1010_baseline.py" in manifest["source_sha256"]


def test_e1010_trained_baseline_run_is_reusable_without_retraining(runner, fresh_training, tmp_path, monkeypatch):
    metadata, utilities, calls = fresh_training
    output = tmp_path / "first"
    assert runner.main(["--baseline_only", "--dataset_json_folder", str(metadata),
                        "--base_n_epochs", "0", "--base_batch_size", "4", "--output_dir", str(output)]) == 0
    root, = output.iterdir()
    monkeypatch.setattr(utilities, "train_model", lambda *a: pytest.fail("Existing B0 must be reused"))
    second = tmp_path / "second"
    assert runner.main(["--base_run", str(root), "--output_dir", str(second), "--arms", "G3_FL_MIL",
                        "--num_tokens", "2", "--memory_layer", "1", "--hidden_dim", "8",
                        "--num_heads", "4", "--depth", "1", "--n_epochs", "1"]) == 0
    second_root, = second.iterdir()
    report = json.loads((second_root / "all_results.json").read_text())
    assert report["arms"]["G3_FL_MIL"]["status"] == "OK"
    assert len(calls) == 1
    assert json.loads((second_root / "manifest.json").read_text())["protocol"]["base_training"] is False


def test_fresh_baseline_failure_stops_before_auxiliary_models(runner, fresh_training, tmp_path, monkeypatch):
    metadata, utilities, _ = fresh_training
    monkeypatch.setattr(utilities, "train_model", lambda *a: None)
    monkeypatch.setattr(runner, "make_train_loader", lambda *a: pytest.fail("B0 failure must stop auxiliary training"))
    output = tmp_path / "failed"
    assert runner.main(["--dataset_json_folder", str(metadata), "--base_n_epochs", "0",
                        "--base_batch_size", "4", "--arms", "G3_PLAIN", "--output_dir", str(output)]) == 1
    root, = output.iterdir()
    result = json.loads((root / "all_results.json").read_text())
    assert result["G0"]["status"] == "FAILED"
    assert result["arms"]["G3_PLAIN"]["status"] == "NOT_RUN"
    assert json.loads((root / "manifest.json").read_text())["status"] == "BASE_TRAIN_FAILED"


def test_fresh_preflight_checks_metadata_without_training(runner, fresh_training, monkeypatch):
    metadata, utilities, _ = fresh_training
    monkeypatch.setattr(utilities, "train_model", lambda *a: pytest.fail("Preflight cannot train"))
    assert runner.main(["--preflight", "--baseline_only", "--dataset_json_folder", str(metadata)]) == 0


def test_fresh_preflight_rejects_invalid_auxiliary_batch_before_training(runner, fresh_training, monkeypatch):
    metadata, utilities, _ = fresh_training
    monkeypatch.setattr(utilities, "train_model", lambda *a: pytest.fail("Invalid auxiliary setup must fail before B0"))
    with pytest.raises(SystemExit):
        runner.main(["--preflight", "--dataset_json_folder", str(metadata),
                     "--arms", "G3_FL_MIL", "--batch_size", "3"])


@pytest.mark.parametrize("arm", ["G3_PLAIN", "G4_TOPK"])
def test_fresh_baseline_and_following_group_run_in_one_invocation(runner, fresh_training, tmp_path, arm):
    metadata, _, calls = fresh_training
    output = tmp_path / "all_stages"
    assert runner.main(["--dataset_json_folder", str(metadata), "--base_n_epochs", "0",
                        "--base_batch_size", "4", "--output_dir", str(output), "--arms", arm,
                        "--n_epochs", "1", "--num_tokens", "2", "--memory_layer", "1",
                        "--hidden_dim", "8", "--num_heads", "4", "--depth", "1",
                        "--prototype_top_k", "2", "--prototype_pool_top_k", "2",
                        "--prototype_frames_per_class", "1"]) == 0
    root, = output.iterdir()
    results = json.loads((root / "all_results.json").read_text())
    assert results["G0"]["baseline_training"]["status"] == "OK"
    assert results["arms"][arm]["status"] == "OK"
    assert results["arms"][arm]["base_scores_bitwise_identical"]
    assert len(calls) == 1


def test_fresh_baseline_evaluation_failure_stops_following_groups(runner, fresh_training, tmp_path, monkeypatch):
    metadata, utilities, _ = fresh_training

    def fail(*args):
        raise RuntimeError("historical evaluation failed")

    monkeypatch.setattr(utilities, "evaluate_model", fail)
    output = tmp_path / "evaluation_failure"
    assert runner.main(["--dataset_json_folder", str(metadata), "--base_n_epochs", "0",
                        "--base_batch_size", "4", "--output_dir", str(output), "--arms", "G3_PLAIN"]) == 1
    root, = output.iterdir()
    assert json.loads((root / "manifest.json").read_text())["status"] == "BASE_EVAL_FAILED"
    assert json.loads((root / "all_results.json").read_text())["arms"]["G3_PLAIN"]["status"] == "NOT_RUN"

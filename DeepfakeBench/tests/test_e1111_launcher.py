"""Server-side scheduling contracts; mocked children never train a model."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/run_e1111.py"


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("e1111_launcher_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dry_run_is_stdlib_only_and_does_not_create_output(tmp_path):
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'transformers', 'experiment_utils'}:
        raise AssertionError('ML import in launcher dry run: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[2], '--dry_run', '--gpu_a', '0', '--gpu_b', '1', '--output_dir', sys.argv[3]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    destination = tmp_path / "unused"
    result = subprocess.run([sys.executable, "-S", "-c", code, str(SCRIPT.parent), str(SCRIPT),
                             str(destination)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schedule_after_B0"] == "parallel"
    assert plan["B0"]["action"] == "retrain once"
    assert plan["B0_command"][plan["B0_command"].index("--arms") + 1] == "B0"
    assert plan["B"]["command"][plan["B"]["command"].index("--arms") + 1] == "PUCE"
    assert plan["A"]["command"][plan["A"]["command"].index("--device") + 1] == "cuda:0"
    assert plan["A"]["command"][plan["A"]["command"].index("--statistic_mode") + 1] == "video_mean"
    assert not destination.exists()


def test_existing_baseline_reuse_checks_native_model_hash_and_conditions(launcher, tmp_path):
    args = launcher.build_parser().parse_args([])
    baseline_dir = tmp_path / "base"
    (baseline_dir / "B0").mkdir(parents=True)
    checkpoint = baseline_dir / "checkpoint.pth"
    checkpoint.write_bytes(b"native CE B0 fixture")
    metadata, images, clip = (tmp_path / name for name in ("metadata", "images", "clip"))
    for folder in (metadata, images, clip):
        folder.mkdir()
    (metadata / "FaceForensics++.json").write_text("{}")
    config = {"model_name": "effort", "manualSeed": 1024, "nEpochs": 10,
              "train_batchSize": 32, "sampler_real_ratio": .3, "compression": "c23",
              "frame_num": {"train": 8, "test": 8}, "dataset_json_folder": str(metadata),
              "rgb_root_override": str(images), "clip_pretrained_path": str(clip)}
    config_path = baseline_dir / "B0/train_config.json"
    config_path.write_text(json.dumps(config))
    baseline = {"status": "OK", "settings": {"model_name": "effort"}, "ckpt": str(checkpoint),
                "checkpoint_sha256": launcher.file_sha256(checkpoint)}
    report = {"experiment": "E1111", "branch": "E1111B", "arms": {"B0": baseline}}
    report_path = baseline_dir / "all_results.json"
    report_path.write_text(json.dumps(report))
    (baseline_dir / "manifest.json").write_text(json.dumps({
        "dataset_sha256": {"FaceForensics++": launcher.file_sha256(metadata / "FaceForensics++.json")}}))
    assert launcher.validate_base_run(baseline_dir, args) == baseline_dir.resolve()
    assert args.rgb_root == images.resolve() and args.clip_pretrained_path == clip.resolve()
    assert args.dataset_json_folder == metadata.resolve()
    mismatched = launcher.build_parser().parse_args(["--rgb_root", str(tmp_path / "different_images")])
    with pytest.raises(ValueError, match="rgb_root"):
        launcher.validate_base_run(baseline_dir, mismatched)
    config["frame_num"]["test"] = 32
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="frame_num"):
        launcher.validate_base_run(baseline_dir, args)
    config["frame_num"]["test"] = 8
    config_path.write_text(json.dumps(config))
    baseline["settings"]["model_name"] = "effort_e1111b"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="original"):
        launcher.validate_base_run(baseline_dir, args)
    baseline["settings"]["model_name"] = "effort"
    report_path.write_text(json.dumps(report))
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="SHA"):
        launcher.validate_base_run(baseline_dir, args)


def test_omitted_legacy_paths_are_read_from_matching_source_without_ml_import(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    owner = tmp_path / "owner.py"
    owner.write_text('path = config.get("clip_pretrained_path", "/old/clip")')
    manifest = {"source_sha256": {"owner.py": launcher.file_sha256(owner)}}
    assert launcher.owning_default("owner.py", "clip_pretrained_path", manifest) == "/old/clip"
    owner.write_text('path = config.get("clip_pretrained_path", "/changed/clip")')
    with pytest.raises(ValueError, match="source changed"):
        launcher.owning_default("owner.py", "clip_pretrained_path", manifest)


@pytest.mark.parametrize("separate_gpus", [False, True])
@pytest.mark.parametrize("a_exit_code", [0, 2])
def test_one_baseline_then_correct_gpu_schedule_and_independent_completion(
        launcher, tmp_path, monkeypatch, separate_gpus, a_exit_code):
    args = launcher.build_parser().parse_args([
        "--output_dir", str(tmp_path / "run"), "--gpu_a", "0",
        "--gpu_b", "1" if separate_gpus else "0"])
    started, finished, seen_environment = [], set(), {}
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    class Process:
        def __init__(self, command, **kwargs):
            arms = command[command.index("--arms") + 1] if "--arms" in command else None
            self.name = "B0" if arms == "B0" else "B" if arms == "PUCE" else "A"
            self.pid = 1000 + len(started)
            started.append(self.name)
            seen_environment[self.name] = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            assert kwargs["env"]["LOCAL_RANK"] == "0"
            assert kwargs["env"]["RANK"] == "0" and kwargs["env"]["WORLD_SIZE"] == "1"
            assert kwargs["cwd"] == launcher.ROOT
            assert kwargs["stdin"] == subprocess.DEVNULL
            if self.name != "B0":
                assert "B0" in finished
            if self.name == "B":
                assert ("A" in finished) is (not separate_gpus)
            folder = Path(command[command.index("--output_dir") + 1]) / "seed_fixture"
            folder.mkdir(parents=True)
            report = {"status": "S0_PAUSED_P_D7" if a_exit_code else "COMPLETE"} if self.name == "A" else {
                "arms": {"B0" if self.name == "B0" else "PUCE_P0.3_L0.01": {"status": "OK"}}}
            (folder / "all_results.json").write_text(json.dumps(report))
            if self.name == "A":
                assert Path(command[command.index("--base_run") + 1]).name == "seed_fixture"
                assert command[command.index("--device") + 1] == "cuda:0"

        def poll(self):
            finished.add(self.name)
            return a_exit_code if self.name == "A" else 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher, "validate_base_run", lambda path, args: Path(path))
    monkeypatch.setattr(launcher, "stop_process", lambda process: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda *args: signal.SIG_DFL)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    code = launcher.run(args)
    assert started == ["B0", "A", "B"]
    assert seen_environment == {"B0": str(args.gpu_b), "A": "0", "B": str(args.gpu_b)}
    assert code == (1 if a_exit_code else 0)
    summary, = (tmp_path / "run").glob("seed*/all_results.json")
    saved = json.loads(summary.read_text())
    assert saved["jobs"]["B"]["status"] == "OK"  # A's gate stop cannot cancel B
    assert saved["jobs"]["B"]["experiment_status"] == "OK"


def test_reuse_never_starts_another_baseline(launcher, tmp_path, monkeypatch):
    args = launcher.build_parser().parse_args([
        "--base_run", str(tmp_path / "previous"), "--output_dir", str(tmp_path / "run")])
    started = []

    class Process:
        def __init__(self, command, **kwargs):
            assert not ("--arms" in command and command[command.index("--arms") + 1] == "B0")
            started.append(command)
            self.pid = len(started)
            folder = Path(command[command.index("--output_dir") + 1]) / "seed_fixture"
            folder.mkdir(parents=True)
            report = {"arms": {"CE_L0": {"status": "OK"}}} if "--arms" in command else {
                "status": "COMPLETE"}
            (folder / "all_results.json").write_text(json.dumps(report))

        def poll(self):
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher, "validate_base_run", lambda path, args: Path(path))
    monkeypatch.setattr(launcher, "stop_process", lambda process: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda *args: signal.SIG_DFL)
    assert launcher.run(args) == 0
    assert len(started) == 2
    summary, = (tmp_path / "run").glob("seed*/all_results.json")
    assert json.loads(summary.read_text())["jobs"]["B0"]["status"] == "REUSED"


def test_baseline_failure_does_not_start_a_or_b(launcher, tmp_path, monkeypatch):
    args = launcher.build_parser().parse_args(["--output_dir", str(tmp_path)])
    called = []

    class Process:
        pid = 999

        def __init__(self, command, **kwargs):
            called.append(command)

        def poll(self):
            return 1

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher, "stop_process", lambda process: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda *args: signal.SIG_DFL)
    assert launcher.run(args) == 1
    assert len(called) == 1


def test_missing_branch_reports_cannot_be_reported_as_success(launcher, tmp_path, monkeypatch):
    args = launcher.build_parser().parse_args([
        "--base_run", str(tmp_path / "previous"), "--output_dir", str(tmp_path / "run")])

    class Process:
        pid = 999

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher, "validate_base_run", lambda path, args: Path(path))
    monkeypatch.setattr(launcher, "stop_process", lambda process: None)
    monkeypatch.setattr(launcher.signal, "signal", lambda *args: signal.SIG_DFL)
    assert launcher.run(args) == 1
    summary, = (tmp_path / "run").glob("seed*/all_results.json")
    assert json.loads(summary.read_text())["jobs"]["A"]["result_valid"] is False


def test_signal_interruption_stops_running_children(launcher, tmp_path, monkeypatch):
    args = launcher.build_parser().parse_args(["--output_dir", str(tmp_path)])
    stopped = []

    class Process:
        pid = 999

        def __init__(self, command, **kwargs):
            pass

        def poll(self):
            return None

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    monkeypatch.setattr(launcher.signal, "signal", lambda *args: signal.SIG_DFL)
    monkeypatch.setattr(launcher.time, "sleep", interrupt)
    monkeypatch.setattr(launcher, "stop_process", lambda process: stopped.append(process.pid))
    assert launcher.run(args) == 130
    assert stopped == [999]


@pytest.mark.skipif(os.name != "posix", reason="Server process groups are POSIX")
def test_group_cleanup_does_not_skip_descendants_when_leader_exited(launcher, monkeypatch):
    calls = []

    class Process:
        pid = 424242

        def poll(self):
            return 0

        def wait(self, timeout):
            return 0

    def killpg(pid, signum):
        calls.append((pid, signum))

    clock = iter([0., 11.])
    monkeypatch.setattr(launcher.os, "killpg", killpg)
    monkeypatch.setattr(launcher.time, "monotonic", lambda: next(clock))
    launcher.stop_process(Process())
    assert (424242, signal.SIGTERM) in calls
    assert (424242, signal.SIGKILL) in calls

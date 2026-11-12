"""Server-side relocation and sampled-file preflight regression tests."""

import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from e1111_paths import input_file, normalize_arguments, relocated


def test_exact_old_prefix_moves_and_other_roots_are_preserved(tmp_path):
    assert relocated("/home/user1/effort/data") == "/mnt/data2/effort/data"
    assert relocated("/home/user1/effort/effort_main/B0.pth") == "/mnt/data2/effort/effort_main/B0.pth"
    assert relocated("/home/user10/effort/data") == "/home/user10/effort/data"
    assert relocated("/custom/data") == "/custom/data"
    args = argparse.Namespace(rgb_root=None, dataset_json_folder=None, clip_pretrained_path=None)
    normalize_arguments(args, tmp_path)
    assert args.rgb_root == "/mnt/data2/effort/data"
    assert args.dataset_json_folder == tmp_path / "preprocessing/dataset_json"
    assert args.clip_pretrained_path == str(tmp_path / "training/models--openai--clip-vit-large-patch14")
    assert isinstance(args.rgb_root, str) and isinstance(args.clip_pretrained_path, str)


def test_explicit_and_saved_paths_relocate_without_modifying_source_config(tmp_path):
    old = {"dataset_json_folder": "/home/user1/effort/json",
           "rgb_root_override": "/home/user1/effort/data",
           "clip_pretrained_path": "/home/user1/effort/clip"}
    args = argparse.Namespace(rgb_root=None, dataset_json_folder=None, clip_pretrained_path=None,
                              base_run=Path("/home/user1/effort/run"))
    normalize_arguments(args, tmp_path, old)
    assert str(args.dataset_json_folder).replace("\\", "/") == "/mnt/data2/effort/json"
    assert args.rgb_root == "/mnt/data2/effort/data" and args.clip_pretrained_path == "/mnt/data2/effort/clip"
    assert str(args.base_run).replace("\\", "/") == "/mnt/data2/effort/run"
    assert old["rgb_root_override"] == "/home/user1/effort/data"
    args.rgb_root = "/another/images"
    normalize_arguments(args, tmp_path, old)
    assert args.rgb_root == "/another/images"


def test_obsolete_absolute_index_is_not_silently_mistaken_for_a_new_file(tmp_path):
    with pytest.raises(ValueError, match="obsolete absolute"):
        input_file("/home/user1/effort/data/FaceForensics++/000.png", tmp_path)
    assert input_file("FaceForensics++\\frames\\000.png", tmp_path) == tmp_path / "FaceForensics++/frames/000.png"


def test_actual_sample_list_is_checked_before_training_without_getitem(tmp_path, monkeypatch):
    script = ROOT / "experiments/run_e1111b.py"
    spec = importlib.util.spec_from_file_location("e1111_path_preflight_test", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    image = tmp_path / "FaceForensics++/frames/001/000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"file-existence fixture; preflight deliberately does not decode")

    class Dataset:
        image_list = ["FaceForensics++\\frames\\001\\000.png"]
        env = None

        def __getitem__(self, index):
            raise AssertionError("Preflight must not load or randomly substitute samples")

    utilities = SimpleNamespace(
        get_train_loader=lambda cfg: SimpleNamespace(dataset=Dataset()),
        get_data_loader=lambda cfg, name: SimpleNamespace(dataset=Dataset()))
    config = {"rgb_root_override": str(tmp_path), "lmdb": False}
    result = runner.check_input_files(config, config, utilities)
    assert result["unique_files_checked"] == 1
    image.unlink()
    with pytest.raises(FileNotFoundError, match="sampled image is missing"):
        runner.check_input_files(config, config, utilities)

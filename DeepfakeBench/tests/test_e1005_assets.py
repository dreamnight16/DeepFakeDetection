"""Conditional E1005 assets require real registration and native pixel detail."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_asset_test", ROOT / "experiments" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    data = load("e1005_data")
    labels = {"FF-real": 0, "FF-DF": 1, "FF-F2F": 1, "FF-FS": 1, "FF-NT": 1}
    metadata = {"FaceForensics++": {label: {"train": {"c23": {}}} for label in labels}}
    receipt = {"schema_version": 1, "receipt_id": "crop-mask-audit",
               "registered_masks": {}, "native_crops": {}}
    for index, method in enumerate(data.METHODS):
        target = f"{index:03}"
        for label, video in (("FF-real", target), ("FF-" + method, target + "_900")):
            frames = [f"rgb/{label}/{video}/{frame:03}.png" for frame in range(2)]
            metadata["FaceForensics++"][label]["train"]["c23"][video] = {"label": label, "frames": frames}
            for path in frames:
                stored = tmp_path / path
                stored.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), (120, 80, 30)).save(stored)
                native = tmp_path / ("native/" + path)
                native.parent.mkdir(parents=True, exist_ok=True)
                pixels = Image.new("RGB", (16, 16))
                pixels.putdata([(255 if (x + y) % 2 else 0, x * 10, y * 10)
                                for y in range(16) for x in range(16)])
                pixels.save(native)
                receipt["native_crops"][path] = {"path": "native/" + path,
                    "same_frame_verified": True, "crop_verified": True,
                    "native_verified": True, "evidence": "original RGB source crop checked"}
                if label != "FF-real":
                    mask_path = "mask/" + path
                    mask_file = tmp_path / mask_path
                    mask_file.parent.mkdir(parents=True, exist_ok=True)
                    mask = Image.new("L", (8, 8))
                    for x in range(2, 6):
                        for y in range(2, 6): mask.putpixel((x, y), 255)
                    mask.save(mask_file)
                    receipt["registered_masks"][path] = {"path": mask_path,
                        "coordinates_verified": True, "evidence": "same face crop coordinates checked"}
    records = data.build_manifest(metadata, "FaceForensics++", "train", labels)
    candidates = data.build_pair_candidates(records)
    pairs = [{**pair, "verified": True, "audit_receipt_id": "time-face-audit"}
             for pair in candidates["pairs"]]
    return data, pairs, receipt


def test_complete_registered_assets_are_audited_with_exact_file_identities(tmp_path):
    _, pairs, receipt = fixture(tmp_path)
    assets = load("e1005_assets")
    report = assets.audit_assets(receipt, pairs, {}, tmp_path, 8, 16)
    assert report["mask"]["eligible"] is True
    assert report["native"]["eligible"] is True
    assert report["mask"]["audited_frames"] == 8
    assert report["native"]["audited_frames"] == 16
    assert len(report["receipt_sha256"]) == 64
    assert all(len(row["sha256"]) == 64 for row in report["audited_identities"]["native"])


def test_absent_native_mapping_does_not_expand_missing_entries_over_the_full_panel(tmp_path):
    _, pairs, receipt = fixture(tmp_path)
    receipt["native_crops"] = {}
    evaluation = [{"frames": [f"unavailable/{index}.png" for index in range(1000)]}]
    report = load("e1005_assets").audit_assets(receipt, pairs, evaluation, tmp_path, 8, 16)
    assert report["mask"]["eligible"]
    assert not report["native"]["eligible"]
    assert len(report["native"]["failures"]) == 1


def test_grayscale_rgb_mask_is_valid_but_color_semantics_are_rejected(tmp_path):
    _, pairs, receipt = fixture(tmp_path)
    assets = load("e1005_assets")
    entry = next(iter(receipt["registered_masks"].values()))
    path = tmp_path / entry["path"]
    with Image.open(path) as image: image.convert("RGB").save(path)
    assert assets.audit_assets(receipt, pairs, {}, tmp_path, 8, 16)["mask"]["eligible"] is True
    Image.new("RGB", (8, 8), (255, 0, 0)).save(path)
    assert assets.audit_assets(receipt, pairs, {}, tmp_path, 8, 16)["mask"]["eligible"] is False


@pytest.mark.parametrize("mutation", ["registration", "missing", "corrupt", "mask_size", "zero", "pair_unverified"])
def test_mask_eligibility_rejects_missing_registration_decode_or_signal(tmp_path, mutation):
    _, pairs, receipt = fixture(tmp_path)
    entry = next(iter(receipt["registered_masks"].values()))
    if mutation == "registration": entry["coordinates_verified"] = False
    if mutation == "missing": (tmp_path / entry["path"]).unlink()
    if mutation == "corrupt": (tmp_path / entry["path"]).write_bytes(b"not a mask")
    if mutation == "mask_size": Image.new("L", (16, 16), 255).save(tmp_path / entry["path"])
    if mutation == "zero":
        for value in receipt["registered_masks"].values(): Image.new("L", (8, 8)).save(tmp_path / value["path"])
    if mutation == "pair_unverified": pairs[0]["verified"] = False
    report = load("e1005_assets").audit_assets(receipt, pairs, {}, tmp_path, 8, 16)
    assert report["mask"]["eligible"] is False
    assert report["mask"]["reason"]


@pytest.mark.parametrize("mutation", ["frame", "crop", "native", "small", "path_escape", "eval_missing"])
def test_native_eligibility_requires_all_train_and_eval_frames_at_actual_resolution(tmp_path, mutation):
    _, pairs, receipt = fixture(tmp_path)
    entry = next(iter(receipt["native_crops"].values()))
    eval_manifests = {}
    if mutation in {"frame", "crop", "native"}: entry[{"frame": "same_frame_verified", "crop": "crop_verified", "native": "native_verified"}[mutation]] = False
    if mutation == "small": Image.new("RGB", (8, 8)).save(tmp_path / entry["path"])
    if mutation == "path_escape": entry["path"] = "../outside.png"
    if mutation == "eval_missing": eval_manifests = {"dev": [{"video_id": "uncovered", "frames": ["missing/000.png"]}]}
    report = load("e1005_assets").audit_assets(receipt, pairs, eval_manifests, tmp_path, 8, 16)
    assert report["native"]["eligible"] is False
    assert report["native"]["reason"]


def test_mask_targets_are_train_only_aligned_real_zero_and_geometry_shared(tmp_path):
    data, pairs, receipt = fixture(tmp_path)
    assets = load("e1005_assets")
    episodes = data.build_episodes(pairs, steps=4)
    reader = data.make_reader(tmp_path, 8, backend="pil")
    dataset = assets.asset_pair_dataset(episodes, reader, [0] * 3, [1] * 3,
        "tamper", receipt, tmp_path, 8, 16)
    row = dataset[2]
    target = row["spatial_target"]
    assert target.shape == (8, 2, 2, 1, 8, 8)
    assert target[row["real_indices"]].sum() == 0
    assert target[row["fake_indices"], 0].sum() > 0
    assert not torch.equal(target[1, 0], target[1, 1])
    broken = copy.deepcopy(receipt)
    broken["registered_masks"] = {}
    with pytest.raises(ValueError, match="registered"):
        assets.asset_pair_dataset(episodes, reader, [0] * 3, [1] * 3,
            "tamper", broken, tmp_path, 8, 16)[0]


def test_boundary_and_shuffle_preserve_defined_spatial_controls(tmp_path):
    data, pairs, receipt = fixture(tmp_path)
    assets = load("e1005_assets")
    episodes = data.build_episodes(pairs, steps=1)
    reader = data.make_reader(tmp_path, 8, backend="pil")
    kwargs = (episodes, reader, [0] * 3, [1] * 3)
    tamper = assets.asset_pair_dataset(*kwargs, "tamper", receipt, tmp_path, 8, 16)[0]
    boundary = assets.asset_pair_dataset(*kwargs, "boundary", receipt, tmp_path, 8, 16)[0]
    shuffled = assets.asset_pair_dataset(*kwargs, "shuffled_mask", receipt, tmp_path, 8, 16)[0]
    mask = tamper["spatial_target"][1, 0, 0]
    expected = torch.nn.functional.max_pool2d(mask[None], 3, 1, 1) + torch.nn.functional.max_pool2d(-mask[None], 3, 1, 1)
    torch.testing.assert_close(boundary["spatial_target"][1, 0, 0], expected[0])
    torch.testing.assert_close(shuffled["spatial_target"].sum(dim=(-1, -2)), tamper["spatial_target"].sum(dim=(-1, -2)))
    assert not torch.equal(shuffled["spatial_target"], tamper["spatial_target"])
    repeated = assets.asset_pair_dataset(*kwargs, "shuffled_mask", receipt, tmp_path, 8, 16)[0]
    assert torch.equal(repeated["spatial_target"], shuffled["spatial_target"])


def test_native_pair_global_image_is_derived_from_same_native_view(tmp_path):
    data, pairs, receipt = fixture(tmp_path)
    assets = load("e1005_assets")
    episodes = data.build_episodes(pairs, steps=3)
    def forbidden_reader(path): raise AssertionError("native arms must use native crop assets")
    row = assets.asset_pair_dataset(episodes, forbidden_reader, [.5] * 3, [.25] * 3,
        "native448", receipt, tmp_path, 8, 16)[2]
    assert row["image"].shape == (8, 2, 2, 3, 8, 8)
    assert row["aux_image"].shape == (8, 2, 2, 3, 16, 16)
    expected = torch.nn.functional.interpolate(row["aux_image"].flatten(0, 2), (8, 8), mode="bilinear", align_corners=False).reshape(row["image"].shape)
    torch.testing.assert_close(row["image"], expected)
    interpolation = assets.asset_pair_dataset(episodes, forbidden_reader, [.5] * 3, [.25] * 3,
        "interpolated448", receipt, tmp_path, 8, 16)[2]
    assert torch.equal(interpolation["image"], row["image"])
    assert torch.equal(interpolation["aux_image"], row["aux_image"])


def test_video_mask_arm_never_reads_gt_and_native_padding_collates(tmp_path):
    data, pairs, receipt = fixture(tmp_path)
    assets = load("e1005_assets")
    records = [pair["fake"] for pair in pairs[:2]]
    reader = data.make_reader(tmp_path, 8, backend="pil")
    mask_receipt = {**receipt, "registered_masks": {}}
    ordinary = assets.asset_video_dataset(records, reader, 4, [0] * 3, [1] * 3,
        "tamper", mask_receipt, tmp_path, 8, 16)[0]
    assert "spatial_target" not in ordinary
    dataset = assets.asset_video_dataset(records, reader, 4, [0] * 3, [1] * 3,
        "native448", receipt, tmp_path, 8, 16)
    batch = assets.asset_collate_videos([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 4, 3, 8, 8)
    assert batch["aux_image"].shape == (2, 4, 3, 16, 16)
    assert batch["mask"].tolist() == [[True, True, False, False]] * 2
    assert batch["aux_image"][:, 2:].sum() == 0

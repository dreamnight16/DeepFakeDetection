"""Image-only E1005 audit establishes lineage and file integrity, not identity."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
PREPROCESS = ROOT / "preprocessing" / "preprocess.py"


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_image_audit_test",
        ROOT / "experiments" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, *, masks=False, landmarks=False):
    data = load("e1005_data")
    labels = {"FF-real": 0, **{"FF-" + name: 1 for name in data.METHODS}}
    metadata = {"FaceForensics++": {
        name: {"train": {"c23": {}}} for name in labels}}
    for number, method in enumerate(data.METHODS):
        target = f"{number:03d}"
        for label, video in (("FF-real", target), ("FF-" + method, target + "_900")):
            folder = f"FaceForensics++/{label}/c23/frames/{video}"
            frames = [folder + f"/{index:03d}.png" for index in (0, 7, 19)]
            record = {"label": label, "frames": frames}
            metadata["FaceForensics++"][label]["train"]["c23"][video] = record
            for path in frames:
                file = tmp_path / path
                file.parent.mkdir(parents=True, exist_ok=True)
                # A completely different fake face is valid; no pixel similarity gate.
                Image.new("RGB", (12, 12), (20, 30, 40) if label == "FF-real"
                          else (250, 180, 90)).save(file)
                if landmarks:
                    land = tmp_path / path.replace("/frames/", "/landmarks/")
                    land = land.with_suffix(".npy")
                    land.parent.mkdir(parents=True, exist_ok=True)
                    np.save(land, np.arange(162, dtype=float).reshape(81, 2))
                if masks and label != "FF-real":
                    mask = tmp_path / path.replace("/frames/", "/masks/")
                    mask.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L", (12, 12), 255).save(mask)
    records = data.build_manifest(metadata, "FaceForensics++", "train", labels)
    return data.build_pair_candidates(records)


def audit(candidates, root, progress=None):
    return load("e1005_image_audit").audit_preprocessed_pairs(
        candidates, root, PREPROCESS, progress)


def test_png_only_all_four_methods_are_eligible_without_identity_claims(tmp_path):
    result = audit(fixture(tmp_path), tmp_path)
    assert len(result["pairs"]) == 4
    assert {pair["method"] for pair in result["pairs"]} == {"DF", "F2F", "FS", "NT"}
    for pair in result["pairs"]:
        assert pair["verified"] is True
        assert pair["common_indices"] == [0, 7, 19]
        assert pair["pairing_mode"] == "ffpp_original_frame_index"
        assert pair["time_verified"] is False
        assert pair["target_face_verified"] is False
        assert pair["pixel_alignment_verified"] is False
        assert pair["audit_receipt_id"] == result["report"]["receipt_id"]
    report = result["report"]
    assert len(report["preprocessing"]["script_sha256"]) == 64
    assert report["counts"]["audited_rgb_files"] == 24
    assert report["landmarks"]["missing"] == 24
    assert result["asset_receipt"]["native_crops"] == {}
    assert result["asset_receipt"]["registered_masks"] == {}


def test_deterministic_audit_binds_content_and_does_not_mutate_candidates(tmp_path):
    candidates = fixture(tmp_path, masks=True, landmarks=True)
    before = copy.deepcopy(candidates)
    first = audit(candidates, tmp_path)
    assert first == audit(candidates, tmp_path)
    assert candidates == before
    path = tmp_path / candidates["pairs"][0]["fake"]["all_frames"][0]
    Image.new("RGB", (12, 12), (3, 4, 5)).save(path)
    changed = audit(candidates, tmp_path)
    assert changed["report"]["receipt_id"] != first["report"]["receipt_id"]
    assert changed["asset_receipt"]["receipt_id"] != first["asset_receipt"]["receipt_id"]


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_bad_frame_excludes_only_its_index_and_insufficient_pair(tmp_path, failure):
    candidates = fixture(tmp_path)
    paths = candidates["pairs"][0]["fake"]["all_frames"]
    for path in paths[:1]:
        if failure == "missing": (tmp_path / path).unlink()
        else: (tmp_path / path).write_bytes(b"broken png")
    result = audit(candidates, tmp_path)
    assert len(result["pairs"]) == 4
    affected = next(pair for pair in result["pairs"] if pair["pair_id"] == candidates["pairs"][0]["pair_id"])
    assert affected["common_indices"] == [7, 19]
    assert result["report"]["frame_failures"][0]["index"] == 0
    (tmp_path / paths[1]).unlink()
    result = audit(candidates, tmp_path)
    assert len(result["pairs"]) == 3
    assert result["report"]["excluded_pairs"][0]["reason"] == "insufficient_decodable_common_indices"


@pytest.mark.parametrize("unsafe", ["../outside/000.png", "symlink/000.png"])
def test_root_containment_excludes_unsafe_frame(tmp_path, unsafe):
    candidates = fixture(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    Image.new("RGB", (12, 12)).save(outside / "000.png")
    if unsafe.startswith("symlink"):
        (tmp_path / "symlink").symlink_to(outside, target_is_directory=True)
    candidates["pairs"][0]["fake"]["all_frames"][0] = unsafe
    result = audit(candidates, tmp_path)
    assert result["report"]["frame_failures"]
    assert any("containment" in row["reason"] for row in result["report"]["frame_failures"])


def test_registered_mask_and_landmarks_derive_same_preprocessing_paths(tmp_path):
    result = audit(fixture(tmp_path, masks=True, landmarks=True), tmp_path)
    receipt = result["asset_receipt"]
    assert receipt["schema_version"] == 1
    assert len(receipt["registered_masks"]) == 12
    assert result["report"]["landmarks"]["valid"] == 24
    for frame, entry in receipt["registered_masks"].items():
        assert entry["path"] == frame.replace("/frames/", "/masks/")
        assert entry["coordinates_verified"] is True
        assert entry["verification_mode"] == "preprocessing_contract"
        assert len(entry["sha256"]) == 64
    assets = load("e1005_assets")
    report = assets.audit_assets(receipt, result["pairs"], {}, tmp_path, 8, 16)
    assert report["mask"]["eligible"] is True
    assert report["native"]["eligible"] is False


@pytest.mark.parametrize("problem", ["path", "dimensions", "color", "corrupt"])
def test_invalid_masks_warn_without_excluding_rgb_pairs(tmp_path, problem):
    candidates = fixture(tmp_path, masks=True)
    fake = candidates["pairs"][0]["fake"]
    frame = fake["all_frames"][0]
    mask_path = frame.replace("/frames/", "/masks/")
    if problem == "path": fake["all_masks"][0] = fake["all_frames"][1].replace("/frames/", "/masks/")
    if problem == "dimensions": Image.new("L", (8, 8), 255).save(tmp_path / mask_path)
    if problem == "color": Image.new("RGB", (12, 12), (255, 0, 0)).save(tmp_path / mask_path)
    if problem == "corrupt": (tmp_path / mask_path).write_bytes(b"broken mask")
    result = audit(candidates, tmp_path)
    assert len(result["pairs"]) == 4
    assert frame not in result["asset_receipt"]["registered_masks"]
    assert result["report"]["masks"]["invalid"] == 1
    assert result["report"]["warnings"]


def test_binary01_and_empty_masks_retain_their_actual_encoding(tmp_path):
    candidates = fixture(tmp_path, masks=True)
    fake = candidates["pairs"][0]["fake"]
    for frame, value in zip(fake["all_frames"][:2], (1, 0)):
        Image.new("L", (12, 12), value).save(tmp_path / frame.replace("/frames/", "/masks/"))
    result = audit(candidates, tmp_path)
    entries = result["asset_receipt"]["registered_masks"]
    assert entries[fake["all_frames"][0]]["mask_encoding"] == "binary_01"
    assert entries[fake["all_frames"][0]]["positive_fraction"] == 1
    assert entries[fake["all_frames"][1]]["positive_fraction"] == 0


@pytest.mark.parametrize("problem", ["shape", "nan", "object", "corrupt"])
def test_optional_invalid_landmarks_do_not_claim_or_block_identity(tmp_path, problem):
    candidates = fixture(tmp_path, landmarks=True)
    frame = candidates["pairs"][0]["fake"]["all_frames"][0]
    land = (tmp_path / frame.replace("/frames/", "/landmarks/")).with_suffix(".npy")
    if problem == "shape": np.save(land, np.ones((2, 2)))
    if problem == "nan": np.save(land, np.full((81, 2), np.nan))
    if problem == "object": np.save(land, np.array([[object(), object()]] * 81))
    if problem == "corrupt": land.write_bytes(b"broken npy")
    result = audit(candidates, tmp_path)
    assert len(result["pairs"]) == 4
    assert result["report"]["landmarks"]["invalid"] == 1
    assert result["report"]["warnings"]


@pytest.mark.parametrize("mutation", ["method", "target", "source", "index"])
def test_lineage_and_numeric_index_are_not_inferred_from_inconsistent_rows(tmp_path, mutation):
    candidates = fixture(tmp_path)
    pair = candidates["pairs"][0]
    if mutation == "method": pair["method"] = "unknown"
    if mutation == "target": pair["fake"]["target_id"] = "900"
    if mutation == "source": pair["fake"]["source_id"] = "901"
    if mutation == "index": pair["fake"]["all_indices"][0] = 1
    result = audit(candidates, tmp_path)
    assert len(result["pairs"]) == 3
    assert result["report"]["excluded_pairs"]


def test_progress_first_every50_and_last_and_shared_real_decode_cache(tmp_path, monkeypatch):
    candidates = fixture(tmp_path)
    original = copy.deepcopy(candidates["pairs"][0])
    for index in range(51):
        extra = copy.deepcopy(original)
        extra["pair_id"] = f"additional-{index:03d}"
        candidates["pairs"].append(extra)
    calls = []
    real_open = Image.open
    def track(*args, **kwargs):
        calls.append(str(args[0]))
        return real_open(*args, **kwargs)
    monkeypatch.setattr(Image, "open", track)
    progress = []
    result = audit(candidates, tmp_path, progress.append)
    assert [event["processed_pairs"] for event in progress] == [1, 50, 55]
    assert result["report"]["counts"]["audited_rgb_files"] == 24
    assert len(calls) == 24


def test_preprocessing_script_requires_original_index_save_contract(tmp_path):
    candidates = fixture(tmp_path)
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("print('not preprocessing')\n")
    with pytest.raises(ValueError, match="contract"):
        load("e1005_image_audit").audit_preprocessed_pairs(candidates, tmp_path, unrelated)


def test_same_numeric_filename_from_a_different_video_is_excluded(tmp_path):
    candidates = fixture(tmp_path)
    affected = candidates["pairs"][0]
    affected["fake"]["all_frames"][0] = candidates["pairs"][1]["fake"]["all_frames"][0]
    result = audit(candidates, tmp_path)
    pair = next(pair for pair in result["pairs"] if pair["pair_id"] == affected["pair_id"])
    assert pair["common_indices"] == [7, 19]
    assert "video identity" in result["report"]["frame_failures"][0]["reason"]


def test_npy_extension_with_npz_bytes_warns_without_breaking_png_audit(tmp_path):
    candidates = fixture(tmp_path, landmarks=True)
    frame = candidates["pairs"][0]["fake"]["all_frames"][0]
    land = (tmp_path / frame.replace("/frames/", "/landmarks/")).with_suffix(".npy")
    with land.open("wb") as handle:
        np.savez(handle, points=np.ones((81, 2)))
    result = audit(candidates, tmp_path)
    assert len(result["pairs"]) == 4
    assert result["report"]["landmarks"]["invalid"] == 1


def test_all_empty_masks_are_recorded_and_asset_eligibility_remains_honest(tmp_path):
    candidates = fixture(tmp_path, masks=True)
    for pair in candidates["pairs"]:
        for frame in pair["fake"]["all_frames"]:
            Image.new("L", (12, 12), 0).save(tmp_path / frame.replace("/frames/", "/masks/"))
    result = audit(candidates, tmp_path)
    assert len(result["asset_receipt"]["registered_masks"]) == 12
    asset_report = load("e1005_assets").audit_assets(result["asset_receipt"], result["pairs"], {}, tmp_path, 8, 16)
    assert asset_report["mask"]["eligible"] is False
    assert "empty" in asset_report["mask"]["reason"]


def test_changed_mask_transform_contract_blocks_auto_registration_only(tmp_path):
    candidates = fixture(tmp_path, masks=True)
    changed_script = tmp_path / "changed_preprocess.py"
    changed_script.write_text(PREPROCESS.read_text().replace(
        "cv2.warpAffine(mask, M,", "cv2.warpAffine(mask, another_transform,"))
    result = load("e1005_image_audit").audit_preprocessed_pairs(candidates, tmp_path, changed_script)
    assert len(result["pairs"]) == 4
    assert result["report"]["preprocessing"]["mask_contract_checked"] is False
    assert result["asset_receipt"]["registered_masks"] == {}
    assert result["report"]["warnings"]


@pytest.mark.parametrize("kind", ["rgb", "mask", "landmarks"])
def test_corrupt_optional_and_rgb_files_have_stable_error_receipts(tmp_path, kind):
    candidates = fixture(tmp_path, masks=True, landmarks=True)
    frame = candidates["pairs"][0]["fake"]["all_frames"][0]
    if kind == "rgb": path = tmp_path / frame
    elif kind == "mask": path = tmp_path / frame.replace("/frames/", "/masks/")
    else: path = (tmp_path / frame.replace("/frames/", "/landmarks/")).with_suffix(".npy")
    path.write_bytes(b"corrupt file bytes")
    first = audit(candidates, tmp_path)
    second = audit(candidates, tmp_path)
    assert first == second
    assert first["report"]["receipt_id"] == second["report"]["receipt_id"]
    assert first["asset_receipt"]["receipt_id"] == second["asset_receipt"]["receipt_id"]

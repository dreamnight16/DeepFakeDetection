"""Strict full-metadata, audited pairing, and E1005 episode contracts."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location("e1005_data_test", ROOT / "experiments/e1005_data.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


LABELS = {"FF-real": 0, "FF-DF": 1, "FF-F2F": 2, "FF-FS": 3, "FF-NT": 4}


def ff_metadata(indices=range(10)):
    root = {label: {"train": {"c23": {}}} for label in LABELS}
    for i, label in enumerate(LABELS):
        video = "001" if i == 0 else f"00{i}_900"
        path = f"FaceForensics++/{label}/c23/frames/{video}"
        root[label]["train"]["c23"][video] = {
            "label": label, "frames": [f"{path}/{idx:03}.png" for idx in indices]}
    return {"FaceForensics++": root}


def records(indices=range(10)):
    return module().build_manifest(ff_metadata(indices), "FaceForensics++", "train", LABELS)


def fixture_pairs():
    data = ff_metadata()
    root = data["FaceForensics++"]
    for i in range(2, 5):
        path = f"FaceForensics++/FF-real/c23/frames/00{i}"
        root["FF-real"]["train"]["c23"][f"00{i}"] = {
            "label": "FF-real", "frames": [f"{path}/{idx:03}.png" for idx in range(10)]}
    m = module()
    candidates = m.build_pair_candidates(m.build_manifest(data, "FaceForensics++", "train", LABELS))
    receipt = {"schema_version": 1, "receipt_id": "human-time-face-audit-1", "pairs": []}
    for pair in candidates["pairs"]:
        receipt["pairs"].append({"pair_id": pair["pair_id"],
            "real_video_id": pair["real"]["video_id"], "fake_video_id": pair["fake"]["video_id"],
            "target_id": pair["fake"]["target_id"], "source_id": pair["fake"]["source_id"],
            "common_indices": pair["common_indices"], "time_verified": True,
            "target_face_verified": True, "evidence": "original timestamps and face track checked"})
    return m, candidates, receipt


def test_metadata_module_import_requires_no_ml_packages():
    code = f"import runpy,sys; runpy.run_path({str(ROOT / 'experiments/e1005_data.py')!r}); assert 'torch' not in sys.modules; assert 'PIL' not in sys.modules"
    result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_uniform_sampling_uses_full_sorted_metadata_and_half_up_rank():
    m = module()
    data = ff_metadata([9, 1, 7, 4, 0, 8, 6, 3, 2, 5])
    result = m.build_manifest(data, "FaceForensics++", "train", LABELS, frames=3)
    assert result[0]["all_indices"] == list(range(10))
    assert result[0]["indices"] == [0, 5, 9]
    assert all("\\" not in path for row in result for path in row["frames"])
    assert result[0]["video_id"] == "FaceForensics++/FF-DF/c23/frames/001_900"
    assert result[0]["target_id"] == "001"
    assert result[0]["source_id"] == "900"


def test_legacy_prefix_preserves_metadata_order_and_short_videos_are_not_repeated():
    m = module()
    result = m.build_manifest(ff_metadata([9, 1, 7]), "FaceForensics++", "train", LABELS,
                              sampling="legacy_prefix8", frames=8)
    assert result[0]["indices"] == [9, 1, 7]
    assert len(result[0]["frames"]) == 3


@pytest.mark.parametrize("mutation,match", [
    ("duplicate_path", "Duplicate frame path"),
    ("duplicate_index", "Duplicate numeric frame"),
    ("missing_label", "label_dict"),
    ("label_conflict", "label"),
    ("split_leak", "source.*overlap"),
    ("path_leak", "path.*overlap"),
])
def test_full_metadata_audit_rejects_identity_errors_before_sampling(mutation, match):
    m = module()
    data, labels = ff_metadata(), dict(LABELS)
    root = data["FaceForensics++"]
    row = root["FF-DF"]["train"]["c23"]["001_900"]
    if mutation == "duplicate_path": row["frames"].append(row["frames"][-1])
    if mutation == "duplicate_index": row["frames"].append(row["frames"][-1].replace("009.png", "frame_009.jpg"))
    if mutation == "missing_label": labels.pop("FF-DF")
    if mutation == "label_conflict": row["label"] = "FF-real"
    if mutation == "split_leak": root["FF-real"]["test"] = {"c23": {"900": {"frames": ["test/frames/900/000.png"]}}}
    if mutation == "path_leak": root["FF-real"]["test"] = {"c23": {"777": {"frames": [row["frames"][0]]}}}
    with pytest.raises(ValueError, match=match):
        m.build_manifest(data, "FaceForensics++", "train", labels, frames=2)


def test_flat_test_schema_and_nested_dfd_compression_are_both_supported():
    m = module()
    info = {"label": "DFDC_Real", "frames": ["DFDC/test/frames/a/010.png", "DFDC/test/frames/a/002.png"]}
    data = {"DFDC": {"DFDC_Real": {"test": {"a": info}}}}
    assert m.build_manifest(data, "DFDC", "test", {"DFDC_Real": 0})[0]["indices"] == [2, 10]
    info = {"label": "DFD_fake", "frames": ["FaceForensics++/DFD/frames/a/000.png"]}
    data = {"DeepFakeDetection": {"DFD_fake": {"test": {"c23": {"a": info}}}}}
    assert m.build_manifest(data, "DeepFakeDetection", "test", {"DFD_fake": 1})[0]["label"] == 1
    with pytest.raises(ValueError, match="compression"):
        m.build_manifest(data, "DeepFakeDetection", "test", {"DFD_fake": 1}, compression="raw")


def test_malformed_flat_split_reports_schema_error_without_attribute_error():
    m = module()
    info = {"label": "DFDC_Real", "frames": ["DFDC/test/frames/a/000.png"]}
    data = {"DFDC": {"DFDC_Real": {"val": info, "test": {"a": info}}}}
    with pytest.raises(ValueError, match="video.*schema"):
        m.build_manifest(data, "DFDC", "test", {"DFDC_Real": 0})


def test_explicit_external_role_scope_records_mirrors_without_using_their_labels():
    m = module()
    info = {"label": "DFDC_Real", "frames": ["DFDC/test/frames/a/000.png"]}
    data = {"DFDC": {"DFDC_Real": {"val": copy.deepcopy(info), "test": {"a": info}}}}
    result = m.build_manifest(data, "DFDC", "test", {"DFDC_Real": 0}, role_scope="selected_split")
    assert result[0]["role_scope"] == "selected_split"
    assert len(result[0]["metadata_sha256"]) == 64
    changed = copy.deepcopy(data)
    changed["DFDC"]["DFDC_Real"]["val"]["label"] = "unused_unknown_label"
    assert m.build_manifest(changed, "DFDC", "test", {"DFDC_Real": 0}, role_scope="selected_split")[0]["label"] == 0
    with pytest.raises(ValueError, match="FF.*all.*split"):
        m.build_manifest(ff_metadata(), "FaceForensics++", "train", LABELS, role_scope="selected_split")


def test_pair_candidates_use_official_target_order_and_report_exclusions():
    m = module()
    result = m.build_pair_candidates(records())
    assert len(result["pairs"]) == 1
    pair = result["pairs"][0]
    assert pair["real"]["target_id"] == pair["fake"]["target_id"] == "001"
    assert pair["common_indices"] == list(range(10))
    assert len(result["exclusions"]) == 3
    assert all(row["reason"] == "missing_target_real" for row in result["exclusions"])
    assert result["coverage"]["DF"] == {"candidates": 1, "included": 1, "excluded": 0}
    with pytest.raises(ValueError, match="train"):
        m.build_pair_candidates([{**row, "split": "test"} for row in records()])


def test_verified_pairs_need_an_explicit_matching_time_and_face_receipt():
    m, candidates, receipt = fixture_pairs()
    assert len(m.verified_pairs(candidates, receipt)) == 4
    assert m.verified_pairs(candidates, {**receipt, "pairs": []}) == []
    for key, value in (("time_verified", False), ("target_face_verified", False),
                       ("fake_video_id", "another/video"), ("target_id", "900"),
                       ("common_indices", [0, 999]), ("evidence", "")):
        broken = copy.deepcopy(receipt)
        broken["pairs"][0][key] = value
        with pytest.raises(ValueError, match="audit"):
            m.verified_pairs(candidates, broken)
    with pytest.raises(ValueError, match="receipt"):
        m.verified_pairs(candidates, {"pairs": receipt["pairs"]})


def test_episodes_are_deterministic_distinct_balanced_and_cycle_full_intersections():
    m, candidates, receipt = fixture_pairs()
    pairs = m.verified_pairs(candidates, receipt)
    episodes = m.build_episodes(pairs, steps=40, seed=1024)
    assert episodes == m.build_episodes(pairs, steps=40, seed=1024)
    observed = {pair["pair_id"]: set() for pair in pairs}
    for episode in episodes:
        assert len({pair["fake"]["target_id"] for pair in episode["pairs"]}) == 4
        assert [pair["method"] for pair in episode["pairs"]] == ["DF", "F2F", "FS", "NT"]
        assert sorted(episode["shuffle_indices"]) == [0, 1, 2, 3]
        assert all(i != j for i, j in enumerate(episode["shuffle_indices"]))
        for pair in episode["pairs"]:
            assert len(pair["indices"]) == 2
            assert pair["indices"][0] in range(5) and pair["indices"][1] in range(5, 10)
            observed[pair["pair_id"]].update(pair["indices"])
    assert all(value == set(range(10)) for value in observed.values())
    with pytest.raises(ValueError, match="verified"):
        m.build_episodes(candidates["pairs"], steps=1)
    with pytest.raises(ValueError, match="distinct"):
        impossible = copy.deepcopy(pairs)
        for pair in impossible: pair["fake"]["target_id"] = "001"
        m.build_episodes(impossible, steps=1)


def test_video_dataset_pads_without_repeating_and_collates_ids():
    torch = pytest.importorskip("torch")
    m = module()
    source = records([0, 5])
    reader = lambda path: torch.full((3, 6, 6), .5)
    dataset = m.VideoDataset(source, reader, frames=8)
    row = dataset[0]
    assert row["image"].shape == (8, 3, 6, 6)
    assert row["mask"].tolist() == [True, True] + [False] * 6
    assert torch.count_nonzero(row["image"][2:]) == 0
    batch = m.collate_videos([row, dataset[1]])
    assert batch["image"].shape == (2, 8, 3, 6, 6)
    assert batch["label"].tolist() == [1, 1]
    assert len(batch["paths"]) == 2
    with pytest.raises(FileNotFoundError):
        m.VideoDataset(source, lambda path: (_ for _ in ()).throw(FileNotFoundError(path)))[0]
    with pytest.raises(ValueError, match="RGB"):
        m.VideoDataset(source, lambda path: torch.full((3, 6, 6), float("nan")))[0]


def test_pair_tensor_budget_common_nuisance_normalization_and_c2_class_sharing():
    torch = pytest.importorskip("torch")
    m, candidates, receipt = fixture_pairs()
    episodes = m.build_episodes(m.verified_pairs(candidates, receipt), steps=4)
    reader = lambda path: torch.full((3, 8, 8), .5)
    dataset = m.PairDataset(episodes, reader, [0, 0, 0], [1, 1, 1])
    row = dataset[3]  # photometry condition, where independent parameters are observable.
    assert row["image"].shape == (8, 2, 2, 3, 8, 8)
    assert row["valid_mask"].all() and row["valid_mask"].shape == (8, 2, 2)
    assert row["labels"].tolist() == [0, 1] * 4
    assert row["methods"].tolist() == [-1, 0, -1, 1, -1, 2, -1, 3]
    assert row["real_indices"].tolist() == [0, 2, 4, 6]
    assert row["fake_indices"].tolist() == [1, 3, 5, 7]
    assert torch.equal(row["image"][0, 1], row["image"][7, 1])
    normalized = m.PairDataset(episodes, reader, [.5] * 3, [.25] * 3)[3]
    torch.testing.assert_close(normalized["image"], (row["image"] - .5) / .25)
    c2 = m.PairDataset(episodes, reader, [0] * 3, [1] * 3, independent_nuisance=True)[3]
    assert torch.equal(c2["image"][0, 1], c2["image"][6, 1])
    assert torch.equal(c2["image"][1, 1], c2["image"][7, 1])
    assert not torch.equal(c2["image"][0, 1], c2["image"][1, 1])
    assert m.collate_pairs([row])["image"].shape == row["image"].shape
    with pytest.raises(ValueError, match="batch_size"):
        m.collate_pairs([row, row])


@pytest.mark.parametrize("mutation", ["method_order", "audit_indices", "repeated_target", "shuffle", "label"])
def test_pair_dataset_revalidates_materialized_episode_contract(mutation):
    torch = pytest.importorskip("torch")
    m, candidates, receipt = fixture_pairs()
    episodes = m.build_episodes(m.verified_pairs(candidates, receipt), steps=1)
    episode = episodes[0]
    if mutation == "method_order": episode["pairs"].reverse()
    if mutation == "audit_indices": episode["pairs"][0]["common_indices"] = [0, 1]
    if mutation == "repeated_target": episode["pairs"][1]["fake"]["target_id"] = "001"
    if mutation == "shuffle": episode["shuffle_indices"] = [0, 1, 2, 3]
    if mutation == "label": episode["pairs"][0]["fake"]["label"] = 0
    with pytest.raises(ValueError, match="episode"):
        m.PairDataset(episodes, lambda path: torch.zeros(3, 8, 8), [0] * 3, [1] * 3)[0]


def test_reader_contains_paths_and_strictly_decodes_rgb_and_masks(tmp_path):
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    m = module()
    (tmp_path / "frames/v").mkdir(parents=True)
    path = tmp_path / "frames/v/000.png"
    Image.new("RGB", (5, 5), (255, 0, 0)).save(path)
    reader = m.make_reader(tmp_path, 8)
    image = reader("./frames\\v\\000.png")
    assert image.shape == (3, 8, 8) and image.dtype == torch.float32
    assert image[0].mean() == 1 and image[1:].sum() == 0
    with pytest.raises(ValueError, match="containment"):
        reader("../outside.png")
    with pytest.raises(FileNotFoundError):
        reader("frames/v/missing.png")
    masks = m.make_mask_reader(tmp_path, 8)
    with pytest.raises(FileNotFoundError):
        masks("masks/v/missing.png")
    path.write_text("not image")
    with pytest.raises(Exception):
        reader("frames/v/000.png")

"""Content pairing contracts; real tiny images, no model and no torch."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))

SIZE = 224
FRAME_COUNT = 8
FACE = (slice(60, 170), slice(80, 144))
FACE_BOX = (slice(38, 190), slice(58, 166))


def _scene(tone):
    import numpy as np

    rows = np.linspace(tone, min(tone + 40, 255), SIZE).astype(np.uint8)
    base = np.repeat(rows[:, None], SIZE, axis=1)
    return np.stack([base, base, base], axis=-1)


def _with_face(image, delta):
    import numpy as np

    out = image.copy()
    out[FACE] = np.clip(out[FACE].astype(int) + delta, 0, 255).astype(np.uint8)
    return out


def _mask(value, region=FACE):
    import numpy as np

    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[region] = value
    return mask


def _write(root, relative, array):
    import cv2

    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise RuntimeError(f"cannot write fixture image: {relative}")


def fake_video(root, prefix, tone=10, delta=60, start=0, masks="face", count=FRAME_COUNT):
    """A fake whose background copies ``tone`` and whose face region is edited."""
    frames, mask_paths = [], []
    for i in range(count):
        index = start + i
        rel = f"{prefix}/{index:03d}.png"
        _write(root, rel, _with_face(_scene(tone), delta))
        frames.append(rel)
        if masks == "face":
            mrel = f"{prefix}/mask_{index:03d}.png"
            _write(root, mrel, _mask(255))
            mask_paths.append(mrel)
    entry = {"frames": frames}
    if masks == "face":
        entry["masks"] = mask_paths
    return entry


def real_video(root, prefix, tone, deltas=(0,) * FRAME_COUNT, count=FRAME_COUNT):
    frames = []
    for i in range(count):
        rel = f"{prefix}/{i:03d}.png"
        _write(root, rel, _with_face(_scene(tone), deltas[i]))
        frames.append(rel)
    return {"frames": frames}


def raw_video(root, prefix, images):
    frames = []
    for i, image in enumerate(images):
        rel = f"{prefix}/{i:03d}.png"
        _write(root, rel, image)
        frames.append(rel)
    return {"frames": frames}


class PairingContracts(unittest.TestCase):
    def test_sample_count_is_clamped_to_the_declared_window(self):
        from g21.pairing import PairingConfig

        self.assertEqual(PairingConfig(sample_frames=100).sample_frames, 16)
        self.assertEqual(PairingConfig(sample_frames=1).sample_frames, 8)
        self.assertEqual(PairingConfig(sample_frames=12).sample_frames, 12)

    def test_evidence_support_is_the_same_index_intersection(self):
        """Evidence is measured where the builder will train: identical indices."""
        from g21.pairing import intersect_indices, subsample

        self.assertEqual(intersect_indices([0, 10, 20], [0, 20, 30]), [0, 20])
        self.assertEqual(intersect_indices([0, 1], [2, 3]), [])
        self.assertEqual(subsample([0, 1, 2, 3], 10), [0, 1, 2, 3])
        self.assertEqual(len(subsample(list(range(100)), 12)), 12)
        self.assertEqual(subsample(list(range(100)), 12), subsample(list(range(100)), 12))

    def test_background_match_verifies_the_matching_candidate(self):
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60)
            a = real_video(root, "real/001", tone=10)
            b = real_video(root, "real/002", tone=210)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertEqual(record["decision"], "verified")
        self.assertTrue(record["verified"])
        self.assertEqual(record["reference_component"], 0)
        self.assertLess(record["d_A"], record["d_B"])
        self.assertEqual(record["win_rate"], 1.0)
        self.assertGreaterEqual(record["margin_ratio"], 0.5)
        self.assertLessEqual(record["d_A"], 12.0)
        self.assertTrue(record["timelines_match"])
        self.assertEqual(record["mask_modes"], ["manipulation_mask"])

    def test_indistinguishable_candidates_are_ambiguous(self):
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60)
            a = real_video(root, "real/001", tone=10)
            b = real_video(root, "real/002", tone=10)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertEqual(record["decision"], "ambiguous_content")
        self.assertFalse(record["verified"])
        self.assertIsNone(record["reference_component"])

    def test_both_candidates_far_apart_is_not_closer(self):
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = raw_video(root, "fake", [_scene(0)] * FRAME_COUNT)
            a = raw_video(root, "real/001", [_scene(100)] * FRAME_COUNT)
            b = raw_video(root, "real/002", [_scene(255)] * FRAME_COUNT)
            blank = _mask(0)
            for entry in (fake, a, b):
                entry["masks"] = []
                for rel in entry["frames"]:
                    mrel = rel.replace(".png", "_mask.png")
                    _write(root, mrel, blank)
                    entry["masks"].append(mrel)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertEqual(record["decision"], "not_closer")
        self.assertGreater(record["d_A"], 12.0)

    def test_small_median_gap_is_weak_margin(self):
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = raw_video(root, "fake", [_scene(0)] * FRAME_COUNT)
            a = raw_video(root, "real/001", [_scene(10)] * FRAME_COUNT)
            b = raw_video(root, "real/002", [_scene(14)] * FRAME_COUNT)
            blank = _mask(0)
            for entry in (fake, a, b):
                entry["masks"] = []
                for rel in entry["frames"]:
                    mrel = rel.replace(".png", "_mask.png")
                    _write(root, mrel, blank)
                    entry["masks"].append(mrel)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertEqual(record["decision"], "weak_margin")
        self.assertLess(record["margin_ratio"], 0.5)
        self.assertLessEqual(record["d_A"], 12.0)

    def test_disjoint_timelines_can_never_confirm_a_candidate(self):
        """A matching background on a disjoint grid has no evidence to stand on."""
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60, start=0)
            a = real_video(root, "real/001", tone=10)
            a["frames"] = [
                rel.replace("/%03d" % i, "/%03d" % (100 + i))
                for i, rel in enumerate(a["frames"])
            ]
            for i in range(FRAME_COUNT):
                _write(root, f"real/001/{100 + i:03d}.png", _scene(10))
            b = real_video(root, "real/002", tone=210)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertFalse(record["verified"])
        self.assertIsNone(record["reference_component"])
        self.assertEqual(record["intersection_A"], 0)
        self.assertEqual(record["measured_A"], 0)

    def test_incompatible_timeline_excludes_an_otherwise_verified_winner(self):
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60, count=16)
            # A carries the fake's background, but only every second frame was
            # extracted, so half the moments have no counterpart.
            a_frames = []
            for i in range(0, 16, 2):
                rel = f"real/001/{i:03d}.png"
                _write(root, rel, _scene(10))
                a_frames.append(rel)
            b = real_video(root, "real/002", tone=210, deltas=(0,) * 16, count=16)
            record = decide_video(
                "FF-DF",
                "001_002",
                fake,
                {"frames": a_frames},
                b,
                root=root,
                config=PairingConfig(),
            )
        self.assertEqual(record["decision"], "timeline_mismatch")
        self.assertFalse(record["verified"])
        self.assertFalse(record["timelines_match"])
        self.assertLess(record["timeline_jaccard"], 0.9)
        # Half the fake's moments have no counterpart at all, which the reported
        # coverage distinguishes from a merely *nested* grid.
        self.assertAlmostEqual(record["timeline_coverage"], 0.5)

    def test_evidence_pairs_are_identical_index_pairs(self):
        """The measurement must cover exactly the pairs the builder will train on."""
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60, count=12)
            # A lacks index 6 (a dropped detection) and is shifted elsewhere.
            a_frames = [f"real/001/{i:03d}.png" for i in range(12) if i != 6]
            for i in range(12):
                if i != 6:
                    _write(root, f"real/001/{i:03d}.png", _scene(10))
            b = real_video(root, "real/002", tone=210, deltas=(0,) * 12, count=12)
            record = decide_video(
                "FF-DF",
                "001_002",
                fake,
                {"frames": a_frames},
                b,
                root=root,
                config=PairingConfig(),
            )
        measured = 0
        for row in record["frame_pairs"]:
            if row["d_A"] is not None:
                self.assertEqual(row["A_index"], row["fake_index"])
                measured += 1
            if row["B_index"] is not None:
                self.assertEqual(row["B_index"], row["fake_index"])
            # Index 6 has no A frame; it must never be paired against one.
            if row["A_index"] is not None or row["d_A"] is not None:
                self.assertNotEqual(row["fake_index"], 6)
        self.assertEqual(measured, record["measured_A"])
        self.assertEqual(record["intersection_A"], 11)
        self.assertEqual(record["mode"], "head_to_head")

    def test_unreadable_candidate_frames_cannot_hand_the_win_to_the_other_side(self):
        """A candidate with almost no readable frames must never be out-voted."""
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60, count=12)
            a = real_video(root, "real/001", tone=10, deltas=(0,) * 12, count=12)
            # B is the true background source but only its last frame survived.
            b = real_video(root, "real/002", tone=10, deltas=(0,) * 12, count=12)
            b["frames"] = [f"real/002/{i:03d}.png" for i in range(11)] + [
                b["frames"][11]
            ]
            for i in range(11):
                (root / f"real/002/{i:03d}.png").unlink()
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertEqual(record["decision"], "unusable_frames")
        self.assertFalse(record["verified"])
        self.assertIsNone(record["reference_component"])
        self.assertEqual(record["measured_B"], 1)
        self.assertLess(record["measured_B"], PairingConfig().min_verification_frames)

    def test_candidate_shorter_than_the_sample_window_is_excluded_not_crashed(self):
        from g21.pairing import DECISION_CODES, PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60, count=8)
            _write(root, "real/001/000.png", _scene(10))
            a = {"frames": ["real/001/000.png"]}
            b = real_video(root, "real/002", tone=10, deltas=(0,) * 8, count=8)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertIn(record["decision"], DECISION_CODES)
        self.assertEqual(record["intersection_A"], 1)
        self.assertEqual(record["measured_A"], 1)
        self.assertEqual(record["mode"], "single_side_confirmation")
        # A is measurable on one frame and B on eight, and on the single frame
        # both share the two backgrounds are identical: that support difference
        # is not evidence, so neither side may be named.
        self.assertEqual(record["shared_pairs"], 1)
        self.assertEqual(record["shared_agreement"], -1)
        self.assertIsNone(record["winner"])
        self.assertEqual(record["decision"], "insufficient_intersection")
        self.assertFalse(record["verified"])

    def test_a_shared_frame_beats_a_bigger_single_sided_support(self):
        """The wrong-candidate regression: a near-by scene must not be confirmed."""
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60, count=8)
            # A is a *different* scene that happens to be 3 gray levels away -
            # inside max_single_side_distance - and is measurable on all eight
            # frames.  B is the true source but only its last frame survived.
            a = real_video(root, "real/001", tone=13, deltas=(0,) * 8, count=8)
            b = real_video(root, "real/002", tone=10, deltas=(0,) * 8, count=8)
            b["frames"] = [b["frames"][7]]
            for i in range(7):
                (root / f"real/002/{i:03d}.png").unlink()
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertEqual(record["mode"], "single_side_confirmation")
        self.assertEqual(record["measured_A"], 8)
        self.assertEqual(record["measured_B"], 1)
        self.assertTrue(record["confirmed_A"])
        self.assertFalse(record["confirmed_B"])
        self.assertLessEqual(record["d_A"], PairingConfig().max_single_side_distance)
        # ... yet on the one frame both sides can be measured, B is closer, so A
        # is contradicted and must not be written into pair_map as the reference.
        self.assertEqual(record["shared_pairs"], 1)
        self.assertEqual(record["shared_agreement"], 1)
        self.assertIsNone(record["reference_component"])
        self.assertFalse(record["verified"])
        self.assertEqual(record["decision"], "insufficient_intersection")
        self.assertEqual(
            record["frame_pairs"][-1]["fake_index"],
            record["frame_pairs"][-1]["B_index"],
        )

    def test_single_sided_confirmation_needs_a_tighter_absolute_match(self):
        """Without a second opinion, a merely close-by scene must not be accepted."""
        from g21.pairing import PairingConfig, decide_video

        config = PairingConfig()
        self.assertLess(
            config.max_single_side_distance, config.max_winner_distance
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60, count=8)
            # A shares only one index; B shares all eight but its background is
            # 6 gray levels off - inside the head-to-head bound, outside the
            # single-sided one.
            _write(root, "real/001/000.png", _scene(10))
            a = {"frames": ["real/001/000.png"]}
            b = real_video(root, "real/002", tone=16, deltas=(0,) * 8, count=8)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=config
            )
        self.assertEqual(record["mode"], "single_side_confirmation")
        self.assertEqual(record["decision"], "insufficient_intersection")
        self.assertFalse(record["verified"])
        self.assertLessEqual(record["measured_B"], 8)

    def test_full_mask_makes_every_frame_unusable(self):
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60)
            full = _mask(255, region=(slice(0, SIZE), slice(0, SIZE)))
            fake["masks"] = []
            for i in range(FRAME_COUNT):
                rel = f"fake/full_{i:03d}.png"
                _write(root, rel, full)
                fake["masks"].append(rel)
            a = real_video(root, "real/001", tone=10)
            b = real_video(root, "real/002", tone=210)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertEqual(record["decision"], "unusable_frames")
        self.assertEqual(
            record["frame_pairs"][0]["note"], "eligible_fraction_below_floor"
        )

    def test_missing_mask_falls_back_to_face_box_and_is_recorded(self):
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60, masks=None)
            a = real_video(root, "real/001", tone=10)
            b = real_video(root, "real/002", tone=210)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertEqual(record["decision"], "verified")
        self.assertEqual(record["mask_modes"], ["central_face_box_fallback"])

    def test_an_out_of_order_frame_list_is_accepted_and_read_by_index(self):
        """FF++ stores 69 real videos out of order; position is not evidence."""
        from g21.pairing import PairingConfig, decide_video, frame_slots

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60, count=8)
            # A's file for index i has a background i gray levels off the fake's,
            # and the list is *reversed*, so a positional read would compare the
            # fake against the wrong image on every row.
            a = {"frames": [f"real/001/{i:03d}.png" for i in range(7, -1, -1)]}
            for i in range(8):
                _write(root, f"real/001/{i:03d}.png", _scene(10 + i))
            b = real_video(root, "real/002", tone=210)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertEqual(record["frames_out_of_order"], ["A"])
        self.assertEqual(frame_slots(a)[0], 7)  # index 0 is stored last
        self.assertEqual(record["measured_A"], 8)
        rows = {row["fake_index"]: row for row in record["frame_pairs"]}
        self.assertEqual(rows[0]["A_index"], 0)
        self.assertLess(rows[0]["d_A"], 0.5)
        self.assertGreater(rows[7]["d_A"], 5.0)
        self.assertEqual(record["decision"], "verified")
        self.assertEqual(record["reference_component"], 0)

    def test_duplicate_frame_indices_are_still_refused(self):
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake", tone=10, delta=60, count=8)
            _write(root, "real/001/000.png", _scene(10))
            a = {"frames": ["real/001/000.png", "real/001/000.png"]}
            b = real_video(root, "real/002", tone=210)
            record = decide_video(
                "FF-DF", "001_002", fake, a, b, root=root, config=PairingConfig()
            )
        self.assertEqual(record["decision"], "malformed_entry")
        self.assertIn("unique", record["error"])

    def test_missing_candidate_is_reported_not_guessed(self):
        from g21.pairing import PairingConfig, decide_video

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = fake_video(root, "fake")
            a = real_video(root, "real/001", tone=10)
            record = decide_video(
                "FF-DF", "001_003", fake, a, None, root=root, config=PairingConfig()
            )
        self.assertEqual(record["decision"], "missing_candidate")
        self.assertEqual(record["missing_candidates"], ["003"])
        self.assertFalse(record["verified"])


class PairingRunContracts(unittest.TestCase):
    def _dataset(self, root):
        real_train = {
            "001": real_video(root, "real/001", tone=10),
            "002": real_video(root, "real/002", tone=210),
        }
        data = {
            label: {"train": {"c23": {}}, "val": {"c23": {}}, "test": {"c23": {}}}
            for label in ("FF-real", "FF-DF", "FF-F2F", "FF-FS", "FF-NT")
        }
        data["FF-real"]["train"]["c23"] = real_train
        for method in ("FF-DF", "FF-F2F", "FF-FS", "FF-NT"):
            data[method]["train"]["c23"]["001_002"] = fake_video(root, f"fake/{method}", tone=10)
            data[method]["train"]["c23"]["001_003"] = {"frames": ["missing/000.png"]}
        return {"FaceForensics++": data}

    def test_run_writes_a_pair_map_the_builder_accepts(self):
        from g21.builder import build_ffpp
        from g21.pairing import PairingConfig, run_pairing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(
                json.dumps(self._dataset(root)), encoding="utf-8"
            )
            report = run_pairing(
                root / "data.json", root, root / "pairing", config=PairingConfig()
            )
            self.assertEqual(report["usable_pairs"], 4)
            self.assertEqual(report["decision_counts"]["missing_candidate"], 4)
            manifest = build_ffpp(
                root / "data.json", root / "pairing/pair_map.json", root / "out"
            )
        self.assertEqual(manifest["pair_count"], 4)
        reasons = {row["reason_code"] for row in manifest["exclusions"]}
        self.assertEqual(reasons, {"explicit_exclusion"})

    def test_repeated_runs_are_byte_identical(self):
        from g21.pairing import PairingConfig, run_pairing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(
                json.dumps(self._dataset(root)), encoding="utf-8"
            )
            run_pairing(root / "data.json", root, root / "one", config=PairingConfig())
            run_pairing(root / "data.json", root, root / "two", config=PairingConfig())
            for name in ("pair_map.json", "pairing_audit.json"):
                self.assertEqual(
                    (root / "one" / name).read_bytes(),
                    (root / "two" / name).read_bytes(),
                )

    def test_a_run_whose_masks_starve_every_row_warns_about_polarity(self):
        """An inverted mask must not be indistinguishable from absent evidence."""
        from g21.pairing import PairingConfig, run_pairing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            full = _mask(255, region=(slice(0, SIZE), slice(0, SIZE)))
            for method in ("FF-DF", "FF-F2F", "FF-FS", "FF-NT"):
                fake = dataset["FaceForensics++"][method]["train"]["c23"]["001_002"]
                fake["masks"] = []
                for i in range(FRAME_COUNT):
                    rel = f"fake/{method}/full_{i:03d}.png"
                    _write(root, rel, full)
                    fake["masks"].append(rel)
            (root / "data.json").write_text(json.dumps(dataset), encoding="utf-8")
            report = run_pairing(
                root / "data.json", root, root / "pairing", config=PairingConfig()
            )
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("eligible_floor", report["warnings"][0])
        self.assertIn("polarity", report["warnings"][0])

    def test_a_run_without_any_readable_mask_warns_about_the_fallback(self):
        from g21.pairing import PairingConfig, run_pairing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._dataset(root)
            for method in ("FF-DF", "FF-F2F", "FF-FS", "FF-NT"):
                fake = dataset["FaceForensics++"][method]["train"]["c23"]["001_002"]
                fake.pop("masks", None)
            (root / "data.json").write_text(json.dumps(dataset), encoding="utf-8")
            report = run_pairing(
                root / "data.json", root, root / "pairing", config=PairingConfig()
            )
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("face box", report["warnings"][0])

    def test_existing_output_is_not_overwritten(self):
        from g21.pairing import PairingConfig, run_pairing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(
                json.dumps(self._dataset(root)), encoding="utf-8"
            )
            run_pairing(root / "data.json", root, root / "out", config=PairingConfig())
            with self.assertRaises(FileExistsError):
                run_pairing(
                    root / "data.json", root, root / "out", config=PairingConfig()
                )


def verifier_module():
    """Load the standalone CLI the way a user would run it from a shell."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "experiments/verify_g21_pairing.py"
    spec = importlib.util.spec_from_file_location("g21_verifier_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_dataset(root, b_tone=210):
    """One fake key per method; the fake background copies real 001's tone."""
    real_train = {
        "001": real_video(root, "real/001", tone=10),
        "002": real_video(root, "real/002", tone=b_tone),
    }
    data = {
        label: {"train": {"c23": {}}, "val": {"c23": {}}, "test": {"c23": {}}}
        for label in ("FF-real", "FF-DF", "FF-F2F", "FF-FS", "FF-NT")
    }
    data["FF-real"]["train"]["c23"] = real_train
    for method in ("FF-DF", "FF-F2F", "FF-FS", "FF-NT"):
        data[method]["train"]["c23"]["001_002"] = fake_video(
            root, f"fake/{method}", tone=10
        )
    return {"FaceForensics++": data}


class PairingCliContracts(unittest.TestCase):
    def _run(self, verifier, argv):
        import contextlib
        import io

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = verifier.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_cli_writes_a_pair_map_builder_accepts_and_reports_the_funnel(self):
        from g21.builder import build_ffpp

        verifier = verifier_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(
                json.dumps(cli_dataset(root)), encoding="utf-8"
            )
            code, out, err = self._run(
                verifier,
                [
                    "--dataset-json",
                    str(root / "data.json"),
                    "--data-root",
                    str(root),
                    "--output",
                    str(root / "pairing"),
                    # This fixture carries a single fake per method, so the
                    # default four-source floor must report the shortfall
                    # honestly; the gate itself is covered by its own test.
                    "--min-sources",
                    "1",
                ],
            )
            summary = json.loads(out)
            self.assertEqual(err, "")
            self.assertEqual(summary["status"], "PAIRING_OK")
            self.assertEqual(summary["usable_pairs"], 4)
            self.assertEqual(summary["methods_without_usable_pairs"], [])
            self.assertEqual(summary["references_by_method"]["FF-DF"], 1)
            self.assertIn("distributions", summary)
            self.assertEqual(summary["warnings"], [])
            for name in ("pair_map.json", "pairing_audit.json"):
                self.assertTrue((root / "pairing" / name).is_file())
            manifest = build_ffpp(
                root / "data.json", root / "pairing/pair_map.json", root / "out"
            )
        self.assertEqual(code, 0)
        self.assertEqual(manifest["pair_count"], 4)

    def test_cli_flags_a_method_below_the_microbatch_source_floor(self):
        """A pair map preflight would reject must not be reported as OK."""
        verifier = verifier_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(
                json.dumps(cli_dataset(root)), encoding="utf-8"
            )
            code, out, err = self._run(
                verifier,
                [
                    "--dataset-json",
                    str(root / "data.json"),
                    "--data-root",
                    str(root),
                    "--output",
                    str(root / "pairing"),
                    "--require-usable",
                ],
            )
            summary = json.loads(out)
        self.assertEqual(err, "")
        self.assertEqual(summary["status"], "PAIRING_INCOMPLETE")
        self.assertEqual(
            summary["methods_below_min_sources"],
            ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"],
        )
        self.assertEqual(code, 1)

    def test_cli_require_usable_fails_closed_when_a_method_is_ambiguous(self):
        verifier = verifier_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(
                json.dumps(cli_dataset(root, b_tone=10)), encoding="utf-8"
            )
            argv = [
                "--dataset-json",
                str(root / "data.json"),
                "--data-root",
                str(root),
                "--output",
                str(root / "pairing"),
                "--require-usable",
            ]
            code, out, err = self._run(verifier, argv)
            summary = json.loads(out)
        self.assertEqual(err, "")
        self.assertEqual(summary["status"], "PAIRING_INCOMPLETE")
        self.assertEqual(summary["usable_pairs"], 0)
        self.assertEqual(
            summary["methods_without_usable_pairs"],
            ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"],
        )
        self.assertEqual(code, 1)

    def test_cli_reports_a_missing_dataset_on_stderr(self):
        verifier = verifier_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code, out, err = self._run(
                verifier,
                [
                    "--dataset-json",
                    str(root / "absent.json"),
                    "--data-root",
                    str(root),
                    "--output",
                    str(root / "pairing"),
                ],
            )
        self.assertEqual(out, "")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(err)["status"], "PAIRING_FAILED")


def ground_truth_dataset(root, b_methods=("FF-DF",)):
    """Four-method FF++ JSON whose fake background copies a known candidate.

    ``b_methods`` names the methods whose fake copies real ``002``; every other
    fake copies real ``001``.  The correct ``reference_component`` is therefore
    known by construction, which is what makes this an end-to-end ground-truth
    check rather than a self-consistency check.
    """
    real_train = {
        "001": real_video(root, "real/001", tone=10),
        "002": real_video(root, "real/002", tone=210),
    }
    data = {
        label: {"train": {"c23": {}}, "val": {"c23": {}}, "test": {"c23": {}}}
        for label in ("FF-real", "FF-DF", "FF-F2F", "FF-FS", "FF-NT")
    }
    data["FF-real"]["train"]["c23"] = real_train
    for method in ("FF-DF", "FF-F2F", "FF-FS", "FF-NT"):
        tone = 210 if method in b_methods else 10
        data[method]["train"]["c23"]["001_002"] = fake_video(
            root, f"fake/{method}", tone=tone
        )
    return {"FaceForensics++": data}


def expected_components(b_methods):
    return {
        method: (1 if method in b_methods else 0)
        for method in ("FF-DF", "FF-F2F", "FF-FS", "FF-NT")
    }


class PairingGroundTruthContracts(unittest.TestCase):
    def test_verifier_recovers_the_injected_candidate_for_every_method(self):
        from g21.pairing import PairingConfig, run_pairing

        b_methods = ("FF-DF", "FF-FS")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(
                json.dumps(ground_truth_dataset(root, b_methods=b_methods)),
                encoding="utf-8",
            )
            report = run_pairing(
                root / "data.json", root, root / "pairing", config=PairingConfig()
            )
            audit = json.loads(
                (root / "pairing/pairing_audit.json").read_text(encoding="utf-8")
            )
        self.assertEqual(report["usable_pairs"], 4)
        records = {row["method"]: row for row in audit["decisions"]}
        for method, component in expected_components(b_methods).items():
            record = records[method]
            self.assertEqual(record["decision"], "verified")
            self.assertTrue(record["verified"])
            self.assertEqual(record["reference_component"], component)
            self.assertEqual(record["winner"], "B" if component else "A")
            self.assertEqual(record["win_rate"], 1.0)
            self.assertGreaterEqual(record["margin_ratio"], 0.5)
            self.assertLessEqual(
                record["d_A"] if component == 0 else record["d_B"], 12.0
            )
            self.assertTrue(record["timelines_match"])

    def test_end_to_end_pair_map_points_at_the_ground_truth_source(self):
        from g21.builder import build_ffpp
        from g21.pairing import PairingConfig, run_pairing

        b_methods = ("FF-DF", "FF-NT")
        components = expected_components(b_methods)
        expected_refs = {
            method: ("002" if component else "001")
            for method, component in components.items()
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(
                json.dumps(ground_truth_dataset(root, b_methods=b_methods)),
                encoding="utf-8",
            )
            run_pairing(
                root / "data.json", root, root / "pairing", config=PairingConfig()
            )
            pair_map = json.loads(
                (root / "pairing/pair_map.json").read_text(encoding="utf-8")
            )
            manifest = build_ffpp(
                root / "data.json", root / "pairing/pair_map.json", root / "out"
            )
            pairs = [
                json.loads(line)
                for line in (root / "out/pairs_train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        # A generated map never asserts verification at the method level; the
        # content decision lives in a per-video override only.
        self.assertTrue(
            all(rule["verified"] is False for rule in pair_map["rules"].values())
        )
        overrides = pair_map["overrides"]
        for method, component in components.items():
            override = overrides[f"{method}/001_002"]
            self.assertTrue(override["verified"])
            self.assertEqual(override["reference_component"], component)
            self.assertTrue(override["same_original_index_verified"])
            # No explicit frame_matches are invented for a verified timeline.
            self.assertNotIn("frame_matches", override)
        self.assertEqual(manifest["pair_count"], 4)
        self.assertEqual(
            {row["method"]: row["content_reference_id"] for row in pairs},
            expected_refs,
        )
        self.assertEqual(
            {row["method"]: row["real_video_id"] for row in pairs},
            {
                method: f"ffpp/c23/FF-real/{ref}"
                for method, ref in expected_refs.items()
            },
        )

    def test_audit_carries_the_required_fields_and_a_partitioning_funnel(self):
        from g21.pairing import PairingConfig, run_pairing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.json").write_text(
                json.dumps(ground_truth_dataset(root)), encoding="utf-8"
            )
            report = run_pairing(
                root / "data.json", root, root / "pairing", config=PairingConfig()
            )
            audit = json.loads(
                (root / "pairing/pairing_audit.json").read_text(encoding="utf-8")
            )
        required = {
            "method",
            "fake_key",
            "candidate_a",
            "candidate_b",
            "reference_component",
            "winner",
            "d_A",
            "d_B",
            "median_A",
            "median_B",
            "trimmed_mean_A",
            "trimmed_mean_B",
            "win_rate",
            "margin_ratio",
            "eligible_fraction",
            "mask_modes",
            "estimated_lengths",
            "intersection_common",
            "measured_A",
            "measured_B",
            "confirmed_A",
            "confirmed_B",
            "shared_pairs",
            "shared_agreement",
            "timeline_jaccard",
            "timeline_coverage",
            "timelines_match",
            "decision",
            "verified",
        }
        for record in audit["decisions"]:
            self.assertTrue(required <= set(record), sorted(required - set(record)))
        for name in ("warnings", "funnel", "distributions", "schema_version"):
            self.assertIn(name, audit)
        # Distributions describe the accepted decisions and their rejected
        # opponent, never the refused videos.
        self.assertEqual(
            set(audit["distributions"]),
            {"d_winner_verified", "d_rejected_on_verified", "margin_ratio_verified"},
        )
        for record in audit["decisions"]:
            if not record["verified"]:
                continue
            self.assertEqual(record["shared_pairs"], len(
                [r for r in record["frame_pairs"] if r["d_A"] is not None and r["d_B"] is not None]
            ))
        overall = audit["funnel"]["overall"]
        named = sum(
            overall[key]
            for key in (
                "verified",
                "ambiguous_content",
                "weak_margin",
                "not_closer",
                "timeline_mismatch",
                "other",
            )
        )
        self.assertEqual(named, overall["total"])
        self.assertEqual(overall["usable_pairs"], overall["verified"])
        self.assertEqual(sum(report["decision_counts"].values()), overall["total"])


if __name__ == "__main__":
    unittest.main()
